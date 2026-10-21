"""0730 source profiles and command timing; no native simulation."""
import ast
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from wheeled_tasks.chassis.tasks.commands.flat_command_sampling import ScutFlatCommands
from wheeled_tasks.chassis.tasks.commands.rough_commands import ScutRoughCommands, command_definition
from wheeled_tasks.chassis.simulation.terrain.rough_terrain import ENV_FILE, ENV_SHA, _read_bound


def terrain_origins():
    result = torch.zeros(10, 14, 3)
    result[:, :, 0] = ((torch.arange(10) + .5) * 9. - 45.)[:, None]
    result[:, :, 1] = ((torch.arange(14) + .5) * 9. - 63.)[None, :]
    return result


def owner(count=14, seed=66):
    tiles = terrain_origins()
    origins = tiles[4, torch.arange(count) % 14].clone()
    commands = ScutRoughCommands(count, "cpu", terrain_origins=tiles, env_origins=origins)
    generator = torch.Generator().manual_seed(seed)
    commands.reset(torch.arange(count), torch.zeros(count), torch.zeros(count, dtype=torch.bool), generator)
    return commands, generator, origins


def advance(commands, generator, positions):
    commands.begin_policy()
    for _ in range(4):
        commands.advance_physics()
    commands.advance(.02, torch.zeros(commands.num_envs), 0, generator, world_positions=positions)


def source_manager(origins):
    path = next((parent / "scut-wheeled-legged-rl/source/agent_tasks/agent_tasks/manager/mdp/terrain/terrain_command_manager.py"
                 for parent in Path(__file__).resolve().parents
                 if (parent / "scut-wheeled-legged-rl/source/agent_tasks/agent_tasks/manager/mdp/terrain/terrain_command_manager.py").is_file()), None)
    if path is None:
        pytest.skip("Pinned source checkout unavailable")
    namespace = dict(torch=torch, Sequence=Sequence, Real=Real, dataclass=dataclass, configclass=dataclass)
    tree = ast.parse(path.read_text())
    # The source manager is pure Torch. Avoid its SDK decorator import; the
    # ordinary dataclass preserves exactly these all-optional config fields.
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
                 or (isinstance(node, ast.ImportFrom) and node.module == "__future__")]
    exec(compile(tree, str(path), "exec"), namespace)
    class Loader(yaml.SafeLoader):
        pass
    for tag in ("python/tuple", "python/object/apply:builtins.slice"):
        Loader.add_constructor("tag:yaml.org,2002:" + tag,
                               lambda loader, node: loader.construct_sequence(node))
    env = yaml.load(_read_bound(ENV_FILE, ENV_SHA), Loader=Loader)
    generator = env["terrain"]["terrain_generator"]
    config = NS(height_range=env["height_range"], reset_heading_axis_aligned_only=False,
        terrain_command_switch_hold_steps=env["terrain_command_switch_hold_steps"],
        commands=NS(ranges=NS(**env["commands"]["ranges"])),
        terrain_command_overrides={k: namespace["TerrainCommandOverrideCfg"](**v)
                                  for k, v in env["terrain_command_overrides"].items()})
    terrain = NS(cfg=NS(terrain_type="generator", terrain_generator=NS(
        num_rows=10, num_cols=14, size=[9., 9.], curriculum=True,
        sub_terrains={k: NS(**v) for k, v in generator["sub_terrains"].items()})),
        terrain_origins=terrain_origins(), env_origins=origins,
        terrain_types=torch.arange(len(origins)) % 14)
    return namespace["TerrainCommandManager"](terrain, config, "cpu", len(origins))


def test_actual_source_manager_height_draws_and_current_location_profiles_match():
    commands, _, origins = owner()
    source = source_manager(origins)
    ids = torch.arange(14)
    source.sync_envs(ids, use_env_origins=True)
    assert torch.equal(source.current_terrain_key_indices, commands._rough["terrain_key_ids"])
    assert torch.equal(source.current_profile_ids, commands._rough["terrain_profile_ids"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        expected = source.sample_height(ids)
    commands._sample_height(ids, torch.Generator().manual_seed(123))
    assert torch.equal(expected, commands.height)
    positions = origins.clone()
    positions[0, 1] = origins[5, 1]  # tiny -> mid stairs: different profile
    positions[1, 1] = origins[2, 1]  # tiny -> slope: same profile
    expected_changed = source.sync_envs(ids, root_pos_w=positions)
    assert commands._sync_terrain(ids, positions).tolist() == expected_changed.tolist() == [0]
    definition = command_definition()
    assert definition["height_ranges"][definition["keys"].index("inv_stair_slope_for_rm_mid")] == (.2, .42)
    assert "stair_for_rm_rot" not in definition["keys"]


def test_rough_special_starts_zero_preserve_readiness_and_double_standing():
    commands, generator, _ = owner(4096)
    assert (commands.special_mode_id == -1).all()  # newborns are not 5s-ready
    commands.episode_physics_ticks[:] = 1000
    commands._resample(torch.arange(4096), 0, generator)
    assert all((commands.special_mode_id == mode).sum() > 250 for mode in range(3))
    second_scope = ~commands.standing_first_draw & (commands.special_mode_id < 0)
    assert torch.equal(commands.standing_second_draw >= 0, second_scope)
    assert torch.equal(commands.is_standing[second_scope], commands.standing_second_draw[second_scope].bool())
    special = commands.special_mode_id >= 0
    before = commands.generator_command3.clone()
    commands._postprocess(torch.ones(4096))
    assert torch.equal(commands.generator_command3[special], before[special])
    assert (commands.generator_command3[commands.is_standing] == 0).all()
    assert [mode["iteration_start"] for mode in commands.summary()["special_modes"]] == [0, 0, 0]
    flat = ScutFlatCommands(4096, "cpu")
    flat.episode_physics_ticks[:] = 1000
    flat._resample(torch.arange(4096), 0, torch.Generator().manual_seed(66))
    assert (flat.special_mode_id == -1).all()


def test_switch_force_resets_command_counter_but_retains_episode_age():
    commands, generator, positions = owner(1)
    advance(commands, generator, positions)  # consume initial height observer
    commands.command_counter[:] = 7
    commands._height_seen_command_counter[:] = 7
    commands.episode_physics_ticks[:] = 1000
    commands.time_left[:] = 10.
    same = positions.clone()
    same[:, 1] = terrain_origins()[4, 2, 1]
    advance(commands, generator, same)
    assert commands.command_counter.item() == 7
    changed = positions.clone()
    changed[:, 1] = terrain_origins()[4, 5, 1]
    advance(commands, generator, changed)
    assert commands.command_counter.item() == commands._height_seen_command_counter.item() == 1
    assert commands.episode_physics_ticks.item() == 1008
    assert commands._rough["profile_switch_count"].item() == 1
    assert .24 <= commands.height.item() <= .38
    assert 5. <= commands.time_left.item() <= 15.
    advance(commands, generator, changed)
    assert commands.command_counter.item() == 1


def test_timer_resample_synchronizes_profile_before_switch_hook():
    commands, generator, positions = owner(1)
    advance(commands, generator, positions)
    commands.command_counter[:] = 7
    commands._height_seen_command_counter[:] = 7
    commands.time_left[:] = .001
    positions[:, 1] = terrain_origins()[4, 5, 1]
    advance(commands, generator, positions)
    assert commands.command_counter.item() == 8
    assert commands._rough["profile_switch_count"].item() == 0
    assert .24 <= commands.height.item() <= .38


def test_complete_restart_replays_and_rejects_profile_or_origin_mismatch_atomically():
    commands, generator, positions = owner()
    advance(commands, generator, positions)
    positions[0, 1] = positions[5, 1]
    advance(commands, generator, positions)
    state = commands.state_dict(generator)
    restored, other_rng, _ = owner(seed=129)
    restored.load_state_dict(state, other_rng)
    advance(commands, generator, positions)
    advance(restored, other_rng, positions)
    for name, value in commands.diagnostics().items():
        assert torch.equal(value, restored.diagnostics()[name]), name
    assert torch.equal(generator.get_state(), other_rng.get_state())
    bad = deepcopy(state)
    bad["rough"]["terrain_profile_ids"][0] = -1
    rng_before, height_before = other_rng.get_state().clone(), restored.height.clone()
    with pytest.raises(ValueError, match="resolved profile"):
        restored.load_state_dict(bad, other_rng)
    assert torch.equal(rng_before, other_rng.get_state()) and torch.equal(height_before, restored.height)
    changed_origins = terrain_origins()[4].clone()
    changed_origins[0, 0] += .1
    different = ScutRoughCommands(14, "cpu", terrain_origins=terrain_origins(), env_origins=changed_origins)
    with pytest.raises(ValueError, match="identity mismatch"):
        different.load_state_dict(state, torch.Generator())


def test_numeric_terminal_position_keeps_profile_and_fixed_requests_survive_switch():
    commands, generator, positions = owner(1)
    advance(commands, generator, positions)
    key = commands._rough["terrain_key_ids"].clone()
    advance(commands, generator, torch.full_like(positions, torch.nan))
    assert torch.equal(key, commands._rough["terrain_key_ids"])
    commands.set_fixed_requests([0], torch.tensor([[.5, 0., .2]]), torch.tensor([.31]), evaluation_only=True)
    commands.reset([0], torch.zeros(1), torch.zeros(1, dtype=torch.bool), generator)
    positions[:, 1] = terrain_origins()[4, 5, 1]
    advance(commands, generator, positions)
    assert torch.equal(commands.command3, torch.tensor([[.5, 0., .2]]))
    assert commands.height.item() == pytest.approx(.31)
