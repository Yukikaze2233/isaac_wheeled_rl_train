"""Final command distributions and source hooks on CPU, without native births."""
import ast
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from numbers import Real
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.tasks.commands.final_rough_commands import FinalRoughCommands, STATE_VERSION
from wheeled_tasks.chassis.tasks.terrain.final_rough_definition import load_final_rough_definition
from wheeled_tasks.chassis.tasks.commands.flat_command_sampling import ScutFlatCommands


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "references/v7_final_terrain_b8ff79f3"


def grid():
    value = torch.zeros(10, 13, 3)
    value[..., 0] = torch.arange(10)[:, None] * 9. + 4.5
    value[..., 1] = torch.arange(13)[None, :] * 9. + 4.5
    return value


def owner(count=13, columns=None):
    terrain = grid()
    columns = torch.arange(count) % 13 if columns is None else torch.as_tensor(columns)
    origins = terrain[0, columns].clone()
    return FinalRoughCommands(count, "cpu", terrain_origins=terrain, env_origins=origins), origins


def tick(commands, generator, positions, heading=None, iteration=0):
    commands.begin_policy()
    for _ in range(4):
        commands.advance_physics()
    heading = torch.zeros(commands.num_envs) if heading is None else heading
    commands.advance(.02, heading, iteration, generator, world_positions=positions)


def ast_classes(filename, names, scope):
    nodes = [node for node in ast.parse((REFERENCE / filename).read_text()).body
             if isinstance(node, ast.ClassDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 str(REFERENCE / filename), "exec"), scope)


def source_manager(origins):
    scope = {"torch": torch, "dataclass": dataclass, "configclass": lambda cls: cls,
             "Sequence": Sequence, "Real": Real}
    ast_classes("terrain_command_manager.py", {"TerrainCommandOverrideCfg", "ResolvedTerrainCommandProfile", "TerrainCommandManager"}, scope)
    definition = load_final_rough_definition()
    command, terrain, birth = definition["commands"], definition["terrain"]["terrain_generator"], definition["birth"]
    cfg = NS(commands=NS(ranges=NS(**command["ranges"])), height_range=command["height_range"],
        reset_heading_axis_aligned_only=birth["reset_heading_axis_aligned_only"], terrain_command_switch_hold_steps=0,
        terrain_command_overrides={key: NS(**value) for key, value in definition["overrides"].items()})
    generated = NS(num_rows=terrain["num_rows"], num_cols=terrain["num_cols"], size=terrain["size"], curriculum=True,
        sub_terrains={key: NS(**value) for key, value in terrain["sub_terrains"].items()})
    bound = NS(cfg=NS(terrain_type="generator", terrain_generator=generated), terrain_origins=grid(), env_origins=origins)
    return scope["TerrainCommandManager"](bound, cfg, "cpu", len(origins))


def test_profiles_world_lookup_and_birth_heading_permissions_match_frozen_manager():
    commands, origins = owner()
    manager = source_manager(origins)
    ids = torch.arange(len(origins))
    manager.sync_envs(ids, root_pos_w=origins)
    keys, profiles = commands.profile_indices(origins)
    assert torch.equal(keys, manager.current_terrain_key_indices)
    assert torch.equal(profiles, manager.current_profile_ids)
    assert commands._final["profiles"] == [asdict(value) for value in manager._profiles_by_id]
    assert torch.equal(commands.axis_reset_mask(), manager.get_reset_heading_mask())
    assert torch.equal(commands.air_birth_allowed(), ~manager.get_disable_predefined_reset_air_mask())
    assert torch.equal(commands.ground_birth_allowed(), ~manager.get_disable_predefined_reset_ground_mask())
    assert torch.equal(commands._final["flags"], torch.stack([manager._command_override_enabled_by_field[field]
        for field in ("lin_vel_x", "lin_vel_y", "ang_vel_z_heading", "ang_vel_z_non_heading")], -1))
    assert commands.summary()["reset_heading_candidates_deg"] == [0., 90., 180., -90.]
    assert commands.summary()["predefined_air_birth_permission_only"] is True
    assert commands.summary()["predefined_air_birth_sampler_implemented"] is False
    assert commands.summary()["airborne_entry_command_resampling_implemented"] is False
    assert commands.summary()["stepup_command_link_implemented"] is False
    # Both XY axes clamp to the edge tile in the source lookup; key columns
    # are independent of terrain row for this generated curriculum mesh.
    outside = origins.clone()
    outside[:, 0] = -10000.
    outside[0, 1], outside[-1, 1] = -10000., 10000.
    manager.sync_envs(ids, root_pos_w=outside)
    assert torch.equal(commands.profile_indices(outside)[0], manager.current_terrain_key_indices)


def test_reset_and_timed_height_draws_respect_ranges_without_constructor_double_draw():
    commands, origins = owner()
    rng = torch.Generator().manual_seed(313)
    commands.reset(torch.arange(13), torch.zeros(13), torch.zeros(13, dtype=torch.bool), rng)
    before = commands.height.clone()
    tick(commands, rng, origins)
    assert torch.equal(commands.height, before)
    assert commands._height_seen_command_counter.eq(1).all()
    assert commands.command_counter.eq(1).all()
    assert not commands.special_mode_id.ge(0).any()
    assert commands.time_left.ge(5. - .02).all() and commands.time_left.le(15.).all()
    for lane, key in enumerate(commands._final["terrain_key_ids"].tolist()):
        ranges = commands._final["profiles"][int(commands._final["key_profile_ids"][key])]["height_range"]
        assert any(low <= commands.height[lane] <= high for low, high in ranges)
    commands.time_left.zero_()
    tick(commands, rng, origins)
    assert commands.command_counter.eq(2).all() and commands._height_seen_command_counter.eq(2).all()
    assert commands.height.ne(before).all()


def test_final_specials_activate_at_iteration_zero_after_five_seconds_on_plane():
    commands, origins = owner(2000, columns=[10] * 2000)
    rng = torch.Generator().manual_seed(17)
    commands.reset(torch.arange(2000), torch.zeros(2000), torch.zeros(2000, dtype=torch.bool), rng, iteration=0)
    commands.episode_physics_ticks.fill_(996)
    commands.time_left.zero_()
    tick(commands, rng, origins, iteration=0)
    assert [mode[1] for mode in commands.special_modes] == [0, 0, 0, 0]
    assert [mode[2] for mode in commands.special_modes] == [.15, .15, .2, .1]
    assert all(commands.special_mode_id.eq(index).any() for index in range(4))
    zero = commands.special_mode_id.eq(3)
    assert not commands.command3[zero].any()
    assert commands.command3[:, 1].eq(0.).all()
    for index, (_, _, _, vx, yaw) in enumerate(commands.special_modes):
        selected = commands.special_mode_id.eq(index)
        for column, ranges in ((0, vx), (2, yaw)):
            values = commands.command3[selected, column]
            assert torch.stack([(values >= low) & (values <= high) for low, high in ranges]).any(0).all()
    early, early_origins = owner(30, columns=[10] * 30)
    early.reset(torch.arange(30), torch.zeros(30), torch.zeros(30, dtype=torch.bool), rng)
    early.episode_physics_ticks.fill_(992)
    early.time_left.zero_()
    tick(early, rng, early_origins)
    assert early.special_mode_id.eq(-1).all()


def test_terrain_overrides_are_sampled_signed_segments_and_heading_yaw_is_restored():
    commands, origins = owner(600, columns=[0] * 600)
    rng = torch.Generator().manual_seed(23)
    commands.reset(torch.arange(600), torch.zeros(600), torch.zeros(600, dtype=torch.bool), rng)
    vx = commands.command3[:, 0]
    assert (vx < 0).any() and (vx > 0).any()
    profile = commands._final["profiles"][int(commands._final["terrain_profile_ids"][0])]
    assert torch.stack([(vx >= low) & (vx <= high) for low, high in profile["lin_vel_x"]]).any(0).all()
    # Force the source labels to prove priority: terrain vx overrides even
    # generator standing, while heading yaw survives its terrain override.
    commands.is_standing[:] = True
    commands.is_heading[0], commands.is_heading[1] = True, False
    commands.generator_command3.zero_()
    commands.generator_command3[0, 2] = .77
    commands._final["override_values"][:, 2] = -.66
    commands._final["override_values"][:, 3] = .04
    commands._publish()
    assert commands.command3[0, 2].item() == pytest.approx(.77)
    assert commands.command3[1, 2].item() == pytest.approx(.04)
    assert torch.equal(commands.command3[:, 0], commands._final["override_values"][:, 0])
    assert torch.equal(commands.generator_command3, commands.command3)


def test_profile_crossing_forces_one_reset_but_simultaneous_timer_resample_does_not_double_draw():
    commands, origins = owner(2, columns=[10, 10])
    rng = torch.Generator().manual_seed(37)
    commands.reset([0, 1], torch.zeros(2), torch.zeros(2, dtype=torch.bool), rng)
    moved = origins.clone()
    moved[:, 1] = grid()[0, 0, 1]
    commands.time_left[1] = 0.
    tick(commands, rng, moved)
    assert commands._final["profile_switch_count"].tolist() == [1, 0]
    assert commands.command_counter.tolist() == [1, 2]
    assert commands._height_seen_command_counter.tolist() == [1, 2]
    assert commands.episode_physics_ticks.tolist() == [4, 4]
    # Reset yaw policy is an episode latch. Crossing into an axis-aligned
    # terrain during this plane-born episode does not rewrite reset heading.
    assert not commands._final["axis_mask"].any()


def test_terrain_special_disable_uses_frozen_normal_resample_and_updates_whole_generator():
    commands, origins = owner(2, columns=[0, 10])
    rng = torch.Generator().manual_seed(43)
    commands.reset([0, 1], torch.zeros(2), torch.zeros(2, dtype=torch.bool), rng)
    commands.special_mode_id[:] = 2
    commands.is_standing.zero_()
    commands.is_heading[1] = True
    commands.heading_target[1] = .1
    commands.generator_command3[1, 2] = 5.
    time_left, counters = commands.time_left.clone(), commands.command_counter.clone()
    commands._disable_special_modes(torch.arange(2), torch.zeros(2), rng)
    assert commands.special_mode_id.tolist() == [-1, 2]
    # Source update_command skips heading for the still-special plane lane.
    assert commands.generator_command3[1, 2].item() == 5.
    assert commands.standing_second_draw[0] in (0, 1)
    assert torch.equal(commands.time_left, time_left) and torch.equal(commands.command_counter, counters)


def test_axis_reset_latches_actual_heading_and_resample_reuses_it():
    commands, origins = owner(3, columns=[0, 0, 10])
    rng = torch.Generator().manual_seed(47)
    actual_heading = torch.tensor([math.pi, math.pi / 2., .7])
    commands.reset([0, 1, 2], actual_heading, torch.zeros(3, dtype=torch.bool), rng)
    assert commands._final["axis_mask"].tolist() == [True, True, False]
    assert torch.equal(commands.heading_target[:2], actual_heading[:2])
    commands.time_left.zero_()
    tick(commands, rng, origins, heading=actual_heading + .1)
    assert torch.equal(commands.heading_target[:2], actual_heading[:2])
    heading = commands.is_heading[:2] & ~commands.is_standing[:2]
    assert torch.allclose(commands.command3[:2, 2][heading], torch.full_like(commands.command3[:2, 2][heading], -.5), atol=1e-6)


def test_partial_reset_accepts_selected_actual_heading_and_retains_other_policy_yaw():
    commands, origins = owner(3, columns=[0, 0, 10])
    rng = torch.Generator().manual_seed(51)
    commands.reset([0, 1, 2], torch.tensor([0., .2, .3]), torch.zeros(3, dtype=torch.bool), rng)
    tick(commands, rng, origins, heading=torch.tensor([.1, .4, .5]))
    commands.reset([1], torch.tensor([math.pi / 2.]), torch.tensor([False]), rng)
    assert torch.equal(commands._final["latest_heading"], torch.tensor([.1, math.pi / 2., .5]))
    assert commands._final["reset_heading"][1].item() == pytest.approx(math.pi / 2.)
    assert commands.heading_target[1].item() == pytest.approx(math.pi / 2.)
    assert commands.episode_physics_ticks.tolist() == [4, 0, 4]


def test_partial_axis_reset_publishes_other_lane_heading_after_ground_expiry():
    commands, _ = owner(2, columns=[10, 0])
    rng = torch.Generator().manual_seed(52)
    commands.reset([0, 1], torch.zeros(2), torch.tensor([True, False]), rng)
    commands.is_heading[0] = True
    commands.is_standing[0] = False
    commands.special_mode_id[0] = -1
    commands.heading_target[0] = .1
    commands.ground_restore_command3[0] = torch.tensor([.4, 0., .8])
    commands.episode_physics_ticks[:] = 300
    commands._publish()
    assert commands.command3[0, 2].item() == pytest.approx(.8)
    commands.reset([1], torch.tensor([math.pi]), torch.tensor([False]), rng)
    assert commands.command3[0, 2].item() == pytest.approx(.5, abs=1e-6)
    assert torch.equal(commands.command3, commands.generator_command3)


def test_final_v13_nonheading_override_excludes_base_axis_zero_mask():
    commands, origins = owner(2, columns=[0, 0])
    manager = source_manager(origins)
    rng = torch.Generator().manual_seed(54)
    commands.reset([0, 1], torch.zeros(2), torch.zeros(2, dtype=torch.bool), rng)
    manager.sync_envs(torch.arange(2), root_pos_w=origins)
    manager.command_override_masks["ang_vel_z_non_heading"].copy_(commands._final["override_masks"][:, 3])
    def method(filename, name):
        cls = next(node for node in ast.parse((REFERENCE / filename).read_text()).body if isinstance(node, ast.ClassDef))
        return deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))
    base = ast.ClassDef(name="SourceBase", bases=[], keywords=[], decorator_list=[],
        body=[method("base_env.py", "_get_non_heading_axis_aligned_zero_mask")])
    final = ast.ClassDef(name="SourceTerrain", bases=[ast.Name(id="SourceBase", ctx=ast.Load())], keywords=[],
        decorator_list=[], body=[method("v13_env.py", "_get_non_heading_axis_aligned_zero_mask")])
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[base, final], type_ignores=[])), "frozen_axis_priority", "exec"), scope)
    value = scope["SourceTerrain"]()
    value._get_axis_aligned_reset_heading_mask = lambda: commands._final["axis_mask"]
    value._get_terrain_command_manager = lambda: manager
    assert scope["SourceBase"]._get_non_heading_axis_aligned_zero_mask(value, torch.zeros(2, dtype=torch.bool)).all()
    assert not value._get_non_heading_axis_aligned_zero_mask(torch.zeros(2, dtype=torch.bool)).any()
    commands.is_heading.zero_()
    commands._final["override_values"][:, 3] = .055
    commands._publish()
    assert torch.allclose(commands.command3[:, 2], torch.tensor([.055, .055]))


def test_fixed_rehearsal_is_nominal_plane_and_retains_public_fixed_requests():
    commands, origins = owner(4, columns=[0, 0, 0, 0])
    rng = torch.Generator().manual_seed(53)
    commands.set_training_rehearsal_requests([0], torch.tensor([[.5, 0., -.3]]), torch.tensor([.305]))
    commands.reset(torch.arange(4), torch.zeros(4), torch.zeros(4, dtype=torch.bool), rng)
    assert commands._final["keys"][int(commands._final["terrain_key_ids"][0])] == "plane_for_rm"
    assert not commands.axis_reset_mask()[0] and not commands.air_birth_allowed()[0]
    assert not commands.ground_birth_allowed()[0]
    assert commands.height[0].item() == pytest.approx(.305)
    tick(commands, rng, origins)
    assert torch.equal(commands.command3[0], torch.tensor([.5, 0., -.3]))
    assert commands.metrics()["/source_command/fixed_training_rehearsal_fraction"] == .25


def test_source_initialization_and_command_hook_order_are_frozen_and_not_unused_helpers():
    path = REFERENCE / "v13_env.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    initialize = ast.unparse(methods["_initialize_terrain_command_state"])
    assert initialize.index("_force_resample_commands") < initialize.index("_sample_height_command")
    assert "_sync_command_generator_command" in initialize
    base = next(node for node in ast.parse((REFERENCE / "base_env.py").read_text()).body if isinstance(node, ast.ClassDef))
    base_methods = {node.name: node for node in base.body if isinstance(node, ast.FunctionDef)}
    force = ast.unparse(base_methods["_force_resample_commands"])
    assert "self.command_counter = self.command_generator.command_counter.clone()" in force
    update = ast.unparse(base_methods["_on_command_updated"])
    assert update.index("_before_state_machine_command_updated") < update.index("_apply_predefined_reset_ground_command_override") < update.index("_after_state_machine_command_updated")
    after = ast.unparse(methods["_on_command_updated"])
    assert after.index("super()._on_command_updated") < after.index("_sync_command_generator_command")


def test_full_restart_restores_profile_overrides_cache_clock_and_caller_rng():
    commands, origins = owner(4, columns=[0, 10, 12, 4])
    rng = torch.Generator().manual_seed(59)
    commands.reset(torch.arange(4), torch.zeros(4), torch.zeros(4, dtype=torch.bool), rng)
    tick(commands, rng, origins)
    state = commands.state_dict(rng)
    assert state["schema"] == STATE_VERSION
    restored, _ = owner(4, columns=[0, 10, 12, 4])
    other_rng = torch.Generator().manual_seed(71)
    restored.load_state_dict(state, other_rng)
    assert torch.equal(rng.get_state(), other_rng.get_state())
    for value in (commands, restored):
        value.time_left.zero_()
    tick(commands, rng, origins)
    tick(restored, other_rng, origins)
    assert torch.equal(commands.command3, restored.command3)
    assert torch.equal(commands.height, restored.height)
    assert torch.equal(rng.get_state(), other_rng.get_state())
    assert commands.state_dict(rng)["source"]["clocks"] == restored.state_dict(other_rng)["source"]["clocks"]
    old = ScutFlatCommands(4, "cpu")
    with pytest.raises(ValueError, match="incompatible"):
        restored.load_state_dict(old.state_dict(other_rng), other_rng)


def test_queries_consume_no_rng_or_profile_state_and_numeric_position_retains_old_profile():
    commands, origins = owner(2, columns=[0, 10])
    rng = torch.Generator().manual_seed(73)
    commands.reset([0, 1], torch.zeros(2), torch.zeros(2, dtype=torch.bool), rng)
    old = commands._final["terrain_key_ids"].clone()
    before = torch.get_rng_state()
    values = origins.clone()
    values[0, 1] = torch.nan
    keys, _ = commands.profile_indices(values)
    commands.axis_reset_mask()
    commands.air_birth_allowed()
    assert torch.equal(keys, old) and torch.equal(commands._final["terrain_key_ids"], old)
    assert torch.equal(torch.get_rng_state(), before)


def test_sampling_uses_only_explicit_caller_generator():
    before = torch.get_rng_state()
    commands, origins = owner(30)
    rng = torch.Generator().manual_seed(75)
    commands.reset(torch.arange(30), torch.zeros(30), torch.zeros(30, dtype=torch.bool), rng)
    commands.episode_physics_ticks.fill_(996)
    commands.time_left.zero_()
    tick(commands, rng, origins)
    assert torch.equal(torch.get_rng_state(), before)


def test_bad_restart_has_no_partial_owner_or_rng_mutation():
    commands, origins = owner(2, columns=[0, 10])
    rng = torch.Generator().manual_seed(79)
    commands.reset([0, 1], torch.zeros(2), torch.zeros(2, dtype=torch.bool), rng)
    before = commands.state_dict(rng)
    bad = deepcopy(before)
    bad["final"]["terrain_profile_ids"][0] = 999
    with pytest.raises(ValueError, match="profiles disagree"):
        commands.load_state_dict(bad, rng)
    after = commands.state_dict(rng)
    assert torch.equal(before["source"]["generator_state"], after["source"]["generator_state"])
    assert all(torch.equal(before["final"][key], after["final"][key]) for key in before["final"])
    assert all(torch.equal(before["source"]["tensors"][key], after["source"]["tensors"][key]) for key in before["source"]["tensors"])
