"""CPU checks of the saved-source map and native sampling semantics."""
from copy import deepcopy
from pathlib import Path
import math
import shutil

import pytest
import torch

from wheeled_tasks.chassis import scut_reuse_curriculum as reuse
from wheeled_tasks.chassis.integrated_sampling import load_sources
from wheeled_tasks.chassis.skill_commands import sample_planar_command, validate_planar_sampling

ROOT = Path(__file__).resolve().parents[1]
SCUT = ROOT / "references/scut_curriculum_b8ff79f3"


def test_actual_saved_files_are_pinned_and_enabled():
    data = reuse.read_sources(SCUT)
    for source in data.values():
        commands = source["commands"]
        assert commands["resampling_time_range"] == [5., 15.]
        assert commands["ranges"]["lin_vel_x"] == [-2.7, 2.7]
        assert commands["ranges"]["ang_vel_z"] == [-math.tau, math.tau]
        assert commands["rel_standing_envs"] == .1
        assert commands["heading_command"] is True
        assert source["nonstandard_birth"]["use_predefined_leg_random_start"] is True
        assert source["nonstandard_birth"]["predefined_reset_ground"]["zero_torque_time_s"] == .2
        assert source["agent"]["max_iterations"] == 20000
    assert data["rough_dash"]["agent"]["policy"]["actor_hidden_dims"] == [256, 128, 64]
    assert data["rough_rotation_stair"]["agent"]["policy"]["actor_hidden_dims"] == [128, 64, 32]
    assert data["rough_dash"]["commands"]["special_modes"][-1]["ranges"]["lin_vel_x"] == [0., 0.]


def test_whole_saved_file_change_is_rejected(tmp_path):
    for profile, (run, _, _) in reuse.SOURCES.items():
        for kind in ("env", "agent"):
            relative = Path("pretrained/26_infantry") / profile / run / "params" / (kind + ".yaml")
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SCUT / relative, target)
    env = next(tmp_path.rglob("env.yaml"))
    with env.open("ab") as stream:
        stream.write(b"\n# changed saved source\n")
    with pytest.raises(ValueError, match="SHA"):
        reuse.read_sources(tmp_path)


@pytest.mark.parametrize("domain", ["speed", "terrain", "jump", "stepjump"])
def test_native_specials_physics_quotas_and_small_stairs_are_exact(domain):
    config = load_sources(ROOT)[domain]
    before = deepcopy(config)
    result = reuse.apply(config, SCUT)
    mapped = result.pop("scut_reuse_candidate")["mapped_groups"]
    assert mapped
    for name, spec in result["skill_specs"].items():
        if name not in mapped:
            assert spec == before["skill_specs"][name]
    expected = deepcopy(before)
    expected["skill_specs"] = result["skill_specs"]
    assert result == expected
    assert config == before
    for name in ("stand_230mm", "stand_430mm", "stand_continuous"):
        assert result["skill_specs"][name] == before["skill_specs"][name]
        assert result["skill_specs"][name]["command"][:2] == [0., 0.]
    if domain == "terrain":
        for name in before["skill_specs"]:
            if name.startswith("small_stairs_"):
                assert result["skill_specs"][name] == before["skill_specs"][name]


@pytest.mark.parametrize("role", ["fixed_eval", "playback", "qualification", "preflight", "rollout_validation"])
def test_nontraining_roles_do_not_read_reference_or_change_contract(role):
    config = load_sources(ROOT)["speed"]
    assert reuse.apply(config, "/not/a/repository", role=role) == config


def test_real_native_normal_sampler_covers_zero_low_high_and_both_directions():
    config = reuse.apply(load_sources(ROOT)["speed"], SCUT)
    spec = config["skill_specs"]["forward_05"]
    validate_planar_sampling(spec)
    assert spec["height_sampling"]["range_m"] == [.23, .42]
    assert spec["command_resampling_scope"] == "skill"
    assert spec["command_resampling_seconds"] == [5., 15.]
    assert spec["standing_command_probability"] == .1
    generator = torch.Generator().manual_seed(290930)
    commands = torch.tensor(spec["command"]).expand(10000, -1).clone()
    sample_planar_command(commands, spec, lambda n: torch.rand(n, generator=generator))
    assert float(commands[:, 0].min()) >= -2.7
    assert float(commands[:, 0].max()) <= 2.7
    assert ((commands[:, 0].abs() < .1) & (commands[:, 1].abs() < .2)).any()
    assert (commands[:, 0] > 2.).any() and (commands[:, 0] < -2.).any()
    assert (commands[:, 1] > 5.).any() and (commands[:, 1] < -5.).any()
    zeros = (commands[:, :2] == 0.).all(dim=1).sum().item()
    assert 800 < zeros < 1200
    exact = torch.tensor(spec["command"]).expand(3, -1).clone()
    assert torch.equal(sample_planar_command(exact.clone(), spec,
        lambda _: (_ for _ in ()).throw(AssertionError("No RNG during fixed evaluation")), evaluation=True), exact)


def test_explicit_terrain_override_selects_source_band_without_convex_hull():
    config = load_sources(ROOT)["terrain"]
    name = "small_stairs_up_005mm"
    binding = {name: dict(override="low_speed_stair_for_rm", vx_interval=0)}
    result = reuse.apply(config, SCUT, terrain_bindings=binding)
    spec = result["skill_specs"][name]
    assert spec["planar_command_sampling"]["vx_range_m_s"] == [.5, 1.5]
    assert spec["planar_command_sampling"]["yaw_range_rad_s"] == [-.1, .1]
    assert spec["height_sampling"]["range_m"] == [.23, .34]
    assert spec["terrain_limits"] == config["skill_specs"][name]["terrain_limits"]
    with pytest.raises(ValueError, match="explicit interval"):
        reuse.apply(config, SCUT, terrain_bindings={name: dict(override="low_speed_stair_for_rm")})
    with pytest.raises(ValueError, match="dedicated or zero-command"):
        reuse.apply(config, SCUT, terrain_bindings={"stand_230mm": dict(override="tiny_step")})


def test_reference_height_distribution_keeps_native_reset_transition_split():
    from test_skill_command_duration import environment
    from wheeled_tasks.chassis.skill_commands import SkillCommands

    config = load_sources(ROOT)["speed"]
    native = config["skill_specs"]["forward_05__height_train"]
    assert native["height_sampling"]["start_at_target_fraction"] == .5
    spec = reuse.apply(config, SCUT)["skill_specs"]["forward_05__height_train"]
    assert spec["height_sampling"]["start_at_target_fraction"] == .5
    assert spec["height_transition_seconds"] == native["height_transition_seconds"]
    env = environment(spec, count=4096)
    skills = SkillCommands(env)
    skills.sample(torch.arange(env.num_envs))
    at_target = skills.height_phase_offset.gt(0.)
    assert at_target.float().mean().item() == pytest.approx(.5, abs=.03)
    assert env.commands[~at_target, 2].eq(.305).all()
    assert (skills.height_target < .26).any() and (skills.height_target > .39).any()
    torch.testing.assert_close(env.commands[at_target, 2], skills.height_target[at_target])


def test_insufficient_standing_exposure_is_not_silently_invented():
    config = load_sources(ROOT)["speed"]
    for row in config["scene_groups"]:
        if config["behavior_pool_membership"][row["name"]] in reuse.STANDING_POOLS:
            row["fraction"] = 0.
    with pytest.raises(ValueError, match="standing floor"):
        reuse.apply(config, SCUT)


@pytest.mark.parametrize("tag", ["mode", "zero_command"])
def test_zero_command_alias_cannot_become_a_moving_normal_group(tag):
    config = load_sources(ROOT)["speed"]
    spec = config["skill_specs"]["forward_05"]
    spec["command"][:2] = [0., 0.]
    if tag == "mode":
        spec["mode"] = 0
    else:
        spec["zero_command"] = True
    result = reuse.apply(config, SCUT)
    assert result["skill_specs"]["forward_05"] == spec
    assert "forward_05" not in result["scut_reuse_candidate"]["mapped_groups"]
