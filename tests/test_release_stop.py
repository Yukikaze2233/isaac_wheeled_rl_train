"""Abrupt release requests retain frozen braking and one long stop per reset."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.release_stop import (
    ReleaseStop, apply_release_stop_contract, release_stop_settings,
    release_stop_runtime_expectation, sample_release_stop_command,
    release_stop_surfaces,
)
from wheeled_tasks.chassis.task import terrain_surfaces
from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.learning_audit import StationaryWindowAudit

ROOT = Path(__file__).resolve().parents[1]


def environment(count=32, *, domain="speed", axis="combined", continuous=False):
    base = "v6_task_campaign_speed_worker_v10.json" if domain == "speed" else "v6_task_campaign_terrain_worker_v6.json"
    cfg = json.loads((ROOT / "contracts" / base).read_text())
    apply_release_stop_contract(cfg, domain)
    name = "release_stop_" + axis + ("__height_train" if continuous else "")
    spec = cfg["skill_specs"][name]
    cfg = {"skill_specs": {name: spec}, "release_stop": cfg["release_stop"],
        "command_slew": cfg["command_slew"], "motion_limits": cfg["motion_limits"],
        "new_asset_curriculum": {"enabled": True, "stage": "S1"}}
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1.
    generator = torch.Generator().manual_seed(1052026)
    env = NS(cfg=cfg, num_envs=count, device="cpu", scene_groups=[name] * count,
        commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
        mode=torch.zeros(count, dtype=torch.long), policy_dt=.02,
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        episode_limits=torch.zeros(count, dtype=torch.long),
        command_clock=torch.zeros(count), height_clock=torch.zeros(count), push_clock=torch.zeros(count),
        push_enabled=torch.zeros(count, dtype=torch.bool),
        robot=NS(data=NS(root_link_pose_w=NS(torch=pose))),
        random=lambda n: torch.rand(n, generator=generator))
    env.skills = SkillCommands(env)
    env.skills.sample(torch.arange(count))
    # Production resample_commands restores applied command zero at reset.
    env.commands[:, :2] = 0.
    return env


def apply_slew(env):
    before = env.commands[:, :2].clone()
    rates = torch.tensor([1.5, 4.])
    change = (env.command_target - before).clamp(-rates * env.policy_dt, rates * env.policy_dt)
    env.commands[:, :2] += change
    return before, change


@pytest.mark.parametrize("domain,limit", [("speed", (3., 6.)), ("terrain", (1.5, 2.))])
def test_contract_pool_preserves_old_relative_sampling_and_fixed_evaluation(domain, limit):
    filename = "v6_task_campaign_speed_worker_v10.json" if domain == "speed" else "v6_task_campaign_terrain_worker_v6.json"
    original = json.loads((ROOT / "contracts" / filename).read_text())
    changed = apply_release_stop_contract(deepcopy(original), domain)
    assert changed["skill_specs"].keys() >= original["skill_specs"].keys()
    assert all(changed["skill_specs"][name] == spec for name, spec in original["skill_specs"].items())
    assert changed["scene_groups"][:len(original["scene_groups"])] == [
        dict(group, fraction=group["fraction"] * .95) for group in original["scene_groups"]]
    assert sum(group["fraction"] for group in changed["scene_groups"][-6:]) == pytest.approx(.05)
    assert sum(group["fraction"] for group in changed["scene_groups"]) == pytest.approx(1.)
    for group in changed["scene_groups"][-6:]:
        spec = changed["skill_specs"][group["name"]]
        assert spec["release_stop_magnitude_ranges"]["vx_m_s"][-1] == limit[0]
        assert spec["release_stop_magnitude_ranges"]["yaw_rad_s"][-1] == limit[1]
        assert not any(kind in ("stairs", "step_up", "step_down", "jump") for kind in group["terrain"])
    exact = deepcopy(original)
    exact["evaluation_exact_cases"] = True
    apply_release_stop_contract(exact, domain)
    assert exact["scene_groups"] == original["scene_groups"]
    assert exact["behavior_pool_fractions"] == original["behavior_pool_fractions"]
    assert exact["evaluation"] == original["evaluation"]
    assert not exact["release_stop"]["enabled"]
    assert changed["command_slew"] == original["command_slew"]
    assert changed["dense_tracking"] == original["dense_tracking"]


@pytest.mark.parametrize("axis", ["translate", "yaw", "combined"])
def test_each_axis_has_unbiased_signs_and_no_hidden_motion_on_inactive_axes(axis):
    env = environment(4096, axis=axis)
    command = env.skills.command_base
    if axis == "yaw":
        assert command[:, 0].eq(0.).all()
    else:
        assert .45 < (command[:, 0] > 0).float().mean() < .55
    if axis == "translate":
        assert command[:, 1].eq(0.).all()
    else:
        assert .45 < (command[:, 1] > 0).float().mean() < .55
    assert (command[:, 0].abs() <= 3.).all() and (command[:, 1].abs() <= 6.).all()
    if axis == "combined":
        assert ((command[:, 0] * command[:, 1]).abs() <= 2.060001).all()
        for signx, signy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            assert .2 < ((command[:, 0] * signx > 0) & (command[:, 1] * signy > 0)).float().mean() < .3


def test_per_lane_timing_partial_resets_and_resampling_never_restart_or_reverse_drive():
    env = environment(128)
    module = env.skills.release_stop
    assert module.wait.unique().numel() > 100 and module.drive.unique().numel() > 100
    assert (module.wait >= 0).all() and (module.wait <= 4).all()
    assert (module.drive >= 4).all() and (module.drive <= 8).all()
    assert (module.requested_dwell >= 12).all() and (module.requested_dwell <= 16).all()
    snapshots = [value.clone() for value in (module.wait, module.release_at, module.end_at, module.drive_command)]
    env.episode_length_buf[:] = ((module.release_at + .1) / .02).ceil().long()
    env.commands[:, :2] = torch.tensor([.2, -.3])
    applied = env.commands.clone()
    env.skills.sample(torch.arange(128), reset_height=False)
    assert env.command_target.eq(0.).all()
    assert torch.equal(applied, env.commands)
    assert env.command_clock.eq(1e9).all()
    assert (env.episode_limits == 1600).all()
    for value, snapshot in zip((module.wait, module.release_at, module.end_at, module.drive_command), snapshots):
        assert torch.equal(value, snapshot)
    env.episode_length_buf[:3] = 0
    env.skills.sample(torch.arange(3), reset_height=True)
    assert not torch.equal(module.wait[:3], snapshots[0][:3])
    for value, snapshot in zip((module.wait, module.release_at, module.end_at, module.drive_command), snapshots):
        assert torch.equal(value[3:], snapshot[3:])
    ids = torch.arange(128)
    assert module.target(ids, module.end_at + 100.)[:, :2].eq(0.).all()


@pytest.mark.parametrize("velocity,yaw", [(3., 0.), (-3., 0.), (0., 6.), (0., -6.), (1., 2.)])
def test_drive_accelerates_release_is_exact_zero_request_actual_slew_and_10s_quiet(velocity, yaw):
    env = environment(1)
    module = env.skills.release_stop
    module.wait[:] = 1.
    module.drive[:] = 4.
    module.release_at[:] = 5.
    module.drive_command[:] = torch.tensor([velocity, yaw, .305])
    module.brake_budget[:] = max(abs(velocity) / 1.5, abs(yaw) / 4.)
    module.end_at[:] = module.release_at + module.brake_budget + 12.
    module.dwell[:] = module.end_at - module.release_at
    audit = StationaryWindowAudit(1, "cpu", .02)
    reached = False
    released = False
    last = torch.zeros(1, 2)
    for tick in range(int((module.end_at / .02).ceil().item()) + 1):
        env.episode_length_buf[:] = tick
        before = env.commands.clone()
        env.skills.update()
        assert torch.equal(env.commands[:, :2], before[:, :2])  # Only target changes.
        if tick * .02 >= 5.:
            assert env.command_target.eq(0.).all()
            released = True
        prior, change = apply_slew(env)
        assert (change[:, 0].abs() <= .030001).all()
        assert (change[:, 1].abs() <= .080001).all()
        if 3. <= tick * .02 < 5.:
            torch.testing.assert_close(env.commands[0, :2], torch.tensor([velocity, yaw]))
            reached = True
        if released:
            assert (env.commands[:, :2].abs() <= last.abs() + 1e-6).all()
        last = env.commands[:, :2].clone()
        if tick * .02 >= 5.:
            audit.observe(dict(commands=env.commands.clone(), done=torch.zeros(1, dtype=torch.bool),
                height=env.commands[:, 2], position=torch.zeros(1, 3), planar_speed_world=torch.zeros(1)))
    assert reached and released
    assert audit.counts[0] >= 1


@pytest.mark.parametrize("fault", ["moving", "yaw", "height", "tilt", "command"])
def test_settled_requires_real_stop_and_pose_not_just_expired_nominal_braking(fault):
    env = environment(1)
    module = env.skills.release_stop
    elapsed = module.release_at + module.brake_budget + 2.
    env.episode_length_buf[:] = (elapsed / .02).ceil().long()
    env.commands[:, :2] = 0.
    velocity, omega, error, gravity = torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1), torch.tensor([[0., 0., -1.]])
    clean = env.skills.diagnostics(velocity, omega=omega, height_error=error, gravity=gravity)
    assert clean["release_requested"].item() and clean["release_stop_settled"].item()
    if fault == "moving": velocity[0, 1] = .031
    if fault == "yaw": omega[0, 2] = .051
    if fault == "height": error[0] = .011
    if fault == "tilt": gravity[:] = torch.tensor([[.2, 0., -.98]])
    if fault == "command": env.commands[0, 0] = .01
    data = env.skills.diagnostics(velocity, omega=omega, height_error=error, gravity=gravity)
    assert data["release_requested"].item() and data["release_stop_braking"].item()
    assert not data["release_stop_settled"].item()
    # Audit the entire requested stationary window, including failed stops.
    assert data["settled_stop_window"].item() == (fault != "command")
    if fault == "moving":
        assert data["settled_stop_speed"].item() > .03  # Failure remains visible to LearningAudit.


def test_height_sampling_and_runtime_settings_snapshot_preserve_frozen_abi():
    env = environment(1024, continuous=True)
    assert env.skills.height_target.unique().numel() > 100
    assert (env.skills.height_target >= .23).all() and (env.skills.height_target <= .43).all()
    assert env.commands.shape == (1024, 3)
    state = env.skills.release_stop.runtime_state()
    assert state == release_stop_runtime_expectation(env.cfg["release_stop"], .02)
    state["settings"]["drive_seconds"][0] = 99.
    assert env.cfg["release_stop"]["drive_seconds"] == [4., 8.]


def test_global_resampling_rule_cannot_restart_release_or_change_declared_horizon():
    env = environment(128)
    env.skills.schedule.cfg["command_resampling_seconds"] = [1., 8.]
    env.skills.schedule.cfg["long_command_fraction"] = .2
    env.skills.schedule.cfg["long_command_resampling_seconds"] = [8., 15.]
    def forbidden_resample(*args, **kwargs):
        raise AssertionError("Release-stop entered ordinary command-duration sampling")
    env.skills.schedule.command_duration = forbidden_resample
    env.skills.sample(torch.arange(128), reset_height=True)
    assert env.command_clock.eq(1e9).all()
    assert env.episode_limits.eq(1600).all()
    env.episode_length_buf[:] = ((env.skills.release_stop.release_at + .2) / .02).ceil().long()
    env.command_clock[:] = -1.
    env.commands[:, :2] = .3
    applied = env.commands.clone()
    env.skills.sample(torch.arange(128), reset_height=False)
    assert torch.equal(applied, env.commands)
    assert env.command_target.eq(0.).all() and env.command_clock.eq(1e9).all()


def test_visible_command_defines_acceleration_cruise_and_deceleration_phases():
    env = environment(1)
    module = env.skills.release_stop
    module.wait[:] = 1.
    module.release_at[:] = 5.
    module.drive_command[:] = torch.tensor([2., -4., .305])
    env.commands[:, :2] = 0.
    velocity = torch.zeros(1, 3)
    data = module.diagnostics(torch.tensor([.5]), env.commands, velocity)
    assert data["release_stop_waiting"].item()
    assert not data["release_stop_accelerating"].item()
    data = module.diagnostics(torch.tensor([1.]), env.commands, velocity)
    assert data["release_stop_accelerating"].item() and not data["release_stop_cruising"].item()
    env.commands[:, :2] = module.drive_command[:, :2]
    data = module.diagnostics(torch.tensor([3.]), env.commands, velocity)
    assert data["release_stop_cruising"].item() and not data["release_stop_accelerating"].item()
    data = module.diagnostics(torch.tensor([5.]), env.commands, velocity)
    assert data["release_requested"].item() and data["release_stop_decelerating"].item()
    env.commands[:, :2] = 0.
    data = module.diagnostics(torch.tensor([8.]), env.commands, velocity)
    assert not data["release_stop_decelerating"].item()


@pytest.mark.parametrize("kind", ["rough", "material", "slope"])
@pytest.mark.parametrize("level,index", [(0., 0), (.5, 1), (1., 2), (1., 3)])
def test_repeated_release_terrain_extent_and_original_surface_geometry(kind, level, index):
    limits = dict(slope_deg=8., roughness_m=.02)
    base = terrain_surfaces(kind, limits, level, index)
    frozen_base = deepcopy(base)
    settings = release_stop_settings()
    repeated = release_stop_surfaces(kind, limits, level, index, settings)
    assert base == frozen_base
    assert repeated[0].x0 == -20. and repeated[-1].x1 == 20.
    assert len(repeated) == 5 * len(base)
    assert all(left.x1 == right.x0 for left, right in zip(repeated, repeated[1:]))
    rise = base[-1].height(base[-1].x1) - base[0].height(base[0].x0)
    for tile, offset in enumerate(range(-2, 3)):
        rows = repeated[tile * len(base):(tile + 1) * len(base)]
        for row, source in zip(rows, base):
            assert row.x0 == source.x0 + offset * 8.
            assert row.x1 == source.x1 + offset * 8.
            assert row.slope == source.slope and row.cross_slope == source.cross_slope
            assert row.friction == source.friction
            assert row.z0 == pytest.approx(source.z0 + (offset * rise if kind == "slope" else 0.))
        if kind == "rough":
            assert [row.z0 for row in rows] == [row.z0 for row in base]
            assert len(rows) == 21  # One original lead-in plus twenty waves.
    if kind in ("slope", "material"):
        for left, right in zip(repeated, repeated[1:]):
            assert left.height(left.x1) == pytest.approx(right.height(right.x0))
    if kind == "material":
        assert [row.friction for row in repeated] == [row.friction for row in base] * 5
        if level == 1.:
            assert min(row.friction for row in repeated) == pytest.approx(.3)
    assert settings["terrain_width_m"] == 4.


@pytest.mark.parametrize("kind", ["flat", "jump", "step_up", "stairs", "cross_slope"])
def test_repeated_release_terrain_cannot_expand_unrelated_or_flat_mesh_scenes(kind):
    with pytest.raises(ValueError, match="only rough, material and slope"):
        release_stop_surfaces(kind, dict(slope_deg=8., roughness_m=.02), 1., 0, release_stop_settings())
