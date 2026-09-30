"""Command-clock tests; they do not claim simulator dynamics were replayed."""
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.skill_commands import SkillCommands


def environment(count=128, bounds=(0., 8.), *, exact=False):
    spec = {"kind": "start_stop", "command": [.5, 0., .305], "mode": 1,
            "segment_seconds": 4., "episode_seconds": 16., "sample_amplitude": False}
    cfg = {"skill_specs": {"stop": spec}, "evaluation_exact_cases": exact,
           "evaluation": {"cases": [{"name": "stop", "command": spec["command"], "skill": spec}]},
           "new_asset_curriculum": {"enabled": True, "stage": "S1"}}
    if bounds is not None:
        cfg["new_asset_curriculum"]["start_stop_start_delay_seconds"] = list(bounds)
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1.
    generator = torch.Generator().manual_seed(290927)
    e = SimpleNamespace(cfg=cfg, num_envs=count, device="cpu", scene_groups=["stop"] * count,
        commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
        mode=torch.zeros(count, dtype=torch.long), policy_dt=.02,
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        episode_limits=torch.full((count,), 800, dtype=torch.long),
        command_clock=torch.zeros(count), height_clock=torch.zeros(count), push_clock=torch.zeros(count),
        push_enabled=torch.zeros(count, dtype=torch.bool),
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))),
        random=lambda n: torch.rand(n, generator=generator))
    e.skills = SkillCommands(e)
    e.skills.sample(torch.arange(count))
    return e


def test_real_wait_precedes_every_acceleration_and_brake_without_clock_jump():
    e = environment(3, (2., 2.))
    s = e.skills
    assert not e.episode_length_buf.any()
    assert not e.command_target.any()
    assert e.episode_limits.tolist() == [900] * 3
    assert s.start_stop_timing_state() == {"enabled": True, "start_delay_seconds": [2., 2.], "policy_dt": .02}
    velocity = torch.zeros(3, 3)
    # The first braking event is after two real waiting seconds and four real
    # commanded motion seconds. No lane is labelled settled at reset.
    for time, command, waiting, braking, stopped in (
            (0., 0., True, False, False), (1.98, 0., True, False, False),
            (2., .5, False, False, False), (5.98, .5, False, False, False),
            (6., 0., False, True, False), (7.02, 0., False, False, True),
            (10., -.5, False, False, False), (14., 0., False, True, False)):
        e.episode_length_buf[:] = round(time / e.policy_dt)
        s.update()
        data = s.diagnostics(velocity)
        assert e.command_target[:, 0].tolist() == [command] * 3
        assert data["start_stop_waiting"].tolist() == [waiting] * 3
        assert data["braking_window"].tolist() == [braking] * 3
        assert data["settled_stop_window"].tolist() == [stopped] * 3


def test_command_resampling_cannot_restart_wait_or_turn_a_stop_back_into_motion():
    e = environment()
    s = e.skills
    delay, limits = s.start_stop_delay.clone(), e.episode_limits.clone()
    ids = torch.arange(e.num_envs)
    for elapsed in (torch.zeros_like(delay), delay + 4.2):
        e.episode_length_buf[:] = (elapsed / e.policy_dt).round().long()
        s.sample(ids, reset_height=False)
        torch.testing.assert_close(s.start_stop_delay, delay)
        torch.testing.assert_close(e.episode_limits, limits)
        assert not e.command_target.any()
        assert (s.command_base[:, 0] == .5).all()


def test_partial_reset_preserves_other_lane_clock_and_delay():
    e = environment()
    s = e.skills
    old_delay = s.start_stop_delay.clone()
    e.episode_length_buf[:] = 300
    e.episode_length_buf[:5] = 0
    s.sample(torch.arange(5), reset_height=True)
    assert not torch.equal(s.start_stop_delay[:5], old_delay[:5])
    torch.testing.assert_close(s.start_stop_delay[5:], old_delay[5:])
    assert (e.episode_length_buf[5:] == 300).all()
    expected = 800 + (s.start_stop_delay / .02).ceil().long()
    torch.testing.assert_close(e.episode_limits, expected)


def test_old_training_and_exact_evaluation_keep_original_commands_and_timing():
    for e in (environment(bounds=None), environment(exact=True)):
        assert not e.skills.start_stop_delay.any()
        assert e.skills.start_stop_delay_bounds is None
        assert (e.episode_limits == 800).all()
        for time, target in ((0., .5), (4., 0.), (8., -.5), (12., 0.)):
            e.episode_length_buf[:] = round(time / .02)
            e.skills.update()
            assert (e.command_target[:, 0] == target).all()


@pytest.mark.parametrize("bounds", [(-1., 2.), (3., 2.), (0., float("nan")), (0., float("inf"))])
def test_invalid_delay_is_rejected(bounds):
    with pytest.raises(ValueError, match="startup delay"):
        environment(bounds=bounds)


@pytest.mark.parametrize("fault", ["time_jump", "short_horizon", "motion_during_wait", "actual_motion_during_wait", "sampling_disabled"])
def test_runtime_probe_rejects_missing_real_wait_or_horizon(fault):
    e = environment()
    if fault == "time_jump":
        e.episode_length_buf[:] = 100
    elif fault == "short_horizon":
        e.episode_limits[:] = 800
    elif fault == "motion_during_wait":
        e.command_target[:, 0] = .5
    elif fault == "actual_motion_during_wait":
        e.commands[:, 0] = .5
    else:
        e.skills.start_stop_delay.zero_()
        e.episode_limits[:] = 800
    with pytest.raises(ValueError, match="real wait"):
        e.skills.start_stop_timing_state()
