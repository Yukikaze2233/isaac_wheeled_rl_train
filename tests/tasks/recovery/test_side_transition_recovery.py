"""Bounded route transitions on prescribed sensing, not physical recovery evidence."""
from copy import deepcopy
import math

import pytest
import torch

from wheeled_tasks.chassis.tasks.recovery.recovery import (
    DIRECT_TAKEOVER_VERSION, LOW_POSE_TAKEOVER_VERSION, SIDE_TRANSITION_TAKEOVER_VERSION,
    RecoveryFSM, RecoveryPhase as P, RecoveryRoute as R,
)
from wheeled_tasks.chassis.application.runtime.recovery_training import validate_direct_takeover
from tests.support.support_direct_recovery_takeover import direct_settings


def completed_side(version=SIDE_TRANSITION_TAKEOVER_VERSION, max_reroutes=1):
    pose = torch.tensor([.3, .4, 0., -.3, -.4, 0.])
    pose[:2] += math.tau
    pose[3:5] -= math.tau
    config = {"takeover_version": version, "dt": .001, "blend_seconds": 0.,
              "prepare_speed_rad_s": 6., "orbit_speed_rad_s": 6., "max_reroutes": max_reroutes}
    fsm = RecoveryFSM(1, "cpu", config, pose, pose, pose, pose, [1., 1., -1., -1.],
                      shell_vertices=torch.tensor([[0., 0., -.17]]))
    fsm.reset(torch.tensor([0]), pose[None], True)
    fsm.phase[:] = P.SIDE
    fsm.route[:] = R.SIDE
    fsm.side_start[:] = pose[list(fsm.LEG_IDS)]
    fsm.side_direction[:] = torch.tensor([1., 1., 0., 0.])
    fsm.script_targets[:, fsm.LEG_IDS] += fsm.side_direction * (math.pi / 2) / fsm.axis_signs
    fsm.continuous_q.copy_(fsm.script_targets)
    fsm.last_raw_q.copy_(fsm.script_targets)
    fsm.phase_ticks[:] = math.ceil((math.pi / 2) / 6. / fsm.dt)
    fsm.age_ticks[:] = 600
    return fsm


def step(fsm, *, pitch=math.radians(150.), support=False, gravity=None, q=None):
    if gravity is None:
        gravity = [math.sin(pitch), 0., -math.cos(pitch)]
    q = fsm.script_targets.clone() if q is None else q
    return fsm.update(q, torch.zeros(1, 6), torch.tensor([gravity]), torch.zeros(1, 3),
        torch.tensor([.15]), torch.tensor([support]), torch.tensor([False]),
        capture_ready=torch.tensor([False]), valid_sensor_history_s=torch.tensor([.5]))


def test_candidate_requires_explicit_version_and_keeps_fixed_capture_gates():
    settings = direct_settings({})
    assert settings["direct_takeover"]["version"] == DIRECT_TAKEOVER_VERSION
    settings["direct_takeover"]["version"] = SIDE_TRANSITION_TAKEOVER_VERSION
    assert validate_direct_takeover(settings)
    changed = deepcopy(settings)
    changed["direct_takeover"]["capture_max_tilt_rad"] = math.pi / 2
    with pytest.raises(ValueError, match="explicit V7 capture settings"):
        validate_direct_takeover(changed)
    fsm = completed_side()
    assert fsm.direct_takeover and fsm.low_pose_capture and fsm.side_transition


def test_completed_reference_and_continuous_dwell_are_both_required():
    fsm = completed_side()
    fsm.script_targets[:, fsm.LEG_IDS] = fsm.side_start
    for _ in range(200):
        step(fsm)
    assert fsm.side_exit_ticks.item() == 0 and fsm.phase.item() == P.SIDE
    for _ in range(100):
        step(fsm)
        if fsm.side_exit_ticks.item() != 0:
            break
    assert fsm.side_exit_ticks.item() == 1
    for _ in range(148):
        step(fsm)
    assert fsm.side_exit_ticks.item() == 149 and fsm.reroute_count.item() == 0
    step(fsm, support=True)
    assert fsm.side_exit_ticks.item() == 0
    for _ in range(149):
        step(fsm)
    assert fsm.phase.item() == P.SIDE
    previous = fsm.script_targets.clone()
    age = fsm.age_ticks.item()
    step(fsm)
    assert fsm.phase.item() == P.SELECT and fsm.reroute_count.item() == 1
    assert fsm.age_ticks.item() == age + 1 and fsm.phase_ticks.item() == 0
    assert torch.equal(fsm.script_targets, previous)
    assert not fsm.pure_rl.item() and not fsm.motion_released.item()


@pytest.mark.parametrize("pitch,expected", [(3.12, R.ORBIT_POSITIVE), (-3.12, R.ORBIT_NEGATIVE)])
def test_reselection_preserves_winding_and_rate_limit_across_inversion(pitch, expected):
    fsm = completed_side()
    fsm.pitch_initialized[:] = True
    fsm.last_pitch[:] = -pitch
    fsm.continuous_pitch[:] = -pitch + 2 * math.tau
    previous_pitch = fsm.continuous_pitch.clone()
    for _ in range(150):
        step(fsm, pitch=pitch)
    assert fsm.phase.item() == P.SELECT
    assert abs(float(fsm.continuous_pitch - previous_pitch)) < .05
    before = fsm.script_targets.clone()
    measured = fsm.continuous_q.clone()
    step(fsm, pitch=pitch, q=measured)
    assert fsm.phase.item() == P.FOLD and fsm.route.item() == expected
    assert torch.equal(fsm.continuous_q, measured)
    assert (fsm.script_targets - before).abs().max() <= .006001
    # The 2*pi winding is shared by both motors, not independently clipped.
    for a, b in ((0, 1), (3, 4)):
        assert torch.allclose(fsm.script_targets[:, a] - fsm.script_targets[:, b],
                              before[:, a] - before[:, b], atol=1e-6)
    for _ in range(400):
        before = fsm.script_targets.clone()
        step(fsm, pitch=pitch)
        assert (fsm.script_targets - before).abs().max() <= .006001
        if fsm.phase.item() == P.ORBIT:
            break
    assert fsm.phase.item() == P.ORBIT and fsm.reroute_count.item() == 1
    assert not fsm.pure_rl.item()


@pytest.mark.parametrize("version", [DIRECT_TAKEOVER_VERSION, LOW_POSE_TAKEOVER_VERSION])
def test_existing_versions_keep_side_timeout(version):
    fsm = completed_side(version)
    for _ in range(151):
        step(fsm)
    assert fsm.phase.item() == P.SIDE and fsm.reroute_count.item() == 0
    fsm.phase_ticks[:] = 2499
    step(fsm)
    assert fsm.failed.item() and fsm.failure_code.item() == 2


def test_supported_side_domain_and_capture_paths_do_not_reselect():
    fsm = completed_side()
    for _ in range(151):
        step(fsm, gravity=[.1, .9, math.sqrt(.18)])
    assert fsm.side_exit_ticks.item() == 0 and fsm.phase.item() == P.SIDE
    for _ in range(149):
        step(fsm)
    step(fsm, pitch=math.radians(64.))
    assert fsm.phase.item() == P.CAPTURE and fsm.reroute_count.item() == 0
    assert not fsm.pure_rl.item()
    # Existing post-capture support-loss failure is deliberately unchanged.
    for _ in range(150):
        step(fsm)
    assert fsm.failed.item() and fsm.failure_code.item() == 2


@pytest.mark.parametrize("max_reroutes,used", [(0, 0), (1, 1)])
def test_shared_reroute_limit_and_whole_script_timeout_cannot_be_extended(max_reroutes, used):
    fsm = completed_side(max_reroutes=max_reroutes)
    fsm.reroute_count[:] = used
    fsm.phase_ticks[:] = 2349
    for _ in range(151):
        step(fsm)
    assert fsm.failed.item() and fsm.failure_code.item() == 2
    assert fsm.reroute_count.item() == used
    fsm = completed_side()
    fsm.age_ticks[:] = 7850
    for _ in range(150):
        step(fsm)
    assert fsm.failed.item() and fsm.failure_code.item() == 6
    assert fsm.age_ticks.item() == 8000
    fsm.reset(torch.tensor([0]), fsm.script_targets.clone(), True)
    assert fsm.side_exit_ticks.item() == 0 and fsm.reroute_count.item() == 0


def test_invalid_sensing_cannot_trigger_reselection():
    fsm = completed_side()
    for _ in range(149):
        step(fsm)
    step(fsm, q=torch.full((1, 6), math.nan))
    assert fsm.failed.item() and fsm.failure_code.item() == 1
    assert fsm.reroute_count.item() == 0 and not fsm.pure_rl.item()
