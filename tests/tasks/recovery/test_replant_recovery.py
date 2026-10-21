"""Bounded replant after a sagittal extension; no physical success claims."""
import math

import pytest
import torch

from wheeled_tasks.chassis.tasks.recovery.recovery import (
    DIRECT_TAKEOVER_VERSION, LOW_POSE_TAKEOVER_VERSION, SIDE_TRANSITION_TAKEOVER_VERSION,
    REPLANT_TAKEOVER_VERSION, RecoveryPhase as P, RecoveryRoute as R,
)
from wheeled_tasks.chassis.application.runtime.recovery_training import validate_direct_takeover
from tests.support.support_direct_recovery_takeover import direct_settings
from tests.tasks.recovery.test_side_transition_recovery import completed_side, step as side_step


def completed_push(version=REPLANT_TAKEOVER_VERSION):
    fsm = completed_side(version)
    fsm.phase[:] = P.THRUST
    fsm.route[:] = R.ORBIT_POSITIVE
    fsm.phase_ticks.zero_()
    fsm.thrust_anchor[:] = fsm.script_targets[:, fsm.LEG_IDS]
    return fsm


def step(fsm, *, height=.15, support=False, pitch=0., q=None):
    return fsm.update(fsm.script_targets.clone() if q is None else q, torch.zeros(1, 6),
        torch.tensor([[math.sin(pitch), 0., -math.cos(pitch)]]), torch.zeros(1, 3),
        torch.tensor([height]), torch.tensor([support]), torch.tensor([False]),
        capture_ready=torch.tensor([False]), valid_sensor_history_s=torch.tensor([.5]))


def test_replant_is_explicit_and_inherits_side_and_low_pose_rules():
    settings = direct_settings({})
    settings["direct_takeover"]["version"] = REPLANT_TAKEOVER_VERSION
    assert validate_direct_takeover(settings)
    settings["direct_takeover"]["capture_max_wheel_velocity_rad_s"] = 20.
    with pytest.raises(ValueError, match="explicit V7 capture settings"):
        validate_direct_takeover(settings)
    fsm = completed_side(REPLANT_TAKEOVER_VERSION)
    assert fsm.low_pose_capture and fsm.side_transition and fsm.replant_capture
    for _ in range(150):
        side_step(fsm)
    assert fsm.phase.item() == P.SELECT and fsm.reroute_count.item() == 1


def test_completed_push_reselects_without_target_jump_or_clock_reset():
    fsm = completed_push()
    for _ in range(149):
        step(fsm)
    assert fsm.phase.item() == P.THRUST and fsm.replant_ticks.item() == 149
    before = fsm.script_targets.clone()
    measured = fsm.continuous_q.clone()
    age = fsm.age_ticks.item()
    step(fsm)
    assert fsm.phase.item() == P.SELECT and fsm.reroute_count.item() == 1
    assert fsm.age_ticks.item() == age + 1 and fsm.phase_ticks.item() == 0
    assert torch.equal(fsm.script_targets, before) and torch.equal(fsm.continuous_q, measured)
    assert not fsm.pure_rl.item() and not fsm.motion_released.item()
    step(fsm)
    assert fsm.phase.item() == P.FOLD and fsm.route.item() == R.PLANT
    assert fsm.push_off_required.item()
    assert (fsm.script_targets - before).abs().max() <= .006001
    for a, b in ((0, 1), (3, 4)):
        assert torch.allclose(fsm.script_targets[:, a] - fsm.script_targets[:, b],
                              before[:, a] - before[:, b], atol=1e-6)
    assert not fsm.pure_rl.item()


@pytest.mark.parametrize("reason", ["height", "support", "tilt", "push_off", "route", "unfinished"])
def test_replant_does_not_replace_an_active_or_supported_extension(reason):
    fsm = completed_push()
    kwargs = {}
    if reason == "height":
        kwargs["height"] = .30
    elif reason == "support":
        kwargs["support"] = True
    elif reason == "tilt":
        kwargs["pitch"] = math.radians(20.)
    elif reason == "push_off":
        fsm.push_off_required.fill_(True)
    elif reason == "route":
        fsm.route[:] = R.PLANT
    else:
        fsm.thrust_anchor += 2.
    for _ in range(151):
        step(fsm, **kwargs)
    assert fsm.phase.item() == P.THRUST and fsm.reroute_count.item() == 0
    assert fsm.replant_ticks.item() == 0


def test_replant_dwell_resets_and_supported_capture_keeps_priority():
    fsm = completed_push()
    for _ in range(149):
        step(fsm)
    step(fsm, height=.30)
    assert fsm.replant_ticks.item() == 0
    for _ in range(149):
        step(fsm)
    step(fsm, height=.25, support=True)
    assert fsm.phase.item() == P.CAPTURE and fsm.reroute_count.item() == 0
    assert not fsm.pure_rl.item()


@pytest.mark.parametrize("version", [DIRECT_TAKEOVER_VERSION, LOW_POSE_TAKEOVER_VERSION,
                                     SIDE_TRANSITION_TAKEOVER_VERSION])
def test_previous_versions_do_not_replant(version):
    fsm = completed_push(version)
    for _ in range(151):
        step(fsm)
    assert fsm.phase.item() == P.THRUST and fsm.reroute_count.item() == 0


def test_shared_allowance_and_whole_script_timeout_still_bound_recovery():
    fsm = completed_push()
    fsm.reroute_count[:] = 1
    for _ in range(151):
        step(fsm)
    assert fsm.phase.item() == P.THRUST and fsm.reroute_count.item() == 1
    fsm = completed_push()
    fsm.age_ticks[:] = 7850
    for _ in range(150):
        step(fsm)
    assert fsm.failed.item() and fsm.failure_code.item() == 6
    assert fsm.age_ticks.item() == 8000
    fsm.reset(torch.tensor([0]), fsm.script_targets.clone(), True)
    assert fsm.replant_ticks.item() == 0 and fsm.reroute_count.item() == 0


def test_invalid_sensing_cannot_replant():
    fsm = completed_push()
    for _ in range(149):
        step(fsm)
    step(fsm, q=torch.full((1, 6), math.nan))
    assert fsm.failed.item() and fsm.failure_code.item() == 1
    assert fsm.reroute_count.item() == 0
