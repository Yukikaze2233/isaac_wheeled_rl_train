"""Low-pose controller transitions; prescribed sensing is not a physics pass."""
from copy import deepcopy
import math

import pytest
import torch

from wheeled_tasks.chassis.tasks.recovery.recovery import (
    LOW_POSE_TAKEOVER_VERSION, SIDE_TRANSITION_TAKEOVER_VERSION, REPLANT_TAKEOVER_VERSION,
    RecoveryFSM, RecoveryPhase as P,
)
from wheeled_tasks.chassis.application.runtime.recovery_training import RecoveryTraining, validate_direct_takeover
from tests.support.support_direct_recovery_takeover import direct_rig, sensor_inputs
from tests.support.support_recovery_training_integration import rig


@pytest.fixture(params=[LOW_POSE_TAKEOVER_VERSION, SIDE_TRANSITION_TAKEOVER_VERSION, REPLANT_TAKEOVER_VERSION])
def low_rig(direct_rig, tmp_path, request):
    env, previous, q = direct_rig
    settings = deepcopy(previous.settings)
    settings['direct_takeover']['version'] = request.param
    settings['wheel_balance'] = {'enabled': True}
    recovery = RecoveryTraining(env, tmp_path, settings)
    recovery.initial_release.zero_()
    recovery.reset(torch.arange(2))
    return env, recovery, q


def step(recovery, q, height, supported, ready=True, gravity=None):
    inputs = sensor_inputs(2)
    if gravity is not None:
        inputs['gravity'] = gravity
    recovery.fsm.update(q, inputs['dq'], inputs['gravity'], inputs['gyro'],
        torch.as_tensor(height).expand(2).float(), torch.as_tensor(supported).expand(2).bool(),
        torch.ones(2, dtype=torch.bool), capture_ready=torch.full((2,), ready),
        valid_sensor_history_s=torch.full((2,), .06, dtype=torch.float64))


def test_low_pose_selection_authenticates_geometry_and_keeps_capture_limits(low_rig):
    _, recovery, _ = low_rig
    assert validate_direct_takeover(recovery.settings)
    settings = deepcopy(recovery.settings)
    settings['direct_takeover']['capture_max_wheel_velocity_rad_s'] = 20.
    with pytest.raises(ValueError, match='explicit V7 capture settings'):
        validate_direct_takeover(settings)
    args = (1, 'cpu', {'takeover_version': LOW_POSE_TAKEOVER_VERSION, 'dt': .001,
                      'blend_seconds': 0.}, *([torch.zeros(6)] * 4), [1., 1., -1., -1.])
    with pytest.raises(ValueError, match='authenticated shell geometry'):
        RecoveryFSM(*args)


def test_push_off_cannot_transfer_while_low_or_unsupported_and_reset_is_local(low_rig):
    _, recovery, q = low_rig
    fsm = recovery.fsm
    shell_depth = -fsm.shell_vertices[:, 2].min()
    # This narrow band is above the old 0.20m entry floor but needs extension.
    low = float(shell_depth + .028)
    assert .20 < low < .29
    for _ in range(65):
        step(recovery, q, low, True)
    assert fsm.phase[0] == P.THRUST and fsm.push_off_required[0]
    assert not fsm.pure_rl[0]
    # Once moving upward, conditional geometry may end extension even while
    # acceleration makes support unavailable. It must not grant actor control.
    step(recovery, q, shell_depth + .02, False)
    assert fsm.phase[0] == P.CAPTURE and not fsm.push_off_required[0]
    assert not fsm.pure_rl[0]
    step(recovery, q, .25, True)
    assert fsm.pure_rl[0]
    fsm.reset(torch.arange(2), q, True)
    step(recovery, q, low, True)
    assert fsm.push_off_required.all()
    fsm.reset(torch.tensor([0]), q, True)
    assert fsm.push_off_required.tolist() == [False, True]


def test_clearance_veto_does_not_remove_support_or_sensor_capture_gate(low_rig):
    _, recovery, q = low_rig
    fsm = recovery.fsm
    # Bypass only trajectory setup to isolate the final transfer boundary.
    fsm.phase[:] = P.CAPTURE
    fsm.age_ticks[:] = 60
    fsm.push_off_required.zero_()
    angle = math.radians(25.)
    gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]]).repeat(2, 1)
    depth = (fsm.shell_vertices @ gravity[0]).max()
    assert .20 < depth + .014 < .29
    for height, support, ready in ((depth + .014, True, True), (.28, False, True),
                                    (.28, True, False)):
        step(recovery, q, height, support, ready, gravity)
        assert not fsm.pure_rl[0]
    step(recovery, q, .28, True, gravity=gravity)
    assert fsm.pure_rl[0]


def test_wheel_balance_extends_only_to_supported_push_off(low_rig):
    _, recovery, q = low_rig
    recovery.release_finished.fill_(True)
    recovery.fsm.phase[:] = P.THRUST
    recovery.fsm.push_off_required[:] = torch.tensor([True, False])
    gravity = torch.tensor([[.1, 0., -(1. - .1 ** 2) ** .5]]).repeat(2, 1)
    recovery._update_wheel_targets(gravity, torch.zeros(2, 3), torch.zeros_like(q),
                                  torch.ones(2, dtype=torch.bool), recovery.fsm.phase.clone())
    assert recovery.wheel_balance_active.tolist() == [True, False]
    recovery._update_wheel_targets(gravity, torch.zeros(2, 3), torch.zeros_like(q),
                                  torch.zeros(2, dtype=torch.bool), recovery.fsm.phase.clone())
    assert not recovery.wheel_balance_active.any()
