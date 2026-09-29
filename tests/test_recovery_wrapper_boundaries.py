"""Regression oracles for reference winding and physical handover time windows."""
import math
import hashlib
import json

import pytest
import torch

from test_recovery_training_integration import rig
from wheeled_tasks.chassis.recovery import RecoveryPhase
from wheeled_tasks.chassis.recovery_training import RecoveryTraining


def motor_state(env):
    q, dq = env.nominal[:, env.ids], torch.zeros(2, 6)
    env.v5.motor_efforts(q, dq, q[:, [0, 1, 3, 4]], dq[:, [2, 5]])
    return q, dq


def test_static_compensation_cannot_be_reinterpreted_with_a_different_pd_gain(rig, tmp_path):
    env, recovery = rig
    env.v5.settings = dict(env.v5.settings, leg_kp=80.)
    with pytest.raises(ValueError, match="native PD stiffness"):
        RecoveryTraining(env, tmp_path, recovery.settings)


def test_zero_release_also_clears_requested_effort_and_action_history(rig):
    env, recovery = rig
    q, dq = motor_state(env)
    env.v5.requested_motor_effort[:] = 5.
    env.actions[:] = .7
    recovery.torque(q, dq, torch.full_like(q, 5.))
    recovery.effective_action_history()
    assert env.v5.requested_motor_effort[0].eq(0.).all() and env.actions[0].eq(0.).all()
    assert env.v5.requested_motor_effort[1].eq(5.).all() and env.actions[1].eq(.7).all()


def test_orbit_error_over_pi_never_reverses_torque(rig):
    env, recovery = rig
    q, dq = motor_state(env)
    recovery.release_finished[:] = True
    recovery.fsm.phase[0] = RecoveryPhase.ORBIT
    recovery.fsm.script_targets[0, [0, 1, 3, 4]] += torch.tensor([4., 4., -4., -4.])
    effort = recovery.torque(q, dq, torch.zeros_like(q))
    assert (effort[0, [0, 1]] > 0.).all() and (effort[0, [3, 4]] < 0.).all()


def test_1khz_wrapped_feedback_uses_common_continuous_encoder_state(rig):
    env, recovery = rig
    q, dq = motor_state(env)
    q = q.clone()
    recovery.release_finished[:] = True
    recovery.fsm.phase[0] = RecoveryPhase.ORBIT
    before_wrap = q.clone()
    before_wrap[0, :2] = 2 * math.pi + .3
    recovery.encoders.reset([0], before_wrap)
    wrapped = q.clone()
    wrapped[0, :2] = .3
    recovery.observe_feedback(wrapped)
    recovery.fsm.script_targets[0, :2] = 2 * math.pi + .5
    q[0, :2] = .4
    recovery.torque(q, dq, torch.zeros_like(q))
    expected = env.v5.settings["leg_kp"] * .1
    assert env.v5.requested_motor_effort[0, 0].item() == pytest.approx(expected, abs=1e-4)


def test_common_feedback_provider_keeps_fk_fsm_pd_coherent_across_raw_axis_wrap(rig):
    env, recovery = rig
    q = env.nominal[:, env.ids].clone()
    q[0] = torch.tensor([-5.981, -5.348, 0., 5.982, 5.349, 0.])
    env.robot.data.joint_pos.torch[:, env.ids] = q
    recovery.reset(torch.arange(2))
    raw = q.clone()
    raw[0] = torch.tensor([5.799, -6.046, 0., -5.799, 6.045, 0.])
    continuous = recovery.observe_feedback(raw).clone()
    assert recovery.geometry.wheel_positions(raw)[1][0].eq(False).all()
    assert recovery.geometry.wheel_positions(continuous)[1].all()
    assert continuous[0, 0] < -math.tau and continuous[0, 3] > math.tau
    recovery.elapsed[0] = .2
    gravity, gyro, dq = torch.tensor([[0., 0., -1.]]).repeat(2, 1), torch.zeros(2, 3), torch.zeros(2, 6)
    recovery.advance_reference(raw, dq, gravity, gyro)
    assert recovery.release_finished[0] and recovery.encoders.valid.all()
    torch.testing.assert_close(recovery.fsm.continuous_q, continuous, atol=1e-6, rtol=0.)
    expected_points = recovery.geometry.wheel_positions(continuous)[0]
    from wheeled_tasks.chassis.step_trigger import estimate_height_if_grounded
    h, _ = estimate_height_if_grounded(expected_points, gravity, recovery.geometry.wheel_radius_m,
        grounded=torch.ones(2, 2, dtype=torch.bool), wheel_axle_b=recovery.geometry.wheel_axles_b)
    torch.testing.assert_close(recovery.observed_height, h.amin(-1))
    snapshot = recovery.encoders.q.clone()
    torch.testing.assert_close(recovery.project_feedback(raw), continuous, atol=1e-6, rtol=0.)
    torch.testing.assert_close(recovery.encoders.q, snapshot)


def test_reset_at_already_split_raw_winding_aligns_only_integer_aux_turns(rig):
    env, recovery = rig
    raw = env.nominal[:, env.ids].clone()
    raw[0] = torch.tensor([5.799, -6.046, 0., -5.799, 6.045, 0.])
    env.robot.data.joint_pos.torch[:, env.ids] = raw
    recovery.reset(torch.arange(2))
    assert recovery.encoders.valid.all()
    torch.testing.assert_close(recovery.encoders.q[0, [0, 3]], raw[0, [0, 3]])
    torch.testing.assert_close(torch.cos(recovery.encoders.q), torch.cos(raw), atol=1e-6, rtol=0.)
    recovery.elapsed[0] = .2
    recovery.observe_feedback(raw)
    recovery.advance_reference(raw, torch.zeros(2, 6), torch.tensor([[0., 0., -1.]]).repeat(2, 1), torch.zeros(2, 3))
    assert recovery.encoders.valid.all() and not recovery.failed[0]


def test_feedback_projection_does_not_relax_actual_illegal_relative_angles(rig):
    env, recovery = rig
    raw = env.nominal[:, env.ids].clone()
    raw[0, 1] = raw[0, 0] + recovery.encoders.relative_limits[0, 0] - .05
    env.robot.data.joint_pos.torch[:, env.ids] = raw
    recovery.reset(torch.arange(2))
    recovery.observe_feedback(raw)
    assert not recovery.encoders.valid[0, 0]
    assert not recovery.geometry.wheel_positions(recovery.project_feedback(raw))[1][0, 0]


def physical_data(recovery):
    recovery.release_finished[:] = True
    recovery.fsm.phase[0] = RecoveryPhase.RL
    recovery.fsm.motion_released[:] = False
    return (torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
            torch.tensor([[0., 0., -1.], [0., 0., -1.]]), torch.zeros(2, 3),
            torch.full((2,), .305), torch.zeros(2, 3))


def test_followup_window_does_not_reuse_the_last_stability_interval(rig):
    _, recovery = rig
    data = physical_data(recovery)
    recovery.followup_command[0, :2] = 0.  # Hold then continue standing.
    for _ in range(49):
        recovery.outcome(*data)
    recovery.fsm.motion_released[0] = True
    recovery.outcome(*data)
    assert recovery.true_stable_peak[0] == pytest.approx(1., abs=1e-6)
    assert recovery.followup_seconds[0] == 0.
    for _ in range(49):
        recovery.outcome(*data)
    assert not recovery.true_success[0]
    recovery.outcome(*data)
    assert recovery.true_success[0]


def test_release_before_real_stability_cannot_be_repaired_by_later_moving_samples(rig):
    _, recovery = rig
    data = physical_data(recovery)
    for _ in range(5):
        recovery.outcome(*data)
    recovery.fsm.motion_released[0] = True
    data[-1][0, 0] = .2  # Within both loose stand and forward-tracking tolerances.
    for _ in range(150):
        recovery.outcome(*data)
    assert recovery.true_stable_peak[0] == pytest.approx(.12, abs=1e-6)
    assert recovery.followup_seconds[0] == 0. and not recovery.true_success[0]


def test_wrapper_timeout_has_explicit_failure_code(rig):
    _, recovery = rig
    recovery.elapsed[0] = 8.
    data = recovery.diagnostics()
    assert data["recovery_failed"][0] and data["recovery_failure_code"][0] == 6


def test_wheel_balance_velocity_target_reproduces_pitch_gyro_effort_through_existing_p(rig):
    env, recovery = rig
    q, dq = motor_state(env)
    recovery.release_finished[:] = True
    recovery.wheel_balance["enabled"] = True
    recovery.fsm.phase[0] = RecoveryPhase.PREPARE
    angle, omega = .2, .3
    gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]]).repeat(2, 1)
    gyro = torch.tensor([[0., omega, 0.]]).repeat(2, 1)
    dq[0, [2, 5]] = torch.tensor([1., -2.])
    recovery._update_wheel_targets(gravity, gyro, dq, torch.ones(2, dtype=torch.bool), recovery.fsm.phase.clone())
    recovery.torque(q, dq, torch.zeros_like(q))
    expected = (8. * angle + 1.5 * omega) * recovery.wheel_axis_signs - .2 * dq[0, [2, 5]]
    torch.testing.assert_close(env.v5.requested_motor_effort[0, [2, 5]], expected)
    assert recovery.wheel_balance_active.tolist() == [True, False]


@pytest.mark.parametrize("phase,supported,angle,expected", [
    (RecoveryPhase.PREPARE, True, .2, True), (RecoveryPhase.PREPARE, False, .2, False),
    (RecoveryPhase.PREPARE, True, math.radians(60.), False),
    (RecoveryPhase.CAPTURE, True, math.radians(70.), True),
    (RecoveryPhase.CAPTURE, True, math.radians(86.), False),
    (RecoveryPhase.ORBIT, True, .2, False), (RecoveryPhase.THRUST, True, .2, False),
    (RecoveryPhase.RL, True, .2, False)])
def test_wheel_balance_is_limited_to_sensor_supported_script_capture(rig, phase, supported, angle, expected):
    _, recovery = rig
    recovery.release_finished[:] = True
    recovery.wheel_balance["enabled"] = True
    recovery.fsm.phase[0] = phase
    gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]]).repeat(2, 1)
    recovery._update_wheel_targets(gravity, torch.zeros(2, 3), torch.zeros(2, 6),
        torch.tensor([supported, False]), recovery.fsm.phase.clone())
    assert recovery.wheel_balance_active[0].item() is expected


def test_blend_retains_last_wheel_reference_but_drops_it_on_lost_support(rig):
    _, recovery = rig
    recovery.release_finished[:] = True
    recovery.wheel_balance["enabled"] = True
    recovery.fsm.phase[0] = RecoveryPhase.PREPARE
    gravity = torch.tensor([[math.sin(.2), 0., -math.cos(.2)]]).repeat(2, 1)
    gyro, dq, supported = torch.zeros(2, 3), torch.zeros(2, 6), torch.ones(2, dtype=torch.bool)
    recovery._update_wheel_targets(gravity, gyro, dq, supported, recovery.fsm.phase.clone())
    previous = recovery.fsm.phase.clone()
    recovery.fsm.phase[0] = RecoveryPhase.BLEND
    recovery._update_wheel_targets(gravity, gyro, dq, supported, previous)
    target = recovery.fsm.script_targets[:, [2, 5]].clone()
    recovery._update_wheel_targets(gravity, gyro + .1, dq + .2, supported, recovery.fsm.phase.clone())
    torch.testing.assert_close(recovery.fsm.script_targets[:, [2, 5]], target)
    recovery._update_wheel_targets(gravity, gyro, dq, ~supported, recovery.fsm.phase.clone())
    assert recovery.fsm.script_targets[:, [2, 5]].eq(0.).all()


def test_recovery_on_a_platform_requires_independent_true_platform_evidence(rig):
    _, recovery = rig
    data = physical_data(recovery)
    recovery.required_platform_height[0] = .2
    for _ in range(60):
        recovery.outcome(*data)  # Missing target-platform evidence fails closed.
    assert recovery.true_stable_peak[0] == 0. and not recovery.goal_platform[0]
    for _ in range(60):
        recovery.outcome(*data, goal_platform=torch.tensor([False, True]))
    assert recovery.true_stable_peak[0] == 0.
    for _ in range(50):
        recovery.outcome(*data, goal_platform=torch.ones(2, dtype=torch.bool))
    recovery.fsm.motion_released[0] = True
    data[-1][0, 0] = .3
    for _ in range(50):
        recovery.outcome(*data, goal_platform=torch.ones(2, dtype=torch.bool))
    assert recovery.true_success[0]
    recovery.outcome(*data, goal_platform=torch.tensor([False, True]))
    assert not recovery.true_success[0]
    assert not recovery.diagnostics()["recovery_platform_retained"][0]


def test_initial_zero_torque_release_cannot_latch_motion_release_before_handover(rig):
    _, recovery = rig
    assert not recovery.release_finished[0] and recovery.fsm.motion_released[0]
    data = (torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
            torch.tensor([[0., 0., -1.], [0., 0., -1.]]), torch.zeros(2, 3),
            torch.full((2,), .305), torch.zeros(2, 3))
    for _ in range(10):
        recovery.outcome(*data)
    assert not recovery.motion_release_observed[0] and recovery.true_stable_peak[0] == 0.
    recovery.release_finished[:] = True
    recovery.fsm.phase[0] = RecoveryPhase.RL
    recovery.fsm.motion_released[0] = False
    for _ in range(50):
        recovery.outcome(*data)
    assert recovery.true_stable_peak[0] >= 1. - 1e-5
    recovery.fsm.motion_released[0] = True
    data[-1][0, 0] = .3
    for _ in range(50):
        recovery.outcome(*data)
    assert recovery.true_success[0]


def test_scripted_action_history_includes_nonzero_wheel_balance_reference(rig):
    env, recovery = rig
    recovery.release_finished[:] = True
    recovery.fsm.phase[0] = RecoveryPhase.PREPARE
    recovery.fsm.script_targets[0, [2, 5]] = torch.tensor([2., -2.])
    recovery.effective_action_history()
    torch.testing.assert_close(env.actions[0, [2, 5]], torch.tensor([2., -2.]) / env.v5.settings["wheel_velocity_scale"])


def test_signed_motor_axes_are_independent_of_unoriented_collision_cylinders(rig):
    _, recovery = rig
    assert recovery.geometry.wheel_axles_b[:, 1].gt(0.).all()
    torch.testing.assert_close(recovery.wheel_axis_signs, torch.tensor([1., -1.]))


def test_native_pd_held_target_does_not_receive_a_wrap_torque_impulse(rig):
    env, recovery = rig
    initial = env.nominal[:, env.ids].clone()
    recovery.observe_feedback(initial)
    target, wheel_target, _ = env.v5.decode(torch.zeros_like(initial), recovery.project_feedback(initial))
    physical = initial + .001
    raw = physical.clone()
    # An independently wrapped hip sample arrives during the 20 ms target hold.
    raw[:, 0] += 2 * math.tau
    raw[:, 3] -= 2 * math.tau
    recovery.observe_feedback(raw)
    continuous = recovery.project_feedback(raw)
    expected = env.v5.motor_efforts(physical, torch.zeros_like(raw), target, wheel_target)
    actual = env.v5.motor_efforts(continuous, torch.zeros_like(raw), target, wheel_target)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0.)
    assert actual[:, list(env.v5.LEGS)].abs().max() < 1.
    raw_effort = env.v5.motor_efforts(raw, torch.zeros_like(raw), target, wheel_target)
    assert raw_effort[:, [0, 3]].abs().eq(40.).all()
    # A snapshot must survive auto-reset, which replaces the provider buffers.
    snapshot = recovery.diagnostics()["recovery_continuous_q"]
    recovery.reset(torch.arange(2))
    torch.testing.assert_close(snapshot, continuous)
    assert not torch.equal(snapshot, recovery.encoders.q)


@pytest.mark.parametrize("signs", [None, [1.], [1., 0.], [1., float("nan")]])
def test_frozen_motor_axis_signs_cannot_be_missing_or_invalid(rig, tmp_path, signs):
    env, recovery = rig
    settings = dict(recovery.settings)
    path = tmp_path / "profiles.json"
    profile = json.loads(path.read_text())
    if signs is None:
        profile.pop("wheel_axis_signs")
    else:
        profile["wheel_axis_signs"] = signs
    path.write_text(json.dumps(profile))
    settings["profiles_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="signed wheel-motor axes"):
        RecoveryTraining(env, tmp_path, settings)
