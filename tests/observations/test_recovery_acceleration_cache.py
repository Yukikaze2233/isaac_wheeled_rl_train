"""Actual cached frame times and controller wiring, without PhysX execution."""
from types import SimpleNamespace

import pytest
import torch

from tests.support.support_direct_recovery_takeover import direct_rig, rig
from tests.support.support_dynamic_birth_env_wiring import methods
from wheeled_tasks.chassis.observations.recovery_sensor_cache import RecoveryAccelerationCache


def test_stored_times_control_interpolation_clamps_and_duplicate_frame_identity():
    cache = RecoveryAccelerationCache(2, "cpu", .005, .03)
    zero = torch.zeros(2, 3)
    cache.reset(torch.arange(2), zero, 0)
    cache.append(torch.full_like(zero, 10.), 1)
    cache.append(torch.full_like(zero, 20.), 2)
    value, stamp = cache.read(torch.tensor([.0025, .0075], dtype=torch.float64))
    torch.testing.assert_close(value, torch.tensor([[5.] * 3, [15.] * 3]))
    torch.testing.assert_close(stamp, torch.tensor([.0025, .0075], dtype=torch.float64))
    before = (cache.values.clone(), cache.stamps_s.clone(), cache.pointer)
    cache.append(torch.full_like(zero, 99.), 2)
    assert cache.pointer == before[2]
    torch.testing.assert_close(cache.values, before[0])
    torch.testing.assert_close(cache.stamps_s, before[1], equal_nan=True)
    _, clamped = cache.read(torch.tensor([-.1, .9], dtype=torch.float64))
    torch.testing.assert_close(clamped, torch.tensor([0., .01], dtype=torch.float64))


def test_partial_reset_cannot_read_old_episode_or_advance_global_time():
    cache = RecoveryAccelerationCache(2, "cpu", .005, .03)
    cache.reset(torch.arange(2), torch.zeros(2, 3), 0)
    cache.append(torch.full((2, 3), 10.), 1)
    cache.append(torch.full((2, 3), 20.), 2)
    other_values, other_stamps = cache.values[:, 1].clone(), cache.stamps_s[:, 1].clone()
    previous = (cache.pointer, cache.last_physical_tick)
    fresh = torch.tensor([[-7.] * 3, [99.] * 3])
    cache.reset(torch.tensor([0]), fresh, 2)
    cache.reset(torch.tensor([0]), fresh, 2)
    assert (cache.pointer, cache.last_physical_tick) == previous
    torch.testing.assert_close(cache.values[:, 1], other_values)
    torch.testing.assert_close(cache.stamps_s[:, 1], other_stamps, equal_nan=True)
    value, stamp = cache.read(torch.full((2,), .001, dtype=torch.float64))
    torch.testing.assert_close(value[0], fresh[0])
    assert stamp[0] == .01 and stamp[1] == .001


def test_empty_or_unavailable_frame_has_no_usable_acceleration_timestamp():
    cache = RecoveryAccelerationCache(2, "cpu", .005, .03)
    value, stamp = cache.read(torch.tensor([0., torch.nan], dtype=torch.float64))
    assert value.eq(0).all() and stamp.isnan().all()


def test_actual_direct_owner_delays_raw_acceleration_without_second_delay_or_resampling(direct_rig):
    env, recovery, _ = direct_rig
    env.physical_tick = 1
    env.robot.data.root_link_lin_vel_w.torch[:, 0] = .005
    env._pc_feedback_transport = SimpleNamespace(sensor_timestamp_s=torch.full((2,), .0025, dtype=torch.float64))
    value, stamp = recovery._direct_acceleration()
    torch.testing.assert_close(value[:, 0], torch.full((2,), .5))
    torch.testing.assert_close(stamp, env._pc_feedback_transport.sensor_timestamp_s)
    previous_velocity = recovery.imu.previous_velocity.clone()
    # Same time is not a new physical sample, even if a CPU double is mutated.
    env.robot.data.root_link_lin_vel_w.torch[:, 0] = 100.
    repeated, repeated_stamp = recovery._direct_acceleration()
    torch.testing.assert_close(repeated, value)
    torch.testing.assert_close(repeated_stamp, stamp)
    torch.testing.assert_close(recovery.imu.previous_velocity, previous_velocity)
    env.physical_tick = 2
    env.robot.data.root_link_lin_vel_w.torch[:, 0] = .015
    env._pc_feedback_transport.sensor_timestamp_s.fill_(.005)
    delayed, delayed_stamp = recovery._direct_acceleration()
    torch.testing.assert_close(delayed[:, 0], torch.ones(2))
    assert delayed_stamp.eq(.005).all()
    previous_velocity = recovery.imu.previous_velocity.clone()
    env.physical_tick = 1
    with pytest.raises(ValueError, match="monotonic actual physical tick"):
        recovery._direct_acceleration()
    torch.testing.assert_close(recovery.imu.previous_velocity, previous_velocity)


def test_invalid_snapshot_diagnostics_use_current_actual_stamp_and_preserve_accepted_identity(direct_rig):
    env, recovery, q = direct_rig
    gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    env.physical_tick = 1
    env._pc_feedback_transport = SimpleNamespace(sensor_timestamp_s=torch.full((2,), .005, dtype=torch.float64))
    recovery.advance_reference(q, torch.zeros_like(q), gravity, torch.zeros(2, 3))
    assert recovery.sensor_status.valid.all()
    accepted_sequences = recovery.sensor_sample_sequences.clone()
    env.physical_tick = 2
    env._pc_feedback_transport.sensor_timestamp_s.fill_(.03)
    recovery.advance_reference(q, torch.zeros_like(q), gravity, torch.zeros(2, 3))
    assert not recovery.sensor_status.valid.any()
    torch.testing.assert_close(recovery.sensor_sample_sequences, accepted_sequences)
    diagnostic = recovery.diagnostics()
    assert diagnostic["recovery_acceleration_stamp_s"].eq(.01).all()
    assert diagnostic["recovery_accepted_acceleration_stamp_s"].eq(.005).all()
    assert diagnostic["recovery_sensor_acceleration_age_ms"].eq(0.).all()
    assert diagnostic["recovery_sensor_imu_age_ms"].eq(-1.).all()
    env.physical_tick = 3
    env._pc_feedback_transport.sensor_timestamp_s.fill_(.015)
    recovery.advance_reference(q, torch.zeros_like(q), gravity, torch.zeros(2, 3))
    assert recovery.sensor_status.valid.all()
    assert recovery.valid_sensor_history_s.eq(0.).all()


def test_large_global_clock_does_not_shorten_sixty_ms_valid_history(direct_rig):
    env, recovery, q = direct_rig
    env.physical_tick = 50_000_000
    recovery.reset(torch.arange(2))
    gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    for _ in range(13):
        recovery.advance_reference(q, torch.zeros_like(q), gravity, torch.zeros(2, 3))
        env.physical_tick += 1
    assert recovery.valid_sensor_history_s[0] >= .06
    assert recovery.valid_sensor_history_s[0] == pytest.approx(.06)


def test_actual_environment_log_reports_age_in_ms_from_the_current_status(direct_rig):
    env, recovery, q = direct_rig
    env.physical_tick = 1
    env._pc_feedback_transport = SimpleNamespace(sensor_timestamp_s=torch.full((2,), .0025, dtype=torch.float64))
    recovery.advance_reference(q, torch.zeros_like(q), torch.tensor([[0., 0., -1.]]).repeat(2, 1),
                               torch.zeros(2, 3))
    env.recovery = recovery
    env.recovery_takeover_settings = None
    env.recovery_hold_cost_diagnostics = {}
    mask = torch.zeros(2, dtype=torch.bool)
    result = methods("_recovery_metrics")["_recovery_metrics"](env, mask, mask, mask, mask)
    for key in ("sensor_motor_age_max_ms", "sensor_imu_age_max_ms", "sensor_acceleration_age_max_ms"):
        assert result["/recovery/" + key] == 2.5
    assert result["/recovery/sensor_time_valid_fraction"] == 1.
