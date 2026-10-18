"""Delayed drive impulse checks against continuous convolution, not a robot model."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.actuator_response import ActuatorResponse, JOINT_NAMES, SCHEMA


def configuration(constants=None):
    return {"enabled": True, "schema": SCHEMA, "version": 1,
            "joint_names": list(JOINT_NAMES), "gain": 1.,
            "armature": [.01] * 6, "frictionloss": [.2] * 6, "damping": [.1] * 6,
            "time_constant_s": constants or [.00386, .00441, 0., .008, .0003, 0.]}


def convolution(initial, previous, current, delay, duration, tau):
    """One input jump expressed as a step response added to the old response."""
    if tau == 0.:
        return (previous * delay + current * (duration - delay)) / duration, (
            current if delay < duration else previous)
    decay = math.exp(-duration / tau)
    remaining = duration - delay
    jump_decay = math.exp(-remaining / tau)
    endpoint = previous + (initial - previous) * decay + (current - previous) * (1. - jump_decay)
    integral = (previous * duration + (initial - previous) * tau * (1. - decay)
                + (current - previous) * (remaining - tau * (1. - jump_decay)))
    return integral / duration, endpoint


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_and_endpoint_match_continuous_step_convolution(dtype):
    cfg = configuration()
    response = ActuatorResponse(3, "cpu", .005, cfg, dtype=dtype)
    initial = torch.tensor([[2., -3., 1., 8., -7., 4.],
                            [-4., 6., -2., 1., 3., -8.],
                            [10., 0., 4., -5., 2., 6.]], dtype=dtype)
    previous = torch.arange(18, dtype=dtype).reshape(3, 6) - 8.
    current = -previous - 1.
    delays = torch.tensor([0., .0017, .005], dtype=torch.float64)
    response.state.copy_(initial)
    untouched = [value.clone() for value in (initial, previous, current, delays)]
    applied = response.apply_delayed_torque(previous, current, delays)
    expected = [convolution(float(initial[row, axis]), float(previous[row, axis]),
        float(current[row, axis]), float(delays[row]), .005, cfg["time_constant_s"][axis])
        for row in range(3) for axis in range(6)]
    means = torch.tensor([value[0] for value in expected], dtype=dtype).reshape(3, 6)
    ends = torch.tensor([value[1] for value in expected], dtype=dtype).reshape(3, 6)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-13
    torch.testing.assert_close(applied, means, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(response.state, ends, rtol=tolerance, atol=tolerance)
    assert applied.data_ptr() != response.state.data_ptr()
    assert not torch.allclose(applied[:, :2], response.state[:, :2])
    for actual, before in zip((initial, previous, current, delays), untouched):
        assert torch.equal(actual, before)


def test_impulse_matches_high_resolution_continuous_quadrature():
    cfg = configuration([.00386, .00441, .001, .008, .0003, .04])
    response = ActuatorResponse(1, "cpu", .005, cfg, dtype=torch.float64)
    initial = np.array([3., -9., 2., 8., 1., -5.])
    previous = np.array([-40., 14., -3., 20., -1., 4.])
    current = np.array([40., -9., 2., -30., 3., -8.])
    delay = .0021
    response.state.copy_(torch.tensor(initial[None]))
    mean = response.apply_delayed_torque(torch.tensor(previous[None]),
        torch.tensor(current[None]), torch.tensor([delay], dtype=torch.float64)).numpy()[0]
    # Integrate the continuous convolution independently over time. Split at
    # the input jump so endpoint placement cannot hide a quadrature error.
    numerical = np.zeros(6)
    for start, end in ((0., delay), (delay, .005)):
        times = np.linspace(start, end, 100_001)[:, None]
        tau = np.array(cfg["time_constant_s"])[None]
        values = (previous + (initial - previous) * np.exp(-times / tau)
            + (current - previous) * (-np.expm1(-np.maximum(times - delay, 0.) / tau)))
        numerical += np.trapezoid(values, times[:, 0], axis=0)
    np.testing.assert_allclose(mean * .005, numerical, rtol=2e-9, atol=2e-11)


def test_zero_lag_reduces_exactly_to_existing_delayed_command_average():
    cfg = configuration([0.] * 6)
    response = ActuatorResponse(2, "cpu", .005, cfg)
    previous = torch.tensor([[1e20, 2., 3., 4., 5., 6.], [-8., -3., 4., 9., 2., 0.]])
    current = -previous + 1.
    delays = torch.tensor([.00071745416553841, .005], dtype=torch.float64)
    expected = current + (delays / .005).float()[:, None] * (previous - current)
    applied = response.apply_delayed_torque(previous, current, delays)
    assert torch.equal(applied, expected)
    assert torch.equal(response.state[0], current[0])
    assert torch.equal(response.state[1], previous[1])


def test_zero_delay_constant_input_impulse_and_subdivision_are_consistent():
    cfg = configuration()
    coarse = ActuatorResponse(1, "cpu", .005, cfg, dtype=torch.float64)
    fine = ActuatorResponse(1, "cpu", .001, cfg, dtype=torch.float64)
    command = torch.full((1, 6), 40., dtype=torch.float64)
    delay = torch.zeros(1, dtype=torch.float64)
    mean = coarse.apply_delayed_torque(torch.zeros_like(command), command, delay)
    fine_impulse = sum(fine.apply_delayed_torque(command, command, delay) * .001 for _ in range(5))
    torch.testing.assert_close(coarse.state, fine.state, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(mean * .005, fine_impulse, rtol=1e-13, atol=1e-13)
    assert mean[0, 0] < coarse.state[0, 0]
    expected_mean = 40. * (1. - cfg["time_constant_s"][0] / .005
                           * (1. - math.exp(-.005 / cfg["time_constant_s"][0])))
    assert float(mean[0, 0]) == pytest.approx(expected_mean, abs=1e-12)


def test_tiny_and_large_duration_over_tau_avoid_cancellation_or_nonfinite_values():
    cfg = configuration([1e20, 1e-20, 0., 1e-12, 1e12, .005])
    response = ActuatorResponse(2, "cpu", .005, cfg, dtype=torch.float64)
    response.state[:, 1] = 1e20
    previous = torch.full((2, 6), 2., dtype=torch.float64)
    current = previous.clone()
    delays = torch.tensor([0., .005], dtype=torch.float64)
    mean = response.apply_delayed_torque(previous, current, delays)
    assert bool(torch.isfinite(mean).all() & torch.isfinite(response.state).all())
    # For very slow response the first impulse is u*h/(2*tau); for a
    # vanishing time constant the large initial state contributes z0*tau/h.
    torch.testing.assert_close(mean[:, 0], torch.full((2,), 5e-23, dtype=torch.float64), rtol=1e-12, atol=0.)
    torch.testing.assert_close(mean[:, 1], torch.full((2,), 202., dtype=torch.float64), rtol=1e-12, atol=1e-12)
    assert torch.equal(response.state[:, 1], current[:, 1])


def test_bounds_apply_to_both_segments_initial_state_and_separate_outputs():
    response = ActuatorResponse(2, "cpu", .005, configuration(), dtype=torch.float64)
    response.state.fill_(80.)
    previous = torch.full((2, 6), -90., dtype=torch.float64)
    current = torch.full((2, 6), 100., dtype=torch.float64)
    bounds = torch.tensor([[40., 30., 4.5, 40., 35., 4.5]], dtype=torch.float64).expand(2, -1).clone()
    delay = torch.tensor([.001, .004], dtype=torch.float64)
    before = [value.clone() for value in (previous, current, bounds)]
    mean = response.apply_delayed_torque(previous, current, delay, effort_bounds=bounds)
    assert bool((mean.abs() <= bounds).all() & (response.state.abs() <= bounds).all())
    for row in range(2):
        for axis, tau in enumerate(response.cfg["time_constant_s"]):
            expected = convolution(float(bounds[row, axis]), -float(bounds[row, axis]),
                float(bounds[row, axis]), float(delay[row]), .005, tau)
            assert float(mean[row, axis]) == pytest.approx(expected[0], abs=1e-12)
            assert float(response.state[row, axis]) == pytest.approx(expected[1], abs=1e-12)
    for actual, original in zip((previous, current, bounds), before):
        assert torch.equal(actual, original)


def test_multistep_interval_impulses_and_selected_reset_preserve_other_lane():
    cfg = configuration()
    response = ActuatorResponse(2, "cpu", .005, cfg, dtype=torch.float64)
    expected = np.zeros((2, 6))
    previous = torch.zeros(2, 6, dtype=torch.float64)
    delay = torch.tensor([.0008, .0024], dtype=torch.float64)
    for step in range(7):
        current = torch.tensor([[20., -10., 2., 14., -6., -1.],
                                [-30., 8., -3., 19., 4., 2.]], dtype=torch.float64) * (-1.) ** step
        means = np.empty_like(expected)
        for row in range(2):
            for axis, tau in enumerate(cfg["time_constant_s"]):
                means[row, axis], expected[row, axis] = convolution(expected[row, axis],
                    float(previous[row, axis]), float(current[row, axis]), float(delay[row]), .005, tau)
        mean = response.apply_delayed_torque(previous, current, delay)
        np.testing.assert_allclose(mean.numpy(), means, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(response.state.numpy(), expected, rtol=1e-12, atol=1e-12)
        previous = current.clone()
    preserved = response.state[1].clone()
    response.reset(torch.tensor([0]))
    assert torch.count_nonzero(response.state[0]) == 0
    assert torch.equal(response.state[1], preserved)
    response.reset()
    assert torch.count_nonzero(response.state) == 0


def test_disabled_interval_response_is_transport_average_without_physical_overrides():
    response = ActuatorResponse(2, "cpu", .005)
    previous, current = torch.ones(2, 6), torch.full((2, 6), -2.)
    delay = torch.tensor([0., .005])
    assert torch.equal(response.apply_delayed_torque(previous, current, delay),
                       torch.stack((current[0], previous[1])))
    assert response.state is None and response.native_parameters() == {}
    bounded = response.apply_delayed_torque(previous * 100., current * 100.,
        delay, effort_bounds=torch.full((2, 6), 4.5))
    assert torch.equal(bounded, torch.stack((torch.full((6,), -4.5), torch.full((6,), 4.5))))


@pytest.mark.parametrize("field,value", [
    ("previous", torch.zeros(2, 5)), ("current", torch.zeros(3, 6)),
    ("current", torch.zeros(2, 6, dtype=torch.float64)), ("previous", [[0.] * 6] * 2),
    ("delay", torch.zeros(2, 1)), ("delay", torch.zeros(2, dtype=torch.long)),
    ("bounds", torch.ones(6)), ("bounds", torch.ones(2, 6, dtype=torch.float64)),
])
def test_structural_mismatches_do_not_change_drive_state(field, value):
    response = ActuatorResponse(2, "cpu", .005, configuration())
    response.state.fill_(3.)
    fields = {"previous": torch.zeros(2, 6), "current": torch.zeros(2, 6),
              "delay": torch.zeros(2), "bounds": torch.full((2, 6), 40.)}
    fields[field] = value
    before = response.state.clone()
    with pytest.raises(ValueError):
        response.apply_delayed_torque(fields["previous"], fields["current"], fields["delay"],
                                     effort_bounds=fields["bounds"])
    assert torch.equal(response.state, before)


def test_legacy_endpoint_update_and_native_parameters_remain_unchanged():
    cfg = configuration()
    original = deepcopy(cfg)
    response = ActuatorResponse(1, "cpu", .001, cfg, dtype=torch.float64)
    command = torch.full((1, 6), 10., dtype=torch.float64)
    for _ in range(5):
        applied = response.apply_torque(command)
    expected = command * command.new_tensor([1. - math.exp(-.005 / tau) if tau else 1.
                                            for tau in cfg["time_constant_s"]])
    torch.testing.assert_close(applied, expected, rtol=1e-13, atol=1e-13)
    assert cfg == original
    assert response.native_parameters()["damping"]["L_joint1"] == .1
