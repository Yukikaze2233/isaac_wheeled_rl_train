"""Exercise the production transport/controller handoff with continuous impulse checks."""
import ast
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.actuator_response import ActuatorResponse, JOINT_NAMES
from wheeled_tasks.chassis.round_trip_transport import (
    EffectiveCommandDelay, RoundTripDelayBudget, round_trip_transport_settings,
)
from support_nonstandard_start_integration import execute, mentions, method


def make_pair():
    cfg = round_trip_transport_settings()
    budget = RoundTripDelayBudget(2, "cpu", cfg, 17)
    budget.reset(budget.rows)
    budget.downlink_s.copy_(torch.tensor([.0004, .0024], dtype=torch.float64))
    transport = EffectiveCommandDelay(2, "cpu", .005, cfg, budget)
    response = ActuatorResponse(2, "cpu", .005, dict(
        enabled=True, schema="chassis-actuator-response", version=1,
        joint_names=list(JOINT_NAMES), gain=1., armature=[.01] * 6,
        frictionloss=[.2] * 6, damping=[.1] * 6,
        time_constant_s=[.004, .004, 0., .004, .004, 0.]))
    return budget, transport, response


def test_transport_integrates_before_replacing_held_input_and_keeps_request_origins():
    budget, transport, response = make_pair()
    current = torch.tensor([[40., -40., 4., 20., -20., -4.]]).repeat(2, 1)
    initial = torch.zeros_like(current)
    for step in range(3):
        previous = transport.last_command.clone()
        expected = ActuatorResponse(2, "cpu", .005, response.cfg)
        expected.state.copy_(initial)
        mean = expected.apply_delayed_torque(previous, current, budget.downlink_s)
        actual = transport.apply_torque(current, pd_generated_s=step * .005,
            actuator_response=response)
        torch.testing.assert_close(actual, mean, rtol=0., atol=0.)
        torch.testing.assert_close(response.state, expected.state, rtol=0., atol=0.)
        assert torch.equal(transport.last_command, current)
        origin = step * .005 - budget.downlink_s * (step > 0)
        torch.testing.assert_close(transport.applied_origins_s[:, 0], origin)
        initial = response.state.clone()
        current = -current
    # Reset one transport/drive lane; preserve the other's history.
    preserved = response.state[1].clone()
    transport.reset(torch.tensor([0]))
    response.reset(torch.tensor([0]))
    assert transport.last_command[0].eq(0.).all() and response.state[0].eq(0.).all()
    assert torch.equal(response.state[1], preserved)


@pytest.mark.parametrize("critic78", [True, False])
def test_production_substep_applies_interval_mean_without_a_second_filter(critic78):
    budget, transport, response = make_pair()
    request = torch.tensor([[40., -40., 4., 20., -20., -4.]]).repeat(2, 1)
    bounds = torch.full_like(request, 40.)
    bounds[:, [2, 5]] = 4.5
    env = NS(torque=request.clone(), motor_strength=torch.ones_like(request),
        skills=None, nonstandard_start=None, command_transport=transport,
        round_trip_transport=budget, actuator_response=response, usb_transport=None,
        recovery=None, critic78=critic78, physical_tick=0, dt=.005,
        policy_generated_s=torch.zeros(2, dtype=torch.float64), _pc_feedback_transport=None,
        v5=NS(current_motor_bounds=bounds, WHEELS=(2, 5)))

    def legacy_filter_is_not_in_the_integrated_path(command):
        raise AssertionError("The native interval mean must not pass through a second endpoint filter")

    response.apply_torque = legacy_filter_is_not_in_the_integrated_path
    loop = next(node for node in method("step").body if isinstance(node, ast.For)
                and mentions(node.target, "substep"))
    start = next(i for i, node in enumerate(loop.body) if mentions(node, "self.torque *= self.motor_strength"))
    end = next(i for i, node in enumerate(loop.body) if mentions(node, "effort = torch.zeros_like(self.nominal)"))
    execute(loop.body[start:end], env, flat=None, dq_now=torch.zeros(2, 6))
    # Continuous step response starts only when the delayed request arrives.
    duration = .005 - budget.downlink_s
    weight = (duration - .004 * (-torch.expm1(-duration / .004))) / .005
    expected = request.clone()
    expected[:, [0, 1, 3, 4]] *= weight.float()[:, None]
    expected[:, [2, 5]] *= (duration / .005).float()[:, None]
    torch.testing.assert_close(env.torque, expected, rtol=2e-6, atol=1e-6)
    assert not torch.allclose(env.torque[:, [0, 1]], response.state[:, [0, 1]])
    assert env.torque.data_ptr() != response.state.data_ptr()
    assert bool((env.torque.abs() <= bounds).all())
    assert bool((response.state.abs() <= bounds).all())
