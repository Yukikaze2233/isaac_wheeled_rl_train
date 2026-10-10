"""Execute the production command-downlink block without starting Isaac."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.command_transport import UsbCommandTransport
from wheeled_tasks.chassis.v5_control import V5Control
from wheeled_tasks.v40 import core

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def command_block():
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    loop = next(node for node in step.body if isinstance(node, ast.For))
    block = next(node for node in loop.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "self.command_transport is not None")
    return compile(ast.Module(body=[block], type_ignores=[]), "production_command_downlink", "exec")


def controller(curve):
    read = lambda path: json.loads((ROOT / path).read_text())
    bundle = "model/纯底盘_v5/urdf/"
    prior = read("contracts/own_v40_v2.json")
    if curve == "cap":
        wheel = prior["actuators"]["wheel"]
        wheel.update(curve_side="wheel_output", effort_limit=4.5,
                     curve_kind="software_cap_only_not_identified", scope="simulation_research_prior",
                     torque_reference="wheel_output")
        for name in ("motor_torque_nm", "motor_speed_rad_s"):
            wheel.pop(name)
    return V5Control(read(bundle + "manifest.json"), read(bundle + "model_spec.json"),
                     read(bundle + "fit_10mpa.json"), prior,
                     read("contracts/v5_foundation_v1.json")["v5_control"], "cpu")


@pytest.mark.parametrize("delayed_feedback", [False, True])
@pytest.mark.parametrize("curve", ["motor", "cap"])
def test_reused_bound_matches_original_for_dropped_held_commands(command_block, monkeypatch,
                                                                delayed_feedback, curve):
    control = controller(curve)
    # Actual transport loses every new packet, retaining the previous drive
    # command even while the physical wheel accelerates and reverses.
    transport = UsbCommandTransport(2, "cpu", .001, {
        "can_period_ms": 1., "delay_ms": [0., 0.], "enabled_fraction": 1.,
        "drop_probability_max": 1., "burst_probability": 0., "burst_packets": [1, 1],
    }, seed=791, profiles=[{"delay_ms": 0., "drop_probability": 1.}] * 2)
    transport.reset(torch.arange(2))
    held = torch.tensor([[11., -12., 3., 13., -14., -3.],
                         [-11., 12., -3., -13., 14., 3.]])
    transport.held.copy_(held)
    env = SimpleNamespace(v5=control, command_transport=transport,
                          usb_transport=object() if delayed_feedback else None,
                          _pc_feedback_transport=object() if delayed_feedback else None)
    original_curve = core.motor_torque_limit
    calls = []

    def counted_curve(speed, cfg):
        calls.append(speed.clone())
        return original_curve(speed, cfg)

    monkeypatch.setattr(core, "motor_torque_limit", counted_curve)
    q = torch.zeros(2, 6)
    legs, wheels = torch.zeros(2, 4), torch.full((2, 2), 20.)
    # Exercise nominal, narrowing and zero-capacity speed domains in both signs.
    for speed in (0., 30., 80., 1000., -80., -1000., 0.):
        physical_dq = torch.zeros_like(q)
        physical_dq[:, 2] = speed
        physical_dq[:, 5] = -speed
        feedback_dq = torch.zeros_like(q) if delayed_feedback else physical_dq
        env.torque = control.motor_efforts(q, feedback_dq, legs, wheels)
        expected_bound = original_curve(physical_dq[:, list(control.WHEELS)], control.wheel_prior)
        expected = held.clone()
        expected[:, list(control.WHEELS)] = expected[:, list(control.WHEELS)].clamp(
            -expected_bound, expected_bound)
        calls.clear()
        exec(command_block, {"self": env, "dq_now": physical_dq, "torch": torch})
        # Only calls introduced by the production command-transport block are
        # counted. motor_efforts still evaluates its own bound on every tick.
        assert len(calls) == int(delayed_feedback)
        if delayed_feedback:
            torch.testing.assert_close(calls[0], physical_dq[:, list(control.WHEELS)], atol=0, rtol=0)
        torch.testing.assert_close(env.torque, expected, atol=0, rtol=0)
        assert torch.equal(torch.signbit(env.torque), torch.signbit(expected))
        torch.testing.assert_close(transport.held, held, atol=0, rtol=0)
        assert (env.torque[:, list(control.WHEELS)].abs() <= expected_bound).all()
        if curve == "motor" and abs(speed) == 1000.:
            assert env.torque[:, list(control.WHEELS)].count_nonzero() == 0
    assert transport.lost.tolist() == [7., 7.]
    assert transport.tick == 7
