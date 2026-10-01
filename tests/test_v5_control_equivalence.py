"""The control fast path must preserve the frozen six-axis controller and ownership."""
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.v5_control import V5Control
from wheeled_tasks.v40.core import motor_torque_limit

ROOT = Path(__file__).resolve().parents[1]


def _controller(wheel_curve):
    read = lambda name: json.loads((ROOT / name).read_text())
    bundle = "model/纯底盘_v5/urdf/"
    prior = read("contracts/own_v40_v2.json")
    if wheel_curve == "output_cap":
        prior["actuators"]["wheel"].update(curve_side="wheel_output", effort_limit=4.5,
            curve_kind="software_cap_only_not_identified", scope="simulation_research_prior",
            torque_reference="wheel_output")
        for field in ("motor_torque_nm", "motor_speed_rad_s"):
            prior["actuators"]["wheel"].pop(field)
    settings = read("contracts/v5_foundation_v1.json")["v5_control"]
    settings.update(leg_kp=160., leg_kd=2.5)
    return V5Control(read(bundle + "manifest.json"), read(bundle + "model_spec.json"),
                     read(bundle + "fit_10mpa.json"), prior, settings, "cpu")


def _frozen_motor_efforts(control, q, dq, legs, wheels):
    """Unchanged pre-optimization implementation, including assignment/cast order."""
    tau = torch.zeros_like(q)
    requested = torch.zeros_like(q)
    requested[:, list(control.LEGS)] = (control.settings["leg_kp"] * (legs - q[:, list(control.LEGS)])
                                      - control.settings["leg_kd"] * dq[:, list(control.LEGS)])
    tau[:, list(control.LEGS)] = requested[:, list(control.LEGS)].clamp(-40., 40.)
    wheel = control.wheel_prior["kd"] * (wheels - dq[:, list(control.WHEELS)])
    bound = motor_torque_limit(dq[:, list(control.WHEELS)], control.wheel_prior)
    tau[:, list(control.WHEELS)] = torch.maximum(torch.minimum(wheel, bound), -bound)
    requested[:, list(control.WHEELS)] = wheel
    bounds = torch.full_like(q, 40.)
    bounds[:, list(control.WHEELS)] = bound
    return tau, requested, bounds


def _exact(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wheel_curve", ["motor", "output_cap"])
@pytest.mark.parametrize("strided", [False, True])
def test_multistep_random_control_matches_frozen_reference(dtype, wheel_curve, strided):
    control = _controller(wheel_curve)
    generator = torch.Generator().manual_seed(739)
    previous = None
    for tick in range(32):
        # Include USB-style row-strided feedback and non-contiguous target views.
        backing = torch.randn(73, 24, dtype=dtype, generator=generator)
        q, dq = backing[:, :12:2], backing[:, 12:24:2]
        target = torch.randn(73, 12, dtype=dtype, generator=generator)
        legs, wheels = target[:, :8:2], target[:, 8:12:2]
        q *= 20.
        dq *= (0., 0.1, 5., 1000.)[tick % 4]
        legs *= (0., 0.01, 1., 100.)[tick % 4]
        wheels *= (0., 0.01, 10., 1000.)[tick % 4]
        if not strided:
            q, dq, legs, wheels = (x.contiguous() for x in (q, dq, legs, wheels))
        inputs = tuple(x.clone() for x in (q, dq, legs, wheels))
        expected = _frozen_motor_efforts(control, q, dq, legs, wheels)
        actual = (control.motor_efforts(q, dq, legs, wheels),
                  control.requested_motor_effort, control.current_motor_bounds)
        for new, old in zip(actual, expected):
            _exact(new, old)
        for original, value in zip(inputs, (q, dq, legs, wheels)):
            _exact(value, original)
        if previous:
            for value, saved in previous:
                _exact(value, saved)
        # Downstream recovery/transport may write all three independently. The
        # next control tick must neither reuse them nor inherit those writes.
        actual[0].mul_(-0.31)
        _exact(actual[1], expected[1])
        _exact(actual[2], expected[2])
        actual[1].add_(7.)
        _exact(actual[2], expected[2])
        actual[2][:, 2::3] = 0.17
        previous = [(x, x.clone()) for x in actual]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wheel_curve", ["motor", "output_cap"])
def test_clipping_sign_reversal_and_motor_curve_breakpoints(dtype, wheel_curve):
    control = _controller(wheel_curve)
    speeds = torch.tensor([0., -0., 0.01, -0.01, 1000., -1000.], dtype=dtype)
    if wheel_curve == "motor":
        knots = torch.tensor(control.wheel_prior["motor_speed_rad_s"], dtype=dtype)
        knots /= control.wheel_prior["gear_ratio"]
        speeds = torch.cat((speeds, knots, -knots, knots + 1e-4, -knots - 1e-4))
    dq = torch.zeros(len(speeds), 6, dtype=dtype)
    dq[:, 2] = speeds
    dq[:, 5] = -speeds
    q = torch.zeros_like(dq)
    legs = torch.tensor([[-1., -.25, .25, 1.]], dtype=dtype).expand(len(speeds), -1)
    for scale in (-100., 0., 100.):
        wheels = torch.full((len(speeds), 2), scale, dtype=dtype)
        expected = _frozen_motor_efforts(control, q, dq, legs, wheels)
        actual = (control.motor_efforts(q, dq, legs, wheels),
                  control.requested_motor_effort, control.current_motor_bounds)
        for new, old in zip(actual, expected):
            _exact(new, old)


def test_nonfinite_wheel_speed_validation_is_preserved():
    control = _controller("output_cap")
    q = torch.zeros(2, 6)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        dq = torch.zeros_like(q)
        dq[1, 5] = invalid
        with pytest.raises(ValueError, match="finite floating tensor"):
            control.motor_efforts(q, dq, torch.zeros(2, 4), torch.zeros(2, 2))
