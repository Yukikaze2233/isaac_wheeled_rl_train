"""CPU contract and dynamical checks for post-USB motor torque response."""
from copy import deepcopy
import math
import subprocess
import sys

import pytest
import torch

from wheeled_tasks.chassis.actuator_response import (
    ActuatorResponse, JOINT_NAMES, SCHEMA, validate_actuator_response,
)


def configuration():
    return {"enabled": True, "schema": SCHEMA, "version": 1,
            "joint_names": list(JOINT_NAMES), "gain": 1.,
            "armature": [0., 0., .0038669808882082915, 0., 0., .003818453742910773],
            "frictionloss": [0., 0., .04950740077966313, 0., 0., .06957712078102896],
            "damping": [0., 0., .0006085318417485766, 0., 0., .0003429025559035852],
            "time_constant_s": [.002, .004, 0., .006, .008, 0.]}


def test_exact_constant_command_response_and_zero_time_axes():
    cfg = configuration()
    response = ActuatorResponse(2, "cpu", .001, cfg, dtype=torch.float64)
    command = torch.tensor([[2., -3., 4., -5., 6., -7.], [-2., 3., -4., 5., -6., 7.]], dtype=torch.float64)
    for _ in range(17):
        applied = response.apply_torque(command)
    factors = torch.tensor([1 - math.exp(-.017 / value) if value else 1. for value in cfg["time_constant_s"]], dtype=torch.float64)
    torch.testing.assert_close(applied, command * factors, rtol=1e-12, atol=1e-12)
    assert torch.equal(applied[:, [2, 5]], command[:, [2, 5]])


def test_exact_update_has_same_result_when_time_step_is_subdivided():
    fine = ActuatorResponse(3, "cpu", .0005, configuration(), dtype=torch.float64)
    coarse = ActuatorResponse(3, "cpu", .001, configuration(), dtype=torch.float64)
    command = torch.arange(18, dtype=torch.float64).reshape(3, 6) - 8
    for _ in range(10):
        fine.apply_torque(command)
    for _ in range(5):
        coarse.apply_torque(command)
    torch.testing.assert_close(fine.state, coarse.state, rtol=1e-12, atol=1e-12)


def test_response_is_driven_by_post_transport_torque_without_changing_command():
    response = ActuatorResponse(2, "cpu", .001, configuration())
    post_usb = torch.tensor([[1., -1., 2., -2., 3., -3.]]).expand(2, -1).clone()
    original = post_usb.clone()
    applied = response.apply_torque(post_usb)
    torch.testing.assert_close(applied, post_usb * response.alpha)
    assert torch.equal(post_usb, original)
    # Falling command retains motor state rather than replacing it with USB or
    # target history, so response cannot be applied at policy-update cadence.
    previous = applied.clone()
    applied = response.apply_torque(torch.zeros_like(post_usb))
    torch.testing.assert_close(applied, previous * (1 - response.alpha))


def test_native_loss_parameters_are_named_and_not_subtracted_twice():
    cfg = configuration()
    cfg["time_constant_s"] = [0.] * 6
    response = ActuatorResponse(1, "cpu", .001, cfg)
    parameters = response.native_parameters()
    assert tuple(parameters["armature"]) == JOINT_NAMES
    assert parameters["armature"]["L_joint3"] == cfg["armature"][2]
    assert parameters["frictionloss"]["R_joint3"] == cfg["frictionloss"][5]
    command = torch.full((1, 6), 1.)
    assert response.apply_torque(command) is command
    parameters["damping"]["L_joint3"] = 100.
    assert response.native_parameters()["damping"]["L_joint3"] == cfg["damping"][2]


def test_subset_reset_does_not_destroy_other_environment_histories():
    response = ActuatorResponse(4, "cpu", .001, configuration())
    response.apply_torque(torch.full((4, 6), 2.))
    before = response.state.clone()
    response.reset(torch.tensor([1, 3]))
    assert torch.equal(response.state[[0, 2]], before[[0, 2]])
    assert torch.count_nonzero(response.state[[1, 3]]) == 0
    response.reset(torch.tensor([], dtype=torch.long))
    assert torch.equal(response.state[[0, 2]], before[[0, 2]])
    response.reset()
    assert torch.count_nonzero(response.state) == 0


@pytest.mark.parametrize("config", [None, {}, {"enabled": False}, {**configuration(), "enabled": False}])
def test_legacy_disabled_is_identity_and_does_not_export_physics_overrides(config):
    response = ActuatorResponse(2, "cpu", .001, config)
    command = torch.randn(2, 6)
    assert response.apply_torque(command) is command
    assert response.native_parameters() == {}
    assert response.state is None


@pytest.mark.parametrize("field", ["armature", "frictionloss", "damping", "time_constant_s"])
@pytest.mark.parametrize("value", [-.001, float("inf"), float("nan"), True, "0.001"])
def test_invalid_physical_parameters_rejected(field, value):
    cfg = configuration()
    cfg[field][1] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        validate_actuator_response(cfg)


@pytest.mark.parametrize("change", [
    {"joint_names": ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"]},
    {"joint_names": "canonical"}, {"armature": [0.] * 5}, {"damping": 0.},
    {"gain": .9}, {"gain": True}, {"gain": float("nan")}, {"schema": "unknown"},
    {"version": 2}, {"version": True}, {"version": 1.}, {"enabled": 1},
    {"motor_lag_typo": .005},
])
def test_contract_identity_shape_and_unknown_fields_rejected(change):
    with pytest.raises(ValueError):
        validate_actuator_response({**configuration(), **change})


def test_missing_enablement_and_missing_parameters_are_not_defaulted():
    for field in configuration():
        cfg = configuration()
        del cfg[field]
        with pytest.raises(ValueError):
            validate_actuator_response(cfg)


def test_validation_does_not_alias_caller_configuration():
    original = configuration()
    expected = deepcopy(original)
    validated = validate_actuator_response(original)
    validated["armature"][2] = 100.
    assert original == expected


@pytest.mark.parametrize("dt", [0., -.001, float("nan"), float("inf"), True])
def test_invalid_physics_time_step_rejected(dt):
    with pytest.raises(ValueError):
        ActuatorResponse(2, "cpu", dt, configuration())


def test_torque_shape_and_dtype_must_match_canonical_state():
    response = ActuatorResponse(2, "cpu", .001, configuration())
    for command in (torch.zeros(2, 4), torch.zeros(3, 6), torch.zeros(2, 6, dtype=torch.float64)):
        with pytest.raises(ValueError, match="shape/device/dtype"):
            response.apply_torque(command)


def test_runtime_probe_verifies_two_steps_without_touching_live_state():
    response = ActuatorResponse(7, "cpu", .001, configuration())
    response.apply_torque(torch.full((7, 6), 8.))
    before = response.state.clone()
    receipt = response.runtime_probe()
    assert receipt["passed"] and all(receipt["checks"].values())
    assert receipt["parameters"] == configuration()
    torch.testing.assert_close(torch.tensor(receipt["observed_second"]),
                               torch.tensor(receipt["expected_second"]))
    assert torch.equal(response.state, before)


def test_runtime_probe_disabled_legacy_is_small_and_identity():
    response = ActuatorResponse(20_000, "cpu", .001, None)
    receipt = response.runtime_probe()
    assert receipt["checks"]["disabled_identity"]
    assert receipt["checks"]["disabled_has_no_state"]
    assert receipt["checks"]["zero_constant_exact_identity"]
    assert response.state is None


def test_preflight_validator_imports_without_torch_or_isaac():
    code = """
import sys
from wheeled_tasks.chassis.actuator_response import validate_actuator_response
assert validate_actuator_response(None) == {'enabled': False}
assert 'torch' not in sys.modules
assert not any(name.startswith(('isaaclab', 'omni')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_protocol_order_matches_runtime_controller():
    from wheeled_tasks.chassis.v5_control import V5Control
    assert JOINT_NAMES == V5Control.ACTIVE
