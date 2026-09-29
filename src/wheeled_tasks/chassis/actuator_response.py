"""Six-axis actuator response contract and post-transport torque dynamics.

Armature, frictionloss and damping belong to native physics. This module only
exports those values and applies an optional first-order motor torque response;
it must be called once per physics step, after USB command transport. All values
refer to active output axes, with no further gearbox or controller-gain scaling.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

SCHEMA = "chassis-actuator-response"
VERSION = 1
# Protocol identity; keep validation importable by the torch-free preflight.
# The runtime controller uses this same output order (V5Control.ACTIVE).
JOINT_NAMES = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
NATIVE_FIELDS = ("armature", "frictionloss", "damping")
VECTOR_FIELDS = (*NATIVE_FIELDS, "time_constant_s")
FIELDS = {"enabled", "schema", "version", "joint_names", "gain", *VECTOR_FIELDS}


def validate_actuator_response(config):
    """Return an independent normalized contract; missing configuration is off.

    An explicitly disabled full contract is still validated so that toggling
    enablement cannot conceal malformed parameters. Legacy absence and the short
    ``{'enabled': False}`` form require no schema or physical parameter defaults.
    """
    if config is None:
        return {"enabled": False}
    if not isinstance(config, Mapping):
        raise ValueError("Actuator response must be a configuration mapping")
    if not config:
        return {"enabled": False}
    if type(config.get("enabled")) is not bool:
        raise ValueError("Actuator response enabled must be an explicit boolean")
    if set(config) == {"enabled"} and config["enabled"] is False:
        return {"enabled": False}
    if set(config) != FIELDS:
        missing, unknown = FIELDS - set(config), set(config) - FIELDS
        raise ValueError(f"Actuator response fields differ: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if config["schema"] != SCHEMA or type(config["version"]) is not int or config["version"] != VERSION:
        raise ValueError("Unsupported actuator response schema/version")
    joints = config["joint_names"]
    if not isinstance(joints, (list, tuple)) or tuple(joints) != JOINT_NAMES:
        raise ValueError("Actuator response requires canonical active joint order")
    gain = config["gain"]
    if type(gain) not in (int, float) or not math.isfinite(gain) or gain != 1.:
        raise ValueError("Actuator response gain is fixed at 1; torque scale is not identified here")
    validated = {"enabled": config["enabled"], "schema": SCHEMA, "version": VERSION,
                 "joint_names": list(JOINT_NAMES), "gain": 1.}
    for field in VECTOR_FIELDS:
        values = config[field]
        if not isinstance(values, (list, tuple)) or len(values) != len(JOINT_NAMES):
            raise ValueError(f"Actuator response {field} requires exactly six values")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Actuator response {field} must be finite and nonnegative")
        validated[field] = [float(value) for value in values]
    return validated


class ActuatorResponse:
    """Batched motor state, independent of Isaac and native passive forces.

    ``apply_torque`` accepts an N x 6 post-USB tensor in ``JOINT_NAMES`` order.
    Its result uses an internal reusable buffer when any axis has lag; callers
    retaining historical results should clone them. Native losses are never
    subtracted here, preventing a second application after the physics engine.
    """

    def __init__(self, count, device, dt, config=None, *, dtype=None):
        import torch

        self._torch = torch
        dtype = torch.float32 if dtype is None else dtype
        self.cfg = validate_actuator_response(config)
        if type(count) is not int or count <= 0:
            raise ValueError("Actuator response environment count must be a positive integer")
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Actuator response physics dt must be finite and positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Actuator response state requires float32 or float64")
        self.enabled, self.dt = self.cfg["enabled"], float(dt)
        self.device, self.dtype = torch.device(device), dtype
        self.state, self.alpha, self.direct_axes = None, None, None
        self.has_lag = False
        if not self.enabled:
            return
        self.state = torch.zeros(count, len(JOINT_NAMES), device=self.device, dtype=dtype)
        constants = self.cfg["time_constant_s"]
        self.has_lag = any(value > 0 for value in constants)
        # expm1 retains precision when dt is much smaller than the time constant.
        alpha = [1. if value == 0 else -math.expm1(-self.dt / value) for value in constants]
        self.alpha = self.state.new_tensor(alpha)[None, :]
        self.direct_axes = self.state.new_tensor([value == 0 for value in constants], dtype=torch.bool)[None, :]

    def native_parameters(self):
        """Return per-joint physical values; disabled response exports no override."""
        if not self.enabled:
            return {}
        return {field: dict(zip(JOINT_NAMES, self.cfg[field])) for field in NATIVE_FIELDS}

    def apply_torque(self, post_usb_torque):
        if not self.enabled:
            return post_usb_torque
        if (post_usb_torque.shape != self.state.shape or post_usb_torque.device != self.state.device
                or post_usb_torque.dtype != self.state.dtype):
            raise ValueError("Actuator response torque shape/device/dtype differs from its state")
        with self._torch.no_grad():
            if not self.has_lag:
                self.state.copy_(post_usb_torque)
                return post_usb_torque
            updated = self.state + self.alpha * (post_usb_torque - self.state)
            # Zero time constant is exactly direct, including cancellation-sensitive
            # values; it is not approximated with a very small positive constant.
            self.state.copy_(self._torch.where(self.direct_axes, post_usb_torque, updated))
        return self.state

    def reset(self, ids=None):
        """Clear all or selected environments' motor state at an episode reset."""
        if not self.enabled:
            return
        with self._torch.no_grad():
            if ids is None:
                self.state.zero_()
            else:
                self.state[ids] = 0.

    def runtime_probe(self):
        """Verify actual update/reset code in isolated small instances.

        The training instance's state is never sampled, advanced or reset. This
        is a deterministic functional check, not evidence of identified physics.
        """
        torch = self._torch
        probe = ActuatorResponse(2, self.device, self.dt, self.cfg, dtype=self.dtype)
        command = torch.tensor([[1., -2., 3., -4., 5., -6.], [-1., 2., -3., 4., -5., 6.]],
                               device=self.device, dtype=self.dtype)
        constants = self.cfg["time_constant_s"] if self.enabled else [0.] * len(JOINT_NAMES)
        expected = [command * command.new_tensor([
            1. if tau == 0 else -math.expm1(-step * self.dt / tau) for tau in constants
        ])[None, :] for step in (1, 2)]
        first = probe.apply_torque(command).clone()
        second = probe.apply_torque(command).clone()
        checks = {"first_step_matches": torch.allclose(first, expected[0], rtol=1e-6, atol=1e-7),
                  "second_step_matches": torch.allclose(second, expected[1], rtol=1e-6, atol=1e-7)}
        probe.reset([0])
        if self.enabled:
            checks["selected_reset_zero"] = bool(torch.count_nonzero(probe.state[0]).item() == 0)
            checks["other_environment_preserved"] = torch.equal(probe.state[1], second[1])
        else:
            checks["disabled_identity"] = probe.apply_torque(command) is command
            checks["disabled_has_no_state"] = probe.state is None
        probe.reset()
        checks["response_after_reset_matches"] = torch.allclose(
            probe.apply_torque(command), expected[0], rtol=1e-6, atol=1e-7)
        # Exercise zero time constants on every axis even when the configured
        # fit has positive constants on all axes; no vacuous zero-axis check.
        direct_cfg = {"enabled": True, "schema": SCHEMA, "version": VERSION,
                      "joint_names": list(JOINT_NAMES), "gain": 1.,
                      **{field: [0.] * len(JOINT_NAMES) for field in VECTOR_FIELDS}}
        direct = ActuatorResponse(2, self.device, self.dt, direct_cfg, dtype=self.dtype)
        direct.state.fill_(1e20)
        checks["zero_constant_exact_identity"] = direct.apply_torque(command) is command
        if not all(checks.values()):
            raise RuntimeError(f"Actuator response runtime probe failed: {checks}")
        return {"enabled": self.enabled, "schema": self.cfg.get("schema"),
                "version": self.cfg.get("version"), "joint_names": list(JOINT_NAMES),
                "physics_dt": self.dt, "parameters": validate_actuator_response(self.cfg),
                "checks": checks, "passed": True,
                "expected_first": expected[0].cpu().tolist(), "observed_first": first.cpu().tolist(),
                "expected_second": expected[1].cpu().tolist(), "observed_second": second.cpu().tolist()}
