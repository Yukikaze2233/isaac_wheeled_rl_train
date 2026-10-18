"""Six-axis actuator response contract and post-transport torque dynamics.

Armature, frictionloss and damping belong to native physics. This module only
exports those values and applies an optional first-order motor torque response.
The legacy update takes a post-transport torque; the opt-in interval update
integrates the actual two delayed command segments once per physics step. All values
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


def merge_wheel_response(base_config, candidate, *, cad_inertia_by_joint):
    """Replace two wheel responses in a complete enabled six-axis contract.

    The v2 candidate is a wheel-output, ratio-15.62 research fit, not a training
    or hardware qualification. ``cad_inertia_by_joint`` must contain the two
    named wheel-axis CAD inertias, excluding any old additional armature; the
    caller owns the provenance of that readback. Algebraic checks here do not
    establish that a physics engine or a particular asset was measured.

    Four leg responses (including lag) are preserved. Wheel native parameters
    replace their old values without gearbox scaling or software loss terms.
    Explicitly unidentified wheel lag becomes zero *additional modeled lag*,
    not a claim of measured zero latency. No gain, PD or clock is inferred.
    All inputs remain unchanged; the returned normalized contract is detached.
    """
    merged = validate_actuator_response(base_config)
    if not merged["enabled"]:
        raise ValueError("Wheel merge requires a complete enabled six-axis base contract")

    required = {"schema", "scope", "coordinate", "wheel_reduction_ratio",
                "reduction_ratio_already_applied", "time_constant_identified", "axes"}
    # Provenance metadata from the current candidate is accepted but never
    # converted into qualification or additional actuator behavior.
    metadata = {"status", "fit_sha256", "commanded_speed_range_rad_s",
                "fit_min_abs_speed_rad_s", "loaded_torque_speed_envelope_identified",
                "static_breakaway_identified", "source_run", "instruction",
                "native_validation_file", "native_validation_report",
                "native_parameter_readback_verified", "friction_mapping_for_isaac"}
    if not isinstance(candidate, Mapping):
        raise ValueError("Wheel response candidate must be a mapping")
    if not required <= set(candidate) or set(candidate) - required - metadata:
        raise ValueError("Wheel response candidate has missing or unknown fields")
    if (candidate["schema"] != "wheel_response_candidate_v2"
            or candidate["scope"] != "nominal_effective_inertia_and_moving_resistance"
            or candidate["coordinate"] != "wheel_output"):
        raise ValueError("Unsupported wheel response candidate schema/scope/coordinate")
    ratio = candidate["wheel_reduction_ratio"]
    if (type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio != 15.62
            or candidate["reduction_ratio_already_applied"] is not True):
        raise ValueError("Wheel response requires ratio 15.62 already applied")
    if candidate["time_constant_identified"] is not False:
        raise ValueError("Wheel response requires explicit unidentified time constant; new lag is unsupported")
    for field in ("loaded_torque_speed_envelope_identified", "static_breakaway_identified"):
        if field in candidate and candidate[field] is not False:
            raise ValueError(f"Wheel response does not support {field}")

    wheels = ("L_joint3", "R_joint3")
    axes = candidate["axes"]
    if not isinstance(axes, Mapping) or set(axes) != set(wheels):
        raise ValueError("Wheel response axes must be exactly L_joint3 and R_joint3")
    if not isinstance(cad_inertia_by_joint, Mapping) or set(cad_inertia_by_joint) != set(wheels):
        raise ValueError("CAD inertia readback must name exactly L_joint3 and R_joint3")
    axis_fields = {"effective_total_inertia_kg_m2", "cad_inertia_kg_m2",
                   "additional_armature_kg_m2", "coulomb_positive_nm", "coulomb_negative_nm",
                   "symmetric_coulomb_approximation_nm", "viscous_nm_per_rad_s"}

    def number(value, label, *, positive=False):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (positive and value == 0)):
            raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
        return float(value)

    for joint in wheels:
        row = axes[joint]
        if not isinstance(row, Mapping) or set(row) != axis_fields:
            raise ValueError(f"Wheel response {joint} has missing or unknown fields")
        values = {key: number(value, f"{joint}.{key}", positive=key in {
            "effective_total_inertia_kg_m2", "cad_inertia_kg_m2"}) for key, value in row.items()}
        cad = number(cad_inertia_by_joint[joint], f"{joint} CAD readback", positive=True)
        # Match the existing wheel replay's 1e-10 absolute inertia tolerance.
        # This is floating-point consistency, not a physical acceptance limit.
        if not math.isclose(cad, values["cad_inertia_kg_m2"], rel_tol=0., abs_tol=1e-10):
            raise ValueError(f"Wheel response {joint} CAD inertia differs from readback")
        if not math.isclose(values["effective_total_inertia_kg_m2"],
                            values["cad_inertia_kg_m2"] + values["additional_armature_kg_m2"],
                            rel_tol=0., abs_tol=1e-10):
            raise ValueError(f"Wheel response {joint} total inertia must equal CAD plus additional armature")
        # Divide before summing to avoid overflow for finite input losses.
        symmetric = values["coulomb_positive_nm"] / 2 + values["coulomb_negative_nm"] / 2
        if not math.isclose(values["symmetric_coulomb_approximation_nm"], symmetric,
                            rel_tol=0., abs_tol=1e-10):
            raise ValueError(f"Wheel response {joint} symmetric Coulomb loss must equal mean(C+, C-)")
        index = JOINT_NAMES.index(joint)
        merged["armature"][index] = values["additional_armature_kg_m2"]
        merged["frictionloss"][index] = values["symmetric_coulomb_approximation_nm"]
        merged["damping"][index] = values["viscous_nm_per_rad_s"]
        merged["time_constant_s"][index] = 0.
    return validate_actuator_response(merged)


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
        self.enabled, self.dt, self.count = self.cfg["enabled"], float(dt), count
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
        self._interval_constants = self.state.new_tensor(constants, dtype=torch.float64)[None, :]

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

    def apply_delayed_torque(self, previous_command, command, delay_s, *, effort_bounds=None):
        """Integrate delayed commands through the drive and return mean effort.

        The previous command acts for ``delay_s`` and the new command for the
        remainder of this physical interval. ``state`` retains the true drive
        endpoint, independently of the returned interval mean. No loss, extra
        delay or physical substep is introduced. The legacy update is unchanged.

        Commands and optional bounds are [N,6] in this instance's dtype/device;
        delay is a float32/float64 [N] tensor on the same device. The transport
        owner validates finite delays in [0,dt] and the physical owner validates
        positive finite bounds. No device-to-host validation occurs here.
        Bounds constrain both held command segments and the initial drive state,
        treating the current native envelope as constant within this interval.
        Inputs are never modified; the returned tensor does not alias ``state``.
        """
        torch = self._torch
        expected_device = self.state.device if self.enabled else self.device
        for value in (previous_command, command):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.count, len(JOINT_NAMES))
                    or value.dtype != self.dtype or value.device.type != expected_device.type
                    or expected_device.index is not None and value.device.index != expected_device.index):
                raise ValueError("Delayed response command shape/device/dtype differs from its state")
        if previous_command.device != command.device:
            raise ValueError("Delayed response commands require the same device")
        if (not isinstance(delay_s, torch.Tensor) or delay_s.shape != (self.count,)
                or delay_s.device != command.device or delay_s.dtype not in (torch.float32, torch.float64)):
            raise ValueError("Delayed response delay requires a floating [N] tensor on the command device")
        if effort_bounds is not None and (not isinstance(effort_bounds, torch.Tensor)
                or effort_bounds.shape != command.shape or effort_bounds.device != command.device
                or effort_bounds.dtype != command.dtype):
            raise ValueError("Delayed response effort bounds must match the command shape/device/dtype")
        with torch.no_grad():
            previous, current = previous_command, command
            if effort_bounds is not None:
                previous = previous.clamp(-effort_bounds, effort_bounds)
                current = current.clamp(-effort_bounds, effort_bounds)
            fraction = (delay_s / self.dt).to(command.dtype)[:, None]
            direct_mean = current + fraction * (previous - current)
            if not self.enabled:
                return (direct_mean if effort_bounds is None
                        else direct_mean.clamp(-effort_bounds, effort_bounds))
            endpoint = self.state.double()
            if effort_bounds is not None:
                endpoint = endpoint.clamp(-effort_bounds.double(), effort_bounds.double())
            constants = self._interval_constants
            safe_constants = torch.where(self.direct_axes, 1., constants)
            impulse = torch.zeros_like(endpoint)
            duration = delay_s.double()[:, None]
            for piece, seconds in ((previous, duration), (current, self.dt - duration)):
                ratio = seconds / safe_constants
                beta = -torch.expm1(-ratio)
                # The small-ratio series resolves a tiny interval impulse
                # without subtracting nearly equal duration and tau*beta.
                safe_ratio = torch.where(ratio == 0., 1., ratio)
                series = ratio * (.5 + ratio * (-1. / 6. + ratio * (
                    1. / 24. + ratio * (-1. / 120. + ratio / 720.))))
                small = ratio.abs() < 1e-3
                input_weight = torch.where(small, series, 1. - beta / safe_ratio)
                state_weight = torch.where(small, 1. - series, beta / safe_ratio)
                piece = piece.double()
                mean = torch.where(self.direct_axes, piece,
                    state_weight * endpoint + input_weight * piece)
                impulse += seconds * mean
                advanced = torch.where(self.direct_axes, piece,
                    torch.exp(-ratio) * endpoint + beta * piece)
                # A zero-length segment cannot replace the physical endpoint,
                # including on axes with an instantaneous response.
                endpoint = torch.where(seconds > 0., advanced, endpoint)
            mean = torch.where(self.direct_axes, direct_mean, (impulse / self.dt).to(command.dtype))
            endpoint = endpoint.to(command.dtype)
            if effort_bounds is not None:
                mean = mean.clamp(-effort_bounds, effort_bounds)
                endpoint = endpoint.clamp(-effort_bounds, effort_bounds)
            self.state.copy_(endpoint)
            return mean

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
