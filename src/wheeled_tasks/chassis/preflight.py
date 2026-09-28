"""Fail-closed design checks and instrumentation for explicit substep control.

The design table is an independent, hash-pinned input, never reconstructed from
the materialized contract. This module has no Isaac or torch dependency.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path


ROLES = ("train", "baseline", "quick", "full", "confirmation")
GAS_SPRING_MODEL_KINDS = frozenset(("v5_gas_spring_closedchain_research",
                                    "v6_gas_spring_closedchain_research"))
REQUIRED_CONTRACT_PATHS = (
    "curriculum_stage", "enabled_stages", "stages", "evaluation",
    "physics_dt", "policy_dt", "v5_control", "usb_transport", "command_transport",
    "precision_tracking", "signal_perturbations", "dynamics_randomization",
    "contact_domain", "step_assist", "command_reference", "reference_reward",
    "reward_profile", "actor_dim", "critic_dim", "action_dim", "history_length",
    "actor_layout", "critic_layout", "asset_manifest_sha256", "target_num_envs",
    "num_steps_per_env", "num_mini_batches", "num_learning_epochs", "total_updates",
    "evaluation.seed", "evaluation.confirmation_seed",
)
MODULES = ("usb_transport", "command_transport", "signal_perturbations",
           "dynamics_randomization", "contact_domain", "step_assist")
_MISSING = object()


def _at(mapping, path):
    for key in path.split("."):
        if not isinstance(mapping, dict) or key not in mapping:
            return _MISSING
        mapping = mapping[key]
    return mapping


def _display(value):
    return "<missing>" if value is _MISSING else repr(value)


def _compare(expected, actual, path, errors):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            _compare(expected.get(key, _MISSING), actual.get(key, _MISSING),
                     f"{path}.{key}", errors)
    elif (expected is _MISSING or actual is _MISSING
          or type(expected) is not type(actual) and not (
              type(expected) in (int, float) and type(actual) in (int, float))
          or expected != actual):
        errors.append(f"{path}: design={_display(expected)}, effective={_display(actual)}")


def control_decimation(physics_dt, policy_dt):
    """Reject silent rounding of a non-integral or invalid control clock."""
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (physics_dt, policy_dt)):
        raise ValueError("physics_dt and policy_dt must be finite positive seconds")
    ratio = policy_dt / physics_dt
    decimation = round(ratio)
    if decimation < 1 or not math.isclose(ratio, decimation, rel_tol=0., abs_tol=1e-9):
        raise ValueError("policy_dt must be an integer multiple of physics_dt")
    return decimation


def validate_design_contract(config, design, control, *, stage, role, seed,
                             num_envs, block_index=0, source="independent design table"):
    """Check one final effective stage/role after CLI and sidecar overrides.

    Each role declares full values for the required mapping-valued fields. An
    explicitly disabled feature is still present, with ``enabled: false``.
    Additional contract paths can be frozen without changing the validator.
    """
    errors = []
    if design.get("schema") != "chassis-design-expectations-v1" or design.get("status") != "frozen":
        raise ValueError(f"{source}: a frozen chassis-design-expectations-v1 table is required")
    if role not in ROLES:
        raise ValueError(f"{source}: unsupported run role {role!r}")
    row = _at(design, f"stages.{stage}.roles.{role}")
    if not isinstance(row, dict):
        raise ValueError(f"{source}: missing design row for stage={stage}, role={role}")
    expected = row.get("contract", {})
    for path in REQUIRED_CONTRACT_PATHS:
        if path not in expected:
            errors.append(f"design.{path}: explicit expectation is required")
    for path, value in expected.items():
        _compare(value, _at(config, path), f"contract.{path}", errors)
    for name in MODULES:
        value = _at(config, name + ".enabled")
        if type(value) is not bool:
            errors.append(f"contract.{name}.enabled: explicit boolean is required")
    if _at(config, "usb_transport.enabled") is True and _at(config, "command_transport.enabled") is True:
        errors.append("contract: legacy usb_transport and command_transport are mutually exclusive")
    control_expected = row.get("control", {})
    for path in ("actuators.wheel",):
        if path not in control_expected:
            errors.append(f"design.control.{path}: explicit expectation is required")
    for path, value in control_expected.items():
        _compare(value, _at(control, path), f"control.{path}", errors)
    invocation = row.get("invocation", {})
    _compare(invocation.get("num_envs", _MISSING), num_envs, "invocation.num_envs", errors)
    seed_spec = invocation.get("seed", {})
    if seed_spec.get("rule") == "root_plus_block_index":
        root_seed = seed_spec.get("root_seed")
        if type(root_seed) is not int or type(block_index) is not int or block_index < 0:
            errors.append("invocation.seed: integer root_seed and nonnegative block_index required")
        else:
            _compare(root_seed + block_index, seed, "invocation.seed", errors)
    elif seed_spec.get("rule") == "fixed" and type(seed_spec.get("value")) is int:
        _compare(seed_spec["value"], seed, "invocation.seed", errors)
    else:
        errors.append("design.invocation.seed: explicit fixed or root_plus_block_index rule required")
    if role != "train":
        _compare(_at(config, "evaluation.seed"), seed, "invocation.evaluation_seed", errors)
    try:
        decimation = control_decimation(config.get("physics_dt"), config.get("policy_dt"))
    except ValueError as error:
        errors.append(str(error))
        decimation = None
    # This schema is deliberately the new 1 kHz PC / 50 Hz target reference,
    # not a relabeling of legacy transport impulse approximations.
    for path, value in (("physics_dt", .001), ("policy_dt", .02),
                        ("v5_control.leg_kp", 160.), ("v5_control.leg_kd", 2.5)):
        _compare(value, _at(config, path), f"reference.{path}", errors)
    _compare(.6, _at(control, "actuators.wheel.kd"), "reference.wheel_velocity_p", errors)
    if not isinstance(row.get("runtime"), dict):
        errors.append("design.runtime: explicit runtime expectations required")
    if errors:
        raise ValueError(f"Preflight rejected ({source}; stage={stage}, role={role}):\n"
                         + "\n".join(errors))
    runtime_expected = deepcopy(row["runtime"])
    if runtime_expected.get("seed") == {"source": "effective_seed"}:
        runtime_expected["seed"] = seed
    return {"status": "static_passed", "source": source, "stage": stage, "role": role,
            "seed": seed, "block_index": block_index, "num_envs": num_envs,
            "physics_dt": config["physics_dt"], "policy_dt": config["policy_dt"],
            "decimation": decimation, "runtime_expected": runtime_expected}


def design_preflight(config, control, root, *, stage, role, seed, num_envs, block_index=None):
    """Shared launch/environment hook; legacy contracts remain explicitly unverified.

    A new executable contract must declare ``requires_design_preflight: true``
    and a ``design_preflight`` binding with file, sha256 and design stage. Callers
    must call again on their effective role config after all overrides.
    """
    binding = config.get("design_preflight")
    if binding is None:
        if config.get("requires_design_preflight"):
            raise ValueError("Required independent design_preflight binding is missing")
        return None
    if not isinstance(binding, dict):
        raise ValueError("design_preflight must be a file/hash/stage binding")
    path = Path(root) / binding["file"]
    contents = path.read_bytes()
    actual_sha = hashlib.sha256(contents).hexdigest()
    if actual_sha != binding["sha256"]:
        raise ValueError(f"Independent design table SHA mismatch: {path}")
    if "control_math_sha256" in config:
        control_path = Path(root) / config["control_math_source"]
        if hashlib.sha256(control_path.read_bytes()).hexdigest() != config["control_math_sha256"]:
            raise ValueError(f"Control prior SHA mismatch: {control_path}")
    receipt = validate_design_contract(config, json.loads(contents), control,
        stage=binding["stage"], role=role, seed=seed, num_envs=num_envs,
        block_index=binding.get("block_index", 0) if block_index is None else block_index,
        source=str(path))
    if stage not in config["enabled_stages"] or stage not in {item["name"] for item in config["stages"]}:
        raise ValueError(f"Environment stage {stage!r} is outside the independently verified contract")
    receipt.update(design_sha256=actual_sha, environment_stage=stage)
    return receipt


def validate_runtime_state(expected, actual):
    """Compare observed instances/clocks, without claiming a functional probe passed."""
    required = {"physics_dt", "policy_dt", "decimation", "pc_control_dt", "seed",
                "modules", "leg_kp", "leg_kd", "wheel_velocity_p", "precision_tracking"}
    errors = [f"runtime.{key}: explicit expectation required" for key in sorted(required - expected.keys())]
    for path, value in expected.items():
        _compare(value, _at(actual, path), "runtime." + path, errors)
    if errors:
        raise ValueError("Runtime preflight rejected:\n" + "\n".join(errors))
    return {"status": "runtime_configuration_passed", "observed": actual,
            "functional_transport_precision_randomization_probes": "not_run"}


class SubstepControlAudit:
    """Count calls around the actual PD/solver/feedback-update execution path."""

    def __init__(self, physics_dt, policy_dt):
        self.physics_dt = physics_dt
        self.policy_dt = policy_dt
        self.decimation = control_decimation(physics_dt, policy_dt)
        self.phase = "idle"
        self.policy_steps = self.physics_steps = 0
        self.last = None

    def begin_policy(self):
        if self.phase != "idle":
            raise RuntimeError("Previous policy step did not complete its control substeps")
        self.controls = self.refreshes = self.decodes = 0
        self.phase = "targets"

    def targets_decoded(self):
        if self.phase != "targets":
            raise RuntimeError("Targets must be decoded exactly once per policy step")
        self.decodes += 1
        self.phase = "feedback"

    def feedback_control(self):
        if self.phase != "feedback" or self.controls >= self.decimation:
            raise RuntimeError("PD requires one refreshed physical state per substep")
        self.controls += 1
        self.phase = "physics"

    def physics_updated(self, dt):
        if self.phase != "physics" or dt != self.physics_dt:
            raise RuntimeError("PD must be followed by one solver step and feedback refresh")
        self.refreshes += 1
        self.physics_steps += 1
        self.phase = "feedback"

    def end_policy(self):
        if self.phase != "feedback" or (self.decodes, self.controls, self.refreshes) != (1, self.decimation, self.decimation):
            raise RuntimeError("Policy step did not execute the declared PD/physics decimation")
        self.policy_steps += 1
        self.phase = "idle"
        self.last = {"policy_steps": self.policy_steps, "physics_steps": self.physics_steps,
                     "target_decodes": self.decodes, "pd_updates": self.controls,
                     "feedback_refreshes": self.refreshes, "physics_dt": self.physics_dt,
                     "policy_dt": self.policy_dt}
        return self.last
