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
NEW_ASSET_CONTRACT_PATHS = (
    "curriculum_plan_id", "budget_stage_id", "budget_spec_file", "budget_spec_sha256",
    "updates_are_actual_not_reference_scaled", "rollback_refunds_consumed_updates",
    "asset_directory", "control_math_source", "control_math_sha256", "actor_frame_dim",
    "actor_observation_source", "policy_action_order", "policy_architecture",
    "ppo_minibatch_samples", "curriculum_reference_batch", "dense_tracking", "jump_assist",
    "new_asset_curriculum", "scene_groups", "skill_specs", "behavior_pool_membership",
    "behavior_pool_fractions", "terrain_limits", "terrain_difficulty_tiers", "motion_limits",
    "task_semantics", "height_range_m", "height_reference", "height_workspace", "command_slew",
    "reward_velocity_reference", "termination_event_cost", "working_margin_weight",
    "height_l1_weight", "zero_command_velocity_scale", "scut_effort_reward_scale",
    "learning_rate", "learning_rate_schedule", "critic_warmup_updates", "transfer_critic",
    "position_iterations", "velocity_iterations", "closure_gap_termination_m",
    "evaluation_exact_cases", "flat_floor_width_m", "flat_triangle_mesh", "flat_half_length_m",
    "solid_step_platforms", "fixed_evaluation_terrain", "terrain_reset_before_entry",
    "step_contact_grace", "manual_context35", "observation_noise_enabled", "fall_confirmation_seconds",
    "research_only", "research_training_authorized", "training_release", "hardware_deployment_ready",
    "physics_prior_scope", "release_scope", "episode_seconds",
)
NEW_ASSET_RUNTIME_MODULES = ("jump_assist", "training_schedule")
_MISSING = object()


def _at(mapping, path):
    for key in path.split("."):
        if not isinstance(mapping, dict) or key not in mapping:
            return _MISSING
        mapping = mapping[key]
    return mapping


def _display(value):
    rendered = "<missing>" if value is _MISSING else repr(value)
    return rendered if len(rendered) <= 500 else rendered[:500] + "...<truncated>"


def _compare(expected, actual, path, errors):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            _compare(expected.get(key, _MISSING), actual.get(key, _MISSING),
                     f"{path}.{key}", errors)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append(f"{path}: design length={len(expected)}, effective length={len(actual)}")
        for index, (before, after) in enumerate(zip(expected, actual)):
            _compare(before, after, f"{path}[{index}]", errors)
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
    # A quick/confirmation subset is a projection of the independent frozen
    # table, never a table reconstructed from the caller's effective values.
    # Case definitions, criteria, canonical slots and replica counts stay fixed.
    row = deepcopy(row)
    expected = row.get("contract", {})
    canonical = expected.get("evaluation", {}).get("cases", [])
    effective = config.get("evaluation", {}).get("cases", [])
    if canonical and [c.get("name") for c in effective] != [c["name"] for c in canonical]:
        from .evaluation import select_evaluation_cases, fixed_suite_contract
        names = [c.get("name") for c in effective]
        if role not in ("quick", "confirmation") or not names or len(set(names)) != len(names):
            raise ValueError("Only quick/confirmation roles may project unique canonical cases")
        if set(expected["evaluation"].get("promotion_case_names", [])) - set(names):
            raise ValueError("Evaluation subset omits a required stage target")
        projected = select_evaluation_cases(expected["evaluation"], names)
        expected["evaluation"] = projected
        layout = fixed_suite_contract(expected)
        expected["scene_groups"] = layout["scene_groups"]
        expected["episode_seconds"] = layout["episode_seconds"]
        row["invocation"]["num_envs"] = len(names) * projected["episodes_per_case"]
    response_declared = ("actuator_response" in config or "actuator_response" in expected
                         or "response_migration" in config)
    if response_declared:
        from .actuator_response import validate_actuator_response
        for key in ("actuator_response", "actuator_response_evidence", "response_migration"):
            if key not in expected:
                errors.append(f"design.{key}: explicit response expectation is required")
        try:
            response = validate_actuator_response(config.get("actuator_response"))
            if response["enabled"] is not True:
                errors.append("contract.actuator_response.enabled: migration requires enabled response")
        except ValueError as error:
            errors.append(str(error))
        if _at(row, "runtime.modules.actuator_response") is not True:
            errors.append("design.runtime.modules.actuator_response: explicit enabled expectation required")
        if _at(row, "runtime.actuator_response") is _MISSING:
            errors.append("design.runtime.actuator_response: explicit parameters required")
    if ("recovery_hold_cost" in config or "recovery_hold_cost" in expected
            or _at(row, "runtime.recovery_hold_cost") is not _MISSING):
        from .rewards import validate_recovery_hold_cost
        hold = config.get("recovery_hold_cost")
        try:
            validate_recovery_hold_cost(hold)
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
        enabled = isinstance(hold, dict) and hold.get("enabled") is True
        if enabled and (not config.get("recovery_training", {}).get("enabled")
                or not config.get("dense_tracking", {}).get("enabled")
                or config.get("reward_profile") != "scut_v14_flat_v5"
                or config.get("recovery_takeover", {}).get("enabled")):
            errors.append("contract.recovery_hold_cost: requires native recovery/dense rewards without legacy takeover")
        if bool(config.get("evaluation_exact_cases")) and enabled:
            errors.append("contract.recovery_hold_cost: fixed evaluation must retain an explicitly disabled object")
        _compare(hold, _at(expected, "recovery_hold_cost"), "design.recovery_hold_cost", errors)
        _compare(enabled, _at(row, "runtime.modules.recovery_hold_cost"),
                 "design.runtime.modules.recovery_hold_cost", errors)
        _compare(hold, _at(row, "runtime.recovery_hold_cost"), "design.runtime.recovery_hold_cost", errors)
        if enabled:
            from .recovery_quality_revision import HOLD_GATE_BINDING
            _compare(HOLD_GATE_BINDING, _at(row, "runtime.recovery_hold_cost_gate_binding"),
                     "design.runtime.recovery_hold_cost_gate_binding", errors)
        elif _at(row, "runtime.recovery_hold_cost_gate_binding") is not _MISSING:
            errors.append("design.runtime.recovery_hold_cost_gate_binding: disabled hold cost has no native binding")
    if config.get("task_campaign", {}).get("revision_migration", {}).get("schema") == "task-worker-revision-v20":
        from .recovery_script_revision import runtime_expectation
        try:
            _compare(runtime_expectation(config), _at(row, "runtime.recovery_script_revision"),
                     "design.runtime.recovery_script_revision", errors)
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
    step_revision = (config.get("response_migration") in ("v6_step_curriculum_v1", "v6_recovery_curriculum_v1")
                     or expected.get("response_migration") in ("v6_step_curriculum_v1", "v6_recovery_curriculum_v1"))
    if config.get("response_migration") == "v6_recovery_curriculum_v1" or "recovery_training" in expected:
        from .response_migration import validate_recovery_configuration
        try:
            validate_recovery_configuration(config.get("recovery_training"))
        except ValueError as error:
            errors.append(str(error))
        if "recovery_training" not in expected:
            errors.append("design.recovery_training: explicit full settings required")
        _compare(True, _at(row, "runtime.modules.recovery_training"),
                 "design.runtime.modules.recovery_training", errors)
        prefix = config.get("recovery_training", {}).get("prefix_curriculum")
        if prefix is not None or _at(row, "runtime.recovery_prefix_curriculum") is not _MISSING:
            from .recovery_prefix_curriculum import runtime_expectation
            try:
                enabled = not bool(config.get("evaluation_exact_cases"))
                _compare(enabled, _at(row, "runtime.modules.recovery_prefix_curriculum"),
                    "design.runtime.modules.recovery_prefix_curriculum", errors)
                _compare(runtime_expectation(prefix, enabled=enabled),
                    _at(row, "runtime.recovery_prefix_curriculum"),
                    "design.runtime.recovery_prefix_curriculum", errors)
            except (TypeError, ValueError, KeyError) as error:
                errors.append(str(error))
    if (step_revision or config.get("response_migration") == "v6_transport_response_v1"
            or expected.get("response_migration") == "v6_transport_response_v1"):
        _compare("phase_aware_hold_v1", _at(config, "command_transport.delivery_model"),
                 "contract.command_transport.delivery_model", errors)
        model = "phase_aware_hold_v1" if _at(config, "command_transport.enabled") is True else None
        _compare(model, _at(row, "runtime.command_transport_model"),
                 "design.runtime.command_transport_model", errors)
    if step_revision:
        from .response_migration import validate_step_task_configuration
        try:
            validate_step_task_configuration(config.get("step_jump_task"))
        except ValueError as error:
            errors.append(str(error))
        if "step_jump_task" not in expected:
            errors.append("design.step_jump_task: explicit sensor/estimator expectation is required")
        for key, value in (("actor_reserved_frame30", {"index": 30,
                            "semantic": "jump_requested_landing_delta_m_times5",
                            "zero_without_step_jump": True}),):
            _compare(value, _at(config, key), "contract." + key, errors)
            _compare(value, expected.get(key, _MISSING), "design." + key, errors)
        _compare(True, _at(row, "runtime.modules.step_jump_task"),
                 "design.runtime.modules.step_jump_task", errors)
        _compare(True, _at(row, "runtime.modules.step_climb_task"),
                 "design.runtime.modules.step_climb_task", errors)
    new_asset = design.get("scope") == "v6_new_asset_formal_research"
    if "signal_delay" in config or "signal_delay" in expected:
        from .signal_delay import validate_signal_delay
        try:
            validate_signal_delay(config.get("signal_delay"), config["physics_dt"])
        except ValueError as error:
            errors.append(str(error))
        if "signal_delay" not in expected:
            errors.append("design.signal_delay: independent full settings required")
        _compare(config.get("signal_delay"), _at(row, "runtime.signal_delay"),
                 "design.runtime.signal_delay", errors)
        _compare(_at(config, "signal_delay.enabled"), _at(row, "runtime.modules.signal_delay"),
                 "design.runtime.modules.signal_delay", errors)
        if (_at(config, "signal_delay.enabled") is True
                and (_at(config, "usb_transport.enabled") is True
                     or _at(config, "signal_perturbations.max_delay_steps") != 0)):
            errors.append("signal_delay: duplicate sensor/policy delay layers are not authorized")
    if "jump_landing" in config or "jump_landing" in expected:
        from .jump_landing import validate_jump_landing
        from .jump_landing_revision import runtime_expectation
        try:
            validate_jump_landing(config.get("jump_landing"))
        except (ValueError, TypeError) as error:
            errors.append(str(error))
        if "jump_landing" not in expected:
            errors.append("design.jump_landing: independent full settings required")
        _compare(config.get("jump_landing", {}).get("enabled"), _at(row, "runtime.modules.jump_landing"),
                 "design.runtime.modules.jump_landing", errors)
        if "jump_landing" in config:
            _compare(runtime_expectation(config), _at(row, "runtime.jump_landing"),
                     "design.runtime.jump_landing", errors)
    if "jump_cycle_reset_on_accepted" in config or "jump_cycle_reset_on_accepted" in expected:
        flag = config.get("jump_cycle_reset_on_accepted")
        if type(flag) is not bool:
            errors.append("contract.jump_cycle_reset_on_accepted: explicit boolean required")
        _compare(flag, _at(row, "runtime.jump_cycle_reset_on_accepted"),
            "design.runtime.jump_cycle_reset_on_accepted", errors)
    if "nonstandard_start" in config or "nonstandard_start" in expected:
        from .nonstandard_start import validate_nonstandard_start, runtime_expectation
        try:
            validate_nonstandard_start(config.get("nonstandard_start"))
        except (ValueError, TypeError) as error:
            errors.append(str(error))
        if "nonstandard_start" not in expected:
            errors.append("design.nonstandard_start: independent full settings required")
        _compare(_at(config, "nonstandard_start.enabled"), _at(row, "runtime.modules.nonstandard_start"),
                 "design.runtime.modules.nonstandard_start", errors)
        if "nonstandard_start" in config:
            try:
                _compare(runtime_expectation(config["nonstandard_start"], config["policy_dt"]),
                         _at(row, "runtime.nonstandard_start"), "design.runtime.nonstandard_start", errors)
            except (ValueError, TypeError) as error:
                errors.append(str(error))
        if config.get("evaluation_exact_cases") and _at(config, "nonstandard_start.enabled") is True:
            errors.append("nonstandard_start: training births cannot alter fixed evaluation")
    if "release_stop" in config or "release_stop" in expected:
        from .release_stop import validate_release_stop, release_stop_runtime_expectation
        try:
            validate_release_stop(config.get("release_stop"))
        except (ValueError, TypeError) as error:
            errors.append(str(error))
        if "release_stop" not in expected:
            errors.append("design.release_stop: independent full settings required")
        _compare(_at(config, "release_stop.enabled"), _at(row, "runtime.modules.release_stop"),
                 "design.runtime.modules.release_stop", errors)
        if "release_stop" in config:
            try:
                _compare(release_stop_runtime_expectation(config["release_stop"], config["policy_dt"]),
                         _at(row, "runtime.release_stop"), "design.runtime.release_stop", errors)
            except (ValueError, TypeError) as error:
                errors.append(str(error))
        if _at(config, "release_stop.enabled") is True:
            if config.get("evaluation_exact_cases"):
                errors.append("release_stop: training profiles cannot alter fixed evaluation")
            rates = config.get("command_slew", {})
            if (rates.get("vx_m_s2"), rates.get("yaw_rad_s2")) != (1.5, 4.):
                errors.append("release_stop: existing 1.5/4 command acceleration limits required")
    if "locomotion_difficulty" in config or "locomotion_difficulty" in expected:
        from .locomotion_difficulty import validate_difficulty
        try:
            validate_difficulty(config.get("locomotion_difficulty"))
        except (ValueError, TypeError) as error:
            errors.append(str(error))
        if "locomotion_difficulty" not in expected:
            errors.append("design.locomotion_difficulty: independent full settings required")
        _compare(dict(enabled=not config.get("evaluation_exact_cases", False),
                      settings=deepcopy(config.get("locomotion_difficulty"))),
                 _at(row, "runtime.locomotion_difficulty"),
                 "design.runtime.locomotion_difficulty", errors)
    quality_keys = ("stand_still_lin_vel", "locomotion_posture")
    quality = {key: deepcopy(config.get("dense_tracking", {}).get(key))
               for key in quality_keys if key in config.get("dense_tracking", {})}
    if quality or _at(row, "runtime.locomotion_quality_rewards") is not _MISSING:
        from .rewards import validate_dense_tracking
        try:
            validate_dense_tracking(config.get("dense_tracking", {}))
        except (ValueError, TypeError) as error:
            errors.append(str(error))
        _compare(quality, _at(row, "runtime.locomotion_quality_rewards"),
                 "design.runtime.locomotion_quality_rewards", errors)
        for name, key in (("stand_still_lin_vel", "stand_still_lin_vel"),
                          ("locomotion_posture_floor", "locomotion_posture")):
            _compare(key in quality, _at(row, "runtime.modules." + name),
                     "design.runtime.modules." + name, errors)
        if "locomotion_posture" in quality and not (
                _at(config, "dense_tracking.enabled") is True
                and config.get("reward_profile") == "scut_v14_flat_v5"
                and isinstance(config.get("command_reference"), dict)
                and bool(config["command_reference"])
                and _at(config, "reference_reward.visible_reference_only") is True):
            errors.append("locomotion_posture: active dense tracking and visible reference rewards required")
    if "ordinary_posture_curriculum" in config or "ordinary_posture_curriculum" in expected:
        from .ordinary_posture_curriculum import validate_posture_settings, runtime_expectation
        settings = config.get("ordinary_posture_curriculum")
        if "ordinary_posture_curriculum" not in expected:
            errors.append("design.ordinary_posture_curriculum: independent full settings required")
        try:
            validate_posture_settings(settings)
            enabled = settings["enabled"] and not bool(config.get("evaluation_exact_cases"))
            if config.get("evaluation_exact_cases") and settings["enabled"]:
                errors.append("ordinary_posture_curriculum: training lessons cannot alter fixed evaluation")
            _compare(enabled, _at(row, "runtime.modules.ordinary_posture_curriculum"),
                     "design.runtime.modules.ordinary_posture_curriculum", errors)
            _compare(runtime_expectation(settings, enabled=enabled),
                     _at(row, "runtime.ordinary_posture_curriculum"),
                     "design.runtime.ordinary_posture_curriculum", errors)
        except (TypeError, ValueError, KeyError) as error:
            errors.append(str(error))
    required_paths = REQUIRED_CONTRACT_PATHS + (NEW_ASSET_CONTRACT_PATHS if new_asset else ())
    for path in required_paths:
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
    if new_asset:
        for path in ("dense_tracking.enabled", "jump_assist.enabled", "new_asset_curriculum.enabled",
                     "evaluation_exact_cases"):
            if type(_at(config, path)) is not bool:
                errors.append(f"contract.{path}: explicit boolean is required")
        if _at(config, "dense_tracking.enabled") is not True:
            errors.append("contract.dense_tracking.enabled: required visible35 dense reward is disabled")
        if _at(config, "evaluation_exact_cases") != (role != "train"):
            errors.append("contract.evaluation_exact_cases: run role and exact evaluation disagree")
        if role != "train" and _at(config, "jump_assist.enabled") is not False:
            errors.append("contract.jump_assist.enabled: evaluation cannot use training assistance")
        if _at(config, "curriculum_plan_id") != "v6-new-asset-curriculum-plan-v1":
            errors.append("contract.curriculum_plan_id: independent new asset campaign required")
        for key in ("stage_actor_update", "global_actor_update", "global_actor_update_offset",
                    "confirmed_jump_bins_m", "stage_start_actor_update"):
            if key in config.get("new_asset_curriculum", {}):
                errors.append(f"contract.new_asset_curriculum.{key}: runtime state cannot redefine the frozen schedule")
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
    # Coarse training is explicitly declared; it is never relabeled as 1 kHz feedback.
    from .control_contract import COARSE_SCOPE, validate_chassis_control
    coarse = control.get("simulation_timing_approximation") == COARSE_SCOPE
    if coarse:
        validate_chassis_control(control)
    for path, value in (("physics_dt", .005 if coarse else .001), ("policy_dt", .02),
                        ("v5_control.leg_kp", 160.), ("v5_control.leg_kd", 2.5)):
        _compare(value, _at(config, path), f"reference.{path}", errors)
    _compare(.6, _at(control, "actuators.wheel.kd"), "reference.wheel_velocity_p", errors)
    if not isinstance(row.get("runtime"), dict):
        errors.append("design.runtime: explicit runtime expectations required")
    elif new_asset:
        for path in ("dense_tracking", "command_reference", "precision_tracking",
                     *("modules." + name for name in NEW_ASSET_RUNTIME_MODULES)):
            if type(_at(row["runtime"], path)) is not bool:
                errors.append(f"design.runtime.{path}: explicit boolean expectation required")
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
        if (config.get("requires_design_preflight") or "actuator_response" in config
                or "response_migration" in config):
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
    if config.get("actuator_response", {}).get("enabled"):
        receipt["actuator_response_evidence"] = validate_response_evidence(config, root)
    if config.get("response_migration") in ("v6_step_curriculum_v1", "v6_recovery_curriculum_v1"):
        from .response_migration import validate_step_budget, validate_step_task_configuration
        if config.get("task_campaign"):
            from .task_campaign import validate_task_budget
            receipt["budget"] = validate_task_budget(config, root)
        else:
            receipt["budget"] = validate_step_budget(config, root)
        receipt["step_jump_task"] = validate_step_task_configuration(config["step_jump_task"], root)
    if config.get("response_migration") == "v6_recovery_curriculum_v1":
        from .response_migration import validate_recovery_configuration
        receipt["recovery_training"] = validate_recovery_configuration(config["recovery_training"], root)
    if stage not in config["enabled_stages"] or stage not in {item["name"] for item in config["stages"]}:
        raise ValueError(f"Environment stage {stage!r} is outside the independently verified contract")
    receipt.update(design_sha256=actual_sha, environment_stage=stage)
    return receipt


def validate_response_evidence(config, root):
    """Authenticate the explicitly released research profile and replay receipts.

    Hashes authenticate reviewed results; they do not turn a failed fit into a
    passed one. Publication of the receipt is a separate model-review decision.
    """
    from .actuator_response import validate_actuator_response
    response = validate_actuator_response(config.get("actuator_response"))
    binding = config.get("actuator_response_evidence")
    if not isinstance(binding, dict) or set(binding) != {"file", "sha256"}:
        raise ValueError("Actuator response requires a frozen evidence file/hash binding")
    contents = (Path(root) / binding["file"]).read_bytes()
    if hashlib.sha256(contents).hexdigest() != binding["sha256"]:
        raise ValueError("Actuator response evidence SHA mismatch")
    evidence = json.loads(contents)
    if (evidence.get("schema") != "actuator-response-evidence-v1"
            or evidence.get("status") != "validated_for_research"
            or evidence.get("hardware_release") is not False):
        raise ValueError("Actuator response evidence is not a reviewed research release")
    if evidence.get("actuator_response") != response:
        raise ValueError("Actuator response parameters differ from the reviewed evidence")
    files = evidence.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Actuator response evidence lacks sealed replay results")
    for filename, digest in files.items():
        if hashlib.sha256((Path(root) / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Actuator response result SHA mismatch: {filename}")
    return {"status": "authenticated", "sha256": binding["sha256"],
            "scope": evidence.get("scope"), "validated_file_count": len(files)}


def validate_runtime_state(expected, actual):
    """Compare observed instances/clocks, without claiming a functional probe passed."""
    required = {"physics_dt", "policy_dt", "decimation", "pc_control_dt", "seed",
                "modules", "leg_kp", "leg_kd", "wheel_velocity_p", "precision_tracking"}
    if "dense_tracking" in expected:
        required.update(("dense_tracking", "command_reference"))
    errors = [f"runtime.{key}: explicit expectation required" for key in sorted(required - expected.keys())]
    if "dense_tracking" in expected:
        for name in NEW_ASSET_RUNTIME_MODULES:
            if type(_at(expected, "modules." + name)) is not bool:
                errors.append(f"runtime.modules.{name}: explicit expectation required")
    if "locomotion_quality_rewards" in expected or "locomotion_quality_rewards" in actual:
        for name in ("stand_still_lin_vel", "locomotion_posture_floor"):
            if type(_at(expected, "modules." + name)) is not bool:
                errors.append(f"runtime.modules.{name}: explicit expectation required")
        if "locomotion_quality_rewards" not in expected:
            errors.append("runtime.locomotion_quality_rewards: explicit expectation required")
    if "locomotion_difficulty" in actual and "locomotion_difficulty" not in expected:
        errors.append("runtime.locomotion_difficulty: explicit expectation required")
    if "recovery_prefix_curriculum" in actual or "recovery_prefix_curriculum" in expected:
        if type(_at(expected, "modules.recovery_prefix_curriculum")) is not bool:
            errors.append("runtime.modules.recovery_prefix_curriculum: explicit boolean expectation required")
        if "recovery_prefix_curriculum" not in expected:
            errors.append("runtime.recovery_prefix_curriculum: explicit expectation required")
    if "recovery_hold_cost" in actual or "recovery_hold_cost" in expected:
        if type(_at(expected, "modules.recovery_hold_cost")) is not bool:
            errors.append("runtime.modules.recovery_hold_cost: explicit boolean expectation required")
        if "recovery_hold_cost" not in expected:
            errors.append("runtime.recovery_hold_cost: explicit complete expectation required")
        try:
            from .rewards import validate_recovery_hold_cost
            validate_recovery_hold_cost(actual.get("recovery_hold_cost"))
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
        if _at(actual, "recovery_hold_cost.enabled") is True:
            from .recovery_quality_revision import HOLD_GATE_BINDING
            _compare(HOLD_GATE_BINDING, _at(actual, "recovery_hold_cost_gate_binding"),
                     "runtime.recovery_hold_cost_gate_binding", errors)
            _compare(HOLD_GATE_BINDING, _at(expected, "recovery_hold_cost_gate_binding"),
                     "runtime.expected.recovery_hold_cost_gate_binding", errors)
        elif "recovery_hold_cost_gate_binding" in actual or "recovery_hold_cost_gate_binding" in expected:
            errors.append("runtime.recovery_hold_cost_gate_binding: disabled hold cost has no native binding")
        if _at(actual, "modules.recovery_hold_cost") != _at(actual, "recovery_hold_cost.enabled"):
            errors.append("runtime.recovery_hold_cost: observed module and complete enabled setting differ")
    if "recovery_script_revision" in actual or "recovery_script_revision" in expected:
        if "recovery_script_revision" not in expected:
            errors.append("runtime.recovery_script_revision: explicit complete expectation required")
        if "recovery_script_revision" not in actual:
            errors.append("runtime.recovery_script_revision: actual native binding required")
    if "jump_cycle_reset_on_accepted" in actual and "jump_cycle_reset_on_accepted" not in expected:
        errors.append("runtime.jump_cycle_reset_on_accepted: explicit expectation required")
    compared = deepcopy(actual)
    if "ordinary_posture_curriculum" in actual or "ordinary_posture_curriculum" in expected:
        if type(_at(expected, "modules.ordinary_posture_curriculum")) is not bool:
            errors.append("runtime.modules.ordinary_posture_curriculum: explicit boolean expectation required")
        if "ordinary_posture_curriculum" not in expected:
            errors.append("runtime.ordinary_posture_curriculum: explicit expectation required")
        observed = compared.get("ordinary_posture_curriculum", {}).pop("runtime_observed", None)
        enabled = _at(expected, "modules.ordinary_posture_curriculum") is True
        if enabled:
            if not isinstance(observed, dict):
                errors.append("runtime.ordinary_posture_curriculum: actual population observation required")
            else:
                for key in ("eligible_count", "birth_eligible_population", "relative_actor_updates",
                            "birth_selected_count", "birth_rejected_count", "push_events"):
                    if type(observed.get(key)) is not int or observed[key] < 0:
                        errors.append("runtime.ordinary_posture_curriculum." + key + ": nonnegative integer required")
                for key in ("birth_angle_cap_rad", "push_speed_cap_m_s", "birth_abs_angle_max_rad",
                            "push_clock_min_s", "push_clock_max_s", "push_clock_std_s", "push_velocity_delta_max_m_s"):
                    value = observed.get(key)
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        errors.append("runtime.ordinary_posture_curriculum." + key + ": finite nonnegative value required")
                for key in ("eligible_count", "birth_eligible_population", "birth_selected_count",
                            "birth_abs_angle_max_rad", "birth_angle_cap_rad", "push_speed_cap_m_s"):
                    if type(observed.get(key)) in (int, float) and observed[key] <= 0:
                        errors.append("runtime.ordinary_posture_curriculum." + key + ": enabled lesson has zero population or cap")
                if observed.get("geometry_bound") is not True:
                    errors.append("runtime.ordinary_posture_curriculum.geometry_bound: authenticated geometry required")
                settings = expected.get("ordinary_posture_curriculum", {}).get("settings", {})
                relative = observed.get("relative_actor_updates")
                if type(relative) is int and relative >= 0 and settings:
                    ramp = settings["ramp"]
                    alpha = min(1., max(0., (relative - ramp["start_actor_update"])
                        / (ramp["end_actor_update"] - ramp["start_actor_update"])))
                    for section, start, end, name in (
                        ("birth", "start_roll_pitch_rad", "end_roll_pitch_rad", "birth_angle_cap_rad"),
                        ("push", "start_speed_m_s", "end_speed_m_s", "push_speed_cap_m_s")):
                        spec = settings[section]
                        cap = spec[start] + alpha * (spec[end] - spec[start])
                        if type(observed.get(name)) not in (int, float) or not math.isclose(
                                observed[name], cap, rel_tol=1e-6, abs_tol=1e-8):
                            errors.append("runtime.ordinary_posture_curriculum." + name + ": actual actor-clock cap differs")
                    lo, hi = settings["push"]["interval_seconds"]
                    if not (lo <= observed.get("push_clock_min_s", -1)
                            <= observed.get("push_clock_max_s", -1) <= hi
                            and observed.get("push_clock_std_s", 0) > 0):
                        errors.append("runtime.ordinary_posture_curriculum: independent push timers differ")
    for path, value in expected.items():
        _compare(value, _at(compared, path), "runtime." + path, errors)
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
