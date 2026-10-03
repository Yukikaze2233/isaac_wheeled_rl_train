"""Validate named response continuations without changing learning semantics.

These checks do not qualify a fitted plant. Its bag/replay validation must be
sealed separately before a caller authorizes this migration.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path


RESPONSE_MIGRATION = "v6_actuator_response_v1"
TRANSPORT_RESPONSE_MIGRATION = "v6_transport_response_v1"
STEP_CURRICULUM_MIGRATION = "v6_step_curriculum_v1"
RECOVERY_CURRICULUM_MIGRATION = "v6_recovery_curriculum_v1"
STEP_BUDGETS = {"S1": 1000, "S2": 2000, "S3": 1000, "S4": 1500,
                "S5": 1000, "S5T": 600, "S5D": 400, "S6": 1000}
RETIRED_STEP_CASES = frozenset(("stairs_200_250", "stairs_down_200_250", "step_up_15", "step_up_20",
                               "step_up_25", "heldout_step_up_payload_grip",
                               "step_up_03", "step_up_06", "step_up_10", "stairs"))
STEP_TASK_SENSOR = {"enabled": True, "sensor_origin_b_m": [.10, -.20],
                    "sensor_pitch_rad": .55, "ground_echo_margin_m": .015,
                    "range_max_m": 4., "range_noise_std_m": .003,
                    "range_dropout_probability": .002, "range_sample_period_s": .02}
_CONTRACT_CHANGES = frozenset((
    "actuator_response", "actuator_response_evidence", "response_migration", "design_preflight",
    "control_math_source", "control_math_sha256", "physics_prior_scope",
    "training_reference", "control_model_status",
))
_CONTROL_CHANGES = frozenset((
    "control_id", "plant_fit_applied", "uncertainty_note",
    "actuators.leg.armature", "actuators.wheel.armature",
    "actuators.leg.source_note", "actuators.wheel.source_note",
))
_METRICS = ("height_mae_m", "vx_mae_m_s", "yaw_mae_rad_s", "stand_drift_max_m",
            "settled_stop_speed", "survival_rate", "task_success_rate")
_BASIC_CHECKS = frozenset(("all_requested_episodes_accounted", "has_post_warmup_samples",
                          "survival", "task_success", "mechanics", "tilt"))
_TRANSPORT_CONTRACT_CHANGES = frozenset((
    "command_transport", "design_preflight", "response_migration", "evaluation",
))
_STEP_CONTRACT_CHANGES = _TRANSPORT_CONTRACT_CHANGES | frozenset((
    "budget_spec_file", "budget_spec_sha256", "step_jump_task", "actor_reserved_frame30",
    "release_scope",
))
_RECOVERY_CONTRACT_CHANGES = _STEP_CONTRACT_CHANGES | frozenset(("recovery_training",))
_EXTENSIBLE_CASE_LISTS = frozenset((
    "canonical_case_names", "promotion_case_names", "quick_case_names", "retention_case_names",
))


def _changed_leaves(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                changes.append(path)
            else:
                changes.extend(_changed_leaves(before[key], after[key], path))
        return changes
    return [] if before == after else [prefix]


def validate_response_controls(source_control, target_control):
    """Permit explicit rotor inertia changes, preserving the real PC controller."""
    changed = _changed_leaves(source_control, target_control)
    unexpected = sorted(set(changed) - _CONTROL_CHANGES)
    if unexpected:
        raise ValueError(f"Response migration changes control semantics: {unexpected}")
    for side in ("leg", "wheel"):
        value = target_control["actuators"][side]["armature"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Response migration has invalid {side} armature")
    return {"changed_control_fields": changed,
            "pd_gains_clock_ratio_and_caps": "unchanged"}


def _case_manifest(settings):
    cases = settings.get("cases")
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, dict) or not isinstance(case.get("name"), str)
                   or not case["name"] for case in cases)):
        raise ValueError("Response comparison requires an explicit nonempty case manifest")
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Response comparison has duplicate case names")
    return {case["name"]: case for case in cases}


def _validate_transport_evaluation_extension(source, target):
    """Keep existing definitions and criteria; only add new cases and their names."""
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError("Transport migration requires explicit evaluation settings")
    old_cases, new_cases = _case_manifest(source), _case_manifest(target)
    added = set(new_cases) - set(old_cases)
    if (set(old_cases) - set(new_cases)
            or [name for name in new_cases if name in old_cases] != list(old_cases)
            or any(old_cases[name] != new_cases[name] for name in old_cases)):
        raise ValueError("Transport migration must preserve every existing case definition and order")
    for name in added:
        case = new_cases[name]
        if (case.get("transport_robustness_case") is not True
                or case.get("reset_seed_key") not in old_cases):
            raise ValueError("Transport migration additions must be marked communication cases paired with an existing reset seed")
    for key in source.keys() | target.keys():
        if key == "cases":
            continue
        if key in _EXTENSIBLE_CASE_LISTS:
            before, after = source.get(key, []), target.get(key, [])
            if before == after and (key in source) == (key in target):
                continue
            if (key not in target or not isinstance(before, list) or not isinstance(after, list)
                    or any(not isinstance(name, str) for name in before + after)
                    or len(set(before)) != len(before) or len(set(after)) != len(after)
                    or [name for name in after if name not in added] != before
                    or not set(after) <= set(new_cases)):
                raise ValueError(f"Transport migration can only append added case names to {key}")
        elif key not in source or key not in target or source[key] != target[key]:
            raise ValueError(f"Transport migration changes fixed evaluation criteria: {key}")
    return sorted(added)


def _validate_step_evaluation_extension(source, target):
    """Retire the named ambiguous old stair/high-step cases, preserving all others."""
    old, new = _case_manifest(source), _case_manifest(target)
    removed, added = set(old) - set(new), set(new) - set(old)
    if removed != RETIRED_STEP_CASES & set(old):
        raise ValueError("Step curriculum may retire only the ten declared stair/up-step cases")
    if ([name for name in new if name in old] != [name for name in old if name not in removed]
            or any(old[name] != new[name] for name in old.keys() & new.keys())):
        raise ValueError("Step curriculum must preserve surviving evaluation definitions and order")
    heights = (3, 6, 10, 15, 20, 25)
    step_names = {f"step_jump_{height:02d}_{source}" for height in heights for source in ("manual", "laser")}
    step_names |= {f"double_step_jump_200_250_{source}" for source in ("manual", "laser")}
    climb_names = {"body_climb_15", "body_climb_20", "body_climb_25", "body_climb_double_200_250",
                   "heldout_body_climb_20_payload_grip"}
    rolling_names = {"rolling_up_03", "rolling_up_06", "rolling_up_10", "rolling_stairs_3x04"}
    link_names = {f"{task}__{profile}" for task in ("stand_305mm", "forward_05", "backward_05", "rotate_1")
                  for profile in ("link_jitter", "link_loss_005", "link_burst_3", "link_combined")}
    task_names = step_names | climb_names | rolling_names
    if not task_names <= added or not added <= task_names | (link_names - set(old)):
        raise ValueError("Step curriculum requires the twenty-three named task cases and only named transport additions")
    for name in added:
        case = new[name]
        if name in step_names:
            if (case.get("step_jump_case") is not True or case.get("task") != "jump"
                    or case.get("skill", {}).get("mode") != 4
                    or case.get("introduced_stage") != ("S5D" if name.startswith("double") else "S5T")
                    or case.get("reset_seed_key") != name):
                raise ValueError("Step curriculum additions require explicit mode4 task and future-stage identity")
            from .skill_curriculum import step_jump_geometry
            spec = case["skill"]
            geometry = step_jump_geometry({"skill": spec["kind"], "step_jump": spec["step_jump"]})
            if any(case["terrain_limits"].get(key) != value for key, value in geometry.items()):
                raise ValueError("Step curriculum geometry must match incremental task rises")
        elif name in climb_names:
            spec = case.get("skill", {})
            stage = "S6" if name.startswith("heldout") else "S5D" if "double" in name else "S4"
            if (case.get("body_climb_case") is not True or case.get("task") != "traverse"
                    or spec.get("kind") != "body_climb" or spec.get("mode") != 0
                    or case.get("introduced_stage") != stage or "trigger_source" in spec.get("body_climb", {})):
                raise ValueError("Body climb cases require continuous forward commands and separate future-stage grading")
            from .skill_curriculum import skill_spec
            terrain, generated = skill_spec({"skill": "body_climb", "body_climb": spec["body_climb"]})
            if case.get("terrain") != terrain or any(case["terrain_limits"].get(key) != value
                                                      for key, value in generated["terrain_limits"].items()):
                raise ValueError("Body climb geometry must match incremental task rises")
        elif name in rolling_names:
            spec = case.get("skill", {})
            if (case.get("untriggered_up_case") is not True or case.get("task") != "traverse"
                    or spec.get("kind") != ("rolling_stairs" if name == "rolling_stairs_3x04" else "rolling_up")
                    or spec.get("mode") != 2 or case.get("introduced_stage") != "S4"
                    or any(key in spec for key in ("manual_request_seconds", "manual_hold_seconds", "step_jump", "body_climb"))):
                raise ValueError("Rolling up cases require continuous forward commands without manual height requests")
        elif (case.get("transport_robustness_case") is not True or case.get("reset_seed_key") not in old):
            raise ValueError("Step curriculum communication additions require old paired seed identity")
    for key in source.keys() | target.keys():
        if key == "cases":
            continue
        if key in _EXTENSIBLE_CASE_LISTS:
            before, after = source.get(key, []), target.get(key, [])
            if (not isinstance(before, list) or not isinstance(after, list)
                    or len(set(after)) != len(after) or not set(after) <= set(new)
                    or [name for name in after if name not in added] != [name for name in before if name not in removed]):
                raise ValueError(f"Step curriculum changed old evaluation coverage: {key}")
        elif key not in source or key not in target or source[key] != target[key]:
            raise ValueError(f"Step curriculum changes fixed evaluation criteria: {key}")
    return sorted(added), sorted(removed)


def validate_step_budget(config, root):
    """Authenticate the separately versioned eight-stage budget without adding updates."""
    path = Path(root) / config["budget_spec_file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.get("budget_spec_sha256"):
        raise ValueError("Step curriculum budget SHA mismatch")
    budget = json.loads(raw)
    formal = budget.get("formal", {})
    stages = formal.get("stages", [])
    budget_id = ("v6-recovery-curriculum-budget-v1" if config.get("response_migration") == RECOVERY_CURRICULUM_MIGRATION
                 else "v6-step-curriculum-budget-v1")
    if (budget.get("budget_id") != budget_id
            or [item.get("id") for item in stages] != list(STEP_BUDGETS)
            or {item["id"]: item.get("update_ceiling") for item in stages} != STEP_BUDGETS
            or any(item.get("transition_ceiling") != STEP_BUDGETS[item["id"]] * 480000 for item in stages)
            or formal.get("total_update_ceiling") != 8500
            or formal.get("total_transition_ceiling") != 4080000000
            or formal.get("num_envs") != 20000 or formal.get("num_steps_per_env") != 24
            or config.get("budget_stage_id") not in STEP_BUDGETS
            or config.get("total_updates") != STEP_BUDGETS[config["budget_stage_id"]]):
        raise ValueError("Step curriculum budget must preserve 8500 updates and the explicit eight-stage split")
    return {"status": "authenticated", "budget_id": budget["budget_id"],
            "sha256": config["budget_spec_sha256"], "total_update_ceiling": 8500}


def validate_recovery_configuration(settings, root=None):
    required = {"enabled": True, "reference_dt": .005, "blend_seconds": .2,
                "stable_seconds": 1., "max_script_seconds": 8.,
                "reroute_tilt_rad": 1.2217304763960306, "reroute_support_lost_seconds": .15, "max_reroutes": 1,
                "learning_mask": "pure_rl_intervals_only", "reset_mode": "initial_state_only",
                "controller": "encoder_imu_recovery_fsm", "action_history": "effective_reference"}
    if not isinstance(settings, dict) or any(settings.get(k) != v for k, v in required.items()):
        raise ValueError("Recovery requires the frozen physical prelude, clocks and pure-RL learning mask")
    if settings.get("wheel_balance") != {"enabled": True, "pitch_kp_nm_rad": 8.,
            "pitch_kd_nm_s_rad": 1.5, "wheel_damping_nm_s_rad": .2,
            "prepare_tilt_limit_deg": 55., "capture_tilt_limit_deg": 85.}:
        raise ValueError("Recovery script wheel capture must use its explicit frozen gains and tilt bounds")
    for prefix in ("profiles", "height_lookup"):
        path, sha = settings.get(prefix + "_file"), settings.get(prefix + "_sha256")
        if (not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts
                or not isinstance(sha, str) or len(sha) != 64):
            raise ValueError("Recovery geometry requires repository-relative frozen files")
        if root is not None and hashlib.sha256((Path(root) / path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Recovery {prefix} SHA mismatch")
    if "complete_started_blend" in settings and type(settings["complete_started_blend"]) is not bool:
        raise ValueError("Recovery blend completion must be an explicit boolean")
    if "pose_curriculum" in settings:
        from .recovery_curriculum import validate_pose_curriculum
        validate_pose_curriculum(settings["pose_curriculum"])
    return {"status": "authenticated" if root is not None else "configuration_validated", **required}


def _validate_recovery_evaluation_extension(source, target):
    old, new = _case_manifest(source), _case_manifest(target)
    added = set(new) - set(old)
    if (not added or set(old) - set(new) or [name for name in new if name in old] != list(old)
            or any(old[name] != new[name] for name in old)):
        raise ValueError("Recovery migration must preserve every existing case and append recovery cases")
    for name in added:
        case = new[name]
        if (case.get("recovery_case") is not True or case.get("skill", {}).get("kind") != "recovery_handoff"
                or case.get("introduced_stage") not in ("S3", "S4", "S5", "S5T", "S5D", "S6")
                or not case.get("skill", {}).get("recovery_pose")):
            raise ValueError("Recovery migration additions must be explicit future-stage recovery cases")
    for key in source.keys() | target.keys():
        if key == "cases":
            continue
        if key in _EXTENSIBLE_CASE_LISTS:
            before, after = source.get(key, []), target.get(key, [])
            if (len(set(after)) != len(after) or not set(after) <= set(new)
                    or [name for name in after if name not in added] != before):
                raise ValueError(f"Recovery changed existing case coverage: {key}")
        elif key not in source or key not in target or source[key] != target[key]:
            raise ValueError(f"Recovery changed fixed evaluation criteria: {key}")
    return sorted(added)


def validate_step_task_configuration(settings, root=None):
    """Keep the deployment-observable height/single-beam model an explicit input."""
    if (not isinstance(settings, dict)
            or set(settings) != set(STEP_TASK_SENSOR) | {"height_lookup_file", "height_lookup_sha256"}
            or any(settings.get(key) != value for key, value in STEP_TASK_SENSOR.items())):
        raise ValueError("step_jump_task requires the explicit reviewed sensor and estimator configuration")
    path, digest = settings["height_lookup_file"], settings["height_lookup_sha256"]
    if (not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("step_jump_task height lookup requires a repository-relative file/SHA binding")
    if root is not None and hashlib.sha256((Path(root) / path).read_bytes()).hexdigest() != digest:
        raise ValueError("step_jump_task height lookup SHA mismatch")
    return {"status": "authenticated" if root is not None else "schema_valid",
            "height_lookup_sha256": digest, "sensor_scope": "engineering_prior_not_identified"}


def validate_response_resume(source, target, *, source_control=None, target_control=None):
    """Authorize only a named same-ABI, same-budget response migration.

    The caller authenticates source checkpoint/contract/control file hashes and
    independently runs target design preflight. Passing this validator never
    marks the target plant as identified or suitable for hardware deployment.
    """
    kind = target.get("response_migration")
    if kind not in (RESPONSE_MIGRATION, TRANSPORT_RESPONSE_MIGRATION, STEP_CURRICULUM_MIGRATION, RECOVERY_CURRICULUM_MIGRATION):
        raise ValueError("Response resume requires a declared actuator or transport migration")
    step_curriculum = kind == STEP_CURRICULUM_MIGRATION
    recovery_curriculum = kind == RECOVERY_CURRICULUM_MIGRATION
    transport_only = kind == TRANSPORT_RESPONSE_MIGRATION or step_curriculum or recovery_curriculum
    if (not source.get("manual_context35") or not target.get("manual_context35")
            or (target.get("actor_dim"), target.get("critic_dim"), target.get("action_dim")) != (35, 81, 6)
            or (target.get("physics_dt"), target.get("policy_dt")) != (.001, .02)):
        raise ValueError("Response resume requires the unchanged35/81 ABI and1kHz/50Hz clocks")
    response = target.get("command_transport" if transport_only else "actuator_response")
    if transport_only:
        old_response = source.get("command_transport")
        if (not isinstance(response, dict) or not isinstance(old_response, dict)
                or type(response.get("enabled")) is not bool
                or response["enabled"] is not old_response.get("enabled")):
            raise ValueError("Transport migration must preserve the explicit stage enabled flag")
        if response.get("delivery_model") != "phase_aware_hold_v1":
            raise ValueError("Transport migration requires phase_aware_hold_v1")
    elif not isinstance(response, dict) or response.get("enabled") is not True:
        raise ValueError("Response migration must explicitly enable its target response model")
    if not (step_curriculum or recovery_curriculum) and response == source.get("command_transport" if transport_only else "actuator_response"):
        raise ValueError("Response migration did not change the transport model" if transport_only
                         else "Response migration did not change the actuator model")
    evidence = target.get("actuator_response_evidence")
    if evidence is not None:
        if (not isinstance(evidence, dict) or set(evidence) != {"file", "sha256"}
                or not isinstance(evidence["file"], str) or not evidence["file"]
                or Path(evidence["file"]).is_absolute() or ".." in Path(evidence["file"]).parts
                or not isinstance(evidence["sha256"], str) or len(evidence["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in evidence["sha256"])):
            raise ValueError("Actuator evidence must be a repository-relative file/hash binding, not a qualification claim")
    changed = sorted(key for key in source.keys() | target.keys()
                     if key not in source or key not in target or source[key] != target[key])
    allowed = (_RECOVERY_CONTRACT_CHANGES if recovery_curriculum else _STEP_CONTRACT_CHANGES if step_curriculum
               else _TRANSPORT_CONTRACT_CHANGES if transport_only else _CONTRACT_CHANGES)
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        raise ValueError(f"Response resume changes unrelated training semantics: {unexpected}")
    retired_cases = []
    if recovery_curriculum:
        if source.get("response_migration") != STEP_CURRICULUM_MIGRATION or any(
                cfg.get("curriculum_stage") != "S1" or cfg.get("total_updates") != 1000 for cfg in (source, target)):
            raise ValueError("Recovery continuation requires the sealed step-curriculum S1 source")
        validate_recovery_configuration(target.get("recovery_training"))
        before, after = dict(source.get("release_scope", {})), dict(target.get("release_scope", {}))
        before.pop("R0", None)
        claim = after.pop("R0", None)
        if before != after or claim != "scripted_physical_recovery_then_RL_handoff_separate_acceptance":
            raise ValueError("Recovery cannot alter unrelated release claims or claim actor-learned flips")
        added_cases = _validate_recovery_evaluation_extension(source["evaluation"], target["evaluation"])
        if set(added_cases) & set(target["evaluation"]["promotion_case_names"]):
            raise ValueError("Recovery cannot require future S3 skills for S1 promotion")
    elif step_curriculum:
        if any(config.get("curriculum_stage") != "S1" or config.get("budget_stage_id") != "S1"
               or config.get("total_updates") != 1000 for config in (source, target)):
            raise ValueError("Step curriculum continuation is authorized only within the unchanged S1 budget")
        validate_step_task_configuration(target.get("step_jump_task"))
        if (target.get("actor_reserved_frame30") != {"index": 30,
                    "semantic": "jump_requested_landing_delta_m_times5", "zero_without_step_jump": True}):
            raise ValueError("Step curriculum requires the explicitly masked task module and unchanged-size frame30 ABI")
        before, after = dict(source.get("release_scope", {})), dict(target.get("release_scope", {}))
        before.pop("stair_200_250_definition", None)
        if after.pop("stair_200_250_definition", None) != "retired_invalid_case_not_retained;new_double_step_rises_0.20_0.25_tops_0.20_0.45":
            raise ValueError("Step curriculum must explicitly retire the incorrect old stair definition")
        if before != after:
            raise ValueError("Step curriculum cannot change other release claims")
        added_cases, retired_cases = _validate_step_evaluation_extension(source.get("evaluation"), target.get("evaluation"))
        if set(added_cases) & set(target["evaluation"]["promotion_case_names"]):
            raise ValueError("Step curriculum cannot promote future new cases during S1 migration")
    else:
        added_cases = (_validate_transport_evaluation_extension(source.get("evaluation"), target.get("evaluation"))
                       if transport_only else [])
    if target.get("critic_warmup_updates", 0) != 0 or target.get("resume_critic_warmup_updates", 0) != 0:
        raise ValueError("This response migration does not authorize critic-only warmup")
    scope = target.get("physics_prior_scope", {})
    if scope.get("g1_release", False) is not False or scope.get("hardware_release", False) is not False:
        raise ValueError("Response continuation cannot grant G1 or hardware release")
    control_changed = any(source.get(key) != target.get(key)
                          for key in ("control_math_source", "control_math_sha256"))
    if control_changed and (source_control is None or target_control is None):
        raise ValueError("Changed control identity requires both expanded authenticated control contracts")
    controls = {}
    if source_control is not None or target_control is not None:
        if source_control is None or target_control is None:
            raise ValueError("Both control contracts are required for comparison")
        if transport_only and source_control != target_control:
            raise ValueError("Transport migration must preserve the complete control contract")
        controls = validate_response_controls(source_control, target_control)
        if controls["changed_control_fields"] and not control_changed:
            raise ValueError("Changed control content must change its frozen contract identity")
    receipt = {"kind": "explicit_transport_response_resume" if transport_only else "explicit_actuator_response_resume",
            "changed_fields": changed, **controls,
            "actuator_response_evidence": dict(evidence) if evidence is not None else None,
            "plant_qualification_evaluated": False,
            "actor_critic_optimizer": "preserved", "environment_state": "new_episodes",
            "rollout_state": "discarded_no_cross_plant_advantages",
            "consumed_budget": "preserved_no_refund", "acceptance": "retest_under_target_plant",
            "plant_qualification": "requires_separate_bag_and_Isaac_replay_receipt"}
    if transport_only:
        receipt.update(transport_qualification="engineering_prior_not_identified_or_G1_release",
                       added_evaluation_case_names=added_cases, actuator_model="unchanged")
    if step_curriculum:
        receipt.update(kind="explicit_step_curriculum_resume", retired_evaluation_case_names=retired_cases,
                       task_module="inactive_for_all_S1_training_groups", total_update_ceiling=8500)
    if recovery_curriculum:
        receipt.update(kind="explicit_recovery_curriculum_resume", task_module="inactive_for_all_S1_training_groups",
                       total_update_ceiling=8500, rollout_mask="all_true_on_S1;script_and_blend_excluded_from_S3")
    return receipt


def response_contract_provenance(source, target, root):
    """Expand and authenticate the control files for CPU-only supervisors."""
    root = Path(root)
    controls = []
    for label, config in (("source", source), ("target", target)):
        path = root / config["control_math_source"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != config.get("control_math_sha256"):
            raise ValueError(f"Response {label} control file hash mismatch")
        controls.append(json.loads(path.read_text()))
    receipt = validate_response_resume(source, target, source_control=controls[0], target_control=controls[1])
    if target.get("response_migration") in (STEP_CURRICULUM_MIGRATION, RECOVERY_CURRICULUM_MIGRATION):
        receipt["budget"] = validate_step_budget(target, root)
        receipt["step_jump_task"] = validate_step_task_configuration(target["step_jump_task"], root)
    if target.get("response_migration") == RECOVERY_CURRICULUM_MIGRATION:
        receipt["recovery_training"] = validate_recovery_configuration(target["recovery_training"], root)
    evidence = target.get("actuator_response_evidence")
    if evidence is not None:
        path = root / evidence["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError("Actuator response evidence file hash mismatch")
        receipt["evidence_file_identity_verified"] = True
    receipt.update(source_control_math_sha256=source["control_math_sha256"],
                   target_control_math_sha256=target["control_math_sha256"])
    return controls, receipt


def response_checkpoint_provenance(checkpoint, infos, target, root):
    """Authenticate both frozen control files before changing checkpoint identity."""
    from .full_curriculum import checkpoint_contract_path, checkpoint_update_count
    checkpoint, root = Path(checkpoint), Path(root)
    if checkpoint_update_count(checkpoint) != infos.get("successful_updates_total"):
        raise ValueError("Response checkpoint update count differs from its sealed learning state")
    source_path = checkpoint_contract_path(checkpoint)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if infos.get("contract_sha256") != source_sha:
        raise ValueError("Response checkpoint source contract provenance mismatch")
    source = json.loads(source_path.read_text())
    controls, receipt = response_contract_provenance(source, target, root)
    if source["control_math_sha256"] != infos.get("control_math_sha256"):
        raise ValueError("Response checkpoint source control provenance mismatch")
    if infos.get("asset_manifest_sha256") != source.get("asset_manifest_sha256"):
        raise ValueError("Response checkpoint asset provenance mismatch")
    receipt.update(source_contract_sha256=source_sha)
    return source, controls, receipt


def assess_response_retention(old_primary, old_confirmation, new_primary, new_confirmation,
                              required_cases, *, migration_kind=None):
    """Compare the same actor under old/new plants using paired fixed seeds.

    Accept complete evaluate_chassis.py reports, not caller-assembled pass lists.
    A case failing only precision remains explicitly distinguished from failure
    of accounting, survival, mechanics or tilt checks.
    Only the explicitly named transport migration permits a strict extension of
    the old suite. Every added case needs both new seeds, without an old credit.
    """
    if migration_kind not in (None, RESPONSE_MIGRATION, TRANSPORT_RESPONSE_MIGRATION, STEP_CURRICULUM_MIGRATION, RECOVERY_CURRICULUM_MIGRATION):
        raise ValueError("Unknown response comparison migration kind")
    step_curriculum = migration_kind == STEP_CURRICULUM_MIGRATION
    recovery_curriculum = migration_kind == RECOVERY_CURRICULUM_MIGRATION
    transport_only = migration_kind == TRANSPORT_RESPONSE_MIGRATION or step_curriculum or recovery_curriculum
    reports = (old_primary, old_confirmation, new_primary, new_confirmation)
    required = set(required_cases)
    if not required:
        raise ValueError("Response review requires explicit stage/retention targets")
    candidates = []
    for report in reports:
        if (report.get("status") != "evaluated" or len(report.get("candidates", [])) != 1
                or report.get("deterministic_actor") is not True
                or report.get("random_pushes") is not False or report.get("case_resampling") is not False):
            raise ValueError("Response comparison requires completed deterministic single-checkpoint evaluations")
        candidates.append(report["candidates"][0])
    hashes = {candidate.get("checkpoint_sha256") for candidate in candidates}
    if len(hashes) != 1 or not all(isinstance(value, str) and len(value) == 64 for value in hashes):
        raise ValueError("Response comparison must evaluate the same authenticated checkpoint")
    settings = [{k: v for k, v in report["settings"].items() if k != "seed"} for report in reports]
    if transport_only:
        if settings[0] != settings[1] or settings[2] != settings[3]:
            raise ValueError("Response comparison changed fixed evaluation criteria or coverage between paired seeds")
        if recovery_curriculum:
            added, retired = set(_validate_recovery_evaluation_extension(settings[0], settings[2])), set()
        elif step_curriculum:
            added, retired = map(set, _validate_step_evaluation_extension(settings[0], settings[2]))
        else:
            added, retired = set(_validate_transport_evaluation_extension(settings[0], settings[2])), set()
    else:
        if any(value != settings[0] for value in settings[1:]):
            raise ValueError("Response comparison changed fixed evaluation criteria or coverage")
        added, retired = set(), set()
    repeats = reports[0].get("episodes_per_case")
    if type(repeats) is not int or repeats <= 0 or any(report.get("episodes_per_case") != repeats for report in reports):
        raise ValueError("Response comparison requires the same explicit episode count")
    seeds = [report["settings"]["seed"] for report in reports]
    if seeds[0] != seeds[2] or seeds[1] != seeds[3] or seeds[0] == seeds[1]:
        raise ValueError("Response comparison requires matching primary and distinct confirmation seeds")
    for first, second in ((old_primary, old_confirmation), (new_primary, new_confirmation)):
        if (not first.get("contract_sha256")
                or first["contract_sha256"] != second.get("contract_sha256")):
            raise ValueError("Response comparison changed plant contract between paired seeds")
    old_names, names = set(candidates[0]["cases"]), set(candidates[2]["cases"])
    if (required - names or set(candidates[1]["cases"]) != old_names
            or set(candidates[3]["cases"]) != names or names != (old_names - retired) | added):
        raise ValueError("Response comparison has invalid evaluation coverage")
    for report, candidate in zip(reports, candidates):
        manifest = _case_manifest(report["settings"])
        if set(candidate["cases"]) != set(manifest):
            raise ValueError("Response comparison differs from the frozen case manifest")
        if transport_only and (report.get("coverage") != "full_manifest"
                or report.get("evaluated_case_names") != list(manifest)):
            raise ValueError("Transport response comparison requires all four full-manifest reports")
        for case in candidate["cases"].values():
            checks = case.get("checks", {})
            if (not checks or any(type(value) is not bool for value in checks.values())
                    or type(case.get("passed")) is not bool or case["passed"] != all(checks.values())):
                raise ValueError("Response comparison contains inconsistent case grading")
            if case.get("requested_episodes") != repeats:
                raise ValueError("Response comparison has inconsistent requested episodes")
    for name in names:
        compared = candidates if name in old_names else candidates[2:]
        if any(set(candidate["cases"][name]["checks"]) != set(compared[0]["cases"][name]["checks"])
               for candidate in compared[1:]):
            raise ValueError("Response comparison changed case grading checks")

    def confirmed(first, second):
        return {name for name in candidates[first]["cases"] if candidates[first]["cases"][name]["passed"]
                and candidates[second]["cases"][name]["passed"]}

    old_passed, new_passed = confirmed(0, 1), confirmed(2, 3)
    regressions = (old_passed - new_passed) & required
    added_unconfirmed = added - new_passed
    precision, basic, deltas = {}, {}, {}
    for name in sorted(required | added):
        failures = {check for candidate in candidates[2:] for check, passed in
                    candidate["cases"][name]["checks"].items() if not passed}
        if failures:
            # Missing a flip/handover or dropping off the required platform is
            # a chain outcome failure, not ordinary velocity/height precision.
            recovery_failure = any(check.startswith("recovery_") for check in failures)
            (basic if failures & _BASIC_CHECKS or recovery_failure else precision)[name] = sorted(failures)
        if name not in old_names:
            continue
        deltas[name] = {}
        for metric in _METRICS:
            values = [candidate["cases"][name].get(metric) for candidate in candidates]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) for value in values):
                deltas[name][metric] = {"old_mean": (values[0] + values[1]) / 2,
                                       "new_mean": (values[2] + values[3]) / 2,
                                       "delta": (values[2] + values[3] - values[0] - values[1]) / 2}
    result = {"schema": "v6-response-retention-comparison-v1", "checkpoint_sha256": next(iter(hashes)),
            "paired_seeds": seeds[:2], "required_case_names": sorted(required),
            "old_confirmed_case_names": sorted(old_passed & required),
            "new_confirmed_case_names": sorted(new_passed & required),
            "response_recovery_case_names": sorted(regressions),
            "precision_not_yet_met": precision, "basic_or_mechanical_check_failures": basic,
            "metric_comparison": deltas,
            "extra_rehearsal_required": bool(regressions | (added_unconfirmed & required)),
            "old_passes_credited_as_new": False,
            "scope": "measured_cases_only_not_unseen_RMCS_handoff_poses"}
    if recovery_curriculum:
        result["recovery_chain_failures"] = {
            name: {"failed_checks": failed, "episodes_by_seed": [
                candidate["cases"][name].get("recovery_episode_diagnostics", []) for candidate in candidates[2:]]}
            for name, failed in basic.items() if any(check.startswith("recovery_") for check in failed)}
    if transport_only:
        result.update(migration_kind=migration_kind,
                      added_case_names=sorted(added),
                      added_confirmed_case_names=sorted(added & new_passed),
                      added_unconfirmed_case_names=sorted(added_unconfirmed))
    if step_curriculum:
        result["retired_case_names_not_credited"] = sorted(retired)
    return result
