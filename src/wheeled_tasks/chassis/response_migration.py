"""Validate actuator-model continuation without changing its learning contract.

These checks do not qualify a fitted plant. Its bag/replay validation must be
sealed separately before a caller authorizes this migration.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path


RESPONSE_MIGRATION = "v6_actuator_response_v1"
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


def validate_response_resume(source, target, *, source_control=None, target_control=None):
    """Authorize only a named same-ABI, same-budget actuator-response migration.

    The caller authenticates source checkpoint/contract/control file hashes and
    independently runs target design preflight. Passing this validator never
    marks the target plant as identified or suitable for hardware deployment.
    """
    if target.get("response_migration") != RESPONSE_MIGRATION:
        raise ValueError("Response resume requires the declared actuator migration")
    if (not source.get("manual_context35") or not target.get("manual_context35")
            or (target.get("actor_dim"), target.get("critic_dim"), target.get("action_dim")) != (35, 81, 6)
            or (target.get("physics_dt"), target.get("policy_dt")) != (.001, .02)):
        raise ValueError("Response resume requires the unchanged35/81 ABI and1kHz/50Hz clocks")
    response = target.get("actuator_response")
    if not isinstance(response, dict) or response.get("enabled") is not True:
        raise ValueError("Response migration must explicitly enable its target actuator model")
    if response == source.get("actuator_response"):
        raise ValueError("Response migration did not change the actuator model")
    evidence = target.get("actuator_response_evidence")
    if evidence is not None:
        if (not isinstance(evidence, dict) or set(evidence) != {"file", "sha256"}
                or not isinstance(evidence["file"], str) or not evidence["file"]
                or Path(evidence["file"]).is_absolute() or ".." in Path(evidence["file"]).parts
                or not isinstance(evidence["sha256"], str) or len(evidence["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in evidence["sha256"])):
            raise ValueError("Actuator evidence must be a repository-relative file/hash binding, not a qualification claim")
    changed = sorted(key for key in source.keys() | target.keys() if source.get(key) != target.get(key))
    unexpected = sorted(set(changed) - _CONTRACT_CHANGES)
    if unexpected:
        raise ValueError(f"Response resume changes unrelated training semantics: {unexpected}")
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
        controls = validate_response_controls(source_control, target_control)
        if controls["changed_control_fields"] and not control_changed:
            raise ValueError("Changed control content must change its frozen contract identity")
    return {"kind": "explicit_actuator_response_resume", "changed_fields": changed, **controls,
            "actuator_response_evidence": dict(evidence) if evidence is not None else None,
            "plant_qualification_evaluated": False,
            "actor_critic_optimizer": "preserved", "environment_state": "new_episodes",
            "rollout_state": "discarded_no_cross_plant_advantages",
            "consumed_budget": "preserved_no_refund", "acceptance": "retest_under_target_plant",
            "plant_qualification": "requires_separate_bag_and_Isaac_replay_receipt"}


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
                              required_cases):
    """Compare the same actor under old/new plants using paired fixed seeds.

    Accept complete evaluate_chassis.py reports, not caller-assembled pass lists.
    A case failing only precision remains explicitly distinguished from failure
    of accounting, survival, mechanics or tilt checks.
    """
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
    if any(value != settings[0] for value in settings[1:]):
        raise ValueError("Response comparison changed fixed evaluation criteria or coverage")
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
    names = set(candidates[0]["cases"])
    if required - names or any(set(candidate["cases"]) != names for candidate in candidates):
        raise ValueError("Response comparison has invalid evaluation coverage")
    expected = {case["name"] for case in reports[0]["settings"]["cases"]}
    if names != expected:
        raise ValueError("Response comparison differs from the frozen case manifest")
    for candidate in candidates:
        for case in candidate["cases"].values():
            checks = case.get("checks", {})
            if (not checks or any(type(value) is not bool for value in checks.values())
                    or type(case.get("passed")) is not bool or case["passed"] != all(checks.values())):
                raise ValueError("Response comparison contains inconsistent case grading")
            if case.get("requested_episodes") != repeats:
                raise ValueError("Response comparison has inconsistent requested episodes")

    def confirmed(first, second):
        return {name for name in names if candidates[first]["cases"][name]["passed"]
                and candidates[second]["cases"][name]["passed"]}

    old_passed, new_passed = confirmed(0, 1), confirmed(2, 3)
    regressions = (old_passed - new_passed) & required
    precision, basic, deltas = {}, {}, {}
    for name in sorted(required):
        failures = {check for candidate in candidates[2:] for check, passed in
                    candidate["cases"][name]["checks"].items() if not passed}
        if failures:
            (basic if failures & _BASIC_CHECKS else precision)[name] = sorted(failures)
        deltas[name] = {}
        for metric in _METRICS:
            values = [candidate["cases"][name].get(metric) for candidate in candidates]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) for value in values):
                deltas[name][metric] = {"old_mean": (values[0] + values[1]) / 2,
                                       "new_mean": (values[2] + values[3]) / 2,
                                       "delta": (values[2] + values[3] - values[0] - values[1]) / 2}
    return {"schema": "v6-response-retention-comparison-v1", "checkpoint_sha256": next(iter(hashes)),
            "paired_seeds": seeds[:2], "required_case_names": sorted(required),
            "old_confirmed_case_names": sorted(old_passed & required),
            "new_confirmed_case_names": sorted(new_passed & required),
            "response_recovery_case_names": sorted(regressions),
            "precision_not_yet_met": precision, "basic_or_mechanical_check_failures": basic,
            "metric_comparison": deltas, "extra_rehearsal_required": bool(regressions),
            "old_passes_credited_as_new": False,
            "scope": "measured_cases_only_not_unseen_RMCS_handoff_poses"}
