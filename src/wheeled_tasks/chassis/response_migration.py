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


def validate_response_resume(source, target, *, source_control=None, target_control=None):
    """Authorize only a named same-ABI, same-budget response migration.

    The caller authenticates source checkpoint/contract/control file hashes and
    independently runs target design preflight. Passing this validator never
    marks the target plant as identified or suitable for hardware deployment.
    """
    kind = target.get("response_migration")
    if kind not in (RESPONSE_MIGRATION, TRANSPORT_RESPONSE_MIGRATION):
        raise ValueError("Response resume requires a declared actuator or transport migration")
    transport_only = kind == TRANSPORT_RESPONSE_MIGRATION
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
    if response == source.get("command_transport" if transport_only else "actuator_response"):
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
    allowed = _TRANSPORT_CONTRACT_CHANGES if transport_only else _CONTRACT_CHANGES
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        raise ValueError(f"Response resume changes unrelated training semantics: {unexpected}")
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
    if migration_kind not in (None, RESPONSE_MIGRATION, TRANSPORT_RESPONSE_MIGRATION):
        raise ValueError("Unknown response comparison migration kind")
    transport_only = migration_kind == TRANSPORT_RESPONSE_MIGRATION
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
        added = set(_validate_transport_evaluation_extension(settings[0], settings[2]))
    else:
        if any(value != settings[0] for value in settings[1:]):
            raise ValueError("Response comparison changed fixed evaluation criteria or coverage")
        added = set()
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
            or set(candidates[3]["cases"]) != names or names != old_names | added):
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
            (basic if failures & _BASIC_CHECKS else precision)[name] = sorted(failures)
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
    if transport_only:
        result.update(migration_kind=migration_kind,
                      added_case_names=sorted(added),
                      added_confirmed_case_names=sorted(added & new_passed),
                      added_unconfirmed_case_names=sorted(added_unconfirmed))
    return result
