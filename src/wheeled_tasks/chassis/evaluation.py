"""Versioned fixed-case evaluation configuration and behavior-based acceptance."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


class RequestProtocolMetrics:
    """Keep per-episode request evidence; one good replica cannot mask another."""

    COUNTERS = ("request_events_delivered", "accepted_request_count", "repeat_request_count",
                "cancel_request_count", "cancel_phase", "repeat_clock_reset_errors",
                "completed_jump_count", "successful_jump_count", "command_sequence_steps", "command_sequence_length")
    PLATFORM_COUNTERS = {
        "step_jump": ("enabled", "requested_count", "required_count", "completed_count", "first_landed", "second_landed"),
        "body_climb": ("enabled", "required_count", "completed_count", "first_supported", "second_supported",
                       "air_peak_s", "flight_violation"),
    }
    RECOVERY_COUNTERS = ("enabled", "phase", "route", "failed", "failure_code", "motion_released",
                         "pure_rl", "rl_stable_s", "followup_s", "true_success", "handover_count",
                         "ever_blended", "script_seconds", "platform_retained")

    def __init__(self, groups, device, policy_dt):
        import torch
        self.groups, self.dt = list(groups), policy_dt
        self.episodes = {name: [] for name in dict.fromkeys(groups)}
        self.platform_episodes = {kind: {name: [] for name in self.episodes} for kind in self.PLATFORM_COUNTERS}
        self.recovery_episodes = {name: [] for name in self.episodes}
        self.tracking = torch.zeros(len(groups), 3, dtype=torch.long, device=device)
        self.seen_flight = torch.zeros(len(groups), dtype=torch.bool, device=device)

    def observe(self, data, active):
        import torch
        # This ledger is independent of jump-request protocol counters: a
        # normal body climb intentionally has no trigger/accepted request.
        if "done" in data:
            keys = ["recovery_" + suffix for suffix in self.RECOVERY_COUNTERS]
            if all(key in data for key in keys):
                ended = data["done"].bool() & active & data["recovery_enabled"].bool()
                for index in ended.nonzero(as_tuple=False).flatten().tolist():
                    row = {suffix: float(data[key][index]) for suffix, key in zip(self.RECOVERY_COUNTERS, keys)}
                    self.recovery_episodes[self.groups[index]].append(row)
            for kind, suffixes in self.PLATFORM_COUNTERS.items():
                keys = [kind + "_" + suffix for suffix in suffixes]
                if any(key not in data for key in keys):
                    continue
                ended = data["done"].bool() & active & data[kind + "_enabled"].bool()
                for index in ended.nonzero(as_tuple=False).flatten().tolist():
                    row = {suffix: (float(data[key][index]) if suffix == "air_peak_s" else int(data[key][index]))
                           for suffix, key in zip(suffixes, keys)}
                    self.platform_episodes[kind][self.groups[index]].append(row)
        if any(name not in data for name in self.COUNTERS) or "physical_phase" not in data:
            return  # Missing evidence must fail grading, never be filled with success.
        first = active & (data["episode_ticks"] == 1)
        self.tracking[first] = 0
        self.seen_flight[first] = False
        phase = data["physical_phase"]
        self.seen_flight |= active & (phase == 2)
        steps, length = data["command_sequence_steps"], data["command_sequence_length"]
        vx_error = (data["velocity"][:, 0] - data["commands"][:, 0]).abs()
        moving = data["commands"][:, 0] > .3
        stationary = data["commands"][:, :2].abs().amax(-1) < .01
        grounded = (phase == 0) | (phase == 4)
        stages = torch.stack(((steps == 1) & moving & (vx_error <= .1),
                              (steps == 2) & stationary & (data["velocity"][:, :2].norm(dim=-1) <= .1),
                              (steps == length) & moving & (vx_error <= .1)), -1)
        self.tracking += stages & (active & grounded & (length > 0))[:, None]
        for index in (data["done"].bool() & active).nonzero(as_tuple=False).flatten().tolist():
            episode = {name: int(data[name][index]) for name in self.COUNTERS}
            episode.update(final_phase=int(phase[index]), seen_flight=bool(self.seen_flight[index]),
                initial_motion_tracking_s=float(self.tracking[index, 0]) * self.dt,
                stop_tracking_s=float(self.tracking[index, 1]) * self.dt,
                resumed_motion_tracking_s=float(self.tracking[index, 2]) * self.dt)
            self.episodes[self.groups[index]].append(episode)

    def report_into(self, metrics):
        for name, episodes in self.episodes.items():
            if name in metrics["groups"]:
                metrics["groups"][name]["request_protocol_episodes"] = deepcopy(episodes)
                for kind, groups in self.platform_episodes.items():
                    if groups[name]:
                        metrics["groups"][name][kind + "_episodes"] = deepcopy(groups[name])
                if self.recovery_episodes[name]:
                    metrics["groups"][name]["recovery_episodes"] = deepcopy(self.recovery_episodes[name])
        return metrics


def request_protocol_checks(case, group, episodes_per_case):
    kind = case.get("request_protocol_case")
    if not kind:
        return {}
    episodes = group.get("request_protocol_episodes", [])
    events = case.get("skill", {}).get("request_events", [])
    evidence = len(episodes) == episodes_per_case and bool(episodes)
    required = set(RequestProtocolMetrics.COUNTERS) | {"final_phase", "seen_flight"}
    evidence &= all(required <= row.keys() for row in episodes)
    checks = {"request_protocol_evidence": bool(evidence)}
    if not evidence:
        return checks
    checks.update(request_events_delivered=all(row["request_events_delivered"] >= len(events) for row in episodes),
                  request_repeat_clock_preserved=all(row["repeat_clock_reset_errors"] == 0 for row in episodes),
                  request_recovered=all(row["final_phase"] == 0 for row in episodes))
    if kind == "request_repeat":
        checks["repeat_exercised"] = all(row["repeat_request_count"] >= 1 and row["accepted_request_count"] == 1
                                         and row["completed_jump_count"] >= 1 and row["successful_jump_count"] >= 1 for row in episodes)
    elif kind == "request_cancel_before_takeoff":
        checks["cancel_before_flight"] = all(row["cancel_request_count"] >= 1 and row["cancel_phase"] in (0, 1)
            and row["accepted_request_count"] == 1 and not row["seen_flight"] and row["completed_jump_count"] == 0
            and row["successful_jump_count"] == 0 for row in episodes)
    elif kind == "request_cancel_airtime":
        checks["airborne_cancel_completed_landing"] = all(row["cancel_request_count"] >= 1 and row["cancel_phase"] == 2
            and row["completed_jump_count"] >= 1 and row["successful_jump_count"] >= 1 and row["accepted_request_count"] == 1 for row in episodes)
    elif kind == "request_repeat_after_landing":
        checks["two_distinct_jumps_completed"] = all(row["accepted_request_count"] >= 2 and row["completed_jump_count"] >= 2
                                                   and row["successful_jump_count"] >= 2 for row in episodes)
    elif kind == "move_stop_jump_resume":
        sequence = case.get("skill", {}).get("command_sequence", [])
        checks["move_stop_jump_resume_completed"] = bool(sequence) and all(
            row["command_sequence_steps"] == len(sequence) and row["completed_jump_count"] >= 1 and row["successful_jump_count"] >= 1
            and row.get("initial_motion_tracking_s", 0.) >= .5 and row.get("stop_tracking_s", 0.) >= .5
            and row.get("resumed_motion_tracking_s", 0.) >= 1. for row in episodes)
    else:
        checks["known_request_protocol_case"] = False
    return checks


def platform_task_checks(case, group, episodes_per_case):
    """Require physical completion of each prescribed platform in every episode.

    A jump's request count is additional command evidence, never a substitute
    for its independently observed landing. Normal climbing has no request gate.
    """
    skill = case.get("skill", {})
    jumping = bool(case.get("step_jump_case") or skill.get("kind") in ("step_jump", "double_step_jump"))
    climbing = bool(case.get("body_climb_case") or skill.get("kind") == "body_climb")
    if not jumping and not climbing:
        return {}
    if jumping and climbing:
        return {"platform_task_kind_unambiguous": False}
    prefix, field = ("step_jump", "step_jump") if jumping else ("body_climb", "body_climb")
    rises = skill.get(field, {}).get("rises_m", [])
    required_count = len(rises) if isinstance(rises, list) else 0
    valid_spec = required_count in (1, 2) and all(
        type(value) in (int, float) and math.isfinite(value) and value > 0 for value in rises)
    if jumping:
        expected_kind = "step_jump" if required_count == 1 else "double_step_jump"
        valid_spec &= skill.get("kind") == expected_kind and case.get("task") == "jump"
    else:
        valid_spec &= skill.get("kind") == "body_climb" and case.get("task") == "traverse"
    checks = {prefix + "_specification": bool(valid_spec)}
    episodes = group.get(prefix + "_episodes", [])
    required_keys = set(RequestProtocolMetrics.PLATFORM_COUNTERS[prefix])
    evidence = (isinstance(episodes, list) and len(episodes) == episodes_per_case and bool(episodes)
                and all(isinstance(row, dict) and required_keys <= row.keys() for row in episodes))
    checks[prefix + "_episode_evidence"] = bool(evidence)
    if not valid_spec or not evidence:
        return checks
    # Do not trust observed required_count to weaken the frozen case definition.
    checks[prefix + "_required_platforms"] = all(
        row["enabled"] == 1 and row["required_count"] == required_count for row in episodes)
    first, second = ("first_landed", "second_landed") if jumping else ("first_supported", "second_supported")
    checks[prefix + "_completed_platforms"] = all(
        row["completed_count"] >= required_count and row[first] == 1
        and (required_count == 1 or row[second] == 1) for row in episodes)
    if jumping:
        checks[prefix + "_requests_exercised"] = all(row["requested_count"] >= required_count for row in episodes)
    else:
        checks[prefix + "_no_flight_violation"] = all(
            row["flight_violation"] == 0 and type(row["air_peak_s"]) in (int, float)
            and math.isfinite(row["air_peak_s"]) and row["air_peak_s"] >= 0 for row in episodes)
    checks[prefix + "_no_failures_or_truncations"] = (
        group["failures"] == 0 and group["boundary_truncations"] == 0 and group.get("blocked_truncations", 0) == 0)
    return checks


def is_recovery_case(case):
    return bool(case.get("recovery_case") or case.get("skill", {}).get("kind") == "recovery_handoff")


def _recovery_row_checks(row):
    from .recovery import RecoveryPhase
    required = set(RequestProtocolMetrics.RECOVERY_COUNTERS)
    evidence = (isinstance(row, dict) and required <= row.keys()
                and all(type(row[key]) in (int, float, bool) and math.isfinite(row[key]) for key in required))
    checks = {"evidence": evidence}
    if not evidence:
        return checks
    discrete = required - {"script_seconds", "rl_stable_s", "followup_s"}
    checks.update(
        finite_valid_state=all(row[key] == int(row[key]) for key in discrete)
            and row["phase"] in [int(phase) for phase in RecoveryPhase] and row["route"] in range(5)
            and row["rl_stable_s"] >= 0. and row["followup_s"] >= 0.,
        enabled=row["enabled"] == 1,
        script_budget=0. <= row["script_seconds"] <= 8. + 1e-6,
        no_failure=row["failed"] == 0 and row["failure_code"] == 0,
        blend_and_single_handover=row["ever_blended"] == 1 and row["handover_count"] == 1,
        pure_rl=row["pure_rl"] == 1 and row["phase"] == int(RecoveryPhase.RL),
        continuous_rl_stability=row["rl_stable_s"] >= 1. - 1e-5,
        motion_release=row["motion_released"] == 1,
        continuous_followup=row["followup_s"] >= 1. - 1e-5,
        physical_success=row["true_success"] == 1,
        platform_retained=row["platform_retained"] == 1,
    )
    return checks


def recovery_task_checks(case, group, episodes_per_case):
    """Grade the complete recovery chain, never static error during the script."""
    if not is_recovery_case(case):
        return {}
    skill = case.get("skill", {})
    checks = {"recovery_specification": (case.get("recovery_case") is True
        and skill.get("kind") == "recovery_handoff" and isinstance(skill.get("recovery_pose"), str)
        and bool(skill["recovery_pose"]))}
    rows = group.get("recovery_episodes", [])
    evidence = (isinstance(rows, list) and len(rows) == episodes_per_case and bool(rows)
                and all(_recovery_row_checks(row)["evidence"] for row in rows))
    checks["recovery_episode_evidence"] = evidence
    if not evidence:
        return checks
    row_checks = [_recovery_row_checks(row) for row in rows]
    for key in row_checks[0]:
        if key != "evidence":
            checks["recovery_" + key] = all(row[key] for row in row_checks)
    checks["recovery_no_failures_or_truncations"] = (group["failures"] == 0
        and group["boundary_truncations"] == 0 and group.get("blocked_truncations", 0) == 0)
    checks["recovery_outcomes_accounted"] = group["successes"] == episodes_per_case
    return checks


def recovery_episode_diagnostics(group, episodes_per_case):
    """Separate script preparation, torque handover, and policy outcome failures."""
    rows = group.get("recovery_episodes", [])
    if not isinstance(rows, list):
        rows = []
    outcomes = []
    for index in range(max(episodes_per_case, len(rows))):
        row = rows[index] if index < len(rows) else None
        checks = _recovery_row_checks(row)
        failed = [key for key, passed in checks.items() if not passed]
        if not checks["evidence"]:
            stage, reason = "evidence", "missing_or_nonfinite_episode_evidence"
        elif not checks["finite_valid_state"] or not checks["enabled"]:
            stage, reason = "evidence", "invalid_episode_state"
        elif not checks["no_failure"] or not checks["script_budget"]:
            stage = "rl_takeover" if row["handover_count"] else ("blend" if row["ever_blended"] else "script")
            reason = "script_timeout" if not checks["script_budget"] or row["failure_code"] == 6 else "controller_or_physical_failure"
        elif not checks["blend_and_single_handover"]:
            stage, reason = ("blend" if row["ever_blended"] else "script"), "missing_or_repeated_handover"
        elif not checks["pure_rl"] or not checks["continuous_rl_stability"]:
            stage, reason = "rl_stabilization", "continuous_stability_not_met"
        elif not checks["platform_retained"]:
            stage, reason = "rl_followup", "required_platform_not_retained"
        elif not checks["motion_release"] or not checks["continuous_followup"] or not checks["physical_success"]:
            stage, reason = "rl_followup", "post_handover_task_not_completed"
        else:
            stage, reason = "complete", "physical_full_chain_completed"
        outcomes.append({"episode": index, "stage": stage, "reason": reason, "failed_checks": failed})
    return outcomes


def validate_cross_asset_actor(source_contract, target_contract, source_manifest, target_manifest):
    """Permit an explicit physics comparison only when the actor/control ABI is unchanged."""
    contract_keys = ("actor_dim", "actor_frame_dim", "action_dim", "actor_layout", "history_length",
                     "actor_observation_source", "policy_action_order", "policy_dt", "physics_dt",
                     "v5_control", "control_math_source", "task_modes", "phases")
    manifest_keys = ("model_kind", "control_frame", "control_joint_names", "tree_joint_names",
                     "rigid_body_names", "spring_joint_names", "nominal_joint_pos")
    for before, after, keys in ((source_contract, target_contract, contract_keys),
                                (source_manifest, target_manifest, manifest_keys)):
        for key in keys:
            if key not in before or key not in after or before[key] != after[key]:
                raise ValueError(f"Cross-asset evaluation changes the actor/control interface: {key}")


def cross_asset_checkpoint_provenance(checkpoint_path, infos, contract, manifest, control_math_sha256):
    """Authenticate a specifically authorized old asset before actor-only migration."""
    from .full_curriculum import checkpoint_contract_path

    expected = contract.get("cross_asset_source_manifest_sha256")
    if not expected or infos.get("asset_manifest_sha256") != expected:
        raise ValueError("Cross-asset source was not authorized by this contract")
    path = Path(checkpoint_path)
    source_manifest_path = path.with_suffix(".asset_manifest.json")
    if not source_manifest_path.is_file():
        source_manifest_path = path.parent / "asset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    # Training sidecars omit the final newline of the canonical exported manifest.
    canonical = (json.dumps(source_manifest, indent=2, allow_nan=False) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != expected:
        raise ValueError("Cross-asset source manifest identity mismatch")
    source_path = checkpoint_contract_path(path)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_contract = json.loads(source_path.read_text())
    if (source_sha != infos.get("contract_sha256") or source_contract.get("asset_manifest_sha256") != expected
            or infos.get("control_math_sha256") != control_math_sha256):
        raise ValueError("Cross-asset checkpoint provenance mismatch")
    validate_cross_asset_actor(source_contract, contract, source_manifest, manifest)
    return {"source_asset_manifest_sha256": expected, "target_asset_manifest_sha256": contract["asset_manifest_sha256"],
            "source_contract_sha256": source_sha, "source_contract": str(source_path),
            "source_manifest": str(source_manifest_path), "scope": "explicit_same_abi_actor_only_asset_migration"}


def summarize_evaluation_tiers(candidate, settings, baseline=None):
    """Report retention and learning separately without rewriting fixed-case grades."""
    tiers = settings.get("tiers")
    if not tiers:
        return None
    result = {"tiers": {}, "baseline_available": baseline is not None, "retention_within_margin": None}
    for tier, names in tiers.items():
        failed = [name for name in names if not candidate["cases"][name]["passed"]]
        result["tiers"][tier] = {"passed": len(names) - len(failed), "total": len(names), "failed_cases": failed}
    if baseline is None:
        return result
    specs = {case["name"]: case for case in settings["cases"]}
    retained = {}
    for name in tiers.get("retain", []):
        before, after, spec = baseline["cases"][name], candidate["cases"][name], specs[name]
        if before["command"] != after["command"]:
            raise ValueError("Retention comparison changed the reference command")
        stationary = spec.get("stationary", spec.get("task", "survive") == "survive"
                              and abs(spec["command"][0]) < .01 and abs(spec["command"][1]) < .01)
        changes = {}
        for metric, margin in settings["retention_margins"].items():
            if metric == "stand_drift_max_m" and not stationary:
                continue
            changes[metric] = {"baseline": before[metric], "current": after[metric], "allowed_delta": margin,
                               "within_margin": after[metric] <= before[metric] + margin}
        accounted = (after["frames"] > 0 and after["episodes"] == after["requested_episodes"]
                     and after["survival_rate"] >= before["survival_rate"]
                     and after["failures"] <= before["failures"])
        retained[name] = {"within_margin": accounted and all(v["within_margin"] for v in changes.values()),
                          "outcomes_retained": accounted, "metrics": changes}
    result["retained_cases"] = retained
    result["retention_within_margin"] = bool(retained) and all(case["within_margin"] for case in retained.values())
    return result


def fixed_suite_contract(contract):
    result = deepcopy(contract)
    suite = result["evaluation"]
    cases = suite["cases"]
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Evaluation case names must be unique and nonempty")
    result["scene_groups"] = [{"name": c["name"], "fraction": 1 / len(cases), "terrain": [c.get("terrain", "flat")]} for c in cases]
    result["episode_seconds"] = suite["episode_seconds"]
    result["record_diagnostics"] = True
    result["evaluation_exact_cases"] = True
    result["diagnostic_logging"] = False
    if "fall_confirmation_seconds" in result:
        result["fall_confirmation_seconds"] = 0.
    result["evaluation_long_corridors"] = bool(result.get("task_semantics"))
    if result.get("signal_perturbations"):
        result["signal_perturbations"]["enabled_fraction"] = 1.
    result.pop("command_curriculum", None)
    for module, enabled in suite.get("frozen_module_enablement", {}).items():
        if module not in ("command_transport", "dynamics_randomization", "contact_domain", "signal_perturbations", "signal_delay") or type(enabled) is not bool:
            raise ValueError("Invalid explicit evaluation module override")
        result[module]["enabled"] = enabled
    # Training assistance can never contribute to an acceptance episode.
    if "jump_assist" in result:
        result["jump_assist"]["enabled"] = False
    if "nonstandard_start" in result:
        result["nonstandard_start"]["enabled"] = False
    if "ordinary_posture_curriculum" in result:
        result["ordinary_posture_curriculum"]["enabled"] = False
    if "release_stop" in result:
        result["release_stop"]["enabled"] = False
        # Preserve the pinned evaluation metadata, without reverse scaling
        # floats or teaching the evaluator new training profiles.
        from .nonstandard_start_revision import SOURCES, read_bound
        domain = result["task_campaign"]["domain"]
        predecessor = read_bound(*SOURCES[domain])
        result["behavior_pool_fractions"] = deepcopy(predecessor["behavior_pool_fractions"])
    if suite.get("frozen_signal_perturbations"):
        result["signal_perturbations"] = deepcopy(suite["frozen_signal_perturbations"])
        result["observation_noise_enabled"] = True
    revision = result.get("task_campaign", {}).get("revision_migration", {})
    if revision.get("schema") == "task-worker-revision-v15":
        # The sampling revision teaches new training groups. Fixed cases keep
        # the independently pinned predecessor catalogue and pool membership.
        from .terrain_rehearsal_revision import SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, read_bound
        predecessor = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA)
        for key in ("skill_specs", "behavior_pool_membership"):
            result[key] = deepcopy(predecessor[key])
    elif revision.get("schema") == "task-worker-revision-v17":
        _restore_v17_fixed_evaluation(result)
    elif revision.get("schema") == "task-worker-revision-v18":
        _restore_v18_fixed_evaluation(result)
    return result


def _restore_v17_fixed_evaluation(result):
    """Separate v17 training lessons from the SHA-bound predecessor's exam.

    New accepted-jump cycle accounting remains active: it resets observed event
    evidence, not physical state, task targets, thresholds or successful flags.
    """
    domain = result.get("task_campaign", {}).get("domain")
    binding = result["task_campaign"]["revision_migration"]
    if domain == "terrain":
        from . import scut_terrain_revision as recipe
        expected = recipe.revision_binding()
    elif domain in ("jump", "stepjump"):
        from . import scut_jump_revision as recipe
        expected = recipe.revision_binding(domain)
    elif domain == "recovery":
        from . import recovery_prefix_revision as recipe
        expected = recipe.revision_binding()
    else:
        raise ValueError("v17 fixed evaluation requires a named revised task domain")
    _restore_predecessor_fixed_evaluation(result, recipe, expected, "v17")


def _restore_v18_fixed_evaluation(result):
    """Project the actual v15 exam, never the unlaunched v17 training recipe."""
    from . import ordinary_posture_revision as recipe

    domain = result.get("task_campaign", {}).get("domain")
    if domain not in recipe.DOMAINS_TO_REVISE:
        raise ValueError("v18 fixed evaluation requires a named revised task domain")
    source_file, source_sha = recipe.source_binding(domain)
    expected = dict(schema=recipe.WORKER_SCHEMA, source_contract_file=source_file,
                    source_contract_sha256=source_sha)
    _restore_predecessor_fixed_evaluation(result, recipe, expected, "v18")


def _restore_predecessor_fixed_evaluation(result, recipe, expected, version):
    domain = result.get("task_campaign", {}).get("domain")
    binding = result["task_campaign"]["revision_migration"]
    if binding != expected:
        raise ValueError(f"{version} fixed evaluation source differs from its frozen revision")
    predecessor = recipe.read_bound(binding["source_contract_file"],
                                    binding["source_contract_sha256"])
    if predecessor.get("task_campaign", {}).get("domain") != domain:
        raise ValueError(f"{version} fixed evaluation cannot borrow another domain's exam")
    cases = {case["name"]: case for case in predecessor["evaluation"]["cases"]}
    if any(case != cases.get(case["name"]) for case in result["evaluation"]["cases"]):
        raise ValueError(f"{version} fixed evaluation changed a predecessor case or criterion")
    # Recursion deliberately applies the predecessor's own historical fixed
    # projection (notably terrain v15's v14 catalogue and standing quotas).
    fixed = fixed_suite_contract(predecessor)
    for key in ("skill_specs", "behavior_pool_membership", "behavior_pool_fractions",
                "new_asset_curriculum", "dense_tracking", "recovery_training",
                "recovery_takeover"):
        if key in fixed:
            result[key] = deepcopy(fixed[key])
        else:
            result.pop(key, None)
    if "nonstandard_start" in result:
        result["nonstandard_start"]["enabled"] = False
    if "ordinary_posture_curriculum" in result:
        result["ordinary_posture_curriculum"]["enabled"] = False
    if domain in ("jump", "stepjump") and result.get("jump_cycle_reset_on_accepted") is not True:
        raise ValueError(f"{version} jump evaluation requires its explicit accepted-cycle evidence reset")


def select_evaluation_cases(settings, names):
    """Select coverage without changing case definitions or canonical placements."""
    selected = set(names)
    available = {case["name"] for case in settings["cases"]}
    if not selected or selected - available:
        raise ValueError(f"Invalid evaluation subset: {sorted(selected - available)}")
    result = deepcopy(settings)
    result.setdefault("canonical_case_names", [case["name"] for case in settings["cases"]])
    result["cases"] = [case for case in result["cases"] if case["name"] in selected]
    result["episode_seconds"] = max(case.get("episode_seconds", settings["episode_seconds"]) for case in result["cases"])
    if "retention_case_names" in result:
        result["retention_case_names"] = [name for name in result["retention_case_names"] if name in selected]
    return result


def quick_evaluation_cases(settings):
    names = set(settings.get("quick_case_names", [])) | set(settings.get("promotion_case_names", []))
    names.update(settings.get("protected_case_names", []))
    available = {case["name"] for case in settings["cases"]}
    if names - available:
        raise ValueError("A required quick/protected case is absent from the canonical manifest")
    return [case["name"] for case in settings["cases"] if case["name"] in names]


def canonical_evaluation_layout(settings, count):
    """Keep each case/replica at the same world slot in subset and full tests."""
    cases = settings["cases"]
    if count % len(cases):
        raise ValueError("Evaluation layout requires equal replicas")
    repeats = count // len(cases)
    canonical = settings.get("canonical_case_names", [case["name"] for case in cases])
    indices = {name: i for i, name in enumerate(canonical)}
    if len(indices) != len(canonical) or {case["name"] for case in cases} - indices.keys():
        raise ValueError("Invalid canonical evaluation layout")
    return [indices[c["name"]] * repeats + replica for c in cases for replica in range(repeats)], len(canonical) * repeats


def grade_fixed_suite(metrics, settings, episodes_per_case):
    result = {"protocol_id": settings["protocol_id"], "cases": {}, "passed": True}
    failure_rates, normalized_errors = [], []
    for case in settings["cases"]:
        name, command = case["name"], case["command"]
        group = metrics["groups"][name]
        full_episodes = group["timeouts"] - group["boundary_truncations"] - group.get("blocked_truncations", 0)
        survival = full_episodes / episodes_per_case
        task = case.get("task", "survive")
        task_rate = group["successes"] / episodes_per_case
        jumping = task == "jump"
        climbing = bool(case.get("body_climb_case") or case.get("skill", {}).get("kind") == "body_climb")
        recovering = is_recovery_case(case)
        stand = case.get("stationary", task == "survive" and abs(command[0]) < .01 and abs(command[1]) < .01)
        checks = {
            "all_requested_episodes_accounted": group["episodes"] == episodes_per_case,
            "has_post_warmup_samples": group["frames"] > 0,
            "survival" if task == "survive" else "task_success": (survival if task == "survive" else task_rate) >= case.get("success_rate_min", settings["survival_rate_min"]),
            # Climbing legitimately raises/folds the chassis before settling.
            # Its platform-support ledger replaces full-episode static height
            # precision, while ordinary height cases keep their existing gate.
            "height": jumping or climbing or group["height_mae_m"] <= case.get("height_mae_m_max", settings["height_mae_m_max"]),
            "velocity": jumping or group["vx_mae_m_s"] <= case.get("velocity_mae_m_s_max", settings["velocity_mae_m_s_max"]),
            "yaw": jumping or group["yaw_mae_rad_s"] <= case.get("yaw_mae_rad_s_max", settings["yaw_mae_rad_s_max"]),
            "tilt": group["tilt_max_deg"] <= settings["tilt_max_deg"],
            "stand_drift": not stand or group["stand_drift_max_m"] <= settings["stand_drift_m_max"],
        }
        if "stand_velocity_mae_m_s_max" in settings:
            checks["stand_velocity"] = not stand or group["vx_mae_m_s"] <= settings["stand_velocity_mae_m_s_max"]
        if recovering:
            # Inverted/low poses and chassis contact are part of the prescribed
            # script. Only the physical full-chain ledger grades this case;
            # aggregate static errors remain diagnostics, not acceptance gates.
            checks = {key: checks[key] for key in ("all_requested_episodes_accounted", "has_post_warmup_samples")}
            checks.update(recovery_task_checks(case, group, episodes_per_case))
        if settings.get("mechanical_checks"):
            checks["mechanics"] = (group["closure_gap_max_m"] <= .003 and all(
                group["termination_reasons"].get(name, 0) == 0 for name in ("knee", "spring_travel", "closure_gap")))
        for metric in ("reference_velocity_error", "settled_stop_speed", "height_velocity_mae_m_s"):
            if metric + "_max" in case and not recovering:
                checks[metric] = group.get(metric, float("inf")) <= case[metric + "_max"]
        if case.get("request_protocol_case"):
            # These episodes deliberately cancel/repeat requests or continue
            # after a first jump. Their ledger defines completion; a conventional
            # single-task success flag cannot substitute for protocol behavior.
            outcome = "survival" if task == "survive" else "task_success"
            checks[outcome] = (group["failures"] == 0 and group["boundary_truncations"] == 0
                               and group.get("blocked_truncations", 0) == 0)
            checks.update(request_protocol_checks(case, group, episodes_per_case))
        checks.update(platform_task_checks(case, group, episodes_per_case))
        result["cases"][name] = {"command": command, "requested_episodes": episodes_per_case,
            "full_horizon_episodes": full_episodes, "survival_rate": survival,
            "task_success_rate": task_rate, "task": task,
            "anchor": case.get("anchor", False),
            "checks": checks, "passed": all(checks.values()), **group}
        if recovering:
            result["cases"][name]["recovery_episode_diagnostics"] = recovery_episode_diagnostics(group, episodes_per_case)
        result["passed"] &= all(checks.values())
        if recovering:
            rows = group.get("recovery_episodes", [])
            completed = sum(all(_recovery_row_checks(row).values()) for row in rows) if isinstance(rows, list) else 0
            failure_rates.append(1. - min(completed, episodes_per_case) / episodes_per_case)
        else:
            failure_rates.append(1. - (survival if task == "survive" else task_rate))
        normalized_errors.append(0. if jumping or recovering else (0. if climbing else group["height_mae_m"] / settings["height_mae_m_max"])
            + group["vx_mae_m_s"] / settings["velocity_mae_m_s_max"]
            + group["yaw_mae_rad_s"] / settings["yaw_mae_rad_s_max"]
            + (group["stand_drift_max_m"] / settings["stand_drift_m_max"] if stand else 0.))
    # Accepted policies outrank rejected ones; then prioritize surviving episodes.
    result["rank_lower_is_better"] = [int(not result["passed"]), sum(failure_rates) / len(failure_rates),
                                     sum(normalized_errors) / len(normalized_errors)]
    anchors = [case for case in result["cases"].values() if case["anchor"]]
    result["anchor_passed"] = bool(anchors) and all(case["passed"] for case in anchors)
    return result


def continuation_assessment(candidate, baseline, settings):
    """Keep already-passing nominal cases while learning harder task distributions."""
    names = settings["retention_case_names"]
    before, after = baseline["cases"], candidate["cases"]
    initial = settings.get("initial_passed_case_names")
    if initial is None:
        initial = [name for name in names if before.get(name, {}).get("passed", False)]
    protected = set(settings.get("protected_case_names", [])) | set(initial)
    lost = sorted(name for name in protected if not after.get(name, {}).get("passed", False))
    checked = (set(settings["promotion_case_names"]) | protected
               if settings.get("promotion_case_names") else set(after))
    mechanical = [name for name in sorted(checked)
                  if name in after and not after[name].get("checks", {}).get("mechanics", False)]
    passed = sum(after[name]["passed"] for name in names if name in after)
    missing = sorted(checked - after.keys())
    return {"eligible": not lost and not mechanical and not missing, "lost_parent_passes": lost,
            "missing_required_cases": missing,
            "mechanical_failures": mechanical, "nominal_passed": passed,
            "rank": [-passed, *candidate["rank_lower_is_better"]]}


def capability_gate(candidate, baseline, settings):
    """Promote declared stage targets while preserving this lineage's passed skills."""
    targets = settings.get("promotion_case_names")
    if not targets:
        return {"passed": candidate["passed"], "lost_parent_passes": []}
    missing = set(targets) - candidate["cases"].keys()
    baseline_missing = set(targets) - (baseline or {}).get("cases", {}).keys()
    if missing or baseline_missing or baseline is None:
        return {"passed": False, "status": "invalid_evaluation_coverage",
                "missing_required_cases": sorted(missing),
                "missing_baseline_cases": sorted(baseline_missing),
                "lost_parent_passes": [], "mechanical_failures": [], "failed_targets": sorted(set(targets))}
    retention = continuation_assessment(candidate, baseline, settings)
    failed = [name for name in targets if not candidate["cases"][name]["passed"]]
    return {"passed": not failed and retention["eligible"], "failed_targets": failed,
            "lost_parent_passes": retention["lost_parent_passes"],
            "mechanical_failures": retention["mechanical_failures"],
            "missing_required_cases": retention["missing_required_cases"],
            "scope": "declared_stage_targets_and_parent_passes_not_all_catalog_skills"}


def severe_retention_regressions(candidate, settings, lost_cases):
    """Distinguish rollback severity from the unchanged pass/fail acceptance gate."""
    ratio = settings.get("regression_recovery", {}).get("trigger_error_ratio")
    if ratio is None:
        return list(lost_cases)
    if not 1 <= ratio <= 5:
        raise ValueError("Rollback severity ratio must be within [1,5]")
    specs = {c["name"]: c for c in settings["cases"]}
    metrics = {"height": ("height_mae_m", "height_mae_m_max"),
               "velocity": ("vx_mae_m_s", "velocity_mae_m_s_max"),
               "yaw": ("yaw_mae_rad_s", "yaw_mae_rad_s_max"),
               "stand_drift": ("stand_drift_max_m", "stand_drift_m_max"),
               "stand_velocity": ("vx_mae_m_s", "stand_velocity_mae_m_s_max"),
               "tilt": ("tilt_max_deg", "tilt_max_deg")}
    severe = []
    for name in lost_cases:
        case = candidate["cases"].get(name)
        if case is not None and case.get("passed"):
            continue
        failures = [] if case is None else [k for k, passed in case.get("checks", {}).items() if not passed]
        if not failures:
            severe.append(name)
            continue
        for check in failures:
            if check not in metrics:
                severe.append(name)
                break
            metric, limit_key = metrics[check]
            limit = specs.get(name, {}).get(limit_key, settings.get(limit_key))
            value = case.get(metric)
            if limit is None or value is None or not math.isfinite(value) or value >= ratio * limit:
                severe.append(name)
                break
    return sorted(set(severe))
