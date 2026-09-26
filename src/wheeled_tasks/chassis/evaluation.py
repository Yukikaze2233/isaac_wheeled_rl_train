"""Versioned fixed-case evaluation configuration and behavior-based acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


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
    if "fall_confirmation_seconds" in result:
        result["fall_confirmation_seconds"] = 0.
    result["evaluation_long_corridors"] = bool(result.get("task_semantics"))
    if result.get("signal_perturbations"):
        result["signal_perturbations"]["enabled_fraction"] = 1.
    result.pop("command_curriculum", None)
    if suite.get("frozen_signal_perturbations"):
        result["signal_perturbations"] = deepcopy(suite["frozen_signal_perturbations"])
        result["observation_noise_enabled"] = True
    return result


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
        stand = case.get("stationary", task == "survive" and abs(command[0]) < .01 and abs(command[1]) < .01)
        checks = {
            "all_requested_episodes_accounted": group["episodes"] == episodes_per_case,
            "has_post_warmup_samples": group["frames"] > 0,
            "survival" if task == "survive" else "task_success": (survival if task == "survive" else task_rate) >= case.get("success_rate_min", settings["survival_rate_min"]),
            "height": jumping or group["height_mae_m"] <= case.get("height_mae_m_max", settings["height_mae_m_max"]),
            "velocity": jumping or group["vx_mae_m_s"] <= case.get("velocity_mae_m_s_max", settings["velocity_mae_m_s_max"]),
            "yaw": jumping or group["yaw_mae_rad_s"] <= case.get("yaw_mae_rad_s_max", settings["yaw_mae_rad_s_max"]),
            "tilt": group["tilt_max_deg"] <= settings["tilt_max_deg"],
            "stand_drift": not stand or group["stand_drift_max_m"] <= settings["stand_drift_m_max"],
        }
        if "stand_velocity_mae_m_s_max" in settings:
            checks["stand_velocity"] = not stand or group["vx_mae_m_s"] <= settings["stand_velocity_mae_m_s_max"]
        if settings.get("mechanical_checks"):
            checks["mechanics"] = (group["closure_gap_max_m"] <= .003 and all(
                group["termination_reasons"].get(name, 0) == 0 for name in ("knee", "spring_travel", "closure_gap")))
        for metric in ("reference_velocity_error", "settled_stop_speed", "height_velocity_mae_m_s"):
            if metric + "_max" in case:
                checks[metric] = group.get(metric, float("inf")) <= case[metric + "_max"]
        result["cases"][name] = {"command": command, "requested_episodes": episodes_per_case,
            "full_horizon_episodes": full_episodes, "survival_rate": survival,
            "task_success_rate": task_rate, "task": task,
            "anchor": case.get("anchor", False),
            "checks": checks, "passed": all(checks.values()), **group}
        result["passed"] &= all(checks.values())
        failure_rates.append(1. - (survival if task == "survive" else task_rate))
        normalized_errors.append(0. if jumping else group["height_mae_m"] / settings["height_mae_m_max"]
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
    initial = settings.get("initial_passed_case_names", [name for name in names if before[name]["passed"]])
    protected = set(settings.get("protected_case_names", [])) | set(initial)
    lost = sorted(name for name in protected if not after[name]["passed"])
    checked = (set(settings["promotion_case_names"]) | protected
               if settings.get("promotion_case_names") else set(after))
    mechanical = [name for name in sorted(checked) if not after[name].get("checks", {}).get("mechanics", False)]
    passed = sum(after[name]["passed"] for name in names)
    return {"eligible": not lost and not mechanical, "lost_parent_passes": lost,
            "mechanical_failures": mechanical, "nominal_passed": passed,
            "rank": [-passed, *candidate["rank_lower_is_better"]]}


def capability_gate(candidate, baseline, settings):
    """Promote declared stage targets while preserving this lineage's passed skills."""
    targets = settings.get("promotion_case_names")
    if not targets:
        return {"passed": candidate["passed"], "lost_parent_passes": []}
    missing = set(targets) - candidate["cases"].keys()
    if missing or baseline is None:
        raise ValueError("Capability gating requires complete cases and an authenticated parent baseline")
    retention = continuation_assessment(candidate, baseline, settings)
    failed = [name for name in targets if not candidate["cases"][name]["passed"]]
    return {"passed": not failed and retention["eligible"], "failed_targets": failed,
            "lost_parent_passes": retention["lost_parent_passes"],
            "mechanical_failures": retention["mechanical_failures"],
            "scope": "declared_stage_targets_and_parent_passes_not_all_catalog_skills"}
