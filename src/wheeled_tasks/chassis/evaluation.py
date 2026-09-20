"""Versioned fixed-case evaluation configuration and behavior-based acceptance."""
from copy import deepcopy


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
    result["evaluation_long_corridors"] = bool(result.get("task_semantics"))
    if result.get("signal_perturbations"):
        result["signal_perturbations"]["enabled_fraction"] = 1.
    result.pop("command_curriculum", None)
    return result


def grade_fixed_suite(metrics, settings, episodes_per_case):
    result = {"protocol_id": settings["protocol_id"], "cases": {}, "passed": True}
    failure_rates, normalized_errors = [], []
    for case in settings["cases"]:
        name, command = case["name"], case["command"]
        group = metrics["groups"][name]
        full_episodes = group["timeouts"] - group["boundary_truncations"]
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
        for metric in ("reference_velocity_error", "settled_stop_speed"):
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
