"""Versioned fixed-case evaluation configuration and behavior-based acceptance."""
from copy import deepcopy


def fixed_suite_contract(contract):
    result = deepcopy(contract)
    suite = result["evaluation"]
    cases = suite["cases"]
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Evaluation case names must be unique and nonempty")
    result["scene_groups"] = [{"name": c["name"], "fraction": 1 / len(cases), "terrain": ["flat"]} for c in cases]
    result["episode_seconds"] = suite["episode_seconds"]
    result["record_diagnostics"] = True
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
        stand = abs(command[0]) < .01 and abs(command[1]) < .01
        checks = {
            "all_requested_episodes_accounted": group["episodes"] == episodes_per_case,
            "has_post_warmup_samples": group["frames"] > 0,
            "survival": survival >= settings["survival_rate_min"],
            "height": group["height_mae_m"] <= settings["height_mae_m_max"],
            "velocity": group["vx_mae_m_s"] <= settings["velocity_mae_m_s_max"],
            "yaw": group["yaw_mae_rad_s"] <= settings["yaw_mae_rad_s_max"],
            "tilt": group["tilt_max_deg"] <= settings["tilt_max_deg"],
            "stand_drift": not stand or group["stand_drift_max_m"] <= settings["stand_drift_m_max"],
        }
        result["cases"][name] = {"command": command, "requested_episodes": episodes_per_case,
            "full_horizon_episodes": full_episodes, "survival_rate": survival,
            "checks": checks, "passed": all(checks.values()), **group}
        result["passed"] &= all(checks.values())
        failure_rates.append(1. - survival)
        normalized_errors.append(group["height_mae_m"] / settings["height_mae_m_max"]
            + group["vx_mae_m_s"] / settings["velocity_mae_m_s_max"]
            + group["yaw_mae_rad_s"] / settings["yaw_mae_rad_s_max"]
            + (group["stand_drift_max_m"] / settings["stand_drift_m_max"] if stand else 0.))
    # Accepted policies outrank rejected ones; then prioritize surviving episodes.
    result["rank_lower_is_better"] = [int(not result["passed"]), sum(failure_rates) / len(failure_rates),
                                     sum(normalized_errors) / len(normalized_errors)]
    return result
