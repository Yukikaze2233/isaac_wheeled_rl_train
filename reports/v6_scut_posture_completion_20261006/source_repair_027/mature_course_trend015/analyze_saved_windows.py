#!/usr/bin/env python3
"""Finite CPU analysis of the fourteen saved native windows; no event access."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError("Aggregation refused: " + message)


def same(a, b, message):
    require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-9), message)


def complete_stat(row, message):
    require(row.get("complete") is True and row.get("records") == 100
            and row.get("expected_records") == 100, message)


def weighted(groups, names, fields, count_key):
    selected = [groups[n] for n in names if groups[n][count_key] > 0]
    total = sum(r[count_key] for r in selected)
    require(total > 0, "empty weighted population")
    return total, {field: math.fsum(r[field] * r[count_key] for r in selected) / total
                   for field in fields}


def stable_identity(raw):
    identity = raw["identity"]
    owner = identity["native_owner"]
    return {"source_commit": raw["source_commit"], "source_root": identity["source_root"],
            "event": identity["event"], "native_owner": {k: owner[k] for k in
                ("pid", "ppid", "starttime_ticks", "command")},
            "source_files_sha256": identity["source_files_sha256"],
            "actual_receipts_sha256": raw["actual_receipts_sha256"],
            "invocation": identity["invocation"], "groups": sorted(identity["groups"]),
            "base_groups": sorted(identity["base_groups"]),
            "group_pools": identity["group_pools"], "population": identity["population"]}


def main():
    preserved = {str(p): sha(p) for p in sorted((BASE / "mature_course_trend014").glob("*"))
                 if p.is_file()}
    require(bool(preserved), "existing500 trend014 not found")
    inputs, review_refs, reviews, windows = {}, {}, [], []
    cohort, prior_end, prior_lesson_end = None, None, None
    for index in range(1, 15):
        directory = BASE / f"learning_status{index:03}"
        raw_path = next(directory.glob("*.raw001.json"))
        summary_path = next(directory.glob("*.summary001.json"))
        arithmetic_path = next(directory.glob("*.arithmetic001.json"))
        review_path = sorted(directory.glob("saved_evidence_review*.json"))[-1]
        raw, summary, arithmetic, review = map(read, (raw_path, summary_path, arithmetic_path, review_path))
        for path in (raw_path, summary_path, arithmetic_path, review_path):
            inputs[str(path)] = sha(path)
        require(all(review["checks"].values()) and review["inputs_unchanged"] is True,
                f"window{index}: saved review not passed")
        require(review["qualification"] is None, f"window{index}: qualification scope")
        for path, expected in review["input_SHA"].items():
            require(Path(path).is_file() and sha(path) == expected,
                    f"window{index}: saved review input SHA mismatch {path}")
            review_refs[path] = expected
        require(summary["report_sha256"] == inputs[str(raw_path)]
                == arithmetic["report_sha256"], f"window{index}: raw projection SHA")
        # Saved summaries append explanatory scope notes; the complete arithmetic
        # payload (including its original limitations separately below) is frozen.
        require(all(summary[k] == value for k, value in arithmetic.items() if k != "limitations"),
                f"window{index}: arithmetic/summary projection mismatch")
        require(raw["complete"] is True and raw["owner_continuity_verified"] is True
                and raw["source_files_unchanged"] is True
                and raw["immutable_metadata_unchanged"] is True
                and raw["PPO_updates_by_reader"] == 0 and raw["qualification"] is None,
                f"window{index}: source/owner/read-only scope")
        identity = stable_identity(raw)
        if cohort is None:
            cohort = identity
        require(identity == cohort, f"window{index}: source/owner/population mismatch")
        clock, native = summary["clock"], raw["window"]
        actor = [clock["baseline_update"] + 1, clock["end_update"]]
        rec, tb = clock["task_actor_updates_inclusive"], clock["event_steps_inclusive"]
        require(actor == native["actor_updates_inclusive"]
                and rec == native["task_actor_updates_inclusive"]
                and tb == native["event_steps_inclusive"]
                and actor == [5301 + index * 100, 5400 + index * 100]
                and rec == [a + 6425 for a in actor]
                and tb == [a + 33629 for a in actor]
                and clock["actor_updates"] == 100
                and clock["critic_warmup_excluded"] is True
                and clock["new_critic_warmup_updates"] == 10,
                f"window{index}: inconsistent clocks")
        if prior_end is not None:
            require([actor[0], rec[0], tb[0]] == [n + 1 for n in prior_end],
                    f"window{index}: gap/overlap")
        prior_end = [actor[1], rec[1], tb[1]]
        w = raw["windows"]["100"]
        groups, pools = w["behavior"]["groups"], w["behavior"]["pools"]
        require(w["complete"] is True and not w["missing"]
                and w["missing_required_tag_count"] == 0,
                f"window{index}: native window incomplete")
        require(len(groups) == 95 and len(pools) == 7
                and len(raw["identity"]["base_groups"]) == 41
                and sum(r["population"] for r in groups.values()) == 4096
                and {n: r["population"] for n, r in groups.items()} == cohort["population"],
                f"window{index}: group/population mismatch")
        counts = ("frames", "observed_frames", "episodes", "successes", "failures", "timeouts",
                  "boundary_truncations", "blocked_truncations")
        for name, row in groups.items():
            require(row["complete"] is True and row["frames"] == row["population"] * 2400
                    and row["observed_frames"] == row["frames"]
                    and row["episodes"] == row["successes"] + row["failures"] + row["timeouts"],
                    f"window{index}: native group counts {name}")
        require(sum(r["frames"] for r in groups.values()) == 9830400
                == clock["expected_observed_frames"], f"window{index}: total frames")
        for pool, pooled in summary["pools"].items():
            names = [n for n in groups if cohort["group_pools"][n] == pool]
            for field in counts:
                require(pooled[field] == sum(groups[n][field] for n in names),
                        f"window{index}: pool counts {pool}/{field}")
            mean_fields = [k for k in pooled if k.endswith("_m_s") or k.endswith("_rad_s")
                           or k in ("height_mae_m", "height_actual_mean_m", "height_command_mean_m")]
            _, means = weighted(groups, names, mean_fields, "frames")
            for field, value in means.items():
                same(pooled[field], value, f"window{index}: pool mean {pool}/{field}")
        rec_names = [n for n in groups if cohort["group_pools"][n] == "scripted_recovery"]
        require(len(rec_names) == 54, f"window{index}: mixed REC54 coverage")
        metrics = {}
        for height in (230, 305, 430):
            name = f"stand_{height}mm"
            for field, suffix in (("planar_speed_mean_m_s", "speed_m_s"), ("height_mae_m", "height_mae_m")):
                same(summary["ordinary_stand"][name][field], groups[name][field], "stand projection")
                metrics[f"stand{height}_{suffix}"] = groups[name][field]
        for name, prefix in (("forward_05", "forward05"), ("backward_05", "backward05"),
                             ("forward_1", "forward1"), ("backward_1", "backward1")):
            for field, suffix in (("vx_actual_mean_m_s", "actual_m_s"),
                                  ("vx_command_mean_m_s", "command_m_s"), ("vx_mae_m_s", "mae_m_s")):
                same(summary["motion"][name][field], groups[name][field], "motion projection")
                metrics[f"{prefix}_{suffix}"] = groups[name][field]
        for name, prefix in (("rotate_1", "yaw_positive1"), ("rotate_1__reverse_train", "yaw_negative1")):
            same(summary["yaw_mae_rad_s"][name], groups[name]["yaw_mae_rad_s"], "yaw projection")
            metrics[f"{prefix}_mae_rad_s"] = groups[name]["yaw_mae_rad_s"]
        stops = w["optional_conditional_stop_phases"]["groups"]
        require(w["optional_conditional_stop_phases"]["complete"] is True, "stop coverage")
        for phase in ("braking", "settled_stop"):
            phase_groups = {n: v[phase] for n, v in stops.items()}
            total, means = weighted(phase_groups, list(phase_groups),
                                    ("mean_speed_m_s", "above_3cm_s_fraction"), "samples")
            saved = summary["conditional_stop_phases"][phase]
            require(saved["samples"] == total, "stop samples")
            for field, value in means.items():
                same(saved[field], value, "stop weighted projection")
                metrics[f"{phase}_{field}"] = value
            peak = max(v["peak_speed_m_s"] for v in phase_groups.values())
            same(saved["peak_speed_m_s"], peak, "stop peak projection")
            metrics[f"{phase}_samples"] = total
            metrics[f"{phase}_peak_speed_m_s"] = peak
        stationary = w["stationary_10s"]["groups"]
        require(w["stationary_10s"]["complete"] is True, "stationary coverage")
        for population, names in (("all", list(groups)), ("base", cohort["base_groups"]),
                                  ("recovery", rec_names)):
            total, means = weighted(stationary, names,
                ("mean_speed_m_s", "endpoint_mean_m", "peak_mean_m"), "samples")
            saved = summary["stationary_10s"][population]
            require(saved["samples"] == total, "stationary samples")
            for field, value in means.items():
                same(saved[field], value, "stationary weighted projection")
                metrics[f"stationary10s_{population}_{field}"] = value
            peak = max(stationary[n]["peak_max_m"] for n in names if stationary[n]["samples"] > 0)
            same(saved["peak_max_m"], peak, "stationary peak projection")
            metrics[f"stationary10s_{population}_samples"] = total
            metrics[f"stationary10s_{population}_peak_max_m"] = peak
        lessons = w["dynamic_lessons"]
        require(lessons["complete"] is True, "lesson complete")
        for level, target in (("prefix_stand", .2), ("prefix_release", .3), ("original_chain", .5)):
            probability = lessons["levels"][level]["fields"]["probability"]
            complete_stat(probability, "lesson probability coverage")
            require(probability["min"] == probability["max"] == probability["mean"]
                    == probability["median"] and abs(probability["mean"] - target) < 1e-7,
                    "difficulty probability mismatch")
            metrics[f"lesson_{level}_probability"] = probability["mean"]
            metrics[f"lesson_{level}_population_mean"] = summary["dynamic_lessons"][level]["population_mean"]
        release = w["teaching"]["release_profile_seconds"]
        complete_stat(release, "release coverage")
        require(release["min"] == release["max"] == release["mean"] == release["median"]
                and abs(release["mean"] - .2) < 1e-7, "release difficulty mismatch")
        metrics["lesson_release_seconds"] = release["mean"]
        lesson_age = summary["lesson_relative_actor_updates"]
        complete_stat(lesson_age, "lesson age coverage")
        require(lesson_age["max"] - lesson_age["min"] == 99 and lesson_age["min"] >= 6000,
                "course not mature throughout")
        if prior_lesson_end is not None:
            require(lesson_age["min"] == prior_lesson_end + 1, "lesson clock gap")
        prior_lesson_end = lesson_age["max"]
        rec_pool = summary["pools"]["scripted_recovery"]
        for source, destination in (("successes", "successes"), ("episodes", "ended"),
                                    ("failures", "failures"), ("timeouts", "timeouts"),
                                    ("boundary_truncations", "boundary_truncations")):
            metrics[f"mixed54_{destination}"] = rec_pool[source]
        metrics["mixed54_native_success_per_ended"] = rec_pool["successes"] / rec_pool["episodes"]
        same(metrics["mixed54_native_success_per_ended"], rec_pool["native_success_per_ended_episode"],
             "mixed54 denominator")
        ownership = summary["learning_ownership"]
        require(ownership == w["learning_ownership"] and ownership["complete"] is True
                and ownership["valid_transitions_total"] + ownership["excluded_transitions_total"]
                == ownership["observed_transitions_total"] == 9830400, "PPO ownership counts")
        for field in ("valid_transitions", "script_or_blend_transitions", "valid_fraction"):
            complete_stat(ownership[field], "PPO ownership coverage")
        metrics.update(valid_PPO_transitions=ownership["valid_transitions_total"],
            excluded_transitions=ownership["excluded_transitions_total"],
            observed_transitions=ownership["observed_transitions_total"],
            valid_PPO_fraction=ownership["valid_transitions_total"] / ownership["observed_transitions_total"])
        require(summary["iteration_timing"] == w["timing"], "timing projection")
        for field, stat in summary["iteration_timing"].items():
            complete_stat(stat, "timing coverage")
            metrics[field + "_mean"] = stat["mean"]
        same(summary["sum_logged_iteration_wall_seconds"], metrics["iteration_wall_seconds_mean"] * 100,
             "iteration sum")
        metrics["sum_logged_iteration_wall_seconds"] = summary["sum_logged_iteration_wall_seconds"]
        windows.append({"window_index": index, "learning_status": directory.name,
            "actor_updates_inclusive": actor, "REC_actor_updates_inclusive": rec,
            "TB_steps_inclusive": tb, "lesson_relative_actor_updates_inclusive":
                [lesson_age["min"], lesson_age["max"]],
            "raw_queried_at_utc": raw["queried_at_utc"],
            "summary_created_at_utc": summary["created_at_utc"], "metrics": metrics,
            "raw_file": str(raw_path), "summary_file": str(summary_path),
            "saved_review_file": str(review_path),
            "saved_summary_limitations": summary["limitations"],
            "arithmetic_limitations": arithmetic["limitations"],
            "summary_scope_notes_differ_from_arithmetic": summary["limitations"] != arithmetic["limitations"]})
        reviews.append({"window_index": index, "review_file": str(review_path),
            "saved_check_count": review["check_count"], "all_saved_checks_passed": True,
            "review_input_SHA_count": len(review["input_SHA"])})
    statistics = {}
    for field in windows[0]["metrics"]:
        values = [w["metrics"][field] for w in windows]
        tail = values[-5:]
        statistics[field] = {"first": values[0], "last": values[-1],
            "last_minus_first": values[-1] - values[0], "all14_min": min(values),
            "all14_max": max(values), "first5_mean": math.fsum(values[:5]) / 5,
            "last5_min": min(tail), "last5_max": max(tail),
            "last5_span": max(tail) - min(tail), "last5_mean": math.fsum(tail) / 5,
            "last5_last_minus_first": tail[-1] - tail[0],
            "monotonic_all14_nondecreasing": all(a <= b for a, b in zip(values, values[1:])),
            "monotonic_all14_nonincreasing": all(a >= b for a, b in zip(values, values[1:]))}
    for path, expected in {**inputs, **review_refs, **preserved}.items():
        require(sha(path) == expected, "input or existing500 trend changed during analysis")
    now = datetime.now(timezone.utc).isoformat()
    result = {"schema": "source023-native-mature-course-saved14-window-trend-v1",
        "status": "complete_saved_evidence_descriptive_trend", "created_at_utc": now,
        "window_count": 14, "actor_updates": 1400,
        "actor_updates_inclusive": [5401, 6800], "REC_actor_updates_inclusive": [11826, 13225],
        "TB_steps_inclusive": [39030, 40429], "source_and_owner": cohort,
        "observed_frames": 14 * 9830400, "windows": windows,
        "first_last_last5_statistics": statistics,
        "totals": {"valid_PPO_transitions": sum(w["metrics"]["valid_PPO_transitions"] for w in windows),
            "excluded_transitions": sum(w["metrics"]["excluded_transitions"] for w in windows),
            "mixed54_successes": sum(w["metrics"]["mixed54_successes"] for w in windows),
            "mixed54_ended": sum(w["metrics"]["mixed54_ended"] for w in windows),
            "sum_logged_iteration_wall_seconds": math.fsum(w["metrics"]["sum_logged_iteration_wall_seconds"] for w in windows)},
        "denominator_scope": {"ordinary": "physical frames weighted within each native group",
            "conditional_stop": "six transition groups; conditional phase samples, not proof of physical stability; peak is this100-update window maximum",
            "stationary10s": "completed eligible10s windows, sample-weighted, separate base/recovery populations",
            "mixed54": "native successes / ended episodes (successes+failures+timeouts), mixed prefix_stand/prefix_release/original_chain; boundary truncations reported separately",
            "valid_PPO": "whole-interval policy_learning_mask valid transitions / observed transitions",
            "timing": "logged actual PPO iteration wall time; includes native diagnostics/logger; not reader duration"},
        "interpretation_limits": ["Descriptions compare14 successive stochastic training windows with the same saved source/owner/population and mature prefix probabilities; random trajectories and DR draws differ.",
            "First-last deltas and last5 ranges do not establish causality, statistical significance, or a proven optimization plateau.",
            "Mixed54 rates are not pure original-chain/L2 ended rates; instantaneous full-chain fractions are not substituted for an unavailable pure-L2 denominator.",
            "No deterministic mean-policy evaluation or qualification follows from this descriptive trend.",
            "No new remote/event/native/GPU/PPO operation was performed; saved window evidence alone was read."],
        "qualification": None, "PPO_updates_by_analysis": 0, "SSH_calls": 0,
        "event_parses": 0, "native_scene_actions": 0, "GPU_actions": 0}
    result["totals"]["mixed54_native_success_per_ended"] = result["totals"]["mixed54_successes"] / result["totals"]["mixed54_ended"]
    result["totals"]["valid_PPO_fraction"] = result["totals"]["valid_PPO_transitions"] / result["observed_frames"]
    path = HERE / "trend001.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    evidence = {"schema": "saved14-window-trend-input-hashes-v1", "created_at_utc": now,
        "status": "all_input_hashes_and_cohort_checks_passed", "primary_input_sha256": inputs,
        "saved_review_referenced_input_sha256": review_refs,
        "saved_review_reference_bindings_checked": sum(r["review_input_SHA_count"] for r in reviews),
        "saved_review_unique_referenced_files_checked": len(review_refs), "saved_reviews": reviews,
        "cohort_checks": ["input hashes", "raw/arithmetic/summary projection", "frozen source and owner identity",
            "contiguous actor/REC/TB clocks", "95 groups/7 pools/4096 population", "per-group frame and episode counts",
            "frame-weighted pool means", "sample-weighted stop and10s means", "mature unchanged probabilities/release",
            "mixed54 denominator", "valid PPO population", "logged PPO timing", "all inputs unchanged after analysis"],
        "preserved500_trend014_sha256": preserved, "output_sha256": {str(path): sha(path)},
        "analysis_script_sha256": sha(Path(__file__)), "PPO_updates": 0, "SSH_calls": 0,
        "event_parses": 0, "qualification": None}
    (HERE / "evidence_sha256001.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "trend": str(path), "SHA256": sha(path),
        "input_reference_bindings": evidence["saved_review_reference_bindings_checked"],
        "unique_input_reference_files": len(review_refs), "totals": result["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
