#!/usr/bin/env python3
"""Compare an authenticated physical-prefix H/C pair using only recorded data.

This stdlib-only reader neither imports the simulator/Torch nor restores a PT.
It does not reconstruct gates, predict gains, select a model or grant qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

CORE_REVISION = "b4ce80964d47c0127e0b3d9929ad35a620bbf204"
RECOVERY_FILE = "src/wheeled_tasks/chassis/recovery.py"
PREPARE_CHANGE = {"file": RECOVERY_FILE,
    "old": 'speed[preparing | thrusting] = self.config["push_speed_rad_s"]',
    "new": 'speed[thrusting] = self.config["push_speed_rad_s"]'}
WORKER_SHA = "f6aa77c6808c9746dc8dee431e8b3331b6b0f883db701335e57069e7df738ee8"
PT_SHA = "7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f"
ENGINEERING_SHA = "a770d28ab43904295bed212b098ae5b51fb94258b68d40ce3b9330a4ae2273ae"
GATES = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")
PARTITIONS = ("success", "failure", "timeout", "boundary_truncation", "right_censored_unfinished")
ACTIVE_ORDER = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
LEG_TARGET_IDS = (0, 1, 3, 4)
DT = .005
EVENTS = ("phase_PREPARE", "phase_BLEND", "blend_entry_pre_effort",
          "blend_first_applied_post_physics", "blend_last_mixed_post_physics",
          "first_pure_rl_post_physics", "sensor_hold_dwell_complete",
          "truth_hold_dwell_complete", "motion_unlock", "followup_success", "terminal_pre_autoreset")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    result = json.loads(Path(path).read_text(), parse_constant=reject)
    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), f"Nonfinite value in {path}")
        elif isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
    finite(result)
    return result


def read_lines(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        require(line.strip(), f"Blank record at {path}:{number}")
        # Use the same strict nonfinite handling as JSON files.
        row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {x}")))
        require(isinstance(row, dict), f"Not a record at {path}:{number}")
        rows.append(row)
    return rows


def stats(values):
    if not values:
        return None
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "Nonfinite/non-numeric statistic")
    ordered = sorted(values)
    return {"samples": len(values), "first": values[0], "last": values[-1],
            "min": ordered[0], "max": ordered[-1], "mean": statistics.fmean(values),
            "mean_abs": statistics.fmean(abs(v) for v in values), "max_abs": max(abs(v) for v in values),
            "p95_abs": sorted(abs(v) for v in values)[max(0, math.ceil(.95 * len(values)) - 1)]}


def vector(row, key, size):
    value = row[key]
    require(isinstance(value, list) and len(value) == size
            and all(type(x) in (int, float) and math.isfinite(x) for x in value), f"Invalid {key}")
    return value


def attitude(row):
    x, y, z, w = vector(row, "root_pose_world_xyzw", 7)[3:]
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    require(abs(norm - 1.) < 2e-4, "Nonunit physical quaternion")
    x, y, z, w = (v / norm for v in (x, y, z, w))
    gx, gy, gz = 2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y))
    return {"gravity_pitch_deg": math.degrees(math.atan2(gx, -gz)),
            "gravity_roll_deg": math.degrees(math.atan2(-gy, -gz)),
            "tilt_deg": math.degrees(math.acos(max(-1., min(1., -gz))))}


def longest(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def authenticated_source(report, expected_mode):
    require(report.get("source_identity_final_verified") is True, "External source identity was not finally verified")
    source = report["frozen_physical_control_sources"]
    require(source.get("mode") == expected_mode, f"Wrong source mode: expected {expected_mode}")
    path = Path(source["source_identity_path"])
    require(sha(path) == source["source_identity_sha256"], "External source identity SHA mismatch")
    identity = read_json(path)
    require(identity["schema"] == "recovery-prefix-source-identity-v1", "Unknown source identity schema")
    for identity_key, report_key in (("source_revision", "revision"), ("base_core_revision", "base_revision"),
                                    ("harness_revision", "harness_revision"), ("mode", "mode"),
                                    ("core_files_sha256", "core_files_sha256"),
                                    ("runtime_sources_sha256", "runtime_sources_sha256"),
                                    ("candidate_change", "candidate_change"), ("candidate_diff_sha256", "candidate_diff_sha256"),
                                    ("engineering_expectations_sha256", "engineering_expectations_sha256")):
        require(identity[identity_key] == source[report_key], f"Source receipt mismatch: {identity_key}")
    require(identity["base_core_revision"] == CORE_REVISION, "Changed baseline physical core")
    require(identity["runtime_sources_sha256"] == report["source_files_sha256"], "Recorded runtime source SHA set mismatch")
    require(identity["engineering_expectations_sha256"] == ENGINEERING_SHA, "Changed independent birth expectations")
    source_root = Path(identity["source_root"])
    require(source_root.is_absolute(), "Source root must be absolute")
    def git(*args):
        return subprocess.run(["git", *args], cwd=source_root, check=True, capture_output=True).stdout.decode().strip()
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == source_root.resolve(), "Source is not a complete Git checkout")
    require(git("rev-parse", "HEAD") == identity["source_revision"], "Frozen source HEAD changed")
    require(not git("status", "--porcelain", "--untracked-files=all"), "Frozen source checkout is no longer clean")
    for name, digest in identity["runtime_sources_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe source filename")
        require(sha(source_root / name) == digest, f"Frozen source bytes changed: {name}")
    require(source["actual_preflight_sha256"] == identity["runtime_sources_sha256"]["src/wheeled_tasks/chassis/preflight.py"],
            "Actual preflight SHA differs from frozen source")
    if expected_mode == "baseline":
        require(identity["source_revision"] == identity["harness_revision"] and identity["candidate_change"] is None,
                "Baseline is not common H")
        require(identity["core_files_sha256"] == identity["base_core_files_sha256"], "H altered baseline core")
    else:
        require(identity["source_revision"] != identity["harness_revision"] and identity["candidate_change"] == PREPARE_CHANGE,
                "Candidate is not the approved one-line PREPARE change")
    return identity, {"path": str(path.resolve()), "sha256": sha(path), "source_root": str(source_root),
                      "harness_revision": identity["harness_revision"], "source_revision": identity["source_revision"]}


def load_run(directory, mode):
    directory = Path(directory).resolve()
    report = read_json(directory / "report.json")
    require(report["schema"] == "v6-native-recovery-physical-prefix-v1", "Unknown prefix report")
    require(report.get("execution_status") == "completed" and report.get("status") == "engineering_probe_completed",
            "Both physical probes must be completed")
    require(report.get("num_envs") == 4 and report.get("run_role") == "full", "N4 full role required")
    require(report.get("formal_updates_consumed") == 0 and report.get("optimizer_steps") == 0, "Diagnostic consumed formal updates")
    require(report.get("actor_tensors_unchanged") is True and report.get("checkpoint_unchanged") is True,
            "PT or actor tensors changed")
    require(report.get("actor_weights_loaded_strictly") is True and report.get("source_files_changed") == [],
            "Actor load/source integrity failed")
    require(report.get("qualification") is False and report.get("physx_internal_state_restore_claim") is False,
            "Unexpected qualification/exact-restore claim")
    require(report.get("randomization_or_delay_overrides") == [], "Diagnostic condition overrides changed")
    require(report.get("source_worker_sha256") == WORKER_SHA and report["checkpoint"]["checkpoint_sha256"] == PT_SHA,
            "Not the frozen worker/flat PT")
    require(report["checkpoint"]["actor_only_deterministic_inference"] is True
            and report["checkpoint"]["seal_and_infos_authenticated"] is True, "Unverified actor/PT provenance")
    require(report.get("requested_episode_seconds") == 20. and report.get("signal_delay_enabled") is False
            and report.get("command_transport_enabled") is True, "Changed bounded full-role delay/time conditions")
    timing = report["control_timing"]
    require(timing["physics_dt"] == DT and timing["policy_dt"] == .02
            and timing["pd_updates"] == timing["feedback_refreshes"] == 4 and timing["target_decodes"] == 1,
            "Changed actual control timing")
    require(report["physics_steps"] == report["policy_steps"] * 4 == timing["physics_steps"]
            and abs(report["elapsed_physics_s"] - report["physics_steps"] * DT) < 1e-9,
            "Physical/policy clock mismatch")
    require(report["startup"]["runtime_preflight"]["status"] == "runtime_configuration_passed"
            and report["startup"]["gpu_dynamics_enabled"] is True, "Native runtime was not independently checked")
    require(tuple(report["startup"]["active_joint_order"]) == ACTIVE_ORDER, "Changed native control order")
    require(len(report["native_joint_names"]) == len(set(report["native_joint_names"])) == 18
            and len(report["joint_position_units"]) == 18, "Invalid named joint ABI")
    identity, source_receipt = authenticated_source(report, mode)
    files = ("report.json", "events.jsonl", "physical_frames.jsonl", "policy_frames.jsonl",
             "effective_contract.json", "engineering_expectations.json", "event_snapshots.pt")
    provenance = {name: {"path": str(directory / name), "sha256": sha(directory / name),
                         "bytes": (directory / name).stat().st_size} for name in files}
    require(provenance["effective_contract.json"]["sha256"] == report["contract_sha256"], "Effective contract SHA mismatch")
    require(provenance["engineering_expectations.json"]["sha256"] == ENGINEERING_SHA, "Engineering projection SHA mismatch")
    contract = read_json(directory / "effective_contract.json")
    binding = contract["design_preflight"]
    require(Path(binding["file"]).is_absolute() and Path(binding["file"]).resolve() == directory / "engineering_expectations.json"
            and binding["sha256"] == ENGINEERING_SHA, "design_preflight.file is not this absolute output")
    events = read_lines(directory / "events.jsonl")
    frames = read_lines(directory / "physical_frames.jsonl")
    policies = read_lines(directory / "policy_frames.jsonl")
    summary = report["summary"]
    require(summary["fixed_birth_denominator"] == 4 and len(summary["rows"]) == 4, "First-cohort denominator changed")
    cases = [r["case"] for r in summary["rows"]]
    require(len(set(cases)) == 4 and set(cases) == set(report["case_names"])
            and report["startup"]["scene_group_counts"] == {name: 1 for name in cases}, "Actual case coverage changed")
    require(len(events) == summary["event_count"] and len(frames) == summary["physical_frame_count"]
            and len(policies) == summary["policy_frame_count"], "Recorded frame/event counts differ from receipt")
    counts = Counter()
    by_case = {}
    lanes = set()
    for row in summary["rows"]:
        name, lane = row["case"], row["lane"]
        require(type(lane) is int and 0 <= lane < 4 and lane not in lanes and row["seed"] == report["seed"]
                and row["birth_id"] == 1 and row["replica"] == 0, "Invalid first-cohort identity")
        lanes.add(lane)
        require(row["partition"] in PARTITIONS, "Unknown first-cohort partition")
        counts[row["partition"]] += 1
        terminal = row["terminal"]
        if terminal is None:
            require(row["partition"] == "right_censored_unfinished"
                    and abs(row["censor_time_s"] - report["elapsed_physics_s"]) < 1e-9, "Invalid right censor")
            until = row["censor_time_s"]
        else:
            expected_partition = ("failure" if terminal["terminated"] else "success" if terminal["success"] else
                                  "boundary_truncation" if terminal["boundary"] else "timeout" if terminal["timeout"] else None)
            require(row["partition"] == expected_partition, "Native terminal partition mismatch")
            until = terminal["time_s"]
        require(type(until) in (int, float) and math.isfinite(until) and 0 < until <= 20., "Invalid cohort window")
        case_events = [e for e in events if e["lane"] == lane]
        first = {}
        for event in case_events:
            require(event["case"] == name and event["birth_id"] == 1, "Event/case identity mismatch")
            require(0 <= event["physical_tick"] <= report["physics_steps"]
                    and abs(event["sim_time_s"] - event["physical_tick"] * DT) < 1e-9
                    and event["sim_time_s"] <= until + 1e-9, "Invalid event physical clock")
            first.setdefault(event["event"], event["sim_time_s"])
        require(first == row["events"] and first.get("birth_complete") == 0., "Events differ from first-cohort ledger")
        if terminal is not None:
            require(first.get("terminal_pre_autoreset") == until, "Terminal was not captured before reset")
        case_frames = [f for f in frames if f["lane"] == lane]
        require(len(case_frames) == round(until / DT), "First-cohort physical coverage is incomplete")
        for tick, frame in enumerate(case_frames, 1):
            require(frame["case"] == name and frame["physical_tick"] == tick
                    and abs(frame["sim_time_s"] - tick * DT) < 1e-9, "Physical frames are not contiguous first cohort")
            require(set(frame["gates"]) == set(GATES) and all(type(v) is bool for v in frame["gates"].values()),
                    "Native eight gate records missing/invalid")
            vector(frame, "q18", 18); vector(frame, "dq18", 18)
            vector(frame, "root_angular_velocity_body_rad_s", 3); vector(frame, "script_targets6", 6)
            attitude(frame)
        require(all(p["case"] == name and 0 < p["sim_time_s"] <= until + 1e-9 for p in policies if p["lane"] == lane),
                "Policy frame escaped its first cohort")
        if "first_pure_rl_post_physics" in first:
            require(summary["actual_blend_physical_substeps"].get(str(lane)) == 40, "Pure RL lacks recorded forty-tick BLEND")
        by_case[name] = {"row": row, "events": case_events, "frames": case_frames,
                         "policies": [p for p in policies if p["lane"] == lane]}
    require(set(lanes) == set(range(4)) and {p: counts[p] for p in PARTITIONS} == summary["partitions"],
            "Fixed-N partitions do not conserve births")
    require(sum(len(c["frames"]) for c in by_case.values()) == len(frames)
            and sum(len(c["events"]) for c in by_case.values()) == len(events)
            and sum(len(c["policies"]) for c in by_case.values()) == len(policies), "Records outside the four first cohorts")
    require(summary["all_first_cohorts_closed"] == (counts["right_censored_unfinished"] == 0)
            and summary["terminal_pending_lanes"] == [], "Unresolved first-cohort accounting")
    for name, rate in summary["events_per_seeded_cohort"].items():
        n = sum(name in r["events"] for r in summary["rows"])
        require(rate["denominator"] == 4 and rate["numerator"] == n and rate["fraction"] == n/4,
                "Milestone rate uses a changed denominator")
    return {"directory": str(directory), "report": report, "contract": contract, "identity": identity,
            "provenance": provenance, "source_identity": source_receipt, "cases": by_case}


def prepare_metrics(run, case):
    frames = case["frames"]
    selected = [(previous, current) for previous, current in zip(frames, frames[1:])
                if current["phase_used"] == 7 and current["effective_control_ownership"] == "script"]
    eligible = sum(f["phase_used"] == 7 and f["effective_control_ownership"] == "script" for f in frames)
    require(len(selected) == eligible, "PREPARE has no preceding physical frame; cannot align pre inputs")
    if not selected:
        return {"physical_steps": 0, "covered_seconds": 0., "pre_physics": None, "post_physics": None,
                "native_gate_counts": None, "leg_target_slew_rad_s": None}
    pre, post = [p for p, _ in selected], [c for _, c in selected]
    joints = run["report"]["native_joint_names"]
    units = run["report"]["joint_position_units"]
    def body(rows):
        attitudes = [attitude(row) for row in rows]
        return {**{key: stats([a[key] for a in attitudes]) for key in attitudes[0]},
                "gyro_body_norm_rad_s": stats([math.sqrt(sum(x*x for x in r["root_angular_velocity_body_rad_s"])) for r in rows]),
                "gyro_body_rad_s": {axis: stats([r["root_angular_velocity_body_rad_s"][i] for r in rows])
                                    for i, axis in enumerate(("x", "y", "z"))},
                "named_dq18": {name: {"unit": "m/s" if units[i] == "m" else "rad/s",
                                      "statistics": stats([r["dq18"][i] for r in rows])} for i, name in enumerate(joints)}}
    patterns = Counter(tuple(g for g in GATES if not f["gates"][g]) for f in post)
    segments = []
    for f in post:
        if not segments or f["physical_tick"] != segments[-1]["last_physical_tick"] + 1:
            segments.append({"first_physical_tick": f["physical_tick"], "last_physical_tick": f["physical_tick"],
                             "first_post_physics_s": f["sim_time_s"], "last_post_physics_s": f["sim_time_s"], "physical_steps": 0})
        segments[-1].update(last_physical_tick=f["physical_tick"], last_post_physics_s=f["sim_time_s"])
        segments[-1]["physical_steps"] += 1
    return {"physical_steps": len(post), "covered_seconds": len(post) * DT, "segments": segments,
            "pre_physics": body(pre), "post_physics": body(post),
            "leg_target_slew_rad_s": {ACTIVE_ORDER[i]: stats([(c["script_targets6"][i] - p["script_targets6"][i])/DT
                                                            for p, c in selected]) for i in LEG_TARGET_IDS},
            "interior_prepare_leg_target_slew_rad_s": {
                ACTIVE_ORDER[i]: stats([(c["script_targets6"][i] - p["script_targets6"][i])/DT for p, c in selected
                                        if p["phase_used"] == 7 and p["effective_control_ownership"] == "script"])
                for i in LEG_TARGET_IDS},
            "phase_entry_leg_target_slew_rad_s": {
                ACTIVE_ORDER[i]: stats([(c["script_targets6"][i] - p["script_targets6"][i])/DT for p, c in selected
                                        if p["phase_used"] != 7 or p["effective_control_ownership"] != "script"])
                for i in LEG_TARGET_IDS},
            "ready_ticks_max": max(f["ready_ticks"] for f in post),
            "native_gate_counts": {g: {"passed": sum(f["gates"][g] for f in post), "denominator_physical_steps": len(post),
                                       "longest_consecutive_pass_steps": longest([f["gates"][g] for f in post])} for g in GATES},
            "native_all_eight_gates_pass_steps": sum(all(f["gates"].values()) for f in post),
            "native_all_eight_gates_longest_consecutive_pass_steps": longest([all(f["gates"].values()) for f in post]),
            "failed_gate_patterns": [{"failed_gates": list(k), "physical_steps": v} for k, v in sorted(patterns.items())]}


def case_summary(run, case):
    row, frames = case["row"], case["frames"]
    first_ready = next((f["sim_time_s"] for f in frames if f["ready_ticks"] > 0), None)
    first_policy_ready = next((p["sim_time_s"] for p in case["policies"]
                               if p["native_recovery"].get("recovery_ready") is True), None)
    return {"lane": row["lane"], "case": row["case"], "seed": row["seed"], "birth_id": row["birth_id"],
            "first_event_time_s": {e: row["events"].get(e) for e in EVENTS},
            "first_native_ready_pre_physics_time_s": None if first_ready is None else first_ready-DT,
            "first_native_ready_record_stamp_s": first_ready,
            "first_policy_sample_native_ready_s": first_policy_ready,
            "partition": row["partition"], "terminal": row["terminal"], "censor_time_s": row.get("censor_time_s"),
            "actual_blend_physical_substeps": run["report"]["summary"]["actual_blend_physical_substeps"].get(str(row["lane"]), 0),
            "blend_entry_ready_ticks": next((f["ready_ticks"] for f in frames if f["effective_control_ownership"] == "script_actor_blend"), None),
            "prepare": prepare_metrics(run, case)}


def prefix_equality(a, b):
    """Compare all recorded fields; never claim complete internal-state equality."""
    af, bf = a["frames"], b["frames"]
    ai = next((i for i, f in enumerate(af) if f["phase_used"] == 7 and f["effective_control_ownership"] == "script"), len(af))
    bi = next((i for i, f in enumerate(bf) if f["phase_used"] == 7 and f["effective_control_ownership"] == "script"), len(bf))
    def fields(frame):
        return {k: v for k, v in frame.items() if k != "lane"}
    first_difference = next(({"physical_tick": x["physical_tick"], "post_physics_s": x["sim_time_s"],
                              "fields": sorted(k for k in set(x) | set(y) if k != "lane" and x.get(k) != y.get(k))}
                             for x, y in zip(af, bf) if fields(x) != fields(y)), None)
    def phases(case):
        return [{"event": e["event"], "sim_time_s": e["sim_time_s"], "physical_tick": e["physical_tick"]}
                for e in case["events"] if e["event"].startswith("phase_")
                and e["sim_time_s"] <= case["row"]["events"].get("phase_PREPARE", math.inf)]
    return {"scope": "all physical-frame JSON fields except lane identity; recorded data only",
            "baseline_physical_steps_before_first_prepare": ai, "candidate_physical_steps_before_first_prepare": bi,
            "all_recorded_prefix_fields_exactly_equal": ai == bi and all(fields(x) == fields(y) for x, y in zip(af[:ai], bf[:bi])),
            "first_prepare_entry_frame_exactly_equal": None if ai == len(af) or bi == len(bf) else fields(af[ai]) == fields(bf[bi]),
            "first_prepare_entry_physical_tick": {"baseline": None if ai == len(af) else af[ai]["physical_tick"],
                                                  "candidate": None if bi == len(bf) else bf[bi]["physical_tick"]},
            "first_recorded_difference": first_difference, "phase_events_through_first_prepare_equal": phases(a) == phases(b),
            "baseline_phase_events_through_first_prepare": phases(a), "candidate_phase_events_through_first_prepare": phases(b)}


def compare(baseline_dir, candidate_dir):
    a, b = load_run(baseline_dir, "baseline"), load_run(candidate_dir, "prepare_slew_1_candidate")
    ar, br = a["report"], b["report"]
    for key in ("seed", "case_names", "stage", "source_worker_sha256", "native_joint_names", "joint_position_units"):
        # Cases are paired by actual names, never lane or declaration order.
        av, bv = (set(ar[key]), set(br[key])) if key == "case_names" else (ar[key], br[key])
        require(av == bv, f"A/B condition mismatch: {key}")
    for key in ("checkpoint_sha256", "checkpoint_contract_sha256", "checkpoint_learning_updates", "checkpoint_clocks"):
        require(ar["checkpoint"][key] == br["checkpoint"][key], f"A/B PT provenance mismatch: {key}")
    ai, bi = a["identity"], b["identity"]
    require(ai["harness_revision"] == bi["harness_revision"] and ai["base_core_files_sha256"] == bi["base_core_files_sha256"],
            "A/B does not share frozen H/base core")
    require(set(ai["core_files_sha256"]) == set(bi["core_files_sha256"]), "Core source set changed")
    require([k for k in ai["core_files_sha256"] if ai["core_files_sha256"][k] != bi["core_files_sha256"][k]] == [RECOVERY_FILE],
            "Only recovery.py core SHA may differ")
    require(set(ar["source_files_sha256"]) == set(br["source_files_sha256"])
            and [k for k in ar["source_files_sha256"] if ar["source_files_sha256"][k] != br["source_files_sha256"][k]] == [RECOVERY_FILE],
            "Runtime sources differ outside recovery.py")
    original = (Path(ai["source_root"]) / RECOVERY_FILE).read_bytes()
    changed = (Path(bi["source_root"]) / RECOVERY_FILE).read_bytes()
    require(original.count(PREPARE_CHANGE["old"].encode()) == 1
            and changed == original.replace(PREPARE_CHANGE["old"].encode(), PREPARE_CHANGE["new"].encode()),
            "Candidate bytes differ beyond approved PREPARE line")
    ac, bc = deepcopy(a["contract"]), deepcopy(b["contract"])
    ac["design_preflight"].pop("file"); bc["design_preflight"].pop("file")
    require(ac == bc, "Effective contracts differ beyond design_preflight.file absolute output")
    require(a["provenance"]["engineering_expectations.json"]["sha256"] == b["provenance"]["engineering_expectations.json"]["sha256"],
            "Independent engineering expectations differ")
    for key in ("runtime_preflight", "mass_kg_each", "dynamics_randomization", "contact_domain", "command_transport",
                "actuator_response", "recovery_training", "active_joint_order", "scene_group_counts"):
        require(ar["startup"][key] == br["startup"][key], f"Actual startup condition mismatch: {key}")
    cases = []
    for name in sorted(a["cases"]):
        x, y = case_summary(a, a["cases"][name]), case_summary(b, b["cases"][name])
        delta = {e: None if x["first_event_time_s"][e] is None or y["first_event_time_s"][e] is None
                 else y["first_event_time_s"][e] - x["first_event_time_s"][e] for e in EVENTS}
        cases.append({"case": name, "baseline": x, "candidate": y, "candidate_minus_baseline_first_event_seconds": delta,
                      "prefix_before_first_prepare": prefix_equality(a["cases"][name], b["cases"][name])})
    return {"schema": "recovery-physical-prefix-paired-comparison-v1", "status": "paired_inputs_verified",
            "candidate_disposition": "candidate_not_merged_or_deployed",
            "qualification": False, "model_selected": False, "gain_prediction": None,
            "formal_updates_consumed": 0, "physx_internal_state_restore_claim": False,
            "sampling": {"physics_dt_s": DT, "policy_dt_s": .02,
                "prepare_selection": "actual phase_used=7 and effective_control_ownership=script",
                "pre_physics": "previous contiguous same-lane post-physics root pose/body gyro/raw dq18; input time=current stamp-0.005",
                "post_physics": "current physical frame root pose/body gyro/raw dq18 after robot.update",
                "gate_ready_and_targets": "recorded native advance_reference before current physics; no reconstructed gates",
                "targetslew": "current minus previous recorded script leg targets divided by 0.005s; no configured gain inference",
                "prepare_gate_denominator": "script-owned phase_used=7 only; final ready/20th tick already marked BLEND is excluded and recorded separately",
                "window_comparison_limit": "PREPARE durations/outcomes differ; entire-phase means are not equal-time improvement evidence",
                "first_event_times": "native event sim_time_s retained; pre-effort/reference events and post-physics events have separate names",
                "gravity_pitch": "atan2(g_body.x,-g_body.z), not Euler pitch for inverted states",
                "not_available": "Full Markov/PhysX warmstart restoration; unobserved substep sensor/effort locals; qualification"},
            "baseline": {"directory": a["directory"], "source_identity": a["source_identity"], "files": a["provenance"],
                         "source_files_sha256": ar["source_files_sha256"], "core_files_sha256": ai["core_files_sha256"],
                         "fixed_birth_denominator": 4, "partitions": ar["summary"]["partitions"],
                         "elapsed_physics_s": ar["elapsed_physics_s"], "wall_seconds": ar["wall_seconds"]},
            "candidate": {"directory": b["directory"], "source_identity": b["source_identity"], "files": b["provenance"],
                          "source_files_sha256": br["source_files_sha256"], "core_files_sha256": bi["core_files_sha256"],
                          "fixed_birth_denominator": 4, "partitions": br["summary"]["partitions"],
                          "elapsed_physics_s": br["elapsed_physics_s"], "wall_seconds": br["wall_seconds"]},
            "cases": cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Comparison output already exists; do not overwrite")
    for source in (args.baseline, args.candidate):
        require(not args.output.resolve().is_relative_to(source.resolve()), "Keep output outside read-only source report directories")
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "qualification": False,
                      "baseline_partitions": result["baseline"]["partitions"], "candidate_partitions": result["candidate"]["partitions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
