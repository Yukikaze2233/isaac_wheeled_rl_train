#!/usr/bin/env python3
"""Wait for one frozen Round3 run, then evaluate its verified final ONNX; no training."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

FLAGS = ("nonfinite", "non_wheel_contact", "knee_limit", "instantaneous_tilt", "low_height",
         "base_visual_bounds_ground", "failure_gravity", "sustained_failure")
CRITERIA = {"height_mae_m_max": .005, "height_p95_m_max": .01, "planar_speed_mean_m_s_max": .02,
            "nonfinite_frames_max": 0, "failure_resets_max": 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("timestamp requires timezone")
    return date


def training_alive(plan):
    started = Path(plan["audit_dir"]) / "train.started.json"
    if not started.exists():
        return False
    record = json.loads(started.read_text())
    try:
        argv = (Path("/proc") / str(record["pid"]) / "cmdline").read_bytes().split(b"\0")
        return str(plan["run_dir"]).encode() in argv
    except FileNotFoundError:
        return False


def verified_final(plan):
    """None means pending. Every terminal failure is blocked, including partial exports."""
    from wheeled_algo.v40_job import validate_completion, strict_json
    from wheeled_tasks.v40.contract import contract_digest, load_contract
    from common import file_record, verify_warm_start

    run = Path(plan["run_dir"])
    path = run / "completion.json"
    if not path.exists():
        if (Path(plan["audit_dir"]) / "worker.status.json").exists():
            raise ValueError("worker ended without completion")
        return None
    completion = validate_completion(strict_json(path.read_bytes()))
    if (completion["status"] != "completed" or completion["export_status"] != "verified"
            or completion["completed_updates"] != plan["updates"]
            or completion["requested_iterations"] != plan["updates"]):
        raise ValueError("training is not a completed, fully verified final export: " + completion["status"])
    started = strict_json((Path(plan["audit_dir"]) / "train.started.json").read_bytes())
    if not utc(started["started_at"]) <= utc(completion["finished_at"]) <= datetime.now(timezone.utc):
        raise ValueError("stale or future completion timestamp")
    for item in completion["artifacts"]:
        if file_record(run / item["path"]) != {key: item[key] for key in ("size", "sha256")}:
            raise ValueError("final artifact hash/size mismatch: " + item["path"])
    manifest = strict_json((run / "run_manifest.json").read_bytes())
    if any(manifest.get(key) != value for key, value in plan["identity"].items()):
        raise ValueError("wrong final run metadata")
    verify_warm_start(manifest, plan["contract_sha256"])
    if contract_digest(load_contract(run / "contract.json")) != plan["contract_sha256"]:
        raise ValueError("wrong final contract")
    sidecar = strict_json((run / "policy.onnx.json").read_bytes())
    if (sidecar["onnx_sha256"] != sha(run / "policy.onnx") or sidecar["run_manifest"] != manifest
            or sidecar["run_manifest_sha256"] != sha(run / "run_manifest.json")):
        raise ValueError("ONNX sidecar identity mismatch")
    return completion


def p95(values):
    ordered = sorted(values)
    index = .95 * (len(ordered) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(csv_path, summary, height, steps):
    if (summary["status"] != "completed" or summary["stop_reason"] != "step_budget"
            or summary["policy_steps"] != steps or summary["diagnostic_flags_version"] != 1):
        raise ValueError("replay incomplete or missing real diagnostics")
    with Path(csv_path).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != steps:
        raise ValueError("CSV frame count mismatch")
    episodes = [[]]
    for index, row in enumerate(rows, 1):
        if (int(row["step"]) != index or int(row["policy_tick"]) != index
                or row["sample_kind"] != "pre_reset"):
            raise ValueError("CSV frame clock/boundary mismatch")
        for key in ("x", "y", "z", "vx", "vy", "episode_time_s", "non_wheel_net_force_max_n"):
            if not math.isfinite(float(row[key])):
                raise ValueError("nonfinite CSV state: " + key)
        for key in ("action_cmd_vx", "action_cmd_wz", "reward_cmd_vx", "reward_cmd_wz"):
            if float(row[key]) != 0:
                raise ValueError("nonzero evaluation velocity command")
        if any(abs(float(row[key]) - height) > 1e-7 for key in ("action_cmd_height", "reward_cmd_height")):
            raise ValueError("wrong evaluation height")
        if episodes[-1] and int(row["episode_step"]) <= int(episodes[-1][-1]["episode_step"]):
            previous = episodes[-1][-1]
            if previous["terminated"] != "True" and previous["timeout"] != "True":
                raise ValueError("unexplained CSV reset boundary")
            episodes.append([])
        episodes[-1].append(row)
    counts = {key: sum(row["diagnostic_" + key] == "True" for row in rows) for key in FLAGS}
    terminations = sum(row["terminated"] == "True" for row in rows)
    timeouts = sum(row["timeout"] == "True" for row in rows)
    if (counts != summary["diagnostic_frames"] or terminations != summary["termination_resets"]
            or timeouts != summary["timeout_resets"]):
        raise ValueError("CSV/summary diagnostic/reset counts differ")
    for key, count in summary["termination_flags"].items():
        if count != sum(row["terminated"] == "True" and row["termination_" + key] == "True" for row in rows):
            raise ValueError("CSV/summary termination reasons differ")

    def phase(selected):
        if not selected:
            return None
        errors = [abs(float(row["z"]) - height) for row in selected]
        speed = [math.hypot(float(row["vx"]), float(row["vy"])) for row in selected]
        forces = [float(row["non_wheel_net_force_max_n"]) for row in selected]
        return {"frames": len(selected), "height_mae_m": statistics.mean(errors), "height_p95_m": p95(errors),
                "planar_speed_mean_m_s": statistics.mean(speed), "planar_speed_p95_m_s": p95(speed),
                "non_wheel_net_force_max_n": max(forces),
                "non_wheel_net_contact_candidate_frames": sum(row["diagnostic_non_wheel_contact"] == "True" for row in selected)}

    initial = phase([row for row in rows if float(row["episode_time_s"]) <= 2])
    steady = phase([row for row in rows if float(row["episode_time_s"]) > 2])
    displacements = []
    for episode in episodes:
        dx = float(episode[-1]["x"]) - float(episode[0]["x"])
        dy = float(episode[-1]["y"]) - float(episode[0]["y"])
        displacements.append({"frames": len(episode), "sampled_xy_displacement_m": math.hypot(dx, dy),
                              "first_step": int(episode[0]["step"]), "last_step": int(episode[-1]["step"])})
    passed = (steady is not None and steady["height_mae_m"] <= .005 and steady["height_p95_m"] <= .01
              and steady["planar_speed_mean_m_s"] <= .02 and counts["nonfinite"] == 0 and terminations == 0)
    return {"height_m": height, "frames": steps, "cumulative_sim_seconds": steps * .01,
            "initialization": initial, "steady": steady, "diagnostic_frames": counts,
            "termination_resets": terminations, "timeout_resets": timeouts, "episodes": displacements,
            "research_criteria_pass": bool(passed) if steps == 6000 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=51 * 3600)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--engineering-smoke", action="store_true", help="one height, 2 steps; never standing acceptance")
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 51 * 3600 or args.poll_seconds < 1:
        parser.error("wait 1..51h and positive poll required")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.repo / "src"), str(args.repo / "scripts/round3")]
    from common import verify_snapshot
    plan = json.loads(args.training_plan.read_text())
    if (plan["repo"] != str(args.repo)
            or (args.engineering_smoke and plan["profile"] != "smoke")
            or (not args.engineering_smoke and (plan["profile"] != "train" or plan["updates"] != 10000))):
        parser.error("wrong frozen training plan/profile")
    args.output.mkdir(exist_ok=False)
    lock = (args.output / "evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + args.wait_seconds
    result = {"schema_version": 1, "evaluation_id": args.output.name, "training_plan_sha256": sha(args.training_plan),
              "run_dir": plan["run_dir"], "git_commit": plan["git_commit"], "seed": 44, "criteria": CRITERIA,
              "classification": "engineering_2_steps_not_policy_quality" if args.engineering_smoke else "final_policy_research_evaluation",
              "script_sha256": sha(__file__), "evaluator_pid": os.getpid(), "cases": [],
              "policy_quality_verified": False,
              "method": "each episode <=2s initialization, >2s steady; cumulative 60s includes 20s timeout resets; net force is not ground-pair/slip proof"}

    def state(value, **extra):
        result.update(state=value, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(args.output / "evaluation_state.json", result)
        print(json.dumps({"state": value, "pid": os.getpid(), **extra}), flush=True)

    try:
        while time.monotonic() < deadline:
            if training_alive(plan):
                state("waiting_training_process_exit")
            else:
                completion = verified_final(plan)
                if completion is not None:
                    break
                state("waiting_verified_final_completion")
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        else:
            state("expired_waiting_completion")
            completion = None
        if completion is not None:
            snapshot = verify_snapshot(Path(plan["snapshot"]))
            if snapshot["git_commit"] != plan["git_commit"]:
                raise ValueError("frozen source commit differs from plan")
            if sha(plan["runtime"]) != plan["runtime_sha256"]:
                raise ValueError("runtime changed")
            write_json(args.output / "training_completion.json", completion)
            run = Path(plan["run_dir"])
            result["policy_sha256"] = sha(run / "policy.onnx")
            result["contract_sha256"] = plan["contract_sha256"]
            result["final_artifacts_verified"] = True
            for height in ((.30,) if args.engineering_smoke else (.29, .30, .31, .32)):
                if time.monotonic() + 640 > deadline:
                    state("expired_before_evaluation_finished")
                    break
                case = args.output / f"h{round(height * 100):03d}"
                steps = 2 if args.engineering_smoke else 6000
                argv = ["timeout", "--signal=TERM", "--kill-after=20s", "620s", "bash", "-c",
                        'source "$1" && shift && exec python -u "$@"', "post-evaluation", plan["runtime"],
                        str(args.repo / "scripts/play_v40_onnx.py"), "--research", "--headless", "--seed", "44",
                        "--num-envs", "1", "--onnx", str(run / "policy.onnx"), "--contract", str(run / "contract.json"),
                        "--command", "0", "0", f"{height:.2f}", "--max-steps", str(steps),
                        "--max-wall-seconds", "580", "--report-dir", str(case)]
                state("evaluating", current_height_m=height)
                with (args.output / (case.name + ".log")).open("x") as log:
                    process = subprocess.run(argv, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
                record = {"height_m": height, "exit_code": process.returncode, "argv": argv}
                try:
                    summary = json.loads((case / "summary.json").read_text())
                    if (process.returncode or summary["onnx_sha256"] != result["policy_sha256"]
                            or summary["contract_sha256"] != plan["contract_sha256"]):
                        raise ValueError("replay failed or policy/contract identity differs")
                    record.update(summarize(case / "telemetry.csv", summary, height, steps), status="completed")
                except (OSError, ValueError, KeyError) as error:
                    record.update(status="failed", error=str(error))
                result["cases"].append(record)
            else:
                state("evaluation_completed" if all(c["status"] == "completed" for c in result["cases"]) else "evaluation_failed")
    except Exception as error:
        state("evaluation_failed" if result.get("final_artifacts_verified") else "blocked_no_verified_final_policy",
              reason=str(error))
    result["research_criteria_all_heights_pass"] = (
        all(c.get("research_criteria_pass") is True for c in result["cases"])
        if not args.engineering_smoke and result["state"] == "evaluation_completed" and len(result["cases"]) == 4 else None)
    write_json(args.output / "evaluation_result.json", result)
    lines = ["# Round3-A 最终策略自动评估", "", f"状态：`{result['state']}`；类型：`{result['classification']}`。",
             "", "累计60秒包含20秒timeout reset，不是连续60秒。SHA回传成功不等于站立通过。",
             "每次reset后前2秒单列；其后为统计窗口。净力候选不是ground-pair，也不能验证真实滑移。",
             "冻结replay的SERVER FINAL/cross_sim字段为旧标签，身份以实际policy SHA、路径和源/目标versions为准。",
             "", "| 高度m | 稳态MAE mm | P95 mm | 平移均速m/s | failure / timeout | 研究指标 |",
             "|---|---:|---:|---:|---|---|"]
    for case in result["cases"]:
        steady = case.get("steady")
        if steady:
            lines.append(f"| {case['height_m']:.2f} | {steady['height_mae_m']*1000:.3f} | {steady['height_p95_m']*1000:.3f} | {steady['planar_speed_mean_m_s']:.5f} | {case['termination_resets']} / {case['timeout_resets']} | {case['research_criteria_pass']} |")
        else:
            lines.append(f"| {case['height_m']:.2f} | — | — | — | — | {case.get('status')}：无有效稳态窗口或失败 |")
    lines.extend(["", "分阶段数值、逐回合XY漂移、真实diagnostics及终止计数见evaluation_result.json与各高度CSV/summary。",
                  "", "原因：" + result.get("reason", "见各case记录。")])
    (args.output / "evaluation_report.md").write_text("\n".join(lines) + "\n")
    files = [args.output / "evaluation_result.json", args.output / "evaluation_report.md"]
    files += [p for p in [args.output / "training_completion.json"] if p.exists()]
    files += [p for p in args.output.glob("h???/*") if p.name in ("summary.json", "telemetry.csv")]
    write_json(args.output / "report_manifest.json", {
        "schema_version": 1, "evaluation_id": result["evaluation_id"],
        "training_plan_sha256": result["training_plan_sha256"], "state": result["state"],
        "files": [{"path": str(p.relative_to(args.output)), "size": p.stat().st_size, "sha256": sha(p)} for p in files]})
    return 0 if result["state"] == "evaluation_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
