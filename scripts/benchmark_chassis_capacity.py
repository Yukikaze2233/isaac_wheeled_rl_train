#!/usr/bin/env python3
"""Sequential real-PPO capacity probes with memory guards and measured throughput."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract


def resources():
    memory = dict((row[0], int(row[1])) for row in
                  (line.split() for line in Path("/proc/meminfo").read_text().splitlines()) if len(row) > 1)
    executable = "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else "nvidia-smi"
    result = subprocess.run([executable, "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    used, free, utilization = map(float, result.stdout.splitlines()[0].split(","))
    return {"available_ram_kib": memory["MemAvailable:"], "gpu_used_mib": used,
            "gpu_free_mib": free, "gpu_utilization_percent": utilization}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--seconds-per-probe", type=float, default=600.)
    args = parser.parse_args()
    if not args.envs or any(n < 32 or n > 4096 for n in args.envs) or args.updates < 8:
        parser.error("Require 32-4096 environments and at least eight measured PPO updates")
    loader = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(json.loads(args.plan.read_text()), loader)
    base = loader(plan["base_contract"])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "engineering_capacity_not_skill_acceptance", "started_at": datetime.now(timezone.utc).isoformat(),
              "probes": [], "recommended_num_envs": None}
    for count in args.envs:
        before = resources()
        if before["available_ram_kib"] < 4 * 1024 ** 2:
            report["stop_reason"] = "insufficient_available_ram_before_next_probe"
            break
        config = stage_contract(base, plan, plan["stages"][0], count)
        path = args.output / f"contract_{count}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        run = args.output / f"envs_{count}"
        command = [sys.executable, "-B", str(ROOT / "scripts/train_chassis.py"), "--contract", str(path.resolve()),
                   "--stage", config["enabled_stages"][0], "--num-envs", str(count), "--updates", str(args.updates),
                   "--research", "--max-runtime-seconds", str(args.seconds_per_probe), "--run-dir", str(run.resolve())]
        samples, reason = [], None
        started = time.monotonic()
        with (args.output / f"envs_{count}.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                sample = resources()
                samples.append(sample)
                if sample["available_ram_kib"] < 2 * 1024 ** 2 or sample["gpu_free_mib"] < 1536:
                    reason = "memory_guard"
                elif time.monotonic() - started > args.seconds_per_probe + 120:
                    reason = "probe_timeout"
                if reason:
                    child.send_signal(signal.SIGTERM)
                    try:
                        child.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                time.sleep(1.)
        completion_path = run / "completion.json"
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
        row = {"num_envs": count, "exit_code": child.returncode, "wall_seconds": time.monotonic() - started,
               "status": completion.get("status", reason or "no_completion"), "resource_stop": reason,
               "successful_updates": completion.get("successful_updates", 0),
               "export_verified": completion.get("export", {}).get("verified", False), "resource_samples": samples}
        if samples:
            row.update(peak_gpu_mib=max(s["gpu_used_mib"] for s in samples),
                       min_available_ram_kib=min(s["available_ram_kib"] for s in samples))
        if completion.get("status") == "completed":
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
            for tag, name in (("Perf/collection_time", "collection_seconds"), ("Perf/learning_time", "learning_seconds"),
                              ("Perf/total_fps", "transitions_per_second")):
                values = [s.value for s in events.Scalars(tag)[3:]]
                row[name] = statistics.median(values)
            row["update_seconds"] = row["collection_seconds"] + row["learning_seconds"]
            row["simulated_seconds_per_wall_second"] = row["transitions_per_second"] * config["policy_dt"]
            row["max_closure_gap_m"] = completion["metrics"]["max_closure_gap_m"]
            row["termination_counts"] = completion["metrics"]["termination_counts"]
        report["probes"].append(row)
        good = [r for r in report["probes"] if r["status"] == "completed" and not r["resource_stop"]]
        if good:
            report["recommended_num_envs"] = max(good, key=lambda r: r["transitions_per_second"])["num_envs"]
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("CAPACITY_PROBE", json.dumps({k: v for k, v in row.items() if k != "resource_samples"}), flush=True)
        if reason or child.returncode or row.get("min_available_ram_kib", 0) < 3 * 1024 ** 2:
            report["stop_reason"] = reason or "insufficient_margin_or_failed_probe"
            break
        time.sleep(3.)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["recommended_num_envs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
