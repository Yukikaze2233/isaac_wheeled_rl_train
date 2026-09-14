#!/usr/bin/env python3
"""Isolated B1 pilot: admission wait, owned-process resource stop/save, and two-hour outer bound."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from common import CODE_DIRECTORY, file_record
from pipeline_common import B_CONTRACT, verify_experiment, verify_transfer
from launch import own_descendants, cleanup_descendants
from start_v40_round2 import child_environment, run_child, verify_run
from wheeled_algo.v40_job import JobError, atomic_bytes, json_bytes, strict_json

GiB = 1024 ** 3


def resources(child=None):
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    result = {"time": datetime.now(timezone.utc).isoformat(),
              "mem_available_bytes": int(values["MemAvailable"].split()[0]) * 1024,
              "pilot_rss_bytes": None, "pilot_peak_rss_bytes": None, "pilot_gpu_mib": None,
              "gpu_free_mib": None, "gpu_used_mib": None, "gpu_total_mib": None}
    if child is not None and child.poll() is None:
        try:
            status = dict(line.split(":", 1) for line in Path(f"/proc/{child.pid}/status").read_text().splitlines())
            for source, key in (("VmRSS", "pilot_rss_bytes"), ("VmHWM", "pilot_peak_rss_bytes")):
                result[key] = int(status[source].split()[0]) * 1024
        except (OSError, KeyError):
            pass
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        query = subprocess.run([smi, "--id=0", "--query-gpu=memory.free,memory.used,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        if query.returncode == 0:
            free, used, total = [float(x.strip()) for x in query.stdout.strip().splitlines()[0].split(",")]
            if not all(math.isfinite(x) and x >= 0 for x in (free, used, total)):
                raise ValueError("invalid GPU memory counters")
            result.update(gpu_free_mib=free, gpu_used_mib=used, gpu_total_mib=total)
        if child is not None:
            query = subprocess.run([smi, "--id=0", "--query-compute-apps=pid,used_gpu_memory",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            for row in query.stdout.splitlines():
                pid, memory = [x.strip() for x in row.split(",", 1)]
                if pid == str(child.pid):
                    try:
                        result["pilot_gpu_mib"] = float(memory)
                    except ValueError:
                        pass
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        result["gpu_query_error"] = type(error).__name__
    result["pilot_gpu_attribution"] = "available" if result["pilot_gpu_mib"] is not None else "unavailable_or_WSL_PID_not_mapped"
    return result


def admission(sample):
    return sample["mem_available_bytes"] >= 6 * GiB and sample["gpu_free_mib"] is not None and sample["gpu_free_mib"] >= 2048


def pressure(sample):
    if sample["mem_available_bytes"] < 2 * GiB:
        return "host_memory_reserve"
    if sample["gpu_free_mib"] is not None and sample["gpu_free_mib"] < 1024:
        return "gpu_memory_reserve"
    return None


def request_pilot_stop(child, audit, reason, sample):
    """Signal only the still-owned Popen handle; never an A PID or process name."""
    child.send_signal(signal.SIGTERM)
    atomic_bytes(Path(audit) / "soft-stop.json", json_bytes({"pid": child.pid, "reason": reason, "sample": sample}))


def wait_for_headroom(telemetry, deadline, stop):
    until = min(time.monotonic() + 1800, deadline - 900)
    while True:
        sample = resources()
        telemetry.write(json.dumps({**sample, "phase": "admission"}) + "\n")
        telemetry.flush()
        if stop or time.monotonic() >= until:
            raise TimeoutError("startup resource admission expired or cancelled")
        if admission(sample):
            return
        time.sleep(10)


def build_plan(experiment, runtime):
    experiment, runtime = Path(experiment).absolute(), Path(runtime).absolute()
    manifest = verify_experiment(experiment)
    repo, root = experiment / CODE_DIRECTORY, experiment / "pilot"
    if root.exists():
        raise JobError("pilot already submitted; inspect it rather than retrying blindly")
    python = runtime.parent.parent / "env/bin/python"
    if not runtime.is_file() or not python.is_file():
        raise JobError("existing Sim runtime/interpreter missing")
    target = strict_json((repo / B_CONTRACT).read_bytes())
    identity = {"contract_id": target["contract_id"], "contract_sha256": manifest["target_contract_sha256"],
                "stage": "locomotion", "seed": 44}
    parent = manifest["parent"]
    plan = {"schema_version": 1, "classification": "experiment_not_promoted", "experiment": str(experiment),
            "repo": str(repo), "stage_root": str(root), "run_dir": str(root / "train"), "audit_dir": str(root / "audit"),
            "git_commit": manifest["git_commit"], "runtime": str(runtime), "runtime_sha256": file_record(runtime)["sha256"],
            "python": str(python), "identity": identity, "contract_sha256": identity["contract_sha256"],
            "source_checkpoint_sha256": parent["source_checkpoint_sha256"], "source_filename": parent["checkpoint"],
            "source_run": parent["source_run"], "updates": 500, "num_envs": 256,
            "session": "b1-pilot-" + uuid.uuid4().hex, "wall_seconds": 7200,
            "resource_policy": {"start_mem_available_GiB": 6, "stop_mem_available_GiB": 2,
                "start_gpu_free_MiB": 2048, "stop_gpu_free_MiB": 1024, "poll_seconds": 5,
                "source": "A RSS 4.14 GiB / observed peak 4.86 GiB; user requests >=2 GiB host reserve and >=6 GiB startup headroom; no cgroup isolation"}}
    plan["command"] = [str(python), str(repo / "scripts/train_v40.py"), "--contract", str(repo / B_CONTRACT),
        "--research", "--headless", "--stage", "locomotion", "--seed", "44", "--device", "cuda:0",
        "--num-envs", "256", "--max-iterations", "500", "--run-dir", plan["run_dir"],
        "--usd-cache-dir", str(root / "usd-cache"), "--stage-transfer", str(experiment / "parent" / parent["checkpoint"]),
        "--source-checkpoint-sha256", parent["source_checkpoint_sha256"]]
    return plan


def worker(path):
    plan = strict_json(Path(path).read_bytes())
    audit = Path(plan["audit_dir"])
    own_descendants()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append("worker_signal"))
    signal.signal(signal.SIGINT, lambda *_: stop.append("worker_signal"))
    deadline = time.monotonic() + 7080  # Cleanup remains inside the 7200s outer timeout.
    child = None
    result = {"status": "failed", "classification": "experiment_not_promoted", "source_checkpoint_sha256": plan["source_checkpoint_sha256"],
              "policy_quality_verified": False, "resource_stop": None}
    atomic_bytes(audit / "worker.started.json", json_bytes({"pid": os.getpid(), "time": datetime.now(timezone.utc).isoformat()}))
    lock = os.open(Path(plan["runtime"]).parent.parent / ".b1-pilot.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Never acquire or change A's lock.
        verify_experiment(plan["experiment"])
        if file_record(Path(plan["runtime"]))["sha256"] != plan["runtime_sha256"]:
            raise JobError("runtime changed")
        with (audit / "resources.jsonl").open("x") as telemetry:
            try:
                wait_for_headroom(telemetry, deadline, stop)
                run_child(plan, "preflight", [*plan["command"], "--preflight-only"], 120)
                preflight = strict_json((audit / "preflight.log").read_bytes())
                if preflight.get("ready") is not True or preflight.get("simulation_started") is not False:
                    raise JobError("B1 preflight rejected")
                verify_transfer(preflight, plan)
                wait_for_headroom(telemetry, deadline, stop)
            except TimeoutError:
                result["resource_stop"] = "startup_wait_expired_or_cancelled"
                raise
            remaining = deadline - time.monotonic()
            cutoff = datetime.now(timezone.utc) + timedelta(seconds=remaining - 600)
            command = [*plan["command"], "--max-runtime-seconds", str(int(min(5400, remaining - 900))),
                       "--stop-at", cutoff.isoformat()]
            with (audit / "train.log").open("xb") as log:
                child = subprocess.Popen(command, cwd=plan["repo"], env=child_environment(plan["python"]),
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic_bytes(audit / "train.started.json", json_bytes({"pid": child.pid, "command": command,
                    "started_at": datetime.now(timezone.utc).isoformat()}))
                sent = None
                while child.poll() is None:
                    sample = resources(child)
                    telemetry.write(json.dumps({**sample, "phase": "running"}) + "\n"); telemetry.flush()
                    reason = pressure(sample) or (stop[0] if stop else None)
                    if reason and sent is None:
                        request_pilot_stop(child, audit, reason, sample)
                        sent = time.monotonic()
                        result["resource_stop"] = reason
                    if time.monotonic() >= deadline or (sent is not None and time.monotonic() - sent >= 600):
                        raise TimeoutError("pilot hard/finalization deadline reached")
                    time.sleep(5)
            result["exit_code"] = child.returncode
            if child.returncode != 0:
                raise JobError(f"pilot exited {child.returncode}")
        result.update(verify_run(plan, {"name": "train", "requested_iterations": 500}))
        manifest = strict_json((Path(plan["run_dir"]) / "run_manifest.json").read_bytes())
        verify_transfer(manifest, plan)
        if manifest.get("round3_stage") != "B1" or manifest.get("num_envs") != 256 or not manifest.get("domain_randomization_report"):
            raise JobError("B1 startup material report missing")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=10)
        result["children_reaped"] = cleanup_descendants()
        if not result["children_reaped"]:
            result["status"] = "failed"
        os.close(lock)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_bytes(audit / "worker.status.json", json_bytes(result))
    print(json.dumps(result), flush=True)
    return 0 if result["status"] in ("completed", "stopped") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--runtime", default="/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.experiment is None:
        parser.error("experiment required")
    plan = build_plan(args.experiment, args.runtime)
    if not args.launch:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2)); return 0
    root = Path(plan["stage_root"]); root.mkdir()
    audit = root / "audit"; audit.mkdir()
    path = audit / "plan.json"; atomic_bytes(path, json_bytes(plan))
    shell = 'unset PYTHONHOME PYTHONPATH CUDA_VISIBLE_DEVICES; source "$1"; export LIVESTREAM=0 ENABLE_CAMERAS=0 PYTHONUNBUFFERED=1; exec "$2" "$3" --worker "$4"'
    command = ["tmux", "new-session", "-d", "-s", plan["session"], "-c", plan["repo"],
               "timeout", "--signal=TERM", "--kill-after=120s", "7200", "bash", "--noprofile", "--norc", "-c", shell,
               "b1-pilot", plan["runtime"], plan["python"], str(Path(plan["repo"]) / "scripts/round3/pipeline_pilot.py"), str(path)]
    reply = subprocess.run(command, capture_output=True, text=True, timeout=15)
    record = {"submitted": reply.returncode == 0, "plan": plan, "stderr": reply.stderr, "training_success": False}
    atomic_bytes(audit / "submission.json", json_bytes(record)); print(json.dumps(record))
    return 0 if reply.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
