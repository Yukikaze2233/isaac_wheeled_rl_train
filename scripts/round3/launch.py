#!/usr/bin/env python3
"""Server-local Round3-A plans and bounded tmux workers; default is dry-run."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from common import CONTRACT, PARENT_SHA256, code_directory, file_record, verify_snapshot, verify_warm_start
from start_v40_round2 import run_child, verify_run
from wheeled_algo.v40_job import JobError, atomic_bytes, json_bytes, strict_json
from wheeled_tasks.v40.contract import contract_digest, load_contract

PROFILES = {
    "smoke": {"num_envs": 2, "updates": 3, "learning_seconds": 180, "initialization_seconds": 600, "finalization_seconds": 600},
    "capacity-256": {"num_envs": 256, "updates": 10, "learning_seconds": 900, "initialization_seconds": 600, "finalization_seconds": 600},
    "capacity-1024": {"num_envs": 1024, "updates": 10, "learning_seconds": 900, "initialization_seconds": 600, "finalization_seconds": 600},
    "train": {"num_envs": 1024, "updates": 10000, "learning_seconds": 172800, "initialization_seconds": 1800, "finalization_seconds": 1800},
}


def checked_gate(path, plan, profiles):
    if path is None:
        raise JobError(f"a verified {profiles} worker status is required")
    record = strict_json(Path(path).read_bytes())
    if record.get("profile") not in profiles or record.get("status") != "completed":
        raise JobError("engineering/capacity gate did not complete")
    for key in ("git_commit", "parent_model_sha256", "contract_sha256"):
        if record.get(key) != plan[key]:
            raise JobError(f"gate came from another source/parent/contract: {key}")
    if record.get("completed_updates") != record.get("requested_updates"):
        raise JobError("gate lacks all requested updates")
    if record.get("artifact_hashes_verified") is not True or record.get("export_verified") is not True:
        raise JobError("gate lacks verified artifacts/export")
    gate_plan = strict_json(Path(path).with_name("plan.json").read_bytes())
    if any(gate_plan.get(key) != record.get(key) for key in
           ("profile", "git_commit", "parent_model_sha256", "contract_sha256", "num_envs")):
        raise JobError("gate status differs from its recorded plan")
    # A saved 'passed' flag is insufficient if its files changed since the short run.
    verify_run(gate_plan, {"name": "train", "requested_iterations": gate_plan["updates"]})
    verify_warm_start(strict_json((Path(gate_plan["run_dir"]) / "run_manifest.json").read_bytes()), plan["contract_sha256"])
    return record


def build_plan(args):
    snapshot = Path(args.snapshot).absolute()
    manifest = verify_snapshot(snapshot)
    repo = code_directory(snapshot)
    spec = dict(PROFILES[args.profile])
    if args.num_envs is not None:
        if args.profile != "train" or args.num_envs not in (256, 1024):
            raise JobError("only formal training may select a benchmarked 256/1024 environment count")
        spec["num_envs"] = args.num_envs
    if args.updates is not None:
        if args.profile != "train" or not 100 <= args.updates <= 1000000:
            raise JobError("formal target must be 100..1000000 updates; short runs use engineering profiles")
        spec["updates"] = args.updates
    if args.seed < 0:
        raise JobError("seed must be nonnegative")
    runtime = Path(args.runtime).absolute()
    python = runtime.parent.parent / "env/bin/python"
    for path in (runtime, python):
        if not path.is_file():
            raise JobError(f"existing runtime/interpreter missing: {path}")
    root = Path(args.run_root).absolute()
    if not root.is_dir():
        raise JobError("run-root must be an existing directory")
    label = args.label or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in label):
        raise JobError("label must contain only letters, digits, '-' and '_'")
    stage_root = root / (args.profile + "-" + label)
    if os.path.lexists(stage_root):
        raise JobError("new run required; existing run is never overwritten or resumed implicitly")
    contract = load_contract(repo / CONTRACT)
    identity = {"contract_id": contract["contract_id"], "contract_sha256": contract_digest(contract),
                "stage": "locomotion", "seed": args.seed}
    plan = {"schema_version": 1, "profile": args.profile, **spec, **identity,
            "identity": identity, "snapshot": str(snapshot), "git_commit": manifest["git_commit"],
            "parent_model_sha256": PARENT_SHA256, "runtime": str(runtime),
            "runtime_sha256": file_record(runtime)["sha256"], "python": str(python),
            "repo": str(repo), "stage_root": str(stage_root),
            "run_dir": str(stage_root / "train"), "audit_dir": str(stage_root / "audit"),
            "session": "r3a-" + args.profile + "-" + uuid.uuid4().hex,
            "policy_quality_verified": False,
            "purpose": "formal_round3_a" if args.profile == "train" else "engineering_not_policy_quality"}
    if args.profile != "smoke":
        plan["smoke_gate"] = checked_gate(args.smoke_status, plan, ("smoke",))
    if args.profile == "train":
        capacity = checked_gate(args.capacity_status, plan, (f"capacity-{spec['num_envs']}",))
        if capacity.get("num_envs") != spec["num_envs"]:
            raise JobError("formal env count differs from benchmark")
        plan["capacity_gate"] = capacity
    plan["command"] = [str(python), str(repo / "scripts/train_v40.py"),
        "--contract", str(repo / CONTRACT), "--research", "--headless",
        "--stage", identity["stage"], "--seed", str(args.seed), "--device", "cuda:0",
        "--num-envs", str(spec["num_envs"]), "--max-iterations", str(spec["updates"]),
        "--max-runtime-seconds", str(spec["learning_seconds"]),
        "--usd-cache-dir", str(stage_root / "usd-cache"), "--run-dir", plan["run_dir"],
        "--warm-start", str(snapshot / "parent/model_final.pt")]
    plan["training_hard_seconds"] = sum(spec[k] for k in ("initialization_seconds", "learning_seconds", "finalization_seconds"))
    plan["worker_hard_seconds"] = plan["training_hard_seconds"] + 120 + 300
    return plan


def own_descendants():
    # Subreaping retains ownership of exporter descendants even if their trainer dies.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enable Linux child subreaper")


def cleanup_descendants():
    import psutil  # Already part of the deployed training runtime; worker-only import.
    children = psutil.Process().children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(children, timeout=20)
    for child in remaining:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(remaining, timeout=10)
    return not remaining


def run_worker(path):
    plan = strict_json(Path(path).read_bytes())
    audit = Path(plan["audit_dir"])
    atomic_bytes(audit / "worker.started.json", json_bytes({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))
    result = {key: plan[key] for key in ("profile", "git_commit", "parent_model_sha256", "contract_sha256", "num_envs", "purpose")}
    result.update(status="failed", requested_updates=plan["updates"], policy_quality_verified=False)
    lock = None
    try:
        own_descendants()
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f"worker signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        lock_path = Path(plan["runtime"]).parent.parent / ".round3-training.lock"
        lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if file_record(Path(plan["runtime"]))["sha256"] != plan["runtime_sha256"]:
            raise JobError("runtime shell script changed after planning")
        verify_snapshot(Path(plan["snapshot"]))
        run_child(plan, "preflight", [*plan["command"], "--preflight-only"], 120)
        preflight = strict_json((audit / "preflight.log").read_bytes())
        if preflight.get("ready") is not True or preflight.get("simulation_started") is not False:
            raise JobError("Round3 preflight rejected")
        if any(preflight.get(key) != plan["identity"][key] for key in ("contract_id", "contract_sha256", "stage")):
            raise JobError("preflight contract/stage identity mismatch")
        verify_warm_start(preflight, plan["contract_sha256"])
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=plan["initialization_seconds"] + plan["learning_seconds"])
        command = [*plan["command"], "--stop-at", cutoff.isoformat()]
        atomic_bytes(audit / "effective-command.json", json_bytes({"command": command, "stop_at": cutoff.isoformat()}))
        run_child(plan, "train", command, plan["training_hard_seconds"])
        checks = verify_run(plan, {"name": "train", "requested_iterations": plan["updates"]})
        verify_warm_start(strict_json((Path(plan["run_dir"]) / "run_manifest.json").read_bytes()), plan["contract_sha256"])
        result.update(checks)
        if plan["profile"] != "train" and checks["status"] != "completed":
            raise JobError("engineering gate must complete every requested update")
        completion = strict_json((Path(plan["run_dir"]) / "completion.json").read_bytes())
        config = strict_json((Path(plan["run_dir"]) / "agent_config.json").read_bytes())
        elapsed = completion["learning_elapsed_seconds"]
        steps = config.get("num_steps_per_env")
        result.update(learning_elapsed_seconds=elapsed, rollout_steps_per_env=steps,
                      aggregate_env_steps_per_second=(checks["completed_updates"] * steps * plan["num_envs"] / elapsed
                                                     if steps and elapsed else None))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            result["children_reaped"] = cleanup_descendants()
        except Exception as error:
            result.update(status="failed", children_reaped=False, cleanup_error=str(error))
        if not result["children_reaped"]:
            result["status"] = "failed"
        if lock is not None:
            os.close(lock)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_bytes(audit / "worker.status.json", json_bytes(result))
    print(json.dumps(result), flush=True)
    return 0 if result["status"] in ("completed", "stopped") else 2


def submit(plan):
    root = Path(plan["stage_root"])
    root.mkdir(mode=0o700)
    audit = root / "audit"
    audit.mkdir(mode=0o700)
    path = audit / "plan.json"
    atomic_bytes(path, json_bytes(plan))
    shell = ('unset PYTHONHOME PYTHONPATH CUDA_VISIBLE_DEVICES; source "$1"; '
             'export PYTHONUNBUFFERED=1 ENABLE_CAMERAS=0 LIVESTREAM=0; exec "$2" "$3" --worker "$4"')
    command = ["/usr/bin/tmux", "new-session", "-d", "-s", plan["session"], "-c", plan["repo"],
               "/usr/bin/timeout", "--signal=TERM", "--kill-after=120s", str(plan["worker_hard_seconds"]),
               "/bin/bash", "--noprofile", "--norc", "-c", shell, "round3-worker", plan["runtime"], plan["python"],
               str(Path(plan["repo"]) / "scripts/round3/launch.py"), str(path)]
    reply = subprocess.run(command, capture_output=True, text=True, timeout=15)
    record = {"submitted": reply.returncode == 0, "session": plan["session"], "run_dir": plan["run_dir"],
              "audit_dir": str(audit), "preflight_pending": True, "training_success": False, "stderr": reply.stderr}
    atomic_bytes(audit / "submission.json", json_bytes(record))
    print(json.dumps(record, indent=2))
    return 0 if reply.returncode == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot")
    parser.add_argument("--run-root")
    parser.add_argument("--runtime", default="/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh")
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--label")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--smoke-status", type=Path)
    parser.add_argument("--capacity-status", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    if not args.snapshot or not args.run_root:
        parser.error("snapshot and existing run-root are required")
    plan = build_plan(args)
    if args.launch:
        return submit(plan)
    print(json.dumps({"dry_run": True, "training_started": False, "plan": plan}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (JobError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Round3 launch not confirmed: {error}; inspect audit before retrying", file=sys.stderr)
        raise SystemExit(2)
