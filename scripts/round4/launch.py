#!/usr/bin/env python3
"""One fresh 1024-env/30000-update Round4 job, bounded tmux worker; default dry-run."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import uuid

from r4_common import (
    CODE_DIRECTORY, CONTRACT, EXPERIMENTS, JobError, atomic_bytes, file_record,
    json_bytes, strict_json, system_snapshot, verify_scratch, verify_snapshot, write_json,
)
from round3.launch import cleanup_descendants, own_descendants
from start_v40_round2 import run_child, verify_run
from wheeled_tasks.v40.contract import contract_digest, load_contract
from wheeled_algo.v40_round4_launch import scratch_initialization


def build_plan(args):
    root = args.experiment.absolute()
    snapshot = verify_snapshot(root)
    if root.parent != EXPERIMENTS or root.name != "round4-full-" + snapshot["git_commit"]:
        raise JobError("Round4 requires its own commit-addressed experiment directory")
    stage = root / "formal"
    if os.path.lexists(stage):
        raise JobError("formal run already submitted; inspect it, never restart implicitly")
    repo = root / CODE_DIRECTORY
    runtime, ground = args.runtime.absolute(), args.ground_usd.absolute()
    python = runtime.parent.parent / "env/bin/python"
    if not python.is_file() or not ground.is_file() or not runtime.is_file():
        raise JobError("existing Sim interpreter/runtime and official cached ground USD are required")
    contract = load_contract(repo / CONTRACT)
    if not contract.get("round4") or args.seed != 44 or args.soft_hours not in (48, 72):
        raise JobError("full Round4 profile, nonnegative seed and finite 48h/72h budget required")
    identity = {"contract_id": contract["contract_id"], "contract_sha256": contract_digest(contract),
                "stage": "locomotion", "seed": args.seed}
    plan = {
        **{key: snapshot[key] for key in ("physics_model", "physics_asset_manifest_sha256", "repaired_dynamics_used",
                                          "explicit_user_authorization", "new_fifteen_body_model_role",
                                          "authorization_sha256", "initial_not_before")},
        "schema_version": 1, "profile": "round4_full", "initialization": "scratch", "parent": None,
        "git_commit": snapshot["git_commit"], "experiment": str(root), "repo": str(repo),
        "stage_root": str(stage), "audit_dir": str(stage / "audit"), "run_dir": str(stage / "train"),
        "identity": identity, "contract_sha256": identity["contract_sha256"],
        "requested_profile": contract["round4"], "num_envs": 1024, "updates": 30000,
        "expected_initialization": scratch_initialization(contract, args.seed),
        "runtime": str(runtime), "runtime_sha256": file_record(runtime)["sha256"], "python": str(python),
        "ground_usd": str(ground), "ground_record": file_record(ground),
        "learning_seconds": args.soft_hours * 3600, "export_seconds": 1800,
        "training_hard_seconds": args.soft_hours * 3600 + 1800,
        "worker_hard_seconds": (args.soft_hours + 1) * 3600,
        "session": "round4-full-" + uuid.uuid4().hex,
        "policy_quality_verified": False, "startup_physical_readback": "trainer_inline_required",
        "evaluation_entry": getattr(args, "evaluation_entry", None),
    }
    plan["command"] = [str(python), str(repo / "scripts/train_v40.py"), "--contract", str(repo / CONTRACT),
        "--research", "--headless", "--stage", "locomotion", "--seed", str(args.seed), "--device", "cuda:0",
        "--num-envs", "1024", "--max-iterations", "30000", "--ground-usd", str(ground),
        "--max-runtime-seconds", str(plan["learning_seconds"]), "--run-dir", plan["run_dir"],
        "--usd-cache-dir", str(stage / "usd-cache")]
    return plan


def run_worker(path):
    plan = strict_json(Path(path).read_bytes())
    audit = Path(plan["audit_dir"])
    with (audit / "worker.claim").open("x") as claim:
        claim.write(str(os.getpid()))
    lock = None
    monitor = None
    finished = threading.Event()
    result = {"status": "failed", "initialization": "scratch", "parent": None,
              **{key: plan[key] for key in ("physics_model", "physics_asset_manifest_sha256", "repaired_dynamics_used",
                                            "explicit_user_authorization")},
              "requested_updates": 30000, "formal_training_target_completed": False,
              "policy_quality_verified": False}

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"worker signal {signum}")

    def telemetry():
        with (audit / "resources.jsonl").open("x") as stream:
            while not finished.is_set():
                stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **system_snapshot()}) + "\n")
                stream.flush()
                finished.wait(30)

    try:
        own_descendants()
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        lock = os.open(Path(plan["runtime"]).parent.parent / ".round4-training.lock",
                       os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_bytes(audit / "worker.started.json", json_bytes({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))
        snapshot = verify_snapshot(plan["experiment"])
        atomic_bytes(audit / "physics-identity.json", json_bytes({
            key: snapshot[key] for key in ("physics_model", "physics_asset_manifest_sha256", "repaired_dynamics_used",
                                          "explicit_user_authorization", "authorization_sha256", "new_fifteen_body_model_role")}))
        if (file_record(Path(plan["runtime"]))["sha256"] != plan["runtime_sha256"]
                or file_record(Path(plan["ground_usd"])) != plan["ground_record"]):
            raise JobError("runtime or cached ground changed since planning")
        atomic_bytes(audit / "resources-start.json", json_bytes(system_snapshot(processes=True)))
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=plan["learning_seconds"])
        command = [*plan["command"], "--stop-at", cutoff.isoformat()]
        atomic_bytes(audit / "effective-command.json", json_bytes({"command": command, "stop_at": cutoff.isoformat(),
            "initialization": "scratch", "parent": None}))
        monitor = threading.Thread(target=telemetry, daemon=True)
        monitor.start()
        # This is the only training child: its normal inline preflight/physics readback precedes PPO.
        run_child(plan, "train", command, plan["training_hard_seconds"])
        checks = verify_run(plan, {"name": "train", "requested_iterations": 30000})
        manifest = strict_json((Path(plan["run_dir"]) / "run_manifest.json").read_bytes())
        verify_scratch(manifest, plan)
        if manifest["training_curriculum"]["completed_updates"] != checks["completed_updates"]:
            raise JobError("saved curriculum update clock differs from completion")
        atomic_bytes(audit / "manifest-evidence.json", json_bytes(manifest))
        result.update(checks, formal_training_target_completed=(checks["status"] == "completed" and checks["completed_updates"] == 30000))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        finished.set()
        if monitor is not None:
            monitor.join(timeout=15)
        try:
            result["children_reaped"] = cleanup_descendants()
        except Exception as error:
            result.update(children_reaped=False, cleanup_error=str(error))
        if not result["children_reaped"]:
            result.update(status="failed", formal_training_target_completed=False)
        if lock is not None:
            os.close(lock)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_bytes(audit / "worker.status.json", json_bytes(result))
    print(json.dumps(result), flush=True)
    if result["formal_training_target_completed"]:
        # Runs remotely after child cleanup; laptop shutdown cannot prevent this hook.
        command = [plan["python"], str(Path(plan["repo"]) / "scripts/round4/evaluation.py"), "--plan", str(path)]
        if plan.get("evaluation_entry"):
            command += ["--evaluator", plan["evaluation_entry"], "--launch"]
        try:
            reply = subprocess.run(command, capture_output=True, text=True, timeout=180)
            hook = {"exit_code": reply.returncode, "stdout": reply.stdout, "stderr": reply.stderr,
                    "policy_quality_verified": False}
        except (OSError, subprocess.TimeoutExpired) as error:
            hook = {"state": "evaluation_dispatch_failed", "reason": str(error), "policy_quality_verified": False}
        write_json(audit / "evaluation-hook.json", hook)
    return 0 if result["formal_training_target_completed"] else 2


def submit(plan):
    root = Path(plan["stage_root"])
    root.mkdir()
    audit = root / "audit"
    audit.mkdir()
    path = audit / "plan.json"
    atomic_bytes(path, json_bytes(plan))
    atomic_bytes(audit / "resources-submission.json", json_bytes(system_snapshot(processes=True)))
    shell = ('unset PYTHONHOME PYTHONPATH CUDA_VISIBLE_DEVICES; source "$1" || exit; '
             'export PYTHONUNBUFFERED=1 ENABLE_CAMERAS=0 LIVESTREAM=0; exec "$2" "$3" --worker "$4"')
    command = ["tmux", "new-session", "-d", "-s", plan["session"], "-c", plan["repo"],
               "timeout", "--signal=TERM", "--kill-after=120s", str(plan["worker_hard_seconds"]),
               "bash", "--noprofile", "--norc", "-c", shell, "round4-worker", plan["runtime"], plan["python"],
               str(Path(plan["repo"]) / "scripts/round4/launch.py"), str(path)]
    reply = subprocess.run(command, capture_output=True, text=True, timeout=15)
    result = {"submitted": reply.returncode == 0, "training_success": False, "plan": plan, "stderr": reply.stderr}
    atomic_bytes(audit / "submission.json", json_bytes(result))
    print(json.dumps(result))
    return 0 if result["submitted"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--ground-usd", type=Path)
    parser.add_argument("--runtime", type=Path, default=Path("/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh"))
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--soft-hours", type=int, choices=(48, 72), default=48)
    parser.add_argument("--evaluation-entry", help="committed evaluator to launch remotely after successful final export")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    if args.experiment is None or args.ground_usd is None:
        parser.error("--experiment and --ground-usd are required")
    plan = build_plan(args)
    if args.launch:
        return submit(plan)
    print(json.dumps({"dry_run": True, "training_started": False, "plan": plan}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
