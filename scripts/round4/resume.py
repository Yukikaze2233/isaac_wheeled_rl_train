#!/usr/bin/env python3
"""Resume one verified paused Round4 scratch run in a fresh, bounded remote run."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import uuid

from r4_common import (
    JobError, atomic_bytes, file_record, json_bytes, strict_json, system_snapshot,
    verify_scratch, verify_snapshot, verify_training_run, resume_parent_updates, write_json,
)
from round4.launch import build_plan as scratch_plan
from round3.launch import cleanup_descendants, own_descendants
from start_v40_round2 import run_child
from wheeled_algo.v40_job import artifact_record, validate_completion, verify_export_sidecar


def check_parent(checkpoint, expected_sha, target_initialization, identity):
    """Read-only weights-only parent validation, including nonempty Adam statistics."""
    from wheeled_algo.v40_export import _read_checkpoint, load_actor_checkpoint
    from wheeled_algo.v40_round4_launch import checked_resume_state
    checkpoint = Path(checkpoint)
    root = checkpoint.parent
    if checkpoint.name != "model_final.pt" or file_record(checkpoint)["sha256"] != expected_sha:
        raise JobError("resume parent checkpoint SHA/name mismatch")
    completion = validate_completion(strict_json((root / "completion.json").read_bytes()))
    if completion["status"] != "stopped" or completion["requested_iterations"] != 30000 or completion["export_status"] != "verified":
        raise JobError("resume requires a verified stopped scratch run")
    for item in completion["artifacts"]:
        if artifact_record(root, item["path"]) != item:
            raise JobError("parent artifact changed: " + item["path"])
    verify_export_sidecar(root)
    _, provenance = load_actor_checkpoint(checkpoint, root / "run_manifest.json")
    manifest_record = next(item for item in completion["artifacts"] if item["path"] == "run_manifest.json")
    if provenance["checkpoint_sha256"] != expected_sha or provenance["run_manifest_sha256"] != manifest_record["sha256"]:
        raise JobError("loaded parent bytes differ from the completion receipt")
    manifest = provenance["run_manifest"]
    verify_scratch(manifest, {"identity": identity, "expected_initialization": target_initialization, "command": []})
    progress = checked_resume_state(checkpoint, provenance, {"initialization": target_initialization})
    if progress["completed_updates"] != completion["completed_updates"] or progress["completed_updates"] != manifest["training_curriculum"]["completed_updates"]:
        raise JobError("parent completed-update clocks disagree")
    saved, digest = _read_checkpoint(checkpoint)
    optimizer = saved.get("optimizer_state_dict", {})
    if (digest != expected_sha or not optimizer.get("state") or not optimizer.get("param_groups")
            or saved["infos"].get("domain_randomization_report") != manifest.get("domain_randomization_report")):
        raise JobError("parent optimizer or actual material mapping evidence missing")
    # Original v40_job receipts have canonical JSON encoding; bind the entire receipt.
    receipt_sha = file_record(root / "completion.json")["sha256"]
    import hashlib
    if receipt_sha != hashlib.sha256(json_bytes(completion)).hexdigest():
        raise JobError("parent completion is not the original canonical job receipt")
    if any(artifact_record(root, item["path"]) != item for item in completion["artifacts"]):
        raise JobError("parent artifact changed during validation")
    return {"checkpoint": str(checkpoint.absolute()), "checkpoint_sha256": digest,
            "completion": completion, "completion_sha256": receipt_sha,
            "contract_sha256": manifest["contract_sha256"], "completed_updates": progress["completed_updates"],
            "run_manifest_sha256": provenance["run_manifest_sha256"], "saved_iter": saved["iter"],
            "optimizer_state_entries": len(optimizer["state"]), "optimizer_learning_rate": optimizer["param_groups"][0]["lr"],
            "material_report_sha256": hashlib.sha256(json_bytes(manifest["domain_randomization_report"])).hexdigest(),
            "resume_scope": "actor_critic_std_optimizer_and_curriculum; not bitwise environment/RNG trajectory"}


def build_plan(args):
    plan = scratch_plan(args)
    parent = check_parent(args.parent_checkpoint, args.parent_sha256, plan["expected_initialization"], plan["identity"])
    stage = Path(plan["experiment"]) / ("resume-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    plan.update(initialization="resume", parent=parent, updates=30000-parent["completed_updates"],
                stage_root=str(stage), audit_dir=str(stage / "audit"), run_dir=str(stage / "train"),
                session="round4-resume-" + uuid.uuid4().hex)
    argv = plan["command"]
    for flag, value in (("--max-iterations", str(plan["updates"])), ("--run-dir", plan["run_dir"]),
                        ("--usd-cache-dir", str(stage / "usd-cache"))):
        argv[argv.index(flag)+1] = value
    argv += ["--resume", parent["checkpoint"]]
    resume_parent_updates(plan)
    return plan


def run_worker(path):
    plan = strict_json(Path(path).read_bytes())
    audit = Path(plan["audit_dir"])
    atomic_bytes(audit / "worker.claim", str(os.getpid()).encode())
    lock = None
    result = {"status": "failed", "initialization": "resume", "parent_checkpoint_sha256": plan["parent"]["checkpoint_sha256"],
              "requested_updates": plan["updates"], "formal_training_target_completed": False, "policy_quality_verified": False}
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"resume worker signal {signum}")
    try:
        own_descendants()
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        lock = os.open(Path(plan["runtime"]).parent.parent / ".round4-training.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_bytes(audit / "worker.started.json", json_bytes({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))
        verify_snapshot(plan["experiment"])
        if (file_record(Path(plan["runtime"]))["sha256"] != plan["runtime_sha256"]
                or file_record(Path(plan["ground_usd"])) != plan["ground_record"]):
            raise JobError("runtime/ground changed since resume planning")
        verified = check_parent(plan["parent"]["checkpoint"], plan["parent"]["checkpoint_sha256"],
                                plan["expected_initialization"], plan["identity"])
        if verified != plan["parent"]:
            raise JobError("parent evidence changed since resume planning")
        atomic_bytes(audit / "parent-validation.json", json_bytes(verified))
        atomic_bytes(audit / "resources-start.json", json_bytes(system_snapshot(processes=True)))
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=plan["learning_seconds"])
        command = [*plan["command"], "--stop-at", cutoff.isoformat()]
        atomic_bytes(audit / "effective-command.json", json_bytes({"command": command, "stop_at": cutoff.isoformat(),
                     "initialization": "resume", "parent_checkpoint_sha256": verified["checkpoint_sha256"]}))
        # One formal training child only; normal inline startup/readback is the gate.
        run_child(plan, "train", command, plan["training_hard_seconds"])
        checks = verify_training_run(plan)
        run = Path(plan["run_dir"])
        atomic_bytes(audit / "cumulative-receipt.json", json_bytes({
            "schema_version": 1, **checks, "requested_invocation_updates": plan["updates"], "target_total_updates": 30000,
            "parent_checkpoint_sha256": verified["checkpoint_sha256"], "parent_completion_sha256": verified["completion_sha256"],
            "child_completion_sha256": file_record(run / "completion.json")["sha256"],
            "child_run_manifest_sha256": file_record(run / "run_manifest.json")["sha256"], "policy_quality_verified": False}))
        result.update(checks)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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
    if result["formal_training_target_completed"]:
        command = [plan["python"], str(Path(plan["repo"]) / "scripts/round4/evaluation.py"), "--plan", str(path),
                   "--evaluator", plan["evaluation_entry"], "--launch"]
        try:
            reply = subprocess.run(command, capture_output=True, text=True, timeout=180)
            hook = {"exit_code": reply.returncode, "stdout": reply.stdout, "stderr": reply.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            hook = {"state": "evaluation_dispatch_failed", "reason": str(error)}
    else:
        hook = {"state": "blocked", "reason": "cumulative target/export/cleanup not verified", "training_status": result}
    write_json(audit / "evaluation-hook.json", hook)
    print(json.dumps(result), flush=True)
    return 0 if result["formal_training_target_completed"] else 2


def submit(plan):
    root = Path(plan["stage_root"])
    root.mkdir()
    audit = root / "audit"
    audit.mkdir()
    path = audit / "plan.json"
    atomic_bytes(path, json_bytes(plan))
    shell = ('unset PYTHONHOME PYTHONPATH CUDA_VISIBLE_DEVICES; source "$1" || exit; '
             'export PYTHONUNBUFFERED=1 ENABLE_CAMERAS=0 LIVESTREAM=0; exec "$2" "$3" --worker "$4"')
    argv = ["tmux", "new-session", "-d", "-s", plan["session"], "-c", plan["repo"],
            "timeout", "--signal=TERM", "--kill-after=120s", str(plan["worker_hard_seconds"]),
            "bash", "--noprofile", "--norc", "-c", shell, "round4-resume", plan["runtime"], plan["python"],
            str(Path(plan["repo"]) / "scripts/round4/resume.py"), str(path)]
    reply = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    record = {"submitted": reply.returncode == 0, "training_success": False, "plan_path": str(path),
              "session": plan["session"], "run_dir": plan["run_dir"], "stderr": reply.stderr}
    atomic_bytes(audit / "submission.json", json_bytes(record))
    print(json.dumps(record), flush=True)
    return 0 if record["submitted"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-sha256")
    parser.add_argument("--ground-usd", type=Path)
    parser.add_argument("--runtime", type=Path, default=Path("/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh"))
    parser.add_argument("--seed", type=int, choices=(44,), default=44)
    parser.add_argument("--soft-hours", type=int, choices=(48,), default=48)
    parser.add_argument("--evaluation-entry", default="scripts/round4/evaluate_policy.py")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    if not all((args.experiment, args.parent_checkpoint, args.parent_sha256, args.ground_usd)):
        parser.error("experiment, parent checkpoint/SHA and official ground are required")
    plan = build_plan(args)
    if args.launch:
        return submit(plan)
    print(json.dumps({"dry_run": True, "training_started": False, "plan": plan}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
