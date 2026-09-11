#!/usr/bin/env python3
"""Server-local round-two tmux workers. Default: read-only plan, no subprocesses."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

# Even a dry-run against a clean checkout must not create __pycache__ directories.
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wheeled_algo.v40_job import (
    JobError, artifact_record, atomic_bytes, json_bytes, positive_seconds, strict_json,
    validate_completion, verify_export_sidecar,
)
from wheeled_tasks.v40.contract import contract_digest, is_round2, load_contract

STAGES = ("stand", "locomotion")
PREFLIGHT_SECONDS, PILOT_SECONDS = 120, 1800
INITIALIZATION_SECONDS, EXPORT_SECONDS, TERM_SECONDS = 1800, 1200, 20
ENV_KEYS = ("PYTHONHOME", "PYTHONPATH", "PATH", "ENABLE_CAMERAS", "LIVESTREAM",
            "PYTHONUNBUFFERED", "OMNI_KIT_ACCEPT_EULA")


def child_environment(python):
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env.update(PATH=f"{Path(python).parent}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
               ENABLE_CAMERAS="0", LIVESTREAM="0", PYTHONUNBUFFERED="1")
    return env


def build_plan(args):
    for key in ("run_root", "python", "repo", "tmux"):
        value = getattr(args, key)
        if not value or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise JobError(f"{key} must be an absolute path without '..'")
    python = Path(args.python)  # Never resolve a venv interpreter symlink.
    if python.parent.name != "bin" or python.name not in {"python", "python3", "python3.11"}:
        raise JobError("python must be explicit ENV/bin/python (or python3/python3.11)")
    if (args.num_envs < 1 or not 0 < args.pilot_iterations < args.total_iterations
            or args.seed_base < 0 or len(set(args.stages)) != len(args.stages)):
        raise JobError("positive counts, total > pilot, nonnegative seed and unique stages required")
    root, repo = Path(args.run_root), Path(args.repo)
    if os.path.lexists(root):
        raise JobError("refusing to overwrite run-root")
    contract_path = repo / "contracts/own_v40_v2.json"
    contract = load_contract(contract_path)
    if not is_round2(contract):
        raise JobError("round two requires the explicit v2 contract")
    workers = []
    for stage in args.stages:
        stage_root = root / stage
        seed = args.seed_base + STAGES.index(stage)
        common = [str(python), str(repo / "scripts/train_v40.py"), "--contract", str(contract_path),
                  "--research", "--headless", "--stage", stage, "--seed", str(seed),
                  "--num-envs", str(args.num_envs), "--usd-cache-dir", str(stage_root / "usd_cache")]
        phases = [{"name": "preflight", "command": [*common, "--preflight-only"],
                   "runtime_seconds": None, "timeout_seconds": PREFLIGHT_SECONDS}]
        for name, count, runtime in (("pilot", args.pilot_iterations, PILOT_SECONDS),
                                     ("train", args.total_iterations - args.pilot_iterations,
                                      positive_seconds(args.training_runtime_seconds))):
            command = [*common, "--run-dir", str(stage_root / name), "--max-iterations", str(count),
                       "--max-runtime-seconds", str(runtime)]
            if name == "train":
                command += ["--resume", str(stage_root / "pilot/model_final.pt")]
            phases.append({"name": name, "command": command, "requested_iterations": count,
                           "runtime_seconds": runtime,
                           "timeout_seconds": INITIALIZATION_SECONDS + runtime + EXPORT_SECONDS})
        workers.append({"repo": str(repo), "python": str(python), "stage_root": str(stage_root),
                        "session": f"v40-r2-{stage}-{uuid.uuid4().hex}", "phases": phases,
                        "identity": {"contract_id": contract["contract_id"],
                                     "contract_sha256": contract_digest(contract), "stage": stage, "seed": seed},
                        "checks": ["preflight_ready_v2", "zero_exit", "completion_schema",
                                   "own_stage_seed_contract", "all_artifact_hashes", "export_sidecar",
                                   "pilot_all_requested_updates_before_resume"]})
    return {"schema_version": 1, "run_root": str(root), "tmux": str(args.tmux), "workers": workers,
            "cutoff_policy": "phase start + initialization + learning; outer timeout adds export budget",
            "initialization_seconds": INITIALIZATION_SECONDS, "export_seconds": EXPORT_SECONDS,
            "term_grace_seconds": TERM_SECONDS}


def run_child(worker, name, command, timeout):
    """Own one new process group; no process-name searches or tmux/server shutdown."""
    audit = Path(worker["stage_root"]) / "audit"
    status = {"command": command, "timeout_seconds": timeout, "started_at": datetime.now(timezone.utc).isoformat(),
              "started_monotonic": time.monotonic(), "pid": None, "exit_code": None, "timed_out": False}
    print(f"[{worker['identity']['stage']}] {name}: starting; log={audit / (name + '.log')}", flush=True)
    with (audit / f"{name}.log").open("xb") as log:
        child = None
        try:
            child = subprocess.Popen(command, cwd=worker["repo"], env=child_environment(worker["python"]),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status["pid"] = child.pid
            atomic_bytes(audit / f"{name}.started.json", json_bytes(status))
            status["exit_code"] = child.wait(timeout=timeout)
        except BaseException as exc:
            status.update(timed_out=isinstance(exc, subprocess.TimeoutExpired), error=f"{type(exc).__name__}: {exc}")
            if child is None or child.returncode is not None:
                raise
            # The Popen handle has not reaped a timed-out child, preserving ownership.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=TERM_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=TERM_SECONDS)
            status["exit_code"] = child.returncode
            raise
        finally:
            status["elapsed_seconds"] = time.monotonic() - status["started_monotonic"]
            atomic_bytes(audit / f"{name}.status.json", json_bytes(status))
    if status["exit_code"] != 0:
        raise JobError(f"{name} exited {status['exit_code']}; inspect {audit / (name + '.log')}")
    return status


def verify_run(worker, phase):
    run = Path(worker["stage_root"]) / phase["name"]
    if run.is_symlink():
        raise JobError("run directory must belong to this worker")
    receipt = validate_completion(strict_json((run / "completion.json").read_bytes()))
    if (receipt["requested_iterations"] != phase["requested_iterations"]
            or receipt["status"] not in ({"completed"} if phase["name"] == "pilot" else {"completed", "stopped"})
            or receipt["export_status"] != "verified"):
        raise JobError("receipt does not authorize this phase/resume")
    for item in receipt["artifacts"]:
        if artifact_record(run, item["path"]) != item:
            raise JobError(f"artifact hash mismatch: {item['path']}")
    manifest = strict_json((run / "run_manifest.json").read_bytes())
    if any(manifest.get(key) != value for key, value in worker["identity"].items()):
        raise JobError("run identity differs from this worker's v2 stage/seed/contract")
    verify_export_sidecar(run)
    return {"status": receipt["status"], "completed_updates": receipt["completed_updates"],
            "artifact_hashes_verified": True, "export_verified": True}


def run_worker(plan_path):
    worker = strict_json(Path(plan_path).read_bytes())
    audit = Path(worker["stage_root"]) / "audit"
    # A second invocation fails before it can start even the preflight child.
    atomic_bytes(audit / "worker.started.json", json_bytes({"pid": os.getpid(),
                 "started_at": datetime.now(timezone.utc).isoformat()}))
    status = {"status": "failed", "checks": {}}
    try:
        for phase in worker["phases"]:
            name = phase["name"]
            if name != "preflight" and os.path.lexists(Path(worker["stage_root"]) / name):
                raise JobError(f"refusing to overwrite {name}")
            command = list(phase["command"])
            if phase["runtime_seconds"] is not None:
                cutoff = datetime.now(timezone.utc) + timedelta(seconds=INITIALIZATION_SECONDS + phase["runtime_seconds"])
                command += ["--stop-at", cutoff.isoformat()]
            run_child(worker, name, command, phase["timeout_seconds"])
            if name == "preflight":
                report = strict_json((audit / "preflight.log").read_bytes())
                if (report.get("ready") is not True or report.get("simulation_started") is not False
                        or any(report.get(key) != worker["identity"][key]
                               for key in ("contract_id", "contract_sha256", "stage"))):
                    raise JobError("preflight rejected or wrong v2 identity")
                status["checks"][name] = {"ready": True}
            else:
                status["checks"][name] = verify_run(worker, phase)
            atomic_bytes(audit / f"{name}.checks.json", json_bytes(status["checks"][name]))
            print(f"[{worker['identity']['stage']}] {name}: verified {status['checks'][name]}", flush=True)
        status["status"] = status["checks"]["train"]["status"]
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        atomic_bytes(audit / "worker.status.json", json_bytes(status))
        print(json.dumps(status), flush=True)
    return 0 if status["status"] in {"completed", "stopped"} else 2


def launch(plan):
    root = Path(plan["run_root"])
    root.mkdir(mode=0o700)  # Existing parent required; exclusive even after dry-run.
    atomic_bytes(root / "plan.json", json_bytes(plan))
    success = True
    for worker in plan["workers"]:
        audit = Path(worker["stage_root"]) / "audit"
        audit.mkdir(parents=True, mode=0o700)
        plan_path = audit / "plan.json"
        atomic_bytes(plan_path, json_bytes(worker))
        env = child_environment(worker["python"])
        env_args = [arg for key in ENV_KEYS for arg in ("-e", f"{key}={env.get(key, '')}")]
        # Multiple command argv entries make tmux exec directly, without a shell.
        command = [plan["tmux"], "new-session", "-d", "-s", worker["session"], "-c", worker["repo"],
                   *env_args, worker["python"], str(Path(worker["repo"]) / "scripts/start_v40_round2.py"),
                   "--worker", str(plan_path)]
        try:
            run_child(worker, "tmux", command, 15)
            print(f"Worker submitted: tmux attach -t {worker['session']}", flush=True)
        except (OSError, JobError, subprocess.SubprocessError) as exc:
            success = False
            print(f"Worker submission failed: {exc}; inspect {audit}", file=sys.stderr)
    return 0 if success else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", help="Absolute NEW directory with existing parent")
    parser.add_argument("--python", help="Absolute server ENV/bin/python")
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--tmux", default="/usr/bin/tmux")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=["locomotion"],
                        help="Unified locomotion policy by default; stand is an optional diagnostic")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--total-iterations", type=int, default=20000)
    parser.add_argument("--pilot-iterations", type=int, default=20)
    parser.add_argument("--training-runtime-seconds", type=positive_seconds, default=86400)
    parser.add_argument("--seed-base", type=int, default=41)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            return run_worker(args.worker)
        plan = build_plan(args)
        if args.launch:
            return launch(plan)
        print(json.dumps({"dry_run": True, "launched": False, "plan": plan}, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Round-two launch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
