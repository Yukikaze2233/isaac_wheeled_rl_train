#!/usr/bin/env python3
"""Submit B1 explicitly, create its LOCAL watcher, and return only verified experiment artifacts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from common import CODE_DIRECTORY, ControlSSH, add_ssh_arguments, file_record
from evaluate_finished import write_json
from pipeline_common import verify_transfer
from pull_v40_artifacts import NotReady, Unavailable, probe, pull_artifacts
from start_v40_round2 import verify_run
from watch import ensure_root, read_worker


def verify_local(directory, plan):
    # Reuse final artifact/export validation while mapping only this local copy's paths.
    if directory.name != "train":
        raise ValueError("local verified artifact directory must be named train")
    from wheeled_algo.v40_job import validate_completion, verify_export_sidecar
    completion = validate_completion(json.loads((directory / "completion.json").read_text()))
    if completion["requested_iterations"] != 500:
        raise ValueError("not this pilot completion")
    for item in completion["artifacts"]:
        if file_record(directory / item["path"]) != {key: item[key] for key in ("size", "sha256")}:
            raise ValueError("local pilot artifact changed")
    if completion["export_status"] == "verified":
        verify_export_sidecar(directory)
    verify_transfer(json.loads((directory / "run_manifest.json").read_text()), plan)
    return completion


def verify_a_return(directory, handoff):
    if not (directory / "local_receipt.json").is_file():
        return False
    from wheeled_algo.v40_job import validate_completion
    receipt = json.loads((directory / "local_receipt.json").read_text())
    completion = validate_completion(json.loads((directory / "completion.json").read_text()))
    if receipt.get("transfer_verified") is not True or receipt["completion_sha256"] != file_record(directory / "completion.json")["sha256"]:
        raise ValueError("A local transfer receipt mismatch")
    for item in completion["artifacts"]:
        if file_record(directory / item["path"]) != {k: item[k] for k in ("size", "sha256")}:
            raise ValueError("A returned artifact changed")
    if file_record(directory / "model_final.pt")["sha256"] != handoff["a_final_checkpoint_sha256"]:
        raise ValueError("A returned checkpoint differs from evaluated final")
    if file_record(directory / "policy.onnx")["sha256"] != handoff["a_evaluation"]["policy_sha256"]:
        raise ValueError("A returned ONNX differs from evaluation")
    return True


def watch(args):
    plan = json.loads(args.plan.read_text())
    if plan.get("classification") != "experiment_not_promoted" or plan.get("updates") != 500 or plan.get("num_envs") != 256:
        raise ValueError("watcher requires this B1 256-env/500-update pilot plan")
    identity = {"plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(), "host": args.host,
                "remote_run": plan["run_dir"], "classification": "experiment_not_promoted"}
    ensure_root(args.destination, identity)
    lock = (args.destination / "watcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client = ControlSSH(args.host, args.control_path, args.ssh_port)
    deadline = time.monotonic() + args.wait_seconds
    final = args.destination / "train"
    transferred = final.exists()
    if transferred:
        completion = verify_local(final, plan)
    def status(value, **extra):
        record = {**identity, "state": value, "watcher_pid": os.getpid(), "B_main_started": False, **extra}
        write_json(args.destination / "watch-state.json", record)
        print(json.dumps(record), flush=True)
    status("watcher_started", artifacts_already_present=transferred)
    while time.monotonic() < deadline:
        try:
            if not transferred:
                health = read_worker(client, plan)
                status("observing_pilot", health=health)
                if health["worker_status"] is not None and not health["completion_present"]:
                    status("pilot_terminal_without_model", health=health)
                    write_json(args.destination / "handoff.json", {"next_action": "blocked_resources" if health["worker_status"].get("resource_stop") else "blocked_pilot_or_export",
                        "B_main_started": False, "classification": "experiment_not_promoted", "worker_status": health["worker_status"]})
                    return 2
                if not health["tmux_session_present"] and not health["completion_present"]:
                    status("pilot_controller_missing", health=health); return 2
                _, completion, _ = probe(client, plan["run_dir"], plan["python"])
                if completion["requested_iterations"] != 500:
                    raise ValueError("not this pilot update target")
                attempt = args.destination / ("attempt-" + uuid.uuid4().hex)
                receipt = pull_artifacts(client, plan["run_dir"], attempt, remote_python=plan["python"])
                manifest = json.loads((attempt / "run_manifest.json").read_text())
                verify_transfer(manifest, plan)
                if any(manifest.get(k) != v for k, v in plan["identity"].items()):
                    raise ValueError("pilot manifest identity differs")
                attempt.rename(final)
                transferred = True
                status("experiment_artifacts_verified", receipt=receipt, completed_updates=completion["completed_updates"])
            if completion["export_status"] != "verified":
                status("checkpoint_returned_but_export_failed", completion=completion)
                write_json(args.destination / "handoff.json", {"next_action": "blocked_pilot_or_export", "B_main_started": False})
                return 2
            if not args.a_plan:
                return 0
            script = str(Path(plan["repo"]) / "scripts/round3/pipeline_handoff.py")
            command = shlex.join([plan["python"], script, "--a-plan", args.a_plan, "--a-evaluation", args.a_evaluation,
                                  "--pilot-plan", str(Path(plan["audit_dir"]) / "plan.json")])
            handoff = json.loads(client.exec(command))
            if handoff["next_action"] == "prepare_B1_main_from_A_final":
                handoff["A_local_return_checked"] = verify_a_return(args.a_return, handoff)
                if not handoff["A_local_return_checked"]:
                    handoff["next_action"] = "waiting_A_local_return"
            write_json(args.destination / "handoff.json", handoff)
            status("handoff_observed", next_action=handoff["next_action"])
            if not handoff["next_action"].startswith("waiting_"):
                return 0
        except (NotReady, Unavailable) as error:
            status("waiting_or_transport_unavailable", reason=str(error), authentication_fallback=False)
        except (ValueError, OSError, RuntimeError) as error:
            status("blocked_evidence", reason=str(error))
            write_json(args.destination / "handoff.json", {"next_action": "blocked_evidence_or_export",
                "B_main_started": False, "reason": str(error)})
            return 2
        time.sleep(min(60, max(0, deadline - time.monotonic())))
    status("watcher_budget_expired", remote_training_affected=False)
    return 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "watch"))
    parser.add_argument("--experiment")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--a-plan", help="remote frozen A plan, optional handoff observation")
    parser.add_argument("--a-evaluation", help="remote four-height evaluation directory")
    parser.add_argument("--a-return", type=Path, help="existing A watcher's local artifacts directory; read-only")
    parser.add_argument("--wait-seconds", type=int, default=52 * 3600)
    parser.add_argument("--launch", action="store_true", help="submit mode: explicitly start pilot AND local watcher")
    add_ssh_arguments(parser)
    args = parser.parse_args()
    if bool(args.a_plan) != bool(args.a_evaluation) or not 1 <= args.wait_seconds <= 52 * 3600:
        parser.error("provide both A evidence paths; finite wait <=52h")
    if args.a_plan and args.a_return is None:
        parser.error("handoff observation also requires the existing A local return directory")
    if args.mode == "watch":
        if args.plan is None:
            parser.error("watch requires plan")
        return watch(args)
    if not args.experiment:
        parser.error("experiment required")
    entry = str(Path(args.experiment) / CODE_DIRECTORY / "scripts/round3/pipeline_pilot.py")
    command = shlex.join(["/home/kaiser/robot-rl-sim60/env/bin/python", entry, "--experiment", args.experiment, "--launch"])
    if not args.launch:
        print(json.dumps({"dry_run": True, "remote_command": command, "local_watcher_created": False})); return 0
    args.destination.mkdir(parents=False, exist_ok=False)
    client = ControlSSH(args.host, args.control_path, args.ssh_port)
    result = json.loads(client.exec(command))
    write_json(args.destination / "submission.json", result)
    if not result["submitted"]:
        return 2
    path = args.destination / "pilot-plan.json"
    write_json(path, result["plan"])
    watcher = [sys.executable, str(Path(__file__).resolve()), "watch", "--plan", str(path),
               "--destination", str(args.destination / "return"), "--host", args.host,
               "--control-path", args.control_path, "--ssh-port", str(args.ssh_port), "--wait-seconds", str(args.wait_seconds)]
    if args.a_plan:
        watcher += ["--a-plan", args.a_plan, "--a-evaluation", args.a_evaluation, "--a-return", str(args.a_return)]
    session = "b1-return-" + uuid.uuid4().hex
    # Only the actual submit path creates a local watcher. A disconnect never stops remote training.
    log = args.destination / "watcher.log"
    shell = shlex.join(["timeout", "--signal=TERM", "--kill-after=30s", str(args.wait_seconds + 120), *watcher]) + " > " + shlex.quote(str(log)) + " 2>&1"
    reply = subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], capture_output=True, text=True, timeout=10)
    record = {"pilot_submitted": True, "watcher_submitted": reply.returncode == 0, "watcher_session": session,
              "plan": str(path), "watcher_log": str(log), "stderr": reply.stderr, "classification": "experiment_not_promoted"}
    write_json(args.destination / "watcher-submission.json", record)
    print(json.dumps(record))
    return 0 if reply.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
