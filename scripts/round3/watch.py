#!/usr/bin/env python3
"""Read-only completion watcher; reuse V40's confined, SHA-verified artifact pull."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import uuid

from common import PARENT_SHA256, ControlSSH, add_ssh_arguments, code_directory, file_record, verify_warm_start
from pull_v40_artifacts import NotReady, Unavailable, open_local_parent, probe, pull_artifacts
from wheeled_algo.v40_job import JobError, json_bytes, strict_json


def read_worker(client, plan):
    script = """import json,pathlib,subprocess
audit=pathlib.Path(AUDIT)
run=pathlib.Path(RUN)
status=audit/'worker.status.json'
started=audit/'train.started.json'
process=json.loads(started.read_text()) if started.exists() else None
alive=False
if process and process.get('pid'):
    try:
        argv=(pathlib.Path('/proc')/str(process['pid'])/'cmdline').read_bytes().split(b'\\0')
        alive=str(run).encode() in argv
    except OSError:
        pass
session=subprocess.run(['/usr/bin/tmux','has-session','-t',SESSION],capture_output=True,timeout=5).returncode==0
checkpoints=sorted((p for p in run.glob('model_*.pt') if p.is_file()),key=lambda p:p.stat().st_mtime)[-3:]
print(json.dumps({'worker_status':json.loads(status.read_text()) if status.exists() else None,
 'training_process':process,'training_pid_present':alive,'tmux_session_present':session,
 'checkpoints':[{'name':p.name,'size':p.stat().st_size,'mtime':p.stat().st_mtime} for p in checkpoints],
 'completion_present':(run/'completion.json').exists()}))
""".replace("AUDIT", repr(plan["audit_dir"])).replace("RUN", repr(plan["run_dir"])).replace("SESSION", repr(plan["session"]))
    return strict_json(client.exec(shlex.join(["python3", "-c", script])))


def verify_expected_manifest(client, plan, completion):
    if completion["requested_iterations"] != plan["updates"]:
        raise JobError("completion update target differs from this profile")
    item = next(i for i in completion["artifacts"] if i["path"] == "run_manifest.json")
    with client.download(plan["run_dir"] + "/run_manifest.json") as source:
        raw = source.read(item["size"] + 1)
    if len(raw) != item["size"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise JobError("run manifest size/hash mismatch")
    manifest = strict_json(raw)
    if any(manifest.get(key) != value for key, value in plan["identity"].items()):
        raise JobError("run manifest differs from expected contract/stage/seed")
    verify_warm_start(manifest, plan["contract_sha256"])


def ensure_root(destination, identity):
    if not destination.exists():
        path, parent = open_local_parent(destination)
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent)
        finally:
            os.close(parent)
        with (destination / "watch-owner.json").open("x") as stream:
            json.dump(identity, stream, indent=2)
    elif (destination.is_symlink() or not destination.is_dir()
          or (destination / "watch-owner.json").is_symlink()
          or strict_json((destination / "watch-owner.json").read_bytes()) != identity):
        raise JobError("destination is not owned by this exact run; refusing reuse")


def already_verified(root, plan):
    final = root / "artifacts"
    if not final.exists():
        return False
    if final.is_symlink():
        raise JobError("published artifact directory is a symlink")
    from wheeled_algo.v40_job import validate_completion
    completion = validate_completion(strict_json((final / "completion.json").read_bytes()))
    if completion["requested_iterations"] != plan["updates"]:
        raise JobError("local completion belongs to another update target")
    receipt = strict_json((final / "local_receipt.json").read_bytes())
    if receipt.get("transfer_verified") is not True or receipt.get("remote_run_dir") != plan["run_dir"]:
        raise JobError("published transfer receipt identity mismatch")
    if receipt.get("completion_sha256") != file_record(final / "completion.json")["sha256"]:
        raise JobError("published completion changed")
    manifest = strict_json((final / "run_manifest.json").read_bytes())
    if any(manifest.get(key) != value for key, value in plan["identity"].items()):
        raise JobError("published model identity differs from watcher plan")
    verify_warm_start(manifest, plan["contract_sha256"])
    for item in completion["artifacts"]:
        if file_record(final / item["path"]) != {"size": item["size"], "sha256": item["sha256"]}:
            raise JobError("published local artifact changed")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="copy of remote audit/plan.json")
    parser.add_argument("--destination", type=Path, required=True, help="new or matching watcher-owned directory; parent exists")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-seconds", type=int, default=180000, help="finite local watcher budget, default 50h")
    parser.add_argument("--once", action="store_true", help="one read/pull attempt; never wait")
    add_ssh_arguments(parser)
    args = parser.parse_args()
    if args.poll_seconds < 30 or not 1 <= args.wait_seconds <= 7 * 86400:
        parser.error("poll >=30 seconds; wait must be 1 second..7 days")
    plan = strict_json(args.plan.read_bytes())
    if plan.get("schema_version") != 1 or plan.get("parent_model_sha256") != PARENT_SHA256:
        raise JobError("invalid Round3 plan/parent")
    if plan.get("repo") != str(code_directory(Path(plan["snapshot"]))):
        raise JobError("watcher plan must use the fixed isaac_wheeled_rl_train code directory")
    identity = {"plan_sha256": hashlib.sha256(json_bytes(plan)).hexdigest(), "host": args.host,
                "remote_run_dir": plan["run_dir"], "profile": plan["profile"],
                "git_commit": plan["git_commit"], "parent_model_sha256": PARENT_SHA256}
    root = args.destination.absolute()
    ensure_root(root, identity)
    lock = os.open(root / "watcher.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client = ControlSSH(args.host, args.control_path, args.ssh_port)
    deadline = time.monotonic() + args.wait_seconds

    def update(state, **extra):
        record = {**identity, "state": state, "watcher_pid": os.getpid(),
                  "updated_at": datetime.now(timezone.utc).isoformat(), "policy_quality_verified": False, **extra}
        temporary = root / ".watch-state.json"
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(root / "watch-state.json")
        print(json.dumps(record), flush=True)

    try:
        if already_verified(root, plan):
            update("already_verified")
            return 0
        while time.monotonic() < deadline:
            try:
                health = read_worker(client, plan)
                update("observed", health=health)
                terminal = health["worker_status"]
                if terminal is not None and not health["completion_present"]:
                    update("terminal_without_completion", health=health)
                    return 2
                if not health["tmux_session_present"] and not health["completion_present"]:
                    update("controller_missing_without_completion", health=health, training_success=False)
                    return 2
                _, completion, _ = probe(client, plan["run_dir"], plan["python"])
                verify_expected_manifest(client, plan, completion)
                attempt = root / ("attempt-" + uuid.uuid4().hex)
                update("transferring", attempt=str(attempt))
                receipt = pull_artifacts(client, plan["run_dir"], attempt, remote_python=plan["python"])
                attempt.rename(root / "artifacts")
                formal_completed = (plan["profile"] == "train" and completion["status"] == "completed"
                                    and completion["completed_updates"] == plan["updates"]
                                    and completion["export_status"] == "verified")
                update("artifacts_verified", receipt=receipt, completed_updates=completion["completed_updates"],
                       requested_updates=plan["updates"], formal_training_target_completed=formal_completed,
                       classification="formal_training_artifacts" if plan["profile"] == "train" else "engineering_only_not_round3_final")
                return 0
            except NotReady as error:
                update("waiting_completion", detail=str(error))
            except Unavailable as error:
                update("transport_unavailable", detail=str(error), action="restore ControlMaster separately; no new authentication attempted")
            except (JobError, OSError, ValueError) as error:
                update("rejected", detail=str(error))
                return 2
            if args.once:
                return 3
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        update("watcher_budget_expired", remote_training_affected=False)
        return 3
    finally:
        os.close(lock)


if __name__ == "__main__":
    raise SystemExit(main())
