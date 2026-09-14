#!/usr/bin/env python3
"""Observe Round4, SHA-return artifacts, then prepare/dispatch final evaluation; never train."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import uuid

from r4_common import (
    CODE_DIRECTORY, ControlSSH, JobError, add_ssh_arguments, file_record, json_bytes,
    strict_json, verify_scratch, write_json,
)
from pull_v40_artifacts import NotReady, Unavailable, probe, pull_artifacts
from round3.watch import ensure_root, read_worker
from wheeled_algo.v40_job import validate_completion, verify_export_sidecar


def verify_local(root, plan):
    completion = validate_completion(strict_json((root / "completion.json").read_bytes()))
    if completion["requested_iterations"] != 30000:
        raise JobError("not this formal update target")
    receipt = strict_json((root / "local_receipt.json").read_bytes())
    if (receipt.get("transfer_verified") is not True or receipt.get("remote_run_dir") != plan["run_dir"]
            or receipt.get("completion_sha256") != file_record(root / "completion.json")["sha256"]):
        raise JobError("local return receipt differs from this run")
    for item in completion["artifacts"]:
        if file_record(root / item["path"]) != {k: item[k] for k in ("size", "sha256")}:
            raise JobError("returned artifact changed")
    manifest = strict_json((root / "run_manifest.json").read_bytes())
    verify_scratch(manifest, plan)
    if manifest["training_curriculum"]["completed_updates"] != completion["completed_updates"]:
        raise JobError("returned curriculum clock differs from completed updates")
    if completion["export_status"] == "verified":
        verify_export_sidecar(root)
    return completion


def pull_audit(client, plan, root):
    names = ("plan.json", "effective-command.json", "worker.started.json", "worker.status.json",
             "train.started.json", "train.status.json", "train.log", "resources-start.json",
             "resources-submission.json", "resources.jsonl", "manifest-evidence.json", "physics-identity.json")
    paths = {name: str(Path(plan["audit_dir"]) / name) for name in names}
    paths["snapshot.json"] = str(Path(plan["experiment"]) / "snapshot.json")
    script = ("import hashlib,json; from pathlib import Path; paths=" + repr(paths) + "; "
              "print(json.dumps({name:{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} "
              "for name,path in paths.items() if (p:=Path(path)).is_file() and not p.is_symlink()}))")
    records = strict_json(client.exec(shlex.join(["python3", "-c", script])))
    final = root / "audit"
    incoming = final if final.exists() else root / ("audit-attempt-" + uuid.uuid4().hex)
    incoming.mkdir(exist_ok=True)
    for name, record in records.items():
        if name not in paths or not 0 <= record["size"] <= 64 * 1024 * 1024:
            raise JobError("unbounded audit artifact")
        path = incoming / name
        if final.exists():
            if file_record(path) != record:
                raise JobError("published audit changed")
            continue
        with client.download(paths[name]) as stream:
            data = stream.read(record["size"] + 1)
        if {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} != record:
            raise JobError("audit changed during return")
        path.write_bytes(data)
    if strict_json(client.exec(shlex.join(["python3", "-c", script]))) != records:
        raise JobError("audit changed during return")
    if not final.exists():
        write_json(incoming / "local_receipt.json", {"transfer_verified": True, "files": records})
        incoming.rename(final)


def pull_reports(client, remote, root, request_sha):
    """Hash-verified reports are evidence, never an automatic policy-quality pass."""
    marker = remote + "/report_manifest.json"
    query = f"from pathlib import Path; print(int(Path({marker!r}).is_file()))"
    if client.exec(shlex.join(["python3", "-c", query])).strip() != "1":
        return None
    with client.download(marker) as stream:
        raw = stream.read(1024 * 1024)
    manifest = strict_json(raw)
    if manifest.get("schema_version") != 1 or manifest.get("request_sha256") != request_sha:
        raise JobError("evaluation report belongs to another request")
    files = manifest["files"]
    names = [item["path"] for item in files]
    if (len(names) != len(set(names)) or "result.json" not in names
            or not any(Path(name).suffix == ".csv" for name in names) or len(names) > 100):
        raise JobError("evaluation report lacks results/raw telemetry or repeats members")
    for item in files:
        path = Path(item["path"])
        if (path.is_absolute() or ".." in path.parts or path.suffix not in {".json", ".csv", ".md"}
                or type(item["size"]) is not int or not 0 < item["size"] <= 32 * 1024 * 1024
                or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64):
            raise JobError("unbounded or unconfined evaluation member")
    final = root / "evaluation-reports"
    incoming = final if final.exists() else root / ("evaluation-attempt-" + uuid.uuid4().hex)
    incoming.mkdir(exist_ok=True)
    for item in files:
        path = incoming / item["path"]
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise JobError("linked evaluation report member")
        expected = {k: item[k] for k in ("size", "sha256")}
        if final.exists():
            if file_record(path) != expected:
                raise JobError("published evaluation report changed")
            continue
        with client.download(remote + "/" + item["path"]) as stream:
            data = stream.read(item["size"] + 1)
        if {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} != expected:
            raise JobError("evaluation transfer hash mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    result = strict_json((incoming / "result.json").read_bytes())
    expected_cases = {f"{kind}_h{height:03d}" for kind in ("standing", "single_push") for height in (29, 30, 31, 32)}
    expected_cases |= {"tracking_vx_pos3", "tracking_vx_neg3", "tracking_wz_pos6", "tracking_wz_neg6"}
    cases = result.get("cases", [])
    if (len(cases) != len(expected_cases) or {case["id"] for case in cases} != expected_cases
            or any(case.get("status") not in {"completed", "failed"} or not case.get("metrics") for case in cases)):
        raise JobError("evaluation result is empty/incomplete; cannot publish a success placeholder")
    with client.download(marker) as stream:
        if stream.read(1024 * 1024) != raw:
            raise JobError("evaluation manifest changed while returning")
    if not final.exists():
        (incoming / "report_manifest.json").write_bytes(raw)
        incoming.rename(final)
    receipt = {"transfer_verified": True, "request_sha256": request_sha,
               "report_manifest_sha256": hashlib.sha256(raw).hexdigest(), "policy_quality_verified": False}
    write_json(root / "evaluation-receipt.json", receipt)
    return receipt


def evaluation_command_after_remote_hook(client, plan, evaluation_entry=None, push_delta=None):
    """Wait for the owned server dispatcher before its idempotent result lookup."""
    if plan.get("evaluation_entry"):
        path = str(Path(plan["audit_dir"]) / "evaluation-hook.json")
        query = f"from pathlib import Path; print(int(Path({path!r}).is_file()))"
        if client.exec(shlex.join(["python3", "-c", query])).strip() != "1":
            raise NotReady("waiting for the server evaluation hook to finish submission")
        with client.download(path) as stream:
            hook = strict_json(stream.read(1024 * 1024))
        if hook.get("exit_code") != 0:
            raise JobError("server evaluation dispatch failed; inspect evaluation-hook.json")
    command = [plan["python"], str(Path(plan["repo"]) / "scripts/round4/evaluation.py"),
               "--plan", str(Path(plan["audit_dir"]) / "plan.json")]
    if push_delta is not None:
        command += ["--push-delta-v", *map(str, push_delta)]
    entry = evaluation_entry or plan.get("evaluation_entry")
    if entry:
        command += ["--evaluator", entry, "--launch"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--wait-hours", type=int, choices=(52, 76), default=52)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--evaluation-entry", help="optional evaluator in this or another verified committed snapshot")
    parser.add_argument("--push-delta-v", type=float, nargs=2)
    add_ssh_arguments(parser)
    args = parser.parse_args()
    plan = strict_json(args.plan.read_bytes())
    if (plan.get("profile") != "round4_full" or plan.get("initialization") != "scratch"
            or plan.get("parent") is not None or plan.get("updates") != 30000 or plan.get("num_envs") != 1024
            or plan["repo"] != str(Path(plan["experiment"]) / CODE_DIRECTORY)):
        raise JobError("watcher requires the Round4 scratch formal plan")
    root = args.destination.absolute()
    identity = {"plan_sha256": hashlib.sha256(json_bytes(plan)).hexdigest(), "host": args.host,
                "remote_run_dir": plan["run_dir"], "initialization": "scratch", "parent": None}
    ensure_root(root, identity)
    lock = os.open(root / "watcher.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client = ControlSSH(args.host, args.control_path, args.ssh_port)
    deadline = time.monotonic() + args.wait_hours * 3600

    def state(value, **extra):
        record = {**identity, "state": value, "watcher_pid": os.getpid(),
                  "policy_quality_verified": False, **extra}
        write_json(root / "watch-state.json", record)
        print(json.dumps(record), flush=True)

    try:
        while time.monotonic() < deadline:
            try:
                health = read_worker(client, plan)
                state("observing", health=health)
                final = root / "artifacts"
                if not final.exists():
                    if health["worker_status"] is not None and not health["completion_present"]:
                        pull_audit(client, plan, root)
                        state("terminal_without_completion", health=health)
                        return 2
                    if not health["tmux_session_present"] and not health["completion_present"]:
                        state("controller_missing", health=health)
                        return 2
                    _, completion, _ = probe(client, plan["run_dir"], plan["python"])
                    if completion["requested_iterations"] != 30000:
                        raise JobError("remote completion differs from formal target")
                    attempt = root / ("attempt-" + uuid.uuid4().hex)
                    pull_artifacts(client, plan["run_dir"], attempt, remote_python=plan["python"])
                    verify_local(attempt, plan)
                    attempt.rename(final)
                completion = verify_local(final, plan)
                completed = (completion["status"] == "completed" and completion["completed_updates"] == 30000
                             and completion["export_status"] == "verified")
                state("artifacts_verified", formal_training_target_completed=completed,
                      completed_updates=completion["completed_updates"], export_status=completion["export_status"])
                if not completed:
                    if health["worker_status"] is not None:
                        pull_audit(client, plan, root)
                    return 2
                # Final export may precede Kit shutdown. Evaluation waits for the owned worker to exit cleanly.
                terminal = health["worker_status"]
                if health["training_pid_present"] or terminal is None:
                    raise NotReady("waiting for formal process/worker cleanup before evaluation")
                if terminal.get("formal_training_target_completed") is not True or terminal.get("children_reaped") is not True:
                    raise JobError("formal controller did not finish cleanly")
                if not (root / "audit").exists():
                    pull_audit(client, plan, root)
                command = evaluation_command_after_remote_hook(
                    client, plan, args.evaluation_entry, args.push_delta_v)
                dispatch = strict_json(client.exec(shlex.join(command)))
                write_json(root / "evaluation-dispatch.json", dispatch)
                with client.download(dispatch["request"]) as stream:
                    request_raw = stream.read(1024 * 1024)
                if hashlib.sha256(request_raw).hexdigest() != dispatch["request_sha256"]:
                    raise JobError("evaluation request transfer hash mismatch")
                (root / "evaluation-request.json").write_bytes(request_raw)
                state(dispatch["state"], formal_training_target_completed=True)
                if dispatch["state"] != "evaluator_submitted":
                    return 3
                receipt = pull_reports(client, str(Path(plan["stage_root"]) / "evaluation/reports"), root, dispatch["request_sha256"])
                if receipt:
                    state("evaluation_reports_verified", receipt=receipt, formal_training_target_completed=True)
                    return 0
            except (NotReady, Unavailable) as error:
                state("waiting_or_transport_unavailable", reason=str(error))
            except (JobError, ValueError, OSError, KeyError) as error:
                state("evidence_rejected", reason=str(error))
                return 2
            if args.once:
                return 3
            time.sleep(min(60, max(0, deadline - time.monotonic())))
        state("watcher_budget_expired", remote_training_affected=False)
        return 3
    finally:
        os.close(lock)


if __name__ == "__main__":
    raise SystemExit(main())
