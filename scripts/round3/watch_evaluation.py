#!/usr/bin/env python3
"""Pull only the bounded post-training report allowlist through an existing SSH master."""
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

from common import ControlSSH, Unavailable, add_ssh_arguments
from watch import ensure_root
from evaluate_finished import write_json

ALLOWED = {"evaluation_result.json", "evaluation_report.md", "training_completion.json"} | {
    f"h{height:03d}/{name}" for height in (29, 30, 31, 32) for name in ("summary.json", "telemetry.csv")}


def validate_report_manifest(manifest, plan_sha, evaluation_id):
    if (manifest.get("schema_version") != 1 or manifest.get("training_plan_sha256") != plan_sha
            or manifest.get("evaluation_id") != evaluation_id):
        raise ValueError("wrong evaluation identity")
    seen = set()
    for item in manifest["files"]:
        if (item["path"] not in ALLOWED or item["path"] in seen or type(item["size"]) is not int
                or not 0 < item["size"] <= 32 * 1024 * 1024
                or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in item["sha256"])):
            raise ValueError("invalid evaluation artifact")
        seen.add(item["path"])
    if not {"evaluation_result.json", "evaluation_report.md"} <= seen:
        raise ValueError("missing evaluation result/report")


def pull(client, remote, root, plan_sha):
    marker = remote + "/report_manifest.json"
    exists = client.exec(shlex.join(["python3", "-c", f"from pathlib import Path; print(int(Path({marker!r}).is_file()))"]))
    if exists.strip() != "1":
        return None
    with client.download(marker) as stream:
        raw = stream.read(1024 * 1024)
    manifest = json.loads(raw)
    validate_report_manifest(manifest, plan_sha, Path(remote).name)
    target = root / "reports"
    incoming = target if target.exists() else root / "incoming"
    if incoming.is_symlink():
        raise ValueError("report directory cannot be a symlink")
    incoming.mkdir(exist_ok=True)
    for item in manifest["files"]:
        path = incoming / item["path"]
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("report member cannot be a symlink")
        path.parent.mkdir(exist_ok=True)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
            continue
        if target.exists():
            raise ValueError("published report was changed; refusing overwrite")
        with client.download(remote + "/" + item["path"]) as stream:
            data = stream.read(item["size"] + 1)
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("report transfer hash/size mismatch")
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(data)
        temporary.replace(path)
    with client.download(marker) as stream:
        if stream.read(1024 * 1024) != raw:
            raise ValueError("remote report manifest changed during transfer")
    if not target.exists():
        (incoming / "report_manifest.json").write_bytes(raw)
        incoming.rename(target)
    receipt = {"transfer_verified": True, "remote_output": remote,
               "report_manifest_sha256": hashlib.sha256(raw).hexdigest(), "evaluation_state": manifest["state"],
               "policy_quality_verified_by_transfer": False, "files_verified": len(manifest["files"])}
    write_json(root / "local_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-output", required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=51 * 3600)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    add_ssh_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 51 * 3600 or args.poll_seconds < 1:
        parser.error("wait 1..51h and positive poll required")
    root = args.destination.absolute()
    plan_sha = hashlib.sha256(args.training_plan.read_bytes()).hexdigest()
    identity = {"training_plan_sha256": plan_sha, "remote_output": args.remote_output, "host": args.host}
    ensure_root(root, identity)
    lock = (root / "watcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + args.wait_seconds
    client = ControlSSH(args.host, args.control_path, args.ssh_port)

    def update(state, **extra):
        record = {**identity, "state": state, "watcher_pid": os.getpid(),
                  "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        write_json(root / "watch-state.json", record)
        print(json.dumps(record), flush=True)

    while time.monotonic() < deadline:
        try:
            receipt = pull(client, args.remote_output, root, plan_sha)
            if receipt:
                update("evaluation_reports_verified", receipt=receipt)
                return 0
            update("waiting_evaluation_report")
        except Unavailable as error:
            update("transport_unavailable_retryable", detail=str(error),
                   action="restore existing ControlMaster separately; no password/key changes; rerun same command if expired")
        except (ValueError, OSError, RuntimeError) as error:
            update("report_rejected", detail=str(error))
            return 2
        if args.once:
            return 3
        time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    update("expired_waiting_evaluation", retryable=True, remote_training_affected=False)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
