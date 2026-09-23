#!/usr/bin/env python3
"""Mirror append-only TensorBoard events so the viewer need not consume trainer RAM."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

from chassis_batch_export import write_json


def mirror_once(owner, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {key: owner[key] for key in ("host", "ssh_port", "remote_root")}
        marker = output / "owner.json"
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise ValueError("TensorBoard cache belongs to another run")
        write_json(marker, identity)
        ssh = ["ssh", "-S", owner.get("control_path", "none"), "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
               "-p", str(owner["ssh_port"])]
        if owner.get("identity_file"):
            ssh += ["-o", "IdentitiesOnly=yes", "-i", owner["identity_file"]]
        command = ["rsync", "-az", "--append-verify", "--partial", "--prune-empty-dirs", "--protect-args", "--timeout=90",
                   "--include=*/", "--include=events.out.tfevents.*", "--exclude=*", "-e", shlex.join(ssh),
                   owner["host"] + ":" + owner["remote_root"].rstrip("/") + "/train/", str(output) + "/"]
        subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
        status = {**identity, "updated_at": datetime.now(timezone.utc).isoformat(),
                  "status": "synced", "scope": "live_event_cache_not_sealed_evidence"}
        write_json(output / "mirror_status.json", status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.)
    parser.add_argument("--seconds", type=float, default=604800.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.seconds <= 0:
        parser.error("Positive interval and runtime required")
    owner = json.loads(args.receipt.read_text())
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            print(json.dumps(mirror_once(owner, args.output)), flush=True)
            if args.once:
                return
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "mirror_error.json", {
                "updated_at": datetime.now(timezone.utc).isoformat(), "error": str(error)})
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
