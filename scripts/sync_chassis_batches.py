#!/usr/bin/env python3
"""Recover completed batches while the rest of the remote curriculum continues."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from chassis_batch_export import digest, write_json


def recover_batch(archive, receipt, destination):
    if digest(archive) != receipt["archive_sha256"]:
        raise ValueError("Batch archive checksum mismatch")
    destination = destination.resolve()
    staging = destination.parent / (".unpack-" + receipt["batch_id"])
    staging.mkdir(exist_ok=True)
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            path = (staging / member.name).resolve()
            expected = receipt["files"].get(member.name)
            if (not member.isfile() or expected is None or member.name in seen
                    or not path.is_relative_to(staging.resolve()) or member.size != expected["size"]):
                raise ValueError("Invalid batch archive member")
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
            if digest(path) != expected["sha256"]:
                raise ValueError(f"Batch file checksum mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(receipt["files"]):
        raise ValueError("Incomplete batch archive")
    for name, expected in receipt["files"].items():
        target = destination / name
        if target.exists() and digest(target) != expected["sha256"]:
            raise ValueError(f"Existing immutable artifact differs: {name}")
    for name in seen:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        (staging / name).replace(target)
    shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=60.)
    parser.add_argument("--seconds", type=float, default=270000.)
    args = parser.parse_args()
    owner = json.loads(args.receipt.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner_path = args.output / "owner.json"
    if owner_path.exists() and json.loads(owner_path.read_text())["remote_root"] != owner["remote_root"]:
        raise ValueError("Recovery destination belongs to another run")
    write_json(owner_path, owner)
    journal_path = args.output / "recovery.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"verified_batches": {}}
    ssh = ["ssh", "-S", owner["control_path"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-p", str(owner["ssh_port"]), owner["host"]]
    exporter = Path(__file__).with_name("chassis_batch_export.py").read_text()
    command = "python3 -B - --run-root " + shlex.quote(owner["remote_root"])
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            process = subprocess.run(ssh + [command], input=exporter, capture_output=True, text=True, timeout=600, check=True)
            index = json.loads(process.stdout)
            if index["remote_root"] != owner["remote_root"]:
                raise ValueError("Remote batch identity mismatch")
            for batch in index["batches"]:
                key = batch["batch_id"]
                if journal["verified_batches"].get(key, {}).get("archive_sha256") == batch["archive_sha256"]:
                    continue
                archive = args.output / (key + ".tar.gz")
                download = archive.with_suffix(".download")
                remote = owner["host"] + ":" + owner["remote_root"] + "/" + batch["archive"]
                subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ControlPath=" + owner["control_path"],
                    "-P", str(owner["ssh_port"]), remote, str(download)], timeout=600, check=True)
                recover_batch(download, batch, args.output / "artifacts")
                download.replace(archive)
                write_json(args.output / (key + ".json"), batch)
                journal["verified_batches"][key] = {"unit": batch["unit"], "kind": batch["kind"],
                    "files": len(batch["files"]), "archive_sha256": batch["archive_sha256"]}
                write_json(journal_path, journal)
                print("BATCH_RECOVERED", batch["unit"], len(batch["files"]), flush=True)
            journal.update(run_finished=index["run_finished"], checked_at_unix=time.time())
            write_json(journal_path, journal)
            if args.once or index["run_finished"]:
                return 0
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            write_json(args.output / "last_error.json", {"time": time.time(), "error": str(error)})
            if args.once:
                raise
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
