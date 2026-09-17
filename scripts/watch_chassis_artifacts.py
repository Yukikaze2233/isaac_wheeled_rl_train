#!/usr/bin/env python3
"""Wait for an owned remote delivery, download it, and verify every recovered file."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.)
    parser.add_argument("--seconds", type=float, default=264000.)
    args = parser.parse_args()
    plan = json.loads(args.receipt.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    owner = args.output / "owner.json"
    if owner.exists() and json.loads(owner.read_text())["remote_root"] != plan["remote_root"]:
        raise ValueError("Destination belongs to another run")
    owner.write_text(json.dumps(plan, indent=2) + "\n")
    ssh = ["ssh", "-S", plan["control_path"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-p", str(plan["ssh_port"]), plan["host"]]
    query = ("import pathlib; p=pathlib.Path(" + repr(plan["remote_root"] + "/delivery.json")
             + "); print(p.read_text() if p.exists() else '{}')")
    started = time.monotonic()
    while time.monotonic() - started < args.seconds:
        try:
            response = subprocess.run(ssh + ["python3 -c " + shlex.quote(query)], capture_output=True, text=True, timeout=30, check=True)
            delivery = json.loads(response.stdout)
            if delivery.get("status") != "ready":
                time.sleep(args.interval)
                continue
            temporary = args.output / "delivery.download"
            subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ControlPath=" + plan["control_path"],
                "-P", str(plan["ssh_port"]), plan["host"] + ":" + plan["remote_root"] + "/delivery.tar.gz", str(temporary)],
                timeout=3600, check=True)
            if sha(temporary) != delivery["archive_sha256"]:
                raise ValueError("Downloaded archive hash mismatch")
            target = (args.output / "artifacts").resolve()
            target.mkdir(exist_ok=True)
            with tarfile.open(temporary, "r:gz") as archive:
                for member in archive:
                    path = (target / member.name).resolve()
                    if (not member.isfile() or member.name not in delivery["files"]
                            or not path.is_relative_to(target) or member.size != delivery["files"][member.name]["size"]):
                        raise ValueError("Unexpected delivery archive member")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, path.open("wb") as output:
                        shutil.copyfileobj(source, output)
            for name, details in delivery["files"].items():
                if sha(target / name) != details["sha256"]:
                    raise ValueError(f"Recovered file mismatch: {name}")
            temporary.replace(args.output / "delivery.tar.gz")
            (args.output / "delivery.json").write_text(json.dumps(delivery, indent=2) + "\n")
            (args.output / "recovery.json").write_text(json.dumps({"verified": True,
                "training_status": delivery["training_status"], "files": len(delivery["files"]),
                "archive_sha256": delivery["archive_sha256"]}, indent=2) + "\n")
            print("RECOVERED", delivery["training_status"], len(delivery["files"]), flush=True)
            return
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            (args.output / "watch_error.json").write_text(json.dumps({"time": time.time(), "error": str(exc)}) + "\n")
            time.sleep(args.interval)
    raise SystemExit("Artifact watcher budget expired; restart with the same receipt/output")


if __name__ == "__main__":
    main()
