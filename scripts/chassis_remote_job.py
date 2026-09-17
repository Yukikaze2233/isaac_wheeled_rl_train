#!/usr/bin/env python3
"""Own one training child; after exit, package its artifacts with exact byte hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(root, exit_code):
    train = root / "train"
    completion = json.loads((train / "completion.json").read_text()) if (train / "completion.json").exists() else {}
    selected = [p for p in train.rglob("*") if p.is_file() and "git" not in p.relative_to(train).parts
                and (p.suffix in (".json", ".jsonl", ".pt", ".onnx", ".py") or p.name.startswith("events.out.tfevents."))]
    if (root / "train.log").exists():
        selected.append(root / "train.log")
    files = {str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": sha(p)} for p in selected}
    temporary = root / "delivery.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz", compresslevel=1) as output:
        for path in selected:
            output.add(path, arcname=str(path.relative_to(root)), recursive=False)
    archive = root / "delivery.tar.gz"
    temporary.replace(archive)
    result = {"status": "ready", "training_status": completion.get("status", "exited_without_completion"),
              "exit_code": exit_code, "created_at": datetime.now(timezone.utc).isoformat(),
              "archive": archive.name, "archive_bytes": archive.stat().st_size,
              "archive_sha256": sha(archive), "files": files,
              "successful_updates": completion.get("successful_updates"),
              "export_verified": completion.get("export", {}).get("verified", False)}
    pending = root / "delivery.tmp"
    pending.write_text(json.dumps(result, indent=2) + "\n")
    pending.replace(root / "delivery.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Training command required")
    with (args.run_root / "train.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

        def forward(signum, _frame):
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        (args.run_root / "job.json").write_text(json.dumps({"pid": process.pid, "command": command,
            "started_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        code = process.wait()
    finalize(args.run_root, code)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
