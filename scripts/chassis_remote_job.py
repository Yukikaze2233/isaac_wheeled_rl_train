#!/usr/bin/env python3
"""Own training and seal recoverable batches independently of any SSH client."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
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


def validate_source_revision(root, command, previous, receipt_path):
    """Authenticate a frozen code repair without changing a campaign command."""
    receipt = json.loads(Path(receipt_path).read_text())
    plan_revision = receipt.get("schema") in ("task-plan-revision-v2", "task-plan-revision-v3", "task-plan-revision-v4", "task-plan-revision-v5", "task-plan-revision-v6", "task-plan-revision-v7", "task-plan-revision-v8", "task-plan-revision-v9", "task-plan-revision-v10", "task-plan-revision-v11", "task-plan-revision-v12", "task-plan-revision-v13", "task-plan-revision-v14", "task-plan-revision-v15", "task-plan-revision-v16")
    if (receipt.get("schema") not in ("task-source-revision-v1", "task-plan-revision-v2", "task-plan-revision-v3", "task-plan-revision-v4", "task-plan-revision-v5", "task-plan-revision-v6", "task-plan-revision-v7", "task-plan-revision-v8", "task-plan-revision-v9", "task-plan-revision-v10", "task-plan-revision-v11", "task-plan-revision-v12", "task-plan-revision-v13", "task-plan-revision-v14", "task-plan-revision-v15", "task-plan-revision-v16")
            or receipt.get("previous_job_sha256") != sha(root / "job.json")
            or receipt.get("command") != command
            or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("source_commit", ""))):
        raise ValueError("Source revision does not authenticate the previous job and new command")
    old, new = (Path(receipt[key]).resolve(strict=True) for key in ("old_source_root", "new_source_root"))
    if old == new or new.is_relative_to(old) or old.is_relative_to(new):
        raise ValueError("Source revision must use a separate frozen source directory")
    original = previous["command"]
    runner = str(old / "scripts/run_task_campaign.py")
    if original.count(runner) != 1 or original.count("--contract") != 1 or original.count("--run-dir") != 1:
        raise ValueError("Source revision supports only the existing task campaign runner")
    contract_index = original.index("--contract") + 1
    old_contract = Path(original[contract_index])
    if (not old_contract.is_absolute() or not old_contract.resolve().is_relative_to(old)
            or Path(original[original.index("--run-dir") + 1]).resolve() != (root / "train").resolve()):
        raise ValueError("Source revision cannot change the campaign run directory")
    relative_contract = old_contract.relative_to(old)
    new_relative = Path(receipt["new_contract_relative_path"]) if plan_revision else relative_contract
    if new_relative.is_absolute() or ".." in new_relative.parts:
        raise ValueError("Source revision contract must stay in the new source directory")
    new_contract = new / new_relative
    expected = list(original)
    expected[expected.index(runner)] = str(new / "scripts/run_task_campaign.py")
    expected[contract_index] = str(new_contract)
    if receipt.get("schema") == "task-plan-revision-v5":
        if original.count("--num-envs") != 1 or original[original.index("--num-envs") + 1] != "20000":
            raise ValueError("Task v5 resizes only the named 20000-environment predecessor")
        expected[expected.index("--num-envs") + 1] = "4096"
    if expected != command:
        raise ValueError("Source revision may change only runner and contract source prefixes")
    old_sha, new_sha = sha(old_contract), sha(new_contract)
    if ((old_sha != new_sha and not plan_revision) or receipt.get("old_contract_sha256") != old_sha
            or receipt.get("new_contract_sha256") != new_sha):
        raise ValueError("Source revision cannot change the frozen campaign contract")
    archive_path = Path(receipt["source_archive"])
    if not archive_path.is_absolute() or sha(archive_path) != receipt.get("source_archive_sha256"):
        raise ValueError("Source revision archive SHA differs")
    names = set()
    with tarfile.open(archive_path, "r:*") as archive:
        archived_commit = archive.pax_headers.get("comment")
        if archived_commit and archived_commit != receipt["source_commit"]:
            raise ValueError("Source revision archive commit differs")
        for member in archive:
            relative = Path(member.name)
            target = new / relative
            if (relative.is_absolute() or ".." in relative.parts
                    or not target.resolve().is_relative_to(new)
                    or not (member.isfile() or member.isdir())):
                raise ValueError("Source revision archive contains an unsupported path or link")
            if member.isdir():
                continue
            if member.name in names:
                raise ValueError("Source revision archive repeats a file")
            names.add(member.name)
            with archive.extractfile(member) as content:
                checksum = hashlib.sha256()
                for block in iter(lambda: content.read(1024 * 1024), b""):
                    checksum.update(block)
            if not target.is_file() or sha(target) != checksum.hexdigest():
                raise ValueError(f"Source revision extracted file differs: {member.name}")
    if not {"scripts/run_task_campaign.py", str(new_relative)} <= names:
        raise ValueError("Source revision archive is missing its runner or contract")
    if plan_revision:
        import sys
        sys.path.insert(0, str(new / "src"))
        from wheeled_tasks.chassis.task_campaign import validate_plan_migration
        validate_plan_migration(json.loads(old_contract.read_text()), json.loads(new_contract.read_text()), root / "train")
    return receipt


def preserve_previous_job(root, log_path, receipt_path=None):
    """Retain each ownership receipt before job.json changes to the next PID."""
    index = log_path.name.removeprefix("train.resume_").removesuffix(".log")
    target = root / f"job.previous_{index}.json"
    with target.open("xb") as output:
        output.write((root / "job.json").read_bytes())
    if receipt_path is not None:
        with (root / f"job.source_revision_{index}.json").open("xb") as output:
            output.write(Path(receipt_path).read_bytes())
    return target.name


def prepare_job_log(root, command, *, resume=False, source_revision_receipt=None):
    """Recover the same frozen job without overwriting its original log."""
    first = root / "train.log"
    if not resume:
        if source_revision_receipt is not None:
            raise ValueError("Source revision requires --resume-existing")
        return first
    previous = json.loads((root / "job.json").read_text())
    if source_revision_receipt is None:
        if previous["command"] != command:
            raise ValueError("Job resume must use exactly the original frozen command")
    else:
        validate_source_revision(root, command, previous, source_revision_receipt)
    process = Path(f"/proc/{previous['pid']}/cmdline")
    if process.exists() and os.fsencode(str(root / "train")) in process.read_bytes().split(b"\0"):
        raise ValueError("The original training job is still alive")
    index = 1
    while any((root / name).exists() for name in (
            f"train.resume_{index:03d}.log", f"job.previous_{index:03d}.json", f"job.source_revision_{index:03d}.json")):
        index += 1
    return root / f"train.resume_{index:03d}.log"


def finalize(root, exit_code):
    train = root / "train"
    completion = json.loads((train / "completion.json").read_text()) if (train / "completion.json").exists() else {}
    if not completion and (train / "report.json").exists():
        capacity = json.loads((train / "report.json").read_text())
        if capacity.get("scope") == "engineering_capacity_not_skill_acceptance":
            completion = {"status": "capacity_probes_finished", "successful_updates": sum(
                item.get("successful_updates", 0) for item in capacity.get("probes", []))}
    selected = [p for p in train.rglob("*") if p.is_file() and "git" not in p.relative_to(train).parts
                and (p.suffix in (".json", ".jsonl", ".pt", ".onnx", ".py", ".log", ".npz", ".csv") or p.name.startswith("events.out.tfevents."))]
    selected.extend(path for pattern in ("train.log", "train.resume_*.log", "job.json",
                    "job.previous_*.json", "job.source_revision_*.json") for path in root.glob(pattern) if path.is_file())
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
    selection = train / "artifact_selection.json"
    if selection.exists():
        result["artifact_selection"] = json.loads(selection.read_text())
    pending = root / "delivery.tmp"
    pending.write_text(json.dumps(result, indent=2) + "\n")
    pending.replace(root / "delivery.json")
    return result


def main():
    from chassis_batch_export import export_ready, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--source-revision-receipt", type=Path,
                        help="Authenticated frozen-source repair for an unchanged campaign/run directory")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Training command required")
    log_path = prepare_job_log(args.run_root, command, resume=args.resume_existing,
                              source_revision_receipt=args.source_revision_receipt)
    with log_path.open("x") as log:
        previous_job = preserve_previous_job(args.run_root, log_path, args.source_revision_receipt) if args.resume_existing else None
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

        def forward(signum, _frame):
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        write_json(args.run_root / "job.json", {"pid": process.pid, "command": command,
            "log": log_path.name, "previous_job": previous_job,
            "source_revision_receipt": ("job.source_revision_" + log_path.name.removeprefix("train.resume_").removesuffix(".log") + ".json"
                                        if args.source_revision_receipt else None),
            "started_at": datetime.now(timezone.utc).isoformat()})
        while process.poll() is None:
            try:
                index = export_ready(args.run_root)
                write_json(args.run_root / "archiver_status.json", {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "pid": process.pid, "sealed_batches": len(index["batches"]),
                    "status": "running", "independent_of_ssh_client": True})
            except (OSError, ValueError, RuntimeError) as error:
                write_json(args.run_root / "archiver_error.json", {
                    "updated_at": datetime.now(timezone.utc).isoformat(), "error": str(error)})
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        code = process.returncode
    index = export_ready(args.run_root)
    delivery = finalize(args.run_root, code)
    write_json(args.run_root / "archiver_status.json", {
        "updated_at": datetime.now(timezone.utc).isoformat(), "pid": process.pid,
        "sealed_batches": len(index["batches"]), "status": "finished",
        "training_status": delivery["training_status"], "independent_of_ssh_client": True})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
