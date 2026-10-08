#!/usr/bin/env python3
"""Four explicit source029 deployment steps; no SSH and no simulator preflight.

Run this copied, SHA-pinned helper in the owned Kaiser experiment. stage safely
extracts a previously copied archive; preflight is CPU-only; correct invokes the
reviewed one-shot origin transaction; launch uses the existing native wrapper.
An existing correction or launch intent is never implicitly retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

EXPECTED_PARENT = "e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967"
CONTRACT = Path("contracts/v6_task_campaign_v26.json")
CHAIN = ["speed", "terrain", "jump", "stepjump"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def durable(path, value):
    """Exclusive publication: partial or ambiguous prior attempts stay visible."""
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def no_owners(source, root):
    sys.path[:0] = [str(source / "scripts"), str(source / "src")]
    from queue_task_campaign_continuation import owned_processes
    owners = owned_processes(root)
    if owners:
        raise ValueError(f"Owned experiment is still active: {owners}")


def stopped_original(root, source):
    no_owners(source, root)
    job = read(root / "job.json")
    original = job["command"]
    if (original.count("--contract") != 1 or original.count("--run-dir") != 1
            or original[original.index("--contract") + 1] != str(source / CONTRACT)
            or original[original.index("--run-dir") + 1] != str(root / "train")
            or original.count(str(source / "scripts/run_task_campaign.py")) != 1):
        raise ValueError("Current job is not the named source028/v26 campaign")
    delivery = read(root / "delivery.json")
    if delivery.get("status") != "ready" or (root / "delivery.tar.gz.tmp").exists():
        raise ValueError("Old wrapper final archive is not ready")
    parent = root / "train/recovery/attempt_010/train/model_final.pt"
    seal = read(parent.with_suffix(".checkpoint.json"))
    complete = read(parent.parent / "completion.json")
    if (sha(parent) != EXPECTED_PARENT or seal.get("checkpoint_sha256") != EXPECTED_PARENT
            or seal.get("learning_updates") != 41393 or seal.get("task_actor_update") != 14188
            or seal.get("actor_updates_consumed") != 41223
            or complete.get("status") != "stopped" or complete.get("successful_updates") != 19):
        raise ValueError("Latest stopped REC final is not the exact 19-actor repair origin")
    for path in (parent, parent.with_suffix(".checkpoint.json"), parent.with_suffix(".contract.json"),
                 parent.parent / "completion.json"):
        item = delivery["files"].get(str(path.relative_to(root)))
        if item is None or item.get("sha256") != sha(path):
            raise ValueError("Latest full REC seal is not in the ready old wrapper manifest")
    return job


def extract_new(archive_path, destination, commit):
    if destination.exists():
        raise ValueError("New frozen source already exists; inspect it instead of restaging")
    with tempfile.TemporaryDirectory(prefix="source029-extract-", dir=destination.parent) as temp:
        staged = Path(temp) / "source"
        staged.mkdir()
        names = set()
        with tarfile.open(archive_path, "r:*") as archive:
            if archive.pax_headers.get("comment") != commit:
                raise ValueError("Archive PAX commit differs from the frozen source029 commit")
            for member in archive:
                relative = Path(member.name)
                target = staged / relative
                if (relative.is_absolute() or ".." in relative.parts
                        or not target.resolve().is_relative_to(staged)
                        or not (member.isfile() or member.isdir())):
                    raise ValueError("Unsafe or linked archive member")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if member.name in names:
                    raise ValueError("Duplicate archived source file")
                names.add(member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open("xb") as dst:
                    while block := src.read(1024 * 1024):
                        dst.write(block)
                target.chmod(member.mode & 0o777)
        if not {str(CONTRACT), "scripts/run_task_campaign.py", "scripts/chassis_remote_job.py",
                "scripts/repair_known_integrated_origin.py"} <= names:
            raise ValueError("Archive lacks the native source029 deployment entrypoints")
        staged.rename(destination)
    return len(names)


def validate_staging(args, directory):
    receipt = read(args.run_root / "source_revision_029.json")
    expected = read(directory / "stage.json")
    if (receipt["source_commit"] != args.source_commit
            or receipt["source_archive"] != str(args.source_archive)
            or receipt["new_source_root"] != str(args.new_source_root)
            or expected["source_commit"] != args.source_commit
            or expected.get("helper_sha256") != sha(__file__)):
        raise ValueError("CLI changed the published source029 identity")
    old = Path(receipt["old_source_root"])
    previous = stopped_original(args.run_root, old)
    # The owner scan is a legacy helper; subsequent authentication must load
    # the new frozen source, including the newly repaired scheduler.
    sys.path[:0] = [str(args.new_source_root / "scripts"), str(args.new_source_root / "src")]
    from chassis_remote_job import validate_source_revision
    validate_source_revision(args.run_root, receipt["command"], previous,
                             args.run_root / "source_revision_029.json")
    old_contracts = {str(p.relative_to(old)): sha(p) for p in (old / "contracts").rglob("*") if p.is_file()}
    new_contracts = {str(p.relative_to(args.new_source_root)): sha(p)
                     for p in (args.new_source_root / "contracts").rglob("*") if p.is_file()}
    if old_contracts != new_contracts:
        raise ValueError("Code-only source repair changed contract files or bytes")
    return receipt


def cpu_dispatch(plan):
    """Exercise actual run_domain dispatch, replacing only the selected worker."""
    from run_task_campaign import TaskCampaign
    campaign = TaskCampaign.__new__(TaskCampaign)
    campaign.plan = plan
    selected = []
    campaign.run_integrated_domain = lambda row: selected.append(row["name"])
    for row in plan["domains"]:
        campaign.run_domain(row)
    if selected != CHAIN:
        raise ValueError(f"Actual dispatch is not speed first / joint-only: {selected}")
    return selected


def correction_command(args, apply):
    return [sys.executable, "-B", str(args.new_source_root / "scripts/repair_known_integrated_origin.py"),
            "--contract", str(args.new_source_root / CONTRACT), "--run-dir", str(args.run_root / "train"),
            "--apply" if apply else "--check-only"]


def checked_command(command, log):
    with Path(log).open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise ValueError(f"CPU/check transaction command failed ({result.returncode}); see {log}")
    return dict(command=command, exit_code=0, log=str(log), log_sha256=sha(log))


def launch_evidence(args, plan):
    from wheeled_tasks.chassis import integrated_production as joint
    from wheeled_tasks.chassis.task_campaign import account_updates
    train = args.run_root / "train"
    evidence = joint.prepare_admission(plan, args.new_source_root, train)
    selection, allowance = evidence["selection"], evidence["allowance"]
    if (selection["domain"] != "speed" or selection["mode"] != "rebase_from_authorized_REC"
            or selection["checkpoint"]["sha256"] != EXPECTED_PARENT
            or selection["checkpoint_consumed_updates"] != 41393
            or selection["checkpoint_actor_updates_consumed"] != 41223
            or selection["origin_recovery_actor_update"] != 14188
            or selection["fresh_joint_actor_updates"] != 0 or selection["fresh_domain_actor_updates"] != 0
            or selection["critic_warmup_remaining"] != 10):
        raise ValueError("First speed entry would not preserve the latest full REC learning state")
    expected = dict(consumed_updates=41393, actor_updates_consumed=41223,
                    requested_total_updates=14907, requested_actor_updates=14897, charged_critic_updates=10)
    if any(allowance.get(key) != value for key, value in expected.items()) or not allowance["admitted"]:
        raise ValueError("First speed allowance loses charged consumption or exceeds its cap")
    if account_updates(plan, evidence["reconciled_attempts"]) != evidence["reconciled_account"]:
        raise ValueError("First speed allowance does not cover every existing attempt")
    # Verify the real worker-facing invocation contract without constructing a
    # runner, scene or a fictitious native attempt. prepare_admission is read-only.
    config = read(args.new_source_root / "contracts/v6_task_campaign_speed_worker_v26.json")
    row = next(row for row in plan["domains"] if row["name"] == "speed")
    receipt = joint.make_admission(plan, args.new_source_root, train, selection, evidence["reconciled_attempts"],
        account_path=evidence["account"]["file"], state_path=evidence["state"]["file"])
    # Admission-file authentication is also checked by the real native worker;
    # here authenticate_receipt verifies its account/source/physical allowance.
    joint.authenticate_receipt(config, receipt, args.new_source_root)
    if row["contract"] != "contracts/v6_task_campaign_speed_worker_v26.json":
        raise ValueError("First worker contract is not the frozen integrated speed worker")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage", "preflight", "correct", "launch"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--new-source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-script", type=Path,
                        default=Path("/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh"))
    args = parser.parse_args()
    for name in ("run_root", "new_source_root", "source_archive", "runtime_script"):
        path = getattr(args, name)
        if not path.is_absolute() or ".." in path.parts:
            parser.error(f"{name} must be an absolute confined path")
        setattr(args, name, path.resolve())
    if (not re.fullmatch(r"[0-9a-f]{40}", args.source_commit)
            or args.new_source_root.parent != args.run_root or args.source_archive.parent != args.run_root):
        parser.error("New source/archive must be immediate children of the owned run, with a 40hex commit")
    directory = args.run_root / "source029_deploy"
    if args.mode == "stage":
        job = read(args.run_root / "job.json")
        old = Path(job["command"][job["command"].index("--contract") + 1]).parents[1]
        stopped_original(args.run_root, old)
        if args.new_source_root == old or directory.exists():
            raise ValueError("Source029 staging already exists or aliases the frozen predecessor")
        count = extract_new(args.source_archive, args.new_source_root, args.source_commit)
        directory.mkdir()
        command = list(job["command"])
        command[command.index(str(old / "scripts/run_task_campaign.py"))] = str(args.new_source_root / "scripts/run_task_campaign.py")
        command[command.index("--contract") + 1] = str(args.new_source_root / CONTRACT)
        receipt = dict(schema="task-source-revision-v1", previous_job_sha256=sha(args.run_root / "job.json"),
            command=command, source_commit=args.source_commit, old_source_root=str(old),
            new_source_root=str(args.new_source_root), old_contract_sha256=sha(old / CONTRACT),
            new_contract_sha256=sha(args.new_source_root / CONTRACT), source_archive=str(args.source_archive),
            source_archive_sha256=sha(args.source_archive))
        if receipt["old_contract_sha256"] != receipt["new_contract_sha256"]:
            raise ValueError("Same-v26 source repair changed the campaign contract")
        durable(args.run_root / "source_revision_029.json", receipt)
        result = dict(schema="source029-staging-v1", source_commit=args.source_commit,
            source_archive_sha256=receipt["source_archive_sha256"], regular_files=count,
            helper_sha256=sha(__file__), receipt_sha256=sha(args.run_root / "source_revision_029.json"),
            qualification=None, native_scenes_created=0, PPO_updates=0)
        durable(directory / "stage.json", result)
    else:
        products = {"preflight": ("preflight.json", "preflight.log", "origin_check.log"),
                    "correct": ("correction_command.json", "origin_apply.log"),
                    "launch": ("admission_before_launch.json", "launch.intent.json", "launch.log", "launch.receipt.json")}
        prior = [name for name in products[args.mode] if (directory / name).exists()]
        if prior:
            raise ValueError(f"Prior {args.mode} products must be inspected, not overwritten or retried: {prior}")
        sys.path[:0] = [str(args.new_source_root / "scripts"), str(args.new_source_root / "src")]
        receipt = validate_staging(args, directory)
        plan = read(args.new_source_root / CONTRACT)
        selected = cpu_dispatch(plan)
        if args.mode == "preflight":
            design = checked_command([*receipt["command"], "--preflight-only"], directory / "preflight.log")
            origin = checked_command(correction_command(args, False), directory / "origin_check.log")
            result = dict(status="CPU_preflight_passed", actual_dispatch=selected, design=design,
                origin=origin, source_revision_sha256=sha(args.run_root / "source_revision_029.json"),
                native_scenes_created=0, PPO_updates=0, qualification=None)
            durable(directory / "preflight.json", result)
        elif args.mode == "correct":
            if read(directory / "preflight.json").get("status") != "CPU_preflight_passed":
                raise ValueError("Source029 CPU preflight has not passed")
            result = checked_command(correction_command(args, True), directory / "origin_apply.log")
            durable(directory / "correction_command.json", result)
        else:
            # Refuse before any additional mutation when submission is ambiguous.
            intent = directory / "launch.intent.json"
            if intent.exists() or (directory / "launch.log").exists():
                raise ValueError("Existing source029 launch intent/log; inspect instead of resubmitting")
            read(directory / "correction_command.json")
            from wheeled_tasks.chassis.task_campaign import CAMPAIGN_ID
            claim = Path(plan["budget_source_stage"]) / (CAMPAIGN_ID + ".account.json")
            with claim.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                no_owners(args.new_source_root, args.run_root)
                evidence = launch_evidence(args, plan)
                durable(directory / "admission_before_launch.json", evidence)
                command = ["/bin/bash", "-c", 'set -e; source "$1"; shift; exec "$@"',
                    "source029-code-only", str(args.runtime_script), sys.executable, "-B",
                    str(args.new_source_root / "scripts/chassis_remote_job.py"), "--run-root", str(args.run_root),
                    "--resume-existing", "--source-revision-receipt", str(args.run_root / "source_revision_029.json"),
                    "--", *receipt["command"]]
                durable(intent, dict(command=command, source_revision_sha256=sha(args.run_root / "source_revision_029.json"),
                    admission_sha256=sha(directory / "admission_before_launch.json"), qualification=None))
                # Release the account before the wrapper/supervisor acquires it.
                # Durable intent forbids a second launch while the lock changes hands.
                fcntl.flock(lock, fcntl.LOCK_UN)
                with (directory / "launch.log").open("x") as log:
                    child = subprocess.Popen(command, cwd=args.run_root, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                result = dict(status="launch_submitted", verification_status="first_actor_pending", pid=child.pid,
                    command=command, source_commit=args.source_commit, actual_dispatch=selected,
                    launched_at_utc=datetime.now(timezone.utc).isoformat(), qualification=None)
                durable(directory / "launch.receipt.json", result)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
