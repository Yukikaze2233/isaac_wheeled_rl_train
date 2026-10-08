#!/usr/bin/env python3
"""Explicit local-to-Kaiser deployment steps. No SSH, signals, polling or scheduler.

static-stage/preflight can run while source029 trains. boundary-prepare requires
the real completed-speed controller receipt and archived, exited old wrapper.
Only the separately invoked launch mode submits the existing native wrapper.
Every ambiguous mutation intent is retained and must be inspected, never retried.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from types import SimpleNamespace

CONTRACT = Path("contracts/v6_task_campaign_v27.json")
OLD_CONTRACT = Path("contracts/v6_task_campaign_v26.json")
ADDED_CONTRACTS = {str(CONTRACT), "contracts/v6_integrated_lineage_v27.json",
    "contracts/v6_integrated_runtime_recipe_v27.json", "contracts/v6_task_campaign_migration_v27.json"} | {
    f"contracts/v6_task_campaign_{domain}_{kind}_v27.json"
    for domain in ("speed", "terrain", "jump", "stepjump") for kind in ("worker", "expectations")}
REQUIRED = ADDED_CONTRACTS | {"scripts/run_task_campaign.py", "scripts/chassis_remote_job.py",
    "scripts/hold_completed_speed_boundary.py", "scripts/queue_task_campaign_continuation.py",
    "src/wheeled_tasks/chassis/integrated_completed_speed_revision.py"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path):
    return dict(file=str(Path(path).resolve(strict=True)), sha256=sha(path))


def flag(command, key):
    if command.count(key) != 1 or command.index(key) + 1 >= len(command):
        raise ValueError("Native command needs exactly one " + key)
    return command[command.index(key) + 1]


def load_native(source):
    """Each CLI mode uses a fresh interpreter; reject an already cached old tree."""
    source = Path(source).resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name.startswith("wheeled_tasks") or name in ("run_task_campaign", "run_chassis_blocks",
                "chassis_remote_job", "queue_task_campaign_continuation"):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(source):
                raise ValueError("Cached native module comes from another frozen source: " + name)
    sys.path[:0] = [str(source / "scripts"), str(source / "src")]
    from queue_task_campaign_continuation import durable_json, owned_processes
    from chassis_remote_job import validate_source_revision
    return SimpleNamespace(publish=lambda p, v: durable_json(p, v, exclusive=True),
        owners=owned_processes, validate_revision=validate_source_revision)


def archive_files(archive_path, commit, *, extract_to=None, verify_tree=None):
    """Reuse source029's strict Git archive extraction; retain per-file hashes."""
    files, seen = {}, set()
    with tarfile.open(archive_path, "r:*") as archive:
        if archive.pax_headers.get("comment") != commit:
            raise ValueError("Archive PAX commit differs")
        for member in archive:
            relative = Path(member.name)
            if (relative.is_absolute() or ".." in relative.parts or relative == Path(".")
                    or str(relative) in seen or not (member.isfile() or member.isdir())):
                raise ValueError("Unsafe, linked or duplicate source member")
            seen.add(str(relative))
            if member.mode not in ((0o755,) if member.isdir() else (0o644, 0o755)):
                raise ValueError("Archive permission differs from Git 0644/0755; use tar.umask=0022")
            target = extract_to / relative if extract_to else None
            if verify_tree is not None:
                actual = verify_tree / relative
                if (actual.is_symlink() or not actual.resolve().is_relative_to(verify_tree)
                        or actual.stat().st_mode & 0o7777 != member.mode):
                    raise ValueError("Staged source permission differs: " + str(relative))
            if member.isdir():
                if target:
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(member.mode)
                continue
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                if target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as output:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                            output.write(block)
                    target.chmod(member.mode & 0o777)
                else:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            files[str(relative)] = digest.hexdigest()
    if not REQUIRED <= files.keys():
        raise ValueError("Source030 archive lacks its native entrypoints or 12 contracts")
    return files


def contract_inventory(old, files):
    legacy = {str(p.relative_to(old)): sha(p) for p in (old / "contracts").rglob("*") if p.is_file()}
    current = {name: value for name, value in files.items() if name.startswith("contracts/")}
    if not legacy or current.keys() - legacy.keys() != ADDED_CONTRACTS:
        raise ValueError("Only the 12 named v27 contracts may be added")
    if any(current.get(name) != value for name, value in legacy.items()):
        raise ValueError("An old contract was changed or removed")


def stage(args):
    directory = args.run_root / "source030_deploy"
    if directory.exists() or args.new_source_root.exists():
        raise ValueError("Prior source030 staging exists; inspect, do not overwrite")
    if sha(args.source_archive) != args.source_archive_sha256:
        raise ValueError("Archive SHA differs from the explicit frozen input")
    directory.mkdir()  # Exclusive reservation; failed stage is not auto-retried.
    with tempfile.TemporaryDirectory(prefix="source030-extract-", dir=args.run_root) as temporary:
        extracted = Path(temporary) / "source"
        extracted.mkdir()
        files = archive_files(args.source_archive, args.source_commit, extract_to=extracted)
        contract_inventory(args.old_source_root, files)
        extracted.rename(args.new_source_root)
    native = load_native(args.new_source_root)
    result = dict(schema="source030-static-stage-v1", status="staged_only", source_commit=args.source_commit,
        source_archive=pin(args.source_archive), old_source_root=str(args.old_source_root),
        new_source_root=str(args.new_source_root), files=files, helper=pin(__file__),
        dynamic_parent_bound=False, PPO_updates=0, qualification=None)
    native.publish(directory / "stage.json", result)
    return result


def validate_stage(args):
    directory = args.run_root / "source030_deploy"
    saved = read(directory / "stage.json")
    if (saved.get("schema") != "source030-static-stage-v1" or saved.get("source_commit") != args.source_commit
            or saved.get("source_archive") != pin(args.source_archive)
            or saved["source_archive"]["sha256"] != args.source_archive_sha256
            or saved.get("new_source_root") != str(args.new_source_root)
            or saved.get("old_source_root") != str(args.old_source_root) or saved.get("helper") != pin(__file__)):
        raise ValueError("CLI or helper differs from the immutable stage identity")
    files = archive_files(args.source_archive, args.source_commit, verify_tree=args.new_source_root)
    if saved["files"] != files:
        raise ValueError("Stage file inventory differs")
    for name, digest in files.items():
        path = args.new_source_root / name
        if path.is_symlink() or not path.resolve().is_relative_to(args.new_source_root) or sha(path) != digest:
            raise ValueError("Staged source differs: " + name)
    contract_inventory(args.old_source_root, files)
    return directory, load_native(args.new_source_root)


@contextmanager
def boundary_locks(train, claim):
    # Same order as the existing controller. Never wait while blocking an owner.
    with (train / "native_speed_boundary_controller.lock").open("a") as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with claim.with_suffix(".lock").open("a") as account:
            fcntl.flock(account, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield account


def closed_boundary(args, native):
    """Validate the actual receipt and immutable archive, not old live ledger SHA."""
    root, train = args.run_root, args.run_root / "train"
    if native.owners(root):
        raise ValueError("Owned wrapper/supervisor/worker/evaluator still alive")
    if args.controller_intent is None or args.controller_receipt_sha256 is None:
        raise ValueError("Actual controller intent and successful receipt SHA are required")
    intent_path = args.controller_intent
    receipt_path = intent_path.with_suffix(".receipt.json")
    if sha(receipt_path) != args.controller_receipt_sha256:
        raise ValueError("Controller receipt SHA differs")
    intent, receipt, job = read(intent_path), read(receipt_path), read(root / "job.json")
    supervisor, worker = intent["supervisor"], intent["worker"]
    command = job["command"]
    if (intent.get("schema") != "native-completed-speed-supervisor-hold-v1"
            or intent.get("operation") != "hold_supervisor_until_normal_speed_end"
            or intent.get("worker_signals") != 0 or intent.get("qualification") is not None
            or supervisor["pid"] != job["pid"] or supervisor["command"] != command
            or worker["ppid"] != supervisor["pid"]
            or command.count(str(args.old_source_root / "scripts/run_task_campaign.py")) != 1
            or flag(command, "--contract") != str(args.old_source_root / OLD_CONTRACT)
            or flag(command, "--run-dir") != str(train)
            or receipt.get("status") != "native_supervisor_stopped_after_normal_speed"
            or receipt.get("worker_signals") != 0 or receipt.get("next_domain_dispatched") is not False
            or receipt.get("qualification") is not None):
        raise ValueError("Controller does not certify this exact old source029 job")
    closure, evidence = receipt["native_completion_recovery"], receipt["evidence"]
    selected, reference = evidence["selection"], evidence["checkpoint"]
    if (closure.get("status") != "closed_native_completed_speed" or closure.get("PPO_updates") != 0
            or closure.get("next_domain_dispatched") is not False or closure.get("qualification") is not None
            or closure.get("checkpoint") != reference or selected.get("checkpoint") != reference
            or evidence.get("PPO_updates_by_controller") != 0 or selected.get("domain") != "speed"
            or selected.get("domain_complete") is not True or selected.get("critic_warmup_remaining") != 0):
        raise ValueError("Controller lacks a normal completed speed and zero-PPO closure")
    checkpoint = Path(reference["path"])
    worker_dir = Path(flag(worker["command"], "--run-dir"))
    if (checkpoint != worker_dir / "model_final.pt" or not checkpoint.resolve().is_relative_to(train / "speed")
            or sha(checkpoint) != reference["sha256"]):
        raise ValueError("Controller checkpoint differs from the actual speed worker")
    state = read(train / "campaign_state.json")
    record = state["domains"]["speed"]
    if (record.get("status") != "joint_domain_finished_unassessed"
            or record.get("candidate_checkpoint") != str(checkpoint)
            or record.get("candidate_checkpoint_sha256") != reference["sha256"]
            or any(state.get(key) != value for key, value in closure["account"].items())):
        raise ValueError("Current account clocks or normal speed record differ from closure")
    delivery, archiver = read(root / "delivery.json"), read(root / "archiver_status.json")
    archive = root / "delivery.tar.gz"
    if (delivery.get("status") != "ready" or delivery.get("archive") != archive.name
            or (root / "delivery.tar.gz.tmp").exists() or (root / "delivery.tmp").exists()
            or archiver.get("status") != "finished" or archiver.get("pid") != job["pid"]
            or delivery.get("exit_code") != 0 or delivery.get("archive_bytes") != archive.stat().st_size
            or delivery.get("archive_sha256") != sha(archive)):
        raise ValueError("Old wrapper did not finish its current immutable delivery archive")
    paths = [checkpoint, checkpoint.with_suffix(".checkpoint.json"), checkpoint.with_suffix(".contract.json"),
        checkpoint.parent / "completion.json", root / "job.json"]
    expected = {}
    for path in paths:
        name = str(path.relative_to(root))
        item = delivery["files"].get(name)
        if item != dict(size=path.stat().st_size, sha256=sha(path)):
            raise ValueError("Delivery lacks the actual immutable speed artifact: " + name)
        expected[name] = item["sha256"]
    found = {}
    with tarfile.open(archive, "r|*") as sealed:
        for member in sealed:
            if member.name in expected:
                if not member.isfile() or member.name in found:
                    raise ValueError("Ambiguous immutable delivery member")
                with sealed.extractfile(member) as content:
                    found[member.name] = hashlib.file_digest(content, "sha256").hexdigest()
    if found != expected:
        raise ValueError("Delivery archive bytes do not contain the current speed final")
    return dict(job=job, job_pin=pin(root / "job.json"), checkpoint=reference,
        controller_intent=pin(intent_path), controller_receipt=pin(receipt_path),
        delivery_manifest=pin(root / "delivery.json"), delivery_archive=pin(archive),
        archiver=pin(root / "archiver_status.json"), immutable_speed_files=expected,
        closed_account=closure["account"])


def revision_receipt(args, boundary):
    command = list(boundary["job"]["command"])
    command[command.index(str(args.old_source_root / "scripts/run_task_campaign.py"))] = str(
        args.new_source_root / "scripts/run_task_campaign.py")
    command[command.index("--contract") + 1] = str(args.new_source_root / CONTRACT)
    return dict(schema="task-plan-revision-v27", previous_job_sha256=boundary["job_pin"]["sha256"],
        command=command, source_commit=args.source_commit, old_source_root=str(args.old_source_root),
        new_source_root=str(args.new_source_root), new_contract_relative_path=str(CONTRACT),
        old_contract_sha256=sha(args.old_source_root / OLD_CONTRACT),
        new_contract_sha256=sha(args.new_source_root / CONTRACT), source_archive=str(args.source_archive),
        source_archive_sha256=args.source_archive_sha256)


def native_admission(args, boundary, snapshot):
    """Use real native account migration, source selection and receipt authentication."""
    from run_task_campaign import TaskCampaign
    from wheeled_tasks.chassis import integrated_production as joint
    job = TaskCampaign(SimpleNamespace(contract=args.new_source_root / CONTRACT,
        run_dir=args.run_root / "train", max_runtime_seconds=600., device="cpu"))
    job.open_account()
    row = next(r for r in job.plan["domains"] if r["name"] == "terrain")
    job.learning_source(row)
    receipt = joint.make_admission(job.plan, args.new_source_root, job.root, job._joint_selection,
        job.records, account_path=job.claim, state_path=job.root / "campaign_state.json",
        snapshot_directory=snapshot)
    joint.authenticate_receipt(read(args.new_source_root / row["contract"]), receipt, args.new_source_root)
    selection, allowance = receipt["selection"], receipt["allowance"]
    if (selection["domain"] != "terrain" or selection["checkpoint"] != boundary["checkpoint"]
            or selection["fresh_domain_actor_updates"] != 0 or selection["critic_warmup_remaining"] != 10
            or not allowance["admitted"] or allowance["charged_critic_updates"] != 10
            or receipt["reconciled_account"] != boundary["closed_account"]):
        raise ValueError("Initial terrain admission changed the paid parent or critic/actor origin")
    return receipt


def no_prior(directory, names):
    prior = [name for name in names if (directory / name).exists()]
    if prior:
        raise ValueError("Prior products require inspection; never overwrite/retry: " + str(prior))


def prepare(args, directory, native, claim):
    with boundary_locks(args.run_root / "train", claim):
        no_prior(directory, ("prepare.intent.json", "prepare.json", "admission.json", "launch.intent.json"))
        if (args.run_root / "source_revision_030.json").exists():
            raise ValueError("Prior dynamic source revision exists")
        if read(directory / "preflight.json")["status"] != "CPU_preflight_passed":
            raise ValueError("Static CPU preflight is missing")
        boundary = closed_boundary(args, native)
        revision = revision_receipt(args, boundary)
        # Validate the exact future wrapper input without prematurely publishing it.
        with tempfile.TemporaryDirectory(prefix="source030-revision-check-", dir=directory) as tmp:
            temporary = Path(tmp) / "revision.json"
            temporary.write_text(json.dumps(revision))
            native.validate_revision(args.run_root, revision["command"], boundary["job"], temporary)
        native.publish(directory / "prepare.intent.json", dict(boundary=boundary, revision=revision,
            migration="native_completed_speed_to_v27", PPO_updates=0, qualification=None))
        snapshot = args.run_root / "train/revisions/source030_deploy_admission"
        if snapshot.exists():
            raise ValueError("Existing admission snapshots must be inspected")
        admission = native_admission(args, boundary, snapshot)
        native.publish(directory / "admission.json", admission)
        # Receipt is only published after actual normal boundary and native migration.
        native.publish(args.run_root / "source_revision_030.json", revision)
        result = dict(status="boundary_prepared_not_launched", boundary=boundary,
            admission=pin(directory / "admission.json"), source_revision=pin(args.run_root / "source_revision_030.json"),
            account=pin(claim), state=pin(args.run_root / "train/campaign_state.json"),
            PPO_updates=0, qualification=None)
        native.publish(directory / "prepare.json", result)
        return result


def launch(args, directory, native, claim):
    with boundary_locks(args.run_root / "train", claim) as account_lock:
        no_prior(directory, ("launch.intent.json", "launch.log", "launch.receipt.json"))
        saved = read(directory / "prepare.json")
        if saved.get("status") != "boundary_prepared_not_launched":
            raise ValueError("Completed boundary preparation is required")
        for item in (saved["admission"], saved["source_revision"], saved["account"], saved["state"]):
            if pin(item["file"]) != item:
                raise ValueError("Prepared source/account/admission changed; do not reuse it")
        if closed_boundary(args, native) != saved["boundary"]:
            raise ValueError("Closed boundary changed before launch")
        revision = read(saved["source_revision"]["file"])
        native.validate_revision(args.run_root, revision["command"], saved["boundary"]["job"],
            Path(saved["source_revision"]["file"]))
        from wheeled_tasks.chassis.integrated_production import authenticate_receipt
        authenticate_receipt(read(args.new_source_root / "contracts/v6_task_campaign_terrain_worker_v27.json"),
            read(saved["admission"]["file"]), args.new_source_root)
        command = ["/bin/bash", "-c", 'set -e; source "$1"; shift; exec "$@"', "source030-native",
            str(args.runtime_script), sys.executable, "-B", str(args.new_source_root / "scripts/chassis_remote_job.py"),
            "--run-root", str(args.run_root), "--resume-existing", "--source-revision-receipt",
            str(args.run_root / "source_revision_030.json"), "--", *revision["command"]]
        native.publish(directory / "launch.intent.json", dict(command=command, prepare=pin(directory / "prepare.json"),
            runtime=pin(args.runtime_script), qualification=None))
        # Do not deadlock the native supervisor. Controller lock and durable intent
        # still prevent a second deployment while account ownership changes hands.
        fcntl.flock(account_lock, fcntl.LOCK_UN)
        with (directory / "launch.log").open("x") as log:
            child = subprocess.Popen(command, cwd=args.run_root, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        result = dict(status="launch_submitted", verification_status="first_terrain_actor_pending",
            pid=child.pid, command=command, source_commit=args.source_commit,
            launched_at_utc=datetime.now(timezone.utc).isoformat(), qualification=None)
        native.publish(directory / "launch.receipt.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("static-stage", "static-preflight", "boundary-prepare", "launch"))
    for name in ("run-root", "old-source-root", "new-source-root", "source-archive"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--controller-intent", type=Path)
    parser.add_argument("--controller-receipt-sha256")
    parser.add_argument("--runtime-script", type=Path, default=Path("/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh"))
    args = parser.parse_args()
    for key in ("run_root", "old_source_root", "new_source_root", "source_archive", "runtime_script", "controller_intent"):
        path = getattr(args, key)
        if path is None:
            continue
        if not path.is_absolute() or ".." in path.parts or path.is_symlink():
            parser.error(key + " must be an absolute non-symlink confined path")
        setattr(args, key, path.resolve())
    if (not re.fullmatch(r"[0-9a-f]{40}", args.source_commit)
            or not re.fullmatch(r"[0-9a-f]{64}", args.source_archive_sha256)
            or args.new_source_root.parent != args.run_root or args.source_archive.parent != args.run_root
            or args.old_source_root == args.new_source_root
            or args.controller_intent is not None and not args.controller_intent.is_relative_to(args.run_root)):
        parser.error("Frozen SHA/path or owned run boundary differs")
    if args.mode == "static-stage":
        result = stage(args)
    else:
        directory, native = validate_stage(args)
        plan = read(args.new_source_root / CONTRACT)
        if args.mode == "static-preflight":
            no_prior(directory, ("preflight.log", "preflight.json"))
            command = [sys.executable, "-B", str(args.new_source_root / "scripts/run_task_campaign.py"),
                "--contract", str(args.new_source_root / CONTRACT), "--run-dir", str(args.run_root / "train"),
                "--num-envs", str(plan["target_num_envs"]), "--research", "--preflight-only", "--device", "cpu"]
            with (directory / "preflight.log").open("x") as log:
                outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=600)
            if outcome.returncode:
                raise ValueError("Static CPU preflight failed; inspect retained log")
            result = dict(status="CPU_preflight_passed", command=command, log=pin(directory / "preflight.log"),
                stage=pin(directory / "stage.json"), PPO_updates=0, qualification=None)
            native.publish(directory / "preflight.json", result)
        else:
            from wheeled_tasks.chassis.task_campaign import CAMPAIGN_ID
            claim = Path(plan["budget_source_stage"]) / (CAMPAIGN_ID + ".account.json")
            result = (prepare if args.mode == "boundary-prepare" else launch)(args, directory, native, claim)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
