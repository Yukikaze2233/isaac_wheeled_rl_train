#!/usr/bin/env python3
"""Cancel one authenticated waiting observer, never an original training owner.

Default is read-only. --apply persists one exclusive intent before PIDFD TERM
to the known controller child, then observes its parent exit naturally. No
retry, subprocess, timeout kill, supervisor signal, PPO or account mutation.
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
import sys
import time
from types import ModuleType

ROOT = Path("/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358")
NEW_COMMIT = "ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a"
NEW_ARCHIVE_SHA = "0b942599d452dc17b9a710349262fcd90218a9c427d3e07df85a55ecdfde59f6"
NEW_HELPER_SHA = "663efe61f1b75ced98cffe5077a5568069b9ee9b7dd03a333ff1dd5dfd4b94e9"
NEW_STAGE_SHA = "b7dae15f3878aa92039580387bc98e526078fc4f617064a35c18da8994f93828"
NEW_PREFLIGHT_SHA = "9c14b08e90dc4eb7a8f3933ba29035e025b3b36083cae37de64002445bc9a83f"
NEW_FILE_COUNT = 4343
OLD_QUEUE_SHA = "0478bfe38c673e7fdb9ad1a5a6fb6151c76848e2a4182714b8910c8c357eb339"
OLD_INTENT_SHA = "8bd3843f4ca8360a60c2af7f66c20a9ad58696f2c5c95857adbb40afba5d34c1"
SCHEMA = "source031-cancel-waiting-source030-v1"
PINS = dict(coordinator=(4166067, 114488334), controller=(4166078, 114488781),
            wrapper=(4141407, 113423130), supervisor=(4141412, 113423416), worker=(4141576, 113428247))
PRODUCTS = ("source030_speed_boundary.json", "source030_speed_boundary.receipt.json", "source_revision_030.json",
    "source030_deploy/prepare.intent.json", "source030_deploy/prepare.json", "source030_deploy/admission.json",
    "source030_deploy/launch.intent.json", "source030_deploy/launch.receipt.json",
    "source030_queue/prepare.started.json", "source030_queue/prepare.execution.json", "source030_queue/prepare.log",
    "source030_queue/launch.started.json", "source030_queue/launch.execution.json", "source030_queue/launch.log",
    "source030_queue/old_delivery.json", "source030_queue/queue.receipt.json",
    "train/revisions/source030_deploy_admission", "train/revisions/v26_completed_speed_to_v27")


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a non-symlink existing file: " + str(path))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(file=str(path.resolve(strict=True)), sha256=digest)


def load(name, path):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def products_absent(root):
    if any((root / name).exists() or (root / name).is_symlink() for name in PRODUCTS):
        raise ValueError("Source030 already produced a transition artifact; waiting-only cancellation is unsafe")


def new_stage(args):
    root = Path(args.run_root)
    file = root / "source031_deploy/stage.json"
    preflight_file = root / "source031_deploy/preflight.json"
    if (args.new_stage_sha256 != NEW_STAGE_SHA or args.new_preflight_sha256 != NEW_PREFLIGHT_SHA
            or pin(file)["sha256"] != args.new_stage_sha256
            or pin(preflight_file)["sha256"] != args.new_preflight_sha256):
        raise ValueError("Actual source031 stage/preflight SHA differs")
    stage, preflight = read(file), read(preflight_file)
    expected = dict(schema="source031-static-stage-v1", status="staged_only", source_commit=NEW_COMMIT,
        old_source_root=str(root / "source_serial_dispatch029_20261009"),
        new_source_root=str(root / "source_serial_dispatch031_20261009"),
        dynamic_parent_bound=False, PPO_updates=0, qualification=None)
    if any(stage.get(key) != value for key, value in expected.items()) or len(stage.get("files", {})) != NEW_FILE_COUNT:
        raise ValueError("Source031 is not the frozen unbound staged candidate")
    archive = root / ("source031-" + NEW_COMMIT + "-umask0022.tar.gz")
    helper = root / "deploy_source031_663efe61.py"
    if (stage.get("source_archive") != pin(archive) or pin(archive)["sha256"] != NEW_ARCHIVE_SHA
            or stage.get("helper") != pin(helper) or pin(helper)["sha256"] != NEW_HELPER_SHA):
        raise ValueError("Source031 archive/helper differs")
    log = root / "source031_deploy/preflight.log"
    if (preflight.get("status") != "CPU_preflight_passed" or preflight.get("stage") != pin(file)
            or preflight.get("log") != pin(log) or preflight.get("PPO_updates") != 0
            or preflight.get("qualification") is not None or read(log) != dict(status="passed", roles=30)):
        raise ValueError("Source031 authenticated 30-role zero-PPO preflight is required")
    helper_module = load("cancel_verified_source031_deployer", helper)
    files = helper_module.archive_files(archive, NEW_COMMIT, verify_tree=Path(stage["new_source_root"]))
    if files != stage["files"]:
        raise ValueError("Source031 staged inventory differs")
    for name, digest in files.items():
        if pin(Path(stage["new_source_root"]) / name)["sha256"] != digest:
            raise ValueError("Source031 staged source differs: " + name)
    return dict(stage=pin(file), preflight=pin(preflight_file), archive=pin(archive), helper=pin(helper))


def native_modules(root):
    queue_file = root / "coordinate_source030_0478bfe3.py"
    if pin(queue_file)["sha256"] != OLD_QUEUE_SHA:
        raise ValueError("Actual waiting coordinator code differs")
    queue = load("cancel_verified_source030_queue", queue_file)
    admitted = queue.inputs(argparse.Namespace(run_root=root,
        runtime_script=Path("/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh")))
    source = admitted["new"]
    sys.path[:0] = [str(source / "scripts"), str(source / "src")]
    boundary = load("cancel_verified_source030_boundary", Path(admitted["controller"]["file"]))
    from queue_task_campaign_boundary_continuation import open_pidfd, send_pidfd_sigterm
    return queue, admitted, boundary, open_pidfd, send_pidfd_sigterm


def checked_process(boundary, role):
    pid, start = PINS[role]
    actual = boundary.process_identity(pid)
    if (actual is None or actual["starttime"] != start
            or boundary._state(pid) in ("Z", "T", "t")):
        raise ValueError("Original " + role + " identity/state differs")
    return actual


def lock_owner(path):
    info = Path(path).stat()
    result = []
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        major, minor, inode = fields[5].split(":")
        if (int(major, 16), int(minor, 16), int(inode)) == (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino):
            result.append(int(fields[4]))
    return result


def account_unmigrated(owners, root, old):
    claim = Path(owners["plan"]["budget_source_stage"]) / "v6-task-domains-20261001.account.json"
    account, state = read(claim), read(root / "train/campaign_state.json")
    source_sha = pin(old / "contracts/v6_task_campaign_v26.json")["sha256"]
    if account.get("contract_sha256") != source_sha or state.get("contract_sha256") != source_sha:
        raise ValueError("Current claim/state has already migrated from the original source029 plan")
    clocks = {key: account.get(key) for key in ("consumed_updates", "actor_updates_consumed")}
    if any(type(value) is not int or value < 0 for value in clocks.values()):
        raise ValueError("Current paid account clocks are invalid")
    return dict(account=pin(claim), clocks=clocks)


def guard(args, modules):
    root = Path(args.run_root)
    if root != ROOT or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("Cancellation is restricted to the actual named experiment")
    products_absent(root)
    directory = root / "source031_queue"
    if any((directory / name).exists() for name in ("cancel_source030.intent.json", "cancel_source030.receipt.json")):
        raise ValueError("A prior cancellation already exists; never retry")
    new = new_stage(args)
    queue, admitted, boundary, _, _ = modules
    queue.unchanged(admitted)
    intent_file = root / "source030_queue/queue.intent.json"
    if pin(intent_file)["sha256"] != OLD_INTENT_SHA:
        raise ValueError("Original immutable queue intent differs")
    old = read(intent_file)
    identity = {key: value for key, value in admitted.items()
                if key not in ("root", "old", "new", "archive", "helper", "runtime")}
    if (old.get("inputs") != identity or old.get("coordinator") != pin(root / "coordinate_source030_0478bfe3.py")
            or old.get("schema") != queue.SCHEMA or old.get("source_commit") != queue.COMMIT
            or old.get("controller_wait_seconds") != 604800 or old.get("retries") != 0):
        raise ValueError("Waiting queue provenance differs")
    status = read(root / "source030_queue/status.json")
    if status != dict(status="waiting_for_normal_speed_boundary", phase="controller", intent=pin(intent_file), qualification=None):
        raise ValueError("Old queue is not waiting in its untouched controller phase")
    coordinator, controller = (checked_process(boundary, key) for key in ("coordinator", "controller"))
    expected_coordinator = [sys.executable, "-B", str(root / "coordinate_source030_0478bfe3.py"),
                            "--run-root", str(root), "--apply"]
    expected_controller = [sys.executable, "-B", admitted["controller"]["file"],
        "--source-root", str(admitted["old"]), "--run-root", str(root / "train"),
        "--intent", str(root / "source030_speed_boundary.json"), "--apply", "--timeout", "604800"]
    if (coordinator["command"] != expected_coordinator or controller["command"] != expected_controller
            or controller["ppid"] != coordinator["pid"]
            or read(root / "source030_queue/controller.started.json") != dict(command=expected_controller, pid=controller["pid"])):
        raise ValueError("Waiting controller argv/parent/start receipt differs")
    owners = boundary.admission(admitted["old"], root / "train")
    account = account_unmigrated(owners, root, admitted["old"])
    training = {key: checked_process(boundary, key) for key in ("wrapper", "supervisor", "worker")}
    if any(training[key] != owners[key] for key in training):
        raise ValueError("Original training owner differs from native admission")
    progress = read(owners["directory"] / "progress.json")
    source = owners["receipt"]["selection"]["fresh_domain_actor_updates"]
    target = source + owners["receipt"]["allowance"]["requested_actor_updates"]
    actor = progress.get("task_actor_update")
    if type(actor) is not int or not source <= actor < target or target - actor <= 1000:
        raise ValueError("Speed is not more than 1000 actor updates away from the normal boundary")
    lock = root / "train/native_speed_boundary_controller.lock"
    if lock_owner(lock) != [controller["pid"]]:
        raise ValueError("Actual controller does not exclusively own its native boundary lock")
    products_absent(root)
    return dict(new_source031=new, old_queue_intent=pin(intent_file), old_coordinator=coordinator,
        old_controller=controller, original_training=training, remaining_speed_actor_updates=target-actor,
        controller_lock=str(lock), account_before=account, qualification=None)


def cancel(args, *, modules=None, sleep=time.sleep, now=time.monotonic):
    modules = native_modules(Path(args.run_root)) if modules is None else modules
    snapshot = guard(args, modules)
    if not args.apply:
        return dict(schema=SCHEMA, status="validated_waiting_source030_not_cancelled", **snapshot,
                    training_signals=0, PPO_updates=0, refunds=0)
    root, directory = Path(args.run_root), Path(args.run_root) / "source031_queue"
    directory.mkdir(exist_ok=True)
    queue, admitted, boundary, open_fd, signal_fd = modules
    publish = queue.native_publish(Path(admitted["publisher"]["file"]))
    intent_file, receipt_file = directory / "cancel_source030.intent.json", directory / "cancel_source030.receipt.json"
    with (directory / "cancel_source030.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = guard(args, modules)
        publish(intent_file, dict(schema=SCHEMA, run_root=str(root), snapshot=snapshot,
            signal="one_PIDFD_SIGTERM_waiting_controller_only", helper=pin(__file__),
            at_utc=datetime.now(timezone.utc).isoformat(), qualification=None), exclusive=True)
        intent_pin = pin(intent_file)
        descriptor, sent = None, False
        try:
            descriptor = open_fd(snapshot["old_controller"]["pid"])
            # Intent now exists, so recheck real safety directly without admitting
            # another transaction. Neither process reuse nor a status transition
            # between opening PIDFD and sending TERM can redirect this operation.
            products_absent(root)
            if (checked_process(boundary, "controller") != snapshot["old_controller"]
                    or checked_process(boundary, "coordinator") != snapshot["old_coordinator"]
                    or read(root / "source030_queue/status.json")["status"] != "waiting_for_normal_speed_boundary"):
                raise ValueError("Waiting identity/phase changed after PIDFD open")
            for role, expected in snapshot["original_training"].items():
                if checked_process(boundary, role) != expected:
                    raise ValueError("Original training owner changed after PIDFD open")
            # Fresh native admission, work, paid-plan and staged provenance are
            # revalidated after PIDFD open, without reopening this transaction.
            owners = boundary.admission(admitted["old"], root / "train")
            actor = read(owners["directory"] / "progress.json")["task_actor_update"]
            source = owners["receipt"]["selection"]["fresh_domain_actor_updates"]
            target = source + owners["receipt"]["allowance"]["requested_actor_updates"]
            account_unmigrated(owners, root, admitted["old"])
            if (type(actor) is not int or not source <= actor < target or target - actor <= 1000
                    or lock_owner(snapshot["controller_lock"]) != [PINS["controller"][0]]):
                raise ValueError("Native boundary work/lock became unsafe after PIDFD open")
            queue.unchanged(admitted)
            if new_stage(args) != snapshot["new_source031"]:
                raise ValueError("Source031 changed after PIDFD open")
            products_absent(root)
            if (pin(intent_file) != intent_pin or pin(root / "source030_queue/queue.intent.json") != snapshot["old_queue_intent"]
                    or read(root / "source030_queue/status.json") != dict(status="waiting_for_normal_speed_boundary",
                        phase="controller", intent=snapshot["old_queue_intent"], qualification=None)):
                raise ValueError("Intent/phase changed immediately before TERM")
            for role in ("controller", "coordinator", "wrapper", "supervisor", "worker"):
                expected = snapshot["old_" + role] if role in ("controller", "coordinator") else snapshot["original_training"][role]
                if checked_process(boundary, role) != expected:
                    raise ValueError(role + " changed immediately before TERM")
            signal_fd(descriptor)
            sent = True
            deadline = now() + 60.
            while any(boundary.process_identity(PINS[key][0]) is not None for key in ("controller", "coordinator")):
                products_absent(root)
                if now() >= deadline:
                    raise TimeoutError("Waiting observers have not naturally exited; no additional signal or retry")
                sleep(.2)
            status = read(root / "source030_queue/status.json")
            execution = read(root / "source030_queue/controller.execution.json")
            if (status.get("status") != "attention_no_retry" or status.get("phase") != "controller"
                    or execution.get("pid") != PINS["controller"][0] or execution.get("exit_code") != -15
                    or execution.get("command") != snapshot["old_controller"]["command"]):
                raise ValueError("Old coordinator did not record its natural controller failure and exit")
            with Path(snapshot["controller_lock"]).open("r") as controller_lock:
                fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for role, expected in snapshot["original_training"].items():
                if checked_process(boundary, role) != expected:
                    raise ValueError("Original training owner changed during observer cancellation")
            products_absent(root)
            queue.unchanged(admitted)
            account_after = account_unmigrated(owners, root, admitted["old"])
            if any(account_after["clocks"][key] < value for key, value in snapshot["account_before"]["clocks"].items()):
                raise ValueError("Original paid account regressed during observer cancellation")
            result = dict(schema=SCHEMA, status="waiting_source030_cancelled_without_training_signal", run_root=str(root),
                intent=pin(intent_file), old_queue_intent=snapshot["old_queue_intent"],
                old_controller=dict(snapshot["old_controller"], exited=True),
                old_coordinator=dict(snapshot["old_coordinator"], exited=True),
                controller_lock_released=True, old_transition_products_absent=True,
                original_training=snapshot["original_training"], new_source031=snapshot["new_source031"],
                account_before=snapshot["account_before"], account_after=account_after,
                old_status=pin(root / "source030_queue/status.json"), old_execution=pin(root / "source030_queue/controller.execution.json"),
                waiting_controller_signals=1, training_signals=0, worker_signals=0, PPO_updates=0, refunds=0, qualification=None)
            publish(receipt_file, result, exclusive=True)
            return result
        except BaseException as error:
            publish(receipt_file, dict(schema=SCHEMA, status="attention_no_retry", intent=pin(intent_file),
                waiting_controller_signal_submitted=sent, error=type(error).__name__+": "+str(error),
                training_signals=0, PPO_updates=0, refunds=0, qualification=None), exclusive=True)
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--new-stage-sha256", default=NEW_STAGE_SHA)
    parser.add_argument("--new-preflight-sha256", default=NEW_PREFLIGHT_SHA)
    parser.add_argument("--apply", action="store_true")
    print(json.dumps(cancel(parser.parse_args()), allow_nan=False))


if __name__ == "__main__":
    main()
