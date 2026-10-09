#!/usr/bin/env python3
"""Once-only, external source029 normal-speed-end -> source031 coordinator.

No SSH, simulation, polling scheduler, kill or signal API. The already frozen
controller owns its explicit seven-day wait and exact supervisor-only signals.
Failures preserve evidence and never retry. Default is read-only; --apply is
required to start the future controller and subsequent native deployment CLI.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = "ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a"
ARCHIVE_SHA = "0b942599d452dc17b9a710349262fcd90218a9c427d3e07df85a55ecdfde59f6"
HELPER_SHA = "663efe61f1b75ced98cffe5077a5568069b9ee9b7dd03a333ff1dd5dfd4b94e9"
STAGE_SHA = "b7dae15f3878aa92039580387bc98e526078fc4f617064a35c18da8994f93828"
STAGE_FILE_COUNT = 4343
PREFLIGHT_SHA = "9c14b08e90dc4eb7a8f3933ba29035e025b3b36083cae37de64002445bc9a83f"
RUNTIME_SHA = "c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7"
WAIT_SECONDS = 604800
SCHEMA = "source031-once-normal-speed-boundary-queue-v1"
CONTROLLER = "scripts/hold_completed_speed_boundary.py"
UTILITY = "scripts/queue_task_campaign_continuation.py"
PROCESS_UTILITY = "scripts/queue_task_campaign_boundary_continuation.py"
CANCEL_SCHEMA = "source031-cancel-waiting-source030-v1"


def native_process_reader(source):
    """Reuse the authenticated native PID/starttime reader without import overlays."""
    utility_spec = importlib.util.spec_from_file_location("source031_superseded_utility", source / UTILITY)
    utility = importlib.util.module_from_spec(utility_spec)
    utility_spec.loader.exec_module(utility)
    prior = sys.modules.get("queue_task_campaign_continuation")
    sys.modules["queue_task_campaign_continuation"] = utility
    try:
        spec = importlib.util.spec_from_file_location("source031_superseded_process", source / PROCESS_UTILITY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            sys.modules.pop("queue_task_campaign_continuation", None)
        else:
            sys.modules["queue_task_campaign_continuation"] = prior
    return module.process_identity, utility.actual_entry


def superseded_waiter(root, new, *, expected=None, prepared=None):
    """Only replace a cancelled waiting observer; recheck live artifacts after waiting."""
    receipt_file = root / "source031_queue/cancel_source030.receipt.json"
    receipt_pin = pin(receipt_file)
    receipt = read(receipt_file)
    intent_file = root / "source031_queue/cancel_source030.intent.json"
    old_intent_file = root / "source030_queue/queue.intent.json"
    intent_pin, old_intent_pin = pin(intent_file), pin(old_intent_file)
    if (receipt.get("schema") != CANCEL_SCHEMA
            or receipt.get("status") != "waiting_source030_cancelled_without_training_signal"
            or receipt.get("run_root") != str(root)
            or receipt.get("intent") != intent_pin or receipt.get("old_queue_intent") != old_intent_pin
            or receipt.get("controller_lock_released") is not True
            or receipt.get("old_transition_products_absent") is not True
            or type(receipt.get("waiting_controller_signals")) is not int
            or receipt["waiting_controller_signals"] != 1
            or receipt.get("qualification") is not None
            or any(type(receipt.get(key)) is not int or receipt[key] != 0
                   for key in ("worker_signals", "training_signals", "PPO_updates", "refunds"))):
        raise ValueError("Source030 waiting-only cancellation receipt is missing or invalid")
    intent, old_intent = read(intent_file), read(old_intent_file)
    snapshot = intent.get("snapshot", {})
    status = read(root / "source030_queue/status.json")
    if (intent.get("schema") != CANCEL_SCHEMA or intent.get("run_root") != str(root)
            or intent.get("signal") != "one_PIDFD_SIGTERM_waiting_controller_only"
            or snapshot.get("old_queue_intent") != old_intent_pin
            or old_intent.get("schema") != "source030-once-normal-speed-boundary-queue-v1"
            or old_intent.get("source_commit") != "81ad124554c4ab45cc274e2d482d46c76ba125b0"
            or status.get("status") != "attention_no_retry" or status.get("phase") != "controller"):
        raise ValueError("Source030 queue was not retired from its original waiting phase")
    new_pins = dict(stage=pin(root / "source031_deploy/stage.json"),
        preflight=pin(root / "source031_deploy/preflight.json"),
        archive=pin(root / ("source031-" + COMMIT + "-umask0022.tar.gz")),
        helper=pin(root / "deploy_source031_663efe61.py"))
    if (snapshot.get("new_source031") != new_pins or receipt.get("new_source031") != new_pins
            or receipt.get("old_status") != pin(root / "source030_queue/status.json")
            or receipt.get("old_execution") != pin(root / "source030_queue/controller.execution.json")):
        raise ValueError("Source030 cancellation is not bound to the staged Source031 or actual old exit")
    products = [root / name for name in ("source_revision_030.json", "source030_speed_boundary.json",
        "source030_speed_boundary.receipt.json", "train/revisions/source030_deploy_admission")]
    products += [root / "source030_deploy" / name for name in (
        "prepare.intent.json", "prepare.json", "admission.json", "launch.intent.json", "launch.receipt.json")]
    products += [root / "source030_queue" / name for name in (
        "prepare.started.json", "prepare.execution.json", "prepare.log", "launch.started.json",
        "launch.execution.json", "launch.log", "old_delivery.json", "queue.receipt.json")]
    if any(path.exists() or path.is_symlink() for path in products):
        raise ValueError("Source030 entered a boundary/prepare/launch or admission claim; cannot supersede")
    migration = root / "train/revisions/v26_completed_speed_to_v27"
    if migration.exists() or migration.is_symlink():
        # Own source031 prepare may create the shared native migration directory.
        # An earlier entry/claim must never be excused merely by a cancel receipt.
        revision = root / "source_revision_031.json"
        entry = migration / "entry.json"
        admission = root / "source031_deploy/admission.json"
        if (prepared is None or migration.is_symlink() or not revision.is_file() or not entry.is_file()
                or prepared.get("status") != "boundary_prepared_not_launched"
                or read(root / "source031_deploy/prepare.json") != prepared
                or prepared.get("source_revision") != pin(revision)
                or prepared.get("admission") != pin(admission)
                or read(revision).get("source_commit") != COMMIT
                or read(revision).get("new_source_root") != str(new)
                or read(entry).get("new_lineage_manifest_sha256") != sha(new / "contracts/v6_integrated_lineage_v27.json")
                or read(admission).get("entry_admission", {}).get("completed_speed_predecessor_entry") != pin(entry)
                or read(admission).get("lineage_manifest", {}).get("sha256") != sha(new / "contracts/v6_integrated_lineage_v27.json")
                or any(pin(prepared[key]["file"]) != prepared[key] for key in ("account", "state"))):
            raise ValueError("An earlier native v27 entry or claim cannot be superseded")
    process_identity, actual_entry = native_process_reader(new)
    for key in ("old_controller", "old_coordinator"):
        identity = receipt.get(key, {})
        if (identity.get("exited") is not True or type(identity.get("pid")) is not int
                or identity["pid"] <= 0 or type(identity.get("starttime")) is not int
                or identity["starttime"] <= 0 or not isinstance(identity.get("command"), list)
                or not identity["command"]
                or {k: v for k, v in identity.items() if k != "exited"} != snapshot.get(key)):
            raise ValueError("Source030 cancelled observer identity or exit evidence differs")
        current = process_identity(identity["pid"])
        if current is not None and current["starttime"] == identity["starttime"]:
            raise ValueError("Source030 observer is still alive after its cancellation receipt")
    started = read(root / "source030_queue/controller.started.json")
    execution = read(root / "source030_queue/controller.execution.json")
    controller, coordinator = receipt["old_controller"], receipt["old_coordinator"]
    if (started != dict(command=controller["command"], pid=controller["pid"])
            or execution.get("command") != controller["command"]
            or execution.get("pid") != controller["pid"] or execution.get("exit_code") != -15
            or controller.get("ppid") != coordinator["pid"]
            or controller.get("entry") != str(root / "source_serial_dispatch030_20261009" / CONTROLLER)
            or coordinator.get("entry") != old_intent.get("coordinator", {}).get("file")):
        raise ValueError("Source030 cancellation does not match the original observer start/exit records")
    old_source = root / "source_serial_dispatch030_20261009"
    job = read(root / "job.json")
    entry = actual_entry(job.get("command", []))
    if entry is not None and entry.is_relative_to(old_source):
        raise ValueError("Current native job already adopted Source030")
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        current = process_identity(int(path.parent.name))
        if current and current["entry"] and Path(current["entry"]).is_relative_to(old_source):
            raise ValueError("A Source030 native entry is still alive")
    # Existing read-only opens preserve the original lock files and inodes.
    # No old worker liveness assertion is made after the potentially long wait.
    for path in (root / "train/native_speed_boundary_controller.lock", root / "source030_queue.lock"):
        with path.open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = dict(receipt=receipt_pin, intent=intent_pin, old_queue_intent=old_intent_pin)
    if expected is not None and result != expected:
        raise ValueError("Source030 cancellation evidence changed while the new queue waited")
    return result


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected an existing non-symlink file: " + str(path))
    return dict(file=str(path.resolve(strict=True)), sha256=sha(path))


def native_publish(path):
    """Reuse the SHA-authenticated existing atomic/no-overwrite publisher."""
    spec = importlib.util.spec_from_file_location("source031_queue_native_utility", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.durable_json


def verify_staged_tree(helper, archive, new, stage):
    # Archive SHA is checked before importing this frozen helper. Reuse its
    # strict archive/mode reader, then its validate_stage byte comparisons,
    # without importing the native account graph or creating a scene.
    spec = importlib.util.spec_from_file_location("source031_queue_verified_deploy_helper", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.archive_files(archive, COMMIT, verify_tree=new)
    if files != stage["files"]:
        raise ValueError("Stage file inventory differs before controller execution")
    for name, digest in files.items():
        path = new / name
        if path.is_symlink() or not path.resolve().is_relative_to(new) or sha(path) != digest:
            raise ValueError("Staged source differs before controller execution: " + name)


def inputs(args):
    if ARCHIVE_SHA is None or STAGE_SHA is None or STAGE_FILE_COUNT is None or PREFLIGHT_SHA is None:
        raise ValueError("Source031 archive/stage pins are not yet frozen")
    root = Path(args.run_root)
    if not root.is_absolute() or root.is_symlink() or ".." in root.parts:
        raise ValueError("Run root must be an absolute non-symlink path")
    root = root.resolve(strict=True)
    stage_file = root / "source031_deploy/stage.json"
    if pin(stage_file)["sha256"] != STAGE_SHA:
        raise ValueError("Actual immutable source031 stage differs")
    stage = read(stage_file)
    old = root / "source_serial_dispatch029_20261009"
    new = root / "source_serial_dispatch031_20261009"
    if old.name != "source_serial_dispatch029_20261009" or new.name != "source_serial_dispatch031_20261009":
        raise ValueError("Source031 candidate or actual source029 parent namespace differs")
    archive = root / ("source031-" + COMMIT + "-umask0022.tar.gz")
    helper = root / "deploy_source031_663efe61.py"
    runtime = Path(args.runtime_script)
    if (not runtime.is_absolute() or runtime.is_symlink() or ".." in runtime.parts
            or runtime.stat().st_mode & 0o7777 != 0o644 or runtime.stat().st_size != 776
            or sha(runtime) != RUNTIME_SHA):
        raise ValueError("Original sourced runtime identity differs; executable permission is not required")
    expected = dict(schema="source031-static-stage-v1", status="staged_only", source_commit=COMMIT,
        old_source_root=str(old), new_source_root=str(new), dynamic_parent_bound=False,
        PPO_updates=0, qualification=None)
    if any(stage.get(k) != v for k, v in expected.items()):
        raise ValueError("Stage is not the frozen unbound source031 candidate")
    if (pin(helper) != stage["helper"] or stage["helper"]["sha256"] != HELPER_SHA
            or pin(archive) != stage["source_archive"] or stage["source_archive"]["sha256"] != ARCHIVE_SHA
            or len(stage["files"]) != STAGE_FILE_COUNT):
        raise ValueError("Helper/archive/staged inventory differs")
    for relative in (CONTROLLER, UTILITY):
        if pin(new / relative)["sha256"] != stage["files"].get(relative):
            raise ValueError("Frozen controller or publisher differs: " + relative)
    verify_staged_tree(helper, archive, new, stage)
    preflight_file = root / "source031_deploy/preflight.json"
    if pin(preflight_file)["sha256"] != PREFLIGHT_SHA:
        raise ValueError("Actual immutable source031 preflight differs")
    preflight = read(preflight_file)
    log = root / "source031_deploy/preflight.log"
    if (preflight.get("status") != "CPU_preflight_passed" or preflight.get("stage") != pin(stage_file)
            or preflight.get("log") != pin(log) or preflight.get("PPO_updates") != 0
            or preflight.get("qualification") is not None or read(log) != dict(status="passed", roles=30)):
        raise ValueError("Actual zero-PPO 30-role CPU preflight is missing or changed")
    products = [root / "source_revision_031.json", root / "source031_speed_boundary.json",
        root / "source031_speed_boundary.receipt.json"] + [root / "source031_deploy" / name for name in (
            "prepare.intent.json", "prepare.json", "admission.json", "launch.intent.json", "launch.receipt.json")]
    if any(path.exists() for path in products):
        raise ValueError("An existing boundary/prepare/launch product requires inspection, never requeue")
    supersedes = superseded_waiter(root, new)
    return dict(root=root, old=old, new=new, archive=archive, helper=helper, runtime=runtime,
        stage=pin(stage_file), preflight=pin(preflight_file), controller=pin(new / CONTROLLER),
        publisher=pin(new / UTILITY), supersedes=supersedes,
        interpreter=dict(file=sys.executable, sha256=sha(sys.executable)))


@contextmanager
def queue_lock(root):
    with (root / "source031_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def unchanged(admitted, *, prepared=None):
    # Seven days may pass before prepare. Do not execute a newly changed helper
    # or source/runtime merely because it was authenticated when queued.
    for key in ("stage", "preflight", "controller", "publisher"):
        item = admitted[key]
        if pin(item["file"]) != item:
            raise ValueError("Queued input changed before its next operation: " + key)
    for key, expected in (("archive", ARCHIVE_SHA), ("helper", HELPER_SHA), ("runtime", RUNTIME_SHA)):
        if pin(admitted[key])["sha256"] != expected:
            raise ValueError("Queued immutable " + key + " changed")
    if (admitted["runtime"].stat().st_mode & 0o7777 != 0o644
            or admitted["runtime"].stat().st_size != 776):
        raise ValueError("Queued runtime mode/size changed")
    if sha(sys.executable) != admitted["interpreter"]["sha256"]:
        raise ValueError("Queued interpreter changed")
    verify_staged_tree(admitted["helper"], admitted["archive"], admitted["new"], read(admitted["stage"]["file"]))
    superseded_waiter(admitted["root"], admitted["new"], expected=admitted["supersedes"], prepared=prepared)


def run_command(command, name, directory, publish, *, popen=subprocess.Popen):
    """Wait without a subprocess timeout/exception context that might kill it.

    The native controller has its own bounded deadline and finally restoration.
    If this coordinator is cancelled, it never signals or kills that child.
    """
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_file = directory / (name + ".log")
    with log_file.open("x") as log:
        child = popen(command, cwd=directory.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
        publish(directory / (name + ".started.json"), dict(command=command, pid=child.pid), exclusive=True)
        code = child.wait()
    publish(directory / (name + ".execution.json"), dict(command=command, pid=child.pid,
        exit_code=code, log=pin(log_file)), exclusive=True)
    if code:
        raise RuntimeError(name + " exited " + str(code) + "; inspect retained raw log, do not retry")


def retain_delivery(directory, prepared, publish):
    """Retain old immutable inodes; native finalize atomically replaces its paths.

    Same-filesystem hard links avoid copying/recompressing the large delivery.
    No original file or content changes. All three original SHA pins are checked
    after publication; partial retention is kept and forbids a retry/launch.
    """
    boundary = prepared["boundary"]
    retained = {}
    for key in ("delivery_archive", "delivery_manifest", "archiver"):
        item = boundary[key]
        source = Path(item["file"])
        if pin(source) != item:
            raise ValueError("Authenticated old delivery changed before retention")
        target = directory / ("old_" + key + "_" + item["sha256"] + (".tar.gz" if key == "delivery_archive" else ".json"))
        os.link(source, target)
        retained_pin = pin(target)
        if retained_pin["sha256"] != item["sha256"]:
            raise ValueError("Retained old delivery bytes differ")
        retained[key] = dict(original=item, retained=retained_pin, method="exclusive_same_filesystem_hardlink")
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    publish(directory / "old_delivery.json", retained, exclusive=True)
    return retained


def coordinate(args, *, execute=run_command, retain=retain_delivery):
    admitted = inputs(args)
    root, directory = admitted["root"], admitted["root"] / "source031_queue"
    if not args.apply:
        return dict(status="validated_once_queue_not_started", controller_wait_seconds=WAIT_SECONDS,
            stage=admitted["stage"], preflight=admitted["preflight"],
            supersedes=admitted["supersedes"], qualification=None)
    with queue_lock(root):
        superseded_waiter(root, admitted["new"], expected=admitted["supersedes"])
        directory.mkdir(exist_ok=True)
        publish = native_publish(Path(admitted["publisher"]["file"]))
        if any((directory / name).exists() for name in ("queue.intent.json", "status.json", "controller.log")):
            raise ValueError("Once-only queue already exists; inspect it and do not retry")
        identity = {k: v for k, v in admitted.items() if k not in ("root", "old", "new", "archive", "helper", "runtime")}
        intent = dict(schema=SCHEMA, source_commit=COMMIT, archive=pin(admitted["archive"]),
            helper=pin(admitted["helper"]), runtime=pin(admitted["runtime"]), coordinator=pin(__file__),
            inputs=identity, controller_wait_seconds=WAIT_SECONDS, started_at_utc=datetime.now(timezone.utc).isoformat(),
            retries=0, additional_signals=0, qualification=None)
        publish(directory / "queue.intent.json", intent, exclusive=True)
        phase = "controller"
        try:
            publish(directory / "status.json", dict(status="waiting_for_normal_speed_boundary",
                phase=phase, intent=pin(directory / "queue.intent.json"), qualification=None))
            controller_intent = root / "source031_speed_boundary.json"
            execute([sys.executable, "-B", str(admitted["new"] / CONTROLLER),
                "--source-root", str(admitted["old"]), "--run-root", str(root / "train"),
                "--intent", str(controller_intent), "--apply", "--timeout", str(WAIT_SECONDS)],
                phase, directory, publish)
            controller_receipt = controller_intent.with_suffix(".receipt.json")
            receipt = read(controller_receipt)
            if (receipt.get("status") != "native_supervisor_stopped_after_normal_speed"
                    or receipt.get("worker_signals") != 0 or receipt.get("next_domain_dispatched") is not False
                    or receipt.get("native_completion_recovery", {}).get("status") != "closed_native_completed_speed"):
                raise ValueError("Controller did not return its successful actual native closure")
            unchanged(admitted)
            common = [sys.executable, "-B", str(admitted["helper"]), "--run-root", str(root),
                "--old-source-root", str(admitted["old"]), "--new-source-root", str(admitted["new"]),
                "--source-archive", str(admitted["archive"]), "--source-commit", COMMIT,
                "--source-archive-sha256", ARCHIVE_SHA, "--controller-intent", str(controller_intent),
                "--controller-receipt-sha256", pin(controller_receipt)["sha256"],
                "--runtime-script", str(admitted["runtime"])]
            phase = "prepare"
            publish(directory / "status.json", dict(status="normal_boundary_closed_preparing", phase=phase, qualification=None))
            execute(common + ["--mode", "boundary-prepare"], phase, directory, publish)
            prepared = read(root / "source031_deploy/prepare.json")
            if prepared.get("status") != "boundary_prepared_not_launched":
                raise ValueError("Native authenticated boundary preparation is missing")
            retain(directory, prepared, publish)
            unchanged(admitted, prepared=prepared)
            phase = "launch"
            publish(directory / "status.json", dict(status="prepared_old_delivery_retained_launching", phase=phase, qualification=None))
            execute(common + ["--mode", "launch"], phase, directory, publish)
            launched = read(root / "source031_deploy/launch.receipt.json")
            if (launched.get("status") != "launch_submitted"
                    or launched.get("verification_status") != "first_terrain_actor_pending"):
                raise ValueError("Native launch has no submitted receipt; inspect intent without retry")
            result = dict(status="launch_submitted_first_terrain_actor_pending", phase="complete",
                launch=pin(root / "source031_deploy/launch.receipt.json"), pid=launched["pid"],
                old_delivery=pin(directory / "old_delivery.json"), retries=0, additional_signals=0, qualification=None)
            publish(directory / "status.json", result)
            publish(directory / "queue.receipt.json", result, exclusive=True)
            return result
        except BaseException as error:
            publish(directory / "status.json", dict(status="attention_no_retry", phase=phase,
                error=type(error).__name__ + ": " + str(error), retries=0, additional_signals=0, qualification=None))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="actual experiment, not experiment/train")
    parser.add_argument("--runtime-script", type=Path, default=Path("/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh"))
    parser.add_argument("--apply", action="store_true", help="start exactly one bounded future controller/prepare/launch sequence")
    args = parser.parse_args()
    print(json.dumps(coordinate(args), allow_nan=False))


if __name__ == "__main__":
    main()
