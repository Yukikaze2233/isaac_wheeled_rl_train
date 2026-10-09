#!/usr/bin/env python3
"""Once-only, external source029 normal-speed-end -> source030 coordinator.

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

COMMIT = "81ad124554c4ab45cc274e2d482d46c76ba125b0"
ARCHIVE_SHA = "7afdf9e0fdb08c217d1f034f88ad52f7dad1eb7af1f86c9d710347cf5b0a9dab"
HELPER_SHA = "612c66e8f7cbdd56d65ed88a1861c0d2add1fcefba33704adfa177664bcea6cb"
STAGE_SHA = "870194901f935c9c024996b0e05317f008fbf3f067039355ec14c5c4fbf42177"
RUNTIME_SHA = "c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7"
WAIT_SECONDS = 604800
SCHEMA = "source030-once-normal-speed-boundary-queue-v1"
CONTROLLER = "scripts/hold_completed_speed_boundary.py"
UTILITY = "scripts/queue_task_campaign_continuation.py"


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
    spec = importlib.util.spec_from_file_location("source030_queue_native_utility", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.durable_json


def verify_staged_tree(helper, archive, new, stage):
    # Archive SHA is checked before importing this frozen helper. Reuse its
    # strict archive/mode reader, then its validate_stage byte comparisons,
    # without importing the native account graph or creating a scene.
    spec = importlib.util.spec_from_file_location("source030_queue_verified_deploy_helper", helper)
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
    root = Path(args.run_root)
    if not root.is_absolute() or root.is_symlink() or ".." in root.parts:
        raise ValueError("Run root must be an absolute non-symlink path")
    root = root.resolve(strict=True)
    stage_file = root / "source030_deploy/stage.json"
    if pin(stage_file)["sha256"] != STAGE_SHA:
        raise ValueError("Actual immutable source030 stage differs")
    stage = read(stage_file)
    old = root / "source_serial_dispatch029_20261009"
    new = root / "source_serial_dispatch030_20261009"
    archive = root / ("source030-" + COMMIT + "-umask0022.tar.gz")
    helper = root / "deploy_source030_612c66e8.py"
    runtime = Path(args.runtime_script)
    if (not runtime.is_absolute() or runtime.is_symlink() or ".." in runtime.parts
            or runtime.stat().st_mode & 0o7777 != 0o644 or runtime.stat().st_size != 776
            or sha(runtime) != RUNTIME_SHA):
        raise ValueError("Original sourced runtime identity differs; executable permission is not required")
    expected = dict(schema="source030-static-stage-v1", status="staged_only", source_commit=COMMIT,
        old_source_root=str(old), new_source_root=str(new), dynamic_parent_bound=False,
        PPO_updates=0, qualification=None)
    if any(stage.get(k) != v for k, v in expected.items()):
        raise ValueError("Stage is not the frozen unbound source030 candidate")
    if (pin(helper) != stage["helper"] or stage["helper"]["sha256"] != HELPER_SHA
            or pin(archive) != stage["source_archive"] or stage["source_archive"]["sha256"] != ARCHIVE_SHA
            or len(stage["files"]) != 4341):
        raise ValueError("Helper/archive/staged inventory differs")
    for relative in (CONTROLLER, UTILITY):
        if pin(new / relative)["sha256"] != stage["files"].get(relative):
            raise ValueError("Frozen controller or publisher differs: " + relative)
    verify_staged_tree(helper, archive, new, stage)
    preflight_file = root / "source030_deploy/preflight.json"
    preflight = read(preflight_file)
    log = root / "source030_deploy/preflight.log"
    if (preflight.get("status") != "CPU_preflight_passed" or preflight.get("stage") != pin(stage_file)
            or preflight.get("log") != pin(log) or preflight.get("PPO_updates") != 0
            or preflight.get("qualification") is not None or read(log) != dict(status="passed", roles=30)):
        raise ValueError("Actual zero-PPO 30-role CPU preflight is missing or changed")
    products = [root / "source_revision_030.json", root / "source030_speed_boundary.json",
        root / "source030_speed_boundary.receipt.json"] + [root / "source030_deploy" / name for name in (
            "prepare.intent.json", "prepare.json", "admission.json", "launch.intent.json", "launch.receipt.json")]
    if any(path.exists() for path in products):
        raise ValueError("An existing boundary/prepare/launch product requires inspection, never requeue")
    return dict(root=root, old=old, new=new, archive=archive, helper=helper, runtime=runtime,
        stage=pin(stage_file), preflight=pin(preflight_file), controller=pin(new / CONTROLLER),
        publisher=pin(new / UTILITY), interpreter=dict(file=sys.executable, sha256=sha(sys.executable)))


@contextmanager
def queue_lock(root):
    with (root / "source030_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def unchanged(admitted):
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
    root, directory = admitted["root"], admitted["root"] / "source030_queue"
    if not args.apply:
        return dict(status="validated_once_queue_not_started", controller_wait_seconds=WAIT_SECONDS,
            stage=admitted["stage"], preflight=admitted["preflight"], qualification=None)
    with queue_lock(root):
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
            controller_intent = root / "source030_speed_boundary.json"
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
            prepared = read(root / "source030_deploy/prepare.json")
            if prepared.get("status") != "boundary_prepared_not_launched":
                raise ValueError("Native authenticated boundary preparation is missing")
            retain(directory, prepared, publish)
            unchanged(admitted)
            phase = "launch"
            publish(directory / "status.json", dict(status="prepared_old_delivery_retained_launching", phase=phase, qualification=None))
            execute(common + ["--mode", "launch"], phase, directory, publish)
            launched = read(root / "source030_deploy/launch.receipt.json")
            if (launched.get("status") != "launch_submitted"
                    or launched.get("verification_status") != "first_terrain_actor_pending"):
                raise ValueError("Native launch has no submitted receipt; inspect intent without retry")
            result = dict(status="launch_submitted_first_terrain_actor_pending", phase="complete",
                launch=pin(root / "source030_deploy/launch.receipt.json"), pid=launched["pid"],
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
