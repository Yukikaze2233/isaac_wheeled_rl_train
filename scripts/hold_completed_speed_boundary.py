#!/usr/bin/env python3
"""Future opt-in native speed boundary: stop supervisor, never the speed worker.

No import performs a signal. ``hold`` uses authenticated Linux pidfds and an
explicit caller-owned durable intent. The native completion validator must
pass before TERM is queued; CONT always targets that same supervisor. A CLI
admission/launch is deliberately separate from this process-control primitive.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from queue_task_campaign_boundary_continuation import (
    _linux_pidfd_syscall, open_pidfd, process_identity,
)
from queue_task_campaign_continuation import durable_json

SCHEMA = "native-completed-speed-supervisor-hold-v1"


def _signal(fd, value):
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(fd, value, None, 0)
    else:
        _linux_pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(value), ctypes.c_void_p(), ctypes.c_uint(0))


def _checked(expected):
    current = process_identity(expected["pid"])
    if current != expected:
        raise ValueError("Boundary process command/starttime changed; possible PID reuse")
    return current


def _state(pid):
    return (Path("/proc") / str(pid) / "stat").read_text().rpartition(")")[2].split()[0]


@contextmanager
def _cancel_as_exception():
    cleaning = False
    def cancel(signum, _frame):
        if not cleaning:
            raise InterruptedError(f"Boundary controller cancelled by signal {signum}")
    def begin_cleanup():
        nonlocal cleaning
        cleaning = True
    old = {value: signal.signal(value, cancel) for value in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield begin_cleanup
    finally:
        for value, handler in old.items():
            signal.signal(value, handler)


def hold(supervisor, worker, *, intent, validate_completed, timeout=3600., poll=.2):
    """Pause only the supervisor after external source/remaining-work admission.

    Both exact command/starttime identities must be obtained from the frozen
    source029 job and its unique live speed child. The caller must admit the
    20–100 remaining-actor window before calling. ``validate_completed`` is
    read-only native full-final/normal-completion validation, not a reward gate.
    A timeout/error/cancellation resumes the original owner; SIGKILL and host
    loss need recovery using the persisted identity intent.
    """
    if not math.isfinite(timeout) or timeout <= 0 or not 0 < poll <= 1 or worker["ppid"] != supervisor["pid"]:
        raise ValueError("Boundary hold needs a bounded wait and the actual speed child")
    if Path(intent).exists():
        raise ValueError("A boundary hold intent already exists; never repeat an ambiguous signal")
    descriptor = worker_fd = None
    paused = False
    with _cancel_as_exception() as begin_cleanup:
        try:
            descriptor = open_pidfd(supervisor["pid"])
            worker_fd = open_pidfd(worker["pid"])
            _checked(supervisor)
            _checked(worker)
            durable_json(Path(intent), dict(schema=SCHEMA, supervisor=supervisor, worker=worker,
                operation="hold_supervisor_until_normal_speed_end", worker_signals=0,
                qualification=None), exclusive=True)
            _checked(supervisor)
            _checked(worker)
            # Arm cleanup before STOP: delivery may succeed before a cancelled
            # Python call returns. CONT through this same pidfd is always safe.
            paused = True
            _signal(descriptor, signal.SIGSTOP)
            deadline = time.monotonic() + timeout
            while _state(supervisor["pid"]) not in ("T", "t"):
                _checked(supervisor)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Supervisor did not enter its authenticated paused state")
                time.sleep(poll)
            _checked(worker)
            while process_identity(worker["pid"]) is not None:
                _checked(supervisor)
                _checked(worker)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Speed worker did not complete within the explicit hold allowance")
                time.sleep(poll)
            _checked(supervisor)
            evidence = validate_completed()
            # A stopped parent cannot reap its child. Zombie means already
            # exited, so the native TERM handler cannot interrupt learning.
            _signal(descriptor, signal.SIGTERM)
            _signal(descriptor, signal.SIGCONT)
            paused = False
            while process_identity(supervisor["pid"]) is not None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native supervisor did not release its owner after the normal speed end")
                time.sleep(poll)
            return dict(status="native_supervisor_stopped_after_normal_speed", evidence=evidence,
                worker_signals=0, next_domain_dispatched=False, qualification=None)
        finally:
            # Keep cancellation handlers installed through CONT/close, and
            # ignore a second cancellation while restoring this original owner.
            begin_cleanup()
            if paused and descriptor is not None and process_identity(supervisor["pid"]) == supervisor:
                try:
                    _signal(descriptor, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            for fd in (worker_fd, descriptor):
                if fd is not None:
                    os.close(fd)


def _flag(command, name):
    if command.count(name) != 1:
        raise ValueError("Boundary native command has a missing or duplicate " + name)
    return command[command.index(name) + 1]


def admission(source_root, run_root):
    """Read-only actual source029 job, native worker admission and PID binding."""
    from wheeled_tasks.chassis import integrated_completed_speed_revision as revision
    from wheeled_tasks.chassis import integrated_production as production
    from wheeled_tasks.chassis import integrated_lineage as lineage
    from wheeled_tasks.chassis import integrated_runtime as runtime
    source_root, run_root = Path(source_root).resolve(strict=True), Path(run_root).resolve(strict=True)
    plan_file = source_root / production.artifact_files(26)["plan"]
    plan = json.loads(plan_file.read_text())
    binding = production.binding_for(plan, source_root, run_root)
    source_job = revision._source029_job(run_root, binding.source_root)
    job = json.loads((run_root.parent / "job.json").read_text())
    if source_job["job"]["file"] != str((run_root.parent / "job.json").resolve()):
        raise ValueError("The boundary must operate on the current original source029 owner")
    supervisor = process_identity(job["pid"])
    if (supervisor is None or supervisor["command"] != job["command"]
            or supervisor["entry"] != str(source_root / "scripts/run_task_campaign.py")
            or Path(_flag(supervisor["command"], "--contract")).resolve() != plan_file):
        raise ValueError("Current supervisor is not the exact source029 v26 job")
    children = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            child = process_identity(int(path.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if child and child["ppid"] == supervisor["pid"] and child["entry"] == str(source_root / "scripts/train_chassis.py"):
            children.append(child)
    if len(children) != 1:
        raise ValueError("Boundary requires exactly one original speed training child")
    worker = children[0]
    directory = Path(_flag(worker["command"], "--run-dir")).resolve(strict=True)
    if not directory.is_relative_to(run_root / "speed") or directory.name != "train":
        raise ValueError("Only the active original speed worker may reach this boundary")
    row = next(r for r in plan["domains"] if r["name"] == "speed")
    if (Path(_flag(worker["command"], "--contract")).resolve() != source_root / row["contract"]
            or int(_flag(worker["command"], "--num-envs")) != 4096):
        raise ValueError("Boundary speed command changed its frozen contract or population")
    file = lineage._within(_flag(worker["command"], "--integrated-source-selection-file"), run_root)
    _, receipt = lineage._json(file, _flag(worker["command"], "--integrated-source-selection-sha256"), run_root)
    selection = runtime._verified_selection(binding, receipt["selection"])
    allowance = lineage.remaining_budget(binding, selection, receipt["reconciled_attempts"],
        requested_actor_updates=receipt["allowance"]["requested_actor_updates"])
    if (selection["domain"] != "speed" or selection["domain_complete"]
            or receipt["schema"] != production.ADMISSION_SCHEMA
            or receipt["campaign_plan"] != dict(file=production.artifact_files(26)["plan"], sha256=lineage.digest(plan_file))
            or receipt["target_contract_sha256"] != row["sha256"] or receipt["allowance"] != allowance
            or receipt["component_clocks_at_source"] != runtime.source_clocks(binding, selection)
            or int(_flag(worker["command"], "--updates")) != allowance["requested_total_updates"]):
        raise ValueError("Boundary worker has no exact frozen original native allowance")
    wrapper = process_identity(supervisor["ppid"])
    if (wrapper is None or Path(wrapper.get("entry") or "").name != "chassis_remote_job.py"
            or str(run_root.parent) not in wrapper["command"]):
        raise ValueError("Boundary supervisor lacks its original identifiable wrapper")
    return dict(binding=binding, plan=plan, directory=directory, receipt=receipt,
        supervisor=supervisor, worker=worker, wrapper=wrapper, source_job=source_job)


def native_completed(admitted):
    from wheeled_tasks.chassis import integrated_lineage as lineage
    from wheeled_tasks.chassis import integrated_runtime as runtime
    binding, receipt = admitted["binding"], admitted["receipt"]
    file = admitted["directory"] / "model_final.pt"
    reference = dict(path=str(file), sha256=lineage.digest(file))
    row = next(r for r in admitted["plan"]["domains"] if r["name"] == "speed")
    selected = lineage.select_learning_source(binding, "speed", receipt["entry_admission"], own_candidates=[reference])
    checkpoint = lineage._checkpoint(binding, reference, "speed", row["sha256"])
    lineage._finished(binding, checkpoint, dict(status="joint_domain_finished_unassessed",
        candidate_checkpoint=str(file), candidate_checkpoint_sha256=reference["sha256"]))
    completion = json.loads((admitted["directory"] / "completion.json").read_text())
    if (not selected["domain_complete"] or selected["critic_warmup_remaining"] != 0
            or selected["checkpoint_consumed_updates"] != receipt["allowance"]["consumed_updates"] + receipt["allowance"]["requested_total_updates"]
            or completion.get("successful_updates") != receipt["allowance"]["requested_total_updates"]
            or selected["fresh_domain_actor_updates"] != receipt["selection"]["fresh_domain_actor_updates"] + receipt["allowance"]["requested_actor_updates"]):
        raise ValueError("Speed worker did not normally finish its exact admitted actor/critic work")
    return dict(checkpoint=reference, selection=selected, component_clocks=runtime.source_clocks(binding, selected),
        successful_updates=completion["successful_updates"], PPO_updates_by_controller=0, qualification=None)


NATIVE_CLOSE = r'''
import fcntl, importlib.util, json, pathlib, sys
from types import SimpleNamespace
source, run, helper = map(pathlib.Path, sys.argv[1:4])
sys.path[:0] = [str(source/'src'), str(source/'scripts')]
from run_task_campaign import TaskCampaign, write_json
spec = importlib.util.spec_from_file_location('wheeled_tasks.chassis.integrated_completed_speed_revision', helper)
revision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revision)
args = SimpleNamespace(contract=source/'contracts/v6_task_campaign_v26.json', run_dir=run,
    max_runtime_seconds=600, device='cpu')
job = TaskCampaign(args)
with job.lock_path.open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job.assert_no_live_workers()
    job.open_account()
    result = revision.close_native_completed_speed(job, source)
    write_json(run/'completion.json', job.report)
print(json.dumps(result, allow_nan=False))
'''


def _run(source_root, run_root, intent, *, apply=False, timeout=3600.):
    admitted = admission(source_root, run_root)
    target = admitted["receipt"]["selection"]["fresh_domain_actor_updates"] + admitted["receipt"]["allowance"]["requested_actor_updates"]
    if not apply:
        progress = json.loads((admitted["directory"] / "progress.json").read_text())
        return dict(status="validated_future_boundary", target_speed_actor=target,
            remaining_actor_updates=target-progress["task_actor_update"], supervisor=admitted["supervisor"],
            worker=admitted["worker"], worker_signals=0, qualification=None)
    deadline = time.monotonic() + timeout
    while True:
        _checked(admitted["supervisor"])
        _checked(admitted["worker"])
        progress = json.loads((admitted["directory"] / "progress.json").read_text())
        actor = progress.get("task_actor_update")
        if type(actor) is not int or actor < admitted["receipt"]["selection"]["fresh_domain_actor_updates"]:
            raise ValueError("Boundary progress has an invalid or regressed native actor clock")
        remaining = target - actor
        if 20 <= remaining <= 100:
            break
        if remaining < 20:
            raise ValueError("Safe last-20–100 update boundary window was missed; no process is signalled")
        if time.monotonic() >= deadline:
            raise TimeoutError("Speed did not reach the future boundary window")
        time.sleep(.5)
    result = hold(admitted["supervisor"], admitted["worker"], intent=intent,
        validate_completed=lambda: native_completed(admitted), timeout=max(.1, deadline-time.monotonic()))
    while process_identity(admitted["wrapper"]["pid"]) is not None:
        _checked(admitted["wrapper"])
        if time.monotonic() >= deadline:
            raise TimeoutError("Old wrapper is still sealing; native zero-PPO closure was not attempted")
        time.sleep(.2)
    helper = Path(__file__).resolve().parents[1] / "src/wheeled_tasks/chassis/integrated_completed_speed_revision.py"
    completed = subprocess.run([admitted["supervisor"]["command"][0], "-B", "-c", NATIVE_CLOSE,
        str(Path(source_root).resolve()), str(Path(run_root).resolve()), str(helper)],
        text=True, capture_output=True, timeout=180, shell=False)
    if completed.returncode:
        raise ValueError("Original native zero-PPO completion recovery failed: " + completed.stderr[-2000:])
    result["native_completion_recovery"] = json.loads(completed.stdout)
    durable_json(Path(intent).with_suffix(".receipt.json"), result)
    return result


def run(source_root, run_root, intent, *, apply=False, timeout=3600.):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Boundary controller requires a finite positive wait allowance")
    if not apply:
        return _run(source_root, run_root, intent, timeout=timeout)
    # Independent from the account lock held by the original supervisor. One
    # fixed run lock covers admission, all signals, sealing and closure, even
    # when two callers choose different intent names.
    run_root = Path(run_root).resolve(strict=True)
    with (run_root / "native_speed_boundary_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(source_root, run_root, intent, apply=True, timeout=timeout)


def main():
    # A future controller runs from the new frozen tree, even when its shell
    # inherited source029 PYTHONPATH. Admission must use this helper's code.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="actual original frozen source029 root")
    parser.add_argument("--run-root", type=Path, required=True, help="original experiment/train")
    parser.add_argument("--intent", type=Path, required=True, help="new once-only hold evidence path")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.run_root, args.intent, apply=args.apply, timeout=args.timeout), allow_nan=False))


if __name__ == "__main__":
    main()
