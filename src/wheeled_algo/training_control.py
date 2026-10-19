"""Cooperative training control at completed optimizer/logging boundaries.

Resident pause preserves the process, including native simulation and rollout
locals. Checkpoints preserve learning state; they are not native scene snapshots.
This module has no simulator, torch, terminal, or service-manager dependencies.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import math
import signal
import types
import json
import os
from pathlib import Path
import tempfile
import time


class JobError(ValueError):
    pass


class PlannedStop(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("seconds must be finite and positive")
    return number


class TrainingBudget:
    def __init__(self, max_runtime_seconds=None, stop_at=None, *, monotonic=time.monotonic,
                 utcnow=lambda: datetime.now(timezone.utc)):
        self.max_runtime_seconds = (None if max_runtime_seconds is None
                                    else positive_seconds(max_runtime_seconds))
        if stop_at is not None and (not isinstance(stop_at, datetime) or stop_at.tzinfo is None
                                    or stop_at.utcoffset() is None):
            raise ValueError("stop_at must be timezone-aware")
        self.stop_at = stop_at
        self.monotonic, self.utcnow = monotonic, utcnow
        self.started = None
        self.learning_finished = None
        self.soft_stop = None
        self.completed_updates = 0
        self.in_update = False
        self.update_failed = False

    def request_stop(self, reason="soft_stop"):
        if reason not in {"sigint", "sigterm", "soft_stop"}:
            raise ValueError("invalid soft stop reason")
        if self.soft_stop is None:
            self.soft_stop = reason

    def mark_signal(self, signum, _frame):
        # No raises, saves, logging, CUDA calls, or process termination here.
        if self.soft_stop is None:
            self.soft_stop = "sigint" if signum == signal.SIGINT else "sigterm"

    @contextmanager
    def signal_handlers(self):
        previous = {}
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, self.mark_signal)
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def check(self):
        # Called before app startup, before learn, and BEFORE official env.step.
        if self.soft_stop:
            raise PlannedStop(self.soft_stop)
        if self.stop_at is not None and self.utcnow() >= self.stop_at:
            raise PlannedStop("stop_at")
        if (self.started is not None and self.max_runtime_seconds is not None
                and self.monotonic() - self.started >= self.max_runtime_seconds):
            raise PlannedStop("max_runtime_seconds")

    def begin_learning(self):
        self.check()  # Startup time never consumes the relative learning budget.
        if self.started is not None:
            raise JobError("budget already started; each invocation needs a fresh budget")
        self.started = self.monotonic()

    @contextmanager
    def bind(self, runner):
        env, alg = runner.env, runner.alg
        original_step, original_update = env.step, alg.update
        missing = object()
        step_attr = vars(env).get("step", missing)
        update_attr = vars(alg).get("update", missing)

        def checked_step(_self, *args, **kwargs):
            self.check()
            return original_step(*args, **kwargs)

        def counted_update(_self, *args, **kwargs):
            self.in_update = True
            try:
                result = original_update(*args, **kwargs)
            except BaseException:
                self.update_failed = True
                raise
            else:
                self.completed_updates += 1
                return result
            finally:
                self.in_update = False

        try:
            env.step = types.MethodType(checked_step, env)
            alg.update = types.MethodType(counted_update, alg)
            yield
        finally:
            for obj, name, old in ((env, "step", step_attr), (alg, "update", update_attr)):
                if old is missing:
                    if name in vars(obj):
                        delattr(obj, name)
                else:
                    setattr(obj, name, old)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process_identity(pid):
    if type(pid) is not int or pid <= 0:
        return None
    try:
        root = Path(f"/proc/{pid}")
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": pid, "start_ticks": fields[19], "uid": root.stat().st_uid,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "argv": [os.fsdecode(x) for x in (root / "cmdline").read_bytes().split(b"\0") if x]}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def identity_alive(identity):
    return bool(identity) and process_identity(identity.get("pid")) == identity


class TrainingControl:
    """One worker owns status; clients publish requests addressed to that worker."""

    def __init__(self, run_directory):
        self.directory = Path(run_directory) / "control"
        self.identity = process_identity(os.getpid())
        self.last_request = None
        self.state = {"schema": "training-control-v1", "identity": self.identity,
                      "run_directory": str(Path(run_directory).resolve()),
                      "phase": "starting", "learning_updates": None,
                      "checkpoint": None, "acknowledged_request": None,
                      "resident_pause_keeps_gpu_memory": True,
                      "checkpoint_restarts_physical_episodes": True}
        self.publish()

    def publish(self, **changes):
        self.state.update(changes, updated_at=utc_now())
        atomic_json(self.directory / "status.json", self.state)

    def _request(self):
        request = read_json(self.directory / "request.json", {})
        if (not request.get("id") or request["id"] == self.last_request
                or request.get("identity") != self.identity):
            return None
        if request.get("action") not in ("pause", "resume", "stop"):
            return None
        self.last_request = request["id"]
        return request

    def boundary(self, learning_updates, *, checkpoint, flush, check_limits, stop):
        """Only call after successful update accounting and SDK logging."""
        self.publish(phase="running", learning_updates=learning_updates)
        request = self._request()
        if request is None:
            return
        while request is not None:
            action = request["action"]
            if action == "resume":
                self.publish(phase="running", acknowledged_request=request["id"])
                return
            self.publish(phase="saving", requested_action=action)
            flush()
            saved = checkpoint(learning_updates)
            self.publish(phase="paused" if action == "pause" else "stopping",
                         checkpoint=str(saved), acknowledged_request=request["id"])
            if action == "stop":
                stop()  # Trainer raises its existing PlannedStop and finalizes.
                raise RuntimeError("Training stop callback returned unexpectedly")
            # No env.step, optimizer operation, resampling, or RNG call here.
            # Existing absolute wall/deadline budgets remain authoritative.
            while True:
                check_limits()
                request = self._request()
                if request is not None:
                    break
                time.sleep(.2)

    def finish(self, phase, **details):
        self.publish(phase=phase, **details)
