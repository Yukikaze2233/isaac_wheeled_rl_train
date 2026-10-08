"""Own CPU subprocesses only; never targets a training process or scene."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import hold_completed_speed_boundary as controller
from wheeled_tasks.chassis import integrated_completed_speed_revision as revision


@contextmanager
def owners(tmp_path, duration=.4):
    worker = tmp_path / "cpu_speed_child.py"
    worker.write_text("import pathlib,sys,time\ntime.sleep(float(sys.argv[2]))\npathlib.Path(sys.argv[1]).write_text('normal completed')\n")
    script = tmp_path / "cpu_native_supervisor.py"
    script.write_text("""import json,pathlib,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); stopping=False
child=subprocess.Popen([sys.executable,str(root/'cpu_speed_child.py'),str(root/'worker.completed'),sys.argv[2]])
def stop(signum, frame):
 global stopping
 stopping=True
 if child.poll() is None: child.send_signal(signum)
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
(root/'ready.json').write_text(json.dumps({'child':child.pid}))
while child.poll() is None: time.sleep(.01)
(root/'worker.exit').write_text(str(child.returncode))
if not stopping: (root/'old_terrain.dispatched').write_text('unexpected on success')
""")
    parent = subprocess.Popen([sys.executable, str(script), str(tmp_path), str(duration)])
    try:
        deadline = time.monotonic() + 3
        while not (tmp_path / "ready.json").exists():
            if parent.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("CPU supervisor did not initialise")
            time.sleep(.005)
        child = json.loads((tmp_path / "ready.json").read_text())["child"]
        yield parent, controller.process_identity(parent.pid), controller.process_identity(child)
    finally:
        if parent.poll() is None:
            parent.send_signal(signal.SIGCONT)
            parent.terminate()
        parent.wait(timeout=3)


def test_native_parent_stop_cont_child_normal_zero_later_dispatch(tmp_path):
    with owners(tmp_path) as (parent, supervisor, worker):
        result = controller.hold(supervisor, worker, intent=tmp_path / "intent.json",
            validate_completed=lambda: dict(normal=(tmp_path / "worker.completed").read_text()), timeout=3, poll=.01)
        parent.wait(timeout=2)
    assert result["worker_signals"] == 0 and result["next_domain_dispatched"] is False
    assert (tmp_path / "worker.exit").read_text() == "0"
    assert not (tmp_path / "old_terrain.dispatched").exists()


@pytest.mark.parametrize("fault", ["post_stop_error", "post_stop_sigterm", "timeout", "validator"])
def test_failure_finally_resumes_exact_parent(tmp_path, monkeypatch, fault):
    with owners(tmp_path, 1. if fault != "validator" else .2) as (parent, supervisor, worker):
        native_signal = controller._signal
        signals = []
        def inject(fd, value):
            signals.append(value)
            native_signal(fd, value)
            if value == signal.SIGSTOP:
                if fault == "post_stop_error": raise OSError("STOP delivered before error")
                if fault == "post_stop_sigterm": os.kill(os.getpid(), signal.SIGTERM)
        monkeypatch.setattr(controller, "_signal", inject)
        def validate():
            if fault == "validator": raise ValueError("final rejected")
            return {}
        with pytest.raises((OSError, InterruptedError, TimeoutError, ValueError)):
            controller.hold(supervisor, worker, intent=tmp_path / "intent.json", validate_completed=validate,
                timeout=.04 if fault == "timeout" else 3, poll=.005)
        assert signal.SIGCONT in signals
        # It may complete/dispatch the old terrain on failure; only the
        # successful native normal-end branch promises no next dispatch.
        identity = controller.process_identity(supervisor["pid"])
        if identity is not None:
            assert identity == supervisor and controller._state(supervisor["pid"]) not in ("T", "t")
        assert controller.process_identity(worker["pid"]) is None or controller.process_identity(worker["pid"]) == worker


def test_reused_pid_identity_refuses_before_signal(tmp_path, monkeypatch):
    with owners(tmp_path, 1) as (_, supervisor, worker):
        supervisor = dict(supervisor, starttime=supervisor["starttime"] + 1)
        monkeypatch.setattr(controller, "_signal", lambda *_: pytest.fail("PID mismatch cannot signal"))
        with pytest.raises(ValueError, match="PID reuse"):
            controller.hold(supervisor, worker, intent=tmp_path / "intent.json", validate_completed=lambda: {}, timeout=1)
        assert not (tmp_path / "intent.json").exists()


def test_existing_once_only_intent_cannot_signal(tmp_path, monkeypatch):
    with owners(tmp_path, 1) as (_, supervisor, worker):
        path = tmp_path / "intent.json"
        path.write_text('{"original":true}')
        before = path.read_bytes()
        monkeypatch.setattr(controller, "_signal", lambda *_: pytest.fail("Existing intent cannot signal"))
        with pytest.raises(ValueError, match="already exists"):
            controller.hold(supervisor, worker, intent=path, validate_completed=lambda: {}, timeout=1)
        assert path.read_bytes() == before


@pytest.mark.parametrize("intent_name", ["same.json", "different.json"])
def test_one_run_controller_lock_covers_all_intent_names(tmp_path, monkeypatch, intent_name):
    monkeypatch.setattr(controller, "_run", lambda *_args, **_kwargs: pytest.fail("Second controller cannot enter"))
    with (tmp_path / "native_speed_boundary_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            controller.run(tmp_path, tmp_path, tmp_path / intent_name, apply=True)


def test_partial_evidence_write_is_never_published_and_can_retry(tmp_path, monkeypatch):
    path = tmp_path / "entry.json"
    actual_fdopen = revision.os.fdopen
    class PartialWrite:
        def __init__(self, fd, mode): self.stream = actual_fdopen(fd, mode)
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def write(self, value):
            self.stream.write(value[:len(value)//2])
            raise OSError("injected partial certificate write")
    with monkeypatch.context() as m:
        m.setattr(revision.os, "fdopen", PartialWrite)
        with pytest.raises(OSError, match="partial certificate"):
            revision._retain(path, b'{"complete":true}\n')
    assert not path.exists() and not list(tmp_path.glob("*.pending.*"))
    revision._retain(path, b'{"complete":true}\n')
    revision._retain(path, b'{"complete":true}\n')
    assert path.read_bytes() == b'{"complete":true}\n'
    with pytest.raises(ValueError, match="different bytes"):
        revision._retain(path, b'{"different":true}\n')
    assert path.read_bytes() == b'{"complete":true}\n'


def test_atomic_exclusive_publish_race_retains_other_evidence(tmp_path, monkeypatch):
    path = tmp_path / "entry.json"
    link = revision.os.link
    def race(source, target):
        Path(target).write_bytes(b"original error evidence")
        link(source, target)
    monkeypatch.setattr(revision.os, "link", race)
    with pytest.raises(ValueError, match="different bytes"):
        revision._retain(path, b"correct candidate")
    assert path.read_bytes() == b"original error evidence"


@pytest.mark.parametrize("timeout", [0., -1., float("nan"), float("inf")])
def test_unbounded_wait_is_rejected_before_admission(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(controller, "_run", lambda *_args, **_kwargs: pytest.fail("Invalid wait cannot admit"))
    with pytest.raises(ValueError, match="finite positive"):
        controller.run(tmp_path, tmp_path, tmp_path / "intent.json", apply=True, timeout=timeout)
    assert not (tmp_path / "native_speed_boundary_controller.lock").exists()
