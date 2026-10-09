import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("script_diagnostic", Path(__file__).parents[1] /
                                            "scripts/replay_recovery_script_diagnostic.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_wrapper_preserves_calls_arguments_and_return_identity():
    class Target:
        def method(self, value, *, scale):
            self.calls += 1
            value.mul_(scale)
            return value
    target = Target()
    target.calls = 0
    value = torch.arange(3.)
    snapshots = []
    with diagnostic.observe_method(target, "method", before=lambda *a, **kw: snapshots.append(a[0].clone()),
                                   after=lambda result, *a, **kw: snapshots.append(result.clone())):
        assert target.method(value, scale=2) is value
    assert target.calls == 1
    assert "method" not in target.__dict__
    assert torch.equal(snapshots[0], torch.arange(3.))
    assert torch.equal(snapshots[1], torch.arange(3.) * 2)


def test_wrapper_restores_on_failure():
    class Target:
        def method(self):
            raise RuntimeError("original failure")
    target = Target()
    with pytest.raises(RuntimeError, match="original failure"):
        with diagnostic.observe_method(target, "method"):
            target.method()
    assert "method" not in target.__dict__


def test_wrapper_restores_existing_instance_method():
    class Target:
        pass
    target = Target()
    original = lambda: 42
    target.method = original
    with diagnostic.observe_method(target, "method"):
        assert target.method() == 42
    assert target.method is original


def test_buffer_owns_copies_and_keeps_exact_selected_order():
    buffer = diagnostic.SelectedBuffer()
    source = torch.arange(28 * 6).reshape(28, 6).float()
    expected = source[list(diagnostic.LANES)].clone()
    buffer.capture("q", source, 0)
    source.zero_()
    assert torch.equal(buffer.arrays["q"][0], expected)


@pytest.mark.parametrize("kwargs", [dict(capacity=1601), dict(device="cuda"), dict(lanes=(12,))])
def test_buffer_rejects_different_scope(kwargs):
    with pytest.raises(ValueError):
        diagnostic.SelectedBuffer(**kwargs)


def test_buffer_rejects_overflow_duplicate_skipped_or_wrong_shape():
    buffer = diagnostic.SelectedBuffer()
    source = torch.zeros(28)
    buffer.capture("q", source, 0)
    for value, row in [(source, 0), (source, 2), (source, 1600), (torch.zeros(4096), 1)]:
        with pytest.raises(ValueError):
            buffer.capture("q", value, row)


def test_export_requires_actual_complete_original_window(tmp_path):
    buffer = diagnostic.SelectedBuffer()
    buffer.capture("q", torch.zeros(28), 0)
    with pytest.raises(ValueError, match="every original"):
        buffer.export(tmp_path, 1600)
    assert not list(tmp_path.iterdir())


def test_no_inner_loop_host_export_or_second_sensor_update():
    import ast
    tree = ast.parse(Path(diagnostic.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ScriptRecorder")
    forbidden = {"cpu", "item", "tolist", "numpy", "sample", "update", "project", "observe", "step"}
    for node in ast.walk(cls):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden


def fake_env():
    from types import SimpleNamespace as S
    from wheeled_tasks.chassis.recovery import RecoveryFSM
    fsm = RecoveryFSM(28, "cpu", {}, [0.] * 6, [0.] * 6, [0.] * 6, [0.] * 6,
                               axis_signs=[1.] * 4)
    class Recovery:
        def advance_reference(self, q, dq, gravity, gyro):
            self.reference_calls += 1
            self.encoders.q.copy_(q)
            self.fsm.continuous_q.copy_(q)
            self.fsm.script_targets.add_(.1)
        def torque(self, q, dq, rl_torque):
            self.torque_calls += 1
            self.elapsed.add_(.005)
            return rl_torque
    class Robot:
        def set_joint_effort_target_index(self, *, target):
            self.calls += 1
            return target
    class Monitor:
        def observe(self, *args):
            self.calls += 1
    rec = Recovery()
    rec.reference_calls = rec.torque_calls = 0
    rec.reference_stride = 1
    rec.fsm = fsm
    rec.elapsed = torch.zeros(28)
    rec.encoders = S(q=torch.zeros(28, 6))
    rec.release_finished = torch.ones(28, dtype=torch.bool)
    rec.observed_height = torch.full((28,), .305)
    rec.observed_support = rec.wheel_balance_active = torch.ones(28, dtype=torch.bool)
    rec.wheel_probe_torque = torch.zeros(28, 2)
    rec.probe = S(evidence=torch.ones(28, 2, 2, dtype=torch.bool), age=torch.zeros(28, 2, 2),
                  bad_samples=torch.zeros(28, 2, 2, dtype=torch.long))
    rec.learning_mask = torch.ones(28, dtype=torch.bool)
    robot, monitor = Robot(), Monitor()
    robot.calls = monitor.calls = 0
    robot.data = S(joint_pos=S(torch=torch.zeros(28, 6)))
    groups = ["stand_305mm"] * 12 + ["recovery_left_side_90_stand"] * 4 + [
        "recovery_right_side_90_stand"] * 4 + ["other"] * 8
    return S(num_envs=28, device="cpu", design_preflight={"role": "baseline"}, dt=.005,
        policy_dt=.02, recovery=rec, robot=robot, torque_monitor=monitor, scene_groups=groups,
        cfg={"auto_reset": False}, ids=list(range(6)), command_transport=None, usb_transport=None,
        actuator_response=None, v5=S(requested_motor_effort=torch.zeros(28, 6),
                                   current_motor_bounds=torch.ones(28, 6)))


def test_recorder_delegates_full_pipeline_once_and_restores():
    env = fake_env()
    recorder = diagnostic.ScriptRecorder(env)
    recorder.install()
    q, dq = torch.zeros(28, 6), torch.ones(28, 6)
    vector = torch.zeros(28, 3)
    result = torch.ones(28, 6)
    env.recovery.advance_reference(q, dq, vector, vector)
    assert env.recovery.torque(q, dq, result) is result
    assert env.robot.set_joint_effort_target_index(target=result) is result
    spring = torch.zeros(28, 2)
    env.torque_monitor.observe(result, dq, spring, spring, spring, result, result)
    result.zero_()
    assert recorder.stages == {"reference": 1, "torque": 1, "issued": 1, "post_physics": 1}
    assert env.recovery.reference_calls == env.recovery.torque_calls == env.robot.calls == env.torque_monitor.calls == 1
    assert recorder.buffer.arrays["issued_motor_effort_nm"][0].eq(1).all()
    assert recorder.buffer.arrays["post_physics_applied_motor_nm"][0].eq(1).all()
    recorder.close()
    for obj, name in [(env.recovery, "advance_reference"), (env.recovery, "torque"),
                      (env.robot, "set_joint_effort_target_index"), (env.torque_monitor, "observe")]:
        assert name not in obj.__dict__


@pytest.mark.parametrize("mutation", ["count", "role", "order", "monitor"])
def test_recorder_refuses_wrong_native_scope(mutation):
    env = fake_env()
    if mutation == "count":
        env.num_envs = 4096
    elif mutation == "role":
        env.design_preflight["role"] = "train"
    elif mutation == "order":
        env.scene_groups[12] = "other"
    else:
        env.torque_monitor = None
    with pytest.raises(ValueError):
        diagnostic.ScriptRecorder(env)


def test_complete_prefix_export_and_channel_aliases(tmp_path):
    import numpy as np
    class Channel:
        def __init__(self):
            self.calls = 0
            self.state = torch.zeros(28, 6)
        def apply_torque(self, torque, velocity):
            self.calls += 1
            self.state.copy_(torque * .5)
            return self.state
    env = fake_env()
    env.command_transport = Channel()
    recorder = diagnostic.ScriptRecorder(env)
    recorder.install()
    q, dq = torch.zeros(28, 6), torch.ones(28, 6)
    vector, spring = torch.zeros(28, 3), torch.zeros(28, 2)
    result = torch.ones(28, 6)
    env.recovery.probe.age.fill_(float("inf"))
    for _ in range(1600):
        env.recovery.advance_reference(q, dq, vector, vector)
        assert env.recovery.torque(q, dq, result) is result
        assert env.command_transport.apply_torque(result, dq) is env.command_transport.state
        env.robot.set_joint_effort_target_index(target=result)
        env.torque_monitor.observe(result, dq, spring, spring, spring, result, result)
    recorder.close()
    env.command_transport.state.zero_()
    receipt = recorder.buffer.export(tmp_path, 1600)
    assert receipt["nonfinite_counts"] == {"probe_age_s": 1600 * 5 * 2 * 2}
    assert env.command_transport.calls == 1600
    assert set(recorder.stages.values()) == {1600}
    with np.load(receipt["file"]) as arrays:
        assert arrays["command_transport_output_nm"].shape == (1600, 5, 6)
        assert np.all(arrays["command_transport_output_nm"] == .5)
        assert np.all(arrays["first_episode_alive"])


def test_unexpected_nonfinite_export_rejected(tmp_path):
    buffer = diagnostic.SelectedBuffer()
    value = torch.full((28,), float("nan"))
    for row in range(1600):
        buffer.capture("q", value, row)
    with pytest.raises(ValueError, match="nonfinite"):
        buffer.export(tmp_path, 1600)
    assert not list(tmp_path.iterdir())


def test_process_identity_waits_for_exec_without_weakening_argv(monkeypatch):
    from types import SimpleNamespace
    observations = iter([[], ["exact", "child"]])
    monkeypatch.setattr(diagnostic, "read_process_identity", lambda pid:
                        dict(pid=pid, starttime=1234, command=next(observations)))
    monkeypatch.setattr(diagnostic.time, "sleep", lambda _: None)
    process = SimpleNamespace(pid=99, poll=lambda: None)
    result = diagnostic.wait_process_identity(process, ["exact", "child"])
    assert result == dict(pid=99, starttime=1234, command=["exact", "child"], identity_read_attempts=2)


def test_process_identity_rejects_exited_child():
    from types import SimpleNamespace
    process = SimpleNamespace(pid=99, poll=lambda: 1)
    with pytest.raises(ValueError, match="exited before"):
        diagnostic.wait_process_identity(process, ["exact", "child"])


def test_export_closes_recorder_before_native_shutdown(tmp_path):
    from types import SimpleNamespace as S
    (tmp_path / "native_replay.json").write_text('{"status":"native_mean_replay_completed_unscored"}\n')
    calls = []
    recorder = S(buffer=S(export=lambda output, count: calls.append((output, count)) or {"rows": count}),
                 stages={"reference": 1600}, close=lambda: calls.append("restored"))
    diagnostic.export_recording(recorder, tmp_path, Path(diagnostic.__file__))
    assert calls == [(tmp_path, 1600), "restored"]
    import json
    assert json.loads((tmp_path / "script_diagnostic.json").read_text())["export_phase"].endswith("before_Kit_shutdown")


@pytest.mark.parametrize("failure_point", ["buffer", "counts", "restore", "write"])
def test_diagnostic_export_failure_allows_native_cleanup(tmp_path, monkeypatch, failure_point):
    from types import SimpleNamespace as S
    import json
    (tmp_path / "native_replay.json").write_text('{}\n')
    calls = []
    def export(output, count):
        if failure_point == "buffer":
            raise ValueError("invalid diagnostic buffer")
        return {"rows": count}
    def close():
        calls.append("restore")
        if failure_point == "restore":
            raise RuntimeError("restoration failed")
    original_write = diagnostic.write
    def write(path, value):
        if failure_point == "write" and path.name == "script_diagnostic.json":
            raise OSError("export disk failure")
        return original_write(path, value)
    monkeypatch.setattr(diagnostic, "write", write)
    recorder = S(buffer=S(export=export), stages={"reference": 1599 if failure_point == "counts" else 1600},
                 close=close)
    assert diagnostic.export_before_shutdown(recorder, tmp_path, Path(diagnostic.__file__)) is False
    calls.append("native_cleanup")
    assert calls[-1] == "native_cleanup" and "restore" in calls
    assert not (tmp_path / "script_diagnostic.json").exists()
    assert json.loads((tmp_path / "diagnostic_export_failed.json").read_text())["status"] == "diagnostic_export_failed"


def test_missing_diagnostic_failure_receipt_still_allows_native_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace as S
    calls = []
    def fail(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(diagnostic, "write", fail)
    recorder = S(buffer=S(export=fail), stages={"reference": 1600}, close=lambda: calls.append("restore"))
    assert diagnostic.export_before_shutdown(recorder, tmp_path, Path(diagnostic.__file__)) is False
    calls.append("native_cleanup")
    assert calls == ["restore", "native_cleanup"]


def test_write_error_after_complete_file_retains_explicit_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace as S
    (tmp_path / "native_replay.json").write_text('{}\n')
    original_write = diagnostic.write
    def write(path, value):
        original_write(path, value)
        if path.name == "script_diagnostic.json":
            raise OSError("late write error")
    monkeypatch.setattr(diagnostic, "write", write)
    recorder = S(buffer=S(export=lambda *args: {"rows": 1600}), stages={"reference": 1600}, close=lambda: None)
    assert diagnostic.export_before_shutdown(recorder, tmp_path, Path(diagnostic.__file__)) is False
    assert not (tmp_path / "script_diagnostic.json").exists()
    assert (tmp_path / "diagnostic_export_failed.json").exists()
