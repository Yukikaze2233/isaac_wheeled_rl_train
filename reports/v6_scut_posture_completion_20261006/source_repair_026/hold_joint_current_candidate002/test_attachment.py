"""CPU attachment checks. No driver.main, App, GPU or native scene executes."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
space = {"__file__": str(HERE / "attach_holding.py")}
exec(compile((HERE / "attach_holding.py").read_bytes(), str(HERE / "attach_holding.py"), "exec"), space)
adapter = SimpleNamespace(**space)
FIXTURE = HERE.parent / "hold_joint_trace001/test_holding_trace.py"
native_fixture = adapter.load_bound(FIXTURE,
    "6dceb8b0ece63e5fbfa876ae12af38985f4605c01a794a1cd88a40300b2b3d90", "bound_original_cpu_fixture")
MANIFEST = HERE / "attachment_manifest001.json"


class Parent:
    def __init__(self, env, output):
        self.env = env
        self.errors = []
        self.parent_observes = 0
        self.parent_cleanup = 0
        self.result = {"original_parent_result": True}
        self.observe_result = object()
        self.parent_hooks = native_fixture.mechanical.Hooks()
        self.parent_begin_calls = 0

        def begin(original):
            def call(*args, **kwargs):
                self.parent_begin_calls += 1
                return original(*args, **kwargs)
            return call
        self.parent_hooks.bind(env.recovery, "begin_policy", begin)

    def observe(self, tick, extras):
        self.parent_observes += 1
        return self.observe_result

    def _invalid(self, phase, error):
        self.errors.append((phase, error))

    def _before_close(self):
        if not self.parent_cleanup:
            self.parent_hooks.restore()
            self.parent_cleanup += 1

    def finish(self, output):
        self._before_close()
        return self.result


def environment():
    env = native_fixture.env()
    env.scene_groups = ["CPU_fixture_REC_" + str(i) for i in range(env.num_envs)]
    env.kinds = ["CPU_fixture_flat"] * env.num_envs
    env.recovery.pose_curriculum = SimpleNamespace(level_ids=torch.tensor([0, 2, 1, 2]),
                                                    profile_ids=torch.arange(env.num_envs))
    return env


def rows(env, lanes=(1, 3)):
    return [{"lane": lane, "group": env.scene_groups[lane], "terrain_kind": env.kinds[lane]} for lane in lanes]


def attached_parent(env, output, expected=None):
    cls = adapter.observer_class(Parent, {}, {}, lanes=(1, 3), expected_rows=rows(env) if expected is None else expected,
                                journal=native_fixture.trace, hooks=native_fixture.mechanical)
    return cls(env, output)


def test_actual_binding_and_original_driver_factory_preserved():
    manifest, binding = adapter.authenticate(MANIFEST, adapter.sha(MANIFEST))
    driver = adapter.load_bound(manifest["driver"], binding[manifest["driver"]], "bound_CPU_original_source023_driver")
    original = driver.source023_runtime_observer
    identity = {"observer_artifact_sha256": {}}
    with adapter.attached(driver, MANIFEST, adapter.sha(MANIFEST)) as value:
        assert value is driver
        cls = driver.source023_runtime_observer(Parent, identity, {})
        assert [base.__name__ for base in cls.__mro__[:3]] == ["SelectedHoldingObserver", "Source023Observer", "Parent"]
        assert identity["observer_artifact_sha256"] == binding
    assert driver.source023_runtime_observer is original
    assert not torch.cuda.is_initialized()


def test_frozen_native_calls_bytes_rng_and_original_return_are_unchanged(tmp_path):
    states = []
    for observed in (False, True):
        output = tmp_path / str(observed)
        output.mkdir()
        torch.manual_seed(2294)
        env = environment()
        original_begin = env.recovery.begin_policy
        observer = attached_parent(env, output) if observed else Parent(env, output)
        start_rng = torch.get_rng_state().clone()
        for _ in range(3):
            extras = native_fixture.cycle(env, broken_substep=3)
            before = native_fixture.tensor_bytes(extras)
            assert observer.observe(env.tick, extras) is observer.observe_result
            assert native_fixture.tensor_bytes(extras) == before
        assert observer.finish(output) is observer.result
        assert not observer.errors
        assert observer.parent_observes == observer.parent_begin_calls == 3
        assert observer.parent_cleanup == 1
        assert env.recovery.begin_policy is original_begin
        assert torch.equal(start_rng, torch.get_rng_state())
        states.append((native_fixture.tensor_bytes(vars(env.recovery.fsm)),
                       native_fixture.tensor_bytes(vars(env.recovery)), env.recovery.original_calls))
    assert states[0] == states[1]
    saved = json.loads((tmp_path / "True/holding_summary.json").read_text())
    assert saved["interval_records_written"] == 3 and saved["hooks_restored"]
    assert saved["records"] == [] and saved["dropped_policies"] == 0
    assert saved["qualification"] is None and not saved["native_execution_verified"]


def test_streamed_clock_lesson_and_birth_alignment(tmp_path):
    env = environment()
    observer = attached_parent(env, tmp_path)
    extras = native_fixture.cycle(env, broken_substep=3, reset_ids=[1])
    observer.observe(env.tick, extras)
    observer.finish(tmp_path)
    with gzip.open(tmp_path / "holding_intervals.jsonl.gz", "rt") as stream:
        record = json.loads(next(stream))
        assert next(stream, None) is None
    assert record["birth_at_begin"] == [0, 0]
    assert record["selected_rows"] == rows(env)
    updates = [x for x in record["events"] if x["kind"] == "fsm_reference_5ms"]
    assert [x["reference_index"] for x in updates] == [1, 2, 3, 4]
    assert [x["nominal_observation_physical_endpoint"] for x in updates] == [0, 1, 2, 3]
    assert all(x["before"]["lesson_id"] == [2, 2] and x["after"]["profile_id"] == [1, 3] for x in updates)
    outcome = next(x for x in record["events"] if x["kind"] == "truth_outcome_20ms")
    assert outcome["physical_endpoint"] == 4
    assert outcome["inputs"]["contact"][0] == [False, True]
    assert "contact" not in updates[-1]["inputs"]
    assert record["actor_valid_whole_policy"] == extras["policy_learning_mask"][[1, 3]].tolist()
    assert record["complete_reference_and_outcome_counts"]


def test_observe_error_is_original_exception_and_cleanup_restores(tmp_path):
    env = environment()
    original = env.recovery.begin_policy
    observer = attached_parent(env, tmp_path)
    failure = RuntimeError("CPU serialization failed")
    observer.hold_trace.after_policy = lambda *args: (_ for _ in ()).throw(failure)
    with pytest.raises(RuntimeError) as caught:
        observer.observe(1, {})
    assert caught.value is failure
    observer._before_close()
    observer._before_close()
    assert env.recovery.begin_policy is original
    assert observer.parent_cleanup == 1 and observer.hold_trace.closed
    assert observer.errors[0][0] == "selected_hold_observe"


def test_failed_constructor_preserves_parent_cleanup_without_native_row_fabrication(tmp_path):
    env = environment()
    original = env.recovery.begin_policy
    with pytest.raises(ValueError, match="scene rows differ"):
        attached_parent(env, tmp_path, expected=[{"lane": 1, "group": "fabricated", "terrain_kind": "flat"}])
    assert env.recovery.begin_policy is original
    assert not (tmp_path / "holding_intervals.jsonl.gz").exists()


def test_failed_stream_constructor_restores_both_hook_layers(tmp_path, monkeypatch):
    env = environment()
    original = (env.recovery.begin_policy, env.recovery.fsm.update, env.recovery.reset)
    failure = OSError("CPU stream construction failed")
    monkeypatch.setattr(adapter.gzip, "open", lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    with pytest.raises(OSError) as caught:
        attached_parent(env, tmp_path)
    assert caught.value is failure
    assert (env.recovery.begin_policy, env.recovery.fsm.update, env.recovery.reset) == original


def test_streaming_memory_remains_bounded_beyond_original_history_capacity(tmp_path):
    env = environment()
    observer = attached_parent(env, tmp_path)
    for _ in range(70):
        extras = native_fixture.cycle(env)
        observer.observe(env.tick, extras)
        assert len(observer.hold_trace.records) == 0
        assert observer.hold_trace.interval is None
        assert len(observer.hold_trace.between) <= observer.hold_trace.capacity
    observer.finish(tmp_path)
    with gzip.open(tmp_path / "holding_intervals.jsonl.gz", "rt") as stream:
        assert sum(1 for _ in stream) == 70
    summary = json.loads((tmp_path / "holding_summary.json").read_text())
    assert summary["calls"]["fsm_update"] == 280
    assert summary["dropped_policies"] == summary["dropped_events"] == 0


def test_factory_cleanup_after_exception_and_reject_alien_owner():
    manifest, binding = adapter.authenticate(MANIFEST, adapter.sha(MANIFEST))
    driver = adapter.load_bound(manifest["driver"], binding[manifest["driver"]], "bound_CPU_factory_cleanup")
    original = driver.source023_runtime_observer
    failure = RuntimeError("original driver body fixture failed")
    with pytest.raises(RuntimeError) as caught:
        with adapter.attached(driver, MANIFEST, adapter.sha(MANIFEST)):
            raise failure
    assert caught.value is failure and driver.source023_runtime_observer is original
    alien = object()
    with pytest.raises(ValueError, match="ownership changed"):
        with adapter.attached(driver, MANIFEST, adapter.sha(MANIFEST)):
            driver.source023_runtime_observer = alien
    assert driver.source023_runtime_observer is alien


def test_bad_manifest_sha_and_missing_real_input_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="manifest identity"):
        adapter.authenticate(MANIFEST, "0" * 64)
    value = json.loads(MANIFEST.read_text())
    value["files_sha256"][str(tmp_path / "absent_callback.py")] = "0" * 64
    altered = tmp_path / "missing_input_manifest.json"
    altered.write_text(json.dumps(value))
    with pytest.raises(FileNotFoundError):
        adapter.authenticate(altered, adapter.sha(altered))


def test_attachment_has_no_driver_gpu_control_or_query_path():
    tree = ast.parse((HERE / "attach_holding.py").read_bytes())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            assert all(key not in name for key in ("driver.main", "AppLauncher", "sim.step", "robot.data", "get_observations",
                                                   "motor_efforts", "set_joint", "manual_seed", "subprocess", "Popen", "cuda"))
    assert not torch.cuda.is_initialized()
