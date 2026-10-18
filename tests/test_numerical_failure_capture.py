"""CPU first-event evidence and direct-Kit-exit cleanup; no native simulation."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.numerical_failure_capture import (
    NumericalFailureCapture, classify_source_numerical,
)
from wheeled_tasks.chassis.scut_flat_reward_runtime import (
    ROOT_VELOCITY_REFERENCE, WHEEL_POSITION_REFERENCE, ScutFlatRewardRuntime,
)


ROOT = Path(__file__).resolve().parents[1]


def test_classification_matches_original_source_and_strict_thresholds():
    count = 7
    runtime = ScutFlatRewardRuntime(count, "cpu", ["base_link", "L_wheel", "R_wheel"], {
        "root_velocity_reference": ROOT_VELOCITY_REFERENCE,
        "root_height_reference": "root_link_z_minus_env_origin",
        "wheel_position_reference": WHEEL_POSITION_REFERENCE,
        "undesired_link_names": ["base_link"], "terminal_link_names": ["base_link"]})
    signals = [torch.zeros(count, width) for width in (6, 3, 3)]
    for i, (value, limit) in enumerate(zip(signals, (500., 100., 200.))):
        value[0, 0] = limit
        value[1 + i, 0] = -limit - 1.
        value[4 + i, 1] = float("nan") if i == 0 else float("inf")
    runtime.observe_physics(signals[0], torch.zeros(count, 6), torch.zeros(count, 6))
    q = torch.tensor([[0., 0., 0., 1.]]).repeat(count, 1)
    _, original, _ = runtime.termination(signals[1], signals[2], q)
    mask, fields = classify_source_numerical(*signals)
    assert torch.equal(mask, original)
    assert mask.tolist() == [False, True, True, True, True, True, True]
    assert fields["active_joint_velocity_rad_s"]["nonfinite"][4]
    assert not fields["active_joint_velocity_rad_s"]["over_limit"][4]
    assert fields["root_com_velocity_body_m_s"]["axis_over_limit"][2, 0]


def test_prior_states_survive_in_place_changes_reset_and_later_events(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 2, {"physics_dt_s": .005})
    q = torch.zeros(2, 3)
    episode = torch.tensor([7, 8])
    sent = torch.zeros(2, 6)
    state = lambda: {"q": q, "episode_id": episode, "sent_effort": sent}
    for tick in range(3):
        q.fill_(10. + tick)
        sent.fill_(20. + tick)
        capture.remember_before_physics(tick, 0, state())
        q.fill_(-999.)
        sent.fill_(-888.)
    q.fill_(501.)
    sent.fill_(22.)
    signals = [torch.zeros(2, width) for width in (6, 3, 3)]
    signals[0][1, 2] = 501.
    capture.observe_after_physics(3, 0, signals, state)
    first_sha = hashlib.sha256(capture.path.read_bytes()).hexdigest()
    # Emulate lane reset and another worse event: neither prior nor current is aliased.
    q.zero_(); episode.add_(100); sent.zero_()
    signals[0].fill_(1000.)
    capture.remember_before_physics(4, 1, state())
    capture.observe_after_physics(5, 1, signals, state)
    assert hashlib.sha256(capture.path.read_bytes()).hexdigest() == first_sha
    saved = torch.load(capture.path, weights_only=True)
    assert saved["lane_ids"].tolist() == [1]
    assert [row["physical_tick"] for row in saved["prior_frames"]] == [0, 1, 2]
    assert [row["next_physical_tick"] for row in saved["prior_frames"]] == [1, 2, 3]
    for tick, row in zip((0, 1, 2), saved["prior_frames"]):
        assert torch.equal(row["state"]["q"], torch.full((1, 3), 10. + tick))
        assert torch.equal(row["state"]["sent_effort"], torch.full((1, 6), 20. + tick))
    assert saved["current"]["state"]["episode_id"].tolist() == [8]
    assert saved["current"]["state"]["q"].tolist() == [[501.] * 3]
    assert saved["original_policy_termination_evaluated_at_dump"] is False
    # A transient substep event is evidence, not proof the policy-end stop fired.
    capture.observe_original_policy_end(4, 1, torch.zeros(2, dtype=torch.bool))
    assert capture.summary["recorded_lanes_original_numerical"] == [False]
    assert capture.summary["termination_changed"] is False


def test_history_wrap_reuses_buffers_and_freezes_selected_lane_first_event(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 2, {"physics_dt_s": .005})
    state = {"q": torch.zeros(2, 3), "episode_id": torch.tensor([7, 8]),
        "sent_effort": torch.zeros(2, 6)}
    capacity = NumericalFailureCapture.HISTORY_FRAMES
    assert capacity == capture.summary["history_frames"] == 64
    with torch.inference_mode():
        for tick in range(capacity + 3):
            state["q"].fill_(float(tick))
            state["sent_effort"].fill_(100. + tick)
            if tick == capacity - 4:
                state["episode_id"][1] = 9
            capture.remember_before_physics(tick, tick // 4, state)
            if tick == capacity - 1:
                buffers = [{key: value.data_ptr() for key, value in frame["state"].items()}
                    for frame in capture.frames]
            # In-place source updates must not change any retained state.
            state["q"].fill_(-999.)
            state["sent_effort"].fill_(-888.)
        assert len(capture.frames) == capacity and capture.cursor == 3
        assert buffers == [{key: value.data_ptr() for key, value in frame["state"].items()}
            for frame in capture.frames]
        state["q"].fill_(501.)
        signals = tuple(torch.zeros(2, width) for width in (6, 3, 3))
        signals[0][1, 2] = 501.
        capture.observe_after_physics(capacity + 3, (capacity + 3) // 4,
            signals, lambda: state)
        first_sha = hashlib.sha256(capture.path.read_bytes()).hexdigest()
        # Continue in record-only mode through another complete ring overwrite.
        for tick in range(capacity + 3, 2 * capacity + 3):
            state["q"].fill_(float(tick))
            capture.remember_before_physics(tick, tick // 4, state)
        capture.observe_after_physics(2 * capacity + 3, (2 * capacity + 3) // 4,
            signals, lambda: state)
    assert hashlib.sha256(capture.path.read_bytes()).hexdigest() == first_sha
    saved = torch.load(capture.path, weights_only=True)
    assert saved["lane_ids"].tolist() == [1]
    assert saved["dump_physical_tick"] == capacity + 3
    assert saved["dump_policy_tick"] == (capacity + 3) // 4
    assert [row["physical_tick"] for row in saved["prior_frames"]] == list(range(3, capacity + 3))
    assert [row["next_physical_tick"] for row in saved["prior_frames"]] == list(range(4, capacity + 4))
    for row in saved["prior_frames"]:
        tick = row["physical_tick"]
        assert row["policy_tick"] == tick // 4
        assert row["state_phase"] == "before_next_physics"
        assert row["state"]["q"].tolist() == [[float(tick)] * 3]
        assert row["state"]["sent_effort"].tolist() == [[100. + tick] * 6]
        assert row["state"]["episode_id"].tolist() == [8 if tick < capacity - 4 else 9]
    current = saved["current"]
    assert current["physical_tick"] == current["control_targets_apply_to_interval_ending_at"] == capacity + 3
    assert current["state"]["q"].tolist() == [[501.] * 3]
    assert current["state"]["episode_id"].tolist() == [9]
    assert current["state"]["sent_effort"].tolist() == [[100. + capacity + 2] * 6]


def test_first_substep_can_be_recorded_with_only_one_prior_frame(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 1, {})
    state = {"q": torch.ones(1, 6)}
    capture.remember_before_physics(0, 0, state)
    capture.observe_after_physics(1, 0,
        (torch.full((1, 6), 501.), torch.zeros(1, 3), torch.zeros(1, 3)), lambda: state)
    saved = torch.load(capture.path, weights_only=True)
    assert len(saved["prior_frames"]) == 1
    assert saved["prior_frames"][0]["physical_tick"] == 0
    capture.observe_original_policy_end(4, 1, torch.ones(1, dtype=torch.bool))
    assert capture.summary["recorded_lanes_original_numerical"] == [True]


def test_closure_first_and_later_peak_keep_physical_frames_across_reset(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 2, {}, stop_on_first_event=True)
    state = {"q": torch.zeros(2, 8), "episode_id": torch.tensor([7, 8])}
    signals = tuple(torch.zeros(2, width) for width in (6, 3, 3))
    # Equality is observation only, and even a gap event must not fast-stop.
    capture.remember_before_physics(0, 0, state)
    capture.observe_after_physics(1, 0, signals, lambda: state,
        closure_gap=torch.tensor([.03, .001]))
    assert not capture.closure_first_path.exists()
    capture.remember_before_physics(1, 0, state)
    state["q"].fill_(11.)
    capture.observe_after_physics(2, 0, signals, lambda: state,
        closure_gap=torch.tensor([.039, .002]))
    first_sha = hashlib.sha256(capture.closure_first_path.read_bytes()).hexdigest()
    first = torch.load(capture.closure_first_path, weights_only=True)
    assert first["lane_ids"].tolist() == [0]
    assert first["dump_physical_tick"] == 2
    assert [row["physical_tick"] for row in first["prior_frames"]] == [0, 1]
    # A later different lane/reset becomes the maximum: preserve both events.
    state["episode_id"][1] = 9
    for tick in (2, 3):
        state["q"].fill_(float(tick))
        capture.remember_before_physics(tick, 0, state)
    state["q"].fill_(44.)
    capture.observe_after_physics(4, 0, signals, lambda: state,
        closure_gap=torch.tensor([.02, .082]))
    state["q"].zero_()
    state["episode_id"].add_(100)
    peak = torch.load(capture.closure_peak_path, weights_only=True)
    assert peak["lane_ids"].tolist() == [1]
    assert peak["dump_physical_tick"] == 4
    assert [row["physical_tick"] for row in peak["prior_frames"]] == [0, 1, 2, 3]
    assert [row["state"]["episode_id"].item() for row in peak["prior_frames"]] == [8, 8, 9, 9]
    assert peak["current"]["state"]["episode_id"].tolist() == [9]
    assert peak["current"]["state"]["q"].tolist() == [[44.] * 8]
    assert hashlib.sha256(capture.closure_first_path.read_bytes()).hexdigest() == first_sha
    assert capture.summary["closure"]["peak_replacements"] == 2
    assert not capture.recorded and not capture.summary["worker_abort_raised"]
    assert not capture.summary["closure"]["termination_changed"]


def test_lower_closure_observation_threshold_preserves_history_and_record_only_behavior(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 2, {}, stop_on_first_event=True,
        closure_observation_threshold_m=.01)
    state = {"q": torch.zeros(2, 8), "episode_id": torch.tensor([7, 8])}
    signals = tuple(torch.zeros(2, width) for width in (6, 3, 3))
    capture.remember_before_physics(0, 0, state)
    capture.observe_after_physics(1, 0, signals, lambda: state,
        closure_gap=torch.tensor([.01, .001]))
    assert not capture.closure_first_path.exists()
    state["q"].fill_(1.)
    capture.remember_before_physics(1, 0, state)
    state["q"].fill_(2.)
    capture.observe_after_physics(2, 0, signals, lambda: state,
        closure_gap=torch.tensor([.001, .0227]))
    saved = torch.load(capture.closure_first_path, weights_only=True)
    assert saved["threshold_m"] == .01 == capture.summary["closure"]["threshold_m"]
    assert saved["lane_ids"].tolist() == [1]
    assert [row["physical_tick"] for row in saved["prior_frames"]] == [0, 1]
    assert saved["prior_frames"][0]["state"]["q"].tolist() == [[0.] * 8]
    assert saved["prior_frames"][1]["state"]["q"].tolist() == [[1.] * 8]
    assert saved["current"]["state"]["q"].tolist() == [[2.] * 8]
    assert saved["current"]["state"]["episode_id"].tolist() == [8]
    assert torch.equal(state["episode_id"], torch.tensor([7, 8]))
    assert not saved["termination_changed"]
    assert not capture.recorded and not capture.summary["worker_abort_raised"]


@pytest.mark.parametrize("threshold", [True, False, 0., -.01, float("nan"),
    float("inf"), -float("inf"), "0.01", None])
def test_closure_observation_threshold_rejects_invalid_distances(tmp_path, threshold):
    with pytest.raises(ValueError, match="finite positive distance"):
        NumericalFailureCapture(tmp_path, 1, {}, closure_observation_threshold_m=threshold)


@pytest.mark.parametrize("options,threshold", [({}, .03),
    ({"closure_observation_threshold_m": .01}, .01),
    ({"stop_on_nonfinite": True}, .03)])
def test_env_enable_captures_cached_knee_and_constraint_metadata(tmp_path, options, threshold):
    source_path = ROOT / "src/wheeled_tasks/chassis/env.py"
    tree = ast.parse(source_path.read_text())
    env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
        and node.name == "ChassisEnv")
    method = next(node for node in env_class.body if isinstance(node, ast.FunctionDef)
        and node.name == "enable_numerical_failure_capture")
    namespace = {"Path": Path, "__file__": str(source_path),
        "__package__": "wheeled_tasks.chassis"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    env = SimpleNamespace(cfg={"formal_v7_flat": True,
        "formal_physics_trace_directory": str(tmp_path)}, scut_reward=object(), critic78=True,
        flat_episode=object(), dt=.005, policy_dt=.02, physical_tick=0, is_v5=True,
        robot=SimpleNamespace(joint_names=["L_joint2", "R_jonit2"], body_names=["base", "L", "R"]),
        model_spec={"constraints": [{"name": "left_closure"}]},
        manifest={"files_sha256": {}}, ids=[0, 1], spring_ids=[], wheel_ids=[], knee_ids=[0, 1],
        knee_bounds={"L_joint2": [-.08616, 1.08321], "R_jonit2": [-1.08321, .08616]},
        _closure_body_ids=torch.tensor([[1, 2]]), _closure_points=torch.zeros(1, 2, 3),
        num_envs=1, startup_report={})
    namespace["enable_numerical_failure_capture"](env, **options)
    capture = env._numerical_failure_capture
    assert capture.closure_observation_threshold_m == threshold
    assert capture.stop_on_nonfinite is options.get("stop_on_nonfinite", False)
    assert capture.summary["closure"]["threshold_m"] == threshold
    assert capture.metadata["closure_constraint_names"] == ["left_closure"]
    assert capture.metadata["knee_joint_ids"] == [0, 1]
    assert capture.metadata["knee_joint_names"] == ["L_joint2", "R_jonit2"]
    assert capture.metadata["knee_joint_bounds_rad"] == env.knee_bounds
    assert env.startup_report["numerical_failure_capture"] is capture.summary


@pytest.mark.parametrize("field", ["joint_position", "joint_velocity"])
def test_passive_nonfinite_stops_without_inventing_finite_speed_limit(tmp_path, field):
    capture = NumericalFailureCapture(tmp_path, 1, {}, stop_on_first_event=True)
    signals = tuple(torch.zeros(1, width) for width in (6, 3, 3))
    joints = torch.zeros(1, 8)
    joints[0, 7] = 1000.  # Finite passive value: no new physical threshold.
    capture.remember_before_physics(0, 0, {field: joints})
    capture.observe_after_physics(1, 0, signals, lambda: {field: joints}, **{field: joints})
    assert not capture.recorded
    joints[0, 7] = float("nan")
    with pytest.raises(FloatingPointError, match="first physical substep 2"):
        capture.observe_after_physics(2, 0, signals, lambda: {field: joints}, **{field: joints})
    saved = torch.load(capture.path, weights_only=True)
    assert saved["classification"]["all_" + field]["axis_nonfinite"][0, 7]
    assert not saved["classification"]["active_joint_velocity_rad_s"]["nonfinite"].any()
    assert capture.summary["all_joint_nonfinite_checked"]


@pytest.mark.parametrize("bad_gap", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("stop", [False, True])
def test_nonfinite_closure_is_numeric_even_when_other_readbacks_are_finite(tmp_path, bad_gap, stop):
    capture = NumericalFailureCapture(tmp_path, 2, {}, stop_on_first_event=stop)
    signals = tuple(torch.zeros(2, width) for width in (6, 3, 3))
    state = {"q": torch.zeros(2, 18), "dq": torch.zeros(2, 18),
        "body_link_pose_world": torch.zeros(2, 19, 7),
        "root_link_pose_world": torch.zeros(2, 7), "episode_id": torch.tensor([3, 4])}
    state["body_link_pose_world"][..., 6] = 1.
    state["root_link_pose_world"][:, 6] = 1.
    capture.remember_before_physics(0, 0, state)
    gap = torch.tensor([.082, bad_gap])

    def observe():
        capture.observe_after_physics(1, 0, signals, lambda: state, closure_gap=gap,
            joint_position=state["q"], joint_velocity=state["dq"])

    if stop:
        with pytest.raises(FloatingPointError, match="first physical substep 1"):
            observe()
    else:
        observe()
    saved = torch.load(capture.path, weights_only=True)
    assert saved["lane_ids"].tolist() == [1]
    assert saved["dump_physical_tick"] == 1
    classification = saved["classification"]
    assert classification["closure_gap_m"]["nonfinite"].tolist() == [True]
    assert not torch.isfinite(classification["closure_gap_m"]["value_m"]).any()
    assert all(not row["nonfinite"].any() for key, row in classification.items()
               if key != "closure_gap_m")
    assert all(torch.isfinite(value).all() for value in saved["current"]["state"].values())
    assert capture.summary["closure_nonfinite_checked"]
    assert capture.summary["worker_abort_raised"] is stop
    # The separate finite82mm lane is captured, never classified as numerical.
    finite_event = torch.load(capture.closure_peak_path, weights_only=True)
    assert finite_event["lane_ids"].tolist() == [0]
    assert finite_event["gap_m"].item() == pytest.approx(.082)
    assert finite_event["termination_changed"] is False


def test_post_physics_feedback_refresh_does_not_relabel_effort_inputs(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 1, {})
    state = {"q": torch.tensor([[1.]]), "feedback_q": torch.tensor([[.5]]),
        "feedback_dq": torch.tensor([[2.]]), "sent_effort": torch.tensor([[3.]])}
    capture.remember_before_physics(0, 0, state)
    # Real feedback transport exposes views then updates them after physics.
    state["q"].fill_(10.)
    state["feedback_q"].fill_(9.)
    state["feedback_dq"].fill_(20.)
    signals = (torch.full((1, 6), 501.), torch.zeros(1, 3), torch.zeros(1, 3))
    capture.observe_after_physics(1, 0, signals, lambda: state, closure_gap=torch.tensor([.04]))
    for path in (capture.path, capture.closure_first_path, capture.closure_peak_path):
        current = torch.load(path, weights_only=True)["current"]["state"]
        assert current["q"].item() == 10.
        assert current["feedback_q"].item() == .5
        assert current["feedback_dq"].item() == 2.
        assert current["sent_effort"].item() == 3.


@pytest.mark.parametrize("existing_peak", [False, True])
def test_closure_write_failure_never_publishes_partial_evidence(tmp_path, monkeypatch, existing_peak):
    capture = NumericalFailureCapture(tmp_path, 1, {})
    state = {"q": torch.zeros(1, 8)}
    signals = tuple(torch.zeros(1, width) for width in (6, 3, 3))
    capture.remember_before_physics(0, 0, state)
    if existing_peak:
        capture.observe_after_physics(1, 0, signals, lambda: state, closure_gap=torch.tensor([.039]))
        prior_sha = hashlib.sha256(capture.closure_peak_path.read_bytes()).hexdigest()
    def fail_save(payload, path):
        Path(path).write_bytes(b"partial invalid snapshot")
        raise OSError("simulated disk-full error")
    monkeypatch.setattr(torch, "save", fail_save)
    capture.observe_after_physics(2, 0, signals, lambda: state, closure_gap=torch.tensor([.082]))
    report = capture.summary["closure"]
    assert not report["latest_dump_succeeded"]
    assert report["observed_peak_m"] == pytest.approx(.082)
    assert report["first_dump_succeeded"] is existing_peak
    if existing_peak:
        assert hashlib.sha256(capture.closure_peak_path.read_bytes()).hexdigest() == prior_sha
        assert report["saved_peak_m"] == pytest.approx(.039)
        assert report["saved_peak_physical_tick"] == 1
    else:
        assert not capture.closure_first_path.exists() and not capture.closure_peak_path.exists()
        assert report["saved_peak_m"] is None
    assert not list(tmp_path.glob("*.tmp"))
    assert not capture.summary["worker_abort_raised"]


def test_explicit_fast_stop_saves_first_state_and_blocks_remaining_physics(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 1, {}, stop_on_first_event=True)
    state = {"q": torch.tensor([[float("nan")]])}
    physical_steps, returned_transitions = [], []
    with pytest.raises(FloatingPointError, match="first physical substep 1"):
        for tick in range(4):
            capture.remember_before_physics(tick, 0, state)
            physical_steps.append(tick + 1)
            capture.observe_after_physics(tick + 1, 0,
                (torch.full((1, 6), float("nan")), torch.zeros(1, 3), torch.zeros(1, 3)),
                lambda: state)
        returned_transitions.append("would_enter_rollout")
    assert physical_steps == [1] and not returned_transitions
    saved = torch.load(capture.path, weights_only=True)
    assert torch.isnan(saved["current"]["state"]["q"]).all()
    assert saved["failure_policy"] == "worker_abort_first_5ms_crossing"
    assert saved["not_root_cause_fix"] and not saved["diagnostic_only_does_not_terminate"]
    assert capture.summary["worker_abort_raised"] and capture.summary["dump_succeeded"]
    assert not capture.summary["source_policy_end_observed"]


def test_nonfinite_fast_stop_writes_completion_without_reading_bad_native_state(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 1, {}, stop_on_first_event=True)
    with pytest.raises(FloatingPointError):
        capture.observe_after_physics(1, 0,
            (torch.full((1, 6), float("nan")), torch.zeros(1, 3), torch.zeros(1, 3)),
            lambda: {"q": torch.full((1, 6), float("nan"))})
    tree = ast.parse((ROOT / "scripts/train_chassis.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = next(node for node in main.body if isinstance(node, ast.Try) and node.finalbody).finalbody[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "env is not None")
    events = []
    def bad_native_read():
        raise AssertionError("Must not read nonfinite terminal native metrics")
    def close_app(*, exit_code):
        events.append("kit")
        raise SystemExit(exit_code)
    env = SimpleNamespace(_numerical_failure_capture=capture, summary=bad_native_read,
        torque_monitor=SimpleNamespace(report=bad_native_read), close=lambda: events.append("env"))
    context = {"report": {"status": "failed", "error": "first physical numerical crossing",
        "successful_updates": 0, "parent_updates": 4701, "learning_lineage_updates": 4701}, "env": env,
        "args": SimpleNamespace(run_dir=tmp_path), "metrics": SimpleNamespace(report=bad_native_read),
        "training_control": SimpleNamespace(finish=lambda *a, **kw: None), "timing": None, "physx_errors": None, "datetime": datetime, "timezone": timezone, "json": json,
        "budget_owner": SimpleNamespace(summary=lambda: {"paid_updates_global": 4750},
            close=lambda: events.append("lease")),
        "launcher": SimpleNamespace(app=SimpleNamespace(close=close_app)), "print": lambda *a, **kw: None}
    with pytest.raises(SystemExit) as result:
        exec(compile(ast.Module(body=body[start:], type_ignores=[]), "actual_failed_cleanup", "exec"), context)
    assert result.value.code == 1 and events == ["env", "lease", "kit"]
    receipt = json.loads((tmp_path / "completion.json").read_text())
    assert receipt["status"] == "failed" and receipt["final_learning_lineage_updates"] == 4701
    assert receipt["training_budget"]["paid_updates_global"] == 4750
    assert receipt["metrics"]["available"] is False
    assert receipt["numerical_failure_capture"]["dump_succeeded"]
    assert not (tmp_path / "torque_monitor.json").exists()


@pytest.mark.parametrize("status,code", [("completed", 0), ("scene_validated", 0),
    ("stopped", 0), ("failed", 1), ("export_failed", 1)])
def test_kit_immediate_exit_follows_budget_close_with_correct_status(status, code):
    # Execute the production terminal cleanup with a Kit close that exits and
    # cannot return to the Python function's final return statement.
    tree = ast.parse((ROOT / "scripts/train_chassis.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = next(node for node in main.body if isinstance(node, ast.Try) and node.finalbody).finalbody[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "print" and any(isinstance(arg, ast.Constant)
            and arg.value == "CHASSIS_COMPLETION" for arg in node.value.args))
    events = []
    def close_app(*, exit_code):
        events.append("kit")
        raise SystemExit(exit_code)
    context = {"report": {"status": status, "successful_updates": 0}, "physx_errors": None,
        "budget_owner": SimpleNamespace(close=lambda: events.append("lease")),
        "launcher": SimpleNamespace(app=SimpleNamespace(close=close_app)), "print": lambda *a, **kw: None}
    with pytest.raises(SystemExit) as result:
        exec(compile(ast.Module(body=body[start:], type_ignores=[]), "actual_final_cleanup", "exec"), context)
    assert result.value.code == code
    assert events == ["lease", "kit"]
    assert context["budget_owner"] is None


@pytest.mark.parametrize("channel", ["active", "root_linear", "root_angular", "root_pose",
    "joint_position", "joint_velocity", "closure_gap"])
def test_recorded_finite_event_does_not_disable_later_nonfinite_stop(tmp_path, channel):
    capture = NumericalFailureCapture(tmp_path, 2, {}, stop_on_nonfinite=True)
    signals = tuple(torch.zeros(2, width) for width in (6, 3, 3))
    q, dq, pose = torch.zeros(2, 18), torch.zeros(2, 18), torch.zeros(2, 7)
    pose[:, 6] = 1.
    gap, episode = torch.zeros(2), torch.tensor([3, 4])
    state = {"q": q, "dq": dq, "root_link_pose_world": pose, "episode_id": episode}

    def observe(tick):
        capture.observe_after_physics(tick, tick // 4, signals, lambda: state,
            closure_gap=gap, joint_position=q, joint_velocity=dq, root_pose=pose)

    capture.remember_before_physics(0, 0, state)
    signals[0][0, 0] = 603.
    observe(1)  # Finite source overspeed is evidence, not a worker stop.
    first_sha = hashlib.sha256(capture.path.read_bytes()).hexdigest()
    first = torch.load(capture.path, weights_only=True)
    assert first["diagnostic_only_does_not_terminate"]
    assert first["lane_ids"].tolist() == [0]
    assert not capture.nonfinite_path.exists() and not capture.summary["worker_abort_raised"]
    capture.observe_original_policy_end(4, 1, torch.tensor([True, False]))
    assert capture.summary["recorded_lanes_original_numerical"] == [True]
    # A subsequent episode must remain protected after first evidence is frozen.
    signals[0].zero_()
    episode[0] += 1
    capture.remember_before_physics(4, 1, state)
    observe(5)
    capture.remember_before_physics(5, 1, state)
    value = {"active": signals[0], "root_linear": signals[1], "root_angular": signals[2],
        "root_pose": pose, "joint_position": q, "joint_velocity": dq, "closure_gap": gap}[channel]
    value[1] = float("nan") if channel in ("joint_position", "root_pose") else float("inf")
    with pytest.raises(FloatingPointError, match="nonfinite physical substep 6"):
        observe(6)
    saved = torch.load(capture.nonfinite_path, weights_only=True)
    assert saved["lane_ids"].tolist() == [1]
    assert saved["trigger"] == "first_nonfinite_physical_substep"
    assert saved["dump_physical_tick"] == 6 and saved["not_root_cause_fix"]
    assert saved["current"]["state"]["episode_id"].tolist() == [4]
    assert not saved["diagnostic_only_does_not_terminate"]
    assert hashlib.sha256(capture.path.read_bytes()).hexdigest() == first_sha
    assert capture.summary["physical_tick"] == 1
    assert capture.summary["nonfinite"]["physical_tick"] == 6
    assert capture.summary["worker_abort_raised"]
    # Even an accidental caught exception may neither replace evidence nor continue silently.
    nonfinite_sha = hashlib.sha256(capture.nonfinite_path.read_bytes()).hexdigest()
    with pytest.raises(FloatingPointError, match="nonfinite physical substep 7"):
        observe(7)
    assert hashlib.sha256(capture.path.read_bytes()).hexdigest() == first_sha
    assert hashlib.sha256(capture.nonfinite_path.read_bytes()).hexdigest() == nonfinite_sha


def test_nonfinite_as_first_event_uses_original_snapshot_once(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 1, {}, stop_on_nonfinite=True)
    signals = (torch.full((1, 6), float("nan")), torch.zeros(1, 3), torch.zeros(1, 3))
    with pytest.raises(FloatingPointError, match="nonfinite physical substep 1"):
        capture.observe_after_physics(1, 0, signals, lambda: {"q": torch.zeros(1, 18)})
    assert capture.path.exists() and not capture.nonfinite_path.exists()
    assert capture.summary["nonfinite"]["file"] == str(capture.path)
    assert capture.summary["nonfinite"]["dump_succeeded"]


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_nonfinite_stop_requires_explicit_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="nonfinite stop requires an explicit boolean"):
        NumericalFailureCapture(tmp_path, 1, {}, stop_on_nonfinite=value)


def test_simultaneous_first_finite_and_nonfinite_events_keep_distinct_lane_labels(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 2, {}, stop_on_nonfinite=True)
    signals = (torch.zeros(2, 6), torch.zeros(2, 3), torch.zeros(2, 3))
    signals[0][0, 0] = 603.
    signals[1][1, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite physical substep 1"):
        capture.observe_after_physics(1, 0, signals, lambda: {"q": torch.zeros(2, 18)})
    saved = torch.load(capture.path, weights_only=True)
    assert saved["lane_ids"].tolist() == [0, 1]
    assert capture.summary["lane_ids"] == [0, 1]
    assert capture.summary["nonfinite"]["lane_ids"] == [1]
    assert capture.summary["nonfinite"]["file"] == str(capture.path)


def test_reset_mode_keeps_latest_event_below_the_old_closure_peak(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 2, {}, stop_on_nonfinite=True)
    signals = tuple(torch.zeros(2, width) for width in (6, 3, 3))
    state = {"q": torch.zeros(2, 6), "episode_id": torch.tensor([3, 4])}

    def observe(tick, gap):
        state["q"].fill_(tick - 1.)
        capture.remember_before_physics(tick - 1, (tick - 1) // 4, state)
        state["q"].fill_(float(tick))
        capture.observe_after_physics(tick, (tick - 1) // 4, signals, lambda: state,
            closure_gap=torch.tensor(gap))

    signals[0][0, 0] = 501.
    observe(1, [.65, .001])
    first_sha = hashlib.sha256(capture.path.read_bytes()).hexdigest()
    peak_sha = hashlib.sha256(capture.closure_peak_path.read_bytes()).hexdigest()
    observe(2, [.65, .001])
    assert not capture.latest_path.exists()
    assert capture.summary["latest"]["additional_events_observed"] == 0

    reset_mask = torch.tensor([True, False])
    capture.observe_original_policy_end(4, 1, reset_mask)
    assert reset_mask.tolist() == [True, False]
    state["episode_id"][0] += 1
    # A new episode can fail before any healthy sample was observed.
    observe(5, [.1, .001])
    second = torch.load(capture.latest_path, weights_only=True)
    assert second["dump_physical_tick"] == 5
    assert second["current"]["state"]["episode_id"].tolist() == [4]
    capture.observe_original_policy_end(8, 2, reset_mask)
    signals[0].zero_()
    signals[2][1, 0] = 201.
    observe(9, [.001, .08])
    latest = torch.load(capture.latest_path, weights_only=True)
    assert latest["dump_physical_tick"] == 9 and latest["lane_ids"].tolist() == [1]
    assert latest["trigger"] == "later_physical_substep_crossing"
    assert latest["classification"]["root_angular_velocity_body_rad_s"]["over_limit"].tolist() == [True]
    assert latest["current"]["state"]["q"].tolist() == [[9.] * 6]
    assert latest["prior_frames"][-1]["state"]["q"].tolist() == [[8.] * 6]
    assert latest["diagnostic_only_does_not_terminate"]
    assert hashlib.sha256(capture.path.read_bytes()).hexdigest() == first_sha
    assert hashlib.sha256(capture.closure_peak_path.read_bytes()).hexdigest() == peak_sha
    assert capture.summary["physical_tick"] == 1
    assert capture.summary["latest"]["additional_events_observed"] == 2
    assert capture.summary["latest"]["replacements"] == 2
    assert capture.summary["latest"]["saved_physical_tick"] == 9
    assert not capture.summary["worker_abort_raised"]


def test_latest_capture_rearms_individual_lanes_without_repeating_bad_substeps(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 2, {}, stop_on_nonfinite=True)
    signals = tuple(torch.zeros(2, width) for width in (6, 3, 3))
    state = {"q": torch.zeros(2, 6)}
    calls = []

    def state_factory():
        calls.append(True)
        return state

    def observe(tick):
        capture.observe_after_physics(tick, 0, signals, state_factory)

    signals[0][0, 0] = 501.
    observe(1)
    signals[0][1, 0] = 501.
    observe(2)
    assert torch.load(capture.latest_path, weights_only=True)["lane_ids"].tolist() == [1]
    signals[0][0, 0] = 0.
    observe(3)
    signals[0][0, 0] = 501.
    observe(4)
    assert torch.load(capture.latest_path, weights_only=True)["lane_ids"].tolist() == [0]
    latest_sha = hashlib.sha256(capture.latest_path.read_bytes()).hexdigest()
    observe(5)
    assert len(calls) == 3
    assert hashlib.sha256(capture.latest_path.read_bytes()).hexdigest() == latest_sha
    assert capture.summary["latest"]["replacements"] == 2


@pytest.mark.parametrize("existing_latest", [False, True])
def test_latest_write_failure_preserves_previous_complete_snapshot(tmp_path, monkeypatch, existing_latest):
    capture = NumericalFailureCapture(tmp_path, 1, {}, stop_on_nonfinite=True)
    state = {"q": torch.zeros(1, 6)}
    signals = (torch.full((1, 6), 501.), torch.zeros(1, 3), torch.zeros(1, 3))

    def new_event(tick):
        capture.observe_original_policy_end(tick - 1, tick // 4, torch.ones(1, dtype=torch.bool))
        capture.observe_after_physics(tick, tick // 4, signals, lambda: state)

    new_event(1)
    first_sha = hashlib.sha256(capture.path.read_bytes()).hexdigest()
    if existing_latest:
        new_event(5)
        latest_sha = hashlib.sha256(capture.latest_path.read_bytes()).hexdigest()

    def fail_save(payload, path):
        Path(path).write_bytes(b"incomplete latest event")
        raise OSError("simulated disk-full error")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_save)
        new_event(9)
    report = capture.summary["latest"]
    assert not report["dump_succeeded"]
    assert report["saved_physical_tick"] == (5 if existing_latest else None)
    if existing_latest:
        assert hashlib.sha256(capture.latest_path.read_bytes()).hexdigest() == latest_sha
    else:
        assert not capture.latest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert not capture.summary["worker_abort_raised"]
    new_event(13)
    assert report["dump_succeeded"] and report["saved_physical_tick"] == 13
    assert "dump_error" not in report
    assert hashlib.sha256(capture.path.read_bytes()).hexdigest() == first_sha
