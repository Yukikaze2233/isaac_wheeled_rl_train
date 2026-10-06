"""Meaningful synthetic CPU boundaries; no App, native evaluation or PPO."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).parent
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load("payment_helper_test", "observe_prefix_native_payments_019.py")
driver = load("payment_driver_test", "probe_prefix_mean_stochastic_payments_019.py")
old_test = load("old_fixture_test", "test_probe_prefix_mean_stochastic_019.py")


def fixture(steps=1):
    n = 8
    enabled = torch.tensor([1, 1, 1, 1, 1, 0, 1, 0], dtype=torch.bool)
    lesson = torch.tensor([0, 1, 2, 0, 1, -1, 0, -1])
    profile = torch.where(enabled, 0, -1)
    live, hold = torch.ones(n, dtype=torch.bool), enabled.clone()
    hold[2] = False
    env = SimpleNamespace(num_envs=n, device="cpu", dt=.005, policy_dt=.02, decimation=4,
        tick=0, cfg={}, recovery_takeover_settings=None,
        recovery=SimpleNamespace(enabled=enabled, pose_curriculum=SimpleNamespace(level_ids=lesson, profile_ids=profile)))
    event_module = SimpleNamespace(calls=0)
    def original(success, terminated, settings):
        event_module.calls += 1
        event_module.last_args = (success, terminated, settings)
        event_module.last_result = {"success_event": 30 * success.float(), "failure_event": -200 * terminated.float()}
        return event_module.last_result
    event_module.episode_event_terms = original
    observer = helper.NativePaymentObserver(env, lambda: live, lambda: hold,
        torch.tensor([0, 1, 2, 1, 3, 4, 0, 4]), event_module=event_module, policy_steps=steps)
    return env, live, hold, event_module, original, observer


def frame(env, observer, module, *, done=None):
    n = env.num_envs
    observer.begin(env.tick + 1)
    enabled = env.recovery.enabled.clone()
    terminated = torch.zeros(n, dtype=torch.bool)
    terminated[3] = True
    success = torch.zeros(n, dtype=torch.bool)
    success[2] = True
    if done is None:
        done = terminated | success
        done[7] = True
    events = module.episode_event_terms(success, terminated, env.cfg)
    components = {"positive_tracking_peak": torch.ones(n), "velocity_cost": -torch.arange(n).float()}
    reward = torch.stack(list(components.values())).sum(0) * .02 + sum(events.values())
    d = dict(reward=reward.clone(), done=done.clone(), terminated=terminated, success=success,
        recovery_enabled=enabled, recovery_prefix_lesson_id=env.recovery.pose_curriculum.level_ids.clone(),
        recovery_actual_profile_id=env.recovery.pose_curriculum.profile_ids.clone(),
        recovery_motion_released=torch.tensor([0, 1, 1, 0, 0, 1, 0, 1], dtype=torch.bool),
        recovery_pure_rl=torch.tensor([1, 1, 1, 0, 1, 1, 0, 1], dtype=torch.bool),
        motor_velocity=torch.ones(n, 6), omega=torch.zeros(n, 3), velocity=torch.zeros(n, 3),
        commands=torch.zeros(n, 3), planar_speed_world=torch.zeros(n),
        height=torch.zeros(n), gravity=torch.zeros(n, 3),
        recovery_rl_stable_s=torch.zeros(n), recovery_stable_peak_seconds=torch.zeros(n),
        recovery_followup_s=torch.zeros(n))
    d.update({"recovery_stable_" + k + "_pass": torch.ones(n, dtype=torch.bool) for k in helper.GATES})
    extras = dict(diagnostics=d, native_reward_components=components,
        policy_learning_mask=torch.tensor([1, 1, 1, 1, 0, 1, 0, 1], dtype=torch.bool))
    env.tick += 1
    return reward, done, extras


def rows(summary, scope="all_native_intervals"):
    return [r for r in summary["rows"] if r["scope"] == scope]


def test_actual_density_events_reconcile_once_and_return_objects_unchanged():
    env, live, hold, module, original, observer = fixture()
    rng = torch.get_rng_state().clone()
    reward, done, extras = frame(env, observer, module)
    assert module.calls == 1
    assert module.last_args[0] is extras["diagnostics"]["success"]
    assert module.last_result["failure_event"] is not observer.active["event"]["payments"]["failure_event"]
    observer.capture(reward, done, extras)
    live &= ~done
    summary = observer.summary()
    assert summary["complete"] and summary["native_lane_intervals"] == 8
    assert summary["first_birth_closed_lanes"] == 3 and summary["first_birth_censored_lanes"] == 5
    assert summary["max_reconciliation_residual"] <= 2e-5
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(r["policy_lane_intervals"] for r in rows(summary)) == 8
    for row in rows(summary):
        assert row["lane_seconds"] == row["policy_lane_intervals"] * .02
        assert row["components"]["positive_tracking_peak"]["mean_payment"] == .02
    observer.close()
    assert module.episode_event_terms is original


@pytest.mark.parametrize("mode", ["double_dt", "event_dt", "duplicated_event", "missing_density"])
def test_wrong_actual_payment_arithmetic_rejected(mode):
    env, _, _, module, _, observer = fixture()
    reward, done, extras = frame(env, observer, module)
    event = observer.active["event"]["payments"]
    density = torch.stack(list(extras["native_reward_components"].values())).sum(0)
    if mode == "double_dt": reward = density * .02 * .02 + sum(event.values())
    if mode == "event_dt": reward = density * .02 + .02 * sum(event.values())
    if mode == "duplicated_event": reward = density * .02 + 2 * sum(event.values())
    if mode == "missing_density": del extras["native_reward_components"]["velocity_cost"]
    extras["diagnostics"]["reward"] = reward.clone()
    with pytest.raises(ValueError, match="does not reconcile"):
        observer.capture(reward, done, extras)
    observer.close()


def test_actual_returned_reward_relation_cannot_be_hidden_by_cached_diag():
    env, _, _, module, _, observer = fixture()
    reward, done, extras = frame(env, observer, module)
    with pytest.raises(ValueError, match="returned reward/done"):
        observer.capture(reward + 1, done, extras)
    observer.close()


def test_final_pure_rl_cannot_hide_actor_failure_or_credit_script_to_rl():
    env, live, _, module, _, observer = fixture()
    reward, done, extras = frame(env, observer, module)
    observer.capture(reward, done, extras)
    live &= ~done
    summary = observer.summary()
    terminal = next(r for r in rows(summary) if r["payment_partition"] == "recovery_actor_valid_terminal")
    assert terminal["components"]["failure_event"]["sum"] == -200
    assert terminal["components"]["actor_eligible_immediate_reward"]["sum"] < -200
    excluded = [r for r in rows(summary) if r["payment_partition"] == "recovery_excluded"]
    assert sum(r["policy_lane_intervals"] for r in excluded) == 2
    assert all(r["components"]["actor_eligible_immediate_reward"]["sum"] == 0 for r in excluded)
    observer.close()


def test_release_interval_and_fully_released_next_interval_have_distinct_bins():
    env, live, hold, module, _, observer = fixture(2)
    reward, done, extras = frame(env, observer, module)
    observer.capture(reward, done, extras)
    live &= ~done
    hold[1] = False
    reward, done, extras = frame(env, observer, module)
    observer.capture(reward, done, extras)
    live &= ~done
    rs = rows(observer.summary())
    assert sum(r["policy_lane_intervals"] for r in rs if r["lesson"] == "prefix_release"
               and r["payment_partition"] == "recovery_actor_valid_release_interval") == 1
    assert sum(r["policy_lane_intervals"] for r in rs if r["lesson"] == "prefix_release"
               and r["payment_partition"] == "recovery_actor_valid_postrelease") == 1
    observer.close()


def test_auto_reset_uses_old_lesson_and_terminal_first_birth_closes_once():
    env, live, _, module, _, observer = fixture(2)
    reward, done, extras = frame(env, observer, module)
    env.recovery.pose_curriculum.level_ids[3] = 2  # Reset inside original env.step.
    observer.capture(reward, done, extras)
    live &= ~done
    reward, done, extras = frame(env, observer, module)
    observer.capture(reward, done, extras)
    live &= ~done
    summary = observer.summary()
    first = rows(summary, "first_birth_intervals")
    assert sum(r["policy_lane_intervals"] for r in first) == 13
    assert sum(r["policy_lane_intervals"] for r in first if r["original_goal"] == "forward"
               and r["payment_partition"] == "recovery_actor_valid_terminal") == 1
    assert any(r["lesson"] == "prefix_stand" and r["payment_partition"] == "recovery_actor_valid_terminal" for r in first)
    assert not any(r["lesson"] == "original_chain" and r["payment_partition"] == "recovery_actor_valid_terminal" for r in first)
    observer.close()


@pytest.mark.parametrize("mode", ["reopened_first", "missing_event", "duplicate_event", "callback_changed"])
def test_event_and_firstbirth_ownership_fail_fast(mode):
    env, live, _, module, _, observer = fixture(2)
    reward, done, extras = frame(env, observer, module)
    if mode == "missing_event":
        observer.active["event"] = None
        with pytest.raises(ValueError, match="Missing native event"):
            observer.capture(reward, done, extras)
    elif mode == "duplicate_event":
        with pytest.raises(ValueError, match="Duplicate"):
            module.episode_event_terms(extras["diagnostics"]["success"], extras["diagnostics"]["terminated"], env.cfg)
        assert module.calls == 1
    elif mode == "callback_changed":
        module.episode_event_terms = lambda *a: None
        with pytest.raises(ValueError, match="ownership changed"):
            observer.close()
        module.episode_event_terms = observer.event_observer
    else:
        observer.capture(reward, done, extras)
        with pytest.raises(ValueError, match="reopened"):
            observer.begin(2)  # Base ledger closure deliberately omitted.
    observer.close()


@pytest.mark.parametrize("mode", ["inactive", "nan", "wrong_old_lesson", "missing_component", "base_excluded"])
def test_inactive_nonfinite_and_missing_native_evidence_rejected(mode):
    env, _, _, module, _, observer = fixture()
    reward, done, extras = frame(env, observer, module)
    if mode == "inactive": extras["native_reward_components"]["recovery_takeover_hold_wheel_velocity_broad"] = torch.zeros(8)
    if mode == "nan": extras["native_reward_components"]["velocity_cost"][4] = float("nan")
    if mode == "wrong_old_lesson": extras["diagnostics"]["recovery_prefix_lesson_id"][0] = 1
    if mode == "missing_component": extras.pop("native_reward_components")
    if mode == "base_excluded": extras["policy_learning_mask"][5] = False
    with pytest.raises(ValueError): observer.capture(reward, done, extras)
    observer.close()


def test_incomplete_window_and_original_goal_not_overwritten_l0():
    env, _, _, module, _, observer = fixture(2)
    with pytest.raises(ValueError, match="Incomplete"):
        observer.summary()
    observer.close()
    env.skills = SimpleNamespace(specs={"r": {"recovery_followup_command": [.3, 0., .305]}, "b": {}})
    env.scene_groups = ["r", "b"]
    env.recovery.enabled = torch.tensor([True, False])
    env.recovery.followup_command = torch.zeros(2, 3)  # L0 actual adapter overwrite.
    assert helper.original_goal_ids(env).tolist() == [1, 4]


def test_tampered_artifact_rejects_before_execution_or_App_import(tmp_path):
    p = tmp_path / "tampered.py"
    p.write_text("raise RuntimeError('must never execute')\n")
    with pytest.raises(ValueError, match="SHA differs"):
        driver.load_bound_module(p, driver.BASE_SHA, "never_exec")


def test_native_context_rejects_wrong_birth_or_sampling_stream():
    with pytest.raises(ValueError, match="birth/environment RNG"):
        driver.native_context({})
    with pytest.raises(ValueError, match="private sampling stream"):
        driver.native_context(dict(initial_physical_birth_sha256=driver.INITIAL_BIRTH_SHA,
            initial_env_rng_sha256=driver.INITIAL_ENV_RNG_SHA), final=True)


def test_closing_stream_cleanup_precedes_original_close():
    calls = []
    stream = SimpleNamespace(close=lambda: calls.append("stream"))
    driver.ClosingStream(stream, lambda: calls.append("before_app_close")).close()
    assert calls == ["before_app_close", "stream"]


def integrated(tmp_path, *, break_payment=False):
    base = driver.load_bound_module(HERE / "probe_prefix_mean_stochastic_019.py", driver.BASE_SHA, "fixture_base")
    base.POLICY_STEPS = 1  # Synthetic lifecycle bound, never a native invocation.
    base.verify_inputs = lambda *a: True
    env, calls = old_test.fake_env()
    env.dt, env.policy_dt, env.decimation, env.tick = .005, .02, 4, 0
    env.cfg, env.recovery_takeover_settings = {}, None
    env.generator = torch.Generator().manual_seed(7)
    env.skills = SimpleNamespace(specs={"a": {"recovery_followup_command": [0, 0, .305]},
        "b": {"recovery_followup_command": [.3, 0, .305]},
        "c": {"recovery_followup_command": [0, .6, .305]}, "ordinary": {}})
    module = SimpleNamespace()
    def event(success, terminated, settings):
        module.args = (success, terminated, settings)
        module.result = dict(success_event=30*success.float(), failure_event=-200*terminated.float())
        return module.result
    module.episode_event_terms = event
    imported = helper.importlib.import_module
    helper.importlib.import_module = lambda name: module if name == "wheeled_tasks.chassis.rewards" else imported(name)
    def step(action):
        calls["step"] = calls.get("step", 0) + 1
        env.recovery.begin_policy()
        for _ in range(4):
            env.recovery.fsm.update()
            env.recovery.torque(None, None, action)
        extras = old_test.fake_extras(env, done0=True)
        d = extras["diagnostics"]
        extras["native_reward_components"] = dict(actual_density=torch.ones(4))
        events = module.episode_event_terms(d["success"], d["terminated"], env.cfg)
        reward = .02 * torch.ones(4) + sum(events.values())
        d.update(reward=reward.clone(), motor_velocity=torch.zeros(4,6))
        if break_payment: extras["native_reward_components"]["actual_density"][0] += 1
        env.tick += 1
        env.recovery.pose_curriculum.level_ids[0] = 2
        env.last_result = ({}, reward, d["done"], extras)
        return env.last_result
    env.step = step
    (tmp_path / "identity.json").write_text('{"synthetic_CPU_fixture":true}\n')
    args = SimpleNamespace(output=tmp_path)
    report = dict(initial_physical_birth_sha256=driver.INITIAL_BIRTH_SHA,
                  initial_env_rng_sha256=driver.INITIAL_ENV_RNG_SHA)
    klass = driver.observer_class(base, helper, args, {}, report, {}, base.write_json)
    try:
        observer = klass(env, tmp_path)
    finally:
        helper.importlib.import_module = imported
    return env, calls, module, event, step, observer


def test_composed_base_observer_keeps_outputs_rng_firstcohort_and_restores_preclose(tmp_path):
    env, calls, module, event, step, observer = integrated(tmp_path)
    rng, env_rng = torch.get_rng_state().clone(), env.generator.get_state().clone()
    result = env.step(torch.ones(4,6))
    assert result is env.last_result
    observer.observe(1, result[3])
    report = observer.finish(tmp_path)
    assert report["first_birth"]["by_lesson"]["prefix_stand"]["partitions"] == {"failure": 1}
    assert calls == dict(begin=1, torque=4, update=4, step=1)
    assert module.episode_event_terms is event and env.step is step
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(env_rng, env.generator.get_state())
    assert report["payment_hooks_restored_before_app_close"]
    assert (tmp_path/"payment_summary.json").exists()
    assert not (tmp_path/"payment_observer_failure.json").exists()


def test_composed_failure_receipt_and_restore_exist_before_any_app_close(tmp_path):
    env, _, module, event, step, observer = integrated(tmp_path, break_payment=True)
    with pytest.raises(ValueError, match="does not reconcile"):
        env.step(torch.ones(4,6))
    observer.sparse.close()  # The unmodified base.run_native finally's preclose operation.
    import json
    failure = json.loads((tmp_path/"payment_observer_failure.json").read_text())
    assert failure["complete"] is False and failure["unfinished_interval"]
    assert failure["app_close_returned"] is None and failure["source_admission"] is False
    assert module.episode_event_terms is event and env.step is step


def test_admission_failure_after_payment_finish_is_persisted_before_close(tmp_path):
    import json
    def write(path, value):
        Path(path).write_text(json.dumps(value))
    guarded = driver.guarded_writer(write, SimpleNamespace(output=tmp_path), {})
    with pytest.raises(ValueError, match="birth/environment RNG"):
        guarded(tmp_path/"report.json", dict(status="native600_completed_preclose"))
    failure = json.loads((tmp_path/"payment_admission_failure.json").read_text())
    assert not failure["complete"] and not failure["source_admission"]
    assert failure["app_close_returned"] is None
    assert not (tmp_path/"report.json").exists()


def test_original_base_finish_failure_is_recorded_before_close(tmp_path, monkeypatch):
    import json
    env, _, module, event, step, observer = integrated(tmp_path)
    result = env.step(torch.ones(4,6))
    observer.observe(1, result[3])
    def failed_finish(*args):
        raise OSError("synthetic original base summary write failure")
    monkeypatch.setattr(type(observer).__mro__[1], "finish", failed_finish)
    with pytest.raises(OSError, match="original base summary"):
        observer.finish(tmp_path)
    observer.sparse.close()
    failure = json.loads((tmp_path/"payment_observer_failure.json").read_text())
    assert failure["native_payments_reconciled"] is True and failure["complete"] is False
    assert module.episode_event_terms is event and env.step is step
