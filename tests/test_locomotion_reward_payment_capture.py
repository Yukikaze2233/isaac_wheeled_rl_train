"""CPU checks for observation of actual native payments, never a second reward."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wheeled_tasks.chassis import rewards


def load(name):
    path = Path(__file__).resolve().parents[1] / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture_module = load("locomotion_reward_payment_capture")
sampling = load("locomotion_exploration_sampling")
Capture = capture_module.NativeRewardPaymentCapture
NAMES = tuple(name for name in sampling.CASES for _ in range(4))


class NativeFixture:
    """Native-shaped CPU stepping, including late density edits and history shift."""
    def __init__(self):
        self.num_envs, self.dt, self.policy_dt, self.decimation = 12, .005, .02, 4
        self.scene_groups, self.tick = NAMES, 0
        self.cfg = {"auto_reset": False, "record_diagnostics": True, "reward_profile": "scut_v14_flat_v5",
                    "dense_tracking": {"enabled": True}, "working_margin_weight": 0.,
                    "success_event_bonus": 30., "termination_event_cost": 200.}
        self.generator = torch.Generator().manual_seed(71891)
        self.actions = torch.arange(72, dtype=torch.float32).reshape(12, 6) / 100
        self.previous_actions = -self.actions.clone()
        self.before_previous_actions = self.actions.clone() * .25
        self.previous_motor_velocity = torch.full((12, 6), .1)
        self.commands = torch.zeros(12, 3)
        self.commands[:, 2] = .305
        self.success_latch = torch.zeros(12, dtype=torch.bool)
        self.density_calls = self.event_calls = 0

    def step(self, action):
        self.before_previous_actions.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(action.clamp(-1, 1))
        q = torch.rand(12, 6, generator=self.generator)
        dq = q * .3
        self.density_calls += 1
        density = {"track_velocity": torch.full((12,), 8.), "pitch": torch.full((12,), -.4)}
        # A native late override/addition after the original reward producer.
        density["pitch"] = torch.full((12,), -.9)
        density["working_margin"] = torch.zeros(12)
        reward = torch.stack(list(density.values())).sum(0) * self.policy_dt
        self.previous_motor_velocity.copy_(dq)
        terminated = torch.zeros(12, dtype=torch.bool)
        success = terminated.clone()
        timeout = terminated.clone()
        terminated[0], success[1], timeout[2] = True, True, True
        success_event = success & ~self.success_latch
        self.success_latch |= success
        self.event_calls += 1
        event = rewards.episode_event_terms(success_event, terminated, self.cfg)
        self.last_event_result = event
        reward += event["success_event"]
        done = terminated | timeout | success
        reward += event["failure_event"]
        self.tick += 1
        diag = {"commands": self.commands.clone(), "velocity": q[:, :3].clone(), "omega": dq[:, :3].clone(),
                "gravity": torch.tensor([0., 0., -1.]).repeat(12, 1), "height": torch.full((12,), .305),
                "motor_position": q.clone(), "motor_velocity": dq.clone(), "motor_effort": q.clone(),
                "physical_phase": torch.zeros(12, dtype=torch.long), "physical_phase_time_s": torch.full((12,), self.tick * .02),
                "actions": self.actions.clone(), "reward": reward.clone(), "done": done.clone(),
                "terminated": terminated.clone(), "success": success.clone(),
                "reasons": {"boundary": torch.zeros(12, dtype=torch.bool), "travel": terminated.clone()}}
        extras = {"native_reward_components": density, "diagnostics": diag, "time_outs": timeout,
                  "log": {"/reward/track_velocity": torch.tensor(-999.)}}
        # Returned environment state already carries the next command; diagnostics do not.
        self.commands[:, 0] = self.tick * 10.
        return reward, done, extras


def setup(role="clean"):
    env, stream = NativeFixture(), io.StringIO()
    ledger = sampling.FirstCohortLedger(NAMES)
    observer = Capture(env, ledger, stream, fixture_role=role)
    return env, ledger, stream, observer


def observe_ledger(ledger, done, extras, tick):
    diag = extras["diagnostics"]
    ledger.observe(done=done, terminated=diag["terminated"], timeouts=extras["time_outs"],
                   success=diag["success"], reasons=diag["reasons"], physical_tick=4 * tick)


def step(env, observer, tick=1):
    observer.begin(tick, 4 * (tick - 1), ["hold"] * 12)
    reward, done, extras = env.step(torch.full((12, 6), 1.4))
    rows = observer.capture(reward, done, extras, 4 * tick)
    return reward, done, extras, rows


def test_native_outputs_histories_and_rng_unchanged_by_capture(monkeypatch):
    original = rewards.episode_event_terms
    seen = []
    def spy(*args):
        result = original(*args)
        seen.append(result)
        return result
    monkeypatch.setattr(rewards, "episode_event_terms", spy)
    baseline = NativeFixture()
    expected = baseline.step(torch.full((12, 6), 1.4))
    env, ledger, stream, observer = setup()
    global_rng = torch.random.get_rng_state().clone()
    try:
        result = step(env, observer)
        assert env.last_event_result is seen[-1]  # No replaced producer output.
        assert len(seen) == 2  # One baseline and exactly one observed producer call.
        assert env.density_calls == env.event_calls == 1
        for a, b in zip(result[:2], expected[:2]):
            assert torch.equal(a, b)
        for name in ("actions", "previous_actions", "before_previous_actions", "previous_motor_velocity"):
            assert torch.equal(getattr(env, name), getattr(baseline, name))
        assert torch.equal(env.generator.get_state(), baseline.generator.get_state())
        assert torch.equal(torch.random.get_rng_state(), global_rng)
        assert observer.summary()["native_event_calls"] == 1
        assert observer.summary()["rows_written"] == 12
    finally:
        observer.close()
    assert rewards.episode_event_terms is spy


def test_final_densities_discrete_events_and_cached_state_reconcile():
    env, ledger, stream, observer = setup("pipeline_age_only")
    old_dq = env.previous_motor_velocity.clone()
    try:
        reward, done, extras, rows = step(env, observer)
        assert rows[0]["fixture_role"] == "pipeline_age_only"
        assert rows[0]["density_per_s"]["pitch"] == pytest.approx(-.9)
        assert rows[0]["density_payment"]["track_velocity"] == pytest.approx(.16)
        assert rows[0]["event_payment"]["failure_event"] == -200.
        assert rows[1]["event_payment"]["success_event"] == 30.
        assert rows[2]["terminal"]["timeout"] and rows[2]["event_payment"]["failure_event"] == 0.
        assert rows[0]["reconciliation"]["reconstructed_total"] == rows[0]["reward_total"]
        assert rows[0]["state"]["commands"][0] == 0. and env.commands[0, 0] == 10.
        assert rows[0]["state"]["previous_motor_velocity_used"] == old_dq[0].tolist()
        assert rows[0]["state"]["actions"] == [1.] * 6
        original_row = deepcopy(rows[0])
        extras["native_reward_components"]["pitch"].fill_(123.)
        extras["diagnostics"]["velocity"].zero_()
        env.actions.zero_()
        assert rows[0] == original_row
        assert json.loads(stream.getvalue().splitlines()[0]) == original_row
        assert observer.summary()["qualification"] is None
    finally:
        observer.close()


def test_first_terminal_frame_included_and_post_terminal_native_repayment_separate():
    env, ledger, stream, observer = setup()
    try:
        _, done, extras, rows = step(env, observer)
        assert all(row["first_cohort"] for row in rows)
        observe_ledger(ledger, done, extras, 1)
        _, _, _, second = step(env, observer, 2)
        assert second[0]["event_payment"]["failure_event"] == -200.  # Actual native repeated payment.
        assert second[1]["terminal"]["success"] and not second[1]["terminal"]["success_event_input"]
        assert second[1]["event_payment"]["success_event"] == 0.
        assert [row["first_cohort"] for row in second[:4]] == [False, False, False, True]
        summary = observer.summary()
        assert summary["first_cohort_rows"] == 21
        assert summary["payment_totals"]["all_native"]["event_total"] == -370.
        assert summary["payment_totals"]["first_cohort"]["event_total"] == -170.
        assert summary["post_first_terminal_native_payments_recorded_but_not_credited"]
        assert not summary["incomplete_policy_interval"]
    finally:
        observer.close()


@pytest.mark.parametrize("change", ("total", "missing_density", "scalar_density", "nonfinite", "event_density", "mask_dtype", "cache", "history", "missing_timeout"))
def test_malformed_or_unclosed_native_payment_rejected(change):
    env, ledger, stream, observer = setup()
    try:
        observer.begin(1, 0, ["hold"] * 12)
        reward, done, extras = env.step(torch.ones(12, 6))
        if change == "total":
            reward[4] += .2
            extras["diagnostics"]["reward"] = reward.clone()
        elif change == "missing_density":
            extras.pop("native_reward_components")
        elif change == "scalar_density":
            extras["native_reward_components"]["pitch"] = torch.tensor(-.9)
        elif change == "nonfinite":
            extras["native_reward_components"]["pitch"][4] = float("nan")
        elif change == "event_density":
            extras["native_reward_components"]["failure_event"] = torch.zeros(12)
        elif change == "mask_dtype":
            extras["diagnostics"]["success"] = extras["diagnostics"]["success"].float()
        elif change == "cache":
            extras["diagnostics"]["reward"][4] += .1
        elif change == "history":
            env.previous_actions.zero_()
        else:
            extras.pop("time_outs")
        with pytest.raises(ValueError):
            observer.capture(reward, done, extras, 4)
        assert stream.getvalue() == ""
        assert observer.summary()["rows_written"] == 0
        assert observer.summary()["incomplete_policy_interval"]
    finally:
        observer.close()


@pytest.mark.parametrize("calls", (0, 2))
def test_event_producer_must_have_one_native_call(calls):
    env, ledger, stream, observer = setup()
    try:
        observer.begin(1, 0, ["hold"] * 12)
        reward, done, extras = env.step(torch.ones(12, 6))
        if calls == 0:
            observer.active["event_calls"] = 0
        else:
            rewards.episode_event_terms(torch.zeros(12, dtype=torch.bool), torch.zeros(12, dtype=torch.bool), env.cfg)
        with pytest.raises(ValueError, match="exactly one"):
            observer.capture(reward, done, extras, 4)
    finally:
        observer.close()


def test_capture_before_ledger_and_native_clock_contract():
    env, ledger, stream, observer = setup()
    try:
        with pytest.raises(ValueError):
            observer.begin(2, 4, ["hold"] * 12)
        with pytest.raises(ValueError):
            observer.begin(1, 1, ["hold"] * 12)
        observer.begin(1, 0, ["hold"] * 12)
        with pytest.raises(ValueError, match="Unclosed"):
            observer.begin(1, 0, ["hold"] * 12)
        reward, done, extras = env.step(torch.ones(12, 6))
        with pytest.raises(ValueError, match="clocks"):
            observer.capture(reward, done, extras, 3)
        observe_ledger(ledger, done, extras, 1)
        with pytest.raises(ValueError, match="precede"):
            observer.capture(reward, done, extras, 4)
    finally:
        observer.close()


def test_hook_restored_after_failed_native_step_and_no_payment_fabricated():
    original = rewards.episode_event_terms
    env, ledger, stream, observer = setup()
    try:
        observer.begin(1, 0, ["hold"] * 12)
        assert observer.summary()["native_event_calls"] == 0
        raise RuntimeError("native physical failure")
    except RuntimeError:
        pass
    finally:
        observer.close()
    assert rewards.episode_event_terms is original
    assert observer.summary()["incomplete_policy_interval"]
    assert observer.summary()["native_policy_steps_captured"] == 0
    assert observer.summary()["rows_written"] == 0
    assert stream.getvalue() == ""
    observer.close()  # Idempotent finally cleanup.


def test_duplicate_hook_and_unexpected_native_event_interval_rejected():
    env, ledger, stream, observer = setup()
    try:
        with pytest.raises(ValueError, match="already installed"):
            Capture(env, ledger, io.StringIO(), fixture_role="clean")
        with pytest.raises(ValueError, match="outside"):
            rewards.episode_event_terms(torch.zeros(12, dtype=torch.bool), torch.zeros(12, dtype=torch.bool), env.cfg)
    finally:
        observer.close()


@pytest.mark.parametrize("change", ("autoreset", "diagnostics", "units", "legacy", "role", "population"))
def test_constructor_rejects_unbounded_or_unexposed_payment_fixture(change):
    env = NativeFixture()
    role = "clean"
    if change == "autoreset":
        env.cfg["auto_reset"] = True
    elif change == "diagnostics":
        env.cfg["record_diagnostics"] = False
    elif change == "units":
        env.cfg.pop("working_margin_weight")
    elif change == "legacy":
        env.cfg["dense_tracking"]["enabled"] = False
    elif change == "role":
        role = "other"
    else:
        env.num_envs = 4
    original = rewards.episode_event_terms
    with pytest.raises(ValueError):
        Capture(env, sampling.FirstCohortLedger(NAMES), io.StringIO(), fixture_role=role)
    assert rewards.episode_event_terms is original
