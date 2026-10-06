"""CPU-only first-cohort boundaries and read-only observer hooks, no App import."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location("prefix_probe_019", Path(__file__).with_name("probe_prefix_mean_stochastic_019.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def birth(lesson="prefix_stand"):
    return dict(lane=0, birth_id=1, group="g", lesson=lesson)


def terminal(**kwargs):
    return dict(done=True, terminated=False, success=False, timeout=True,
                termination_reasons=[], **kwargs)


def test_alive_12s_is_censored_not_failure_or_timeout():
    ledger = probe.FirstBirthLedger([birth()])
    report = ledger.finish(600, {0: dict(done=False, terminated=False, success=False, timeout=False)})
    assert report["partitions"] == {"right_censored_unfinished": 1}
    assert report["rows"][0]["censor_policy_tick"] == 600
    assert report["by_lesson"]["prefix_stand"]["fixed_birth_denominator"] == 1


@pytest.mark.parametrize("flags,part", [
    ({"terminated": True, "success": True, "timeout": True}, "failure"),
    ({"success": True, "timeout": True}, "success"),
    ({"termination_reasons": ["boundary"], "timeout": True}, "boundary_truncation"),
    ({"termination_reasons": ["blocked"], "timeout": True}, "boundary_truncation"),
    ({"timeout": True}, "timeout"),
])
def test_native_outcome_precedence(flags, part):
    row = terminal()
    row.update(flags)
    ledger = probe.FirstBirthLedger([birth()])
    ledger.close(0, 17, row)
    report = ledger.finish(600, {})
    assert report["rows"][0]["partition"] == part
    assert report["rows"][0]["terminal_policy_tick"] == 17


def test_native_rebirth_never_reopens_first_cohort():
    ledger = probe.FirstBirthLedger([birth()])
    ledger.close(0, 5, terminal())
    with pytest.raises(ValueError, match="reopened"):
        ledger.close(0, 10, terminal())
    with pytest.raises(ValueError, match="censor population"):
        ledger.finish(600, {0: terminal()})


def test_unknown_native_done_cannot_be_invented_as_timeout():
    ledger = probe.FirstBirthLedger([birth()])
    row = terminal()
    row["timeout"] = False
    with pytest.raises(ValueError, match="lacks a partition"):
        ledger.close(0, 5, row)


def fake_env():
    n = 4
    enabled = torch.tensor([True, True, True, False])
    course = SimpleNamespace(level_ids=torch.tensor([0, 1, 2, -1]), profile_ids=torch.tensor([0, 0, 0, -1]),
                             profile_names=["upright"])
    calls = dict(begin=0, torque=0, update=0)
    fsm = SimpleNamespace(pure_rl=torch.ones(n, dtype=torch.bool), failed=torch.zeros(n, dtype=torch.bool),
                          motion_released=torch.zeros(n, dtype=torch.bool), holding_rl=enabled.clone(),
                          stable_pass_masks={g: torch.ones(n, dtype=torch.bool) for g in probe.GATES})
    def update(*args, **kwargs):
        calls["update"] += 1
        return "original_update_output"
    def begin():
        calls["begin"] += 1
    def torque(q, dq, effort):
        calls["torque"] += 1
        return effort
    fsm.update = update
    recovery = SimpleNamespace(enabled=enabled, pure_rl=torch.ones(n, dtype=torch.bool),
        release_finished=torch.ones(n, dtype=torch.bool), policy_motion_hold=enabled.clone(),
        pose_curriculum=course, fsm=fsm, begin_policy=begin, torque=torque)
    env = SimpleNamespace(num_envs=n, device="cpu", recovery=recovery,
        scene_groups=["a", "b", "c", "ordinary"], kinds=["flat"]*n,
        startup_report={"control_timing": {"runtime_substep_verification": {"pd_updates": 4, "feedback_refreshes": 4}}})
    return env, calls


def fake_extras(env, done0=False):
    n = env.num_envs
    data = {k: torch.zeros(n) for k in probe.ROW_FIELDS}
    for k in ("commands", "velocity", "omega", "gravity"):
        data[k] = torch.zeros(n, 3)
    data["recovery_enabled"] = env.recovery.enabled.clone()
    data["recovery_prefix_lesson_id"] = torch.tensor([0, 1, 2, -1])
    data["recovery_actual_profile_id"] = torch.tensor([0, 0, 0, -1])
    data["recovery_holding_rl"] = env.recovery.enabled.clone()
    data["recovery_pure_rl"] = torch.ones(n, dtype=torch.bool)
    for k in ("done", "terminated", "success", "recovery_motion_released"):
        data[k] = torch.zeros(n, dtype=torch.bool)
    data["done"][0] = done0
    data["terminated"][0] = done0
    data["reasons"] = {"boundary": torch.zeros(n, dtype=torch.bool), "fall": data["terminated"].clone()}
    for gate in probe.GATES:
        data[f"recovery_stable_{gate}_pass"] = torch.ones(n, dtype=torch.bool)
    return dict(diagnostics=data, time_outs=torch.zeros(n, dtype=torch.bool),
                policy_learning_mask=torch.ones(n, dtype=torch.bool),
                policy_bootstrap_mask=~data["terminated"])


def run_one_policy(env):
    env.recovery.begin_policy()
    effort = torch.randn(env.num_envs, 6)
    for _ in range(4):
        assert env.recovery.fsm.update() == "original_update_output"
        assert env.recovery.torque(None, None, effort) is effort


def test_hook_calls_original_once_keeps_outputs_and_rng(tmp_path):
    env, calls = fake_env()
    observer = probe.NativeObserver(env, tmp_path)
    effort = torch.ones(4, 6)
    before = torch.get_rng_state().clone()
    env.recovery.begin_policy()
    for _ in range(4):
        assert env.recovery.fsm.update() == "original_update_output"
        assert env.recovery.torque(None, None, effort) is effort
    assert torch.equal(before, torch.get_rng_state())
    assert calls == dict(begin=1, torque=4, update=4)
    observer.observe(1, fake_extras(env))
    report = observer.finish(tmp_path)
    assert report["reference_update_calls"] == 4
    assert report["raw_instantaneous_hold_gates"]["reference_200Hz"]["first_birth_only"]["prefix_stand"]["hold_frames"] == 4
    assert report["raw_instantaneous_hold_gates"]["diagnostic_50Hz"]["first_birth_only"]["prefix_stand"]["hold_frames"] == 1


def test_old_terminal_lesson_survives_postreset_lesson_change(tmp_path):
    env, _ = fake_env()
    observer = probe.NativeObserver(env, tmp_path)
    run_one_policy(env)
    old = fake_extras(env, done0=True)
    # Mimic native autoreset already happening inside env.step: old extras are
    # lesson0 but the current/new physical birth is now lesson2.
    env.recovery.pose_curriculum.level_ids[0] = 2
    observer.observe(1, old)
    run_one_policy(env)
    data = fake_extras(env)
    data["diagnostics"]["recovery_prefix_lesson_id"][0] = 2
    observer.observe(2, data)
    report = observer.finish(tmp_path)
    assert report["first_birth"]["by_lesson"]["prefix_stand"]["partitions"] == {"failure": 1}
    assert report["subsequent_births_by_lesson"]["original_chain"] == 1
    assert report["policy_diagnostics"]["first_birth_only"]["prefix_stand"]["frames"] == 1
    assert report["policy_diagnostics"]["first_birth_only"]["original_chain"]["frames"] == 2
    assert report["policy_diagnostics"]["all_native_births"]["original_chain"]["frames"] == 3


def test_missing_substep_ownership_is_rejected(tmp_path):
    env, _ = fake_env()
    observer = probe.NativeObserver(env, tmp_path)
    with pytest.raises(ValueError, match="ownership"):
        observer.observe(1, fake_extras(env))
    observer.sparse.close()


def test_passive_default_rl_flags_are_not_handover_or_release(tmp_path):
    env, _ = fake_env()
    observer = probe.NativeObserver(env, tmp_path)
    r = env.recovery
    r.release_finished[0] = False
    r.fsm.motion_released[0] = True
    assert r.fsm.pure_rl[0]
    r.fsm.update()
    assert observer.handover_ref[0] == observer.release_ref[0] == -1
    r.release_finished[0] = True
    r.fsm.pure_rl[0] = False  # Actual PREPARE after the release reset.
    r.fsm.motion_released[0] = False
    r.fsm.update()
    assert observer.handover_ref[0] == observer.release_ref[0] == -1
    r.fsm.pure_rl[0] = True  # Real actor ownership.
    r.fsm.update()
    assert observer.handover_ref[0] == 3 and observer.release_ref[0] == -1
    r.fsm.motion_released[0] = True  # Real qualified native unlock.
    r.fsm.update()
    assert observer.handover_ref[0] == 3 and observer.release_ref[0] == 4
    observer.sparse.close()


def test_nonzero_command_exposure_requires_whole_actor_interval(tmp_path):
    env, _ = fake_env()
    observer = probe.NativeObserver(env, tmp_path)
    run_one_policy(env)
    data = fake_extras(env)
    data["diagnostics"]["commands"][1, 0] = .3
    data["policy_learning_mask"][1] = False
    observer.expected[1] = False
    observer.observe(1, data)
    report = observer.finish(tmp_path)
    metrics = report["policy_diagnostics"]["first_birth_only"]["prefix_release"]
    assert metrics["nonzero_vx_command_frames"] == 1
    assert metrics["actor_valid_and_nonzero_vx_command_frames"] == 0


def test_missing_native_profile_field_is_rejected(tmp_path):
    env, _ = fake_env()
    observer = probe.NativeObserver(env, tmp_path)
    run_one_policy(env)
    data = fake_extras(env)
    del data["diagnostics"]["recovery_actual_profile_id"]
    with pytest.raises(ValueError, match="diagnostic field"):
        observer.observe(1, data)
    observer.sparse.close()


def test_censored_final_rows_keep_actual_held_zero_commands(tmp_path):
    env, _ = fake_env()
    observer = probe.NativeObserver(env, tmp_path)
    run_one_policy(env)
    observer.observe(600, fake_extras(env))
    report = observer.finish(tmp_path)
    assert report["first_birth"]["partitions"] == {"right_censored_unfinished": 4}
    assert report["policy_diagnostics"]["first_birth_only"]["prefix_release"]["nonzero_vx_command_frames"] == 0
