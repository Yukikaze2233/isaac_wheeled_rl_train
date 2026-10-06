"""Meaningful CPU control/ownership checks; no Isaac app/physics/GPU/PPO."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

HERE = Path(__file__).resolve().parent
SOURCE = Path("/tmp/scut-script-frozen-022-pbifeg_2")
AUDIT = HERE.parent
sys.path.insert(0, str(SOURCE / "src"))
from wheeled_tasks.chassis.recovery import RecoveryFSM, RecoveryPhase
from wheeled_tasks.chassis.recovery_training import RecoveryTraining


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load(HERE / "immediate_handoff_hook.py", "immediate_candidate_hook")
B = load(AUDIT / "probe_prefix_mean_stochastic_019.py", "immediate_actual55_observer")
M = load(AUDIT / "exploration_gate_review_021/mechanical_remaining_capture_candidate_021/mechanical_remaining_capture.py",
         "immediate_frozen_hookset")


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, allow_nan=False)


def setup(tmp_path, count=3, lag=.0):
    nominal = torch.tensor([0., .2, 0., 0., .2, 0.])
    fsm = RecoveryFSM(count, "cpu", {"side_speed_rad_s": 1.5, "complete_started_blend": True},
                      nominal, nominal, nominal, nominal, [1., 1., -1., -1.])
    q = nominal.expand(count, -1).clone()
    q[:, fsm.LEG_IDS] += lag
    fsm.reset(torch.arange(count), q, True)
    recovery = object.__new__(RecoveryTraining)
    recovery.fsm, recovery.enabled = fsm, fsm.enabled
    recovery.settings = {"complete_started_blend": True, "max_script_seconds": 8.}
    recovery.release_finished = torch.ones(count, dtype=torch.bool)
    recovery.premature_release = torch.zeros(count, dtype=torch.bool)
    recovery.reference_tick = 0
    recovery.policy_motion_hold = torch.zeros(count, dtype=torch.bool)
    recovery.learning_mask = torch.ones(count, dtype=torch.bool)
    recovery.elapsed = torch.zeros(count)
    recovery.wheel_probe_torque = torch.zeros(count, 2)
    recovery.project = lambda value: value
    recovery.pose_curriculum = types.SimpleNamespace(level_ids=torch.zeros(count, dtype=torch.long),
        profile_ids=torch.zeros(count, dtype=torch.long), profile_names=("fixture",))
    v5 = types.SimpleNamespace(LEGS=fsm.LEG_IDS, WHEELS=fsm.WHEEL_IDS,
        settings={"leg_kp": 160., "leg_kd": 5.}, wheel_prior={"kd": .6},
        current_motor_bounds=torch.tensor([40., 40., 4.5, 40., 40., 4.5]),
        requested_motor_effort=torch.zeros(count, 6))
    env = types.SimpleNamespace(recovery=recovery, num_envs=count, device="cpu", dt=.005, v5=v5,
                              scene_groups=["fixture"] * count, kinds=["flat"] * count)
    recovery.env = env
    observer = B.NativeObserver(env, tmp_path)
    return fsm, q, env, observer, nominal


def attach(tmp, fsm, env, observer, arm="immediate", expected_calls=2400):
    hook = H.ImmediateHandoffHook(env, observer, M.HookSet(), tmp, write, M.plain,
        arm=arm, expected_calls=expected_calls, cohort=lambda: [(i, 1) for i in range(len(fsm.phase))])
    hook.apply()
    return hook


def inputs(q, *, gyro=.1, dq=0., height=.305, support=True, clear=True):
    n = len(q)
    return (q, torch.full_like(q, dq), q.new_tensor([0., 0., -1.]).expand(n, 3),
            q.new_tensor([0., gyro, 0.]).expand(n, 3), q.new_full((n,), height),
            torch.full((n,), support), torch.full((n,), clear))


def compare_state(first, second):
    keys = set(vars(first)) - {"update", "_diagnostic_prepare_endpoint"}
    assert keys == set(vars(second)) - {"update", "_diagnostic_prepare_endpoint"}
    for name in keys:
        a, b = getattr(first, name), getattr(second, name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), name
        elif isinstance(a, dict) and any(isinstance(v, torch.Tensor) for v in a.values()):
            assert all(torch.equal(a[k], b[k]) for k in a), name
        else:
            assert a == b, name


def close(hook, observer):
    assert not hook.cleanup()
    assert observer.original_update is hook.original
    assert not hasattr(hook.fsm, "_diagnostic_prepare_endpoint")
    observer.sparse.close()


def test_baseline_real_observer_original_native_full_state_bitwise_unchanged(tmp_path):
    a, b = tmp_path / "native", tmp_path / "baseline"
    a.mkdir(); b.mkdir()
    f, q, e, o, _ = setup(a)
    g, _, ee, oo, _ = setup(b)
    hook = attach(b, g, ee, oo, arm="baseline", expected_calls=261)
    before_rng = torch.get_rng_state().clone()
    for tick in range(261):
        e.recovery.reference_tick = ee.recovery.reference_tick = tick
        f.update(*inputs(q)); g.update(*inputs(q))
        compare_state(f, g)
        assert torch.equal(o.gates, oo.gates)
        assert torch.equal(o.handover_ref, oo.handover_ref)
        assert torch.equal(o.release_ref, oo.release_ref)
    assert f.motion_released.all()
    assert hook.calls_completed == oo.reference_calls == 261
    assert torch.equal(before_rng, torch.get_rng_state())
    assert not hook.finish()
    assert hook.complete and oo.original_update is hook.original
    oo.sparse.close(); o.sparse.close()


def test_reference_endpoint_immediately_owns_native_torque_despite_bad_gates_and_q_lag(tmp_path):
    f, q, env, observer, nominal = setup(tmp_path, lag=.5)
    f.phase[:] = int(RecoveryPhase.PREPARE)
    f.script_targets[:] = nominal
    env.recovery.begin_policy()  # A PREPARE-start policy interval cannot gain learning credit.
    assert not env.recovery.learning_mask.any()
    hook = attach(tmp_path, f, env, observer)
    before = [v.clone() for v in inputs(q, gyro=2., dq=3., height=.18, support=False, clear=False)]
    original_raw_q = q.clone()
    f.update(*before)
    assert f.pure_rl.all() and f.blend.eq(1).all() and not f.motion_released.any()
    assert f.phase_ticks.eq(0).all() and not f.blending.any()
    assert torch.equal(q, original_raw_q) and torch.equal(f.continuous_q, original_raw_q)
    assert not f.stable_pass.any() and not f.stable_ticks.any()
    assert hook.first_global_intervention == 1 and observer.handover_ref.eq(1).all()
    actor_effort = torch.tensor([1., 2., .4, -1., -2., -.4]).expand(len(q), -1).clone()
    env.v5.requested_motor_effort[:] = actor_effort
    effort = env.recovery.torque(q, before[1], actor_effort)
    assert torch.equal(effort, actor_effort)  # Actual frozen native torque provider, all six axes.
    assert not env.recovery.learning_mask.any()
    env.recovery.begin_policy()
    assert env.recovery.learning_mask.all() and env.recovery.policy_motion_hold.all()
    close(hook, observer)
    with gzip.open(tmp_path / "immediate_handoff_endpoint_events.jsonl.gz", "rt") as stream:
        event = json.loads(next(stream))
    assert all(v["native_measured_pair_error_rad"] > .12 for v in event["selected"])
    assert all(v["native_remaining_rad"] < .02 for v in event["selected"])
    assert event["no_mechanical_convergence_or_functional_success_claim"]


def test_immediate_no_early_handoff_before_reference_completes_and_prefix_exact(tmp_path):
    a, b = tmp_path / "native", tmp_path / "candidate"
    a.mkdir(); b.mkdir()
    f, q, e, o, nominal = setup(a)
    g, _, ee, oo, _ = setup(b)
    for machine in (f, g):
        machine.phase[:] = int(RecoveryPhase.PREPARE)
        machine.script_targets[:, machine.LEG_IDS] = nominal[list(machine.LEG_IDS)] + .2
    hook = attach(b, g, ee, oo)
    for tick in range(8):
        e.recovery.reference_tick = ee.recovery.reference_tick = tick
        f.update(*inputs(q)); g.update(*inputs(q))
        compare_state(f, g)
        assert not g.pure_rl.any()
    while not g.pure_rl.all():
        ee.recovery.reference_tick += 1
        g.update(*inputs(q))
        assert hook.calls_completed <= 11
    assert hook.first_global_intervention > 8
    close(hook, oo); o.sparse.close()


def test_post_rl_continuous_one_second_still_required_for_motion_release(tmp_path):
    f, q, env, observer, _ = setup(tmp_path)
    f.phase[:] = int(RecoveryPhase.PREPARE)
    hook = attach(tmp_path, f, env, observer)
    f.update(*inputs(q))
    assert f.pure_rl.all() and not f.motion_released.any()
    for tick in range(199):
        env.recovery.reference_tick += 1
        f.update(*inputs(q))
    assert f.stable_ticks.eq(199).all() and not f.motion_released.any()
    env.recovery.reference_tick += 1
    f.update(*inputs(q))
    assert f.stable_ticks.eq(200).all() and f.motion_released.all()
    close(hook, observer)


@pytest.mark.parametrize("fault", ["nonfinite", "budget_unfinished", "takeover_tilt"])
def test_original_failure_guards_retained_and_rejection_evidence_preserved(tmp_path, fault):
    f, q, env, observer, nominal = setup(tmp_path)
    f.phase[:] = int(RecoveryPhase.PREPARE)
    hook = attach(tmp_path, f, env, observer)
    values = list(inputs(q))
    if fault == "nonfinite":
        values[0] = q.clone(); values[0][:, 0] = float("nan")
        expected = 1
    elif fault == "budget_unfinished":
        f.script_targets[:, f.LEG_IDS] += 1.
        f.age_ticks[:] = 1599
        expected = 6
    else:
        # Immediate actor takeover activates the already-existing >45deg guard.
        values[2] = q.new_tensor([1., 0., 0.]).expand(len(q), 3)
        values[5] = torch.zeros(len(q), dtype=torch.bool)
        f.script_targets[:] = nominal
        expected = 5
    f.update(*values)
    assert f.failed.all() and f.failure_code.eq(expected).all()
    assert not f.motion_released.any()
    close(hook, observer)
    receipt = json.loads((tmp_path / "immediate_handoff_hook_preclose.json").read_text())
    assert receipt["status"] == "incomplete_preserved" and not receipt["complete"]
    assert receipt["hooks_restored"] and receipt["config_unchanged"]


@pytest.mark.parametrize("arm", ["baseline", "immediate"])
def test_ast_scope_and_real_source_hash_are_bound(tmp_path, arm):
    f, _, _, observer, _ = setup(tmp_path)
    _, receipt = H.compile_update(observer.original_update, arm)
    assert receipt["source_sha256"] == hashlib.sha256((SOURCE / "src/wheeled_tasks/chassis/recovery.py").read_bytes()).hexdigest()
    assert len(receipt["modified_control_statements"]) == (0 if arm == "baseline" else 2)
    assert receipt["native_body_copies_executed_per_call"] == 1
    observer.sparse.close()


def test_partial_reset_and_foreign_hook_ownership_not_silently_restored(tmp_path):
    f, q, env, observer, _ = setup(tmp_path)
    hook = attach(tmp_path, f, env, observer)
    f.phase[:] = int(RecoveryPhase.PREPARE)
    f.update(*inputs(q))
    f.reset(torch.tensor([0]), q, True)
    assert f.phase[0] == int(RecoveryPhase.SELECT) and f.pure_rl[1:].all()
    observer.original_update = lambda *_: None
    errors = hook.cleanup()
    assert errors and any("ownership changed" in v for v in errors)
    assert observer.original_update is not hook.original
    observer.sparse.close()
