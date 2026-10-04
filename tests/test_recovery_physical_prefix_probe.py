"""CPU proof of authentication, birth guards and fixed-cohort accounting.

Fake physical buffers check capture ordering; they never certify physical FSM
success. The BLEND fractions separately come from the native CPU FSM.
"""
import builtins
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_recovery_physical_prefix as probe
from recovery_prefix_support import (
    ABI_KEYS, CASES, CohortLedger, StateWriteGuard, materialize, project_role,
    read_json, validate_checkpoint_payload, effective_control_ownership,
)
from wheeled_tasks.chassis.recovery import RecoveryFSM, RecoveryPhase as P
from test_recovery_training_integration import rig

WORKER = ROOT / "contracts/v6_task_campaign_recovery_worker_v8.json"
BASELINE = ROOT / "models/v6_model_delivery_20261003T2235/recommended_flat/model_final.pt"


def fake_env():
    data = NS(**{name: NS(torch=torch.zeros(4, n)) for name, n in (
        ("joint_pos", 18), ("joint_vel", 18), ("root_link_pose_w", 7),
        ("root_link_lin_vel_w", 3), ("root_com_ang_vel_b", 3), ("applied_torque", 18))})
    robot = NS(data=data, update=lambda dt: None)
    robot.write_calls = []
    for name in probe.WRITE_METHODS:
        setattr(robot, name, lambda *args, _name=name, **kwargs: robot.write_calls.append(_name))
    fsm = NS(phase=torch.zeros(4, dtype=torch.long), blend=torch.zeros(4), ready_ticks=torch.zeros(4),
             stable_ticks=torch.zeros(4), stable_peak_ticks=torch.zeros(4), stable_pass=torch.zeros(4, dtype=torch.bool),
             script_targets=torch.zeros(4, 6), motion_hold=torch.zeros(4, dtype=torch.bool),
             stable_pass_masks={k: torch.zeros(4, dtype=torch.bool) for k in probe.GATES})
    recovery = NS(fsm=fsm, release_finished=torch.zeros(4, dtype=torch.bool), reference_dt=.005,
                  reference_tick=0, advance_reference=lambda *args: None,
                  torque=lambda q, dq, torque: torque, outcome=lambda *args, **kwargs: False,
                  encoders=NS(), imu=NS(), probe=NS(), initial_release=torch.ones(4), elapsed=torch.zeros(4),
                  enabled=torch.ones(4, dtype=torch.bool), failed=torch.zeros(4, dtype=torch.bool),
                  pure_rl=torch.zeros(4, dtype=torch.bool), policy_motion_hold=torch.zeros(4, dtype=torch.bool),
                  learning_mask=torch.zeros(4, dtype=torch.bool))
    env = NS(num_envs=4, dt=.005, ids=[0, 1, 2, 3, 4, 5], scene_groups=list(CASES),
             robot=robot, history=NS(), recovery=recovery, v5=NS(motor_efforts=lambda q, dq, legs, wheels: q),
             actuator_response=NS(apply_torque=lambda torque: torque), command_transport=None,
             usb_transport=None, signal_delay=None, generator=torch.Generator().manual_seed(1),
             sim=NS(step=lambda **kwargs: None), closure_gap=lambda: torch.zeros(4),
             wheel_centers=lambda: torch.zeros(4, 2, 3))

    def reset(ids):
        robot.write_joint_position_to_sim_index(joint_pos=torch.zeros(len(ids), 18), env_ids=ids)
        robot.data.joint_pos.torch[ids] = 100.
    env.reset = reset
    return env


def recorder(tmp_path):
    result = probe.PrefixRecorder(fake_env(), tmp_path, 290929)
    result.guard.close_birth()
    return result


def test_native_v8_full_then_four_case_projection_and_no_app_import(tmp_path, monkeypatch):
    original = builtins.__import__
    forbidden = []

    def guarded(name, *args, **kwargs):
        if name == "isaaclab.app" or name == "wheeled_tasks.chassis.eval_env":
            forbidden.append(name)
            raise AssertionError("CPU preflight imported physical scene")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    cfg, manifest, control, stage, seed, count, receipt = materialize(ROOT, WORKER, tmp_path)
    assert (stage, seed, count) == ("mixed", 290929, 4)
    assert receipt["original_full_role_verified"]["num_envs"] == 252
    assert {c["name"] for c in cfg["evaluation"]["cases"]} == set(CASES)
    assert cfg["episode_seconds"] == 20. and cfg["physics_dt"] == .005 and cfg["policy_dt"] == .02
    assert cfg["signal_delay"]["enabled"] is False and cfg["command_transport"]["enabled"] is True
    assert manifest["urdf"] == "robot.urdf" and not forbidden
    from wheeled_tasks.chassis.preflight import design_preflight
    bad = deepcopy(cfg)
    bad["recovery_training"]["max_script_seconds"] = 7.
    with pytest.raises(ValueError):
        design_preflight(bad, control, ROOT, stage=stage, role="full", seed=seed, num_envs=count)


def test_preflight_entry_authenticates_relocated_full_pt_without_simulator(tmp_path, monkeypatch):
    if not BASELINE.exists():
        pytest.skip("Local ignored delivery checkpoint is absent")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("isaaclab.app"):
            raise AssertionError("Preflight-only must not initialize AppLauncher")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "preflight"
    assert probe.main(["--contract", str(WORKER), "--checkpoint", str(BASELINE),
                       "--output", str(output), "--preflight-only"]) == 0
    report = read_json(output / "report.json")
    assert report["status"] == "preflight_passed_scene_not_run" and report["execution_status"] == "not_run"
    assert report["num_envs"] == 4 and report["formal_updates_consumed"] == 0 and not report["qualification"]
    assert report["checkpoint"]["checkpoint_learning_updates"] == 14020
    assert report["checkpoint"]["checkpoint_clocks"]["task_actor_update"] == 13000
    assert not report["source_files_changed"]


def test_failed_scene_start_reports_failed_execution_and_unchanged_source_pt(tmp_path, monkeypatch):
    if not BASELINE.exists():
        pytest.skip("Local ignored delivery checkpoint is absent")
    original = builtins.__import__

    def fail_scene(name, *args, **kwargs):
        if name == "isaaclab.app":
            raise RuntimeError("Injected scene startup failure; no AppLauncher initialization")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fail_scene)
    output = tmp_path / "failed_startup"
    assert probe.main(["--contract", str(WORKER), "--checkpoint", str(BASELINE), "--output", str(output)]) == 1
    report = read_json(output / "report.json")
    assert report["status"] == report["execution_status"] == "failed"
    assert report["checkpoint_unchanged"] and report["physics_steps"] == 0
    assert report["elapsed_physics_s"] == 0. and report["optimizer_steps"] == 0


def test_wrong_worker_and_projection_population_are_rejected(tmp_path):
    path = tmp_path / "worker.json"
    bad = read_json(WORKER)
    bad["physics_dt"] = .001
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="V8 recovery worker SHA"):
        materialize(ROOT, path, tmp_path)
    with pytest.raises(ValueError, match="replicas"):
        project_role({}, {}, 2)


def payload():
    source = {key: key for key in ABI_KEYS}
    saved = {name: {"weight": torch.zeros(1)} for name in
             ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")}
    saved["infos"] = {"contract_sha256": "source", "asset_manifest_sha256": source["asset_manifest_sha256"],
                      "control_math_sha256": source["control_math_sha256"], "successful_updates_total": 14,
                      "task_actor_update": 10}
    return saved, source


@pytest.mark.parametrize("defect", ["empty_actor", "missing_optimizer", "infos_sha", "clock", "asset", "abi"])
def test_full_learning_state_and_separate_clocks_are_authenticated(defect):
    saved, source = payload()
    target = deepcopy(source)
    assert validate_checkpoint_payload(saved, source, target, "source", 14)["task_actor_update"] == 10
    if defect == "empty_actor": saved["actor_state_dict"] = {}
    elif defect == "missing_optimizer": del saved["optimizer_state_dict"]
    elif defect == "infos_sha": saved["infos"]["contract_sha256"] = "wrong"
    elif defect == "clock": saved["infos"]["successful_updates_total"] = 10
    elif defect == "asset": saved["infos"]["asset_manifest_sha256"] = "wrong"
    else: target["actor_dim"] = "wrong"
    with pytest.raises(ValueError):
        validate_checkpoint_payload(saved, source, target, "source", 14)


def test_birth_transaction_guard_and_closed_lane_reset_do_not_reopen_cohort():
    guard = StateWriteGuard(4)
    guard.write("q18", range(4))
    guard.write("q18", range(4))  # parent and FixedCaseEnv may both initialize.
    guard.close_birth()
    with pytest.raises(ValueError, match="after birth"):
        guard.write("q18", [0])
    with pytest.raises(ValueError, match="Invalid"):
        guard.write("q18", [4])
    guard.close_terminal([0])
    guard.write("q18", [0])
    assert guard.live == {1, 2, 3}
    with pytest.raises(ValueError, match="Duplicate"):
        guard.close_terminal([0])


def test_fixed_birth_denominator_includes_unfinished_and_native_overlap():
    ledger = CohortLedger(list(CASES), 290929)
    ledger.event(0, "first_pure_rl_post_physics", 2.)
    ledger.finish(0, success=True, terminated=False, boundary=False, timeout=False, time_s=4., reasons={})
    ledger.finish(1, success=True, terminated=True, boundary=True, timeout=False, time_s=4., reasons={"fall": True})
    summary = ledger.summary(censor_time_s=7.5)
    assert summary["fixed_birth_denominator"] == 4 and summary["success_per_seeded_cohort"] == .25
    assert summary["partitions"]["failure"] == 1 and summary["partitions"]["right_censored_unfinished"] == 2
    assert summary["events_per_seeded_cohort"]["first_pure_rl_post_physics"]["fraction"] == .25
    assert summary["events_per_seeded_cohort"]["phase_BLEND"] == {"numerator": 0, "denominator": 4, "fraction": 0.}
    assert summary["rows"][2]["censor_time_s"] == 7.5 and "censor_time_s" not in summary["rows"][0]
    assert not summary["qualification"] and not summary["physx_internal_state_restore_claim"]
    with pytest.raises(ValueError, match="after"):
        ledger.event(0, "new_episode", 5.)


def test_write_wrappers_reject_before_mutation_and_capture_before_autoreset(tmp_path):
    r = recorder(tmp_path)
    robot = r.env.robot
    with pytest.raises(ValueError, match="after birth"):
        robot.write_joint_position_to_sim_index(joint_pos=torch.zeros(1, 18), env_ids=torch.tensor([0]))
    assert not robot.write_calls
    with pytest.raises(ValueError, match="outside native terminal"):
        r.env.reset(torch.tensor([0]))
    robot.data.joint_pos.torch[0] = 3.
    r.in_policy, r.tick, r.outcome_tick = True, 1, 1
    r.env.reset(torch.tensor([0]))
    assert r.pending_terminals == {0} and r.guard.live == {1, 2, 3}
    assert r.snapshots[0]["physics.joint_pos"][0].eq(3.).all()
    assert robot.data.joint_pos.torch[0].eq(100.).all()
    data = {"reasons": {"fall": torch.tensor([True, False, False, False])},
            "recovery_true_success": torch.zeros(4, dtype=torch.bool),
            "success": torch.zeros(4, dtype=torch.bool),
            "terminated": torch.tensor([True, False, False, False])}
    extras = {"diagnostics": data, "policy_learning_mask": torch.zeros(4, dtype=torch.bool),
              "time_outs": torch.zeros(4, dtype=torch.bool)}
    r.after_policy(torch.zeros(4), torch.tensor([True, False, False, False]), extras,
                   NS(torque_calls=4, expected=extras["policy_learning_mask"]))
    assert r.ledger.rows[0]["partition"] == "failure"
    assert r.completed_policy_intervals == 1
    r.restore_wrappers()
    robot.write_joint_position_to_sim_index(joint_pos=torch.zeros(1, 18), env_ids=torch.tensor([1]))
    assert len(robot.write_calls) == 2


def test_control_wrappers_are_identity_and_physics_clock_has_no_reset_steps(tmp_path):
    r = recorder(tmp_path)
    q, dq, legs, wheels = torch.zeros(4, 6), torch.zeros(4, 6), torch.zeros(4, 4), torch.zeros(4, 2)
    r.before_policy(1, {"policy": torch.zeros(4, 35)}, torch.zeros(4, 6))
    effort = r.env.v5.motor_efforts(q, dq, legs, wheels)
    assert effort is q and r.env.recovery.torque(q, dq, effort) is effort
    assert r.env.actuator_response.apply_torque(effort) is effort
    r.env.robot.update(.005)
    assert r.physics_tick == 0  # no preceding sim.step; reset-only update.
    r.env.sim.step(render=False)
    r.env.robot.update(.005)
    assert r.physics_tick == 1 and len(r.frames) == 4
    assert torch.equal(q, torch.zeros_like(q))


def native_blend_fsm():
    target = torch.tensor([.1, .7, 0., -.1, -.7, 0.])
    fsm = RecoveryFSM(4, "cpu", {"complete_started_blend": True}, target, target, target, target,
                      [1., 1., -1., -1.])
    fsm.reset(torch.arange(4), target.repeat(4, 1), True)
    # This CPU fixture starts at a reference boundary. The real entry never
    # writes this phase; it records transitions made by the physical FSM.
    fsm.phase.fill_(int(P.BLEND))
    return fsm


def physical_control_cache(phase, fraction, *, releasing=False, hold=True):
    return {"phase_used": phase.clone(), "blend_fraction": fraction.clone(),
        "post_native_mix_torque": torch.zeros(4, 6), "recovery_enabled_used": torch.ones(4, dtype=torch.bool),
        "release_finished_used": torch.full((4,), not releasing), "motor_enabled_used": torch.full((4,), not releasing),
        "native_pure_rl_used": (phase == 9) & ~torch.full((4,), releasing),
        "policy_motion_hold_used": torch.full((4,), hold), "fsm_motion_hold_used": torch.full((4,), hold),
        "native_policy_learning_mask_after_torque": (phase == 9) & ~torch.full((4,), releasing),
        "initial_release_s": torch.ones(4), "release_elapsed_s_before_torque": torch.zeros(4)}


def test_native_blend_40_ticks_preserves_last_mixed_and_first_pure_rl(tmp_path):
    r = recorder(tmp_path)
    fsm = native_blend_fsm()
    r.env.recovery.fsm = fsm
    for step in range(41):
        r.physics_tick = step + 1
        r.env.robot.data.joint_pos.torch.fill_(step)
        r.cache = physical_control_cache(fsm.phase, fsm.blend)
        r.after_physics()
        fsm.update(fsm.script_targets.clone(), torch.zeros(4, 6), torch.tensor([[0., 0., -1.]]).repeat(4, 1),
                   torch.zeros(4, 3), torch.full((4,), .305), torch.ones(4, dtype=torch.bool),
                   torch.ones(4, dtype=torch.bool))
    assert set(r.blend_counts.values()) == {40}
    events = [e for e in r.events if e["lane"] == 0]
    last = next(e for e in events if e["event"] == "blend_last_mixed_post_physics")
    pure = next(e for e in events if e["event"] == "first_pure_rl_post_physics")
    assert last["physical_tick"] == 40 and pure["physical_tick"] == 41
    assert r.snapshots[last["snapshot_index"]]["physics.joint_pos"][0].eq(39.).all()
    assert r.snapshots[pure["snapshot_index"]]["physics.joint_pos"][0].eq(40.).all()
    assert len([e for e in events if e["event"] == "first_pure_rl_post_physics"]) == 1


def test_pure_rl_without_complete_physical_blend_is_rejected(tmp_path):
    r = recorder(tmp_path)
    r.cache = physical_control_cache(torch.full((4,), 9), torch.ones(4))
    with pytest.raises(ValueError, match="full native 200ms physical BLEND"):
        r.after_physics()


def test_uninitialized_history_infinity_is_marked_without_hiding_physics_nan(tmp_path):
    r = recorder(tmp_path)
    r.env.recovery.probe.age = torch.full((4, 2, 2), float("inf"))
    snapshot = r.snapshot()
    assert torch.isinf(snapshot["support_probe.age"]).all()
    assert probe.json_value(snapshot["support_probe.age"])[0][0][0] is None
    r.env.robot.data.joint_pos.torch[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite physical state"):
        r.snapshot()


def test_timeout_and_boundary_use_native_outcome_without_inferring_success():
    ledger = CohortLedger(list(CASES), 290929)
    ledger.finish(0, success=False, terminated=False, boundary=False, timeout=True, time_s=20., reasons={})
    ledger.finish(1, success=False, terminated=False, boundary=True, timeout=True, time_s=4., reasons={"boundary": True})
    # Native episode_outcome permits success overlapping a boundary reason.
    ledger.finish(2, success=True, terminated=False, boundary=True, timeout=False, time_s=5., reasons={"boundary": True})
    with pytest.raises(ValueError, match="not a native terminal"):
        ledger.finish(3, success=False, terminated=False, boundary=False, timeout=False, time_s=6., reasons={})
    summary = ledger.summary(censor_time_s=20.)
    assert summary["partitions"] == {"success": 1, "failure": 0, "timeout": 1,
                                     "boundary_truncation": 1, "right_censored_unfinished": 1}


def test_initial_release_raw_rl_placeholder_records_zero_without_handover(tmp_path):
    r = recorder(tmp_path)
    r.before_policy(1, {"policy": torch.zeros(4, 35)}, torch.zeros(4, 6))
    r.env.recovery.fsm.phase.fill_(9)
    r.env.recovery.fsm.blend.fill_(1.)
    # This reproduces the native reset placeholder; release is unfinished,
    # motor/actor ownership false even though the raw FSM reads RL/blend=1.
    r.cache = physical_control_cache(torch.full((4,), 9), torch.ones(4), releasing=True, hold=False)
    for tick in range(1, 201):
        r.physics_tick = tick
        r.after_physics()
    summary = r.summary()
    assert summary["fixed_birth_denominator"] == 4 and not r.blend_counts
    assert summary["events_per_seeded_cohort"]["first_pure_rl_post_physics"]["numerator"] == 0
    assert summary["events_per_seeded_cohort"]["passive_zero_effort_post_physics"]["numerator"] == 4
    assert all(frame["raw_fsm_phase_used"] == 9 and frame["blend_fraction_used"] == 1.
               and frame["effective_control_ownership"] == "passive_release" for frame in r.frames)
    assert all(v == 200 for v in r.release_evidence_counts.values())


def test_passive_upstream_zero_does_not_assert_away_retained_drive_output(tmp_path):
    r = recorder(tmp_path)
    r.cache = physical_control_cache(torch.full((4,), 9), torch.ones(4), releasing=True, hold=False)
    r.env.robot.data.applied_torque.torch[:, r.env.ids] = .25
    r.after_physics()
    assert all(frame["effective_control_ownership"] == "passive_release" for frame in r.frames)
    assert all(not e["applied_zero"] for e in r.events)
    assert all(r.release_evidence_counts[lane, "zero_applied_effort_ticks"] == 0 for lane in range(4))


@pytest.mark.parametrize("mask", [False, True])
def test_instantaneous_rl_hold_does_not_require_whole_interval_learning_credit(mask):
    # mask is the native interval's separate qualification, possibly still
    # false because an earlier substep in this interval was BLEND.
    cache = physical_control_cache(torch.full((4,), 9), torch.ones(4))
    cache["native_policy_learning_mask_after_torque"].fill_(mask)
    assert effective_control_ownership(phase=9, enabled=True, release_finished=True, motor_enabled=True,
                                      native_pure_rl=True, policy_motion_hold=True) == "rl_hold"


def test_true_native_initial_release_uses_rl_placeholder_without_actor_ownership(rig):
    env, recovery = rig
    # Reuse the existing native RecoveryTraining fixture, not a replacement
    # controller. It resets a released pose through the same implementation.
    assert recovery.fsm.phase[0] == P.RL and recovery.fsm.blend[0] == 1.
    assert not recovery.release_finished[0] and not recovery.pure_rl[0]
    recovery.begin_policy()
    q, dq = env.nominal[:, env.ids], torch.zeros(2, 6)
    env.v5.motor_efforts(q, dq, q[:, [0, 1, 3, 4]], dq[:, [2, 5]])
    enabled_before = bool((recovery.release_finished & ~recovery.failed)[0])
    output = recovery.torque(q, dq, torch.ones(2, 6))
    assert output[0].eq(0.).all() and not recovery.learning_mask[0]
    assert effective_control_ownership(phase=int(recovery.fsm.phase[0]), enabled=True,
        release_finished=bool(recovery.release_finished[0]), motor_enabled=enabled_before,
        native_pure_rl=bool(recovery.pure_rl[0]), policy_motion_hold=bool(recovery.policy_motion_hold[0])) == "passive_release"
