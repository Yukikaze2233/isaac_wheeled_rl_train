"""CPU execution of production reward observers and env dispatch; no Kit/PPO."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.scut_flat_reward import WEIGHTS
from wheeled_tasks.chassis.scut_flat_reward_runtime import (
    ROOT_VELOCITY_REFERENCE, WHEEL_POSITION_REFERENCE, ScutFlatRewardRuntime,
    root_angles_xyzw, wheel_link_positions_body,
)
from wheeled_tasks.chassis.preflight import SubstepControlAudit
from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL
from wheeled_tasks.chassis.task import PhaseTracker


ROOT = Path(__file__).resolve().parents[1]


def runtime(n=2):
    return ScutFlatRewardRuntime(n, "cpu", ["base_link", "L_wheel", "R_wheel"], {
        "root_velocity_reference": ROOT_VELOCITY_REFERENCE,
        "root_height_reference": "root_link_z_minus_env_origin",
        "wheel_position_reference": WHEEL_POSITION_REFERENCE,
        "undesired_link_names": ["base_link"], "terminal_link_names": ["base_link"]})


def methods(*names):
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    nodes = [deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    context = {"torch": torch, "stage_actor_updates": lambda env: 0,
               "root_link_linear_velocity": lambda env: env.robot.data.root_com_lin_vel_b.torch,
               "__name__": "wheeled_tasks.chassis.env", "__package__": "wheeled_tasks.chassis"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "production_env_reward_methods", "exec"), context)
    return context


def field(value):
    return SimpleNamespace(torch=value)


def env(n=2):
    r = runtime(n)
    root = torch.zeros(n, 7)
    root[:, 2], root[:, 6] = .42, 1.
    bodies = torch.zeros(n, 3, 7)
    bodies[:, :, 6] = 1.
    bodies[:, 1:, 0] = .2
    data = SimpleNamespace(root_link_pose_w=field(root),
        root_com_lin_vel_b=field(torch.zeros(n, 3)), root_com_ang_vel_b=field(torch.zeros(n, 3)),
        projected_gravity_b=field(torch.tensor([[0., 0., -1.]]).repeat(n, 1)),
        joint_pos=field(torch.zeros(n, 6)), joint_vel=field(torch.zeros(n, 6)),
        joint_acc=field(torch.zeros(n, 6)), applied_torque=field(torch.zeros(n, 6)),
        body_link_pose_w=field(bodies))
    e = SimpleNamespace(scut_reward=r, robot=SimpleNamespace(data=data), num_envs=n,
        origins=torch.zeros(n, 3), cfg={"record_diagnostics": True, "closure_gap_termination_m": .003},
        closure_gap_peak=torch.zeros(n), max_closure_gap=0.,
        max_closure_gap_all_finite=0., max_closure_gap_numerical_reset=0.,
        knee_ids=[0, 3], knee_bounds={"L_joint2": (-1., 1.), "R_jonit2": (-1., 1.)},
        spring_ids=[1, 4], v5=SimpleNamespace(stroke=.08,
            spring_state=lambda q, dq: (torch.zeros(n, 2), torch.zeros(n, 2))),
        spring_pressure_element=None, _spring_pressure_frame=None,
        spring_reserve_frames=0, episode_length_buf=torch.ones(n, dtype=torch.long),
        episode_limits=torch.full((n,), 1000, dtype=torch.long), wheel_ids=[1, 2], ids=list(range(6)),
        commands=torch.tensor([[0., 0., .42]]).repeat(n, 1),
        contact_force=torch.zeros(n, 3, 3), phase=PhaseTracker(n, "cpu", .02),
        phase_counts=torch.zeros(5, dtype=torch.long), termination_counts={}, timeout_count=0,
        sampling_metrics={}, command_transport=None, feedback_transport=None,
        signal_delay=None, usb_transport=None, round_trip_transport=None, dynamics_randomization=None,
        nonstandard_start=None, ordinary_posture=None,
        get_observations=lambda: {"fake_observations": True})
    e._advance_policy_state = lambda done, terminated, extras: None
    e.finish = lambda: methods("_finish_scut_flat_step")["_finish_scut_flat_step"](e)
    return e


def test_raw_history_is_owned_and_selective_reset_does_not_destroy_other_lanes():
    r = runtime()
    a = torch.arange(12.).reshape(2, 6)
    r.begin_policy(a)
    a.zero_()
    assert r.raw_action[1, 5] == 11.
    r.begin_policy(torch.full((2, 6), 20.))
    r.begin_policy(torch.full((2, 6), -30.))
    assert r.previous_raw_action[1, 5] == 20.
    assert r.before_previous_raw_action[1, 5] == 11.
    r.terminal_counter[:] = 19
    r.contact_history[:] = 10.
    r.reset(torch.tensor([0]))
    assert not r.raw_action[0].any() and not r.contact_history[0].any()
    assert r.raw_action[1, 5] == -30. and r.terminal_counter[1] == 19


def test_native_physical_axes_and_final_5ms_sample_are_preserved():
    r = runtime()
    for sample in range(4):
        qd = torch.arange(12.).reshape(2, 6) + 10. * sample
        r.observe_physics(qd, 100. + qd, 200. + qd)
    assert torch.equal(r.joint_velocity, qd[:, POLICY_FROM_CONTROL])
    assert torch.equal(r.joint_acceleration, (100. + qd)[:, POLICY_FROM_CONTROL])
    assert torch.equal(r.applied_torque, (200. + qd)[:, POLICY_FROM_CONTROL])


def test_three_physics_contacts_and_current_terminal_are_distinct():
    r = runtime()
    for force in (10., 0., 0.):
        r.net_contact[:, 0, 2] = force
        r.finish_contacts()
    assert torch.equal(r.contact_history[:, :, 0, 2], torch.tensor([[10., 0., 0.]]).repeat(2, 1))
    q = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
    terminated, _, _ = r.termination(torch.zeros(2, 3), torch.zeros(2, 3), q)
    assert not terminated.any() and not r.raw_terminal.any()
    r.finish_contacts()
    assert not r.contact_history.any()


@pytest.mark.parametrize("cause", ["contact", "roll", "pitch", "alternating"])
def test_source_or_condition_requires_twenty_policy_ticks(cause):
    r = runtime()
    v = torch.zeros(2, 3)
    for tick in range(20):
        q = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
        r.net_contact.zero_()
        if cause == "contact" or cause == "alternating" and tick % 2 == 0:
            r.net_contact[:, 0, 2] = 1.01
        else:
            axis = 1 if cause == "pitch" else 0
            angle = math.radians(41.)
            q[:, axis], q[:, 3] = math.sin(angle / 2), math.cos(angle / 2)
        terminated, numeric, confirmed = r.termination(v, v, q)
        assert bool(terminated.all()) == (tick == 19)
        assert torch.equal(confirmed, terminated) and not numeric.any()
    r.net_contact.zero_()
    r.termination(v, v, torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1))
    assert not r.terminal_counter.any()


@pytest.mark.parametrize("field_name,limit", [("joint", 500.), ("velocity", 100.), ("omega", 200.)])
@pytest.mark.parametrize("bad", ["above", "nan", "inf"])
def test_numeric_reset_is_immediate_per_lane_and_strict_above_limit(field_name, limit, bad):
    r = runtime()
    v, omega = torch.zeros(2, 3), torch.zeros(2, 3)
    value = {"joint": r.joint_velocity, "velocity": v, "omega": omega}[field_name]
    value[0, 0] = limit
    q = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
    assert not r.termination(v, omega, q)[1].any()
    value[0, 0] = {"above": limit + 1., "nan": float("nan"), "inf": float("inf")}[bad]
    terminated, numerical, confirmed = r.termination(v, omega, q)
    assert torch.equal(numerical, torch.tensor([True, False]))
    assert torch.equal(terminated, numerical) and not confirmed.any()


def test_xyzw_angles_and_wheel_link_origin_inverse_rotation():
    q = torch.tensor([[0., math.sin(math.pi / 8), 0., math.cos(math.pi / 8)]])
    roll, pitch = root_angles_xyzw(q)
    assert roll.item() == 0. and pitch.item() == pytest.approx(math.pi / 4)
    root = torch.tensor([[3., 2., 1., 0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]])
    wheels = torch.tensor([[[3., 3., 1.], [3., 1., 1.]]])
    assert torch.allclose(wheel_link_positions_body(root, wheels),
                          torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]]), atol=2e-7)


def test_numeric_failure_does_not_advance_delayed_terminal_counter():
    r = runtime()
    r.terminal_counter[0] = 19
    r.net_contact[0, 0, 2] = 2.
    r.joint_velocity[0, 0] = 501.
    terminated, numeric, confirmed = r.termination(torch.zeros(2, 3), torch.zeros(2, 3),
        torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1))
    assert terminated[0] and numeric[0] and not confirmed[0]
    assert r.terminal_counter[0] == 0


def test_exclusive_source_payments_are_logged_once_and_sum_to_returned_reward():
    e = env()
    _, reward, done, extras = e.finish()
    assert not done.any()
    assert set(extras["native_reward_components"]) == set(WEIGHTS)
    assert torch.equal(reward, torch.stack(list(extras["native_reward_components"].values()), -1).sum(-1))
    assert reward[0].item() == pytest.approx(.1)
    for name, paid in extras["native_reward_components"].items():
        assert torch.equal(extras["log"]["/reward/" + name], paid.mean())
    assert "already_dt_scaled" in extras["native_reward_component_units"]
    assert not any("event" in name or "tail" in name for name in extras["native_reward_components"])


@pytest.mark.parametrize("kind", ["time", "boundary", "knee", "spring", "closure", "confirmed", "numeric"])
def test_real_termination_vs_truncation_and_numerical_zero_row(kind):
    e = env()
    if kind == "time":
        e.episode_length_buf[0] = 999
    elif kind == "boundary":
        e.robot.data.root_link_pose_w.torch[0, 0] = 4.
    elif kind == "knee":
        e.robot.data.joint_pos.torch[0, 0] = 2.
    elif kind == "spring":
        compression = torch.tensor([[.082, 0.], [0., 0.]])
        e.v5.spring_state = lambda q, dq: (compression, torch.zeros_like(compression))
    elif kind == "closure":
        e.closure_gap_peak[0] = .00301
    elif kind == "confirmed":
        e.scut_reward.terminal_counter[0] = 19
        e.scut_reward.net_contact[0, 0, 2] = 2.
    else:
        e.scut_reward.joint_velocity[0, 0] = 501.
    _, reward, done, extras = e.finish()
    assert torch.equal(done, torch.tensor([True, False]))
    pay = extras["native_reward_components"]["termination"]
    assert pay[0].item() == (0. if kind in ("time", "boundary", "numeric") else -4.)
    assert bool(extras["time_outs"][0]) == (kind in ("time", "boundary"))
    if kind == "numeric":
        assert reward[0] == 0.
        assert all(paid[0] == 0. for paid in extras["native_reward_components"].values())
    assert reward[1].item() == pytest.approx(.1)


def test_mechanical_corruption_remains_a_named_local_abort():
    e = env()
    e.robot.data.joint_pos.torch[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="outside the source numerical reset"):
        e.finish()


def test_raw_closure_peak_keeps_numeric_reset_lane_without_changing_payments_or_done():
    e = env(3)
    e.cfg.update(formal_v7_flat=True, formal_closure_monitor_only=True,
                 closure_gap_termination_m=None)
    e.closure_gap_peak[:] = torch.tensor([.002, .08, 1.136453])
    e.scut_reward.joint_velocity[2, 0] = 501.
    _, reward, done, extras = e.finish()
    assert done.tolist() == [False, False, True]
    assert reward.tolist() == pytest.approx([.1, .1, 0.])
    assert e.max_closure_gap == pytest.approx(.08)
    assert e.max_closure_gap_all_finite == pytest.approx(1.136453)
    assert e.max_closure_gap_numerical_reset == pytest.approx(1.136453)
    assert extras["log"]["/diagnostics/closure/all_finite_peak_since_start_m"] == pytest.approx(1.136453)
    assert extras["log"]["/diagnostics/closure/numerical_reset_peak_since_start_m"] == pytest.approx(1.136453)
    assert extras["diagnostics"]["gap"].tolist() == pytest.approx([.002, .08, 1.136453])
    e.scut_reward.joint_velocity.zero_()
    e.closure_gap_peak.fill_(.001)
    _, _, done, _ = e.finish()
    assert not done.any()
    assert e.max_closure_gap_all_finite == pytest.approx(1.136453)
    assert e.max_closure_gap_numerical_reset == pytest.approx(1.136453)


def test_formal_gap_reset_is_per_lane_failure_and_historical_peak_does_not_retrigger():
    e = env(3)
    e.cfg.update(formal_v7_flat=True, formal_closure_monitor_only=False)
    e.closure_gap_peak[:] = torch.tensor([.003, .0031, .05])
    e.episode_length_buf[:] = 999  # A simultaneous timeout must not mask failure.
    observed = []
    def advance(done, terminated, extras):
        observed.append((done.clone(), terminated.clone()))
        e.closure_gap_peak[done] = 0.
        e.episode_length_buf[done] = 0
        e.scut_reward.reset(done.nonzero().flatten())
    e._advance_policy_state = advance
    _, _, done, extras = e.finish()
    assert done.tolist() == [True, True, True]
    assert observed[0][1].tolist() == [False, True, True]
    assert extras["time_outs"].tolist() == [True, False, False]
    assert extras["native_reward_components"]["termination"].tolist() == [0., -4., -4.]
    assert not extras["diagnostics"]["reasons"]["numerical_safety"].any()
    assert e.max_closure_gap == pytest.approx(.05)
    _, _, done, extras = e.finish()
    assert not done.any()
    assert not extras["native_reward_components"]["termination"].any()
    assert e.max_closure_gap == pytest.approx(.05)


def test_concurrent_failure_and_timeout_uses_explicit_local_failure_precedence():
    e = env()
    e.episode_length_buf[0] = 999
    e.robot.data.joint_pos.torch[0, 0] = 2.
    _, _, _, extras = e.finish()
    assert extras["diagnostics"]["terminated"][0] and not extras["time_outs"][0]
    assert e.scut_reward.summary()["concurrent_terminal_precedence"] == "termination"


def test_masks_and_raw_history_diagnostics_survive_auto_reset():
    e = env()
    e.scut_reward.begin_policy(torch.full((2, 6), 3.))
    e.scut_reward.joint_velocity[0, 0] = 501.
    held_mask = torch.tensor([False, True])
    e.nonstandard_start = SimpleNamespace(learning_mask=held_mask,
        metrics=lambda clock: {}, diagnostics=lambda: {})
    def advance(done, terminated, extras):
        e.scut_reward.reset(done.nonzero().flatten())
        held_mask.fill_(True)
    e._advance_policy_state = advance
    _, _, _, extras = e.finish()
    assert torch.equal(extras["policy_learning_mask"], torch.tensor([False, True]))
    assert torch.equal(extras["diagnostics"]["raw_actor_actions"], torch.full((2, 6), 3.))
    assert not e.scut_reward.raw_action[0].any()


def test_actual_numeric_reset_diagnostics_do_not_poison_episode_metrics():
    from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
    e = env()
    e.robot.data.root_com_lin_vel_b.torch[0, 0] = float("nan")
    metrics = EpisodeMetrics(["left_case", "right_case"], "cpu", .02,
                             height_range_m=(.3, .5))
    diagnostics = e.finish()[3]["diagnostics"]
    assert torch.isnan(diagnostics["velocity"][0, 0])  # Preserve raw failure evidence.
    metrics.observe(diagnostics)
    assert torch.equal(metrics.frames, torch.tensor([0, 1]))
    assert torch.equal(metrics.observed_frames, torch.tensor([1, 1]))
    assert torch.equal(metrics.failures, torch.tensor([1, 0]))
    assert torch.isfinite(metrics.sums).all() and torch.isfinite(metrics.torque_peak).all()
    e.robot.data.root_com_lin_vel_b.torch.zero_()
    e.robot.data.root_link_pose_w.torch[0, 2] = .44
    metrics.observe(e.finish()[3]["diagnostics"])
    assert metrics.frames[0] == 1 and metrics.failures[0] == 1
    assert metrics.sums[0, 4].item() == pytest.approx(.02, abs=1e-7)


def test_actual_step_collects_four_native_samples_and_bypasses_legacy_reward():
    e = env(1)
    e.cfg = {}
    e.critic78 = False
    e.physical_tick = e.tick = 0
    e.episode_length_buf.zero_()
    e.decimation, e.dt = 4, .005
    e.control_audit = SubstepControlAudit(.005, .02)
    e.skills = e.recovery = e.signal_delay = e.jump_assist = e.perturbations = None
    e._pc_feedback_transport = None
    e.scut35, e.is_v5 = True, True
    e.actions = torch.zeros(1, 6)
    e.previous_actions, e.before_previous_actions = torch.zeros_like(e.actions), torch.zeros_like(e.actions)
    e.v5.decode = lambda actions, q: (actions[:, :4], actions[:, 4:], actions.clamp(-1., 1.))
    e.v5.motor_efforts = lambda *args: torch.ones(1, 6)
    e.v5.spring_efforts = lambda position: torch.zeros_like(position)
    e.actuator_response = SimpleNamespace(enabled=False)
    e.nominal, e.motor_strength, e.spring_strength = torch.zeros(1, 6), 1., 1.
    e.spring_ids = []
    e.sim = SimpleNamespace(step=lambda **kwargs: None)
    e.torque_monitor = e.references = e.wheel_force_peak_n = None
    e.contact_peak = torch.zeros(1)
    e.body_count = 3
    e.startup_report = {"control_timing": {}}
    calls = []
    def refresh(env, dt):
        assert dt == .005
        sample = len(calls) + 1
        calls.append(sample)
        e.robot.data.joint_vel.torch[:] = torch.arange(6.) + sample * 10.
        e.robot.data.joint_acc.torch[:] = torch.arange(6.) + sample * 100.
        e.robot.data.applied_torque.torch[:] = torch.arange(6.) + sample
    e.robot.set_joint_effort_target_index = lambda **kwargs: None
    e.robot.write_data_to_sim = lambda: None
    contact_view = SimpleNamespace(
        get_contact_force_matrix=lambda **kwargs: torch.zeros(3, 1, 3),
        get_net_contact_forces=lambda **kwargs: torch.full((3, 3), float(len(calls))))
    e.contact_views = [(torch.tensor([0]), contact_view)]
    gaps = [.0001, .004, .0002, .0001]
    e.closure_gap = lambda: torch.tensor([gaps[len(calls) - 1]])
    e.nonstandard_start = SimpleNamespace(begin_policy=lambda: None, holding=torch.tensor([True]),
        apply_torque=lambda torque: torch.zeros_like(torque), advance=lambda: None,
        effective_action_history=lambda actions: torch.zeros_like(actions))
    e._finish_scut_flat_step = lambda: "source-exclusive-return"
    e.state = lambda: pytest.fail("Source reward fell through to legacy state/reward")
    context = methods("step")
    context.update(update_robot_data=refresh, wp=SimpleNamespace(to_torch=lambda value: value))
    incoming = torch.tensor([[2., 3., 4., 5., 6., 7.]])
    assert context["step"](e, incoming) == "source-exclusive-return"
    assert calls == [1, 2, 3, 4] and e.physical_tick == 4
    assert torch.equal(e.scut_reward.raw_action, incoming)
    assert not e.actions.any()  # Hold history is distinct from source raw history.
    assert torch.equal(e.scut_reward.joint_acceleration, (torch.arange(6.) + 400.)[None, POLICY_FROM_CONTROL])
    assert torch.equal(e.scut_reward.applied_torque, (torch.arange(6.) + 4.)[None, POLICY_FROM_CONTROL])
    assert torch.equal(e.scut_reward.contact_history[0, :, 0, 2], torch.tensor([2., 3., 4.]))
    assert e.closure_gap_peak[0].item() == pytest.approx(.004)
    assert e.startup_report["control_timing"]["runtime_substep_verification"]["pd_updates"] == 4
