"""Execute production feedback consumers with native CPU math and fake solver data.

Only the policy/control portion of step() is extracted; reward, scene creation
and PPO are outside these tests. Extraction preserves the production statements
and their ordering, and the fractional transport uses the native history buffer.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import zlib

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from wheeled_tasks.chassis.actuator_response import ActuatorResponse, JOINT_NAMES
from wheeled_tasks.chassis.command_transport import UsbCommandTransport
from wheeled_tasks.chassis.feedback_transport import FractionalFeedbackTransport
from wheeled_tasks.chassis.pc_communication import (
    command_profiles, command_transport_settings, feedback_transport_settings,
)
from wheeled_tasks.chassis.preflight import SubstepControlAudit
from wheeled_tasks.chassis.root_kinematics import root_link_linear_velocity, update_robot_data
from wheeled_tasks.chassis.scut_observation import build_sensor18
from wheeled_tasks.chassis.step_sensors import StepTaskSensors
from wheeled_tasks.chassis.v5_control import V5Control
from wheeled_tasks.v40 import core


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/wheeled_tasks/chassis/env.py"
EVAL_SOURCE = ROOT / "src/wheeled_tasks/chassis/eval_env.py"


def feedback_settings(delay):
    return dict(version="complete_one_way_feedback_v1", enabled=True, units="seconds",
        scope="engineering_prior_complete_sensor_to_host_path_not_measured_rtt",
        delay_range_s=[delay, delay], enabled_fraction=1., jitter_s=0., jitter_correlation=.9,
        ordered_delivery="monotonic_sensor_timestamp", reconstruction="linear_physics_history")


def production_class(*, evaluation=False, base=None):
    source = EVAL_SOURCE if evaluation else SOURCE
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
        and n.name == ("FixedCaseEnv" if evaluation else "ChassisEnv"))
    selected = {"reset", "reset_suite"} if evaluation else {
        "_pc_feedback_transport", "state", "_transport_sensors", "get_observations", "_update_step_jump", "step"}
    nodes = [deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    if not evaluation:
        step = next(n for n in nodes if n.name == "step")
        # Keep decoding, the complete physics loop and the control-audit receipt.
        # End before reward computation, which legitimately reads physical truth.
        end = next(i for i, n in enumerate(step.body) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "self.state")
        step.body = step.body[:end]
    definition = ast.ClassDef(name="ExtractedEnv", bases=[ast.Name(id="Base", ctx=ast.Load())]
        if evaluation else [], keywords=[], body=nodes, decorator_list=[])
    scope = dict(torch=torch, F=F, TensorDict=TensorDict, zlib=zlib,
        update_robot_data=update_robot_data, root_link_linear_velocity=root_link_linear_velocity,
        __package__="wheeled_tasks.chassis", Base=base)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), str(source), "exec"), scope)
    return scope["ExtractedEnv"]


class RecordedControl(V5Control):
    def __init__(self):
        read = lambda path: json.loads((ROOT / path).read_text())
        bundle = "model/纯底盘_v6_232mm_stop105/urdf/"
        super().__init__(read(bundle + "manifest.json"), read(bundle + "model_spec.json"),
            read(bundle + "fit_10mpa.json"), read("contracts/own_v40_v2.json"),
            read("contracts/v6_new_asset_training_v1.json")["runtime_base"]["v5_control"], "cpu")
        self.decoded, self.pd, self.springs = [], [], []

    def decode(self, actions, q):
        self.decoded.append(q.clone())
        return super().decode(actions, q)

    def motor_efforts(self, q, dq, legs, wheels):
        self.pd.append((q.clone(), dq.clone(), legs.clone(), wheels.clone()))
        return super().motor_efforts(q, dq, legs, wheels)

    def spring_efforts(self, position):
        self.springs.append(position.clone())
        return super().spring_efforts(position)


class RecordedRecovery:
    has_tasks = True
    reference_stride = 1

    def __init__(self):
        self.projections, self.feedback, self.references, self.torques = [], [], [], []
        self.policies = self.histories = 0

    def begin_policy(self):
        self.policies += 1

    def project_feedback(self, q):
        self.projections.append(q.clone())
        return q

    def observe_feedback(self, q):
        self.feedback.append(q.clone())
        return q

    def advance_reference(self, q, dq, gravity, gyro):
        self.references.append(tuple(value.clone() for value in (q, dq, gravity, gyro)))

    def torque(self, q, dq, torque):
        self.torques.append((q.clone(), dq.clone()))
        return torque

    def effective_action_history(self):
        self.histories += 1


def make_env(*, delay=.0123, recovery=False, reference=False):
    env = production_class()()
    count = env.num_envs = 2
    env.device, env.dt, env.policy_dt, env.decimation = "cpu", .005, .02, 4
    env.cfg = dict(actor_frame_dim=35, critic_dim=81,
        task_semantics={"jump_request_seconds": 1.}, manual_context35=reference,
        command_reference={"preload_seconds": .1, "preload_depth_m": .01, "release_offset_m": .02})
    env.scut35 = env.is_v5 = True
    env.v5 = RecordedControl()
    env.ids, env.spring_ids, env.wheel_ids = list(range(6)), [6, 7], [0, 1]
    q = torch.zeros(count, 18)
    q[:, :6] = env.v5.nominal
    q[:, 6:8] = .01
    pose = torch.tensor([[-.24, 0., .32, 0., 0., 0., 1.]]).repeat(count, 1)
    data = NS(joint_pos=NS(torch=q), joint_vel=NS(torch=torch.zeros_like(q)),
        root_link_pose_w=NS(torch=pose), projected_gravity_b=NS(torch=torch.tensor([[0., 0., -1.]]).repeat(count, 1)),
        root_com_ang_vel_b=NS(torch=torch.tensor([[.1, .2, .3]]).repeat(count, 1)),
        root_com_lin_vel_b=NS(torch=torch.tensor([[1., 2., 3.]]).repeat(count, 1)),
        root_link_lin_vel_w=NS(torch=torch.zeros(count, 3)),
        body_link_pose_w=NS(torch=pose[:, None].repeat(1, 2, 1)))
    env.robot = NS(data=data)
    env.events, env.physical_before, env.applied_efforts, env.observed_packets = [], [], [], []
    env.native_updates = 0

    def update(dt):
        assert dt == .005
        env.native_updates += 1
        env.events.append("native_update")
        data.joint_pos.torch[:, :6] += .02
        data.joint_pos.torch[:, 6:8] += .001
        data.joint_vel.torch[:, :6] += .1
        data.root_com_ang_vel_b.torch += .01
        data.root_com_lin_vel_b.torch += 1.

    def effort_target(*, target):
        env.physical_before.append(data.joint_pos.torch.clone())
        env.applied_efforts.append(target.clone())

    env.robot.update = update
    env.robot.set_joint_effort_target_index = effort_target
    env.robot.write_data_to_sim = lambda: env.events.append("write_effort")
    env.sim = NS(step=lambda **kwargs: env.events.append("physics_step"))
    env.origins = torch.zeros(count, 3)
    env.ground_height = lambda xy: torch.zeros(len(xy))
    env.surface_data = torch.tensor([[[-100., 100., 0., 0., 0.]]]).repeat(count, 1, 1)
    env.surface_valid = torch.ones(count, 1, dtype=torch.bool)
    env.surface_mu = torch.full((count, 1), .7)
    env.nominal = torch.zeros(count, 18)
    env.commands = torch.tensor([[.4, .2, .23]]).repeat(count, 1)
    env.actions = torch.zeros(count, 6)
    env.previous_actions = torch.zeros_like(env.actions)
    env.before_previous_actions = torch.zeros_like(env.actions)
    env.targets = torch.zeros(count, 4)
    env.mode = torch.ones(count, dtype=torch.long)
    env.jump_requested = torch.zeros(count, dtype=torch.bool)
    env.episode_length_buf = torch.zeros(count, dtype=torch.long)
    env.contact_force = torch.zeros(count, 2, 3)
    env.contact_peak = torch.zeros(count)
    env.wheel_force_peak_n = None
    env.motor_strength = torch.ones(count, 6)
    env.spring_strength = torch.ones(count, 1)
    env.control_audit = SubstepControlAudit(env.dt, env.policy_dt)
    env.startup_report = {"control_timing": {}}
    env.tick = 0
    env.history = NS(update=lambda frame, tick: frame)
    env.recovery = RecordedRecovery() if recovery else None
    env.references = NS(elapsed=torch.zeros(count), terrain_mode=torch.ones(count),
        phase=torch.zeros(count, dtype=torch.long), STEP_LOWER=4, landing_delta=torch.full((count,), .1)) if reference else None
    env.has_jump_tasks = False
    env.skills = env.signal_delay = env.usb_transport = env.command_transport = None
    env.nonstandard_start = env.jump_assist = env.perturbations = env.torque_monitor = None
    env.actuator_response = NS(enabled=False)
    env.contact_views = []
    env.step_sensors = env.step_jump = None
    env.feedback_transport = FractionalFeedbackTransport(count, "cpu", env.dt, feedback_settings(delay), 617)
    env.feedback_transport.reset(torch.arange(count), env._transport_sensors())
    original_observe = env.feedback_transport.observe

    def observe(packet):
        env.events.append("feedback_observe")
        env.observed_packets.append(packet.clone())
        original_observe(packet)

    env.feedback_transport.observe = observe
    return env


@pytest.mark.parametrize("delay", [.00004, .0123])
@pytest.mark.parametrize("recovery", [False, True])
def test_production_four_physics_pc_pd_and_recovery_use_one_cached_fractional_stream(delay, recovery):
    env = make_env(delay=delay, recovery=recovery)
    original = env._transport_sensors().clone()
    env.step(torch.full((2, 6), .1))
    assert len(env.v5.decoded) == 1 and torch.equal(env.v5.decoded[0], original[:, :6])
    assert len(env.v5.pd) == len(env.observed_packets) == env.native_updates == 4
    assert env.feedback_transport.physics_steps == 4
    assert env.feedback_transport.history._num_pushes.eq(5).all()
    for substep, (q, dq, _, _) in enumerate(env.v5.pd):
        now = substep * .005
        source_time = max(0., now - delay)
        torch.testing.assert_close(q, original[:, :6] + source_time / .005 * .02, rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(dq, torch.full((2, 6), source_time / .005 * .1), rtol=1e-6, atol=1e-7)
        # The native spring force deliberately uses the solver's current state.
        assert torch.equal(env.v5.springs[substep], env.physical_before[substep][:, env.spring_ids])
    assert not torch.equal(env.v5.pd[-1][0], env.physical_before[-1][:, env.ids])
    assert env.events == ["write_effort", "physics_step", "native_update", "feedback_observe"] * 4
    receipt = env.startup_report["control_timing"]["runtime_substep_verification"]
    assert (receipt["pd_updates"], receipt["feedback_refreshes"], receipt["target_decodes"]) == (4, 4, 1)
    if recovery:
        assert len(env.recovery.feedback) == len(env.recovery.references) == len(env.recovery.torques) == 4
        assert env.recovery.policies == env.recovery.histories == 1
        for index, (q, dq, gravity, gyro) in enumerate(env.recovery.references):
            torch.testing.assert_close(q, env.v5.pd[index][0], rtol=0, atol=0)
            torch.testing.assert_close(dq, env.v5.pd[index][1], rtol=0, atol=0)
            source_time = max(0., index * .005 - delay)
            torch.testing.assert_close(gyro, original[:, 12:15] + source_time / .005 * .01)
            torch.testing.assert_close(gravity, original[:, 15:18])
            assert torch.equal(env.recovery.torques[index][0], q)


@pytest.mark.parametrize("reference", [False, True])
def test_actual_actor_uses_delayed_feedback_while_commands_and_privileged_critic_are_fresh(reference):
    env = make_env(reference=reference)
    env.step(torch.full((2, 6), .1))
    fresh = env._transport_sensors().clone()
    delayed = env.feedback_transport.feedback.clone()
    observations = env.get_observations()
    expected_actor_sensors = build_sensor18(delayed[:, 12:15], delayed[:, 15:18],
        delayed[:, :6], delayed[:, 6:12], env.v5.nominal)
    expected_clean_sensors = build_sensor18(fresh[:, 12:15], fresh[:, 15:18],
        fresh[:, :6], fresh[:, 6:12], env.v5.nominal)
    torch.testing.assert_close(observations["policy"][:, 4:22], expected_actor_sensors, rtol=0, atol=0)
    torch.testing.assert_close(observations["critic"][:, 4:22], expected_clean_sensors, rtol=0, atol=0)
    assert not torch.equal(expected_actor_sensors, expected_clean_sensors)
    assert torch.equal(observations["critic"][:, 35:38], env.robot.data.root_com_lin_vel_b.torch)
    assert torch.equal(observations["critic"][:, 41:59], env.robot.data.joint_pos.torch)
    assert torch.equal(observations["critic"][:, 59:77], env.robot.data.joint_vel.torch * .1)
    assert observations["critic"].shape[-1] == 81
    enabled_sensor_time = env.feedback_transport.sensor_timestamp_s.clone()
    random_state = env.feedback_transport.generator.get_state().clone()
    env.commands[:, 0] = -.7
    for _ in range(10):
        reread = env.get_observations()
        assert reread["policy"][:, 0].eq(-.7).all()
        assert torch.equal(reread["policy"][:, 4:22], expected_actor_sensors)
    assert env.feedback_transport.physics_steps == 4
    assert torch.equal(env.feedback_transport.sensor_timestamp_s, enabled_sensor_time)
    assert torch.equal(env.feedback_transport.generator.get_state(), random_state)


def test_physical_wheel_capacity_after_transport_uses_current_shaft_speed(monkeypatch):
    env = make_env()
    calls = []
    native_limit = core.motor_torque_limit

    def capacity(speed, cfg):
        calls.append(speed.clone())
        return native_limit(speed, cfg)

    monkeypatch.setattr(core, "motor_torque_limit", capacity)
    env.command_transport = NS(apply_torque=lambda command: command)
    env.step(torch.full((2, 6), .1))
    assert len(calls) == 4
    for index, speed in enumerate(calls):
        torch.testing.assert_close(speed, torch.full((2, 2), index * .1), atol=1e-7, rtol=0)
        if index:
            assert not torch.equal(speed, env.v5.pd[index][1][:, list(env.v5.WHEELS)])


def test_native_retained_motor_response_cannot_reintroduce_torque_above_current_drive_capacity():
    env = make_env()
    # The feedback still reports zero speed while physical shafts have left the
    # identified curve domain. A prior interval retained 1 N*m of motor state.
    env.robot.data.joint_vel.torch[:, [2, 5]] = 1000.
    env.command_transport = NS(apply_torque=lambda torque: torque)
    env.actuator_response = ActuatorResponse(2, "cpu", .005, dict(
        enabled=True, schema="chassis-actuator-response", version=1,
        joint_names=list(JOINT_NAMES), gain=1., armature=[0.] * 6,
        frictionloss=[0.] * 6, damping=[0.] * 6, time_constant_s=[.02] * 6))
    env.actuator_response.state[:, [2, 5]] = 1.
    capacity = core.motor_torque_limit(torch.full((2, 2), 1000.), env.v5.wheel_prior)
    assert capacity.eq(0).all()
    env.step(torch.zeros(2, 6))
    assert env.v5.pd[0][1][:, [2, 5]].eq(0).all()
    # Before the fix, the native response produced .7788008, .6065307 and
    # .4723666 N*m for the first three updates despite zero physical capacity.
    for effort in env.applied_efforts:
        assert (effort[:, [2, 5]].abs() <= capacity).all()
    assert env.actuator_response.state[:, [2, 5]].eq(0).all()


@pytest.mark.parametrize("recovery", [False, True])
def test_real_step_height_observer_consumes_delayed_encoders_and_imu_without_advancing_stream(recovery):
    env = make_env(recovery=recovery)
    lookup = ROOT / "contracts/v6_height_lookup_v1.json"
    settings = dict(height_lookup_file=str(lookup.relative_to(ROOT)),
        height_lookup_sha256=hashlib.sha256(lookup.read_bytes()).hexdigest(),
        range_sample_period_s=.02, sensor_origin_b_m=[.1, -.2], sensor_pitch_rad=.55,
        ground_echo_margin_m=.015, range_max_m=4., range_noise_std_m=0., range_dropout_probability=0.)
    env.step_sensors = StepTaskSensors(2, "cpu", .02, settings, ROOT, 19, evaluation=True)
    env.step_sensors.reset(torch.arange(2), env.robot.data.root_link_lin_vel_w.torch)
    captured = []
    native_update = env.step_sensors.update

    def observer(*args):
        captured.append(tuple(value.clone() for value in args[1:5]))
        return native_update(*args)

    env.step_sensors.update = observer
    env.step_jump = NS(control=lambda *args, **kwargs: torch.zeros(2, dtype=torch.bool))
    cached = env.feedback_transport.feedback.clone()
    env.robot.data.joint_pos.torch[:, 1] = 20.
    env.robot.data.joint_vel.torch[:, :6] = 50.
    env.robot.data.root_com_ang_vel_b.torch[:] = 10.
    env.robot.data.projected_gravity_b.torch[:, 2] = 1.
    env._update_step_jump()
    for actual, expected in zip(captured[0], (cached[:, :6], cached[:, 6:12], cached[:, 12:15], cached[:, 15:18])):
        assert torch.equal(actual, expected)
    assert env.step_sensors.height_valid.all() and env.step_sensors.support.all()
    _, fresh_valid = env.step_sensors.model.wheel_positions(env.robot.data.joint_pos.torch[:, :6])
    assert not fresh_valid[:, 0].any()
    assert env.feedback_transport.physics_steps == 0
    assert torch.equal(env.feedback_transport.feedback, cached)
    if recovery:
        assert len(env.recovery.projections) == 1 and not env.recovery.feedback


@pytest.mark.parametrize("diagnostic", [False, True])
def test_fixed_case_final_pose_is_primed_after_native_refresh_with_paired_profile_activation(diagnostic):
    class Base:
        def reset(self, ids):
            self.events.append("base_reset")
            self.feedback_transport.reset(ids, self._transport_sensors())
            self.command_transport.reset(ids)

    env = production_class(evaluation=True, base=Base)()
    env.device, env.dt, env.num_envs, env.eval_seed = "cpu", .005, 2, 93
    env.eval_generator = torch.Generator().manual_seed(env.eval_seed)
    env.generator = torch.Generator().manual_seed(env.eval_seed)
    env.origins = torch.zeros(2, 3)
    env.fixed_reset_samples = None
    env.scene_groups, env.kinds = ["a", "b"], ["flat", "flat"]
    cases = [dict(name="a"), dict(name="b")]
    if not diagnostic:
        cases[0]["transport_enabled"] = False
        cases[1].update(transport_enabled=True, communication_profile={"delay_ms": .08, "can_phase_ms": .5})
    env.cases = {case["name"]: case for case in cases}
    env.cfg = dict(physics_dt=.005, policy_dt=.02, evaluation_exact_cases=True,
        feedback_transport=feedback_transport_settings(), command_transport=command_transport_settings(),
        signal_delay={"enabled": False, "version": "physical_signal_delay_v1"},
        usb_transport={"enabled": False}, signal_perturbations={"max_delay_steps": 0},
        evaluation={"cases": cases})
    if diagnostic:
        env.cfg.update(evaluation_transport_mode="training_domain_prior", run_scope="interactive_unscored")
    env.nominal = torch.zeros(2, 18)
    env.ground_height = lambda xy: torch.zeros(len(xy))
    env.events = []
    data = NS(root_link_pose_w=NS(torch=torch.tensor([[0., 0., .32, 0., 0., 0., 1.]]).repeat(2, 1)),
        joint_pos=NS(torch=torch.zeros(2, 6)), joint_vel=NS(torch=torch.zeros(2, 6)),
        root_com_ang_vel_b=NS(torch=torch.zeros(2, 3)),
        projected_gravity_b=NS(torch=torch.tensor([[0., 0., -1.]]).repeat(2, 1)))
    env.robot = NS(data=data)
    env._transport_sensors = lambda: torch.cat((data.joint_pos.torch, data.joint_vel.torch,
        data.root_com_ang_vel_b.torch, data.projected_gravity_b.torch), -1)

    def write_pose(*, root_pose, env_ids):
        env.events.append("final_pose_write")
        data.root_link_pose_w.torch[env_ids] = root_pose

    def update(dt):
        env.events.append("native_refresh")
        data.root_com_ang_vel_b.torch.copy_(data.root_link_pose_w.torch[:, 3:6] * 10)
        gravity = data.root_link_pose_w.torch[:, 3:6].clone()
        gravity[:, 2] = -1.
        data.projected_gravity_b.torch.copy_(gravity / gravity.norm(dim=-1, keepdim=True))

    env.robot.write_root_link_pose_to_sim_index = write_pose
    env.robot.write_root_com_velocity_to_sim_index = lambda **kwargs: env.events.append("velocity_write")
    env.robot.update = update
    env.skills = env.step_jump = env.step_sensors = env.recovery = env.usb_transport = None
    env.full_tasks = env.perturbations = None
    env.update_targets = lambda: None
    env.update_command_reference = lambda ids: None
    env.get_observations = lambda: {"policy": env.feedback_transport.feedback.clone()}
    shared_draw = torch.tensor([.1, .9])
    env.feedback_transport = FractionalFeedbackTransport(2, "cpu", .005,
        env.cfg["feedback_transport"], 19, domain_draw=shared_draw)
    env.command_transport = UsbCommandTransport(2, "cpu", .005, env.cfg["command_transport"],
        19, profiles=command_profiles(env.cfg, env.scene_groups), domain_draw=shared_draw)
    prime = env.feedback_transport.prime

    def record_prime(ids, packet):
        env.events.append("feedback_prime")
        prime(ids, packet)

    env.feedback_transport.prime = record_prime
    env.reset(torch.arange(2))
    assert env.events == ["base_reset", "feedback_prime", "final_pose_write", "velocity_write",
        "native_refresh", "feedback_prime"]
    assert torch.equal(env.feedback_transport.feedback, env._transport_sensors())
    expected_enabled = [True, False] if diagnostic else [False, True]
    assert env.feedback_transport.enabled.tolist() == expected_enabled
    assert torch.equal(env.feedback_transport.enabled, env.command_transport.enabled)
    assert env.feedback_transport.physics_steps == 0
    first = env.reset_suite()["policy"]
    second = env.reset_suite()["policy"]
    assert torch.equal(first, second)
    assert torch.equal(env.feedback_transport.feedback, env._transport_sensors())
    assert env.feedback_transport.enabled.tolist() == expected_enabled
    assert torch.equal(env.feedback_transport.enabled, env.command_transport.enabled)
