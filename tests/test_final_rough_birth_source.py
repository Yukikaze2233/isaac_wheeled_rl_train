"""Execute frozen reset methods on CPU buffers, without claiming native IK."""
import ast
import __future__
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.final_rough_definition import (
    REFERENCE, load_final_rough_birth_definition, source_manifest,
)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, cp, cy = (value.mul(.5).cos() for value in (roll, pitch, yaw))
    sr, sp, sy = (value.mul(.5).sin() for value in (roll, pitch, yaw))
    return torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)


def quat_mul(a, b):
    w = a[:, :1] * b[:, :1] - (a[:, 1:] * b[:, 1:]).sum(-1, keepdim=True)
    xyz = a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] + torch.cross(a[:, 1:], b[:, 1:], dim=-1)
    return torch.cat((w, xyz), -1)


def quat_apply(q, value):
    cross = 2 * torch.cross(q[:, 1:], value, dim=-1)
    return value + q[:, :1] * cross + torch.cross(q[:, 1:], cross, dim=-1)


def euler_xyz_from_quat(q):
    w, x, y, z = q.unbind(-1)
    return (torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            torch.asin((2 * (w * y - z * x)).clamp(-1, 1)),
            torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def sample_uniform(low, high, shape, device):
    return low + torch.rand(shape, device=device) * (high - low)


def source_methods():
    source_manifest()
    tree = ast.parse((REFERENCE / "base_env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Wheelbipe25V3Env")
    names = {"_custom_reset_random", "_range_pair_as_float", "_apply_root_state_uniform_vel_b",
             "_get_reset_training_iteration", "_get_active_predefined_reset_air_modes",
             "_apply_axis_aligned_reset_heading", "_as_env_ids_tensor"}
    methods = [deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(torch=torch, Mapping=Mapping, sample_uniform=sample_uniform,
                 quat_from_euler_xyz=quat_from_euler_xyz, quat_mul=quat_mul,
                 quat_apply=quat_apply, euler_xyz_from_quat=euler_xyz_from_quat)
    events = ast.parse((REFERENCE / "reset_events.py").read_text())
    methods.append(deepcopy(next(node for node in events.body if isinstance(node, ast.FunctionDef)
                                and node.name == "reset_root_state_uniform_vel_b")))
    scope["math_utils"] = NS(sample_uniform=sample_uniform, quat_from_euler_xyz=quat_from_euler_xyz,
                             quat_mul=quat_mul, quat_apply=quat_apply)
    scope["SceneEntityCfg"] = lambda name: NS(name=name)
    exec(compile(ast.Module(body=methods, type_ignores=[]), "frozen_birth_methods", "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return {**{name: scope[name] for name in names},
            "reset_root_event": scope["reset_root_state_uniform_vel_b"]}


class Robot:
    def __init__(self, n, init, origins):
        root = torch.tensor(init["pos"] + init["rot"] + init["lin_vel"] + init["ang_vel"]).repeat(n, 1)
        state = root.clone()
        state[:, :3] += origins
        self.data = NS(default_root_state=root, root_state_w=state)
        self.device = "cpu"
        self._ALL_INDICES = torch.arange(n)
        self.joint_velocity = torch.zeros(n, 14)

    def write_root_pose_to_sim(self, pose, env_ids):
        self.data.root_state_w[env_ids, :7] = pose

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.data.root_state_w[env_ids, 7:] = velocity

    def write_root_state_to_sim(self, state, env_ids):
        self.data.root_state_w[env_ids] = state

    def write_joint_state_to_sim(self, q, dq, joint_ids, env_ids):
        self.write_joint_velocity_to_sim(dq, joint_ids, env_ids)

    def write_joint_velocity_to_sim(self, dq, joint_ids, env_ids):
        self.joint_velocity[env_ids[:, None], joint_ids] = dq


def source_owner(n=32768, *, air_allowed=True, ground_allowed=True):
    birth = load_final_rough_birth_definition()
    owner = type("FrozenBirthOwner", (), source_methods())()
    owner.cfg = NS(**birth)
    owner.device, owner.num_envs, owner.use_spring = "cpu", n, False
    owner.terrain = NS(env_origins=torch.zeros(n, 3))
    owner.terrain.env_origins[:, 2] = torch.arange(n) % 2
    owner.robot = Robot(n, birth["source_root_init_state"], owner.terrain.env_origins)
    owner.scene = {"robot": owner.robot}
    owner.finish_init_time = torch.zeros(n)
    owner.episode_length_buf = torch.zeros(n, dtype=torch.long)
    owner.step_dt = .02
    owner.predefined_reset_ground_zero_torque_until_time = torch.zeros(n)
    owner._legs_act_idx, owner._wheel_idx = torch.tensor([0, 1, 6, 7]), torch.tensor([12, 13])
    owner.reorder_reset_joint_idx = torch.arange(12)
    owner._get_training_iteration = lambda: 0
    owner._get_predefined_reset_air_disabled_mask = lambda ids: torch.full((len(ids),), not air_allowed)
    owner._get_predefined_reset_ground_disabled_mask = lambda ids: torch.full((len(ids),), not ground_allowed)
    owner._get_axis_aligned_reset_heading_mask = lambda ids: torch.ones(len(ids), dtype=torch.bool)
    owner.air_ids, owner.ground_ids = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    def air_limits(ids, limits, *, mode_idx):
        owner.air_ids, owner.air_limits = ids.clone(), deepcopy(limits)
        assert mode_idx == 0

    def ground_override(ids, ranges, duration):
        owner.ground_ids, owner.ground_commands = ids.clone(), (deepcopy(ranges), duration)

    def record_ik(length, angle):
        owner.sampled_length, owner.sampled_angle = length.clone(), angle.clone()
        # Only capture source requests; the source and local mechanisms differ.
        return torch.zeros(len(length), 2, 6)

    owner._set_predefined_reset_air_command_limits = air_limits
    owner._set_predefined_reset_ground_command_override = ground_override
    owner._inverse_kinematics = record_ik
    return owner


@pytest.fixture(autouse=True)
def isolated_torch_rng_and_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("air_allowed,ground_allowed", [(True, True), (False, True), (True, False), (False, False)])
def test_actual_source_mode_intervals_keep_disabled_probability_as_ordinary_birth(air_allowed, ground_allowed):
    owner = source_owner(air_allowed=air_allowed, ground_allowed=ground_allowed)
    ids = owner.robot._ALL_INDICES
    torch.manual_seed(1739)
    owner.reset_root_event(ids, **owner.cfg.root_reset_event)
    initial = owner.robot.data.root_state_w.clone()
    state = torch.get_rng_state()
    owner._custom_reset_random(ids)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(state)
        torch.empty(len(ids), 2).uniform_(*owner.cfg.leg_length_range)
        torch.empty(len(ids), 2).uniform_(*owner.cfg.leg_angle_range)
        draws = torch.empty(len(ids)).uniform_(0, 1)
    assert torch.equal(owner.air_ids, ids[(draws < .3) & air_allowed])
    assert torch.equal(owner.ground_ids, ids[(draws >= .3) & (draws < .6) & ground_allowed])
    assert len(owner.air_ids) / len(ids) == pytest.approx(.3 if air_allowed else 0., abs=.012)
    assert len(owner.ground_ids) / len(ids) == pytest.approx(.3 if ground_allowed else 0., abs=.012)
    dq = owner.robot.joint_velocity
    assert bool((dq[:, owner._legs_act_idx].abs().sum(-1) > 0).all())
    assert bool((dq[:, owner._wheel_idx].abs().sum(-1) > 0).all())
    assert float(dq[:, owner._legs_act_idx].abs().max()) <= torch.pi / 2
    assert float(dq[:, owner._wheel_idx].abs().max()) <= 50
    if ground_allowed:
        g = owner.ground_ids
        assert float((owner.sampled_angle[g] > 0).float().mean()) == pytest.approx(2 / 3, abs=.02)
        assert torch.equal(owner.finish_init_time[g], torch.full((len(g),), 1.5))
        assert torch.equal(owner.predefined_reset_ground_zero_torque_until_time[g], torch.full((len(g),), .2))
    if air_allowed:
        a = owner.air_ids
        root = owner.robot.data.root_state_w[a]
        relative_z = root[:, 2] - owner.terrain.env_origins[a, 2]
        assert bool(((relative_z >= .50) & (relative_z <= .80)).all())
        assert owner.air_limits["duration_s"] == 3.
        assert owner.air_limits["height"] == [.18, .43]
        # Linear velocity rotates by reset yaw only; world vertical stays zero.
        assert torch.equal(root[:, 9], torch.zeros(len(a)))
        yaw = euler_xyz_from_quat(root[:, 3:7])[2]
        vx = yaw.cos() * root[:, 7] + yaw.sin() * root[:, 8]
        vy = -yaw.sin() * root[:, 7] + yaw.cos() * root[:, 8]
        assert float(vx.abs().max()) <= 2.5 + 1e-6
        assert float(vy.abs().max()) <= .5 + 1e-6
    ordinary = torch.ones(len(ids), dtype=torch.bool)
    ordinary[owner.air_ids], ordinary[owner.ground_ids] = False, False
    relative_z = owner.robot.data.root_state_w[:, 2] - owner.terrain.env_origins[:, 2]
    assert torch.allclose(relative_z[ordinary], torch.full((int(ordinary.sum()),), .38))
    assert torch.equal(owner.robot.data.root_state_w[ordinary], initial[ordinary])


def test_actual_source_heading_snap_preserves_roll_pitch_and_existing_world_velocity():
    owner = source_owner(n=512)
    torch.manual_seed(18)
    owner.reset_root_event(owner.robot._ALL_INDICES, **owner.cfg.root_reset_event)
    owner._custom_reset_random(owner.robot._ALL_INDICES)
    before = owner.robot.data.root_state_w.clone()
    owner._apply_axis_aligned_reset_heading(owner.robot._ALL_INDICES)
    after = owner.robot.data.root_state_w
    old_rpy, new_rpy = euler_xyz_from_quat(before[:, 3:7]), euler_xyz_from_quat(after[:, 3:7])
    assert torch.allclose(new_rpy[0], old_rpy[0], atol=1e-6, rtol=0)
    assert torch.allclose(new_rpy[1], old_rpy[1], atol=1e-6, rtol=0)
    assert torch.equal(after[:, 7:], before[:, 7:])
    assert bool(new_rpy[2].isfinite().all())
    scaled = new_rpy[2] / (torch.pi / 2)
    assert torch.allclose(scaled, scaled.round(), atol=1e-6, rtol=0)


def test_actual_reset_calls_sample_then_snap_then_record_actual_heading():
    tree = ast.parse((REFERENCE / "base_env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Wheelbipe25V3Env")
    reset = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx")
    calls = [node.value.func.attr for node in reset.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)]
    ordered = ["_reset_idx", "_custom_reset_random", "_apply_axis_aligned_reset_heading",
               "_record_reset_heading_target", "_sync_heading_command_target_to_reset_heading"]
    assert [calls.index(name) for name in ordered] == sorted(calls.index(name) for name in ordered)
