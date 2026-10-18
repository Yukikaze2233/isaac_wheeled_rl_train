"""Nominal axis reset and actual rotated collision envelope on CPU fixtures.

The asymmetric shape and raised strip distinguish a 90-degree actual root pose
from an unchanged request yaw. No native scene, contacts or PPO are exercised.
"""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch
import warp as wp

from wheeled_tasks.chassis.final_rough_birth import apply_nominal_axis_heading
from wheeled_tasks.chassis.flat_episode_runtime import FlatResetState
from wheeled_tasks.chassis.nonsteady_initial_state import FlatBirthRequests
from wheeled_tasks.chassis.scut_rough_support import RoughBirthSupport


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "src/wheeled_tasks/chassis/env.py"


def pending(ids=(0,), dtype=torch.float64):
    ids = torch.tensor(ids, dtype=torch.long)
    n = len(ids)
    requests = FlatBirthRequests(ids.clone(), torch.zeros(n, dtype=torch.bool),
        torch.full((n, 2), -1, dtype=torch.int8), torch.full((n, 2), .22),
        torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n, 6),
        torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n, 6))
    pose = torch.zeros(n, 7, dtype=dtype)
    pose[:, 2], pose[:, 6] = .213, 1.
    return FlatResetState(ids, requests, pose, torch.zeros(n, 6, dtype=dtype),
        torch.arange(18, dtype=dtype).repeat(n, 1) / 100., torch.zeros(n, 18, dtype=dtype),
        torch.ones(n, dtype=torch.long), torch.full((n,), .213, dtype=dtype), torch.full((n,), .003, dtype=dtype))


class Permissions:
    reset_heading_candidates_deg = (0., 90., 180., -90.)

    def __init__(self, mask):
        self.mask = mask
        self.calls = []

    def axis_reset_mask(self, ids, *, use_env_origins):
        self.calls.append((ids.clone(), use_env_origins))
        return self.mask.clone()


def snapshot(state):
    return {name: ({key: value.clone() for key, value in vars(state.requests).items()} if name == "requests"
                  else getattr(state, name).clone()) for name in FlatResetState.__dataclass_fields__}


@pytest.mark.parametrize("explicit_nominal", [False, True])
def test_caller_rng_four_candidates_selected_order_and_all_other_state_preserved(explicit_nominal):
    state = pending(tuple(range(400, 600)))
    selected = torch.arange(200) % 2 == 0
    commands = Permissions(selected)
    before, global_rng = snapshot(state), torch.get_rng_state()
    rng, oracle = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
    indices = torch.randint(4, (int(selected.sum()),), generator=oracle)
    assert set(indices.tolist()) == {0, 1, 2, 3}
    result = apply_nominal_axis_heading(state, commands, rng,
        **({"nominal_mask": torch.ones(200, dtype=torch.bool)} if explicit_nominal else {}))
    assert torch.equal(result, selected) and torch.equal(rng.get_state(), oracle.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(commands.calls[0][0], state.env_ids) and commands.calls[0][1] is True
    yaw = torch.tensor([0., math.pi / 2., math.pi, -math.pi / 2.], dtype=torch.float64)[indices]
    assert torch.allclose(state.root_pose[selected, 5], torch.sin(yaw / 2.), rtol=0., atol=1e-15)
    assert torch.allclose(state.root_pose[selected, 6], torch.cos(yaw / 2.), rtol=0., atol=1e-15)
    assert torch.equal(state.root_pose[~selected], before["root_pose"][~selected])
    assert torch.equal(state.root_pose[:, :5], before["root_pose"][:, :5])
    for name in before:
        if name == "root_pose":
            continue
        if name == "requests":
            assert all(torch.equal(value, before[name][key]) for key, value in vars(state.requests).items())
        else:
            assert torch.equal(getattr(state, name), before[name])


def test_no_permission_draws_no_rng_and_changes_no_pose():
    state = pending((3, 1))
    commands, rng = Permissions(torch.zeros(2, dtype=torch.bool)), torch.Generator().manual_seed(7)
    before, saved = snapshot(state), rng.get_state()
    assert not apply_nominal_axis_heading(state, commands, rng).any()
    assert torch.equal(state.root_pose, before["root_pose"]) and torch.equal(saved, rng.get_state())


@pytest.mark.parametrize("mutate", [
    lambda s: s.requests.ground_birth.fill_(True),
    lambda s: s.requests.source_root_rpy_rad.__setitem__((0, 2), .1),
    lambda s: s.root_velocity.__setitem__((0, 0), .1),
    lambda s: s.joint_velocity.__setitem__((0, 0), .1),
    lambda s: s.root_pose.__setitem__((0, slice(3, 7)), torch.tensor([0., 0., 2**-.5, 2**-.5])),
])
def test_reject_nonstandard_or_nonfresh_birth_without_partial_change_or_rng_draw(mutate):
    state = pending()
    mutate(state)
    before = snapshot(state)
    rng = torch.Generator().manual_seed(11)
    saved = rng.get_state()
    with pytest.raises(ValueError, match="fresh explicit nominal"):
        apply_nominal_axis_heading(state, Permissions(torch.ones(1, dtype=torch.bool)), rng)
    assert torch.equal(state.root_pose, before["root_pose"]) and torch.equal(rng.get_state(), saved)


def test_explicit_nominal_subset_preserves_other_birth_poses_and_velocity_in_actual_id_order():
    state = pending((9, 2, 7, 4, 1))
    nominal = torch.tensor([True, False, True, False, True])
    permitted = torch.tensor([True, True, False, True, True])
    state.requests.ground_birth[~nominal] = True
    state.requests.source_root_rpy_rad[~nominal, 0] = .7
    state.root_pose[~nominal, 3], state.root_pose[~nominal, 6] = math.sin(.35), math.cos(.35)
    state.root_velocity[~nominal, 0] = .8
    state.joint_velocity[~nominal, 0] = 1.2
    before = snapshot(state)
    rng, oracle = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    choices = torch.randint(4, (2,), generator=oracle)
    result = apply_nominal_axis_heading(state, Permissions(permitted), rng, nominal_mask=nominal)
    selected = nominal & permitted
    assert torch.equal(result, selected)
    assert torch.equal(rng.get_state(), oracle.get_state())
    angles = torch.tensor([0., math.pi / 2., math.pi, -math.pi / 2.], dtype=state.root_pose.dtype)[choices]
    torch.testing.assert_close(state.root_pose[selected, 5], (angles / 2.).sin(), rtol=0, atol=1e-15)
    torch.testing.assert_close(state.root_pose[selected, 6], (angles / 2.).cos(), rtol=0, atol=1e-15)
    assert torch.equal(state.root_pose[~selected], before["root_pose"][~selected])
    assert torch.equal(state.root_pose[:, :5], before["root_pose"][:, :5])
    for name in before:
        if name == "requests":
            assert all(torch.equal(value, before[name][key]) for key, value in vars(state.requests).items())
        elif name != "root_pose":
            assert torch.equal(getattr(state, name), before[name])


@pytest.mark.parametrize("nominal_mask", [torch.tensor([False]), torch.tensor([True])])
def test_mixed_subset_does_not_silently_accept_an_incorrect_nominal_declaration(nominal_mask):
    state, rng = pending(), torch.Generator().manual_seed(1)
    state.root_velocity[0, 0] = .3
    before, saved = snapshot(state), rng.get_state()
    if nominal_mask.item():
        with pytest.raises(ValueError, match="fresh explicit nominal"):
            apply_nominal_axis_heading(state, Permissions(torch.ones(1, dtype=torch.bool)), rng,
                nominal_mask=nominal_mask)
    else:
        assert not apply_nominal_axis_heading(state, Permissions(torch.ones(1, dtype=torch.bool)), rng,
            nominal_mask=nominal_mask).any()
    assert torch.equal(state.root_pose, before["root_pose"]) and torch.equal(rng.get_state(), saved)


@pytest.mark.parametrize("mask", [[True], torch.tensor([1]), torch.tensor([[True]]), torch.tensor([])])
def test_nominal_subset_rejects_malformed_masks_before_pose_or_rng_changes(mask):
    state, rng = pending(), torch.Generator().manual_seed(1)
    before, saved = snapshot(state), rng.get_state()
    with pytest.raises(ValueError, match="ordered boolean nominal birth mask"):
        apply_nominal_axis_heading(state, Permissions(torch.ones(1, dtype=torch.bool)), rng,
            nominal_mask=mask)
    assert torch.equal(state.root_pose, before["root_pose"]) and torch.equal(rng.get_state(), saved)


@pytest.fixture
def asymmetric_support():
    wp.init()
    # The narrow raised strip is at positive world Y, outside the unrotated
    # shape's footprint. A quarter turn moves that same shape onto the strip.
    points = torch.tensor([[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.],
        [-.03, .28, .4], [.03, .28, .4], [.03, .34, .4], [-.03, .34, .4]])
    triangles = torch.tensor([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype=torch.int32)
    mesh = wp.Mesh(points=wp.from_torch(points, dtype=wp.vec3), indices=wp.from_torch(triangles))
    vertices = torch.tensor([[x, y, z] for x in (.27, .36) for y in (-.04, .06)
                             for z in (-.21, -.16)], dtype=torch.float64)
    placement = NS(device=torch.device("cpu"), dtype=torch.float64, clearance_m=.003,
        kinematics=NS(frames=lambda q: {"shape": torch.eye(4, dtype=torch.float64).expand(len(q), -1, -1)}),
        _shapes=[("shape", 0, "mesh", torch.eye(4, dtype=torch.float64), vertices)])
    terrain = NS(warp_mesh=mesh, bounds_m=[[-2., -2., 0.], [2., 2., .4]], mesh_sha256="cpu-asymmetric-fixture",
        points=points, triangles=triangles)
    return RoughBirthSupport(placement, terrain), vertices, points


def test_actual_quarter_turn_envelope_finds_strip_while_old_default_keeps_request_rpy(asymmetric_support, monkeypatch):
    import wheeled_tasks.chassis.scut_rough_support as support_module
    support, vertices, points = asymmetric_support
    actual_query, bounds = support_module.mesh_upper_bounds, []
    def record_query(mesh, query_bounds, low, high):
        bounds.append((low.clone(), high.clone()))
        return actual_query(mesh, query_bounds, low, high)
    monkeypatch.setattr(support_module, "mesh_upper_bounds", record_query)
    actual, legacy = pending(), pending()
    for state in (actual, legacy):
        state.root_pose[0, 5:7] = 2**-.5
    before = snapshot(actual)
    support.adjust(actual, actual_root_pose_orientation=True)
    support.adjust(legacy)
    # Independent matrix: swap/negate XY directly rather than reusing the
    # production quaternion formula or its AABB calculations.
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    rotated = vertices @ rotation.T
    torch.testing.assert_close(bounds[0][0][0, 0], rotated[:, :2].amin(0), rtol=0., atol=1e-15)
    torch.testing.assert_close(bounds[0][1][0, 0], rotated[:, :2].amax(0), rtol=0., atol=1e-15)
    torch.testing.assert_close(bounds[1][0][0, 0], vertices[:, :2].amin(0), rtol=0., atol=1e-15)
    torch.testing.assert_close(bounds[1][1][0, 0], vertices[:, :2].amax(0), rtol=0., atol=1e-15)
    assert rotated[:, 0].min() < points[4:8, 0].max() and rotated[:, 0].max() > points[4:8, 0].min()
    assert rotated[:, 1].min() < points[4:8, 1].max() and rotated[:, 1].max() > points[4:8, 1].min()
    expected = float(points[4:8, 2].max()) - float(rotated[:, 2].min()) + .003
    assert actual.root_pose[0, 2].item() == pytest.approx(expected, abs=2e-9)
    assert actual.root_pose[0, 2].item() - legacy.root_pose[0, 2].item() == pytest.approx(.4, abs=2e-8)
    assert legacy.root_pose[0, 2].item() == pytest.approx(.213, abs=2e-9)
    assert actual.minimum_shape_clearance_m[0] >= .003
    for name in ("joint_position", "joint_velocity", "root_velocity"):
        assert torch.equal(getattr(actual, name), before[name])
    assert torch.equal(actual.root_pose[:, [0, 1, 3, 4, 5, 6]], before["root_pose"][:, [0, 1, 3, 4, 5, 6]])
    assert torch.equal(actual.requests.source_root_rpy_rad, before["requests"]["source_root_rpy_rad"])


def env_method(name):
    cls = next(node for node in ast.parse(ENV.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    return deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))


@pytest.mark.parametrize("explicit_nominal", [False, True])
def test_actual_env_reset_snippets_snap_support_write_refresh_then_commit_actual_heading(asymmetric_support, monkeypatch, explicit_nominal):
    import wheeled_tasks.chassis.final_rough_birth as birth_module
    support, _, _ = asymmetric_support
    state = pending((2,))
    trace, committed = [], []
    commands = Permissions(torch.ones(1, dtype=torch.bool))
    real_apply = birth_module.apply_nominal_axis_heading
    def apply(*args, **kwargs):
        trace.append("axis")
        if explicit_nominal:
            assert torch.equal(kwargs["nominal_mask"], torch.tensor([True]))
        else:
            assert kwargs == {}
        result = real_apply(*args, **kwargs)
        assert state.root_pose[0, 5].item() == pytest.approx(2**-.5)
        return result
    monkeypatch.setattr(birth_module, "apply_nominal_axis_heading", apply)
    class ActualSupport:
        def adjust(self, actual, **kwargs):
            trace.append("support")
            assert kwargs == {"actual_root_pose_orientation": True}
            support.adjust(actual, **kwargs)
    def prepare(ids, generator, **kwargs):
        trace.append("prepare")
        assert torch.equal(ids, state.env_ids) and "nominal_q" in kwargs
        if explicit_nominal:
            assert torch.equal(kwargs["nominal_mask"], torch.tensor([True]))
        return state
    def commit(actual, heading, generator, iteration):
        trace.append("commit")
        assert actual is state and iteration == 0
        committed.append(heading.clone())
    flat = NS(commands=commands, tree_joint_names=[f"q{i}" for i in range(18)], prepare_reset=prepare,
        commit_reset=commit, publish=lambda *args: trace.append("publish"))
    poses = torch.zeros(3, 7, dtype=torch.float64)
    poses[:, 6] = 1.
    robot = NS(joint_names=flat.tree_joint_names, data=NS(root_link_pose_w=NS(torch=poses)))
    def write_pose(*, root_pose, env_ids):
        trace.append("native_pose")
        assert root_pose[0, 2] > .6
        poses[env_ids] = root_pose
    robot.write_root_link_pose_to_sim_index = write_pose
    robot.write_root_com_velocity_to_sim_index = lambda **kw: trace.append("native_velocity")
    robot.write_joint_position_to_sim_index = lambda **kw: trace.append("native_q")
    robot.write_joint_velocity_to_sim_index = lambda **kw: trace.append("native_dq")
    robot.reset = lambda ids: trace.append("native_reset")
    e = NS(flat_episode=flat, cfg={"formal_foundation_nominal_birth": True,
        "formal_terrain": {"type": "rough", "source": "v7_final_b8ff79f3"}},
        device="cpu", dt=.005, nominal=state.joint_position.expand(3, -1), robot=robot, generator=torch.Generator().manual_seed(1),
        origins=torch.zeros(3, 3), reset_positions=torch.zeros(3, 3), ground_height=lambda xy: torch.zeros(3),
        rough_birth_support=ActualSupport(), flat_tree_to_native=torch.arange(18), commands=torch.zeros(3, 3),
        mode=torch.zeros(3, dtype=torch.long), targets=torch.zeros(3, 6), jump_requested=torch.zeros(3, dtype=torch.bool))
    e._flat_command_iteration = lambda: 0
    if explicit_nominal:
        e._dynamic_birth_nominal_mask = torch.tensor([False, False, True])
    scope = {"torch": torch, "self": e, "ids": state.env_ids, "count": 1,
        "__name__": "wheeled_tasks.chassis.env", "__package__": "wheeled_tasks.chassis",
        "update_robot_data": lambda *args: trace.append("refresh")}
    heading = env_method("_flat_heading")
    exec(compile(ast.Module(body=[heading], type_ignores=[]), str(ENV), "exec"), scope)
    e._flat_heading = MethodType(scope["_flat_heading"], e)
    body = env_method("reset").body
    first = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == "flat")
    write = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
                 and ast.unparse(node.value.func) == "self.robot.write_root_link_pose_to_sim_index")
    nodes = body[first:write + 5]
    nodes.append(next(node for node in body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "update_robot_data"))
    commit_node = next(node for node in body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "flat is not None" and "flat.commit_reset" in ast.unparse(node))
    nodes.append(commit_node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(ENV), "exec"), scope)
    assert trace == ["prepare", "axis", "support", "native_pose", "native_velocity", "native_q", "native_dq", "native_reset", "refresh", "commit", "publish"]
    assert committed[0].item() == pytest.approx(math.pi / 2.)
    assert not state.requests.source_root_rpy_rad.any()
