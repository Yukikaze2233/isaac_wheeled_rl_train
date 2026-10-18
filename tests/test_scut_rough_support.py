"""CPU mesh/scan geometry; no Kit, PhysX stepping or GPU simulation."""
from types import SimpleNamespace as NS

import pytest
import torch
import warp as wp

from wheeled_tasks.chassis.scut_rough_support import (
    RoughBirthSupport, base_ground_scan, mesh_upper_bounds,
)
from support_nonstandard_start_integration import execute, method
from test_formal_shared_plane import env_for_guard, run_pre_reset_guards


def test_source_nine_ray_yaw_scan_finite_mean_and_origin_fallback():
    pose = torch.tensor([[10., 20., .4, 0., 0., 2**-.5, 2**-.5],
                         [30., 40., .8, 0., 0., 0., 1.]])
    seen = []

    def cast(starts, directions):
        seen.append((starts.clone(), directions.clone()))
        hits = starts.clone()
        hits[:, :, 2] = float("inf")
        hits[0, :2, 2] = torch.tensor([.2, .4])
        hits[0, 2, 2] = float("nan")
        return hits

    height, count = base_ground_scan(pose, torch.tensor([9., 1.25]), cast)
    torch.testing.assert_close(height, torch.tensor([.3, 1.25]))
    assert count.tolist() == [2, 0]
    starts, directions = seen[0]
    assert starts.shape == (2, 9, 3)
    torch.testing.assert_close(starts[:, :, 2], pose[:, 2:3].expand(-1, 9) + 5.)
    # First unrotated offset (-.01,-.01) becomes (+.01,-.01) at 90 degrees.
    torch.testing.assert_close(starts[0, 0, :2], torch.tensor([10.01, 19.99]))
    assert directions[:, :, :2].eq(0).all() and directions[:, :, 2].eq(-1).all()


@pytest.fixture
def terrain():
    wp.init()
    # A 2 mm-wide bar lies away from the root ray and between footprint corners.
    points = torch.tensor([[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.],
                           [.299, -.01, .08], [.301, -.01, .08], [.301, .01, .08], [.299, .01, .08]])
    faces = torch.tensor([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype=torch.int32)
    mesh = wp.Mesh(points=wp.from_torch(points, dtype=wp.vec3), indices=wp.from_torch(faces))
    return NS(warp_mesh=mesh, bounds_m=[[-2., -2., 0.], [2., 2., .08]],
              mesh_sha256="cpu_fixture", points=points, faces=faces)


def test_actual_mesh_bvh_finds_bar_beyond_root_sample(terrain):
    low = torch.tensor([[.2, -.1], [-.1, -.1]], dtype=torch.float64)
    high = torch.tensor([[.4, .1], [.1, .1]], dtype=torch.float64)
    height = mesh_upper_bounds(terrain.warp_mesh, terrain.bounds_m, low, high)
    torch.testing.assert_close(height, torch.tensor([.08, 0.], dtype=torch.float64), atol=2e-9, rtol=0)


@pytest.mark.parametrize("shape", ["cylinder", "mesh"])
def test_birth_raises_whole_shape_envelope_without_changing_q_dq_or_orientation(terrain, shape):
    transform = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    transform[:, :3, 3] = torch.tensor([.3, 0., -.2])
    geometry = ((.1, .04) if shape == "cylinder" else torch.tensor(
        [[x, y, z] for x in (-.1, .1) for y in (-.1, .1) for z in (-.02, .02)], dtype=torch.float64))
    placement = NS(device=torch.device("cpu"), dtype=torch.float64, clearance_m=.003,
        kinematics=NS(frames=lambda q: {"wheel": transform.expand(len(q), -1, -1)}),
        _shapes=[("wheel", 0, shape, torch.eye(4, dtype=torch.float64), geometry)])
    pending = NS(joint_position=torch.arange(18, dtype=torch.float64).reshape(1, -1),
        joint_velocity=torch.ones(1, 18, dtype=torch.float64),
        root_pose=torch.tensor([[0., 0., .223, 0., 0., 0., 1.]], dtype=torch.float64),
        root_velocity=torch.zeros(1, 6, dtype=torch.float64),
        placement_raise_m=torch.tensor([.223], dtype=torch.float64),
        minimum_shape_clearance_m=torch.tensor([.003], dtype=torch.float64),
        requests=NS(source_root_rpy_rad=torch.zeros(1, 3)))
    before = {name: getattr(pending, name).clone() for name in
              ("joint_position", "joint_velocity", "root_pose", "root_velocity")}
    RoughBirthSupport(placement, terrain).adjust(pending)
    assert pending.root_pose[0, 2] == pytest.approx(.303, abs=2e-9)
    assert pending.minimum_shape_clearance_m[0] >= .003
    for name in ("joint_position", "joint_velocity", "root_velocity"):
        assert torch.equal(getattr(pending, name), before[name])
    assert torch.equal(pending.root_pose[:, [0, 1, 3, 4, 5, 6]], before["root_pose"][:, [0, 1, 3, 4, 5, 6]])
    assert pending.placement_raise_m[0] == pytest.approx(.303, abs=2e-9)


def test_critic_height_keeps_rough_world_z_and_flat_legacy_origin_datum():
    body = method("_critic78_frame").body
    start = next(i for i, node in enumerate(body)
                 if getattr(node, "targets", []) and getattr(node.targets[0], "id", "") == "root_height")
    pose = torch.tensor([[0., 0., 1.5, 0., 0., 0., 1.]])
    env = NS(robot=NS(data=NS(root_link_pose_w=NS(torch=pose))), origins=torch.tensor([[0., 0., 1.]]), cfg={})
    assert execute(body[start:start + 2], env)["root_height"].item() == .5
    env.cfg["critic_height_reference"] = "root_link_world_z"
    assert execute(body[start:start + 2], env)["root_height"].item() == 1.5


def test_actual_finish_consumer_passes_world_scan_to_reward_once():
    env = env_for_guard()
    env.cfg["reward_profile"] = "scut0730_rough_v1"
    env.origins[:, 2] = 1.4
    env.robot.data.root_link_pose_w.torch[:, 2] = 1.8

    def raycast(starts, directions, mesh):
        hits = starts.clone()
        hits[:, :, 2] = 1.5
        return (hits,)

    env.shared_terrain = NS(raycast=raycast, warp_mesh=object(), boundary=lambda pose: torch.zeros(2, dtype=torch.bool))
    env.scut_reward.reward_height = lambda pose, origins, ground: pose[:, 2] - ground
    scope = run_pre_reset_guards(env)
    torch.testing.assert_close(scope["height"], torch.full((2,), .3))
    assert scope["reward_ground_height_world"].eq(1.5).all()
    assert scope["rough_scan_hit_count"].eq(9).all()
    captured = {}
    env.scut_reward.pay = lambda **kwargs: captured.update(kwargs)
    env.commands = torch.zeros(2, 3)
    env.robot.data.body_link_pose_w = NS(torch=torch.zeros(2, 2, 7))
    node = next(node for node in method("_finish_scut_flat_step").body
                if getattr(node, "targets", []) and getattr(node.targets[0], "id", "") == "result")
    execute([node], env, **{key: value for key, value in scope.items() if key not in ("self", "__builtins__", "__package__", "torch")})
    assert captured["reward_ground_height_world"].eq(1.5).all()
    assert captured["root_pose_world"][:, 2].eq(1.8).all()
    assert captured["origins"][:, 2].eq(1.4).all()
