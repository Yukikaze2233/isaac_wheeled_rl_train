"""CPU geometry/reduction checks; no native raycast or sensor qualification."""
import ast
from copy import deepcopy
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.wheel_ground_scan import wheel_ground_scan


@lru_cache
def sdk_oracle():
    """Execute only actual SDK Torch pattern/math AST, leaving native imports out."""
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        pytest.skip("Installed IsaacLab source is needed for the independent SDK witness")
    sdk = Path(spec.origin).parent
    scope = {"torch": torch}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    requested = {
        sdk / "sensors/ray_caster/patterns/patterns.py": {"grid_pattern"},
        sdk / "utils/math.py": {"normalize", "yaw_quat", "quat_apply", "quat_apply_yaw", "quat_from_euler_xyz"},
    }
    for path, names in requested.items():
        tree = ast.parse(path.read_text())
        methods = [deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in methods} == names
        for method in methods:
            method.decorator_list = []
        nodes.extend(methods)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "actual_sdk_torch_grid_and_quaternion_oracle", "exec"), scope)
    return scope


def poses(n=1, dtype=torch.float32):
    value = torch.zeros(n, 2, 7, dtype=dtype)
    value[..., 6] = 1.
    value[..., 2] = .3
    value[:, 1, 1] = .4
    return value


def capture(callback):
    calls = []

    def run(starts, directions):
        calls.append((starts.clone(), directions.clone()))
        return callback(starts, directions)

    return run, calls


def flat_hits(starts, directions, height=0.):
    result = starts.clone()
    result[..., 2] = height
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_four_point_grid_matches_actual_sdk_order_and_world_down_rays(dtype):
    value = poses(2, dtype)
    value[:, 0, :3] = torch.tensor([[1., 2., .3], [-2., 3., .8]], dtype=dtype)
    raycast, calls = capture(flat_hits)
    ground, count, height = wheel_ground_scan(value, torch.tensor([.7, -.2], dtype=dtype), raycast)
    assert len(calls) == 1
    starts, directions = calls[0]
    source_grid, source_directions = sdk_oracle()["grid_pattern"](
        NS(size=(.015, .015), resolution=.01, ordering="xy", direction=(0., 0., -1.)), "cpu")
    assert source_grid.shape == (4, 3)
    torch.testing.assert_close(source_grid[:, :2], torch.tensor(
        [[-.0075, -.0075], [.0025, -.0075], [-.0075, .0025], [.0025, .0025]]), rtol=0., atol=1e-9)
    expected = value[..., :3, None].transpose(-1, -2).expand(-1, -1, 4, -1).clone()
    expected += source_grid.to(dtype)[None, None]
    expected[..., 2] += 5.
    torch.testing.assert_close(starts, expected.reshape(2, 8, 3), rtol=0., atol=1e-9 if dtype == torch.float64 else 3e-7)
    assert torch.equal(directions, source_directions.to(dtype).repeat(2, 2, 1))
    assert ground.shape == count.shape == height.shape == (2, 2)
    assert ground.dtype == height.dtype == dtype and count.dtype == torch.long
    assert torch.equal(count, torch.full((2, 2), 4, dtype=torch.long))
    assert not ground.any()
    assert torch.equal(height, value[..., 2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_attachment_wheel_quaternion_supplies_yaw_without_tilting_rays(dtype):
    value = poses(3, dtype)
    roll = torch.tensor([.7, -.6, 0., .2, -.4, .1], dtype=dtype)
    pitch = torch.tensor([.4, -.2, 0., 1.1, -.6, .3], dtype=dtype)
    yaw = torch.tensor([.3, -1.2, math.pi / 2, -math.pi / 2, 2.4, .8], dtype=dtype)
    value[..., 3:] = sdk_oracle()["quat_from_euler_xyz"](roll, pitch, yaw).reshape(3, 2, 4)
    raycast, calls = capture(flat_hits)
    wheel_ground_scan(value, torch.zeros(3, dtype=dtype), raycast)
    starts, directions = calls[0]
    local_grid, _ = sdk_oracle()["grid_pattern"](
        NS(size=(.015, .015), resolution=.01, ordering="xy", direction=(0., 0., -1.)), "cpu")
    local_grid = local_grid.to(dtype)
    local_grid[:, 2] += 5.
    attachment = value[..., 3:].reshape(6, 4).repeat_interleave(4, dim=0)
    expected = sdk_oracle()["quat_apply_yaw"](attachment, local_grid.repeat(6, 1)).reshape(3, 2, 4, 3)
    expected += value[..., None, :3]
    torch.testing.assert_close(starts, expected.reshape(3, 8, 3), rtol=0., atol=3e-7 if dtype == torch.float32 else 1e-9)
    assert torch.equal(starts[..., 2], (value[..., 2] + 5.).repeat_interleave(4, dim=1))
    assert torch.equal(directions, torch.tensor([0., 0., -1.], dtype=dtype).expand(3, 8, 3))


def test_wheel_spin_can_change_attachment_yaw_and_wheels_keep_separate_footprints():
    value = poses()
    value[0, 1, 3:] = torch.tensor([0., .8, 0., .6])
    raycast, calls = capture(flat_hits)
    wheel_ground_scan(value, torch.zeros(1), raycast)
    starts = calls[0][0].reshape(1, 2, 4, 3)
    first_offset = starts[0, 0, 0, :2] - value[0, 0, :2]
    second_offset = starts[0, 1, 0, :2] - value[0, 1, :2]
    torch.testing.assert_close(second_offset, -first_offset, rtol=0., atol=5e-8)
    assert not torch.allclose(first_offset, second_offset)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_per_wheel_plane_means_keep_link_origin_height_without_collision_offsets(dtype):
    value = poses(2, dtype)
    value[..., 0] = torch.tensor([[1., 2.], [-1., -2.]], dtype=dtype)
    value[..., 1] = torch.tensor([[3., 4.], [-3., -4.]], dtype=dtype)
    value[..., 2] = torch.tensor([[.3, .4], [.5, .6]], dtype=dtype)

    def slope(starts, directions):
        hit = starts.clone()
        hit[..., 2] = .12 * starts[..., 0] + .07 * starts[..., 1] + .1
        return hit

    raycast, calls = capture(slope)
    means, counts, heights = wheel_ground_scan(value, torch.full((2,), 50., dtype=dtype), raycast)
    starts = calls[0][0].reshape(2, 2, 4, 3)
    expected = (.12 * starts[..., 0] + .07 * starts[..., 1] + .1).mean(-1)
    assert torch.equal(means, expected)
    assert torch.equal(counts, torch.full((2, 2), 4, dtype=torch.long))
    assert torch.equal(heights, value[..., 2] - expected)
    assert bool((heights[0] < 0.).all())


def test_finite_zero_hits_and_partial_misses_do_not_use_fallback():
    value = poses(2)
    hit_z = torch.tensor([[0., torch.nan, torch.inf, -torch.inf, 2., 4., torch.nan, 6.],
                         [torch.nan] * 8])

    def supplied(starts, directions):
        result = starts.clone()
        result[..., 2] = hit_z
        return result

    means, counts, heights = wheel_ground_scan(value, torch.tensor([100., .7]), supplied)
    assert torch.equal(counts, torch.tensor([[1, 3], [0, 0]]))
    assert torch.equal(means, torch.tensor([[0., 4.], [.7, .7]]))
    assert torch.equal(heights, value[..., 2] - means)
    assert counts[0, 0] == 1 and means[0, 0] == 0.


def test_source_reduction_validity_uses_finite_z_and_exposes_all_miss_counts():
    def hit(starts, directions):
        result = torch.full_like(starts, torch.nan)
        result[..., 2] = torch.tensor([[0., .1, .2, .3, torch.nan, torch.inf, -torch.inf, torch.nan]])
        return result

    means, count, _ = wheel_ground_scan(poses(), torch.tensor([.8]), hit)
    torch.testing.assert_close(means, torch.tensor([[.15, .8]]))
    assert torch.equal(count, torch.tensor([[4, 0]]))


def test_repeated_calls_are_stateless_and_inputs_are_not_changed_by_callback():
    value, ground = poses(2), torch.tensor([.1, .2])
    before = value.clone(), ground.clone()

    def hit(starts, directions):
        starts[..., 2] = 0.
        directions.zero_()
        return starts

    first = wheel_ground_scan(value, ground, hit)
    second = wheel_ground_scan(value, ground, hit)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(value, before[0]) and torch.equal(ground, before[1])


@pytest.mark.parametrize("bad_pose", [
    None, torch.zeros(0, 2, 7), torch.zeros(1, 7), torch.zeros(1, 3, 7),
    torch.zeros(1, 2, 6), torch.zeros(1, 2, 7, dtype=torch.long),
    torch.zeros(1, 2, 7, dtype=torch.bool),
])
def test_bad_pose_layout_or_dtype_is_rejected_before_raycast(bad_pose):
    with pytest.raises(ValueError):
        wheel_ground_scan(bad_pose, torch.zeros(1), lambda *_: pytest.fail("callback must not run"))


@pytest.mark.parametrize("bad_ground", [None, torch.zeros(2), torch.zeros(1, 1),
                                       torch.zeros(1, dtype=torch.float64), torch.tensor([torch.nan]),
                                       torch.tensor([torch.inf]), torch.empty(1, device="meta")])
def test_bad_base_ground_is_not_a_silent_zero_or_untyped_fallback(bad_ground):
    with pytest.raises(ValueError):
        wheel_ground_scan(poses(), bad_ground, lambda *_: pytest.fail("callback must not run"))


@pytest.mark.parametrize("coordinate,value", [(0, torch.nan), (1, torch.inf), (2, -torch.inf),
                                             (3, torch.nan), (6, 0.), (6, 2.)])
def test_invalid_link_pose_or_quaternion_is_rejected(coordinate, value):
    bad = poses()
    bad[0, 0, coordinate] = value
    with pytest.raises(ValueError):
        wheel_ground_scan(bad, torch.zeros(1), lambda *_: pytest.fail("callback must not run"))


@pytest.mark.parametrize("bad_callback", [None, False, torch.zeros(1)])
def test_raycast_callback_is_explicit(bad_callback):
    with pytest.raises(TypeError, match="callback"):
        wheel_ground_scan(poses(), torch.zeros(1), bad_callback)


@pytest.mark.parametrize("bad_hit", [None, (torch.zeros(1, 8, 3),), torch.zeros(1, 2, 4, 3),
                                    torch.zeros(1, 8), torch.zeros(2, 8, 3),
                                    torch.zeros(1, 8, 3, dtype=torch.float64),
                                    torch.zeros(1, 8, 3, dtype=torch.bool),
                                    torch.empty(1, 8, 3, device="meta")])
def test_callback_must_return_typed_hit_points_in_the_same_batch_order(bad_hit):
    with pytest.raises(ValueError, match="hit points"):
        wheel_ground_scan(poses(), torch.zeros(1), lambda *_: bad_hit)


def test_module_and_measurement_require_no_native_imports():
    code = """
import sys, torch
from wheeled_tasks.chassis.wheel_ground_scan import wheel_ground_scan
pose = torch.zeros(1, 2, 7)
pose[..., 2] = .3
pose[..., 6] = 1.
def raycast(starts, directions):
    hits = starts.clone()
    hits[..., 2] = 0.
    return hits
ground, count, height = wheel_ground_scan(pose, torch.tensor([.9]), raycast)
assert count.tolist() == [[4, 4]] and ground.tolist() == [[0., 0.]]
assert not any(name.startswith(('isaaclab', 'omni', 'pxr', 'warp')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
