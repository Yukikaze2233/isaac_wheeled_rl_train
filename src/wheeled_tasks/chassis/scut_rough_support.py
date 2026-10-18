"""0730 base scan and conservative whole-robot birth support on the actual mesh."""
from __future__ import annotations

import math

import torch
import warp as wp

from .flat_birth_placement import _rpy_rotation_and_xyzw


def base_ground_scan(root_pose_world, origin_z_world, raycast):
    """Nine source rays: 2 cm square, 1 cm spacing, yaw only, root z + 5 m."""
    xy = root_pose_world[:, :2]
    x, y, z, w = root_pose_world[:, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    axis = torch.arange(-.01, .01 + 1.e-9, .01, device=xy.device, dtype=xy.dtype)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    offsets = torch.stack((gx.flatten(), gy.flatten()), -1)
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    starts = root_pose_world.new_empty(len(xy), 9, 3)
    starts[:, :, 0] = xy[:, :1] + c * offsets[:, 0] - s * offsets[:, 1]
    starts[:, :, 1] = xy[:, 1:2] + s * offsets[:, 0] + c * offsets[:, 1]
    starts[:, :, 2] = root_pose_world[:, 2:3] + 5.
    directions = torch.zeros_like(starts)
    directions[:, :, 2] = -1.
    hits = raycast(starts, directions)[..., 2]
    valid = torch.isfinite(hits)
    counts = valid.sum(-1)
    mean = hits.masked_fill(~valid, 0.).sum(-1) / counts.clamp_min(1)
    return torch.where(counts > 0, mean, origin_z_world), counts


@wp.kernel
def _mesh_upper_bounds(mesh: wp.uint64, lower: wp.array(dtype=wp.vec3),
                       upper: wp.array(dtype=wp.vec3), height: wp.array(dtype=float)):
    i = wp.tid()
    value = float(-wp.inf)
    query = wp.mesh_query_aabb(mesh, lower[i], upper[i])
    for face in query:
        a = wp.mesh_eval_position(mesh, face, 1., 0.)
        b = wp.mesh_eval_position(mesh, face, 0., 1.)
        c = wp.mesh_eval_position(mesh, face, 0., 0.)
        value = wp.max(value, wp.max(a[2], wp.max(b[2], c[2])))
    height[i] = value


def mesh_upper_bounds(mesh, bounds_m, lower_xy, upper_xy):
    """Conservative upper z of triangles whose XY bounds overlap each footprint.

    The mesh is the retained native collider. Triangle AABB overlap may raise
    a birth more than exact triangle/shape intersection requires, but cannot
    miss a narrow bar between sampled rays. No regular sampling grid is used.
    """
    shape = lower_xy.shape[:-1]
    lower = torch.empty((*shape, 3), device=lower_xy.device, dtype=torch.float32)
    upper = torch.empty_like(lower)
    lower[..., :2] = torch.nextafter(lower_xy.float(), lower_xy.new_full((), -math.inf).float())
    upper[..., :2] = torch.nextafter(upper_xy.float(), upper_xy.new_full((), math.inf).float())
    lower[..., 2], upper[..., 2] = bounds_m[0][2] - 1., bounds_m[1][2] + 1.
    result = torch.empty(shape, device=lower_xy.device, dtype=torch.float32)
    wp.launch(_mesh_upper_bounds, dim=result.numel(),
        inputs=[mesh.id, wp.from_torch(lower.reshape(-1, 3), dtype=wp.vec3),
                wp.from_torch(upper.reshape(-1, 3), dtype=wp.vec3), wp.from_torch(result.reshape(-1))],
        device=str(lower_xy.device))
    return result.to(lower_xy.dtype)


class RoughBirthSupport:
    """Raise a prepared reachable pose using every authenticated collision shape.

    Joint positions, joint velocities, root orientation and velocity stay intact.
    The existing explicit placement clearance is retained. This geometric
    envelope is conservative and is not native contact/dynamics qualification.
    """

    def __init__(self, placement, terrain):
        self.placement, self.terrain = placement, terrain

    def adjust(self, pending, *, actual_root_pose_orientation=False):
        placement = self.placement
        q = pending.joint_position.to(device=placement.device, dtype=placement.dtype)
        root = pending.root_pose
        frames = placement.kinematics.frames(q)
        if type(actual_root_pose_orientation) is not bool:
            raise ValueError("Rough support orientation selection requires a boolean")
        if actual_root_pose_orientation:
            quaternion = root[:, 3:7].to(q)
            if (not bool(torch.isfinite(quaternion).all())
                    or bool(((quaternion.square().sum(-1) - 1.).abs() > 1e-6).any())):
                raise ValueError("Rough support requires the actual unit root quaternion")
            x, y, z, w = quaternion.unbind(-1)
            rotation = torch.stack((
                1. - 2. * (y.square() + z.square()), 2. * (x * y - w * z), 2. * (x * z + w * y),
                2. * (x * y + w * z), 1. - 2. * (x.square() + z.square()), 2. * (y * z - w * x),
                2. * (x * z - w * y), 2. * (y * z + w * x), 1. - 2. * (x.square() + y.square()),
            ), -1).reshape(-1, 3, 3)
        else:
            rotation, _ = _rpy_rotation_and_xyzw(pending.requests.source_root_rpy_rad.to(q))
        lows, highs = [], []
        for body, _, kind, origin, geometry in placement._shapes:
            transform = frames[body] @ origin
            axes = rotation @ transform[:, :3, :3]
            center = (rotation @ transform[:, :3, 3:4]).squeeze(-1)
            if kind == "mesh":
                vertices = axes @ geometry.T
                lower, upper = vertices.amin(-1) + center, vertices.amax(-1) + center
            else:
                radius, length = geometry
                extent = radius * axes[:, :, :2].norm(dim=-1) + .5 * length * axes[:, :, 2].abs()
                lower, upper = center - extent, center + extent
            lows.append(lower)
            highs.append(upper)
        lower, upper = torch.stack(lows, 1), torch.stack(highs, 1)
        # The optional rehearsal path retains both native meshes in warp_mesh.
        # Its query bounds are not the boundary region: the intervening gap has
        # no triangles and cannot silently become support for a reset.
        query_bounds = getattr(self.terrain, "query_bounds_m", self.terrain.bounds_m)
        support = mesh_upper_bounds(self.terrain.warp_mesh, query_bounds,
            lower[:, :, :2] + root[:, None, :2], upper[:, :, :2] + root[:, None, :2])
        if not bool(torch.isfinite(support).all()):
            raise RuntimeError("Rough birth collision footprint has no finite mesh support")
        required = (support - lower[:, :, 2] + placement.clearance_m).amax(-1)
        z = torch.maximum(root[:, 2], required)
        clearance = (z[:, None] + lower[:, :, 2] - support).amin(-1)
        z = torch.where(clearance < placement.clearance_m,
                        torch.nextafter(z, torch.full_like(z, math.inf)), z)
        clearance = (z[:, None] + lower[:, :, 2] - support).amin(-1)
        if not bool((torch.isfinite(z) & (clearance >= placement.clearance_m)).all()):
            raise RuntimeError("Rough birth collision envelope clearance is not representable")
        # FlatResetState owns mutable tensors and is still pending: update the
        # actual state that commit_reset will record, before any native write.
        pending.placement_raise_m.add_(z - root[:, 2])
        root[:, 2].copy_(z)
        pending.minimum_shape_clearance_m.copy_(clearance)

    def summary(self):
        result = {"source": "bound_robot_collision_shapes_and_retained_native_terrain_mesh",
                "method": "conservative_per_shape_world_AABB_mesh_triangle_upper_envelope",
                "clearance_m": self.placement.clearance_m,
                "clearance_reference": "float64_geometric_bound_before_float32_native_write",
                "mesh_sha256": self.terrain.mesh_sha256,
                "changes": "root_world_z_only_before_native_reset_write",
                "native_contact_qualified": False}
        if getattr(self.terrain, "rehearsal_bounds_m", None) is not None:
            result.update(source="bound_robot_collision_shapes_and_both_actual_native_terrain_meshes",
                          query_geometry="rough_and_finite_rehearsal_plane_triangle_union",
                          rough_mesh_sha256=self.terrain.source_rough_mesh_sha256,
                          rehearsal_mesh_sha256=self.terrain.rehearsal_mesh_sha256)
        return result
