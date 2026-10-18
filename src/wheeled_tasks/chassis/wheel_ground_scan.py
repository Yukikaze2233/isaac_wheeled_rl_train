"""Pure Torch wheel-ground measurements for a future explicit V7 composition.

The pinned b8ff79f3 V14 wheel scanners use a 0.015 m square with 0.01 m
spacing, default xy ordering, a local +5 m origin and world-down rays. The
installed IsaacLab-3.0.0-beta2 GridPattern and yaw kernel establish four rays
per wheel and attachment-link quaternion yaw. The supplied wheel order is
preserved; no root-yaw, collision-center, terrain-kind or contact substitution
is performed. This stateless adapter is not connected to a training profile
and does not establish any deployment sensor or airborne qualification.
"""

import torch


def wheel_ground_scan(wheel_link_pose_world, base_ground_height_world, raycast):
    """Return ground mean, finite-Z hit count and link-origin height [N,2].

    Inputs are wheel LINK poses [N,2,7] (world xyz + xyzw) and the already
    measured base ground [N], sharing float32/float64 dtype and device. The
    callback receives starts/directions [N,8,3] and returns hit points of the
    same shape, dtype and device. Rays 0..3 belong to input wheel 0; rays 4..7
    belong to wheel 1. The caller binds the actual collision mesh and raycast
    distance; this module imports neither Warp nor the simulator.

    As in the source scanner reduction, finite hit Z alone determines validity.
    A finite zero is a ground measurement. An all-miss wheel falls back to base
    ground while retaining hit_count=0; fallback is not measured support.
    """
    pose, ground = wheel_link_pose_world, base_ground_height_world
    if (not isinstance(pose, torch.Tensor) or pose.ndim != 3 or pose.shape[1:] != (2, 7)
            or pose.shape[0] == 0):
        raise ValueError("Wheel ground scan requires nonempty LINK poses [N,2,7]")
    if pose.dtype not in (torch.float32, torch.float64):
        raise ValueError("Wheel ground scan LINK poses require float32 or float64")
    if (not isinstance(ground, torch.Tensor) or ground.shape != (pose.shape[0],)
            or ground.dtype != pose.dtype or ground.device != pose.device):
        raise ValueError("Wheel ground scan base ground must share [N], dtype and device")
    if not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(ground).all()):
        raise ValueError("Wheel ground scan requires finite LINK poses and base ground")
    quaternion = pose[..., 3:7]
    if not bool((quaternion.square().sum(-1) - 1.).abs().le(1e-3).all()):
        raise ValueError("Wheel ground scan requires unit xyzw LINK quaternions")
    if not callable(raycast):
        raise TypeError("Wheel ground scan requires an explicit raycast callback")

    axis = torch.arange(-.015 / 2., .015 / 2. + 1e-9, .01,
                        dtype=pose.dtype, device=pose.device)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    offsets_x, offsets_y = grid_x.flatten(), grid_y.flatten()
    x, y, z, w = quaternion.unbind(-1)
    yaw = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y.square() + z.square()))
    cosine, sine = yaw.cos()[..., None], yaw.sin()[..., None]
    starts = pose.new_empty(pose.shape[0], 2, 4, 3)
    starts[..., 0] = pose[..., 0, None] + cosine * offsets_x - sine * offsets_y
    starts[..., 1] = pose[..., 1, None] + sine * offsets_x + cosine * offsets_y
    starts[..., 2] = pose[..., 2, None] + 5.
    starts = starts.reshape(pose.shape[0], 8, 3)
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.
    hits = raycast(starts, directions)
    if (not isinstance(hits, torch.Tensor) or hits.shape != starts.shape
            or hits.dtype != pose.dtype or hits.device != pose.device):
        raise ValueError("Wheel ground scan raycast must return hit points [N,8,3] with matching dtype/device")
    hit_z = hits[..., 2].reshape(pose.shape[0], 2, 4)
    valid = torch.isfinite(hit_z)
    counts = valid.sum(-1)
    means = hit_z.masked_fill(~valid, 0.).sum(-1) / counts.clamp_min(1)
    ground_means = torch.where(counts > 0, means, ground[:, None])
    return ground_means, counts, pose[..., 2] - ground_means
