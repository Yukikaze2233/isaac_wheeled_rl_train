"""Reset-only recovery geometry construction; no native or curriculum ownership.

The caller authenticates the endpoint poses, supplies interpolation/yaw and
world-COM velocity requests, and owns any decision to admit the returned state.
Only leg requests are randomly sampled here, using the existing source sampler.
Geometric validity does not certify self-collision, recovery or native dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .nonsteady_initial_state import CONTROL_JOINT_NAMES, FlatBirthRequests, FlatBirthSampler


@dataclass(frozen=True)
class RecoveryBirthResult:
    env_ids: torch.Tensor
    tree_joint_names: tuple[str, ...]
    asset_manifest_sha256: str
    root_pose: torch.Tensor
    root_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    valid: torch.Tensor
    proposal_counts: torch.Tensor
    accepted_counts: torch.Tensor
    rejection_counts: torch.Tensor
    rejection_reasons: dict[str, torch.Tensor]  # Per-lane, nonexclusive counts over attempted proposals.
    leg_requests: FlatBirthRequests
    interpolation_fraction: torch.Tensor
    yaw_offset_rad: torch.Tensor
    requested_root_position_m: torch.Tensor
    requested_root_quaternion_xyzw: torch.Tensor
    placement_raise_m: torch.Tensor
    minimum_shape_clearance_m: torch.Tensor


class RecoveryBirthSampler:
    """Combine explicit body poses with mapped source legs at episode reset.

    Endpoint positions are offsets from origins.z and reset_xy_world, exactly as
    PhysicalFlatPlacement requests. They are interpolated using the supplied
    fraction; source leg-sampler root heights/orientations are not substituted.
    Rejection redraws only legs with the caller's existing bounded policy. On
    exhaustion, invalid rows remain unavailable and the last raw leg request is
    retained for diagnosis. This object never commits an episode or changes RNG
    beyond sampling from the supplied environment generator.
    Native reset admission remains the caller's responsibility.
    """

    def __init__(self, num_envs, device, mapper, placement, *, birth_rejection):
        self.sampler = FlatBirthSampler(num_envs, device)
        self.mapper, self.placement = mapper, placement
        self.device = self.sampler.device
        self.dtype = placement.dtype
        if (mapper.device != self.device or placement.device != self.device
                or mapper.spec_canonical_sha256 != placement.spec_canonical_sha256
                or mapper.manifest_canonical_sha256 != placement.manifest_canonical_sha256
                or mapper.kinematics.tree_joint_names != placement.kinematics.tree_joint_names
                or mapper.kinematics.control_joint_names != CONTROL_JOINT_NAMES):
            raise ValueError("Recovery birth components require the same bound asset, device and named order")
        self.rejection = dict(birth_rejection)
        if self.rejection == {"mode": "abort"}:
            self.max_attempts = 1
        elif (set(self.rejection) == {"mode", "max_attempts"}
                and self.rejection["mode"] == "redraw"
                and type(self.rejection["max_attempts"]) is int
                and self.rejection["max_attempts"] > 0):
            self.max_attempts = self.rejection["max_attempts"]
        else:
            raise ValueError("Recovery birth requires the existing explicit bounded rejection policy")

    def _check(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.device != self.device or value.dtype != self.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Recovery {name} requires finite {shape} on the placement dtype/device")

    def _attitude(self, start, end, fraction, yaw):
        """Shortest-arc SLERP followed by an extrinsic world-Z rotation."""
        start = start / start.norm(dim=-1, keepdim=True)
        end = end / end.norm(dim=-1, keepdim=True)
        dot = (start * end).sum(-1)
        end = torch.where((dot < 0.)[:, None], -end, end)
        dot = dot.abs().clamp(0., 1.)
        theta = dot.acos()
        denominator = theta.sin().clamp_min(torch.finfo(self.dtype).eps)
        curved = (((1. - fraction) * theta).sin()[:, None] * start
                  + (fraction * theta).sin()[:, None] * end) / denominator[:, None]
        linear = (1. - fraction[:, None]) * start + fraction[:, None] * end
        quaternion = torch.where((1. - dot <= 8 * torch.finfo(self.dtype).eps)[:, None], linear, curved)
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
        x, y, z, w = quaternion.unbind(-1)
        sine, cosine = (yaw * .5).sin(), (yaw * .5).cos()
        return torch.stack((cosine * x - sine * y, cosine * y + sine * x,
                            cosine * z + sine * w, cosine * w - sine * z), -1)

    def _placement_rpy(self, quaternion):
        """Equivalent extrinsic XYZ angles, including pitch gimbal endpoints."""
        x, y, z, w = quaternion.unbind(-1)
        r00, r10 = 1. - 2. * (y * y + z * z), 2. * (x * y + w * z)
        r20 = 2. * (x * z - w * y)
        r21, r22 = 2. * (y * z + w * x), 1. - 2. * (x * x + y * y)
        cosine_pitch = (r00.square() + r10.square()).sqrt()
        roll, yaw = torch.atan2(r21, r22), torch.atan2(r10, r00)
        singular = cosine_pitch <= 8 * torch.finfo(self.dtype).eps
        r01, r11 = 2. * (x * y - w * z), 1. - 2. * (x * x + z * z)
        roll = torch.where(singular, torch.zeros_like(roll), roll)
        yaw = torch.where(singular, torch.atan2(-r01, r11), yaw)
        return torch.stack((roll, torch.atan2(-r20, cosine_pitch), yaw), -1)

    def sample(self, ids, generator, *, start_root_position_m, end_root_position_m,
               start_root_quaternion_xyzw, end_root_quaternion_xyzw,
               interpolation_fraction, yaw_offset_rad, root_velocity_com_world6,
               root_velocity_frame, origins, reset_xy_world, ground_z_world):
        ids = self.sampler._ids(ids)
        count = len(ids)
        if (not isinstance(generator, torch.Generator)
                or torch.device(generator.device) != self.device):
            raise ValueError("Recovery births require the environment generator on the sampler device")
        if not isinstance(root_velocity_frame, str) or root_velocity_frame != "com_world":
            raise ValueError("Recovery births require explicit root_velocity_frame=com_world")
        fields = ((start_root_position_m, (count, 3), "start_root_position_m"),
                  (end_root_position_m, (count, 3), "end_root_position_m"),
                  (start_root_quaternion_xyzw, (count, 4), "start_root_quaternion_xyzw"),
                  (end_root_quaternion_xyzw, (count, 4), "end_root_quaternion_xyzw"),
                  (interpolation_fraction, (count,), "interpolation_fraction"),
                  (yaw_offset_rad, (count,), "yaw_offset_rad"),
                  (root_velocity_com_world6, (count, 6), "root_velocity_com_world6"),
                  (origins, (count, 3), "origins"),
                  (reset_xy_world, (count, 2), "reset_xy_world"),
                  (ground_z_world, (count,), "ground_z_world"))
        for value, shape, name in fields:
            self._check(value, shape, name)
        if bool(((interpolation_fraction < 0.) | (interpolation_fraction > 1.)).any()):
            raise ValueError("Recovery interpolation_fraction must lie in [0,1]")
        for quaternion in (start_root_quaternion_xyzw, end_root_quaternion_xyzw):
            if not torch.allclose(quaternion.norm(dim=-1), torch.ones(count, dtype=self.dtype, device=self.device),
                                  atol=1e-6, rtol=0.):
                raise ValueError("Recovery endpoint quaternions must be unit XYZW orientations")
        fraction = interpolation_fraction[:, None]
        position = (1. - fraction) * start_root_position_m + fraction * end_root_position_m
        quaternion = self._attitude(start_root_quaternion_xyzw, end_root_quaternion_xyzw,
                                    interpolation_fraction, yaw_offset_rad)
        rpy = self._placement_rpy(quaternion)
        counts = torch.zeros(count, dtype=torch.long, device=self.device)
        rejected_counts = torch.zeros_like(counts)
        valid = torch.zeros(count, dtype=torch.bool, device=self.device)
        root_pose = torch.full((count, 7), torch.nan, dtype=self.dtype, device=self.device)
        root_velocity = torch.full((count, 6), torch.nan, dtype=self.dtype, device=self.device)
        q = torch.full((count, 18), torch.nan, dtype=self.mapper.dtype, device=self.device)
        dq = torch.full_like(q, torch.nan)
        raise_m = torch.full((count,), torch.nan, dtype=self.dtype, device=self.device)
        clearance = torch.full_like(raise_m, torch.nan)
        reasons, saved_requests = {}, None
        remaining = torch.arange(count, device=self.device)
        for _ in range(self.max_attempts):
            if not len(remaining):
                break
            requests = self.sampler.sample(ids[remaining], generator)
            mapped = self.mapper.map_requests(
                requests.virtual_leg_length_m.to(dtype=self.mapper.dtype),
                -requests.virtual_leg_angle_rad.to(dtype=self.mapper.dtype),
                requests.wheel_angle_rad.to(dtype=self.mapper.dtype),
                requests.control_velocity_rad_s.to(dtype=self.mapper.dtype))
            placed = self.placement.propose(mapped.q.to(dtype=self.dtype), position[remaining], rpy[remaining],
                root_velocity_com_world6[remaining], origins[remaining], reset_xy_world[remaining],
                ground_z_world[remaining], root_velocity_frame="com_world")
            accepted = mapped.valid & placed.valid
            rejected = ~accepted
            counts[remaining] += 1
            rejected_counts[remaining] += rejected.long()
            for prefix, observed in (("mapping/", mapped.reasons), ("placement/", placed.reasons)):
                for name, mask in observed.items():
                    key = prefix + name
                    if key not in reasons:
                        reasons[key] = torch.zeros_like(counts)
                    reasons[key][remaining] += (mask & rejected).long()
            if saved_requests is None:
                saved_requests = {name: value.new_empty((count, *value.shape[1:]))
                                  for name, value in vars(requests).items()}
            for name, value in vars(requests).items():
                saved_requests[name][remaining] = value
            selected = remaining[accepted]
            valid[selected] = True
            root_pose[selected] = placed.root_pose[accepted]
            root_velocity[selected] = placed.root_velocity[accepted]
            q[selected], dq[selected] = mapped.q[accepted], mapped.dq[accepted]
            raise_m[selected], clearance[selected] = placed.placement_raise_m[accepted], placed.minimum_shape_clearance_m[accepted]
            remaining = remaining[rejected]
        leg_requests = (self.sampler.sample(ids, generator) if saved_requests is None
                        else FlatBirthRequests(**saved_requests))
        return RecoveryBirthResult(ids.clone(), self.mapper.kinematics.tree_joint_names,
            self.placement.manifest_sha256, root_pose, root_velocity, q, dq, valid,
            counts, valid.long(), rejected_counts, reasons, leg_requests,
            interpolation_fraction.clone(), yaw_offset_rad.clone(), position, quaternion,
            raise_m, clearance)
