"""Place a mapped chassis configuration above a flat plane using its geometry.

This module reads only the supplied asset's authenticated model spec and
collision meshes. It imports no Isaac/Kit or historical recovery-pose builder.
Mesh support is taken over actual OBJ vertices; wheels use the exact support
of their finite cylinders, including their end caps and collision origins.
The source virtual leg height is never used as world ground clearance.

``valid`` means finite plane placement with the requested minimum clearance,
not a closed-linkage, self-collision, route or native-dynamics qualification.
The caller must also require the mechanical mapper's position/velocity validity
and separately authenticate the manifest identity for the intended candidate.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

import torch

from .closed_chain_kinematics import ClosedChainKinematics


PLACEMENT_VERSION = "physical-flat-collision-placement-v1"


class FlatPlacementProposal(NamedTuple):
    root_pose: torch.Tensor  # [N,7], world position and quaternion xyzw
    root_velocity: torch.Tensor  # [N,6], root-COM linear/angular velocity in world axes
    valid: torch.Tensor  # [N], plane-placement validity only
    minimum_shape_clearance_m: torch.Tensor  # [N], actual lower shape support minus ground z
    placement_raise_m: torch.Tensor  # [N], increase over origins.z + source request z
    reasons: dict[str, torch.Tensor]  # [N] masks; no implicit qualification flags


def _sha(contents):
    return hashlib.sha256(contents).hexdigest()


def _rpy_rotation_and_xyzw(rpy):
    """Extrinsic xyz RPY, i.e. Rz(yaw) Ry(pitch) Rx(roll), without angle wrapping."""
    roll, pitch, yaw = rpy.unbind(-1)
    cr, cp, cy = roll.cos(), pitch.cos(), yaw.cos()
    sr, sp, sy = roll.sin(), pitch.sin(), yaw.sin()
    rotation = torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                            -sp, cp * sr, cp * cr), -1).reshape(-1, 3, 3)
    hr, hp, hy = (rpy * .5).unbind(-1)
    cr, cp, cy, sr, sp, sy = hr.cos(), hp.cos(), hy.cos(), hr.sin(), hp.sin(), hy.sin()
    xyzw = torch.stack((sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy), -1)
    return rotation, xyzw


class PhysicalFlatPlacement:
    """Pure Torch batched placement for a supplied 18-coordinate tree asset.

    ``clearance_m`` is mandatory, finite and nonnegative. It is an explicit
    caller-owned placement parameter, never an inferred safety threshold.
    Input tensors share configured dtype/device and selected-batch size N.
    All linear values are metres and all RPY/joint angles are radians.

    ``origins`` is [N,3]; requested world z is origins.z + source_request.z.
    ``reset_xy_world`` is the selected world reset anchor [N,2], to which the
    source request's x/y offsets are added exactly once. It already includes
    environment x/y origins. ``ground_z_world`` is [N] for a horizontal plane.
    By default source zero root velocity is required; nonzero requests are
    invalid rather than silently zeroed. Explicit ``root_velocity_frame="com_world"``
    accepts finite root-COM linear/angular velocity in world axes unchanged.
    Root RPY is preserved, including ordinary yaw/tilt and the sampler's
    ground-birth identity orientation. q18 is never changed.
    """

    def __init__(self, spec, manifest, asset_directory, *, clearance_m,
                 device="cpu", dtype=torch.float64):
        if (type(clearance_m) not in (int, float) or not math.isfinite(clearance_m)
                or clearance_m < 0):
            raise ValueError("Placement clearance_m must be explicit, finite and nonnegative")
        if not isinstance(spec, Mapping) or not isinstance(manifest, Mapping):
            raise ValueError("Placement requires explicit model spec and manifest mappings")
        self.asset_directory = Path(asset_directory).resolve(strict=True)
        manifest_bytes = (self.asset_directory / "manifest.json").read_bytes()
        if json.loads(manifest_bytes) != dict(manifest):
            raise ValueError("Supplied placement manifest differs from bound asset bytes")
        self.manifest_sha256 = _sha(manifest_bytes)
        self._files = dict(manifest.get("files_sha256", {}))
        spec_bytes = self._read_bound_file("model_spec.json")
        if json.loads(spec_bytes) != dict(spec):
            raise ValueError("Supplied placement spec differs from bound model_spec.json")
        self.model_spec_sha256 = _sha(spec_bytes)
        self.spec_canonical_sha256 = _sha(json.dumps(
            spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        self.manifest_canonical_sha256 = _sha(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        self.clearance_m = float(clearance_m)
        self.kinematics = ClosedChainKinematics(spec, manifest["tree_joint_names"],
            manifest["control_joint_names"], device, dtype=dtype)
        self.device, self.dtype = self.kinematics.device, self.kinematics.dtype
        if self.kinematics.joint_count != 18:
            raise ValueError("Flat birth placement requires the explicit complete q18 tree")
        bodies = spec.get("bodies", [])
        body_names = [body["name"] for body in bodies]
        frames = self.kinematics.frames(torch.empty(0, 18, dtype=dtype, device=self.device))
        if (len(set(body_names)) != len(body_names) or set(body_names) != set(frames)
                or set(body_names) != set(manifest["rigid_body_names"])):
            raise ValueError("Collision bodies must match the supplied complete joint tree/manifest")
        self._shapes, self.collision_file_sha256 = [], {}
        identity = torch.eye(4, dtype=dtype, device=self.device)
        for body in bodies:
            collisions = body.get("collisions")
            if not isinstance(collisions, list) or not collisions:
                raise ValueError("Every bound chassis body requires explicit collision geometry")
            for index, collision in enumerate(collisions):
                origin = torch.tensor(collision["origin"], dtype=dtype, device=self.device)
                if (origin.shape != (4, 4) or not bool(torch.isfinite(origin).all())
                        or not torch.allclose(origin[3], identity[3], atol=1e-6, rtol=0.)
                        or not torch.allclose(origin[:3, :3].T @ origin[:3, :3], identity[:3, :3], atol=1e-6, rtol=0.)
                        or not bool(torch.linalg.det(origin[:3, :3]) > 0)):
                    raise ValueError("Invalid collision rigid transform")
                kind = collision.get("type")
                if kind == "mesh":
                    contents = self._read_bound_file(collision["file"])
                    self.collision_file_sha256[collision["file"]] = _sha(contents)
                    vertices = []
                    for line in contents.decode("utf-8").splitlines():
                        tokens = line.split()
                        if tokens and tokens[0] == "v":
                            if len(tokens) != 4:
                                raise ValueError("Collision OBJ requires explicit Cartesian xyz vertices")
                            vertices.append([float(value) for value in tokens[1:]])
                    points = torch.tensor(vertices, dtype=dtype, device=self.device)
                    if points.ndim != 2 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
                        raise ValueError("Collision mesh requires nonempty finite xyz vertices")
                    self._shapes.append((body["name"], index, kind, origin, points))
                elif kind == "cylinder":
                    radius, length = collision.get("radius"), collision.get("length")
                    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                           for value in (radius, length)):
                        raise ValueError("Finite collision cylinder needs positive radius and length")
                    self._shapes.append((body["name"], index, kind, origin, (float(radius), float(length))))
                else:
                    raise ValueError(f"Unsupported collision geometry: {kind!r}")

    def _read_bound_file(self, name):
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or ".." in Path(name).parts or name not in self._files):
            raise ValueError("Collision dependency must be a manifest-bound relative asset path")
        path = (self.asset_directory / name).resolve(strict=True)
        if not path.is_relative_to(self.asset_directory):
            raise ValueError("Collision dependency escapes the bound asset directory")
        contents = path.read_bytes()
        if _sha(contents) != self._files[name]:
            raise ValueError(f"Placement dependency SHA mismatch: {name}")
        return contents

    def _check(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != self.dtype or value.device != self.device):
            raise ValueError(f"{name} must have shape {shape} on configured dtype/device")

    def propose(self, q18, source_root_position_m, source_root_rpy_rad,
                source_root_velocity6, origins, reset_xy_world, ground_z_world,
                *, root_velocity_frame=None):
        if (root_velocity_frame is not None
                and (not isinstance(root_velocity_frame, str) or root_velocity_frame != "com_world")):
            raise ValueError("Placement root_velocity_frame must be None or com_world")
        if not isinstance(q18, torch.Tensor) or q18.ndim != 2:
            raise ValueError("q18 must be an explicit [N,18] tensor")
        count = len(q18)
        fields = ((q18, (count, 18), "q18"),
            (source_root_position_m, (count, 3), "source_root_position_m"),
            (source_root_rpy_rad, (count, 3), "source_root_rpy_rad"),
            (source_root_velocity6, (count, 6), "source_root_velocity6"),
            (origins, (count, 3), "origins"), (reset_xy_world, (count, 2), "reset_xy_world"),
            (ground_z_world, (count,), "ground_z_world"))
        for value, shape, name in fields:
            self._check(value, shape, name)
        finite = torch.ones(count, dtype=torch.bool, device=self.device)
        for value, _, _ in fields:
            finite &= torch.isfinite(value).reshape(count, -1).all(-1) if count else finite
        zero_velocity = (source_root_velocity6 == 0.).all(-1)
        velocity_allowed = (torch.ones_like(zero_velocity)
                            if root_velocity_frame == "com_world" else zero_velocity)
        # Invalid lanes are isolated for arithmetic, then returned as NaN poses
        # with explicit invalid reasons; they are never replaced by nominal poses.
        safe = lambda value: torch.where(torch.isfinite(value), value, 0.)
        frames = self.kinematics.frames(safe(q18))
        rotation, quaternion = _rpy_rotation_and_xyzw(safe(source_root_rpy_rad))
        lower = torch.full((count,), math.inf, dtype=self.dtype, device=self.device)
        for body, _, kind, origin, geometry in self._shapes:
            transform = frames[body] @ origin
            row = (rotation[:, 2:3, :] @ transform[:, :3, :3]).squeeze(1)
            center_z = (rotation[:, 2, :] * transform[:, :3, 3]).sum(-1)
            if kind == "mesh":
                bottom = (row @ geometry.T).amin(-1) + center_z
            else:
                radius, length = geometry
                extent = radius * row[:, :2].norm(dim=-1) + .5 * length * row[:, 2].abs()
                bottom = center_z - extent
            lower = torch.minimum(lower, bottom)
        requested_z = safe(origins[:, 2]) + safe(source_root_position_m[:, 2])
        ground = safe(ground_z_world)
        root_z = torch.maximum(requested_z, ground - lower + self.clearance_m)
        clearance = root_z + lower - ground
        # The first successor can be lost when root_z+lower rounds again.
        # Retry once without changing the physical threshold; any remaining
        # failure is still returned explicitly rather than accepted by tolerance.
        for _ in range(2):
            root_z = torch.where(clearance < self.clearance_m,
                torch.nextafter(root_z, torch.full_like(root_z, math.inf)), root_z)
            clearance = root_z + lower - ground
        raised = root_z - requested_z
        pose = torch.cat((safe(reset_xy_world) + safe(source_root_position_m[:, :2]),
                          root_z[:, None], quaternion), -1)
        geometric_finite = torch.isfinite(pose).all(-1) & torch.isfinite(clearance) & torch.isfinite(raised)
        clearance_ok = clearance >= self.clearance_m
        valid = finite & velocity_allowed & geometric_finite & clearance_ok
        reasons = {"nonfinite_input": ~finite, "nonzero_source_root_velocity": ~velocity_allowed,
                   "nonfinite_geometry": ~geometric_finite, "ground_clearance_not_representable": ~clearance_ok}
        mask = lambda value: torch.where(valid.reshape((count,) + (1,) * (value.ndim - 1)), value, math.nan)
        return FlatPlacementProposal(mask(pose), mask(source_root_velocity6.clone()), valid,
            mask(clearance), mask(raised), reasons)

    def summary(self):
        return {"version": PLACEMENT_VERSION, "manifest_sha256": self.manifest_sha256,
            "model_spec_sha256": self.model_spec_sha256,
            "collision_file_sha256": dict(self.collision_file_sha256),
            "shape_count": len(self._shapes), "clearance_m": self.clearance_m,
            "tree_joint_names": list(self.kinematics.tree_joint_names),
            "root_pose_quaternion_order": "xyzw", "ground_model": "horizontal_world_plane",
            "scope": "CPU_or_Torch_collision_shape_support_and_plane_placement_only",
            "validity_requires_separate_mapper_result": True,
            "native_qualified": False,
            "pending_validation": ["native_filtered_self_collision_pairs_and_contacts",
                "native_full_tree_reset_and_physical_readback", "route_or_nonplanar_support",
                "mechanical_asset_and_training_admission"]}
