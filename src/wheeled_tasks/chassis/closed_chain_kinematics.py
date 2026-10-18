"""Batched tree geometry and tangent velocities for a supplied closed linkage.

The FK and endpoint residual follow the sealed asset's ``tools/v6_mechanism.py``
and ``tools/build_v6_closedchain.py``: joint origin followed by joint motion,
then body-local spherical-closure endpoints. Dimensions, axes, limits and
coordinate names come from model_spec/manifest, never another robot's IK.

This is reset preparation, not a simulator, collision checker, or branch
selector. The caller must authenticate the input asset and provide a nearby
initial configuration on the intended assembly branch. No limit is extended,
no angle is wrapped, and an invalid result must not be written to physics.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch


class PositionSolution(NamedTuple):
    q: torch.Tensor
    valid: torch.Tensor
    residual: torch.Tensor


class VelocitySolution(NamedTuple):
    dq: torch.Tensor
    valid: torch.Tensor
    closure_velocity_residual: torch.Tensor


class ClosedChainKinematics:
    """Pure Torch FK and local passive-coordinate solves, with lane validity.

    Revolute coordinates/velocities are rad/rad s^-1; prismatic coordinates
    are m/m s^-1. Closure residuals are m and m s^-1. Default 1e-8 numerical
    tolerances match the existing geometric pose check; they are not native
    dynamic acceptance thresholds. Float64 is recommended for reset solves.
    """

    def __init__(self, spec, tree_joint_names, control_joint_names, device,
                 dtype=torch.float64, *, position_tolerance_m=1e-8,
                 velocity_tolerance_m_s=1e-8, limit_tolerance=1e-10,
                 rank_rtol=None):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Geometry requires float32 or float64")
        # Resolve aliases such as "cuda" to the actual tensor device (cuda:0).
        self.device, self.dtype = torch.empty(0, device=device).device, dtype
        self.tree_joint_names = tuple(tree_joint_names)
        self.control_joint_names = tuple(control_joint_names)
        joints = spec.get("joints", [])
        names = [joint["name"] for joint in joints]
        if (not names or len(set(names)) != len(names)
                or len(self.tree_joint_names) != len(names)
                or len(set(self.tree_joint_names)) != len(names)
                or set(self.tree_joint_names) != set(names)
                or not self.control_joint_names
                or len(set(self.control_joint_names)) != len(self.control_joint_names)
                or not set(self.control_joint_names) < set(names)):
            raise ValueError("Expected unique tree/control names and passive coordinates")
        self.joint_count = len(names)
        self.control_indices = torch.tensor([self.tree_joint_names.index(n)
            for n in self.control_joint_names], device=self.device)
        self.passive_indices = torch.tensor([i for i, n in enumerate(self.tree_joint_names)
            if n not in self.control_joint_names], device=self.device)
        for value in (position_tolerance_m, velocity_tolerance_m_s, limit_tolerance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Numerical tolerances must be finite and positive")
        self.position_tolerance_m = position_tolerance_m
        self.velocity_tolerance_m_s = velocity_tolerance_m_s
        self.limit_tolerance = limit_tolerance

        children = [joint["child"] for joint in joints]
        roots = set(joint["parent"] for joint in joints) - set(children)
        if len(set(children)) != len(children) or len(roots) != 1:
            raise ValueError("Expected one connected joint tree")
        self.root_name = next(iter(roots))
        known, remaining, ordered = {self.root_name}, list(joints), []
        while remaining:
            ready = [j for j in remaining if j["parent"] in known]
            if not ready:
                raise ValueError("Disconnected or cyclic joint tree")
            for joint in ready:
                known.add(joint["child"])
                ordered.append(joint)
                remaining.remove(joint)

        self._eye = torch.eye(4, dtype=dtype, device=self.device)
        self._eye3 = self._eye[:3, :3]
        self._joints, ancestors = [], {self.root_name: []}
        lower, upper, speeds = [-math.inf] * len(names), [math.inf] * len(names), [math.inf] * len(names)
        prismatic = [False] * len(names)
        for joint in ordered:
            index = self.tree_joint_names.index(joint["name"])
            kind = joint["type"]
            if kind not in ("continuous", "revolute", "prismatic"):
                raise ValueError("Only scalar revolute/prismatic tree joints are supported")
            origin = torch.as_tensor(joint["origin"], dtype=dtype, device=self.device).clone()
            axis = torch.as_tensor(joint["axis"], dtype=dtype, device=self.device).clone()
            if (origin.shape != (4, 4) or axis.shape != (3,)
                    or not bool(torch.isfinite(origin).all() & torch.isfinite(axis).all())
                    or not bool(torch.isfinite(axis.norm()) & (axis.norm() > 0))
                    or not torch.allclose(origin[3], self._eye[3], atol=1e-6, rtol=0.)
                    or not torch.allclose(origin[:3, :3].T @ origin[:3, :3],
                                          self._eye3, atol=1e-6, rtol=0.)
                    or not bool(torch.linalg.det(origin[:3, :3]) > 0)):
                raise ValueError("Invalid joint origin or axis")
            # Preserve the sealed FK's rotvec semantics, including axis norm.
            norm = axis.norm()
            x, y, z = axis / norm
            zero = torch.zeros((), dtype=dtype, device=self.device)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
            self._joints.append((index, joint["parent"], joint["child"], kind,
                                 origin, axis, norm, skew, skew @ skew))
            ancestors[joint["child"]] = ancestors[joint["parent"]] + [index]
            prismatic[index] = kind == "prismatic"
            limit = joint.get("limit", {})
            if kind != "continuous":
                lower[index], upper[index] = float(limit["lower"]), float(limit["upper"])
                if not (math.isfinite(lower[index]) and math.isfinite(upper[index])
                        and lower[index] < upper[index]):
                    raise ValueError("Invalid bounded joint limits")
            if "velocity" in limit:
                speeds[index] = float(limit["velocity"])
                if not math.isfinite(speeds[index]) or speeds[index] <= 0:
                    raise ValueError("Invalid joint velocity limit")
        # Check physical slide travel as well as the declared joint bounds.
        for name, binding in spec.get("spring_binding", {}).items():
            index = self.tree_joint_names.index(name)
            compression = float(binding["compression_at_q_zero_m"])
            stroke = float(binding["stroke_m"])
            if not prismatic[index] or not math.isfinite(compression) or not math.isfinite(stroke) or stroke <= 0:
                raise ValueError("Invalid spring slide binding")
            lower[index] = max(lower[index], compression - stroke)
            upper[index] = min(upper[index], compression)
            if lower[index] >= upper[index]:
                raise ValueError("Spring travel and joint domain do not overlap")
        self.lower = torch.tensor(lower, dtype=dtype, device=self.device)
        self.upper = torch.tensor(upper, dtype=dtype, device=self.device)
        self.speed_limits = torch.tensor(speeds, dtype=dtype, device=self.device)
        self._prismatic = torch.tensor(prismatic, device=self.device)

        constraints = spec.get("constraints", [])
        self._endpoint_bodies, local_points, endpoint_ancestors = [], [], []
        for constraint in constraints:
            if constraint.get("type") != "spherical":
                raise ValueError("Expected spherical point-closure constraints")
            for end in (0, 1):
                body = constraint[f"body{end}"]
                if body not in known:
                    raise ValueError("Closure endpoint is not in the joint tree")
                self._endpoint_bodies.append(body)
                local_points.append(constraint[f"local_pos{end}_m"])
                endpoint_ancestors.append([i in ancestors[body] for i in range(len(names))])
        self.residual_count = len(constraints) * 3
        if self.residual_count < len(self.passive_indices):
            raise ValueError("Closure must determine all passive coordinates")
        self._local_points = torch.as_tensor(local_points, dtype=dtype, device=self.device).clone()
        if self._local_points.shape != (len(constraints) * 2, 3) or not bool(torch.isfinite(self._local_points).all()):
            raise ValueError("Invalid local closure points")
        self._endpoint_ancestors = torch.tensor(endpoint_ancestors, device=self.device)
        self.rank_rtol = (max(self.residual_count, len(self.passive_indices))
                          * torch.finfo(dtype).eps if rank_rtol is None else rank_rtol)
        if not math.isfinite(self.rank_rtol) or not 0 < self.rank_rtol < 1:
            raise ValueError("Rank tolerance must be finite and between zero and one")
        self._safe_matrix = torch.eye(self.residual_count, len(self.passive_indices),
                                      dtype=dtype, device=self.device)

    def _check(self, value, width, name):
        if (not isinstance(value, torch.Tensor) or value.ndim != 2
                or value.shape[1] != width or value.device != self.device or value.dtype != self.dtype):
            raise ValueError(f"{name} must be [N,{width}] on the configured device/dtype")

    def _kinematics(self, q):
        self._check(q, self.joint_count, "q")
        count = len(q)
        frames = {self.root_name: self._eye.expand(count, -1, -1)}
        origins, axes = [None] * self.joint_count, [None] * self.joint_count
        for index, parent, child, kind, origin, axis, norm, skew, skew2 in self._joints:
            before = frames[parent] @ origin
            origins[index] = before[:, :3, 3]
            axes[index] = (before[:, :3, :3] @ axis).reshape(count, 3)
            motion = self._eye.expand(count, -1, -1).clone()
            if kind == "prismatic":
                motion[:, :3, 3] = q[:, index, None] * axis
            else:
                angle = q[:, index] * norm
                motion[:, :3, :3] = (self._eye3 + angle.sin()[:, None, None] * skew
                                       + (1. - angle.cos())[:, None, None] * skew2)
            frames[child] = before @ motion
        return frames, torch.stack(origins, 1), torch.stack(axes, 1)

    def frames(self, q):
        """Return body-name -> [N,4,4] poses relative to the tree root."""
        return self._kinematics(q)[0]

    def _points(self, frames):
        return torch.stack([frames[body][:, :3, :3] @ self._local_points[i]
                            + frames[body][:, :3, 3]
                            for i, body in enumerate(self._endpoint_bodies)], 1)

    def residual(self, q):
        """Return endpoint0 - endpoint1, flattened as [N,3*C] in meters."""
        points = self._points(self.frames(q))
        return (points[:, 0::2] - points[:, 1::2]).reshape(len(q), self.residual_count)

    def _geometry(self, q):
        frames, origins, axes = self._kinematics(q)
        points = self._points(frames)
        arms = points[:, :, None, :] - origins[:, None, :, :]
        derivatives = torch.linalg.cross(axes[:, None, :, :].expand_as(arms), arms)
        derivatives = torch.where(self._prismatic[None, None, :, None],
                                  axes[:, None, :, :], derivatives)
        derivatives = torch.where(self._endpoint_ancestors[None, :, :, None], derivatives, 0.)
        jacobian = (derivatives[:, 0::2] - derivatives[:, 1::2]).transpose(-1, -2)
        residual = (points[:, 0::2] - points[:, 1::2]).reshape(len(q), self.residual_count)
        return residual, jacobian.reshape(len(q), self.residual_count, self.joint_count)

    def jacobian(self, q):
        """Analytic endpoint Jacobian [N,3*C,J]; no per-lane autograd."""
        return self._geometry(q)[1]

    def _domain(self, q):
        return (torch.isfinite(q).all(-1)
                & (q >= self.lower - self.limit_tolerance).all(-1)
                & (q <= self.upper + self.limit_tolerance).all(-1))

    def _norm(self, residual):
        return residual.reshape(len(residual), self.residual_count // 3, 3).norm(dim=-1).amax(-1)

    def _passive_solve(self, jacobian, rhs):
        matrix = jacobian[:, :, self.passive_indices]
        finite = torch.isfinite(matrix).all(dim=(-2, -1)) & torch.isfinite(rhs).all(-1)
        matrix = torch.where(finite[:, None, None], matrix, self._safe_matrix)
        singular = torch.linalg.svdvals(matrix)
        full_rank = finite & (singular[:, -1] > self.rank_rtol * singular[:, 0])
        # One bad lane must not make a batched QR/triangular solve fail for all
        # lanes. This placeholder never changes the returned validity mask.
        matrix = torch.where(full_rank[:, None, None], matrix, self._safe_matrix)
        rhs = torch.where(full_rank[:, None], rhs, 0.)
        orthogonal, triangular = torch.linalg.qr(matrix, mode="reduced")
        value = torch.linalg.solve_triangular(triangular,
            (orthogonal.transpose(-1, -2) @ rhs[:, :, None]), upper=True).squeeze(-1)
        return value, full_rank & torch.isfinite(value).all(-1)

    def solve_passive(self, initial_q, prescribed_control6, max_iterations=12):
        """Local Newton solve changing only passive coordinates.

        Input controls are preserved exactly, even on failed lanes. No redraw,
        clipping, nominal fallback, or claimed global assembly-branch proof.
        A step outside the asset domain fails that lane rather than clipping it.
        ``residual`` contains the final actual position residual, not a mask.
        """
        self._check(initial_q, self.joint_count, "initial_q")
        self._check(prescribed_control6, len(self.control_indices), "prescribed_control6")
        if len(initial_q) != len(prescribed_control6):
            raise ValueError("Initial and prescribed coordinates need the same lane count")
        if type(max_iterations) is not int or max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        q = initial_q.clone()
        q[:, self.control_indices] = prescribed_control6
        eligible = self._domain(initial_q) & self._domain(q)
        for _ in range(max_iterations):
            safe_q = torch.where(torch.isfinite(q).all(-1)[:, None], q, 0.)
            residual, jacobian = self._geometry(safe_q)
            correction, rank_valid = self._passive_solve(jacobian, -residual)
            eligible &= rank_valid
            working = eligible & (self._norm(residual) > self.position_tolerance_m)
            candidate = q.clone()
            candidate[:, self.passive_indices] += correction
            q = torch.where(working[:, None], candidate, q)
            eligible &= self._domain(q)
            # Rank and domain checks have already run. With no working lanes,
            # another iteration would solve the same geometry without writing q.
            if not bool(working.any()):
                break
        residual = self.residual(q)
        safe_q = torch.where(torch.isfinite(q).all(-1)[:, None], q, 0.)
        _, rank_valid = self._passive_solve(self.jacobian(safe_q),
            torch.zeros(len(q), self.residual_count, dtype=self.dtype, device=self.device))
        valid = eligible & rank_valid & self._domain(q) & (self._norm(residual) <= self.position_tolerance_m)
        return PositionSolution(q, valid, residual)

    def compatible_velocities(self, q, control_dq6):
        """Preserve active velocities and solve the passive closure tangent.

        A closed position with an outward velocity at a finite position limit
        is invalid. Spec velocity limits are checked too. The returned raw
        residual and active commands remain available for failure diagnostics.
        """
        self._check(q, self.joint_count, "q")
        self._check(control_dq6, len(self.control_indices), "control_dq6")
        if len(q) != len(control_dq6):
            raise ValueError("Coordinates and velocities need the same lane count")
        finite = torch.isfinite(q).all(-1) & torch.isfinite(control_dq6).all(-1)
        safe_q = torch.where(finite[:, None], q, 0.)
        safe_control = torch.where(finite[:, None], control_dq6, 0.)
        _, jacobian = self._geometry(safe_q)
        rhs = -(jacobian[:, :, self.control_indices] @ safe_control[:, :, None]).squeeze(-1)
        passive, rank_valid = self._passive_solve(jacobian, rhs)
        dq = torch.empty_like(q)
        dq[:, self.control_indices] = control_dq6
        dq[:, self.passive_indices] = torch.where((finite & rank_valid)[:, None], passive, torch.nan)
        # Use the real q here so invalid physical evidence remains invalid.
        velocity_residual = (self.jacobian(q) @ dq[:, :, None]).squeeze(-1)
        outward = (((q <= self.lower + self.limit_tolerance) & (dq < -self.limit_tolerance))
                   | ((q >= self.upper - self.limit_tolerance) & (dq > self.limit_tolerance))).any(-1)
        valid = (finite & rank_valid & self._domain(q)
                 & (self._norm(self.residual(q)) <= self.position_tolerance_m)
                 & (self._norm(velocity_residual) <= self.velocity_tolerance_m_s)
                 & (dq.abs() <= self.speed_limits + self.limit_tolerance).all(-1)
                 & ~outward)
        return VelocitySolution(dq, valid, velocity_residual)
