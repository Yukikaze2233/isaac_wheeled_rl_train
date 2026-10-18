"""Map explicit virtual-leg geometry to this asset's closed-chain reset state.

Source-request lengths are never accepted as own geometry without this mapping.
Length is explicitly either hip-to-actual-wheel-center 3D distance or its
projection perpendicular to the real hip axis. Angle zero is base-down in that
plane and positive points toward base-forward. L*cos(angle) is therefore a
local projection, not world wheel-bottom height or a root-height prescription.

Tables only seed a local same-assembly solve. Collision/support placement and
native reset admission remain outside this module. Invalid requests are not
redrawn, clipped, or replaced by a nominal pose.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import NamedTuple

import torch

from .closed_chain_kinematics import ClosedChainKinematics


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class LegPoseResult(NamedTuple):
    q: torch.Tensor
    dq: torch.Tensor
    valid: torch.Tensor
    reasons: dict[str, torch.Tensor]
    requested: dict[str, torch.Tensor]
    actual: dict[str, torch.Tensor]


class LegPoseMapper:
    """Full-tree local pose mapping; caller must preserve returned validity.

    Default 1e-8 tolerances are geometric numerical tolerances, not recovery or
    tracking acceptance metrics. Float64 CPU is the currently tested path.
    The sampler owns any subsequent redraw decision and its distribution.
    """

    def __init__(self, spec, manifest, branch_table, device, dtype=torch.float64,
                 *, length_tolerance_m=1e-8, angle_tolerance_rad=1e-8):
        table = branch_table
        if (table.get("schema") != "own_leg_pose_branch_v1"
                or table.get("spec_canonical_sha256") != _canonical_sha(spec)
                or table.get("manifest_canonical_sha256") != _canonical_sha(manifest)
                or table.get("tree_joint_names") != manifest["tree_joint_names"]
                or table.get("control_joint_names") != manifest["control_joint_names"]
                or table.get("physical_knee_domain_deg") != manifest["knee_inner_limits_deg"]
                or table.get("angle_definition") != "hip_axis_plane_base_down_toward_base_forward_positive_rad"
                or table.get("length_definition") not in ("hip_axis_projected", "hip_to_wheel_3d")):
            raise ValueError("Leg table is not bound to the supplied asset/coordinate definition")
        for value in (length_tolerance_m, angle_tolerance_rad):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Mapping tolerances must be finite and positive")
        self.kinematics = ClosedChainKinematics(spec, manifest["tree_joint_names"],
                                               manifest["control_joint_names"], device, dtype)
        self.spec_canonical_sha256 = table["spec_canonical_sha256"]
        self.manifest_canonical_sha256 = table["manifest_canonical_sha256"]
        self.device, self.dtype = self.kinematics.device, dtype
        self.length_definition = table["length_definition"]
        self.length_tolerance_m, self.angle_tolerance_rad = length_tolerance_m, angle_tolerance_rad
        self.names = self.kinematics.tree_joint_names
        self._joints = {j["name"]: j for j in spec["joints"]}
        self._bodies = {b["name"]: b for b in spec["bodies"]}
        self._nominal = torch.tensor([manifest["nominal_joint_pos"][n] for n in self.names],
                                     dtype=dtype, device=self.device)
        self._side_data = []
        all_indices = []
        for row in table["sides"]:
            hip = self._joints[row["hip_joint"]]
            auxiliary = self._joints[row["auxiliary_joint"]]
            knee = self._joints[row["knee_joint"]]
            sign = float(row["knee_coordinate_sign"])
            expected_zero = (math.radians(manifest["knee_inner_limits_deg"][0])
                - sign * float(knee["limit"]["lower" if sign == 1. else "upper"]))
            if (sign not in (-1., 1.) or not math.isclose(row["knee_zero_rad"], expected_zero, abs_tol=1e-12, rel_tol=0.)
                    or not {row["hip_joint"], row["auxiliary_joint"], row["wheel_joint"]}
                           <= set(manifest["control_joint_names"])
                    or row["knee_joint"] in manifest["control_joint_names"]
                    or self._joints[row["wheel_joint"]]["child"] != row["wheel_body"]):
                raise ValueError("Named leg coordinate roles or knee zero differ from the asset")
            if hip["parent"] != self.kinematics.root_name or auxiliary["parent"] != self.kinematics.root_name:
                raise ValueError("Leg mapper requires named hip/auxiliary axes on the base")
            origin = torch.tensor(hip["origin"], dtype=dtype, device=self.device)
            raw_axis = origin[:3, :3] @ torch.tensor(hip["axis"], dtype=dtype, device=self.device)
            norm, axis = raw_axis.norm(), raw_axis / raw_axis.norm()
            down = torch.tensor([0., 0., -1.], dtype=dtype, device=self.device)
            down = down - axis * torch.dot(down, axis)
            if down.norm() < .5:
                raise ValueError("Hip axis does not define a sagittal leg plane")
            down /= down.norm()
            forward = torch.linalg.cross(axis, down)
            if forward[0] < 0:
                forward = -forward
            handedness = torch.dot(axis, torch.linalg.cross(down, forward))
            collisions = self._bodies[row["wheel_body"]]["collisions"]
            if len(collisions) != 1 or collisions[0]["type"] != "cylinder":
                raise ValueError("Explicit actual wheel-center geometry is required")
            center = torch.tensor(collisions[0]["origin"], dtype=dtype, device=self.device)[:3, 3]
            descendants = {hip["child"], auxiliary["child"]}
            while True:
                grown = descendants | {j["child"] for j in spec["joints"] if j["parent"] in descendants}
                if grown == descendants:
                    break
                descendants = grown
            indices = [i for i, n in enumerate(self.names) if self._joints[n]["child"] in descendants]
            all_indices += indices
            lengths = torch.tensor(row["length_m"], dtype=dtype, device=self.device)
            q = torch.tensor(row["full_q"], dtype=dtype, device=self.device)
            if (lengths.ndim != 1 or len(lengths) < 3 or q.shape != (len(lengths), len(self.names))
                    or not bool(torch.isfinite(lengths).all() & torch.isfinite(q).all())
                    or not bool((lengths[1:] > lengths[:-1]).all()) or not bool((lengths > 0).all())
                    or not bool(self.kinematics._domain(q).all())
                    or not bool((self.kinematics._norm(self.kinematics.residual(q)) <= 1e-8).all())):
                raise ValueError("Invalid full-tree reference branch")
            self._side_data.append({"name": row["name"], "hip": self.names.index(row["hip_joint"]),
                "auxiliary": self.names.index(row["auxiliary_joint"]),
                "wheel": self.names.index(row["wheel_joint"]), "wheel_body": row["wheel_body"],
                "knee": self.names.index(row["knee_joint"]),
                "knee_sign": float(row["knee_coordinate_sign"]), "knee_zero": float(row["knee_zero_rad"]),
                "axis": axis, "axis_norm": norm, "pivot": origin[:3, 3], "down": down,
                "forward": forward, "handedness": handedness, "center": center,
                "indices": torch.tensor(indices, device=self.device), "lengths": lengths, "q": q})
        if ([s["name"] for s in self._side_data] != ["left", "right"]
                or sorted(all_indices) != list(range(len(self.names)))):
            raise ValueError("Exactly two independent named leg subtrees must cover the tree")
        self.wheel_indices = torch.tensor([s["wheel"] for s in self._side_data], device=self.device)
        self._free = torch.tensor([i for i in range(len(self.names)) if i not in self.wheel_indices], device=self.device)
        self._safe_matrix = torch.eye(self.kinematics.residual_count + 4, len(self._free), dtype=dtype, device=self.device)
        self.length_bounds_m = torch.stack([s["lengths"][[0, -1]] for s in self._side_data])
        physical = manifest["knee_inner_limits_deg"]
        training = table["training_knee_domain_deg"]
        if (len(training) != 2 or not all(math.isfinite(v) for v in training)
                or not physical[0] <= training[0] < training[1] <= physical[1]):
            raise ValueError("Training domain must be within the asset physical domain")
        self.training_knee_domain_deg = torch.tensor(training, dtype=dtype, device=self.device)
        # Verify the table's length labels rather than trusting its advertised range.
        for side_index, side in enumerate(self._side_data):
            metrics = self.metrics(side["q"])
            if not torch.allclose(metrics["length_m"][:, side_index], side["lengths"], atol=1e-10, rtol=0.):
                raise ValueError("Reference length labels differ from actual sealed geometry")
            beta = metrics["knee_inner_deg"][:, side_index]
            declared_beta = torch.tensor(table["sides"][side_index]["beta_deg"], dtype=dtype, device=self.device)
            if (declared_beta.shape != beta.shape
                    or not torch.allclose(beta, declared_beta, atol=1e-8, rtol=0.)
                    or not torch.allclose(beta[[0, -1]], self.training_knee_domain_deg, atol=1e-8, rtol=0.)):
                raise ValueError("Reference branch does not span the declared training knee domain")

    def metrics(self, q):
        frames = self.kinematics.frames(q)
        centers, vectors, lengths, angles, betas = [], [], [], [], []
        for side in self._side_data:
            frame = frames[side["wheel_body"]]
            center = frame[:, :3, :3] @ side["center"] + frame[:, :3, 3]
            vector = center - side["pivot"]
            projected = vector - (vector @ side["axis"])[:, None] * side["axis"]
            length_vector = projected if self.length_definition == "hip_axis_projected" else vector
            centers.append(center); vectors.append(vector); lengths.append(length_vector.norm(dim=-1))
            angles.append(torch.atan2(vector @ side["forward"], vector @ side["down"]))
            betas.append((side["knee_sign"] * q[:, side["knee"]] + side["knee_zero"]) * (180. / math.pi))
        return {"length_m": torch.stack(lengths, -1), "angle_rad": torch.stack(angles, -1),
                "wheel_center_b_m": torch.stack(centers, 1), "hip_to_wheel_b_m": torch.stack(vectors, 1),
                "knee_inner_deg": torch.stack(betas, -1)}

    def _task_jacobian(self, q, metrics, target_length):
        frames = self.kinematics.frames(q)
        origins, axes, prism = [], [], []
        for name in self.names:
            joint = self._joints[name]
            before = frames[joint["parent"]] @ q.new_tensor(joint["origin"])
            origins.append(before[:, :3, 3])
            axes.append(before[:, :3, :3] @ q.new_tensor(joint["axis"]))
            prism.append(joint["type"] == "prismatic")
        origins, axes = torch.stack(origins, 1), torch.stack(axes, 1)
        prism = torch.tensor(prism, device=self.device)
        rows = []
        for i, side in enumerate(self._side_data):
            ancestors, body = set(), side["wheel_body"]
            by_child = {j["child"]: j for j in self._joints.values()}
            while body in by_child:
                joint = by_child[body]; ancestors.add(joint["name"]); body = joint["parent"]
            mask = torch.tensor([n in ancestors for n in self.names], device=self.device)
            point = metrics["wheel_center_b_m"][:, i]
            point_j = torch.linalg.cross(axes, point[:, None] - origins)
            point_j = torch.where(prism[None, :, None], axes, point_j)
            point_j = torch.where(mask[None, :, None], point_j, 0.)
            vector = metrics["hip_to_wheel_b_m"][:, i]
            gradient = vector
            if self.length_definition == "hip_axis_projected":
                gradient = vector - (vector @ side["axis"])[:, None] * side["axis"]
            gradient = gradient / metrics["length_m"][:, i, None]
            forward, down = vector @ side["forward"], vector @ side["down"]
            angle_gradient = (down[:, None] * side["forward"] - forward[:, None] * side["down"])
            angle_gradient /= (forward.square() + down.square())[:, None]
            rows.extend([(point_j * gradient[:, None]).sum(-1),
                         (point_j * angle_gradient[:, None]).sum(-1) * target_length[:, i, None]])
        return torch.stack(rows, 1)

    @staticmethod
    def _angle_error(actual, requested):
        return torch.atan2((actual - requested).sin(), (actual - requested).cos())

    def _training_domain(self, metrics):
        beta = metrics["knee_inner_deg"]
        # Coordinate numerical tolerance expressed in degrees, not a new domain.
        tolerance = math.degrees(self.kinematics.limit_tolerance)
        return ((beta >= self.training_knee_domain_deg[0] - tolerance)
                & (beta <= self.training_knee_domain_deg[1] + tolerance)).all(-1)

    def map_requests(self, length_m, angle_rad, wheel_q, control_dq, *, max_iterations=8):
        for value, width, name in ((length_m, 2, "length_m"), (angle_rad, 2, "angle_rad"),
                (wheel_q, 2, "wheel_q"), (control_dq, len(self.kinematics.control_indices), "control_dq")):
            self.kinematics._check(value, width, name)
        if len({len(x) for x in (length_m, angle_rad, wheel_q, control_dq)}) != 1:
            raise ValueError("Request tensors must have equal lane counts")
        if type(max_iterations) is not int or max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        count = len(length_m)
        finite = torch.isfinite(torch.cat((length_m, angle_rad, wheel_q, control_dq), -1)).all(-1)
        reachable = ((length_m >= self.length_bounds_m[:, 0]) & (length_m <= self.length_bounds_m[:, 1])).all(-1)
        accepted_input = finite & reachable
        # Internal algebra placeholders are discarded, not returned as fallback poses.
        target_length = torch.where(accepted_input[:, None], length_m, self.length_bounds_m.mean(-1))
        target_angle = torch.where(accepted_input[:, None], angle_rad, 0.)
        q = self._nominal.expand(count, -1).clone()
        for i, side in enumerate(self._side_data):
            index = torch.searchsorted(side["lengths"], target_length[:, i].contiguous()).clamp(1, len(side["lengths"]) - 1)
            weight = ((target_length[:, i] - side["lengths"][index - 1])
                      / (side["lengths"][index] - side["lengths"][index - 1]))
            seed = side["q"][index - 1] + weight[:, None] * (side["q"][index] - side["q"][index - 1])
            q[:, side["indices"]] = seed[:, side["indices"]]
        reference = self.metrics(q)
        for i, side in enumerate(self._side_data):
            shift = side["handedness"] * (target_angle[:, i] - reference["angle_rad"][:, i]) / side["axis_norm"]
            q[:, side["hip"]] += shift
            q[:, side["auxiliary"]] += shift
        q[:, self.wheel_indices] = torch.where(accepted_input[:, None], wheel_q, 0.)
        # Correct task geometry and closure together. A passive-only correction
        # at an exact knee endpoint can move that knee outward before the length
        # equation has a chance to restore it; it is not an admissible seed step.
        seed_domain = self.kinematics._domain(q)
        eligible = accepted_input & seed_domain
        singular_seen = torch.zeros(count, dtype=torch.bool, device=self.device)
        for _ in range(max_iterations):
            metrics = self.metrics(q)
            closure = self.kinematics.residual(q)
            length_error = metrics["length_m"] - target_length
            angle_error = self._angle_error(metrics["angle_rad"], target_angle)
            tasks = torch.stack((length_error, angle_error * target_length), -1).reshape(count, 4)
            residual = torch.cat((closure, tasks), -1)
            jacobian = torch.cat((self.kinematics.jacobian(q),
                                  self._task_jacobian(q, metrics, target_length)), 1)[:, :, self._free]
            matrix_finite = torch.isfinite(jacobian).all(dim=(-2, -1)) & torch.isfinite(residual).all(-1)
            matrix = torch.where(matrix_finite[:, None, None], jacobian, self._safe_matrix)
            singular = torch.linalg.svdvals(matrix)
            rank = matrix_finite & (singular[:, -1] > self.kinematics.rank_rtol * singular[:, 0])
            singular_seen |= eligible & ~rank
            eligible &= rank
            matrix = torch.where(rank[:, None, None], matrix, self._safe_matrix)
            rhs = torch.where(rank[:, None], -residual, 0.)
            orthogonal, triangular = torch.linalg.qr(matrix, mode="reduced")
            delta = torch.linalg.solve_triangular(triangular,
                orthogonal.transpose(-1, -2) @ rhs[:, :, None], upper=True).squeeze(-1)
            converged = ((length_error.abs() <= self.length_tolerance_m).all(-1)
                & (angle_error.abs() <= self.angle_tolerance_rad).all(-1)
                & (self.kinematics._norm(closure) <= self.kinematics.position_tolerance_m))
            candidate = q.clone(); candidate[:, self._free] += delta
            q = torch.where((eligible & ~converged)[:, None], candidate, q)
            eligible &= self.kinematics._domain(q)
            # Eligibility only shrinks. Once every eligible lane is converged,
            # later iterations leave q and all diagnostic masks unchanged.
            if not bool((eligible & ~converged).any()):
                break
        final_closure = self.kinematics.solve_passive(q, q[:, self.kinematics.control_indices])
        q = final_closure.q
        eligible &= final_closure.valid
        metrics = self.metrics(q)
        length_error = metrics["length_m"] - target_length
        angle_error = self._angle_error(metrics["angle_rad"], target_angle)
        closure = self.kinematics.residual(q)
        position_domain = self.kinematics._domain(q) & self._training_domain(metrics)
        geometry_ok = ((length_error.abs() <= self.length_tolerance_m).all(-1)
            & (angle_error.abs() <= self.angle_tolerance_rad).all(-1)
            & (self.kinematics._norm(closure) <= self.kinematics.position_tolerance_m))
        velocity = self.kinematics.compatible_velocities(q, control_dq)
        valid = eligible & position_domain & geometry_ok & velocity.valid
        reasons = {"nonfinite_request": ~finite, "unreachable_length": finite & ~reachable,
            "seed_domain_failed": accepted_input & ~seed_domain,
            "closure_solve_failed": accepted_input & ~final_closure.valid,
            "singular_mapping": singular_seen, "position_domain_failed": accepted_input & ~position_domain,
            "geometry_tolerance_failed": accepted_input & ~geometry_ok,
            "velocity_invalid": accepted_input & ~velocity.valid}
        # Geometric outputs for rejected input are explicitly unavailable; raw
        # requested wheel positions and all six active velocities remain visible.
        q = torch.where(accepted_input[:, None], q, torch.nan)
        q[:, self.wheel_indices] = wheel_q
        dq = torch.where(accepted_input[:, None], velocity.dq, torch.nan)
        dq[:, self.kinematics.control_indices] = control_dq
        actual = {key: torch.where(accepted_input.reshape(count, *([1] * (value.ndim - 1))), value, torch.nan)
                  for key, value in metrics.items()}
        actual.update({"length_error_m": torch.where(accepted_input[:, None], length_error, torch.nan),
            "angle_error_rad": torch.where(accepted_input[:, None], angle_error, torch.nan),
            "closure_residual_m": torch.where(accepted_input[:, None], closure, torch.nan),
            "closure_velocity_residual_m_s": torch.where(accepted_input[:, None], velocity.closure_velocity_residual, torch.nan)})
        return LegPoseResult(q, dq, valid, reasons,
            {"length_m": length_m.clone(), "angle_rad": angle_rad.clone(),
             "wheel_q": wheel_q.clone(), "control_dq": control_dq.clone()}, actual)
