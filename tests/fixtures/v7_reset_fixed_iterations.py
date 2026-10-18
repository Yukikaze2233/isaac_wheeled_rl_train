"""V7 reset numerical oracles from commit de64f54.

Methods are dedented and renamed for binding; executable statements are unchanged.
These fixed-iteration methods describe CPU geometry, not native physics.
"""
import torch

from wheeled_tasks.chassis.closed_chain_kinematics import PositionSolution
from wheeled_tasks.chassis.leg_pose_mapping import LegPoseResult


def fixed_iteration_solve_passive(self, initial_q, prescribed_control6, max_iterations=12):
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
    residual = self.residual(q)
    safe_q = torch.where(torch.isfinite(q).all(-1)[:, None], q, 0.)
    _, rank_valid = self._passive_solve(self.jacobian(safe_q),
        torch.zeros(len(q), self.residual_count, dtype=self.dtype, device=self.device))
    valid = eligible & rank_valid & self._domain(q) & (self._norm(residual) <= self.position_tolerance_m)
    return PositionSolution(q, valid, residual)


def fixed_iteration_map_requests(self, length_m, angle_rad, wheel_q, control_dq, *, max_iterations=8):
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
