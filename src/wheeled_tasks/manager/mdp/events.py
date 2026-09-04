"""Domain-randomization events for the direct RL env (self-implemented).

Each function writes randomized parameters into the physics engine for the
given env_ids. Startup-mode events are applied once over all envs; reset-mode
events run for the envs being reset. Only Isaac Lab write APIs with stable
signatures are used (joint stiffness/damping/friction, root state, masses).

Not implemented yet (extension points):
COM offsets, per-body materials, actuator effort scale/bias/noise.
"""
import torch


def randomize_root_state(robot, env_ids: torch.Tensor, pose_range: dict, velocity_range: dict) -> None:
    """Randomize root pose/velocity around defaults. pose_range keys: roll/pitch/yaw (rad) or z (m)."""
    n = env_ids.numel()
    root_state = robot.data.default_root_state[env_ids].clone()

    if "roll" in pose_range or "pitch" in pose_range or "yaw" in pose_range:
        r = pose_range.get("roll", (0.0, 0.0))
        p = pose_range.get("pitch", (0.0, 0.0))
        y = pose_range.get("yaw", (0.0, 0.0))
        roll = _u(r, n, robot.device)
        pitch = _u(p, n, robot.device)
        yaw = _u(y, n, robot.device)
        root_state[:, 3:7] = _quat_from_euler_xyz(roll, pitch, yaw)
    if "z" in pose_range:
        root_state[:, 2] += _u(pose_range["z"], n, robot.device)

    lin = _u3(velocity_range.get("lin", (0.0, 0.0)), n, robot.device)
    ang = _u3(velocity_range.get("ang", (0.0, 0.0)), n, robot.device)
    root_state[:, 7:10] = lin
    root_state[:, 10:13] = ang

    robot.write_root_state_to_sim(root_state, env_ids=env_ids)


def randomize_body_mass(robot, env_ids: torch.Tensor, body_names: list[str], scale_range: tuple[float, float]) -> torch.Tensor:
    """Scale masses of the named bodies (keep total mass realistic: scale, don't offset).
    Returns the per-env sampled scale (N, 1) for privileged readout."""
    body_ids = robot.find_bodies(body_names)[0]
    masses = robot.root_physx_view.get_masses()[env_ids]  # (N, num_bodies)
    scale = _u(scale_range, (env_ids.numel(), 1), robot.device)
    masses[:, body_ids] = masses[:, body_ids] * scale
    robot.root_physx_view.set_masses(masses, env_ids)
    return scale


def randomize_actuator_gains(robot, env_ids: torch.Tensor, scale_range: tuple[float, float]) -> torch.Tensor:
    """Scale stiffness/damping of ALL actuated joints (defaults 0.75-1.25).
    Returns the per-env sampled scale (N, 1) for privileged readout."""
    n = env_ids.numel()
    scale = _u(scale_range, (n, 1), robot.device)
    stiffness = robot.data.default_joint_stiffness[env_ids] * scale
    damping = robot.data.default_joint_damping[env_ids] * scale
    robot.write_joint_stiffness_to_sim(stiffness, env_ids=env_ids)
    robot.write_joint_damping_to_sim(damping, env_ids=env_ids)
    return scale


def randomize_joint_friction(robot, env_ids: torch.Tensor, joint_names: list[str],
                             add_range: tuple[float, float]) -> torch.Tensor:
    """Add Coulomb friction to the named joints (Coulomb-friction randomization core).
    Returns the per-env sampled friction (N, len(joint_names)) for privileged readout."""
    joint_ids = robot.find_joints(joint_names)[0]
    n, m = env_ids.numel(), len(joint_ids)
    friction = robot.data.default_joint_friction_coeff[env_ids][:, joint_ids] + _u(add_range, (n, m), robot.device)
    robot.write_joint_friction_coefficient_to_sim(friction, env_ids=env_ids, joint_ids=joint_ids)
    return friction


def randomize_body_com(robot, env_ids: torch.Tensor, body_names: list[str],
                       offset_ranges: dict[str, tuple[float, float]]) -> None:
    """Offset the COM of the named bodies along x/y/z (defaults: base x±4cm, y/z±2cm)."""
    body_ids = robot.find_bodies(body_names)[0]
    coms = robot.root_physx_view.get_coms()[env_ids]  # (N, num_bodies, 7): pos(3)+quat(4)
    for axis, key in enumerate(("x", "y", "z")):
        if key in offset_ranges:
            lo, hi = offset_ranges[key]
            coms[:, body_ids, axis] += lo + (hi - lo) * torch.rand((env_ids.numel(), len(body_ids)), device=robot.device)
    robot.root_physx_view.set_coms(coms, env_ids)


def randomize_body_material(robot, env_ids: torch.Tensor, body_names: list[str],
                            static_friction: tuple[float, float], dynamic_friction: tuple[float, float],
                            restitution: tuple[float, float]) -> None:
    """Randomize rigid-body contact material (static/dynamic friction, restitution).

    Property layout follows Isaac Lab's randomize_rigid_body_material: the
    material tensor is (N, num_bodies, 3) in [static, dynamic, restitution]
    order; if your IsaacLab version shapes it differently, adjust here only.
    """
    body_ids = robot.find_bodies(body_names)[0]
    n, m = env_ids.numel(), len(body_ids)
    props = robot.root_physx_view.get_material_properties()[env_ids]
    props[:, body_ids, 0] = _u(static_friction, (n, m), robot.device)
    props[:, body_ids, 1] = _u(dynamic_friction, (n, m), robot.device)
    props[:, body_ids, 2] = _u(restitution, (n, m), robot.device)
    robot.root_physx_view.set_material_properties(props, env_ids)


def sample_effort_noise(num_envs: int, joint_count: int, scale_range: tuple[float, float],
                        bias_range: tuple[float, float], noise_std: tuple[float, float],
                        device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-env actuator output disturbance parameters actuator output disturbance (effort_noise:
    tau = clip(tau_nominal * scale + bias + N(0, std))). The env multiplies its
    commanded torques by these each step; resampled on reset."""
    scale = _u(scale_range, (num_envs, joint_count), device)
    bias = _u(bias_range, (num_envs, joint_count), device)
    std = _u(noise_std, (num_envs, joint_count), device)
    return scale, bias, std


def _u(range_pair: tuple[float, float], shape, device) -> torch.Tensor:
    lo, hi = range_pair
    return lo + (hi - lo) * torch.rand(shape, device=device)


def _u3(range_pair: tuple[float, float], n: int, device) -> torch.Tensor:
    """Per-axis uniform in all 3 dims."""
    lo, hi = range_pair
    return lo + (hi - lo) * torch.rand((n, 3), device=device)


def _quat_from_euler_xyz(roll, pitch, yaw) -> torch.Tensor:
    """ZYX intrinsic euler -> quaternion (w, x, y, z). Matches Isaac Lab convention."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)
