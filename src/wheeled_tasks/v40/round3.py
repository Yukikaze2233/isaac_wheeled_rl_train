"""Round3-A command distribution and opt-in wheel diagnostics, not a new policy API.

The validator stays stdlib-only. Torch is imported only by tensor operations.
"""
from copy import deepcopy
import json
import math
from pathlib import Path


def is_round3_a(contract):
    return contract.get("round3", {}).get("stage") == "A"


def validate_round3(contract):
    """Permit only the reviewed A changes to the unchanged v2 control contract."""
    config = contract["round3"]
    keys = {"stage", "spin_probability", "height_endpoint_probability_each",
            "height_resample_seconds", "physics_material"}
    if not isinstance(config, dict) or set(config) != keys or config["stage"] != "A":
        raise ValueError("round3 requires exactly the supported stage A fields")

    def finite(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("round3 numeric fields must be finite, not bool")
        return value

    standing = finite(contract["commands"]["stages"]["locomotion"]["standing_probability"])
    spin = finite(config["spin_probability"])
    endpoint = finite(config["height_endpoint_probability_each"])
    if not (0 <= standing <= 1 and 0 <= spin <= 1 and standing + spin <= 1 and 0 <= endpoint <= .5):
        raise ValueError("round3 invalid probability or bucket probability sum")
    periods = config["height_resample_seconds"]
    if not isinstance(periods, list) or len(periods) != 2:
        raise ValueError("round3 height clock requires two bounds")
    low, high = map(finite, periods)
    dt = contract["timing"]["policy_dt"]
    if not 0 < low <= high <= contract["termination"]["episode_seconds"]:
        raise ValueError("round3 height clock outside episode bounds")
    if any(not math.isclose(x / dt, round(x / dt), rel_tol=0, abs_tol=1e-9) for x in periods):
        raise ValueError("round3 height clock must use whole policy ticks")
    material = config["physics_material"]
    expected_material = {"static_friction": .5, "dynamic_friction": .5, "restitution": 0.,
                         "friction_combine_mode": "average", "restitution_combine_mode": "average"}
    if not isinstance(material, dict) or set(material) != set(expected_material):
        raise ValueError("round3 material fields mismatch")
    for key in ("static_friction", "dynamic_friction", "restitution"):
        finite(material[key])
    if material != expected_material:
        raise ValueError("round3 A requires explicit nominal material; DR is not enabled")
    for stage in ("height", "locomotion"):
        bounds = contract["commands"]["stages"][stage]["height"]
        if not .29 <= finite(bounds[0]) < finite(bounds[1]) <= .32:
            raise ValueError("round3 A height domain must stay inside .29..32")
    weight = finite(contract["rewards"]["weights"]["zero_command_translation"])
    if weight != -1:
        raise ValueError("round3 A requires the single -1 zero-vx L1 translation term")

    # Comparison is at validation/startup, never in rollout tensor math. Prevent an
    # A-labelled file from silently changing PD, actions, noise, reward or failure rules.
    baseline = json.loads((Path(__file__).resolve().parents[3] / "contracts/own_v40_v2.json").read_text())
    normalized = deepcopy(contract)
    del normalized["round3"]
    for stage in ("height", "locomotion"):
        normalized["commands"]["stages"][stage]["height"] = baseline["commands"]["stages"][stage]["height"]
    normalized["commands"]["stages"]["locomotion"]["standing_probability"] = .1
    normalized["rewards"]["weights"]["zero_command_translation"] = 0.
    if normalized != baseline:
        raise ValueError("round3 A may not alter the remaining v2 control/reward contract")


class Round3Commands:
    """Per-env height clock, independent of the existing velocity command clock."""

    def __init__(self, contract, stage, num_envs, device):
        import torch
        self.config = contract["round3"]
        self.bounds = contract["commands"]["stages"][stage]
        self.mixed_velocity = stage == "locomotion"
        self.height_period = [round(x / contract["timing"]["policy_dt"])
                              for x in self.config["height_resample_seconds"]]
        self.height_ticks_left = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.velocity_bucket = torch.zeros_like(self.height_ticks_left)

    def reset(self, env_ids):
        self.height_ticks_left[env_ids] = 0

    def advance(self):
        # Called exactly once after the old-command reward; sampling happens in obs.
        self.height_ticks_left -= 1

    def sample_velocity(self, commands, env_ids):
        import torch
        for column, key in enumerate(("vx", "wz")):
            lo, hi = self.bounds[key]
            commands[env_ids, column] = lo + (hi - lo) * torch.rand(len(env_ids), device=commands.device)
        if self.mixed_velocity:
            draw = torch.rand(len(env_ids), device=commands.device)
            p_stand = self.bounds["standing_probability"]
            standing = draw < p_stand
            spin = (draw >= p_stand) & (draw < p_stand + self.config["spin_probability"])
            commands[env_ids[standing], :2] = 0.
            commands[env_ids[spin], 0] = 0.
            self.velocity_bucket[env_ids] = torch.where(standing, 0, torch.where(spin, 1, 2))

    def sample_height(self, commands, env_ids=None):
        import torch
        if env_ids is None:
            env_ids = (self.height_ticks_left <= 0).nonzero(as_tuple=False).flatten()
        else:
            env_ids = env_ids[self.height_ticks_left[env_ids] <= 0]
        if env_ids.numel() == 0:
            return
        lo, hi = self.bounds["height"]
        draw = torch.rand(len(env_ids), device=commands.device)
        continuous = lo + (hi - lo) * torch.rand(len(env_ids), device=commands.device)
        p = self.config["height_endpoint_probability_each"]
        commands[env_ids, 2] = torch.where(draw < p, lo, torch.where(draw < 2 * p, hi, continuous))
        low_tick, high_tick = self.height_period
        self.height_ticks_left[env_ids] = torch.randint(
            low_tick, high_tick + 1, (len(env_ids),), device=commands.device,
        )


def wheel_slip_proxy(link_pos, quat_xyzw, com_pos, com_velocity, net_force, ground_z):
    """Lowest-cylinder-point velocity [m/s], NOT measured ground-contact slip.

    Inputs use [N,2,3/4/6] world-frame wheel data in L/R order; total angular
    velocity is included. The net-force gate is only a contact candidate.
    """
    import torch
    n = link_pos.shape[0]
    for tensor, width in ((link_pos, 3), (quat_xyzw, 4), (com_pos, 3), (com_velocity, 6)):
        if tensor.shape != (n, 2, width):
            raise ValueError("wheel proxy requires N x 2 world-frame state tensors")
    if net_force.shape != (n, 2) or ground_z.shape != (n,):
        raise ValueError("wheel proxy force/ground shape mismatch")
    qxyz, qw = quat_xyzw[..., :3], quat_xyzw[..., 3:]
    local_axis = torch.zeros_like(link_pos)
    local_axis[..., 2] = 1.
    axis = local_axis + 2 * torch.cross(qxyz, torch.cross(qxyz, local_axis, dim=-1)
                                      + qw * local_axis, dim=-1)
    offset = link_pos.new_tensor([.020350000821053982, -.020350000821053982])
    center = link_pos + axis * offset[None, :, None]
    normal = torch.zeros_like(axis)
    normal[..., 2] = 1.
    radial = -normal + axis * axis[..., 2:3]
    radial_norm = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    point = center + .06 * radial / radial_norm.clamp_min(1e-8)
    point -= .0125 * torch.sign(axis[..., 2:3]) * axis
    lateral = axis - normal * axis[..., 2:3]
    lateral = lateral / torch.linalg.vector_norm(lateral, dim=-1, keepdim=True).clamp_min(1e-8)
    longitudinal = torch.cross(lateral, normal, dim=-1)
    velocity = com_velocity[..., :3] + torch.cross(com_velocity[..., 3:], point - com_pos, dim=-1)
    finite = (torch.isfinite(velocity).all(-1) & torch.isfinite(quat_xyzw).all(-1)
              & torch.isfinite(net_force))
    return {
        "semantics": "lowest_cylinder_point_proxy_not_ground_confirmed; stationary_flat_ground_assumed",
        "wheel_order": ["L_link3", "R_link3"], "quaternion_order": "xyzw",
        "wheel_link_pos_w_m": link_pos, "wheel_link_quat_xyzw": quat_xyzw,
        "wheel_com_pos_w_m": com_pos, "wheel_com_velocity_w_m_s_rad_s": com_velocity,
        "wheel_collision_center_w_m": center, "lowest_point_proxy_w_m": point,
        "lowest_point_clearance_proxy_m": point[..., 2] - ground_z[:, None],
        "point_velocity_proxy_w_m_s": velocity,
        "longitudinal_velocity_proxy_m_s": (velocity * longitudinal).sum(-1),
        "lateral_velocity_proxy_m_s": (velocity * lateral).sum(-1),
        "geometry_valid": finite & (radial_norm.squeeze(-1) > 1e-6),
        "wheel_net_force_n": net_force, "net_contact_candidate_not_ground": net_force > 1.,
        "ground_contact_confirmed": torch.zeros_like(net_force, dtype=torch.bool),
    }
