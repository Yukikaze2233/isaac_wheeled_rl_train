#!/usr/bin/env python3
"""Record actual reward-function slices in TensorBoard, with explicit SI axes."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.full_curriculum import resolve_plan
from wheeled_tasks.chassis.rewards import StationaryAnchor, reward_terms
from wheeled_tasks.chassis.v5_control import V5Control


def terms(error, axis, width=None):
    n = len(error)
    velocity, omega, gravity, command = [error.new_zeros(n, 3) for _ in range(4)]
    motors, acceleration, torque, actions, previous, before_previous = [error.new_zeros(n, 6) for _ in range(6)]
    wheels = error.new_zeros(n, 2, 3)
    gravity[:, 2], command[:, 2] = -1., .305
    height = command[:, 2] + (error if axis == "height" else 0.)
    if axis == "velocity":
        velocity[:, 0] = error
    elif axis == "yaw":
        omega[:, 2] = error
    elif axis in ("pitch", "roll"):
        gravity[:, 0 if axis == "pitch" else 1] = error.sin()
        gravity[:, 2] = -error.cos()
    elif axis == "vertical_velocity":
        velocity[:, 2] = error
    elif axis == "horizontal_angular_velocity":
        omega[:, 0] = error
    elif axis == "fork":
        wheels[:, 0, 0] = error
    elif axis.startswith(("leg_", "wheel_")):
        side, quantity = axis.split("_", 1)
        indices = list(V5Control.LEGS if side == "leg" else V5Control.WHEELS)
        target = {"velocity": motors, "acceleration": acceleration, "torque": torque, "action": actions}[quantity]
        target[:, indices] = error[:, None]
    return reward_terms(velocity, omega, gravity, height, command,
        motors, acceleration, torque, actions, previous, before_previous, wheels, error.new_ones(n),
        torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool),
        error.new_full((n,), width) if width is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "contracts/v5_scut35_repair_v51.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(json.loads(args.plan.read_text()), load)
    cfg = plan["stages"][0]["performance_curriculum"]
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"scope": "reward_shape_audit_not_policy_gradient_or_convergence_proof",
               "height_axis": "error_mm", "velocity_axis": "error_mm_per_second", "height_levels": []}
    for level, (scale, width) in enumerate(zip(cfg["height_scales"], cfg["height_kernel_widths_m"])):
        error = torch.linspace(0., .15, 151, dtype=torch.float64, requires_grad=True)
        values = terms(error, "height", width)
        reward = scale * (values["track_height"] + values["height_square"])
        slope = torch.autograd.grad(reward.sum(), error)[0]
        with SummaryWriter(str(args.output / f"height_level_{level}")) as writer:
            for mm, density, derivative in zip(range(151), reward.tolist(), slope.tolist()):
                writer.add_scalar("RewardShape/height_reward_density", density, mm)
                writer.add_scalar("RewardShape/height_loss_from_target", scale - density, mm)
                writer.add_scalar("RewardShape/height_derivative_per_m", derivative, mm)
        summary["height_levels"].append({"scale": scale, "kernel_width_m": width,
            "near_target_quadratic_loss_coefficient_per_m2": scale * (1 / width**2 + 100),
            "loss_at_10mm_per_second": scale - reward[10].item(),
            "loss_at_50mm_per_second": scale - reward[50].item(),
            "loss_at_100mm_per_second": scale - reward[100].item(),
            "derivative_at_30mm_per_m": slope[30].item()})
    for axis, maximum, step, names in (("velocity", 1., .01, ("track_lin_vel", "lin_vel_square")),
                                     ("yaw", 2., .01, ("track_yaw", "yaw_square"))):
        error = torch.arange(0., maximum + step / 2, step, dtype=torch.float64)
        values = terms(error, axis)
        reward = values[names[0]] + values[names[1]]
        with SummaryWriter(str(args.output / axis)) as writer:
            for e, value in zip(error.tolist(), reward.tolist()):
                writer.add_scalar("RewardShape/" + axis + "_tracking_loss", 1 - value, round(e * 1000))
        if axis == "velocity":
            summary["velocity_tracking_loss_at_005m_s_per_second"] = 1 - reward[5].item()
    stationary = plan["stationary_tracking"]
    summary["stationary_velocity_loss_at_005m_s_per_second"] = stationary["velocity_scale"] * .05
    anchor = StationaryAnchor(301, "cpu", stationary["position_weight"], stationary["position_band_m"])
    active = torch.ones(301, dtype=torch.bool)
    positions = torch.zeros(301, 2)
    anchor.reward(positions, active, active)
    positions[:, 0] = torch.arange(301) / 1000
    residence = -anchor.reward(positions, active, active)
    summary["stationary_position_loss_at_50mm_per_second"] = residence[50].item()
    with SummaryWriter(str(args.output / "stationary_distance_mm")) as writer:
        for mm, value in enumerate(residence.tolist()):
            writer.add_scalar("RewardShape/stationary_position_loss", value, mm)
    # Each sweep excites all motors in the named family, with action history at zero.
    # These are state-space slices; achievable trajectories and PPO gradients differ.
    axes = (("pitch", .8, "mrad", 1000), ("roll", .8, "mrad", 1000),
            ("vertical_velocity", 2., "mm_s", 1000), ("horizontal_angular_velocity", 4., "mrad_s", 1000),
            ("leg_velocity", 10., "mrad_s", 1000), ("wheel_velocity", 100., "mrad_s", 1000),
            ("leg_acceleration", 1000., "rad_s2", 1), ("wheel_acceleration", 10000., "rad_s2", 1),
            ("leg_torque", 40., "mNm", 1000), ("wheel_torque", 3.84, "mNm", 1000),
            ("leg_action", 4., "milli_action", 1000), ("wheel_action", 4., "milli_action", 1000),
            ("fork", .15, "mm", 1000))
    for axis, maximum, unit, units_per_si in axes:
        error = torch.linspace(0., maximum, 101, dtype=torch.float64)
        values = terms(error, axis)
        with SummaryWriter(str(args.output / (axis + "_" + unit))) as writer:
            for name, value in values.items():
                if torch.equal(value, value[:1].expand_as(value)):
                    continue
                for x, density in zip(error.tolist(), value.tolist()):
                    writer.add_scalar("RewardShape/" + name + "_density", density, round(x * units_per_si))
    summary["regularizer_axes"] = {axis: unit for axis, _, unit, _ in axes}
    for item in summary["height_levels"]:
        item["height_10mm_to_velocity_005_tracking_loss_ratio"] = (
            item["loss_at_10mm_per_second"] / summary["velocity_tracking_loss_at_005m_s_per_second"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
