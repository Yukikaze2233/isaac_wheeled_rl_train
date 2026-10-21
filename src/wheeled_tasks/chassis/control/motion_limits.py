"""Differential-drive command feasibility in physical units."""
import torch


def wheel_speeds(vx, yaw, radius=.06, track=.4373):
    return (vx - track * yaw / 2) / radius, (vx + track * yaw / 2) / radius


def validate_command(command, limits):
    left, right = wheel_speeds(command[0], command[1], limits["wheel_radius_m"], limits["track_width_m"])
    if max(abs(left), abs(right)) > limits["max_wheel_speed_rad_s"] + 1e-5:
        raise ValueError(f"Command exceeds differential wheel-speed envelope: {command}")
    if abs(command[0] * command[1]) > limits["lateral_acceleration_m_s2"] + 1e-5:
        raise ValueError(f"Command exceeds lateral acceleration envelope: {command}")


def project_commands(commands, limits):
    result = commands.clone()
    speed = result[:, 0].abs() + limits["track_width_m"] * result[:, 1].abs() / 2
    cap = limits["wheel_radius_m"] * limits["max_wheel_speed_rad_s"]
    scale = (cap / speed.clamp_min(1e-6)).clamp_max(1.)
    result[:, :2] *= scale[:, None]
    result[:, 1] *= (limits["lateral_acceleration_m_s2"] / (result[:, 0] * result[:, 1]).abs().clamp_min(1e-6)).clamp_max(1.)
    return result
