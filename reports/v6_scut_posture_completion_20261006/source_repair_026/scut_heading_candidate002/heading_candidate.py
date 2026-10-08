"""Unwired CPU-checkable adapter for SCUT's heading command equation.

Source: SCUTRobotLab b8ff79f3, commands.py:577–604 (MIT).
wrap convention: frozen inspected IsaacLab utils/math.py:100–122
(Copyright 2022–2026 Isaac Lab Project Developers, BSD-3-Clause).
The caller owns target/mask lifecycle and ordinary-row eligibility. This module
does not sample commands, own a manager, or access simulator state.
"""
import math

import torch


def resolve_heading_commands(commands, *, heading_target, heading_w,
                             is_heading_env, is_standing_env, special_mode_id,
                             ordinary_mask, heading_command=True,
                             stiffness=5., yaw_range=(-2 * math.pi, 2 * math.pi)):
    """Return native [vx, yaw, height] commands without changing protected rows.

    ``ordinary_mask`` must identify only the frozen 24 normal mapped groups.
    Explicit terrain bindings and dedicated/native-high rows are excluded.
    Absolute world heading is a generator input, never a new actor input.
    Output remains upstream of the native slew and feasibility transforms.
    """
    if not isinstance(commands, torch.Tensor) or commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError("native commands must be an N x 3 tensor")
    if not commands.is_floating_point():
        raise ValueError("native commands must have a floating dtype")
    size = (len(commands),)
    for name, value in (("heading_target", heading_target), ("heading_w", heading_w)):
        if (not isinstance(value, torch.Tensor) or value.shape != size or
                value.dtype != commands.dtype or value.device != commands.device):
            raise ValueError(name + " must match native commands' rows, dtype and device")
    for name, value in (("is_heading_env", is_heading_env),
                        ("is_standing_env", is_standing_env), ("ordinary_mask", ordinary_mask)):
        if (not isinstance(value, torch.Tensor) or value.shape != size or
                value.dtype != torch.bool or value.device != commands.device):
            raise ValueError(name + " must be a bool vector on the native commands' device")
    if (not isinstance(special_mode_id, torch.Tensor) or special_mode_id.shape != size or
            special_mode_id.dtype not in (torch.int32, torch.int64) or
            special_mode_id.device != commands.device):
        raise ValueError("special_mode_id must be an integer vector on the native commands' device")
    if (type(heading_command) is not bool or not math.isfinite(stiffness) or stiffness < 0 or
            len(yaw_range) != 2 or not all(math.isfinite(x) for x in yaw_range) or
            yaw_range[0] > yaw_range[1]):
        raise ValueError("invalid heading configuration")

    result = commands.clone()
    if heading_command:
        heading_ids = (ordinary_mask & is_heading_env & (special_mode_id < 0)).nonzero().flatten()
        angles = heading_target[heading_ids] - heading_w[heading_ids]
        # Exact inspected IsaacLab convention: positive odd pi -> +pi.
        wrapped = (angles + torch.pi) % (2 * torch.pi)
        error = torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)
        result[heading_ids, 1] = torch.clip(stiffness * error, min=yaw_range[0], max=yaw_range[1])

    # Source priority is standing last; native column 2 is height, not velocity.
    standing = ordinary_mask & is_standing_env
    result[standing, :2] = 0.
    return result
