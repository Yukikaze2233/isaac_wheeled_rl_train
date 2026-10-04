"""Command timing for interactive playback, without changing training semantics."""
import math

import torch

from .motion_limits import project_commands


def advance_playback_command(previous, requested, dt, slew, limits=None):
    """Apply the frozen command slope once before the actor sees its command.

    Requested sliders and keyboard targets may move instantly; applied vx/yaw
    obey the same SI rate limits as the loaded policy contract. The existing
    feasibility projection can reduce yaw near the lateral envelope. Back off
    the proposed increment there so that projection cannot break either rate
    bound. Height retains the GUI's existing separate command timing.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Playback requires a positive finite policy dt")
    rates = [slew["vx_m_s2"], slew["yaw_rad_s2"]]
    if any(not math.isfinite(value) or value <= 0 for value in rates):
        raise ValueError("Playback requires the loaded contract's positive command rates")
    if previous.shape != requested.shape or previous.ndim != 2 or previous.shape[1] != 3:
        raise ValueError("Playback commands must have matching [batch,3] shapes")
    if not bool(torch.isfinite(previous).all() and torch.isfinite(requested).all()):
        raise ValueError("Playback commands must be finite")
    target = project_commands(requested, limits) if limits else requested.clone()
    maximum = previous.new_tensor(rates) * dt
    delta = (target[:, :2] - previous[:, :2]).clamp(-maximum, maximum)
    scale = previous.new_ones((len(previous), 1))
    for _ in range(20):
        result = target.clone()
        result[:, :2] = previous[:, :2] + scale * delta
        if limits:
            result = project_commands(result, limits)
        valid = ((result[:, :2] - previous[:, :2]).abs() <= maximum + 1e-6).all(-1)
        if bool(valid.all()):
            return result
        scale[~valid] *= .5
    # A previously feasible command always supplies a bounded fallback.
    result[~valid, :2] = previous[~valid, :2]
    return result


def attitude_from_gravity(gravity):
    """Return roll/pitch in radians from body projected gravity near upright."""
    roll = math.atan2(-gravity[1], -gravity[2])
    pitch = math.atan2(gravity[0], math.hypot(gravity[1], gravity[2]))
    return roll, pitch
