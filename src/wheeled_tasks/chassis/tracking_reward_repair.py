"""Explicit V7 velocity-target repair; historical contracts retain their payments."""

KEY = "formal_tracking_reward_repair"
VERSION = "v7_exact_command_tracking_v1"
LINEAR_SQUARE_WEIGHT = -2.0


def validate(value):
    if value is not None and value != VERSION:
        raise ValueError("Unknown V7 tracking reward repair")
    return value


def pay(frame, *, rough):
    """Keep physical terms and payment units, changing only reviewed tracking."""
    from .scut_flat_reward import WEIGHTS, ROUGH_WEIGHTS, _source_reward

    weights = dict(ROUGH_WEIGHTS if rough else WEIGHTS)
    weights["track_lin_vel_xy_square"] = LINEAR_SQUARE_WEIGHT
    return _source_reward(frame, weights=weights, velocity_height_gate=not rough,
                          exact_commands=True)
