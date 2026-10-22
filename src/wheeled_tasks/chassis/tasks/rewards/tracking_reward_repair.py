"""Versioned tracking payments; historical contracts retain their exact costs."""

KEY = "formal_tracking_reward_repair"
VERSION = "v7_exact_command_tracking_v1"
CONTINUOUS_COST_VERSION = "continuous_tracking_cost_v2"
SUPPORTED_VERSIONS = (VERSION, CONTINUOUS_COST_VERSION)
LINEAR_SQUARE_WEIGHT = -2.0


def validate(value):
    if value is not None and value not in SUPPORTED_VERSIONS:
        raise ValueError("Unknown tracking reward repair")
    return value


def validate_forward_migration(before, after):
    """Never relabel a saved version or silently restore an older objective."""
    if (before, after) not in (
            (None, VERSION), (None, CONTINUOUS_COST_VERSION),
            (VERSION, CONTINUOUS_COST_VERSION)):
        raise ValueError("Mixed tracking repair must be an explicit forward migration")


def pay(frame, *, rough, version=VERSION):
    """Keep physical terms and payment units, changing only reviewed tracking."""
    from wheeled_tasks.chassis.tasks.rewards.flat_reward import WEIGHTS, ROUGH_WEIGHTS, _source_reward

    if validate(version) is None:
        raise ValueError("Tracking payment requires an explicit repair version")
    weights = dict(ROUGH_WEIGHTS if rough else WEIGHTS)
    weights["track_lin_vel_xy_square"] = LINEAR_SQUARE_WEIGHT
    # Posture error must not make velocity errors cheaper by hiding the costs.
    return _source_reward(frame, weights=weights, velocity_height_gate=not rough,
                          exact_commands=True,
                          squared_tracking_height_gate=version == VERSION)
