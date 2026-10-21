"""Pure-Python validation for the complete one-way feedback engineering prior."""
from copy import deepcopy
import math


def validate_feedback_transport(settings, physics_dt=.005):
    """Validate sensor-to-host timing without importing torch or Isaac Lab.

    This prior describes the complete one-way feedback path. It is not an RTT
    measurement and adds no packet loss. Bounds and jitter are in seconds;
    jitter_s is each physics tick's uniform innovation half-width, not a bound
    on the accumulated deviation from the sampled base delay.
    """
    if not isinstance(settings, dict):
        raise ValueError("Feedback transport requires explicit settings")
    expected = {
        "version": "complete_one_way_feedback_v1",
        "units": "seconds",
        "scope": "engineering_prior_complete_sensor_to_host_path_not_measured_rtt",
        "ordered_delivery": "monotonic_sensor_timestamp",
        "reconstruction": "linear_physics_history",
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Feedback transport requires the complete one-way engineering prior in seconds")
    if type(settings.get("enabled")) is not bool:
        raise ValueError("Feedback transport requires an explicit boolean enable")
    required = {*expected, "enabled", "delay_range_s", "enabled_fraction", "jitter_s", "jitter_correlation"}
    if set(settings) != required:
        raise ValueError("Feedback transport settings contain missing or unsupported fields; packet loss is excluded")
    bounds = settings["delay_range_s"]
    numeric = (settings["enabled_fraction"], settings["jitter_s"], settings["jitter_correlation"], physics_dt)
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in (*bounds, *numeric))
            or not 0 <= bounds[0] <= bounds[1] <= .015
            or not 0 <= settings["enabled_fraction"] <= 1
            or settings["jitter_s"] < 0
            or not 0 <= settings["jitter_correlation"] <= 1
            or physics_dt != .005):
        raise ValueError("Invalid feedback transport bounds, jitter or 5 ms physics clock")
    return deepcopy(settings)
