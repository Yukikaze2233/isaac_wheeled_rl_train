"""Startup-only support geometry for native terrain training route rows.

The original terrain segments remain intact. This module does not change task
completion, commands, rewards, negative/lateral boundaries, or exact exams.
"""
from dataclasses import replace
import math

from .task import Surface


ROUTE_TERRAIN_KINDS = frozenset(("step_up", "step_down", "stairs", "stairs_down",
    "slope_up", "slope_down", "cross_slope", "rough", "material"))
_SETTING_KEYS = frozenset(("enabled", "support_end_x_m", "positive_boundary_x_m"))


def _finite_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Terrain route runout {name} must be a finite number")
    return float(value)


def _settings(settings):
    if settings is None:
        return None
    if not isinstance(settings, dict) or type(settings.get("enabled")) is not bool:
        raise ValueError("Terrain route runout enabled must be an explicit bool")
    if not settings["enabled"]:
        return None
    if set(settings) - _SETTING_KEYS:
        raise ValueError("Unknown terrain route runout setting")
    end = _finite_number(settings.get("support_end_x_m", 6.2), "support_end_x_m")
    boundary = _finite_number(settings.get("positive_boundary_x_m", 5.6),
                              "positive_boundary_x_m")
    if end <= 4. or boundary <= 3.6 or end - boundary < .6 - 1e-9:
        raise ValueError("Terrain route runout must extend the 4m support and 3.6m "
                         "positive boundary with at least 0.6m support padding")
    return dict(enabled=True, support_end_x_m=end, positive_boundary_x_m=boundary)


def validate_runout(config):
    """Return enabled settings only for terrain training in a separated layout.

    The long-corridor spacing below is the formula used by ChassisEnv. Checking
    static extents prevents a runout from accidentally joining adjacent scenes.
    Disabled configurations and exact evaluations preserve the original geometry.
    """
    if (config.get("evaluation_exact_cases")
            or config.get("task_campaign", {}).get("domain") != "terrain"):
        return None
    settings = _settings(config.get("terrain_route_runout"))
    if settings is None:
        return None
    if config.get("evaluation_long_corridors") is not True:
        raise ValueError("Terrain route runout requires the long-corridor layout")
    half = _finite_number(config.get("flat_half_length_m", 44.), "flat_half_length_m")
    width = _finite_number(config.get("flat_floor_width_m", 4.), "flat_floor_width_m")
    if half <= 0. or not 4. <= width <= 8.:
        raise ValueError("Terrain route runout requires positive corridor length "
                         "and the validated 4-8m floor width")
    maximum_extent = max(half, 4., settings["support_end_x_m"])
    release = config.get("release_stop", {})
    if release.get("enabled"):
        release_half = _finite_number(release.get("terrain_half_length_m"),
                                      "release_stop.terrain_half_length_m")
        if release_half <= 0.:
            raise ValueError("Release-stop support extent must be positive")
        maximum_extent = max(maximum_extent, release_half)
    if 2 * half + 12. <= 2 * maximum_extent:
        raise ValueError("Terrain route runout support overlaps adjacent corridors")
    # Terrain boxes stay 4m wide; flat/jump boxes or meshes use floor width.
    if width + 1. <= max(width, 4.):
        raise ValueError("Terrain route runout support overlaps adjacent rows")
    return settings


def route_runout_selection(config, group, kind):
    """Select the frozen native route rows, independent of live robot state."""
    if validate_runout(config) is None or kind not in ROUTE_TERRAIN_KINDS:
        return False
    spec = config.get("skill_specs", {}).get(group, {})
    row = spec.get("terrain_training_row")
    return (isinstance(row, dict) and bool(row)
            and type(spec.get("mode")) is int and spec["mode"] in (2, 3)
            and spec.get("kind") != "body_climb" and "body_climb" not in spec)


def append_route_runout(surfaces, settings):
    """Append a matching short support segment; keep old Surface objects.

    Callers apply route_runout_selection before calling this function. The original
    terminal plane continues across the seam, including cross slope and friction.
    """
    settings = _settings(settings)
    if settings is None:
        return surfaces
    if (not surfaces or any(not isinstance(surface, Surface) for surface in surfaces)
            or not math.isclose(surfaces[-1].x1, 4., abs_tol=1e-9)):
        raise ValueError("Terrain route runout requires the original terminal x=4 support")
    last = surfaces[-1]
    values = (last.x0, last.x1, last.z0, last.slope, last.cross_slope, last.friction)
    if not all(math.isfinite(value) for value in values) or last.x0 >= last.x1:
        raise ValueError("Terrain route runout terminal surface must be finite and nonempty")
    if last.slope and last.cross_slope:
        raise ValueError("Compound runout slopes cannot be represented by collision boxes")
    extension = replace(last, x0=last.x1, x1=settings["support_end_x_m"],
                        z0=last.height(last.x1))
    return [*surfaces, extension]


def apply_route_boundary(local_x, x_limit, mask, settings):
    """The policy-step boundary path, also exercised by startup readback."""
    if mask is None:
        return x_limit
    import torch
    return torch.where(mask & (local_x >= 0.), settings["positive_boundary_x_m"], x_limit)
