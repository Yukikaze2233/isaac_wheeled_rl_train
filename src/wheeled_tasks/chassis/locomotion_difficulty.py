"""Expand training commands on the native domain actor clock, only at sampling."""
from copy import deepcopy
import math


def difficulty_settings(origin_actor_update):
    if type(origin_actor_update) is not int or origin_actor_update < 0:
        raise ValueError("Locomotion difficulty needs a nonnegative frozen actor origin")
    return dict(schema="locomotion-difficulty-v1", enabled=True,
        clock="domain_actor_updates", origin_actor_update=origin_actor_update,
        interpolation="linear", apply_at="command_sampling",
        scope="moving_commands_only_static_heights_unchanged",
        exact_evaluation="bypass", keyframes=[
            dict(actor_updates=0, vx_max_m_s=2., yaw_max_rad_s=math.tau,
                 motion_height_range_m=[.28, .35]),
            dict(actor_updates=120, vx_max_m_s=3., yaw_max_rad_s=1.625 * math.tau,
                 motion_height_range_m=[.25, .38]),
            dict(actor_updates=240, vx_max_m_s=4., yaw_max_rad_s=2.25 * math.tau,
                 motion_height_range_m=[.24, .40]),
            dict(actor_updates=360, vx_max_m_s=5., yaw_max_rad_s=3. * math.tau,
                 motion_height_range_m=[.23, .43])])


def validate_difficulty(settings):
    """Explicit frozen semantics: a missing/disabled field cannot default silently."""
    if settings is None:
        return None
    if (not isinstance(settings, dict) or settings.get("enabled") is not True
            or settings != difficulty_settings(settings.get("origin_actor_update"))):
        raise ValueError("Locomotion difficulty differs from its explicit frozen rule")
    return deepcopy(settings)


def difficulty_envelope(settings, actor_update):
    if type(actor_update) is not int or actor_update < 0:
        raise ValueError("Locomotion difficulty uses the nonnegative domain actor clock")
    elapsed = max(0, actor_update - settings["origin_actor_update"])
    nodes = settings["keyframes"]
    for index, (left, right) in enumerate(zip(nodes, nodes[1:])):
        if elapsed < right["actor_updates"]:
            fraction = max(0., (elapsed - left["actor_updates"]) /
                           (right["actor_updates"] - left["actor_updates"]))
            return dict(grade=1. + index + fraction,
                vx_max_m_s=left["vx_max_m_s"] + fraction * (right["vx_max_m_s"] - left["vx_max_m_s"]),
                yaw_max_rad_s=left["yaw_max_rad_s"] + fraction * (right["yaw_max_rad_s"] - left["yaw_max_rad_s"]),
                motion_height_range_m=[a + fraction * (b - a) for a, b in
                    zip(left["motion_height_range_m"], right["motion_height_range_m"])])
    return dict(grade=float(len(nodes)), **{key: deepcopy(value) for key, value in nodes[-1].items()
        if key != "actor_updates"})


def moving_spec(spec):
    return any(spec.get("command", [0., 0.])[:2]) or spec.get("kind") == "spin_translate"


def bounded_height_sampling(spec, envelope):
    settings = spec["height_sampling"]
    if envelope is None or not moving_spec(spec):
        return settings
    low, high = settings["range_m"]
    bounds = envelope["motion_height_range_m"]
    result = dict(settings, range_m=[max(low, bounds[0]), min(high, bounds[1])])
    if not result["range_m"][0] < result["range_m"][1]:
        raise ValueError("Moving height sampler and difficulty envelope do not overlap")
    return result


def scale_planar_command(command, spec, envelope):
    """Scale the declared distribution; clipping would pile samples at the cap."""
    if envelope is None or not moving_spec(spec):
        return command
    joint = spec.get("planar_command_sampling", {})
    for axis, key, limit in ((0, "vx_range_m_s", envelope["vx_max_m_s"]),
                             (1, "yaw_range_rad_s", envelope["yaw_max_rad_s"])):
        extent = max(abs(spec["command"][axis]), *(abs(x) for x in joint.get(key, [0.])))
        if extent > limit:
            command[:, axis] *= limit / extent
    return command
