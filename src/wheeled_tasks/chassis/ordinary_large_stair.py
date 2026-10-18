"""Opt-in static ordinary stairs, taught on the terrain component actor clock.

This is a local course candidate. It does not freeze a contract, admit a worker,
change a budget or produce jump requests. Geometry is built once by the native
terrain builder; only forward commands and assessment exposure are scheduled.
"""
from copy import deepcopy
import math

import torch

from .skill_curriculum import skill_spec


SCHEMA = "ordinary-large-stair-candidate-v1"
POOL = "ordinary_body_climb"
ROWS = (
    ("ordinary_climb_050mm", (.05,), (.8,), 1000, 1200),
    ("ordinary_climb_100mm", (.10,), (.8,), 1400, 1600),
    ("ordinary_climb_150mm", (.15,), (.8,), 1800, 2000),
    ("ordinary_climb_200mm", (.20,), (.8,), 2200, 2400),
    ("ordinary_double_050_062mm", (.05, .0625), (.8, .6), 2600, 2800),
    ("ordinary_double_100_125mm", (.10, .125), (.8, .6), 3000, 3200),
    ("ordinary_double_150_187mm", (.15, .1875), (.8, .6), 3400, 3600),
    ("ordinary_double_200_250mm", (.20, .25), (.8, .6), 3800, 4000),
)


def row_settings(index, *, origin_actor_update, actor_allowance):
    """Freeze a bounded clock; leave at least 65% of allowance at full difficulty.

    Allowance is prospective actor learning, not total updates including critic
    adaptation. A caller must derive it from the already debited account.
    """
    if (type(index) is not int or not 0 <= index < len(ROWS)
            or type(origin_actor_update) is not int or origin_actor_update < 0
            or type(actor_allowance) is not int or actor_allowance < 1000):
        raise ValueError("Ordinary stairs need a valid row, actor origin and >=1000 actor allowance")
    finish = min(4000, math.floor(.35 * actor_allowance))
    _, _, _, start, end = ROWS[index]
    return dict(schema=SCHEMA, enabled=True, clock="terrain_component_actor_updates",
        origin_actor_update=origin_actor_update, actor_allowance=actor_allowance,
        row_index=index, activation_start=math.ceil(start * finish / 4000),
        activation_end=math.ceil(end * finish / 4000))


def validate_row(spec):
    settings = spec.get("ordinary_large_stair_curriculum")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("Ordinary stair curriculum must be an explicit row setting")
    expected = row_settings(settings.get("row_index"),
        origin_actor_update=settings.get("origin_actor_update"),
        actor_allowance=settings.get("actor_allowance"))
    _, rises, treads, _, _ = ROWS[expected["row_index"]]
    conflicts = ("command_sequence", "command_curriculum", "planar_command_sampling",
                 "height_sampling", "height_motion", "height_transition_seconds",
                 "standing_command_probability", "reference_velocity")
    body = spec.get("body_climb", {})
    if (settings != expected or spec.get("kind") != "body_climb" or spec.get("mode") != 0
            or spec.get("command") != [.3, 0., .305] or spec.get("sample_amplitude", False)
            or any(key in spec for key in conflicts) or "trigger_source" in body
            or body.get("rises_m") != list(rises) or body.get("treads_m") != list(treads)):
        raise ValueError("Ordinary stair row has another command owner, trigger or geometry")
    return deepcopy(settings)


def activation_fraction(settings, actor_update):
    if type(actor_update) is not int or actor_update < settings["origin_actor_update"]:
        raise ValueError("Ordinary stair component actor clock regressed before its origin")
    elapsed = actor_update - settings["origin_actor_update"]
    return min(1., max(0., (elapsed - settings["activation_start"]) /
        (settings["activation_end"] - settings["activation_start"])))


def command_profile(elapsed, settings, actor_update):
    """Preteach approach/reverse/stop on the pad; active climbs use forward only."""
    fraction = activation_fraction(settings, actor_update)
    command = torch.zeros(len(elapsed), 3, device=elapsed.device, dtype=elapsed.dtype)
    command[:, 2] = .305
    if fraction == 0.:
        phase = elapsed.remainder(5.)
        command[:, 0] = torch.where(phase < 1., .1, torch.where(phase < 2., -.1, 0.))
    else:
        command[:, 0] = .1 + .2 * fraction
    return command, fraction > 0.


def sampling_candidate(config, *, enabled=False, origin_actor_update=0, actor_allowance=15000):
    """Copy a native terrain source; retain small stairs and every rehearsal row.

    The new 10% pool comes exclusively from surface_traversal (10% -> 2%)
    and surface_height_joint (4% -> 2%). No formal contract is written here.
    """
    if type(enabled) is not bool:
        raise ValueError("Ordinary large stairs require an explicit opt-in")
    result = deepcopy(config)
    if not enabled:
        return result
    pools, membership = result["behavior_pool_fractions"], result["behavior_pool_membership"]
    if (config.get("evaluation_exact_cases") or config["task_campaign"]["domain"] != "terrain"
            or not config.get("step_jump_task", {}).get("enabled") or POOL in pools
            or pools.get("untriggered_steps") != .42 or pools.get("surface_traversal") != .1
            or pools.get("surface_height_joint") != .04):
        raise ValueError("Ordinary stair candidate requires the unchanged native small-stair source")
    for group in result["scene_groups"]:
        pool = membership[group["name"]]
        if pool in ("surface_traversal", "surface_height_joint"):
            group["fraction"] *= .02 / pools[pool]
    pools.update(surface_traversal=.02, surface_height_joint=.02, ordinary_body_climb=.10)
    for index, (name, rises, treads, _, _) in enumerate(ROWS):
        _, spec = skill_spec(dict(skill="body_climb", body_climb=dict(
            rises_m=list(rises), treads_m=list(treads)), episode_seconds=32.))
        # Explicit heights/treads also use the native builder for single steps.
        spec.update(terrain_level=1., episode_seconds=32., terrain_limits=dict(
            stair_heights_m=[sum(rises[:i + 1]) for i in range(len(rises))],
            stair_tread_m=list(treads)), ordinary_large_stair_curriculum=row_settings(index,
                origin_actor_update=origin_actor_update, actor_allowance=actor_allowance))
        validate_row(spec)
        result["scene_groups"].append(dict(name=name, terrain=["stairs"], fraction=.10 / len(ROWS)))
        result["skill_specs"][name], membership[name] = spec, POOL
    if not math.isclose(math.fsum(row["fraction"] for row in result["scene_groups"]), 1., abs_tol=1e-12):
        raise ValueError("Ordinary stair candidate lost the native population quota")
    for pool, fraction in pools.items():
        actual = math.fsum(row["fraction"] for row in result["scene_groups"]
            if membership[row["name"]] == pool)
        if not math.isclose(actual, fraction, abs_tol=1e-12):
            raise ValueError("Ordinary stair pool quota differs: " + pool)
    return result
