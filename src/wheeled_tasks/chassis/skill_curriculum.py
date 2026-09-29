"""Materialize single-task SCUT-style recipes with immutable regression cases."""
from copy import deepcopy
import math
import zlib


SKILLS = {
    "stand": ("foundation", "flat", [0., 0., .305], 0),
    "height": ("foundation", "flat", [0., 0., .305], 0),
    "height_pulse": ("foundation", "flat", [0., 0., .305], 0),
    "forward": ("speed", "flat", [.5, 0., .305], 1),
    "backward": ("speed", "flat", [-.5, 0., .305], 1),
    "start_stop": ("speed", "flat", [.5, 0., .305], 1),
    "velocity_curve": ("speed", "flat", [.5, 0., .305], 1),
    "rotate": ("speed", "flat", [0., 1., .305], 1),
    "curve": ("speed", "flat", [.5, .6, .305], 1),
    "weave": ("speed", "flat", [3., .4, .305], 1),
    "spin_translate": ("speed", "flat", [0., 2., .305], 1),
    "push_recovery": ("foundation", "flat", [0., 0., .305], 0),
    "slope_up": ("terrain", "slope_up", [.4, 0., .305], 2),
    "slope_down": ("terrain", "slope_down", [.4, 0., .305], 3),
    "cross_slope": ("terrain", "cross_slope", [.4, 0., .305], 2),
    "rough": ("terrain", "rough", [.4, 0., .305], 2),
    "step_up": ("terrain", "step_up", [.4, 0., .305], 2),
    "step_down": ("terrain", "step_down", [.4, 0., .305], 3),
    "stairs": ("terrain", "stairs", [.4, 0., .305], 2),
    "stairs_down": ("terrain", "stairs_down", [.4, 0., .305], 3),
    "airborne": ("foundation", "flat", [0., 0., .305], 1),
    "landing": ("foundation", "flat", [0., 0., .305], 1),
    "jump": ("jump", "jump", [0., 0., .305], 4),
    "running_jump": ("jump", "jump", [.4, 0., .305], 4),
    "step_jump": ("jump", "step_up", [.4, 0., .305], 4),
    "double_step_jump": ("jump", "stairs", [.4, 0., .305], 4),
    "body_climb": ("terrain", "step_up", [.3, 0., .305], 0),
    "rolling_up": ("terrain", "step_up", [.3, 0., .305], 2),
    "rolling_stairs": ("terrain", "stairs", [.3, 0., .305], 2),
}


def step_jump_geometry(recipe):
    """Validate incremental rises and produce the terrain's absolute top levels."""
    params = recipe.get("step_jump")
    keys = {"rises_m", "treads_m", "landing_clearance_m", "launch_distance_m",
            "approach_speed_m_s", "landing_margin_m", "settle_seconds", "trigger_source"}
    if not isinstance(params, dict) or set(params) != keys:
        raise ValueError("step_jump requires the complete explicit task parameter schema")
    count = 2 if recipe["skill"] == "double_step_jump" else 1
    for key in ("rises_m", "treads_m"):
        if not isinstance(params[key], list) or len(params[key]) != count:
            raise ValueError(f"step_jump.{key} requires {count} platform values")
    values = params["rises_m"] + params["treads_m"] + [params[key] for key in keys
        - {"rises_m", "treads_m", "trigger_source"}]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("step_jump dimensions, approach speed and settling time must be finite positive")
    if params["trigger_source"] not in ("manual", "laser"):
        raise ValueError("step_jump trigger_source must be manual or laser")
    if any(tread <= 2 * params["landing_margin_m"] for tread in params["treads_m"]):
        raise ValueError("step_jump has no landing interior after margins")
    if count == 1:
        return {"step_up_m": params["rises_m"][0]}
    return {"stair_heights_m": [sum(params["rises_m"][:i + 1]) for i in range(count)],
            "stair_tread_m": list(params["treads_m"])}


def skill_spec(recipe):
    skill = recipe["skill"]
    _, terrain, default, mode = SKILLS[skill]
    command = list(recipe.get("command", default))
    result = {"kind": skill, "command": command, "mode": mode,
              "sample_amplitude": skill in ("forward", "backward", "rotate", "curve")}
    for key in ("height_range_m", "reference_velocity", "push_m_s", "drop_height_m",
                "reset_pitch_rad", "descent_speed_m_s", "reset_vx_m_s", "segment_seconds", "stop_settle_seconds",
                "height_motion", "drop_height_range_m", "height_sampling", "height_pulse", "terrain_level",
                "manual_request_seconds", "manual_hold_seconds", "velocity_transition_seconds", "acceleration_m_s2"):
        if key in recipe:
            result[key] = deepcopy(recipe[key])
    if skill in ("jump", "running_jump"):
        apex = recipe.get("jump_apex_delta_m", .06)
        result["semantics"] = {"jump_apex_delta_m": apex, "release_height_offset_m": .025,
            "jump_com_rise_m": max(.01, apex - .04), "jump_min_clearance_m": .01,
            "jump_forward_distance_m": .15 if skill == "running_jump" else 0.,
            "jump_landing_radius_m": 1.2 if skill == "running_jump" else .25,
             "jump_landing_speed_max_m_s": .8 if skill == "running_jump" else .1}
        if recipe.get("jump_apex_frame") == "com_release":
            result["semantics"]["jump_com_rise_m"] = max(.005, apex - .015)
    if skill in ("step_jump", "double_step_jump"):
        result["terrain_limits"] = step_jump_geometry(recipe)
        params = result["step_jump"] = deepcopy(recipe["step_jump"])
        apex = params["rises_m"][0] + params["landing_clearance_m"]
        result["semantics"] = {"jump_apex_delta_m": apex, "release_height_offset_m": .025,
            "jump_com_rise_m": max(.005, apex - .015), "jump_min_clearance_m": .01,
            "jump_forward_distance_m": params["launch_distance_m"] + params["treads_m"][0] / 2,
            "jump_landing_radius_m": params["treads_m"][0] / 2 - params["landing_margin_m"],
            "jump_landing_speed_max_m_s": .8}
    if skill == "body_climb":
        params = result["body_climb"] = deepcopy(recipe["body_climb"])
        rises, treads = params["rises_m"], params["treads_m"]
        if (len(rises) not in (1, 2) or len(treads) != len(rises)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in rises + treads)
                or "trigger_source" in params):
            raise ValueError("body_climb requires one/two positive platforms and no explicit trigger")
        terrain = "stairs" if len(rises) == 2 else "step_up"
        result["terrain_limits"] = ({"stair_heights_m": [sum(rises[:i + 1]) for i in range(len(rises))],
                                     "stair_tread_m": list(treads)} if len(rises) == 2 else {"step_up_m": rises[0]})
    return terrain, result


def skill_cases(base, recipe):
    terrain, spec = skill_spec(recipe)
    skill = recipe["skill"]
    task = "jump" if spec["mode"] == 4 else "traverse" if spec["mode"] in (2, 3) or skill == "body_climb" else "land" if skill in ("airborne", "landing") else "survive"
    limits = {**base["terrain_limits"], **recipe.get("terrain_limits", {}), **spec.get("terrain_limits", {})}
    case = {"name": recipe["name"], "command": spec["command"], "terrain": terrain,
            "task": task, "skill": spec, "terrain_limits": limits,
            "terrain_seed": zlib.crc32(recipe["name"].encode()),
            "episode_seconds": 15. if task == "traverse" or skill == "start_stop" else 10.,
            "height_mae_m_max": .02 if task == "traverse" else .01,
            "velocity_mae_m_s_max": max(.10, abs(spec["command"][0]) * .075),
            "yaw_mae_rad_s_max": max(.15, abs(spec["command"][1]) * .075)}
    if skill in ("step_jump", "double_step_jump"):
        case["step_jump_case"] = True
    if skill == "body_climb":
        case["body_climb_case"] = True
    if skill in ("rolling_up", "rolling_stairs"):
        case["untriggered_up_case"] = True
    if skill == "push_recovery":
        case.update(push_velocity_m_s=[recipe.get("push_m_s", .15), 0.], push_at_s=3.,
                    stationary=False)
    if skill == "spin_translate":
        case.update(reference_velocity_error_max=.10,
                    velocity_mae_m_s_max=.5, stationary=False)
    if skill in ("start_stop", "velocity_curve"):
        case.update(settled_stop_speed_max=.15, stationary=False, velocity_mae_m_s_max=.15)
    if skill == "weave":
        case["stationary"] = False
    if "episode_seconds" in recipe:
        case["episode_seconds"] = recipe["episode_seconds"]
    if "warmup_seconds" in recipe:
        case["warmup_seconds"] = recipe["warmup_seconds"]
    if recipe.get("evaluation_heights_m"):
        if skill != "stand":
            raise ValueError("Parallel height holds require a stationary skill")
        cases = []
        for height in recipe["evaluation_heights_m"]:
            hold = deepcopy(case)
            hold["name"] += f"_{round(height * 1000):03d}mm"
            hold["command"][2] = height
            hold["skill"]["command"] = list(hold["command"])
            hold["skill"].pop("height_sampling", None)
            hold["height_mae_m_max"] = .005
            cases.append(hold)
        return cases
    cases = [case]
    if skill in ("rotate", "curve", "spin_translate"):
        opposite = deepcopy(case)
        opposite["name"] += "_reverse"
        opposite["command"][1] *= -1
        opposite["skill"]["command"] = list(opposite["command"])
        cases.append(opposite)
    return cases


def configure_skill_contract(config, base, plan, recipe):
    """Preserve each predecessor's actual terrain dimensions and command profile."""
    prior = {}
    for item in plan["stages"]:
        if item["name"] == recipe["name"]:
            break
        if "skill" in item:
            prior[item["skill"]] = item
    cases = deepcopy(base["evaluation"]["cases"]) if plan.get("include_untrained_foundation_anchors", True) else []
    for case in cases:
        case.update(anchor=True, episode_seconds=base["evaluation"]["episode_seconds"])
    current_skill = recipe.get("skill")
    for skill, previous in prior.items():
        for case in skill_cases(base, previous):
            case["anchor"] = True
            cases.append(case)
    specs, groups = {}, []
    if current_skill:
        terrain, spec = skill_spec(recipe)
        spec["terrain_limits"] = {**base["terrain_limits"], **recipe.get("terrain_limits", {})}
        specs[recipe["name"]] = spec
        groups = [{"name": recipe["name"], "fraction": 1., "terrain": [terrain]}]
        cases.extend(skill_cases(base, recipe))
        config["terrain_limits"] = {**base["terrain_limits"], **recipe.get("terrain_limits", {})}
        rehearsal = plan.get("rehearsal_fraction", 0.)
        if not 0. <= rehearsal < 1.:
            raise ValueError("Rehearsal fraction must be in [0, 1)")
        if rehearsal and prior:
            groups[0]["fraction"] = 1. - rehearsal
            for previous in prior.values():
                previous_terrain, previous_spec = skill_spec(previous)
                previous_spec["terrain_limits"] = {**base["terrain_limits"], **previous.get("terrain_limits", {})}
                name = "rehearsal_" + previous["name"]
                specs[name] = previous_spec
                groups.append({"name": name, "fraction": rehearsal / len(prior), "terrain": [previous_terrain]})
        config["rehearsal_fraction"] = rehearsal if prior else 0.
    else:
        for previous in prior.values():
            terrain, spec = skill_spec(previous)
            spec["terrain_limits"] = {**base["terrain_limits"], **previous.get("terrain_limits", {})}
            specs[previous["name"]] = spec
            groups.append({"name": previous["name"], "fraction": 1. / len(prior), "terrain": [terrain]})
        # Mixed scenes use the hardest accepted dimensions for each terrain family.
        config["terrain_limits"] = dict(base["terrain_limits"])
        for previous in prior.values():
            for key, value in previous.get("terrain_limits", {}).items():
                config["terrain_limits"][key] = max(value, config["terrain_limits"][key])
        specs["surface_transfer"] = {"kind": "constant", "command": [.4, 0., .305], "mode": 2}
        groups.append({"name": "surface_transfer", "terrain": ["material"]})
        for group in groups:
            group["fraction"] = 1. / len(groups)
        cases.append({"name": "surface_transfer", "terrain": "material", "command": [.4, 0., .305],
                      "skill": specs["surface_transfer"], "task": "traverse", "episode_seconds": 15.,
                      "height_mae_m_max": .02, "velocity_mae_m_s_max": .15})
        if recipe.get("robust"):
            for case in deepcopy(cases):
                if case.get("skill"):
                    case.update(name=case["name"] + "_delayed", perturbed=True, anchor=False)
                    cases.append(case)
    config.update(skill_specs=specs, scene_groups=groups, episode_seconds=recipe.get("episode_seconds", 15.),
                  flat_floor_width_m=plan.get("flat_floor_width_m", 4.),
                  evaluation_long_corridors=True, height_range_m=[.305, .305],
                  position_iterations=64, velocity_iterations=32,
                  closure_gap_termination_m=.003,
                  curriculum="single_skill_then_consolidation_with_fixed_regression_gates")
    config["task_semantics"].update(success_hold_seconds=1., release_height_offset_m=.025,
                                    jump_com_rise_m=.02)
    if current_skill in ("jump", "running_jump"):
        config["task_semantics"].update(specs[recipe["name"]]["semantics"])
    config["stages"][0].update(terrain=list(dict.fromkeys(t for g in groups for t in g["terrain"])),
                                push_max=recipe.get("push_m_s", .15 if recipe.get("robust") else 0.))
    config["evaluation"].update(cases=cases, episode_seconds=15., protocol_id="v5-scut-skills-" + recipe["name"])
    config["training_reference"] = plan["training_reference"]
    config["upper_target_layout"] = ["route_distance_or_reference_vx", "platform_delta_or_reference_vy",
                                     "landing_distance", "jump_request"]
    return config
