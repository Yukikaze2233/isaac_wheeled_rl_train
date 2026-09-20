"""Materialize single-task SCUT-style recipes with immutable regression cases."""
from copy import deepcopy
import zlib


SKILLS = {
    "stand": ("foundation", "flat", [0., 0., .305], 0),
    "height": ("foundation", "flat", [0., 0., .305], 0),
    "forward": ("speed", "flat", [.5, 0., .305], 1),
    "backward": ("speed", "flat", [-.5, 0., .305], 1),
    "start_stop": ("speed", "flat", [.5, 0., .305], 1),
    "rotate": ("speed", "flat", [0., 1., .305], 1),
    "curve": ("speed", "flat", [.5, .6, .305], 1),
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
}


def skill_spec(recipe):
    skill = recipe["skill"]
    _, terrain, default, mode = SKILLS[skill]
    command = list(recipe.get("command", default))
    result = {"kind": skill, "command": command, "mode": mode,
              "sample_amplitude": skill in ("forward", "backward", "rotate", "curve")}
    for key in ("height_range_m", "reference_velocity", "push_m_s", "drop_height_m",
                "reset_pitch_rad", "descent_speed_m_s", "reset_vx_m_s"):
        if key in recipe:
            result[key] = deepcopy(recipe[key])
    if skill in ("jump", "running_jump"):
        apex = recipe.get("jump_apex_delta_m", .06)
        result["semantics"] = {"jump_apex_delta_m": apex, "release_height_offset_m": .025,
            "jump_com_rise_m": max(.01, apex - .04), "jump_min_clearance_m": .01,
            "jump_forward_distance_m": .15 if skill == "running_jump" else 0.,
            "jump_landing_radius_m": 1.2 if skill == "running_jump" else .25,
            "jump_landing_speed_max_m_s": .8 if skill == "running_jump" else .1}
    return terrain, result


def skill_cases(base, recipe):
    terrain, spec = skill_spec(recipe)
    skill = recipe["skill"]
    task = "jump" if spec["mode"] == 4 else "traverse" if spec["mode"] in (2, 3) else "land" if skill in ("airborne", "landing") else "survive"
    limits = {**base["terrain_limits"], **recipe.get("terrain_limits", {})}
    case = {"name": recipe["name"], "command": spec["command"], "terrain": terrain,
            "task": task, "skill": spec, "terrain_limits": limits,
            "terrain_seed": zlib.crc32(recipe["name"].encode()),
            "episode_seconds": 15. if task == "traverse" or skill == "start_stop" else 10.,
            "height_mae_m_max": .02 if task == "traverse" else .01,
            "velocity_mae_m_s_max": max(.10, abs(spec["command"][0]) * .075),
            "yaw_mae_rad_s_max": max(.15, abs(spec["command"][1]) * .075)}
    if skill == "push_recovery":
        case.update(push_velocity_m_s=[recipe.get("push_m_s", .15), 0.], push_at_s=3.,
                    stationary=False)
    if skill == "spin_translate":
        case.update(reference_velocity_error_max=.10,
                    velocity_mae_m_s_max=.5, stationary=False)
    if skill == "start_stop":
        case.update(settled_stop_speed_max=.15, stationary=False, velocity_mae_m_s_max=.15)
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
    cases = deepcopy(base["evaluation"]["cases"])
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
    config.update(skill_specs=specs, scene_groups=groups, episode_seconds=15.,
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
