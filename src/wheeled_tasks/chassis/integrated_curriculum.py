"""SCUT-style simultaneous skill distributions with sample-count-based scheduling."""
from copy import deepcopy
import math

from .skill_curriculum import skill_cases, skill_spec


def integrated_contract(config, base, plan, recipe, num_envs):
    """Populate skill distributions and acceptance cases on a prepared contract."""
    if "performance_curriculum" not in recipe and plan.get("performance_curriculum"):
        recipe = {**recipe, "performance_curriculum": deepcopy(plan["performance_curriculum"])}
    catalog = {item["name"]: item for item in plan["skill_catalog"] if "skill" in item}
    prior = []
    for phase in plan["stages"]:
        if phase["name"] == recipe["name"]:
            break
        prior.extend(phase["new_skills"])
    added = recipe["new_skills"]
    names = prior + added
    if not names or len(set(names)) != len(names) or set(names) - catalog.keys():
        raise ValueError("Integrated phases require unique known skills")
    phase_names = [phase["name"] for phase in plan["stages"]]
    height_stage = plan.get("height_locomotion_stage")
    if height_stage and (height_stage not in phase_names or not plan.get("repair_sampling")
                         or not plan.get("locomotion_height_sampling")):
        raise ValueError("Separate height locomotion requires a known stage, repair sampling and a height domain")
    height_active = bool(height_stage and phase_names.index(recipe["name"]) >= phase_names.index(height_stage))
    introducing_height = recipe["name"] == height_stage
    transition_stage = plan.get("height_transition_stage")
    if transition_stage and (not height_stage or transition_stage not in phase_names
                             or phase_names.index(transition_stage) <= phase_names.index(height_stage)
                             or plan.get("cross_height_evaluation", {}).get("transition_seconds", 0.) <= 0):
        raise ValueError("Height transitions require a later stage and a positive reference duration")
    transition_active = bool(transition_stage and phase_names.index(recipe["name"]) >= phase_names.index(transition_stage))
    introducing_transition = recipe["name"] == transition_stage
    rehearsal_fraction = recipe.get("rehearsal_fraction", plan.get("rehearsal_fraction", .5))
    if not math.isfinite(rehearsal_fraction) or not 0 < rehearsal_fraction < 1:
        raise ValueError("Integrated rehearsal fraction must be in (0,1)")
    specs, groups, cases = {}, [], []
    height_endpoints = {}
    for name in names:
        item = catalog[name]
        terrain, spec = skill_spec(item)
        spec["terrain_limits"] = {**base["terrain_limits"], **item.get("terrain_limits", {})}
        spec["episode_seconds"] = item.get("episode_seconds", 20.)
        if name in recipe.get("command_curricula", {}):
            spec["command_curriculum"] = deepcopy(recipe["command_curricula"][name])
        specs[name] = spec
        resampling = plan.get("command_resampling_seconds")
        if resampling and spec["kind"] in ("forward", "backward", "rotate", "curve", "spin_translate"):
            if len(resampling) != 2 or not 0 < resampling[0] <= resampling[1] < float("inf"):
                raise ValueError("Command resampling requires a finite positive time range")
            spec["command_resampling_seconds"] = list(resampling)
        if (not height_stage and plan.get("locomotion_height_sampling") and spec["mode"] == 1
                and item["skill"] not in ("airborne", "landing")):
            spec["height_sampling"] = deepcopy(plan["locomotion_height_sampling"])
            spec["height_transition_seconds"] = plan["locomotion_height_sampling"]["transition_seconds"]
        if added and prior:
            fraction = (1. - rehearsal_fraction) / len(added) if name in added else rehearsal_fraction / len(prior)
        else:
            fraction = 1. / len(names)
        groups.append({"name": name, "fraction": fraction, "terrain": [terrain]})
        for case in skill_cases(base, item):
            case["anchor"] = name in prior
            if item["skill"] == "height":
                case["height_mae_m_max"] = .005
                if spec.get("height_motion"):
                    case["height_velocity_mae_m_s_max"] = .03
                for suffix, height in zip(("low", "high"), spec.get("height_range_m", [.29, .32])):
                    endpoint = deepcopy(case)
                    endpoint.update(name=name + "_hold_" + suffix, command=[0., 0., height])
                    endpoint["skill"] = {"kind": "stand", "mode": 0, "command": [0., 0., height]}
                    if spec.get("height_motion"):
                        endpoint["skill"].update(height_range_m=list(spec["height_range_m"]),
                            height_motion={**deepcopy(spec["height_motion"]), "endpoint": suffix})
                        endpoint.update(episode_seconds=18., warmup_seconds=spec["height_motion"]["transition_seconds"])
                    height_endpoints[endpoint["name"]] = name
                    cases.append(endpoint)
            cases.append(case)
    if plan.get("repair_sampling"):
        is_new = {name: name in added for name in specs}
        owners = {name: name for name in specs}
        for case in cases:
            if case["name"] not in height_endpoints:
                continue
            name = case["name"]
            specs[name] = {**deepcopy(case["skill"]), "episode_seconds": 20.,
                           "terrain_limits": deepcopy(case["terrain_limits"])}
            groups.append({"name": name, "terrain": ["flat"]})
            is_new[name] = is_new[height_endpoints[name]]
            owners[name] = height_endpoints[name]
        for group in groups[:]:
            name = group["name"]
            spec = specs[name]
            if spec["kind"] not in ("rotate", "spin_translate"):
                continue
            spec["sample_yaw_sign"] = False
            if recipe.get("performance_curriculum") and spec["kind"] == "rotate":
                spec["sample_amplitude"] = False
            reverse = deepcopy(spec)
            reverse["command"][1] *= -1
            if reverse.get("command_curriculum"):
                reverse["command_curriculum"]["initial"][1] *= -1
            key = name + "__reverse_train"
            specs[key] = reverse
            groups.append({"name": key, "terrain": list(group["terrain"])})
            is_new[key] = is_new[name]
            owners[key] = owners[name]
        nominal_groups = groups[:]
        height_distributions = []
        if height_active:
            height_distributions.append(("__height_train", introducing_height, plan["locomotion_height_sampling"]))
        if transition_active:
            height_distributions.append(("__height_transition_train", introducing_transition,
                {**plan["locomotion_height_sampling"], "transition_seconds": plan["cross_height_evaluation"]["transition_seconds"],
                 "start_at_target_fraction": 0.}))
        for suffix, introduced, sampling in height_distributions:
            # Keep nominal-height rehearsal as a distinct distribution. Enabling
            # a compound task must not silently redefine an accepted skill.
            for group in nominal_groups:
                name = group["name"]
                spec = specs[name]
                if spec["mode"] != 1 or spec["kind"] in ("airborne", "landing"):
                    continue
                key = name + suffix
                specs[key] = {**deepcopy(spec), "height_sampling": deepcopy(sampling),
                              "height_transition_seconds": sampling["transition_seconds"]}
                groups.append({"name": key, "terrain": list(group["terrain"])})
                is_new[key] = introduced
                owners[key] = key
        weights = recipe.get("sampling_weights", {})
        for name, spec in specs.items():
            if name in weights and spec["kind"] in ("forward", "backward"):
                spec["sample_amplitude"] = False
            if (recipe.get("performance_curriculum") or {}).get("kind") == "adaptive_commands":
                spec["sample_amplitude"] = False
        pools = ([(True, 1. - rehearsal_fraction), (False, rehearsal_fraction)]
                 if prior and (added or introducing_height or introducing_transition) else [(None, 1.)])
        focus = recipe.get("focus_skills")
        if focus:
            if added or introducing_height or introducing_transition or recipe.get("rehearsal_groups"):
                raise ValueError("Skill revisits cannot also introduce skills or override rehearsal groups")
            if len(set(focus)) != len(focus) or not set(focus) < set(owners.values()):
                raise ValueError("Skill revisits require known unique focus skills and a rehearsal remainder")
            is_new = {name: owner in focus for name, owner in owners.items()}
            pools = [(True, 1. - rehearsal_fraction), (False, rehearsal_fraction)]
        if recipe.get("rehearsal_groups"):
            rehearsal = set(recipe["rehearsal_groups"])
            fraction = recipe["rehearsal_fraction"]
            if not rehearsal < set(specs) or not 0 < fraction < 1:
                raise ValueError("Rehearsal requires known groups, a repair remainder and a fraction in (0,1)")
            is_new = {name: name not in rehearsal for name in specs}
            pools = [(True, 1. - fraction), (False, fraction)]
        for membership, fraction in pools:
            selected = [g for g in groups if membership is None or is_new[g["name"]] == membership]
            values = [weights.get(g["name"], weights.get("reverse_rotation", 1.)
                      if g["name"].endswith("__reverse_train") else weights.get("default", 1.)) for g in selected]
            if not values or min(values) <= 0:
                raise ValueError("Sampling pools require positive weights")
            for group, weight in zip(selected, values):
                group["fraction"] = fraction * weight / sum(values)
    cross_heights = plan.get("cross_height_evaluation", {}) if not height_stage or height_active else {}
    for case in cases[:]:
        if case["name"] not in cross_heights.get("cases", []):
            continue
        for height in cross_heights["height_m"]:
            cross = deepcopy(case)
            cross["name"] += f"_height_{round(height * 1000):03d}mm"
            cross["command"][2] = height
            cross["skill"]["command"] = list(cross["command"])
            cross["skill"]["height_transition_seconds"] = cross_heights["transition_seconds"]
            cross.update(episode_seconds=18., warmup_seconds=cross_heights["transition_seconds"],
                          height_velocity_mae_m_s_max=.03)
            if height_stage:
                cross["anchor"] = not introducing_height
            if transition_stage:
                hold = deepcopy(cross)
                hold["name"] += "_hold"
                hold["skill"]["height_transition_seconds"] = 0.
                hold.pop("height_velocity_mae_m_s_max")
                cases.append(hold)
                cross["anchor"] = not introducing_transition
            if not transition_stage or transition_active:
                cases.append(cross)
    if recipe.get("robust"):
        specs["surface_transfer"] = {"kind": "constant", "command": [.4, 0., .305], "mode": 2}
        for group in groups:
            group["fraction"] *= .95
        groups.append({"name": "surface_transfer", "fraction": .05, "terrain": ["material"]})
        cases.append({"name": "surface_transfer", "terrain": "material", "command": [.4, 0., .305],
                      "skill": specs["surface_transfer"], "task": "traverse", "episode_seconds": 15.,
                      "height_mae_m_max": .02, "velocity_mae_m_s_max": .15})
        for case in deepcopy(cases):
            case.update(name=case["name"] + "_perturbed", perturbed=True, anchor=False)
            cases.append(case)
    from .motion_limits import validate_command
    for case in cases:
        validate_command(case["command"], config["motion_limits"])
    anchors = set(recipe.get("evaluation_anchor_cases", []))
    if anchors:
        if anchors - {case["name"] for case in cases}:
            raise ValueError("Unknown protected evaluation case")
        for case in cases:
            case["anchor"] = case.get("anchor", False) or case["name"] in anchors
        config["evaluation"]["require_passing_anchors"] = True
    if plan.get("require_passing_anchors") and any(case.get("anchor") for case in cases):
        config["evaluation"]["require_passing_anchors"] = True
    config.update(skill_specs=specs, scene_groups=groups, episode_seconds=max(
        spec.get("episode_seconds", 20.) for spec in specs.values()),
        curriculum_reference_batch=plan["target_num_envs"] * plan["num_steps_per_env"],
        checkpoint_interval=plan.get("checkpoint_interval", 100), checkpoint_first_update=10,
        save_interval=500, checkpoint_snapshots=True, curriculum="integrated_on_policy_skills",
        terrain_difficulty_tiers=[.25, .5, .75, 1.] if recipe["kind"] in ("terrain", "jump", "mixed") else [1.])
    config["stages"][0]["terrain"] = list(dict.fromkeys(t for group in groups for t in group["terrain"]))
    config["evaluation"].update(cases=cases, episode_seconds=max(c["episode_seconds"] for c in cases),
        block_updates=plan["block_updates"], regression_patience=recipe.get("regression_patience", plan.get("regression_patience")),
        minimum_updates=math.ceil(recipe.get("minimum_updates", 0) * plan["target_num_envs"] / num_envs),
        skip_training_if_initially_accepted=recipe.get("skip_training_if_initially_accepted", False))
    if recipe["kind"] in ("terrain", "mixed"):
        config["scut_effort_reward_scale"] = .1
    if plan.get("stationary_tracking"):
        config["stationary_tracking"] = deepcopy(plan["stationary_tracking"])
    if "fall_confirmation_seconds" in plan:
        config["fall_confirmation_seconds"] = plan["fall_confirmation_seconds"]
    for key in ("height_workspace", "height_tracking", "landing_tracking", "special_mode_activation", "transfer_curriculum"):
        if key in plan:
            config[key] = deepcopy(plan[key])
    if plan.get("special_mode_activation"):
        for kind, activation in plan["special_mode_activation"].items():
            if (kind != "spin_translate" or any(not math.isfinite(value) or value < 0 for value in activation.values())
                    or activation["min_episode_seconds"] <= 0 or activation["height_error_m"] <= 0
                    or activation["gravity_xy_max"] <= 0):
                raise ValueError("Invalid spin-translation activation timing or stability bounds")
        if any(catalog[name]["skill"] == "spin_translate" for name in prior):
            config["special_mode_activation"]["spin_translate"]["start_reference_updates"] = 0.
    if plan.get("height_workspace"):
        heights = plan["height_workspace"]["height_m"]
        samplers = [plan.get("locomotion_height_sampling"), *(spec.get("height_sampling") for spec in specs.values())]
        for sampling in samplers:
            if sampling and not (heights[0] <= sampling["range_m"][0] < sampling["range_m"][1] <= heights[-1]
                                 and 0 <= sampling["nominal_fraction"] <= 1 and 0 <= sampling["endpoint_fraction"] <= 1
                                 and sampling["nominal_fraction"] + sampling["endpoint_fraction"] <= 1
                                 and 0 <= sampling.get("start_at_target_fraction", 0.) <= 1
                                 and math.isfinite(sampling.get("transition_seconds", 0.))
                                 and sampling.get("transition_seconds", 0.) >= 0):
                raise ValueError("Invalid full-range locomotion height sampling")
        for spec in specs.values():
            targets = spec.get("height_range_m", [spec["command"][2]])
            if not all(heights[0] <= h <= heights[-1] for h in targets):
                raise ValueError("Commanded height is outside the verified workspace")
        for case in cases:
            targets = case.get("skill", {}).get("height_range_m", [case["command"][2]])
            if not all(heights[0] <= h <= heights[-1] for h in targets):
                raise ValueError("Evaluation height is outside the verified workspace")
        config["height_range_m"] = [heights[0], heights[-1]]
    for key in ("learning_rate", "critic_warmup_updates", "transfer_critic", "transfer_noise_floor", "performance_curriculum"):
        if key in recipe:
            config[key] = deepcopy(recipe[key])
    return config
