"""SCUT-style simultaneous skill distributions with sample-count-based scheduling."""
from copy import deepcopy
import math

from .skill_curriculum import skill_cases, skill_spec


def integrated_contract(config, base, plan, recipe, num_envs):
    """Populate skill distributions and acceptance cases on a prepared contract."""
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
    specs, groups, cases = {}, [], []
    for name in names:
        item = catalog[name]
        terrain, spec = skill_spec(item)
        spec["terrain_limits"] = {**base["terrain_limits"], **item.get("terrain_limits", {})}
        spec["episode_seconds"] = item.get("episode_seconds", 20.)
        if name in recipe.get("command_curricula", {}):
            spec["command_curriculum"] = deepcopy(recipe["command_curricula"][name])
        specs[name] = spec
        if added and prior:
            fraction = .5 / len(added) if name in added else .5 / len(prior)
        else:
            fraction = 1. / len(names)
        groups.append({"name": name, "fraction": fraction, "terrain": [terrain]})
        for case in skill_cases(base, item):
            case["anchor"] = name in prior
            if item["skill"] == "height":
                case["height_mae_m_max"] = .005
                for suffix, height in (("low", .29), ("high", .32)):
                    endpoint = deepcopy(case)
                    endpoint.update(name=name + "_hold_" + suffix, command=[0., 0., height])
                    endpoint["skill"] = {"kind": "stand", "mode": 0, "command": [0., 0., height]}
                    cases.append(endpoint)
            cases.append(case)
    if plan.get("repair_sampling"):
        is_new = {name: name in added for name in specs}
        for case in cases:
            if case["name"] not in ("height_hold_low", "height_hold_high"):
                continue
            name = case["name"]
            specs[name] = {**deepcopy(case["skill"]), "episode_seconds": 20.,
                           "terrain_limits": deepcopy(case["terrain_limits"])}
            groups.append({"name": name, "terrain": ["flat"]})
            is_new[name] = not bool(prior)
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
        weights = recipe.get("sampling_weights", {})
        for name, spec in specs.items():
            if name in weights and spec["kind"] in ("forward", "backward"):
                spec["sample_amplitude"] = False
        pools = [(True, .5), (False, .5)] if prior and added else [(None, 1.)]
        for membership, fraction in pools:
            selected = [g for g in groups if membership is None or is_new[g["name"]] == membership]
            values = [weights.get(g["name"], weights.get("reverse_rotation", 1.)
                      if g["name"].endswith("__reverse_train") else weights.get("default", 1.)) for g in selected]
            if not values or min(values) <= 0:
                raise ValueError("Sampling pools require positive weights")
            for group, weight in zip(selected, values):
                group["fraction"] = fraction * weight / sum(values)
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
    config.update(skill_specs=specs, scene_groups=groups, episode_seconds=max(
        spec.get("episode_seconds", 20.) for spec in specs.values()),
        curriculum_reference_batch=plan["target_num_envs"] * plan["num_steps_per_env"],
        checkpoint_interval=plan.get("checkpoint_interval", 100), checkpoint_first_update=10,
        save_interval=500, checkpoint_snapshots=True, curriculum="integrated_on_policy_skills",
        terrain_difficulty_tiers=[.25, .5, .75, 1.] if recipe["kind"] in ("terrain", "jump", "mixed") else [1.])
    config["stages"][0]["terrain"] = list(dict.fromkeys(t for group in groups for t in group["terrain"]))
    config["evaluation"].update(cases=cases, episode_seconds=max(c["episode_seconds"] for c in cases),
        block_updates=plan["block_updates"], regression_patience=None,
        minimum_updates=math.ceil(recipe.get("minimum_updates", 0) * plan["target_num_envs"] / num_envs),
        skip_training_if_initially_accepted=False)
    if recipe["kind"] in ("terrain", "mixed"):
        config["scut_effort_reward_scale"] = .1
    if plan.get("stationary_tracking"):
        config["stationary_tracking"] = deepcopy(plan["stationary_tracking"])
    if "fall_confirmation_seconds" in plan:
        config["fall_confirmation_seconds"] = plan["fall_confirmation_seconds"]
    for key in ("learning_rate", "critic_warmup_updates", "transfer_critic", "transfer_noise_floor", "performance_curriculum"):
        if key in recipe:
            config[key] = deepcopy(recipe[key])
    return config
