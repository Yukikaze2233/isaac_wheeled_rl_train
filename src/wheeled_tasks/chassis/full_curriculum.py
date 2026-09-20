"""Materialize auditable per-stage contracts for one progressively trained policy."""
from copy import deepcopy
import math
from pathlib import Path


FULL_CONTRACT_ID = "v5-gas-spring-full-stage-research-v2"


def checkpoint_contract_path(checkpoint):
    checkpoint = Path(checkpoint)
    sidecar = checkpoint.with_suffix(".contract.json")
    return sidecar if sidecar.exists() else checkpoint.parent / "contract.json"


def compatible_control_transfer(old, new):
    """Allow only wheel-bound widening; action units, gains and leg bounds stay exact."""
    old, new = dict(old), dict(new)
    old_clip = old.pop("wheel_action_clip", old["action_clip"])
    new_clip = new.pop("wheel_action_clip", new["action_clip"])
    return old == new and new_clip >= old_clip


def stage_contract(base, plan, recipe, num_envs):
    if plan["contract_id"] in ("v5-complete-curriculum-plan-v3", "v5-complete-curriculum-plan-v4"):
        recipe = {"vx_max": 3., "yaw_max": 8., "terrain_scale": 1., **recipe}
    config = deepcopy(base)
    kind = recipe["kind"]
    updates = max(1, math.ceil(recipe["updates"] * plan["target_num_envs"] / num_envs))
    config.update(contract_id=FULL_CONTRACT_ID, curriculum_stage=recipe["name"], target_num_envs=num_envs,
                  total_updates=updates, enabled_stages=[kind], centered_locomotion_resets=True,
                  height_reference="wheel_support_mean", command_slew={"vx_m_s2": 1.5, "yaw_rad_s2": 4.0})
    config["v5_control"]["wheel_action_clip"] = plan["wheel_action_clip"]
    config.update(learning_rate=3e-5, learning_rate_schedule="fixed", critic_warmup_updates=50,
                  zero_critic_wheel_positions=True)
    config["critic_layout"][4] = "tree_q18_wheel_positions_zeroed"
    config["upper_target_layout"][1] = "mode_conditioned_target_height_delta_m"
    config["task_semantics"] = {"route_goal_x_m": .65, "success_hold_seconds": .5,
        "jump_request_seconds": .8, "preload_seconds": .25, "preload_depth_m": .02,
        "release_height_offset_m": .035, "jump_apex_delta_m": recipe.get("jump_apex_delta_m", .06),
        "jump_min_clearance_m": recipe.get("jump_min_clearance_m", .01),
        "jump_min_air_seconds": .06, "jump_landing_radius_m": .25,
        "jump_release_velocity_min_m_s": .2 if recipe.get("jump_apex_delta_m", .06) <= .06 else .4}
    scale = recipe["terrain_scale"]
    config["terrain_limits"] = {name: value * scale for name, value in base["terrain_limits"].items()}
    groups = [
        {"name": "stand", "fraction": .3, "terrain": ["flat"]},
        {"name": "translate", "fraction": .3, "terrain": ["flat"]},
        {"name": "rotate", "fraction": .3, "terrain": ["flat"]},
        {"name": "combined", "fraction": .1, "terrain": ["flat"]},
    ]
    if kind == "terrain":
        groups = [
            {"name": "stand", "fraction": .2, "terrain": ["flat"]},
            {"name": "translate", "fraction": .15, "terrain": ["flat"]},
            {"name": "rotate", "fraction": .15, "terrain": ["flat"]},
            {"name": "terrain_move", "fraction": .25, "terrain": ["slope", "rough", "material"]},
            {"name": "step_up", "fraction": .15, "terrain": ["low_step", "step_up", "stairs"]},
            {"name": "step_down", "fraction": .1, "terrain": ["step_down"]},
        ]
    if kind in ("jump", "mixed"):
        groups = [
            {"name": "stand", "fraction": .15, "terrain": ["flat"]},
            {"name": "translate", "fraction": .1, "terrain": ["flat"]},
            {"name": "rotate", "fraction": .1, "terrain": ["flat"]},
            {"name": "step_up", "fraction": .075, "terrain": ["low_step", "step_up", "stairs"]},
            {"name": "step_down", "fraction": .075, "terrain": ["step_down"]},
            {"name": "terrain_move", "fraction": .1, "terrain": ["slope", "rough", "material"]},
            {"name": "jump", "fraction": .4, "terrain": ["jump"]},
        ]
    config["scene_groups"] = groups
    config["stages"] = [{"name": kind, "updates": updates, "vx_max": recipe["vx_max"],
        "yaw_max": recipe["yaw_max"], "push_max": .15 if recipe.get("robust") else .05,
        "terrain": list(dict.fromkeys(t for g in groups for t in g["terrain"]))}]
    config["signal_perturbations"] = {"enabled": recipe.get("robust", False), "max_delay_steps": 2,
        "noise_scale": 1., "enabled_fraction": .7, "scope": "research_priors_not_hardware_identification"}
    config["observation_noise_enabled"] = recipe.get("robust", False)
    cases = deepcopy(base["evaluation"]["cases"])
    for case in cases:
        case["anchor"] = True
    if kind != "foundation":
        cases += [
            {"name": "speed_forward", "command": [recipe["vx_max"], 0., .305], "velocity_mae_m_s_max": .15},
            {"name": "speed_backward", "command": [-recipe["vx_max"], 0., .305], "velocity_mae_m_s_max": .15},
            {"name": "spin_left", "command": [0., recipe["yaw_max"], .305], "yaw_mae_rad_s_max": .35},
            {"name": "spin_right", "command": [0., -recipe["yaw_max"], .305], "yaw_mae_rad_s_max": .35},
            {"name": "combined_motion", "command": [min(recipe["vx_max"], 1.), min(recipe["yaw_max"], 1.), .305],
             "velocity_mae_m_s_max": .15, "yaw_mae_rad_s_max": .25},
        ]
        for case in cases[len(base["evaluation"]["cases"]):]:
            case["anchor"] = kind not in ("foundation", "speed")
    if kind in ("terrain", "jump", "mixed"):
        for terrain in ("slope", "rough", "material", "step_up", "step_down", "stairs"):
            route = terrain in ("step_up", "step_down", "stairs")
            cases.append({"name": terrain, "terrain": terrain, "task": "traverse" if route else "survive",
                "command": [.4 if route else .2, 0., .305], "height_mae_m_max": .02,
                "velocity_mae_m_s_max": .15, "success_rate_min": .9, "anchor": kind in ("jump", "mixed")})
    if kind in ("jump", "mixed"):
        cases.append({"name": "jump", "terrain": "jump", "task": "jump", "command": [0., 0., .305], "success_rate_min": .9, "anchor": kind == "mixed"})
    if recipe.get("robust"):
        for name, command in (("delayed_stand", [0., 0., .305]), ("delayed_forward", [.5, 0., .305]),
                              ("delayed_turn", [0., 1., .305])):
            cases.append({"name": name, "command": command, "perturbed": True, "height_mae_m_max": .015,
                          "velocity_mae_m_s_max": .15, "yaw_mae_rad_s_max": .2})
        for terrain in ("step_up", "step_down", "jump"):
            cases.append({"name": "delayed_" + terrain, "terrain": terrain, "perturbed": True,
                "task": "jump" if terrain == "jump" else "traverse",
                "command": [0. if terrain == "jump" else .4, 0., .305],
                "height_mae_m_max": .02, "velocity_mae_m_s_max": .15, "success_rate_min": .9})
        cases.append({"name": "pushed_stand", "command": [0., 0., .305], "perturbed": True,
                      "push_velocity_m_s": [.15, 0.], "push_at_s": 3., "height_mae_m_max": .015})
    config["evaluation"].update(protocol_id="v5-full-" + recipe["name"] + "-v2", cases=cases,
        seed=plan["evaluation_seeds"][0], confirmation_seed=plan["evaluation_seeds"][1],
        episodes_per_case=plan["evaluation_episodes_per_case"], block_updates=plan["block_updates"],
        skip_training_if_initially_accepted=True, consecutive_passes_required=1, protect_anchor_cases=True)
    if plan["contract_id"] in ("v5-complete-curriculum-plan-v3", "v5-complete-curriculum-plan-v4"):
        from .skill_curriculum import configure_skill_contract
        config = configure_skill_contract(config, base, plan, recipe)
    return config
