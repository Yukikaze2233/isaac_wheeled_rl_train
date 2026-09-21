"""Materialize auditable per-stage contracts for one progressively trained policy."""
from copy import deepcopy
import math
from pathlib import Path


FULL_CONTRACT_ID = "v5-gas-spring-full-stage-research-v2"


def resolve_plan(plan, loader, seen=()):
    """Resolve a committed parent plan without duplicating its stage catalogue."""
    plan = deepcopy(plan)
    parent_name = plan.pop("extends_plan", None)
    if parent_name:
        if parent_name in seen:
            raise ValueError("Cyclic curriculum plan inheritance")
        parent = resolve_plan(loader(parent_name), loader, (*seen, parent_name))
        reference = {**parent.get("training_reference", {}), **plan.get("training_reference", {})}
        parent.update(plan)
        parent["training_reference"] = reference
        plan = parent
    overrides = plan.pop("stage_overrides", {})
    for recipe in plan["stages"]:
        recipe.update(overrides.get(recipe["name"], {}))
    return plan


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
    if plan["contract_id"] in ("v5-complete-curriculum-plan-v3", "v5-complete-curriculum-plan-v4", "v5-complete-curriculum-plan-v5"):
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
    if plan["contract_id"] in ("v5-complete-curriculum-plan-v3", "v5-complete-curriculum-plan-v4", "v5-complete-curriculum-plan-v5"):
        from .skill_curriculum import configure_skill_contract
        config = configure_skill_contract(config, base, plan, recipe)
    if plan.get("actor_observation_source") == "scut35_encoders_imu_commands":
        config.update(physics_dt=plan["physics_dt"], policy_dt=plan["policy_dt"],
            num_steps_per_env=plan["num_steps_per_env"], history_length=1, actor_frame_dim=35,
            actor_dim=35, critic_dim=81, actor_observation_source=plan["actor_observation_source"],
            learning_rate=plan.get("learning_rate", 1e-4), learning_rate_schedule="adaptive",
            critic_warmup_updates=0, initial_noise_std=1.,
            contact_estimate_source="privileged_only_not_actor_input")
        config["actor_layout"] = ["command_xyz3", "height_command1_times5", "imu_gyro3_times0.5",
            "imu_projected_gravity3", "motor_position_delta6_wheels_zeroed", "motor_velocity6_times0.1",
            "previous_policy_action6", "command_context7_no_contact_phase"]
        config["policy_action_order"] = ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"]
        config["critic_layout"][0] = "clean_frame35"
        config["v5_control"]["leg_position_scale"] = .25
        config["signal_perturbations"]["max_delay_steps"] = round(.02 / config["policy_dt"])
        config["transfer_critic"] = plan.get("transfer_critic", False)
    if plan.get("verify_height_endpoints"):
        cases = config["evaluation"]["cases"]
        for case in cases[:]:
            if case.get("skill", {}).get("kind") != "height":
                continue
            case["height_mae_m_max"] = plan.get("height_profile_mae_m_max", .005)
            for suffix, height in (("low", .29), ("high", .32)):
                endpoint = deepcopy(case)
                endpoint.update(name=case["name"] + "_hold_" + suffix,
                                command=[0., 0., height], height_mae_m_max=.005)
                endpoint["skill"] = {"kind": "stand", "mode": 0, "command": [0., 0., height]}
                cases.append(endpoint)
    return config
