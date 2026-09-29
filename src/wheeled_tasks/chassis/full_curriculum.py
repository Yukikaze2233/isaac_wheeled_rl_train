"""Materialize auditable per-stage contracts for one progressively trained policy."""
from copy import deepcopy
import json
import hashlib
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
    insertions = plan.pop("insert_stages_after", {})
    if insertions:
        if set(insertions) - {s["name"] for s in plan["stages"]}:
            raise ValueError("Unknown stage insertion point")
        plan["stages"] = [item for stage in plan["stages"] for item in (stage, *insertions.get(stage["name"], []))]
        if len({s["name"] for s in plan["stages"]}) != len(plan["stages"]):
            raise ValueError("Duplicate curriculum stage names")
    phases = plan.pop("phase_schedule", None)
    if phases is not None:
        if "skill_catalog" not in plan:
            plan["skill_catalog"] = plan["stages"]
        plan["stages"] = phases
    additions = plan.pop("catalog_additions", [])
    if additions:
        catalog = plan["skill_catalog"] + additions
        if len({item["name"] for item in catalog}) != len(catalog):
            raise ValueError("Duplicate skill catalog entries")
        plan["skill_catalog"] = catalog
    catalog_overrides = plan.pop("catalog_overrides", {})
    if catalog_overrides:
        if set(catalog_overrides) - {item["name"] for item in plan["skill_catalog"]}:
            raise ValueError("Unknown skill catalog override")
        for item in plan["skill_catalog"]:
            item.update(catalog_overrides.get(item["name"], {}))
    reference = plan.pop("height_workspace_file", None)
    if reference:
        plan["height_workspace"] = loader(reference)
    transport_profile = plan.pop("usb_transport_profile_file", None)
    if transport_profile:
        plan["usb_transport"]["profile"] = loader(transport_profile)
    dynamics_profile = plan.pop("dynamics_randomization_file", None)
    if dynamics_profile:
        plan["dynamics_randomization"] = loader(dynamics_profile)
    return plan


def checkpoint_contract_path(checkpoint):
    checkpoint = Path(checkpoint)
    sidecar = checkpoint.with_suffix(".contract.json")
    return sidecar if sidecar.exists() else checkpoint.parent / "contract.json"


def checkpoint_update_count(checkpoint):
    """Read immutable completion metadata without loading torch in supervisors."""
    checkpoint = Path(checkpoint)
    seal = checkpoint.with_suffix(".checkpoint.json")
    if seal.exists():
        data = json.loads(seal.read_text())
        if (data.get("schema") != "chassis-learning-checkpoint-v1" or data.get("sealed") is not True
                or data.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                or type(data.get("learning_updates")) is not int or data["learning_updates"] < 0):
            raise ValueError("Invalid or stale per-file learning checkpoint manifest")
        contract = checkpoint_contract_path(checkpoint)
        if data.get("contract_sha256") != hashlib.sha256(contract.read_bytes()).hexdigest():
            raise ValueError("Per-file checkpoint contract mismatch")
        return data["learning_updates"]
    marker = checkpoint.parent / "completion.json"
    data = json.loads(marker.read_text())
    if checkpoint.name == "model.pt" and data["status"] == "checkpoint_sealed":
        return int(data["successful_updates"])
    if checkpoint.name == "model_final.pt" and data["status"] in ("completed", "stopped"):
        return int(data.get("parent_updates", 0)) + int(data["successful_updates"])
    raise ValueError("Resume requires an immutable snapshot or finalized training block")


def compatible_control_transfer(old, new):
    """Allow only wheel-bound widening; action units, gains and leg bounds stay exact."""
    old, new = dict(old), dict(new)
    old_clip = old.pop("wheel_action_clip", old["action_clip"])
    new_clip = new.pop("wheel_action_clip", new["action_clip"])
    return old == new and new_clip >= old_clip


def stage_contract(base, plan, recipe, num_envs, *, protected_cases=None):
    """Compose a stage without introducing a reverse planner dependency."""
    if plan.get("contract_id") == "v6-new-asset-curriculum-plan-v1":
        return _new_asset_stage_contract(plan, recipe, num_envs, protected_cases=protected_cases)
    limit = recipe.get("num_envs", num_envs)
    if "num_envs" in recipe and (not isinstance(limit, int) or not 32 <= limit <= 20000):
        raise ValueError("Stage environment ceiling must be an integer in [32,20000]")
    num_envs = min(num_envs, limit)
    warmup = recipe.get("critic_warmup_reference_updates", plan.get("critic_warmup_reference_updates"))
    if warmup is not None:
        if not math.isfinite(warmup) or warmup < 0:
            raise ValueError("Critic warmup reference budget must be finite and nonnegative")
        recipe = {**recipe, "critic_warmup_updates": math.ceil(warmup * plan["target_num_envs"] / num_envs)}
    if "new_skills" in recipe:
        from .integrated_curriculum import integrated_contract
        foundation = {key: value for key, value in recipe.items() if key != "new_skills"}
        config = _specialist_contract(base, {**plan, "stages": [foundation]}, foundation, num_envs)
        config = integrated_contract(config, base, plan, recipe, num_envs)
    else:
        config = _specialist_contract(base, plan, recipe, num_envs)
    for key in ("asset_directory", "solid_step_platforms", "fixed_evaluation_terrain", "step_assist", "step_contact_grace",
                 "zero_command_velocity_scale", "dynamics_randomization", "command_reference", "reference_reward",
                 "actor_migration", "terrain_reset_before_entry", "manual_context35", "contact_domain", "command_transport",
                 "precision_tracking", "reward_migration", "resume_critic_warmup_updates", "budget_migration"):
        if key in plan:
            config[key] = deepcopy(plan[key])
        if key in recipe:
            config[key] = deepcopy(recipe[key])
    if "dynamics_randomization_overrides" in recipe:
        config["dynamics_randomization"].update(recipe["dynamics_randomization_overrides"])
    if config.get("manual_context35"):
        config["contact_domain"]["enabled_fraction"] = recipe.get("contact_domain_fraction", .5)
        config["command_transport"]["enabled_fraction"] = recipe.get("transport_fraction", .5)
        config["reference_reward"]["upright_denominator"] = recipe.get("upright_denominator", .05)
    if plan.get("retention_case_names"):
        present = {c["name"] for c in config["evaluation"]["cases"]}
        if set(plan["retention_case_names"]) - present:
            raise ValueError("Retention suite must exist in every adaptation stage")
        config["evaluation"]["retention_case_names"] = list(plan["retention_case_names"])
        config["evaluation"]["continuation_selection"] = "retain_parent_passes_then_rank"
        config["evaluation"].update(stand_velocity_mae_m_s_max=.03, mechanical_checks=True)
    if config.get("dynamics_randomization", {}).get("enabled") and plan.get("dynamics_evaluation"):
        suite = plan["dynamics_evaluation"]
        for case in config["evaluation"]["cases"][:]:
            if case["name"] not in suite["cases"]:
                continue
            for profile in suite["profiles"]:
                varied = deepcopy(case)
                varied.update(name=case["name"] + "__dynamics_" + profile["name"], anchor=False,
                    dynamics_profile=deepcopy(profile["parameters"]), reset_seed_key=case["name"])
                config["evaluation"]["cases"].append(varied)
    if config.get("step_assist", {}).get("enabled"):
        for case in config["evaluation"]["cases"][:]:
            if case.get("terrain") == "step_up" and case.get("terrain_limits", {}).get("step_up_m", 0.) >= .15:
                unassisted = deepcopy(case)
                unassisted.update(name=case["name"] + "__unassisted", step_assist_enabled=False,
                                  reset_seed_key=case["name"], anchor=False)
                config["evaluation"]["cases"].append(unassisted)
    if plan.get("usb_transport") and plan.get("usb_evaluation_pairs"):
        for case in config["evaluation"]["cases"][:]:
            case["transport_enabled"] = False
            case.setdefault("reset_seed_key", case["name"])
            delayed = deepcopy(case)
            delayed.update(name=case["name"] + "_usb", transport_enabled=True, anchor=False)
            config["evaluation"]["cases"].append(delayed)
        if config["evaluation"].get("retention_case_names"):
            config["evaluation"]["retention_case_names"] += [name + "_usb" for name in plan["retention_case_names"]]
    config["evaluation"]["block_updates"] = recipe.get("block_updates", config["evaluation"]["block_updates"])
    if config.get("command_reference"):
        if config.get("step_assist", {}).get("enabled"):
            raise ValueError("Manual command-reference training cannot enable terrain-oracle assistance")
        config["scut_effort_reward_scale"] = 1.
        if not config.get("manual_context35"):
            config.update(actor_dim=36, actor_frame_dim=36, critic_dim=114,
                          actor_observation_source="encoders_imu_command_reference36")
            config["actor_layout"][-1] = "manual_skill_reference_context7"
            config["actor_layout"].append("longitudinal_acceleration_reference1")
            config["critic_layout"][0] = "clean_reference_frame36"
            config["critic_layout"] += ["body_mass_ratio19", "base_com_offset3_times10", "landing_displacement_xy2_times0.2",
                                       "whole_com_vz1", "whole_com_rise1_times5", "physical_phase_onehot5", "requested_com_displacement1_times5"]
        config["task_semantics"].update(jump_apex_frame="com_release", success_hold_seconds=2.,
            preload_seconds=config["command_reference"]["preload_seconds"],
            preload_depth_m=config["command_reference"]["preload_depth_m"],
            release_height_offset_m=config["command_reference"]["release_offset_m"])
        targets = recipe.get("promotion_cases", plan.get("promotion_cases", []))
        if not targets:
            raise ValueError("Remedial stages require explicit capability gates")
        names = {case["name"] for case in config["evaluation"]["cases"]}
        paired = list(targets) + ([name + "_usb" for name in targets] if plan.get("usb_evaluation_pairs") else [])
        if recipe.get("promote_all_cases"):
            paired = sorted(names)
        if set(paired) - names and not plan.get("unified_evaluation"):
            raise ValueError("Unknown promotion case")
        config["evaluation"].update(promotion_case_names=paired, mode="gate",
            protocol_id="v6-manual-" + recipe["name"] + "-v1",
            protect_anchor_cases=False, require_passing_anchors=False,
            regression_patience=2, consecutive_passes_required=1,
            skip_training_if_initially_accepted=False)
        if "evaluation_episodes_per_case" in recipe:
            config["evaluation"]["episodes_per_case"] = recipe["evaluation_episodes_per_case"]
    if plan.get("unified_evaluation"):
        # Compile one final-domain manifest, independent of stage sampling.
        full = stage_contract(base, {**plan, "unified_evaluation": False}, plan["stages"][-1], num_envs)
        cases = deepcopy(full["evaluation"]["cases"])
        sources = {c["name"]: c for c in cases}
        for name in ("stand", "forward_05", "backward_05", "forward_2", "forward_3", "forward_5", "rotate_1", "jump_cold_03", "step_up_03"):
            for mu in (.1, .2, .3, .4, .8, 1.2):
                case = deepcopy(sources[name])
                case.update(name=f"{name}__mu_{round(mu * 100):03d}", contact_profile={"friction": mu}, reset_seed_key=name)
                if mu < .4 and "skill" in case:
                    case["skill"]["acceleration_m_s2"] = .5
                cases.append(case)
            for delay in (.1, 1., 3., 5.):
                case = deepcopy(sources[name])
                case.update(name=f"{name}__delay_{delay:g}", communication_profile={"delay_ms": delay}, reset_seed_key=name)
                cases.append(case)
            case = deepcopy(sources[name])
            case.update(name=name + "__loss", communication_profile={"delay_ms": 5., "drop_probability": .02}, reset_seed_key=name)
            cases.append(case)
            case = deepcopy(sources[name])
            case.update(name=name + "__burst", communication_profile={"delay_ms": 5.,
                "burst_every_packets": 100, "burst_packets": 3}, reset_seed_key=name)
            cases.append(case)
            case = deepcopy(sources[name])
            case.update(name=name + "__low_grip_loss", contact_profile={"friction": .1},
                communication_profile={"delay_ms": 5., "drop_probability": .02}, reset_seed_key=name)
            if "skill" in case:
                case["skill"]["acceleration_m_s2"] = .5
            cases.append(case)
        config["evaluation"].update(cases=cases, protocol_id=plan["evaluation_protocol_id"],
            episode_seconds=max(c.get("episode_seconds", 10.) for c in cases),
            retention_case_names=[c["name"] for c in cases],
            cumulative_retention=bool(plan.get("cumulative_retention")),
            zero_success_fuse_updates=recipe.get("zero_success_fuse_updates"),
            frozen_signal_perturbations={"enabled": True, "max_delay_steps": 0, "noise_scale": 1., "enabled_fraction": 1.})
        if recipe.get("promote_all_cases"):
            config["evaluation"]["promotion_case_names"] = [c["name"] for c in cases]
        if set(config["evaluation"]["promotion_case_names"]) - {c["name"] for c in cases}:
            raise ValueError("Promotion case absent from the fixed V6 manifest")
    if plan.get("evaluation_strategy"):
        config["evaluation"].update(deepcopy(plan["evaluation_strategy"]))
        if config["evaluation"].get("stable_case_layout"):
            config["evaluation"]["canonical_case_names"] = [c["name"] for c in config["evaluation"]["cases"]]
    if plan.get("regression_recovery"):
        config["evaluation"]["regression_recovery"] = deepcopy(plan["regression_recovery"])
    if config.get("manual_context35"):
        if (config["physics_dt"], config["policy_dt"]) != (plan["physics_dt"], plan["policy_dt"]):
            raise ValueError("Materialized control clocks differ from the declared plan")
    if config.get("precision_tracking"):
        if not config.get("manual_context35") or not config.get("command_reference"):
            raise ValueError("Precision tracking requires the manual35 command-reference interface")
        for name, value in config["precision_tracking"].items():
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Precision tracking requires a positive finite parameter: {name}")
    if plan.get("protected_rehearsal"):
        if protected_cases is None:
            protected_cases = plan.get("initial_protected_case_names", [])
        config["evaluation"]["protected_case_names"] = sorted(protected_cases)
        from .integrated_curriculum import apply_protected_rehearsal
        apply_protected_rehearsal(config, plan["protected_rehearsal"])
    return config


def _new_asset_stage_contract(plan, recipe, num_envs, *, protected_cases=None):
    """Materialize the independent campaign without legacy stage defaults.

    Reuse the tested skill/reference/terrain vocabulary, but make the complete
    control ABI, budgets, sampling pools and promotion catalogue local inputs.
    The caller's legacy ``base`` argument is intentionally never consulted.
    """
    from .skill_curriculum import skill_cases, skill_spec
    from .motion_limits import validate_command

    recovery_revision = plan.get("curriculum_revision") == "v6_recovery_curriculum_v1"
    step_revision = plan.get("curriculum_revision") in ("v6_step_curriculum_v1", "v6_recovery_curriculum_v1")
    ceilings = (dict(zip(("S1", "S2", "S3", "S4", "S5", "S5T", "S5D", "S6"),
                         (1000, 2000, 1000, 1500, 1000, 600, 400, 1000))) if step_revision else
                dict(zip(("S1", "S2", "S3", "S4", "S5", "S6"),
                         (1000, 2000, 1000, 1500, 2000, 1000))))
    if any(key in plan for key in ("extends_plan", "phase_schedule", "stage_overrides")):
        raise ValueError("The new asset campaign cannot inherit legacy stages")
    if (type(num_envs) is not int or num_envs != 20000
            or plan["target_num_envs"] != 20000
            or plan["num_steps_per_env"] != 24
            or plan["ppo_minibatch_samples"] != 48000
            or plan["num_mini_batches"] != 10 or plan["num_learning_epochs"] != 5):
        raise ValueError("New asset formal budget is fixed at 20000x24, 10x48000 minibatches and 5 epochs")
    if {item["name"]: item["updates"] for item in plan["stages"]} != ceilings:
        raise ValueError("Independent stage budgets do not match the frozen specification")
    if [item["name"] for item in plan["stages"]] != list(ceilings):
        raise ValueError("New asset stages must execute in the frozen stage order")
    for item in plan["stages"]:
        if item["budget_stage_id"] != item["name"] or item["num_envs"] != 20000:
            raise ValueError("Every stage needs an exact budget-stage and environment mapping")
    if sum(ceilings.values()) != plan["total_update_ceiling"]:
        raise ValueError("New asset total budget mismatch")
    if recipe not in plan["stages"]:
        raise ValueError("Materialization requires an exact declared stage recipe")
    base = deepcopy(plan["runtime_base"])
    if (base["physics_dt"], base["policy_dt"], base["actor_dim"], base["critic_dim"],
            base["history_length"], base["action_dim"]) != (.001, .02, 35, 81, 1, 6):
        raise ValueError("New asset control clock or policy ABI changed")
    if (base["v5_control"]["leg_kp"], base["v5_control"]["leg_kd"]) != (160., 2.5):
        raise ValueError("New asset PC PD differs from the recorded baseline")
    if any(base["policy_architecture"][key] != value for key, value in (
            ("actor_hidden_dims", [256, 128, 64]), ("critic_hidden_dims", [256, 128, 64]),
            ("activation", "elu"))):
        raise ValueError("The unchanged MLP architecture is part of the campaign ABI")
    config = deepcopy(base)
    sid = recipe["name"]
    stage_index = list(ceilings).index(sid)
    config.update(contract_id="v6-gas-spring-full-stage-research-v1", curriculum_stage=sid,
        budget_stage_id=sid, target_num_envs=num_envs, total_updates=ceilings[sid],
        num_steps_per_env=24, num_mini_batches=10, num_learning_epochs=5,
        ppo_minibatch_samples=48000, curriculum_reference_batch=480000,
        enabled_stages=[recipe["kind"]], zero_critic_wheel_positions=True,
        curriculum_plan_id=plan["contract_id"], budget_spec_file=plan["budget_spec_file"],
        budget_spec_sha256=plan["budget_spec_sha256"], updates_are_actual_not_reference_scaled=True,
        rollback_refunds_consumed_updates=False, release_scope=deepcopy(plan["release_scope"]))
    schedule = deepcopy(plan["new_asset_curriculum"])
    schedule.update(stage=sid, edge_reset_fraction=(.05 if recovery_revision else .1) if stage_index >= 2 else 0.,
                    unilateral_degradation_fraction=.05 if stage_index >= 2 else 0.)
    config["jump_assist"] = {**deepcopy(plan["jump_assist"]), "enabled": sid == "S5"}
    schedule["request_protocol"]["stress_episode_fraction"] = (.5 if sid == "S6" else .2 if stage_index >= 3 else 0.)
    schedule["command_resampling_seconds"] = [.5, 4.] if sid == "S6" else [1., 8.]
    schedule["long_command_fraction"] = .2 if sid == "S6" else 0.
    config["new_asset_curriculum"] = schedule
    for module in ("dynamics_randomization", "contact_domain", "command_transport", "signal_perturbations"):
        config[module]["enabled"] = bool(recipe.get("robust"))
    config["observation_noise_enabled"] = bool(recipe.get("robust"))

    catalog = {item["name"]: item for item in plan["skill_catalog"]}
    if len(catalog) != len(plan["skill_catalog"]):
        raise ValueError("Duplicate new asset skill names")
    specs, terrains, cases, case_owner = {}, {}, [], {}
    for name, item in catalog.items():
        validate_command(item.get("command", [0., 0., .305]), config["motion_limits"])
        terrain, spec = skill_spec(item)
        terrain = item.get("terrain", terrain)
        if "mode" in item:
            spec["mode"] = item["mode"]
        spec.update(terrain_limits={**base["terrain_limits"], **deepcopy(item.get("terrain_limits", {})),
                                    **spec.get("terrain_limits", {})},
                    episode_seconds=item.get("episode_seconds", 20.),
                    sample_amplitude=item.get("sample_amplitude", False), sample_yaw_sign=False,
                    push_m_s=item.get("push_m_s", 0.))
        for key in ("request_events", "command_sequence"):
            if key in item:
                spec[key] = deepcopy(item[key])
        if item["skill"] in ("forward", "backward", "rotate", "curve", "spin_translate"):
            spec["command_resampling_seconds"] = schedule["command_resampling_seconds"]
        specs[name], terrains[name] = spec, terrain
        if item["skill"] in ("rotate", "curve", "spin_translate"):
            reverse = deepcopy(spec)
            reverse["command"][1] *= -1
            specs[name + "__reverse_train"] = reverse
            terrains[name + "__reverse_train"] = terrain
        for case in skill_cases(base, item):
            case.update(terrain=terrain, task=item.get("task", case["task"]),
                        success_rate_min=.95, survival_rate_min=.95, anchor=False,
                        reset_seed_key=case["name"])
            case["skill"]["mode"] = spec["mode"]
            if case["task"] == "traverse" and spec["mode"] not in (2, 3) and item["skill"] != "body_climb":
                raise ValueError("Traversal cases require a route-completion skill mode")
            for key in ("request_events", "command_sequence"):
                if key in item:
                    case["skill"][key] = deepcopy(item[key])
            if item.get("request_events"):
                case["request_protocol_case"] = name
            if item["skill"] == "stand":
                case.update(height_mae_m_max=.005, velocity_mae_m_s_max=.03,
                            stationary=True, stand_drift_m_max=.1)
            if item["skill"] == "height":
                case["height_mae_m_max"] = .005
            if item["skill"] in ("start_stop", "velocity_curve"):
                case["settled_stop_speed_max"] = .03
            if item["skill"] in ("jump", "running_jump"):
                case.update(jump_assistance_enabled=False, episode_seconds=20.)
            if item["skill"] in ("step_jump", "double_step_jump"):
                case.update(jump_assistance_enabled=False, introduced_stage=item["introduced_stage"])
            if item["skill"] == "body_climb":
                case["introduced_stage"] = item["introduced_stage"]
            if item["skill"] in ("rolling_up", "rolling_stairs"):
                case["introduced_stage"] = item["introduced_stage"]
            case_owner[case["name"]] = name
            cases.append(case)

    groups, membership = [], {}
    pools = recipe["sampling_pools"]
    if not math.isclose(sum(pool["fraction"] for pool in pools), 1., abs_tol=1e-9):
        raise ValueError("New asset behavior pool fractions must sum to one")
    active_specs = {}
    for pool in pools:
        if pool["fraction"] <= 0 or not pool["skills"]:
            raise ValueError("Training pools must be positive and nonempty")
        selected = []
        for name in pool["skills"]:
            if name not in catalog or list(ceilings).index(catalog[name]["introduced_stage"]) > stage_index:
                raise ValueError("Sampling an unknown or not-yet-introduced skill")
            selected.append(name)
            if name + "__reverse_train" in specs:
                selected.append(name + "__reverse_train")
        cross_fraction = plan["height_combination_fraction_of_motion"] if pool["name"] == "motion" else 0.
        sequence_fraction = recipe.get("sequence_fraction_of_motion", 0.) if pool["name"] == "motion" else 0.
        for name in selected:
            if name in membership:
                raise ValueError("A skill cannot belong to two behavior pools")
            active_specs[name] = deepcopy(specs[name])
            membership[name] = pool["name"]
            groups.append({"name": name, "fraction": pool["fraction"] * (1 - sequence_fraction) * (1 - cross_fraction) / len(selected),
                           "terrain": [terrains[name]]})
            if cross_fraction:
                cross_name = name + "__height_train"
                active_specs[cross_name] = {**deepcopy(specs[name]),
                    "height_sampling": deepcopy(plan["height_combination_sampling"]),
                    "height_transition_seconds": plan["height_combination_sampling"]["transition_seconds"]}
                membership[cross_name] = pool["name"]
                groups.append({"name": cross_name, "fraction": pool["fraction"] * (1 - sequence_fraction) * cross_fraction / len(selected),
                               "terrain": [terrains[name]]})
        sequence_names = pool.get("sequence_skills", [])
        if bool(sequence_fraction) != bool(sequence_names) or not 0 <= sequence_fraction < 1:
            raise ValueError("Sequence sampling needs explicit nonempty bounded allocation")
        for name in sequence_names:
            if name not in catalog or catalog[name]["introduced_stage"] != sid or name in membership:
                raise ValueError("Sequence sampling needs unique current-stage skills")
            active_specs[name] = deepcopy(specs[name])
            membership[name] = pool["name"]
            groups.append({"name": name, "fraction": pool["fraction"] * sequence_fraction / len(sequence_names),
                           "terrain": [terrains[name]]})

    # Fixed, independent cross-height tests preserve the commanded endpoints;
    # runtime projection cannot silently redefine the declared target domain.
    for case in cases[:]:
        item = catalog[case_owner[case["name"]]]
        if item["skill"] not in ("forward", "backward", "rotate", "curve", "spin_translate"):
            continue
        if item.get("sample_amplitude"):
            continue
        for height in (.23, .32, .43):
            cross = deepcopy(case)
            cross.update(name=case["name"] + f"_height_{round(height * 1000):03d}mm", episode_seconds=20., warmup_seconds=4.)
            cross["command"][2] = height
            cross["skill"].update(command=list(cross["command"]), height_transition_seconds=4.)
            case_owner[cross["name"]] = case_owner[case["name"]]
            cases.append(cross)
    # Explicit payload, contact and communication conditions are fixed research
    # profiles, never interpreted as a successful hardware identification.
    for name in ("stand_305mm", "forward_05", "backward_05", "rotate_1"):
        source = next(case for case in cases if case["name"] == name)
        for suffix, fields in (
            ("payload_light", {"dynamics_profile": {"base_mass_scale": .9}}),
            ("payload_heavy", {"dynamics_profile": {"base_mass_scale": 1.1}}),
            ("com_forward", {"dynamics_profile": {"base_com_offset_m": [.01, 0., 0.]}}),
            ("low_grip", {"contact_profile": {"friction": .3}}),
            ("usbhs", {"communication_profile": {"delay_ms": .16}})):
            case = {**deepcopy(source), **fields, "name": name + "__" + suffix,
                    "introduced_stage": "S3", "reset_seed_key": name}
            case_owner[case["name"]] = name
            cases.append(case)
    holdouts = (
        ("stand_320mm", "stand_payload_grip", None,
         {"dynamics_profile": {"base_mass_scale": 1.1}, "contact_profile": {"friction": .4}}),
        ("forward_3", "forward_height_com", .35,
         {"dynamics_profile": {"base_com_offset_m": [.006, -.003, 0.]}}),
        ("backward_4", "backward_height_grip_usb", .28,
         {"contact_profile": {"friction": .4}, "communication_profile": {"delay_ms": .08}}),
        ("rotate_2rps_reverse", "spin_height_payload_usb", .38,
         {"dynamics_profile": {"base_mass_scale": .95}, "communication_profile": {"delay_ms": .12}}),
        ("step_up_20", "step_up_payload_grip", None,
         {"dynamics_profile": {"base_mass_scale": 1.1}, "contact_profile": {"friction": .4}}),
        ("step_down_15", "step_down_com_usb", None,
         {"dynamics_profile": {"base_com_offset_m": [-.006, .003, 0.]}, "communication_profile": {"delay_ms": .12}}),
        ("jump_10", "jump_payload_grip", None,
         {"dynamics_profile": {"base_mass_scale": .95}, "contact_profile": {"friction": .4}}),
        ("running_jump_10", "running_jump_payload_usb", None,
         {"dynamics_profile": {"base_mass_scale": 1.05}, "communication_profile": {"delay_ms": .08}}),
    )
    for source_name, suffix, height, profiles in holdouts:
        if step_revision and source_name == "step_up_20":
            source_name, suffix = "body_climb_20", "body_climb_20_payload_grip"
        case = deepcopy(next(c for c in cases if c["name"] == source_name))
        case.update(profiles)
        case.update(name="heldout_" + suffix, introduced_stage="S6", reset_seed_key=source_name,
                    heldout_parameter_combination=True)
        if height is not None:
            case["command"][2] = height
            case["skill"].update(command=list(case["command"]), height_transition_seconds=4.)
            case.update(episode_seconds=20., warmup_seconds=4.)
        case_owner[case["name"]] = case_owner[source_name]
        cases.append(case)
    # Explicitly declared additions leave the original fixed suite intact.
    # Communication stress is introduced with S3; earlier stages still observe
    # these cases without treating them as already taught capabilities.
    for profile in plan.get("transport_evaluation_profiles", []):
        if (set(profile) != {"suffix", "communication_profile"}
                or not isinstance(profile["suffix"], str)
                or not profile["suffix"].startswith("link_")
                or not isinstance(profile["communication_profile"], dict)):
            raise ValueError("Transport evaluation additions require named explicit profiles")
        for name in ("stand_305mm", "forward_05", "backward_05", "rotate_1"):
            source = next(case for case in cases if case["name"] == name)
            case = deepcopy(source)
            case.update(name=name + "__" + profile["suffix"], introduced_stage="S3",
                        reset_seed_key=name, transport_robustness_case=True,
                        communication_profile=deepcopy(profile["communication_profile"]))
            case_owner[case["name"]] = name
            cases.append(case)
    if step_revision:
        # New capabilities are an explicit suffix; all surviving old cases retain
        # their definition and relative order for paired checkpoint review.
        def new_task(case):
            return case.get("step_jump_case") or case.get("body_climb_case") or case.get("untriggered_up_case")
        cases = [case for case in cases if not new_task(case)] + [case for case in cases if new_task(case)]
    if recovery_revision:
        # Keep the full previous 268-case prefix identical, including step tasks.
        cases = [case for case in cases if not case.get("recovery_case")] + [
            case for case in cases if case.get("recovery_case")]
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("The fixed new asset evaluation manifest has duplicate cases")
    promotions = []
    for case in cases:
        introduced = case.get("introduced_stage", catalog[case_owner[case["name"]]]["introduced_stage"])
        if "_height_" in case["name"] and "introduced_stage" not in case:
            introduced = "S2"
        if list(ceilings).index(introduced) <= stage_index:
            promotions.append(case["name"])
    config.update(skill_specs=active_specs, scene_groups=groups,
                  behavior_pool_membership=membership,
                  behavior_pool_fractions={pool["name"]: pool["fraction"] for pool in pools},
                  episode_seconds=max(spec["episode_seconds"] for spec in active_specs.values()))
    config["stages"] = [{"name": recipe["kind"], "updates": ceilings[sid],
                         "vx_max": recipe["vx_max"], "yaw_max": recipe["yaw_max"], "push_max": 0.,
                         "terrain": list(dict.fromkeys(t for group in groups for t in group["terrain"]))}]
    config["evaluation"].update(cases=cases, canonical_case_names=[c["name"] for c in cases],
        promotion_case_names=promotions, quick_case_names=[c["name"] for c in cases],
        retention_case_names=[c["name"] for c in cases], protected_case_names=[],
        episodes_per_case=recipe["evaluation_episodes_per_case"],
        episode_seconds=max(c["episode_seconds"] for c in cases), minimum_updates=40,
        zero_success_fuse_updates=None)
    # Additional passed cases belong to the authenticated gate ledger. They
    # cannot rewrite this independently frozen training distribution or ABI.
    # Binding is filled by the independent design-table generation step. Until
    # then the mandatory runtime preflight fails closed before sampling.
    if plan.get("design_preflight"):
        config["design_preflight"] = {**deepcopy(plan["design_preflight"]), "stage": sid}
    return config


def _specialist_contract(base, plan, recipe, num_envs):
    if plan["contract_id"] in ("v5-complete-curriculum-plan-v3", "v5-complete-curriculum-plan-v4", "v5-complete-curriculum-plan-v5", "v5-complete-curriculum-plan-v6"):
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
    for key in ("preload_depth_m", "preload_seconds", "jump_tuck_extension_m", "jump_tuck_sigma_m"):
        if key in plan.get("jump_reference", {}):
            value = plan["jump_reference"][key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Jump references require positive finite parameters")
            config["task_semantics"][key] = value
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
    evaluation_mode = plan.get("evaluation_mode", "gate")
    if evaluation_mode not in ("gate", "monitor"):
        raise ValueError("Evaluation mode must be gate or monitor")
    if evaluation_mode == "monitor":
        config["evaluation"]["mode"] = "monitor"
    if plan["contract_id"] in ("v5-complete-curriculum-plan-v3", "v5-complete-curriculum-plan-v4", "v5-complete-curriculum-plan-v5", "v5-complete-curriculum-plan-v6"):
        from .skill_curriculum import configure_skill_contract
        config = configure_skill_contract(config, base, plan, recipe)
    if plan.get("actor_observation_source") in ("scut35_encoders_imu_commands", "encoders_imu_command_reference36"):
        config.update(physics_dt=plan["physics_dt"], policy_dt=plan["policy_dt"],
            num_steps_per_env=plan["num_steps_per_env"], history_length=1, actor_frame_dim=35,
            actor_dim=35, critic_dim=81, actor_observation_source=plan["actor_observation_source"],
            learning_rate=plan.get("learning_rate", 1e-4), learning_rate_schedule=plan.get("learning_rate_schedule", "adaptive"),
            critic_warmup_updates=plan.get("critic_warmup_updates", 0), initial_noise_std=1.,
            contact_estimate_source="privileged_only_not_actor_input")
        config["actor_layout"] = ["command_xyz3", "height_command1_times5", "imu_gyro3_times0.5",
            "imu_projected_gravity3", "motor_position_delta6_wheels_zeroed", "motor_velocity6_times0.1",
            "previous_policy_action6", "command_context7_no_contact_phase"]
        config["policy_action_order"] = ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"]
        config["critic_layout"][0] = "clean_frame35"
        config["v5_control"]["leg_position_scale"] = .25
        config["signal_perturbations"]["max_delay_steps"] = round(.02 / config["policy_dt"])
        config["transfer_critic"] = plan.get("transfer_critic", False)
        config["command_slew"] = deepcopy(plan.get("command_slew", config["command_slew"]))
    if plan.get("ppo_minibatch_samples"):
        samples = plan["ppo_minibatch_samples"]
        batch = num_envs * config["num_steps_per_env"]
        if not isinstance(samples, int) or samples <= 0 or (batch >= samples and batch % samples):
            raise ValueError("PPO minibatch samples must divide the rollout batch")
        config["num_mini_batches"] = max(1, batch // samples)
    if plan.get("reward_profile"):
        config.update(reward_profile=plan["reward_profile"], reward_velocity_reference="base_link_origin",
                      termination_event_cost=200. * config["policy_dt"], height_l1_weight=0.)
    if plan.get("flat_triangle_mesh"):
        config.update(flat_triangle_mesh=True, flat_half_length_m=plan.get("flat_half_length_m", 160.))
    if plan.get("motion_limits"):
        from .motion_limits import validate_command
        config["motion_limits"] = deepcopy(plan["motion_limits"])
        for case in config["evaluation"]["cases"]:
            validate_command(case["command"], config["motion_limits"])
    config["evaluation"]["episode_seconds"] = max(c.get("episode_seconds", config["evaluation"]["episode_seconds"]) for c in config["evaluation"]["cases"])
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
    if plan.get("usb_transport"):
        config["usb_transport"] = deepcopy(plan["usb_transport"])
        config["usb_transport"].update(recipe.get("usb_transport_overrides", {}))
        # Retain optional noise in robust scenes without imposing 20 ms USB delays.
        config["signal_perturbations"]["max_delay_steps"] = 0
    if plan.get("cross_asset_source_manifest_sha256"):
        config["cross_asset_source_manifest_sha256"] = plan["cross_asset_source_manifest_sha256"]
    return config
