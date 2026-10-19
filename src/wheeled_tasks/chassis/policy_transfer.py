"""Explicit actor ABI and learning-state migrations with provenance validation."""
import torch


REFERENCE_SOURCE = "encoders_imu_command_reference36"
LEGACY_SOURCE = "scut35_encoders_imu_commands"


def is_reference_migration(source, target):
    return (source.get("actor_observation_source") == LEGACY_SOURCE
            and target.get("actor_observation_source") == REFERENCE_SOURCE
            and target.get("actor_migration") == "scut35_to_reference36_zero_new_columns")


def transfer_actor_state(state, source, target, *, response_controls=None):
    if target.get("manual_context35"):
        for key in ("actor_dim", "actor_frame_dim", "history_length", "actor_observation_source", "actor_layout",
                    "action_dim", "policy_action_order", "v5_control", "policy_dt", "control_math_source", "asset_manifest_sha256"):
            if source.get(key) != target.get(key):
                if key == "control_math_source" and response_controls is not None:
                    from .response_migration import RESPONSE_MIGRATION, validate_response_controls
                    if target.get("response_migration") != RESPONSE_MIGRATION:
                        raise ValueError("Changed control identity requires a named response migration")
                    validate_response_controls(*response_controls)
                    continue
                raise ValueError(f"35D continuation changes the actor/control ABI: {key}")
        if state["mlp.0.weight"].shape != (256, 35) or target["critic_dim"] != 81:
            raise ValueError("Continuation requires the original35D actor and81D critic")
        if source.get("physics_dt") != target.get("physics_dt"):
            clocks = (source.get("physics_dt"), target.get("physics_dt"), target.get("actor_migration"))
            if clocks not in ((.005, .001, "scut35_explicit_1khz_clock_transfer"),
                              (.001, .005, "scut35_explicit_200hz_clock_transfer")):
                raise ValueError("Unapproved control-clock transfer")
        return state, {"source_dim": 35, "target_dim": 35, "actor_parameters": "unchanged",
                       "physics_dt_old": source.get("physics_dt"), "physics_dt_new": target["physics_dt"],
                       "critic_optimizer": "fresh_at_stage_transfer"}
    if not is_reference_migration(source, target):
        if source["actor_dim"] != target["actor_dim"]:
            raise ValueError("Actor dimension change requires an explicit observation migration")
        if target.get("actor_observation_source") == REFERENCE_SOURCE:
            for key in ("actor_frame_dim", "history_length", "actor_observation_source", "actor_layout",
                        "action_dim", "policy_action_order", "v5_control", "policy_dt", "physics_dt",
                        "asset_manifest_sha256", "control_math_source"):
                if source.get(key) != target.get(key):
                    raise ValueError(f"Reference actor interface mismatch: {key}")
        return state, None
    if (source["actor_dim"], source["actor_frame_dim"], source["history_length"],
            target["actor_dim"], target["actor_frame_dim"], target["history_length"]) != (35, 35, 1, 36, 36, 1):
        raise ValueError("Only single-frame 35D to 36D command-reference migration is supported")
    for key in ("action_dim", "policy_action_order", "v5_control", "policy_dt", "physics_dt", "control_math_source",
                "asset_manifest_sha256", "task_modes", "phases"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Reference migration changes the mechanical/control interface: {key}")
    original = state["mlp.0.weight"]
    if original.shape != (256, 35):
        raise ValueError("Reference migration requires the declared 256/128/64 actor")
    result = {key: value.clone() for key, value in state.items()}
    weight = original.new_zeros((256, 36))
    weight[:, :35] = original
    weight[:, 29:32] = 0.
    result["mlp.0.weight"] = weight
    return result, {"source_dim": 35, "target_dim": 36, "zero_initialized_columns": [29, 30, 31, 35],
                    "exploration_parameters": "preserved", "critic_optimizer": "fresh"}


def validate_physics_resume(source, target):
    """Authorize one explicit clock migration without relabeling the checkpoint."""
    if (source.get("physics_dt"), target.get("physics_dt"), target.get("actor_migration")) != (
            .001, .005, "scut35_explicit_200hz_clock_transfer"):
        raise ValueError("Learning-state migration only supports declared1kHz-to200Hz continuation")
    if not target.get("manual_context35") or (target["actor_dim"], target["critic_dim"]) != (35, 81):
        raise ValueError("Physics resume requires the original35/81 learning architecture")
    allowed = {"physics_dt", "actor_migration", "evaluation", "training_reference"}
    changed = sorted(key for key in source.keys() | target.keys() if source.get(key) != target.get(key))
    unexpected = set(changed) - allowed
    if unexpected:
        raise ValueError(f"Physics resume also changes training semantics: {sorted(unexpected)}")
    for key in ("cases", "promotion_case_names", "survival_rate_min", "height_mae_m_max",
                "velocity_mae_m_s_max", "yaw_mae_rad_s_max", "tilt_max_deg", "stand_drift_m_max",
                "stand_velocity_mae_m_s_max", "mechanical_checks", "episodes_per_case", "seed", "confirmation_seed"):
        if source["evaluation"].get(key) != target["evaluation"].get(key):
            raise ValueError(f"Physics resume changes acceptance criteria: {key}")
    return {"kind": "explicit_physics_clock_resume", "source_physics_dt": .001,
            "target_physics_dt": .005, "policy_dt": target["policy_dt"],
            "changed_fields": changed, "actor_critic_optimizer": "preserved",
            "environment_state": "new_episodes_not_mid_rollout_restore"}


def formal_stage_load_kwargs(contract, checkpoint_infos, *, stage):
    """Select first-handoff Adam and the explicitly reviewed critic treatment."""
    selection = contract.get("formal_stage_optimizer")
    from .scut_reward_selection import V7_FINAL_ROUGH_PROFILE
    if contract.get("reward_profile") == V7_FINAL_ROUGH_PROFILE and selection is not None:
        from .training_profile import FINAL_ROUGH_OPTIMIZER, validate_final_rough_stage_selection
        validate_final_rough_stage_selection(contract)
        if selection != FINAL_ROUGH_OPTIMIZER or stage != "terrain":
            raise ValueError("V7 final rough finetune requires its explicit terrain optimizer")
        if checkpoint_infos.get("stage") == "terrain":
            return {}
        if checkpoint_infos.get("stage") != "foundation":
            raise ValueError("V7 final rough finetune requires an authenticated foundation parent")
        return {"load_cfg": {"actor": True, "critic": True, "optimizer": False,
                             "iteration": False, "rnd": False}, "strict": True}
    if selection is None:
        return {}
    if (selection != "scut_finetune_v1"
            or contract.get("formal_v7_flat") is not True
            or contract.get("formal_learning_state_resume") is not True
            or contract.get("training_profile") != "v6_78d_scratch_v1"
            or contract.get("curriculum_stage") != stage):
        raise ValueError("reference stage finetune requires the explicit formal 78D rough handoff")
    source_stage = checkpoint_infos.get("stage")
    adaptation = None
    if "formal_critic_adaptation" in contract:
        from .critic_adaptation import selection as critic_selection, remaining_updates
        adaptation = critic_selection(contract)
        remaining_updates(contract, checkpoint_infos)
    if stage == "foundation":
        from .flat_episode_selection import validate_scut_flat_recovery
        from .training_profile import formal_recovery_stage_transition
        if (source_stage in ("terrain", "foundation")
                and formal_recovery_stage_transition(contract, "terrain", "foundation")
                and validate_scut_flat_recovery(contract) is not None):
            # This retained marker records the earlier rough initialization;
            # authenticated scripted recovery and its restarts inherit Adam.
            return {}
        raise ValueError("Retained finetune marker requires explicit authenticated plane recovery")
    if stage != "terrain":
        raise ValueError("reference stage finetune marker is unsupported for this stage")
    if source_stage == "terrain":
        # The declaration remains in the checkpoint contract. It must never
        # reset Adam again at an assessment restart within the same stage.
        return {}
    if (source_stage != "foundation" or contract.get("formal_stage_transition")
            != {"from": "foundation", "to": "terrain", "reset_stage_actor_clock": True}):
        raise ValueError("reference stage finetune requires an authenticated foundation parent")
    return {"load_cfg": {"actor": True, "critic": adaptation is None, "optimizer": False,
                         "iteration": False, "rnd": False}, "strict": True}


def formal_resume_rng_selection(contract, checkpoint_infos, *, stage, mode, seed):
    """Start a new course stream once; cold restarts retain its saved stream."""
    if mode not in ("inherit", "fresh_fork"):
        raise ValueError("Unknown formal resume RNG mode")
    from .small_rough_course import KEY, MOTION_VERSION, selection
    settings = selection(contract) if KEY in contract else None
    if settings is None or settings["version"] != MOTION_VERSION:
        if mode != "inherit":
            raise ValueError("Fresh formal RNG requires the explicit motion-course handoff")
        return {"mode": mode, "phase": "existing_resume"}
    load = formal_stage_load_kwargs(contract, checkpoint_infos, stage=stage)
    entering = bool(load)
    expected = "fresh_fork" if entering else "inherit"
    if mode != expected:
        raise ValueError(f"Motion-course {'entry' if entering else 'restart'} requires RNG mode {expected}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Motion-course sampling seed must be uint32")
    if not entering and (checkpoint_infos.get("sampling_seed") != seed
            or checkpoint_infos.get("terrain_geometry_seed") != settings["geometry_seed"]):
        raise ValueError("Motion-course restart must retain saved sampling and geometry seeds")
    return {"mode": mode, "phase": "new_course_entry" if entering else "course_restart",
            "geometry_seed": settings["geometry_seed"], "sampling_seed": seed,
            "sampling_seed_scope": "Torch, CUDA, environment and communication streams",
            "physical_episodes": "recreated_not_live_pause_restore"}


def verify_learning_state_restore(algorithm, checkpoint, *, fresh_optimizer_learning_rate=None,
                                  fresh_critic_state=None):
    """Check exact policy weights and either restored Adam or declared fresh Adam."""
    def compare(actual, expected, path):
        if isinstance(expected, torch.Tensor):
            equal = isinstance(actual, torch.Tensor) and torch.equal(actual.detach().cpu(), expected.cpu())
        elif isinstance(expected, dict):
            equal = isinstance(actual, dict) and actual.keys() == expected.keys()
            if equal:
                for key in expected:
                    compare(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            equal = isinstance(actual, (list, tuple)) and len(actual) == len(expected)
            if equal:
                for i, (a, b) in enumerate(zip(actual, expected)):
                    compare(a, b, f"{path}[{i}]")
        else:
            equal = actual == expected
        if not equal:
            raise ValueError(f"Learning-state restore mismatch: {path}")

    restored = algorithm.save()
    compare(restored["actor_state_dict"], checkpoint["actor_state_dict"], "actor_state_dict")
    compare(restored["critic_state_dict"],
        checkpoint["critic_state_dict"] if fresh_critic_state is None else fresh_critic_state,
        "critic_state_dict")
    if fresh_critic_state is not None and fresh_optimizer_learning_rate is None:
        raise ValueError("Fresh critic initialization requires a fresh optimizer")
    if fresh_optimizer_learning_rate is not None:
        # Actor equality includes its distribution/std parameters. The fresh
        # optimizer must not retain moments or steps from the parent stage.
        expected_rate = float(fresh_optimizer_learning_rate)
        groups = algorithm.optimizer.param_groups
        if (not expected_rate > 0 or not torch.isfinite(torch.tensor(expected_rate))
                or restored["optimizer_state_dict"]["state"] or not groups
                or algorithm.learning_rate != expected_rate
                or any(group.get("lr") != expected_rate for group in groups)):
            raise ValueError("Stage finetune requires empty Adam state and the configured initial learning rate")
        receipt = {"actor_exact": True, "critic_exact": fresh_critic_state is None, "optimizer_exact": False,
                "optimizer_initialization": "fresh_on_foundation_to_terrain_only",
                "optimizer_state_entries": 0, "initial_learning_rate": expected_rate}
        if fresh_critic_state is not None:
            receipt["critic_initialization"] = "verified_fresh_before_parent_load"
        return receipt
    compare(restored["optimizer_state_dict"], checkpoint["optimizer_state_dict"], "optimizer_state_dict")
    return {"actor_exact": True, "critic_exact": True, "optimizer_exact": True}


def validate_reward_resume(source, target):
    """Permit the named precision repair, retaining the physical and actor ABI."""
    if target.get("continuation"):
        from .continuation import validate_resume
        return validate_resume(source, target)
    if target.get("remediation"):
        from .remediation import validate_remediation_resume
        return validate_remediation_resume(source, target)
    if (target.get("reward_migration") != "scut35_precision_reward_v1"
            or not target.get("manual_context35")
            or (target.get("actor_dim"), target.get("critic_dim")) != (35, 81)
            or (source.get("physics_dt"), target.get("physics_dt")) not in ((.001, .005), (.005, .005))):
        raise ValueError("Reward resume requires the declared35/81 precision repair at200Hz")
    allowed = {"physics_dt", "actor_migration", "evaluation", "training_reference", "precision_tracking",
               "reward_migration", "resume_critic_warmup_updates", "scene_groups",
               "behavior_pool_membership", "behavior_pool_fractions", "protected_rehearsal"}
    changed = sorted(key for key in source.keys() | target.keys() if source.get(key) != target.get(key))
    if set(changed) - allowed:
        raise ValueError(f"Reward resume changes undeclared semantics: {sorted(set(changed) - allowed)}")
    for key in ("cases", "promotion_case_names", "survival_rate_min", "height_mae_m_max",
                "velocity_mae_m_s_max", "yaw_mae_rad_s_max", "tilt_max_deg", "stand_drift_m_max",
                "stand_velocity_mae_m_s_max", "mechanical_checks", "episodes_per_case", "seed", "confirmation_seed"):
        if source["evaluation"].get(key) != target["evaluation"].get(key):
            raise ValueError(f"Reward resume changes acceptance criteria: {key}")
    return {"kind": "explicit_precision_reward_resume", "changed_fields": changed,
            "source_physics_dt": source["physics_dt"], "target_physics_dt": target["physics_dt"],
            "actor_critic_optimizer": "preserved", "environment_state": "new_episodes"}


def resume_budget(infos, consumed_updates, batch_transitions):
    """Charge discarded optimization work without pretending its weights survived."""
    loaded = int(infos["successful_updates_total"])
    spent = loaded if consumed_updates is None else consumed_updates
    if type(spent) is not int or spent < loaded:
        raise ValueError("Consumed updates cannot precede the restored checkpoint")
    transitions = int(infos.get("training_transitions", 0))
    saved_batch = infos.get("batch_transitions")
    compatible_batch = (saved_batch == batch_transitions if saved_batch is not None
                        else transitions == loaded * batch_transitions)
    if spent > loaded and not compatible_batch:
        raise ValueError("Rollback budget rebasing requires the same historical batch size")
    return {"parent_updates": spent, "restored_checkpoint_updates": loaded,
            "parent_training_transitions": transitions + (spent - loaded) * batch_transitions,
            "learning_lineage_updates": int(infos.get("learning_lineage_updates", loaded)),
            "discarded_updates_charged": spent - loaded}


def validate_budget_resume(source, target):
    """Extend a frozen stage budget without weakening its model or acceptance ABI."""
    if (target.get("budget_migration") != "scut35_budget_retry_v1"
            or not target.get("manual_context35")
            or (target.get("actor_dim"), target.get("critic_dim"), target.get("physics_dt")) != (35, 81, .005)
            or target["total_updates"] < source["total_updates"]):
        raise ValueError("Budget resume requires the declared35/81 nondecreasing200Hz budget")
    allowed = {"budget_migration", "total_updates", "stages", "evaluation", "training_reference"}
    changed = sorted(k for k in source.keys() | target.keys() if source.get(k) != target.get(k))
    if set(changed) - allowed:
        raise ValueError(f"Budget resume changes nonbudget semantics: {sorted(set(changed) - allowed)}")
    if len(source["stages"]) != len(target["stages"]):
        raise ValueError("Budget resume cannot change stage identities")
    for old, new in zip(source["stages"], target["stages"]):
        if ({k: v for k, v in old.items() if k != "updates"}
                != {k: v for k, v in new.items() if k != "updates"} or new["updates"] < old["updates"]):
            raise ValueError("Budget resume changes stage semantics")
    operational = {"block_updates", "regression_patience", "regression_recovery"}
    before = {k: v for k, v in source["evaluation"].items() if k not in operational}
    after = {k: v for k, v in target["evaluation"].items() if k not in operational}
    if before != after:
        raise ValueError("Budget resume changes acceptance criteria or evaluation coverage")
    return {"kind": "explicit_budget_resume", "changed_fields": changed,
            "source_update_ceiling": source["total_updates"], "target_update_ceiling": target["total_updates"],
            "actor_critic_optimizer": "preserved", "acceptance": "unchanged"}
