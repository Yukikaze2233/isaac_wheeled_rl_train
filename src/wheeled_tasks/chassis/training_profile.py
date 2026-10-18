"""Versioned runner settings and explicit diagnostic RNG ownership.

Legacy workers keep their existing defaults. A new profile must be requested by
name; dimensions and physical/control clocks cannot silently drift.
"""
from copy import deepcopy
from collections.abc import Mapping
import hashlib
import json
import math


PROFILE = "v6_78d_scratch_v1"
CRITIC_VERSION = "v6_scut78_v1"
# Bounds in the installed RSL5 PPO KL scheduler, not a cap at the initial LR.
ADAPTIVE_LR_BOUNDS = (1e-5, 1e-2)
ALGORITHM = {
    "class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4,
    "learning_rate": 1e-4, "schedule": "adaptive", "gamma": .99,
    "lam": .95, "clip_param": .2, "entropy_coef": .005,
    "value_loss_coef": 4., "use_clipped_value_loss": True,
    "desired_kl": .01, "max_grad_norm": 1.,
    "normalize_advantage_per_mini_batch": False,
}
FINAL_ROUGH_STAGE_VERSION = "v7_final_rough_stage_v1"
FINAL_TERRAIN_STAGE_VERSION = "v7_final_rough_terrain_stage_v1"
FINAL_ROUGH_OPTIMIZER = "v7_final_rough_finetune_v1"
FINAL_RUNNER_SOURCE_COMMIT = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
FINAL_RUNNER_REFERENCE = "references/v7_final_runner_b8ff79f3"
GAS_PRESSURE_APPLICATION = "main_link_mounts_v1"
GAS_PRESSURE_ASSET_SHA256 = "088e699266fa3dc666379a911b23a30e2038ab3d7f113cab4dcd4bdd0a14279b"
FOUNDATION_CONTACT_LINKS_BEFORE = ("base_link", "L_link1", "R_link1", "LL_link1", "RR_link1")
FOUNDATION_CONTACT_LINKS_COMPLETE = (
    "base_link", "L_link1", "L_link2", "R_link1", "R_link2", "LL_link1", "LL_link2",
    "LL_link3", "LL_link4", "RR_link1", "RR_link2", "RR_link3", "RR_link4")


def effective_profile_algorithm(contract):
    """Select the registered final rough runner without changing old defaults."""
    from .scut_reward_selection import V7_FINAL_ROUGH_PROFILE
    result = deepcopy(ALGORITHM)
    if contract.get("reward_profile") == V7_FINAL_ROUGH_PROFILE:
        result["value_loss_coef"] = 2.
    return result


def uses_profile(contract):
    return contract.get("training_profile") == PROFILE


def validate_profile_selection(contract):
    """Bind the privileged layout and runner profile in both directions.

    Evaluation uses the same contract layout with a smaller native batch. PPO
    authorization and actual training batch size belong to the worker CLI.
    """
    if "formal_physics_trace_substeps" in contract:
        ticks = contract["formal_physics_trace_substeps"]
        if (type(ticks) is not list or not 0 < len(ticks) <= 64
                or any(type(tick) is not int or tick <= 0 for tick in ticks)
                or any(left >= right for left, right in zip(ticks, ticks[1:]))):
            raise ValueError("Formal physics trace substeps require 1-64 strictly ascending positive integer ticks")
    validate_gas_pressure_application(contract)
    validate_physx_solver_selection(contract)
    if "formal_contact_generation_distance" in contract:
        from .contact_generation_distance import selected
        selected(contract)
    from .fixed_flat_rehearsal import validate_dynamic_nonsteady_birth
    validate_dynamic_nonsteady_birth(contract)
    if "round_trip_transport" in contract:
        from .pc_communication import validate_pc_communication
        validate_pc_communication(contract)
    capacity_key = "gpu_found_lost_aggregate_pairs_capacity"
    if capacity_key in contract:
        capacity = contract[capacity_key]
        if (contract.get("formal_v7_flat") is not True or not uses_profile(contract)
                or type(capacity) is not int or not 0 < capacity <= 2**32 - 1):
            raise ValueError("GPU aggregate pair capacity requires a positive uint32 on formal V7")
    if "formal_flat_rehearsal" in contract:
        # Keep the existing frozen default independent of this opt-in module.
        from .fixed_flat_rehearsal import (validate_fixed_flat_rehearsal,
            FLAT_REFRESH_VERSION, FLAT_MIXED_REFRESH_VERSION)
        rehearsal = validate_fixed_flat_rehearsal(contract)
        from .scut_reward_selection import SCUT_ROUGH_PROFILES
        rough = (contract.get("reward_profile") in SCUT_ROUGH_PROFILES
                 and contract.get("curriculum_stage") == "terrain"
                 and contract.get("enabled_stages") == ["terrain"])
        flat_refresh = contract.get("formal_flat_refresh") in (FLAT_REFRESH_VERSION, FLAT_MIXED_REFRESH_VERSION)
        recovery = False
        if not (rough or flat_refresh):
            from .flat_episode_selection import validate_scut_flat_recovery
            recovery = (validate_scut_flat_recovery(contract) is not None
                        and contract.get("formal_terrain") == {
                            "type": "plane", "source": "scut0719", "env_spacing_m": 4.})
        if (contract.get("formal_v7_flat") is not True
                or contract.get("formal_learning_state_resume") is not True
                or not uses_profile(contract)
                or not (rough or recovery or flat_refresh)
                or "flat_episode_runtime" not in contract):
            raise ValueError("Fixed flat rehearsal requires formal rough, flat refresh or scripted plane recovery")
        if rehearsal["version"] == "v7_source0719_plane_rehearsal_v2" and not rough:
            raise ValueError("Source flat rehearsal is an explicit rough-stage course repair only")
    from .scut_reward_selection import validate_reward_selection
    validate_reward_selection(contract)
    from .flat_episode_selection import validate_flat_episode_selection
    validate_flat_episode_selection(contract)
    from .flat_interval_events import validate_interval_selection
    validate_interval_selection(contract)
    name = contract.get("training_profile")
    if name is not None and name != PROFILE:
        raise ValueError("Unknown next-generation training profile")
    # These owners are instantiated only with the new privileged runtime. A
    # legacy scene must not author source materials yet silently skip sampling.
    for key in ("wheel_material_priors", "pc_relative_priors"):
        if key not in contract:
            continue
        settings = contract[key]
        if not isinstance(settings, Mapping) or type(settings.get("enabled", False)) is not bool:
            raise ValueError(f"{key} requires mapping settings and boolean selection")
        if settings.get("enabled") and not uses_profile(contract):
            raise ValueError(f"{key} requires the named 78D training profile")
        if key == "pc_relative_priors" and settings.get("enabled"):
            if settings.get("version") != "pc_relative_priors_v1":
                raise ValueError("PC DR requires a versioned prior before native construction")
        if key == "wheel_material_priors":
            from .privileged_runtime import ContactMaterialPolicy
            ContactMaterialPolicy.from_config(contract,
                legacy_contact_domain_active=bool(contract.get("contact_domain", {}).get("enabled")))
    if (contract.get("critic_dim") == 78
            or contract.get("critic_observation_version") == CRITIC_VERSION) and not uses_profile(contract):
        raise ValueError("78D critic requires the named training profile")
    if uses_profile(contract):
        validate_profile(contract)
    return uses_profile(contract)


def validate_gas_pressure_application(contract):
    """Admit the mount-load implementation only on its authenticated asset."""
    if "gas_pressure_application" not in contract:
        return None
    solver = contract.get("physx_solver_type")
    external = contract.get("physx_external_forces_every_iteration")
    if type(external) is not bool or (type(solver) is int and solver == 0 and external):
        raise ValueError("Mount gas pressure requires a boolean force-iteration flag; PGS requires False")
    required = {
        "gas_pressure_application": GAS_PRESSURE_APPLICATION,
        "formal_v7_flat": True, "training_profile": PROFILE,
        "actor_dim": 35, "critic_dim": 78, "action_dim": 6,
        "physics_dt": .005, "policy_dt": .02,
        "position_iterations": 12, "velocity_iterations": 6,
        "asset_manifest_sha256": GAS_PRESSURE_ASSET_SHA256,
        "formal_closure_monitor_only": True, "closure_gap_termination_m": None,
    }
    if any(key not in contract or contract[key] != value
            or isinstance(contract[key], bool) != isinstance(value, bool)
            for key, value in required.items()):
        raise ValueError("Mount gas pressure requires its frozen 35D/78D 200Hz asset and monitor-only physics")
    pressure = contract.get("v5_control", {}).get("spring_pressure_mpa")
    if type(pressure) not in (int, float) or pressure != 10.:
        raise ValueError("Mount gas pressure must retain the authenticated 10MPa curve")
    return GAS_PRESSURE_APPLICATION


def validate_physx_solver_selection(contract):
    """Keep historical defaults and bind explicit solvers to inherited physics."""
    key = "physx_solver_type"
    if key not in contract:
        return None
    selected = contract[key]
    if type(selected) is not int or selected not in (0, 1):
        raise ValueError("physx_solver_type requires integer 0 (PGS) or 1 (TGS)")
    if "formal_final_rough_stage" in contract:
        validate_final_rough_stage_selection(contract)
        if (selected != 1 or contract.get("physx_external_forces_every_iteration") is not False
                or any(key in contract for key in ("formal_scripted_recovery_mix", "recovery_training"))
                or validate_gas_pressure_application(contract) != GAS_PRESSURE_APPLICATION):
            raise ValueError("Explicit final rough requires inherited TGS/False mount-pressure physics")
        return selected
    if (contract.get("reward_profile") == "scut0730_rough_v1"
            and contract.get("formal_terrain") == {"type": "rough", "source": "scut0730"}
            and contract.get("curriculum_stage") == "terrain"
            and contract.get("enabled_stages") == ["terrain"]):
        mixed = False
        if "formal_small_rough_mix" in contract:
            from .small_rough_course import selection
            mixed = selection(contract) is not None
        if (selected != 1 or contract.get("physx_external_forces_every_iteration") is not False
                or not mixed and any(key in contract for key in ("formal_scripted_recovery_mix", "recovery_training"))
                or validate_gas_pressure_application(contract) != GAS_PRESSURE_APPLICATION
                or contract.get("scut_reward_mapping", {}).get("undesired_link_names")
                   != list(FOUNDATION_CONTACT_LINKS_COMPLETE)):
            raise ValueError("Explicit small rough requires inherited TGS/False, mount pressure and complete contact links")
        return selected
    scripted = "formal_scripted_recovery_mix" in contract or "recovery_training" in contract
    if scripted:
        from .fixed_flat_rehearsal import validate_scripted_flat_mix
        if (selected != 1 or contract.get("physx_external_forces_every_iteration") is not False
                or validate_scripted_flat_mix(contract) is None
                or contract.get("scut_reward_mapping", {}).get("undesired_link_names")
                   != list(FOUNDATION_CONTACT_LINKS_COMPLETE)):
            raise ValueError("Explicit recovery requires the inherited TGS/False direct mix and complete contact list")
    if (validate_gas_pressure_application(contract) != GAS_PRESSURE_APPLICATION
            or contract.get("curriculum_stage") != "foundation"
            or contract.get("enabled_stages") != ["foundation"]
            or "formal_stage_optimizer" in contract):
        raise ValueError("Explicit solver selection requires the authenticated mount-pressure foundation")
    return selected


def validate_profile(contract):
    if not uses_profile(contract):
        raise ValueError("Unknown next-generation training profile")
    required = {"actor_dim": 35, "actor_frame_dim": 35, "critic_dim": 78,
                "action_dim": 6, "history_length": 1, "physics_dt": .005,
                "policy_dt": .02, "position_iterations": 12, "velocity_iterations": 6,
                "num_steps_per_env": 24, "target_num_envs": 4096}
    for key, value in required.items():
        if contract.get(key) != value or isinstance(contract.get(key), bool):
            raise ValueError(f"Next-generation profile mismatch: {key}")
    # Both historical monitor-only checkpoints and the user-selected 3 mm
    # lane-reset policy remain readable. Resume separately binds any change.
    if contract.get("formal_closure_monitor_only") is True:
        if contract.get("formal_v7_flat") is not True or contract.get("closure_gap_termination_m") is not None:
            raise ValueError("Formal closure monitoring requires the gap termination to be disabled")
    elif contract.get("closure_gap_termination_m") != .003:
        raise ValueError("Next-generation profile mismatch: closure_gap_termination_m")
    if "formal_rough_initial_level" in contract:
        level = contract["formal_rough_initial_level"]
        from .scut_reward_selection import V7_FINAL_ROUGH_PROFILE, V7_FINAL_TERRAIN_SOURCE
        final_geometry = (contract.get("reward_profile") == V7_FINAL_ROUGH_PROFILE
                          and contract.get("formal_terrain") == {"type": "rough", "source": V7_FINAL_TERRAIN_SOURCE})
        if (type(level) is not int or not 0 <= level <= 9
                or contract.get("formal_v7_flat") is not True
                or not final_geometry and contract.get("formal_terrain") != {"type": "rough", "source": "scut0730"}):
            raise ValueError("Formal rough initial level requires an integer from 0 to 9 on reference rough")
    if "formal_stage_optimizer" in contract and (
            contract["formal_stage_optimizer"] not in ("scut_finetune_v1", FINAL_ROUGH_OPTIMIZER)
            or contract.get("formal_v7_flat") is not True):
        raise ValueError("Formal stage optimizer requires scut_finetune_v1")
    if ("formal_final_rough_stage" in contract
            or contract.get("formal_stage_optimizer") == FINAL_ROUGH_OPTIMIZER
            or contract.get("formal_terrain", {}).get("source") == "v7_final_b8ff79f3"):
        validate_final_rough_stage_selection(contract)
    if contract.get("critic_observation_version") != CRITIC_VERSION:
        raise ValueError("78D profile requires the explicit critic field schema")
    if (contract.get("actor_observation_source") not in (
            "scut35_encoders_imu_commands", "encoders_imu_command_reference36")
            or contract.get("manual_context35") is not True):
        raise ValueError("78D profile requires the frozen clean35 command-context interface")
    for key in ("remediation", "continuation", "task_campaign"):
        if contract.get(key):
            raise ValueError(f"78D profile cannot use a legacy experiment identity: {key}")
    for key in ("jump_assist", "step_assist"):
        if contract.get(key, {}).get("enabled", False):
            raise ValueError("Scratch profile forbids external jump assistance")
    # Reject contradictory declared values before the final profile overrides.
    # A missing optional value means the named profile owns that value.
    declared = {key: value for key, value in effective_profile_algorithm(contract).items() if key != "class_name"}
    declared["learning_rate_schedule"] = declared.pop("schedule")
    declared["initial_noise_std"] = 1.
    for key, value in declared.items():
        if key in contract and (contract[key] != value
                or isinstance(contract[key], bool) != isinstance(value, bool)):
            raise ValueError(f"78D profile PPO mismatch: {key}")
    return required


def profile_source_files(contract):
    from .scut_reward_selection import SCUT_SOURCE_PROFILES, reward_source_files
    if not validate_profile_selection(contract):
        return reward_source_files(contract) if contract.get("reward_profile") in SCUT_SOURCE_PROFILES else []
    sources = [f"src/wheeled_tasks/chassis/{name}.py" for name in (
        "training_profile", "critic_observation", "privileged_runtime", "pc_parameters",
        "env", "root_kinematics", "scut_observation", "v5_control", "control_contract",
        "preflight", "actuator_response", "task", "training_schedule", "terrain_runout", "rewards",
        "flat_episode_selection", "flat_interval_events")]
    if contract.get("formal_v7_flat") is True:
        sources.append("src/wheeled_tasks/chassis/com_motion_measurement.py")
    if "formal_contact_generation_distance" in contract:
        sources.append("src/wheeled_tasks/chassis/contact_generation_distance.py")
    if "formal_small_rough_mix" in contract:
        sources.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in (
            "small_rough_course", "temporal_flat_rehearsal"))
    sources.extend(("src/wheeled_tasks/v40/core.py", "src/wheeled_tasks/v40/contract.py",
                    "src/wheeled_tasks/agents/v40_ppo_cfg.py"))
    sources.extend(reward_source_files(contract))
    if validate_gas_pressure_application(contract) is not None:
        sources.append("src/wheeled_tasks/chassis/spring_force_element.py")
    if contract.get("v5_control", {}).get("leg_winding_version") in ("paired_common_v1", "paired_branch107_v1"):
        sources.append("src/wheeled_algo/leg_winding_export.py")
    if contract.get("v5_control", {}).get("leg_winding_version") == "paired_branch107_v1":
        sources.extend(("src/wheeled_tasks/chassis/policy_transfer.py",
                        "src/wheeled_algo/recovery_rollout.py"))
    from .scut_reward_selection import V7_FINAL_ROUGH_PROFILE
    if contract.get("reward_profile") == V7_FINAL_ROUGH_PROFILE:
        sources.extend(f"{FINAL_RUNNER_REFERENCE}/{name}" for name in (
            "task_registration.py", "ppo_cfg.py", "train_finetune.py", "source_manifest.json", "LICENSE"))
    if "formal_stepup_command" in contract:
        sources.append("src/wheeled_tasks/chassis/stepup_command.py")
    if "formal_flat_rehearsal" in contract:
        sources.extend(("src/wheeled_tasks/chassis/fixed_flat_rehearsal.py",
                        contract["formal_flat_rehearsal"]["cases_file"]))
    if contract.get("formal_terrain"):
        sources.append("src/wheeled_tasks/chassis/formal_terrain.py")
        if contract["formal_terrain"].get("type") == "rough":
            sources.extend(("src/wheeled_tasks/chassis/scut_rough_terrain.py",
                "src/wheeled_tasks/chassis/scut_rough_support.py",
                "src/wheeled_tasks/chassis/rough_commands.py",
                "references/scut_curriculum_b8ff79f3/pretrained/26_infantry/rough_rotation_stair/2026-07-30_09-16-27/params/env.yaml",
                "references/scut_curriculum_b8ff79f3/source/agent_world/agent_world/terrains/height_field.py"))
            if contract["formal_terrain"].get("source") == "v7_final_b8ff79f3":
                from .final_rough_definition import final_rough_source_files
                sources.extend(final_rough_source_files())
                sources.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in (
                    "final_rough_definition", "final_rough_terrain", "final_rough_commands", "final_rough_birth"))
    from .flat_episode_selection import flat_episode_source_files
    sources.extend(flat_episode_source_files(contract))
    if "formal_dynamic_nonsteady_birth" in contract or "final_rough_birth_adapter" in contract:
        sources.append("src/wheeled_tasks/chassis/source_birth_exposure.py")
    from .flat_episode_selection import validate_scut_flat_recovery
    if validate_scut_flat_recovery(contract) is not None:
        sources.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in (
            "recovery", "recovery_training", "recovery_evidence", "recovery_observer",
            "recovery_curriculum", "recovery_prefix_curriculum", "continuous_encoders",
            "mechanism_observer", "step_trigger", "response_migration"))
        sources.extend(contract["recovery_training"][key] for key in ("profiles_file", "height_lookup_file"))
        if "direct_takeover" in contract["recovery_training"]:
            sources.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in (
                "recovery_sensor_guard", "recovery_sensor_cache"))
    for key in ("nonstandard_start", "ordinary_posture_curriculum"):
        if key in contract:
            sources.append(f"src/wheeled_tasks/chassis/{key}.py")
    if "ordinary_posture_curriculum" in contract:
        sources.append("src/wheeled_tasks/chassis/nonstandard_start.py")
    if contract.get("signal_perturbations", {}).get("enabled"):
        sources.append("src/wheeled_tasks/chassis/robustness.py")
    if "round_trip_transport" in contract:
        sources.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in
            ("round_trip_transport", "feedback_transport", "feedback_transport_config",
             "pc_communication", "delay_buffer"))
    if contract.get("nonstandard_start", {}).get("enabled"):
        sources.append("src/wheeled_algo/recovery_rollout.py")
        if contract["nonstandard_start"].get("profiles_file"):
            sources.append(contract["nonstandard_start"]["profiles_file"])
    if contract.get("step_jump_task", {}).get("enabled"):
        sources.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in (
            "step_jump", "step_climb", "step_sensors", "mechanism_observer", "recovery_observer", "step_trigger"))
    if contract.get("skill_specs"):
        sources.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in (
            "skill_commands", "skill_curriculum", "integrated_curriculum",
            "locomotion_difficulty", "release_stop", "ordinary_large_stair"))
    if contract.get("diagnostic_logging"):
        sources.append("src/wheeled_tasks/chassis/diagnostic_curriculum.py")
    return list(dict.fromkeys(sources))


def profile_identity(contract, source_hashes):
    """Bind exact restarts to all captured worker/control/observer sources."""
    if not validate_profile_selection(contract):
        return {}
    from .critic_observation import SCHEMA_VERSION
    missing = set(profile_source_files(contract)) - set(source_hashes)
    if missing or not all(isinstance(v, str) and len(v) == 64
            and all(ch in "0123456789abcdef" for ch in v) for v in source_hashes.values()):
        raise ValueError("78D profile requires complete SHA-256 source provenance")
    encoded = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    result = {"training_profile": PROFILE, "critic_schema_version": SCHEMA_VERSION,
              "training_source_sha256": hashlib.sha256(encoded).hexdigest()}
    from .scut_reward_selection import V7_FINAL_ROUGH_PROFILE
    if contract.get("reward_profile") == V7_FINAL_ROUGH_PROFILE:
        result["effective_algorithm"] = effective_profile_algorithm(contract)
        result["runner_reference_commit"] = FINAL_RUNNER_SOURCE_COMMIT
    return result


def validate_final_rough_stage_selection(contract):
    """One named nominal rough course; checkpoint approval is separately owned."""
    from .fixed_flat_rehearsal import (FIXED17_VERSION, FIXED17_CASE_FILE, FIXED17_CASE_SHA)
    from .scut_reward_selection import V7_FINAL_ROUGH_PROFILE, V7_FINAL_TERRAIN_SOURCE
    stage_version = contract.get("formal_final_rough_stage")
    terrain_settings = {"type": "rough", "source": V7_FINAL_TERRAIN_SOURCE if
        stage_version == FINAL_TERRAIN_STAGE_VERSION else "scut0730"}
    marker = contract.get("formal_stage_transition", {})
    rehearsal = {"version": FIXED17_VERSION, "fraction": .25,
        "cases_file": FIXED17_CASE_FILE, "cases_sha256": FIXED17_CASE_SHA,
        "allocation": "equal_case_lanes", "reward_profile": V7_FINAL_ROUGH_PROFILE}
    if (stage_version not in (FINAL_ROUGH_STAGE_VERSION, FINAL_TERRAIN_STAGE_VERSION)
            or contract.get("formal_stage_optimizer") != FINAL_ROUGH_OPTIMIZER
            or contract.get("formal_v7_flat") is not True
            or contract.get("formal_learning_state_resume") is not True
            or contract.get("training_profile") != PROFILE
            or contract.get("reward_profile") != V7_FINAL_ROUGH_PROFILE
            or contract.get("curriculum_stage") != "terrain"
            or contract.get("enabled_stages") != ["terrain"]
            or contract.get("formal_terrain") != terrain_settings
            or contract.get("formal_foundation_nominal_birth") is not True
            or type(contract.get("formal_rough_initial_level")) is not int or contract["formal_rough_initial_level"] != 0
            or type(contract.get("total_updates")) is not int or contract["total_updates"] != 20000
            or contract.get("formal_flat_rehearsal") != rehearsal
            or marker != {"from": "foundation", "to": "terrain", "reset_stage_actor_clock": True}
            or marker.get("reset_stage_actor_clock") is not True
            or any(key in contract for key in ("formal_flat_refresh", "formal_robustness_revision",
                "formal_dynamic_nonsteady_birth", "v7_curriculum_runtime", "recovery_training"))):
        raise ValueError("V7 final rough requires its explicit nominal fixed17 stage and finetune markers")
    return True


def final_rough_stage_transition(contract, source_stage, target_stage):
    if "formal_final_rough_stage" not in contract:
        return False
    validate_final_rough_stage_selection(contract)
    return source_stage == "foundation" and target_stage == "terrain"


def _validate_final_rough_stage_resume(old, new):
    """Preserve the selected flat lineage or restart this exact final course."""
    from .fixed_flat_rehearsal import FLAT_MIXED_REFRESH_VERSION
    from .scut_reward_selection import SCUT_FLAT_PROFILE
    from .pc_communication import validate_pc_communication
    validate_profile_selection(old)
    validate_profile_selection(new)
    validate_final_rough_stage_selection(new)
    validate_pc_communication(new)
    if (new.get("round_trip_transport", {}).get("enabled") is not True
            or new.get("wheel_material_priors") != {"enabled": True, "version": "v7_low_grip_wheel_buckets_v2"}):
        raise ValueError("V7 final rough requires the inherited shared delay and low-grip materials")
    changed = {key for key in old.keys() | new.keys()
        if (key in old) != (key in new)
        or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
    transition = final_rough_stage_transition(new, old.get("curriculum_stage"), new.get("curriculum_stage"))
    if not transition:
        validate_final_rough_stage_selection(old)
        if changed - {"formal_physics_trace_directory"}:
            raise ValueError("V7 final rough restart must retain the exact course, algorithm and interface")
    else:
        if (old.get("reward_profile") != SCUT_FLAT_PROFILE
                or old.get("formal_flat_refresh") != FLAT_MIXED_REFRESH_VERSION
                or old.get("formal_foundation_nominal_birth") is not True
                or old.get("enabled_stages") != ["foundation"]
                or old.get("formal_terrain") != {"type": "plane", "source": "scut0719", "env_spacing_m": 4.}
                or any(key in old for key in ("formal_final_rough_stage", "formal_stage_optimizer",
                    "formal_stage_transition", "flat_special_mode_iteration_cap", "formal_dynamic_nonsteady_birth"))):
            raise ValueError("V7 final rough requires the existing nominal mixed flat parent contract")
        validate_pc_communication(old)
        if (new["formal_flat_rehearsal"] != {**old["formal_flat_rehearsal"], "reward_profile": new["reward_profile"]}
                or new.get("stages") != [{**item, "name": "terrain", "terrain": ["rough"]} for item in old["stages"]]
                or new.get("scene_groups") != [{"name": "source_rough", "terrain": ["rough"], "fraction": 1.}]
                or new.get("critic_height_reference") != "root_link_world_z"
                or "evaluation" in new):
            raise ValueError("V7 final rough must retain fixed17, update allowance and the explicit rough course")
        policy_keys = {"later_skills_enabled", "first_assessment_update", "no_automatic_iteration_based_promotion",
            "deployment_policy_hz", "deployment_pd_hz", "assessment_interval_updates"}
        prose_keys = {"communication_scope", "terrain_scope", "initialization", "progression", "birth_scope"}
        before, after = old["formal_stage_policy"], new["formal_stage_policy"]
        if (set(after) - policy_keys - prose_keys - {"assessment_mode"}
                or not policy_keys <= set(after)
                or any(json.dumps(before.get(key)) != json.dumps(after[key]) for key in policy_keys)
                or after.get("assessment_mode", "pause_for_assessment") not in (
                    "pause_for_assessment", "record_and_continue")
                or before.get("flat_special_mode_schedule") != "scut0719_pinned_v1"):
            raise ValueError("V7 final rough must retain assessment, promotion and deployment clocks")
        allowed = {"formal_final_rough_stage", "formal_stage_optimizer", "formal_stage_transition",
            "reward_profile", "scut_reward_mapping", "critic_height_reference", "curriculum_stage",
            "enabled_stages", "stages", "scene_groups", "formal_terrain", "formal_flat_refresh",
            "formal_flat_rehearsal", "formal_stage_policy", "formal_rough_initial_level",
            "value_loss_coef", "evaluation", "formal_physics_trace_directory"}
        capacity = "gpu_found_lost_aggregate_pairs_capacity"
        if capacity in changed:
            if type(new.get(capacity)) is not int or new[capacity] <= old.get(capacity, 0):
                raise ValueError("V7 final rough may only increase the native aggregate pair capacity")
            allowed.add(capacity)
        if changed - allowed:
            raise ValueError(f"V7 final rough changes unrelated inherited semantics: {sorted(changed - allowed)}")
    return {"scope": "same_35D_78D_learning_state_with_fresh_native_episodes",
        "formal_final_rough_stage": new["formal_final_rough_stage"], "stage_transition": transition,
        "actor_critic_std": "exact", "optimizer_initialization": "fresh_first_handoff" if transition else "inherit",
        "initial_learning_rate": effective_profile_algorithm(new)["learning_rate"] if transition else None,
        "rng_policy": "inherit", "local_stage_clock": "reset_zero" if transition else "inherit",
        "global_learning_and_paid_consumption": "inherit", "fixed_flat_rehearsal_fraction": .25,
        "changed_contract_fields": sorted(changed), "exact_physics_resume": False,
        "source_final_environment_equivalent": False, "checkpoint_approved": False}


def formal_rough_stage_transition(contract, source_stage, target_stage):
    """One explicit flat-to-rough transition; same-stage resumes never reset clocks."""
    from .scut_reward_selection import SCUT_ROUGH_PROFILE
    return (source_stage == "foundation" and target_stage == "terrain"
            and contract.get("formal_learning_state_resume") is True
            and contract.get("reward_profile") == SCUT_ROUGH_PROFILE
            and contract.get("formal_stage_transition") == {
                "from": "foundation", "to": "terrain", "reset_stage_actor_clock": True})


def formal_recovery_stage_transition(contract, source_stage, target_stage):
    """Explicit rough-to-plane scripted recovery; not a generic stage switch."""
    from .flat_episode_selection import SCUT_FLAT_RECOVERY
    recovery = contract.get("recovery_training", {})
    marker = contract.get("formal_stage_transition", {})
    return (source_stage == "terrain" and target_stage == "foundation"
            and contract.get("formal_learning_state_resume") is True
            and contract.get("reward_profile") == "scut0719_flat_v1"
            and isinstance(recovery, Mapping) and recovery.get("enabled") is True
            and recovery.get("source_flat_handoff") == SCUT_FLAT_RECOVERY
            and isinstance(marker, Mapping) and marker == {
                "from": "terrain", "to": "foundation", "reset_stage_actor_clock": True}
            and marker["reset_stage_actor_clock"] is True)


def _validate_gas_pressure_resume(old, new):
    """Move pressure to the main links without changing the learning task."""
    field = "gas_pressure_application"
    if (field in old or validate_gas_pressure_application(new) != GAS_PRESSURE_APPLICATION
            or new.get("formal_learning_state_resume") is not True
            or any(config.get("curriculum_stage") != "foundation"
                   or config.get("enabled_stages") != ["foundation"]
                   or type(config.get("research_training_authorized")) is not bool
                   or "formal_scripted_recovery_mix" in config
                   for config in (old, new))):
        raise ValueError("Gas pressure migration requires the existing unscripted foundation parent")
    changed = {key for key in old.keys() | new.keys()
               if (key in old) != (key in new)
               or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
    if changed - {field, "research_training_authorized", "formal_physics_trace_directory",
                  "formal_physics_trace_substeps"}:
        raise ValueError("Gas pressure migration cannot change the course, rewards, control, physics clocks or budget")
    previous, current = deepcopy(old), deepcopy(new)
    current.pop(field)
    current["research_training_authorized"] = previous["research_training_authorized"]
    receipt = validate_formal_learning_state_resume(previous, current)
    return {**receipt, field: {
                "before": "slider_effort", "after": GAS_PRESSURE_APPLICATION,
                "pressure_mpa": 10., "slider_damping_N_s_per_m": .002,
                "actor_critic_std_adam_adaptive_lr_rng": "inherit",
                "local_global_and_consumption_clocks": "inherit",
                "native_validation": "required_before_activation"},
            "changed_contract_fields": sorted(changed), "exact_physics_resume": False}


def _validate_physx_solver_resume(old, new):
    """Validate the solver-specific force flag before reusing the original guard."""
    key = "physx_solver_type"
    external_key = "physx_external_forces_every_iteration"
    before = validate_physx_solver_selection(old)
    after = validate_physx_solver_selection(new)
    before = 1 if before is None else before
    after = 1 if after is None else after
    if (before, after) not in ((1, 0), (0, 0), (1, 1)):
        raise ValueError("Solver migration only admits TGS to PGS, retaining PGS thereafter")
    if key not in new or validate_physx_solver_selection(new) is None:
        raise ValueError("An explicit solver selection must be retained on continuation")
    final_restart = all("formal_final_rough_stage" in contract for contract in (old, new))
    if final_restart and ((before, after) != (1, 1)
            or any(contract.get(external_key) is not False for contract in (old, new))):
        raise ValueError("Final rough continuation requires unchanged TGS/False")
    for contract, solver in ((old, before), (new, after)):
        external = contract.get(external_key)
        if type(external) is not bool or (solver == 0 and external):
            raise ValueError("External forces every iteration requires a boolean; PGS requires False")
        if (not validate_profile_selection(contract)
                or contract.get("formal_v7_flat") is not True
                or contract.get("formal_closure_monitor_only") is not True
                or contract.get("closure_gap_termination_m") is not None
                or not final_restart and (
                    contract.get("curriculum_stage") != "foundation"
                    or contract.get("enabled_stages") != ["foundation"]
                    or "recovery_training" in contract
                    or "formal_scripted_recovery_mix" in contract
                    or "formal_stage_optimizer" in contract)):
            raise ValueError("Solver migration requires the unscripted foundation or exact final rough continuation")
    if old[external_key] is False and new[external_key] is True:
        raise ValueError("Force-iteration repair cannot re-enable the per-iteration force flag")
    physics_changed = before != after or old[external_key] != new[external_key]
    changed = {field for field in old.keys() | new.keys()
               if (field in old) != (field in new)
               or json.dumps(old.get(field), sort_keys=True) != json.dumps(new.get(field), sort_keys=True)}
    if physics_changed and changed - {key, "gas_pressure_application",
            external_key, "research_training_authorized", "formal_physics_trace_directory",
            "formal_physics_trace_substeps"}:
        raise ValueError("Solver migration cannot change unrelated control, task or budget fields")
    if final_restart:
        # The final-stage guard checks the actual physics fields unchanged;
        # no solver or force flag is normalized on this exact continuation.
        receipt = _validate_final_rough_stage_resume(old, new)
    else:
        previous, current = deepcopy(old), deepcopy(new)
        previous.pop(key, None)
        current.pop(key, None)
        # The legacy guard describes TGS. Both actual flags were checked above;
        # normalize only this algorithm-specific field, retaining every other diff.
        previous[external_key] = current[external_key] = True
        # This also retains the original pressure-entry restrictions when the
        # registered parent still applies pressure on the slider coordinate.
        receipt = validate_formal_learning_state_resume(previous, current)
    return {**receipt, key: {
                "before": before, "after": after,
                "before_name": "TGS" if before else "PGS",
                "after_name": "TGS" if after else "PGS",
                "physics_dt_s": .005, "policy_dt_s": .02,
                "position_velocity_iterations": [12, 6],
                "external_forces_every_iteration": {
                    "before": old[external_key], "after": new[external_key],
                    "compatibility": "TGS supports either boolean; PGS requires False",
                    "force_torque_enabled": True,
                    "control_forces_and_torques": "unchanged",
                    "solver_force_timing": "algorithm-dependent; not exact physical equivalence"},
                "actor_critic_std_adam_adaptive_lr_rng": "inherit",
                "local_global_and_consumption_clocks": "inherit",
                "native_validation": ("required_before_activation" if physics_changed
                    else "unchanged_algorithm_no_repeat_required")},
            "changed_contract_fields": sorted(changed), "exact_physics_resume": False}


def _validate_foundation_contact_mapping_resume(old, new):
    """Complete this chassis' fixed contact list, preserving every other input."""
    field = "scut_reward_mapping"
    before, after = old[field], new[field]
    if any(not validate_profile_selection(config)
           or config.get("formal_v7_flat") is not True
           or config.get("asset_manifest_sha256") != GAS_PRESSURE_ASSET_SHA256
           or config.get("formal_learning_state_resume") is not True
           or config.get("curriculum_stage") != "foundation"
           or config.get("enabled_stages") != ["foundation"]
           or config.get("reward_profile") != "scut0719_flat_v1"
           or any(key in config for key in ("recovery_training", "formal_scripted_recovery_mix",
                                           "formal_stage_optimizer"))
           for config in (old, new)):
        raise ValueError("Contact mapping repair requires the existing unscripted formal foundation")
    if (before.get("undesired_link_names") != list(FOUNDATION_CONTACT_LINKS_BEFORE)
            or after.get("undesired_link_names") != list(FOUNDATION_CONTACT_LINKS_COMPLETE)
            or before.get("terminal_link_names") != ["base_link"]
            or after.get("terminal_link_names") != ["base_link"]
            or {key: value for key, value in before.items() if key != "undesired_link_names"}
               != {key: value for key, value in after.items() if key != "undesired_link_names"}):
        raise ValueError("Contact mapping repair only admits the fixed five-to-thirteen link completion")
    previous, current = deepcopy(old), deepcopy(new)
    current[field] = deepcopy(previous[field])
    # Re-enter the existing solver/pressure guards after checking this one
    # reward-input difference; neither physics nor other reward fields vanish.
    receipt = validate_formal_learning_state_resume(previous, current)
    changed = {key for key in old.keys() | new.keys()
               if (key in old) != (key in new)
               or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
    return {**receipt, "undesired_contact_mapping": {
                "before": list(FOUNDATION_CONTACT_LINKS_BEFORE),
                "after": list(FOUNDATION_CONTACT_LINKS_COMPLETE),
                "reward_coefficients": "unchanged", "terminal_link_names": ["base_link"],
                "scope": "reward_contact_input_coverage_not_a_force_or_constraint_change"},
            "changed_contract_fields": sorted(changed)}


def validate_formal_learning_state_resume(old, new):
    """Same-ABI repair or explicit skill stage: retain learning, start fresh episodes.

    The shared budget separately authenticates the exact registered parent.
    This is deliberately not an exact native episode/physics restart.
    """
    if "formal_small_rough_mix" in old or "formal_small_rough_mix" in new:
        from .small_rough_course import validate_resume
        return validate_resume(old, new)
    before_mapping, after_mapping = old.get("scut_reward_mapping", {}), new.get("scut_reward_mapping", {})
    if (isinstance(before_mapping, Mapping) and isinstance(after_mapping, Mapping)
            and all(config.get("curriculum_stage") == "foundation"
                    and config.get("enabled_stages") == ["foundation"] for config in (old, new))
            and before_mapping.get("undesired_link_names") != after_mapping.get("undesired_link_names")
            and any(mapping.get("undesired_link_names") == list(FOUNDATION_CONTACT_LINKS_COMPLETE)
                    for mapping in (before_mapping, after_mapping))):
        return _validate_foundation_contact_mapping_resume(old, new)
    scripted_mix_key = "formal_scripted_recovery_mix"
    if scripted_mix_key in old or scripted_mix_key in new:
        return _validate_formal_scripted_flat_mix_resume(old, new)
    if "physx_solver_type" in old or "physx_solver_type" in new:
        return _validate_physx_solver_resume(old, new)
    if "formal_final_rough_stage" in old or "formal_final_rough_stage" in new:
        return _validate_final_rough_stage_resume(old, new)
    pressure_key = "gas_pressure_application"
    if ((pressure_key in old) != (pressure_key in new)
            or old.get(pressure_key) != new.get(pressure_key)):
        return _validate_gas_pressure_resume(old, new)
    before_policy, after_policy = old.get("formal_stage_policy", {}), new.get("formal_stage_policy", {})
    if (isinstance(before_policy, Mapping) and isinstance(after_policy, Mapping)
            and any("assessment_mode" in policy for policy in (before_policy, after_policy))):
        modes = tuple(policy.get("assessment_mode", "pause_for_assessment")
                      for policy in (before_policy, after_policy))
        if any(type(mode) is not str or mode not in ("pause_for_assessment", "record_and_continue")
               for mode in modes):
            raise ValueError("Unknown formal assessment mode")
        if modes[0] != modes[1]:
            if (modes != ("pause_for_assessment", "record_and_continue")
                    or any(config.get("formal_v7_flat") is not True
                           or config.get("curriculum_stage") != "foundation"
                           or config.get("enabled_stages") != ["foundation"]
                           for config in (old, new))
                    or new.get("formal_learning_state_resume") is not True):
                raise ValueError("Assessment mode migration requires the same formal foundation")
            previous, current = deepcopy(old), deepcopy(new)
            for config in (previous, current):
                config["formal_stage_policy"].pop("assessment_mode", None)
            if (json.dumps(previous["formal_stage_policy"], sort_keys=True)
                    != json.dumps(current["formal_stage_policy"], sort_keys=True)):
                raise ValueError("Assessment mode migration cannot change other stage policy fields")
            changed = {key for key in old.keys() | new.keys()
                       if (key in old) != (key in new)
                       or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
            unexpected = changed - {"formal_stage_policy", "formal_physics_trace_directory"}
            if unexpected:
                raise ValueError(f"Assessment mode migration changes unrelated fields: {sorted(unexpected)}")
            # Remove only the monitoring choice; the original continuation
            # guard still authenticates the selected profile and flat physics.
            receipt = validate_formal_learning_state_resume(previous, current)
            return {**receipt, "formal_assessment_mode": {
                        "before": modes[0], "after": modes[1],
                        "actor_critic_std_adam_adaptive_lr_rng": "inherit",
                        "local_global_and_consumption_clocks": "inherit"},
                    "changed_contract_fields": sorted(changed)}
    dynamic_birth_key = "formal_dynamic_nonsteady_birth"
    if dynamic_birth_key in old or dynamic_birth_key in new:
        from .fixed_flat_rehearsal import (
            FLAT_MIXED_REFRESH_VERSION, validate_dynamic_nonsteady_birth,
        )
        before = validate_dynamic_nonsteady_birth(old)
        after = validate_dynamic_nonsteady_birth(new)
        if (any(config.get("formal_v7_flat") is not True
                or config.get("curriculum_stage") != "foundation"
                or config.get("formal_flat_refresh") != FLAT_MIXED_REFRESH_VERSION
                or type(config.get("formal_foundation_nominal_birth")) is not bool
                for config in (old, new))
                or new.get("formal_learning_state_resume") is not True):
            raise ValueError("Dynamic birth resume requires the same formal mixed foundation")
        if not validate_profile_selection(old) or not validate_profile_selection(new):
            raise ValueError("Dynamic birth resume requires the same 35D/78D training profile")
        previous, current = deepcopy(old), deepcopy(new)
        for config in (previous, current):
            config.pop(dynamic_birth_key, None)
            config["formal_foundation_nominal_birth"] = True
        winding_change = None
        before_winding = old["v5_control"].get("leg_winding_version", "independent_nearest_v1")
        after_winding = new["v5_control"].get("leg_winding_version", "independent_nearest_v1")
        if any(value not in ("independent_nearest_v1", "paired_common_v1", "paired_branch107_v1")
               for value in (before_winding, after_winding)):
            raise ValueError("Dynamic birth requires a known leg target winding version")
        if before_winding != after_winding:
            if (before or not after or before_winding != "independent_nearest_v1"
                    or after_winding not in ("paired_common_v1", "paired_branch107_v1")):
                raise ValueError("Dynamic birth target repair only admits the paired decoder at entry")
            if (after_winding == "paired_branch107_v1"
                    and new["asset_manifest_sha256"] != "088e699266fa3dc666379a911b23a30e2038ab3d7f113cab4dcd4bdd0a14279b"):
                raise ValueError("Branch107 target projection requires its frozen mechanical asset")
            # Two continuous root axes per side must take the same full turn.
            # Normalize this one reviewed decoder choice; gains, scales, torque
            # bounds and all other action semantics still pass the old guard.
            if "leg_winding_version" in previous["v5_control"]:
                current["v5_control"]["leg_winding_version"] = previous["v5_control"]["leg_winding_version"]
            else:
                current["v5_control"].pop("leg_winding_version", None)
            winding_change = {"before": before_winding, "after": after_winding,
                              "scope": "common full-turn target per side; no relative-knee projection",
                              "learning_state": "inherit", "native_validation": "required_before_activation"}
            if after_winding == "paired_branch107_v1":
                winding_change.update(scope="same-side common turn plus the frozen107 assembly relative-target projection",
                    control_change="PD preload may change; this is not a closed-chain physics fix",
                    native_validation="matched_full_resume_sampled_policy_required_before_activation")
        # Normalize only the reviewed lane birth selector. The original mixed
        # guard continues to own physics, commands, budgets and learning clocks.
        receipt = validate_formal_learning_state_resume(previous, current)
        return {**receipt, dynamic_birth_key: {
                    "enabled_before": before, "enabled_after": after,
                    "fixed_nominal_rehearsal_fraction": .25,
                    "dynamic_background_fraction": .75,
                "dynamic_background_birth": "source_requests_with_mechanical_rejection" if after else "nominal",
                "scope": "reset lane selection; not measured accepted distribution or PPO exposure"},
                **({"leg_target_winding_repair": winding_change} if winding_change is not None else {}),
                "changed_contract_fields": sorted(key for key in old.keys() | new.keys()
                    if (key in old) != (key in new)
                    or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)),
                "exact_physics_resume": False}
    stepup_key = "formal_stepup_command"
    if stepup_key in old or stepup_key in new:
        from .stepup_command import validate_stepup_command
        before = validate_stepup_command(old.get(stepup_key))
        after = validate_stepup_command(new.get(stepup_key))
        if not (old.get("formal_v7_flat") is True and new.get("formal_v7_flat") is True
                and new.get("formal_learning_state_resume") is True
                and validate_profile_selection(old) and validate_profile_selection(new)):
            raise ValueError("V7 StepUp requires explicit same-profile formal learning-state resume")
        previous, current = deepcopy(old), deepcopy(new)
        previous.pop(stepup_key, None)
        current.pop(stepup_key, None)
        receipt = validate_formal_learning_state_resume(previous, current)
        return {**receipt, stepup_key: {"before": before, "after": after,
                    "scope": "explicit event height modulation; same actor and PPO ownership"},
                "changed_contract_fields": sorted(key for key in old.keys() | new.keys()
                    if (key in old) != (key in new)
                    or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True))}
    decoupling_key = "formal_command_decoupling"
    if decoupling_key in old or decoupling_key in new:
        if not (old.get("formal_v7_flat") is True and new.get("formal_v7_flat") is True
                and new.get("formal_learning_state_resume") is True):
            raise ValueError("V7 command decoupling requires explicit formal learning-state resume")
        if not validate_profile_selection(old) or not validate_profile_selection(new):
            raise ValueError("V7 command decoupling requires the same 35D/78D training profile")
        from .scut_reward_selection import validate_command_decoupling
        before = validate_command_decoupling(old.get(decoupling_key))
        after = validate_command_decoupling(new.get(decoupling_key))
        previous, current = deepcopy(old), deepcopy(new)
        previous.pop(decoupling_key, None)
        current.pop(decoupling_key, None)
        # Only this additive payment selector is normalized. The existing
        # stage/physics/budget guards still own every other contract field.
        receipt = validate_formal_learning_state_resume(previous, current)
        return {**receipt,
                decoupling_key: {"before": before, "after": after,
                    "source_reward_terms_unchanged": True,
                    "source_nonzero_term_count": 26,
                    "additional_payment_term_count": int(after is not None and after["enabled"])},
                "changed_contract_fields": sorted(key for key in old.keys() | new.keys()
                    if (key in old) != (key in new)
                    or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True))}
    if "formal_flat_refresh" in new:
        from .fixed_flat_rehearsal import FLAT_MIXED_REFRESH_VERSION
        if new["formal_flat_refresh"] == FLAT_MIXED_REFRESH_VERSION:
            return _validate_formal_flat_mixed_refresh_resume(old, new)
        return _validate_formal_flat_refresh_resume(old, new)
    if "formal_robustness_revision" in new:
        return _validate_formal_robustness_resume(old, new)
    if not (old.get("formal_v7_flat") is True and new.get("formal_v7_flat") is True
            and new.get("formal_learning_state_resume") is True):
        raise ValueError("Explicit formal V7 learning-state repair required")
    validate_profile_selection(old)
    validate_profile_selection(new)
    birth_key = "formal_foundation_nominal_birth"
    if any(birth_key in config and type(config[birth_key]) is not bool for config in (old, new)):
        raise ValueError("Formal foundation nominal birth selection requires strict boolean values")
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    cap_key = "flat_special_mode_iteration_cap"
    if (type(old.get(cap_key)) is not type(new.get(cap_key))
            or (cap_key in old) != (cap_key in new)):
        changed.add(cap_key)
    allowed = {"actuator_response", "actuator_response_evidence", "formal_learning_state_resume",
               "formal_stage_policy", "control_model_status", "formal_physics_trace_directory",
               "formal_physics_trace_substeps",
               "physx_external_forces_every_iteration", "formal_terrain",
               "formal_closure_monitor_only", "closure_gap_termination_m",
               "evaluation_long_corridors", "flat_triangle_mesh", "flat_floor_width_m",
               "flat_half_length_m"}
    capacity_key = "gpu_found_lost_aggregate_pairs_capacity"
    capacity_increase = (capacity_key in changed and capacity_key in new
                         and new[capacity_key] > old.get(capacity_key, 0))
    if capacity_key in changed:
        if not capacity_increase:
            raise ValueError("Formal GPU aggregate pair capacity repair must increase capacity")
        allowed.add(capacity_key)
    rehearsal_key = "formal_flat_rehearsal"
    rehearsal_release = rehearsal_key not in old and rehearsal_key in new
    rough_transition = formal_rough_stage_transition(new, old.get("curriculum_stage"), new.get("curriculum_stage"))
    source_rehearsal = new.get(rehearsal_key, {}).get("version") == "v7_source0719_plane_rehearsal_v2"
    uniform_reward_repair = False
    recovery_transition = formal_recovery_stage_transition(new, old.get("curriculum_stage"), new.get("curriculum_stage"))
    rehearsal_removal = recovery_transition and rehearsal_key in old and rehearsal_key not in new
    rehearsal_retained = (recovery_transition and rehearsal_key in old and rehearsal_key in new
                          and json.dumps(old[rehearsal_key], sort_keys=True)
                          == json.dumps(new[rehearsal_key], sort_keys=True))
    from .flat_episode_selection import validate_scut_flat_recovery
    old_recovery = validate_scut_flat_recovery(old)
    new_recovery = validate_scut_flat_recovery(new)
    rough_initial_key = "formal_rough_initial_level"
    stage_optimizer_key = "formal_stage_optimizer"
    initial_level_release = (rough_transition and rough_initial_key not in old
                             and type(new.get(rough_initial_key)) is int and new[rough_initial_key] == 0)
    optimizer_release = (rough_transition and stage_optimizer_key not in old
                         and new.get(stage_optimizer_key) == "scut_finetune_v1")
    initial_level_removal = (recovery_transition and rough_initial_key in old and rough_initial_key not in new)
    for key, released in ((rough_initial_key, initial_level_release or initial_level_removal),
                          (stage_optimizer_key, optimizer_release)):
        different = ((key in old) != (key in new)
                     or json.dumps(old.get(key)) != json.dumps(new.get(key)))
        if different and not released:
            raise ValueError(f"Formal rough initialization must be retained after its flat-to-rough release: {key}")
        if released:
            changed.add(key)
            allowed.add(key)
    closure_keys = {"formal_closure_monitor_only", "closure_gap_termination_m"}
    closure_changed = {key for key in closure_keys
                       if (key in old) != (key in new)
                       or json.dumps(old.get(key)) != json.dumps(new.get(key))}
    closure_reset_revision = (old.get("formal_closure_monitor_only") is True
                              and old.get("closure_gap_termination_m") is None
                              and new.get("formal_closure_monitor_only") is False
                              and new.get("closure_gap_termination_m") == .003)
    # A recovery handoff inherits physics; it cannot introduce, undo or hide
    # the closure-policy repair while changing its scripted birth population.
    if closure_changed and (old_recovery is not None or new_recovery is not None):
        raise ValueError("Source recovery must retain the parent's closure policy")
    if source_rehearsal:
        # Admit this course change and the explicit monitor -> 3 mm reset
        # revision only, not the broader physical-repair list below.
        strict_changed = {key for key in old.keys() | new.keys()
                          if (key in old) != (key in new)
                          or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
        course_changes = {rehearsal_key, "formal_physics_trace_directory"}
        if capacity_increase:
            course_changes.add(capacity_key)
        if initial_level_release:
            course_changes.add(rough_initial_key)
        if optimizer_release:
            course_changes.add(stage_optimizer_key)
        if closure_reset_revision:
            course_changes.update(closure_keys)
        if rough_transition:
            course_changes.update(("reward_profile", "scut_reward_mapping", "critic_height_reference",
                "curriculum_stage", "enabled_stages", "stages", "scene_groups", "formal_stage_transition",
                "formal_stage_policy", "formal_learning_state_resume", "formal_terrain", "evaluation"))
            if (rehearsal_key in old or old.get("enabled_stages") != ["foundation"]
                    or old.get("formal_terrain") != {"type": "plane", "source": "scut0719", "env_spacing_m": 4.}
                    or new.get("formal_terrain") != {"type": "rough", "source": "scut0730"}
                    or cap_key in old or cap_key in new
                    or new.get("stages") != [{**item, "name": "terrain", "terrain": ["rough"]}
                                             for item in old.get("stages", [])]):
                raise ValueError("Source flat rehearsal handoff requires the existing uncapped flat course and unchanged update allowance")
        elif not all(config.get("curriculum_stage") == "terrain"
                     and config.get("enabled_stages") == ["terrain"]
                     and config.get("reward_profile") == "scut0730_rough_v1"
                     for config in (old, new)):
            raise ValueError("Source flat rehearsal requires the explicit flat-to-rough or same-rough handoff")
        previous = old.get(rehearsal_key)
        if previous is not None:
            current = new[rehearsal_key]
            uniform_reward_repair = (
                previous.get("version") == "v7_source0719_plane_rehearsal_v2"
                and previous.get("reward_profile") == "scut0719_flat_v1"
                and current == {**previous, "reward_profile": "scut0730_rough_v1"})
            if (previous["version"] == "v7_source0719_plane_rehearsal_v2"
                    and rehearsal_key in strict_changed
                    and not uniform_reward_repair
                    or any(json.dumps(previous[key]) != json.dumps(current[key])
                           for key in ("fraction", "cases_file", "cases_sha256"))):
                raise ValueError("Source flat rehearsal repair must retain its allocation and bound evaluation cases")
            if uniform_reward_repair:
                # This repair changes one payment selector, not the sampler,
                # physics, optimizer or clocks. Only its prose may follow it.
                course_changes = {rehearsal_key, "formal_physics_trace_directory"}
                if "formal_stage_policy" in strict_changed:
                    old_policy, new_policy = old["formal_stage_policy"], new["formal_stage_policy"]
                    if (set(old_policy) != set(new_policy)
                            or any(old_policy[key] != new_policy[key]
                                   for key in old_policy if key != "terrain_scope")):
                        raise ValueError("Uniform rough reward repair may update only terrain_scope documentation")
                    course_changes.add("formal_stage_policy")
        if strict_changed - course_changes:
            raise ValueError("Source flat rehearsal cannot bundle unrelated contract changes")
        changed = strict_changed
        allowed.add(rehearsal_key)
    elif (rehearsal_key in old or rehearsal_key in new) and not (rehearsal_removal or rehearsal_retained):
        # Allocation release remains isolated. An existing rough allocation
        # may additionally adopt the explicit monitor -> 3 mm reset revision.
        strict_changed = {key for key in old.keys() | new.keys()
                          if (key in old) != (key in new)
                          or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
        rehearsal_changes = {rehearsal_key, "formal_physics_trace_directory"}
        if closure_reset_revision and not rehearsal_release:
            rehearsal_changes.update(closure_keys)
        if (rehearsal_key not in new
                or rehearsal_key in old and rehearsal_key in strict_changed
                or strict_changed - rehearsal_changes
                or not (all(config.get("curriculum_stage") == "terrain"
                            and config.get("enabled_stages") == ["terrain"]
                            and config.get("reward_profile") == "scut0730_rough_v1"
                            for config in (old, new))
                        or old_recovery is not None and new_recovery is not None
                        and not rehearsal_release)):
            raise ValueError("Fixed flat rehearsal permits only initial rough allocation or unchanged same-stage allocation")
        changed = strict_changed
        allowed.add(rehearsal_key)
    transition = rough_transition or recovery_transition
    recovery_release = old_recovery is None and new_recovery is not None
    if recovery_transition:
        # Presence and JSON types matter here: a null field or boolean is not
        # permission to hide an unrelated change in the ordinary repair list.
        changed = {key for key in old.keys() | new.keys()
                   if (key in old) != (key in new)
                   or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
        recovery_changes = {"reward_profile", "scut_reward_mapping", "critic_height_reference",
            "curriculum_stage", "enabled_stages", "stages", "scene_groups", "formal_stage_transition",
            "recovery_training", "formal_stage_policy", "formal_learning_state_resume",
            "formal_physics_trace_directory"}
        if rehearsal_removal:
            # Plane recovery replaces this rough-only lane owner. All other
            # transition checks below still apply to the actual parent.
            recovery_changes.add(rehearsal_key)
        if initial_level_removal:
            recovery_changes.add(rough_initial_key)
        if (not recovery_release or old.get("reward_profile") != "scut0730_rough_v1"
                or old.get("enabled_stages") != ["terrain"]
                or old.get("formal_terrain") != {"type": "rough", "source": "scut0730"}
                or new.get("formal_terrain") != {"type": "plane", "source": "scut0719", "env_spacing_m": 4.}
                or any(config.get(birth_key) is not True or cap_key in config for config in (old, new))
                or old.get("critic_height_reference") != "root_link_world_z"
                or new.get("critic_height_reference", "root_link_z_minus_env_origin") != "root_link_z_minus_env_origin"
                or new.get("enabled_stages") != ["foundation"]
                or "evaluation" in new or changed - (recovery_changes | {"formal_terrain"})):
            raise ValueError("Rough-to-recovery requires the pinned plane/0719 handoff, unchanged nominal births and only its necessary contract differences")
        old_mapping, new_mapping = dict(old["scut_reward_mapping"]), dict(new["scut_reward_mapping"])
        old_mapping.pop("root_height_reference")
        new_mapping.pop("root_height_reference")
        if old_mapping != new_mapping:
            raise ValueError("Rough-to-recovery cannot change chassis velocity/contact mappings")
        stages = old.get("stages", [])
        if (len(stages) != 1 or stages[0].get("name") != "terrain"
                or new.get("stages") != [{**stages[0], "name": "foundation", "terrain": ["flat"]}]):
            raise ValueError("Rough-to-recovery must retain the stage update allowance")
        from .flat_episode_selection import formal_flat_special_schedule
        if not formal_flat_special_schedule(new):
            raise ValueError("Rough-to-recovery must retain the pinned flat schedule on the inherited global clock")
        allowed.update(recovery_changes)
    elif recovery_release:
        if (old.get("reward_profile") != "scut0719_flat_v1"
                or old.get("curriculum_stage") != "foundation"
                or old.get("enabled_stages") != ["foundation"]
                or old.get("formal_terrain") != new.get("formal_terrain")
                or changed - {"recovery_training", "scene_groups", "formal_stage_policy",
                              "formal_learning_state_resume", "formal_physics_trace_directory"}):
            raise ValueError("Source recovery handoff permits only explicit flat recovery births and scene quotas with unchanged learning/physics")
        allowed.update(("recovery_training", "scene_groups"))
    elif old_recovery is not None and new_recovery != old_recovery:
        raise ValueError("Source recovery continuation must retain the frozen handoff and profile allocation")
    from .flat_episode_selection import formal_flat_special_schedule
    old_special = formal_flat_special_schedule(old)
    new_special = formal_flat_special_schedule(new)
    special_release = new_special and not old_special and not recovery_transition
    if special_release:
        # Uncap the existing global actor clock only. This is neither a rough
        # stage transition nor permission to combine unrelated physical edits.
        if (old.get("reward_profile") != "scut0719_flat_v1"
                or old.get("curriculum_stage") != "foundation"
                or old.get("enabled_stages") != ["foundation"]
                or old.get("formal_terrain") != new["formal_terrain"]
                or type(old.get("flat_special_mode_iteration_cap")) is not int
                or old["flat_special_mode_iteration_cap"] != 0
                or changed - {"formal_stage_policy", "formal_learning_state_resume",
                              "formal_physics_trace_directory",
                              "flat_special_mode_iteration_cap"}):
            raise ValueError("Pinned flat special release permits only foundation/plane cap 0 removal with unchanged learning interface")
        allowed.add("flat_special_mode_iteration_cap")
    elif old_special and not new_special and not transition:
        raise ValueError("Same-stage resume must retain the pinned flat special schedule")
    birth_release = old.get(birth_key) is True and new.get(birth_key) is False
    if birth_key in changed:
        terrain = new.get("formal_terrain", {})
        if (not birth_release
                or any(config.get("reward_profile") != "scut0719_flat_v1"
                       or config.get("curriculum_stage") != "foundation"
                       or config.get("enabled_stages") != ["foundation"] for config in (old, new))
                or not isinstance(terrain, Mapping) or terrain.get("type") != "plane"
                or terrain.get("source") != "scut0719" or old.get("formal_terrain") != terrain
                or old_special != new_special
                or (not old_special and (type(old.get(cap_key)) is not int or old[cap_key] != 0))
                or changed - {birth_key, "formal_stage_policy", "formal_learning_state_resume",
                              "formal_physics_trace_directory"}):
            raise ValueError("Formal birth release permits only boolean true to false on the same foundation/plane with unchanged special schedule and learning interface")
        allowed.add(birth_key)
    if rough_transition:
        from .scut_reward_selection import SCUT_FLAT_PROFILE
        if old.get("reward_profile") != SCUT_FLAT_PROFILE:
            raise ValueError("Rough stage handoff requires the source flat parent")
        allowed.update(("reward_profile", "scut_reward_mapping", "critic_height_reference",
                        "curriculum_stage", "enabled_stages", "stages", "scene_groups",
                        "formal_stage_transition", "flat_special_mode_iteration_cap", "evaluation"))
        old_mapping, new_mapping = dict(old["scut_reward_mapping"]), dict(new["scut_reward_mapping"])
        old_mapping.pop("root_height_reference")
        new_mapping.pop("root_height_reference")
        if old_mapping != new_mapping:
            raise ValueError("Rough handoff cannot change chassis velocity/contact mappings")
        if (new.get("flat_special_mode_iteration_cap") is not None
                or new.get("enabled_stages") != ["terrain"] or "evaluation" in new):
            raise ValueError("Rough stage must use the saved 0730 commands and explicit terrain stage")
    if changed - allowed:
        raise ValueError(f"V7 physical repair changes unrelated interface fields: {sorted(changed - allowed)}")
    return {"scope": "same_35D_78D_learning_state_with_fresh_native_episodes",
            "stage_transition": transition,
            "flat_special_schedule_release": special_release,
            "formal_nonsteady_birth_release": birth_release,
            "formal_scripted_recovery_release": recovery_release,
            "formal_rough_to_recovery_transition": recovery_transition,
            **({"formal_fixed_flat_rehearsal_removed_for_recovery": True}
               if rehearsal_removal else {}),
            **({"formal_fixed_flat_rehearsal_retained_for_recovery": True}
               if rehearsal_retained else {}),
            **({"formal_fixed_flat_rehearsal_release": rehearsal_release,
                "rehearsal_clock_scope": (
                    "explicit stage transition; reset local stage clock, inherit global learning and consumption"
                    if transition else
                    "same stage; inherit all local and global actor clocks")}
               if rehearsal_key in new else {}),
            **({"formal_source_flat_rehearsal": True,
                "source_flat_command_iteration_offset": new[rehearsal_key]["source_iteration_offset"],
                "rehearsal_reward_profile": new[rehearsal_key]["reward_profile"],
                "formal_uniform_rough_reward_repair": uniform_reward_repair}
               if source_rehearsal else {}),
            "critic_height_semantics_changed": transition,
            **({"height_reference_transition": {
                "critic_from": "root_link_world_z", "critic_to": "root_link_z_minus_env_origin",
                "reward_from": "root_link_z_minus_base_ground_scan", "reward_to": "root_link_z_minus_env_origin",
                "scope": "same 78D fields and unchanged critic weights; destination shared plane origins z=0"}}
               if recovery_transition else {}),
            "changed_contract_fields": sorted(changed), "exact_physics_resume": False}


def _validate_formal_scripted_flat_mix_resume(old, new):
    """Add the frozen scripted owner without changing the ordinary task."""
    from .fixed_flat_rehearsal import (
        validate_dynamic_nonsteady_birth, validate_scripted_flat_mix,
    )
    field = "formal_scripted_recovery_mix"
    before = validate_scripted_flat_mix(old)
    after = validate_scripted_flat_mix(new)
    if (after is None or not validate_dynamic_nonsteady_birth(old)
            or before is None and "recovery_training" in old):
        raise ValueError("Scripted mix requires an existing dynamic foundation and cannot be removed")
    if not validate_profile_selection(old) or not validate_profile_selection(new):
        raise ValueError("Scripted mix requires the same 35D/78D training profile")
    changed = {key for key in old.keys() | new.keys()
               if (key in old) != (key in new)
               or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
    allowed = {field, "recovery_training", "scene_groups", "research_training_authorized",
               "formal_stage_policy", "formal_physics_trace_directory"}
    if changed - allowed:
        raise ValueError(f"Scripted mix changes unrelated fields: {sorted(changed - allowed)}")
    previous, current = deepcopy(old), deepcopy(new)
    controller_revision = None
    if before is not None:
        from .recovery import DIRECT_TAKEOVER_VERSION, REPLANT_TAKEOVER_VERSION
        old_version = previous["recovery_training"]["direct_takeover"]["version"]
        new_version = current["recovery_training"]["direct_takeover"]["version"]
        if old_version != new_version:
            if (old_version, new_version) != (DIRECT_TAKEOVER_VERSION, REPLANT_TAKEOVER_VERSION):
                raise ValueError("Scripted mix only admits the explicit v1 to v4 controller revision")
            # This version selects the verified script repair. Keep the same
            # lane population, capture gates, physics and full learning state.
            current["recovery_training"]["direct_takeover"]["version"] = old_version
            controller_revision = {"before": old_version, "after": new_version,
                                   "capture_settings": "unchanged"}
    if before is not None and any(
            json.dumps(previous[key], sort_keys=True) != json.dumps(current[key], sort_keys=True)
            for key in ("scene_groups", "recovery_training")):
        raise ValueError("Scripted mix continuation must retain its allocation and recovery settings")

    for config, selection in ((previous, before), (current, after)):
        if selection is None:
            continue
        config.pop(field)
        config.pop("recovery_training")
        ordinary = [scene for scene in config["scene_groups"] if "recovery_pose" not in scene]
        for scene in ordinary:
            scene["fraction"] /= selection["ordinary_fraction"]
        config["scene_groups"] = ordinary
    # Only the validated script owner and its lane share are normalized. Names,
    # terrain and all ordinary scene fields still pass the original guard.
    receipt = validate_formal_learning_state_resume(previous, current)
    return {**receipt, "dynamic_source_fraction": after["dynamic_source_fraction"],
            "formal_dynamic_nonsteady_birth": {
                **receipt["formal_dynamic_nonsteady_birth"],
                "dynamic_background_fraction": after["dynamic_source_fraction"]},
            field: {
                "enabled_before": before is not None, "enabled_after": True,
                "script_fraction": after["script_fraction"],
                "fixed_nominal_rehearsal_fraction": .25,
                "ordinary_dynamic_fraction": after["dynamic_source_fraction"],
                "population_changed": before is None,
                "actor_critic_std_adam_adaptive_lr_rng": "inherit",
                "local_global_and_consumption_clocks": "inherit"},
            **({"recovery_controller_revision": controller_revision}
               if controller_revision is not None else {}),
            "changed_contract_fields": sorted(changed),
            "exact_physics_resume": False}


def _validate_formal_flat_refresh_resume(old, new):
    """Refresh the existing flat policy without resetting its learning clocks."""
    from .fixed_flat_rehearsal import FLAT_REFRESH_VERSION, FIXED17_VERSION, validate_fixed_flat_rehearsal
    from .pc_communication import validate_pc_communication
    validate_profile_selection(old)
    validate_profile_selection(new)
    rehearsal = validate_fixed_flat_rehearsal(new)
    if (new.get("formal_flat_refresh") != FLAT_REFRESH_VERSION
            or new.get("formal_learning_state_resume") is not True
            or "round_trip_transport" not in new
            or rehearsal is None or rehearsal["version"] != FIXED17_VERSION
            or rehearsal["fraction"] != 1.
            or new.get("wheel_material_priors") != {
                "enabled": True, "version": "v7_low_grip_wheel_buckets_v2"}):
        raise ValueError("Flat refresh requires the shared delay, low-grip wheels and all-flat fixed cases")
    validate_pc_communication(new)
    for cfg in (old, new):
        if (cfg.get("formal_v7_flat") is not True
                or cfg.get("curriculum_stage") != "foundation"
                or cfg.get("enabled_stages") != ["foundation"]
                or cfg.get("reward_profile") != "scut0719_flat_v1"
                or cfg.get("formal_terrain") != {"type": "plane", "source": "scut0719", "env_spacing_m": 4.}
                or cfg.get("formal_closure_monitor_only") is not True
                or cfg.get("closure_gap_termination_m") is not None
                or "recovery_training" in cfg or "formal_stage_optimizer" in cfg):
            raise ValueError("Flat refresh inherits the existing foundation, learning state and closure policy")
    fields = {"formal_flat_refresh", "formal_flat_rehearsal", "round_trip_transport", "feedback_transport",
              "command_transport", "signal_delay", "usb_transport", "wheel_material_priors",
              "formal_stage_policy", "formal_learning_state_resume", "formal_physics_trace_directory",
              "formal_physics_trace_substeps"}
    changed = {key for key in old.keys() | new.keys()
               if (key in old) != (key in new)
               or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
    if changed - fields:
        raise ValueError(f"Flat refresh changes unrelated fields: {sorted(changed - fields)}")
    prose = {"communication_scope", "terrain_scope", "initialization", "progression", "birth_scope",
             "flat_special_scope"}
    before, after = old["formal_stage_policy"], new["formal_stage_policy"]
    if (set(before) != set(after) or any(json.dumps(before[key]) != json.dumps(after[key])
            for key in before if key not in prose)):
        raise ValueError("Flat refresh cannot change assessment, promotion, schedule or deployment clocks")
    if "formal_flat_refresh" in old:
        if old.get("formal_flat_refresh") != FLAT_REFRESH_VERSION or any(
                json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)
                for key in fields - {"formal_stage_policy", "formal_physics_trace_directory",
                                     "formal_physics_trace_substeps"}):
            raise ValueError("Flat refresh continuation must retain its selected configuration")
    elif (old.get("command_transport", {}).get("enabled") is not False
            or "round_trip_transport" in old or "formal_flat_rehearsal" in old
            or old.get("wheel_material_priors") != {"enabled": True, "version": "wheel_shape_buckets_v1"}):
        raise ValueError("Flat refresh release requires the registered zero-communication flat parent")
    return {"scope": "same_35D_78D_learning_state_with_fresh_native_episodes",
            "stage_transition": False, "formal_flat_refresh": FLAT_REFRESH_VERSION,
            "rehearsal_clock_scope": "same stage; inherit all local and global actor clocks",
            "communication_total_delay_range_s": [.0001, .005], "communication_uplink_fraction": .5,
            "wheel_friction_lower_bound": .2, "flat_rehearsal_version": FIXED17_VERSION,
            "flat_rehearsal_allocation": "equal_case_lanes", "changed_contract_fields": sorted(changed),
            "exact_physics_resume": False}


def _validate_formal_flat_mixed_refresh_resume(old, new):
    """Change only the ordinary flat sampling allocation, retaining learning state."""
    from .fixed_flat_rehearsal import (FLAT_REFRESH_VERSION, FLAT_MIXED_REFRESH_VERSION,
                                       validate_fixed_flat_rehearsal)
    validate_profile_selection(old)
    validate_profile_selection(new)
    if (old.get("formal_flat_refresh") not in (FLAT_REFRESH_VERSION, FLAT_MIXED_REFRESH_VERSION)
            or new.get("formal_flat_refresh") != FLAT_MIXED_REFRESH_VERSION
            or any(type(cfg.get("research_training_authorized")) is not bool for cfg in (old, new))):
        raise ValueError("Mixed flat refresh requires the existing refresh parent and an explicit authorization flag")
    before, after = validate_fixed_flat_rehearsal(old), validate_fixed_flat_rehearsal(new)
    if any(before[key] != after[key] for key in before if key != "fraction"):
        raise ValueError("Mixed flat refresh cannot change the pinned rehearsal requests or rewards")
    allowed = {"formal_flat_refresh", "formal_flat_rehearsal", "formal_stage_policy",
               "research_training_authorized", "formal_physics_trace_directory",
               "formal_physics_trace_substeps"}
    changed = {key for key in old.keys() | new.keys()
               if (key in old) != (key in new)
               or json.dumps(old.get(key), sort_keys=True) != json.dumps(new.get(key), sort_keys=True)}
    if changed - allowed:
        raise ValueError(f"Mixed flat refresh changes unrelated fields: {sorted(changed - allowed)}")
    # Normalize only the reviewed sampler selection and the draft execution
    # flag. The existing guard still owns physics, stage and clock invariants.
    previous, current = deepcopy(old), deepcopy(new)
    for cfg in (previous, current):
        cfg["formal_flat_refresh"] = FLAT_REFRESH_VERSION
        cfg["formal_flat_rehearsal"]["fraction"] = 1.
    current["research_training_authorized"] = previous["research_training_authorized"]
    receipt = _validate_formal_flat_refresh_resume(previous, current)
    return {**receipt, "formal_flat_refresh": FLAT_MIXED_REFRESH_VERSION,
            "fixed_rehearsal_fraction": .25, "dynamic_source_fraction": .75,
            "background_commands": "existing source sampler without fixed overrides; ordinary PPO samples",
            "changed_contract_fields": sorted(changed)}


def _validate_formal_robustness_resume(old, new):
    """Admit the selected delay/material/rehearsal revision through the old stage guard."""
    marker = "v7_delay_low_grip_rehearsal_v1"
    if new.get("formal_robustness_revision") != marker:
        raise ValueError("Unknown formal robustness revision")
    validate_profile_selection(old)
    validate_profile_selection(new)
    from .pc_communication import validate_pc_communication
    from .fixed_flat_rehearsal import (
        FIXED17_VERSION, FLAT_REFRESH_VERSION, FLAT_MIXED_REFRESH_VERSION,
        validate_fixed_flat_rehearsal,
    )
    validate_pc_communication(new)
    rehearsal = validate_fixed_flat_rehearsal(new)
    if ("round_trip_transport" not in new or rehearsal is None
            or rehearsal["version"] != FIXED17_VERSION or rehearsal["fraction"] != .25
            or new.get("wheel_material_priors") != {
                "enabled": True, "version": "v7_low_grip_wheel_buckets_v2"}
            or new.get("formal_closure_monitor_only") is not True
            or new.get("closure_gap_termination_m") is not None):
        raise ValueError("Robustness revision requires the shared delay, low-grip wheels and fixed flat rehearsal")
    for key in ("v5_control", "actuator_response", "actuator_response_evidence",
                "asset_manifest_sha256", "control_math_sha256", "control_math_source",
                "position_iterations", "velocity_iterations", "physics_dt", "policy_dt"):
        if old.get(key) != new.get(key):
            raise ValueError(f"Robustness revision cannot change mechanical/control semantics: {key}")
    policy_keys = {"later_skills_enabled", "first_assessment_update",
                  "no_automatic_iteration_based_promotion", "deployment_policy_hz",
                  "deployment_pd_hz", "assessment_interval_updates"}
    prose_keys = {"communication_scope", "terrain_scope", "initialization", "progression", "birth_scope"}
    if (set(new["formal_stage_policy"]) - policy_keys - prose_keys
            or any(type(old["formal_stage_policy"].get(key)) is not type(new["formal_stage_policy"].get(key))
                   or old["formal_stage_policy"].get(key) != new["formal_stage_policy"].get(key)
                   for key in policy_keys)):
        raise ValueError("Robustness revision cannot change assessment, promotion or deployment timing")
    previous = deepcopy(old)
    current = deepcopy(new)
    current.pop("formal_robustness_revision")
    if "formal_robustness_revision" in previous:
        if previous.pop("formal_robustness_revision") != marker:
            raise ValueError("Unknown parent robustness revision")
        receipt = validate_formal_learning_state_resume(previous, current)
    else:
        flat_parent = old.get("formal_flat_refresh") in (
            FLAT_REFRESH_VERSION, FLAT_MIXED_REFRESH_VERSION)
        if old.get("formal_flat_refresh") == FLAT_MIXED_REFRESH_VERSION:
            if ("round_trip_transport" not in old
                    or not formal_rough_stage_transition(
                        new, old.get("curriculum_stage"), new.get("curriculum_stage"))):
                raise ValueError("Mixed flat-to-rough requires the existing shared "
                                 "communication parent and explicit stage handoff")
            validate_pc_communication(old)
            # The parent already uses the selected delay and materials. Do
            # not hide a communication change behind the legacy release.
            for key in ("round_trip_transport", "feedback_transport", "command_transport",
                        "signal_delay", "usb_transport", "wheel_material_priors"):
                if ((key in old) != (key in new)
                        or json.dumps(old.get(key), sort_keys=True)
                        != json.dumps(new.get(key), sort_keys=True)):
                    raise ValueError("Mixed flat-to-rough must retain the parent's "
                                     f"communication and materials: {key}")
        if not flat_parent and old.get("command_transport", {}).get("enabled") is not False:
            raise ValueError("Robustness release requires the authenticated zero-communication parent")
        for key in ("round_trip_transport", "feedback_transport", "command_transport",
                    "signal_delay", "usb_transport", "wheel_material_priors"):
            if key in previous:
                current[key] = deepcopy(previous[key])
            else:
                current.pop(key, None)
        if old.get("curriculum_stage") == "foundation":
            if flat_parent:
                previous.pop("formal_flat_refresh")
                previous.pop("formal_flat_rehearsal")
            from .fixed_flat_rehearsal import SOURCE_VERSION, CASE_FILE, CASE_SHA
            current["formal_flat_rehearsal"] = {
                "version": SOURCE_VERSION, "fraction": .25, "cases_file": CASE_FILE,
                "cases_sha256": CASE_SHA, "allocation": "source_command_distribution",
                "source_iteration_offset": 4500, "reward_profile": "scut0730_rough_v1"}
        else:
            current["formal_flat_rehearsal"] = deepcopy(previous["formal_flat_rehearsal"])
            current["formal_flat_rehearsal"]["reward_profile"] = "scut0730_rough_v1"
            if (set(old["formal_stage_policy"]) != set(new["formal_stage_policy"])
                    or any(old["formal_stage_policy"][key] != new["formal_stage_policy"][key]
                           for key in old["formal_stage_policy"] if key not in prose_keys)):
                raise ValueError("Robustness release cannot change assessment or stage budgets")
            current["formal_stage_policy"] = deepcopy(previous["formal_stage_policy"])
        receipt = validate_formal_learning_state_resume(previous, current)
    for key in ("formal_source_flat_rehearsal", "source_flat_command_iteration_offset",
                "rehearsal_reward_profile", "formal_uniform_rough_reward_repair"):
        receipt.pop(key, None)
    return {**receipt, "formal_robustness_revision": marker,
            "communication_total_delay_range_s": [.0001, .005],
            "communication_uplink_fraction": .5,
            "wheel_friction_lower_bound": .2,
            "flat_rehearsal_version": FIXED17_VERSION,
            "flat_rehearsal_allocation": "equal_case_lanes",
            "changed_contract_fields": sorted(key for key in old.keys() | new.keys()
                if old.get(key) != new.get(key)), "exact_physics_resume": False}


def verify_profile_evaluation(contract, checkpoint_infos, runtime_hashes):
    """Authenticate training sources, then compare the shared runtime sources.

    Training and fixed evaluation have different entry points and source lists.
    Their whole-list digests must be recorded separately, never equated.
    """
    if not validate_profile_selection(contract):
        return None
    saved = checkpoint_infos.get("profile_source_sha256")
    if not isinstance(saved, dict) or not saved:
        raise ValueError("78D evaluation requires checkpoint source provenance")
    origin = profile_identity(contract, saved)
    if any(checkpoint_infos.get(key) != value for key, value in origin.items()):
        raise ValueError("78D checkpoint profile/schema/source provenance mismatch")
    runtime = profile_identity(contract, runtime_hashes)
    required = set(profile_source_files(contract))
    if contract.get("control_math_source"):
        required.add(contract["control_math_source"])
    if "signal_delay" in contract:
        required.update(f"src/wheeled_tasks/chassis/{name}.py" for name in ("signal_delay", "delay_buffer"))
    if "feedback_transport" in contract:
        required.update(f"src/wheeled_tasks/chassis/{name}.py" for name in (
            "feedback_transport", "feedback_transport_config", "pc_communication", "delay_buffer"))
    if required - saved.keys() or required - runtime_hashes.keys():
        raise ValueError("78D evaluation is missing required runtime source provenance")
    shared = saved.keys() & runtime_hashes.keys()
    changed = sorted(name for name in shared if saved[name] != runtime_hashes[name])
    if changed:
        raise ValueError(f"78D evaluation changes checkpoint runtime sources: {changed}")
    return {"checkpoint_identity": origin, "runtime_identity": runtime,
            "shared_runtime_sources_checked": sorted(shared),
            "evaluation_only_sources": sorted(runtime_hashes.keys() - saved.keys()),
            "scope": "same_profile_shared_runtime_sources; not native capability qualification"}


def verify_learning_rate(algorithm, contract, *, actor_enabled=True, before_rate=None):
    """Check the committed optimizer state without resetting restored Adam."""
    rate = algorithm.learning_rate
    groups = algorithm.optimizer.param_groups
    def valid(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0)
    if not groups or not valid(rate) or any(
            not valid(g["lr"]) or g["lr"] != rate for g in groups):
        raise RuntimeError("PPO scalar and optimizer learning rates differ or are invalid")
    if uses_profile(contract):
        expected = effective_profile_algorithm(contract)["schedule"] if actor_enabled else "fixed"
        if algorithm.schedule != expected:
            raise RuntimeError("78D profile learning-rate schedule differs from the active phase")
        if not ADAPTIVE_LR_BOUNDS[0] <= rate <= ADAPTIVE_LR_BOUNDS[1]:
            raise RuntimeError("78D profile learning rate exceeds the RSL adaptive bounds")
        if not actor_enabled and before_rate is not None and rate != before_rate:
            raise RuntimeError("Critic warmup changed its fixed learning rate")
    elif any(contract.get(key) for key in ("remediation", "continuation", "task_campaign")):
        if algorithm.schedule != "fixed" or rate != contract["learning_rate"]:
            raise RuntimeError("Fixed remediation learning rate changed during optimization")
    return rate


def restore_profile_clocks(contract, infos, *, stage, requested):
    """Exact restarts inherit all actor clocks; explicit CLI conflicts fail."""
    if not validate_profile_selection(contract):
        return None
    if "formal_final_rough_stage" in contract and stage != "terrain":
        raise ValueError("V7 final rough clocks require the explicit terrain destination")
    transition = (final_rough_stage_transition(contract, infos.get("stage"), stage)
                  or formal_rough_stage_transition(contract, infos.get("stage"), stage)
                  or formal_recovery_stage_transition(contract, infos.get("stage"), stage))
    if infos.get("stage") != stage and not transition:
        raise ValueError("78D exact restart cannot change the checkpoint stage")
    keys = ("stage_actor_update", "global_actor_update_offset", "global_actor_update", "actor_updates_consumed")
    clocks = {key: infos.get(key) for key in keys}
    if any(type(value) is not int or value < 0 for value in clocks.values()):
        raise ValueError("78D restart requires complete nonnegative saved actor clocks")
    if (clocks["global_actor_update"] != clocks["global_actor_update_offset"] + clocks["stage_actor_update"]
            or clocks["actor_updates_consumed"] < clocks["stage_actor_update"]):
        raise ValueError("78D restart has inconsistent saved actor clocks")
    if transition:
        # Global learning and paid consumption remain inherited; only the new
        # stage's exposure clock starts at zero. BudgetOwner never uses this
        # local clock to refund or reallocate already consumed PPO updates.
        clocks["stage_actor_update"] = 0
        clocks["global_actor_update_offset"] = clocks["global_actor_update"]
    for key, value in requested.items():
        if key not in clocks or value is not None and (type(value) is not int or value != clocks[key]):
            raise ValueError(f"Requested 78D restart clock differs from the checkpoint: {key}")
    return clocks


def configure_runner(config, contract, *, stage):
    """Operate on the final RSL5 dictionary, after all legacy overrides."""
    result = deepcopy(config)
    if not validate_profile_selection(contract):
        return result
    result.update(num_steps_per_env=24, logger="tensorboard",
                  obs_groups={"actor": ["policy"], "critic": ["critic"]})
    result["algorithm"].update(effective_profile_algorithm(contract))
    result["algorithm"].update(rnd_cfg=None, symmetry_cfg=None)
    for group in ("actor", "critic"):
        network = result[group]
        # RSL5 maps the old ActorCritic cfg to two MLPModel dictionaries.
        network.update(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)
    result["actor"]["distribution_cfg"].update(init_std=1., std_type="scalar")
    return result


def verify_runner(algorithm, config, contract):
    """Read back the instantiated runner instead of trusting the YAML alone."""
    if not validate_profile_selection(contract):
        return None
    expected = {key: value for key, value in effective_profile_algorithm(contract).items() if key != "class_name"}
    actual = {key: getattr(algorithm, key, None) for key in expected}
    mismatches = {key: {"expected": value, "actual": actual[key]}
                  for key, value in expected.items() if actual[key] != value}
    if mismatches:
        raise RuntimeError(f"Effective PPO settings differ: {mismatches}")
    actor = algorithm.actor
    layout = {"actor_dim": actor.obs_dim, "critic_dim": algorithm.critic.obs_dim,
              "action_dim": algorithm.storage.actions.shape[-1],
              "num_steps_per_env": algorithm.storage.num_transitions_per_env}
    if any(layout[key] != contract[key] for key in layout):
        raise RuntimeError(f"Instantiated PPO layout differs from the contract: {layout}")
    std = actor.distribution.std_param.detach()
    if not bool((std == 1.).all()):
        raise RuntimeError("Scratch actor did not initialize action std to 1")
    return {"profile": PROFILE, "algorithm": actual, **layout,
            "initial_std": std.cpu().tolist(), "scope": "instantiated_before_learning"}


def restore_rng(torch, env, infos, *, seed, mode="inherit"):
    """Fork RNG explicitly without changing restored model or optimizer state."""
    if mode not in ("inherit", "fresh_fork"):
        raise ValueError("Unknown RNG restore mode")
    if mode == "fresh_fork":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        env.generator.manual_seed(seed)
        if getattr(env, "round_trip_transport", None) is not None:
            env.round_trip_transport.reseed(seed)
        return {"mode": mode, "seed": seed, "parent_rng_restored": False,
                "transport_rng": "fresh_constructor_seed", "course_clock": "separately_owned"}
    if "rng_state" in infos:
        torch.set_rng_state(infos["rng_state"])
        env.generator.set_state(infos["env_rng_state"])
        if infos.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(infos["cuda_rng_states"])
    budget = getattr(env, "round_trip_transport", None)
    budget_rng = "not_selected"
    if budget is not None:
        state = infos.get("round_trip_rng_state")
        if state is None:
            if "src/wheeled_tasks/chassis/round_trip_transport.py" in infos.get("profile_source_sha256", {}):
                raise ValueError("Shared communication checkpoint is missing its owned RNG state")
            budget_rng = "new_component_stream_for_zero_communication_parent"
        else:
            if infos.get("round_trip_transport_configuration") != budget.cfg:
                raise ValueError("Shared communication checkpoint configuration differs")
            budget.generator.set_state(state)
            budget_rng = "inherited_for_future_episode_draws"
    return {"mode": mode, "parent_rng_restored": "rng_state" in infos,
            **({"round_trip_rng": budget_rng} if budget is not None else {})}
