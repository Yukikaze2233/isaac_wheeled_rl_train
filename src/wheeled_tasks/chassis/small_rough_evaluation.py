"""Evaluate a flat/recovery actor on small rough without rewriting its parent."""
from copy import deepcopy


SURFACE = "small_rough_regression_v1"


def configuration(parent):
    """Change the measurement surface and owners, preserving physical inputs.

    This is an inference overlay, not a learning-state migration. Nominal
    births and the existing rough source sampler measure retained locomotion;
    they do not assess recovery on uneven ground or define training fractions.
    """
    from .training_profile import validate_profile_selection
    from .small_rough_course import KEY, selection
    mixed = selection(parent) is not None
    if (not mixed and (parent.get("reward_profile") != "scut0719_flat_v1"
            or parent.get("curriculum_stage") != "foundation"
            or parent.get("enabled_stages") != ["foundation"]
            or parent.get("formal_terrain", {}).get("type") != "plane")
            or parent.get("physx_solver_type") != 1
            or not validate_profile_selection(parent)):
        raise ValueError("Small rough regression requires an authenticated TGS retained-skill parent")
    result = deepcopy(parent)
    for key in ("recovery_training", "formal_scripted_recovery_mix", "formal_flat_refresh",
                "formal_flat_rehearsal", "formal_dynamic_nonsteady_birth", "evaluation",
                "flat_special_mode_iteration_cap", KEY, "formal_stage_transition",
                "formal_stage_optimizer"):
        result.pop(key, None)
    result.update(
        reward_profile="scut0730_rough_v1", curriculum_stage="terrain", enabled_stages=["terrain"],
        stages=[dict(stage, name="terrain", terrain=["rough"]) for stage in parent["stages"]],
        scene_groups=[{"name": "source_rough", "terrain": ["rough"], "fraction": 1.}],
        formal_terrain={"type": "rough", "source": "scut0730"},
        formal_rough_initial_level=0, formal_foundation_nominal_birth=True,
        critic_height_reference="root_link_world_z",
        formal_stage_policy={"later_skills_enabled": False,
            "no_automatic_iteration_based_promotion": True,
            "scope": "inference only; original actor clocks retained, no PPO or optimizer migration"})
    result["scut_reward_mapping"]["root_height_reference"] = "root_link_z_minus_base_ground_scan"
    validate_profile_selection(result)
    return result


def retained_flat_configuration(parent, scenes, *, scripted):
    """Project an authenticated retained-skill actor onto nominal plane cases.

    Only inference ownership, scene allocation and flat payment/reference
    selection change. In particular the checkpoint clocks, mechanics, transport
    and recovery controller are not replaced by an older training contract.
    """
    from .small_rough_course import KEY, selection
    from .training_profile import validate_profile_selection
    from .flat_episode_selection import FLAT_SPECIAL_SCHEDULE
    mixed = selection(parent) is not None
    if (not validate_profile_selection(parent) or not mixed and (
            parent.get("reward_profile") != "scut0719_flat_v1"
            or parent.get("curriculum_stage") != "foundation"
            or parent.get("formal_terrain", {}).get("type") != "plane")):
        raise ValueError("Retained flat measurements require a flat/recovery or mixed parent")
    result = deepcopy(parent)
    for key in (KEY, "formal_scripted_recovery_mix", "formal_flat_refresh",
                "formal_flat_rehearsal", "formal_dynamic_nonsteady_birth", "evaluation",
                "formal_stage_transition", "formal_stage_optimizer", "formal_rough_initial_level"):
        result.pop(key, None)
    if not scripted:
        result.pop("recovery_training", None)
    result.update(
        reward_profile="scut0719_flat_v1", curriculum_stage="foundation", enabled_stages=["foundation"],
        stages=[dict(stage, name="foundation", terrain=["flat"]) for stage in parent["stages"]],
        scene_groups=deepcopy(scenes),
        formal_terrain={"type": "plane", "source": "scut0719", "env_spacing_m": 4.},
        formal_foundation_nominal_birth=True, critic_height_reference="root_link_world_z",
        formal_stage_policy={"later_skills_enabled": False,
            "no_automatic_iteration_based_promotion": True,
            "flat_special_mode_schedule": FLAT_SPECIAL_SCHEDULE,
            "scope": "inference only; retained clocks and nominal plane births, no PPO"})
    result["scut_reward_mapping"]["root_height_reference"] = "root_link_z_minus_env_origin"
    if scripted:
        from .fixed_flat_rehearsal import SCRIPTED_RECOVERY_MIX_VERSION, FLAT_MIXED_REFRESH_VERSION
        # The existing typed evaluator suppresses training allocation and
        # makes ordinary births nominal; retain the recovery selector's
        # complete configuration rather than weakening its admission checks.
        result.update(formal_scripted_recovery_mix=SCRIPTED_RECOVERY_MIX_VERSION,
            formal_flat_refresh=FLAT_MIXED_REFRESH_VERSION,
            formal_foundation_nominal_birth=False,
            formal_flat_rehearsal=deepcopy(parent["formal_flat_rehearsal"]),
            formal_dynamic_nonsteady_birth=parent["formal_dynamic_nonsteady_birth"])
    validate_profile_selection(result)
    return result


def validate_overlay(protocol, overlay):
    """Reuse the existing 480-lane mean-policy, one-episode rough protocol."""
    expected = dict(schema="flat-policy-evaluation-v1", mode="source_distribution",
        actor_mode="mean", episodes_per_lane=1, environment_seed=290929, actor_seed=290929,
        policy_dt_s=.02, horizon_seconds=20., center_reference="root_link_origin_world",
        assessment="measurement_only")
    if (protocol != expected or overlay.get("num_envs") != 480
            or overlay.get("lane_case_ids") != ["source_distribution"] * 480
            or overlay.get("record_diagnostics") is not True
            or overlay.get("evaluation_surface") != SURFACE):
        raise ValueError("Small rough regression requires the unchanged 480-lane mean source protocol")
    selected = overlay.get("contact_generation_distance")
    from .contact_generation_distance import SUPPORTED_VERSIONS
    if selected is not None and selected not in SUPPORTED_VERSIONS:
        raise ValueError("Unknown small rough contact-distance candidate")


def receipt(parent, effective):
    changed = sorted(key for key in parent.keys() | effective.keys()
                     if (key in parent) != (key in effective) or parent.get(key) != effective.get(key))
    return {"surface": SURFACE, "changed_configuration_fields": changed,
            "checkpoint_contract_rewritten": False, "optimizer_or_training_state_loaded": False,
            "physics_and_control": "inherited; optional named contact-distance intervention is recorded separately",
            "births": "nominal pose on actual terrain, initial row 0; full generated mesh remains accessible",
            "commands_and_rewards": "existing small rough source owners; no fixed flat or scripted recovery lanes",
            "actor_clock": "parent clocks held constant during inference",
            "qualification": "measurement only; not proof of rough recovery or a complete training course"}
