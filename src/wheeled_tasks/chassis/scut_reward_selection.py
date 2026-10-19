"""Torch-free selection of the pinned rewards and V7 final rough adaptation.

This validates configuration intent, not runtime wiring, source command sampling
or skill qualification. The runtime must independently resolve body names and
all 18 physical inputs, then replace every legacy payment with this reward.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from types import MappingProxyType

TRACKING_REPAIR_KEY = "formal_tracking_reward_repair"


SCUT_FLAT_PROFILE = "scut0719_flat_v1"
SCUT_ROUGH_PROFILE = "scut0730_rough_v1"
V7_FINAL_ROUGH_PROFILE = "v7_final_rough_v1"
V7_FINAL_TERRAIN_SOURCE = "v7_final_b8ff79f3"
V7_FINAL_ROUGH_REWARD_VERSION = "v7-final-rough-chassis-payments-v1"
SCUT_ROUGH_PROFILES = (SCUT_ROUGH_PROFILE, V7_FINAL_ROUGH_PROFILE)
SCUT_SOURCE_PROFILES = (SCUT_FLAT_PROFILE, *SCUT_ROUGH_PROFILES)
V7_FINAL_UNDESIRED_BODY_NAMES = (
    "base_link", "L_link1", "L_link2", "R_link1", "R_link2",
    "LL_link1", "LL_link2", "LL_link3", "LL_link4",
    "RR_link1", "RR_link2", "RR_link3", "RR_link4",
)
V7_FINAL_ROUGH_WEIGHTS = MappingProxyType({
    "termination": -200., "leg_joint_acc": -1e-7, "leg_joint_vel": -.001,
    "joint_torque": -1e-5, "wheel_acc": -2e-9, "wheel_vel": -2e-6,
    "wheel_power": -1e-5, "lin_vel_z": -.5, "ang_vel_xy": -.05,
    "action_smoothness_leg": -.005, "action_rate": -.002, "action_smoothness_wheel": -.001,
    "flat_orientation_y_v": -2., "flat_orientation_y_exp": 1.,
    "flat_orientation_x_v": -2., "flat_orientation_x_exp": 1.,
    "track_lin_vel_xy": 1.25, "track_lin_vel_xy_square": -1.,
    "track_ang_vel_z": 1., "track_ang_vel_z_square": -1., "stand_still_lin_vel": -1.,
    "track_height_exp_tight": 1., "track_height_square": -1., "no_fork": -1.,
    "no_fork_square": -1., "undesired_contact": -2.,
    "airborne_joint_pos_limits": -10., "airborne_wheel_heading_x_centering": 10.,
    "airborne_air_wheel_zero_torque_exp": 20., "airborne_precontact_wheel_directional_speed": 10.,
    "airborne_precontact_wheel_directional_speed_shortfall": -10.,
})
LEGACY_FLAT_PROFILE = "scut_v14_flat_v5"
SOURCE_REWARD_VERSION = "scut0719-flat-source-v1"
SOURCE_COMMIT = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
MAPPING_FIELDS = frozenset(("root_velocity_reference", "root_height_reference",
                            "wheel_position_reference", "undesired_link_names", "terminal_link_names"))
FLAT_SKILL_KINDS = frozenset(("stand", "forward", "backward", "curve", "height",
                              "rotate", "start_stop", "velocity_curve"))

# These objects have inconsistent legacy constructor activation rules (key
# presence, truthiness, or enabled). The new profile requires ABSENCE, not a
# disabled shell; never silently rewrite the reviewed contract to remove them.
INCOMPATIBLE_OBJECTS = (
    "dense_tracking", "reference_reward", "landing_tracking", "jump_landing",
    "reward_payment", "payment_runtime", "episode_payment", "reward_overrides",
    "rewards", "command_reference", "task_semantics", "full_task_fsm",
    "jump_assist", "step_assist", "step_jump_task", "step_climb_task", "ramp_flight",
    "recovery_training", "recovery_takeover", "recovery_hold_cost", "recovery_curriculum",
    "recovery_prefix_curriculum", "release_stop", "new_asset_curriculum",
    "terrain_route_runout", "scut_heading_command", "integrated_production",
    "integrated_runtime_candidate", "task_campaign", "remediation", "continuation",
    "stationary_tracking", "performance_curriculum", "special_mode_activation",
)
INCOMPATIBLE_FLAGS = ("precision_tracking", "height_tracking", "track_height_during_recovery",
                      "jump_cycle_reset_on_accepted", "step_jump_landing_stop_target")
INCOMPATIBLE_PAYMENTS = ("success_event_bonus", "termination_event_cost", "working_margin_weight",
                         "height_l1_weight", "scut_effort_reward_scale", "success_bonus",
                         "failure_tail_weight", "tail_weight")


def validate_command_decoupling(config):
    """Validate the explicit V7 pure-translation yaw-cost candidate."""
    if config is None:
        return None
    if (not isinstance(config, Mapping)
            or config.get("version") != "v7-pure-translation-yaw-cost-v2"
            or type(config.get("enabled")) is not bool):
        raise ValueError("V7 command decoupling requires its version and explicit boolean enabled")
    fields = {"version", "enabled"}
    if config["enabled"]:
        fields.add("coefficient_per_rad2_s2")
        coefficient = config.get("coefficient_per_rad2_s2")
        if (type(coefficient) not in (int, float)
                or not math.isfinite(coefficient) or coefficient <= 0):
            raise ValueError("V7 command decoupling requires a positive finite coefficient_per_rad2_s2")
    if set(config) != fields:
        raise ValueError("V7 command decoupling contains missing or unsupported fields")
    return deepcopy(dict(config))


def _inert(value):
    return (value is None or value is False
            or isinstance(value, Mapping) and (not value or dict(value) == {"enabled": False}
                                                and type(value["enabled"]) is bool))


def _flat_terrain(value, label, terrain_kind="flat"):
    if value != terrain_kind and not (isinstance(value, (list, tuple)) and list(value) == [terrain_kind]):
        raise ValueError(f"SCUT reward requires explicit {terrain_kind} terrain: {label}")


def _ordinary_spec(spec, label, *, require_kind=False, terrain_kind="flat"):
    if not isinstance(spec, Mapping):
        raise ValueError(f"SCUT0719 flat specification must be a mapping: {label}")
    if require_kind or "kind" in spec:
        if spec.get("kind") not in FLAT_SKILL_KINDS:
            raise ValueError(f"SCUT0719 flat reward has no adapter for skill: {label}")
    if "terrain" in spec:
        _flat_terrain(spec["terrain"], label, terrain_kind)
    if "episode_seconds" in spec:
        if type(spec["episode_seconds"]) not in (int, float) or spec["episode_seconds"] != 20.:
            raise ValueError(f"SCUT0719 flat reward requires episode_seconds=20: {label}")
    if "mode" in spec and (type(spec["mode"]) is not int or spec["mode"] not in (0, 1)):
        raise ValueError(f"SCUT0719 flat reward forbids jump/step/recovery modes: {label}")
    if "task" in spec and spec["task"] != "survive":
        raise ValueError(f"SCUT0719 flat reward has no adapter for task: {label}")
    for key in ("reference_velocity", "world_velocity", "world_velocity_xy", "command_world_xy",
                "request_events", "command_sequence", "jump", "step", "recovery", "ramp_flight"):
        if key in spec and not _inert(spec[key]):
            raise ValueError(f"SCUT0719 flat reward has no adapter for {label}.{key}")
    if "skill" in spec:
        _ordinary_spec(spec["skill"], label + ".skill", require_kind=True, terrain_kind=terrain_kind)


def _check_cases(cases, label, terrain_kind="flat"):
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError(f"SCUT0719 flat reward requires nonempty explicit cases: {label}")
    names = []
    for index, case in enumerate(cases):
        at = f"{label}[{index}]"
        _ordinary_spec(case, at, terrain_kind=terrain_kind)
        _flat_terrain(case.get("terrain"), at, terrain_kind)
        if not isinstance(case.get("name"), str) or not case["name"].strip():
            raise ValueError(f"SCUT0719 flat case requires an explicit name: {at}")
        names.append(case["name"])
    if len(set(names)) != len(names):
        raise ValueError(f"SCUT0719 flat case names must be unique: {label}")
    return len(cases)


def validate_reward_selection(config, *, stage_name=None, case_specs=None):
    """Validate reward dispatch before native imports and after role overrides.

    Legacy absence and the known legacy profile retain their existing semantics.
    Every other name is rejected, preventing an unknown profile from falling
    through to the old basic reward. For the new profile this function inspects
    all declared stage/scene/skill/case types, not just a friendly stage name.
    ``case_specs`` adds an effective fixed-case check; it never hides the cases
    already declared by the contract. Nonstandard birth/capture modules remain
    available. Scripted flat recovery needs its explicit birth/control overlay;
    jumps, routes and world-XY spin translation have no source reward adapter.

    Returned fields describe the selected static contract. They are not evidence
    that payments, contacts, termination or source sampling ran successfully.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Reward selection requires a configuration mapping")
    profile = config.get("reward_profile")
    if TRACKING_REPAIR_KEY in config:
        from .tracking_reward_repair import SUPPORTED_VERSIONS
        if (config[TRACKING_REPAIR_KEY] not in SUPPORTED_VERSIONS
                or profile not in (SCUT_FLAT_PROFILE, SCUT_ROUGH_PROFILE)
                or config.get("formal_v7_flat") is not True
                or (config.get("actor_dim"), config.get("critic_dim")) != (35, 78)):
            raise ValueError("Tracking repair requires an explicit V7 flat or small-rough contract")
    if profile is not None and profile not in (LEGACY_FLAT_PROFILE, *SCUT_SOURCE_PROFILES):
        raise ValueError("Unknown reward_profile; refusing legacy reward fallback")
    if profile not in SCUT_SOURCE_PROFILES:
        if "formal_stepup_command" in config:
            raise ValueError("V7 StepUp requires the explicit source reward profile")
        if "formal_command_decoupling" in config:
            raise ValueError("V7 command decoupling requires the explicit source reward profile")
        if "scut_reward_mapping" in config:
            raise ValueError("reference physical mapping requires the explicit flat baseline reward profile")
        return {"reward_profile": profile, "scut_flat_selected": False,
                "scut_rough_selected": False, "scut_source_selected": False,
                "selection_scope": "legacy_behavior_unchanged", "runtime_verified": False}

    decoupling = validate_command_decoupling(config.get("formal_command_decoupling"))
    stepup = None
    if "formal_stepup_command" in config:
        from .stepup_command import validate_stepup_command
        stepup = validate_stepup_command(config["formal_stepup_command"])
    if "formal_stepup_command" in config and (config.get("formal_v7_flat") is not True
            or config.get("actor_dim") != 35 or config.get("critic_dim") != 78):
        raise ValueError("V7 StepUp requires the formal35D/78D source command owner")

    final_rough = profile == V7_FINAL_ROUGH_PROFILE
    rough = profile in SCUT_ROUGH_PROFILES
    if final_rough:
        if decoupling is not None and decoupling["enabled"]:
            raise ValueError("V7 final rough does not implement the additional command decoupling payment")
        for field in config:
            if (not isinstance(field, str)
                    or field.startswith(("formal_airborne_", "airborne_"))
                    or field in ("airborne", "formal_airborne", "terrain_command_resample",
                                 "stair_state_machine", "stair_state_machine_cfg", "wheel_forward_scan_cfg")):
                raise ValueError(f"V7 final rough has no adapter for {field}")
        if "formal_flat_rehearsal" in config:
            from .fixed_flat_rehearsal import validate_fixed_flat_rehearsal
            rehearsal = validate_fixed_flat_rehearsal(config)
        else:
            rehearsal = None
        if rehearsal is not None and "reward_profile" in rehearsal:
            if rehearsal["reward_profile"] != V7_FINAL_ROUGH_PROFILE:
                raise ValueError("V7 final rough requires unified final rough rewards on every rehearsal lane")
        if ("flat_rehearsal_reward_profile" in config
                and config["flat_rehearsal_reward_profile"] != V7_FINAL_ROUGH_PROFILE):
            raise ValueError("V7 final rough requires flat_rehearsal_reward_profile=v7_final_rough_v1")
    terrain_kind = "rough" if rough else "flat"
    if rough:
        terrain = config.get("formal_terrain")
        final_geometry = final_rough and terrain == {"type": "rough", "source": V7_FINAL_TERRAIN_SOURCE}
        if not final_geometry and terrain != {"type": "rough", "source": "scut0730"}:
            raise ValueError("rough baseline rough requires its pinned shared terrain generator")
        if config.get("critic_height_reference") != "root_link_world_z":
            raise ValueError("rough baseline critic height must remain root_link_world_z")
    elif "formal_terrain" in config:
        terrain = config["formal_terrain"]
        if not isinstance(terrain, Mapping) or terrain.get("type") != "plane":
            raise ValueError("Flat reward cannot silently train on rough terrain")

    for field, expected in (("physics_dt", .005), ("policy_dt", .02), ("episode_seconds", 20.)):
        if type(config.get(field)) not in (int, float) or config[field] != expected:
            raise ValueError(f"SCUT0719 flat reward requires {field}={expected}")
    mapping = config.get("scut_reward_mapping")
    if not isinstance(mapping, Mapping) or set(mapping) != MAPPING_FIELDS:
        raise ValueError("reference reward mapping requires exactly the five explicit physical mapping fields")
    for field in ("root_velocity_reference", "root_height_reference", "wheel_position_reference"):
        if not isinstance(mapping[field], str) or not mapping[field].strip():
            raise ValueError(f"SCUT reward mapping {field} must be explicit")
    height_reference = "root_link_z_minus_base_ground_scan" if rough else "root_link_z_minus_env_origin"
    if mapping["root_height_reference"] != height_reference:
        raise ValueError(f"SCUT reward height must use {height_reference}")
    if mapping["root_velocity_reference"] != "root_rigid_body_com_in_root_link_frame":
        raise ValueError("reference reward velocity must use root_rigid_body_com_in_root_link_frame")
    if mapping["wheel_position_reference"] != "wheel_link_origins_in_root_link_frame":
        raise ValueError("reference reward wheels must use wheel_link_origins_in_root_link_frame")
    for field in ("undesired_link_names", "terminal_link_names"):
        names = mapping[field]
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError(f"SCUT reward mapping {field} requires unique nonempty body names")
    if final_rough:
        if set(mapping["undesired_link_names"]) != set(V7_FINAL_UNDESIRED_BODY_NAMES):
            raise ValueError("V7 final rough requires the complete explicit 13-body undesired contact set")
        if tuple(mapping["terminal_link_names"]) != ("base_link",):
            raise ValueError("V7 final rough requires base_link as its only terminal contact body")

    from .flat_episode_selection import validate_scut_flat_recovery
    recovery = validate_scut_flat_recovery(config)
    for field in INCOMPATIBLE_OBJECTS:
        if field == "recovery_training" and recovery is not None:
            continue
        if field in config:
            raise ValueError(f"SCUT0719 flat reward requires excluded configuration field to be absent: {field}")
    for field in INCOMPATIBLE_FLAGS:
        if field in config and config[field] is not False:
            raise ValueError(f"SCUT0719 flat reward excludes legacy flag: {field}")
    for field in INCOMPATIBLE_PAYMENTS:
        if field in config and (type(config[field]) not in (int, float)
                                or not math.isfinite(config[field]) or config[field] != 0.):
            raise ValueError(f"SCUT0719 flat reward excludes legacy payment: {field}")
    for field in ("reward_payment_profile", "reward_velocity_reference"):
        if field in config and config[field] is not None:
            raise ValueError(f"SCUT0719 flat reward has no adapter for {field}")

    stages = config.get("stages")
    if not isinstance(stages, (list, tuple)) or not stages:
        raise ValueError("flat baseline flat reward requires explicit nonempty stages")
    stage_names = []
    for stage in stages:
        _ordinary_spec(stage, "stages", terrain_kind=terrain_kind)
        _flat_terrain(stage.get("terrain"), "stages", terrain_kind)
        if not isinstance(stage.get("name"), str) or not stage["name"].strip():
            raise ValueError("flat baseline flat stages require explicit names")
        stage_names.append(stage["name"])
    if len(set(stage_names)) != len(stage_names):
        raise ValueError("flat baseline flat stage names must be unique")
    if stage_name is not None and stage_name not in stage_names:
        raise ValueError("Selected flat baseline flat stage is not declared")
    enabled = config.get("enabled_stages", stage_names)
    if (not isinstance(enabled, (list, tuple)) or not enabled
            or any(name not in stage_names for name in enabled)
            or stage_name is not None and stage_name not in enabled):
        raise ValueError("flat baseline enabled stages differ from declared/selected stages")
    if "formal_small_rough_mix" in config:
        from .small_rough_course import selection
        selection(config)
        scene_count = sum(_check_cases([scene], "scene_groups",
            "rough" if scene["name"] == "source_rough" else "flat")
            for scene in config["scene_groups"])
    else:
        scene_count = _check_cases(config.get("scene_groups"), "scene_groups", terrain_kind)
    specs = config.get("skill_specs", {})
    if not isinstance(specs, Mapping):
        raise ValueError("flat baseline flat skill_specs must be a mapping")
    for name, spec in specs.items():
        _ordinary_spec(spec, f"skill_specs.{name}", require_kind=True, terrain_kind=terrain_kind)
    evaluation_count = 0
    if "evaluation" in config:
        evaluation = config["evaluation"]
        if not isinstance(evaluation, Mapping):
            raise ValueError("flat baseline flat evaluation must be a mapping")
        _ordinary_spec(evaluation, "evaluation", terrain_kind=terrain_kind)
        evaluation_count = _check_cases(evaluation.get("cases"), "evaluation.cases", terrain_kind)
    effective_count = _check_cases(case_specs, "case_specs", terrain_kind) if case_specs is not None else 0
    result = {"reward_profile": profile, "scut_flat_selected": not rough,
            "scut_rough_selected": rough, "scut_source_selected": True,
            "selection_scope": ("flat_source_commands_with_scripted_recovery" if recovery is not None else
                "rough_body_commands" if rough else "flat_body_commands_and_nonstandard_capture_only"),
            "source_reward_version": "scut0730-rough-source-v1" if rough else SOURCE_REWARD_VERSION, "source_commit": SOURCE_COMMIT,
            "nonzero_term_count": 26, "physics_dt_s": .005, "policy_dt_s": .02,
            "episode_horizon_seconds": 20.,
            "payment_dt_multiplications": 1, "legacy_reward_merge": False,
            "excluded_fields_must_be_absent": True,
            "excluded_configuration_fields": [field for field in INCOMPATIBLE_OBJECTS
                if recovery is None or field != "recovery_training"],
            **({"scripted_recovery_selected": True} if recovery is not None else {}),
            "success_bonus": False, "failure_tail_settlement": False,
            "termination_weight_per_s": -200., "termination_payment_per_event": -4.,
            "scut_reward_mapping": deepcopy(dict(mapping)),
            "stage_names": stage_names, "scene_count": scene_count,
            "evaluation_case_count": evaluation_count, "effective_case_count": effective_count,
            "source_command_sampler_verified": False, "physical_inputs_verified": False,
            "runtime_verified": False, "all_skill_coverage": False}
    if "formal_command_decoupling" in config:
        result["formal_command_decoupling"] = decoupling
    if "formal_stepup_command" in config:
        result["formal_stepup_command"] = stepup
    if final_rough:
        result.update(v7_final_rough_selected=True,
            source_reward_version=V7_FINAL_ROUGH_REWARD_VERSION, nonzero_term_count=31,
            nonzero_weights=dict(V7_FINAL_ROUGH_WEIGHTS),
            local_reward_adaptations=["passive_knee_inner_angle", "relative_height_clip"],
            source_reward_terms_unchanged=False,
            relative_height_clip_m=[.05, .45],
            knee_coordinate="passive_thigh_shank_inner_angle_rad",
            termination_confirmation_policy_ticks=10, termination_confirmation_seconds=.2,
            termination_airborne_payment_per_event=-24.,
            local_terrain_adaptation=("final_source_geometry_with_own_birth_and_rehearsal" if final_geometry
                                      else "shared_0730_rough_geometry"),
            source_terrain_generator_equivalent=False,
            airborne_command_sampling_enabled=False, stair_state_machine_enabled=False,
            stepup_command_owner="separate_explicit_component",
            source_final_environment_equivalent=False)
        if final_geometry:
            result["final_source_geometry_parameters_selected"] = True
        from .small_rough_course import FINAL_MOTION_VERSION
        exact = config.get("formal_small_rough_mix", {}).get("version") == FINAL_MOTION_VERSION
        result["tracking_commands"] = ("actor_visible_targets_without_reward_deadzone"
            if exact else "source_command_deadzone")
        result["velocity_height_gate_enabled"] = False
        if exact:
            result["local_reward_adaptations"].append("exact_body_command_tracking")
            result["standing_command_gate"] = "vx_exactly_zero"
    if decoupling is not None and decoupling["enabled"]:
        result.update(local_reward_adaptations=["v7_pure_translation_yaw_cost"],
                      additional_payment_term_count=1,
                      source_reward_terms_unchanged=True)
    if TRACKING_REPAIR_KEY in config:
        result[TRACKING_REPAIR_KEY] = config[TRACKING_REPAIR_KEY]
        result["source_reward_terms_unchanged"] = False
        result["linear_square_weight"] = -2.0
        result["tracking_commands"] = "actor_visible_targets_without_reward_deadzone"
        result["standing_command_gate"] = "vx_exactly_zero"
    return result


def reward_source_files(config):
    """Return only this selection's dependencies for source-provenance capture."""
    selected = validate_reward_selection(config)
    sources = ["src/wheeled_tasks/chassis/scut_reward_selection.py"]
    if TRACKING_REPAIR_KEY in config:
        sources.append("src/wheeled_tasks/chassis/tracking_reward_repair.py")
    if selected["scut_source_selected"]:
        sources.extend(("src/wheeled_tasks/chassis/scut_flat_reward.py",
                        "src/wheeled_tasks/chassis/scut_flat_reward_runtime.py"))
        if selected.get("v7_final_rough_selected", False):
            sources.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in (
                "final_rough_reward", "final_rough_reward_runtime", "source_airborne_state",
                "wheel_ground_scan", "relative_knee_geometry"))
            sources.extend(f"references/v7_final_rough_b8ff79f3/{name}" for name in (
                "airborne.py", "env_cfg.py", "source_manifest.json", "LICENSE"))
    elif selected["reward_profile"] == LEGACY_FLAT_PROFILE:
        sources.append("src/wheeled_tasks/chassis/rewards.py")
    return sources
