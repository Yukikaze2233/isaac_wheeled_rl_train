"""Select source flat command/birth ownership without granting native admission.

Source: reference b8ff79f3, 0719 archive. Mechanical reachability, ground placement
and whole-interval PPO masking are explicit local adapters, not source settings.
The CPU component contract is separate from a native experiment contract.
"""
from collections.abc import Mapping
import math

VERSION = "scut0719-flat-episode-cpu-v1"
FIELDS = frozenset(("version", "branch_table", "branch_table_sha256", "length_definition",
                    "training_knee_upper_deg", "minimum_clearance_m", "birth_rejection"))
EXCLUDED = ("skill_specs", "nonstandard_start", "ordinary_posture_curriculum",
            "command_curriculum", "command_slew", "motion_limits", "push_interval",
            "push_randomization", "height_curriculum")
FLAT_SPECIAL_SCHEDULE = "scut0719_pinned_v1"
SCUT_FLAT_RECOVERY = "scut0719_scripted_handoff_v1"


def validate_scut_flat_recovery(config, root=None):
    """Explicit recovery births within the source command/reward owner.

    Scene fractions allocate lanes; recovery_pose selects a frozen flat profile.
    Neither this admission nor the existing FSM grades hardware qualification.
    """
    settings = config.get("recovery_training", {})
    selected = isinstance(settings, Mapping) and "source_flat_handoff" in settings
    scenes = config.get("scene_groups", [])
    declared = (isinstance(scenes, (list, tuple))
                and any(isinstance(scene, Mapping) and "recovery_pose" in scene for scene in scenes))
    if not selected:
        if declared and config.get("reward_profile") == "scut0719_flat_v1":
            raise ValueError("Source recovery poses require explicit scripted handoff selection")
        return None
    terrain = config.get("formal_terrain", {})
    from .training_profile import formal_recovery_stage_transition
    recovery_transition = formal_recovery_stage_transition(config, "terrain", "foundation")
    mixed_rough = False
    if "formal_small_rough_mix" in config:
        from .small_rough_course import selection as small_rough_selection
        mixed_rough = small_rough_selection(config) is not None
    if (settings["source_flat_handoff"] != SCUT_FLAT_RECOVERY or settings.get("enabled") is not True
            or config.get("formal_v7_flat") is not True
            or not mixed_rough and (config.get("reward_profile") != "scut0719_flat_v1"
                or config.get("curriculum_stage") != "foundation"
                or config.get("enabled_stages") != ["foundation"]
                or not isinstance(terrain, Mapping) or terrain.get("type") != "plane"
                or terrain.get("source") != "scut0719")
            or "flat_episode_runtime" not in config
            or "formal_stage_transition" in config and not (recovery_transition or mixed_rough)
            or any(key in config for key in ("skill_specs", "v7_curriculum_runtime"))):
        raise ValueError("Scripted source recovery requires same-ABI foundation/0719 plane with one source command owner")
    from .response_migration import validate_recovery_configuration
    receipt = validate_recovery_configuration(settings, root)
    script_fixes = {"side_speed_rad_s", "complete_started_blend"}
    if script_fixes & settings.keys() and "direct_takeover" not in settings:
        # Explicitly reuse the deployed V6 pair; absence retains the FSM defaults.
        side_speed = settings.get("side_speed_rad_s")
        if (not script_fixes <= settings.keys() or type(side_speed) not in (int, float)
                or not math.isfinite(side_speed) or side_speed != 1.5
                or settings.get("complete_started_blend") is not True):
            raise ValueError("Source recovery script fixes require side_speed_rad_s=1.5 and complete_started_blend=true together")
        receipt.update(side_speed_rad_s=side_speed, complete_started_blend=True)
    if any(type(value) in (int, float) and type(settings[key]) is bool
           for key, value in receipt.items() if key != "status"):
        raise ValueError("Source recovery numeric settings cannot be booleans")
    allowed = set(receipt) - {"status"} | {"profiles_file", "profiles_sha256", "height_lookup_file",
        "height_lookup_sha256", "wheel_balance", "source_flat_handoff"}
    if set(settings) - allowed:
        raise ValueError("Source recovery cannot add legacy samplers or change the frozen FSM")
    if not isinstance(scenes, (list, tuple)) or any(not isinstance(scene, Mapping) for scene in scenes):
        raise ValueError("Source recovery requires explicit scene metadata")
    fractions = [scene.get("fraction") for scene in scenes]
    if (not scenes or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1
                          for v in fractions) or not math.isclose(math.fsum(fractions), 1., abs_tol=1e-9)):
        raise ValueError("Source recovery requires explicit positive scene fractions summing to one")
    profiles = {}
    names = [scene.get("name") for scene in scenes]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Source recovery scene names must be explicit and unique")
    for scene in scenes:
        if set(scene) - {"name", "terrain", "fraction", "recovery_pose"}:
            raise ValueError("Source recovery scene metadata cannot introduce another command or task owner")
        if scene.get("terrain") != (["rough"] if mixed_rough and scene["name"] == "source_rough" else ["flat"]):
            raise ValueError("Source recovery scene groups must explicitly use flat terrain")
        if "recovery_pose" in scene:
            name = scene["recovery_pose"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Recovery scenes require an explicit frozen profile name")
            profiles[scene["name"]] = name
    ordinary_groups = [scene["name"] for scene in scenes if "recovery_pose" not in scene]
    if not profiles or not ordinary_groups:
        raise ValueError("Source recovery requires explicit recovery profiles and an ordinary source rehearsal allocation")
    if root is not None:
        import hashlib
        import json
        from pathlib import Path
        root = Path(root)
        table = json.loads((root / settings["profiles_file"]).read_bytes())
        lookup = json.loads((root / settings["height_lookup_file"]).read_bytes())
        manifest_bytes = (root / config["asset_directory"] / "manifest.json").read_bytes()
        spec_bytes = (root / config["asset_directory"] / "model_spec.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if (table.get("asset_manifest_sha256") != config["asset_manifest_sha256"]
                or hashlib.sha256(manifest_bytes).hexdigest() != config["asset_manifest_sha256"]
                or table.get("model_spec_sha256") != hashlib.sha256(spec_bytes).hexdigest()
                or table.get("control_joint_names") != manifest["control_joint_names"]
                or lookup.get("asset_manifest_sha256") != config["asset_manifest_sha256"]
                or lookup.get("model_spec_sha256") != table.get("model_spec_sha256")):
            raise ValueError("Source recovery profiles/lookup must bind the actual mechanical asset")
        for name in profiles.values():
            profile = table.get("profiles", {}).get(name, {})
            if (profile.get("enabled") is not True or profile.get("terrain") != "flat"
                    or profile.get("required_platform_height_m", 0.) != 0.):
                raise ValueError(f"Shared plane requires an enabled flat recovery profile: {name}")
    return {**receipt, "source_flat_handoff": SCUT_FLAT_RECOVERY,
            "profiles_by_scene": profiles, "ordinary_source_groups": ordinary_groups,
            "physical_qualification": False}


def formal_flat_special_schedule(config):
    """Explicit source schedule selection; never a skill or stage promotion."""
    policy = config.get("formal_stage_policy", {})
    if not isinstance(policy, Mapping) or "flat_special_mode_schedule" not in policy:
        return False
    terrain = config.get("formal_terrain", {})
    from .training_profile import formal_recovery_stage_transition
    if (policy["flat_special_mode_schedule"] != FLAT_SPECIAL_SCHEDULE
            or config.get("formal_v7_flat") is not True
            or config.get("formal_learning_state_resume") is not True
            or config.get("reward_profile") != "scut0719_flat_v1"
            or config.get("curriculum_stage") != "foundation"
            or config.get("enabled_stages") != ["foundation"]
            or not isinstance(terrain, Mapping) or terrain.get("type") != "plane"
            or terrain.get("source") != "scut0719"
            or "formal_stage_transition" in config and not formal_recovery_stage_transition(config, "terrain", "foundation")
            or "flat_special_mode_iteration_cap" in config):
        raise ValueError("Pinned 0719 special schedule requires same-stage formal flat/plane learning resume and absent cap")
    return True


def validate_flat_episode_selection(config):
    settings = config.get("flat_episode_runtime")
    if "flat_episode_runtime" not in config:
        return None
    from .scut_reward_selection import SCUT_SOURCE_PROFILES, validate_reward_selection
    validate_reward_selection(config)
    if config.get("reward_profile") not in SCUT_SOURCE_PROFILES:
        raise ValueError("Source flat episode ownership requires its exclusive source reward")
    from .training_profile import PROFILE, CRITIC_VERSION
    if (config.get("training_profile") != PROFILE or config.get("critic_dim") != 78
            or config.get("critic_observation_version") != CRITIC_VERSION
            or config.get("actor_observation_source") != "scut35_encoders_imu_commands"
            or config.get("actor_frame_dim") != 35 or config.get("manual_context35") is not True):
        raise ValueError("Source flat episode requires the named actor35/critic78 scratch interface")
    if not isinstance(settings, Mapping) or set(settings) != FIELDS or settings["version"] != VERSION:
        raise ValueError("Unknown or incomplete source flat episode component contract")
    for key in EXCLUDED:
        if key in config:
            raise ValueError(f"Source flat episode requires legacy owner field absent: {key}")
    for key in ("branch_table", "length_definition"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Source flat episode requires explicit {key}")
    if settings["length_definition"] != "hip_axis_projected":
        raise ValueError("Source planar IK length requires the explicit hip-axis projected mapping")
    digest = settings["branch_table_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Source branch table needs its exact SHA-256")
    for key in ("training_knee_upper_deg", "minimum_clearance_m"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Source mechanical adapter requires positive finite {key}")
    reject = settings["birth_rejection"]
    if (not isinstance(reject, Mapping) or reject.get("mode") not in ("abort", "redraw")
            or set(reject) != ({"mode"} if reject["mode"] == "abort" else {"mode", "max_attempts"})):
        raise ValueError("Birth rejection requires an explicit abort or bounded redraw adapter")
    if reject["mode"] == "redraw" and (type(reject["max_attempts"]) is not int or reject["max_attempts"] <= 0):
        raise ValueError("Birth redraw must have an explicit positive attempt bound")
    from .flat_interval_events import validate_interval_selection
    validate_interval_selection(config)
    return dict(settings)


def require_native_flat_admission(config, manifest=None):
    """Select the authorized formal entry; generic CPU components stay blocked.

    This validates configuration only. Native contact/reset evidence and skill
    qualification are separate from selecting the formal training entry.
    """
    settings = validate_flat_episode_selection(config)
    if config.get("formal_v7_flat") is True:
        if settings is None or config.get("research_training_authorized") is not True:
            raise ValueError("Formal V7 requires the explicit source flat profile")
        from .training_profile import validate_profile
        validate_profile(config)
        special = formal_flat_special_schedule(config)
        if (config.get("reward_profile") == "scut0719_flat_v1" and not special
                and (type(config.get("flat_special_mode_iteration_cap")) is not int
                     or config["flat_special_mode_iteration_cap"] != 0)):
            raise ValueError("Initial V7 foundation must hold later special modes for capability assessment")
        if manifest is not None and settings["training_knee_upper_deg"] > manifest["knee_inner_limits_deg"][1]:
            raise ValueError("Training knee range exceeds the authenticated asset")
        return
    if settings is not None:
        raise ValueError("Source flat episode is CPU-only; native experiment admission is not implemented")
    if isinstance(manifest, Mapping) and (manifest.get("candidate_only") is True
            or "candidate_parameters_sha256" in manifest):
        raise ValueError("Mechanical candidate has no native experiment admission")


def flat_episode_source_files(config):
    settings = validate_flat_episode_selection(config)
    if settings is None:
        return []
    return [f"src/wheeled_tasks/chassis/{name}.py" for name in (
        "flat_episode_selection", "flat_episode_runtime", "flat_command_sampling",
        "nonsteady_initial_state", "leg_pose_mapping", "closed_chain_kinematics",
        "flat_birth_placement", "flat_interval_events")] + ["src/wheeled_algo/recovery_rollout.py", settings["branch_table"]]
