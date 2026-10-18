"""Named v14 locomotion repair, reusing the v13 birth and braking rules."""
from copy import deepcopy

from .locomotion_difficulty import difficulty_settings
from .nonstandard_start_revision import (
    ENDPOINTS, ORIGINS, ROOT, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, SOURCES,
    apply_start_revision, plan_sampling_pools, read_bound,
    release_stop_runtime_expectation, runtime_expectation,
)

PLAN_FILE = "contracts/v6_task_campaign_v14.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v14.json"
SCHEMA = "task-worker-revision-v14"
SCOPE = "ordinary_locomotion_stationary_l1_posture_floor_graded_speed_and_full_speed_release_stop"
STATIONARY = dict(version="scut-flat-linear-velocity-v2", weight=-2.0,
    command_threshold_m_s=.1, exclude_world_reference_motion=True)
POSTURE = dict(version="bounded-posture-floor-v1", pitch_scale_floor=3.0, roll_scale_floor=4.0)


def worker_file(domain):
    if domain not in SOURCES:
        raise ValueError("Locomotion quality revision is limited to speed and terrain")
    return f"contracts/v6_task_campaign_{domain}_worker_v14.json"


def design_file(domain):
    worker_file(domain)
    return f"contracts/v6_task_campaign_{domain}_expectations_v14.json"


def revision_binding(domain):
    file, sha = SOURCES[domain]
    return dict(schema=SCHEMA, source_contract_file=file, source_contract_sha256=sha,
        origin_actor_update=ORIGINS[domain])


def apply_quality_revision(config):
    """Apply the exact repair to an independent frozen v10/v6 role input."""
    domain = config.get("task_campaign", {}).get("domain")
    if domain not in SOURCES or config["task_campaign"]["clock"] != "domain_actor_updates":
        raise ValueError("Locomotion quality requires its unchanged ordinary domain and actor clock")
    apply_start_revision(config)
    config["task_campaign"]["revision_migration"] = revision_binding(domain)
    config["dense_tracking"]["stand_still_lin_vel"] = deepcopy(STATIONARY)
    config["dense_tracking"]["locomotion_posture"] = deepcopy(POSTURE)
    if domain == "speed":
        config["locomotion_difficulty"] = difficulty_settings(ORIGINS[domain])
        for spec in config["skill_specs"].values():
            if spec.get("kind") != "release_stop":
                continue
            # Both forms of translation must teach release from the full domain;
            # the yaw-only spec retains its original magnitude distribution.
            if spec["command"][0] != 0.:
                spec["command"][0] = 5.0
                spec["release_stop_magnitude_ranges"]["vx_m_s"] = [.5, 5.0]


def quality_runtime_expectation(config):
    return {key: deepcopy(config["dense_tracking"][key])
        for key in ("stand_still_lin_vel", "locomotion_posture")}


def revised_design(domain, root=ROOT):
    """Transform five separately frozen role inputs, never the output worker."""
    source = read_bound(*SOURCES[domain], root)
    binding = source["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_locomotion_quality_design_sha256"] = binding["sha256"]
    for row in design["stages"][domain]["roles"].values():
        config, runtime = row["contract"], row["runtime"]
        apply_quality_revision(config)
        runtime["modules"]["nonstandard_start"] = config["nonstandard_start"]["enabled"]
        runtime["nonstandard_start"] = runtime_expectation(config)
        runtime["modules"]["release_stop"] = config["release_stop"]["enabled"]
        runtime["release_stop"] = release_stop_runtime_expectation(config)
        runtime["modules"].update(stand_still_lin_vel=True, locomotion_posture_floor=True)
        runtime["locomotion_quality_rewards"] = quality_runtime_expectation(config)
        if domain == "speed":
            runtime["locomotion_difficulty"] = dict(enabled=not config["evaluation_exact_cases"],
                settings=deepcopy(config["locomotion_difficulty"]))
    return design
