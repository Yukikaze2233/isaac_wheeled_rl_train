"""One named v10→v11 patch: command difficulty only, with full mixed retention."""
from copy import deepcopy

from .locomotion_difficulty import difficulty_settings
from .locomotion_revision import ROOT, read_bound

SOURCE_FILE = "contracts/v6_task_campaign_speed_worker_v10.json"
SOURCE_SHA = "37837b286de10a66404ef155ae84ead613a90f76375d4abe6874a042eae6b406"
WORKER_FILE = "contracts/v6_task_campaign_speed_worker_v11.json"
DESIGN_FILE = "contracts/v6_task_campaign_speed_expectations_v11.json"
SCHEMA = "task-worker-revision-v11"
SCOPE = "actor_clock_graded_commands_preserve_v10_population_rewards_and_final_targets"


def revision_binding(origin_actor_update):
    return dict(schema=SCHEMA, source_contract_file=SOURCE_FILE,
        source_contract_sha256=SOURCE_SHA, origin_actor_update=origin_actor_update)


def apply_graded_revision(config, origin_actor_update):
    if (config.get("task_campaign", {}).get("domain") != "speed"
            or config["task_campaign"]["clock"] != "domain_actor_updates"):
        raise ValueError("Graded locomotion must retain the speed domain and actor clock")
    config["locomotion_difficulty"] = difficulty_settings(origin_actor_update)
    config["task_campaign"]["revision_migration"] = revision_binding(origin_actor_update)


def revised_design(origin_actor_update, root=ROOT):
    source = read_bound(SOURCE_FILE, SOURCE_SHA, root)
    binding = source["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_graded_locomotion_design_sha256"] = binding["sha256"]
    for row in design["stages"]["speed"]["roles"].values():
        apply_graded_revision(row["contract"], origin_actor_update)
        row["runtime"]["locomotion_difficulty"] = dict(
            enabled=not row["contract"]["evaluation_exact_cases"],
            settings=deepcopy(row["contract"]["locomotion_difficulty"]))
    return design
