#!/usr/bin/env python3
"""Freeze a graded command curriculum at the latest sealed task actor clock."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.graded_locomotion_revision import (
    DESIGN_FILE, SCOPE, SOURCE_FILE, SOURCE_SHA, WORKER_FILE,
    apply_graded_revision, revised_design,
)
from wheeled_tasks.chassis.locomotion_revision import read_bound, validate_integer_allocation
from wheeled_tasks.chassis.task_campaign import digest, validate_plan_migration


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return digest(ROOT / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-actor-update", type=int, required=True)
    args = parser.parse_args()
    source_file, target_file = "contracts/v6_task_campaign_v10.json", "contracts/v6_task_campaign_v11.json"
    manifest_file = "contracts/v6_task_campaign_migration_v11.json"
    source = json.loads((ROOT / source_file).read_text())
    endpoint = source["domain_completion"]["schedule"]["end_actor_updates"]["speed"]
    if not 0 <= args.origin_actor_update <= endpoint - 460:
        raise ValueError("Need 360 actor updates to expand plus at least 100 at the full target before the frozen endpoint")
    worker = read_bound(SOURCE_FILE, SOURCE_SHA)
    apply_graded_revision(worker, args.origin_actor_update)
    allocation = validate_integer_allocation(worker)
    worker["design_preflight"] = dict(file=DESIGN_FILE,
        sha256=write(DESIGN_FILE, revised_design(args.origin_actor_update)), stage="speed")
    worker_sha = write(WORKER_FILE, worker)
    target = deepcopy(source)
    target["contract_id"] = "v6-task-campaign-v11"
    row = next(row for row in target["domains"] if row["name"] == "speed")
    row.update(contract=WORKER_FILE, sha256=worker_sha)
    target["revision_migration"].update(manifest_file=manifest_file,
        source_plan_sha256=digest(ROOT / source_file))
    target_sha = write(target_file, target)
    write(manifest_file, dict(schema="task-plan-revision-v11", source_plan_file=source_file,
        source_plan_sha256=digest(ROOT / source_file), target_plan_file=target_file,
        target_plan_sha256=target_sha, run_directory=target["revision_migration"]["run_directory"],
        difficulty_scope=SCOPE, origin_actor_update=args.origin_actor_update,
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)]))
    validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    print(json.dumps(dict(plan_sha256=target_sha, origin_actor_update=args.origin_actor_update,
        full_target_actor_update=args.origin_actor_update + 360,
        pool_environment_counts_4096=allocation["pool_environment_counts"])))


if __name__ == "__main__":
    main()
