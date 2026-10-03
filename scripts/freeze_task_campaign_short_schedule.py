#!/usr/bin/env python3
"""Freeze user-requested first-pass actor budgets without changing any worker."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.task_campaign import (
    SHORT_SCHEDULE_ENDPOINTS, SHORT_SCHEDULE_SCOPE, digest, validate_plan_migration,
)


def write(name, value):
    path = ROOT / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return digest(path)


def main():
    source_file = "contracts/v6_task_campaign_v8.json"
    target_file = "contracts/v6_task_campaign_v9.json"
    manifest_file = "contracts/v6_task_campaign_migration_v9.json"
    source = json.loads((ROOT / source_file).read_text())
    plan = deepcopy(source)
    plan["contract_id"] = "v6-task-campaign-v9"
    plan["domain_completion"]["schedule"]["end_actor_updates"] = deepcopy(SHORT_SCHEDULE_ENDPOINTS)
    plan["revision_migration"].update(
        manifest_file=manifest_file, source_plan_sha256=digest(ROOT / source_file))
    target_sha = write(target_file, plan)
    write(manifest_file, dict(
        schema="task-plan-revision-v9", source_plan_file=source_file,
        source_plan_sha256=digest(ROOT / source_file), target_plan_file=target_file,
        target_plan_sha256=target_sha, run_directory=plan["revision_migration"]["run_directory"],
        schedule_scope=SHORT_SCHEDULE_SCOPE,
        workers=[dict(source_file=row["contract"], source_sha256=row["sha256"],
                      target_file=row["contract"], target_sha256=row["sha256"])
                 for row in source["domains"]]))
    validate_plan_migration(source, plan, plan["revision_migration"]["run_directory"])
    print(target_sha)


if __name__ == "__main__":
    main()
