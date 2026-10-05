#!/usr/bin/env python3
"""Freeze the v16 terrain endpoint extension without writing any worker."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.task_campaign import digest, validate_plan_migration
from wheeled_tasks.chassis.terrain_schedule_revision import (
    ENDPOINTS, MANIFEST_FILE, ORIGINS, PLAN_FILE, SCOPE, SOURCE_PLAN_FILE,
    SOURCE_PLAN_SHA, read_bound, revised_plan,
)


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return digest(ROOT / name)


def main():
    source = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT)
    target = revised_plan(ROOT)
    target_sha = write(PLAN_FILE, target)
    write(MANIFEST_FILE, dict(schema="task-plan-revision-v16", source_plan_file=SOURCE_PLAN_FILE,
        source_plan_sha256=SOURCE_PLAN_SHA, target_plan_file=PLAN_FILE, target_plan_sha256=target_sha,
        run_directory=target["revision_migration"]["run_directory"], schedule_scope=SCOPE,
        reopen_actor_origins=ORIGINS, reopen_actor_endpoints={name: ENDPOINTS[name] for name in ORIGINS},
        workers=[dict(source_file=row["contract"], source_sha256=row["sha256"],
            target_file=row["contract"], target_sha256=row["sha256"]) for row in source["domains"]]))
    validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    print(json.dumps(dict(plan_sha256=target_sha, unchanged_worker_count=6,
        terrain_actor_updates_added=12814, scope="frozen_candidate_not_deployment_or_qualification")))


if __name__ == "__main__":
    main()
