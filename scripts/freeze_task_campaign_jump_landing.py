#!/usr/bin/env python3
"""Freeze only the v10 jump landing repair and its named v12 migration."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.jump_landing_revision import (
    DESIGN_FILE, MANIFEST_FILE, PLAN_FILE, SCOPE, SOURCE_FILE, SOURCE_PLAN_FILE,
    SOURCE_PLAN_SHA, SOURCE_SHA, WORKER_FILE, apply_landing_revision, read_bound, revised_design,
)
from wheeled_tasks.chassis.task_campaign import digest, validate_plan_migration


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return digest(ROOT / name)


def main():
    source = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA)
    worker = read_bound(SOURCE_FILE, SOURCE_SHA)
    apply_landing_revision(worker)
    worker["design_preflight"] = dict(file=DESIGN_FILE,
        sha256=write(DESIGN_FILE, revised_design()), stage="jump")
    worker_sha = write(WORKER_FILE, worker)
    target = deepcopy(source)
    target["contract_id"] = "v6-task-campaign-v12"
    row = next(row for row in target["domains"] if row["name"] == "jump")
    row.update(contract=WORKER_FILE, sha256=worker_sha)
    target["revision_migration"].update(manifest_file=MANIFEST_FILE,
        source_plan_sha256=SOURCE_PLAN_SHA)
    target_sha = write(PLAN_FILE, target)
    write(MANIFEST_FILE, dict(schema="task-plan-revision-v12", source_plan_file=SOURCE_PLAN_FILE,
        source_plan_sha256=SOURCE_PLAN_SHA, target_plan_file=PLAN_FILE, target_plan_sha256=target_sha,
        run_directory=target["revision_migration"]["run_directory"], landing_scope=SCOPE,
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)]))
    validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    print(json.dumps(dict(plan_sha256=target_sha, jump_worker_sha256=worker_sha,
        source_plan=SOURCE_PLAN_FILE, unchanged_worker_count=5,
        scope="frozen_candidate_not_deployment_or_qualification")))


if __name__ == "__main__":
    main()
