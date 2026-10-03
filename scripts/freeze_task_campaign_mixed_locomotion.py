#!/usr/bin/env python3
"""Freeze the constant mixed locomotion allocation without changing other tasks."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.locomotion_revision import (
    DESIGN_FILE, SCHEMA, SCOPE, SPEED_FILE, SPEED_SHA, WORKER_FILE,
    apply_mixed_locomotion_revision, plan_sampling_pools, read_bound, revised_design, validate_integer_allocation,
)
from wheeled_tasks.chassis.task_campaign import digest, validate_plan_migration


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return digest(ROOT / name)


def main():
    source_file = "contracts/v6_task_campaign_v9.json"
    target_file = "contracts/v6_task_campaign_v10.json"
    manifest_file = "contracts/v6_task_campaign_migration_v10.json"
    source = json.loads((ROOT / source_file).read_text())
    worker = read_bound(SPEED_FILE, SPEED_SHA)
    apply_mixed_locomotion_revision(worker)
    allocation = validate_integer_allocation(worker)
    worker["task_campaign"]["revision_migration"] = dict(schema=SCHEMA,
        source_contract_file=SPEED_FILE, source_contract_sha256=SPEED_SHA)
    worker["design_preflight"] = dict(file=DESIGN_FILE,
        sha256=write(DESIGN_FILE, revised_design()), stage="speed")
    worker_sha = write(WORKER_FILE, worker)
    target = deepcopy(source)
    target["contract_id"] = "v6-task-campaign-v10"
    row = next(row for row in target["domains"] if row["name"] == "speed")
    row.update(contract=WORKER_FILE, sha256=worker_sha, sampling_pools=plan_sampling_pools())
    target["revision_migration"].update(manifest_file=manifest_file,
        source_plan_sha256=digest(ROOT / source_file))
    target_sha = write(target_file, target)
    write(manifest_file, dict(schema="task-plan-revision-v10", source_plan_file=source_file,
        source_plan_sha256=digest(ROOT / source_file), target_plan_file=target_file,
        target_plan_sha256=target_sha, run_directory=target["revision_migration"]["run_directory"],
        sampling_scope=SCOPE, workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)]))
    validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    print(json.dumps(dict(plan_sha256=target_sha,
        pool_environment_counts_4096=allocation["pool_environment_counts"])))


if __name__ == "__main__":
    main()
