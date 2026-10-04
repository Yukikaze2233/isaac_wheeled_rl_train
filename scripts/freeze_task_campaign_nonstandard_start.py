#!/usr/bin/env python3
"""Freeze two ordinary restart-teaching windows from the deployed v12 plan."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.nonstandard_start_revision import (
    ENDPOINTS, MANIFEST_FILE, ORIGINS, PLAN_FILE, SCOPE, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, SOURCES,
    apply_start_revision, design_file, plan_sampling_pools, read_bound, revised_design, worker_file,
)
from wheeled_tasks.chassis.task_campaign import digest, validate_plan_migration


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return digest(ROOT / name)


def main():
    source = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA)
    target = deepcopy(source)
    target["contract_id"] = "v6-task-campaign-v13"
    target["domain_completion"]["schedule"]["end_actor_updates"] = deepcopy(ENDPOINTS)
    for row in target["domains"]:
        domain = row["name"]
        if domain not in ORIGINS:
            continue
        worker = read_bound(*SOURCES[domain])
        apply_start_revision(worker)
        file = design_file(domain)
        worker["design_preflight"] = dict(file=file, sha256=write(file, revised_design(domain)), stage=domain)
        file = worker_file(domain)
        row.update(contract=file, sha256=write(file, worker), sampling_pools=plan_sampling_pools(domain))
    target["revision_migration"].update(manifest_file=MANIFEST_FILE, source_plan_sha256=SOURCE_PLAN_SHA)
    target_sha = write(PLAN_FILE, target)
    write(MANIFEST_FILE, dict(schema="task-plan-revision-v13", source_plan_file=SOURCE_PLAN_FILE,
        source_plan_sha256=SOURCE_PLAN_SHA, target_plan_file=PLAN_FILE, target_plan_sha256=target_sha,
        run_directory=target["revision_migration"]["run_directory"], start_scope=SCOPE,
        reopen_actor_origins=ORIGINS, reopen_actor_endpoints={name: ENDPOINTS[name] for name in ORIGINS},
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)]))
    validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    print(json.dumps(dict(plan_sha256=target_sha, teaching_actor_updates=dict(speed=1500, terrain=1000),
        unchanged_worker_count=4, scope="frozen_candidate_not_deployment_or_qualification")))


if __name__ == "__main__":
    main()
