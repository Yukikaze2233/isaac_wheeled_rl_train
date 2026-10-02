#!/usr/bin/env python3
"""Freeze actor-clock task endpoints while preserving every v6 worker byte."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.task_campaign import digest


def write(name, value):
    path = ROOT / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return digest(path)


def main():
    source_file = "contracts/v6_task_campaign_v6.json"
    target_file = "contracts/v6_task_campaign_v7.json"
    plan = deepcopy(json.loads((ROOT / source_file).read_text()))
    plan["contract_id"] = "v6-task-campaign-v7"
    plan["domain_completion"].update(
        early_finish="iteration_schedule_or_explicit_evidence_review",
        schedule=dict(clock="domain_actor_updates", end_actor_updates=dict(
            flat=13000, recovery=15000, speed=15000, terrain=15000, jump=15000, stepjump=15000),
            advance_requires_evaluation=False, result="unassessed_research_candidate"))
    manifest_file = "contracts/v6_task_campaign_migration_v7.json"
    plan["revision_migration"].update(manifest_file=manifest_file,
        source_plan_sha256=digest(ROOT / source_file))
    target_sha = write(target_file, plan)
    write(manifest_file, dict(schema="task-plan-revision-v7", source_plan_file=source_file,
        source_plan_sha256=digest(ROOT / source_file), target_plan_file=target_file,
        target_plan_sha256=target_sha, run_directory=plan["revision_migration"]["run_directory"],
        workers=[dict(source_file=r["contract"], source_sha256=r["sha256"],
                      target_file=r["contract"], target_sha256=r["sha256"]) for r in plan["domains"]]))
    print(target_sha)


if __name__ == "__main__":
    main()
