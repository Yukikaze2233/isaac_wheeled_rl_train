#!/usr/bin/env python3
"""Freeze one authenticated recovery-worker patch without changing other domains."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.recovery_revision import apply_recovery_takeover_revision
from wheeled_tasks.chassis.task_campaign import digest, validate_plan_migration


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return digest(ROOT / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-actor-update", type=int, required=True)
    args = parser.parse_args()
    before = read("contracts/v6_task_campaign_v7.json")
    plan = deepcopy(before)
    pins = []
    for old, row in zip(before["domains"], plan["domains"], strict=True):
        if row["name"] == "recovery":
            worker = read(old["contract"])
            apply_recovery_takeover_revision(worker, args.origin_actor_update)
            binding = dict(schema="task-worker-revision-v8", source_contract_file=old["contract"],
                source_contract_sha256=old["sha256"], origin_actor_update=args.origin_actor_update)
            worker["task_campaign"]["revision_migration"] = binding
            design = read(worker["design_preflight"]["file"])
            design["reviewed_change"] = "reset_pose_curriculum_dense_actor_takeover_full_flat_rehearsal_v8"
            for role, expectation in design["stages"]["recovery"]["roles"].items():
                apply_recovery_takeover_revision(expectation["contract"], args.origin_actor_update)
                expectation["contract"]["task_campaign"] = deepcopy(worker["task_campaign"])
                runtime = expectation["runtime"]
                runtime.update(recovery_takeover=deepcopy(worker["recovery_takeover"]),
                    recovery_pose_curriculum=deepcopy(worker["recovery_training"]["pose_curriculum"]) if role == "train" else None,
                    recovery_complete_started_blend=True)
            design_file = "contracts/v6_task_campaign_recovery_expectations_v8.json"
            worker["design_preflight"].update(file=design_file, sha256=write(design_file, design))
            row["contract"] = "contracts/v6_task_campaign_recovery_worker_v8.json"
            row["sha256"] = write(row["contract"], worker)
        pins.append(dict(source_file=old["contract"], source_sha256=old["sha256"],
                         target_file=row["contract"], target_sha256=row["sha256"]))
    plan["contract_id"] = "v6-task-campaign-v8"
    plan["revision_migration"].update(manifest_file="contracts/v6_task_campaign_migration_v8.json",
        source_plan_sha256=digest(ROOT / "contracts/v6_task_campaign_v7.json"))
    target_sha = write("contracts/v6_task_campaign_v8.json", plan)
    write("contracts/v6_task_campaign_migration_v8.json", dict(schema="task-plan-revision-v8",
        source_plan_file="contracts/v6_task_campaign_v7.json",
        source_plan_sha256=plan["revision_migration"]["source_plan_sha256"],
        target_plan_file="contracts/v6_task_campaign_v8.json", target_plan_sha256=target_sha,
        run_directory=plan["revision_migration"]["run_directory"], workers=pins))
    validate_plan_migration(before, plan, plan["revision_migration"]["run_directory"])
    print(target_sha)


if __name__ == "__main__":
    main()
