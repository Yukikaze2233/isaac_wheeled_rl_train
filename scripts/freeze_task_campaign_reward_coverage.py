#!/usr/bin/env python3
"""Freeze the reviewed v4 coverage/events patch without changing the campaign caps."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.task_campaign import digest, repair_coverage_and_events


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return digest(ROOT / name)


def main():
    before = read("contracts/v6_task_campaign_v3.json")
    plan = deepcopy(before)
    parents = {"A": (plan["source_contract_file"], plan["source_contract_sha256"])}
    pins = []
    for old, row in zip(before["domains"], plan["domains"], strict=True):
        worker = read(old["contract"])
        repair_coverage_and_events(worker)
        meta = worker["task_campaign"]
        meta["source_contract_file"], meta["source_contract_sha256"] = parents[row["parent_domain"]]
        meta["revision_migration"] = dict(schema="task-worker-revision-v4",
            source_contract_file=old["contract"], source_contract_sha256=old["sha256"])
        design = read(worker["design_preflight"]["file"])
        design["reviewed_change"] = "continuous_low_joint_commands_explicit_event_rewards_v4"
        for role in design["stages"][row["name"]]["roles"].values():
            # Independently amend the frozen design, including exact-case roles.
            repair_coverage_and_events(role["contract"])
            role["contract"]["task_campaign"] = deepcopy(meta)
        design_file = worker["design_preflight"]["file"].replace("_v3.json", "_v4.json")
        worker["design_preflight"].update(file=design_file, sha256=write(design_file, design))
        row["contract"] = old["contract"].replace("_v3.json", "_v4.json")
        row["sha256"] = write(row["contract"], worker)
        parents[row["name"]] = (row["contract"], row["sha256"])
        pins.append(dict(source_file=old["contract"], source_sha256=old["sha256"],
                         target_file=row["contract"], target_sha256=row["sha256"]))
    plan["contract_id"] = "v6-task-campaign-v4"
    plan["revision_migration"].update(manifest_file="contracts/v6_task_campaign_migration_v4.json",
        source_plan_sha256=digest(ROOT / "contracts/v6_task_campaign_v3.json"))
    new_sha = write("contracts/v6_task_campaign_v4.json", plan)
    write("contracts/v6_task_campaign_migration_v4.json", dict(schema="task-plan-revision-v4",
        source_plan_file="contracts/v6_task_campaign_v3.json",
        source_plan_sha256=plan["revision_migration"]["source_plan_sha256"],
        target_plan_file="contracts/v6_task_campaign_v4.json", target_plan_sha256=new_sha,
        run_directory=plan["revision_migration"]["run_directory"], workers=pins))
    print(new_sha)


if __name__ == "__main__":
    main()
