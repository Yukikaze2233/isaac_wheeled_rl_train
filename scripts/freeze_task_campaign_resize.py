#!/usr/bin/env python3
"""Freeze 4096-env task workers and nonblocking explicit review, with unchanged caps."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.task_campaign import digest, resize_reviewed_worker, REVIEW_MODE


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return digest(ROOT / name)


def main():
    before = read("contracts/v6_task_campaign_v4.json")
    plan = deepcopy(before)
    budget = read(before["budget_spec_file"])
    budget.update(schema="task-domain-budget-v2", update_unit="one_PPO_optimizer_update",
        transition_accounting="sum_actual_num_envs_times_rollout_steps_per_attempt_no_refund",
        rollout_steps_per_env=24, previous_num_envs=20000, current_num_envs=4096)
    budget_file = "contracts/v6_task_campaign_budget_v2.json"
    plan.update(budget_spec_file=budget_file, budget_spec_sha256=write(budget_file, budget))
    parents = {"A": (plan["source_contract_file"], plan["source_contract_sha256"])}
    pins = []
    for old, row in zip(before["domains"], plan["domains"], strict=True):
        worker = read(old["contract"])
        resize_reviewed_worker(worker)
        meta = worker["task_campaign"]
        meta["source_contract_file"], meta["source_contract_sha256"] = parents[row["parent_domain"]]
        meta["revision_migration"] = dict(schema="task-worker-revision-v5",
            source_contract_file=old["contract"], source_contract_sha256=old["sha256"])
        design = read(worker["design_preflight"]["file"])
        design["reviewed_change"] = "4096x24_four_minibatches_nonblocking_explicit_review_v5"
        for name, role in design["stages"][row["name"]]["roles"].items():
            resize_reviewed_worker(role["contract"])
            role["contract"]["task_campaign"] = deepcopy(meta)
            if name == "train":
                role["invocation"]["num_envs"] = 4096
        design_file = worker["design_preflight"]["file"].replace("_v4.json", "_v5.json")
        worker["design_preflight"].update(file=design_file, sha256=write(design_file, design))
        row["contract"] = old["contract"].replace("_v4.json", "_v5.json")
        row["sha256"] = write(row["contract"], worker)
        parents[row["name"]] = (row["contract"], row["sha256"])
        pins.append(dict(source_file=old["contract"], source_sha256=old["sha256"],
                         target_file=row["contract"], target_sha256=row["sha256"]))
    plan.update(contract_id="v6-task-campaign-v5", target_num_envs=4096,
        runtime_estimate=dict(seconds_per_update=None,
            basis="4096x24_requires_native_wall_measurement; caps_not_required_run_length"))
    plan["domain_completion"].update(execution_mode=REVIEW_MODE, review_interval_actor_updates=500,
        early_finish="explicit_evidence_review", review_blocks_training=False)
    plan["revision_migration"].update(manifest_file="contracts/v6_task_campaign_migration_v5.json",
        source_plan_sha256=digest(ROOT / "contracts/v6_task_campaign_v4.json"))
    new_sha = write("contracts/v6_task_campaign_v5.json", plan)
    write("contracts/v6_task_campaign_migration_v5.json", dict(schema="task-plan-revision-v5",
        source_plan_file="contracts/v6_task_campaign_v4.json",
        source_plan_sha256=plan["revision_migration"]["source_plan_sha256"],
        target_plan_file="contracts/v6_task_campaign_v5.json", target_plan_sha256=new_sha,
        run_directory=plan["revision_migration"]["run_directory"], workers=pins))
    print(new_sha)


if __name__ == "__main__":
    main()
