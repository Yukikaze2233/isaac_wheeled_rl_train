#!/usr/bin/env python3
"""Freeze paired S1 contracts against the previously reviewed independent table."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.remediation import ARMS, REVISION, SOURCE_SHA, CHECKPOINT_SHA, arm_changes, arm_contract


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write(path, value):
    (ROOT / path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return sha(path)


def build():
    directory = "reports/v6_s1_remediation_20260930/source/"
    source = read(directory + "contract.json")
    if sha(directory + "contract.json") != SOURCE_SHA:
        raise ValueError("Source S1 contract changed")
    old = read(source["design_preflight"]["file"])
    if sha(source["design_preflight"]["file"]) != source["design_preflight"]["sha256"]:
        raise ValueError("Source independent table changed")
    completion, retention = (read(directory + name) for name in ("completion.json", "retention_state.json"))
    if (completion["successful_updates"] != 618 or completion["recovery_exhausted"] != "retry_limit"
            or len(completion["recovery_attempts"]) != 1):
        raise ValueError("Remediation must inherit the exhausted 618 ledger")
    design = {k: deepcopy(v) for k, v in old.items() if k != "stages"}
    design["inherited_design_sha256"] = source["design_preflight"]["sha256"]
    design["inherited_reviewed_plan_payload_sha256"] = design.pop("reviewed_plan_payload_sha256")
    design["reviewed_change"] = "docs/V6_S1_REMEDIATION_AND_ORDER_20260930.md"
    design["stages"] = {}
    design_file = "contracts/v6_s1_remediation_expectations_v1.json"
    for arm in ARMS:
        row = deepcopy(old["stages"]["S1"])
        for role, values in row["roles"].items():
            # Independent reviewed values, not reconstructed from arm_contract().
            expected = values["contract"]
            expected.update(learning_rate=1e-4, learning_rate_schedule="fixed",
                resume_critic_warmup_updates=10, checkpoint_interval=10,
                checkpoint_first_update=619 if arm == "A_fixed" else 659,
                remediation=deepcopy(arm_changes(arm)["remediation"]))
            if arm == "B_stationary":
                expected["dense_tracking"]["stationary"].update(broad_weight=3., fine_weight=1.5)
        design["stages"][arm] = row
    design_sha = write(design_file, design)
    arms = []
    for arm in ARMS:
        filename = f"contracts/v6_s1_{arm}_v1.json"
        config = arm_contract(source, arm, {"file": design_file, "sha256": design_sha, "stage": arm})
        arms.append({"name": arm, "updates": 40, "contract": filename, "sha256": write(filename, config)})
    remote = "/home/kaiser/robot-rl-sim60/experiments/v6-new-asset-20260929T142940Z-201411/train/stage_00_S1"
    names = sorted(set(source["evaluation"]["promotion_case_names"]) | set(retention["protected_case_names"]))
    source_files = {name: sha(directory + name) for name in ("completion.json", "retention_state.json")}
    plan = {"contract_id": REVISION, "scope": "bounded_S1_remediation_charged_to_formal_budget",
        "target_num_envs": 20000, "num_steps_per_env": 24, "seed": 290927,
        "source_stage": remote, "source_checkpoint": remote + "/block_002/model_final.pt",
        "source_checkpoint_sha256": CHECKPOINT_SHA, "source_contract_sha256": SOURCE_SHA,
        "source_files_sha256": source_files, "source_consumed_updates": 618,
        "source_actor_consumed_updates": 618, "source_stage_actor_update": 338,
        "stage_ceiling": 1000, "total_ceiling": 8500, "study_update_ceiling": 80,
        "rollback_refunds_consumed_updates": False, "inherited_recovery_attempts": completion["recovery_attempts"],
        "protected_case_names": retention["protected_case_names"], "evaluation_case_names": names,
        "evaluation_seeds": [290929, 290930], "stages": arms,
        "automatic_promotion": False, "automatic_formal_resume": False}
    write("contracts/v6_s1_remediation_v1.json", plan)
    print(json.dumps({"arms": arms, "coverage": len(names), "design_sha256": design_sha}))


if __name__ == "__main__":
    build()
