#!/usr/bin/env python3
"""Freeze one formal S1 continuation from A; discarded B work remains charged."""
from copy import deepcopy

from freeze_s1_remediation import ROOT, read, sha, write
from wheeled_tasks.chassis.continuation import (
    REVISION, SOURCE_SHA, CHECKPOINT_SHA, ORIGIN, settings, worker_contract,
)


def build():
    source_file = "contracts/v6_s1_A_fixed_v1.json"
    source = read(source_file)
    if sha(source_file) != SOURCE_SHA:
        raise ValueError("A contract changed")
    origin = read(ORIGIN)
    old = read(source["design_preflight"]["file"])
    if sha(source["design_preflight"]["file"]) != source["design_preflight"]["sha256"]:
        raise ValueError("Independent A design table changed")
    design = {k: deepcopy(v) for k, v in old.items() if k != "stages"}
    design["inherited_design_sha256"] = source["design_preflight"]["sha256"]
    design["reviewed_change"] = "docs/V6_S1_CONTINUATION_20260930.md"
    row = deepcopy(old["stages"]["A_fixed"])
    for values in row["roles"].values():
        # Explicit amendments to the independent A table, not re-materialized expectations.
        expected = values["contract"]
        expected.pop("remediation")
        expected.update(resume_critic_warmup_updates=0, checkpoint_first_update=699,
                        continuation=settings())
    design["stages"] = {"A_continue": row}
    filename = "contracts/v6_s1_continuation_expectations_v1.json"
    binding = dict(file=filename, sha256=write(filename, design), stage="A_continue")
    config = worker_contract(source, binding)
    worker = "contracts/v6_s1_continuation_worker_v1.json"
    previous = read("contracts/v6_s1_remediation_v1.json")
    plan = {**{k: deepcopy(previous[k]) for k in (
        "target_num_envs", "num_steps_per_env", "seed", "evaluation_case_names",
        "evaluation_seeds", "inherited_recovery_attempts", "stage_ceiling", "total_ceiling",
        "rollback_refunds_consumed_updates", "automatic_promotion", "automatic_formal_resume")},
        "contract_id": REVISION, "scope": "formal_S1_continuation_charged_to_original_budget",
        "orchestration": "formal_S1_continuation", "origin_file": ORIGIN, "origin_sha256": sha(ORIGIN),
        "source_stage": origin["source_stage"], "source_checkpoint": origin["source_checkpoint"],
        "source_checkpoint_sha256": CHECKPOINT_SHA, "source_contract_sha256": SOURCE_SHA,
        "source_files_sha256": origin["source_files_sha256"],
        "source_consumed_updates": 698, "source_actor_consumed_updates": 678,
        "source_stage_actor_update": 368, "source_learning_lineage_updates": 378,
        "study_update_ceiling": 40,
        "protected_case_names": sorted(set(origin["old_protected"]) | set(origin["A_passed"])),
        "stages": [dict(name="A_continue", updates=40, contract=worker, sha256=write(worker, config))]}
    write("contracts/v6_s1_continuation_v1.json", plan)
    print({"worker": plan["stages"], "protected": len(plan["protected_case_names"]), "consumption": [698, 738]})


if __name__ == "__main__":
    build()
