#!/usr/bin/env python3
"""Freeze one formal S1 continuation from A; discarded B work remains charged."""
from copy import deepcopy
import argparse

from freeze_s1_remediation import ROOT, read, sha, write
from wheeled_tasks.chassis.continuation import (
    REVISION, REPAIR_REVISION, PERSISTENT_REVISION, SOURCE_SHA, CHECKPOINT_SHA, origin_file, recipe_name,
    settings, worker_contract,
)


def build(revision=REVISION):
    persistent = revision == PERSISTENT_REVISION
    repair = revision in (REPAIR_REVISION, PERSISTENT_REVISION)
    frozen = settings(revision)
    origin_path, recipe = origin_file(revision), recipe_name(revision)
    prefix = ("v6_s1_persistent_recovery" if persistent else
              "v6_s1_landing_repair" if repair else "v6_s1_continuation")
    source_file = "contracts/v6_s1_A_fixed_v1.json"
    source = read(source_file)
    if sha(source_file) != SOURCE_SHA:
        raise ValueError("A contract changed")
    origin = read(origin_path)
    old = read(source["design_preflight"]["file"])
    if sha(source["design_preflight"]["file"]) != source["design_preflight"]["sha256"]:
        raise ValueError("Independent A design table changed")
    design = {k: deepcopy(v) for k, v in old.items() if k != "stages"}
    design["inherited_design_sha256"] = source["design_preflight"]["sha256"]
    design["reviewed_change"] = ("docs/V6_S1_PERSISTENT_RECOVERY_20261001.md" if persistent else
                                 "docs/V6_S1_LANDING_REPAIR_20260930.md" if repair
                                 else "docs/V6_S1_CONTINUATION_20260930.md")
    row = deepcopy(old["stages"]["A_fixed"])
    for role, values in row["roles"].items():
        # Explicit amendments to the independent A table, not re-materialized expectations.
        expected = values["contract"]
        expected.pop("remediation")
        expected.update(resume_critic_warmup_updates=0,
                        checkpoint_first_update=frozen["source_consumed_updates"] + 1,
                        continuation=frozen)
        if repair:
            expected["landing_tracking"] = dict(height_weight=2., velocity_weight=1., impact_weight=.05,
                                                 passive_semantics="finite_task_cost_v1", window_seconds=.6)
            values["runtime"]["finite_passive_landing"] = dict(enabled=True, window_seconds=.6, policy_dt=.02)
        if persistent:
            expected["new_asset_curriculum"]["start_stop_start_delay_seconds"] = [0., 8.]
            values["runtime"]["start_stop_timing"] = dict(
                enabled=role == "train", start_delay_seconds=[0., 8.] if role == "train" else None,
                policy_dt=.02)
    design["stages"] = {recipe: row}
    filename = f"contracts/{prefix}_expectations_v1.json"
    binding = dict(file=filename, sha256=write(filename, design), stage=recipe)
    config = worker_contract(source, binding, revision=revision)
    worker = f"contracts/{prefix}_worker_v1.json"
    previous = read("contracts/v6_s1_remediation_v1.json")
    plan = {**{k: deepcopy(previous[k]) for k in (
        "target_num_envs", "num_steps_per_env", "seed", "evaluation_case_names",
        "evaluation_seeds", "inherited_recovery_attempts", "stage_ceiling", "total_ceiling",
        "rollback_refunds_consumed_updates", "automatic_promotion", "automatic_formal_resume")},
        "contract_id": revision, "scope": "formal_S1_continuation_charged_to_original_budget",
        "orchestration": "formal_S1_continuation", "origin_file": origin_path, "origin_sha256": sha(origin_path),
        "source_stage": origin["source_stage"], "source_checkpoint": origin["source_checkpoint"],
        "source_checkpoint_sha256": CHECKPOINT_SHA, "source_contract_sha256": SOURCE_SHA,
        "source_files_sha256": origin["source_files_sha256"],
        **{key: frozen[key] for key in ("source_consumed_updates", "source_actor_consumed_updates",
                                        "source_stage_actor_update", "source_learning_lineage_updates")},
        "study_update_ceiling": frozen["updates"],
        "protected_case_names": sorted(set(origin["old_protected"]) | set(origin["A_passed"])),
        "stages": [dict(name=recipe, updates=frozen["updates"], contract=worker, sha256=write(worker, config))]}
    if repair:
        plan.update(budget_source_stage=origin["budget_source"]["stage"],
                    budget_source_files_sha256=origin["budget_source"]["files_sha256"])
    write(f"contracts/{prefix}_v1.json", plan)
    print({"worker": plan["stages"], "protected": len(plan["protected_case_names"]),
           "consumption": [frozen["source_consumed_updates"], frozen["source_consumed_updates"] + frozen["updates"]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=(REVISION, REPAIR_REVISION, PERSISTENT_REVISION), default=REVISION)
    build(parser.parse_args().revision)
