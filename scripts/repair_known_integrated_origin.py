#!/usr/bin/env python3
"""Correct source028's one misdirected REC origin, without refunding or learning."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path

from run_task_campaign import write_json
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis.task_campaign import CAMPAIGN_ID, account_updates


DIRECTORY = "integrated_early_origin_v26_dispatch_correction029"
EXPECTED = dict(previous_checkpoint_sha256="ed24fe4e5e62fd8c8e01f81427c0741ae46799a237dee2b0fbd4b288df4e1085",
    previous_consumed_updates=41374, previous_REC_actor=14169,
    checkpoint_sha256="e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967",
    consumed_updates=41393, actor_updates_consumed=41223, REC_actor=14188, learning_lineage_updates=27646,
    checkpoint_relative_path="recovery/attempt_010/train/model_final.pt")


def read(path):
    return json.loads(Path(path).read_text())


def write_snapshot(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def no_owners(run_root, proc_root):
    for file in Path(proc_root).glob("[0-9]*/cmdline"):
        try:
            argv = file.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (any(arg.endswith((b"/train_chassis.py", b"/evaluate_chassis.py", b"/run_task_campaign.py", b"/chassis_remote_job.py")) for arg in argv)
                and any(str(run_root).encode() in arg for arg in argv)):
            raise ValueError("Campaign owner is still alive; correction cannot change a live origin")


def correct_origin(plan, source_root, run_root, *, apply=False, proc_root=Path("/proc")):
    source_root, run_root = Path(source_root).resolve(), Path(run_root).resolve()
    if production.revision_number(plan) != 26:
        raise ValueError("This one correction is only for source028's v26 dispatch error")
    binding = production.binding_for(plan, source_root, run_root)
    claim = Path(plan["budget_source_stage"]) / (CAMPAIGN_ID + ".account.json")
    directory = run_root / DIRECTORY
    with claim.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_owners(run_root, proc_root)
        if directory.exists():
            raise ValueError("Named correction already has artifacts; no automatic repeat")
        files = dict(account=claim, consumption=run_root / "consumption.json",
            state=run_root / "campaign_state.json", completion=run_root / "completion.json",
            migration=run_root / "revisions/v21_to_v26_account_migration.json")
        before = {key: path.read_bytes() for key, path in files.items()}
        values = {key: json.loads(value) for key, value in before.items()}
        account, state, migration = values["account"], values["state"], values["migration"]
        totals = account_updates(plan, account["attempts"])
        records = production.reconcile_records(plan, run_root, account["attempts"])
        if (account.get("campaign_id") != CAMPAIGN_ID or account.get("run_directory") != str(run_root)
                or account.get("contract_sha256") != lineage.digest(source_root / production.artifact_files(26)["plan"])
                or account.get("rollback_refunds_consumed_updates") is not False
                or records != account["attempts"] or totals != account_updates(plan, records)
                or any(values[key].get(k) != v for key in ("account", "consumption", "state", "completion") for k, v in totals.items())
                or totals["consumed_updates"] != EXPECTED["consumed_updates"]
                or totals["actor_updates_consumed"] != EXPECTED["actor_updates_consumed"]
                or state != values["completion"] or state.get("status") != "stopped"):
            raise ValueError("Stopped campaign/account must retain all latest charged work")
        if account.get("joint_learning_sources") or state.get("joint_learning_sources"):
            raise ValueError("A joint source has already been selected; its origin cannot be replaced")
        for domain, _ in lineage.CHAIN:
            if state["domains"][domain].get("status") != "joint_domain_pending":
                raise ValueError("Joint domain has already started or completed")
            for path in (run_root / domain).glob("attempt_*/train/**/*.checkpoint.json"):
                if "integrated_lineage" in read(path):
                    raise ValueError("A joint own seal exists; origin correction is prohibited")
        previous = migration["entry_admission"]
        identity = state.get("integrated_continuation", {})
        if (previous["checkpoint"]["sha256"] != EXPECTED["previous_checkpoint_sha256"]
                or previous["record"] != state.get("authorized_REC_early_stop")
                or identity != account.get("integrated_continuation")
                or identity.get("origin_checkpoint_sha256") != EXPECTED["previous_checkpoint_sha256"]):
            raise ValueError("The original authenticated source028 origin differs")
        original = lineage._checkpoint(binding, previous["checkpoint"], "recovery", binding.read()["recovery_entry"]["contract_sha256"])
        if (original["seal"]["learning_updates"] != EXPECTED["previous_consumed_updates"]
                or original["seal"]["task_actor_update"] != EXPECTED["previous_REC_actor"]):
            raise ValueError("The original REC full-state clocks differ")
        path = run_root / EXPECTED["checkpoint_relative_path"]
        reference = dict(path=str(path), sha256=EXPECTED["checkpoint_sha256"])
        checkpoint = lineage._checkpoint(binding, reference, "recovery", binding.read()["recovery_entry"]["contract_sha256"])
        seal, completed = checkpoint["seal"], read(path.parent / "completion.json")
        if ("integrated_lineage" in checkpoint["infos"] or "integrated_lineage" in seal
                or seal["learning_updates"] != EXPECTED["consumed_updates"]
                or seal["actor_updates_consumed"] != EXPECTED["actor_updates_consumed"]
                or seal["task_actor_update"] != EXPECTED["REC_actor"]
                or seal["learning_lineage_updates"] != EXPECTED["learning_lineage_updates"]
                or completed.get("status") != "stopped" or completed.get("successful_updates") != 19
                or completed.get("parent_updates") != EXPECTED["previous_consumed_updates"]
                or completed.get("checkpoint_sha256") != reference["sha256"]
                or completed.get("task_actor_update") != EXPECTED["REC_actor"]):
            raise ValueError("Latest stopped REC is not the exact nineteen-update full-state seal")
        for other in (run_root / "recovery").glob("attempt_*/train/model_final.pt"):
            if lineage.checkpoint_update_count(other) > seal["learning_updates"]:
                raise ValueError("Correction would replace the latest own REC with an older parent")
        intent = dict(schema="known-integrated-dispatch-origin-correction029-v1",
            action="retain_latest_misdirected_REC_then_joint_chain", lineage_id=binding.read()["lineage_id"],
            lineage_manifest_sha256=binding.sha256, checkpoint=reference, previous_entry=previous,
            expected=EXPECTED, preserved_account=totals,
            before={key: dict(original_file=str(files[key]), file=str(directory / ("before_" + key + ".json")),
                sha256=lineage.digest(files[key])) for key in files},
            PPO_updates_by_repair=0, qualification=None)
        result = dict(schema="known-integrated-dispatch-origin-correction029-result-v1", status="validated", checkpoint=reference, consumed_updates=totals["consumed_updates"],
            actor_updates_consumed=totals["actor_updates_consumed"], REC_actor=seal["task_actor_update"],
            retained_new_actor_updates=19, PPO_updates_by_repair=0, qualification=None)
        if not apply:
            return result
        directory.mkdir(exist_ok=False)
        for key, content in before.items():
            write_snapshot(directory / ("before_" + key + ".json"), content)
        for name in ("campaign_state.json", "completion.json"):
            write_snapshot(directory / name, before["state"])
        write_json(directory / "intent.json", intent)
        record = deepcopy(previous["record"])
        record.update(candidate_checkpoint=reference["path"], candidate_checkpoint_sha256=reference["sha256"],
            early_stop_origin_correction=dict(file=str(directory / "intent.json"), sha256=lineage.digest(directory / "intent.json")))
        for key, name in (("early_stop_campaign_state", "campaign_state.json"), ("early_stop_campaign_completion", "completion.json")):
            record[key] = dict(file=str(directory / name), sha256=lineage.digest(directory / name))
        entry = lineage.admit_entry(binding, reference, record)
        after_account, after_state, after_migration = deepcopy(account), deepcopy(state), deepcopy(migration)
        after_state["authorized_REC_early_stop"] = record
        after_state["authorized_REC_origin_correction_history"] = [previous]
        for value in (after_account, after_state):
            value["integrated_continuation"]["origin_checkpoint_sha256"] = reference["sha256"]
        after_migration["entry_admission"] = entry
        after_migration["origin_correction"] = record["early_stop_origin_correction"]
        after_state["revision_migration"] = after_migration
        write_json(directory / "after_account.json", after_account)
        write_json(directory / "after_state.json", after_state)
        admission = production.prepare_admission(plan, source_root, run_root,
            account_path=directory / "after_account.json", state_path=directory / "after_state.json")
        if admission["selection"]["checkpoint"] != reference or admission["selection"]["fresh_joint_actor_updates"] != 0:
            raise ValueError("Correction did not select the latest REC as the fresh joint origin")
        write_json(directory / "admission.json", admission)
        after_migration["initial_admission"] = admission
        after_state["revision_migration"] = after_migration
        # All admission/full-state checks precede changing any campaign file.
        for key, value in (("account", after_account), ("consumption", after_account), ("state", after_state),
                           ("completion", after_state), ("migration", after_migration)):
            write_json(files[key], value)
        result.update(status="corrected", intent_sha256=lineage.digest(directory / "intent.json"),
            entry_admission=entry, admission_sha256=lineage.digest(directory / "admission.json"),
            preserved_original_files=intent["before"],
            after={key: dict(file=str(path), sha256=lineage.digest(path)) for key, path in files.items()})
        write_json(directory / "receipt.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = correct_origin(read(args.contract), args.contract.resolve().parents[1], args.run_dir, apply=args.apply)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
