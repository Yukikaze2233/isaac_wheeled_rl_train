#!/usr/bin/env python3
"""Read-only v19 role receipt using the real campaign and preflight entry points."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    manifest = json.loads(args.candidate_manifest.read_text())
    selected = sorted(str(path.relative_to(root))
        for folder, suffix in (("src", ".py"), ("scripts", ".py"), ("contracts", ".json"), ("tests", ".py"))
        for path in (root / folder).rglob("*" + suffix) if path.is_file())
    before = {name: digest(root / name) for name in selected}
    for row in manifest["files"]:
        if before.get(row["file"]) != row["candidate_sha256"]:
            raise ValueError("Candidate identity mismatch: " + row["file"])
    for name, sha in manifest["existing_contract_files_preserved_sha256"].items():
        if before.get(name) != sha:
            raise ValueError("Predecessor contract bytes changed: " + name)
    for key, value in dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1").items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from wheeled_tasks.chassis import recovery_quality_revision as recipe
    from wheeled_tasks.chassis import task_campaign as campaign
    from wheeled_tasks.chassis import evaluation
    from wheeled_tasks.chassis.preflight import ROLES
    import run_task_campaign as runner
    import train_chassis as train_entry
    modules = (recipe, campaign, evaluation, runner, train_entry)
    if any(not Path(module.__file__).resolve().is_relative_to(root) for module in modules):
        raise ValueError("Imported source outside selected root")
    started = time.monotonic()
    source = json.loads((root / recipe.SOURCE_PLAN_FILE).read_text())
    target = json.loads((root / recipe.PLAN_FILE).read_text())
    migration = campaign.validate_plan_migration(source, target,
        target["revision_migration"]["run_directory"])
    workers = recipe.validate_frozen_workers(root)
    receipts = runner.validate_contracts(target)
    old_receipts = runner.validate_contracts(source)
    expected_roles = {(row["name"], role) for row in target["domains"] for role in ROLES}
    if len(receipts) != 30 or {(r["stage"], r["role"]) for r in receipts} != expected_roles:
        raise ValueError("Missing independent role")
    if any(r["status"] != "static_passed" for r in receipts):
        raise ValueError("Real static preflight did not pass")
    old_by_role = {(r["stage"], r["role"]): r for r in old_receipts}
    role_checks = []
    for row in receipts:
        key = (row["stage"], row["role"])
        runtime = row["runtime_expected"]
        if key[0] != "recovery":
            if row != old_by_role[key]:
                raise ValueError("Unchanged domain role receipt differs: " + str(key))
            role_checks.append(dict(domain=key[0], role=key[1], predecessor_receipt_exact=True))
            continue
        enabled = key[1] == "train"
        if runtime["recovery_hold_cost"] != dict(recipe.HOLD_SETTINGS, enabled=enabled):
            raise ValueError("Recovery role cost settings differ")
        if runtime["modules"]["recovery_hold_cost"] is not enabled:
            raise ValueError("Recovery role cost module differs")
        if enabled:
            if runtime["recovery_hold_cost_gate_binding"] != recipe.HOLD_GATE_BINDING:
                raise ValueError("Native gate mapping differs")
            if runtime["recovery_prefix_curriculum"]["settings"]["goal_bridge"] != recipe.GOAL_BRIDGE:
                raise ValueError("Native original goal bridge missing")
        elif "recovery_hold_cost_gate_binding" in runtime:
            raise ValueError("Disabled fixed role acquired native gate binding")
        role_checks.append(dict(domain=key[0], role=key[1], enabled=enabled,
            complete_cost_object=True, authenticated_native_gates=enabled,
            unchanged_final_success_seconds=1.0))
    old_worker = json.loads((root / recipe.SOURCE_WORKER_FILE).read_text())
    worker = json.loads((root / recipe.WORKER_FILE).read_text())
    original = deepcopy(worker)
    old_fixed, fixed = map(evaluation.fixed_suite_contract, (old_worker, worker))
    fields = ("evaluation", "scene_groups", "skill_specs", "behavior_pool_membership",
        "behavior_pool_fractions", "new_asset_curriculum", "dense_tracking", "recovery_training",
        "recovery_takeover", "ordinary_posture_curriculum")
    if any(old_fixed.get(key) != fixed.get(key) for key in fields):
        raise ValueError("Fixed evaluation predecessor criteria or sampling changed")
    if worker != original or fixed["recovery_hold_cost"] != dict(recipe.HOLD_SETTINGS, enabled=False):
        raise ValueError("Fixed evaluation mutated training or enabled new cost")
    pins = train_entry.scut_curriculum_source_files(worker, root)
    if any(name not in before for name in pins):
        raise ValueError("Selected-contract source snapshot escaped verification")
    after = {name: digest(root / name) for name in selected}
    if after != before:
        raise ValueError("Frozen candidate bytes changed during validation")
    report = dict(schema="recovery-quality-v19-thirty-role-receipt-v1", status="passed",
        source_root=str(root), candidate_manifest_sha256=digest(args.candidate_manifest),
        combined_patch_sha256=manifest["combined_patch_sha256"], planned_source_revision="021",
        actual_source_commit=None, baseline_source_commit=manifest["baseline_source_commit"],
        contract_revision="v19", role_count=30, migration=migration, frozen_workers=workers,
        receipts=receipts, role_checks=role_checks, other_five_domain_role_receipts_exact=True,
        fixed_evaluation_predecessor_fields_exact=list(fields), fixed_cost_object_disabled=True,
        source_files_before=before, source_files_after=after, source_files_unchanged=True,
        runtime_source_pins=pins, duration_seconds=time.monotonic()-started,
        imported_modules={m.__name__: str(Path(m.__file__).resolve()) for m in modules},
        validator_sha256=digest(__file__), command=[sys.executable, *sys.argv],
        environment={name: os.environ[name] for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")},
        formal_updates_consumed=0, GPU_scene_started=False, qualification=None,
        limitations=["Static source and role validation only; new recovery requires actual native 600-step admission.",
            "Candidate identity is bound to the manifest, not yet to a source021 commit/archive."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(dict(status="passed", roles=30, output=str(args.output),
        other_five_domain_role_receipts_exact=True, source_files_unchanged=True,
        seconds=report["duration_seconds"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
