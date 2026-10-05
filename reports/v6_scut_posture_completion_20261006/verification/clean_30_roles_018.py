#!/usr/bin/env python3
"""CPU-only v18 migration, 30 roles and actual-v15 fixed-exam isolation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("source commit must be the frozen 40-character SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256):
        parser.error("archive SHA-256 must be the frozen 64-character digest")
    if digest(args.archive) != args.archive_sha256:
        raise ValueError("Frozen source archive SHA differs")
    clean = args.source_root.resolve(strict=True)
    # Hash before imports or recipe validation, and match every selected byte.
    exact_files = sorted(str(path.relative_to(clean))
        for folder, suffix in (("src", ".py"), ("scripts", ".py"), ("contracts", ".json"))
        for path in (clean / folder).rglob("*" + suffix) if path.is_file())
    before = {name: digest(clean / name) for name in exact_files}
    matched = {}
    with tarfile.open(args.archive, "r:gz") as bundle:
        archive_commit = bundle.pax_headers.get("comment")
        if archive_commit != args.source_commit:
            raise ValueError("Archive commit header differs")
        for member in bundle:
            if member.name in before:
                if not member.isfile():
                    raise ValueError("Expected source file is not an archive file")
                matched[member.name] = hashlib.sha256(bundle.extractfile(member).read()).hexdigest()
    if matched != before:
        raise ValueError("Role validation source bytes differ from frozen archive")
    # Set before torch/native task imports; no Isaac/GPU launch.
    for name, value in dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                            MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1").items():
        os.environ[name] = value
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(clean / "src"), str(clean / "scripts")]
    from wheeled_tasks.chassis import task_campaign as campaign
    from wheeled_tasks.chassis import ordinary_posture_revision as revision
    from wheeled_tasks.chassis import ordinary_posture_curriculum as posture
    from wheeled_tasks.chassis import evaluation
    from wheeled_tasks.chassis.preflight import ROLES
    import run_task_campaign as runner
    import train_chassis as train_entry
    modules = (campaign, revision, posture, evaluation, runner, train_entry)
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(clean):
            raise ValueError("Preflight imported a module outside the frozen source")
    started = time.monotonic()
    source = json.loads((clean / revision.SOURCE_PLAN_FILE).read_text())
    target = json.loads((clean / revision.PLAN_FILE).read_text())
    migration = campaign.validate_plan_migration(
        source, target, target["revision_migration"]["run_directory"])
    roles = runner.validate_contracts(target)
    expected_roles = {(row["name"], role) for row in target["domains"] for role in ROLES}
    if len(roles) != 30 or {(row["stage"], row["role"]) for row in roles} != expected_roles:
        raise ValueError("All six tasks must retain five independent preflight roles")
    workers = revision.validate_frozen_workers(clean)
    old_flat = next(row for row in source["domains"] if row["name"] == "flat")
    new_flat = next(row for row in target["domains"] if row["name"] == "flat")
    if (old_flat["contract"], old_flat["sha256"]) != (new_flat["contract"], new_flat["sha256"]):
        raise ValueError("The sealed legacy flat worker changed")
    if digest(clean / new_flat["contract"]) != old_flat["sha256"]:
        raise ValueError("The sealed flat worker bytes differ")
    runtime_checks = []
    for row in roles:
        runtime = row["runtime_expected"]
        domain, role = row["stage"], row["role"]
        if domain == "flat":
            if "ordinary_posture_curriculum" in runtime or "ordinary_posture_curriculum" in runtime["modules"]:
                raise ValueError("The legacy flat role acquired new posture metadata")
            continue
        enabled = role == "train"
        settings = posture.posture_settings(revision.TEACHING_ORIGINS[domain], enabled=enabled)
        expected = posture.runtime_expectation(settings)
        if runtime["modules"].get("ordinary_posture_curriculum") is not enabled:
            raise ValueError("Independent role posture runtime enablement differs")
        if runtime.get("ordinary_posture_curriculum") != expected:
            raise ValueError("Independent role posture runtime settings differ")
        runtime_checks.append(dict(domain=domain, role=role, enabled=enabled,
            runtime_expected=deepcopy(expected)))
    restored_fields = ("skill_specs", "behavior_pool_membership", "behavior_pool_fractions",
        "new_asset_curriculum", "dense_tracking", "recovery_training", "recovery_takeover")
    projected_fields = (*restored_fields, "scene_groups", "episode_seconds", "evaluation",
        "nonstandard_start", "ordinary_posture_curriculum", "jump_cycle_reset_on_accepted")
    exam_checks, source_pins = [], {}
    for domain in revision.DOMAINS_TO_REVISE:
        worker = json.loads((clean / revision.worker_file(domain)).read_text())
        original = deepcopy(worker)
        source_file, source_sha = revision.source_binding(domain, clean)
        if digest(clean / source_file) != source_sha:
            raise ValueError("Actual predecessor exam source SHA differs")
        old_fixed = evaluation.fixed_suite_contract(json.loads((clean / source_file).read_text()))
        fixed = evaluation.fixed_suite_contract(worker)
        if any(fixed.get(name) != old_fixed.get(name) for name in restored_fields):
            raise ValueError("Fixed exam sampling or rewards differ from the actual v15 source")
        if fixed["ordinary_posture_curriculum"]["enabled"] is not False or worker != original:
            raise ValueError("Fixed exam enabled training posture or mutated the worker")
        design = json.loads((clean / worker["design_preflight"]["file"]).read_text())
        source_pins[domain] = train_entry.scut_curriculum_source_files(worker)
        if any(name not in before for name in source_pins[domain]):
            raise ValueError("Runtime source pin escaped archive byte verification")
        for role in ("baseline", "quick", "full", "confirmation"):
            expected = design["stages"][domain]["roles"][role]["contract"]
            configured = deepcopy(worker)
            configured["evaluation"] = deepcopy(expected["evaluation"])
            actual = evaluation.fixed_suite_contract(configured)
            if any(actual.get(name) != expected.get(name) for name in projected_fields):
                raise ValueError("Fixed exam projection differs from independent role input")
            exam_checks.append(dict(domain=domain, role=role, source_file=source_file,
                source_sha256=source_sha, training_posture_disabled=True,
                actual_predecessor_sampling_and_rewards_preserved=True,
                independent_role_projection_matched=True,
                projection_sha256=value_digest({name: actual.get(name) for name in projected_fields})))
    after = {name: digest(clean / name) for name in exact_files}
    if before != after:
        raise ValueError("Role checks changed frozen source bytes")
    report = dict(status="passed", source_commit=args.source_commit,
        archive_sha256=args.archive_sha256, archive_header_commit=archive_commit,
        all_selected_source_bytes_match_archive=True, clean_root=str(clean), role_count=len(roles),
        migration=migration, frozen_workers=workers, duration_seconds=time.monotonic() - started,
        receipts=roles, ordinary_posture_runtime_checks=runtime_checks,
        fixed_evaluation_isolation=exam_checks, legacy_flat_worker_preserved=True,
        runtime_source_pins=source_pins,
        imported_modules={module.__name__: str(Path(module.__file__).resolve()) for module in modules},
        exactsource_files=before, exactsource_files_before=before, exactsource_files_after=after,
        source_files_unchanged=True, validator_sha256=digest(__file__),
        command=[sys.executable, *sys.argv], exitcode=0,
        environment={name: os.environ[name] for name in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "PYTHONDONTWRITEBYTECODE")},
        limitations=["Static migration and role validation do not establish native rollout or hardware qualification."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(status="passed", roles=len(roles), fixed_exam_roles=len(exam_checks),
        output=str(args.output), source_files_unchanged=True,
        duration_seconds=report["duration_seconds"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
