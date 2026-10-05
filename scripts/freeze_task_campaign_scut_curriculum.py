#!/usr/bin/env python3
"""Freeze v17 recipes; no process, SSH, checkpoint or historical file mutation."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis import scut_curriculum_revision as revision


def write(file, value):
    (ROOT / file).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    for domain in revision.DOMAINS_TO_REVISE:
        source_file, source_sha, worker_file, design_file, apply, design_recipe, _ = revision._recipe(domain)
        write(design_file, design_recipe(ROOT))
        worker = revision.read_bound(source_file, source_sha, ROOT)
        apply(worker, ROOT)
        worker["design_preflight"] = dict(file=design_file, sha256=revision.digest(ROOT / design_file), stage=domain)
        write(worker_file, worker)
    worker_receipts = revision.validate_frozen_workers(ROOT)
    write(revision.PLAN_FILE, revision.revised_plan(ROOT))
    write(revision.MANIFEST_FILE, revision.revised_manifest(ROOT))
    print(json.dumps(dict(plan_sha256=revision.digest(ROOT / revision.PLAN_FILE),
        manifest_sha256=revision.digest(ROOT / revision.MANIFEST_FILE),
        revised_workers=worker_receipts, unchanged_worker_count=2,
        scope="frozen_candidate_not_deployment_or_qualification")))


if __name__ == "__main__":
    main()
