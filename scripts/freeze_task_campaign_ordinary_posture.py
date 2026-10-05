#!/usr/bin/env python3
"""Freeze direct v15-to-v18 posture lessons; no remote or learning mutations."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis import ordinary_posture_revision as revision


def write(file, value):
    (ROOT / file).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    for domain in revision.DOMAINS_TO_REVISE:
        write(revision.design_file(domain), revision.revised_design(domain, ROOT))
        write(revision.worker_file(domain), revision.canonical_worker(domain, ROOT))
    receipts = revision.validate_frozen_workers(ROOT)
    write(revision.PLAN_FILE, revision.revised_plan(ROOT))
    write(revision.MANIFEST_FILE, revision.revised_manifest(ROOT))
    print(json.dumps(dict(plan_sha256=revision.digest(ROOT / revision.PLAN_FILE),
        manifest_sha256=revision.digest(ROOT / revision.MANIFEST_FILE), revised_workers=receipts,
        unchanged_worker_count=1, scope="frozen_candidate_not_deployment_or_qualification")))


if __name__ == "__main__":
    main()
