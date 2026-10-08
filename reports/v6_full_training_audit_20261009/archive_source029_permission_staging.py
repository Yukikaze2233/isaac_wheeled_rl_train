#!/usr/bin/env python3
"""Archive this one permission-mismatched source029 staging; no SSH/delete.

Retains four original objects by same-filesystem rename. This is not a retry
runner: an existing archival intent/directory requires manual inspection.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import stat

BAD_COMMIT = "e5bd9c1367b875d7f229b9a1930647013d023372"
BAD_ARCHIVE = "55e3af79ede31160e0cf2f6993cc535bf384ef0ad0e52dbfbe31ccbad3518423"
BAD_REVISION = "f825f71278babaeba59eaed81c2c2ef545386f2a33bc0bba80e3b320bc5156dd"
HELPER = "3864f047ffea30989cce30db1bd697c20d5f217d70286d09c77c12198c05b963"
PREDECESSOR = "335542ddc753b84b2778609b92803064682198a9"
PARENT = "e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967"
NAMES = ("source_serial_dispatch029_20261009", "source_serial_dispatch029_20261009.tar.gz",
         "source_revision_029.json", "source029_deploy")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    if (not args.run_root.is_absolute()
            or root.name != "v6-task-campaign-20261001T110045Z-282358"):
        parser.error("Only this exact owned experiment may be archived")
    rejected = root / "source029_staging_rejected_archive002"
    if rejected.exists():
        raise ValueError("Rejected archive/intent already exists; inspect, never repeat implicitly")
    old = root / "source_integrated_v26_20261008"
    sys.path[:0] = [str(old / "scripts"), str(old / "src")]
    from chassis_remote_job import validate_source_revision
    from queue_task_campaign_continuation import owned_processes
    from wheeled_tasks.chassis.task_campaign import CAMPAIGN_ID, account_updates
    plan = read(old / "contracts/v6_task_campaign_v26.json")
    claim = Path(plan["budget_source_stage"]) / (CAMPAIGN_ID + ".account.json")
    with claim.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if owned_processes(root):
            raise ValueError("Campaign owner still active")
        if read(root / "source_revision_028.json")["source_commit"] != PREDECESSOR:
            raise ValueError("Current predecessor is not the actual source028")
        job = read(root / "job.json")
        if str(old / "scripts/run_task_campaign.py") not in job["command"]:
            raise ValueError("The current job already adopted a different source")
        if read(root / "delivery.json").get("status") != "ready" or (root / "delivery.tar.gz.tmp").exists():
            raise ValueError("Old wrapper archive is not ready")
        state, completion, account = read(root / "train/campaign_state.json"), read(root / "train/completion.json"), read(claim)
        totals = account_updates(plan, account["attempts"])
        if (state != completion or state.get("status") != "stopped"
                or account != read(root / "train/consumption.json")
                or account.get("rollback_refunds_consumed_updates") is not False
                or totals["consumed_updates"] != 41393 or totals["actor_updates_consumed"] != 41223
                or any(value.get(key) != count for value in (state, account) for key, count in totals.items())):
            raise ValueError("Stopped claim/state lost or changed the 19 retained updates")
        if (root / "train/integrated_early_origin_v26_dispatch_correction029").exists():
            raise ValueError("Origin correction already occurred; this is not staging-only rejection")
        stage_directory = root / NAMES[3]
        if sorted(p.name for p in stage_directory.iterdir()) != ["origin_check.log", "preflight.log", "stage.json"]:
            raise ValueError("Only this pinned staging and failed CPU preflight logs may be archived")
        logs = [stage_directory / name for name in ("origin_check.log", "preflight.log")]
        if any(not path.is_file() or path.is_symlink() for path in logs):
            raise ValueError("CPU preflight logs are absent or redirected")
        expected_error = "ValueError: Source029 changed a frozen non-repair file: contracts/v6_task_campaign_migration_v24.json"
        if (not logs[0].read_text().rstrip().endswith(expected_error)
                or logs[1].stat().st_size != 34):
            raise ValueError("CPU preflight is not the authenticated source029 permission failure")
        same_file = Path("contracts/v6_task_campaign_migration_v24.json")
        old_file, new_file = old / same_file, root / NAMES[0] / same_file
        expected_sha = "2acddfb3b49540e2a882873100b155f24839f0f7bc5c63ea80dc3c3bfc2f1234"
        if (sha(old_file) != expected_sha or sha(new_file) != expected_sha
                or stat.S_IMODE(old_file.stat().st_mode) != 0o644
                or stat.S_IMODE(new_file.stat().st_mode) != 0o664):
            raise ValueError("Named old/new contract must differ only by archive permissions")
        for name in NAMES:
            path = root / name
            if not path.exists() or path.is_symlink():
                raise ValueError("A required staging object is absent or redirected")
        stage, revision = read(stage_directory / "stage.json"), read(root / NAMES[2])
        if (stage.get("source_commit") != BAD_COMMIT or stage.get("source_archive_sha256") != BAD_ARCHIVE
                or stage.get("receipt_sha256") != BAD_REVISION or stage.get("helper_sha256") != HELPER
                or stage.get("regular_files") != 4313 or sha(root / NAMES[1]) != BAD_ARCHIVE
                or sha(root / NAMES[2]) != BAD_REVISION or revision.get("source_commit") != BAD_COMMIT
                or stage.get("PPO_updates") != 0 or stage.get("native_scenes_created") != 0):
            raise ValueError("Rejected staging is not the pinned e5bd9/4313-file permission artifact")
        # Existing native wrapper authenticator verifies all extracted bytes
        # against the retained tar and authenticates unchanged current job argv.
        validate_source_revision(root, revision["command"], job, root / NAMES[2])
        parent = root / "train/recovery/attempt_010/train/model_final.pt"
        if sha(parent) != PARENT:
            raise ValueError("Latest REC parent changed")
        unchanged = {str(path): sha(path) for path in (
            root / "job.json", claim, root / "train/consumption.json", root / "train/campaign_state.json",
            root / "train/completion.json", parent, root / "source_revision_028.json")}
        rejected.mkdir(exist_ok=False)
        intent = dict(schema="source029-staging-rejection-archive-v1",
            reason="CPU origin preflight refused tar permission mismatch 0644/0664; byte-identical source; no origin correction or PPO",
            rejected_commit=BAD_COMMIT, rejected_archive_sha256=BAD_ARCHIVE,
            rejected_source_revision_sha256=BAD_REVISION, unchanged=unchanged,
            rejected_cpu_logs_sha256={path.name: sha(path) for path in logs},
            moves=[dict(original=str(root / name), archived=str(rejected / name)) for name in NAMES],
            qualification=None, PPO_updates=0, source_files_modified=False)
        publish(rejected / "intent.json", intent)
        for name in NAMES:
            (root / name).rename(rejected / name)
            sync_directory(root)
            sync_directory(rejected)
        if any((root / name).exists() for name in NAMES) or any(sha(path) != digest for path, digest in unchanged.items()):
            raise ValueError("Archival verification failed; preserve intent and inspect partial state")
        if (sha(rejected / NAMES[1]) != BAD_ARCHIVE or sha(rejected / NAMES[2]) != BAD_REVISION
                or read(rejected / NAMES[3] / "stage.json") != stage):
            raise ValueError("Archived bytes differ from the rejected originals")
        result = dict(status="rejected_staging_archived_no_training_changes", **intent,
            intent_sha256=sha(rejected / "intent.json"), original_paths_clear=True,
            next_step="Copy same e5bd9 commit archived with tar.umask=0022 to original basename, then unchanged helper mode stage with its new archive SHA")
        publish(rejected / "receipt.json", result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
