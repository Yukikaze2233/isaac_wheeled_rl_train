"""V10 reaches the named plan validator without weakening job ownership."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from wheeled_tasks.chassis import task_campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from chassis_remote_job import prepare_job_log, preserve_previous_job, sha, validate_source_revision


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def revision(tmp_path, monkeypatch):
    old, new = tmp_path / "source_old", tmp_path / "source_new"
    source, target = ({"contract_id": f"v6-task-campaign-v{version}"} for version in (9, 10))
    old_name = "contracts/v6_task_campaign_v9.json"
    new_name = "contracts/v6_task_campaign_v10.json"
    for directory, name, plan in ((old, old_name, source), (new, new_name, target)):
        write(directory / name, plan)
        (directory / "scripts").mkdir()
        (directory / "scripts/run_task_campaign.py").write_text("# Fixture runner is never executed.\n")
    command = [sys.executable, "-B", str(old / "scripts/run_task_campaign.py"),
               "--contract", str(old / old_name), "--num-envs", "4096", "--device", "cuda:0",
               "--research", "--max-runtime-seconds", "15552000", "--run-dir", str(tmp_path / "train")]
    previous = dict(pid=2147483647, command=command)
    write(tmp_path / "job.json", previous)
    (tmp_path / "train.log").write_text("Original evidence must survive.\n")
    revised = [arg.replace(str(old), str(new)) for arg in command]
    revised[revised.index("--contract") + 1] = str(new / new_name)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(new.rglob("*")):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(new)), recursive=False)
    receipt = dict(schema="task-plan-revision-v10", previous_job_sha256=sha(tmp_path / "job.json"),
        command=revised, source_commit="a" * 40, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path=new_name, old_contract_sha256=sha(old / old_name),
        new_contract_sha256=sha(new / new_name), source_archive=str(archive), source_archive_sha256=sha(archive))
    path = tmp_path / "source_revision_011.json"
    write(path, receipt)
    calls = []

    def validate_plan_migration(before, after, run_directory):
        assert before == source and after == target
        assert run_directory == tmp_path / "train"
        calls.append((before, after, run_directory))

    # Scope/worker semantics are independently covered by the plan validator's
    # tests. Here we verify the remote entry authenticates and delegates to it.
    monkeypatch.setattr(task_campaign, "validate_plan_migration", validate_plan_migration)
    return dict(root=tmp_path, old=old, new=new, source=source, target=target,
                previous=previous, command=revised, receipt=receipt, path=path, calls=calls)


def test_v10_changes_source_and_named_plan_only_and_preserves_old_job(revision):
    f = revision
    result = validate_source_revision(f["root"], f["command"], f["previous"], f["path"])
    assert result == f["receipt"]
    assert len(f["calls"]) == 1
    log = prepare_job_log(f["root"], f["command"], resume=True, source_revision_receipt=f["path"])
    assert log.name == "train.resume_001.log"
    preserved = preserve_previous_job(f["root"], log, f["path"])
    assert json.loads((f["root"] / preserved).read_text()) == f["previous"]
    assert (f["root"] / "job.source_revision_001.json").read_bytes() == f["path"].read_bytes()
    assert (f["root"] / "train.log").read_text() == "Original evidence must survive.\n"


@pytest.mark.parametrize("option,value", [("--num-envs", "20000"), ("--device", "cuda:1"),
    ("--run-dir", "/other/train"), ("--max-runtime-seconds", "1")])
def test_v10_receipt_cannot_authorize_other_command_changes(revision, option, value):
    f = revision
    revised = list(f["command"])
    revised[revised.index(option) + 1] = value
    receipt = deepcopy(f["receipt"])
    receipt["command"] = revised
    write(f["path"], receipt)
    with pytest.raises(ValueError, match="only runner and contract"):
        validate_source_revision(f["root"], revised, f["previous"], f["path"])
    assert not f["calls"]


@pytest.mark.parametrize("field", ["previous_job_sha256", "old_contract_sha256",
    "new_contract_sha256", "source_archive_sha256"])
def test_v10_retains_receipt_hash_authentication(revision, field):
    f = revision
    receipt = deepcopy(f["receipt"])
    receipt[field] = "0" * 64
    write(f["path"], receipt)
    with pytest.raises(ValueError):
        validate_source_revision(f["root"], f["command"], f["previous"], f["path"])
    assert not f["calls"]


def test_v10_retains_extracted_source_archive_verification(revision):
    f = revision
    (f["new"] / "scripts/run_task_campaign.py").write_text("# Undeclared overlay.\n")
    with pytest.raises(ValueError, match="extracted file differs"):
        validate_source_revision(f["root"], f["command"], f["previous"], f["path"])
    assert not f["calls"]


def test_v10_does_not_launch_over_a_live_existing_owner(revision):
    f = revision
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(f["root"] / "train")])
    try:
        previous = dict(f["previous"], pid=process.pid)
        write(f["root"] / "job.json", previous)
        receipt = dict(f["receipt"], previous_job_sha256=sha(f["root"] / "job.json"))
        write(f["path"], receipt)
        with pytest.raises(ValueError, match="still alive"):
            prepare_job_log(f["root"], f["command"], resume=True, source_revision_receipt=f["path"])
        assert not (f["root"] / "train.resume_001.log").exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_v10_cannot_bypass_named_plan_migration_validation(revision, monkeypatch):
    f = revision

    def reject(*args):
        raise ValueError("Named migration failed")

    monkeypatch.setattr(task_campaign, "validate_plan_migration", reject)
    with pytest.raises(ValueError, match="Named migration failed"):
        prepare_job_log(f["root"], f["command"], resume=True, source_revision_receipt=f["path"])
    assert not (f["root"] / "job.previous_001.json").exists()
