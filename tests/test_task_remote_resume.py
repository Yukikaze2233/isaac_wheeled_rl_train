import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chassis_remote_job import prepare_job_log


def test_resume_preserves_logs_and_requires_same_command(tmp_path):
    command = ["python", "frozen/run_task_campaign.py", "--run-dir", str(tmp_path / "train")]
    (tmp_path / "job.json").write_text(json.dumps({"pid": 2147483647, "command": command}))
    (tmp_path / "train.log").write_text("original")
    path = prepare_job_log(tmp_path, command, resume=True)
    assert path.name == "train.resume_001.log"
    path.write_text("first recovery")
    assert prepare_job_log(tmp_path, command, resume=True).name == "train.resume_002.log"
    assert (tmp_path / "train.log").read_text() == "original"
    with pytest.raises(ValueError, match="original frozen command"):
        prepare_job_log(tmp_path, [*command, "--hot-change"], resume=True)


def test_resume_rejects_live_owner(tmp_path, monkeypatch):
    command = ["python", "run_task_campaign.py", "--run-dir", str(tmp_path / "train")]
    (tmp_path / "job.json").write_text(json.dumps({"pid": os.getpid(), "command": command}))
    native = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"python\0" + os.fsencode(str(tmp_path / "train"))
                        if str(p).startswith("/proc/") else native(p))
    with pytest.raises(ValueError, match="still alive"):
        prepare_job_log(tmp_path, command, resume=True)


def revision(tmp_path):
    import tarfile
    from chassis_remote_job import sha
    old, new = tmp_path / "source_old", tmp_path / "source_new"
    for source in (old, new):
        (source / "scripts").mkdir(parents=True)
        (source / "contracts").mkdir()
        (source / "src").mkdir()
        (source / "scripts/run_task_campaign.py").write_text("# orchestrate\n")
        (source / "contracts/plan.json").write_text('{"campaign":"unchanged"}\n')
        (source / "src/env.py").write_text("original\n" if source == old else "fixed\n")
    command = ["python", "-B", str(old / "scripts/run_task_campaign.py"),
               "--contract", str(old / "contracts/plan.json"), "--run-dir", str(tmp_path / "train"),
               "--num-envs", "20000", "--max-runtime-seconds", "15552000"]
    (tmp_path / "job.json").write_text(json.dumps({"pid": 2147483647, "command": command}))
    (tmp_path / "train.log").write_text("failed original attempt\n")
    revised = [arg.replace(str(old), str(new)) for arg in command]
    archive = tmp_path / "source.tar.gz"
    commit = "a" * 40
    with tarfile.open(archive, "w:gz", pax_headers={"comment": commit}) as target:
        for path in sorted(new.rglob("*")):
            if path.is_file():
                target.add(path, arcname=str(path.relative_to(new)), recursive=False)
    receipt = dict(schema="task-source-revision-v1", previous_job_sha256=sha(tmp_path / "job.json"),
        old_source_root=str(old), new_source_root=str(new), command=revised,
        old_contract_sha256=sha(old / "contracts/plan.json"),
        new_contract_sha256=sha(new / "contracts/plan.json"), source_commit=commit,
        source_archive=str(archive), source_archive_sha256=sha(archive))
    receipt_path = tmp_path / "revision.json"
    receipt_path.write_text(json.dumps(receipt))
    return revised, receipt, receipt_path, old, new


def test_source_revision_changes_only_frozen_prefix_and_preserves_all_evidence(tmp_path):
    from chassis_remote_job import finalize, preserve_previous_job
    command, receipt, receipt_path, old, new = revision(tmp_path)
    previous = (tmp_path / "job.json").read_bytes()
    log = prepare_job_log(tmp_path, command, resume=True, source_revision_receipt=receipt_path)
    assert log.name == "train.resume_001.log"
    log.write_text("new attempt")
    history = preserve_previous_job(tmp_path, log, receipt_path)
    assert history == "job.previous_001.json"
    assert (tmp_path / history).read_bytes() == previous
    assert (tmp_path / "job.source_revision_001.json").read_bytes() == receipt_path.read_bytes()
    assert (old / "src/env.py").read_text() == "original\n"
    (tmp_path / "train").mkdir()
    (tmp_path / "train/completion.json").write_text('{"status":"stopped","successful_updates":1000}')
    files = finalize(tmp_path, 0)["files"]
    assert {"train.log", "train.resume_001.log", "job.previous_001.json", "job.source_revision_001.json", "job.json"} <= set(files)


@pytest.mark.parametrize("change", ["argv", "run_dir", "contract", "archive", "extracted", "job", "commit", "same_source"])
def test_source_revision_rejects_changes_outside_authenticated_code_repair(tmp_path, change):
    command, receipt, receipt_path, old, new = revision(tmp_path)
    if change == "argv":
        command[command.index("--num-envs") + 1] = "8192"
        receipt["command"] = command
    elif change == "run_dir":
        command[command.index("--run-dir") + 1] = str(tmp_path / "different")
        receipt["command"] = command
    elif change == "contract":
        (new / "contracts/plan.json").write_text('{"campaign":"changed"}')
    elif change == "archive":
        receipt["source_archive_sha256"] = "0" * 64
    elif change == "extracted":
        (new / "src/env.py").write_text("unfrozen hot patch")
    elif change == "job":
        receipt["previous_job_sha256"] = "0" * 64
    elif change == "commit":
        receipt["source_commit"] = "b" * 40
    elif change == "same_source":
        receipt["new_source_root"] = str(old)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        prepare_job_log(tmp_path, command, resume=True, source_revision_receipt=receipt_path)
    assert not (tmp_path / "train.resume_001.log").exists()


def test_revision_needs_explicit_resume_and_rejects_live_owner(tmp_path, monkeypatch):
    command, receipt, receipt_path, old, new = revision(tmp_path)
    with pytest.raises(ValueError, match="resume-existing"):
        prepare_job_log(tmp_path, command, source_revision_receipt=receipt_path)
    native_exists, native_read = Path.exists, Path.read_bytes
    process = "/proc/2147483647/cmdline"
    monkeypatch.setattr(Path, "exists", lambda p: True if str(p) == process else native_exists(p))
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"python\0" + os.fsencode(str(tmp_path / "train"))
                        if str(p) == process else native_read(p))
    with pytest.raises(ValueError, match="still alive"):
        prepare_job_log(tmp_path, command, resume=True, source_revision_receipt=receipt_path)
