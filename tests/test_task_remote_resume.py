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
