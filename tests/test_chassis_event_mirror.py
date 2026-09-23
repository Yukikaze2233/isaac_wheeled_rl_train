"""A live event mirror is append-only, isolated by run, and never a model archive."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mirror_chassis_events as mirror


def test_event_mirror_copies_only_events_and_rejects_another_owner(tmp_path, monkeypatch):
    owner = {"host": "user@host", "ssh_port": 2222, "remote_root": "/owned/run", "identity_file": "/key path"}
    calls = []
    monkeypatch.setattr(mirror.subprocess, "run", lambda command, **kwargs: calls.append(command))
    result = mirror.mirror_once(owner, tmp_path)
    command = calls[0]
    assert "--append-verify" in command and "--delete" not in command
    assert "--include=events.out.tfevents.*" in command and "--exclude=*" in command
    assert command[-2] == "user@host:/owned/run/train/"
    assert "'/key path'" in command[command.index("-e") + 1]
    assert result["scope"] == "live_event_cache_not_sealed_evidence"
    assert json.loads((tmp_path / "mirror_status.json").read_text())["status"] == "synced"
    with pytest.raises(ValueError, match="another run"):
        mirror.mirror_once({**owner, "remote_root": "/different/run"}, tmp_path)
    assert len(calls) == 1
