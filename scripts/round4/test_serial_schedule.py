"""Exact cron supersession and elapsed-slot submission without PPO or SSH."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import r4_common as common
from round4 import schedule_serial as scheduler


def test_cancel_only_exact_cron_and_preserve_previous_evidence(tmp_path):
    old = tmp_path / "round4-repaired-20260915"
    old.mkdir()
    (old / "state.json").write_text('{"status":"waiting_repaired_asset"}')
    authorization = json.loads((common.REPO / common.AUTHORIZATION).read_text())
    unrelated = "# user job\n3 4 * * * /other/job\n"
    similar = "* * * * * /other/job # robot-rl-schedule:round4-repaired-20260915-other\n"
    cancelled = f"* * * * * python {old}/schedule.py tick {scheduler.OLD_MARKER}\n"
    cron = [unrelated + cancelled + similar]
    writes = []

    def run(argv, **kwargs):
        if argv == ["crontab", "-l"]:
            return subprocess.CompletedProcess(argv, 0, cron[0], "")
        assert argv == ["crontab", "-"]
        writes.append(kwargs["input"])
        cron[0] = kwargs["input"]
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = scheduler.cancel_old(old, scheduler.OLD_MARKER, authorization, run=run)
    assert writes == [unrelated + similar]
    assert result["status"] == "superseded" and result["removed_lines"] == [cancelled]
    assert result["authorization"]["user_authorization_text"] == authorization["user_authorization_text"]
    assert result["repaired_asset_validation_passed"] is False
    assert json.loads((old / "state-before-supersession.json").read_text())["status"] == "waiting_repaired_asset"
    scheduler.cancel_old(old, scheduler.OLD_MARKER, authorization, run=run)
    assert len(writes) == 1


def test_concurrent_cron_change_is_not_overwritten(tmp_path):
    authorization = json.loads((common.REPO / common.AUTHORIZATION).read_text())
    entry = f"* * * * * python {tmp_path}/schedule.py {scheduler.OLD_MARKER}\n"
    reads = iter([entry, entry + "0 0 * * * /new/user/job\n"])

    def run(argv, **kwargs):
        assert argv == ["crontab", "-l"], "must not overwrite a concurrent user's edit"
        return subprocess.CompletedProcess(argv, 0, next(reads), "")

    with pytest.raises(RuntimeError, match="concurrently"):
        scheduler.cancel_old(tmp_path, scheduler.OLD_MARKER, authorization, run=run)
    assert not (tmp_path / "cancelled.json").exists()


@pytest.mark.parametrize("at,expected", [
    ("2026-09-14T23:59:59+08:00", "waiting_initial_not_before"),
    ("2026-09-15T05:03:00+08:00", "submitted"),
])
def test_elapsed_midnight_is_not_delayed_to_next_day(tmp_path, monkeypatch, at, expected):
    snapshot = {"initial_not_before": "2026-09-15T00:00:00+08:00", "physics_model": common.PHYSICS_MODEL,
                "git_commit": "a" * 40}
    plan = {"python": "/env/bin/python", "repo": str(tmp_path / common.CODE_DIRECTORY), "ground_usd": "/ground.usd"}
    monkeypatch.setattr(scheduler, "verify_snapshot", lambda _: snapshot)
    monkeypatch.setattr(scheduler.launch, "build_plan", lambda _: plan)
    args = SimpleNamespace(experiment=tmp_path, execute=True, soft_hours=48, evaluation_entry="scripts/eval.py")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps({"submitted": True, "plan": plan}), "")

    now = datetime.fromisoformat(at)
    result = scheduler.tick(args, now=now, run=run)
    assert result["status"] == expected
    assert result["initial_not_before"] == snapshot["initial_not_before"]
    assert len(calls) == int(expected == "submitted")
    if calls:
        assert "--launch" in calls[0] and "--evaluation-entry" in calls[0]
        assert scheduler.tick(args, now=now, run=run)["status"] == "submitted"
        assert len(calls) == 1


def test_ambiguous_prior_claim_never_relaunches(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "verify_snapshot", lambda _: {
        "initial_not_before": "2026-09-15T00:00:00+08:00", "physics_model": common.PHYSICS_MODEL, "git_commit": "a" * 40})
    (tmp_path / "serial-schedule.json").write_text('{"status":"launch_claimed"}')
    args = SimpleNamespace(experiment=tmp_path, execute=True)
    result = scheduler.tick(args, now=datetime.fromisoformat("2026-09-15T05:03:00+08:00"),
                            run=lambda *_, **__: pytest.fail("must not duplicate a claimed launch"))
    assert result["status"] == "launch_claimed"
