"""Scheduler tests use synthetic receipts, never launch a simulator or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("repaired_schedule", Path(__file__).with_name("schedule.py"))
schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schedule)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    base = tmp_path / "deployment"
    request = tmp_path / "job" / "request.json"
    put(request, {"schema_version": 1, "id": "repaired-r4-20260915", "deployment_root": str(base),
                  "not_before": "2026-09-15T00:00:00+08:00", "expires_at": "2026-09-18T00:00:00+08:00",
                  "require_repaired_dynamics": True})
    commit = "a" * 40
    experiment = base / "experiments" / ("round4-full-" + commit)
    repo = experiment / "isaac_wheeled_rl_train"
    put(base / "env/bin/python", "test executable placeholder")
    ground_hash = put(base / "ground.usd", "test ground placeholder")
    source = {
        "scripts/round4/launch.py": "test launcher placeholder",
        "contracts/repaired.json": {"asset": {"directory": "assets/new", "manifest": "manifest.json"}},
        "assets/new/manifest.json": {"test_fixture": True},
    }
    hashes = {name: put(repo / name, value) for name, value in source.items()}
    snapshot_hash = put(experiment / "snapshot.json", {
        "git_commit": commit, "contract": "contracts/repaired.json",
        "files": {"isaac_wheeled_rl_train/" + name: {"sha256": value} for name, value in hashes.items()},
    })
    validation = {"scope": "v40_repaired_dynamics", "passed": True,
                  "checks": dict.fromkeys(schedule.CHECKS, True),
                  "contract_file_sha256": hashes["contracts/repaired.json"],
                  "asset_manifest_sha256": hashes["assets/new/manifest.json"]}
    validation_path = experiment / "repair-validation.json"
    validation_hash = put(validation_path, validation)
    ready = {"git_commit": commit, "snapshot_sha256": snapshot_hash,
             "validation_path": str(validation_path), "validation_sha256": validation_hash,
             "ground_usd": str(base / "ground.usd"), "ground_sha256": ground_hash}
    return request, repo, ready, validation


def no_launch(*args, **kwargs):
    pytest.fail("a waiting/invalid schedule invoked a launcher")


@pytest.mark.parametrize("time,status", [
    ("2026-09-14T23:59:59+08:00", "scheduled_waiting_time"),
    ("2026-09-15T00:00:00+08:00", "waiting_repaired_asset"),
    ("2026-09-18T00:00:00+08:00", "expired"),
])
def test_time_and_absent_asset_never_launch(setup, time, status):
    request, _, _, _ = setup
    assert schedule.tick(request, now=schedule.stamp(time), run=no_launch)["status"] == status


def test_ready_asset_still_waits_until_midnight(setup):
    request, _, ready, _ = setup
    put(request.parent / "ready.json", ready)
    assert schedule.tick(request, now=schedule.stamp("2026-09-14T15:59:59+00:00"),
                         run=no_launch)["status"] == "scheduled_waiting_time"


def test_wait_until_ready_has_no_implicit_deadline(setup):
    request, _, _, _ = setup
    value = schedule.load(request)
    value["expires_at"] = None
    put(request, value)
    result = schedule.tick(request, now=schedule.stamp("2026-09-25T12:00:00+08:00"), run=no_launch)
    assert result["status"] == "waiting_repaired_asset"


@pytest.mark.parametrize("fault", ["mass", "old_asset", "source_changed", "contract_changed"])
def test_geometry_only_old_or_tampered_asset_is_blocked(setup, monkeypatch, fault):
    request, repo, ready, validation = setup
    if fault == "mass":
        validation["checks"]["mass_properties"] = False
        ready["validation_sha256"] = put(Path(ready["validation_path"]), validation)
    elif fault == "old_asset":
        monkeypatch.setattr(schedule, "OLD_ASSET", validation["asset_manifest_sha256"])
    else:
        name = "scripts/round4/launch.py" if fault == "source_changed" else "contracts/repaired.json"
        put(repo / name, "changed after validation")
    put(request.parent / "ready.json", ready)
    result = schedule.tick(request, now=schedule.stamp("2026-09-15T00:01:00+08:00"), run=no_launch)
    assert result["status"] == "blocked_readiness"


@pytest.mark.parametrize("submission_code,status", [(0, "submitted"), (2, "launch_failed")])
def test_exactly_once_submission_even_after_failure(setup, submission_code, status):
    request, repo, ready, _ = setup
    put(request.parent / "ready.json", ready)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "--launch" in argv:
            value = {"submitted": submission_code == 0, "training_success": False}
            return subprocess.CompletedProcess(argv, submission_code, json.dumps(value), "")
        plan = {"initialization": "scratch", "parent": None, "num_envs": 1024,
                "updates": 30000, "git_commit": ready["git_commit"], "command": ["python", "train_v40.py"]}
        return subprocess.CompletedProcess(argv, 0, json.dumps({"dry_run": True, "plan": plan}), "")

    now = schedule.stamp("2026-09-15T00:01:00+08:00")
    result = schedule.tick(request, now=now, run=run)
    assert result["status"] == status and result["training_success"] is False
    assert len(calls) == 2 and "--launch" not in calls[0] and "--launch" in calls[1]
    assert schedule.tick(request, now=now, run=no_launch)["status"] == status


def test_cron_install_preserves_existing_jobs(setup):
    request, _, _, _ = setup
    original = "# Existing user job\n3 4 * * * /existing/job\n"
    writes = []

    def run(argv, **kwargs):
        if argv == ["crontab", "-l"]:
            return subprocess.CompletedProcess(argv, 0, original, "")
        writes.append(kwargs["input"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = schedule.install(request, run=run)
    assert result["installed"] and writes == [original + result["cron_entry"] + "\n"]


def test_timezone_is_required():
    with pytest.raises(ValueError, match="timezone"):
        schedule.stamp("2026-09-15T00:00:00")
