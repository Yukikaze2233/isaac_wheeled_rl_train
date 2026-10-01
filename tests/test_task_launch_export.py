"""Exercise task campaign CLI construction and actual immutable batch archives."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import chassis_batch_export as batch_export
import launch_v5_remote as launcher
from run_task_campaign import TaskCampaign

DEADLINE = 180 * 86400
PLAN_FILE = "contracts/v6_task_campaign_v1.json"


def json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def launch_fixture(monkeypatch, output, *, deadline=DEADLINE, execute=False, extra=()):
    calls = []
    commit = "a" * 40

    def check_output(command, **kwargs):
        calls.append(command)
        if command[:2] == ["git", "rev-parse"]:
            return commit + "\n"
        if command[:2] == ["git", "show"]:
            path = command[2].split(":", 1)[1]
            return (ROOT / path).read_bytes()
        if command[0] == "ssh":
            return json.dumps({"mem_available_kib": 64 * 1024**2})
        raise AssertionError(f"Unexpected external command: {command}")

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "check_output", check_output)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "freeze_source", lambda commit, overlays, archive: archive.write_bytes(b"frozen commit archive"))
    args = ["launch_v5_remote.py", "--commit", commit, "--contract", PLAN_FILE,
            "--num-envs", "20000", "--seed", "290927", "--output", str(output), *extra]
    if deadline is not None:
        args += ["--max-runtime-seconds", str(deadline)]
    if execute:
        args.append("--execute")
    monkeypatch.setattr(sys, "argv", args)
    return calls


def test_task_launcher_builds_campaign_command_with_explicit_180_day_cap(tmp_path, monkeypatch):
    output = tmp_path / "dry"
    calls = launch_fixture(monkeypatch, output)
    launcher.main()
    plan = json.loads((output / "plan.json").read_text())
    command = plan["command"]
    assert command[2].endswith("/scripts/run_task_campaign.py")
    assert command[command.index("--max-runtime-seconds") + 1] == str(DEADLINE)
    assert command[command.index("--num-envs") + 1] == "20000"
    assert "--updates" not in command and "--resume" not in command
    assert plan["updates"] == 120000 and plan["consumed_ceiling"] == 121000
    assert plan["training_transitions"] == 120000 * 20000 * 24
    assert plan["updates_are_caps"] and plan["historical_consumed_updates"] == 1000
    assert plan["initialization"] == "A_then_named_parent_full_learning_state"
    assert plan["scope"] == "independent_task_domain_research"
    assert all(command[0] == "git" for command in calls)


def test_execute_passes_long_deadline_to_preflight_worker_and_outer_timeout(tmp_path, monkeypatch):
    output = tmp_path / "execute"
    calls = launch_fixture(monkeypatch, output, execute=True)
    launcher.main()
    receipt = json.loads((output / "launch.json").read_text())
    assert receipt["archive_sha256"] == hashlib.sha256(b"frozen commit archive").hexdigest()
    remote_calls = [call[-1] for call in calls if call[0] == "ssh"]
    preflight = next(command for command in remote_calls if "--preflight-only" in command)
    assert str(DEADLINE) in preflight and "run_task_campaign.py" in preflight
    launch = next(command for command in remote_calls if "new-session" in command)
    inner = shlex.split(launch)[-1]
    assert f"timeout --signal=TERM --kill-after=120s {DEADLINE + 1200}s" in inner
    assert f"--max-runtime-seconds {DEADLINE}" in inner
    assert "172800" not in inner
    assert remote_calls.index(preflight) < remote_calls.index(launch)


@pytest.mark.parametrize("extra,deadline", [([], None), (["--updates", "40"], DEADLINE),
    (["--capacity-envs", "20000"], DEADLINE), (["--transfer", "/model.pt"], DEADLINE),
    (["--num-envs", "4096"], DEADLINE), (["--seed", "617"], DEADLINE)])
def test_task_launcher_rejects_legacy_default_deadline_and_partial_overrides(tmp_path, monkeypatch, extra, deadline):
    output = tmp_path / "invalid"
    calls = launch_fixture(monkeypatch, output, extra=extra, deadline=deadline)
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2
    assert not output.exists()
    assert all(command[0] == "git" for command in calls)


def test_campaign_worker_uses_remaining_campaign_deadline_not_default_48_hours(tmp_path, monkeypatch):
    import run_task_campaign as campaign
    plan = json.loads((ROOT / PLAN_FILE).read_text())
    args = SimpleNamespace(contract=ROOT / PLAN_FILE, run_dir=tmp_path / "run", device="cpu",
                           max_runtime_seconds=DEADLINE)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: 100.)
    task = TaskCampaign(args)
    source = tmp_path / "parent" / "model_final.pt"
    source.parent.mkdir()
    source.write_bytes(b"checkpoint")
    source.with_suffix(".checkpoint.json").write_text(json.dumps({"stage_actor_update": 368}))
    source_contract = json_file(source.parent / "contract.json", {"source": "A"})
    monkeypatch.setattr(campaign, "checkpoint_contract_path", lambda _: source_contract)
    task.report.update(consumed_updates=1000, actor_updates_consumed=980)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: 3700.)
    command = task.train_command(plan["domains"][0], tmp_path / "worker", source, 20000)
    assert float(command[command.index("--max-runtime-seconds") + 1]) == DEADLINE - 3600
    assert command[command.index("--task-actor-update") + 1] == "0"
    assert command[command.index("--stage-actor-update") + 1] == "368"
    assert "--resume-task-change" in command


def test_task_hierarchy_exports_sealed_snapshots_and_two_seed_evaluations(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    worker = root / "train/flat/attempt_000/train"
    snapshot = worker / "checkpoints/update_00001001"
    json_file(snapshot / "completion.json", {"status": "checkpoint_sealed"})
    (snapshot / "model.pt").write_bytes(b"immutable model")
    json_file(snapshot / "contract.json", {"contract": "frozen task"})
    pending = worker / "checkpoints/update_00001100.pending"
    json_file(pending / "completion.json", {"status": "checkpoint_sealed"})
    json_file(worker / "progress.json", {"successful_updates": 1})
    json_file(worker / "completion.json", {"status": "completed"})
    (worker / "events.out.tfevents.test").write_bytes(b"event stream")
    eval_dirs = []
    for seed in (290929, 290930):
        directory = root / f"train/flat/evaluations/task_00001000_model/{seed}"
        json_file(directory / "evaluation.json", {"status": "evaluated", "seed": seed})
        eval_dirs.append(directory)
    json_file(root / "train/completion.json", {"status": "task_domains_finished"})
    live = {worker, eval_dirs[1]}
    monkeypatch.setattr(batch_export, "writer_alive", lambda directory: directory in live)
    first = batch_export.export_ready(root)
    assert not first["run_finished"]
    units = {r["unit"] for r in first["batches"]}
    assert str(snapshot.relative_to(root)) in units
    assert str(eval_dirs[0].relative_to(root)) in units
    assert str(eval_dirs[1].relative_to(root)) not in units
    assert str(worker.relative_to(root)) not in units
    assert not any(unit.endswith(".pending") for unit in units)
    live.clear()
    second = batch_export.export_ready(root)
    assert second["run_finished"]
    by_unit = {r["unit"]: r for r in second["batches"]}
    assert by_unit[str(worker.relative_to(root))]["kind"] == "training_block"
    assert by_unit[str(eval_dirs[1].relative_to(root))]["kind"] == "evaluation"
    for receipt in second["batches"]:
        archive = root / receipt["archive"]
        assert batch_export.digest(archive) == receipt["archive_sha256"]
        with tarfile.open(archive) as bundle:
            assert set(bundle.getnames()) == set(receipt["files"])
            for name, meta in receipt["files"].items():
                assert hashlib.sha256(bundle.extractfile(name).read()).hexdigest() == meta["sha256"]
    third = batch_export.export_ready(root)
    assert third["batches"] == second["batches"]
