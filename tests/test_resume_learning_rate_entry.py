"""Explicit resumed LR reaches one worker without changing frozen contracts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_new_asset_blocks import Campaign, argument
from test_new_asset_preflight import load_full_entry

ROOT = Path(__file__).resolve().parents[1]
SCALE = 1.9509221155311695


def entry(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name + "_resume_lr", ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["launch_v5_remote", "run_full_chassis", "run_chassis_blocks"])
@pytest.mark.parametrize("scale,resume", [("0", True), ("-1", True), ("nan", True),
    ("inf", True), ("-inf", True), ("0.5", False)])
def test_resume_lr_entry_rejects_invalid_values_before_reading_or_launching(
        tmp_path, monkeypatch, capsys, script, scale, resume):
    module = entry(script)
    args = [script, "--contract", str(tmp_path / "unused.json"),
            "--resume-learning-rate-scale=" + scale]
    if script == "launch_v5_remote":
        args += ["--commit", "HEAD", "--output", str(tmp_path / "unused")]
    else:
        args += ["--run-dir", str(tmp_path / "unused")]
    if resume:
        args += ["--resume", str(tmp_path / "unused.pt")]
    monkeypatch.setattr(module.sys, "argv", args)
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2
    assert "Resume learning-rate scale requires --resume" in capsys.readouterr().err
    assert not (tmp_path / "unused").exists()


def test_launcher_to_full_stage_preserves_explicit_scale_and_consumption(
        tmp_path, monkeypatch):
    launch = entry("launch_v5_remote")
    out = tmp_path / "launch_plan"
    monkeypatch.setattr(launch.sys, "argv", ["launch_v5_remote", "--commit", "HEAD",
        "--contract", "contracts/v6_recovery_curriculum_training_v1.json", "--stage", "curriculum",
        "--num-envs", "20000", "--seed", "290927", "--output", str(out),
        "--resume", "/sealed/initial_state/model_final.pt", "--resume-response-change",
        "--consumed-updates", "378", "--start-stage", "S1", "--global-actor-update-offset", "0",
        "--resume-learning-rate-scale", str(SCALE)])
    launch.main()  # Plan only: no SSH, file staging or execution.
    plan = json.loads((out / "plan.json").read_text())
    assert plan["execute"] is False
    assert plan["resume_learning_rate_scale"] == SCALE
    assert plan["updates"] == 8500 and plan["consumed_updates"] == 378
    argv = plan["command"][2:]
    argv[argv.index("--contract") + 1] = str(ROOT / "contracts/v6_recovery_curriculum_training_v1.json")
    argv[argv.index("--run-dir") + 1] = str(tmp_path / "full")
    full = load_full_entry()
    monkeypatch.setattr(full.signal, "signal", lambda *_: None)
    calls = []

    def execute(self, command, log_path, training_directory=None):
        calls.append(command)
        training_directory.mkdir()
        (training_directory / "completion.json").write_text(json.dumps({
            "status": "stopped", "successful_updates": 378, "actor_updates_consumed": 378}))
        return 0

    monkeypatch.setattr(full.FullCurriculum, "execute", execute)
    monkeypatch.setattr(full.sys, "argv", argv)
    assert full.main() == 0
    assert len(calls) == 1
    assert float(argument(calls[0], "--resume-learning-rate-scale")) == SCALE
    assert argument(calls[0], "--consumed-updates") == "378"
    assert argument(calls[0], "--updates") == "1000"
    assert argument(calls[0], "--resume") == "/sealed/initial_state/model_final.pt"


def test_launcher_rejects_resume_scale_for_direct_foundation_worker(tmp_path, monkeypatch, capsys):
    launch = entry("launch_v5_remote")
    output = tmp_path / "launch_plan"
    monkeypatch.setattr(launch.sys, "argv", ["launch_v5_remote", "--commit", "HEAD",
        "--contract", "contracts/v5_foundation_v1.json", "--output", str(output),
        "--resume", "/sealed/model_final.pt", "--resume-learning-rate-scale", str(SCALE)])
    with pytest.raises(SystemExit) as caught:
        launch.main()
    assert caught.value.code == 2
    assert "Resume learning-rate scale requires a full curriculum" in capsys.readouterr().err
    assert not output.exists()


def test_explicit_resume_scale_reaches_only_first_worker_and_keeps_ledger(
        tmp_path, monkeypatch):
    first = Campaign(tmp_path, monkeypatch, [{"stand"}], updates=2)
    assert first.run()[0] == 0
    checkpoint = first.args.run_dir / "block_000/model_final.pt"
    second = Campaign(tmp_path, monkeypatch, [{"stand"}, {"stand"}], updates=6,
                      resume=checkpoint, run_name="resumed")
    second.args.resume_learning_rate_scale = SCALE
    second.blocks = second.module.TrainingBlocks(second.args)
    monkeypatch.setattr(second.blocks, "execute", second.execute)
    code, report = second.run()
    assert code == 0 and len(second.calls) == 2
    assert float(argument(second.calls[0], "--learning-rate-scale")) == SCALE
    assert "--learning-rate-scale" not in second.calls[1]
    assert argument(second.calls[0], "--consumed-updates") == "2"
    assert argument(second.calls[1], "--consumed-updates") == "4"
    assert report["successful_updates"] == report["actor_updates_consumed"] == 6
    receipt = json.loads((second.args.run_dir / "resume_learning_rate.json").read_text())
    assert receipt["scale_to_contract"] == SCALE and receipt["block"] == "block_000"
    assert receipt["requested_learning_rate"] == pytest.approx(1e-4 * SCALE)
    assert report["resume_learning_rate_scale"] == SCALE
