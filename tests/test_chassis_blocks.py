"""Test block handoff and regression protection with mocked simulator processes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def run_blocks(tmp_path, monkeypatch, outcomes):
    spec = importlib.util.spec_from_file_location("chassis_blocks", ROOT / "scripts/run_chassis_blocks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"contract_id": "test", "evaluation": {
        "block_updates": 2, "consecutive_passes_required": 2}}))
    baseline = tmp_path / "old_final.pt"
    baseline.write_bytes(b"verified old actor")
    args = SimpleNamespace(contract=contract, run_dir=tmp_path / "run", transfer=baseline,
        max_runtime_seconds=100., updates=10, stage="foundation", num_envs=8, seed=617,
        device="cpu", publish_state=True)
    blocks = module.TrainingBlocks(args)
    training_calls = []
    evaluations = iter(outcomes)

    def execute(command, log_path, training_directory=None):
        log_path.write_text("mock simulator log\n")
        if training_directory is not None:
            training_calls.append(command)
            training_directory.mkdir()
            count = len(training_calls)
            for name in ("model_final.pt", "policy.onnx", "policy.onnx.json", "agent_config.json"):
                (training_directory / name).write_text(f"block {count}")
            completion = {"status": "completed", "parent_updates": 2 * (count - 1),
                          "successful_updates": 2, "checkpoint_sha256": str(count),
                          "export": {"verified": True}}
            (training_directory / "completion.json").write_text(json.dumps(completion))
        else:
            directory = Path(command[command.index("--output") + 1])
            directory.mkdir()
            passed = True if directory.name == "baseline_evaluation" else next(evaluations)
            evaluation = {"candidates": [{"passed": passed,
                "rank_lower_is_better": [int(not passed), 0., .5 if passed else 2.]}]}
            (directory / "evaluation.json").write_text(json.dumps(evaluation))
        return 0

    monkeypatch.setattr(blocks, "execute", execute)
    assert blocks.run() == 0
    return args.run_dir, blocks.report, training_calls


def test_regression_stops_and_preserves_accepted_initial_actor(tmp_path, monkeypatch):
    directory, report, calls = run_blocks(tmp_path, monkeypatch, [False, False, False])
    assert report["status"] == "regression_hold_best_preserved"
    assert report["successful_updates"] == 6
    assert (directory / "baseline_actor.pt").read_bytes() == b"verified old actor"
    assert not (directory / "model_best.pt").exists()
    assert "--transfer-actor-only" in calls[0]
    assert "--resume" in calls[1] and "--transfer" not in calls[1]


def test_acceptance_requires_consecutive_passes(tmp_path, monkeypatch):
    directory, report, calls = run_blocks(tmp_path, monkeypatch, [True, False, True, True])
    assert report["status"] == "foundation_accepted"
    assert report["consecutive_evaluation_passes"] == 2
    assert len(calls) == 4 and report["successful_updates"] == 8
    # Equal later scores must not replace the first accepted checkpoint.
    assert (directory / "model_best.pt").read_text() == "block 1"
