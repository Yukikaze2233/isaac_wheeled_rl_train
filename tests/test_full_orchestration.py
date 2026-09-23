"""Scene handoff respects the selected monitoring/gating and batch-size semantics."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def exercise(tmp_path, monkeypatch, statuses, start_stage=None, monitor=False):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("full_runner_test", ROOT / "scripts/run_full_chassis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    plan = json.loads((ROOT / "contracts/v5_full_curriculum_v2.json").read_text())
    plan["stages"] = plan["stages"][:2]
    if monitor:
        plan["evaluation_mode"] = "monitor"
        plan["stages"][1]["num_envs"] = 32
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"accepted actor")
    checkpoint.with_suffix(".contract.json").write_bytes((ROOT / plan["base_contract"]).read_bytes())
    args = SimpleNamespace(contract=plan_path, run_dir=tmp_path / "run", transfer=checkpoint,
        max_runtime_seconds=60., updates=100, num_envs=64, seed=17, device="cpu", stage="curriculum",
        start_stage=start_stage)
    supervisor = module.FullCurriculum(args)
    calls = []

    def execute(command, log_path, training_directory=None):
        calls.append(command)
        training_directory.mkdir()
        result = {"status": statuses[len(calls) - 1], "successful_updates": 2,
                   "accepted_checkpoint": str(checkpoint)}
        if monitor:
            (training_directory / "model_final.pt").write_bytes(b"latest candidate")
            (training_directory / "contract.json").write_bytes(Path(command[command.index("--contract") + 1]).read_bytes())
        (training_directory / "completion.json").write_text(json.dumps(result))
        return 0

    monkeypatch.setattr(supervisor, "execute", execute)
    assert supervisor.run() == 0
    return supervisor.report, calls


def test_pending_stage_blocks_later_training(tmp_path, monkeypatch):
    report, calls = exercise(tmp_path, monkeypatch, ["budget_exhausted_gate_pending"])
    assert len(calls) == 1
    assert report["status"] == "stage_gate_pending"
    assert report["blocked_stage"] == "foundation"


def test_accepted_policy_is_carried_to_next_stage(tmp_path, monkeypatch):
    report, calls = exercise(tmp_path, monkeypatch, ["stage_accepted", "stage_accepted"])
    assert len(calls) == 2
    assert report["status"] == "full_curriculum_accepted"
    assert report["successful_updates"] == 4
    assert "--transfer" in calls[1]


def test_named_continuation_does_not_retrain_preceding_stages(tmp_path, monkeypatch):
    report, calls = exercise(tmp_path, monkeypatch, ["stage_accepted"], start_stage="speed_1")
    assert len(calls) == 1
    assert report["preceding_stages_not_retrained"] == ["foundation"]
    assert report["start_stage"] == "speed_1"
    assert report["successful_updates"] == 2


def test_monitoring_advances_with_latest_candidate_and_counts_mixed_batch_sizes(tmp_path, monkeypatch):
    report, calls = exercise(tmp_path, monkeypatch, ["training_budget_completed"] * 2, monitor=True)
    assert report["status"] == "full_training_completed"
    assert [command[command.index("--num-envs") + 1] for command in calls] == ["64", "32"]
    assert calls[1][calls[1].index("--transfer") + 1].endswith("stage_00_foundation/model_final.pt")
    assert report["training_transitions"] == 2 * (64 + 32) * 48
    assert not (tmp_path / "run/accepted_policy.pt").exists()


def test_missing_child_progress_does_not_double_count_completed_updates(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_full_chassis import FullCurriculum
    args = SimpleNamespace(contract=ROOT / "contracts/v5_full_curriculum_v2.json", run_dir=tmp_path,
                           transfer=None, max_runtime_seconds=60., num_envs=256)
    supervisor = FullCurriculum(args)
    supervisor.completed_updates = 250
    supervisor.steps_per_env = 48
    old = {"successful_updates": 250, "training_transitions": 3072000}
    (tmp_path / "progress.json").write_text(json.dumps(old))
    supervisor.publish(tmp_path / "not_created_yet")
    assert json.loads((tmp_path / "progress.json").read_text()) == old
