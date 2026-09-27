"""Test block handoff and regression protection with mocked simulator processes."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("count", [12288, 16384])
def test_block_cli_accepts_supported_large_batches(tmp_path, monkeypatch, count):
    spec = importlib.util.spec_from_file_location("large_batch_blocks", ROOT / "scripts/run_chassis_blocks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"contract_id": "test"}))
    monkeypatch.setattr(sys, "argv", ["blocks", "--contract", str(contract), "--run-dir", str(tmp_path / "run"),
        "--num-envs", str(count), "--updates", "10", "--research"])
    counts = []
    monkeypatch.setattr(module.TrainingBlocks, "run", lambda self: counts.append(self.args.num_envs) or 0)
    assert module.main() == 0
    assert counts == [count]


def run_blocks(tmp_path, monkeypatch, outcomes, transfer_critic=False, resumed_updates=0, worker_source=None,
               evaluation_settings=None, baseline_passed=True, baseline_anchor_passed=False, expected_code=0,
               baseline_cases=None):
    spec = importlib.util.spec_from_file_location("chassis_blocks", ROOT / "scripts/run_chassis_blocks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"contract_id": "test", "transfer_critic": transfer_critic,
        "learning_rate": 3e-5, "num_steps_per_env": 24, "evaluation": {
        "block_updates": 2, "consecutive_passes_required": 2, **(evaluation_settings or {})}}))
    baseline = tmp_path / "old_final.pt"
    baseline.write_bytes(b"verified old actor")
    args = SimpleNamespace(contract=contract, run_dir=tmp_path / "run", transfer=baseline,
        max_runtime_seconds=100., updates=10, stage="foundation", num_envs=8, seed=617,
        device="cpu", publish_state=True)
    args.worker_source = worker_source
    if resumed_updates:
        sealed = tmp_path / "sealed"
        sealed.mkdir()
        (sealed / "completion.json").write_text(json.dumps({"status": "checkpoint_sealed",
                                                          "successful_updates": resumed_updates}))
        args.resume = sealed / "model.pt"
        args.resume.write_bytes(b"actor critic optimizer")
        args.transfer = None
    blocks = module.TrainingBlocks(args)
    training_calls = []
    evaluations = iter(outcomes)

    def execute(command, log_path, training_directory=None):
        if worker_source is not None:
            assert Path(command[2]).parent.parent == worker_source.resolve()
        log_path.write_text("mock simulator log\n")
        if training_directory is not None:
            training_calls.append(command)
            training_directory.mkdir()
            count = len(training_calls)
            for name in ("model_final.pt", "policy.onnx", "policy.onnx.json", "agent_config.json"):
                (training_directory / name).write_text(f"block {count}")
            (training_directory / "contract.json").write_bytes(contract.read_bytes())
            initial = training_directory / "initial_state"
            initial.mkdir()
            (initial / "model_final.pt").write_bytes(b"initial full learning state")
            (initial / "contract.json").write_bytes(contract.read_bytes())
            scale = float(command[command.index("--learning-rate-scale") + 1]) if "--learning-rate-scale" in command else 1.
            parent_updates = (int(command[command.index("--consumed-updates") + 1])
                              if "--consumed-updates" in command else resumed_updates + 2 * (count - 1))
            completion = {"status": "completed", "parent_updates": parent_updates, "learning_rate": 3e-5 * scale,
                          "successful_updates": 2, "checkpoint_sha256": str(count),
                          "export": {"verified": True}}
            (training_directory / "completion.json").write_text(json.dumps(completion))
        else:
            directory = Path(command[command.index("--output") + 1])
            directory.mkdir()
            outcome = ({"passed": baseline_passed, "anchor_passed": baseline_anchor_passed}
                       if directory.name == "baseline_evaluation" else next(evaluations))
            if isinstance(outcome, bool):
                outcome = {"passed": outcome}
            passed = outcome["passed"]
            if directory.name == "baseline_evaluation" and baseline_cases is not None:
                outcome["cases"] = baseline_cases
            if "--cases" in command and "cases" in outcome:
                selected = command[command.index("--cases") + 1:]
                selected = selected[:next((i for i, value in enumerate(selected) if value.startswith("--")), len(selected))]
                outcome = {**outcome, "cases": {k: v for k, v in outcome["cases"].items() if k in selected}}
                passed = all(case["passed"] for case in outcome["cases"].values())
                outcome["passed"] = passed
            evaluation = {"candidates": [{**outcome,
                "rank_lower_is_better": [int(not passed), 0., .5 if passed else 2.]}]}
            (directory / "evaluation.json").write_text(json.dumps(evaluation))
        return 0

    monkeypatch.setattr(blocks, "execute", execute)
    assert blocks.run() == expected_code
    return args.run_dir, blocks.report, training_calls


def test_capability_gate_accepts_required_cases_without_claiming_untrained_catalog(tmp_path, monkeypatch):
    baseline = {name: {"passed": name == "stand", "checks": {"mechanics": True}}
                for name in ("stand", "forward", "fast")}
    candidate = {"passed": False, "cases": {
        name: {"passed": name != "fast", "checks": {"mechanics": True}} for name in baseline}}
    directory, report, calls = run_blocks(tmp_path, monkeypatch, [candidate], baseline_passed=False,
        baseline_cases=baseline, evaluation_settings={"mode": "gate", "promotion_case_names": ["stand", "forward"],
            "retention_case_names": list(baseline), "continuation_selection": "retain_parent_passes_then_rank",
            "consecutive_passes_required": 1})
    assert report["status"] == "foundation_accepted" and len(calls) == 1
    assert (directory / "model_best.pt").read_text() == "block 1"
    assert not report["blocks"][0]["evaluation_passed"]
    assert report["blocks"][0]["capability_gate"]["passed"]


def test_capability_gate_stops_after_two_actual_retention_regressions(tmp_path, monkeypatch):
    baseline = {"stand": {"passed": True, "checks": {"mechanics": True}}}
    failed = {"passed": False, "cases": {"stand": {"passed": False, "checks": {"mechanics": True}}}}
    _, report, calls = run_blocks(tmp_path, monkeypatch, [failed, failed], baseline_passed=False,
        baseline_cases=baseline, evaluation_settings={"mode": "gate", "promotion_case_names": ["stand"],
            "retention_case_names": ["stand"], "continuation_selection": "retain_parent_passes_then_rank",
            "regression_patience": 2})
    assert report["status"] == "regression_hold_best_preserved" and len(calls) == 2


@pytest.mark.parametrize("recovers", [False, True])
def test_bounded_full_state_rollback_keeps_spent_budget_and_publishes_terminal(tmp_path, monkeypatch, recovers):
    baseline = {"stand": {"passed": True, "checks": {"mechanics": True}},
                "forward": {"passed": False, "checks": {"mechanics": True}}}
    failed = {"passed": False, "cases": {name: {"passed": False, "checks": {"mechanics": True}} for name in baseline}}
    good = {"passed": True, "cases": {name: {"passed": True, "checks": {"mechanics": True}} for name in baseline}}
    outcomes = [failed, failed, good] if recovers else [failed] * 4
    directory, report, calls = run_blocks(tmp_path, monkeypatch, outcomes, baseline_passed=False,
        baseline_cases=baseline, evaluation_settings={"promotion_case_names": ["stand", "forward"],
            "retention_case_names": ["stand", "forward"],
            "regression_patience": 2, "consecutive_passes_required": 1,
            "regression_recovery": {"max_retries": 1, "learning_rate_factor": .5}})
    assert len(report["recovery_attempts"]) == 1
    retry = calls[2]
    assert retry[retry.index("--resume") + 1].endswith("block_000/initial_state/model_final.pt")
    assert retry[retry.index("--consumed-updates") + 1] == "4"
    assert retry[retry.index("--learning-rate-scale") + 1] == "0.5"
    assert report["successful_updates"] == (6 if recovers else 8)
    assert report["status"] == ("foundation_accepted" if recovers else "regression_hold_best_preserved")
    progress = json.loads((directory / "progress.json").read_text())
    assert progress["phase"] == "finished" and progress["status"] == report["status"]
    assert progress["worker_pid"] is None


def test_quick_checks_do_not_confirm_every_unqualified_block(tmp_path, monkeypatch):
    baseline = {name: {"passed": name == "stand", "checks": {"mechanics": True}}
                for name in ("stand", "forward", "future")}
    unchanged = {"passed": False, "cases": baseline}
    settings = {"mode": "gate", "quick_checks": True, "confirmation_on_candidate_or_regression": True,
                "cumulative_retention": True, "confirmation_seed": 19,
                "quick_case_names": ["stand"], "promotion_case_names": ["stand", "forward"],
                "retention_case_names": list(baseline), "cases": [{"name": n} for n in baseline]}
    directory, report, calls = run_blocks(tmp_path, monkeypatch, [unchanged] * 6, baseline_passed=False,
        baseline_cases=baseline, evaluation_settings=settings)
    assert len(calls) == 5 and report["status"] == "budget_exhausted_gate_pending"
    assert not list(directory.glob("evaluation_*_confirmation"))
    assert set(report["baseline_evaluation"]["cases"]) == {"stand", "forward"}


def test_quick_promotion_requires_full_exit_audit_and_retains_new_cases(tmp_path, monkeypatch):
    baseline = {name: {"passed": name == "stand", "checks": {"mechanics": True}}
                for name in ("stand", "forward", "future")}
    good = {"passed": True, "cases": {name: {"passed": True, "checks": {"mechanics": True}} for name in baseline}}
    settings = {"mode": "gate", "quick_checks": True, "confirmation_on_candidate_or_regression": True,
                "full_audit_at_stage_exit": True, "cumulative_retention": True, "confirmation_seed": 19,
                "consecutive_passes_required": 1, "quick_case_names": ["stand"],
                "promotion_case_names": ["stand", "forward"], "retention_case_names": list(baseline),
                "cases": [{"name": n} for n in baseline]}
    directory, report, calls = run_blocks(tmp_path, monkeypatch,
        [{"passed": False, "cases": baseline}, good, good, good], baseline_passed=False,
        baseline_cases=baseline, evaluation_settings=settings)
    assert report["status"] == "foundation_accepted" and len(calls) == 1
    assert report["blocks"][0]["full_evaluation"] == "evaluation_000_full"
    assert report["protected_case_names"] == ["forward", "future", "stand"]
    assert (directory / "evaluation_000_full/evaluation.json").exists()


def test_quick_checks_confirm_and_protect_new_skills_before_stage_promotion(tmp_path, monkeypatch):
    baseline = {name: {"passed": name == "stand", "checks": {"mechanics": True}}
                for name in ("stand", "forward", "fast")}
    unchanged = {"passed": False, "cases": baseline}
    gained = {"passed": False, "cases": {**baseline, "fast": {"passed": True, "checks": {"mechanics": True}}}}
    settings = {"mode": "gate", "quick_checks": True, "confirmation_on_candidate_or_regression": True,
                "cumulative_retention": True, "confirmation_seed": 19, "regression_patience": 2,
                "quick_case_names": ["stand", "fast"], "promotion_case_names": ["stand", "forward"],
                "retention_case_names": list(baseline), "cases": [{"name": n} for n in baseline]}
    _, report, calls = run_blocks(tmp_path, monkeypatch,
        [unchanged, gained, gained, unchanged, unchanged, unchanged, unchanged],
        baseline_passed=False, baseline_cases=baseline, evaluation_settings=settings)
    assert len(calls) == 3 and report["status"] == "regression_hold_best_preserved"
    assert "fast" in report["protected_case_names"]
    assert report["blocks"][0]["new_primary_passes"] == ["fast"]


def test_recovery_uses_last_confirmed_learning_state_including_new_protection(tmp_path, monkeypatch):
    baseline = {name: {"passed": name == "stand", "checks": {"mechanics": True}}
                for name in ("stand", "forward", "fast")}
    unchanged = {"passed": False, "cases": baseline}
    gained = {"passed": False, "cases": {**baseline, "fast": {"passed": True, "checks": {"mechanics": True}}}}
    good = {"passed": True, "cases": {name: {"passed": True, "checks": {"mechanics": True}} for name in baseline}}
    settings = {"mode": "gate", "cumulative_retention": True, "confirmation_seed": 19,
                "regression_patience": 2, "consecutive_passes_required": 1,
                "promotion_case_names": ["stand", "forward"], "retention_case_names": list(baseline),
                "regression_recovery": {"max_retries": 1, "learning_rate_factor": .5}}
    directory, report, calls = run_blocks(tmp_path, monkeypatch,
        [unchanged, gained, gained, unchanged, unchanged, unchanged, unchanged, good, good],
        baseline_passed=False, baseline_cases=baseline, evaluation_settings=settings)
    assert report["status"] == "foundation_accepted" and report["successful_updates"] == 8
    retry = calls[3]
    assert retry[retry.index("--resume") + 1].endswith("block_000/model_final.pt")
    assert retry[retry.index("--consumed-updates") + 1] == "6"
    saved = json.loads((directory / "retention_state.json").read_text())
    assert saved["retention_learning_checkpoint"].endswith("block_003/model_final.pt")
    assert saved["protected_case_names"] == ["fast", "forward", "stand"]


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


def test_compatible_network_transfer_keeps_the_critic(tmp_path, monkeypatch):
    _, report, calls = run_blocks(tmp_path, monkeypatch, [True, True], transfer_critic=True)
    assert report["status"] == "foundation_accepted"
    assert "--transfer" in calls[0]
    assert "--transfer-actor-only" not in calls[0]
    assert "--resume" in calls[1]


def test_sealed_resume_continues_optimizer_and_counts_prior_updates(tmp_path, monkeypatch):
    _, report, calls = run_blocks(tmp_path, monkeypatch, [True, True], resumed_updates=4)
    assert report["successful_updates"] == 8
    assert report["resumed_updates"] == 4
    assert all("--resume" in command and "--transfer" not in command for command in calls)


def test_incomplete_baseline_still_protects_learned_cases(tmp_path, monkeypatch):
    failed = {"passed": False, "anchor_passed": False, "cases": {"stand": {"anchor": True, "passed": False}}}
    _, report, calls = run_blocks(tmp_path, monkeypatch, [failed, failed],
        baseline_passed=False, baseline_anchor_passed=True,
        evaluation_settings={"require_passing_anchors": True, "regression_patience": 2, "minimum_updates": 1000})
    assert report["status"] == "regression_hold_best_preserved"
    assert len(calls) == 2 and report["successful_updates"] == 4
    assert all(block["failed_anchor_cases"] == ["stand"] for block in report["blocks"])


def test_required_anchors_survive_optimizer_resume_without_a_new_baseline(tmp_path, monkeypatch):
    _, report, calls = run_blocks(tmp_path, monkeypatch, [False, False], resumed_updates=4,
        evaluation_settings={"require_passing_anchors": True, "regression_patience": 2})
    assert report["status"] == "regression_hold_best_preserved"
    assert report["successful_updates"] == 8 and len(calls) == 2


def test_unlearned_cases_do_not_stop_training_while_anchors_still_pass(tmp_path, monkeypatch):
    _, report, calls = run_blocks(tmp_path, monkeypatch, [{"passed": False, "anchor_passed": True}] * 5,
        baseline_passed=False, baseline_anchor_passed=True,
        evaluation_settings={"require_passing_anchors": True, "regression_patience": 2})
    assert report["status"] == "budget_exhausted_gate_pending"
    assert len(calls) == 5


def test_required_initial_anchors_are_verified_before_any_update(tmp_path, monkeypatch):
    _, report, calls = run_blocks(tmp_path, monkeypatch, [], baseline_passed=False,
        evaluation_settings={"require_passing_anchors": True}, expected_code=1)
    assert report["status"] == "failed" and not calls
    assert "Initial actor does not pass" in report["error"]


@pytest.mark.parametrize("passed", [False, True])
def test_monitoring_neither_stops_on_regression_nor_skips_training_on_success(tmp_path, monkeypatch, passed):
    directory, report, calls = run_blocks(tmp_path, monkeypatch, [passed] * 5,
        baseline_passed=passed, baseline_anchor_passed=False,
        evaluation_settings={"mode": "monitor", "require_passing_anchors": True,
                             "skip_training_if_initially_accepted": True, "regression_patience": 1})
    assert report["status"] == "training_budget_completed"
    assert len(calls) == 5 and report["successful_updates"] == 10
    assert "accepted_checkpoint" not in report
    selection = json.loads((directory / "artifact_selection.json").read_text())
    assert not selection["latest_is_accepted"]


def test_rollback_selects_one_frozen_worker_source_for_training_and_evaluation(tmp_path, monkeypatch):
    _, report, _ = run_blocks(tmp_path, monkeypatch, [True, True],
                              resumed_updates=4, worker_source=tmp_path / "reference")
    assert report["worker_source"] == str(tmp_path / "reference")


@pytest.mark.parametrize("confirmation_passed", [True, False])
def test_initial_actor_skip_requires_confirmation_and_preserves_source_contract(tmp_path, monkeypatch, confirmation_passed):
    spec = importlib.util.spec_from_file_location("chassis_skip", ROOT / "scripts/run_chassis_blocks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"contract_id": "test", "curriculum_stage": "stand", "evaluation": {
        "skip_training_if_initially_accepted": True, "confirmation_seed": 19}}))
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"accepted actor")
    actor.with_suffix(".contract.json").write_text('{"source": "original"}')
    args = SimpleNamespace(contract=contract, run_dir=tmp_path / "run", transfer=actor,
        max_runtime_seconds=100., updates=0, stage="foundation", num_envs=8, seed=17,
        device="cpu", publish_state=False)
    blocks = module.TrainingBlocks(args)
    calls = []

    def execute(command, log_path, training_directory=None):
        assert training_directory is None
        calls.append(command)
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir()
        for name in ("policy.onnx", "policy.onnx.json", "policy.onnx.contract.json"):
            (directory / name).write_text("mock export")
        passed = len(calls) == 1 or confirmation_passed
        candidate = {"passed": passed, "anchor_passed": passed, "checkpoint_sha256": "test",
                     "rank_lower_is_better": [int(not passed), 0., 0.],
                     "export_directory": str(directory), "export": {"verified": True}}
        (directory / "evaluation.json").write_text(json.dumps({"candidates": [candidate]}))
        return 0

    monkeypatch.setattr(blocks, "execute", execute)
    assert blocks.run() == 0
    assert len(calls) == 2 and "--seed" in calls[1]
    assert (blocks.report["status"] == "stage_accepted") == confirmation_passed
    if confirmation_passed:
        assert (args.run_dir / "model_final.contract.json").read_text() == '{"source": "original"}'
