"""Required coverage, frozen-table projection and restart-safe evaluation cadence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_new_asset_blocks import Campaign, contract, module
from test_recovery_curriculum_contract import ROOT, config_for, read
from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
from wheeled_tasks.chassis.preflight import design_preflight


def subset(index=0, role="quick"):
    cfg, seed, _ = config_for(index, role)
    cfg["evaluation"] = select_evaluation_cases(cfg["evaluation"], cfg["evaluation"]["promotion_case_names"])
    cfg = fixed_suite_contract(cfg)
    return cfg, seed, len(cfg["evaluation"]["cases"]) * cfg["evaluation"]["episodes_per_case"]


def validate(cfg, seed, count, role="quick"):
    return design_preflight(cfg, read(cfg["control_math_source"]), ROOT,
        stage=cfg["enabled_stages"][0], role=role, seed=seed, num_envs=count)


@pytest.mark.parametrize("index", range(8))
@pytest.mark.parametrize("role", ["quick", "confirmation"])
def test_all_stage_subsets_project_the_independent_table(index, role):
    cfg, seed, count = subset(index, role)
    assert validate(cfg, seed, count, role)["status"] == "static_passed"


@pytest.mark.parametrize("mutation", ["physics", "usb", "seed", "count", "case", "order", "target"])
def test_subset_does_not_bypass_frozen_fields_or_required_targets(mutation):
    cfg, seed, count = subset()
    if mutation == "physics":
        cfg["physics_dt"] = .005
    elif mutation == "usb":
        cfg["command_transport"]["enabled"] = False
    elif mutation == "seed":
        seed += 1
    elif mutation == "count":
        count += 4
    elif mutation == "case":
        cfg["evaluation"]["cases"][0]["command"][0] += .1
    elif mutation == "order":
        cfg["evaluation"]["cases"].reverse()
    else:
        cfg["evaluation"]["cases"].pop()
    with pytest.raises(ValueError):
        validate(cfg, seed, count)


@pytest.mark.parametrize("role", ["baseline", "full"])
def test_baselines_and_full_audits_cannot_be_subsets(role):
    cfg, seed, count = subset(role=role)
    with pytest.raises(ValueError, match="Only quick/confirmation"):
        validate(cfg, seed, count, role)


def campaign(tmp_path, monkeypatch, outcomes, **kwargs):
    cfg = contract()
    cfg["evaluation"].update(quick_checks=True,
        quick_case_names=[case["name"] for case in cfg["evaluation"]["cases"]])
    (tmp_path / "contract.json").write_text(json.dumps(cfg))
    run = Campaign(tmp_path, monkeypatch, outcomes, **kwargs)
    original = run.execute
    calls = []

    def execute(command, log_path, training_directory=None):
        calls.append(command)
        code = original(command, log_path, training_directory)
        if "--cases" in command:
            names = command[command.index("--cases") + 1:command.index("--no-traces")]
            output = Path(command[command.index("--output") + 1]) / "evaluation.json"
            data = json.loads(output.read_text())
            candidate = data["candidates"][0]
            candidate["cases"] = {name: case for name, case in candidate["cases"].items() if name in names}
            output.write_text(json.dumps(data))
        return code

    monkeypatch.setattr(run.blocks, "execute", execute)
    return run, calls


def test_quick_union_and_confirmation_match_and_full_keeps_catalog(tmp_path, monkeypatch):
    run, calls = campaign(tmp_path, monkeypatch, [{"stand", "jump_06"}], updates=2, protected='["jump_03"]')
    run.blocks.report["response_recovery_case_names"] = ["running_jump_03"]
    run.blocks.contract["curriculum_stage"] = "S4"
    run.frozen = json.dumps(run.blocks.contract, indent=2).encode()
    run.path.write_bytes(run.frozen)
    assert run.run()[0] == 0
    eval_calls = [c for c in calls if "--output" in c]
    assert all("--cases" not in c for c in eval_calls[:2])
    for command in eval_calls[2:]:
        names = command[command.index("--cases") + 1:command.index("--no-traces")]
        assert set(names) == {"stand", "jump_06", "jump_03", "running_jump_03"}


def test_partial_worker_report_is_invalid_not_pass(tmp_path, monkeypatch):
    run, _ = campaign(tmp_path, monkeypatch, [], updates=2)
    original = run.blocks.execute
    def drop_required(command, log, training_directory=None):
        code = original(command, log, training_directory)
        if "--output" in command:
            path = Path(command[command.index("--output") + 1]) / "evaluation.json"
            data = json.loads(path.read_text())
            del data["candidates"][0]["cases"]["stand"]
            path.write_text(json.dumps(data))
        return code
    monkeypatch.setattr(run.blocks, "execute", drop_required)
    code, report = run.run()
    assert code == 1 and "invalid_evaluation_coverage" in report["error"]


def test_completed_review_no_longer_forces_second_seed_or_overwrites_receipt(tmp_path, monkeypatch):
    first, _ = campaign(tmp_path, monkeypatch, [{"stand"}], updates=2)
    assert first.run()[0] == 0
    state_path = first.args.run_dir / "retention_state.json"
    state = json.loads(state_path.read_text())
    state["response_review_start_consumed_updates"] = 0
    state_path.write_text(json.dumps(state))
    (first.args.run_dir / "response_review.json").write_text(json.dumps({
        "confirmation_available": True, "remaining_recovery_case_names": [],
        "updates_consumed_since_model_change": 2}))
    second = Campaign(tmp_path, monkeypatch, [{"stand"}], updates=4,
        resume=first.args.run_dir / "block_000/model_final.pt", run_name="resumed")
    # Reuse the subset-capable worker, leaving the restored bookkeeping real.
    original = second.execute
    def selected(command, log, training_directory=None):
        result = original(command, log, training_directory)
        if "--cases" in command:
            names = command[command.index("--cases") + 1:command.index("--no-traces")]
            path = Path(command[command.index("--output") + 1]) / "evaluation.json"
            data = json.loads(path.read_text())
            data["candidates"][0]["cases"] = {n: c for n, c in data["candidates"][0]["cases"].items() if n in names}
            path.write_text(json.dumps(data))
        return result
    monkeypatch.setattr(second.blocks, "execute", selected)
    code, report = second.run()
    assert code == 0
    assert report["response_review_completed_at_updates"] == 2
    assert [e[0] for e in second.evaluations] == ["evaluation_000"]
    assert report["actor_updates_consumed"] == 4
    assert second.blocks.record_response_review({}, None) is None


def test_pending_review_still_requires_a_real_confirmation(tmp_path, monkeypatch):
    run, _ = campaign(tmp_path, monkeypatch, [{"stand"}], updates=2)
    run.blocks.report["response_review_start_consumed_updates"] = 0
    assert run.run()[0] == 0
    assert run.evaluations[-1][0] == "evaluation_000_confirmation"
    assert run.blocks.report["response_review_completed_at_updates"] == 2


@pytest.mark.parametrize("updates", [2, 200])
def test_ceiling_or_periodic_audit_runs_once_before_confirmation(tmp_path, monkeypatch, updates):
    run, calls = campaign(tmp_path, monkeypatch, [{"stand", "jump_06"}], updates=updates)
    run.blocks.evaluation_settings.update(block_updates=updates, full_audit_at_stage_exit=True)
    assert run.run()[0] == 0
    assert [e[0] for e in run.evaluations] == ["baseline_evaluation", "baseline_confirmation",
        "evaluation_000", "evaluation_000_confirmation"]
    assert all("--cases" not in c for c in calls if "--output" in c)


def test_restart_preserves_adjudicated_regressions_but_not_rolled_back_history(monkeypatch):
    restore = module(monkeypatch).TrainingBlocks.restored_regression_streak
    bad = lambda n: {"successful_updates_total": n, "rollback_trigger_cases": ["stand"], "capability_gate": {}}
    report = {"blocks": [bad(40), bad(80), {"successful_updates_total": 120}]}
    assert restore(report) == 2
    report["recovery_attempts"] = [{"consumed_updates": 80}]
    assert restore(report) == 0
    report["blocks"].append(bad(160))
    assert restore(report) == 1
    report["regression_streak"] = 0
    assert restore(report) == 0


def test_resume_does_not_reset_two_existing_regressions(tmp_path, monkeypatch):
    cfg = contract()
    cfg["evaluation"]["regression_patience"] = 3
    (tmp_path / "contract.json").write_text(json.dumps(cfg))
    first = Campaign(tmp_path, monkeypatch, [set(), set()], updates=4)
    assert first.run()[0] == 0
    assert first.blocks.report["regression_streak"] == 2
    second = Campaign(tmp_path, monkeypatch, [set()], updates=6,
        resume=first.args.run_dir / "block_001/model_final.pt", run_name="resumed")
    code, report = second.run()
    assert code == 0 and report["status"] == "regression_hold_best_preserved"
    assert report["regression_streak"] == 3 and report["actor_updates_consumed"] == 6


def test_s5_quick_and_confirmation_keep_all_airtime_diagnostic_cases(tmp_path, monkeypatch):
    run, calls = campaign(tmp_path, monkeypatch, [{"stand"}], updates=2)
    assert run.run()[0] == 0
    for command in calls:
        if "--cases" in command:
            names = command[command.index("--cases") + 1:command.index("--no-traces")]
            assert {"jump_03", "jump_06", "assisted_10"} <= set(names)
