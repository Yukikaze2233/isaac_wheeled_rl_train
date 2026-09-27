"""A budget extension is not permission to change the controller or hide missing cases."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis.evaluation import capability_gate, severe_retention_regressions
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.policy_transfer import validate_budget_resume

ROOT = Path(__file__).resolve().parents[1]


def first_stage(name):
    load = lambda path: json.loads((ROOT / path).read_text())
    plan = resolve_plan(load(name), load)
    return stage_contract(load(plan["base_contract"]), plan, plan["stages"][0], 16384)


def test_only_budget_and_recovery_scheduling_may_change():
    source = first_stage("contracts/v6_scut35_precision_v1.json")
    target = first_stage("contracts/v6_scut35_budget_retry_v1.json")
    receipt = validate_budget_resume(source, target)
    assert (source["total_updates"], target["total_updates"]) == (500, 800)
    assert receipt["acceptance"] == "unchanged"
    for key, value in (("precision_tracking", {}), ("actor_dim", 36), ("physics_dt", .001)):
        changed = deepcopy(target)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_budget_resume(source, changed)
    for key, value in (("height_mae_m_max", .02), ("seed", 123), ("cases", [])):
        changed = deepcopy(target)
        changed["evaluation"][key] = value
        with pytest.raises(ValueError, match="acceptance"):
            validate_budget_resume(source, changed)


def test_minor_regression_blocks_promotion_without_forcing_rollback():
    baseline = {"cases": {"rotate": {"passed": True}}, "rank_lower_is_better": [0, 0, 0]}
    settings = {"cases": [{"name": "rotate"}], "promotion_case_names": ["rotate"],
                "retention_case_names": ["rotate"], "protected_case_names": ["rotate"],
                "height_mae_m_max": .01, "regression_recovery": {"trigger_error_ratio": 2.}}
    candidate = {"passed": False, "rank_lower_is_better": [1, 0, 1],
                 "cases": {"rotate": {"passed": False, "height_mae_m": .014,
                                      "checks": {"mechanics": True, "height": False}}}}
    gate = capability_gate(candidate, baseline, settings)
    assert not gate["passed"] and gate["lost_parent_passes"] == ["rotate"]
    assert severe_retention_regressions(candidate, settings, gate["lost_parent_passes"]) == []
    candidate["cases"]["rotate"]["height_mae_m"] = .025
    assert severe_retention_regressions(candidate, settings, ["rotate"]) == ["rotate"]
    candidate["cases"]["rotate"].update(height_mae_m=.014, checks={"mechanics": False, "height": False})
    assert severe_retention_regressions(candidate, settings, ["rotate"]) == ["rotate"]


def test_missing_protected_case_is_not_silently_accepted():
    baseline = {"cases": {"stand": {"passed": True}, "stand_usb": {"passed": True}},
                "rank_lower_is_better": [0, 0, 0]}
    candidate = {"passed": True, "rank_lower_is_better": [0, 0, 0],
                 "cases": {"stand": {"passed": True, "checks": {"mechanics": True}}}}
    settings = {"promotion_case_names": ["stand"], "retention_case_names": ["stand", "stand_usb"],
                "protected_case_names": ["stand_usb"]}
    gate = capability_gate(candidate, baseline, settings)
    assert not gate["passed"]
    assert gate["missing_required_cases"] == ["stand_usb"]
    assert gate["lost_parent_passes"] == ["stand_usb"]


def test_interrupted_cross_budget_rollback_uses_frozen_contract_not_live_selection(tmp_path):
    spec = importlib.util.spec_from_file_location("rollback_contract_test", ROOT / "scripts/run_chassis_blocks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = first_stage("contracts/v6_scut35_precision_v1.json")
    target = first_stage("contracts/v6_scut35_budget_retry_v1.json")
    source_dir = tmp_path / "retained"
    source_dir.mkdir()
    parent = source_dir / "model_final.pt"
    parent.write_bytes(b"verified full learning state")
    (source_dir / "contract.json").write_text(json.dumps(source))
    contract = tmp_path / "target.json"
    contract.write_text(json.dumps(target))
    args = SimpleNamespace(contract=contract, run_dir=tmp_path, transfer=None, resume=tmp_path / "model600.pt",
                           max_runtime_seconds=60., updates=800, resume_budget_change=True)
    blocks = module.TrainingBlocks(args)
    settings = blocks.contract["evaluation"]
    # Real evaluation narrows this field and adds inherited clock-recovery goals.
    settings["retention_case_names"] = ["stand", "rotate_1"]
    settings["promotion_case_names"] += ["rotate_8"]
    settings["initial_passed_case_names"] = ["rotate_1"]
    blocks.report.update(successful_updates=600, resumed_updates=600,
                         retention_learning_checkpoint=str(parent), learning_rate_scale=.5,
                         recovery_attempts=[{"consumed_updates": 400}])
    previous = {"status": "failed", "blocks": [
        {"successful_updates_total": step, "rollback_trigger_cases": ["rotate_1"],
         "capability_gate": {"lost_parent_passes": ["rotate_1"]}} for step in (550, 600)]}
    assert blocks.pending_regression_recovery(settings, previous) == parent
    assert blocks.report["successful_updates"] == 600
    assert blocks.report["learning_rate_scale"] == .25
    assert len(blocks.report["recovery_attempts"]) == 2
    assert blocks.report["completed_interrupted_recovery"]
    assert settings["retention_case_names"] == ["stand", "rotate_1"]
    assert "rotate_8" in settings["promotion_case_names"]
    # A different checkpoint or a normal completion cannot replay that decision.
    previous["status"] = "completed"
    assert blocks.pending_regression_recovery(settings, previous) is None
