"""A budget extension is not permission to change the controller or hide missing cases."""
from copy import deepcopy
import json
from pathlib import Path

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
