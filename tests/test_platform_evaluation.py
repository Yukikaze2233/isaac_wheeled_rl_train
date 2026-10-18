"""Every replica must physically finish the named climb or jump platforms."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.chassis.evaluation import RequestProtocolMetrics, grade_fixed_suite, platform_task_checks


def case(jumping=True, double=True):
    field = "step_jump" if jumping else "body_climb"
    kind = ("double_step_jump" if double else "step_jump") if jumping else "body_climb"
    return {"name": "platform", "command": [.4, 0., .305], "task": "jump" if jumping else "traverse",
            field + "_case": True,
            "skill": {"kind": kind, field: {"rises_m": [.2, .25] if double else [.2]}}}


def episode(jumping=True, double=True, **updates):
    suffix = "landed" if jumping else "supported"
    value = {"enabled": 1, "required_count": 2 if double else 1, "completed_count": 2 if double else 1,
             "first_" + suffix: 1, "second_" + suffix: int(double)}
    if jumping:
        value["requested_count"] = 2 if double else 1
    else:
        value.update(air_peak_s=.02, flight_violation=0)
    return {**value, **updates}


def group(jumping=True, double=True):
    field = "step_jump" if jumping else "body_climb"
    return {"episodes": 2, "timeouts": 0, "successes": 2, "failures": 0, "boundary_truncations": 0,
            "frames": 100, "height_mae_m": 0., "vx_mae_m_s": 0., "yaw_mae_rad_s": 0., "tilt_max_deg": 0.,
            "stand_drift_max_m": 0., "closure_gap_max_m": 0., "termination_reasons": {},
            field + "_episodes": [episode(jumping, double), episode(jumping, double)]}


def settings(c):
    return {"protocol_id": "test", "cases": [c], "survival_rate_min": .95, "height_mae_m_max": .02,
            "velocity_mae_m_s_max": .1, "yaw_mae_rad_s_max": .1, "tilt_max_deg": 20.,
            "stand_drift_m_max": .1, "mechanical_checks": True}


@pytest.mark.parametrize("jumping", [True, False])
@pytest.mark.parametrize("double", [True, False])
def test_grade_accepts_actual_single_and_double_completions(jumping, double):
    c, g = case(jumping, double), group(jumping, double)
    assert grade_fixed_suite({"groups": {c["name"]: g}}, settings(c), 2)["passed"]


@pytest.mark.parametrize("jumping", [True, False])
def test_success_and_high_reward_cannot_replace_each_replica_completion(jumping):
    c, g = case(jumping), group(jumping)
    field = "step_jump" if jumping else "body_climb"
    g["mean_reward"] = 1e9
    g[field + "_episodes"][1]["completed_count"] = 1
    result = grade_fixed_suite({"groups": {c["name"]: g}}, settings(c), 2)
    assert not result["passed"]
    assert not result["cases"][c["name"]]["checks"][field + "_completed_platforms"]


@pytest.mark.parametrize("jumping", [True, False])
def test_missing_or_cross_kind_evidence_and_weakened_required_count_fail(jumping):
    c, g = case(jumping), group(jumping)
    field = "step_jump" if jumping else "body_climb"
    other = "body_climb" if jumping else "step_jump"
    bad = deepcopy(g)
    bad[other + "_episodes"] = bad.pop(field + "_episodes")
    assert not all(platform_task_checks(c, bad, 2).values())
    g[field + "_episodes"][0]["required_count"] = 1
    assert not all(platform_task_checks(c, g, 2).values())
    g[field + "_episodes"] = g[field + "_episodes"][:1]
    assert not all(platform_task_checks(c, g, 2).values())


def test_jump_needs_requests_but_climbing_does_not():
    jump = group()
    jump["step_jump_episodes"][0]["requested_count"] = 1
    assert not platform_task_checks(case(), jump, 2)["step_jump_requests_exercised"]
    assert all(platform_task_checks(case(False), group(False), 2).values())


def test_climb_height_excursion_is_not_a_static_height_precision_failure():
    c, g = case(False), group(False)
    g["height_mae_m"] = .2
    result = grade_fixed_suite({"groups": {c["name"]: g}}, settings(c), 2)
    assert result["passed"] and result["rank_lower_is_better"][-1] == 0.
    g["vx_mae_m_s"] = .2
    assert not grade_fixed_suite({"groups": {c["name"]: g}}, settings(c), 2)["passed"]


def test_climb_historical_completions_cannot_hide_later_flight_violation():
    c, g = case(False), group(False)
    g["body_climb_episodes"][1].update(air_peak_s=.08, flight_violation=1)
    result = grade_fixed_suite({"groups": {c["name"]: g}}, settings(c), 2)
    assert not result["passed"]
    checks = result["cases"][c["name"]]["checks"]
    assert checks["body_climb_completed_platforms"]
    assert not checks["body_climb_no_flight_violation"]


def test_missing_climb_flight_evidence_is_invalid_even_with_all_platforms_complete():
    c, g = case(False), group(False)
    del g["body_climb_episodes"][0]["flight_violation"]
    checks = platform_task_checks(c, g, 2)
    assert not checks["body_climb_episode_evidence"]


@pytest.mark.parametrize("peak", [-.01, float("nan"), float("inf")])
def test_invalid_climb_airtime_evidence_cannot_pass(peak):
    c, g = case(False), group(False)
    g["body_climb_episodes"][0]["air_peak_s"] = peak
    assert not platform_task_checks(c, g, 2)["body_climb_no_flight_violation"]


@pytest.mark.parametrize("jumping", [True, False])
@pytest.mark.parametrize("violation", ["failure", "boundary", "blocked", "knee", "tilt"])
def test_new_completion_does_not_bypass_common_failure_standards(jumping, violation):
    c, g = case(jumping), group(jumping)
    if violation == "failure":
        g["failures"] = 1
    elif violation == "boundary":
        g["boundary_truncations"] = 1
    elif violation == "blocked":
        g["blocked_truncations"] = 1
    elif violation == "knee":
        g["termination_reasons"]["knee"] = 1
    else:
        g["tilt_max_deg"] = 35.
    assert not grade_fixed_suite({"groups": {c["name"]: g}}, settings(c), 2)["passed"]


@pytest.mark.parametrize("jumping", [True, False])
def test_metrics_capture_terminal_diagnostics_without_request_protocol_fields(jumping):
    prefix = "step_jump" if jumping else "body_climb"
    ledger = RequestProtocolMetrics(["platform", "platform", "ordinary"], "cpu", .02)
    row = episode(jumping)
    data = {prefix + "_" + key: torch.tensor([value, value, value]) for key, value in row.items()}
    data[prefix + "_enabled"][2] = 0
    data["done"] = torch.tensor([False, True, True])
    ledger.observe(data, torch.tensor([True, True, True]))
    data["done"] = torch.tensor([True, True, True])
    ledger.observe(data, torch.tensor([True, False, False]))
    metrics = ledger.report_into({"groups": {"platform": {}, "ordinary": {}}})
    measured = metrics["groups"]["platform"][prefix + "_episodes"]
    assert len(measured) == 2
    for actual in measured:
        assert actual == pytest.approx(row)
        if not jumping:
            assert actual["air_peak_s"] > 0  # Preserve seconds, do not cast to integer.
    assert prefix + "_episodes" not in metrics["groups"]["ordinary"]
    assert all(platform_task_checks(case(jumping), {**group(jumping), **metrics["groups"]["platform"]}, 2).values())


def test_flag_does_not_hide_conflicting_kind_or_invalid_platform_specification():
    c, g = case(), group()
    c["skill"]["kind"] = "step_jump"
    assert not all(platform_task_checks(c, g, 2).values())
    c = case()
    c["body_climb_case"] = True
    assert platform_task_checks(c, g, 2) == {"platform_task_kind_unambiguous": False}
