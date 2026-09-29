"""Recovery full-chain acceptance is per episode and separate from static error."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.chassis.evaluation import (
    RequestProtocolMetrics, grade_fixed_suite, recovery_episode_diagnostics, recovery_task_checks)
from wheeled_tasks.chassis.recovery import RecoveryPhase as P


def case():
    return {"name": "recovery", "command": [.3, 0., .305], "task": "survive", "recovery_case": True,
            "skill": {"kind": "recovery_handoff", "recovery_pose": "supine"}}


def row(**updates):
    return {"enabled": 1, "phase": int(P.RL), "route": 2, "failed": 0, "failure_code": 0,
        "motion_released": 1, "pure_rl": 1, "rl_stable_s": 1., "followup_s": 1., "true_success": 1,
        "handover_count": 1, "ever_blended": 1, "script_seconds": 4., "platform_retained": 1, **updates}


def group():
    return {"episodes": 2, "timeouts": 0, "successes": 2, "failures": 0, "boundary_truncations": 0,
            "frames": 100, "height_mae_m": .3, "vx_mae_m_s": 1., "yaw_mae_rad_s": 1., "tilt_max_deg": 180.,
            "stand_drift_max_m": 2., "closure_gap_max_m": 0., "termination_reasons": {},
            "recovery_episodes": [row(), row()]}


def settings(c):
    return {"protocol_id": "test", "cases": [c], "survival_rate_min": .95, "height_mae_m_max": .02,
            "velocity_mae_m_s_max": .1, "yaw_mae_rad_s_max": .1, "tilt_max_deg": 20.,
            "stand_drift_m_max": .1, "stand_velocity_mae_m_s_max": .05, "mechanical_checks": True}


def grade(g=None, c=None):
    c = c or case()
    return grade_fixed_suite({"groups": {c["name"]: g or group()}}, settings(c), 2)


def test_complete_recovery_is_not_rejected_by_script_pose_or_full_horizon_survival():
    result = grade()
    assert result["passed"] and result["rank_lower_is_better"] == [0, 0., 0.]
    checked = result["cases"]["recovery"]
    assert checked["survival_rate"] == 0. and checked["tilt_max_deg"] == 180.
    assert not {"height", "velocity", "yaw", "tilt", "stand_drift", "survival"} & checked["checks"].keys()
    assert all(item["stage"] == "complete" for item in checked["recovery_episode_diagnostics"])


@pytest.mark.parametrize("updates,failed_check,stage", [
    ({"failed": 1, "phase": int(P.FAILED), "failure_code": 6, "handover_count": 0, "ever_blended": 0}, "no_failure", "script"),
    ({"script_seconds": 8.01, "handover_count": 0}, "script_budget", "blend"),
    ({"handover_count": 0, "phase": int(P.BLEND), "pure_rl": 0}, "blend_and_single_handover", "blend"),
    ({"handover_count": 2}, "blend_and_single_handover", "blend"),
    ({"ever_blended": 0}, "blend_and_single_handover", "script"),
    ({"rl_stable_s": .98}, "continuous_rl_stability", "rl_stabilization"),
    ({"pure_rl": 0}, "pure_rl", "rl_stabilization"),
    ({"motion_released": 0}, "motion_release", "rl_followup"),
    ({"followup_s": .98}, "continuous_followup", "rl_followup"),
    ({"true_success": 0}, "physical_success", "rl_followup"),
    ({"platform_retained": 0}, "platform_retained", "rl_followup"),
])
def test_one_good_replica_never_masks_missing_stage_in_another(updates, failed_check, stage):
    g = group()
    g["recovery_episodes"][1].update(updates)
    result = grade(g)
    assert not result["passed"]
    assert not result["cases"]["recovery"]["checks"]["recovery_" + failed_check]
    assert result["cases"]["recovery"]["recovery_episode_diagnostics"][1]["stage"] == stage


def test_mechanical_limits_and_physical_truncations_still_apply_to_recovery():
    for key, value in (("closure_gap_max_m", .004), ("failures", 1), ("boundary_truncations", 1),
                       ("blocked_truncations", 1), ("successes", 1), ("episodes", 1), ("frames", 0)):
        g = group()
        g[key] = value
        assert not grade(g)["passed"], key


@pytest.mark.parametrize("bad", [[], [row()], [row(), {}], [row(), row(script_seconds=float("nan"))],
    [row(), row(rl_stable_s="1")], [row(), row(phase=999)], [row(), row(handover_count=1.5)], None])
def test_absent_nonfinite_or_invalid_evidence_cannot_pass(bad):
    g = group()
    g["recovery_episodes"] = bad
    assert not grade(g)["passed"]


def test_ordinary_cases_retain_their_existing_static_precision_gates():
    c = {"name": "ordinary", "command": [0., 0., .305]}
    g = group()
    g.update(timeouts=2, successes=0)
    result = grade(g, c)
    assert not result["passed"]
    assert not result["cases"]["ordinary"]["checks"]["height"]
    assert "recovery_episode_diagnostics" not in result["cases"]["ordinary"]


def test_ledger_collects_only_done_active_recovery_lanes_without_jump_protocol():
    ledger = RequestProtocolMetrics(["recovery", "recovery", "ordinary"], "cpu", .02)
    rows = [row(), row(rl_stable_s=.5, true_success=0), row(enabled=0)]
    data = {"recovery_" + key: torch.tensor([item[key] for item in rows]) for key in ledger.RECOVERY_COUNTERS}
    data["done"] = torch.tensor([True, False, True])
    ledger.observe(data, torch.ones(3, dtype=torch.bool))
    assert len(ledger.recovery_episodes["recovery"]) == 1
    data["done"] = torch.ones(3, dtype=torch.bool)
    ledger.observe(data, torch.tensor([False, True, True]))
    metrics = ledger.report_into({"groups": {"recovery": {}, "ordinary": {}}})
    assert len(metrics["groups"]["recovery"]["recovery_episodes"]) == 2
    assert metrics["groups"]["recovery"]["recovery_episodes"][1]["rl_stable_s"] == .5
    assert "recovery_episodes" not in metrics["groups"]["ordinary"]
    ledger.recovery_episodes["recovery"].clear()
    assert len(metrics["groups"]["recovery"]["recovery_episodes"]) == 2  # Sealed copy.


def test_incomplete_ledger_fields_fail_closed_without_filling_success():
    ledger = RequestProtocolMetrics(["recovery"], "cpu", .02)
    data = {"done": torch.tensor([True]), "recovery_enabled": torch.tensor([True])}
    ledger.observe(data, torch.tensor([True]))
    g = group()
    del g["recovery_episodes"]
    result = ledger.report_into({"groups": {"recovery": g}})
    assert not recovery_task_checks(case(), result["groups"]["recovery"], 2)["recovery_episode_evidence"]
    assert all(item["stage"] == "evidence" for item in recovery_episode_diagnostics(g, 2))


def test_explicit_recovery_specification_is_required():
    for mutate in (lambda c: c.pop("recovery_case"), lambda c: c["skill"].pop("recovery_pose"),
                   lambda c: c["skill"].update(kind="body_climb")):
        c = deepcopy(case())
        mutate(c)
        assert not recovery_task_checks(c, group(), 2)["recovery_specification"]
