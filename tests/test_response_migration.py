"""Response changes must preserve learning semantics and re-test retained skills."""
from copy import deepcopy

import pytest

from wheeled_tasks.chassis.response_migration import (
    RESPONSE_MIGRATION, assess_response_retention, validate_response_resume,
)


def contracts():
    source = {"manual_context35": True, "actor_dim": 35, "critic_dim": 81, "action_dim": 6,
              "physics_dt": .001, "policy_dt": .02, "critic_warmup_updates": 0,
              "total_updates": 1000, "control_math_source": "old.json", "control_math_sha256": "a" * 64,
              "evaluation": {"height_mae_m_max": .005}, "dense_tracking": {"enabled": True}}
    target = deepcopy(source)
    target.update(response_migration=RESPONSE_MIGRATION, actuator_response={"enabled": True, "tau_s": .005})
    return source, target


def control():
    return {"control_id": "old", "actuators": {
        "leg": {"armature": 0., "kp": 160., "kd": 2.5, "effort_limit": 40.},
        "wheel": {"armature": 0., "kd": .6, "gear_ratio": 15.8, "effort_limit": 4.5}},
        "timing": {"physics_dt": .001, "policy_dt": .02}}


def test_response_migration_preserves_full_state_without_claiming_qualification():
    source, target = contracts()
    result = validate_response_resume(source, target)
    assert result["actor_critic_optimizer"] == "preserved"
    assert result["consumed_budget"] == "preserved_no_refund"
    assert result["plant_qualification"] == "requires_separate_bag_and_Isaac_replay_receipt"
    assert result["plant_qualification_evaluated"] is False
    assert result["actuator_response_evidence"] is None


def test_evidence_binding_cannot_self_certify_model_acceptance():
    source, target = contracts()
    evidence = {"file": "reports/actuator_calibration.json", "sha256": "e" * 64}
    target["actuator_response_evidence"] = evidence
    receipt = validate_response_resume(source, target)
    assert receipt["actuator_response_evidence"] == evidence
    assert receipt["plant_qualification_evaluated"] is False
    assert receipt["plant_qualification"] == "requires_separate_bag_and_Isaac_replay_receipt"
    target["actuator_response_evidence"]["passed"] = True
    with pytest.raises(ValueError, match="not a qualification claim"):
        validate_response_resume(source, target)


@pytest.mark.parametrize("key,value", [
    ("actor_dim", 36), ("physics_dt", .005), ("total_updates", 2000),
    ("evaluation", {"height_mae_m_max": .02}), ("dense_tracking", {"enabled": False}),
    ("critic_warmup_updates", 20), ("scene_groups", [{"name": "stand", "fraction": 1.}]),
])
def test_response_migration_rejects_bundled_semantic_changes(key, value):
    source, target = contracts()
    target[key] = value
    with pytest.raises(ValueError):
        validate_response_resume(source, target)


def test_response_migration_requires_an_enabled_changed_model():
    source, target = contracts()
    target["actuator_response"]["enabled"] = False
    with pytest.raises(ValueError, match="explicitly enable"):
        validate_response_resume(source, target)
    target["actuator_response"]["enabled"] = True
    source["actuator_response"] = deepcopy(target["actuator_response"])
    with pytest.raises(ValueError, match="did not change"):
        validate_response_resume(source, target)


def test_control_identity_changes_require_expanded_controls_and_only_allow_armature():
    source, target = contracts()
    target.update(control_math_source="new.json", control_math_sha256="b" * 64)
    before, after = control(), control()
    after["actuators"]["leg"]["armature"] = .025
    with pytest.raises(ValueError, match="both expanded"):
        validate_response_resume(source, target)
    assert validate_response_resume(source, target, source_control=before, target_control=after)[
        "changed_control_fields"] == ["actuators.leg.armature"]
    for path, value in (("kp", 80.), ("effort_limit", 50.)):
        bad = deepcopy(after)
        bad["actuators"]["leg"][path] = value
        with pytest.raises(ValueError, match="control semantics"):
            validate_response_resume(source, target, source_control=before, target_control=bad)
    after["actuators"]["wheel"]["gear_ratio"] = 11.
    with pytest.raises(ValueError, match="control semantics"):
        validate_response_resume(source, target, source_control=before, target_control=after)


def test_armature_content_cannot_change_behind_the_same_control_identity():
    source, target = contracts()
    before, after = control(), control()
    after["actuators"]["wheel"]["armature"] = .02
    with pytest.raises(ValueError, match="frozen contract identity"):
        validate_response_resume(source, target, source_control=before, target_control=after)


def evaluations():
    reports = []
    for index, seed in enumerate((11, 12, 11, 12)):
        cases = {name: {"passed": True, "height_mae_m": .002, "requested_episodes": 4,
                       "checks": {"height": True, "survival": True, "mechanics": True}}
                 for name in ("stand", "forward")}
        reports.append({"status": "evaluated", "deterministic_actor": True, "random_pushes": False,
                        "case_resampling": False, "episodes_per_case": 4,
                        "contract_sha256": ("a" if index < 2 else "b") * 64,
                        "settings": {"seed": seed, "cases": [{"name": name} for name in cases]},
                        "candidates": [{"checkpoint_sha256": "f" * 64, "cases": cases}]})
    return reports


def test_old_pass_is_not_credited_after_new_plant_precision_regression():
    reports = evaluations()
    case = reports[2]["candidates"][0]["cases"]["stand"]
    case.update(passed=False, height_mae_m=.009)
    case["checks"]["height"] = False
    result = assess_response_retention(*reports, required_cases=["stand", "forward"])
    assert result["new_confirmed_case_names"] == ["forward"]
    assert result["response_recovery_case_names"] == ["stand"]
    assert result["precision_not_yet_met"] == {"stand": ["height"]}
    assert result["basic_or_mechanical_check_failures"] == {}
    assert result["extra_rehearsal_required"]
    assert result["old_passes_credited_as_new"] is False
    assert result["metric_comparison"]["stand"]["height_mae_m"]["delta"] == pytest.approx(.0035)


def test_mechanical_failure_is_not_described_as_precision_only():
    reports = evaluations()
    case = reports[3]["candidates"][0]["cases"]["stand"]
    case["passed"] = case["checks"]["mechanics"] = False
    result = assess_response_retention(*reports, required_cases=["stand"])
    assert result["basic_or_mechanical_check_failures"] == {"stand": ["mechanics"]}
    assert result["precision_not_yet_met"] == {}


def test_unchanged_measured_retention_does_not_force_critic_warmup_or_extra_rehearsal():
    result = assess_response_retention(*evaluations(), required_cases=["stand", "forward"])
    assert not result["extra_rehearsal_required"]
    assert result["response_recovery_case_names"] == []


@pytest.mark.parametrize("mutation,error", [
    (lambda r: r[2]["candidates"][0].update(checkpoint_sha256="e" * 64), "same authenticated"),
    (lambda r: r[2]["settings"].update(seed=13), "matching primary"),
    (lambda r: r[3].update(contract_sha256="c" * 64), "plant contract"),
    (lambda r: r[2].update(episodes_per_case=8), "episode count"),
    (lambda r: r[2]["candidates"][0]["cases"].pop("stand"), "coverage"),
    (lambda r: r[2]["settings"].update(height_mae_m_max=.02), "criteria or coverage"),
    (lambda r: r[2]["candidates"][0]["cases"]["stand"].update(passed=False), "inconsistent"),
])
def test_paired_comparison_rejects_noncomparable_or_incomplete_evidence(mutation, error):
    reports = evaluations()
    mutation(reports)
    with pytest.raises(ValueError, match=error):
        assess_response_retention(*reports, required_cases=["stand", "forward"])
