"""Transport continuation preserves calibrated actuators and old acceptance rules."""
from copy import deepcopy

import pytest

from wheeled_tasks.chassis.response_migration import (
    RESPONSE_MIGRATION,
    TRANSPORT_RESPONSE_MIGRATION,
    assess_response_retention,
    validate_response_resume,
)


def contracts(enabled=False):
    source = {
        "manual_context35": True, "actor_dim": 35, "critic_dim": 81, "action_dim": 6,
        "physics_dt": .001, "policy_dt": .02, "total_updates": 1000,
        "control_math_source": "control.json", "control_math_sha256": "a" * 64,
        "actuator_response": {"enabled": True, "wheel_armature": [.0038, .0039]},
        "command_transport": {"enabled": enabled, "delay_ms": [.04, .16]},
        "response_migration": RESPONSE_MIGRATION,
        "dense_tracking": {"enabled": True},
        "physics_prior_scope": {"g1_release": False, "hardware_release": False},
        "evaluation": {
            "height_mae_m_max": .01,
            "cases": [{"name": "stand", "command": [0., 0., .305]},
                      {"name": "forward", "command": [.5, 0., .305]}],
            "promotion_case_names": ["stand", "forward"],
            "canonical_case_names": ["stand", "forward"],
            "quick_case_names": ["stand", "forward"],
            "retention_case_names": ["stand", "forward"],
            "protected_case_names": ["stand"],
            "initial_passed_case_names": ["stand"],
        },
    }
    target = deepcopy(source)
    target["response_migration"] = TRANSPORT_RESPONSE_MIGRATION
    target["command_transport"].update(delivery_model="phase_aware_hold_v1", drop_probability_max=.005)
    target["evaluation"]["cases"].append({
        "name": "stand__link_loss_005", "command": [0., 0., .305],
        "transport_robustness_case": True, "introduced_stage": "S3", "reset_seed_key": "stand",
    })
    for key in ("canonical_case_names", "quick_case_names", "retention_case_names"):
        target["evaluation"][key].append("stand__link_loss_005")
    return source, target


def control():
    return {"actuators": {"leg": {"armature": 0., "kp": 160., "kd": 2.5},
                          "wheel": {"armature": .0038, "kd": .6}},
            "timing": {"physics_dt": .001, "policy_dt": .02}}


@pytest.mark.parametrize("enabled", [False, True])
def test_transport_migration_preserves_stage_switch_and_model_identity(enabled):
    source, target = contracts(enabled)
    receipt = validate_response_resume(source, target, source_control=control(), target_control=control())
    assert receipt["kind"] == "explicit_transport_response_resume"
    assert receipt["changed_control_fields"] == []
    assert receipt["actuator_model"] == "unchanged"
    assert receipt["actor_critic_optimizer"] == "preserved"
    assert receipt["consumed_budget"] == "preserved_no_refund"
    assert receipt["plant_qualification_evaluated"] is False
    assert receipt["transport_qualification"] == "engineering_prior_not_identified_or_G1_release"
    assert receipt["added_evaluation_case_names"] == ["stand__link_loss_005"]


@pytest.mark.parametrize("key,value", [
    ("actuator_response", {"enabled": True, "wheel_armature": [0., 0.]}),
    ("actuator_response_evidence", {"file": "new.json", "sha256": "f" * 64}),
    ("control_math_source", "new.json"), ("control_math_sha256", "b" * 64),
    ("total_updates", 2000), ("actor_dim", 36), ("physics_dt", .005),
    ("policy_dt", .005), ("dense_tracking", {"enabled": False}),
    ("resume_critic_warmup_updates", 20),
    ("physics_prior_scope", {"g1_release": True, "hardware_release": False}),
])
def test_transport_cannot_bundle_plant_controller_reward_budget_or_release_changes(key, value):
    source, target = contracts()
    target[key] = value
    with pytest.raises(ValueError):
        validate_response_resume(source, target)


@pytest.mark.parametrize("value", [True, 0, None])
def test_transport_cannot_change_or_implicitly_coerce_stage_enabled(value):
    source, target = contracts(False)
    target["command_transport"]["enabled"] = value
    with pytest.raises(ValueError, match="stage enabled"):
        validate_response_resume(source, target)


def test_transport_requires_actual_model_change_and_named_mode():
    source, target = contracts()
    target["command_transport"]["delivery_model"] = "unreviewed"
    with pytest.raises(ValueError, match="phase_aware_hold_v1"):
        validate_response_resume(source, target)
    target["command_transport"]["delivery_model"] = "phase_aware_hold_v1"
    source["command_transport"] = deepcopy(target["command_transport"])
    with pytest.raises(ValueError, match="did not change"):
        validate_response_resume(source, target)


def test_transport_rejects_changed_control_content_behind_same_identity():
    source, target = contracts()
    changed = control()
    changed["actuators"]["wheel"]["armature"] = 0.
    with pytest.raises(ValueError, match="complete control contract"):
        validate_response_resume(source, target, source_control=control(), target_control=changed)


@pytest.mark.parametrize("mutation", [
    lambda x: x["cases"][0].update(command=[0., 0., .31]),
    lambda x: x["cases"].pop(0),
    lambda x: x["cases"].reverse(),
    lambda x: x["cases"].append(deepcopy(x["cases"][-1])),
    lambda x: x["cases"][-1].update(transport_robustness_case=False),
    lambda x: x["cases"][-1].update(reset_seed_key="unknown"),
    lambda x: x.update(height_mae_m_max=.05),
    lambda x: x.update(promotion_case_names=["stand"]),
    lambda x: x.update(quick_case_names=["forward", "stand", "stand__link_loss_005"]),
    lambda x: x.update(protected_case_names=["stand", "stand__link_loss_005"]),
    lambda x: x.update(initial_passed_case_names=["stand", "stand__link_loss_005"]),
])
def test_transport_only_allows_added_communication_cases_and_their_nomination(mutation):
    source, target = contracts()
    mutation(target["evaluation"])
    with pytest.raises(ValueError):
        validate_response_resume(source, target)


def test_new_transport_cases_may_be_added_to_promotion_when_the_stage_requires_them():
    source, target = contracts(True)
    target["evaluation"]["promotion_case_names"].append("stand__link_loss_005")
    validate_response_resume(source, target)


def evaluations():
    source, target = contracts()
    reports = []
    for index, seed in enumerate((11, 12, 11, 12)):
        settings = deepcopy((source if index < 2 else target)["evaluation"])
        settings["seed"] = seed
        cases = {case["name"]: {"passed": True, "height_mae_m": .002, "requested_episodes": 4,
                 "checks": {"height": True, "survival": True, "mechanics": True}}
                 for case in settings["cases"]}
        reports.append({"status": "evaluated", "deterministic_actor": True, "random_pushes": False,
                        "case_resampling": False, "episodes_per_case": 4,
                        "contract_sha256": ("a" if index < 2 else "b") * 64,
                        "coverage": "full_manifest", "evaluated_case_names": list(cases),
                        "settings": settings,
                        "candidates": [{"checkpoint_sha256": "f" * 64, "cases": cases}]})
    return reports


def compare(reports, required=("stand", "forward", "stand__link_loss_005")):
    return assess_response_retention(*reports, required_cases=required,
                                     migration_kind=TRANSPORT_RESPONSE_MIGRATION)


def test_transport_report_compares_common_cases_and_requires_two_new_seeds_for_additions():
    reports = evaluations()
    result = compare(reports)
    assert result["old_confirmed_case_names"] == ["forward", "stand"]
    assert result["added_confirmed_case_names"] == ["stand__link_loss_005"]
    assert result["added_unconfirmed_case_names"] == []
    assert set(result["metric_comparison"]) == {"stand", "forward"}
    assert not result["old_passes_credited_as_new"]
    assert not result["extra_rehearsal_required"]
    case = reports[3]["candidates"][0]["cases"]["stand__link_loss_005"]
    case["passed"] = case["checks"]["height"] = False
    result = compare(reports)
    assert result["extra_rehearsal_required"]
    assert result["response_recovery_case_names"] == []
    assert result["added_confirmed_case_names"] == []
    assert result["added_unconfirmed_case_names"] == ["stand__link_loss_005"]
    assert result["precision_not_yet_met"] == {"stand__link_loss_005": ["height"]}
    # S1/S2 still observe S3 cases, without silently making them early promotion gates.
    assert not compare(reports, ("stand", "forward"))["extra_rehearsal_required"]


def test_old_common_case_regression_still_requires_recovery():
    reports = evaluations()
    case = reports[2]["candidates"][0]["cases"]["stand"]
    case["passed"] = case["checks"]["mechanics"] = False
    result = compare(reports)
    assert result["response_recovery_case_names"] == ["stand"]
    assert result["basic_or_mechanical_check_failures"] == {"stand": ["mechanics"]}
    assert result["extra_rehearsal_required"]


@pytest.mark.parametrize("kind", [None, RESPONSE_MIGRATION, "unknown"])
def test_extended_coverage_is_rejected_without_explicit_transport_kind(kind):
    with pytest.raises(ValueError):
        assess_response_retention(*evaluations(), required_cases=["stand"], migration_kind=kind)


@pytest.mark.parametrize("mutation", [
    lambda r: r[2]["candidates"][0].update(checkpoint_sha256="e" * 64),
    lambda r: r[2]["settings"].update(seed=13),
    lambda r: r[3].update(contract_sha256="c" * 64),
    lambda r: r[2].update(episodes_per_case=8),
    lambda r: r[0].update(coverage="selected_cases"),
    lambda r: r[2].pop("evaluated_case_names"),
    lambda r: r[2]["candidates"][0]["cases"].pop("stand"),
    lambda r: r[3]["candidates"][0]["cases"].pop("stand__link_loss_005"),
    lambda r: r[0]["candidates"][0]["cases"].update(extra=deepcopy(r[0]["candidates"][0]["cases"]["stand"])),
    lambda r: r[2]["settings"].update(height_mae_m_max=.02),
    lambda r: r[2]["candidates"][0]["cases"]["stand"].update(passed=False),
    lambda r: r[2]["candidates"][0]["cases"]["stand"]["checks"].pop("height"),
    lambda r: r[3]["candidates"][0]["cases"]["stand__link_loss_005"]["checks"].pop("height"),
])
def test_transport_reports_reject_missing_cases_changed_criteria_or_unpaired_evidence(mutation):
    reports = evaluations()
    mutation(reports)
    with pytest.raises(ValueError):
        compare(reports)
