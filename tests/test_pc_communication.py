"""Pure independent declarations, fixed fault overrides and preserved exams."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.pc_communication import (
    FEEDBACK_CONSUMERS, apply_pc_communication, command_profiles, command_transport_settings,
    feedback_profile_enablement, feedback_transport_settings,
    retire_legacy_fault_cases, runtime_expectation,
    validate_no_loss_projection, validate_pc_communication,
)


ROOT = Path(__file__).resolve().parents[1]
BASES = ("stand_305mm", "forward_05", "backward_05", "rotate_1")
FAULTS = {
    "link_loss_005": {"delay_ms": .08, "can_phase_ms": .5, "drop_probability": .005},
    "link_burst_3": {"delay_ms": .08, "can_phase_ms": .5,
                     "burst_every_packets": 2000, "burst_packets": 3},
    "link_combined": {"delay_ms": .08, "can_phase_ms": .08, "jitter_ms": .02,
                      "jitter_correlation": .9, "drop_probability": .005,
                      "burst_every_packets": 5000, "burst_packets": 3},
}
RETIRED = [f"{base}__{suffix}" for suffix in FAULTS for base in BASES]


def feedback(enabled=True):
    return {
        "version": "complete_one_way_feedback_v1", "enabled": enabled,
        "units": "seconds", "scope": "engineering_prior_complete_sensor_to_host_path_not_measured_rtt",
        "ordered_delivery": "monotonic_sensor_timestamp", "reconstruction": "linear_physics_history",
        "delay_range_s": [.00004, .015], "enabled_fraction": .75,
        "jitter_s": .00002, "jitter_correlation": .9,
    }


def command(enabled=True):
    return {
        "version": "complete_one_way_command_v1", "enabled": enabled,
        "delay_ms": [.04, 14.], "can_period_ms": 1., "enabled_fraction": .75,
        "drop_probability_max": 0., "burst_probability": 0., "burst_packets": [2, 3],
        "scope": "engineering_prior_complete_host_to_drive_path_not_measured_rtt;"
                 "USB_arrival_plus_CAN_phase;substep_torque_impulse_approximation",
        "delivery_model": "phase_aware_hold_v1", "can_phase_ms": [0., 1.],
        "jitter_ms": .02, "jitter_correlation": .9,
    }


def clean(enabled=True):
    return {
        "physics_dt": .005, "policy_dt": .02, "feedback_transport": feedback(enabled),
        "command_transport": command(enabled),
        "signal_delay": {"version": "physical_signal_delay_v1", "enabled": False,
                         "profiles": [{"observation_ms": [40., 80.], "action_ms": [30., 60.]}]},
        "usb_transport": {"enabled": False},
        "signal_perturbations": {"enabled": True, "max_delay_steps": 0, "noise_scale": 1.},
        "evaluation_exact_cases": True,
        "evaluation": {"cases": [{"name": "stand_305mm", "command": [0., 0., .305]}],
            "frozen_module_enablement": {"dynamics_randomization": False, "contact_domain": False,
                "signal_perturbations": False, "signal_delay": False,
                "command_transport": enabled, "feedback_transport": enabled}},
    }


def catalogue():
    cases = [{"name": f"retained_{index:03d}", "command": [0., 0., .305],
              "skill": {"kind": "constant", "cohort": index}, "height_mae_m_max": .005}
             for index in range(310)]
    cases[0]["communication_profile"] = {"delay_ms": .08, "can_phase_ms": .08,
        "jitter_ms": .02, "jitter_correlation": .9}
    for suffix, profile in FAULTS.items():
        for base in BASES:
            cases.append({"name": f"{base}__{suffix}", "communication_profile": deepcopy(profile),
                          "command": [0., 0., .305], "skill": {"kind": "constant"}})
    names = [case["name"] for case in cases]
    return dict(cases=cases, canonical_case_names=names, quick_case_names=names[:2] + RETIRED,
        promotion_case_names=names, protected_case_names=names, retention_case_names=names,
        frozen_module_enablement=dict(command_transport=True, signal_delay=False,
            signal_perturbations=False, contact_domain=False, dynamics_randomization=False))


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def retired(document, *, require_full=True):
    return retire_legacy_fault_cases(document, "contracts/frozen_predecessor.json",
        hashlib.sha256(encoded(document)).hexdigest(), require_full_catalogue=require_full)


def test_canonical_prior_units_path_budget_and_independent_runtime_expectation():
    config = clean()
    before = encoded(config)
    assert feedback_transport_settings() == feedback()
    assert command_transport_settings() == command()
    receipt = validate_pc_communication(config)
    assert config["command_transport"]["delay_ms"][1] + config["command_transport"]["can_phase_ms"][1] == 15.
    assert receipt["complete_one_way_feedback_range_s"] == [.00004, .015]
    assert receipt["complete_one_way_command_range_s"] == [.00004, .015]
    assert receipt["synthetic_rtt_range_s"] == [.00008, .030]
    assert receipt["rtt_provenance"] == "sum_of_two_engineering_priors_not_measured_rtt"
    assert FEEDBACK_CONSUMERS == "policy_pc_pd_recovery"
    assert runtime_expectation(config) == dict(feedback_transport=feedback(),
        feedback_transport_model="complete_one_way_feedback_v1", feedback_consumers="policy_pc_pd_recovery")
    assert encoded(config) == before


def test_absent_feedback_preserves_legacy_without_validating_or_patching_it():
    config = {"signal_delay": {"enabled": True}, "command_transport": {"drop_probability_max": .005}}
    before = deepcopy(config)
    assert validate_pc_communication(config) is None
    assert runtime_expectation(config) == {}
    assert config == before


def diagnostic(enabled=True):
    config = clean(enabled)
    config.update(evaluation_transport_mode="training_domain_prior", run_scope="interactive_unscored")
    return config


def test_profile_selection_preserves_legacy_exact_clean_and_fixed_lanes_without_mutation():
    profile = {"delay_ms": .08, "drop_probability": .005}
    config = dict(evaluation_exact_cases=True,
        evaluation=dict(cases=[dict(name="clean"), dict(name="fixed", communication_profile=profile)]))
    before = encoded(config)
    selected = command_profiles(config, ["clean", "fixed", "clean"])
    assert selected == [None, profile, None]
    selected[1]["drop_probability"] = 0.
    assert encoded(config) == before
    assert command_profiles(dict(evaluation_exact_cases=False), ["unlisted"]) is None
    # Merely adding the new feedback settings does not change exact-case semantics.
    assert command_profiles(clean(), ["stand_305mm"] * 6) == [None] * 6


@pytest.mark.parametrize("enabled", [True, False])
def test_explicit_unscored_diagnostic_prior_selects_native_random_domain(enabled):
    config = diagnostic(enabled)
    config.update(qualification=None, domain_qualified=None, qualified=None, run_role="playback")
    config["evaluation"]["cases"][0]["communication_profile"] = None
    before = encoded(config)
    assert command_profiles(config, ["stand_305mm"] * 6) is None
    assert validate_pc_communication(config) is not None
    assert encoded(config) == before


@pytest.mark.parametrize("field,value", [
    ("evaluation_transport_mode", None), ("evaluation_transport_mode", "random"),
    ("evaluation_transport_mode", True), ("run_scope", "training"),
    ("run_scope", "formal_evaluation"), ("run_scope", None),
    ("evaluation_exact_cases", False), ("evaluation_exact_cases", 1),
    ("qualification", "passed"), ("domain_qualified", True), ("qualified", True),
    ("run_role", "train"), ("role", "full"),
])
def test_diagnostic_mode_cannot_escape_unscored_exact_playback(field, value):
    config = diagnostic()
    config[field] = value
    with pytest.raises(ValueError):
        command_profiles(config, ["stand_305mm"])
    with pytest.raises(ValueError):
        validate_pc_communication(config)


@pytest.mark.parametrize("profile", [{"delay_ms": .08}, {"delay_ms": .08, "drop_probability": 0.}, {}])
def test_diagnostic_mode_rejects_any_fixed_profile_even_on_an_unselected_case(profile):
    config = diagnostic()
    config["evaluation"]["cases"].append(dict(name="unselected", communication_profile=profile))
    with pytest.raises(ValueError, match="conflicts with a fixed"):
        command_profiles(config, ["stand_305mm"])
    with pytest.raises(ValueError, match="conflicts with a fixed"):
        validate_pc_communication(config)


def test_diagnostic_mode_requires_valid_feedback_and_canonical_paired_settings():
    config = diagnostic()
    config.pop("feedback_transport")
    with pytest.raises(ValueError, match="explicit PC feedback"):
        command_profiles(config, ["stand_305mm"])
    with pytest.raises(ValueError, match="explicit PC feedback"):
        validate_pc_communication(config)
    config = diagnostic()
    config["feedback_transport"]["delay_range_s"] = [.00004, .08]
    with pytest.raises(ValueError, match="canonical prior"):
        command_profiles(config, ["stand_305mm"])
    config = diagnostic()
    config["command_transport"]["enabled"] = False
    with pytest.raises(ValueError, match="paired feedback"):
        command_profiles(config, ["stand_305mm"])


@pytest.mark.parametrize("cases", [None, [], [dict(name="same"), dict(name="same")], [dict(name=1)]])
def test_diagnostic_mode_requires_unambiguous_case_catalogue(cases):
    config = diagnostic()
    config["evaluation"]["cases"] = cases
    with pytest.raises(ValueError, match="unique explicit"):
        command_profiles(config, ["stand_305mm"])


@pytest.mark.parametrize("config", [clean(), diagnostic()])
def test_exact_profile_selection_rejects_unknown_scene_group(config):
    with pytest.raises(ValueError, match="declared scene-group"):
        command_profiles(config, ["missing"])


def test_feedback_activation_preserves_legacy_and_pairs_new_fixed_clean_and_delayed_lanes():
    legacy = dict(evaluation_exact_cases=True, evaluation=dict(cases=[
        dict(name="clean", transport_enabled=True),
        dict(name="fixed", communication_profile=dict(delay_ms=.08, drop_probability=.005))]))
    assert feedback_profile_enablement(legacy, ["clean", "fixed"]) is None
    config = clean()
    config["evaluation"]["cases"] = [dict(name="clean", transport_enabled=False),
        dict(name="fixed", communication_profile=dict(delay_ms=3., can_phase_ms=.5), transport_enabled=True)]
    groups = ["clean", "fixed", "clean", "fixed"]
    before = encoded(config)
    assert feedback_profile_enablement(config, groups) == [False, True, False, True]
    assert [profile is not None for profile in command_profiles(config, groups)] == [False, True, False, True]
    assert encoded(config) == before
    assert feedback_profile_enablement(clean(), ["stand_305mm"] * 6) == [False] * 6


@pytest.mark.parametrize("diagnostic_mode", [False, True])
def test_disabled_exact_feedback_has_no_active_lane(diagnostic_mode):
    config = diagnostic(False) if diagnostic_mode else clean(False)
    if not diagnostic_mode:
        config["evaluation"]["cases"].append(dict(name="fixed", communication_profile=dict(delay_ms=3.),
            transport_enabled=False))
    assert feedback_profile_enablement(config, ["stand_305mm"] * 6) == [False] * 6


@pytest.mark.parametrize("enabled", [False, True])
def test_training_keeps_shared_draw_and_enabled_diagnostic_has_no_fixed_mask(enabled):
    config = clean(enabled)
    config["evaluation_exact_cases"] = False
    assert feedback_profile_enablement(config, ["stand_305mm"] * 6) is None
    config = diagnostic()
    assert feedback_profile_enablement(config, ["stand_305mm"] * 6) is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("value", [False, True, None, 1])
def test_diagnostic_mode_forbids_every_explicit_case_activation_override(enabled, value):
    config = diagnostic(enabled)
    config["evaluation"]["cases"][0]["transport_enabled"] = value
    with pytest.raises(ValueError, match="forbids explicit transport_enabled"):
        feedback_profile_enablement(config, ["stand_305mm"])
    with pytest.raises(ValueError, match="forbids explicit transport_enabled"):
        validate_pc_communication(config)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_new_fixed_case_activation_requires_explicit_boolean(value):
    config = clean()
    config["evaluation"]["cases"][0]["transport_enabled"] = value
    with pytest.raises(ValueError, match="explicit boolean"):
        feedback_profile_enablement(config, ["stand_305mm"])


@pytest.mark.parametrize("enabled,profile,override", [
    (True, None, True), (True, dict(delay_ms=3.), False),
    (False, dict(delay_ms=3.), True), (False, None, True),
])
def test_new_fixed_case_override_cannot_diverge_from_paired_profile_activation(enabled, profile, override):
    config = clean(enabled)
    config["evaluation"]["cases"][0].update(communication_profile=profile, transport_enabled=override)
    with pytest.raises(ValueError, match="paired native profile activation"):
        feedback_profile_enablement(config, ["stand_305mm"])
    with pytest.raises(ValueError, match="paired native profile activation"):
        validate_pc_communication(config)


def test_feedback_activation_rejects_unknown_lane_and_invalid_canonical_settings():
    with pytest.raises(ValueError, match="declared scene-group"):
        feedback_profile_enablement(clean(), ["missing"])
    config = clean()
    config["feedback_transport"]["delay_range_s"] = [.00004, .08]
    with pytest.raises(ValueError, match="canonical prior"):
        feedback_profile_enablement(config, ["stand_305mm"])


@pytest.mark.parametrize("feedback_enabled,command_enabled", [(True, False), (False, True)])
def test_complete_pc_prior_requires_paired_path_enablement(feedback_enabled, command_enabled):
    config = clean(feedback_enabled)
    config["command_transport"] = command(command_enabled)
    config["evaluation"]["frozen_module_enablement"]["command_transport"] = command_enabled
    before = encoded(config)
    with pytest.raises(ValueError, match="paired feedback and command"):
        validate_pc_communication(config)
    assert encoded(config) == before


@pytest.mark.parametrize("enabled", [True, False])
def test_authoring_preserves_other_flags_legacy_evidence_and_disables_both_paths(enabled):
    source = clean()
    source.pop("feedback_transport")
    source["command_transport"] = {"enabled": True, "drop_probability_max": .005}
    source["signal_delay"]["enabled"] = True
    source["usb_transport"]["enabled"] = True
    source["signal_perturbations"]["max_delay_steps"] = 3
    source["measured_usb_rtt_evidence"] = {"file": "contracts/old_rtt.json", "sha256": "a" * 64}
    source["evaluation"]["frozen_module_enablement"]["usb_transport"] = True
    original = encoded(source)
    authored = apply_pc_communication(source, enabled=enabled)
    assert encoded(source) == original
    assert authored["feedback_transport"] == feedback(enabled)
    assert authored["command_transport"] == command(enabled)
    expected_delay = deepcopy(source["signal_delay"])
    expected_delay["enabled"] = False
    assert authored["signal_delay"] == expected_delay
    assert authored["signal_perturbations"] == dict(source["signal_perturbations"], max_delay_steps=0)
    assert authored["measured_usb_rtt_evidence"] == source["measured_usb_rtt_evidence"]
    mask = authored["evaluation"]["frozen_module_enablement"]
    assert mask["feedback_transport"] is mask["command_transport"] is enabled
    assert mask["usb_transport"] is mask["signal_delay"] is False
    assert all(mask[name] is False for name in ("dynamics_randomization", "contact_domain", "signal_perturbations"))
    assert runtime_expectation(authored)["feedback_transport_model"] == (
        "complete_one_way_feedback_v1" if enabled else None)


@pytest.mark.parametrize("section,key,value", [
    ("feedback_transport", "version", "physical_signal_delay_v1"),
    ("feedback_transport", "scope", "measured_usb_rtt"),
    ("feedback_transport", "delay_range_s", [.00004, .080]),
    ("feedback_transport", "enabled_fraction", .5),
    ("feedback_transport", "jitter_s", float("nan")),
    ("feedback_transport", "jitter_correlation", float("inf")),
    ("feedback_transport", "enabled", 1),
    ("feedback_transport", "drop_probability", 0.),
    ("command_transport", "delay_ms", [.04, 15.]),
    ("command_transport", "can_phase_ms", [0., 2.]),
    ("command_transport", "delivery_model", "legacy_tick_hold"),
    ("command_transport", "enabled_fraction", True),
    ("command_transport", "drop_probability_max", .005),
    ("command_transport", "burst_probability", .0001),
    ("command_transport", "burst_packets", [2., 3.]),
    ("command_transport", "burst_packets", [True, 3]),
    ("command_transport", "jitter_ms", float("inf")),
    ("signal_delay", "enabled", True), ("usb_transport", "enabled", True),
    ("signal_perturbations", "max_delay_steps", 1),
    ("signal_perturbations", "max_delay_steps", False),
])
def test_canonical_contract_rejects_drift_nonfinite_bool_and_delay_stacks(section, key, value):
    config = clean()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_pc_communication(config)


@pytest.mark.parametrize("profile", [
    {"delay_ms": .08, "drop_probability": .005},
    {"delay_ms": .08, "burst_every_packets": 2000, "burst_packets": 3},
    {"delay_ms": .08, "burst_packets": 1},
    {"delay_ms": .08, "drop_probability": True},
    {"delay_ms": .08, "burst_every_packets": float("nan")},
    {"delay_ms": 15., "can_phase_ms": 1.},
    {"delay_ms": .08, "loss_probability": 0.},
])
@pytest.mark.parametrize("enabled", [True, False])
def test_zero_global_loss_cannot_be_overridden_even_in_disabled_fixed_roles(profile, enabled):
    config = clean(enabled)
    config["evaluation"]["cases"][0]["communication_profile"] = profile
    with pytest.raises(ValueError):
        validate_pc_communication(config)


@pytest.mark.parametrize("module", ["signal_delay", "usb_transport"])
def test_frozen_evaluation_toggle_cannot_reenable_a_legacy_delay(module):
    config = clean()
    config["evaluation"]["frozen_module_enablement"][module] = True
    with pytest.raises(ValueError, match="Evaluation override"):
        validate_pc_communication(config)


@pytest.mark.parametrize("evaluation", [None, [], False])
def test_malformed_evaluation_container_fails_with_configuration_error(evaluation):
    config = clean()
    config["evaluation"] = evaluation
    with pytest.raises(ValueError, match="evaluation mapping"):
        validate_pc_communication(config)
    with pytest.raises(ValueError, match="evaluation mapping"):
        apply_pc_communication(config)


def test_retirement_covers_nested_roles_plan_names_and_preserves_every_survivor():
    source = dict(evaluation=catalogue(), plan=dict(domains=[dict(required_case_names=RETIRED + ["retained_000"],
        monitor_case_names=["retained_001"])]), stages=dict(speed=dict(roles=dict(
        full=dict(contract=dict(evaluation=dict(cases=catalogue()["cases"][:3] + catalogue()["cases"][-12:],
            promotion_case_names=["retained_000", *RETIRED])))))))
    before = encoded(source)
    result, provenance = retired(source)
    assert encoded(source) == before
    assert provenance["schema"] == "pc-communication-case-retirement-v1"
    assert provenance["predecessor"] == dict(file="contracts/frozen_predecessor.json",
        sha256=hashlib.sha256(before).hexdigest())
    assert provenance["retired_names"] == RETIRED
    assert provenance["predecessor_full_case_count"] == 322
    assert provenance["successor_full_case_count"] == 310
    assert encoded(result["evaluation"]["cases"]) == encoded(source["evaluation"]["cases"][:310])
    for key in ("canonical_case_names", "quick_case_names", "promotion_case_names",
                "protected_case_names", "retention_case_names"):
        assert result["evaluation"][key] == [name for name in source["evaluation"][key] if name not in RETIRED]
    assert result["plan"]["domains"][0] == dict(required_case_names=["retained_000"], monitor_case_names=["retained_001"])
    role = result["stages"]["speed"]["roles"]["full"]["contract"]["evaluation"]
    assert role["cases"] == source["evaluation"]["cases"][:3]
    assert role["promotion_case_names"] == ["retained_000"]


def test_retirement_refuses_unknown_faults_modified_legacy_profiles_and_unpinned_subsets():
    source = catalogue()
    source["cases"][0]["communication_profile"]["drop_probability"] = .001
    with pytest.raises(ValueError, match="packet loss"):
        retired(source)
    source = catalogue()
    source["cases"][-1]["communication_profile"]["drop_probability"] = 0.
    with pytest.raises(ValueError, match="predecessor profile"):
        retired(source)
    subset = dict(cases=catalogue()["cases"][:3], required_case_names=["retained_000", *RETIRED])
    with pytest.raises(ValueError, match="322-case"):
        retired(subset)
    projected, _ = retired(subset, require_full=False)
    assert projected["required_case_names"] == ["retained_000"]


@pytest.mark.parametrize("domain", ["speed", "terrain", "jump", "stepjump"])
def test_actual_v26_catalogue_retires_only_twelve_and_preserves_frozen_source_bytes(domain):
    path = ROOT / f"contracts/v6_task_campaign_{domain}_worker_v26.json"
    raw = path.read_bytes()
    source = json.loads(raw)
    result, provenance = retire_legacy_fault_cases(source, str(path.relative_to(ROOT)), hashlib.sha256(raw).hexdigest())
    assert path.read_bytes() == raw
    assert len(result["evaluation"]["cases"]) == 310
    expected = [case for case in source["evaluation"]["cases"] if case["name"] not in RETIRED]
    assert encoded(result["evaluation"]["cases"]) == encoded(expected)
    assert provenance["retired_names"] == RETIRED
    authored = apply_pc_communication(result)
    assert validate_pc_communication(authored)["feedback_consumers"] == "policy_pc_pd_recovery"


def test_fixed_projection_rejects_fault_reentry_relabeling_or_surviving_definition_changes():
    predecessor = catalogue()
    new_suite, _ = retired(predecessor)
    config = clean()
    config["evaluation"] = new_suite
    config = apply_pc_communication(config)
    assert validate_no_loss_projection(config, predecessor)["surviving_case_count"] == 310
    bad = deepcopy(config)
    bad["evaluation"]["cases"].append(deepcopy(predecessor["cases"][-1]))
    with pytest.raises(ValueError):
        validate_no_loss_projection(bad, predecessor)
    bad = deepcopy(config)
    bad["evaluation"]["cases"][0]["height_mae_m_max"] = .01
    with pytest.raises(ValueError, match="definitions or coverage"):
        validate_no_loss_projection(bad, predecessor)
    bad = deepcopy(config)
    bad["evaluation"]["canonical_case_names"].append(RETIRED[0])
    with pytest.raises(ValueError, match="retired legacy"):
        validate_no_loss_projection(bad, predecessor)
    bad = deepcopy(config)
    bad["evaluation"]["cases"].reverse()
    with pytest.raises(ValueError, match="definitions or coverage"):
        validate_no_loss_projection(bad, predecessor)
