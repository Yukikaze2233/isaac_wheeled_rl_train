"""Independent declarations for the optional shared PC feedback path."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_chassis_preflight import fixture
from wheeled_tasks.chassis.pc_communication import (
    command_transport_settings, feedback_transport_settings,
)
from wheeled_tasks.chassis.preflight import validate_design_contract, validate_runtime_state


def feedback_fixture(enabled=True):
    config, design, control = fixture()
    row = design["stages"]["S1"]["roles"]["train"]
    control = json.loads((Path(__file__).resolve().parents[1] /
        "contracts/v6_wheel_response_control_200hz_v1.json").read_text())
    row["control"]["actuators.wheel"] = deepcopy(control["actuators"]["wheel"])
    expected = row["contract"]
    signal = dict(version="physical_signal_delay_v1", enabled=False,
        clock="physics_steps", jitter_clock="policy_steps", feedback_pd="fresh_physics_feedback",
        enabled_fraction=.75, jitter_ms=5.,
        profiles=[dict(probability=.60, observation_ms=[0., 20.], action_ms=[0., 15.]),
                  dict(probability=.35, observation_ms=[20., 40.], action_ms=[15., 30.]),
                  dict(probability=.05, observation_ms=[40., 80.], action_ms=[30., 60.])],
        schedule=dict(clock="domain_actor_updates", start=0, end=360))
    # Declare the expected role first; candidate changes cannot alter it.
    expected.update(physics_dt=.005, signal_delay=signal,
        feedback_transport=feedback_transport_settings(enabled),
        command_transport=command_transport_settings(enabled),
        signal_perturbations=dict(enabled=False, max_delay_steps=0))
    config.update({key: deepcopy(expected[key]) for key in (
        "physics_dt", "signal_delay", "feedback_transport", "command_transport", "signal_perturbations")})
    runtime = row["runtime"]
    runtime.update(physics_dt=.005, pc_control_dt=.005, decimation=4,
        signal_delay=deepcopy(signal), feedback_transport=feedback_transport_settings(enabled),
        feedback_transport_model="complete_one_way_feedback_v1" if enabled else None,
        feedback_consumers="policy_pc_pd_recovery")
    runtime["modules"].update(signal_delay=False, feedback_transport=enabled, command_transport=enabled)
    return config, design, control


def validate(config, design, control):
    return validate_design_contract(config, design, control,
        stage="S1", role="train", seed=617, num_envs=20000)


@pytest.mark.parametrize("enabled", [False, True])
def test_complete_new_role_declares_feedback_even_when_disabled(enabled):
    config, design, control = feedback_fixture(enabled)
    validate(config, design, control)
    expected = design["stages"]["S1"]["roles"]["train"]["runtime"]
    assert validate_runtime_state(expected, deepcopy(expected))["status"] == "runtime_configuration_passed"


@pytest.mark.parametrize("key", ["feedback_transport", "feedback_transport_model", "feedback_consumers"])
def test_runtime_cannot_omit_feedback_expectation(key):
    _, design, _ = feedback_fixture()
    expected = design["stages"]["S1"]["roles"]["train"]["runtime"]
    actual = deepcopy(expected)
    del expected[key]
    with pytest.raises(ValueError, match="independent complete expectation"):
        validate_runtime_state(expected, actual)


@pytest.mark.parametrize("key,value", [
    ("feedback_transport_model", "substep_rtt_sensor_interpolation_and_torque_impulse"),
    ("feedback_consumers", "policy_only_fresh_pc_pd"),
])
def test_runtime_rejects_substitution_of_feedback_model_or_consumer(key, value):
    _, design, _ = feedback_fixture()
    expected = design["stages"]["S1"]["roles"]["train"]["runtime"]
    actual = deepcopy(expected)
    actual[key] = value
    with pytest.raises(ValueError, match=key):
        validate_runtime_state(expected, actual)


def test_enabled_instance_cannot_be_reported_as_disabled():
    _, design, _ = feedback_fixture()
    expected = design["stages"]["S1"]["roles"]["train"]["runtime"]
    actual = deepcopy(expected)
    actual["modules"]["feedback_transport"] = False
    with pytest.raises(ValueError, match="modules.feedback_transport"):
        validate_runtime_state(expected, actual)


def test_candidate_requires_independent_complete_feedback_field():
    config, design, control = feedback_fixture()
    del design["stages"]["S1"]["roles"]["train"]["contract"]["feedback_transport"]
    with pytest.raises(ValueError, match="independent complete settings"):
        validate(config, design, control)


@pytest.mark.parametrize("key,value", [
    ("signal_delay", True), ("usb_transport", True),
])
def test_disabled_legacy_layer_cannot_be_reenabled(key, value):
    config, design, control = feedback_fixture()
    config[key]["enabled"] = value
    with pytest.raises(ValueError, match="cannot stack"):
        validate(config, design, control)


def test_independently_frozen_feedback_limit_cannot_drift():
    config, design, control = feedback_fixture()
    config["feedback_transport"]["delay_range_s"][1] = .080
    with pytest.raises(ValueError, match="canonical prior"):
        validate(config, design, control)


def test_runtime_extra_enabled_path_without_design_is_rejected():
    _, design, _ = fixture()
    expected = design["stages"]["S1"]["roles"]["train"]["runtime"]
    actual = deepcopy(expected)
    actual.update(physics_dt=.005, feedback_transport=feedback_transport_settings(),
        feedback_transport_model="complete_one_way_feedback_v1",
        feedback_consumers="policy_pc_pd_recovery")
    actual["modules"]["feedback_transport"] = True
    with pytest.raises(ValueError, match="independent complete expectation"):
        validate_runtime_state(expected, actual)
