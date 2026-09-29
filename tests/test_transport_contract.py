"""Audit real frozen transport contracts, not copied synthetic expectations."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.command_transport import UsbCommandTransport
from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.full_curriculum import stage_contract
from wheeled_tasks.chassis.preflight import (
    design_preflight, validate_design_contract, validate_runtime_state,
)
from wheeled_tasks.chassis.response_migration import response_contract_provenance


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("train", "baseline", "quick", "full", "confirmation")
MODEL = "phase_aware_hold_v1"
BASE_CASES = ("stand_305mm", "forward_05", "backward_05", "rotate_1")
PROFILES = {
    "link_jitter": {"delay_ms": .08, "can_phase_ms": .08,
                    "jitter_ms": .02, "jitter_correlation": .9},
    "link_loss_005": {"delay_ms": .08, "can_phase_ms": .5, "drop_probability": .005},
    "link_burst_3": {"delay_ms": .08, "can_phase_ms": .5,
                     "burst_every_packets": 2000, "burst_packets": 3},
    "link_combined": {"delay_ms": .08, "can_phase_ms": .08,
                      "jitter_ms": .02, "jitter_correlation": .9,
                      "drop_probability": .005, "burst_every_packets": 5000,
                      "burst_packets": 3},
}
ADDED_NAMES = [f"{name}__{suffix}" for suffix in PROFILES for name in BASE_CASES]


def read(path):
    return json.loads((ROOT / path).read_text())


@lru_cache(maxsize=1)
def inputs():
    old = read("contracts/v6_wheel_response_training_v1.json")
    plan = read("contracts/v6_transport_response_training_v1.json")
    design = read("contracts/v6_transport_response_expectations_v1.json")
    control = read(plan["runtime_base"]["control_math_source"])
    return old, plan, design, control


def effective(index, role="train", *, previous=False):
    old, plan, _, _ = inputs()
    value = old if previous else plan
    config = stage_contract({}, value, value["stages"][index], 20000)
    seed = 290927 if role == "train" else 290930 if role == "confirmation" else 290929
    if role != "train":
        config["evaluation"]["seed"] = seed
        config = fixed_suite_contract(config)
    count = (20000 if role == "train" else
             len(config["evaluation"]["cases"]) * config["evaluation"]["episodes_per_case"])
    return config, seed, count


def validate(config, index, role, seed, count, design=None):
    _, _, frozen, control = inputs()
    return validate_design_contract(config, frozen if design is None else design, control,
        stage=f"S{index + 1}", role=role, seed=seed, num_envs=count)


def test_plan_and_design_are_hash_bound_and_only_authorize_transport_changes():
    old, plan, design, _ = inputs()
    assert design["status"] == "frozen"
    assert set(design["stages"]) == {f"S{i}" for i in range(1, 7)}
    assert all(set(stage["roles"]) == set(ROLES) for stage in design["stages"].values())
    assert plan["design_preflight"]["sha256"] == hashlib.sha256(
        (ROOT / plan["design_preflight"]["file"]).read_bytes()).hexdigest()
    reviewed = deepcopy(plan)
    reviewed.pop("design_preflight")
    payload = json.dumps(reviewed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(payload).hexdigest() == design["reviewed_plan_payload_sha256"]
    # This independent exclusion list catches any bundled reward/network/asset,
    # controller, actuator-response, sampling or budget edit in the whole plan.
    before, after = deepcopy(old), deepcopy(plan)
    for value in (before, after):
        value.pop("design_preflight")
        value.pop("transport_evaluation_profiles", None)
        value["runtime_base"].pop("command_transport")
        value["runtime_base"].pop("response_migration")
    assert before == after
    assert plan["transport_evaluation_profiles"] == [
        {"suffix": suffix, "communication_profile": profile} for suffix, profile in PROFILES.items()]


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("role", ROLES)
def test_all_thirty_roles_pass_real_design_preflight_and_runtime_model_readback(index, role):
    config, seed, count = effective(index, role)
    _, _, design, control = inputs()
    active = role != "train" or index >= 2
    row = design["stages"][f"S{index + 1}"]["roles"][role]
    assert config["command_transport"]["enabled"] is active
    assert config["command_transport"]["delivery_model"] == MODEL
    assert row["runtime"]["command_transport_model"] == (MODEL if active else None)
    assert row["invocation"]["num_envs"] == count == (
        20000 if role == "train" else 5100 if index == 5 else 1020)
    # Includes frozen design/control/evidence file hashes and all role overrides.
    receipt = design_preflight(config, control, ROOT, stage=config["enabled_stages"][0],
        role=role, seed=seed, num_envs=count)
    assert receipt["status"] == "static_passed"
    assert receipt["actuator_response_evidence"]["status"] == "authenticated"
    instance = (UsbCommandTransport(1, "cpu", config["physics_dt"],
                                   config["command_transport"], seed) if active else None)
    observed = {
        "physics_dt": .001, "policy_dt": .02, "pc_control_dt": .001, "decimation": 20,
        "seed": seed, "leg_kp": 160., "leg_kd": 2.5, "wheel_velocity_p": .6,
        "precision_tracking": False, "dense_tracking": True, "command_reference": True,
        "modules": {"usb_transport": False, "step_assist": False,
                    "command_transport": active,
                    "contact_domain": role != "train" or index >= 2,
                    "dynamics_randomization": role != "train" or index >= 2,
                    "signal_perturbations": role == "train" and index >= 2,
                    "jump_assist": role == "train" and index == 4,
                    "training_schedule": True, "actuator_response": True},
        "actuator_response": config["actuator_response"],
        "command_transport_model": instance.delivery_model if instance else None,
    }
    assert validate_runtime_state(receipt["runtime_expected"], observed)["status"] == (
        "runtime_configuration_passed")


@pytest.mark.parametrize("index", range(6))
def test_old_239_cases_and_every_noncommunication_training_field_are_unchanged(index):
    source, _, _ = effective(index, previous=True)
    target, _, _ = effective(index)
    before, after = deepcopy(source), deepcopy(target)
    for config in (before, after):
        for field in ("command_transport", "response_migration", "design_preflight", "evaluation"):
            config.pop(field)
    assert before == after
    old, new = source["evaluation"], target["evaluation"]
    assert len(old["cases"]) == 239
    assert len(new["cases"]) == 255
    assert new["cases"][:239] == old["cases"]
    assert [case["name"] for case in new["cases"][239:]] == ADDED_NAMES
    lookup = {case["name"]: case for case in old["cases"]}
    for case in new["cases"][239:]:
        base, suffix = case["name"].split("__")
        assert case == {**lookup[base], "name": case["name"], "introduced_stage": "S3",
                        "reset_seed_key": base, "transport_robustness_case": True,
                        "communication_profile": PROFILES[suffix]}
    for field in ("canonical_case_names", "quick_case_names", "retention_case_names"):
        assert new[field] == old[field] + ADDED_NAMES
    assert new["promotion_case_names"] == old["promotion_case_names"] + (
        ADDED_NAMES if index >= 2 else [])
    for field in ("cases", "canonical_case_names", "quick_case_names",
                  "retention_case_names", "promotion_case_names"):
        old = {k: v for k, v in old.items() if k != field}
        new = {k: v for k, v in new.items() if k != field}
    assert old == new  # Includes every previous acceptance threshold.
    _, receipt = response_contract_provenance(source, target, ROOT)
    assert receipt["changed_fields"] == [
        "command_transport", "design_preflight", "evaluation", "response_migration"]
    assert receipt["added_evaluation_case_names"] == sorted(ADDED_NAMES)
    assert receipt["actuator_model"] == "unchanged"
    assert receipt["actor_critic_optimizer"] == "preserved"
    assert receipt["consumed_budget"] == "preserved_no_refund"


@pytest.mark.parametrize("mutation", ["missing_module", "missing_model", "off", "wrong_model", "unexpected_field"])
@pytest.mark.parametrize("index,role", [(2, "train"), (0, "baseline")])
def test_frozen_preflight_rejects_transport_omission_disable_drift_and_extra_fields(index, role, mutation):
    config, seed, count = effective(index, role)
    transport = config["command_transport"]
    if mutation == "missing_module":
        config.pop("command_transport")
    elif mutation == "missing_model":
        transport.pop("delivery_model")
    elif mutation == "off":
        transport["enabled"] = False
    elif mutation == "wrong_model":
        transport["delivery_model"] = "legacy_tick_hold"
    else:
        transport["unreviewed_extra_jitter_ms"] = .4
    with pytest.raises(ValueError, match="command_transport"):
        validate(config, index, role, seed, count)


@pytest.mark.parametrize("mutation", ["missing", "off", "wrong_model"])
@pytest.mark.parametrize("index,role", [(2, "train"), (0, "baseline")])
def test_runtime_instance_cannot_omit_disable_or_use_another_transport(index, role, mutation):
    config, seed, count = effective(index, role)
    expected = validate(config, index, role, seed, count)["runtime_expected"]
    observed = deepcopy(expected)
    if mutation == "missing":
        observed.pop("command_transport_model")
    elif mutation == "off":
        observed["modules"]["command_transport"] = False
        observed["command_transport_model"] = None
    else:
        observed["command_transport_model"] = "legacy_tick_hold"
    with pytest.raises(ValueError, match="command_transport"):
        validate_runtime_state(expected, observed)


@pytest.mark.parametrize("index", [0, 2])
def test_transport_revision_requires_explicit_runtime_model_expectation_even_when_disabled(index):
    config, seed, count = effective(index)
    design = deepcopy(inputs()[2])
    design["stages"][f"S{index + 1}"]["roles"]["train"]["runtime"].pop("command_transport_model")
    with pytest.raises(ValueError, match="command_transport_model"):
        validate(config, index, "train", seed, count, design)


def test_legacy_wheel_response_contract_without_model_expectation_still_passes():
    config, seed, count = effective(0, previous=True)
    control = read(config["control_math_source"])
    receipt = design_preflight(config, control, ROOT, stage=config["enabled_stages"][0],
        role="train", seed=seed, num_envs=count)
    assert receipt["status"] == "static_passed"
    assert "command_transport_model" not in receipt["runtime_expected"]
