"""Clock migration retains learned state and keeps millisecond transport semantics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.command_transport import UsbCommandTransport
from wheeled_tasks.chassis.evaluation import canonical_evaluation_layout, quick_evaluation_cases, select_evaluation_cases
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.policy_transfer import validate_physics_resume, verify_learning_state_restore

ROOT = Path(__file__).resolve().parents[1]


def stage(plan_name):
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load(plan_name), load)
    return stage_contract(load(plan["base_contract"]), plan, plan["stages"][0], 16384)


def test_200hz_contract_has_only_declared_training_changes():
    old = stage("contracts/v6_scut35_continuation_v1.json")
    new = stage("contracts/v6_scut35_200hz_v1.json")
    result = validate_physics_resume(old, new)
    assert result["changed_fields"] == ["actor_migration", "evaluation", "physics_dt", "training_reference"]
    assert new["physics_dt"] == .005 and new["policy_dt"] == .02
    assert new["total_updates"] == 500 and new["critic_warmup_updates"] == 25
    assert new["command_transport"] == old["command_transport"]
    assert new["evaluation"]["cases"] == old["evaluation"]["cases"]
    bad = deepcopy(new)
    bad["v5_control"]["leg_kp"] += 1
    with pytest.raises(ValueError, match="training semantics"):
        validate_physics_resume(old, bad)
    bad = deepcopy(new)
    bad["evaluation"]["stand_drift_m_max"] = 1.
    with pytest.raises(ValueError, match="acceptance criteria"):
        validate_physics_resume(old, bad)


@pytest.mark.parametrize("loss,burst", [(0., 0.), (.2, .1)])
def test_5ms_physics_preserves_1ms_packet_sequence_and_torque_impulse(loss, burst):
    cfg = {"delay_ms": [.1, 5.], "can_period_ms": 1., "enabled_fraction": 1.,
           "drop_probability_max": loss, "burst_probability": burst, "burst_packets": [2, 3]}
    fine = UsbCommandTransport(8, "cpu", .001, cfg, 617)
    coarse = UsbCommandTransport(8, "cpu", .005, cfg, 617)
    generator = torch.Generator().manual_seed(55)
    for tick in range(40):
        command = torch.randn(8, 6, generator=generator)
        expected = torch.stack([fine.apply_torque(command) for _ in range(5)]).mean(0)
        actual = coarse.apply_torque(command)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(coarse.age, fine.age)
        torch.testing.assert_close(coarse.lost, fine.lost)
        if tick == 15:
            ids = torch.tensor([1, 3, 7])
            fine.reset(ids)
            coarse.reset(ids)
    assert coarse.tick == fine.tick == 200


def test_quick_suite_keeps_targets_protection_and_world_slots():
    config = stage("contracts/v6_scut35_200hz_v1.json")
    settings = config["evaluation"]
    settings["protected_case_names"] = ["stand__mu_010", "rotate_1_reverse_perturbed"]
    names = quick_evaluation_cases(settings)
    assert set(settings["promotion_case_names"]) <= set(names)
    assert set(settings["protected_case_names"]) <= set(names)
    assert len(names) < len(settings["cases"]) / 4
    subset = select_evaluation_cases(settings, names)
    slots, count = canonical_evaluation_layout(subset, len(names) * 4)
    assert count == len(settings["cases"]) * 4
    for i, case in enumerate(subset["cases"]):
        original = settings["canonical_case_names"].index(case["name"])
        assert slots[i * 4:(i + 1) * 4] == list(range(original * 4, original * 4 + 4))


def test_restore_verification_catches_lost_optimizer_moments():
    checkpoint = {"actor_state_dict": {"weight": torch.tensor([1.])},
                  "critic_state_dict": {"weight": torch.tensor([2.])},
                  "optimizer_state_dict": {"state": {0: {"step": torch.tensor(100.), "exp_avg": torch.tensor([.3])}},
                                           "param_groups": [{"lr": 3e-5, "params": [0]}]}}

    class Algorithm:
        def __init__(self):
            self.state = deepcopy(checkpoint)

        def save(self):
            return self.state

    algorithm = Algorithm()
    assert verify_learning_state_restore(algorithm, checkpoint)["optimizer_exact"]
    algorithm.state["optimizer_state_dict"]["state"][0]["exp_avg"].zero_()
    with pytest.raises(ValueError, match="optimizer_state_dict"):
        verify_learning_state_restore(algorithm, checkpoint)
