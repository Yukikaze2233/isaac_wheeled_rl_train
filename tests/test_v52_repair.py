"""Broad-kernel progression and recoverable falls cannot relax fixed acceptance."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.performance_curriculum import PerformanceCurriculum
from wheeled_tasks.chassis.task import FallConfirmation

ROOT = Path(__file__).resolve().parents[1]


def plan(version):
    load = lambda name: json.loads((ROOT / name).read_text())
    resolved = resolve_plan(load(f"contracts/v5_scut35_repair_{version}.json"), load)
    return resolved, load(resolved["base_contract"])


def test_full_queue_keeps_fixed_cases_and_enables_robustness_only_at_final_stage():
    old, base = plan("v51")
    new, _ = plan("v52")
    configs = [stage_contract(base, new, recipe, 6144) for recipe in new["stages"]]
    assert sum(c["total_updates"] for c in configs) == 25336
    for before, after in zip(old["stages"], configs):
        assert after["evaluation"]["cases"] == stage_contract(base, old, before, 6144)["evaluation"]["cases"]
        assert after["fall_confirmation_seconds"] == .1
        assert fixed_suite_contract(after)["fall_confirmation_seconds"] == 0.
        assert after["height_l1_weight"] == 0.
    assert all(not c["signal_perturbations"]["enabled"] for c in configs[:-1])
    robust = configs[-1]
    assert robust["signal_perturbations"]["enabled"]
    assert robust["signal_perturbations"]["max_delay_steps"] == 1
    assert any(c.get("perturbed") for c in robust["evaluation"]["cases"])


def test_coarse_to_fine_thresholds_and_hysteresis_survive_resume():
    config, _ = plan("v52")
    settings = deepcopy(config["stages"][0]["performance_curriculum"])
    settings.update(window_episodes=1, min_reference_updates=0)
    specs = {"stand": {"kind": "stand", "command": [0., 0., .305]}}
    model = PerformanceCurriculum(["stand"], specs, settings, "cpu")
    yes, no = torch.tensor([True]), torch.tensor([False])

    def episode(error):
        model.observe(torch.tensor([error]), torch.zeros(1), torch.zeros(1), yes, yes, no, 1.)
        model.reset_episodes(torch.tensor([0]))

    episode(.012)
    assert model.env_height_width.item() == pytest.approx(.05)
    episode(.02)
    assert model.env_height_width.item() == pytest.approx(.05)  # Hysteresis, not immediate regression.
    episode(.004)
    assert model.state["stand"]["height_level"] == 2
    restored = PerformanceCurriculum(["stand"], specs, settings, "cpu")
    restored.load_state_dict(model.state_dict())
    assert restored.state_dict() == model.state_dict()
    assert restored.env_height_width.item() == pytest.approx(.001 ** .5)
    episode(.04)
    assert model.env_height_width.item() == pytest.approx(.15)


@pytest.mark.parametrize("dt,ticks", [(.02, 5), (.01, 10)])
def test_fall_requires_consecutive_time_and_reset_clears_pending_state(dt, ticks):
    fall = FallConfirmation(2, "cpu", dt, .1)
    gravity, height, flight = torch.tensor([-.4, -1.]), torch.tensor([.3, .3]), torch.zeros(2, dtype=torch.bool)
    for _ in range(ticks - 1):
        assert not fall.update(gravity, height, flight).any()
    assert fall.update(gravity, height, flight).tolist() == [True, False]
    fall.reset(torch.tensor([0]))
    assert not fall.update(gravity, height, flight).any()
    fall.update(torch.tensor([-1., -1.]), height, flight)
    assert fall.ticks.tolist() == [0, 0]


def test_critical_falls_stay_immediate_and_flight_does_not_trigger_low_height():
    fall = FallConfirmation(3, "cpu", .02, .1)
    assert fall.update(torch.tensor([.1, -1., -1.]), torch.tensor([.3, .09, .09]),
                       torch.tensor([True, False, True])).tolist() == [True, True, False]
    strict = FallConfirmation(1, "cpu", .02, 0.)
    assert strict.update(torch.tensor([-.4]), torch.tensor([.3]), torch.tensor([False])).item()
