"""Outcome accounting and fixed-suite acceptance must not confuse truncation with survival."""
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
from wheeled_tasks.chassis.evaluation import fixed_suite_contract, grade_fixed_suite


ROOT = Path(__file__).resolve().parents[1]


def sample(count=2):
    return {"velocity": torch.zeros(count, 3), "omega": torch.zeros(count, 3),
            "gravity": torch.tensor([[0., 0., -1.]]).repeat(count, 1),
            "height": torch.full((count,), .3), "position": torch.zeros(count, 3),
            "commands": torch.tensor([[0., 0., .3]]).repeat(count, 1),
            "episode_ticks": torch.ones(count, dtype=torch.long),
            "motor_effort": torch.full((count, 6), 3.), "reward": torch.full((count,), .05),
            "gap": torch.zeros(count), "done": torch.ones(count, dtype=torch.bool),
            "terminated": torch.zeros(count, dtype=torch.bool), "success": torch.zeros(count, dtype=torch.bool),
            "reasons": {"boundary": torch.tensor([True, False]), "fall": torch.tensor([False, True])}}


def test_outcomes_have_disjoint_denominators_and_preserve_boundaries():
    metrics = EpisodeMetrics(["a", "a"], "cpu", .01)
    data = sample()
    data["terminated"][1] = True
    metrics.observe(data)
    report = metrics.report()["groups"]["a"]
    assert report["episodes"] == 2
    assert report["failures"] == 1
    assert report["timeouts"] == report["boundary_truncations"] == 1
    assert report["successes"] == 0
    assert report["episodes"] == report["failures"] + report["timeouts"] + report["successes"]


def test_finished_env_mask_and_warmup_do_not_inflate_samples():
    metrics = EpisodeMetrics(["a", "b"], "cpu", .01, warmup_seconds=.01)
    data = sample()
    metrics.observe(data, torch.tensor([True, False]))
    assert metrics.report()["groups"]["a"]["frames"] == 0
    data["episode_ticks"][:] = 2
    data["done"][:] = False
    metrics.observe(data, torch.tensor([True, False]))
    report = metrics.report()["groups"]
    assert report["a"]["frames"] == 1 and report["b"]["frames"] == 0
    assert report["a"]["rms_motor_torque_nm"] == [3.] * 6
    assert report["b"]["episodes"] == 0


def test_large_accumulators_preserve_each_sample():
    metrics = EpisodeMetrics(["a", "a"], "cpu", .01)
    metrics.frames.fill_(2**29)
    metrics.torque_square.fill_(2**29)
    metrics.observe(sample())
    assert int(metrics.frames[0]) == 2**29 + 1
    assert float(metrics.torque_square[0, 0]) == 2**29 + 9


def test_boundary_exit_is_not_a_passed_fixed_horizon():
    settings = json.loads((ROOT / "contracts/v5_locomotion_v2.json").read_text())["evaluation"]
    settings["cases"] = [{"name": "a", "command": [0., 0., .3]}]
    metrics = EpisodeMetrics(["a", "a"], "cpu", .01)
    data = sample()
    data["reasons"]["fall"][:] = False
    metrics.observe(data)
    result = grade_fixed_suite(metrics.report(), settings, 2)
    assert result["cases"]["a"]["survival_rate"] == .5
    assert not result["passed"]
    assert not result["cases"]["a"]["checks"]["survival"]


def test_fixed_suite_preserves_physical_actuator_grouping():
    original = json.loads((ROOT / "contracts/v5_locomotion_v2.json").read_text())
    config = fixed_suite_contract(original)
    assert config["monitor_applied_effort"] is True
    assert config["v5_control"] == original["v5_control"]
    assert config["episode_seconds"] == 10. and original["episode_seconds"] == 20.
    assert sum(g["fraction"] for g in config["scene_groups"]) == pytest.approx(1.)
    assert {g["name"] for g in original["scene_groups"]} == {"stand", "translate", "rotate"}


def test_spin_translation_uses_mean_reference_velocity_per_episode_instance():
    metrics = EpisodeMetrics(["spin", "spin"], "cpu", .01)
    data = sample()
    data["done"][:] = False
    data["reference_velocity_error_vector"] = torch.tensor([[.3, 0.], [.4, 0.]])
    metrics.observe(data)
    data["episode_ticks"][:] = 2
    data["reference_velocity_error_vector"] = torch.tensor([[-.1, 0.], [0., 0.]])
    metrics.observe(data)
    assert metrics.report()["groups"]["spin"]["reference_velocity_error"] == pytest.approx(.2)
