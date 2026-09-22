"""Repair curricula preserve acceptance and never disguise unsupported episodes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.performance_curriculum import PerformanceCurriculum
from wheeled_tasks.chassis.rewards import StationaryAnchor

ROOT = Path(__file__).resolve().parents[1]


def plans():
    load = lambda name: json.loads((ROOT / name).read_text())
    old = resolve_plan(load("contracts/v5_scut35_integrated_v1.json"), load)
    new = resolve_plan(load("contracts/v5_scut35_repair_v51.json"), load)
    return old, new, load(old["base_contract"])


def test_repair_changes_training_coverage_without_relaxing_cases_or_interface():
    old, new, base = plans()
    before = stage_contract(base, old, old["stages"][0], 6144)
    after = stage_contract(base, new, new["stages"][0], 6144)
    assert sum(s["updates"] for s in new["stages"]) == 38000
    assert after["evaluation"]["cases"] == before["evaluation"]["cases"]
    for key in ("v5_control", "actor_layout", "actor_dim", "critic_dim", "physics_dt", "policy_dt",
                "asset_manifest_sha256", "policy_action_order", "height_l1_weight"):
        assert after[key] == before[key]
    assert after["critic_warmup_updates"] == 50 and not after["transfer_critic"]
    specs = after["skill_specs"]
    assert specs["height_hold_low"]["command"] == [0., 0., .29]
    assert specs["height_hold_high"]["command"] == [0., 0., .32]
    assert specs["rotate_2rps__reverse_train"]["command"][1] < 0
    assert not specs["rotate_2rps__reverse_train"]["sample_yaw_sign"]
    assert not specs["backward_05"]["sample_amplitude"]
    for recipe in new["stages"]:
        config = stage_contract(base, new, recipe, 6144)
        assert sum(g["fraction"] for g in config["scene_groups"]) == pytest.approx(1.)
        assert len({c["name"] for c in config["evaluation"]["cases"]}) == len(config["evaluation"]["cases"])
        if recipe["name"] != "flat_repair":
            assert "performance_curriculum" not in config
            current = set(recipe["new_skills"])
            if current:
                # Signed copies retain the source skill's rehearsal partition.
                assert sum(g["fraction"] for g in config["scene_groups"]
                           if g["name"].removesuffix("__reverse_train") in current) == pytest.approx(.5)


def curriculum(groups=("positive", "negative")):
    _, plan, _ = plans()
    cfg = deepcopy(plan["stages"][0]["performance_curriculum"])
    cfg.update(window_episodes=2, min_reference_updates=1, spin_levels_rad_s=[4., 6., 8., 12.])
    specs = {name: {"kind": "rotate", "command": [0., -12. if name == "negative" else 12., .305]}
             for name in set(groups)}
    return PerformanceCurriculum(groups, specs, cfg, "cpu")


def test_directional_windows_advance_independently_and_resume_exactly():
    model = curriculum(("positive", "positive", "negative", "negative"))
    model.observe(torch.tensor([.002, .002, .03, .03]), torch.zeros(4), torch.zeros(4),
                  torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool),
                  torch.zeros(4, dtype=torch.bool), 10)
    assert model.limit_command("positive", [0., 12., .305])[1] == 6.
    assert model.limit_command("negative", [0., -12., .305])[1] == -4.
    state = model.state_dict()
    restored = curriculum(("positive", "positive", "negative", "negative"))
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.limit_command("positive", [0., 12., .305])[1] == 6.


def test_failures_and_airborne_only_episodes_cannot_unlock_spin():
    model = curriculum(("negative", "negative"))
    model.observe(torch.zeros(2), torch.zeros(2), torch.zeros(2),
                  torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                  torch.zeros(2, dtype=torch.bool), 10)
    assert model.limit_command("negative", [0., -12., .305])[1] == -4.
    model.observe(torch.zeros(2), torch.zeros(2), torch.zeros(2),
                  torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                  torch.ones(2, dtype=torch.bool), 20)
    assert model.limit_command("negative", [0., -12., .305])[1] == -4.


def test_reward_scale_is_latched_until_each_environment_resets():
    _, plan, _ = plans()
    cfg = deepcopy(plan["stages"][0]["performance_curriculum"])
    cfg.update(window_episodes=1, min_reference_updates=0)
    model = PerformanceCurriculum(["stand", "stand"],
        {"stand": {"kind": "stand", "command": [0., 0., .305]}}, cfg, "cpu")
    model.observe(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.bool),
                  torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool), 1)
    assert model.height_scale.tolist() == [1.5]
    assert model.env_height_scale.tolist() == [2., 2.]
    model.reset_episodes(torch.tensor([0]))
    assert model.env_height_scale.tolist() == [1.5, 2.]


def test_stationary_anchor_is_bounded_and_not_reset_by_contact_loss():
    anchor = StationaryAnchor(1, "cpu", 1., .1)
    active, absent = torch.tensor([True]), torch.tensor([False])
    assert anchor.reward(torch.tensor([[0., 0.]]), active, active).item() == 0.
    moved = torch.tensor([[.1, 0.]])
    cost = anchor.reward(moved, active, active).item()
    assert cost == pytest.approx(-.63212056)
    assert anchor.reward(moved, active, absent).item() == 0.
    assert anchor.reward(moved, active, active).item() == pytest.approx(cost)
    assert -1. <= anchor.reward(torch.tensor([[10., 10.]]), active, active).item() <= 0.
    anchor.reward(moved, absent, active)
    assert anchor.reward(moved, active, active).item() == 0.
