"""Conservative recovery preserves case thresholds and guarantees rehearsal coverage."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract

ROOT = Path(__file__).resolve().parents[1]


def plans():
    load = lambda name: json.loads((ROOT / name).read_text())
    old = resolve_plan(load("contracts/v5_scut35_repair_v52.json"), load)
    new = resolve_plan(load("contracts/v5_scut35_repair_v53.json"), load)
    return old, new, load(new["base_contract"])


def test_recovery_keeps_all_acceptance_thresholds_and_adds_explicit_anchors():
    old, new, base = plans()
    before = stage_contract(base, old, old["stages"][0], 6144)
    recipe = new["stages"][0]
    after = stage_contract(base, new, recipe, 6144)
    assert not after.get("performance_curriculum")
    assert after["height_l1_weight"] == 0.
    assert after["transfer_noise_floor"] == .03
    assert after["evaluation"]["require_passing_anchors"]
    assert {c["name"] for c in after["evaluation"]["cases"] if c["anchor"]} == set(recipe["evaluation_anchor_cases"])
    assert len(recipe["evaluation_anchor_cases"]) == 18
    for old_case, new_case in zip(before["evaluation"]["cases"], after["evaluation"]["cases"]):
        assert {k: v for k, v in new_case.items() if k != "anchor"} == {k: v for k, v in old_case.items() if k != "anchor"}
    for key in ("v5_control", "actor_layout", "actor_dim", "critic_dim", "physics_dt", "policy_dt", "asset_manifest_sha256"):
        assert after[key] == before[key]


def test_half_of_flat_sampling_rehearses_previous_capabilities():
    _, new, base = plans()
    recipe = new["stages"][0]
    after = stage_contract(base, new, recipe, 6144)
    rehearsal = set(recipe["rehearsal_groups"])
    groups = after["scene_groups"]
    assert len(rehearsal) == 16
    assert sum(g["fraction"] for g in groups) == pytest.approx(1.)
    assert sum(g["fraction"] for g in groups if g["name"] in rehearsal) == pytest.approx(.5)
    assert all(g["fraction"] > 0 for g in groups)


def test_all_stages_use_fixed_learning_rate_and_finite_regression_patience():
    _, new, base = plans()
    configs = [stage_contract(base, new, recipe, 6144) for recipe in new["stages"]]
    assert sum(c["total_updates"] for c in configs) == 25336
    for config in configs:
        assert config["learning_rate"] == 3e-5
        assert config["learning_rate_schedule"] == "fixed"
        assert config["critic_warmup_updates"] == 50
        assert not config["transfer_critic"]
        assert config["evaluation"]["regression_patience"] == 2
        assert config["evaluation"]["block_updates"] == 250


@pytest.mark.parametrize("field,value", [("rehearsal_groups", ["unknown"]),
                                       ("rehearsal_fraction", 1.),
                                       ("evaluation_anchor_cases", ["unknown"])])
def test_unknown_rehearsal_or_anchor_configuration_is_rejected(field, value):
    _, new, base = plans()
    recipe = deepcopy(new["stages"][0])
    recipe[field] = value
    with pytest.raises(ValueError):
        stage_contract(base, new, recipe, 6144)
