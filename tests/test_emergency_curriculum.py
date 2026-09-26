"""Emergency budgets, immutable interfaces, task quotas and paired controls."""
import hashlib
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.evaluation import continuation_assessment
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.task import choose_scene_groups


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stages():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v6_emergency_v1.json"), load)
    return [stage_contract(load(plan["base_contract"]), plan, recipe, 16384) for recipe in plan["stages"]]


def test_actual_budgets_and_actor_interface(stages):
    assert [c["total_updates"] for c in stages] == [500, 750, 250, 500, 2500, 3000, 2000]
    assert sum(c["total_updates"] * c["target_num_envs"] * c["num_steps_per_env"] for c in stages) == 2260992000
    for c in stages:
        assert (c["actor_dim"], c["critic_dim"], c["action_dim"]) == (35, 81, 6)
        assert (c["policy_dt"], c["physics_dt"]) == (.02, .005)
        assert c["asset_manifest_sha256"] == hashlib.sha256((ROOT / c["asset_directory"] / "manifest.json").read_bytes()).hexdigest()
        assert c["critic_warmup_updates"] == (25 if c["target_num_envs"] == 16384 else 50)
        assert c["transfer_critic"] is False and c["learning_rate"] == 3e-5
        assert c["evaluation"]["mode"] == "monitor"
        assert len(c["evaluation"]["retention_case_names"]) == 54
        assert len(choose_scene_groups(c["scene_groups"], c["target_num_envs"])) == c["target_num_envs"]


def test_startup_variation_keeps_nominal_phase_and_fixed_eval_controls(stages):
    assert not stages[0]["dynamics_randomization"]["enabled"]
    assert not stages[0]["step_assist"]["enabled"]
    c = stages[1]
    assert c["dynamics_randomization"]["enabled_fraction"] == .5
    assert c["dynamics_randomization"]["inertia_scale"] == [1., 1.]
    cases = {case["name"]: case for case in c["evaluation"]["cases"]}
    assert cases["stand"].get("dynamics_profile") is None
    for name, case in cases.items():
        if "__dynamics_" not in name or name.endswith("_usb"):
            continue
        assert cases[name + "_usb"]["dynamics_profile"] == case["dynamics_profile"]
        assert cases[name + "_usb"]["reset_seed_key"] == case["reset_seed_key"]


def test_core_quotas_and_full_step_heights_are_not_diluted_by_group_count(stages):
    for c in (stages[0], stages[1], stages[2], stages[5]):
        for pool, fraction in c["behavior_pool_fractions"].items():
            actual = sum(group["fraction"] for group in c["scene_groups"]
                         if c["behavior_pool_membership"][group["name"]] == pool)
            assert actual == pytest.approx(fraction)
    terrain = stages[5]
    cases = {case["name"]: case for case in terrain["evaluation"]["cases"]}
    for name, height in (("step_up_15", .15), ("step_up_20", .20), ("step_up_25", .25)):
        assert terrain["skill_specs"][name]["terrain_level"] == 1.
        assert terrain["skill_specs"][name]["terrain_limits"]["step_up_m"] == height
        assert cases[name + "__unassisted"]["step_assist_enabled"] is False
        assert cases[name + "__unassisted"]["reset_seed_key"] == cases[name]["reset_seed_key"]
    assert "step_up_03" in cases and "step_up_06" in cases and "step_up_10" in cases


def test_candidate_cannot_trade_parent_passes_or_mechanics_for_average_score():
    settings = {"retention_case_names": ["stand", "forward"]}
    def result(stand=True, forward=False, mechanics=True):
        return {"rank_lower_is_better": [1, .1, .2], "cases": {
            "stand": {"passed": stand, "checks": {"mechanics": mechanics}},
            "forward": {"passed": forward, "checks": {"mechanics": True}}}}
    baseline = result()
    assert not continuation_assessment(result(False, True), baseline, settings)["eligible"]
    assert not continuation_assessment(result(True, True, False), baseline, settings)["eligible"]
    better = continuation_assessment(result(True, True), baseline, settings)
    assert better["eligible"] and better["nominal_passed"] == 2
