"""Scene quotas cover every group without concentrating rounding in the last skill."""
from collections import Counter
import ast
import json
import math
from pathlib import Path

import pytest

from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
from wheeled_tasks.chassis.task import choose_scene_groups


ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "contracts/v6_task_campaign_v7.json").read_text())


def groups(fractions):
    return [{"name": f"skill_{index}", "fraction": fraction, "terrain": ["flat", "rough"]}
            for index, fraction in enumerate(fractions)]


def test_largest_remainders_keep_group_order_and_restart_terrain_cycles():
    scene_groups = groups([.26, .27, .24, .23])
    scenes = choose_scene_groups(scene_groups, 10)
    assert scenes == [("skill_0", "flat"), ("skill_0", "rough"), ("skill_0", "flat"),
                      ("skill_1", "flat"), ("skill_1", "rough"), ("skill_1", "flat"),
                      ("skill_2", "flat"), ("skill_2", "rough"),
                      ("skill_3", "flat"), ("skill_3", "rough")]


def test_equal_remainders_break_ties_in_original_order():
    scene_groups = groups([.25] * 4)
    scenes = choose_scene_groups(scene_groups, 6)
    assert Counter(name for name, _ in scenes) == dict(skill_0=2, skill_1=2, skill_2=1, skill_3=1)
    assert choose_scene_groups(scene_groups, 6) == scenes


def test_rounding_can_give_a_subunit_quota_its_first_environment():
    scenes = choose_scene_groups(groups([.2, .3, .5]), 3)
    assert Counter(name for name, _ in scenes) == dict.fromkeys(["skill_0", "skill_1", "skill_2"], 1)


@pytest.mark.parametrize("fractions,count", [([.1, .9], 2), ([.2 / 3, .9 / 3, 1.9 / 3], 3)])
def test_feasible_small_populations_reserve_one_environment_before_distributing_remainders(fractions, count):
    scenes = choose_scene_groups(groups(fractions), count)
    assert Counter(name for name, _ in scenes) == {f"skill_{index}": 1 for index in range(count)}


@pytest.mark.parametrize("domain", PLAN["domains"], ids=lambda domain: domain["name"])
@pytest.mark.parametrize("count", [4096, 20000])
def test_frozen_worker_allocations_have_exact_totals_and_floor_ceil_quotas(domain, count):
    worker = json.loads((ROOT / domain["contract"]).read_text())
    scene_groups = worker["scene_groups"]
    scenes = choose_scene_groups(scene_groups, count)
    counts = Counter(name for name, _ in scenes)
    assert len(scenes) == count
    assert set(counts) == {group["name"] for group in scene_groups}
    for group in scene_groups:
        quota = count * group["fraction"]
        assert counts[group["name"]] in (math.floor(quota), math.ceil(quota))
    assert scenes == [(group["name"], group["terrain"][i % len(group["terrain"])])
                      for group in scene_groups for i in range(counts[group["name"]])]


def test_recovery_yaw_directions_no_longer_absorb_a_last_group_rounding_bias():
    worker = json.loads((ROOT / "contracts/v6_task_campaign_recovery_worker_v6.json").read_text())
    counts = Counter(name for name, _ in choose_scene_groups(worker["scene_groups"], 4096))
    assert counts["rotate_1"] + counts["rotate_1__height_train"] == 42
    assert counts["rotate_1__reverse_train"] + counts["rotate_1__reverse_train__height_train"] == 42


@pytest.mark.parametrize("domain", PLAN["domains"], ids=lambda domain: domain["name"])
def test_integer_four_env_per_case_domain_evaluation_assignments_stay_unchanged(domain):
    worker = json.loads((ROOT / domain["contract"]).read_text())
    worker["evaluation"] = select_evaluation_cases(worker["evaluation"], domain["required_case_names"])
    fixed = fixed_suite_contract(worker)
    count = 4 * len(fixed["scene_groups"])
    # The old allocator already assigned four per case when these quotas are integers.
    assert all(count * group["fraction"] == 4 for group in fixed["scene_groups"])
    assert choose_scene_groups(fixed["scene_groups"], count) == [
        (group["name"], group["terrain"][0]) for group in fixed["scene_groups"] for _ in range(4)]


@pytest.mark.parametrize("case_count", [45, 322])
def test_fixed_evaluation_four_replicas_include_equal_fraction_float_roundoff(case_count):
    worker = json.loads((ROOT / PLAN["domains"][0]["contract"]).read_text())
    names = [case["name"] for case in worker["evaluation"]["cases"][:case_count]]
    worker["evaluation"] = select_evaluation_cases(worker["evaluation"], names)
    fixed = fixed_suite_contract(worker)
    if case_count == 322:
        # A raw floor would make each of the first 321 cases lose one replica.
        assert 4 * case_count * fixed["scene_groups"][0]["fraction"] < 4
    assert choose_scene_groups(fixed["scene_groups"], 4 * case_count) == [
        (group["name"], group["terrain"][0]) for group in fixed["scene_groups"] for _ in range(4)]


@pytest.mark.parametrize("case_count", [45, 322])
def test_production_fixed_evaluation_uses_integer_replicas_without_training_allocator(case_count):
    worker = json.loads((ROOT / PLAN["domains"][0]["contract"]).read_text())
    names = [case["name"] for case in worker["evaluation"]["cases"][:case_count]]
    worker["evaluation"] = select_evaluation_cases(worker["evaluation"], names)
    fixed = fixed_suite_contract(worker)
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    branch = next(node for node in init.body if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Call)
                  and ast.unparse(node.test) == "config.get('evaluation_exact_cases')")
    code = compile(ast.Module(body=[branch], type_ignores=[]), "production_scene_allocation", "exec")

    def reject_training_allocation(*args):
        pytest.fail("Fixed evaluation must use its independent integer replica assignment")

    scope = dict(config=fixed, num_envs=4 * case_count, choose_scene_groups=reject_training_allocation)
    exec(code, scope)
    assert scope["scenes"] == [(case["name"], case.get("terrain", "flat"))
                                for case in fixed["evaluation"]["cases"] for _ in range(4)]
    scope["num_envs"] += 1
    with pytest.raises(ValueError, match="equal integer replicas"):
        exec(code, scope)


@pytest.mark.parametrize("fraction", [0., -.1, float("nan"), float("inf"), float("-inf"),
                                    True, "1", None, 1.1, 10 ** 1000])
def test_invalid_fractions_are_rejected(fraction):
    with pytest.raises(ValueError, match="finite and positive"):
        choose_scene_groups(groups([fraction, 1.]), 10)


@pytest.mark.parametrize("scene_groups", [[], groups([.4, .4])])
def test_empty_or_nonunit_total_fractions_are_rejected(scene_groups):
    with pytest.raises(ValueError, match="sum to one"):
        choose_scene_groups(scene_groups, 10)


@pytest.mark.parametrize("count", [0, -1, 1, 2., True, float("nan")])
def test_invalid_or_insufficient_environment_counts_are_rejected(count):
    with pytest.raises(ValueError, match="at least one environment"):
        choose_scene_groups(groups([.5, .5]), count)


def test_minimum_populations_that_exceed_raw_ceil_quotas_or_no_terrain_are_rejected():
    with pytest.raises(ValueError, match="within floor/ceil quotas"):
        choose_scene_groups(groups([.05, .05, .9]), 3)
    scene_groups = groups([.5, .5])
    scene_groups[-1]["terrain"] = []
    with pytest.raises(ValueError, match="Empty scene group"):
        choose_scene_groups(scene_groups, 10)


def test_near_unit_sum_preserves_raw_quotas_when_they_can_fill_the_count():
    scenes = choose_scene_groups(groups([.5, .5000000001]), 10)
    assert Counter(name for name, _ in scenes) == dict(skill_0=5, skill_1=5)


@pytest.mark.parametrize("fraction", [.5000000009, .4999999991, .4999999999])
def test_near_unit_sum_that_cannot_fill_raw_floor_ceil_quotas_is_rejected(fraction):
    # Last example has integer quotas: adding one would exceed its ceil even though
    # the missing count is smaller than the number of groups.
    with pytest.raises(ValueError, match="within floor/ceil quotas"):
        choose_scene_groups(groups([.5, fraction]), 10_000_000_000)
