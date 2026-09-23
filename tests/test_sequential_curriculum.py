"""Task isolation, parallel height holds and immutable predecessor acceptance."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.task import choose_scene_groups

ROOT = Path(__file__).resolve().parents[1]


def configuration(name="v5_sequential_v55"):
    def load(path):
        return json.loads((ROOT / path).read_text())

    plan = resolve_plan(load(f"contracts/{name}.json"), load)
    return plan, load(plan["base_contract"])


def stages():
    plan, base = configuration()
    return plan, {recipe["name"]: stage_contract(base, plan, recipe, 6144) for recipe in plan["stages"]}


def test_first_task_is_stand_and_motion_retains_nominal_height():
    plan, configs = stages()
    assert plan["initialization"].startswith("scratch_")
    assert list(configs["stand"]["skill_specs"]) == ["stand"]
    assert len(configs["stand"]["evaluation"]["cases"]) == 1
    assert configs["stand"]["learning_rate"] == 1e-4
    assert configs["stand"]["critic_warmup_updates"] == 0
    for name in ("translation_nominal", "rotation_nominal", "translation_revisit", "spin_translate_nominal"):
        config = configs[name]
        for spec in config["skill_specs"].values():
            if spec["mode"] == 1:
                assert spec["command"][2] == .305
                assert "height_sampling" not in spec
        assert config["learning_rate"] == 3e-5
        assert config["critic_warmup_updates"] == 50
        assert config["transfer_critic"] is False
        assert config["evaluation"]["require_passing_anchors"]
        assert config["evaluation"]["regression_patience"] == 2
    specs = configs["translation_nominal"]["skill_specs"]
    assert specs["forward_5"]["command"][0] == 5
    assert specs["backward_5"]["command"][0] == -5
    assert not any(spec.get("command_curriculum") for spec in specs.values())


def test_actor_transfer_variant_only_changes_initial_optimization_and_verified_skip():
    original, base = configuration()
    plan, _ = configuration("v5_sequential_transfer_v55")
    assert plan["initialization"].startswith("compatible_actor_transfer_")
    for index, recipe in enumerate(plan["stages"]):
        config = stage_contract(base, plan, recipe, 6144)
        old = stage_contract(base, original, original["stages"][index], 6144)
        assert config["skill_specs"] == old["skill_specs"]
        assert config["evaluation"]["cases"] == old["evaluation"]["cases"]
        assert config["learning_rate"] == 3e-5
        assert config["critic_warmup_updates"] == 50
        assert config["transfer_critic"] is False
        assert config["evaluation"]["skip_training_if_initially_accepted"] == (index == 0)
        assert config["evaluation"]["confirmation_seed"] != config["evaluation"]["seed"]


def test_new_tasks_and_revisits_receive_seventy_percent_after_expansion():
    plan, configs = stages()
    prior = set()
    for recipe in plan["stages"]:
        config = configs[recipe["name"]]
        groups = config["scene_groups"]
        assert sum(g["fraction"] for g in groups) == pytest.approx(1.)
        if recipe["name"] == "height_locomotion":
            primary = {name for name in config["skill_specs"] if name.endswith("__height_train")}
        elif recipe.get("focus_skills"):
            primary = {name for name in config["skill_specs"] if name.split("__reverse_train")[0] in recipe["focus_skills"]}
        else:
            primary = set(config["skill_specs"]) - prior
        if prior and primary and not recipe.get("robust"):
            assert sum(g["fraction"] for g in groups if g["name"] in primary) == pytest.approx(.7)
        prior = set(config["skill_specs"])
        for count in (512, 6144):
            scenes = choose_scene_groups(groups, count)
            assert {name for name, _ in scenes} == set(config["skill_specs"])


def test_height_locomotion_adds_groups_instead_of_mutating_nominal_rehearsal():
    _, configs = stages()
    before, combined = configs["spin_translate_nominal"], configs["height_locomotion"]
    for name, spec in before["skill_specs"].items():
        assert combined["skill_specs"][name] == spec
    for name, spec in combined["skill_specs"].items():
        if name.endswith("__height_train"):
            assert spec["height_sampling"]["range_m"] == [.21, .35]
            assert spec["height_transition_seconds"] == 0
    before_cases = {case["name"] for case in before["evaluation"]["cases"]}
    for case in combined["evaluation"]["cases"]:
        assert case["anchor"] == (case["name"] in before_cases)
    assert all(case["anchor"] for case in configs["ground_consolidation"]["evaluation"]["cases"])


def test_moving_height_transitions_are_taught_after_parallel_holds():
    _, configs = stages()
    holds = configs["height_locomotion"]
    transitions = configs["height_transition_locomotion"]
    assert all(not name.endswith("__height_transition_train") for name in holds["skill_specs"])
    for name, spec in holds["skill_specs"].items():
        assert transitions["skill_specs"][name] == spec
    moving = {name: spec for name, spec in transitions["skill_specs"].items() if name.endswith("__height_transition_train")}
    assert moving
    assert all(spec["height_transition_seconds"] == 6. and spec["height_sampling"]["start_at_target_fraction"] == 0. for spec in moving.values())
    hold_cases = {case["name"] for case in holds["evaluation"]["cases"]}
    assert "forward_05_height_210mm_hold" in hold_cases
    assert "forward_05_height_210mm" not in hold_cases
    new_cases = [case for case in transitions["evaluation"]["cases"] if case["name"] not in hold_cases]
    assert len(new_cases) == 8 and not any(case["anchor"] for case in new_cases)


def test_all_original_final_cases_and_thresholds_are_preserved():
    _, configs = stages()
    original, base = configuration("v5_full_range_v54")
    for old_stage, new_stage in (("ground_full_range", "ground_consolidation"), ("mixed_robust", "mixed_robust")):
        recipe = next(r for r in original["stages"] if r["name"] == old_stage)
        old = stage_contract(base, original, recipe, 6144)
        actual = {c["name"]: c for c in configs[new_stage]["evaluation"]["cases"]}
        for case in old["evaluation"]["cases"]:
            expected = deepcopy(case)
            current = deepcopy(actual[case["name"]])
            expected.pop("anchor", None)
            current.pop("anchor", None)
            assert current == expected
        for key in ("v5_control", "motion_limits", "actor_dim", "critic_dim", "physics_dt", "policy_dt"):
            assert configs[new_stage][key] == old[key]


def test_parallel_height_targets_are_constant_until_reset_and_evaluation_is_exact():
    _, configs = stages()
    config = configs["height_parallel"]
    spec = config["skill_specs"]["height_parallel"]
    count = 10
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1
    env = SimpleNamespace(num_envs=count, device="cpu", cfg={"skill_specs": {"hold": spec}},
        scene_groups=["hold"] * count, commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
        mode=torch.zeros(count, dtype=torch.long), training_transitions=0, command_clock=torch.zeros(count),
        height_clock=torch.zeros(count), push_clock=torch.zeros(count), push_enabled=torch.zeros(count, dtype=torch.bool),
        episode_limits=torch.zeros(count), policy_dt=.02, episode_length_buf=torch.zeros(count),
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))),
        random=lambda n: torch.linspace(0., .99, n))
    commands = SkillCommands(env)
    commands.sample(torch.arange(count))
    targets = env.commands.clone()
    assert targets[:, 2].min() == pytest.approx(.21)
    assert targets[:, 2].max() == pytest.approx(.35)
    for time in (0., 2., 6., 17.):
        env.episode_length_buf[:] = time / env.policy_dt
        commands.update()
        assert torch.equal(env.commands, targets)
    cases = config["evaluation"]["cases"]
    holds = [c for c in cases if c["name"].startswith("height_parallel_")]
    assert [c["command"][2] for c in holds] == [.21, .25, .305, .32, .35]
    assert all("height_sampling" not in c["skill"] and c["height_mae_m_max"] == .005 for c in holds)


@pytest.mark.parametrize("change", ["unknown_height_stage", "unknown_focus", "bad_fraction", "negative_transition"])
def test_invalid_sequential_configuration_is_rejected(change):
    plan, base = configuration()
    if change == "unknown_height_stage":
        plan["height_locomotion_stage"] = "typo"
    elif change == "unknown_focus":
        plan["stages"][0]["focus_skills"] = ["typo"]
    elif change == "bad_fraction":
        plan["rehearsal_fraction"] = 1.
    else:
        plan["locomotion_height_sampling"]["transition_seconds"] = -1.
    with pytest.raises(ValueError):
        stage_contract(base, plan, plan["stages"][0], 6144)
