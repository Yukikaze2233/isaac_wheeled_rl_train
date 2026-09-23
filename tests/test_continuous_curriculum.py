"""Periodic command conditioning, delayed modes and budget-driven scene training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.adaptive_commands import AdaptiveCommandCurriculum
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.task import choose_scene_groups

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_continuous_v57.json"), load)
    return plan, load(plan["base_contract"])


def environment(kind="forward", count=32):
    plan, base = configuration()
    c = stage_contract(base, plan, plan["stages"][0], 6144)
    name = "spin_translate" if kind == "spin_translate" else "forward_5"
    spec = c["skill_specs"][name]
    cfg = {"skill_specs": {name: spec}, "special_mode_activation": c["special_mode_activation"],
           "curriculum_reference_batch": 4096 * 24}
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1
    generator = torch.Generator().manual_seed(617)
    env = SimpleNamespace(num_envs=count, device="cpu", cfg=cfg, scene_groups=[name] * count,
        commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2), mode=torch.zeros(count, dtype=torch.long),
        command_clock=torch.zeros(count), height_clock=torch.zeros(count), push_clock=torch.zeros(count),
        push_enabled=torch.zeros(count, dtype=torch.bool), episode_limits=torch.zeros(count),
        policy_dt=.02, episode_length_buf=torch.zeros(count), training_transitions=0,
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))),
        random=lambda n: torch.rand(n, generator=generator))
    env.performance_curriculum = AdaptiveCommandCurriculum(env.scene_groups, cfg["skill_specs"], c["performance_curriculum"], "cpu")
    return env, SkillCommands(env), name


def test_full_queue_monitors_without_gates_and_caps_later_scene_memory():
    plan, base = configuration()
    assert plan["initialization"].startswith("scratch_")
    assert len(plan["stages"]) == 5
    seen = set()
    for index, recipe in enumerate(plan["stages"]):
        c = stage_contract(base, plan, recipe, 12288)
        assert c["evaluation"]["mode"] == "monitor"
        assert not c["evaluation"].get("require_passing_anchors", False)
        assert c["target_num_envs"] == (12288 if index == 0 else 4096)
        assert c["total_updates"] == (6500 if index == 0 else recipe["updates"])
        assert seen <= set(c["skill_specs"])
        seen.update(c["skill_specs"])
        assert {name for name, _ in choose_scene_groups(c["scene_groups"], 512)} == set(c["skill_specs"])
        assert c["special_mode_activation"]["spin_translate"]["start_reference_updates"] == (2000 if index == 0 else 0)
        assert c["transfer_curriculum"] == "shared_frontiers"
    assert stage_contract(base, plan, plan["stages"][1], 128)["target_num_envs"] == 128


def test_periodic_resampling_changes_velocity_not_height_or_episode_curriculum_identity():
    env, skills, name = environment()
    ids = torch.arange(env.num_envs)
    skills.sample(ids)
    height = env.commands[:, 2].clone()
    old_velocity = env.commands[:, 0].clone()
    course = env.performance_curriculum
    old_caps, old_pool, old_revision = course.env_caps.clone(), course.env_pool.clone(), course.env_revision.clone()
    course.state[name]["caps"][0] = 3.
    course.state[name]["revision"] += 1
    course.age[:] = 500
    skills.sample(ids, reset_height=False)
    assert torch.equal(env.commands[:, 2], height)
    assert not torch.equal(env.commands[:, 0], old_velocity)
    assert torch.equal(course.env_caps, old_caps)
    assert torch.equal(course.env_pool, old_pool)
    assert torch.equal(course.env_revision, old_revision)
    assert torch.all(course.age == 500)
    assert torch.all((env.command_clock >= 5.) & (env.command_clock <= 15.))
    assert torch.unique(env.commands[course.env_pool == 1, 0]).numel() > 1


def test_special_mode_waits_for_training_age_episode_age_and_stability():
    env, skills, _ = environment("spin_translate")
    ids = torch.arange(env.num_envs)
    skills.sample(ids)
    assert not skills.spin.any()
    assert env.commands[:, :2].abs().max() == 0.
    assert not env.performance_curriculum.env_pool.any()
    env.training_transitions = 2000 * 4096 * 24
    env.episode_length_buf[:] = 300
    skills.sample(ids, reset_height=False)
    assert not skills.spin.any()
    skills.stable[:] = True
    env.performance_curriculum.error_sum[:] = 100.
    skills.sample(ids, reset_height=False)
    assert skills.spin.all()
    assert skills.reference_target[:, 0].abs().min() > 0
    assert env.performance_curriculum.error_sum.abs().max() == 0.
    assert env.commands[:, 1].abs().min() > 0


def test_fixed_evaluation_bypasses_training_special_mode_conditions():
    env, _, name = environment("spin_translate")
    spec = env.cfg["skill_specs"][name]
    env.cfg.update(evaluation_exact_cases=True, evaluation={"cases": [{"name": name, "command": spec["command"], "skill": spec}]})
    skills = SkillCommands(env)
    skills.sample(torch.arange(env.num_envs))
    assert skills.spin.all()
    assert torch.all(env.commands[:, 1] == spec["command"][1])
    assert torch.all(env.command_clock == 1e9)


def test_height_course_is_performance_driven_and_latched_until_reset():
    env, skills, name = environment()
    model = env.performance_curriculum
    model.windows[name].extend([[.001, 0., 0., 1.]] * model.window_size)
    model._adapt(name, 100.)
    assert model.state[name]["height_level"] == 1
    assert model.env_height_width[0] == pytest.approx(.06)
    model.reset_episodes(torch.tensor([0]))
    assert model.env_height_width[0] == pytest.approx(.045)


def test_scene_transfer_inherits_learned_frontiers_without_stale_windows():
    env, _, name = environment()
    model = env.performance_curriculum
    model.state[name].update(caps=[3., 0.], height_level=2, last_change=5000., revision=8)
    model.windows[name].append([.001, 0., 0., 1.])
    model.pool_frames[:] = 1000
    target, _, _ = environment()
    restored = target.performance_curriculum
    assert restored.inherit_frontiers(model.state_dict()) == 1
    assert restored.state[name]["caps"] == [3., 0.]
    assert restored.state[name]["height_level"] == 2
    assert restored.state[name]["last_change"] == 0.
    assert len(restored.windows[name]) == 0 and restored.pool_frames.sum() == 0


def test_v56_materialized_contract_is_unchanged():
    import hashlib
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_adaptive_v56.json"), load)
    c = stage_contract(load(plan["base_contract"]), plan, plan["stages"][0], 6144)
    digest = hashlib.sha256((json.dumps(c, indent=2) + "\n").encode()).hexdigest()
    assert digest == "26494ef4924a48e1e93b389b3006a948f1624fc28652911ea7cf0c97bf43f749"
