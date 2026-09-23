"""Adaptive command scheduling cannot confuse exposure, failure and mastery."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.adaptive_commands import AdaptiveCommandCurriculum
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.skill_commands import SkillCommands, profile_command
from wheeled_tasks.chassis.task import choose_scene_groups

ROOT = Path(__file__).resolve().parents[1]


def config():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_adaptive_v56.json"), load)
    return plan, stage_contract(load(plan["base_contract"]), plan, plan["stages"][0], 6144)


def curriculum(groups=("forward", "backward")):
    _, materialized = config()
    settings = deepcopy(materialized["performance_curriculum"])
    settings.update(window_episodes=2, min_reference_updates=1, settle_seconds=.02, minimum_scored_seconds=.02)
    specs = {name: {"kind": name, "command": [-5. if name == "backward" else 5., 0., .305]} for name in groups}
    model = AdaptiveCommandCurriculum(groups, specs, settings, "cpu")
    model.env_pool[:] = 1
    return model


def episode(model, *, failed=False, error=0., update=10, pool=1, revision=None):
    count = len(model.group_ids)
    model.env_pool[:] = pool
    model.env_revision[:] = torch.tensor([model.state[model.names[int(i)]]["revision"] for i in model.group_ids]) if revision is None else revision
    for tick in range(2):
        model.observe(torch.zeros(count), torch.full((count,), error), torch.zeros(count),
                      torch.ones(count, dtype=torch.bool), torch.full((count,), tick == 1),
                      torch.full((count,), failed), update)


def test_one_stage_full_parallel_domain_and_original_fixed_cases():
    plan, c = config()
    assert len(plan["stages"]) == 1 and c["total_updates"] == 13000
    assert len(c["evaluation"]["cases"]) == 59
    assert c["evaluation"]["require_passing_anchors"]
    assert len([case for case in c["evaluation"]["cases"] if case["anchor"]]) == 7
    assert c["skill_specs"]["forward_5"]["command"][0] == 5.
    assert c["skill_specs"]["backward_5"]["command"][0] == -5.
    assert c["skill_specs"]["rotate_3rps"]["command"][1] == pytest.approx(6 * torch.pi)
    assert not any(spec["sample_amplitude"] for spec in c["skill_specs"].values() if "sample_amplitude" in spec)
    for count in (128, 6144):
        assert {n for n, _ in choose_scene_groups(c["scene_groups"], count)} == set(c["skill_specs"])


def test_sampling_keeps_full_domain_and_signed_frontiers_and_easy_commands():
    model = curriculum(("backward",) * 3)
    command = torch.tensor([[-5., 0., .305]]).repeat(3, 1)
    model.sample_commands("backward", torch.arange(3), command, lambda n: torch.tensor([.01, .3, .99]))
    assert command[:, 0].tolist() == pytest.approx([-5., -.5, -.497])
    assert model.env_pool.tolist() == [2, 1, 0]


def test_frontier_advances_but_full_and_easy_episodes_do_not_unlock_it():
    model = curriculum(("forward",) * 2)
    episode(model, pool=2)
    episode(model, pool=0)
    assert model.state["forward"]["caps"] == [.5, 0.]
    episode(model, pool=1)
    assert model.state["forward"]["caps"] == [.75, 0.]
    assert model.state["forward"]["full_domain_episodes"] == 2


def test_failure_regresses_frontier_and_stale_episodes_cannot_repromote():
    model = curriculum(("forward",) * 2)
    episode(model)
    episode(model, update=20, failed=True)
    assert model.state["forward"]["caps"] == [.5, 0.]
    episode(model, update=30, revision=0)
    assert model.state["forward"]["caps"] == [.5, 0.]
    assert model.state["forward"]["stale_frontier_episodes"] == 2


def test_directions_adapt_independently_and_checkpoint_roundtrip_is_exact():
    model = curriculum(("forward", "forward", "backward", "backward"))
    model.windows["forward"].extend([[0., 0., 0., 1.]] * 2)
    model._adapt("forward", 10)
    assert model.state["forward"]["caps"][0] == .75
    assert model.state["backward"]["caps"][0] == .5
    restored = curriculum(("forward", "forward", "backward", "backward"))
    restored.load_state_dict(model.state_dict())
    assert restored.state_dict() == model.state_dict()
    broken = model.state_dict()
    broken["groups"]["forward"]["caps"][0] = 50.
    with pytest.raises(ValueError):
        restored.load_state_dict(broken)


def test_warmup_only_and_unsupported_episodes_do_not_count_as_success():
    model = curriculum(("forward",) * 2)
    model.observe(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                  torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), 10)
    assert model.state["forward"]["caps"][0] == .5
    assert model.state["forward"]["frontier_fraction"] < .7


def test_time_profiles_use_the_sampled_per_environment_cap():
    _, c = config()
    spec = c["skill_specs"]["start_stop_5"]
    pose = torch.zeros(2, 7)
    pose[:, 6] = 1.
    env = SimpleNamespace(num_envs=2, device="cpu", cfg={"skill_specs": {"move": spec}},
        scene_groups=["move"] * 2, policy_dt=.02, episode_length_buf=torch.tensor([1., 1300.]),
        command_target=torch.zeros(2, 2), commands=torch.zeros(2, 3),
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))))
    commands = SkillCommands(env)
    commands.command_base[:] = torch.tensor([[.5, 0., .305], [1., 0., .305]])
    commands.update()
    assert env.command_target[:, 0].tolist() == [.5, -1.]
    assert profile_command(torch.tensor([0.]), [.5, 0., .305], {"kind": "constant"})[0, 0] == .5
