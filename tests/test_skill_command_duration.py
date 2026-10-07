"""Ordinary command dwell is independent of special task and global clocks."""
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.skill_commands import SkillCommands, sample_planar_command, validate_planar_sampling
from wheeled_tasks.chassis.training_schedule import TrainingSchedule


def environment(spec, count=512):
    generator = torch.Generator().manual_seed(1028)
    env = SimpleNamespace(num_envs=count, device="cpu", global_actor_update=0,
        stage_actor_update=0, training_transitions=0, scene_groups=["movement"] * count,
        cfg={"skill_specs": {"movement": spec}, "new_asset_curriculum": {
            "enabled": True, "command_resampling_seconds": [1., 8.],
            "long_command_fraction": 1., "long_command_resampling_seconds": [30., 40.]}},
        random=lambda n: torch.rand(n, generator=generator), policy_dt=.02,
        commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
        mode=torch.zeros(count, dtype=torch.long), push_enabled=torch.zeros(count, dtype=torch.bool))
    for name in ("command_clock", "height_clock", "push_clock", "episode_limits", "episode_length_buf"):
        setattr(env, name, torch.zeros(count))
    return env


def test_real_skill_sampler_honors_saved_dwell_without_global_long_tail():
    spec = {"kind": "forward", "command": [.5, 0., .305],
        "command_resampling_seconds": [5., 15.], "command_resampling_scope": "skill"}
    env = environment(spec)
    commands = SkillCommands(env)
    commands.sample(torch.arange(env.num_envs))
    assert env.command_clock.min() >= 5 and env.command_clock.max() <= 15
    assert env.command_clock.min() < 6 and env.command_clock.max() > 14
    assert env.command_target[:, 0].eq(.5).all()
    commands.sample(torch.arange(env.num_envs), reset_height=False)
    assert env.command_clock.min() >= 5 and env.command_clock.max() <= 15


def test_unmarked_skills_retain_the_global_distribution():
    env = environment({"kind": "forward", "command": [.5, 0., .305],
        "command_resampling_seconds": [5., 15.]})
    commands = SkillCommands(env)
    commands.sample(torch.arange(env.num_envs))
    assert env.command_clock.min() >= 30 and env.command_clock.max() <= 40


def test_explicit_default_retains_existing_rng_sequence():
    first = environment({"kind": "forward", "command": [.5, 0., .305]})
    second = environment({"kind": "forward", "command": [.5, 0., .305]})
    a, b = TrainingSchedule(first), TrainingSchedule(second)
    torch.testing.assert_close(a.command_duration(512, [5., 15.]),
                               b.command_duration(512, [5., 15.], per_skill=False), rtol=0, atol=0)
    torch.testing.assert_close(first.random(10), second.random(10), rtol=0, atol=0)


@pytest.mark.parametrize("spec", [{"command_resampling_scope": "all"},
    {"command_resampling_scope": "skill"},
    {"command_resampling_scope": "skill", "command_resampling_seconds": [0., 15.]},
    {"command_resampling_scope": "skill", "command_resampling_seconds": [15., 5.]},
    {"command_resampling_scope": "skill", "command_resampling_seconds": [5., float("inf")]}])
def test_undeclared_or_invalid_skill_override_is_rejected(spec):
    with pytest.raises(ValueError, match="Skill-local command duration"):
        validate_planar_sampling(spec)


def test_disabled_curriculum_does_not_consume_duration_rng():
    first = environment({"kind": "forward", "command": [.5, 0., .305]})
    second = environment({"kind": "forward", "command": [.5, 0., .305]})
    first.cfg["new_asset_curriculum"]["enabled"] = False
    assert TrainingSchedule(first).command_duration(512, [5., 15.], per_skill=True) is None
    torch.testing.assert_close(first.random(10), second.random(10), rtol=0, atol=0)


def test_standing_draw_is_resampled_and_preserves_height():
    count = 50000
    generator = torch.Generator().manual_seed(1028)
    random = lambda n: torch.rand(n, generator=generator)
    spec = {"kind": "forward", "standing_command_probability": .1}
    command = torch.tensor([.5, .4, .305]).expand(count, -1).clone()
    sample_planar_command(command, spec, random)
    first = command[:, :2].eq(0).all(-1)
    assert first.float().mean().item() == pytest.approx(.1, abs=.005)
    assert command[:, 2].eq(.305).all()
    command[:, :2] = torch.tensor([.5, .4])
    sample_planar_command(command, spec, random)
    second = command[:, :2].eq(0).all(-1)
    assert (first & second).float().mean().item() == pytest.approx(.01, abs=.002)


def test_fixed_evaluation_keeps_command_and_rng_with_standing_enabled():
    def reject(_):
        raise AssertionError("fixed evaluation must not draw a standing mask")
    command = torch.tensor([[.5, .4, .305]])
    before = command.clone()
    sample_planar_command(command, {"standing_command_probability": 1.}, reject, evaluation=True)
    torch.testing.assert_close(command, before, rtol=0, atol=0)


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan"), None])
def test_invalid_standing_probability_is_rejected(probability):
    with pytest.raises(ValueError, match="Standing command probability"):
        validate_planar_sampling({"standing_command_probability": probability})
