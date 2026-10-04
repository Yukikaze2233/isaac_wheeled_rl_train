"""Exercise the native sampler, its restored clock, and fixed-evaluation bypass."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.locomotion_difficulty import (
    difficulty_envelope, difficulty_settings, scale_planar_command, validate_difficulty,
)
from wheeled_tasks.chassis.skill_commands import SkillCommands, sample_planar_command
from wheeled_tasks.chassis.task import choose_scene_groups

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "contracts/v6_task_campaign_speed_worker_v10.json").read_text())


def environment(*, actor=2600, origin=2600, exact=False, legacy=False):
    cfg = deepcopy(BASE)
    if not legacy:
        cfg["locomotion_difficulty"] = difficulty_settings(origin)
    if exact:
        names = ["forward_5", "rotate_3rps", "forward_5__height_train", "stand_230mm"]
        cfg["evaluation_exact_cases"] = True
        cfg["evaluation"]["cases"] = [dict(name=name, command=cfg["skill_specs"][name]["command"],
            skill=cfg["skill_specs"][name]) for name in names]
        groups = [name for name in names for _ in range(32)]
    else:
        groups = [name for name, _ in choose_scene_groups(cfg["scene_groups"], 4096)]
    n = len(groups)
    generator = torch.Generator().manual_seed(290927)
    pose = torch.zeros(n, 7)
    pose[:, 6] = 1.
    env = SimpleNamespace(cfg=cfg, num_envs=n, device="cpu", scene_groups=groups,
        task_actor_update=actor, stage_actor_update=99999, global_actor_update=99999,
        training_transitions=123456789, policy_dt=.02,
        commands=torch.zeros(n, 3), command_target=torch.zeros(n, 2),
        mode=torch.zeros(n, dtype=torch.long), episode_length_buf=torch.zeros(n, dtype=torch.long),
        episode_limits=torch.zeros(n, dtype=torch.long), command_clock=torch.zeros(n), height_clock=torch.zeros(n),
        push_clock=torch.zeros(n), push_enabled=torch.zeros(n, dtype=torch.bool),
        random=lambda count: torch.rand(count, generator=generator), generator=generator,
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))))
    env.skills = SkillCommands(env)
    env.skills.sample(torch.arange(n))
    return env


def members(env, name):
    return torch.tensor([i for i, group in enumerate(env.scene_groups) if group == name])


def test_monotonic_difficulty_preserves_lower_tasks_and_saturates_only_at_final_target():
    settings = difficulty_settings(2600)
    states = [difficulty_envelope(settings, count) for count in range(2500, 3101)]
    for left, right in zip(states, states[1:]):
        assert left["grade"] <= right["grade"]
        assert left["vx_max_m_s"] <= right["vx_max_m_s"]
        assert left["yaw_max_rad_s"] <= right["yaw_max_rad_s"]
        assert left["motion_height_range_m"][0] >= right["motion_height_range_m"][0]
        assert left["motion_height_range_m"][1] <= right["motion_height_range_m"][1]
    assert difficulty_envelope(settings, 2660)["vx_max_m_s"] == 2.5
    assert difficulty_envelope(settings, 2960) == difficulty_envelope(settings, 9000)
    assert states[-1]["motion_height_range_m"] == [.23, .43]
    assert states[-1]["yaw_max_rad_s"] == 3 * math.tau


@pytest.mark.parametrize("delta,vx,yaw,height", [
    (0, 2., 1., [.28, .35]), (120, 3., 1.625, [.25, .38]),
    (240, 4., 2.25, [.24, .40]), (360, 5., 3., [.23, .43])])
def test_actual_sampler_applies_each_grade_and_retains_all_static_height_anchors(delta, vx, yaw, height):
    env = environment(actor=2600 + delta)
    report = env.skills.verify_difficulty_samples()
    assert report["sample_check"] == "passed" and report["actor_update"] == 2600 + delta
    assert torch.allclose(env.skills.command_base[members(env, "forward_5"), 0], torch.tensor(vx))
    assert torch.allclose(env.skills.command_base[members(env, "rotate_3rps"), 1], torch.tensor(yaw * math.tau))
    targets = env.skills.height_target[members(env, "forward_5__height_train")]
    assert targets.min() >= height[0] - 1e-6 and targets.max() <= height[1] + 1e-6
    for mm in (230, 250, 280, 305, 320, 350, 380, 400, 430):
        command = env.skills.command_base[members(env, f"stand_{mm}mm")]
        assert len(command) == 55 and not command[:, :2].any()
        assert torch.allclose(command[:, 2], torch.tensor(mm / 1000))
    assert torch.allclose(env.skills.command_base[members(env, "forward_05"), 0], torch.tensor(.5))
    assert env.skills.specs["forward_5"]["command"][0] == 5.


def test_sampling_is_continuous_directionally_symmetric_and_retains_legacy_rng_at_full_target():
    graded = environment(actor=2960)
    legacy = environment(actor=2960, legacy=True)
    for key in ("commands", "command_target", "command_clock", "episode_limits"):
        torch.testing.assert_close(getattr(graded, key), getattr(legacy, key), rtol=0, atol=0)
    assert torch.equal(graded.generator.get_state(), legacy.generator.get_state())
    # The inherited high-speed named anchors are fixed targets. For samplers
    # with amplitude variation, grade scaling must keep their continuous law.
    spec = dict(kind="forward", command=[5., 0., .305], sample_amplitude=True,
                amplitude_range=[.05, 1.])
    draw = torch.Generator().manual_seed(35)
    random = lambda count: torch.rand(count, generator=draw)
    command = torch.tensor([spec["command"]] * 4096)
    sample_planar_command(command, spec, random)
    original = command.clone()
    scale_planar_command(command, spec, difficulty_envelope(difficulty_settings(2600), 2600))
    forward, backward = command[:, 0], -command[:, 0]
    assert torch.unique(forward).numel() > 2 and forward.max() <= 2
    assert torch.unique(backward).numel() > 2 and backward.min() >= -2
    assert not (forward == 2).any()
    torch.testing.assert_close(command[:, 0] / 2., original[:, 0] / 5.)


def test_critic_consumption_other_clocks_and_episode_resets_cannot_advance_difficulty():
    env = environment()
    first = env.skills.difficulty_metrics()
    env.global_actor_update += 100000
    env.stage_actor_update += 100000
    env.training_transitions += 100000
    env.skills.sample(torch.arange(env.num_envs))
    assert env.skills.difficulty_metrics() == first
    env.task_actor_update += 180
    env.skills.sample(torch.arange(env.num_envs))
    assert env.skills.verify_difficulty_samples()["envelope"]["vx_max_m_s"] == 3.5
    assert env.skills.difficulty_metrics()["/curriculum/locomotion/grade"] == 2.5


def test_command_resampling_preserves_height_episode_and_real_braking_phase():
    env = environment()
    old_height = env.commands[:, 2].clone()
    old_delay = env.skills.start_stop_delay.clone()
    old_limits = env.episode_limits.clone()
    env.task_actor_update += 120
    env.episode_length_buf[:] = 500
    env.skills.sample(torch.arange(env.num_envs), reset_height=False)
    torch.testing.assert_close(env.commands[:, 2], old_height)
    torch.testing.assert_close(env.skills.start_stop_delay, old_delay)
    torch.testing.assert_close(env.episode_limits, old_limits)
    ids = members(env, "start_stop_5")
    assert torch.allclose(env.skills.command_base[ids, 0], torch.tensor(3.))
    assert env.skills.verify_difficulty_samples()["sample_check"] == "passed"


def test_exact_evaluation_keeps_full_requested_commands_and_no_difficulty_metrics():
    env = environment(exact=True)
    assert env.skills.verify_difficulty_samples()["sample_check"] == "exact_evaluation_bypass"
    assert env.skills.difficulty_metrics() == {}
    assert (env.commands[members(env, "forward_5"), 0] == 5).all()
    assert torch.allclose(env.commands[members(env, "rotate_3rps"), 1], torch.tensor(3 * math.tau))
    assert torch.allclose(env.commands[members(env, "stand_230mm"), 2], torch.tensor(.23))


@pytest.mark.parametrize("fault", ["disabled", "missing_clock", "consumption_clock", "boolean_origin", "extra_field"])
def test_malformed_rule_never_silently_defaults(fault):
    settings = difficulty_settings(2600)
    if fault == "disabled":
        settings["enabled"] = False
    elif fault == "missing_clock":
        del settings["clock"]
    elif fault == "consumption_clock":
        settings["clock"] = "successful_updates_total"
    elif fault == "boolean_origin":
        settings["origin_actor_update"] = True
    else:
        settings["reset_on_resume"] = True
    with pytest.raises(ValueError):
        validate_difficulty(settings)


@pytest.mark.parametrize("fault", ["velocity_bypass", "height_bypass", "nonfinite"])
def test_runtime_probe_detects_actual_sampler_bypass_even_when_module_enabled(fault):
    env = environment()
    if fault == "velocity_bypass":
        env.skills.command_base[members(env, "forward_5"), 0] = 5.
    elif fault == "height_bypass":
        env.skills.height_target[members(env, "forward_5__height_train")] = .43
    else:
        env.skills.command_base[0, 0] = float("nan")
    with pytest.raises(ValueError):
        env.skills.verify_difficulty_samples()
