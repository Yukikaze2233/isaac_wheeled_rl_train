"""Actor-clock lessons retain the native physical prefix and truthful outcomes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_recovery_pose_curriculum import make_env
from test_recovery_training_integration import rig
from test_recovery_takeover_integration import activate, advance, initialize, rebuild, v8_rig
from wheeled_tasks.chassis.recovery import RecoveryPhase as Phase
from wheeled_tasks.chassis.recovery_prefix_curriculum import (
    LESSONS, RecoveryPrefixCurriculum, prefix_settings, release_seconds,
    validate_prefix_curriculum,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def table():
    return json.loads((ROOT / "contracts/v6_recovery_profiles_v2.json").read_text())


def learner(table, *, count=8192, only_profile="front_down_90", clock=2294):
    env = make_env(table, count=count, only_profile=only_profile)
    env.skills.specs["lane_0"]["recovery_followup_command"] = [.3, .2, .305]
    sampler = RecoveryPrefixCurriculum(env, table, prefix_settings())
    sampler.reset(torch.arange(count), clock)
    return env, sampler


def test_actor_origin_probability_clock_and_permanent_rehearsal(table):
    _, sampler = learner(table)
    assert not sampler.level_ids.eq(2).any()
    metrics = sampler.metrics(2294)
    assert [metrics[f"/recovery/lesson/{name}/probability"] for name in LESSONS] == [.95, .05, 0.]
    sampler.reset(torch.arange(sampler.env.num_envs), 2294 + 6000)
    fractions = torch.bincount(sampler.level_ids, minlength=3) / sampler.env.num_envs
    assert fractions.tolist() == pytest.approx([.20, .30, .50], abs=.02)
    hard = sampler.profile_index["front_down_90"]
    assert torch.equal(sampler.profile_ids.eq(hard), sampler.full_chain_mask)
    assert sampler.metrics(2294 + 3000)["/recovery/lesson/relative_actor_updates"] == 3000
    assert sampler.settings["origin_actor_update"] == 2294


@pytest.mark.parametrize("clock,expected", [(0, .03), (2294, .03),
    (3294, .085), (3794, .115), (5294, .2), (99999, .2)])
def test_passive_release_clock_is_real_200hz_ticks(clock, expected):
    value = release_seconds(prefix_settings(), clock)
    assert value == pytest.approx(expected)
    assert value / .005 == pytest.approx(round(value / .005))


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(origin_actor_update=True),
    lambda s: s.update(state_source="dynamic_import"),
    lambda s: s.update(handoff_state="teleport_rl"),
    lambda s: s.update(blend_seconds=.02),
    lambda s: s.update(stable_seconds=.5),
    lambda s: s["keyframes"][1]["probabilities"].update(prefix_stand=.98, prefix_release=.01, original_chain=.01),
    lambda s: s["keyframes"][-1]["probabilities"].update(prefix_stand=0., prefix_release=.5, original_chain=.5),
    lambda s: s["release_schedule"][-1].update(seconds=.205),
    lambda s: s["release_schedule"][0].update(seconds=.031),
    lambda s: s["release_schedule"][-1].update(seconds=.02),
])
def test_bad_clocks_hidden_state_changes_and_gate_shortcuts_are_rejected(mutation):
    settings = prefix_settings()
    mutation(settings)
    with pytest.raises(ValueError):
        validate_prefix_curriculum(settings)


def test_lesson_only_changes_reset_local_release_and_visible_followup(table):
    env, sampler = learner(table, clock=5294)
    owner = SimpleNamespace(initial_release=torch.zeros(env.num_envs),
        followup_command=torch.zeros(env.num_envs, 3))
    sampler.apply_lesson(owner, torch.arange(env.num_envs))
    stand, release, original = (sampler.level_ids == i for i in range(3))
    assert stand.any() and release.any() and original.any()
    assert owner.followup_command[stand, :2].eq(0).all()
    torch.testing.assert_close(owner.followup_command[~stand],
        sampler.original_followup[~stand])
    assert owner.initial_release[stand].eq(0).all()
    assert owner.initial_release[release].eq(.2).all()
    torch.testing.assert_close(owner.initial_release[original],
        sampler.initial_release_bank[sampler.profile_ids[original]])
    assert env.nominal.eq(0).all()
    assert sampler.runtime_state()["imported_dynamic_bank"] is False
    state = sampler.runtime_state()
    state["settings"]["origin_actor_update"] = 0
    assert sampler.settings["origin_actor_update"] == 2294


@pytest.mark.parametrize("name,height", [("step200_lip_near_shell", .2),
    ("double_step_first_riser_split", .2), ("double_step_second_side_top", .45)])
def test_lessons_keep_target_platform_and_do_not_cross_ground(name, height, table):
    _, sampler = learner(table, only_profile=name, clock=8294)
    nonoriginal = ~sampler.full_chain_mask
    names = {sampler.profile_names[i] for i in sampler.profile_ids[nonoriginal]}
    assert len(names) == 1 and next(iter(names)).endswith("upright_top")
    assert sampler.required_platform_height_bank[sampler.profile_ids].eq(height).all()
    assert sampler.profile_ids[sampler.full_chain_mask].eq(sampler.profile_index[name]).all()


def test_outcomes_count_once_per_real_episode_and_never_credit_l3_with_l1(table):
    env, sampler = learner(table, clock=8294)
    done = torch.ones(env.num_envs, dtype=torch.bool)
    success = sampler.level_ids < 2
    failure = sampler.level_ids == 2
    timeout = torch.zeros_like(done)
    sampler.record_outcomes(done, success, failure, timeout)
    before = sampler.outcome_counts.clone()
    sampler.record_outcomes(done, success, failure, timeout)
    assert torch.equal(before, sampler.outcome_counts)
    assert before.sum(dim=1).tolist() == [2 * int((sampler.level_ids == i).sum()) for i in range(3)]
    metrics = sampler.lesson_metrics(success)
    assert metrics["/recovery/lesson/prefix_stand/success_fraction"] == 1.
    assert metrics["/recovery/full_chain_success"] == 0.
    assert metrics["/recovery/full_chain_population"] == failure.sum()
    # An ordinary lane has neither ownership nor outcome counts.
    ordinary_env = make_env(table)
    ordinary = RecoveryPrefixCurriculum(ordinary_env, table, prefix_settings())
    ordinary.reset(torch.arange(10), 2294)
    zero = torch.zeros(10, dtype=torch.bool)
    ordinary.record_outcomes(torch.ones_like(zero), torch.ones_like(zero), zero, zero)
    assert ordinary.outcome_counts[:, 0].sum() == 8
    assert ordinary.lesson_metrics(torch.ones_like(zero))["/recovery/full_chain_population"] == 0
    assert ordinary.lesson_metrics(torch.ones_like(zero))["/recovery/full_chain_success"] == 0.


def test_partial_reset_retains_other_lessons_state_and_outcome_account(table):
    _, sampler = learner(table, count=10, clock=8294)
    ids = torch.tensor([0, 3])
    before = tuple(t.clone() for t in (sampler.profile_ids, sampler.level_ids,
        sampler.selected_followup, sampler.selected_release))
    sampler.outcome_recorded[:] = True
    sampler.reset(ids, 2294)
    keep = torch.tensor([1, 2, 4, 5, 6, 7, 8, 9])
    for tensor, old in zip((sampler.profile_ids, sampler.level_ids,
            sampler.selected_followup, sampler.selected_release), before):
        torch.testing.assert_close(tensor[keep], old[keep])
    assert not sampler.outcome_recorded[ids].any() and sampler.outcome_recorded[keep].all()


@pytest.fixture
def prefix_rig(v8_rig):
    env, _, table = v8_rig
    env.cfg["recovery_training"].pop("pose_curriculum")
    env.cfg["recovery_training"]["prefix_curriculum"] = prefix_settings()
    env.cfg.pop("recovery_takeover")
    env.task_actor_update = 2294
    # First recovery lane is intentionally a forward original case. L1 must
    # override its followup visibly, while retaining its original case identity.
    return env, rebuild(env, env.scene_groups, env.kinds), table


def test_real_adapter_l1_keeps_native_script_and_stand_followup(prefix_rig):
    env, recovery, _ = prefix_rig
    before = deepcopy((env.scene_groups, env.skills.specs))
    initialize(env, recovery)
    sampler = recovery.pose_curriculum
    assert sampler.level_ids[0] == 0
    assert sampler.profile_names[sampler.profile_ids[0]] == "upright"
    assert recovery.initial_release[0] == 0
    assert recovery.followup_command[0, :2].eq(0).all()
    assert recovery.fsm.phase[0] == Phase.SELECT
    recovery.begin_policy()
    assert not recovery.pure_rl[0] and not recovery.learning_mask[0]
    assert (env.scene_groups, env.skills.specs) == before
    assert sampler.diagnostics()["recovery_prefix_lesson"][0]
    assert not sampler.full_chain_mask[0]


def test_physical_prefix_has_no_mid_episode_state_write_and_40_blend_ticks(prefix_rig):
    env, recovery, _ = prefix_rig
    q, dq = activate(env, recovery, Phase.BLEND)
    physical = [env.robot.data.root_link_pose_w.torch, env.robot.data.joint_pos.torch,
        env.robot.data.root_link_lin_vel_w.torch]
    frozen = [value.clone() for value in physical]
    rng = env.generator.get_state().clone()
    samples = recovery.pose_curriculum.profile_reset_counts.clone()
    env.task_actor_update += 6000
    for tick in range(40):
        if tick % 4 == 0:
            recovery.begin_policy()
        advance(recovery, q, dq)
        recovery.torque(q, dq, torch.ones_like(q))
        assert not recovery.learning_mask[0] and not recovery.failed[0]
    assert recovery.fsm.phase[0] == Phase.RL
    recovery.begin_policy()
    assert recovery.learning_mask[0]
    for tensor, expected in zip(physical, frozen):
        torch.testing.assert_close(tensor, expected)
    assert torch.equal(env.generator.get_state(), rng)
    assert torch.equal(recovery.pose_curriculum.profile_reset_counts, samples)
    assert recovery.followup_command[0, :2].eq(0).all()


def test_fixed_eval_uses_original_pose_release_followup_and_no_lesson(prefix_rig):
    env, _, table = prefix_rig
    env.task_actor_update += 6000
    recovery = rebuild(env, env.scene_groups, env.kinds, exact=True)
    rng = env.generator.get_state().clone()
    initialize(env, recovery)
    assert recovery.pose_curriculum is None
    profile = table["profiles"]["front_down_90"]
    assert recovery.initial_release[0] == pytest.approx(profile["initial_release_s"])
    assert recovery.followup_command[0, 0] == pytest.approx(.3)
    assert recovery.profile_names[0] == "front_down_90"
    assert torch.equal(env.generator.get_state(), rng)
