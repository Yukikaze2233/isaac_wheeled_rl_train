"""CPU checks of explicit event ownership, paid commands and failure priority.

Fixture heights and hold times are arithmetic inputs, not selected training
parameters. No scanner, physics application or learned climb is qualified.
"""
from types import SimpleNamespace

import pytest
import torch

from tests.simulation.runtime.test_flat_reward_runtime import env, methods
from wheeled_tasks.chassis.application.runtime.flat_episode_runtime import FlatEpisodeRuntime
from wheeled_tasks.chassis.observations.actor import build_scut35
from wheeled_tasks.chassis.tasks.commands.stepup_command import StepUpCommand, VERSION


def settings(**changes):
    return {"version": VERSION, "enabled": True,
            "event_source": "external_step_wall_events", "height_bias_m": .16,
            "hold_seconds": .04, "maximum_height_m": .40, **changes}


def command(count=2, **changes):
    return StepUpCommand(count, "cpu", settings(**changes))


def events(owner, step=(), wall=()):
    steps, walls = torch.zeros_like(owner.pending_step), torch.zeros_like(owner.pending_wall)
    steps[list(step)], walls[list(wall)] = True, True
    owner.submit(steps, walls)


def advance(owner, height, tick, eligible=None):
    raw = torch.as_tensor(height, dtype=torch.float32)
    allow = torch.ones_like(owner.pending_step) if eligible is None else torch.tensor(eligible)
    return owner.advance(raw, allow, policy_tick=tick)


def test_latch_uses_raw_height_retrigger_does_not_accumulate_and_expiry_uses_current_height():
    owner = command()
    events(owner, step=[0, 1])
    torch.testing.assert_close(advance(owner, [.22, .30], 1), torch.tensor([.38, .40]))
    assert owner.remaining.tolist() == [2, 2]
    torch.testing.assert_close(advance(owner, [.20, .28], 2), torch.tensor([.38, .40]))
    events(owner, step=[0])
    torch.testing.assert_close(advance(owner, [.20, .28], 3), torch.tensor([.36, .28]))
    assert owner.remaining.tolist() == [2, 0]
    torch.testing.assert_close(advance(owner, [.25, .26], 4), torch.tensor([.36, .26]))
    torch.testing.assert_close(advance(owner, [.27, .24], 5), torch.tensor([.27, .24]))
    assert not owner.active.any()


def test_wall_wins_over_accumulated_step_events_and_ineligible_events_are_discarded():
    owner = command(3)
    events(owner, step=[0, 1, 2])
    events(owner, wall=[0])
    torch.testing.assert_close(advance(owner, [.22] * 3, 1, [True, True, False]),
                               torch.tensor([.22, .38, .22]))
    assert owner.wall_event.tolist() == [True, False, False]
    assert owner.step_event.tolist() == [False, True, False]
    assert not owner.pending_step.any() and not owner.pending_wall.any()
    advance(owner, [.22] * 3, 2, [True, False, True])
    assert not owner.active.any() and not owner.step_event.any()


def test_partial_reset_discards_pending_events_and_retains_other_lane_clock():
    owner = command()
    events(owner, step=[0, 1])
    advance(owner, [.22, .22], 8)
    events(owner, step=[0], wall=[0, 1])
    owner.reset(torch.tensor([0]))
    assert owner.remaining.tolist() == [0, 2]
    assert owner.pending_wall.tolist() == [False, True]
    torch.testing.assert_close(advance(owner, [.24, .24], 9), torch.tensor([.24, .24]))
    assert owner.wall_event.tolist() == [False, True]


def test_readonly_observations_do_not_advance_hold_and_zero_hold_never_changes_height():
    owner = command()
    events(owner, step=[0])
    advance(owner, [.22, .22], 1)
    for _ in range(5):
        torch.testing.assert_close(owner.effective_height(torch.tensor([.20, .22])), torch.tensor([.38, .22]))
        assert owner.remaining.tolist() == [2, 0]
    snapshot = owner.diagnostics()
    snapshot["hold_remaining_policy_ticks"].zero_()
    assert owner.remaining.tolist() == [2, 0]
    zero = command(hold_seconds=0.)
    events(zero, step=[0], wall=[1])
    torch.testing.assert_close(advance(zero, [.22, .25], 1), torch.tensor([.22, .25]))
    assert zero.step_event[0] and zero.wall_event[1] and not zero.active.any()


@pytest.mark.parametrize("tick", [1, 3, True, -1])
def test_duplicate_or_nonconsecutive_policy_advance_is_rejected_without_consuming_state(tick):
    owner = command()
    events(owner, step=[0])
    advance(owner, [.22, .22], 1)
    before = owner.diagnostics()
    with pytest.raises(ValueError, match="consecutive policy tick"):
        advance(owner, [.20, .20], tick)
    for name, value in owner.diagnostics().items():
        assert torch.equal(value, before[name])


def production_env():
    e = env()
    e.cfg.update(formal_closure_monitor_only=True, closure_gap_termination_m=None)
    e.generator = torch.Generator().manual_seed(23)
    e.policy_dt, e.tick = .02, 0
    e.mode = torch.ones(2, dtype=torch.long)
    e.targets = torch.zeros(2, 2)
    e.jump_requested = torch.zeros(2, dtype=torch.bool)
    e.raw_height = torch.zeros(2)
    e._flat_heading = lambda: torch.zeros(2)
    e._flat_command_iteration = lambda: 6000
    e.v7_stepup = command()
    e.flat_episode = FlatEpisodeRuntime(2, "cpu", None, None,
        [f"axis{i}" for i in range(18)], birth_rejection={"mode": "abort"})
    requests = torch.tensor([[.5, 0., .25], [-.5, 0., -.25]])
    e.flat_episode.commands.set_fixed_requests(torch.arange(2), requests,
        torch.tensor([.22, .22]), evaluation_only=True)
    e.flat_episode.commands.reset(torch.arange(2), torch.zeros(2),
        torch.zeros(2, dtype=torch.bool), e.generator, iteration=6000)
    e.flat_episode.episode_id.zero_()
    e.flat_episode.publish(e.commands, e.mode, e.targets, e.jump_requested)
    e.raw_height.copy_(e.commands[:, 2])
    scope = methods("submit_stepup_events", "_advance_flat_episode")
    e.submit = lambda step, wall, ids: scope["submit_stepup_events"](e, step, wall, episode_ids=ids)
    e._advance_flat_episode = lambda done, terminated, extras: scope["_advance_flat_episode"](e, done, terminated, extras)
    e.get_observations = lambda: {"policy": build_scut35(
        e.robot.data.root_com_ang_vel_b.torch, e.robot.data.projected_gravity_b.torch,
        e.commands, torch.zeros(2, 6), torch.zeros(2, 6), torch.zeros(2, 6), torch.zeros(6),
        e.jump_requested, torch.zeros(2), torch.zeros(2))}
    return e


def paid_interval(e):
    e.flat_episode.begin_policy()
    for _ in range(4):
        e.flat_episode.advance_physics(.005)
    e.tick += 1
    return e.finish()


def test_actual_returned_command_and_actor35_change_only_after_paying_observed_command():
    baseline, candidate = production_env(), production_env()
    observed = candidate.get_observations()["policy"].clone()
    candidate.submit(torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool),
                     candidate.flat_episode.episode_id.clone())
    assert torch.equal(candidate.get_observations()["policy"], observed)
    _, old_reward, _, old_extras = paid_interval(baseline)
    frame, reward, done, extras = paid_interval(candidate)
    assert not done.any() and torch.equal(reward, old_reward)
    assert torch.equal(extras["diagnostics"]["commands"][:, 2], torch.tensor([.22, .22]))
    assert not extras["diagnostics"]["command_override_active"].any()
    torch.testing.assert_close(frame["policy"][:, 3], torch.tensor([1.9, 1.1]))
    assert frame["policy"].shape == (2, 35)
    assert torch.equal(frame["policy"][:, :3], observed[:, :3])
    assert torch.equal(frame["policy"][:, 28:], observed[:, 28:])
    assert torch.equal(candidate.raw_height, torch.tensor([.22, .22]))
    assert extras["returned_episode_state"]["v7_stepup_command"]["height_override_active"].tolist() == [True, False]
    assert torch.equal(extras["policy_learning_mask"], old_extras["policy_learning_mask"])
    assert extras["policy_learning_mask"].all()
    # The following paid interval targets the same height the actor observed;
    # the untouched lane keeps exact source payments and no new reward term.
    _, _, _, next_extras = paid_interval(candidate)
    _, _, _, next_baseline = paid_interval(baseline)
    torch.testing.assert_close(next_extras["diagnostics"]["commands"][:, 2], torch.tensor([.38, .22]))
    assert next_extras["diagnostics"]["command_override_active"].tolist() == [True, False]
    assert next_extras["native_reward_components"].keys() == next_baseline["native_reward_components"].keys()
    for name, value in next_extras["native_reward_components"].items():
        assert value[1] == next_baseline["native_reward_components"][name][1]


def test_production_event_endpoint_rejects_stale_episode_ids_and_disabled_owner():
    e = production_env()
    step, wall = torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="stale events"):
        e.submit(step, wall, e.flat_episode.episode_id + 1)
    assert not e.v7_stepup.pending_step.any()
    e.v7_stepup = None
    with pytest.raises(RuntimeError, match="not explicitly enabled"):
        e.submit(step, wall, e.flat_episode.episode_id)


@pytest.mark.parametrize("cause", [None, "numeric", "knee", "spring"])
def test_actual_wall_timeout_suppresses_source_failure_but_never_numeric_or_mechanical_failure(cause):
    e = env()
    e.cfg.update(formal_closure_monitor_only=True, closure_gap_termination_m=None)
    e.v7_stepup = command()
    events(e.v7_stepup, wall=[0])
    advance(e.v7_stepup, [.22, .22], 1)
    e.scut_reward.terminal_counter[0] = 19
    e.scut_reward.net_contact[0, 0, 2] = 2.
    if cause == "numeric":
        e.scut_reward.joint_velocity[0, 0] = 501.
    elif cause == "knee":
        e.robot.data.joint_pos.torch[0, 0] = 2.
    elif cause == "spring":
        e.v5.spring_state = lambda q, dq: (torch.tensor([[.082, 0.], [0., 0.]]), torch.zeros(2, 2))
    _, reward, done, extras = e.finish()
    assert done.tolist() == [True, False]
    assert bool(extras["time_outs"][0]) == (cause is None)
    assert bool(extras["diagnostics"]["reasons"]["wall_blocked"][0]) == (cause is None)
    assert extras["native_reward_components"]["termination"][0].item() == (0. if cause in (None, "numeric") else -4.)
    if cause == "numeric":
        assert reward[0] == 0.
    assert not extras["time_outs"][1]


def test_returned_stepup_is_cancelled_during_recovery_motion_hold_without_changing_masks():
    e = production_env()
    e.recovery = SimpleNamespace(source_flat_handoff=True,
        policy_motion_hold=torch.tensor([True, False]),
        override_commands=lambda: None, bootstrap_mask=lambda terminated: ~terminated)
    e.submit(torch.tensor([True, True]), torch.zeros(2, dtype=torch.bool), e.flat_episode.episode_id)
    e.flat_episode.begin_policy()
    for _ in range(4):
        e.flat_episode.advance_physics(.005)
    e.tick = 1
    extras = {"log": {}}
    e._advance_flat_episode(torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), extras)
    torch.testing.assert_close(e.commands[:, 2], torch.tensor([.22, .38]))
    assert not e.v7_stepup.pending_step.any()
    assert e.flat_episode.learning_mask.all() and extras["policy_bootstrap_mask"].all()
