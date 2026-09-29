"""Actor-owned terminal transitions and privileged recovery outcome grading."""
import math

import pytest
import torch

from test_recovery_training_integration import rig
from test_recovery_rollout_mask import algorithm, observation
from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
from wheeled_tasks.chassis.recovery import RecoveryPhase as Phase
from wheeled_tasks.chassis.task import episode_outcome


def activate(env, recovery, phase=Phase.RL):
    q = env.nominal[:, env.ids].clone()
    recovery.release_finished[:] = True
    recovery.fsm.reset(torch.arange(2), q, recovery.enabled)
    recovery.fsm.phase[0] = phase
    recovery.handover_count[0] = int(phase == Phase.RL)
    dq = torch.zeros_like(q)
    env.v5.motor_efforts(q, dq, q[:, list(env.v5.LEGS)], dq[:, list(env.v5.WHEELS)])
    return q, dq


def lose_upright(recovery, q, dq):
    gravity = torch.tensor([[math.sin(math.pi / 3), 0., -.5], [0., 0., -1.]])
    recovery.fsm.update(q, dq, gravity, torch.zeros(2, 3), torch.full((2,), .305),
                        torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))


def terminal_outcome(recovery):
    reasons = {"boundary": torch.zeros(2, dtype=torch.bool), **recovery.failure_reasons()}
    return episode_outcome(reasons, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize("failure_tick", [0, 1, 9, 19])
def test_actor_failure_keeps_elapsed_interval_but_never_bootstraps(rig, failure_tick):
    env, recovery = rig
    q, dq = activate(env, recovery)
    recovery.begin_policy()
    for tick in range(20):
        if tick == failure_tick:
            lose_upright(recovery, q, dq)
        effort = recovery.torque(q, dq, torch.ones_like(q))
        if tick >= failure_tick:
            assert effort[0].eq(0.).all()  # Existing terminal disarm is unchanged.
    assert recovery.learning_mask.tolist() == [True, True]
    assert recovery.fsm.failure_code[0] == 5
    reasons = recovery.failure_reasons()
    assert reasons["recovery_actor_failure"].tolist() == [True, False]
    assert not reasons["recovery_script"].any()
    terminated, timeouts, success = terminal_outcome(recovery)
    assert terminated.tolist() == [True, False] and not timeouts.any() and not success.any()
    assert recovery.bootstrap_mask(terminated).tolist() == [False, True]
    learning = recovery.learning_mask.clone()
    recovery.reset(torch.tensor([0]))
    assert learning.tolist() == [True, True]
    assert recovery.bootstrap_mask(terminated).tolist() == [False, True]


@pytest.mark.parametrize("initial_phase", [Phase.PREPARE, Phase.BLEND])
def test_any_script_or_blend_tick_excludes_even_a_later_actor_failure(rig, initial_phase):
    env, recovery = rig
    q, dq = activate(env, recovery, initial_phase)
    recovery.begin_policy()
    for tick in range(20):
        if tick == 10:
            recovery.fsm.phase[0] = Phase.RL
            recovery.handover_count[0] = 1
        if tick == 19:
            lose_upright(recovery, q, dq)
        recovery.torque(q, dq, torch.ones_like(q))
    assert recovery.learning_mask.tolist() == [False, True]
    assert recovery.failure_reasons()["recovery_actor_failure"][0]
    recovery.begin_policy()  # A terminal state cannot create another actor sample.
    assert recovery.learning_mask.tolist() == [False, True]


def test_failed_script_is_never_enabled_for_learning(rig):
    env, recovery = rig
    q, dq = activate(env, recovery, Phase.PREPARE)
    recovery.begin_policy()
    recovery.fsm._fail(torch.tensor([True, False]), 6)
    recovery.torque(q, dq, torch.ones_like(q))
    assert recovery.learning_mask.tolist() == [False, True]
    assert recovery.failure_reasons()["recovery_script"].tolist() == [True, False]
    assert not recovery.failure_reasons()["recovery_actor_failure"].any()


@pytest.mark.parametrize("hold_intervals", [19, 29, 31, 43])
def test_early_sensor_unlock_terminates_without_changing_fsm_or_filling_credit(rig, hold_intervals):
    env, recovery = rig
    activate(env, recovery)
    recovery.fsm.motion_released[0] = False
    data = (torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
            torch.tensor([[0., 0., -1.]]).repeat(2, 1), torch.zeros(2, 3),
            torch.full((2,), .305), torch.zeros(2, 3))
    for _ in range(hold_intervals):
        recovery.outcome(*data)
    recovery.begin_policy()
    targets = recovery.fsm.script_targets.clone()
    recovery.fsm.motion_released[0] = True
    recovery.outcome(*data)
    peak = (hold_intervals + 1) * .02  # Boundary interval still belongs to hold.
    assert recovery.premature_release.tolist() == [True, False]
    assert recovery.failed.tolist() == [True, False]
    assert recovery.diagnostics()["recovery_failure_code"][0] == 7
    assert recovery.fsm.phase[0] == Phase.RL and recovery.fsm.motion_released[0]
    assert recovery.fsm.failure_code[0] == 0  # Truth has not entered the controller.
    torch.testing.assert_close(recovery.fsm.script_targets, targets)
    assert recovery.learning_mask.tolist() == [True, True]
    terminated, timeouts, _ = terminal_outcome(recovery)
    assert terminated[0] and not timeouts[0] and not recovery.bootstrap_mask(terminated)[0]
    assert recovery.failure_reasons()["recovery_premature_release"][0]
    for _ in range(100):
        recovery.outcome(*data)
    assert recovery.true_stable_peak[0] == pytest.approx(peak, abs=1e-6)
    assert recovery.followup_seconds[0] == 0. and not recovery.true_success[0]
    recovery.reset(torch.tensor([0]))
    assert not recovery.premature_release.any()


def test_native_ppo_keeps_actor_terminal_penalty_and_propagates_it_backward(rig):
    env, recovery = rig
    q, dq = activate(env, recovery)
    alg = algorithm(per_batch=True)
    adapter = attach_recovery_rollout_mask(alg, num_envs=2)
    for step in range(6):
        recovery.begin_policy()
        with torch.no_grad():
            alg.act(observation(2, step / 10))
        if step == 5:
            lose_upright(recovery, q, dq)
        for _ in range(20):
            recovery.torque(q, dq, torch.ones_like(q))
        terminated, _, _ = terminal_outcome(recovery)
        reward = torch.where(terminated, -200., 1.)
        next_obs = observation(2, (step + 1) / 10)
        alg.process_env_step(next_obs, reward, terminated,
            {"policy_learning_mask": recovery.learning_mask.clone(),
             "policy_bootstrap_mask": recovery.bootstrap_mask(terminated)})
    alg.compute_returns(next_obs)
    assert adapter.learning[-1, 0] and not adapter.bootstrap[-1, 0]
    assert alg.storage.returns[-1, 0, 0] == -200.
    assert alg.storage.returns[-2, 0, 0] == pytest.approx(1. + .9 * -200.)
    assert adapter.metrics["valid_transitions"] == 12
