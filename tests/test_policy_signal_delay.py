"""Policy lag uses physical samples and preserves each control clock."""
import json
from pathlib import Path

import pytest
import torch

from test_actor_sensor_noise import observation_rig
from support_scut_flat_scripted_recovery import rig
from wheeled_tasks.chassis.feedback_transport import FractionalFeedbackTransport
from wheeled_tasks.chassis.policy_signal_delay import KEY, SETTINGS, PolicySignalDelay
from wheeled_tasks.chassis.round_trip_transport import (
    EffectiveCommandDelay, RoundTripDelayBudget, command_settings, feedback_settings,
)
from wheeled_tasks.chassis.training_profile import profile_source_files, validate_profile_selection


ROOT = Path(__file__).resolve().parents[1]


def make_delay(count=8, probability=1.):
    cfg = json.loads((ROOT / 'configs/v7_formal_20261012/motion_retention_review.json').read_text())
    cfg[KEY] = dict(SETTINGS, extra_episode_probability=probability)
    return cfg, PolicySignalDelay(count, 'cpu', cfg, torch.Generator().manual_seed(51))


def sensors(count, value):
    result = torch.full((count, 18), float(value))
    result[:, 15:18] = torch.tensor([0., 0., -1.])
    return result


def test_episode_selection_is_one_joint_profile_with_independent_group_lags():
    cfg, delay = make_delay(4096, .5)
    delay.reset(delay.rows, sensors(4096, 0), torch.zeros(4096, dtype=torch.float64))
    assert .47 < delay.enabled.float().mean() < .53
    assert delay.observation_lags[~delay.enabled].eq(0).all()
    assert delay.target_lags[~delay.enabled].eq(0).all()
    assert delay.observation_lags[delay.enabled].unique().tolist() == [1, 2, 3]
    assert delay.target_lags[delay.enabled].unique().tolist() == [1, 2]
    assert (delay.observation_lags[:, 0] != delay.observation_lags[:, 1]).sum() > 1000
    assert 'src/wheeled_tasks/chassis/policy_signal_delay.py' in profile_source_files(cfg)


@pytest.mark.parametrize('field,value', [
    ('enabled', False), ('buffer_physics_samples', 3), ('clock', 'policy_steps'),
    ('observation_steps', [True, 2, 3]), ('target_steps', [1, 2, 3]),
    ('extra_episode_probability', float('nan')), ('extra_episode_probability', True),
    ('jitter_s', .001), ('critic_action_age', 'original_pd_time'),
])
def test_partial_or_ambiguous_delay_contracts_are_rejected(field, value):
    cfg, _ = make_delay()
    cfg[KEY][field] = value
    with pytest.raises(ValueError):
        validate_profile_selection(cfg)


def test_actual_sensor_age_includes_local_feedback_once_and_only_actor_is_extra_delayed():
    cfg, delay = make_delay(2)
    budget = RoundTripDelayBudget(2, 'cpu', cfg['round_trip_transport'], 51)
    budget.reset(budget.rows)
    budget.uplink_s[:] = torch.tensor([.001, .002], dtype=torch.float64)
    feedback = FractionalFeedbackTransport(2, 'cpu', .005, feedback_settings(), 51,
        round_trip_budget=budget)
    feedback.reset(delay.rows, sensors(2, 0))
    delay.reset(delay.rows, feedback.feedback, feedback.sensor_timestamp_s)
    delay.observation_lags[:] = torch.tensor([[1, 2, 3, 1], [3, 2, 1, 3]])
    for tick in range(1, 11):
        feedback.observe(sensors(2, tick))
        pd_input = feedback.feedback.clone()
        delay.observe(feedback.feedback, feedback.sensor_timestamp_s, tick)
        assert torch.equal(feedback.feedback, pd_input)
    expected = delay.observation_lags.double() * .005 + budget.uplink_s[:, None]
    torch.testing.assert_close(delay.observation_age_s, expected, rtol=0, atol=1e-16)
    assert delay.feedback[0, :6].tolist() == pytest.approx([8.8] * 6)
    assert delay.feedback[0, 6:12].tolist() == pytest.approx([7.8] * 6)
    assert delay.feedback[0, 12:15].tolist() == pytest.approx([6.8] * 3)
    before = delay.readback()
    state = delay.generator.get_state().clone()
    delay.readback()
    assert torch.equal(delay.generator.get_state(), state)
    assert torch.equal(before['sensor_timestamp_s'], delay.sensor_timestamp_s)


def test_target_lag_and_downlink_have_distinct_pd_sample_and_policy_generation_times():
    cfg, delay = make_delay(1)
    budget = RoundTripDelayBudget(1, 'cpu', cfg['round_trip_transport'], 5)
    budget.reset(budget.rows)
    budget.downlink_s.fill_(.001)
    command = EffectiveCommandDelay(1, 'cpu', .005, command_settings(), budget)
    delay.reset(delay.rows, sensors(1, 0), torch.zeros(1, dtype=torch.float64))
    delay.target_lags[:] = torch.tensor([[1, 2]])
    for tick in range(6):
        now = tick * .005
        generated = torch.tensor([(tick // 4) * .02], dtype=torch.float64)
        legs, wheels = delay.targets(torch.full((1, 4), float(tick)),
            torch.full((1, 2), float(10 * tick)), generated, tick)
        command.apply_torque(torch.ones(1, 6), pd_generated_s=now,
            policy_generated_s=generated, target_sample_s=delay.target_sample_s,
            target_policy_generated_s=delay.target_policy_generated_s)
        if tick == 5:
            assert legs.tolist()[0] == [4.] * 4
            assert wheels.tolist()[0] == [30.] * 2
            assert delay.target_sample_s.tolist()[0] == pytest.approx([.020, .015])
            assert delay.target_policy_generated_s.tolist()[0] == pytest.approx([.020, .000])
            assert command.applied_target_sample_s.tolist()[0] == pytest.approx([.019, .014])
            assert command.applied_target_policy_generated_s.tolist()[0] == pytest.approx([.016, .000])
            assert command.applied_origins_s.tolist()[0] == pytest.approx([.024, .020, .024])
            assert command.pd_age_s.item() == pytest.approx(.006)
            actual_target_age = command.applied_at_s[:, None] - command.applied_target_sample_s
            assert actual_target_age.tolist()[0] == pytest.approx([.011, .016])
        delay.observe(sensors(1, tick + 1), torch.full((1,), now + .005, dtype=torch.float64), tick + 1)


def test_partial_reset_has_no_sensor_or_target_leak_from_the_previous_episode():
    _, delay = make_delay(2)
    delay.reset(delay.rows, sensors(2, 0), torch.zeros(2, dtype=torch.float64))
    delay.observation_lags.fill_(3)
    delay.target_lags.fill_(2)
    for tick in range(5):
        delay.targets(torch.full((2, 4), float(tick)), torch.full((2, 2), float(tick)),
            torch.full((2,), tick * .005, dtype=torch.float64), tick)
        delay.observe(sensors(2, tick + 1), torch.full((2,), (tick + 1) * .005, dtype=torch.float64), tick + 1)
    before = delay.feedback[1].clone()
    delay.reset(torch.tensor([0]), sensors(2, 99), torch.full((2,), .025, dtype=torch.float64))
    assert delay.feedback[0, 0] == 99.
    assert torch.equal(delay.feedback[1], before)
    assert delay.observation_age_s[0].eq(0).all()
    legs, wheels = delay.targets(torch.full((2, 4), 77.), torch.full((2, 2), 88.),
        torch.full((2,), .025, dtype=torch.float64), 5)
    assert legs[0].eq(77).all() and wheels[0].eq(88).all()
    assert legs[1].eq(3).all() and wheels[1].eq(3).all()
    assert delay.target_sample_s[0].tolist() == [.025, .025]
    assert delay.target_sample_s[1].tolist() == [.015, .015]


def test_actual_observation_path_adds_noise_after_policy_lag_without_touching_pd_or_critic():
    env, *_ = observation_rig()
    _, delay = make_delay(env.num_envs)
    delay.reset(delay.rows, sensors(env.num_envs, .7), torch.zeros(env.num_envs, dtype=torch.float64))
    env.policy_signal_delay = delay
    pd = env._pc_feedback_transport.feedback.clone()
    observations = env.get_observations()
    assert torch.equal(env.actor_sensor_noise.input, delay.feedback)
    assert torch.equal(env._pc_feedback_transport.feedback, pd)
    assert torch.equal(observations['critic'][:, :28], env._actor_sensor_noise_clean_frame[:, :28])
    assert torch.equal(observations['policy'][:, :4], env._actor_sensor_noise_clean_frame[:, :4])
    assert torch.equal(observations['policy'][:, 22:], env._actor_sensor_noise_clean_frame[:, 22:])


def test_source_reward_finish_publishes_actual_policy_delay_metrics(rig):
    from support_scut_flat_scripted_recovery import ready, tick
    ready(rig)
    _, rig.policy_signal_delay = make_delay(2, .5)
    rig.policy_signal_delay.reset(torch.arange(2), sensors(2, 0), torch.zeros(2, dtype=torch.float64))
    _, _, _, info = tick(rig)
    assert '/policy_delay/actual_observation_age_mean_s' in info['log']
    assert info['log']['/policy_delay/extra_episode_fraction'] == rig.policy_signal_delay.enabled.float().mean()
