"""Analytic COM derivatives and pre-reset continuity; no physics or PPO."""
import math
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.observations.com_motion_measurement import ComMotionMeasurement
from wheeled_tasks.chassis.application.evaluation.flat_policy_evaluation import FlatEpisodeLedger
from wheeled_tasks.chassis.application.evaluation.flat_evaluation_runtime import PreResetLedgerAdapter
from wheeled_tasks.chassis.tasks.rewards.flat_reward import WEIGHTS


def fixed():
    return dict(schema='flat-policy-evaluation-v1', mode='fixed_measurement', actor_mode='mean',
        episodes_per_lane=1, environment_seed=17, actor_seed=31, policy_dt_s=.02,
        horizon_seconds=20., center_reference='root_link_origin_world', assessment='measurement_only',
        cases=[{'case_id': 'case', 'base_request': [1., 0., .3]}])


def begin(ledger, episode_id=0):
    ledger.begin_episode(0, episode_id, root_link_position_world=[0., 0., 0.],
                         yaw_rad=0., reset_metadata={'fixture': 'explicit initial state'})


def frame(tick=1, **changes):
    row = dict(episode_id=0, episode_tick=tick, reward=.02,
        payments={name: .02 if name == 'track_lin_vel_xy' else 0. for name in WEIGHTS},
        done=False, terminated=False, timeout=False, boundary=False, termination_reasons=[],
        learning_mask=True, bootstrap_mask=True, measurement_valid=True, command_override_active=False,
        effective_command=[1., 0., .3], base_request=[1., 0., .3],
        root_link_position_world=[tick * .02, 0., 0.], root_com_velocity_body=[1., 0., 0.],
        angular_velocity_body=[0., 0., 0.], yaw_rad=0.)
    row.update(changes)
    return row


def terminal(tick=1, **changes):
    return frame(tick, **{'done': True, 'terminated': True,
                         'termination_reasons': ['source_attitude_or_contact'], **changes})


def test_runtime_derivatives_reject_reset_invalid_and_nonfinite_pairs():
    measurement = ComMotionMeasurement(torch.zeros(2, 3))
    episode = torch.zeros(2, dtype=torch.long)
    valid = torch.ones(2, dtype=torch.bool)
    tags = lambda field: '/motion/world_root_COM_' + field
    velocity = torch.zeros(2, 3)
    assert measurement.observe(velocity, episode, valid)[tags('acceleration_samples_per_policy_frame')] == 0
    velocity[:, 0] = .02
    second = measurement.observe(velocity, episode, valid)
    assert second[tags('acceleration_frame_rms_m_s2')] == pytest.approx(1.)
    assert second[tags('jerk_samples_per_policy_frame')] == 0
    velocity[:, 0] = .06
    third = measurement.observe(velocity, episode, valid)
    assert third[tags('acceleration_frame_rms_m_s2')] == pytest.approx(2.)
    assert third[tags('jerk_frame_rms_m_s3')] == pytest.approx(50., rel=1e-6)
    # A native reset and an invalid physics frame independently break history.
    episode[0] += 1
    valid[1] = False
    velocity[:, 0] = 100.
    gap = measurement.observe(velocity, episode, valid)
    assert gap[tags('acceleration_samples_per_policy_frame')] == gap[tags('jerk_samples_per_policy_frame')] == 0
    valid.fill_(True)
    velocity[:, 0] = 100.02
    resumed = measurement.observe(velocity, episode, valid)
    assert resumed[tags('acceleration_samples_per_policy_frame')] == 1
    assert resumed[tags('jerk_samples_per_policy_frame')] == 0
    velocity[0, 0] = math.nan
    nonfinite = measurement.observe(velocity, episode, valid)
    assert nonfinite[tags('acceleration_samples_per_policy_frame')] == 1
    assert all(torch.isfinite(value) for value in nonfinite.values())


def test_ledger_uses_world_COM_velocity_and_terminal_frame_not_body_rotation():
    ledger = FlatEpisodeLedger(fixed(), ['case'], list(WEIGHTS))
    begin(ledger)
    for tick, yaw in ((1, 0.), (2, .2)):
        ledger.observe(0, frame(tick, yaw_rad=yaw,
            root_com_velocity_body=[math.cos(yaw), -math.sin(yaw), 0.],
            root_com_velocity_world=[1., 0., 0.]))
    record = ledger.observe(0, terminal(3, yaw_rad=.4,
        root_com_velocity_body=[math.cos(.4), -math.sin(.4), 0.],
        root_com_velocity_world=[1., 0., 0.], learning_mask=False))
    motion = record['measurements']['all']['world_COM_motion']
    assert motion['acceleration_samples'] == 2 and motion['jerk_samples'] == 1
    assert motion['acceleration_max_m_s2'] == motion['jerk_max_m_s3'] == 0.


def test_ledger_invalid_or_missing_world_state_breaks_derivative_history():
    protocol = fixed()
    protocol['episodes_per_lane'] = 2
    ledger = FlatEpisodeLedger(protocol, ['case'], list(WEIGHTS))
    begin(ledger)
    ledger.observe(0, frame(1, root_com_velocity_world=[0., 0., 0.]))
    ledger.observe(0, frame(2, measurement_valid=False,
        root_com_velocity_world=[math.nan] * 3))
    ledger.observe(0, frame(3, root_com_velocity_world=[100., 0., 0.]))
    ledger.observe(0, frame(4))  # Absent data cannot stand in for zero velocity.
    ledger.observe(0, frame(5, root_com_velocity_world=[-100., 0., 0.]))
    record = ledger.observe(0, terminal(6, root_com_velocity_world=[-99.98, 0., 0.]))
    motion = record['measurements']['all']['world_COM_motion']
    assert motion['acceleration_samples'] == 1 and motion['jerk_samples'] == 0
    assert motion['acceleration_rms_m_s2'] == pytest.approx(1.)
    assert motion['missing_world_velocity_frames'] == 1
    begin(ledger, episode_id=1)
    newborn = ledger.observe(0, terminal(episode_id=1, root_com_velocity_world=[1000., 0., 0.]))
    assert newborn['measurements']['all']['world_COM_motion']['acceleration_samples'] == 0


def test_production_finish_captures_world_COM_before_autoreset_and_uses_existing_log():
    from tests.simulation.runtime.test_flat_reward_runtime import env, field

    instance = env()
    ids = torch.zeros(2, dtype=torch.long)
    instance.flat_episode = SimpleNamespace(
        commands=SimpleNamespace(observe_motion_interval=lambda *args: None),
        learning_mask=torch.ones(2, dtype=torch.bool),
        metrics=lambda: {}, diagnostics=lambda: {'episode_id': ids.clone(),
            'base_request_command3': torch.zeros(2, 3), 'height': instance.commands[:, 2].clone(),
            'paid_override_active': torch.zeros(2, dtype=torch.bool)})
    instance._flat_heading = lambda: torch.zeros(2)
    world = torch.tensor([[1., 2., 3.]]).repeat(2, 1)
    instance.robot.data.root_com_lin_vel_w = field(world)
    instance.episode_length_buf = instance.episode_limits - 1
    def autoreset(done, terminated, extras):
        world.fill_(1000.)
        ids.add_(1)
    instance._advance_flat_episode = autoreset
    _, _, done, extras = instance.finish()
    assert done.all()
    assert torch.equal(extras['diagnostics']['root_COM_velocity_world'],
                       torch.tensor([[1., 2., 3.]]).repeat(2, 1))
    assert extras['log']['/motion/world_root_COM_acceleration_samples_per_policy_frame'] == 0
    _, _, _, newborn = instance.finish()
    assert newborn['log']['/motion/world_root_COM_acceleration_samples_per_policy_frame'] == 0


def test_adapter_keeps_pre_reset_world_COM_velocity_and_does_not_invent_a_pair():
    adapter = PreResetLedgerAdapter(fixed(), ['case'])
    adapter.begin_initial({'episode_id': torch.tensor([0]), 'episode_tick': torch.tensor([0])},
        root_position_world=torch.zeros(1, 3), yaw_rad=torch.zeros(1),
        reset_metadata=[{'fixture': 'explicit initial state'}])
    row = terminal(root_com_velocity_world=[.2, .3, .4])
    names = {'measurement_valid': 'measurement_valid_mask', 'learning_mask': 'policy_learning_mask',
             'bootstrap_mask': 'policy_bootstrap_mask', 'root_com_velocity_body': 'root_COM_velocity_body',
             'root_com_velocity_world': 'root_COM_velocity_world', 'yaw_rad': 'root_yaw_rad'}
    diagnostics = {names.get(name, name): torch.tensor([value]) for name, value in row.items()
                   if name not in ('payments', 'termination_reasons')}
    diagnostics['payments'] = {name: torch.tensor([value]) for name, value in row['payments'].items()}
    diagnostics['reasons'] = {'source_attitude_or_contact': torch.tensor([True])}
    extras = {'diagnostics': diagnostics, 'returned_episode_state': {'episode_id': torch.tensor([1]),
        'episode_tick': torch.tensor([0]), 'reset_mask': torch.tensor([True])}}
    record, = adapter.observe_paid(extras, diagnostics['reward'], diagnostics['done'],
        returned_root_position_world=torch.zeros(1, 3), returned_yaw_rad=torch.zeros(1),
        reset_metadata=[{'fixture': 'newborn state'}])
    motion = record['measurements']['all']['world_COM_motion']
    assert motion['missing_world_velocity_frames'] == 0
    assert motion['status'] == 'insufficient_contiguous_frames'
