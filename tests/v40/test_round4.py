"""CPU-only full-profile, maneuver-budget, reward and impulse boundary regressions."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.v40 import core
from wheeled_tasks.v40.round4 import PushCurriculum, FullManeuverCommands, effective_dynamic_from_readback, verify_ground_usd


@pytest.fixture
def contract():
    return core.load_contract(ROOT / 'contracts/own_v40_round4_full.json')


@pytest.mark.parametrize('update,vx,wz,push', [
    (0, .5, 1., .1), (3000, 1.25, 1.5, .19), (5000, 1.75, 11/6, .25),
    (6000, 2., 2., .3), (10000, 2 + 4000/19000, 2 + 16000/19000, .5),
    (25000, 3., 6., .5), (30000, 3., 6., .5),
])
def test_curricula_are_completed_PPO_update_based(contract, update, vx, wz, push):
    commands = FullManeuverCommands(contract, 'locomotion', 2, 'cpu', [.4, .6])
    impulses = PushCurriculum(contract, 2, 'cpu')
    commands.set_training_iteration(update)
    impulses.set_training_iteration(update)
    assert commands.caps == pytest.approx((vx, wz))
    assert impulses.max_delta_v == pytest.approx(push)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            commands.set_training_iteration(invalid)
        with pytest.raises(ValueError):
            impulses.set_training_iteration(invalid)


@pytest.mark.parametrize('change', ['schedule', 'probability', 'mu_margin', 'wheel_margin', 'new_reward', 'pd', 'terrain', 'nan'])
def test_full_profile_rejects_unreviewed_changes(contract, change):
    if change == 'schedule':
        contract['round4']['command_curriculum']['schedule'][-1][2] = 7.
    elif change == 'probability':
        contract['round4']['command_curriculum']['bucket_probabilities']['straight'] = .4
    elif change == 'mu_margin':
        contract['round4']['command_curriculum']['traction_fraction'] = 1.
    elif change == 'wheel_margin':
        contract['round4']['command_curriculum']['wheel_speed_fraction'] = True
    elif change == 'new_reward':
        contract['rewards']['weights']['height'] = 3.
    elif change == 'pd':
        contract['actuators']['leg']['kp'] = 61.
    elif change == 'terrain':
        contract['round4']['terrain'] = 'rough'
    else:
        contract['round4']['stabilization']['wheel_scale_m_s'] = float('nan')
    with pytest.raises(ValueError):
        core.validate_contract(contract)


def test_four_buckets_and_actual_per_env_traction_budget(contract):
    torch.manual_seed(44)
    n = 50000
    mu = torch.linspace(.4, .6, n)
    sampler = FullManeuverCommands(contract, 'locomotion', n, 'cpu', mu)
    sampler.set_training_iteration(25000)
    commands = torch.zeros(n, 3)
    commands[:, 2] = .31
    sampler.sample_velocity(commands, torch.arange(n))
    for index, probability in enumerate((.3, .2, .3, .2)):
        mask = sampler.velocity_bucket == index
        assert mask.float().mean().item() == pytest.approx(probability, abs=.008)
    assert commands[sampler.velocity_bucket == 0, :2].eq(0).all()
    assert commands[sampler.velocity_bucket == 1, 0].eq(0).all()
    assert commands[sampler.velocity_bucket == 2, 1].eq(0).all()
    assert commands[:, 2].eq(.31).all()
    assert commands[:, 0].abs().max() <= 3 and commands[:, 1].abs().max() <= 6
    assert (commands[:, 0].abs() * commands[:, 1].abs() <= .7 * mu * 9.81 + 1e-6).all()
    for sign in (-1, 1):
        speed = (commands[:, 0] + sign * commands[:, 1] * .4373 / 2).abs()
        assert (speed <= sampler.wheel_ground_speed_cap + 1e-6).all()
    assert sampler.traction_clipped.any()
    assert sampler.requested_velocity.abs().max() > commands.abs().max() or sampler.traction_clipped.any()
    # Exercise wheel-speed clipping separately; current valid profile already has
    # conservative ceilings, so this pure helper stress request is not a train command.
    request = torch.tensor([[10., 0.], [3., 6.]])
    bounded = sampler.constrain(request, torch.tensor([0, n-1]))
    assert bounded[0, 0] == pytest.approx(sampler.wheel_ground_speed_cap)
    assert sampler.wheel_speed_clipped[0]
    assert bounded[1, 0] * bounded[1, 1] <= .7 * .6 * 9.81 + 1e-6


def test_mu_budget_uses_actual_readback_order_not_bucket_requests():
    report = {'stage': 'B1', 'passed': True, 'wheel_env_ids': [1, 0, 1, 0],
              'wheel_physx_coefficients': [[[.8, .3, 0.]], [[.9, .7, 0.]], [[.8, .3, 0.]], [[.9, .7, 0.]]],
              'expected_mapping': {'unused': 'not the source of command budgets'}}
    assert effective_dynamic_from_readback(report, 2) == pytest.approx([.6, .4])
    report['wheel_physx_coefficients'][2][0][1] = .4
    with pytest.raises(ValueError, match='left/right'):
        effective_dynamic_from_readback(report, 2)


def test_actual_dq_quiet_penalty_zero_command_and_legacy_reward_compatibility(contract):
    b1 = core.load_contract(ROOT / 'contracts/own_v40_round3_b1.json')
    v, w, gravity = torch.zeros(4, 3), torch.tensor([[2., 3., 7.]]).repeat(4, 1), torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    command = torch.tensor([[0., 0., .31], [0., 1., .31], [1., 0., .31], [0., 0., .31]])
    action, q = torch.full((4, 6), 99.), torch.tensor(contract['joints']['nominal_positions']).repeat(4, 1)
    dq = torch.zeros(4, 6)
    dq[:, [2, 5]] = 2.  # measured wheel linear speed .12 m/s, not the huge policy target
    dq[3, [2, 5]] = .3  # .018 m/s deadzone
    args = (v, w, gravity, torch.full((4,), .31), command, action, action, torch.zeros(4, 6), q)
    full = core.compute_reward_terms(*args, contract, joint_vel6=dq)
    legacy = core.compute_reward_terms(*args, b1)
    for key in legacy:
        torch.testing.assert_close(full[key], legacy[key], atol=0, rtol=0)
    torch.testing.assert_close(full['body_angular_rate'], torch.full((4,), -.05 * 13 * .01))
    assert full['wheel_quiet'][0].item() == pytest.approx(-.02 * ((.12-.018)/.1)**2 * .01)
    assert full['wheel_quiet'][1:].abs().max() < 1e-14
    still = core.compute_reward_terms(*args, contract, joint_vel6=torch.zeros_like(dq))
    assert still['wheel_quiet'].eq(0).all()  # action target does not replace measured dq
    with pytest.raises(ValueError, match='actual joint'):
        core.compute_reward_terms(*args, contract)
    dq[0, 2] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        core.compute_reward_terms(*args, contract, joint_vel6=dq)


def test_push_rng_grace_repeat_tick_partial_reset_and_eval_disable(contract):
    torch.manual_seed(19)
    global_rng = torch.random.get_rng_state().clone()
    push = PushCurriculum(contract, 10000, 'cpu')
    push.reset(torch.arange(10000))
    assert push.episode_push_enabled.float().mean().item() == pytest.approx(.5, abs=.02)
    active = push.episode_push_enabled
    assert push.next_push_episode_step[active].min() >= 500
    assert push.next_push_episode_step[active].max() <= 700
    assert push.sample(200, torch.full((10000,), 200))[0].numel() == 0
    push.next_push_episode_step[active] = 500
    ids, delta = push.sample(500, torch.full((10000,), 500))
    assert len(ids) == active.sum()
    assert torch.linalg.vector_norm(delta, dim=-1).max() <= .1 + 1e-7
    assert push.sample(500, torch.full((10000,), 500)) is None
    unchanged = push.next_push_episode_step[1:].clone()
    push.reset(torch.tensor([0]))
    torch.testing.assert_close(push.next_push_episode_step[1:], unchanged)
    state = push.generator.get_state().clone()
    push.sample(1000, torch.full((10000,), 1000), automatic=False)
    push.reset(torch.arange(10000), automatic=False)
    assert torch.equal(state, push.generator.get_state())
    assert not push.episode_push_enabled.any()
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert push.state()['exact_trajectory_resume_supported'] is False


@pytest.fixture
def env(contract):
    from test_round2 import adapter
    from test_env_contract_static import isolated_tensor_method
    obj = adapter(contract)
    for name in ('_get_rewards', '_read_root_impulse_state', 'apply_velocity_impulse',
                 '_apply_scheduled_pushes', 'set_training_iteration'):
        setattr(obj, name, MethodType(isolated_tensor_method(name)[2], obj))
    obj._push_curriculum = PushCurriculum(contract, 2, 'cpu')
    obj._full_commands = FullManeuverCommands(contract, 'locomotion', 2, 'cpu', [.4, .6])
    obj._round3_commands = obj._full_commands
    obj.reset_terminated = torch.zeros(2, dtype=torch.bool)
    obj._last_reward_tick = -1
    obj.first_push_report = None
    velocity = obj.robot.data.root_com_vel_w.torch
    pose = obj.robot.data.root_link_pose_w.torch
    old_data = obj.robot.data

    class CachedData(SimpleNamespace):
        @property
        def root_com_lin_vel_b(self):
            if self._root_com_lin_vel_b.timestamp < 0:
                self.cached = self.root_com_vel_w.torch[:, :3].clone()
                self._root_com_lin_vel_b.timestamp = 0
            return SimpleNamespace(torch=self.cached)

    fields = {key: value for key, value in vars(old_data).items() if key != 'root_com_lin_vel_b'}
    obj.robot.data = CachedData(**fields, _root_com_lin_vel_b=SimpleNamespace(timestamp=-1))
    obj.robot.root_view = SimpleNamespace(get_root_velocities=lambda: velocity, get_root_transforms=lambda: pose)
    obj.writes = []

    def write(*, root_velocity, env_ids, full_data):
        assert env_ids.dtype == torch.int32 and full_data is False
        obj.writes.append((env_ids.clone(), root_velocity.clone()))
        velocity[env_ids.long()] = root_velocity

    obj.robot.write_root_com_velocity_to_sim_index = write
    return obj


def test_impulse_is_world_COM_additive_preserves_pose_and_angular_velocity(env):
    velocity = env.robot.data.root_com_vel_w.torch
    velocity[:] = torch.tensor([[.4, -.2, .3, 1., 2., 3.], [3., 4., 5., 6., 7., 8.]])
    before = velocity.clone()
    pose = env.robot.data.root_link_pose_w.torch
    pose[0, 3:] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
    before_pose = pose.clone()
    env._evaluation_snapshot = {'sentinel': 'previous transition'}
    result = env.apply_velocity_impulse(torch.tensor([[.1, -.15]]), [0])
    torch.testing.assert_close(velocity[0, :2], before[0, :2] + torch.tensor([.1, -.15]))
    torch.testing.assert_close(velocity[0, 2:], before[0, 2:])
    torch.testing.assert_close(velocity[1], before[1])
    torch.testing.assert_close(pose, before_pose)
    assert result['events_count'] == 1 and env.first_push_report['readback_passed']
    assert env._evaluation_snapshot == {'sentinel': 'previous transition'}


@pytest.mark.parametrize('delta,ids', [(torch.tensor([[float('nan'), 0.]]), [0]),
    (torch.zeros(2, 2), [0, 0]), (torch.zeros(1, 2), [2]), (torch.zeros(1, 2), [True]),
    (torch.zeros(1, 2, dtype=torch.float64), [0])])
def test_bad_impulse_never_writes_physics(env, delta, ids):
    with pytest.raises(ValueError):
        env.apply_velocity_impulse(delta, ids)
    assert not env.writes


def test_old_reward_next_critic_observation_and_no_duplicate_push(env):
    env.commands[:] = torch.tensor([[0., 0., .31], [0., 0., .31]])
    env._commands_due.zero_()
    env._command_ticks_left.fill_(100)
    env._round3_commands.height_ticks_left.fill_(100)
    env.episode_length_buf.fill_(500)
    env._push_curriculum.episode_push_enabled[:] = True
    env._push_curriculum.next_push_episode_step[:] = 500
    env.common_step_counter = 500
    env._get_rewards()
    old_velocity_error = env.extras['log']['Tracking/vx_abs_error'].clone()
    obs = env._get_observations()
    assert len(env.writes) == 1
    torch.testing.assert_close(obs['critic'][:, -4:-1], env.robot.data.root_com_vel_w.torch[:, :3])
    assert env.extras['log']['Tracking/vx_abs_error'] == old_velocity_error == 0
    assert env.extras['log']['Push/events_count'] == 2
    env._get_observations()
    assert len(env.writes) == 1 and env.extras['log']['Push/events_count'] == 2
    env.set_training_iteration(10000)
    assert env._full_commands.training_iteration == env._push_curriculum.training_iteration == 10000
    assert env._full_commands.caps[0] > 2 and env._push_curriculum.max_delta_v == .5


def test_evaluation_override_disables_random_push_and_partial_reset_reinitializes_only_subset(env):
    env._evaluation_command_override = (0., 0., .31)
    env._push_curriculum.episode_push_enabled[:] = True
    env._push_curriculum.next_push_episode_step[:] = 500
    env.episode_length_buf[:] = 500
    env.common_step_counter = 500
    env._get_observations()
    assert not env.writes
    before = env._push_curriculum.next_push_episode_step[1].clone()
    env._reset_idx([0])
    assert not env._push_curriculum.episode_push_enabled[0]
    assert env._push_curriculum.next_push_episode_step[1] == before


def test_full_command_update_waits_for_resampling_and_preserves_history(env):
    env.commands[:] = torch.tensor([[.49, .99, .31], [-.49, -.99, .31]])
    env._command_ticks_left.fill_(1)
    env._commands_due.zero_()
    env._round3_commands.height_ticks_left.fill_(100)
    first = env._get_observations()['policy'].reshape(2, 5, 25).clone()
    old = env.commands.clone()
    env.set_training_iteration(25000)
    torch.testing.assert_close(env.commands, old)
    env.common_step_counter += 1
    env._get_rewards()
    torch.testing.assert_close(env.commands, old)
    assert env.extras['log']['Tracking/vx_abs_error'] == pytest.approx(.49)
    next_obs = env._get_observations()['policy'].reshape(2, 5, 25)
    torch.testing.assert_close(next_obs[:, :-1], first[:, 1:])
    torch.testing.assert_close(next_obs[:, -1, 6:9], env.commands * torch.tensor([1., 1., 5.]))
    assert not torch.equal(env.commands[:, :2], old[:, :2])


def test_empty_impulse_does_not_read_or_write_and_bad_readback_fails_loudly(env):
    assert env.first_push_report is None
    assert env.apply_velocity_impulse(torch.empty(0, 2), []) == {'events_count': 0}
    assert not env.writes and env.first_push_report is None
    env.robot.write_root_com_velocity_to_sim_index = lambda **kwargs: None
    with pytest.raises(RuntimeError, match='readback'):
        env.apply_velocity_impulse(torch.tensor([[.1, 0.]]), [0])
    assert env.first_push_report['readback_passed'] is False


def test_ground_cache_rechecks_exact_bytes_without_simulator(tmp_path, monkeypatch):
    import wheeled_tasks.v40.round4 as r4
    path = tmp_path / 'ground.usd'
    path.write_bytes(b'CPU-only fixture')
    with pytest.raises(ValueError, match='SHA256'):
        verify_ground_usd(path)
    monkeypatch.setattr(r4, 'GROUND_ASSET_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    assert verify_ground_usd(path)['path'] == str(path.resolve())
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        verify_ground_usd(path)
