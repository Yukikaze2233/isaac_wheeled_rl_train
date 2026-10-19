"""Physical noise/encoding boundaries and independent reset-force ownership."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from support_flat_episode_env_wiring import execute, method
from support_scut_flat_scripted_recovery import ready, rig, tick
from wheeled_tasks.chassis.actor_sensor_noise import ActorSensorNoise, KEY, SETTINGS, validate_actor_sensor_noise
from wheeled_tasks.chassis.critic_observation import build_scut78
from wheeled_tasks.chassis.dynamics import (
    EPISODE_SPRING_KEY, EPISODE_SPRING_SETTINGS, validate_episode_spring_force_randomization,
)
from wheeled_tasks.chassis.scut_observation import build_scut35, build_sensor18
from wheeled_tasks.chassis.training_profile import profile_source_files, validate_profile_selection
from wheeled_tasks.chassis.control_core import HistoryStack


ROOT = Path(__file__).resolve().parents[1]


def contract():
    return json.loads((ROOT / 'configs/v7_formal_20261012/motion_retention_review.json').read_text())


def noise(count=4096):
    return ActorSensorNoise(count, 'cpu', contract(), torch.Generator().manual_seed(91))


def test_uniform_physical_noise_has_the_reviewed_units_and_preserves_input():
    producer = noise()
    sensors = torch.zeros(4096, 18)
    sensors[:, 17] = -1.
    before = sensors.clone()
    output = producer.apply(sensors, 0).clone()
    delta = output - before
    assert torch.equal(sensors, before)
    assert bool((producer.noise.abs() <= producer.half_width).all())
    assert producer.noise[:, [2, 5]].eq(0).all()
    assert producer.noise[:, 12:15].std().item() == pytest.approx(.25 / math.sqrt(3), rel=.04)
    assert producer.noise[:, 12:15].mean().abs() < .01
    encoded = build_sensor18(output[:, 12:15], output[:, 15:18], output[:, :6], output[:, 6:12], torch.zeros(6))
    assert encoded[:, 10:12].eq(0).all()
    assert encoded[:, :3].abs().max() <= .125
    assert encoded[:, 12:16].abs().max() <= .05
    assert encoded[:, 16:18].abs().max() <= .1
    assert bool(torch.isfinite(delta).all())


def test_repeated_read_does_not_draw_again_and_partial_reset_changes_only_its_rows():
    producer = noise(8)
    sensors = torch.zeros(8, 18)
    first = producer.apply(sensors, 0).clone()
    rng = producer.generator.get_state().clone()
    for _ in range(3):
        assert torch.equal(producer.apply(sensors, 0), first)
        assert torch.equal(producer.generator.get_state(), rng)
    producer.reset(torch.tensor([2, 5]))
    changed = producer.apply(sensors, 0).clone()
    assert torch.equal(changed[[0, 1, 3, 4, 6, 7]], first[[0, 1, 3, 4, 6, 7]])
    assert not torch.equal(changed[[2, 5]], first[[2, 5]])
    assert producer.samples == 10
    producer.apply(sensors, 1)
    assert producer.samples == 18


@pytest.mark.parametrize('update', [
    {'observation_noise_enabled': False},
    {'signal_perturbations': {'enabled': True, 'max_delay_steps': 0}},
    {'signal_perturbations': {'enabled': False, 'max_delay_steps': 1}},
    {KEY: dict(SETTINGS, gyro_rad_s=.025)},
    {KEY: dict(SETTINGS, wheel_velocity_rad_s=True)},
    {KEY: dict(SETTINGS, height_command_m=.01)},
    {KEY: dict(SETTINGS, enabled=False)},
])
def test_partial_false_or_double_noise_selection_is_rejected(update):
    with pytest.raises(ValueError):
        validate_profile_selection({**contract(), **update})


def critic_frame(frame):
    n = len(frame)
    return build_scut78(clean_common28=frame[:, :28], root_linear_velocity_b=torch.zeros(n, 3),
        root_height=torch.full((n, 1), .305), command_context7=frame[:, 28:35],
        pc_kp=torch.ones(n, 6), pc_kd=torch.ones(n, 6), applied_torque=torch.zeros(n, 6),
        observation_age_ticks=torch.zeros(n, 4), action_age_ticks=torch.zeros(n, 2),
        wheel_relative_velocity_b=torch.zeros(n, 2, 3), wheel_contact=torch.zeros(n, 2),
        base_mass_ratio=torch.ones(n, 1), wheel_material=torch.ones(n, 2, 3))


def observation_rig():
    n = 8
    q, dq, omega = torch.zeros(n, 6), torch.ones(n, 6), torch.zeros(n, 3)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(n, 1)
    # Keep raw source and delivered feedback distinct to exercise consumer order.
    delivered = torch.cat((q + .1, dq + .2, omega + .3, gravity), -1)
    env = SimpleNamespace(scut35=True, critic78=True, num_envs=n,
        state=lambda: (torch.zeros(n, 3), omega, gravity, torch.full((n,), .305), q, dq, None),
        jump_requested=torch.zeros(n, dtype=torch.bool), mode=torch.zeros(n, dtype=torch.long),
        scut_reward=object(), commands=torch.tensor([[.05, 0., .305]]).repeat(n, 1),
        targets=torch.zeros(n, 4), actions=torch.ones(n, 6) * .25,
        skills=None, references=None, v5=SimpleNamespace(nominal=torch.zeros(6)),
        cfg={'actor_frame_dim': 35, 'critic_dim': 78}, signal_delay=None, perturbations=None,
        _pc_feedback_transport=SimpleNamespace(feedback=delivered),
        _critic78_frame=critic_frame, actor_sensor_noise=noise(n),
        history=HistoryStack(n, 'cpu', length=1, dim=35), tick=0)
    source = method('get_observations')
    scope = {'torch': torch, 'TensorDict': TensorDict, '__package__': 'wheeled_tasks.chassis'}
    exec(compile(ast.Module(body=[source], type_ignores=[]), '<production-observations>', 'exec'), scope)
    env.get_observations = lambda: scope['get_observations'](env)
    return env, q, dq, omega, gravity


def test_actual_observation_builder_keeps_clean_critic_pd_and_context():
    env, q, dq, omega, gravity = observation_rig()
    delivered = env._pc_feedback_transport.feedback.clone()
    observed = env.get_observations()
    clean = build_scut35(omega, gravity, env.commands, q, dq, env.actions, env.v5.nominal,
        env.jump_requested, torch.zeros(8), torch.zeros(8))
    assert torch.equal(observed['critic'], critic_frame(clean))
    assert torch.equal(env._pc_feedback_transport.feedback, delivered)
    assert torch.equal(env.actor_sensor_noise.input, delivered)
    policy = observed['policy']
    assert torch.equal(policy[:, :4], clean[:, :4])
    assert torch.equal(policy[:, 22:], clean[:, 22:])
    assert policy[:, 14:16].eq(0).all()
    noisy = env.actor_sensor_noise.output
    assert torch.equal(policy[:, 4:22], build_sensor18(noisy[:, 12:15], noisy[:, 15:18],
        noisy[:, :6], noisy[:, 6:12], env.v5.nominal))
    rng = env.actor_sensor_noise.generator.get_state().clone()
    again = env.get_observations()
    assert torch.equal(again['policy'], policy)
    assert torch.equal(env.actor_sensor_noise.generator.get_state(), rng)


def test_independent_force_prior_still_runs_when_legacy_noise_is_absent():
    c = contract()
    assert validate_actor_sensor_noise(c) == SETTINGS
    assert validate_episode_spring_force_randomization(c) == EPISODE_SPRING_SETTINGS
    assert not c['signal_perturbations']['enabled']
    nodes = method('reset').body
    branch = next(node for node in nodes if isinstance(node, ast.If)
        and 'getattr(self, \'episode_spring_strength_settings\'' in ast.unparse(node.test))
    env = SimpleNamespace(episode_spring_strength_settings=deepcopy(EPISODE_SPRING_SETTINGS),
        motor_strength=torch.zeros(4096, 6), spring_strength=torch.zeros(4096, 1),
        generator=torch.Generator().manual_seed(93), device='cpu')
    execute([branch], env, ids=torch.arange(4096), count=4096)
    assert env.motor_strength.eq(1).all()
    assert env.spring_strength.min() >= .9 and env.spring_strength.max() <= 1.1
    assert env.spring_strength.unique().numel() > 4000
    before = env.spring_strength.clone()
    execute([branch], env, ids=torch.tensor([7, 9]), count=2)
    assert torch.equal(env.spring_strength[:7], before[:7])
    assert not torch.equal(env.spring_strength[[7, 9]], before[[7, 9]])
    sources = profile_source_files(c)
    assert 'src/wheeled_tasks/chassis/actor_sensor_noise.py' in sources
    assert 'src/wheeled_tasks/chassis/dynamics.py' in sources
    assert 'src/wheeled_tasks/chassis/robustness.py' not in sources


def test_angle_noise_is_encoded_through_the_existing_wrapped_position_abi():
    producer = noise(4096)
    sensors = torch.zeros(4096, 18)
    sensors[:, 0] = math.pi - .001
    output = producer.apply(sensors, 0)
    encoded = build_sensor18(output[:, 12:15], output[:, 15:18], output[:, :6], output[:, 6:12], torch.zeros(6))
    assert bool((encoded[:, 6] < 0).any() & (encoded[:, 6] > 0).any())
    difference = encoded[:, 6] - sensors[:, 0]
    wrapped = torch.atan2(difference.sin(), difference.cos())
    assert wrapped.abs().max() <= .025001


def test_active_source_reward_path_publishes_noise_and_independent_force_metrics(rig):
    ready(rig)
    rig.actor_sensor_noise = noise(2)
    rig.actor_sensor_noise.apply(torch.zeros(2, 18), rig.tick)
    rig.episode_spring_strength_settings = deepcopy(EPISODE_SPRING_SETTINGS)
    rig.spring_strength = torch.tensor([[.9], [1.1]])
    _, _, _, info = tick(rig)
    values = info['log']
    assert values['/sensor_noise/enabled_fraction'] == 1.
    assert values['/sensor_noise/noise_vector_draws'] == 2
    assert values['/domain/spring_force_scale_mean'] == 1.
    assert values['/domain/spring_force_scale_min'] == pytest.approx(.9)
    assert values['/domain/spring_force_scale_max'] == pytest.approx(1.1)
