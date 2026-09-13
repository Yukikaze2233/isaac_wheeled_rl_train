"""CPU tests of the A contract, real command/reward boundaries and slip proxy."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MethodType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.v40 import core
from wheeled_tasks.v40.round3 import Round3Commands, wheel_slip_proxy


@pytest.fixture
def contract():
    return core.load_contract(ROOT / 'contracts/own_v40_round3_a.json')


def test_contract_preserves_v2_identity_but_changes_semantic_digest(contract):
    v2 = core.load_contract(ROOT / 'contracts/own_v40_v2.json')
    assert contract['contract_id'] == v2['contract_id']
    assert core.contract_digest(contract) != core.contract_digest(v2)
    for key in ('observations', 'actions', 'actuators', 'joints', 'timing', 'termination', 'reset', 'policy'):
        assert contract[key] == v2[key]
    assert contract['commands']['stages']['locomotion']['height'] == [.29, .32]


@pytest.mark.parametrize('field,value', [
    ('spin_probability', True), ('spin_probability', float('nan')),
    ('spin_probability', .71), ('spin_probability', -.01),
    ('height_endpoint_probability_each', .51), ('height_endpoint_probability_each', float('inf')),
    ('height_resample_seconds', [8, 5]), ('height_resample_seconds', [0, 5]),
    ('height_resample_seconds', [5.001, 8]), ('height_resample_seconds', [5, 21]),
    ('stage', 'B'), ('unknown_switch', True), ('physics_material', {}),
])
def test_strict_new_fields_reject_invalid_configuration(contract, field, value):
    contract['round3'][field] = value
    with pytest.raises(ValueError):
        core.validate_contract(contract)


@pytest.mark.parametrize('mutation', ['pd', 'reward', 'height', 'standing', 'material', 'boolean_material', 'double_penalty'])
def test_A_does_not_hide_unreviewed_changes(contract, mutation):
    if mutation == 'pd':
        contract['actuators']['leg']['kp'] = 61
    elif mutation == 'reward':
        contract['rewards']['weights']['height'] = 3
    elif mutation == 'height':
        contract['commands']['stages']['locomotion']['height'][0] = .28
    elif mutation == 'standing':
        contract['commands']['stages']['locomotion']['standing_probability'] = .95
    elif mutation == 'material':
        contract['round3']['physics_material']['friction_combine_mode'] = 'multiply'
    elif mutation == 'boolean_material':
        contract['round3']['physics_material']['restitution'] = False
    else:
        contract['rewards']['weights']['lateral_velocity'] = -1
    with pytest.raises(ValueError):
        core.validate_contract(contract)


def test_contract_validation_without_torch_import():
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "from wheeled_tasks.v40.contract import load_contract; "
            f"load_contract({str(ROOT / 'contracts/own_v40_round3_a.json')!r}); "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True, timeout=10)


def test_command_distribution_endpoints_and_independent_height_clock(contract):
    torch.manual_seed(42)
    count = 50000
    sampler = Round3Commands(contract, 'locomotion', count, 'cpu')
    commands = torch.zeros(count, 3)
    ids = torch.arange(count)
    sampler.sample_velocity(commands, ids)
    sampler.sample_height(commands, ids)
    for bucket, expected in enumerate((.3, .1, .6)):
        assert (sampler.velocity_bucket == bucket).float().mean().item() == pytest.approx(expected, abs=.008)
    assert commands[sampler.velocity_bucket == 0, :2].eq(0).all()
    assert commands[sampler.velocity_bucket == 1, 0].eq(0).all()
    assert commands[sampler.velocity_bucket == 1, 1].abs().mean() > .9
    assert commands[:, :2].abs().max() <= 2
    for endpoint in (.29, .32):
        assert commands[:, 2].eq(endpoint).float().mean().item() == pytest.approx(.15, abs=.006)
    assert commands[:, 2].min() >= .29 and commands[:, 2].max() <= .32
    assert sampler.height_ticks_left.min() >= 500 and sampler.height_ticks_left.max() <= 800
    height = commands[:, 2].clone()
    sampler.sample_velocity(commands, ids)
    torch.testing.assert_close(commands[:, 2], height)
    rng = torch.random.get_rng_state().clone()
    sampler.sample_height(commands)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_old_reward_new_height_observation_and_partial_reset(contract):
    from test_round2 import adapter
    from test_env_contract_static import isolated_tensor_method

    contract['round3']['height_endpoint_probability_each'] = .5
    env = adapter(contract)
    env._round3_commands = Round3Commands(contract, 'locomotion', 2, 'cpu')
    env._get_rewards = MethodType(isolated_tensor_method('_get_rewards')[2], env)
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    ids = torch.arange(2)
    env._sample_commands(ids)
    env.commands[:] = torch.tensor([[1., .5, .305], [-1., -.5, .305]])
    env._command_ticks_left.fill_(250)
    env._round3_commands.height_ticks_left.fill_(1)
    first = env._get_observations()['policy'].reshape(2, 5, 25).clone()
    old_command = env.commands.clone()
    env.common_step_counter = 1
    env._last_reward_tick = 0
    env._get_rewards()
    assert env._round3_commands.height_ticks_left.tolist() == [0, 0]
    torch.testing.assert_close(env.commands, old_command)
    assert env.extras['log']['Tracking/height_abs_error'] == pytest.approx(.015, abs=1e-7)
    env._get_rewards()  # Duplicate reward call must not advance either timer twice.
    assert env._round3_commands.height_ticks_left.tolist() == [0, 0]
    new = env._get_observations()['policy'].reshape(2, 5, 25)
    torch.testing.assert_close(new[:, :-1], first[:, 1:])
    torch.testing.assert_close(env.commands[:, :2], old_command[:, :2])
    assert env.commands[:, 2].ne(.305).all()
    torch.testing.assert_close(new[:, -1, 8], env.commands[:, 2] * 5)
    rng, timers = torch.random.get_rng_state().clone(), env._round3_commands.height_ticks_left.clone()
    torch.testing.assert_close(env._get_observations()['policy'], new.flatten(1))
    assert torch.equal(rng, torch.random.get_rng_state())
    env._reset_idx([0])
    assert env._round3_commands.height_ticks_left[1] == timers[1]
    reset_obs = env._get_observations()['policy'].reshape(2, 5, 25)
    torch.testing.assert_close(reset_obs[1], new[1])
    torch.testing.assert_close(reset_obs[0], reset_obs[0, -1:].expand(5, 25))


def test_fixed_evaluation_command_remains_fixed_across_height_due_and_reset(contract):
    from test_round2 import adapter
    env = adapter(contract)
    env._round3_commands = Round3Commands(contract, 'locomotion', 2, 'cpu')
    env._evaluation_command_override = (0., 0., .31)
    env._sample_commands(torch.arange(2))
    for tick in range(4):
        env.common_step_counter = tick
        env._round3_commands.height_ticks_left.fill_(-1)
        env._get_observations()
        assert env.commands[:, 2].eq(.31).all()
    env._reset_idx([0])
    assert env.commands[:, 2].eq(.31).all()


def test_only_zero_translation_term_changes_and_is_L1_even_during_yaw(contract):
    v2 = core.load_contract(ROOT / 'contracts/own_v40_v2.json')
    v = torch.tensor([[.2, -.3, 0.], [.2, -.3, 0.], [.2, -.3, 0.]])
    command = torch.tensor([[0., 1., .31], [.049, 0., .31], [.05, 0., .31]])
    zeros = torch.zeros(3, 6)
    arguments = (v, torch.zeros(3, 3), torch.tensor([[0., 0., -1.]]).repeat(3, 1),
                 torch.full((3,), .31), command, zeros, zeros, zeros,
                 torch.tensor(contract['joints']['nominal_positions']).repeat(3, 1))
    new = core.compute_reward_terms(*arguments, contract)
    old = core.compute_reward_terms(*arguments, v2)
    torch.testing.assert_close(new['zero_command_translation'], torch.tensor([-.005, -.005, 0.]))
    for key in new.keys() - {'zero_command_translation'}:
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    assert old['zero_command_translation'].eq(0).all()


def test_wheel_proxy_uses_total_world_twist_and_contact_is_not_ground():
    # Both cylinder axes point +Y. Left/right joint signs are irrelevant when
    # using actual total world angular velocity instead of relative joint dq.
    pos = torch.zeros(1, 2, 3)
    pos[..., 2] = .06
    quat = torch.tensor([[[-math.sqrt(.5), 0., 0., math.sqrt(.5)]]]).repeat(1, 2, 1)
    com = pos.clone()
    vel = torch.tensor([[[1., 0., 0., 0., 1/.06, 0.], [-1., 0., 0., 0., -1/.06, 0.]]])
    proxy = wheel_slip_proxy(pos, quat, com, vel, torch.tensor([[10., 0.]]), torch.zeros(1))
    assert proxy['geometry_valid'].all()
    torch.testing.assert_close(proxy['longitudinal_velocity_proxy_m_s'], torch.zeros(1, 2), atol=2e-6, rtol=0)
    assert proxy['net_contact_candidate_not_ground'].tolist() == [[True, False]]
    assert not proxy['ground_contact_confirmed'].any()
    vel[..., :2] += torch.tensor([.2, .3])
    slip = wheel_slip_proxy(pos, quat, com, vel, torch.ones(1, 2) * 10, torch.zeros(1))
    torch.testing.assert_close(slip['longitudinal_velocity_proxy_m_s'], torch.full((1, 2), .2), atol=2e-6, rtol=0)
    torch.testing.assert_close(slip['lateral_velocity_proxy_m_s'], torch.full((1, 2), .3), atol=2e-6, rtol=0)


def test_vertical_wheel_axis_does_not_claim_valid_rolling_proxy():
    p = torch.zeros(1, 2, 3)
    q = torch.tensor([[[0., 0., 0., 1.]]]).repeat(1, 2, 1)
    result = wheel_slip_proxy(p, q, p, torch.zeros(1, 2, 6), torch.ones(1, 2), torch.zeros(1))
    assert not result['geometry_valid'].any()
    assert torch.isfinite(result['point_velocity_proxy_w_m_s']).all()
