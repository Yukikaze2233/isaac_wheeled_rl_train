"""B1 startup mapping, inherited control semantics and restore validation (CPU)."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.v40 import core
from wheeled_tasks.v40.round3 import B1MaterialBuckets, Round3Commands, _material_digest


@pytest.fixture
def contract():
    return core.load_contract(ROOT / 'contracts/own_v40_round3_b1.json')


def saved_report(buckets):
    return {'stage': 'B1', 'expected_mapping': deepcopy(buckets.mapping), 'mapping_sha256': buckets.mapping_sha256}


def test_B1_inherits_A_exactly_except_stage_and_material_randomization(contract):
    a = core.load_contract(ROOT / 'contracts/own_v40_round3_a.json')
    inherited = deepcopy(contract)
    inherited['round3'].pop('material_randomization')
    inherited['round3']['stage'] = 'A'
    assert inherited == a
    assert core.contract_digest(a) == 'e31ba1538be4f26d5a5c6452ca8b06e1d09f41db999118957909d13f7248e73e'
    assert core.contract_digest(contract) != core.contract_digest(a)
    assert core.is_round2(contract)
    for name in ('own_v40_v1.json', 'own_v40_v2.json'):
        core.load_contract(ROOT / 'contracts' / name)


@pytest.mark.parametrize('field,value', [
    ('bucket_count', 63), ('bucket_count', True), ('generator_seed', True), ('generator_seed', -1),
    ('generator_seed', 2**63), ('nominal_probability', float('nan')), ('nominal_probability', .4),
    ('effective_static_range', [.5, .9]), ('effective_dynamic_range', [.4, float('inf')]),
    ('restitution', False), ('restitution', .1), ('resample', 'every_reset'),
    ('assignment', 'per_wheel'), ('pd_randomization', True),
])
def test_B1_rejects_unknown_or_unreviewed_material_configuration(contract, field, value):
    contract['round3']['material_randomization'][field] = value
    with pytest.raises(ValueError):
        core.validate_contract(contract)


def test_B1_cannot_smuggle_command_PD_or_reward_changes(contract):
    for mutate in (
        lambda c: c['actuators']['leg'].update(kp=61),
        lambda c: c['commands']['stages']['locomotion'].update(standing_probability=.4),
        lambda c: c['rewards']['weights'].update(zero_command_translation=0),
        lambda c: c['round3'].update(delay_ms=10),
    ):
        changed = deepcopy(contract)
        mutate(changed)
        with pytest.raises(ValueError):
            core.validate_contract(changed)


def test_independent_seed_reproducible_mapping_and_nominal_mask(contract):
    config = contract['round3']['material_randomization']
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    first = B1MaterialBuckets(config, 20000)
    second = B1MaterialBuckets(config, 20000)
    assert first.mapping == second.mapping and first.mapping_sha256 == second.mapping_sha256
    assert random.getstate() == python_state and torch.equal(torch.random.get_rng_state(), torch_state)
    mask = first.mapping['nominal_mask']
    assert sum(mask) / len(mask) == pytest.approx(.3, abs=.012)
    assert len(first.mapping['effective_buckets']) == len(first.mapping['wheel_buckets']) == 64
    for i, nominal in enumerate(mask):
        s, d, e = first.wheel_coefficients(i)
        assert .5 <= s <= .9 and .3 - 1e-12 <= d <= .7 and d <= s and e == 0
        if nominal:
            assert [s, d, e] == [.5, .5, 0.]
        else:
            bucket = first.mapping['env_bucket_ids'][i]
            es, ed, _ = first.mapping['effective_buckets'][bucket]
            assert (s + .5) / 2 == pytest.approx(es)
            assert (d + .5) / 2 == pytest.approx(ed)
    different = B1MaterialBuckets({**config, 'generator_seed': config['generator_seed'] + 1}, 20000)
    assert different.mapping != first.mapping
    json.dumps(saved_report(first), allow_nan=False)


def test_restore_uses_saved_table_without_invoking_rng(contract, monkeypatch):
    import wheeled_tasks.v40.round3 as round3
    config = contract['round3']['material_randomization']
    original = B1MaterialBuckets(config, 8)
    report = json.loads(json.dumps(saved_report(original)))
    monkeypatch.setattr(round3.random, 'Random', lambda *args: pytest.fail('restore must not resample'))
    restored = B1MaterialBuckets(config, 8, report)
    assert restored.mapping == original.mapping
    report['expected_mapping']['nominal_mask'][0] = not report['expected_mapping']['nominal_mask'][0]
    assert restored.mapping == original.mapping  # Owned copy, not mutable caller metadata.
    with pytest.raises(ValueError, match='digest mismatch'):
        B1MaterialBuckets(config, 8, report)


@pytest.mark.parametrize('mutation', ['wheel_table', 'out_of_range', 'bucket_id', 'mask_type', 'env_count', 'seed'])
def test_restored_mapping_is_validated_even_if_digest_is_recomputed(contract, mutation):
    config = contract['round3']['material_randomization']
    report = saved_report(B1MaterialBuckets(config, 8))
    m = report['expected_mapping']
    if mutation == 'wheel_table':
        m['wheel_buckets'][0][0] += .1
    elif mutation == 'out_of_range':
        m['effective_buckets'][0][1] = 1.
    elif mutation == 'bucket_id':
        m['env_bucket_ids'][0] = 64
    elif mutation == 'mask_type':
        m['nominal_mask'][0] = 1
    elif mutation == 'env_count':
        m['num_envs'] = 4
    else:
        m['generator_seed'] += 1
    report['mapping_sha256'] = _material_digest(m)
    with pytest.raises(ValueError):
        B1MaterialBuckets(config, 8, report)


def test_episode_partial_reset_preserves_mapping(contract, monkeypatch):
    from test_round2 import adapter
    import wheeled_tasks.v40.round3 as round3
    env = adapter(contract)
    env._round3_commands = Round3Commands(contract, 'locomotion', 2, 'cpu')
    env._b1_materials = B1MaterialBuckets(contract['round3']['material_randomization'], 2)
    before = deepcopy(env._b1_materials.mapping)
    monkeypatch.setattr(round3.random, 'Random', lambda *args: pytest.fail('reset resampled friction'))
    for ids in ([0], [1], [0, 1]):
        env._reset_idx(ids)
        assert env._b1_materials.mapping == before


def test_A_and_B1_command_rng_and_L1_rewards_are_identical(contract):
    a = core.load_contract(ROOT / 'contracts/own_v40_round3_a.json')
    outputs = []
    for c in (a, contract):
        torch.manual_seed(91)
        sampler = Round3Commands(c, 'locomotion', 128, 'cpu')
        commands = torch.zeros(128, 3)
        sampler.sample_velocity(commands, torch.arange(128))
        sampler.sample_height(commands)
        outputs.append(commands)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    v = torch.tensor([[.2, -.3, 0.]])
    zero6 = torch.zeros(1, 6)
    args = (v, torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]), torch.tensor([.30]),
            torch.tensor([[0., 1., .30]]), zero6, zero6, zero6,
            torch.tensor([a['joints']['nominal_positions']]))
    ra, rb = core.compute_reward_terms(*args, a), core.compute_reward_terms(*args, contract)
    for key in ra:
        torch.testing.assert_close(ra[key], rb[key], atol=0, rtol=0)
    assert rb['zero_command_translation'].item() == pytest.approx(-.005)
