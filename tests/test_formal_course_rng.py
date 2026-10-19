"""First course entry owns new sampling; restarts and paired replays retain RNG."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.policy_transfer import formal_resume_rng_selection
from wheeled_tasks.chassis.round_trip_transport import RoundTripDelayBudget, round_trip_transport_settings
from wheeled_tasks.chassis.small_rough_course import KEY, selection, validate_resume
from wheeled_tasks.chassis.training_profile import restore_rng


ROOT = Path(__file__).resolve().parents[1]


def candidate():
    return json.loads((ROOT / 'configs/v7_formal_20261012/motion_retention_review.json').read_text())


def parent():
    cfg = candidate()['formal_critic_adaptation']
    return {'stage': 'foundation', 'contract_sha256': cfg['source_contract_sha256'],
            'successful_updates_total': cfg['source_learning_updates']}


def restarted(**extra):
    return {'stage': 'terrain', 'sampling_seed': 20261020, 'terrain_geometry_seed': 290929,
            'formal_critic_adaptation': candidate()['formal_critic_adaptation'],
            'critic_warmup_remaining': 0, **extra}


def test_first_motion_course_requires_fresh_rng_then_inherits_on_restart():
    config = candidate()
    chosen = formal_resume_rng_selection(config, parent(),
        stage='terrain', mode='fresh_fork', seed=20261020)
    assert chosen['phase'] == 'new_course_entry'
    assert chosen['geometry_seed'] == 290929
    with pytest.raises(ValueError, match='entry requires RNG mode fresh_fork'):
        formal_resume_rng_selection(config, parent(), stage='terrain', mode='inherit', seed=20261020)
    restart = restarted()
    chosen = formal_resume_rng_selection(config, restart, stage='terrain', mode='inherit', seed=20261020)
    assert chosen['phase'] == 'course_restart'
    with pytest.raises(ValueError, match='restart requires RNG mode inherit'):
        formal_resume_rng_selection(config, restart, stage='terrain', mode='fresh_fork', seed=20261020)
    missing_seed = dict(restart)
    missing_seed.pop('sampling_seed')
    for infos, seed in ((missing_seed, 20261020), (restart, 20261021),
                        ({**restart, 'terrain_geometry_seed': 290930}, 20261020)):
        with pytest.raises(ValueError, match='retain saved sampling and geometry seeds'):
            formal_resume_rng_selection(config, infos, stage='terrain', mode='inherit', seed=seed)


def test_legacy_contracts_keep_their_existing_resume_mode():
    for path in ('mixed_tracking_repair_063.json', 'recovery_after10000_056.json'):
        config = json.loads((ROOT / 'configs/v7_formal_20261012' / path).read_text())
        assert formal_resume_rng_selection(config, {'stage': 'terrain'},
            stage=config['curriculum_stage'], mode='inherit', seed=290929) == {'mode': 'inherit', 'phase': 'existing_resume'}
        with pytest.raises(ValueError, match='explicit motion-course handoff'):
            formal_resume_rng_selection(config, {'stage': 'terrain'},
                stage=config['curriculum_stage'], mode='fresh_fork', seed=290929)


@pytest.mark.parametrize('seed', [True, -1, 2**32, 290929., None])
def test_geometry_seed_is_explicit_and_strict(seed):
    config = candidate()
    config[KEY]['geometry_seed'] = seed
    with pytest.raises(ValueError, match='uint32 terrain geometry seed'):
        selection(config)


def test_geometry_seed_cannot_drift_during_a_course_restart():
    config = candidate()
    changed = deepcopy(config)
    changed[KEY]['geometry_seed'] += 1
    with pytest.raises(ValueError, match='unrelated physical or learning semantics'):
        validate_resume(config, changed)
    entry = ast.parse((ROOT / 'src/wheeled_tasks/chassis/env.py').read_text())
    call = next(node for node in ast.walk(entry) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'create_shared_terrain')
    assert ast.unparse(call.args[4]) == 'terrain_seed'


def test_fresh_entry_reseeds_policy_environment_and_shared_transport_without_old_rng():
    config = candidate()
    old_torch = torch.Generator().manual_seed(14).get_state()
    old_env = torch.Generator().manual_seed(15).get_state()
    transport = RoundTripDelayBudget(8, 'cpu', round_trip_transport_settings(), 16)
    old_transport = transport.generator.get_state().clone()
    env = SimpleNamespace(generator=torch.Generator().manual_seed(999), round_trip_transport=transport)
    infos = {**parent(), 'rng_state': old_torch, 'env_rng_state': old_env,
        'round_trip_rng_state': old_transport, 'round_trip_transport_configuration': transport.cfg}
    chosen = formal_resume_rng_selection(config, infos, stage='terrain', mode='fresh_fork', seed=20261020)

    def draws(seed):
        receipt = restore_rng(torch, env, infos, seed=seed, mode=chosen['mode'])
        transport.reset(torch.arange(8))
        return receipt, torch.rand(8), torch.rand(8, generator=env.generator), transport.total_s.clone()

    first, a, b, c = draws(20261020)
    _, aa, bb, cc = draws(20261020)
    assert all(torch.equal(x, y) for x, y in ((a, aa), (b, bb), (c, cc)))
    _, aaa, bbb, ccc = draws(20261021)
    assert all(not torch.equal(x, y) for x, y in ((a, aaa), (b, bbb), (c, ccc)))
    assert not first['parent_rng_restored']
    assert first['transport_rng'] == 'fresh_reseeded_stream'
    assert config[KEY]['geometry_seed'] == 290929
    assert torch.equal(infos['rng_state'], old_torch)

    chosen = formal_resume_rng_selection(config, {**infos, **restarted()},
        stage='terrain', mode='inherit', seed=20261020)
    restored = restore_rng(torch, env, infos, seed=999, mode=chosen['mode'])
    assert restored['parent_rng_restored']
    assert torch.equal(torch.get_rng_state(), old_torch)
    assert torch.equal(env.generator.get_state(), old_env)
    assert torch.equal(transport.generator.get_state(), old_transport)


@pytest.mark.parametrize('mode,parent_restored,phase,validation,accepted', [
    ('inherit', True, 'course_restart', None, True),
    ('inherit', False, 'course_restart', None, False),
    ('fresh_fork', False, 'new_course_entry', None, True),
    ('fresh_fork', False, 'new_course_entry', 'sampled', False),
    ('fresh_fork', False, 'new_course_entry', 'mean', False),
    ('fresh_fork', False, 'diagnostic_fork', None, False),
])
def test_actual_capture_guard_keeps_matched_replays_distinct(
        tmp_path, mode, parent_restored, phase, validation, accepted):
    tree = ast.parse((ROOT / 'scripts/train_chassis.py').read_text())
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and any(isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
            and isinstance(child.value.func, ast.Name) and child.value.func.id == 'bind_sampled_policy_inputs'
            for child in node.body))
    called = []
    scope = dict(c={'v5_control': {'leg_winding_version': 'paired_branch107_v1'}},
        args=SimpleNamespace(capture_numerical_failure=True, run_dir=tmp_path),
        report={'rng_restore': {'mode': mode, 'parent_rng_restored': parent_restored,
            'selection': {'phase': phase}}},
        sampled_policy_validation=validation == 'sampled', restored_policy_validation=validation == 'mean',
        runner=SimpleNamespace(alg=object()), env=object(),
        bind_sampled_policy_inputs=lambda *values: called.append(values))
    code = compile(ast.Module(body=[guard], type_ignores=[]), '<actual-capture-guard>', 'exec')
    if not accepted:
        with pytest.raises(ValueError, match='RNG'):
            exec(code, scope)
        assert not called
    else:
        exec(code, scope)
        assert len(called) == 1
        expected = ('fresh_motion_course_not_paired_parent_replay' if mode == 'fresh_fork'
                    else 'inherited_parent_stream')
        assert scope['report']['physical_capture_rng_scope'] == expected
