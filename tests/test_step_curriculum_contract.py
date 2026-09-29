"""Independent invariants for the reviewed continuous-climb / triggered-jump plan."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.full_curriculum import stage_contract
from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.preflight import design_preflight, validate_design_contract, validate_runtime_state
from wheeled_tasks.chassis.response_migration import (
    assess_response_retention, response_contract_provenance, validate_response_resume,
)

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('S1', 'S2', 'S3', 'S4', 'S5', 'S5T', 'S5D', 'S6')
CAPS = (1000, 2000, 1000, 1500, 1000, 600, 400, 1000)
ROLES = ('train', 'baseline', 'quick', 'full', 'confirmation')
RETIRED = {'stairs_200_250', 'stairs_down_200_250', 'step_up_15', 'step_up_20',
           'step_up_25', 'heldout_step_up_payload_grip', 'step_up_03', 'step_up_06', 'step_up_10', 'stairs'}


def read(name):
    return json.loads((ROOT / name).read_text())


@lru_cache(maxsize=1)
def inputs():
    plan = read('contracts/v6_step_curriculum_training_v1.json')
    return (plan, read(plan['design_preflight']['file']),
            read(plan['runtime_base']['control_math_source']))


def config_for(index, role='train', source=None):
    plan = read(source) if source else inputs()[0]
    recipe = next(item for item in plan['stages'] if item['name'] == STAGES[index])
    config = stage_contract({}, plan, recipe, 20000)
    seed = 290927 if role == 'train' else 290930 if role == 'confirmation' else 290929
    if role != 'train':
        config['evaluation']['seed'] = seed
        config = fixed_suite_contract(config)
    count = 20000 if role == 'train' else len(config['evaluation']['cases']) * config['evaluation']['episodes_per_case']
    return config, seed, count


def test_hashes_and_independent_budget_split_are_frozen():
    plan, design, _ = inputs()
    assert plan['curriculum_revision'] == 'v6_step_curriculum_v1'
    assert list(design['stages']) == list(STAGES)
    assert all(set(row['roles']) == set(ROLES) for row in design['stages'].values())
    assert [item['updates'] for item in plan['stages']] == list(CAPS)
    assert [item['name'] for item in plan['stages']] == list(STAGES)
    assert sum(CAPS) == plan['total_update_ceiling'] == 8500
    sensor = plan['runtime_base']['step_jump_task']
    assert sensor['sensor_origin_b_m'] == [.10, -.20]
    assert sensor['sensor_pitch_rad'] == .55
    assert sensor['ground_echo_margin_m'] == .015
    budget = read(plan['budget_spec_file'])
    assert [row['update_ceiling'] for row in budget['formal']['stages']] == list(CAPS)
    assert budget['formal']['total_transition_ceiling'] == 4080000000
    for name, digest in [(plan['budget_spec_file'], plan['budget_spec_sha256']),
                         (plan['design_preflight']['file'], plan['design_preflight']['sha256'])]:
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    bare = deepcopy(plan)
    bare.pop('design_preflight')
    assert hashlib.sha256(json.dumps(bare, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest() == design['reviewed_plan_payload_sha256']


@pytest.mark.parametrize('index', range(8))
@pytest.mark.parametrize('role', ROLES)
def test_all_forty_effective_roles_pass_preflight(index, role):
    config, seed, count = config_for(index, role)
    _, design, control = inputs()
    assert len(config['evaluation']['cases']) == 268
    assert count == (20000 if role == 'train' else 5360 if index == 7 else 1072)
    result = design_preflight(config, control, ROOT, stage=config['enabled_stages'][0],
                              role=role, seed=seed, num_envs=count)
    assert result['status'] == 'static_passed'
    assert result['budget']['total_update_ceiling'] == 8500
    active = role != 'train' or index >= 2
    assert config['command_transport']['enabled'] is active
    observed = deepcopy(result['runtime_expected'])
    assert observed['modules']['step_jump_task'] is True
    assert observed['modules']['step_climb_task'] is True
    assert observed['command_transport_model'] == ('phase_aware_hold_v1' if active else None)
    assert validate_runtime_state(result['runtime_expected'], observed)['status'] == 'runtime_configuration_passed'
    for module in ('step_jump_task', 'step_climb_task'):
        broken = deepcopy(observed)
        broken['modules'].pop(module)
        with pytest.raises(ValueError, match=module):
            validate_runtime_state(result['runtime_expected'], broken)


@pytest.mark.parametrize('index', range(3))
def test_s1_to_s3_training_fields_are_unchanged(index):
    before, _, _ = config_for(index, source='contracts/v6_transport_response_training_v1.json')
    after, _, _ = config_for(index)
    for cfg in (before, after):
        for field in ('evaluation', 'design_preflight', 'budget_spec_file', 'budget_spec_sha256',
                      'response_migration', 'step_jump_task', 'actor_reserved_frame30', 'release_scope'):
            cfg.pop(field, None)
    assert before == after
    assert not any('step_jump' in spec or 'body_climb' in spec for spec in after['skill_specs'].values())


def test_all_old_surviving_cases_unchanged_new_methods_separate_and_stage_gated():
    old, _, _ = config_for(0, source='contracts/v6_transport_response_training_v1.json')
    before = {case['name']: case for case in old['evaluation']['cases']}
    for index in range(8):
        config, _, _ = config_for(index)
        cases = {case['name']: case for case in config['evaluation']['cases']}
        assert set(before) - set(cases) == RETIRED
        assert [name for name in cases if name in before] == [name for name in before if name not in RETIRED]
        assert all(cases[name] == case for name, case in before.items() if name not in RETIRED)
        assert len(set(cases) - set(before)) == 23
        for name, case in cases.items():
            if not (case.get('step_jump_case') or case.get('body_climb_case') or case.get('untriggered_up_case')):
                continue
            trained = STAGES.index(case['introduced_stage']) <= index
            assert (name in config['evaluation']['promotion_case_names']) is trained
            spec = case['skill']
            if case.get('body_climb_case'):
                assert spec['mode'] == 0 and case['task'] == 'traverse'
                assert 'step_jump' not in spec and 'trigger_source' not in spec['body_climb']
                assert 'request_protocol_case' not in case
            elif case.get('untriggered_up_case'):
                assert spec['mode'] == 2 and case['task'] == 'traverse'
                assert spec['kind'] in ('rolling_up', 'rolling_stairs')
                assert not {'step_jump', 'body_climb', 'manual_request_seconds', 'manual_hold_seconds'} & spec.keys()
            else:
                assert spec['mode'] == 4 and case['task'] == 'jump'
                assert spec['step_jump']['trigger_source'] in ('manual', 'laser')
                assert spec['semantics']['jump_apex_delta_m'] == pytest.approx(spec['step_jump']['rises_m'][0] + .08)
        if index == 6:
            assert config['behavior_pool_fractions']['double_body_climb'] == .15
            assert config['behavior_pool_fractions']['double_step_jumps'] == .15
        assert sum(config['behavior_pool_fractions'].values()) == pytest.approx(1.)
        assert sum(group['fraction'] for group in config['scene_groups']) == pytest.approx(1.)


def test_double_geometry_is_two_rises_and_apex_not_cumulative():
    config, _, _ = config_for(6)
    for name in ('double_step_jump_200_250_manual', 'double_step_jump_200_250_laser', 'body_climb_double_200_250'):
        spec = config['skill_specs'][name]
        assert spec['terrain_limits']['stair_heights_m'] == [.2, .45]
        assert spec['terrain_limits']['stair_tread_m'] == [.8, .6]
        params = spec.get('step_jump', spec.get('body_climb'))
        assert params['rises_m'] == [.2, .25]
        if 'step_jump' in spec:
            assert spec['semantics']['jump_apex_delta_m'] == pytest.approx(.28)


@pytest.mark.parametrize('source', ['contracts/v6_wheel_response_training_v1.json',
                                    'contracts/v6_transport_response_training_v1.json'])
def test_only_named_s1_same_learning_state_migration_is_allowed(source):
    before, _, _ = config_for(0, source=source)
    after, _, _ = config_for(0)
    _, receipt = response_contract_provenance(before, after, ROOT)
    assert receipt['kind'] == 'explicit_step_curriculum_resume'
    assert receipt['consumed_budget'] == 'preserved_no_refund'
    assert receipt['actor_critic_optimizer'] == 'preserved'
    assert set(receipt['retired_evaluation_case_names']) == RETIRED
    assert receipt['total_update_ceiling'] == 8500
    for field, value in [('total_updates', 1001), ('learning_rate', 1e-2), ('physics_dt', .005),
                         ('policy_dt', .005), ('actor_dim', 36), ('reward_profile', 'unreviewed')]:
        broken = deepcopy(after)
        broken[field] = value
        with pytest.raises(ValueError):
            validate_response_resume(before, broken)
    for index in (1, 4, 6):
        other, _, _ = config_for(index)
        with pytest.raises(ValueError):
            validate_response_resume(before, other)


@pytest.mark.parametrize('mutation', ['missing_module', 'disabled', 'missing_sensor', 'extra_sensor', 'frame30',
                                     'changed_old_case', 'future_promotion', 'geometry', 'budget_sha',
                                     'horizontal_beam', 'missing_ground_rejection'])
def test_fault_injection_rejected(mutation):
    config, seed, count = config_for(0)
    _, design, control = inputs()
    if mutation == 'missing_module':
        config.pop('step_jump_task')
    elif mutation == 'disabled':
        config['step_jump_task']['enabled'] = False
    elif mutation == 'missing_sensor':
        config['step_jump_task'].pop('range_sample_period_s')
    elif mutation == 'extra_sensor':
        config['step_jump_task']['unreviewed'] = True
    elif mutation == 'frame30':
        config['actor_reserved_frame30']['index'] = 31
    elif mutation == 'changed_old_case':
        config['evaluation']['cases'][0]['height_mae_m_max'] = 1.
    elif mutation == 'future_promotion':
        config['evaluation']['promotion_case_names'].append('step_jump_25_manual')
    elif mutation == 'geometry':
        next(case for case in config['evaluation']['cases'] if case['name'] == 'double_step_jump_200_250_manual')['terrain_limits']['stair_heights_m'] = [.2, .25]
    elif mutation == 'horizontal_beam':
        config['step_jump_task']['sensor_pitch_rad'] = 0.
    elif mutation == 'missing_ground_rejection':
        config['step_jump_task'].pop('ground_echo_margin_m')
    else:
        config['budget_spec_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        validate_design_contract(config, design, control, stage='S1', role='train', seed=seed, num_envs=count)


@pytest.mark.parametrize('key', ['step_jump_task', 'step_climb_task'])
def test_task_runtime_expectations_cannot_be_omitted(key):
    config, seed, count = config_for(0)
    _, design, control = inputs()
    broken = deepcopy(design)
    broken['stages']['S1']['roles']['train']['runtime']['modules'].pop(key)
    with pytest.raises(ValueError, match=key):
        validate_design_contract(config, broken, control, stage='S1', role='train', seed=seed, num_envs=count)


def test_paired_retention_retires_old_claims_without_requiring_future_new_skills_at_s1():
    old, _, _ = config_for(0, source='contracts/v6_transport_response_training_v1.json')
    new, _, _ = config_for(0)
    old_names = {case['name'] for case in old['evaluation']['cases']}

    def report(config, seed, target):
        settings = deepcopy(config['evaluation'])
        settings['seed'] = seed
        names = [case['name'] for case in settings['cases']]
        results = {}
        for name in names:
            passed = name in old_names  # New future skills deliberately fail.
            results[name] = {'checks': {'all_requested_episodes_accounted': True,
                                       'task_success': passed}, 'passed': passed,
                             'requested_episodes': 4}
        return {'status': 'evaluated', 'deterministic_actor': True, 'random_pushes': False,
                'case_resampling': False, 'coverage': 'full_manifest', 'evaluated_case_names': names,
                'settings': settings, 'contract_sha256': ('2' if target else '1') * 64,
                'episodes_per_case': 4, 'candidates': [{'checkpoint_sha256': 'a' * 64, 'cases': results}]}

    review = assess_response_retention(report(old, 11, False), report(old, 12, False),
        report(new, 11, True), report(new, 12, True), new['evaluation']['promotion_case_names'],
        migration_kind='v6_step_curriculum_v1')
    assert set(review['retired_case_names_not_credited']) == RETIRED
    assert review['extra_rehearsal_required'] is False
    assert len(review['added_unconfirmed_case_names']) == 23
    assert review['response_recovery_case_names'] == []


@pytest.mark.parametrize('exact', [False, True])
def test_all_normal_up_commands_have_no_hidden_timed_raise_and_down_is_preserved(exact):
    from types import SimpleNamespace
    import torch
    from wheeled_tasks.chassis.skill_commands import SkillCommands, profile_command

    config, _, _ = config_for(3)
    names = ['rolling_up_03', 'rolling_up_06', 'rolling_up_10', 'rolling_stairs_3x04',
             'body_climb_15', 'body_climb_20', 'body_climb_25', 'step_down_05']
    specs = {name: config['skill_specs'][name] for name in names}
    cfg = {'skill_specs': specs, 'command_reference': config['command_reference'],
           'evaluation_exact_cases': exact, 'evaluation': {'cases': [case for case in
              config['evaluation']['cases'] if case['name'] in names]}}
    n = len(names)
    env = SimpleNamespace(cfg=cfg, num_envs=n, device='cpu', scene_groups=names,
        commands=torch.zeros(n, 3), command_target=torch.zeros(n, 2), mode=torch.zeros(n, dtype=torch.long),
        training_transitions=0, episode_length_buf=torch.zeros(n), policy_dt=.02,
        random=lambda count: torch.full((count,), .5), push_enabled=torch.zeros(n, dtype=torch.bool))
    for field in ('command_clock', 'height_clock', 'push_clock', 'episode_limits'):
        setattr(env, field, torch.zeros(n))
    runtime = SkillCommands(env)
    runtime.sample(torch.arange(n))
    assert runtime.manual_step_mode.tolist() == [0.] * 7 + [-1.]
    assert torch.isinf(runtime.manual_step_start[:7]).all()
    assert (runtime.manual_step_duration[:7] == 0).all()
    assert torch.isfinite(runtime.manual_step_start[-1])
    for name in names[:7]:
        values = profile_command(torch.arange(0., 25., .05), specs[name]['command'], specs[name])
        torch.testing.assert_close(values[:, 0], torch.full_like(values[:, 0], .3))
        torch.testing.assert_close(values[:, 2], torch.full_like(values[:, 2], .305))
