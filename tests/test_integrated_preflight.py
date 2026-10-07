"""Independent scoped expectations and real native CPU instances, no PhysX/PPO."""
import ast
from copy import deepcopy

import pytest
import torch

from test_integrated_native_routing import environment, bind_env_methods, later
from test_integrated_runtime import sources, lineage_fixture, first
from test_integrated_lineage import fixture as original_lineage_fixture
from test_ordinary_posture_curriculum import ROOT, ASSET, MANIFEST, NAMES, reset_buffers
from wheeled_tasks.chassis.integrated_preflight import (
    FIELD, OBSERVED, scoped_expectation, with_scoped_expectation,
    observe_component_clocks, validate_integrated_runtime_state)
from wheeled_tasks.chassis.nonstandard_start import NonstandardStart, runtime_expectation as birth_expected
from wheeled_tasks.chassis.ordinary_posture_curriculum import (
    OrdinaryPostureCurriculum, ordinary_eligibility, runtime_expectation as posture_expected)
from wheeled_tasks.chassis.task import choose_scene_groups


def base_expected(config):
    # Independent fixture physics. These atoms are not read from actual instances.
    return dict(physics_dt=.005, policy_dt=.02, decimation=4, pc_control_dt=.005,
        seed=1729, modules=dict(ordinary_posture_curriculum=True, nonstandard_start=True),
        leg_kp=160., leg_kd=2.5, wheel_velocity_p=.6, precision_tracking=False,
        ordinary_posture_curriculum=posture_expected(config['ordinary_posture_curriculum']),
        nonstandard_start=birth_expected(config['nonstandard_start'], .02),
        locomotion_difficulty=dict(enabled=True, settings=deepcopy(config['locomotion_difficulty'])))


@pytest.fixture
def prepared(lineage_fixture, sources):
    config = later(lineage_fixture, sources, 'terrain')
    env = environment(config)
    population = choose_scene_groups(config['scene_groups'], 4096)
    env.scene_groups, kinds = map(list, zip(*population))
    env.num_envs = len(population)
    bind_env_methods(env)
    spin = torch.tensor([config['skill_specs'][name].get('kind') == 'spin_translate'
        and not config.get('special_mode_activation', {}).get('spin_translate') for name in env.scene_groups])
    eligible = ordinary_eligibility(config, env.scene_groups, kinds, 'cpu', spin=spin)
    posture = OrdinaryPostureCurriculum(env.num_envs, 'cpu', config['ordinary_posture_curriculum'], eligible)
    owners, settings = env._component_configuration('ordinary_posture_curriculum')
    posture.configure_components(owners, settings)
    birth = NonstandardStart(env.num_envs, 'cpu', .005, config['nonstandard_start'], ROOT, NAMES)
    owners, settings = env._component_configuration('nonstandard_start')
    birth.configure_components(owners, settings)
    env.ordinary_posture, env.nonstandard_start = posture, birth
    posture.bind_geometry(ASSET, MANIFEST, NAMES, birth)
    ids, root, joints, origins, surfaces, valid = reset_buffers(posture)
    clocks = env._curriculum_clock('ordinary_posture_curriculum')
    posture.reset(ids, clocks, generator=env.generator)
    posture.sample_intervals(ids, generator=env.generator)
    posture.initial_orientation(ids, root)
    posture.align_support(ids, root, joints, origins, surfaces, valid)
    birth.reset(ids, env._curriculum_clock('nonstandard_start'), eligible=eligible, generator=env.generator)
    actual = dict(physics_dt=.005, policy_dt=.02, decimation=4, pc_control_dt=.005,
        seed=env.generator.initial_seed(), modules=dict(ordinary_posture_curriculum=posture.enabled,
            nonstandard_start=birth.enabled), leg_kp=160., leg_kd=2.5, wheel_velocity_p=.6,
        precision_tracking=False, ordinary_posture_curriculum=posture.runtime_state(),
        nonstandard_start=birth.runtime_state(.02), locomotion_difficulty=dict(enabled=True,
            settings=deepcopy(config['locomotion_difficulty']),
            row_components={name: owner['difficulty_component'] for name, owner in
                config['integrated_runtime_candidate']['row_ownership'].items()}))
    actual['ordinary_posture_curriculum']['runtime_observed'] = posture.observed(clocks)
    actual[OBSERVED] = observe_component_clocks(env)
    expected = with_scoped_expectation(base_expected(config), config, num_envs=env.num_envs)
    return env, expected, actual


def validate(prepared):
    env, expected, actual = prepared
    return validate_integrated_runtime_state(expected, actual, config=env.cfg,
        num_envs=env.num_envs, actor_update=env.task_actor_update)


def test_actual_cpu_instances_validate_without_a_global_posture_cap(prepared):
    env, expected, actual = prepared
    result = validate(prepared)
    observed = actual['ordinary_posture_curriculum']['runtime_observed']
    assert 'birth_angle_cap_rad' not in observed
    assert observed['component/recovery/birth_angle_cap_rad'] == .15
    assert observed['component/terrain/birth_angle_cap_rad'] == .03
    assert observed['component/recovery/relative_actor_updates'] == 20000 - 2294
    assert observed['component/terrain/relative_actor_updates'] == 0
    assert result['status'] == 'runtime_configuration_passed'
    assert result['runtime_ready'] is False and result['qualification'] is None
    assert result['functional_transport_precision_randomization_probes'] == 'not_run'
    assert actual[OBSERVED]['component_actor_updates']['speed'] == 5000
    assert actual[OBSERVED]['nonstandard_start']['terrain'] == 0
    assert expected[FIELD]['runtime_fields']['nonstandard_start']['component_settings'] == env.nonstandard_start.component_settings


@pytest.mark.parametrize('patch', [
    'missing_design', 'design_birth_hold', 'actual_birth_origin', 'actual_clock',
    'mature_cap', 'global_cap', 'population', 'geometry', 'timer',
    'physics', 'pd', 'module', 'extra_birth_field', 'wrong_difficulty_owner',
    'positive_population', 'component_mask_population', 'missing_birth_component',
])
def test_bad_component_or_native_physical_state_is_rejected(prepared, patch):
    env, expected, actual = prepared
    if patch == 'missing_design':
        expected.pop(FIELD)
    elif patch == 'design_birth_hold':
        expected[FIELD]['runtime_fields']['nonstandard_start']['component_settings']['terrain']['hold_seconds'] = .3
    elif patch == 'actual_birth_origin':
        actual['nonstandard_start']['component_settings']['terrain']['origin_actor_update'] = 5000
    elif patch == 'actual_clock':
        actual[OBSERVED]['nonstandard_start']['terrain'] = 5000
    elif patch == 'mature_cap':
        actual['ordinary_posture_curriculum']['runtime_observed']['component/recovery/birth_angle_cap_rad'] = .03
    elif patch == 'global_cap':
        actual['ordinary_posture_curriculum']['runtime_observed']['birth_angle_cap_rad'] = .15
    elif patch == 'population':
        actual['ordinary_posture_curriculum']['runtime_observed']['birth_selected_count'] = 0
    elif patch == 'geometry':
        actual['ordinary_posture_curriculum']['runtime_observed']['geometry_bound'] = False
    elif patch == 'timer':
        actual['ordinary_posture_curriculum']['runtime_observed']['push_clock_min_s'] = 2.
    elif patch == 'physics':
        actual['physics_dt'] = .01
    elif patch == 'pd':
        actual['leg_kp'] = 80.
    elif patch == 'module':
        actual['modules']['nonstandard_start'] = False
    elif patch == 'extra_birth_field':
        actual['nonstandard_start']['component_settings']['terrain']['unvalidated'] = True
    elif patch == 'wrong_difficulty_owner':
        actual['locomotion_difficulty']['row_components']['stand_305mm'] = None
    elif patch == 'positive_population':
        actual['ordinary_posture_curriculum']['runtime_observed']['eligible_count'] -= 1
    elif patch == 'component_mask_population':
        actual[OBSERVED]['module_populations']['nonstandard_start']['component_mask_counts']['terrain'] -= 1
    elif patch == 'missing_birth_component':
        del actual['nonstandard_start']['component_settings']['terrain']
    with pytest.raises(ValueError, match='preflight rejected'):
        validate(prepared)


def test_named_design_can_carry_scope_without_changing_native_baselines(lineage_fixture, sources):
    config = first(lineage_fixture, sources)
    native = base_expected(config)
    before = deepcopy(native)
    declared = with_scoped_expectation(native, config, num_envs=4096)
    assert native == before
    assert {key: value for key, value in declared.items() if key != FIELD} == before
    assert declared[FIELD] == scoped_expectation(config, num_envs=4096)
    assert declared[FIELD]['clocks_at_source']['component_actor_updates'] == {'recovery': 15000, 'speed': 0}
    assert declared[FIELD]['runtime_fields']['ordinary_posture_curriculum']['component_settings']['recovery']['origin_actor_update'] == 2294
    assert sum(declared[FIELD]['row_population'].values()) == 4096


@pytest.mark.parametrize('config', [{}, {'evaluation_exact_cases': True, 'integrated_runtime_candidate': {'schema': 'wrong'}}])
def test_default_and_fixed_eval_delegate_to_unchanged_native_validator(config):
    expected = dict(physics_dt=.005, policy_dt=.02, decimation=4, pc_control_dt=.005,
        seed=7, modules={}, leg_kp=160., leg_kd=2.5, wheel_velocity_p=.6, precision_tracking=False)
    actual = deepcopy(expected)
    result = validate_integrated_runtime_state(expected, actual, config=config, num_envs=1, actor_update=0)
    assert 'scoped_validation' not in result
    actual['physics_dt'] = .01
    with pytest.raises(ValueError, match='Runtime preflight rejected'):
        validate_integrated_runtime_state(expected, actual, config=config, num_envs=1, actor_update=0)


def test_env_keeps_original_validator_in_non_candidate_branch():
    tree = ast.parse((ROOT / 'src/wheeled_tasks/chassis/env.py').read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute) and node.test.attr == 'integrated_runtime_enabled'
        and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == 'validate_integrated_runtime_state' for body in node.body for call in ast.walk(body)))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'validate_runtime_state' for statement in branch.orelse for node in ast.walk(statement))
