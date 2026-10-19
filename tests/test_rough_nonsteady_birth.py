"""Continuous rough births retain ownership, migration and inference semantics."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.fixed_flat_rehearsal import reorder_fixed_flat_rehearsal
from wheeled_tasks.chassis.rough_commands import ScutRoughCommands
from wheeled_tasks.chassis.small_rough_course import (
    ROLES, ROUGH_ANCHOR_KEY, ROUGH_ANCHOR_VERSION, ROUGH_BIRTH_KEY, ROUGH_BIRTH_VERSION,
    allocate, install, selection, validate_resume,
)
from wheeled_tasks.chassis.small_rough_evaluation import configuration, retained_flat_configuration
from wheeled_tasks.chassis.task import choose_scene_groups


ROOT = Path(__file__).resolve().parents[1]


def candidate():
    return json.loads((ROOT / 'configs/v7_formal_20261012/motion_retention_review.json').read_text())


def birth_methods():
    tree = ast.parse((ROOT / 'src/wheeled_tasks/chassis/env.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ChassisEnv')
    names = {'_configure_dynamic_birth', '_dynamic_birth_runtime_state', '_configure_rough_birth_anchor'}
    scope = {'torch': torch, '__package__': 'wheeled_tasks.chassis'}
    nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(nodes, type_ignores=[]), '<native-birth-ownership>', 'exec'), scope)
    return scope


def build(config=None):
    config = candidate() if config is None else config
    count = 256
    groups = [name for name, _ in choose_scene_groups(config['scene_groups'], count)]
    spec = allocate(config, ROOT, count, groups)
    order = sorted(range(count), key=lambda index: f'/World/envs/env_{index}/Robot')
    spec = reorder_fixed_flat_rehearsal(spec, order)
    roles = torch.tensor(spec['population_index'])
    tiles = torch.zeros(10, 14, 3)
    tiles[:, :, 0] = (torch.arange(10) * 9. - 40.5)[:, None]
    tiles[:, :, 1] = (torch.arange(14) * 9. - 58.5)[None, :]
    origins = tiles[0, torch.arange(count) % 14].clone()
    commands = ScutRoughCommands(count, 'cpu', terrain_origins=tiles, env_origins=origins)
    script = roles == ROLES.index('recovery')
    install(spec, commands, script)
    env = NS(cfg=config, flat_episode=NS(commands=commands, device=torch.device('cpu')),
             recovery=NS(enabled=script), rehearsal_spec=spec, num_envs=count)
    return env, roles


def test_actual_native_ownership_randomizes_both_source_surfaces_only():
    env, roles = build()
    methods = birth_methods()
    before = torch.get_rng_state()
    methods['_configure_dynamic_birth'](env, training=True)
    source = (roles == ROLES.index('dynamic_flat')) | (roles == ROLES.index('source_rough'))
    assert torch.equal(env._dynamic_birth_source_mask, source)
    assert torch.equal(env._dynamic_birth_nominal_mask, ~source)
    state = methods['_dynamic_birth_runtime_state'](env)
    assert state['source_lanes'] == 128
    assert state['flat_source_lanes'] == state['rough_source_lanes'] == 64
    assert state['nominal_lanes'] == 96 and state['scripted_lanes'] == 32
    assert not (source & env.recovery.enabled).any()
    assert not (source & env.flat_episode.commands.fixed_request_mask).any()
    assert torch.equal(torch.get_rng_state(), before)


def test_legacy_motion_selector_keeps_rough_nominal():
    config = candidate()
    config.pop(ROUGH_BIRTH_KEY)
    config.pop(ROUGH_ANCHOR_KEY)
    env, roles = build(config)
    methods = birth_methods()
    methods['_configure_dynamic_birth'](env, training=True)
    assert torch.equal(env._dynamic_birth_source_mask, roles == ROLES.index('dynamic_flat'))
    state = methods['_dynamic_birth_runtime_state'](env)
    assert state['source_lanes'] == 64 and 'rough_source_lanes' not in state


def test_inference_keeps_all_source_births_nominal_even_without_training_allocation():
    env, _ = build()
    env.rehearsal_spec = None
    methods = birth_methods()
    methods['_configure_dynamic_birth'](env, training=False)
    state = methods['_dynamic_birth_runtime_state'](env)
    assert env._dynamic_birth_nominal_mask.all() and not env._dynamic_birth_source_mask.any()
    assert state['source_lanes'] == state['flat_source_lanes'] == state['rough_source_lanes'] == 0


def test_selector_changes_only_birth_population_and_preserves_parent_handoff():
    new = candidate()
    old = json.loads((ROOT / 'configs/v7_formal_20261012/recovery_after10000_056.json').read_text())
    receipt = validate_resume(old, new)
    assert receipt['stage_transition'] and receipt['source_birth_population_changed']
    assert receipt['budget_counter_reset'] is False
    previous = deepcopy(new)
    previous.pop(ROUGH_BIRTH_KEY)
    previous.pop(ROUGH_ANCHOR_KEY)
    receipt = validate_resume(previous, new)
    assert not receipt['stage_transition'] and receipt['optimizer'] == 'inherited'
    assert receipt['source_birth_population_changed']
    assert receipt['changed_contract_fields'] == sorted([ROUGH_BIRTH_KEY, ROUGH_ANCHOR_KEY])
    assert validate_resume(new, new)['source_birth_population_changed'] is False
    with pytest.raises(ValueError, match='forward migration'):
        validate_resume(new, previous)
    assert new['research_training_authorized'] is False


@pytest.mark.parametrize('value', [None, False, 'unknown', {'enabled': True}])
def test_unknown_birth_selectors_are_not_silently_ignored(value):
    config = candidate()
    config[ROUGH_BIRTH_KEY] = value
    with pytest.raises(ValueError, match='explicit motion-course selector'):
        selection(config)
    with pytest.raises(ValueError, match='explicit mixed motion course'):
        selection({ROUGH_BIRTH_KEY: ROUGH_BIRTH_VERSION})


def test_nominal_regression_overlays_do_not_inherit_training_random_births():
    parent = candidate()
    rough = configuration(parent)
    assert ROUGH_BIRTH_KEY not in rough
    assert ROUGH_ANCHOR_KEY not in rough
    flat = retained_flat_configuration(parent,
        [{'name': 'stand', 'terrain': ['flat'], 'fraction': 1., 'vx': 0., 'yaw_rate': 0., 'height': .305}],
        scripted=False)
    assert ROUGH_BIRTH_KEY not in flat
    assert ROUGH_ANCHOR_KEY not in flat
    assert rough['formal_foundation_nominal_birth'] and flat['formal_foundation_nominal_birth']
    assert parent[ROUGH_BIRTH_KEY] == ROUGH_BIRTH_VERSION


def test_center_anchor_changes_only_native_rough_lanes_and_legacy_remains_untouched():
    env, roles = build()
    count = len(roles)
    env.origins = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
    env.reset_positions = env.origins + torch.tensor([-1.5, 0., .32])
    selected = roles == ROLES.index('source_rough')
    env.kinds = ['rough' if lane else 'flat' for lane in selected]
    env.startup_report = {}
    before = env.reset_positions.clone()
    method = birth_methods()['_configure_rough_birth_anchor']
    method(env)
    assert torch.equal(env.reset_positions[selected, :2], env.origins[selected, :2])
    assert torch.equal(env.reset_positions[~selected], before[~selected])
    assert torch.equal(env.reset_positions[:, 2], before[:, 2])
    assert env.startup_report['rough_birth_anchor'] == {
        'version': ROUGH_ANCHOR_VERSION, 'rough_lanes': 64,
        'x_offset_m': 0., 'y_offset_m': 0., 'maximum_xy_error_m': 0.}
    env.cfg.pop(ROUGH_ANCHOR_KEY)
    env.reset_positions.copy_(before)
    method(env)
    assert torch.equal(env.reset_positions, before)


def test_anchor_migration_preserves_learning_state_and_rejects_silent_rollback():
    new = candidate()
    old = deepcopy(new)
    old.pop(ROUGH_ANCHOR_KEY)
    receipt = validate_resume(old, new)
    assert receipt['rough_birth_anchor_changed']
    assert not receipt['source_birth_population_changed']
    assert receipt['optimizer'] == 'inherited' and not receipt['budget_counter_reset']
    assert receipt['changed_contract_fields'] == [ROUGH_ANCHOR_KEY]
    with pytest.raises(ValueError, match='forward migration'):
        validate_resume(new, old)
    new[ROUGH_ANCHOR_KEY] = 'unknown'
    with pytest.raises(ValueError, match='explicit continuous-birth selector'):
        selection(new)


def test_center_anchor_rejects_mismatched_native_surface_ownership():
    env, roles = build()
    env.origins = torch.zeros(len(roles), 3)
    env.reset_positions = env.origins.clone()
    env.kinds = ['flat'] * len(roles)
    env.startup_report = {}
    with pytest.raises(ValueError, match='native rough ownership'):
        birth_methods()['_configure_rough_birth_anchor'](env)
