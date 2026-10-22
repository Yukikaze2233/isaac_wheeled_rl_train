"""Mixed-course ownership and accounting; these CPU fixtures do not launch PPO."""
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.application.configuration.small_rough_course import (
    KEY, ROLES, MixedCourseExposure, allocate, install, selection, validate_resume)
from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import reorder_fixed_flat_rehearsal
from wheeled_tasks.chassis.tasks.commands.rough_commands import ScutRoughCommands
from wheeled_tasks.chassis.tasks.episodes import choose_scene_groups
from wheeled_tasks.chassis.application.configuration.training_profile import restore_profile_clocks
from wheeled_tasks.chassis.application.training.policy_transfer import formal_stage_load_kwargs


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'reports/v7_formal_training_20261012'


def candidate():
    return json.loads((ROOT / 'configs/v7_formal_20261012/mixed_tracking_repair_063.json').read_text())


@pytest.mark.parametrize('fractions,count,expected', [
    ([.98, .01, .01], 3, [1, 1, 1]),
    ([.8, .1, .1], 5, [3, 1, 1]),
    ([.5, .5], 3, [2, 1]),
])
def test_small_scene_allocations_keep_coverage_and_exact_population(fractions, count, expected):
    groups = [dict(name=str(i), fraction=fraction, terrain=['flat'])
              for i, fraction in enumerate(fractions)]
    scenes = choose_scene_groups(groups, count)
    counts = Counter(name for name, _ in scenes)
    assert len(scenes) == count
    assert [counts[group['name']] for group in groups] == expected


def test_recovery_scene_minimum_includes_every_configured_group():
    config = json.loads((ROOT / 'configs/v7_formal_20261012/recovery_after10000_056.json').read_text())
    groups = config['scene_groups']
    scenes = choose_scene_groups(groups, len(groups))
    assert [name for name, _ in scenes] == [group['name'] for group in groups]


def setup(count=256):
    config = candidate()
    groups = [group for group, _ in choose_scene_groups(config['scene_groups'], count)]
    spec = allocate(config, ROOT, count, groups)
    order = list(reversed(range(count)))
    spec = reorder_fixed_flat_rehearsal(spec, order)
    roles = torch.tensor(spec['population_index'])
    tiles = torch.zeros(10, 14, 3)
    tiles[:, :, 0] = ((torch.arange(10) + .5) * 9. - 45.)[:, None]
    tiles[:, :, 1] = ((torch.arange(14) + .5) * 9. - 63.)[None, :]
    origins = tiles[0, torch.arange(count) % 14].clone()
    for native, original in enumerate(order):
        if original in spec['origins_by_env']:
            origins[native] = torch.tensor(spec['origins_by_env'][original])
    commands = ScutRoughCommands(count, 'cpu', terrain_origins=tiles, env_origins=origins)
    recover = roles == ROLES.index('recovery')
    install(spec, commands, recover)
    return config, spec, commands, roles, recover, origins


def test_population_reordering_preserves_geometry_requests_and_recovery():
    config, spec, commands, roles, recover, origins = setup()
    plane = torch.tensor(spec['flat_lane_mask'])
    assert torch.equal(plane, roles != ROLES.index('source_rough'))
    assert (origins[plane, 1] > 100).all() and (origins[~plane, 1] < 100).all()
    assert commands.fixed_request_mask.sum() == 64
    assert commands._temporal_rehearsal.mask.sum() == 32
    assert commands._rough['source_flat_rehearsal']['mask'].sum() == 96
    assert recover.sum() == 32
    rng = torch.Generator().manual_seed(1)
    commands.reset(torch.arange(256), torch.zeros(256), torch.zeros(256, dtype=torch.bool), rng, 0)
    selected = roles == ROLES.index('temporal_flat')
    assert torch.unique(commands.command3[selected, 0], return_counts=True)[1].tolist() == [4] * 8
    original = commands.command3[selected, 0].clone()
    for _ in range(250):
        commands.begin_policy()
        for _ in range(4):
            commands.advance_physics()
        phase = commands.diagnostics()['temporal_published_phase']
        assert (phase[selected] == 0).all()
        torch.testing.assert_close(commands.command3[selected, 0], original)
        commands.advance(.02, torch.zeros(256), 0, rng, world_positions=origins)
    assert (commands.command3[selected, 0] == 0).all()
    assert (commands.diagnostics()['temporal_published_phase'][selected] == 1).all()


def test_native_birth_masks_keep_rough_temporal_and_fixed_nominal():
    config, spec, commands, roles, recover, _ = setup()
    tree = ast.parse((ROOT / 'src/wheeled_tasks/chassis/application/environment.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ChassisEnv')
    names = {'_configure_dynamic_birth', '_dynamic_birth_runtime_state'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'torch': torch, '__package__': 'wheeled_tasks.chassis'}
    exec(compile(ast.Module(methods, type_ignores=[]), '<actual-env-methods>', 'exec'), scope)
    env = NS(cfg=config, flat_episode=NS(commands=commands, device=torch.device('cpu')),
             recovery=NS(enabled=recover), rehearsal_spec=spec, num_envs=256)
    scope['_configure_dynamic_birth'](env, training=True)
    assert torch.equal(env._dynamic_birth_source_mask, roles == ROLES.index('dynamic_flat'))
    assert (env._dynamic_birth_nominal_mask[roles == ROLES.index('temporal_flat')]).all()
    state = scope['_dynamic_birth_runtime_state'](env)
    assert state['source_lanes'] == 64 and state['scripted_lanes'] == 32


def test_final_loss_mask_and_paid_phase_accounting_commit_only_after_optimizer():
    _, spec, _, roles, _, _ = setup()
    exposure = MixedCourseExposure(spec, 4, 'cpu')
    exposure.required_frames.fill_(1)
    selected = roles == ROLES.index('temporal_flat')
    for i in range(4):
        phase = torch.where(selected, i, -1)
        exposure.observe(i, {'diagnostics': {'source_flat_episode': {
            'temporal_published_phase': phase, 'episode_id': torch.zeros(256, dtype=torch.long)},
            'done': torch.full((256,), i == 3), 'timeout': torch.full((256,), i == 3),
            'boundary': torch.zeros(256, dtype=torch.bool),
            'terminated': torch.zeros(256, dtype=torch.bool)}})
    mask = torch.ones(4, 256, dtype=torch.bool)
    mask[:, roles == ROLES.index('recovery')] = False
    mask[1, selected] = False
    prepared = exposure.prepare(mask)
    assert all(v == 0 for v in exposure.counts.values())
    complete = exposure.prepare(torch.ones_like(mask))
    assert complete['completed'] == 32
    counts = exposure.commit(prepared)
    assert exposure.complete_loss_sequences == 0
    assert sum(counts[k] for k in ROLES) == int(mask.sum())
    assert counts['stop'] == 0 and counts['drive'] == counts['reverse'] == counts['final_stop'] == 32
    restored = MixedCourseExposure(spec, 4, 'cpu')
    restored.load_state_dict(exposure.state_dict())
    assert restored.state_dict() == exposure.state_dict()
    bad = deepcopy(exposure.state_dict())
    bad['direct_loss_transitions']['drive'] += 1
    with pytest.raises(ValueError):
        restored.load_state_dict(bad)


def test_entry_restarts_only_local_clock_and_first_stage_optimizer():
    config = candidate()
    old = json.loads((ROOT / 'configs/v7_formal_20261012/recovery_after10000_056.json').read_text())
    assert validate_resume(old, config)['stage_transition']
    assert not validate_resume(config, config)['stage_transition']
    infos = dict(stage='foundation', stage_actor_update=14841, global_actor_update_offset=0,
                 global_actor_update=14841, actor_updates_consumed=16594)
    clocks = restore_profile_clocks(config, infos, stage='terrain', requested={})
    assert clocks == dict(stage_actor_update=0, global_actor_update_offset=14841,
                          global_actor_update=14841, actor_updates_consumed=16594)
    load = formal_stage_load_kwargs(config, infos, stage='terrain')['load_cfg']
    assert load['actor'] and load['critic'] and not load['optimizer']
    assert formal_stage_load_kwargs(config, dict(infos, stage='terrain'), stage='terrain') == {}
    for change in ({'velocity_iterations': 0}, {'v5_control': {}}, {'recovery_training': {}},
                   {'formal_closure_monitor_only': False}, {'entropy_coef': 0.}):
        with pytest.raises(ValueError):
            validate_resume(old, dict(config, **change))


def test_unknown_or_incomplete_mixture_is_rejected():
    config = candidate()
    for key, value in (('phase_policy_ticks', [250] * 4), ('speeds_m_s', [.05]), ('height_m', .6)):
        changed = deepcopy(config)
        changed[KEY]['temporal'][key] = value
        with pytest.raises(ValueError):
            selection(changed)
    config['scene_groups'][0]['terrain'] = ['rough']
    with pytest.raises(ValueError):
        selection(config)


@pytest.mark.parametrize("revision", ["upper", "leg", "rough_wheel"])
def test_contact_repair_keeps_optimizer_and_all_course_clocks(revision):
    from wheeled_tasks.chassis.simulation.terrain.contact_generation_distance import (
        KEY as CONTACT_KEY, VERSION as BASE_CONTACT_VERSION, LEG_LINK_VERSION,
        ROUGH_WHEEL_VERSION, UPPER_LINK_VERSION)

    old = candidate()
    before, after = {"upper": (BASE_CONTACT_VERSION, UPPER_LINK_VERSION),
                     "leg": (UPPER_LINK_VERSION, LEG_LINK_VERSION),
                     "rough_wheel": (LEG_LINK_VERSION, ROUGH_WHEEL_VERSION)}[revision]
    old[CONTACT_KEY] = before
    repaired = dict(old, **{CONTACT_KEY: after})
    receipt = validate_resume(old, repaired)
    assert receipt["changed_contract_fields"] == [CONTACT_KEY]
    assert receipt["optimizer"] == "inherited"
    assert not receipt["stage_transition"] and not receipt["source_birth_population_changed"]
    assert not receipt["budget_counter_reset"]
    infos = dict(stage="terrain", stage_actor_update=177, global_actor_update_offset=14841,
                 global_actor_update=15018, actor_updates_consumed=16772)
    assert restore_profile_clocks(repaired, infos, stage="terrain", requested={}) == {
        k: infos[k] for k in infos if k != "stage"}
    assert formal_stage_load_kwargs(repaired, infos, stage="terrain") == {}
    with pytest.raises(ValueError, match="explicit contact revision"):
        validate_resume(repaired, old)
    for change in ({"velocity_iterations": 0}, {"entropy_coef": 0.},
                   {"formal_closure_monitor_only": False}):
        with pytest.raises(ValueError):
            validate_resume(old, dict(repaired, **change))


@pytest.mark.parametrize("repair", [False, True])
def test_actual_constructor_selects_flat_payments_for_all_four_flat_roles(repair):
    tree = ast.parse((ROOT / 'src/wheeled_tasks/chassis/application/environment.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ChassisEnv')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    branch = next(n for n in ast.walk(init) if isinstance(n, ast.If)
        and any(isinstance(child, ast.ImportFrom) and child.module == 'wheeled_tasks.chassis.simulation.runtime.flat_reward_runtime'
                for child in n.body))
    config = candidate()
    config.pop('formal_tracking_reward_repair', None)
    if repair:
        config['formal_tracking_reward_repair'] = 'v7_exact_command_tracking_v1'
    config['scut_reward_mapping'].update(undesired_link_names=['base'], terminal_link_names=['base'])
    mask = (True, True, True, True, False)
    env = NS(reward_selection={'scut_source_selected': True},
        rehearsal_spec={'settings': config['formal_flat_rehearsal'], 'flat_lane_mask': mask},
        robot=NS(body_names=['base', 'left', 'right']), startup_report={}, dt=.005,
        source_flat_rehearsal_mask=None)
    scope = {'__package__': 'wheeled_tasks.chassis', 'self': env, 'torch': torch,
             'config': config, 'num_envs': 5, 'device': 'cpu'}
    exec(compile(ast.Module([branch], type_ignores=[]), '<actual-reward-constructor>', 'exec'), scope)
    assert torch.equal(env.source_flat_rehearsal_mask, torch.tensor(mask))
    assert env.scut_reward.flat_rehearsal_reward_profile == 'scut0719_flat_v1'
    assert env.scut_reward.tracking_reward_repair == (
        'v7_exact_command_tracking_v1' if repair else None)


def test_optimization_readback_uses_only_direct_loss_samples_and_keeps_state():
    _, spec, _, roles, _, _ = setup()
    exposure = MixedCourseExposure(spec, 4, 'cpu')
    temporal = roles == ROLES.index('temporal_flat')
    for index in range(4):
        exposure.observe(index, {'diagnostics': {'source_flat_episode': {
            'temporal_published_phase': torch.where(temporal, index, -1),
            'episode_id': torch.zeros(256, dtype=torch.long)},
            'done': torch.zeros(256, dtype=torch.bool),
            'timeout': torch.zeros(256, dtype=torch.bool),
            'boundary': torch.zeros(256, dtype=torch.bool),
            'terminated': torch.zeros(256, dtype=torch.bool)}})
    mask = torch.ones(4, 256, dtype=torch.bool)
    mask[:, roles == ROLES.index('recovery')] = False
    mask[1, temporal] = False
    mask[-1, -3:] = False
    values = torch.arange(1024, dtype=torch.float32).reshape(4, 256, 1) / 100.
    residual = torch.arange(1024, dtype=torch.float32).reshape(4, 256, 1) % 7 - 3.
    storage = NS(values=values.clone(), returns=values + residual, advantages=residual / 2.)
    for tensor in vars(storage).values():
        tensor[~mask] = float('nan')
    original = {key: value.clone() for key, value in vars(storage).items()}
    rng, state = torch.get_rng_state(), deepcopy(exposure.state_dict())
    fixed = torch.tensor(spec['case_index']) + 1
    names = ('non_rehearsal', *spec['case_ids'])
    metrics = exposure.optimization_metrics(storage, mask, fixed, names)
    prefix = 'MixedCourse/optimization/'
    assert metrics[prefix + 'role/recovery/transitions'] == 0
    assert metrics[prefix + 'phase/stop/transitions'] == 0
    assert not metrics[prefix + 'role/recovery/explained_variance_valid']
    assert all(torch.isfinite(value) for value in metrics.values())
    assert sum(metrics[prefix + 'role/' + name + '/transitions'] for name in ROLES) == mask.sum()
    for label, selected in (
            ('role/dynamic_flat', mask & (roles == ROLES.index('dynamic_flat'))),
            ('phase/drive', mask & temporal[None, :] & (torch.arange(4)[:, None] == 0)),
            ('fixed/' + names[1], mask & (fixed == 1))):
        expected = residual.squeeze(-1)[selected]
        assert metrics[prefix + label + '/transitions'] == expected.numel()
        torch.testing.assert_close(metrics[prefix + label + '/gae_residual_mean'], expected.mean())
        torch.testing.assert_close(metrics[prefix + label + '/normalized_advantage_mean'], expected.mean() / 2.)
        torch.testing.assert_close(metrics[prefix + label + '/positive_advantage_fraction'], (expected > 0).float().mean())
        target = storage.returns.squeeze(-1)[selected]
        torch.testing.assert_close(metrics[prefix + label + '/target_variance'], target.var(unbiased=False))
        torch.testing.assert_close(metrics[prefix + label + '/explained_variance'],
            1 - expected.var(unbiased=False) / target.var(unbiased=False))
    for key, value in original.items():
        torch.testing.assert_close(vars(storage)[key], value, equal_nan=True)
    assert torch.equal(torch.get_rng_state(), rng) and exposure.state_dict() == state
