#!/usr/bin/env python3
"""Freeze the reviewed eight-stage task extension from prior independent rows.

This generator does not call stage_contract or fixed_suite_contract. It copies
unchanged expectations and constructs changed pools/cases explicitly; the CPU
preflight tests compare these rows against the executable materializer.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = dict(zip(('S1', 'S2', 'S3', 'S4', 'S5', 'S5T', 'S5D', 'S6'),
                  (1000, 2000, 1000, 1500, 1000, 600, 400, 1000)))
RETIRED = {'stairs_200_250', 'stairs_down_200_250', 'step_up_15', 'step_up_20',
           'step_up_25', 'heldout_step_up_payload_grip', 'step_up_03', 'step_up_06', 'step_up_10', 'stairs'}
ROLLING_RENAMES = {'step_up_03': 'rolling_up_03', 'step_up_06': 'rolling_up_06',
                   'step_up_10': 'rolling_up_10', 'stairs': 'rolling_stairs_3x04'}
REVISION = 'v6_step_curriculum_v1'
SCOPE = 'retired_invalid_case_not_retained;new_double_step_rises_0.20_0.25_tops_0.20_0.45'


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    path = ROOT / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jump_recipe(rises, source):
    double = len(rises) == 2
    return {'name': (f'double_step_jump_200_250_{source}' if double else
                     f'step_jump_{round(rises[0] * 100):02d}_{source}'),
            'skill': 'double_step_jump' if double else 'step_jump',
            'introduced_stage': 'S5D' if double else 'S5T',
            'command': [.4, 0., .305], 'episode_seconds': 32. if double else 20.,
            'step_jump': {'rises_m': rises, 'treads_m': [.8, .6] if double else [.8],
                'landing_clearance_m': .08, 'launch_distance_m': .25,
                'approach_speed_m_s': .4, 'landing_margin_m': .10,
                'settle_seconds': .35, 'trigger_source': source}}


def climb_recipe(rises):
    double = len(rises) == 2
    return {'name': 'body_climb_double_200_250' if double else f'body_climb_{round(rises[0]*100):02d}',
            'skill': 'body_climb', 'introduced_stage': 'S5D' if double else 'S4',
            'command': [.3, 0., .305], 'episode_seconds': 32. if double else 20.,
            'body_climb': {'rises_m': rises, 'treads_m': [.8, .6] if double else [.8],
                'approach_speed_m_s': .3, 'body_contact_min_force_n': 20.,
                'support_hold_seconds': .10, 'settle_seconds': .35, 'landing_margin_m': .06}}


def explicit_task_spec(item, base_limits):
    """Reviewed task row, deliberately independent of skill_spec implementation."""
    if item['skill'] in ('rolling_up', 'rolling_stairs'):
        limits = {**base_limits, **item['terrain_limits']}
        bare = {'kind': item['skill'], 'command': list(item['command']), 'mode': 2, 'sample_amplitude': False}
        train = {**deepcopy(bare), 'terrain_limits': limits, 'episode_seconds': item['episode_seconds'],
                 'sample_yaw_sign': False, 'push_m_s': 0.}
        case = {'name': item['name'], 'command': list(item['command']),
                'terrain': 'stairs' if item['skill'] == 'rolling_stairs' else 'step_up',
                'task': 'traverse', 'skill': bare, 'terrain_limits': limits,
                'terrain_seed': zlib.crc32(item['name'].encode()), 'episode_seconds': item['episode_seconds'],
                'height_mae_m_max': .02, 'velocity_mae_m_s_max': .1, 'yaw_mae_rad_s_max': .15,
                'success_rate_min': .95, 'survival_rate_min': .95, 'anchor': False,
                'reset_seed_key': item['name'], 'introduced_stage': 'S4', 'untriggered_up_case': True}
        return train, case
    climb = item['skill'] == 'body_climb'
    field = 'body_climb' if climb else 'step_jump'
    params = deepcopy(item[field])
    double = len(params['rises_m']) == 2
    limits = dict(base_limits)
    limits.update({'stair_heights_m': [.20, .45], 'stair_tread_m': [.8, .6]} if double
                  else {'step_up_m': params['rises_m'][0]})
    apex = params['rises_m'][0] + .08
    semantic = {'jump_apex_delta_m': apex, 'release_height_offset_m': .025,
        'jump_com_rise_m': max(.005, apex - .015), 'jump_min_clearance_m': .01,
        'jump_forward_distance_m': .65, 'jump_landing_radius_m': .30000000000000004,
        'jump_landing_speed_max_m_s': .8}
    bare = {'kind': item['skill'], 'command': list(item['command']), 'mode': 0 if climb else 4,
            'sample_amplitude': False, 'terrain_limits': {key: limits[key] for key in
                (('stair_heights_m', 'stair_tread_m') if double else ('step_up_m',))},
            field: params}
    if not climb:
        bare['semantics'] = semantic
    train = {**deepcopy(bare), 'terrain_limits': limits, 'episode_seconds': item['episode_seconds'],
             'sample_yaw_sign': False, 'push_m_s': 0.}
    case = {'name': item['name'], 'command': list(item['command']),
            'terrain': 'stairs' if double else 'step_up', 'task': 'traverse' if climb else 'jump', 'skill': bare,
            'terrain_limits': limits, 'terrain_seed': zlib.crc32(item['name'].encode()),
            'episode_seconds': item['episode_seconds'], 'height_mae_m_max': .02 if climb else .01,
            'velocity_mae_m_s_max': .1, 'yaw_mae_rad_s_max': .15,
            'body_climb_case' if climb else 'step_jump_case': True, 'success_rate_min': .95, 'survival_rate_min': .95,
            'anchor': False, 'reset_seed_key': item['name'],
            'introduced_stage': item['introduced_stage']}
    if not climb:
        case['jump_assistance_enabled'] = False
    return train, case


def rewrite_pools(recipe, source_row, new_specs):
    """Preserve old per-group allocation except the named changed task pools."""
    old = source_row['contract']
    specs, groups, membership = {}, [], {}
    for pool in recipe['sampling_pools']:
        name = pool['name']
        if name not in ('jumps', 'step_jumps', 'double_step_jumps', 'double_body_climb', 'steps'):
            names = [g['name'] for g in old['scene_groups'] if old['behavior_pool_membership'][g['name']] == name]
            ratio = pool['fraction'] / old['behavior_pool_fractions'][name]
            selected = [{**deepcopy(g), 'fraction': g['fraction'] * ratio}
                        for g in old['scene_groups'] if g['name'] in names]
        else:
            names = list(pool['skills'])
            selected = [{'name': skill, 'fraction': pool['fraction'] / len(names),
                         'terrain': [('stairs' if (new_specs[skill]['kind'] == 'rolling_stairs' or
                                      len(new_specs[skill].get('step_jump', new_specs[skill].get('body_climb', {})).get('rises_m', [])) == 2)
                                      else 'step_up') if skill in new_specs else
                                     next(g['terrain'][0] for g in old['scene_groups'] if g['name'] == skill)]}
                        for skill in names]
        groups.extend(selected)
        for skill in names:
            specs[skill] = deepcopy(new_specs[skill] if skill in new_specs else old['skill_specs'][skill])
            membership[skill] = name
    return specs, groups, membership


def build():
    previous = read('contracts/v6_transport_response_training_v1.json')
    design = read('contracts/v6_transport_response_expectations_v1.json')
    plan = deepcopy(previous)
    plan.pop('design_preflight')
    plan['curriculum_revision'] = REVISION
    plan['retired_case_names'] = sorted(RETIRED)
    plan['runtime_base']['response_migration'] = REVISION
    lookup = 'contracts/v6_height_lookup_v1.json'
    plan['runtime_base']['step_jump_task'] = {
        'enabled': True, 'height_lookup_file': lookup,
        'height_lookup_sha256': hashlib.sha256((ROOT / lookup).read_bytes()).hexdigest(),
        'sensor_origin_b_m': [.10, -.20], 'sensor_pitch_rad': .55,
        'ground_echo_margin_m': .015, 'range_max_m': 4.,
        'range_noise_std_m': .003, 'range_dropout_probability': .002, 'range_sample_period_s': .02}
    plan['runtime_base']['actor_reserved_frame30'] = {'index': 30,
        'semantic': 'jump_requested_landing_delta_m_times5', 'zero_without_step_jump': True}
    plan['release_scope']['stair_200_250_definition'] = SCOPE
    old_stages = {item['name']: item for item in previous['stages']}
    new_items = [{**deepcopy(item), 'name': ROLLING_RENAMES[item['name']],
                  'skill': 'rolling_stairs' if item['name'] == 'stairs' else 'rolling_up'}
                 for item in previous['skill_catalog'] if item['name'] in ROLLING_RENAMES]
    new_items += [climb_recipe([height / 100]) for height in (15, 20, 25)] + [climb_recipe([.20, .25])]
    new_items += [jump_recipe([height / 100], trigger) for height in (3, 6, 10, 15, 20, 25)
                 for trigger in ('manual', 'laser')]
    new_items += [jump_recipe([.20, .25], trigger) for trigger in ('manual', 'laser')]
    plan['skill_catalog'] = [item for item in plan['skill_catalog'] if item['name'] not in RETIRED] + new_items
    single = [item['name'] for item in new_items if item['introduced_stage'] == 'S5T']
    double = [item['name'] for item in new_items if item['skill'] == 'double_step_jump']
    plan['stages'] = []
    for sid, ceiling in BUDGETS.items():
        stage = deepcopy(old_stages['S5' if sid in ('S5T', 'S5D') else sid])
        stage.update(name=sid, budget_stage_id=sid, updates=ceiling)
        for pool in stage['sampling_pools']:
            pool['skills'] = [ROLLING_RENAMES.get(name, name) for name in pool['skills']
                              if name not in RETIRED or name in ROLLING_RENAMES]
            if pool['name'] == 'steps':
                pool['skills'] += ['body_climb_15', 'body_climb_20', 'body_climb_25']
        if sid in ('S5T', 'S5D', 'S6'):
            allocation = {'S5T': (.15, .35, 0.), 'S5D': (.05, .15, .15), 'S6': (.10, .06, .02)}[sid]
            next(pool for pool in stage['sampling_pools'] if pool['name'] == 'jumps')['fraction'] = allocation[0]
            stage['sampling_pools'].insert(1, {'name': 'step_jumps', 'fraction': allocation[1], 'skills': single})
            if allocation[2]:
                stage['sampling_pools'].insert(2, {'name': 'double_step_jumps', 'fraction': allocation[2], 'skills': double})
                stage['sampling_pools'].insert(3, {'name': 'double_body_climb', 'fraction': allocation[2],
                                                  'skills': ['body_climb_double_200_250']})
        plan['stages'].append(stage)
    budget = read('contracts/v6_new_asset_budget_v1.json')
    budget.update(budget_id='v6-step-curriculum-budget-v1', status='frozen_executable_research_design',
                  executable_training_contract=True)
    old_budget = {row['id']: row for row in budget['formal']['stages']}
    budget['formal']['stages'] = [{**deepcopy(old_budget.get(sid, old_budget['S5'])),
        'id': sid, 'update_ceiling': cap, 'transition_ceiling': cap * 480000,
        'capability': ({'S5': 'flat_standing_and_running_jumps', 'S5T': 'single_step_jumps',
                       'S5D': 'double_step_body_climb_and_jumps'}[sid] if sid in ('S5', 'S5T', 'S5D')
                       else old_budget[sid]['capability'])}
        for sid, cap in BUDGETS.items()]
    budget['release_requirements'][1] = 'Record the exact ordered S1/S2/S3/S4/S5/S5T/S5D/S6 mapping; no extra updates.'
    plan['budget_spec_file'] = 'contracts/v6_step_curriculum_budget_v1.json'
    plan['budget_spec_sha256'] = write(plan['budget_spec_file'], budget)
    new_specs, additions = {}, []
    for item in new_items:
        new_specs[item['name']], case = explicit_task_spec(item, plan['runtime_base']['terrain_limits'])
        additions.append(case)
    holdout = deepcopy(next(case for case in additions if case['name'] == 'body_climb_20'))
    holdout.update(name='heldout_body_climb_20_payload_grip', introduced_stage='S6',
        reset_seed_key='body_climb_20', heldout_parameter_combination=True,
        dynamics_profile={'base_mass_scale': 1.1}, contact_profile={'friction': .4})
    additions.append(holdout)
    old_design_stages = deepcopy(design['stages'])
    design['stages'] = {}
    for stage in plan['stages']:
        sid = stage['name']
        source_sid = 'S5' if sid in ('S5T', 'S5D') else sid
        source_train = old_design_stages[source_sid]['roles']['train']
        specs, groups, membership = rewrite_pools(stage, source_train, new_specs)
        rows = deepcopy(old_design_stages[source_sid])
        for role, row in rows['roles'].items():
            config = row['contract']
            config.update(curriculum_stage=sid, budget_stage_id=sid, total_updates=BUDGETS[sid],
                budget_spec_file=plan['budget_spec_file'], budget_spec_sha256=plan['budget_spec_sha256'],
                release_scope=deepcopy(plan['release_scope']), response_migration=REVISION,
                step_jump_task=deepcopy(plan['runtime_base']['step_jump_task']),
                actor_reserved_frame30=deepcopy(plan['runtime_base']['actor_reserved_frame30']))
            config['new_asset_curriculum']['stage'] = sid
            config['jump_assist']['enabled'] = role == 'train' and sid == 'S5'
            row['runtime']['modules'].update(step_jump_task=True, step_climb_task=True,
                                              jump_assist=role == 'train' and sid == 'S5')
            config['skill_specs'] = deepcopy(specs)
            config['behavior_pool_membership'] = deepcopy(membership)
            config['behavior_pool_fractions'] = {pool['name']: pool['fraction'] for pool in stage['sampling_pools']}
            config['stages'][0].update(updates=BUDGETS[sid],
                terrain=list(dict.fromkeys(t for group in groups for t in group['terrain'])))
            ev = config['evaluation']
            ev['cases'] = [case for case in ev['cases'] if case['name'] not in RETIRED] + deepcopy(additions)
            for key in ('canonical_case_names', 'quick_case_names', 'retention_case_names', 'promotion_case_names'):
                ev[key] = [name for name in ev[key] if name not in RETIRED]
                ev[key] += [case['name'] for case in additions if key != 'promotion_case_names'
                            or list(BUDGETS).index(case['introduced_stage']) <= list(BUDGETS).index(sid)]
            if role == 'train':
                config['scene_groups'] = deepcopy(groups)
                config['episode_seconds'] = max(spec['episode_seconds'] for spec in specs.values())
            else:
                count = len(ev['cases'])
                row['invocation']['num_envs'] = count * ev['episodes_per_case']
                config['scene_groups'] = [{'name': case['name'], 'fraction': 1. / count,
                                          'terrain': [case['terrain']]} for case in ev['cases']]
        design['stages'][sid] = rows
    design['review_basis'] += (' Step task revision: independently preserve unchanged fields and cases; split the same '
        '8500-update ceiling into eight stages; explicit retired cases carry no credit. S1 pools/rewards/control remain '
        'identical, task module inactive there. New normal climbs and airborne jumps are independently named and graded.')
    design['reviewed_plan_payload_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    path = 'contracts/v6_step_curriculum_expectations_v1.json'
    sha = write(path, design)
    plan['design_preflight'] = {**previous['design_preflight'], 'file': path, 'sha256': sha}
    write('contracts/v6_step_curriculum_training_v1.json', plan)
    print('Frozen eight-stage candidate. Run the independent materialization and fault-injection tests before use.')


if __name__ == '__main__':
    build()
