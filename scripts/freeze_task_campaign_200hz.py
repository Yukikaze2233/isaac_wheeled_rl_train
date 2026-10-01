#!/usr/bin/env python3
"""Freeze the authorized 200 Hz research approximation without changing learning scope."""
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.chassis.control_contract import COARSE_TIMING, COARSE_SCOPE
from wheeled_tasks.chassis.task_campaign import digest

def read(name):
    return json.loads((ROOT / name).read_text())

def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    return digest(ROOT / name)

def main():
    before = read('contracts/v6_task_campaign_v2.json')
    plan = deepcopy(before)
    control_file = 'contracts/v6_wheel_response_control_200hz_v1.json'
    control = read('contracts/v6_wheel_response_control_v1.json')
    control.update(control_id='v6-wheel-response-control-200hz-v1', timing=COARSE_TIMING,
                   simulation_timing_approximation=COARSE_SCOPE)
    control_sha = write(control_file, control)
    parents = {'A': (plan['source_contract_file'], plan['source_contract_sha256'])}
    pins = []
    for old, row in zip(before['domains'], plan['domains'], strict=True):
        worker = read(old['contract'])
        worker.update(physics_dt=.005, control_math_source=control_file, control_math_sha256=control_sha)
        meta = worker['task_campaign']
        meta['source_contract_file'], meta['source_contract_sha256'] = parents[row['parent_domain']]
        meta['revision_migration'] = dict(schema='task-worker-revision-v3',
            source_contract_file=old['contract'], source_contract_sha256=old['sha256'])
        design = read(worker['design_preflight']['file'])
        design['reviewed_change'] = 'explicit_200hz_simulation_pd_50hz_policy_v3'
        for role in design['stages'][row['name']]['roles'].values():
            for key in ('physics_dt', 'control_math_source', 'control_math_sha256', 'task_campaign'):
                role['contract'][key] = deepcopy(worker[key])
            role['control']['timing'] = deepcopy(COARSE_TIMING)
            role['control']['simulation_timing_approximation'] = deepcopy(COARSE_SCOPE)
            role['runtime'].update(COARSE_TIMING)
        design_file = worker['design_preflight']['file'].replace('_v2.json', '_v3.json')
        worker['design_preflight'].update(file=design_file, sha256=write(design_file, design))
        row['contract'] = old['contract'].replace('_v2.json', '_v3.json')
        row['sha256'] = write(row['contract'], worker)
        parents[row['name']] = (row['contract'], row['sha256'])
        pins.append(dict(source_file=old['contract'], source_sha256=old['sha256'],
                         target_file=row['contract'], target_sha256=row['sha256']))
    plan['contract_id'] = 'v6-task-campaign-v3'
    plan['revision_migration'].update(manifest_file='contracts/v6_task_campaign_migration_v3.json',
        source_plan_sha256=digest(ROOT / 'contracts/v6_task_campaign_v2.json'))
    new_sha = write('contracts/v6_task_campaign_v3.json', plan)
    write('contracts/v6_task_campaign_migration_v3.json', dict(schema='task-plan-revision-v3',
        source_plan_file='contracts/v6_task_campaign_v2.json',
        source_plan_sha256=plan['revision_migration']['source_plan_sha256'],
        target_plan_file='contracts/v6_task_campaign_v3.json', target_plan_sha256=new_sha,
        run_directory=plan['revision_migration']['run_directory'], workers=pins))
    print(new_sha)

if __name__ == '__main__':
    main()
