"""Read frozen source031 effective sampling; no simulator, PPO or remote writes."""
from pathlib import Path
import hashlib, json, math, subprocess

base = Path(__file__).resolve().parent
root = base.parents[2] / '.worktrees/stepjump-landing-goal-20261009'
expected = 'ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a'
actual = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
assert actual == expected
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
recipe_path = root / 'contracts/v6_integrated_runtime_recipe_v27.json'
recipe = json.loads(recipe_path.read_text())
report = dict(schema='source031-frozen-effective-goal-coverage-readonly-v2',
    source_commit=actual, recipe_sha256=sha(recipe_path), domains={},
    qualification=None, PPO_updates=0, native_executions=0, remote_writes=0,
    scope='Positive static effective sampling is evidence of configured exposure, not learned ability or physical reachability.')
for domain, row in recipe['domains'].items():
    path=root / f'contracts/v6_task_campaign_{domain}_worker_v27.json'
    worker=json.loads(path.read_text())
    assert worker['integrated_production']['recipe_sha256'] == report['recipe_sha256']
    effective=dict(worker); effective.update(row['patch'])
    scenes=effective['scene_groups']; specs=effective['skill_specs']
    assert math.isclose(sum(s['fraction'] for s in scenes),1.0,abs_tol=1e-12)
    active={s['name']:s['fraction'] for s in scenes if s['fraction']>0}
    assert set(active)<=specs.keys()
    kinds={}
    for name,fraction in active.items():
        kind=specs[name]['kind'];kinds[kind]=kinds.get(kind,0)+fraction
    small=[]; ordinary_large=[]; ordinary_all=[]; active_jumps=[]
    for name in active:
        s=specs[name];kind=s['kind']; limits=s.get('terrain_limits',{})
        if kind=='body_climb':
            assert s['mode']==0 and 'trigger_source' not in s['body_climb']
            ordinary_all.append(dict(name=name,fraction=active[name],body_climb=s['body_climb'],curriculum=s.get('ordinary_large_stair_curriculum'),command=s['command']))
        if kind=='rolling_stairs':
            small.append(dict(name=name,fraction=active[name],mode=s.get('mode'),limits=limits))
        if kind in ('body_climb','rolling_stairs','step_up','step_down') and (
                max([0]+limits.get('stair_heights_m',[]))>=.2 or limits.get('step_up_m',0)>=.2):
            ordinary_large.append(dict(name=name,kind=kind,fraction=active[name]))
        if kind in ('step_jump','double_step_jump'):
            active_jumps.append(dict(name=name,kind=kind,fraction=active[name],step_jump=s.get('step_jump')))
    report['domains'][domain]=dict(worker_sha256=sha(path),positive_scene_count=len(active),
        effective_kind_fractions=kinds,ordinary_stairs=small,ordinary_large=ordinary_large,ordinary_all=ordinary_all,
        active_step_jumps=active_jumps,
        double_ordinary_evaluation_present=any(c['name']=='body_climb_double_200_250' for c in worker['evaluation']['cases']),
        moving_jump_groups=[dict(name=n,fraction=active[n],semantics=specs[n].get('semantics'))
            for n in active if specs[n]['kind']=='running_jump'],
        controller_settings=effective['recovery_training'],
        signal_delay=effective.get('signal_delay'),contact_domain=effective.get('contact_domain'))
report['findings']=[
    'Reading only the worker JSON misses integrated recipe overrides. The effective terrain/jump/stepjump each include all eight progressively activated ordinary single/double body-climb rows in addition to 5/10/20/40 mm stairs.',
    'ordinary_double_200_250mm has positive configured sampling, mode0, forward .3 m/s, and +.200/+.250 m rises with .800/.600 m treads; fixed evaluation alone would not prove this configured exposure.',
    'Active single/double step jumps are explicitly sampled in stepjump and must not be credited as ordinary climbing.',
    'Recovery on existing 200/450 mm platforms is sampled as recovery; it is not an ordinary approach/body-on-platform/leg-retraction traversal task.',
    'All future domain abilities remain unproven; source031 was observed waiting, not adopted, in queue readback004.'
]
report['next_action']='Maintain actual source031 normal-boundary wait; verify native admission binds the eight row clocks once from the preserved account, then capture actual ordinary geometry/zero-request exposure and end outcomes during terrain. No extra duplicate ordinary course, source029 hot change or duplicate launch.'
report['source_scope_correction']=dict(rejected='source031_complete_scope_readonly001.json', reason='v1 manual findings contradicted its own effective recipe extraction; v2 findings follow integrated overrides. No runtime or trained state changed.', sha256=sha(base/'source031_complete_scope_readonly001.json'))
assert all(len(report['domains'][d]['ordinary_all'])==8 for d in ('terrain','jump','stepjump'))
report['source_module_sha256']={name:sha(root/name) for name in ['src/wheeled_tasks/chassis/ordinary_large_stair.py','src/wheeled_tasks/chassis/integrated_completed_speed_revision.py','src/wheeled_tasks/chassis/integrated_production.py']}
out=base/'source031_complete_scope_readonly002.json'
with out.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(source_commit=actual,ordinary_large_counts={k:len(v['ordinary_large']) for k,v in report['domains'].items()},output=str(out)),ensure_ascii=False))
