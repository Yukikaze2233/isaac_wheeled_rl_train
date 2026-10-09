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
report = dict(schema='source031-frozen-effective-goal-coverage-readonly-v1',
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
    small=[]; ordinary_large=[]; active_jumps=[]
    for name in active:
        s=specs[name];kind=s['kind']; limits=s.get('terrain_limits',{})
        if kind=='rolling_stairs':
            small.append(dict(name=name,fraction=active[name],mode=s.get('mode'),limits=limits))
        if kind in ('body_climb','rolling_stairs','step_up','step_down') and (
                max([0]+limits.get('stair_heights_m',[]))>=.2 or limits.get('step_up_m',0)>=.2):
            ordinary_large.append(dict(name=name,kind=kind,fraction=active[name]))
        if kind in ('step_jump','double_step_jump'):
            active_jumps.append(dict(name=name,kind=kind,fraction=active[name],step_jump=s.get('step_jump')))
    report['domains'][domain]=dict(worker_sha256=sha(path),positive_scene_count=len(active),
        effective_kind_fractions=kinds,ordinary_stairs=small,ordinary_large=ordinary_large,
        active_step_jumps=active_jumps,
        double_ordinary_evaluation_present=any(c['name']=='body_climb_double_200_250' for c in worker['evaluation']['cases']),
        moving_jump_groups=[dict(name=n,fraction=active[n],semantics=specs[n].get('semantics'))
            for n in active if specs[n]['kind']=='running_jump'],
        controller_settings=effective['recovery_training'],
        signal_delay=effective.get('signal_delay'),contact_domain=effective.get('contact_domain'))
report['findings']=[
    'Frozen source031 ordinary terrain remains 5/10/20/40 mm stairs; no ordinary large/double climb has positive effective sampling in any of the four next domains.',
    'body_climb_double_200_250 remains a fixed evaluation case; this alone does not train ordinary double climbing.',
    'Active single/double step jumps are explicitly sampled in stepjump and must not be credited as ordinary climbing.',
    'Recovery on existing 200/450 mm platforms is sampled as recovery; it is not an ordinary approach/body-on-platform/leg-retraction traversal task.',
    'All future domain abilities remain unproven; source031 was observed waiting, not adopted, in queue readback004.'
]
report['next_action']='Design and native-check progressive ordinary large/double stair exposure after small stairs, preserving zero STEP/JUMP triggers and existing budget/one actor; freeze a separate reviewed revision before any deployment. No current source029 hot change or duplicate launch.'
out=base/'source031_complete_scope_readonly001.json'
with out.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(source_commit=actual,ordinary_large_counts={k:len(v['ordinary_large']) for k,v in report['domains'].items()},output=str(out)),ensure_ascii=False))
