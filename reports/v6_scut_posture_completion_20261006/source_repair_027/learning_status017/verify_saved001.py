"""Independent stdlib verification of saved JSON only; no event/helper execution."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import math

OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent
PRIOR = BASE / 'source_repair_027/learning_status016'
STEM = 'latest100_actor7001_7100'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-10), (a, b)


def payload(path):
    tree = ast.parse(path.read_text())
    return tree, json.loads(tree.body[-1].exc.args[0].args[0].args[0].value)


def sample_summary(groups):
    eligible = [g for g in groups.values() if g['samples'] > 0]
    n = sum(g['samples'] for g in eligible)
    result = dict(samples=n, positive_groups=len(eligible), zero_sample_groups=len(groups)-len(eligible))
    for key in ('mean_speed_m_s', 'above_3cm_s_fraction', 'endpoint_mean_m', 'peak_mean_m'):
        if eligible and all(key in g and g[key] is not None for g in eligible):
            result[key] = sum(g[key]*g['samples'] for g in eligible)/n
    for key in ('peak_speed_m_s', 'peak_max_m'):
        if eligible and all(key in g for g in eligible):
            result[key] = max(g[key] for g in eligible)
    return result


inputs = [OUT/(STEM+suffix) for suffix in ('.raw001.json', '.arithmetic001.json', '.summary001.json', '.analysis001.receipt.json', '.read001.execution.json', '.read001.stdout', '.read001.stderr', '.reader001.remote.py')]
inputs += [OUT/'readiness001.json', Path(read(OUT/'readiness001.json')['live_file']), OUT/'reader_preparation001.json', OUT/'summarize_latest100.py', PRIOR/'latest100_actor6901_7000.arithmetic001.json', PRIOR/'latest100_actor6901_7000.reader001.remote.py']
inputs += [OUT/'reward_tag_manifest001.json', BASE/'source_repair_027/velocity_payment_tag_spec001.json']
before = {str(p): sha(p) for p in inputs}
raw, arithmetic, summary, analysis, execution = [read(OUT/(STEM+suffix)) for suffix in ('.raw001.json', '.arithmetic001.json', '.summary001.json', '.analysis001.receipt.json', '.read001.execution.json')]
prior = read(PRIOR/'latest100_actor6901_7000.arithmetic001.json')
ready, prep = [read(OUT/name) for name in ('readiness001.json','reader_preparation001.json')]
live = read(Path(read(OUT/'readiness001.json')['live_file']))
checks = {}
oldtree, oldpayload = payload(PRIOR/'latest100_actor6901_7000.reader001.remote.py')
newtree, newpayload = payload(OUT/(STEM+'.reader001.remote.py'))
assert ast.dump(oldtree.body[1],include_attributes=False)==ast.dump(newtree.body[1],include_attributes=False)
oldhelper=ast.get_source_segment((PRIOR/'latest100_actor6901_7000.reader001.remote.py').read_text(),oldtree.body[2])
newhelper=ast.get_source_segment((OUT/(STEM+'.reader001.remote.py')).read_text(),newtree.body[2])
assert newhelper.replace(prep['scope_literal_current'],prep['scope_literal_previous'])==oldhelper
oldpayload.pop('window');newpayload.pop('window');assert oldpayload==newpayload
specs=newpayload['reward_tag_specs'];manifest=read(OUT/'reward_tag_manifest001.json');assert specs==manifest['specs'] and len(specs)==16
assert prep['remote_main_AST_identical'] and prep['reward_helper_identical_except_prior_availability_scope_literal']
assert prep['reward_manifest_sha256']==sha(OUT/'reward_tag_manifest001.json')
checks['reader_window_and_prior_scope_literal_only_native_parser_unchanged']=True
assert execution['exit_code']==0 and execution['stdout_sha256']==sha(OUT/(STEM+'.read001.stdout')) and execution['stderr_sha256']==sha(OUT/(STEM+'.read001.stderr'))
assert execution['script_sha256']==sha(OUT/(STEM+'.reader001.remote.py'))
assert read(OUT/(STEM+'.read001.stdout'))==raw
assert ready['ready'] and ready['end7100_actual'] and live['progress']['actor_updates_in_block']>=7100
assert ready['live_sha256']==sha(Path(read(OUT/'readiness001.json')['live_file']))
assert {r['kind']:(r['pid'],r['starttime_ticks']) for r in live['owners']} == {'wrapper':(3662137,96629148),'supervisor':(3662142,96629351),'worker':(3662149,96629601)}
checks['actual_exit_stdout_and_ready_same_three_owners'] = True
assert raw['complete'] and raw['owner_continuity_verified'] and raw['source_files_unchanged'] and raw['immutable_metadata_unchanged']
assert raw['source_commit']==prior['source_commit']=='6f07ec298e01d2686216787cfba34b394151f1aa'
w=raw['windows']['100']; c=w['window']
assert w['complete'] and w['missing_required_tag_count']==0
assert c['actor_updates']==100 and c['baseline_update']==7000 and c['end_update']==7100
assert c['task_actor_updates_inclusive']==[13426,13525] and c['event_steps_inclusive']==[40630,40729]
assert c['dense_steps']==list(range(40630,40730)) and c['behavior_steps']==list(range(40629,40730,10))
assert prior['clock']['end_update']==7000
checks['source_and_nonoverlapping100_actor_REC_TB'] = True
identity=raw['identity']; groups=w['behavior']['groups']; pools=w['behavior']['pools']; mapping=identity['group_pools']; population=identity['population']
assert len(groups)==95 and len(pools)==7 and len(identity['base_groups'])==41 and sum(population.values())==4096
assert set(groups)==set(population)==set(mapping) and sum(mapping[g]=='scripted_recovery' for g in groups)==54
counts=('frames','observed_frames','episodes','successes','failures','timeouts','boundary_truncations','blocked_truncations')
means=('vx_mae_m_s','yaw_mae_rad_s','height_mae_m','planar_speed_mean_m_s','vx_actual_mean_m_s','vx_command_mean_m_s','vx_bias_m_s','height_actual_mean_m','height_command_mean_m')
for g,v in groups.items():
    assert v['complete'] and v['frames']==v['observed_frames']==population[g]*2400
    assert all(isinstance(v[k],int) and v[k]>=0 for k in counts)
    assert v['episodes']==v['successes']+v['failures']+v['timeouts']
for p,v in pools.items():
    members=[groups[g] for g in groups if mapping[g]==p]
    for k in counts: assert v[k]==sum(m[k] for m in members)
    for k in means: close(v[k],sum(m[k]*m['frames'] for m in members)/v['frames'])
totals={k:sum(v[k] for v in groups.values()) for k in counts}
assert totals['frames']==totals['observed_frames']==9830400 and totals==arithmetic['arithmetic']['totals']
checks['95groups7pools4096_frame_weighted_counts_and_means'] = True
ownership=w['learning_ownership']
assert ownership['complete'] and ownership['observed_transitions_total']==9830400
assert ownership['valid_transitions_total']+ownership['excluded_transitions_total']==9830400
assert math.isclose(ownership['valid_transitions_total']/9830400,ownership['valid_fraction']['mean'],rel_tol=0,abs_tol=1e-8)
for phase in ('braking','settled_stop'):
    expected=sample_summary({g:v[phase] for g,v in w['optional_conditional_stop_phases']['groups'].items()})
    for k,v in expected.items(): close(v,arithmetic['conditional_stop_phases'][phase][k])
stationary=w['stationary_10s']['groups']; base=set(identity['base_groups'])
for label,selected in (('all',stationary),('base',{g:v for g,v in stationary.items() if g in base}),('recovery',{g:v for g,v in stationary.items() if g not in base})):
    expected=sample_summary(selected)
    for k,v in expected.items(): close(v,arithmetic['stationary_10s'][label][k])
checks['learning_mask_and_sample_weighted_stops_stationary10s'] = True
for key,value in arithmetic.items():
    if key=='limitations':
        assert summary[key][:len(value)]==value and summary[key][len(value):]==['No v25 PPO yet; this is unchanged source023 REC.','L0/L1/L2 averaged counter deltas are not integer ended episodes; full-chain instantaneous fraction is not completed-chain rate.','Training behavior statistics include Gaussian exploration and domain randomization; deterministic deployment quality remains unassessed.']
    else: assert summary[key]==value
pair_count=0
def check_pairs(node):
    global pair_count
    if isinstance(node,dict):
        if set(node)=={'old','new','new_minus_old'}:
            close(node['new']-node['old'],node['new_minus_old']);pair_count+=1
        else:
            for v in node.values():check_pairs(v)
check_pairs(summary['versus_actor6901_7000'])
assert pair_count==77
for g,fields in arithmetic['ordinary_stand'].items():
    for k,v in fields.items(): close(summary['versus_actor6901_7000']['ordinary_stand'][g][k]['old'],prior['ordinary_stand'][g][k]);close(summary['versus_actor6901_7000']['ordinary_stand'][g][k]['new'],v)
assert analysis['input_SHA'][str(OUT/(STEM+'.raw001.json'))]==sha(OUT/(STEM+'.raw001.json'))
for name,value in analysis['output_SHA'].items(): assert sha(OUT/name)==value
checks['summary_raw_projection_and_77_comparison_deltas'] = True
for level,values in w['dynamic_lessons']['levels'].items():
    for k,d in values['endpoint_rollout_average_deltas'].items():close(d['after']-d['before'],d['delta'])
commands=w['teaching']['static_group_command_means'];assert len(commands)==54
nonzero=[g for g,fields in commands.items() if any(fields[k][f]!=0 for k in fields for f in ('mean','min','max'))]
assert len(nonzero)==summary['recovery_nonzero_command_static_groups_count'] and set(nonzero)==set(summary['recovery_nonzero_command_static_groups'])
assert summary['qualification'] is None and summary['PPO_updates_by_reader']==0 and summary['remote_writes']==0
checks['mixed54_deltas_and_exposure_no_pureL2_qualification'] = True
value=raw['reward_value_audit'];assert value==summary['reward_value_audit']
assert value['actor_updates_inclusive']==[7001,7100] and value['event_steps_inclusive']==[40630,40729]
assert value['first50_event_steps']==[40630,40679] and value['second50_event_steps']==[40680,40729]
assert set(value['rows'])=={spec['tag'] for spec in specs}
static_spec=read(BASE/'source_repair_027/velocity_payment_tag_spec001.json')
assert sha(BASE/'source_repair_027/velocity_payment_tag_spec001.json')=='1895a0833affa5ed9d6b748608ed8ae298e995675fd7d6541c8b9eecd9063b42'
assert set(value['rows'])==set(static_spec['exact_tags'])
for tag,row in value['rows'].items():
    observed={item['TB_step']:item['value'] for item in row['values']}
    assert len(observed)==len(row['values'])==row['records'] and row['records']<=100
    assert row['expected_records']==100 and row['expected_group_frames_per_record']==24*population[row['group']]==552
    assert row['missing_steps']==[step for step in range(40630,40730) if step not in observed]
    assert row['complete']==(row['records']==100)
    for key,steps in [('full100',range(40630,40730)),('first50',range(40630,40680)),('second50',range(40680,40730))]:
        vals=[observed[step] for step in steps if step in observed]; stats=row[key]
        assert stats['records']==len(vals) and stats['expected_records']==len(steps) and stats['complete']==(len(vals)==len(steps))
        if stats['complete']:
            close(stats['mean'],sum(vals)/len(vals));close(stats['min'],min(vals));close(stats['max'],max(vals))
        else:assert stats['mean'] is None and stats['median'] is None and stats['min'] is None and stats['max'] is None
    if row['field']=='rollout_samples':assert all(item['value']==552 for item in row['values'])
assert value['all_tags_complete']==all(row['complete'] for row in value['rows'].values())
assert value['missing_tags']==[tag for tag,row in value['rows'].items() if not row['complete']]
checks['additional16_actual_values_complete_units_denominators_halves_or_missing'] = True
assert {str(p):sha(p) for p in inputs}==before
checks['all_inputs_unchanged'] = True
review=dict(schema='source027-fixed100-saved-evidence-review-v1',reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed_saved_evidence_arithmetic_and_scope_review',review_scope='Independent stdlib JSON/AST/hash and arithmetic checks only; no helper execution, events, remote, torch, model, simulator or GPU.',checks=checks,check_count=len(checks),numeric_comparison_pairs=pair_count,input_SHA=before,inputs_unchanged=True,reader_wall_seconds=execution['wall_seconds'],PPO_iteration_wall_seconds=w['timing']['iteration_wall_seconds']['mean'],scope_notes=['Reader wall is separate from native PPO iteration wall.','95 groups / 7 pools / 4096 environments / 9,830,400 observer frames.','54 recovery groups mix L0/L1/L2; instantaneous L2 fraction and averaged counter deltas do not give ended pure-L2 success rate.','Training Gaussian exploration and domain randomization are not deterministic deployment qualification.','Posture group maxima are since observer birth, not current mean pitch/roll.'],PPO_updates=0,SSH_calls=0,remote_writes=0,GPU_actions=0,qualification=None)
with (OUT/'saved_evidence_review001.json').open('x') as f:json.dump(review,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:review[k] for k in ('status','check_count','numeric_comparison_pairs','reader_wall_seconds','PPO_iteration_wall_seconds')},ensure_ascii=False))
