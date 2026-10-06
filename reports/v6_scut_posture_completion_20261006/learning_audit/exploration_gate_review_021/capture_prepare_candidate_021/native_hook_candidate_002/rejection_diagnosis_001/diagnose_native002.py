from pathlib import Path
from copy import deepcopy
import hashlib,gzip,json,struct,collections
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT.parents[1]/'mechanical_side_speed_candidate_021/native_SIDE_001/native'
RUN=ROOT/'native_CAPTURE_PREPARE_002'
ACT=RUN/'native'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def norm(x):
 if type(x) is float:return ('float64bits',struct.pack('!d',x).hex())
 if isinstance(x,dict):return ('dict',tuple((k,norm(v)) for k,v in sorted(x.items())))
 if isinstance(x,list):return ('list',tuple(norm(v) for v in x))
 return (type(x).__name__,x)
def diffs(x,y,p=''):
 if type(x)!=type(y):yield p,x,y;return
 if isinstance(x,dict):
  if x.keys()!=y.keys():yield p+'/keys',list(x),list(y);return
  for k in x:yield from diffs(x[k],y[k],p+'/'+k)
 elif isinstance(x,list):
  if len(x)!=len(y):yield p+'/length',len(x),len(y);return
  for i,(v,w) in enumerate(zip(x,y)):yield from diffs(v,w,p+'/'+str(i))
 elif norm(x)!=norm(y):yield p,x,y
def comparable(row):
 x=deepcopy(row)
 if x.get('applied_control') is not None:x['applied_control'].pop('side_reference_audit',None)
 if x.get('contacts') is not None:
  for k in ('sum_reported_normal_patch_counts','observed_count_at_capacity'):x['contacts']['capacity_indicator'].pop(k,None)
 return x
def joint_patch_projection(row):
 x=comparable(row)
 if x.get('contacts') is None:return x
 for sensor in x['contacts']['sensor_body_rows']:
  for kind,fields in (
   ('normal_patches',('force_scalar_n','point_physx_m','normal_physx','separation_m','impulse_scalar_ns_derived_force_times_dt')),
   ('friction_patches',('force_physx_n','point_physx_m','impulse_physx_ns_derived_force_times_dt'))):
   for patch in sensor[kind]:
    count=patch['count'];assert all(len(patch[k])==count for k in fields)
    records=[{k:patch[k][i] for k in fields} for i in range(count)]
    # Sort whole records by their exact type/bit representation. Never sort one field independently.
    patch['joint_records_exact_unordered']=sorted(records,key=lambda v:repr(norm(v)))
    for k in fields:del patch[k]
 return x
inputs=[ROOT/'candidate_manifest.json',ROOT/'capture_prepare_hook.py',ROOT/'run_capture_prepare_021.py',ROOT/'test_native_hook_021.py',ROOT/'native_CAPTURE_PREPARE_002.process.json',RUN/'run_plan.json',RUN/'capture_prepare_rejection.json',RUN/'pre_intervention_bitwise_review.json',RUN/'coupled_reference_review.json']
for root in (BASE,ACT):
 inputs.extend(root/name for name in ('identity.json','report.json','mechanical_summary.json','first_birth_rows.jsonl.gz','events.jsonl.gz','mechanical_global_peaks.jsonl.gz','mechanical_5ms.jsonl.gz'))
inputs.extend(ACT/name for name in ('capture_prepare_hook_applied.json','capture_prepare_hook_preclose.json','capture_prepare_enter_events.jsonl.gz'))
before={str(p):sha(p) for p in inputs}
assert sha(ROOT/'candidate_manifest.json')=='ed03fa1d06be18b665e13d667b905c187599acc33a8ef11540b7b87a80b2f1f7'
process=load(ROOT/'native_CAPTURE_PREPARE_002.process.json');assert process['process_exit_code']==0 and process['process_exit_observed'] and process['process_present_after_wait'] is False
hook=load(ACT/'capture_prepare_hook_preclose.json');assert hook['complete'] and hook['hook_callbacks_restored'] and hook['original_update_calls_attempted']==hook['original_update_calls_completed']==2400
enters=rows(ACT/'capture_prepare_enter_events.jsonl.gz');first=next(x for x in enters if x['all4096_blocked_capture_transitions']>0)
assert first['update_call']==hook['first_global_intervention_update_call']==531 and first['native_reference_tick']==530
assert sum(x['all4096_blocked_capture_transitions'] for x in enters)==hook['blocked_lane_transitions']==2562
oldrows=rows(BASE/'mechanical_5ms.jsonl.gz');newrows=rows(ACT/'mechanical_5ms.jsonl.gz')
old={(x['state']['lane'],x['physical_step']):x for x in oldrows};new={(x['state']['lane'],x['physical_step']):x for x in newrows}
end=first['update_call']-1
results=[]
for lane in (1708,2216):
 item=dict(lane=lane,compared_physical_steps=[0,end],compared_rows=end+1,raw_float_bit_difference_count=0,raw_difference_categories={},first_raw_mismatch=None,
  physical_state_and_applied_control_projection_exact=True,joint_contact_records_and_all_nonlayout_fields_exact=True,contact_permutation_only_row_count=0,
  first_joint_record_mismatch=None,first_noncontact_mismatch=None)
 counts=collections.Counter();max_abs=0
 for step in range(end+1):
  assert (lane,step) in old and (lane,step) in new
  a,b=comparable(old[lane,step]),comparable(new[lane,step]);raw=list(diffs(a,b))
  for path,x,y in raw:
   item['raw_float_bit_difference_count']+=1;counts['/'.join(path.split('/')[:3])]+=1
   if isinstance(x,(int,float)) and isinstance(y,(int,float)):max_abs=max(max_abs,abs(x-y))
   if item['first_raw_mismatch'] is None:item['first_raw_mismatch']=dict(physical_step=step,simulated_s=step*.005,path=path,baseline=x,actual=y)
   if not path.startswith('/contacts'):
    item['physical_state_and_applied_control_projection_exact']=False
    if item['first_noncontact_mismatch'] is None:item['first_noncontact_mismatch']=dict(physical_step=step,path=path,baseline=x,actual=y)
  joint=list(diffs(joint_patch_projection(old[lane,step]),joint_patch_projection(new[lane,step])))
  if joint:
   item['joint_contact_records_and_all_nonlayout_fields_exact']=False
   if item['first_joint_record_mismatch'] is None:
    path,x,y=joint[0];item['first_joint_record_mismatch']=dict(physical_step=step,path=path,baseline=x,actual=y)
  elif raw:item['contact_permutation_only_row_count']+=1
 item['raw_difference_categories']=dict(counts);item['raw_max_slot_numeric_difference']=max_abs
 local_first=hook['selected_first_birth_first_blocked_update_call'][str(lane)]
 item['selected_first_actual_block']=next(x for x in enters if any(y['lane']==lane and y['birth_id']==1 and y['blocked'] for y in x['selected']))
 item['first_noncontact_difference_after_global']=None
 for step in range(first['update_call'],min(max(k[1] for k in old if k[0]==lane),max(k[1] for k in new if k[0]==lane))+1):
  raw=list(diffs(comparable(old[lane,step]),comparable(new[lane,step])))
  m=next(((p,x,y) for p,x,y in raw if not p.startswith('/contacts')),None)
  if m:
   p,x,y=m;item['first_noncontact_difference_after_global']=dict(physical_step=step,path=p,baseline=x,actual=y);break
 assert item['first_noncontact_difference_after_global']['physical_step']==local_first
 results.append(item)
old_ev=rows(BASE/'events.jsonl.gz');new_ev=rows(ACT/'events.jsonl.gz')
pre_old=[x for x in old_ev if 4*x['policy_tick']<first['update_call']]
pre_new=[x for x in new_ev if 4*x['policy_tick']<first['update_call']]
old_peaks=rows(BASE/'mechanical_global_peaks.jsonl.gz');new_peaks=rows(ACT/'mechanical_global_peaks.jsonl.gz')
peaks_old=[x for x in old_peaks if x['physical_step']<first['update_call']];peaks_new=[x for x in new_peaks if x['physical_step']<first['update_call']]
oldreport,newreport=load(BASE/'report.json'),load(ACT/'report.json')
identity_keys=('initial_physical_birth_tensor_sha256','initial_physical_birth_sha256','initial_env_rng_sha256','native_group_population','sampling_rng_initial_sha256','sampling_rng_final_sha256','first_native_sampling','final_native_sampling','native_effective_std')
assert all(norm(oldreport[k])==norm(newreport[k]) for k in identity_keys) and norm(load(BASE/'identity.json'))==norm(load(ACT/'identity.json'))
final_old={x['lane']:x for x in rows(BASE/'first_birth_rows.jsonl.gz')};final_new={x['lane']:x for x in rows(ACT/'first_birth_rows.jsonl.gz')}
assert len(final_old)==len(final_new)==4096
fields=('partition','sampled_profile','recovery_prefix_lesson_id','episode_ticks','recovery_phase','recovery_pure_rl','recovery_motion_released','recovery_true_success','recovery_platform_retained','recovery_stable_peak_seconds','first_handover_reference_tick','first_motion_release_reference_tick','commands','height','omega','planar_speed_world')
selected_final=[dict(lane=l,baseline={k:final_old[l][k] for k in fields},actual={k:final_new[l][k] for k in fields}) for l in (1708,2216)]
assert before=={str(p):sha(p) for p in inputs}
report=dict(schema='source021-gyro002-preserved-rejection-independent-diagnosis-v1',
 status='raw_contact_layout_rejection_preserved_with_narrow_measurement_diagnosis',
 original_strict_whole_reproduction=False,original_counterfactual_admission=False,qualification=None,PPO_updates=0,formal_updates_consumed=0,
 root_native_process=dict(pid=process['pid'],exit_code=0,wall_seconds=process['process_wall_seconds']),
 inputs_sha256=before,input_count=len(inputs),all_inputs_before_after_unchanged=True,
 earliest_all4096_intervention=dict(update_call=531,native_reference_tick=530,input_physical_step=530,after_physics_step=531,simulated_after_s=2.655,first_event=first,
  exact_intervened_lane_ID_available=False,explanation='Global count records one blocked transition, but selected array is empty and this observer did not retain its lane ID.'),
 selected_prefix=results,
 broader_evidence=dict(initial4096_birth_identity_private_rng_std_exact=True,recorded_preintervention_event_count=len(pre_old),recorded_preintervention_events_exact=norm(pre_old)==norm(pre_new),
  preintervention_global_peak_count=len(peaks_old),recorded_global_peak_records_exact=norm(peaks_old)==norm(peaks_new),
  full_all4096_dynamic_state_control_history_available=False,
  limitation='Full dynamic state/control/contact arrays across all4096 lanes were not retained; two selected physical trajectories plus existing event/peak records cannot prove every unrecorded lane was bitwise equal.'),
 contact_projection=dict(method='Per-sensor/per-surface/per-kind multiset of WHOLE correlated records with exact type/float64-bit keys. Normal record=(force,point,normal,separation,derived impulse); friction record=(force vector,point,derived impulse). Scalar counts/matrix/net/aggregate and all nonlayout fields are compared exactly. No tolerance and no independent field sorting.',
  raw_ordered_layout_exact=False,whole_contact_record_set_exact_for_recorded_selected_prefix=all(x['joint_contact_records_and_all_nonlayout_fields_exact'] for x in results)),
 clock_diagnosis=dict(clock_or_off_by_one_failure_observed=False,explanation='2400 inherited updates/reference0..2399; first blocked update531 uses state530. Selected first phase/control differences667/710 coincide with their actual blocks. Raw contacts already differ319/352 before any blocked transition, in reordered slots only within measured prefix.'),
 actual_mechanical_outcomes_descriptive_only=selected_final,
 all4096_max_closure_gap_m_baseline=load(BASE/'mechanical_summary.json')['full_window_max_pin_gap_m'],
 all4096_max_closure_gap_m_actual=load(ACT/'mechanical_summary.json')['full_window_max_pin_gap_m'],
 interpretation=['This is a raw contact-list ordering rejection, with no recorded selected state/control difference before earliest global intervention. Exact joint-record permutation is independently diagnosed, not admitted by altering the frozen strict comparator.',
 'After531 any selected evolution may include other-lane intervention/reset/RNG coupling. Actual1708 reaches pureRL by12s, but both selected first births still hold zero motion, lack original task success, and are right-censored.',
 'The unchanged27.620mm all-lane peak remains a real guard excursion; neither complete600/2400 engineering coverage nor these selected trajectories imply whole-domain mechanical repair.',
 'No newnative, CPUtest, SSH, production/contract edit, failed-review overwrite, or causal/success qualification was performed.'])
with (HERE/'diagnosis.json').open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
admission=dict(schema='source021-gyro002-narrow-mechanical-analysis-admission-v1',status='independent_descriptive_analysis_only',diagnosis_sha256=sha(HERE/'diagnosis.json'),
 original_strict_whole_reproduction=False,counterfactual_admission=False,qualification=None,
 selected_physical_control_projection_exact=all(x['physical_state_and_applied_control_projection_exact'] for x in results),
 selected_joint_contact_record_set_exact=all(x['joint_contact_records_and_all_nonlayout_fields_exact'] for x in results),
 raw_contact_layout_bitwise_exact=False,scope='Only recorded selected lanes1708/2216 physical steps0..530; no full4096 dynamic-history assertion.',
 original_rejection_sha256=sha(RUN/'capture_prepare_rejection.json'),native_root_exit_code=0,PPO_updates=0)
with (HERE/'analysis_admission.json').open('x') as f:json.dump(admission,f,indent=2);f.write('\n')
lines=['# gyro002 实际拒绝诊断','', '原严格准入仍拒绝；本诊断不补写通过，qualification=null，0 PPO。', '',f'实际 root 进程 exit0，墙钟 {process["process_wall_seconds"]:.3f}s；2400次原生更新/600策略步完整，新增 hook 已在关闭前恢复。全4096第一次真正拦截发生在 update531/reference530：消费物理state530，在physics531前改变转换mask。该条记录未保留被拦截lane ID。','', '|已记录lane|原首差异物理步|原有序字段差异数|仅完整接触记录重排的帧数|首次phase/control变化|','|---|---:|---:|---:|---:|']
for x in results:lines.append(f'|{x["lane"]}|{x["first_raw_mismatch"]["physical_step"]}|{x["raw_float_bit_difference_count"]}|{x["contact_permutation_only_row_count"]}|{x["first_noncontact_difference_after_global"]["physical_step"]}|')
lines+=['','已记录两lane在物理0..530的state、实际控制、净力／矩阵力／聚合力及所有非layout字段逐位一致。接触记录以同一point关联的force、normal、separation及派生impulse整体逐bit核对，完整记录集合一致，只有有序slot不同。2216首次319将同一摩擦force从slot0换到slot2；1708首次352将同一normal记录从slot2换到slot3。不能把这描述成力差末位舍入，也未单独排序force或point制造等价。','',f'原始前介入事件 {len(pre_old)} 条及全域峰记录 {len(peaks_old)} 条逐位一致；4096初始出生／私有采样／std身份一致。完整4096逐步state/control未保留，因此上述投影证据只能覆盖已记录两lane与已有事件／峰记录，不能声称所有未记录lane全过程逐位一致。','','未观察到时钟错位：2216/1708的第一次非接触字段变化分别是update667/710的phase PREPARE7→CAPTURE6，正好对应各自被新门拦截，输入gyro约7.015/6.967rad/s，超过原.75门。531后可存在其他lane介入／reset／共享RNG影响，本诊断不授单lane反事实因果。','','实际12s两lane均pureRL但仍未释放motion、未成功、平台未保留，均右删失。1708原SIDE1.5参考在8s脚本超时；当前这条轨迹到达pureRL只是一项描述性差异，不是本冻结反事实准入通过，也不能归因成任务收益。全4096最大闭链偏差仍27.620118mm，不能称机械全域修好。','','原review、rejection、candidate及所有原始输入保持不变；新增analysis_admission仅允许这一窄描述性分析，strictwhole=false/counterfactual=false。']
with (HERE/'diagnosis.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'diagnosis':str(HERE/'diagnosis.json'),'sha256':sha(HERE/'diagnosis.json'),'admission_sha256':sha(HERE/'analysis_admission.json'),'md_sha256':sha(HERE/'diagnosis.md'),'prefix':results,'broader':report['broader_evidence']},ensure_ascii=False))
