from pathlib import Path
import copy,datetime,hashlib,importlib.util,json,time
OUT=Path(__file__).resolve().parent
BASE=OUT.parent.parent
HELPER=BASE/'source_repair_023/incremental_reader_audit001/recompute_latest003.py'
PRIOR=OUT.parent/'learning_status002/latest100_actor5001_5100.arithmetic001.json'
RAW=OUT/'latest100_actor5101_5200.raw001.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(HELPER)=='cbc55b924d4bbfb9469fa1d89a47934d93f5df1ed65b9c338b8c7a465684d91a'
t=time.monotonic();before={str(p):sha(p) for p in (HELPER,PRIOR,RAW)}
spec=importlib.util.spec_from_file_location('frozen_latest003_for026_read002',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
old,audit=h.load_arithmetic();new=audit('source023_actor5101_5200',RAW,'100',[11526,11625],[38730,38829]);prior=json.loads(PRIOR.read_text());raw=json.loads(RAW.read_text());comparison=old.paired(new,prior);exposure=h.exposure(raw)
assert new['clock']['baseline_update']==5100 and new['clock']['end_update']==5200
assert prior['clock']['end_update']==5100 and new['clock']['baseline_update']-prior['clock']['end_update']==0
assert new['source_commit']==prior['source_commit']=='6f07ec298e01d2686216787cfba34b394151f1aa'
receipt=json.loads((OUT/'latest100_actor5101_5200.read001.execution.json').read_text())
assert receipt['exit_code']==0 and receipt['stdout_sha256']==sha(OUT/'latest100_actor5101_5200.read001.stdout')
summary=copy.deepcopy(new)
summary.update(schema='source026-REC11526_11625-training-window-status-v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='complete_native_fixed100_training_statistics',latest_observed_progress={k:raw['progress_after'][k] for k in ('updated_at','task_actor_update','successful_updates','actor_updates_in_block')},versus_actor5001_5100=comparison,comparison_scope='Adjacent nonoverlapping100 stochastic training windows; same frozen source and mature course shares, different random rollouts/DR, not a causal or qualification comparison.',recovery_nonzero_command_static_groups_count=exposure['nonzero_command_static_groups_count'],recovery_nonzero_command_static_groups=exposure['nonzero_command_static_groups'],source_and_owner_unchanged={k:raw[k] for k in ('owner_continuity_verified','source_files_unchanged','immutable_metadata_unchanged')},reader_execution=receipt,qualification=None,PPO_updates_by_reader=0,remote_writes=0,posture_scope='Native group maxima are since observer birth; current-window mean/separated pitch-roll angles are unavailable. Reward densities do not supply measured posture angles.',posture_since_birth={g:v['posture'] for g,v in raw['windows']['100']['behavior']['groups'].items()},posture_reward_densities=raw['windows']['100']['optional_posture_reward_density'])
summary['limitations'] += ['No v24 PPO yet; this is unchanged source023 REC.', 'L0/L1/L2 averaged counter deltas are not integer ended episodes; full-chain instantaneous fraction is not completed-chain rate.', 'Training behavior statistics include Gaussian exploration and domain randomization; deterministic deployment quality remains unassessed.']
for name,x in [('latest100_actor5101_5200.arithmetic001.json',new),('latest100_actor5101_5200.summary001.json',summary)]:
 with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert {str(p):sha(p) for p in (HELPER,PRIOR,RAW)}==before
analysis=dict(schema='source026-fixed100-arithmetic-invocation-v1',status='saved_report_arithmetic_consistent',input_SHA=before,audit_reuse='Exact reviewed load_arithmetic/paired/exposure; only explicit window ranges changed.',full100_dense_records=True,group95_pool7_frames9830400=True,window_overlap_actor_updates=0,intervening_actor_updates=0,inputs_unchanged=True,output_SHA={name:sha(OUT/name) for name in ('latest100_actor5101_5200.arithmetic001.json','latest100_actor5101_5200.summary001.json')},wall_seconds=time.monotonic()-t,PPO_updates=0,SSH_calls=0,GPU_actions=0,remote_writes=0,qualification=None)
with (OUT/'latest100_actor5101_5200.analysis001.receipt.json').open('x') as f:json.dump(analysis,f,indent=2);f.write('\n')
print(json.dumps(analysis));print(json.dumps({k:summary[k] for k in ('ordinary_stand','motion','yaw_mae_rad_s','conditional_stop_phases','dynamic_lessons','full_chain_success','policy_mean_std','iteration_timing')},ensure_ascii=False))
