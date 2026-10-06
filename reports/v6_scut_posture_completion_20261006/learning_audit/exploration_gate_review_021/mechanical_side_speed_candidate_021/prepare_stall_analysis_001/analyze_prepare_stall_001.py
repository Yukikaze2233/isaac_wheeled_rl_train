"""CPU-only readback of sealed native SIDE001 lane1708. No simulator or PPO."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,math

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 import torch
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 frozen=Path('/tmp/scut-recovery-frozen-021-qv1ukmbr');base=a.root/'reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/mechanical_side_speed_candidate_021';native=base/'native_SIDE_001/native'
 comparison=base/'analysis_candidate_001/actual_comparison_001/comparison.json';cp=json.loads(comparison.read_text());inputs={Path(k):v for k,v in cp['actual_inputs_sha256'].items()};inputs[comparison]=sha(comparison)
 files=['src/wheeled_tasks/chassis/recovery.py','src/wheeled_tasks/chassis/recovery_training.py','src/wheeled_tasks/chassis/continuous_encoders.py','src/wheeled_tasks/chassis/mechanism_observer.py','src/wheeled_tasks/chassis/step_trigger.py','contracts/v6_task_campaign_recovery_worker_v19.json','contracts/v6_recovery_profiles_v2.json','contracts/v6_height_lookup_v1.json']
 for name in files:inputs[frozen/name]=sha(frozen/name)
 inputs[Path(__file__).resolve()]=sha(Path(__file__).resolve())
 for p,h in inputs.items():assert sha(p)==h,(p,'changed input')
 report=json.loads((native/'report.json').read_text());assert report['native_startup']['runtime_preflight']['observed']['modules']['usb_transport'] is False
 assert report['PPO_updates']==report['formal_updates_consumed']==0 and report['qualification'] is None
 layout=json.loads((native/'mechanical_layout.json').read_text());rows=[]
 with gzip.open(native/'mechanical_5ms.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line)
   if r['state']['lane']==1708:rows.append(r)
 assert [r['physical_step'] for r in rows]==list(range(1601))
 ids=[layout['full_joint_names'].index(n) for n in layout['control_joint_names']];bodyidx={n:layout['body_names'].index(n) for n in ('base_link','L_link3','R_link3')}
 table=json.loads((frozen/'contracts/v6_height_lookup_v1.json').read_text());limits=[[table['sides'][s]['delta_rad'][0],table['sides'][s]['delta_rad'][-1]] for s in ('left','right')]
 enc=load_module('_read_enc',frozen/files[2]).ContinuousEncoderAngles(1,'cpu',limits);geo=load_module('_read_geo',frozen/files[3]).ClosedChainHeightModel(table,'cpu');height_fn=load_module('_read_height',frozen/files[4]).estimate_height_if_grounded
 enc.reset([0],torch.tensor([[rows[0]['state']['full_q'][i] for i in ids]],dtype=torch.float32));legs=[0,1,3,4];released=False;frames=[];reqdiff=[];hdiff=[]
 def norm(v):return math.sqrt(sum(x*x for x in v))
 for previous,r in zip(rows,rows[1:]):
  state=r['state'];c=r['applied_control'];assert c['feedback_full_q_prephysics']==previous['state']['full_q'];assert c['feedback_full_dq_prephysics']==previous['state']['full_dq']
  q=torch.tensor([[c['feedback_full_q_prephysics'][i] for i in ids]],dtype=torch.float32);dq=torch.tensor([[c['feedback_full_dq_prephysics'][i] for i in ids]],dtype=torch.float32);cont=enc.update(q)
  if state['release_finished'] and not released:enc.realign([0]);cont=enc.project(q)
  released=state['release_finished'];target=torch.tensor([c['script_targets_control6']],dtype=torch.float32)
  gravity=torch.tensor([previous['state']['gravity_body']],dtype=torch.float32);gyro=previous['state']['gyro_body'];unit=gravity/gravity.norm(dim=-1,keepdim=True);tilt=math.degrees(float(torch.acos((-unit[:,2]).clamp(-1,1))))
  points,valid=geo.wheel_positions(cont);heights,hvalid=height_fn(points,gravity,geo.wheel_radius_m,grounded=valid,sensor_valid=valid,wheel_axle_b=geo.wheel_axles_b)
  if released:
   requested=torch.tensor([c['native_requested_motor_effort_nm_at_setter']],dtype=torch.float32);reqdiff.append(float((160*(target[:,legs]-cont[:,legs])-2.5*dq[:,legs]-requested[:,legs]).abs().max()));hdiff.append(abs(float(heights.amin(-1))-state['observed_height_sensor_m']))
  forces={n:[sum(v[k] for v in r['contacts']['sensor_body_rows'][i]['matrix_force_n']) for k in range(3)] for n,i in bodyidx.items()}
  err=(target-cont)[0,legs].tolist();limit=85. if state['phase']==6 else 55.
  frames.append({'physical_step':r['physical_step'],'s':r['simulated_s'],'phase':state['phase'],'route':state['route'],'reference_input_tilt_deg':tilt,'reference_input_gyro_norm_rad_s':norm(gyro),'postphysics_tilt_deg':math.degrees(math.acos(max(-1,min(1,-state['gravity_body'][2])))),'cached_reference_support_proxy':state['observed_support_sensor'],'cached_reference_height_m':state['observed_height_sensor_m'],'CPU_replayed_encoder_control6_rad':cont[0].tolist(),'observed_target_control6':c['script_targets_control6'],'target_minus_CPU_replayed_leg_q_rad':err,'max_abs_leg_target_error_rad':max(map(abs,err)),'actual_terrain_contact_force_norm_n':{n:norm(v) for n,v in forces.items()},'both_actual_wheels_over_native2N_contact_threshold':norm(forces['L_link3'])>2 and norm(forces['R_link3'])>2,'source_wheel_balance_eligible_derived':state['phase'] in (6,7) and state['observed_support_sensor'] and tilt<limit,'source_prepare_partial_gate_blockers_derived':[k for k,ok in {'support':state['observed_support_sensor'],'tilt':tilt<8.,'gyro':norm(gyro)<.75,'height':.27<state['observed_height_sensor_m']<.36,'leg_target_error':max(map(abs,err))<.12}.items() if not ok],'full_ready_predicate_available':False})
 assert max(reqdiff)==0.,'CPU encoder replay differs from actual preclip PD request'
 assert max(hdiff)<2e-6,'CPU FK height replay differs from cached actual observer'
 selected_steps=[8,117,118,683,709,710,715,720,725,730,740,760,780,800,820,840,880,900,1000,1200,1600]
 late=[f for f in frames if f['physical_step']>=710];postsettle=[f for f in frames if f['physical_step']>=1000]
 phase_segments=[]
 for f in frames:
  if not phase_segments or (phase_segments[-1]['phase'],phase_segments[-1]['route'])!=(f['phase'],f['route']):phase_segments.append({'phase':f['phase'],'route':f['route'],'first_step':f['physical_step'],'last_step':f['physical_step']})
  else:phase_segments[-1]['last_step']=f['physical_step']
 final_t=frames[-1]['observed_target_control6'];lastchange=max(f['physical_step'] for f in frames if f['observed_target_control6']!=final_t)
 result={'schema':'source021-SIDE001-lane1708-prepare-stall-readback-v1','inputs_sha256':{str(p):h for p,h in inputs.items()},'clock_and_sensor_scope':{'native_physics_poststate':'same-step q/dq/pins/contact after PhysX','reference_sensor_input':'previous postphysics gyro/gravity and setter prephysics q/dq; native USB disabled and sequential q/dq equality verified','cached_support_and_height':'native RecoveryTraining.advance_reference values used before this same physical step','derived_FK':'frozen native encoder/geometry/height helpers replayed on CPU, not new physical observations','native_missing':['body_clear/probe confirmed','FSM ready_ticks/reroute_count/remaining serialized per5ms','actual randomized mass/COM/friction draw','selfcontact pair identity']},'CPU_readback_reconciliation':{'actual_preclip_leg_PD_request_max_diff_Nm':max(reqdiff),'request_samples':len(reqdiff),'cached_height_max_abs_diff_m':max(hdiff),'no_isaac_or_PPO':True},'phase_segments':phase_segments,'selected_native_frames':[f for f in frames if f['physical_step'] in selected_steps],'late_PREPARE':{'steps':len(late),'duration_s':len(late)*.005,'proxy_support_true_steps':sum(f['cached_reference_support_proxy'] for f in late),'proxy_true_but_not_both_actual_wheels_gt2N_steps':sum(f['cached_reference_support_proxy'] and not f['both_actual_wheels_over_native2N_contact_threshold'] for f in late),'proxy_false_but_both_actual_wheels_gt2N_steps':sum(not f['cached_reference_support_proxy'] and f['both_actual_wheels_over_native2N_contact_threshold'] for f in late),'last_target_change_step':lastchange,'exact_final_target_constant_from_step':lastchange+1,'settled1000_to1600_max_leg_target_error_rad':max(f['max_abs_leg_target_error_rad'] for f in postsettle),'settled1000_to1600_proxy_support_true_steps':sum(f['cached_reference_support_proxy'] for f in postsettle)},'source_inferences':['PREPARE8→FOLD118 with SIDE route4, no reset after first release, is the native unsupported-pose reroute path. Contract max_reroutes1 means the only reroute is spent before SIDE; actual counter itself was not serialized.','CAPTURE→PREPARE710 source gate checks tilt<20deg, cached support and cached height>.22m, not angular speed. Actual incoming gyro≈6.97rad/s exceeds the unchanged .75 handover gyro gate. This is a design weakness candidate, not a proved unique cause.','Late PREPARE reaches nominal-equivalent motor targets while the body is inverted. Small leg tracking residual does not imply uprightness or real two-wheel support.','Proxy support is conditional FK/acceleration/gyro evidence, not native contact. Opposite proxy/contact cases in this trace do not by themselves prove an observer bug or illegal contact.'],'next_counterfactual_proposed_only':{'change':'CAPTURE→PREPARE additionally requires gyro.norm < existing config.handover_gyro_max (.75rad/s)','held_fixed':['SIDE speed1.5','sameREC3076 mean/private sampling and firstbirth RNG','ready8 predicates, .1s prepare, .2s blend,1s RL hold and original followup','8s global script budget','max_reroutes1','solver64/32,3mm','all other transitions/PD/actuator/terrain/DR'],'validation':'No guarantee of recovery: existing plausible gate still cuts support at gyro8rad/s, and contact/geometry can remain inadequate. Must run isolated0PPO paired native counterfactual before any deployment.'},'PPO_updates':0,'formal_updates_consumed':0,'qualification':None}
 for p,h in inputs.items():assert sha(p)==h,(p,'changed during analysis')
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(sha(a.output/'report.json'))
if __name__=='__main__':main()
