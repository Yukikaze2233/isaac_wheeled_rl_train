#!/usr/bin/env python3
"""Local0PPO REC3076 mean600; explicit stricter CAPTURE→PREPARE runtime hook."""
from __future__ import annotations
import argparse,inspect,json
from pathlib import Path
import sys,traceback
HERE=Path(__file__).resolve().parent
SIDE=HERE.parent.parent/'mechanical_side_speed_candidate_021'
SIDE_SHA='2d04ffc45d6d12e7f3bac610f24f7c3f88e2af2493af24ee4f8cefc611b487c1'
BASELINE=SIDE/'native_SIDE_001/native'
PYTHON='/home/yukikaze/isaacsim60-venv/bin/python'

def require(v,message):
 if not bool(v):raise ValueError(message)

def load(path,name):
 import importlib.util
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def frozen_side():
 side=load(SIDE/'run_side_speed_021.py','prepare_frozen_SIDE');capture,_=side.verify_candidate(SIDE_SHA);return side,capture

def verify_candidate(expected_sha):
 side,capture=frozen_side();mp=HERE/'candidate_manifest.json';require(capture.sha(mp)==expected_sha,'CapturePrepare manifest differs');manifest=capture.read(mp)
 require(manifest['schema']=='source021-CAPTURE-PREPARE-runtime-hook-candidate-v1','Hook schema differs')
 for relative,digest in manifest['files_sha256'].items():
  p=Path(relative);require(not p.is_absolute() and '..' not in p.parts,'Candidate path escape');require(capture.sha(HERE/p)==digest,'Changed candidate: '+relative)
 for path,digest in manifest['external_sha256'].items():require(Path(path).is_absolute() and capture.sha(path)==digest,'Changed source/evidence: '+path)
 return side,capture,manifest

def authenticate_hook_fsm(fsm,env,source_root,source_sha,sha):
 name='wheeled_tasks.chassis.recovery';m=sys.modules.get(name)
 require(type(fsm).__module__==name and m is not None and getattr(m,'RecoveryFSM',None) is type(fsm),'Native FSM not constructed/registered')
 p=Path(inspect.getsourcefile(type(fsm))).resolve();require(p==Path(source_root)/'src/wheeled_tasks/chassis/recovery.py' and sha(p)==source_sha,'Hook native source differs')
 for name in ('update','_enter'):
  cb=getattr(fsm,name);require(cb.__self__ is fsm and cb.__func__ is getattr(type(fsm),name) and cb.__func__.__module__==type(fsm).__module__
    and Path(inspect.getsourcefile(cb.__func__)).resolve()==p,'Native callback implementation differs: '+name)
 require(fsm.count==4096 and env.recovery.reference_dt==.005 and env.recovery.reference_stride==1,'Nativepopulation/reference clock differs')
 expected=fsm.DEFAULTS|{k:v for k,v in env.recovery.settings.items() if k in fsm.DEFAULTS};expected['dt']=.005;expected['side_speed_rad_s']=1.5
 require(fsm.config==expected,'Hook config differs from admitted SIDE1.5 baseline')

def make_observer(parent,side,capture,mechanical,side_helper,hook_helper,writer,source_root,source_sha,authenticate=None):
 base=side.make_side_observer(parent,capture,mechanical,side_helper,writer,source_root,source_sha)
 class WithCapturePrepare(base):
  def __init__(self,env,output):
   self.capture_prepare=None;super().__init__(env,output)
   try:
    auth=authenticate or (lambda fsm:authenticate_hook_fsm(fsm,env,source_root,source_sha,capture.sha))
    self.capture_prepare=hook_helper.CapturePrepareHook(env,mechanical.HookSet(),writer,output,mechanical.plain,auth,
      lambda:[(lane,int(self.mechanical.birth_ids[lane])) for lane in side_helper.LANES])
    self.capture_prepare.apply()
   except BaseException:self._before_close();raise
  def _before_close(self):
   try:
    if self.capture_prepare is not None:require(not self.capture_prepare.cleanup(),'Hook preclose restore rejected')
   finally:super()._before_close()
  def finish(self,output):
   try:
    require(not self.capture_prepare.finish(),'Hook complete restore rejected')
    result=super().finish(output)
   except BaseException:
    writer(output/'capture_prepare_observer_failure.json',dict(status='capture_prepare_finish_rejected_preclose',error=traceback.format_exc(),PPO_updates=0,qualification=None));self._before_close();raise
   result.update(isolated_capture_prepare_hook_complete=True,
     isolated_capture_prepare_hook_summary_sha256=capture.sha(output/'capture_prepare_hook_preclose.json'),
     isolated_capture_prepare_runtime_control_override=True,original_recovery_source_modified=False)
   return result
 return WithCapturePrepare

def review(output,candidate_sha,process_receipt):
 side,capture,manifest=verify_candidate(candidate_sha);hook_helper=load(HERE/'capture_prepare_hook.py','prepare_readback_hook');side_helper=load(SIDE/'side_speed_capture.py','prepare_readback_side')
 pair=capture.load(capture.ORCHESTRATOR,'prepare_original_pair');process=capture.read(process_receipt);plan=capture.read(output/'run_plan.json')
 require(process['process_exit_observed'] is True and process['process_exit_code']==0 and process.get('process_present_after_wait') is False,'Actualexclusive process exit absent')
 require(process['candidate_manifest_sha256']==candidate_sha and process['command']==plan['native_command']
  and process['PPO_updates']==process['formal_updates_consumed']==0,'Actual process identity/scope differs')
 native=output/'native';integrity=pair.diagnostic_review(pair.bound_driver(),native);report=capture.read(native/'report.json');old=capture.read(BASELINE/'report.json')
 require(capture.read(native/'identity.json')==capture.read(BASELINE/'identity.json'),'Same actualREC3076 source/seal identity differs')
 for key in ('initial_physical_birth_tensor_sha256','initial_physical_birth_sha256','initial_env_rng_sha256','native_group_population',
  'sampling_rng_initial_sha256','sampling_rng_final_sha256','first_native_sampling','final_native_sampling','native_effective_std'):
  require(report[key]==old[key],'Same sealedbirth/private sampling differs: '+key)
 summary=capture.read(native/'mechanical_summary.json');hook=capture.read(native/'capture_prepare_hook_preclose.json');rate=capture.read(native/'side_override_preclose.json')
 require(summary['complete'] and summary['policy_steps']==600 and summary['physics_steps']==summary['trace_cutoff_physics_steps']==2400
  and summary['hooks_restored'] and not summary['contact_count_at_capacity_indicators'],'Full600/2400 mechanical observation incomplete')
 require(hook['complete'] and hook['config_unchanged'] and hook['hook_callbacks_restored'] and hook['no_pending_update']
  and hook['original_update_calls_attempted']==hook['original_update_calls_completed']==2400
  and hook['native_reference_tick_last']-hook['native_reference_tick_first']==2399 and not hook['errors']
  and hook['explicit_runtime_control_override'] and not hook['original_source_code_modified']
  and report['isolated_capture_prepare_hook_complete'],'Explicit native mask override/cleanup incomplete')
 require(rate['complete'] and rate['config_restored'] and rate['override_hooks_restored'] and not rate['errors']
  and rate['sole_changed_keys']==['side_speed_rad_s'] and rate['before']|{'side_speed_rad_s':1.5}==rate['effective'],'SIDE1.5 baseline/restoration changed')
 require(not any((native/name).exists() for name in ('capture_prepare_observer_failure.json','side_observer_failure.json','mechanical_capture_failure.json')),'Native observer failure sidecar')
 pre=hook_helper.pre_intervention_review(side_helper,BASELINE/'mechanical_5ms.jsonl.gz',native/'mechanical_5ms.jsonl.gz',hook)
 coupled=side_helper.coupled_reference_review(native/'mechanical_5ms.jsonl.gz');capture.write(output/'pre_intervention_bitwise_review.json',pre);capture.write(output/'coupled_reference_review.json',coupled)
 require(pre['status']=='pre_intervention_equal','Actual pre-intervention mismatch: preserve result, reject single-variable admission')
 require(coupled['status']=='paired_reference_preserved','SIDE reference changed')
 inputs=list(BASELINE.glob('mechanical*'))+list(native.glob('*'))+[Path(process_receipt),output/'run_plan.json',output/'pre_intervention_bitwise_review.json',output/'coupled_reference_review.json',HERE/'candidate_manifest.json']
 return dict(schema='source021-CAPTURE-PREPARE-hook-native-readback-v1',status='native_counterfactual_integrity_checked_not_functionally_qualified',
  native_integrity=integrity,initial_birth_private_sampling_std_exact=True,pre_intervention=pre,coupled_reference=coupled,hook=hook,
  inherited_SIDE1p5=rate,mechanical_capture=summary,inputs_sha256={str(p):capture.sha(p) for p in inputs if p.is_file()},
  original_source_code_modified=False,explicit_runtime_control_override=True,PPO_updates=0,formal_updates_consumed=0,qualification=None,
  limitations=['12s incomplete firstbirths are rightcensored','Allbirth guards differ from fixed4096firstbirth outcomes',
   'Original frozen source and its old runtimepreflight remain unchanged; additional diagnostic override is separately authenticated',
   'Closure3mm/original1s/8s/reroute unchanged; this review confers no mechanical/function/PPO/deployment qualification',
   'No actual selfcontact pair IDs/native overflow/randomized massCOMfriction draw','Kit.close may exit Python; independent actual exit receipt required'])

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('cpu','native','review'),default='cpu');p.add_argument('--candidate-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--process-receipt',type=Path);args=p.parse_args(argv)
 output=args.output.resolve();require(HERE in output.parents,'New candidate-only output required');side,capture,manifest=verify_candidate(args.candidate_sha)
 if args.mode=='review':
  require(args.process_receipt is not None and output.is_dir(),'Actualprocess receipt/output required')
  try:capture.write(output/'capture_prepare_review.json',review(output,args.candidate_sha,args.process_receipt))
  except BaseException:capture.write(output/'capture_prepare_rejection.json',dict(status='preserved_actual_counterfactual_rejected',error=traceback.format_exc(),PPO_updates=0,qualification=None));return 2
  return 0
 require(args.process_receipt is None and not output.exists(),'New output; no execute under review receipt')
 pair=capture.load(capture.ORCHESTRATOR,'prepare_pair');plan,admission=pair.load_plan(capture.PAIR);source=Path(admission['source_root'])/'src/wheeled_tasks/chassis/recovery.py';output.mkdir(parents=True)
 resource=None if args.mode=='cpu' else pair.local_GPU_empty();command=[PYTHON,'-B',str(HERE/'run_capture_prepare_021.py'),'--mode','native','--candidate-sha',args.candidate_sha,'--output',str(output)]
 capture.write(output/'run_plan.json',dict(schema='source021-CAPTURE-PREPARE-run-plan-v1',mode=args.mode,candidate_manifest_sha256=args.candidate_sha,
  inherited_SIDE_manifest_sha256=SIDE_SHA,source_commit=plan['source_commit'],native_recovery_source_sha256=capture.sha(source),source_code_modified=False,
  actual_REC3076_checkpoint_sha256=plan['checkpoint_sha256'],actual_SIDE1p5_baseline_review_sha256=capture.sha(SIDE/'native_SIDE_001/side_speed_review.json'),
  native_command=command,resource_check=resource,inherited_baseline_control_override={'side_speed_rad_s':1.5},
  sole_new_control_override='Original instance CAPTURE→PREPARE mask additionally gyro.norm<existing handover_gyro_max(.75)',
  runtime_requested=dict(lanes=4096,policy_steps=600,physics_steps=2400,policy_dt=.02,physics_dt=.005,solver=[64,32],guard_m=.003,selected_lanes=[1708,2216],arm='mean'),
  PPO_updates=0,formal_updates_consumed=0,qualification=None))
 driver=pair.bound_driver()
 if args.mode=='native':
  mechanical=capture.load(side.CAPTURE/'mechanical_capture.py','prepare_private_capture003');side_helper=load(SIDE/'side_speed_capture.py','prepare_SIDE_override');side_helper.expand_capture_module(mechanical)
  hook_helper=load(HERE/'capture_prepare_hook.py','prepare_hook001');original_factory=driver.make_observer
  def factory(base,producer,composition,native_args,identity,report,binding,writer):
   parent=original_factory(base,producer,composition,native_args,identity,report,binding,writer)
   return make_observer(parent,side,capture,mechanical,side_helper,hook_helper,writer,admission['source_root'],capture.sha(source))
  driver.make_observer=factory
 inherited_args=['--admission',str(capture.PAIR/'admission.json'),'--admission-sha',plan['admission_sha256'],'--arm','mean','--output',str(output/'native')]
 if args.mode=='cpu':inherited_args.append('--preflight-only')
 code=driver.main(inherited_args);verify_candidate(args.candidate_sha)
 if code!=0:return code
 if args.mode=='cpu':
  report=capture.read(output/'native/report.json');require(report['status']=='CPU_preflight_passed_scene_not_run' and not report['native_scene_run'],'Appfree native CPUpreflight failed')
  capture.write(output/'CPU_admission.json',dict(status='CPU_authenticated_runtime_hook_not_applied_scene_not_run',candidate_manifest_sha256=args.candidate_sha,
   source_commit=plan['source_commit'],original_identity_sha256=capture.sha(output/'native/identity.json'),native_scene_run=False,runtime_override_applied=False,
   original_source_code_modified=False,PPO_updates=0,qualification=None))
 return 0
if __name__=='__main__':raise SystemExit(main())
