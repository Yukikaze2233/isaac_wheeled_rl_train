"""One diagnostic instance hook, CPU-proven equivalent to the single source patch.

Source file, actions, RNG, config and physics are unchanged. The original native
FSM.update is called once; only its CAPTURE→PREPARE mask is made stricter.
"""
from __future__ import annotations
from copy import deepcopy
import gzip,json,types

def require(v,message):
 if not bool(v):raise ValueError(message)

class CapturePrepareHook:
 def __init__(self,env,hooks,writer,output,plain,authenticate,cohort):
  self.env,self.fsm,self.hooks,self.writer,self.output,self.plain,self.cohort=env,env.recovery.fsm,hooks,writer,output,plain,cohort
  authenticate(self.fsm)
  require('update' not in vars(self.fsm) and '_enter' not in vars(self.fsm),'FSM already has local callback owner')
  self.config=deepcopy(self.fsm.config)
  require(self.config['side_speed_rad_s']==1.5 and self.config['handover_gyro_max']==.75
    and self.config['dt']==.005 and self.config['max_script_seconds']==8. and self.config['max_reroutes']==1,'Diagnostic config binding differs')
  self.calls_attempted=self.calls_completed=self.enter_calls=self.blocked_lane_transitions=0
  self.first_interventions={};self.first_global_intervention=None;self.current_gyro=self.current_gate=None
  self.in_update=self.applied=self.complete=self.closed=False
  self.first_reference_tick=self.last_reference_tick=None
  self.events=None

 def assert_owner(self):
  require(self.applied and not self.closed and self.fsm.config==self.config,'Diagnostic hook/config changed')

 def apply(self):
  require(not self.applied and not self.closed,'Hook duplicate or closed')
  require(self.fsm.config==self.config,'Config changed before hook')
  original_update,original_enter=self.fsm.update,self.fsm._enter
  require(isinstance(original_update,types.MethodType) and isinstance(original_enter,types.MethodType)
    and original_update.__self__ is self.fsm and original_enter.__self__ is self.fsm,'Native callback owners differ')
  self.events=gzip.open(self.output/'capture_prepare_enter_events.jsonl.gz','wt')
  self.applied=True
  def enter(owner,mask,phase):
   require(owner is self.fsm,'Foreign FSM _enter owner')
   self.enter_calls+=1
   if self.in_update and int(phase)==7:
    if self.current_gate is None:self.current_gate=self.current_gyro.norm(dim=-1)<self.fsm.config['handover_gyro_max']
    applicable=mask & (self.fsm.phase==6)
    blocked=applicable & ~self.current_gate
    n=int(blocked.sum())
    if n:
     self.blocked_lane_transitions+=n
     if self.first_global_intervention is None:self.first_global_intervention=self.calls_attempted
    # Gate only the original native CAPTURE population, never another PREPARE route.
    result=mask & (~(self.fsm.phase==6) | self.current_gate)
    if bool(applicable.any()):
     selected=[]
     for lane,birth_id in self.cohort():
      if bool(applicable[lane]):
       selected.append(dict(lane=lane,birth_id=birth_id,blocked=bool(blocked[lane]),original_mask=True,
         effective_mask=bool(result[lane]),phase_before_native_enter=int(self.fsm.phase[lane]),
         gyro_norm_rad_s=float(self.current_gyro[lane].norm()),gyro_limit_rad_s=self.fsm.config['handover_gyro_max']))
       if birth_id==1 and bool(blocked[lane]) and lane not in self.first_interventions:self.first_interventions[lane]=self.calls_attempted
     self.events.write(json.dumps(dict(update_call=self.calls_attempted,native_reference_tick=self.last_reference_tick,
      selected=selected,all4096_attempted_capture_transitions=int(applicable.sum()),all4096_blocked_capture_transitions=n),allow_nan=False)+'\n')
    mask=result
   return original_enter(mask,phase)
  def update(owner,q_control,dq_control,gravity,gyro,estimated_height,support,body_clear):
   require(owner is self.fsm,'Foreign FSM update owner');self.assert_owner();require(not self.in_update,'Recursive diagnostic FSM update')
   self.calls_attempted+=1;self.current_gyro=gyro;self.current_gate=None;self.in_update=True
   self.last_reference_tick=int(self.env.recovery.reference_tick)
   if self.first_reference_tick is None:self.first_reference_tick=self.last_reference_tick
   try:
    result=original_update(q_control,dq_control,gravity,gyro,estimated_height,support,body_clear)
    self.calls_completed+=1;self.assert_owner();return result
   finally:self.in_update=False;self.current_gyro=self.current_gate=None
  try:
   self.hooks.install(self.fsm,'_enter',types.MethodType(enter,self.fsm))
   self.hooks.install(self.fsm,'update',types.MethodType(update,self.fsm))
   self.writer(self.output/'capture_prepare_hook_applied.json',self.receipt('applied_before_first_policy'))
  except BaseException:
   self.cleanup();raise

 def receipt(self,status):
  return dict(schema='source021-isolated-CAPTURE-PREPARE-runtime-hook-v1',status=status,
   explicit_runtime_control_override=True,original_source_code_modified=False,
   rule='Only native current-phase CAPTURE mask requesting PREPARE additionally gyro.norm<handover_gyro_max',
   gyro_limit_rad_s=self.config['handover_gyro_max'],config_unchanged=self.fsm.config==self.config,
   original_update_calls_attempted=self.calls_attempted,original_update_calls_completed=self.calls_completed,
   original_enter_calls=self.enter_calls,blocked_lane_transitions=self.blocked_lane_transitions,
   first_global_intervention_update_call=self.first_global_intervention,
   selected_first_birth_first_blocked_update_call=self.first_interventions,
   native_reference_tick_first=self.first_reference_tick,native_reference_tick_last=self.last_reference_tick,
   complete=self.complete,hook_callbacks_restored=self.closed and self.hooks.closed,
   no_pending_update=not self.in_update,PPO_updates=0,formal_updates_consumed=0,qualification=None)

 def finish(self):
  self.assert_owner();require(not self.in_update and self.calls_attempted==self.calls_completed==2400,'Original2400 reference updates not completed')
  require(self.last_reference_tick-self.first_reference_tick==2399,'Native reference cadence differs')
  self.complete=True;return self.cleanup()

 def cleanup(self):
  if self.closed:return []
  errors=self.hooks.restore()
  if self.in_update:errors.append('Cleanup during a pending original FSM update')
  if self.fsm.config!=self.config:errors.append('Foreign FSM config mutation; hook owns no config restoration')
  self.closed=True
  if self.events is not None:self.events.close()
  self.writer(self.output/'capture_prepare_hook_preclose.json',self.receipt(
    'complete_restored_preclose' if self.complete and not errors else 'incomplete_or_restore_rejected')|{'errors':errors})
  return errors

def pre_intervention_review(side,baseline_path,actual_path,hook):
 baseline={lane:{} for lane in side.LANES};actual={lane:{} for lane in side.LANES}
 for path,dst in ((baseline_path,baseline),(actual_path,actual)):
  for row in side.rows(path):
   lane=row['state']['lane'];step=row['physical_step'];require(lane in dst and step not in dst[lane],'Unknown/duplicate capture row');dst[lane][step]=row
 result=[]
 for lane in side.LANES:
  first=hook['selected_first_birth_first_blocked_update_call'].get(str(lane))
  if first is None:
   # No intervention on the selected first birth: compare only common observed history.
   end=min(max(baseline[lane]),max(actual[lane]))
  else:end=first-1
  require(all(i in baseline[lane] and i in actual[lane] for i in range(end+1)),'Pre-intervention history incomplete')
  mismatch=None
  for step in range(end+1):
   why=side.compare_bits(side.comparable(baseline[lane][step]),side.comparable(actual[lane][step]))
   if why and mismatch is None:mismatch={'physical_step':step,'reason':why}
  result.append(dict(lane=lane,first_blocked_update_call=first,compared_through_physical_step=end,
    compared_rows=end+1,bitwise_identical=mismatch is None,first_mismatch=mismatch))
 return dict(status='pre_intervention_equal' if all(x['bitwise_identical'] for x in result) else 'pre_intervention_rejected',lanes=result,
  scope='All selected native state/control/contact through the frame before actual first blocked transition; source-phase and control changes are expected on intervention frame',
  declared_exclusions=['Inherited side_reference_audit telemetry','Allview contact capacity count indicators'],tolerance_used=False)
