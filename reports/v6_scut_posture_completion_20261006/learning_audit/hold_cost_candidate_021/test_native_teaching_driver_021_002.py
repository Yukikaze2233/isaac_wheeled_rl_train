"""CPU boundaries for the artifact observer. No App, CUDA, PPO, or deployment."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch
torch.set_num_threads(1)

HERE=Path(__file__).parent
AUDIT=HERE.parent
ROOT=Path('/tmp/scut-recovery-frozen-021-qv1ukmbr')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

driver=load('teaching_driver_cpu_v2',HERE/'validate_native_teaching_021_002.py')
from wheeled_tasks.chassis import rewards as module
old=load('teaching_old_fixture',AUDIT/'test_probe_prefix_mean_stochastic_019.py')


def test_exact_engineering_admission_and_bound_tamper_before_execution(tmp_path):
    a=HERE/'engineering_flat_prefix.admission.json'
    admitted=driver.read_admission(a,driver.digest(a))
    assert admitted['context']=='engineering_flat_prefix'
    p=tmp_path/'never.py';p.write_text("raise AssertionError('never execute')")
    with pytest.raises(ValueError,match='Bound artifact bytes differ'):
        driver.load_bound(p,'0'*64,'never_exec')


@pytest.mark.parametrize('field,value', [('context','implicit'),('source_commit','0'*40),('contract_sha256','0'*64),
    ('source_manifest_sha256','0'*64),('contract_relative','../escape'),('seed',True),('block_index',1),('extra',0)])
def test_admission_rejects_context_identity_escape_or_extra_schema(tmp_path,field,value):
    a=json.loads((HERE/'engineering_flat_prefix.admission.json').read_text());a[field]=value
    p=tmp_path/'admission.json';p.write_text(json.dumps(a))
    with pytest.raises(ValueError):driver.read_admission(p,driver.digest(p))


def full_state(context='engineering_flat_prefix'):
    config={'task_campaign':{'domain':'recovery'},'asset_manifest_sha256':'asset','control_math_sha256':'control','learning_rate':1e-4}
    campaign={'domain':'flat' if context=='engineering_flat_prefix' else 'recovery'}
    pred={**config,'task_campaign':campaign}
    info={k:i+1 for i,k in enumerate(driver.CLOCK_NAMES)}
    info.update(contract_sha256='parent',task_campaign=campaign,asset_manifest_sha256='asset',control_math_sha256='control',
        rng_state=torch.get_rng_state(),env_rng_state=torch.Generator().get_state(),cuda_rng_states=[torch.zeros(5,dtype=torch.uint8)],
        training_transitions=100,confirmed_jump_bins_m=[])
    saved={'infos':info,'iter':42}
    for name,dims in [('actor',(35,256,128,64,6)),('critic',(81,256,128,64,1))]:
        state={'distribution.std_param':torch.full((6,),.42)} if name=='actor' else {}
        for i,(ins,outs) in enumerate(zip(dims,dims[1:])):
            state[f'mlp.{2*i}.weight']=torch.zeros(outs,ins);state[f'mlp.{2*i}.bias']=torch.zeros(outs)
        saved[name+'_state_dict']=state
    params=[*saved['actor_state_dict'].values(),*saved['critic_state_dict'].values()]
    saved['optimizer_state_dict']={'param_groups':[{'params':list(range(17)),'lr':1e-4}],
        'state':{i:{'step':torch.tensor(9.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.ones_like(p)} for i,p in enumerate(params)}}
    seal={k:info[k] for k in driver.CLOCK_NAMES};seal.update(task_campaign=campaign,learning_updates=info['successful_updates_total'])
    return saved,seal,pred,config,{'checkpoint_contract_sha256':'parent','context':context}


def test_full_flat_infos_clock_not_rewritten_to_engineering_origin():
    saved,seal,pred,cfg,a=full_state();before=deepcopy(saved)
    clocks=driver.validate_state(saved,seal,pred,cfg,a)
    assert clocks['task_actor_update']==1 and clocks['task_actor_update']!=2294
    for key,value in saved['infos'].items():
        expected=before['infos'][key]
        if isinstance(value,torch.Tensor):assert torch.equal(value,expected)
        elif key=='cuda_rng_states':assert all(torch.equal(a,b) for a,b in zip(value,expected))
        else:assert value==expected
    assert torch.equal(saved['actor_state_dict']['distribution.std_param'],before['actor_state_dict']['distribution.std_param'])
    a['context']='sealed_recovery_resume'
    with pytest.raises(ValueError,match='parent task'):driver.validate_state(saved,seal,pred,cfg,a)


@pytest.mark.parametrize('mode',['clock','std','critic','adam_catalog','adam_moment','adam_lr','rng','plant','domain'])
def test_full_learning_state_negative_cases(mode):
    saved,seal,pred,cfg,a=full_state()
    if mode=='clock':seal['task_actor_update']+=1
    elif mode=='std':saved['actor_state_dict']['distribution.std_param'][0]=0
    elif mode=='critic':saved['critic_state_dict']['mlp.0.weight'][0,0]=float('nan')
    elif mode=='adam_catalog':del saved['optimizer_state_dict']['state'][4]
    elif mode=='adam_moment':saved['optimizer_state_dict']['state'][0]['exp_avg']=torch.zeros(7)
    elif mode=='adam_lr':saved['optimizer_state_dict']['param_groups'][0]['lr']=4e-4
    elif mode=='rng':del saved['infos']['env_rng_state']
    elif mode=='plant':saved['infos']['control_math_sha256']='other'
    else:pred['task_campaign']['domain']='speed'
    with pytest.raises(ValueError):driver.validate_state(saved,seal,pred,cfg,a)


def hold_result():
    n=8;dq=torch.arange(n*6).float().reshape(n,6)/7;omega=torch.arange(n*3).float().reshape(n,3)/9
    enabled=torch.tensor([1,1,1,1,0,1,0,1],dtype=torch.bool)
    owned=torch.tensor([1,1,0,1,1,1,1,1],dtype=torch.bool)
    held=torch.tensor([1,0,1,1,0,1,0,0],dtype=torch.bool)
    settings=deepcopy(module.RECOVERY_HOLD_COST_V1)
    before={k:-torch.arange(n).float()/10 for k in driver.CHANGED_KEYS}
    before.update(track_height=torch.ones(n),failure_event=-200*torch.tensor([0,0,0,1,0,0,0,0]))
    after={k:v.clone() for k,v in before.items()};diag={}
    module.apply_recovery_hold_costs(after,dq,omega,enabled,owned,held,settings,diagnostics=diag)
    return before,after,dq,omega,enabled,owned,held,settings,diag


def test_observed_return_matches_native_tax_preserves_actor_terminal_and_other_payments():
    args=hold_result();mask=driver.check_cost_result(module,*args)
    assert mask.tolist()==[1,0,0,1,0,1,0,0]
    assert args[1]['failure_event'][3]==-200
    assert torch.equal(args[1]['failure_event'],args[0]['failure_event'])


@pytest.mark.parametrize('mode',['order','ordinary','excluded','postrelease','event','yaw','units','mask'])
def test_actual_cost_observer_rejects_nonheld_mutation_or_double_charge(mode):
    args=list(hold_result());before,after,dq,omega,e,o,h,c,d=args
    if mode=='order':after['leg_velocity']=after.pop('leg_velocity')
    elif mode in ('ordinary','excluded','postrelease'):after['wheel_velocity'][{'ordinary':4,'excluded':2,'postrelease':1}[mode]]-=1
    elif mode=='event':after['failure_event'][3]+=1
    elif mode=='yaw':after['dense_yaw_broad'][0]=-1
    elif mode=='units':d['gyro/gate_relative_error'][0]+=1
    else:d['mask'][0]=False
    with pytest.raises(ValueError):driver.check_cost_result(module,*args)


def test_first_release_remains_zero_then_next_interval_exposes_all_original_goals():
    goal=torch.tensor([[.3,0,.305],[0,.6,.305],[0,-.6,.305],[0,0,.305]])
    enabled=torch.ones(4,dtype=torch.bool);release=enabled.clone();held=enabled.clone()
    command=goal.clone();command[:,:2]=0
    assert not driver.check_goals(command,enabled,release,goal,held).any()
    held.zero_()
    assert driver.check_goals(goal,enabled,release,goal,held).tolist()==[1,1,1,0]
    with pytest.raises(ValueError,match='Released native command'):
        driver.check_goals(command,enabled,release,goal,held)


def terminal():
    report=dict(status='native600_completed_preclose',native_teaching_complete=True,policy_steps_completed=600,
        physical_steps_completed=2400,PPO_updates=0,optimizer_steps=0,actor_tensors_and_saved_std_unchanged=True,
        curriculum_clocks_unchanged=True,learning_transition_clock_unchanged=True,payment_hooks_restored_before_app_close=True)
    payment=dict(complete=True,native_lane_intervals=2457600,event_producer_calls=600,max_reconciliation_residual=0)
    teaching=dict(complete=True,hold_producer_calls=600,native_hold_lane_intervals=3,ordinary_unchanged_lane_intervals=5,
        excluded_unchanged_lane_intervals=7,teaching_hooks_restored_preclose=True,L0_all_three_nonzero_followups_observed=False,
        native_formal_probe_structural_audit={'native_scut_train_structure_checked':True,
            'posture_native_readback':{'actual_native_velocity_delta_observed':True}})
    return report,payment,teaching


def test_terminal_is_engineering_credit_only_goal_absence_retains_gap():
    assert driver.validate_terminal(*terminal())=={'engineering_mechanisms_passed':True,
        'actual_L0_followup_exposure_complete':False,'behavior_qualification':None,'PPO_updates':0}


@pytest.mark.parametrize('owner,key,value', [('report','status','starting'),('report','physical_steps_completed',2399),
    ('report','actor_tensors_and_saved_std_unchanged',False),('report','PPO_updates',1),
    ('payment','max_reconciliation_residual',1e-9),('payment','event_producer_calls',599),
    ('teaching','hold_producer_calls',599),('teaching','native_hold_lane_intervals',0),
    ('teaching','teaching_hooks_restored_preclose',False)])
def test_exit_zero_missing_native_evidence_or_cleanup_cannot_admit(owner,key,value):
    r,p,t=terminal();{'report':r,'payment':p,'teaching':t}[owner][key]=value
    with pytest.raises(ValueError):driver.validate_terminal(r,p,t)


def integrated(tmp_path,writer=None,*,native_descriptors=False):
    base=driver.load_bound(AUDIT/'probe_prefix_mean_stochastic_019.py',driver.BASE_SHA,'teaching_base')
    helper=driver.load_bound(AUDIT/'observe_prefix_native_payments_019.py',driver.PAYMENT_SHA,'teaching_helper')
    composition=driver.load_bound(AUDIT/'probe_prefix_mean_stochastic_payments_019.py',driver.COMPOSITION_SHA,'teaching_composition')
    base.POLICY_STEPS=1;base.verify_inputs=lambda *args:True
    env,calls=old.fake_env();env.dt,env.policy_dt,env.decimation,env.tick=.005,.02,4,0
    if native_descriptors:
        class NativeDescriptorEnv(NS):
            def _reset_jump_evidence(self,*args):return 'actual native method'
        env=NativeDescriptorEnv(**vars(env))
    env.cfg={'recovery_hold_cost':deepcopy(module.RECOVERY_HOLD_COST_V1)}
    env.recovery_takeover_settings=None;env.recovery_hold_cost_settings=env.cfg['recovery_hold_cost'];env.recovery_hold_cost_diagnostics={}
    env.generator=torch.Generator().manual_seed(7);env.nonstandard_start=None
    env.recovery.pose_curriculum.settings={'goal_bridge':driver.BRIDGE}
    env.recovery.fsm.config={'handover_gyro_max':.75}
    env.recovery.fsm.WHEEL_RADIUS_M=.06;env.recovery.fsm.WHEEL_RIM_MAX_M_S=.25;env.recovery.fsm.LEG_SPEED_MAX_RAD_S=2.
    env.recovery.geometry=NS(wheel_radius_m=torch.full((2,),.06))
    env.skills=NS(specs={'a':{'recovery_followup_command':[.3,0,.305]},'b':{'recovery_followup_command':[0,.6,.305]},
        'c':{'recovery_followup_command':[0,-.6,.305]},'ordinary':{}})
    env.recovery.followup_command=torch.tensor([[.3,0,.305],[0,.6,.305],[0,-.6,.305],[0,0,.305]])
    env.recovery.learning_mask=torch.tensor([1,1,0,1],dtype=torch.bool)
    env.recovery.pure_rl[2]=False
    env.apply_push=lambda ids:calls.update(push=calls.get('push',0)+1)
    if not native_descriptors:env._reset_jump_evidence=lambda *args:None
    class Structural:
        def __init__(self,env,domain):
            self.counts=Counter();self.max_gap=0.;self.env=env
            begin,torque,push=env.recovery.begin_policy,env.recovery.torque,env.apply_push
            env.recovery.begin_policy=lambda:begin()
            env.recovery.torque=lambda *args:torque(*args)
            env.apply_push=lambda ids:push(ids)
        def observe(self,tick,reward,done,extras):self.counts['all_policy_intervals']+=4
        def finish_posture(self):return {'actual_native_velocity_delta_observed':True}
    before_cost=module.apply_recovery_hold_costs;before_event=module.episode_event_terms
    def step(action):
        calls['step']=calls.get('step',0)+1;env.recovery.begin_policy()
        for _ in range(4):env.recovery.fsm.update();env.recovery.torque(None,None,action)
        extras=old.fake_extras(env,done0=True);d=extras['diagnostics']
        d.update(motor_velocity=torch.ones(4,6),omega=torch.zeros(4,3),reward=torch.zeros(4),planar_speed_world=torch.zeros(4))
        components={k:torch.zeros(4) for k in driver.CHANGED_KEYS};components['positive_baseline']=torch.ones(4)
        module.apply_recovery_hold_costs(components,d['motor_velocity'],d['omega'],env.recovery.enabled,
            env.recovery.learning_mask,env.recovery.policy_motion_hold,env.recovery_hold_cost_settings,diagnostics=env.recovery_hold_cost_diagnostics)
        events=module.episode_event_terms(d['success'],d['terminated'],env.cfg)
        reward=torch.stack(list(components.values())).sum(0)*.02+sum(events.values());d['reward']=reward.clone()
        extras['native_reward_components']=components;extras['policy_learning_mask']=env.recovery.learning_mask.clone()
        mask=env.recovery_hold_cost_diagnostics['mask'];den=mask.sum().clamp_min(1)
        extras['log']={'/HoldCost/actor_hold_population':mask.sum()}
        for axis in ('leg','wheel','gyro'):
            for name in ('gate_relative_error','broad_density','fine_density'):
                extras['log']['/HoldCost/'+axis+'/'+name+'_mean']=env.recovery_hold_cost_diagnostics[axis+'/'+name].sum()/den
        env.tick+=1;env.recovery.pose_curriculum.level_ids[0]=2
        env.last_result=({},reward,d['done'],extras);return env.last_result
    env.step=step
    report={'runtime_preflight_passed':True,'initial_physical_birth_sha256':'synthetic',
        'native_startup':{'runtime_preflight':{'observed':{'modules':{'recovery_hold_cost':True},
        'recovery_hold_cost':env.recovery_hold_cost_settings,
        'recovery_hold_cost_gate_binding':module.validate_recovery_hold_gates(env.recovery_hold_cost_settings,env.recovery.fsm,env.recovery.geometry)}}}}
    (tmp_path/'identity.json').write_text('{"CPU_fixture_only":true}')
    identity=dict(checkpoint_clocks=dict(base.CLOCKS),observation_actor_clocks={k:2294 for k in ('task_actor_update','stage_actor_update','global_actor_update')},
        observation_context='engineering_flat_prefix',forced_engineering_actor_update=2294,forced_clock_is_not_resume_or_learning_progress=True)
    klass=driver.make_observer(base,helper,composition,NS(output=tmp_path,formal_probe=NS(StructuralAudit=Structural)),identity,report,{},writer or base.write_json)
    obs=klass(env,tmp_path)
    return env,calls,obs,step,before_cost,before_event


def test_original_composition_once_only_exact_output_and_old_terminal_lesson(tmp_path):
    env,calls,obs,step,cost,event=integrated(tmp_path)
    rng=torch.get_rng_state().clone();env_rng=env.generator.get_state().clone()
    result=env.step(torch.ones(4,6));assert result is env.last_result
    obs.observe(1,result[3]);assert obs.cost_calls==obs.payment.event_calls==1
    assert calls==dict(begin=1,torque=4,update=4,step=1)
    assert obs.total_hold==2 and obs.total_excluded==1 and obs.total_ordinary==1
    assert obs.payment.first_closed[0] and not obs.live[0]
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(env_rng,env.generator.get_state())
    assert obs._restore()==[]
    assert env.step is step and module.apply_recovery_hold_costs is cost and module.episode_event_terms is event
    obs.sparse.close()
    assert json.loads((tmp_path/'payment_observer_failure.json').read_text())['completed_policy_steps']==1


def test_teaching_summary_failure_after_parent_cleanup_still_persists_preclose(tmp_path):
    def writer(path,value):
        if path.name=='teaching_summary.json':raise OSError('synthetic publication fault')
        path.write_text(json.dumps(value))
    env,calls,obs,step,cost,event=integrated(tmp_path,writer)
    result=env.step(torch.ones(4,6));obs.observe(1,result[3])
    obs.cost_calls=600  # Synthetic late-failure entry only, no native-count claim.
    with pytest.raises(OSError,match='publication fault'):obs.finish(tmp_path)
    assert obs.payment_cleanup_done and obs.payment_complete
    obs.sparse.close()
    failure=json.loads((tmp_path/'teaching_observer_failure.json').read_text())
    assert failure['status']=='teaching_finish_rejected_preclose' and failure['app_close_returned'] is None
    ledger=json.loads((tmp_path/'payment_summary.json').read_text())
    assert ledger['checkpoint_clocks']['task_actor_update']==2579
    assert ledger['observation_actor_clocks']['task_actor_update']==2294
    assert env.step is step and module.apply_recovery_hold_costs is cost and module.episode_event_terms is event


def review_fixture(tmp_path,*,exposure):
    r,p,t=terminal();t['L0_all_three_nonzero_followups_observed']=exposure
    clocks={'task_actor_update':13000};observation={'task_actor_update':2294}
    identity=dict(checkpoint_clocks=clocks,observation_actor_clocks=observation,observation_context='engineering_flat_prefix',
        observer_artifact_sha256={},qualification=None,PPO_updates=0)
    p.update(checkpoint_clocks=clocks,observation_actor_clocks=observation,observation_context='engineering_flat_prefix')
    r.update(source021_actual_observation_preclose=True,checkpoint_unchanged=True,source_files_unchanged=True,
        native_effective_std_matches_saved=True,formal_updates_consumed=0)
    for name,value in [('identity.json',identity),('payment_summary.json',p),('teaching_summary.json',t)]:
        (tmp_path/name).write_text(json.dumps(value))
    r.update(identity_sha256=driver.digest(tmp_path/'identity.json'),payment_summary_sha256=driver.digest(tmp_path/'payment_summary.json'),
        teaching_summary_sha256=driver.digest(tmp_path/'teaching_summary.json'))
    (tmp_path/'report.json').write_text(json.dumps(r))


def test_external_collector_rejects_clean_mechanisms_without_all_actual_nonzero_goals(tmp_path):
    review_fixture(tmp_path,exposure=False)
    with pytest.raises(ValueError,match='coverage gap'):driver.review_output(tmp_path)


def test_external_collector_requires_preclose_receipts_and_leaves_qualification_null(tmp_path):
    review_fixture(tmp_path,exposure=True)
    r=driver.review_output(tmp_path)
    assert r['all_required_passed'] and r['qualification'] is None and r['formal_updates_consumed']==0
    (tmp_path/'teaching_observer_failure.json').write_text('{}')
    with pytest.raises(ValueError,match='failure sidecar'):driver.review_output(tmp_path)


def test_incomplete_native_finish_failure_receipt_written_before_close(tmp_path):
    env,calls,obs,step,cost,event=integrated(tmp_path)
    with pytest.raises(ValueError,match='coverage incomplete'):obs.finish(tmp_path)
    assert json.loads((tmp_path/'teaching_observer_failure.json').read_text())['status']=='teaching_finish_rejected_preclose'
    obs.sparse.close()
    failure=json.loads((tmp_path/'payment_observer_failure.json').read_text())
    assert not failure['complete'] and failure['app_close_returned'] is None
    assert env.step is step and module.apply_recovery_hold_costs is cost and module.episode_event_terms is event


def test_descriptor_equivalence_only_same_bound_owner_and_function():
    class Owner:
        def callback(self):return 1
        def other(self):return 1
    one,two=Owner(),Owner()
    assert one.callback is not one.callback  # The actual native001 failure.
    assert driver.same_callback(one.callback,one.callback)
    assert not driver.same_callback(one.callback,two.callback)
    assert not driver.same_callback(one.callback,one.other)
    plain=lambda:1
    assert driver.same_callback(plain,plain)
    assert not driver.same_callback(plain,lambda:1)


def test_real_class_descriptor_native_cleanup_and_second_close_are_idempotent(tmp_path):
    env,calls,obs,step,cost,event=integrated(tmp_path,native_descriptors=True)
    assert '_reset_jump_evidence' not in vars(env)
    before=env._reset_jump_evidence
    result=env.step(torch.ones(4,6));obs.observe(1,result[3])
    assert obs._restore()==[]
    assert obs._restore()==[]  # ClosingStream after prior finish teardown.
    assert '_reset_jump_evidence' not in vars(env)  # No spurious method shadow.
    assert driver.same_callback(before,env._reset_jump_evidence)
    assert env.step is step and module.apply_recovery_hold_costs is cost and module.episode_event_terms is event
    obs.sparse.close();obs.sparse.close()
    failure=json.loads((tmp_path/'payment_observer_failure.json').read_text())
    assert failure['callback_restore_errors']==[]


@pytest.mark.parametrize('replacement',['owner','function'])
def test_changed_native_descriptor_owner_or_function_still_rejects(tmp_path,replacement):
    env,calls,obs,step,cost,event=integrated(tmp_path,native_descriptors=True)
    if replacement=='owner':
        other=type(env)();env._reset_jump_evidence=other._reset_jump_evidence
    else:
        def different(self,*args):return 'actual native method'
        env._reset_jump_evidence=type(env._reset_jump_evidence)(different,env)
    errors=obs._restore()
    assert errors==['Formal probe hook ownership changed: _reset_jump_evidence']
    assert obs._restore()==[]  # Prior rejection persists separately; no false second ownership errors.
    obs.sparse.close()


def test_parent_success_then_teaching_failure_with_native_descriptor_has_no_false_cleanup(tmp_path):
    def writer(path,value):
        if path.name=='teaching_summary.json':raise OSError('late native-descriptor publication')
        path.write_text(json.dumps(value))
    env,calls,obs,step,cost,event=integrated(tmp_path,writer,native_descriptors=True)
    result=env.step(torch.ones(4,6));obs.observe(1,result[3]);obs.cost_calls=600
    with pytest.raises(OSError,match='late native-descriptor'):obs.finish(tmp_path)
    assert obs.payment_complete and obs.payment_cleanup_done and obs.teaching_cleanup_done
    obs.sparse.close()
    assert not (tmp_path/'payment_observer_failure.json').exists()
    assert (tmp_path/'teaching_observer_failure.json').exists()
    assert env.step is step and module.apply_recovery_hold_costs is cost and module.episode_event_terms is event
