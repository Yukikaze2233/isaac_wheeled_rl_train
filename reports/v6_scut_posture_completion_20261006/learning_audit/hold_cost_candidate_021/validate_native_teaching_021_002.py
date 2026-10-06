#!/usr/bin/env python3
"""Revision002 bound source021 observation; no PPO, CPU preflight is App-free.

The immutable55 inference loop/native observer and51580 payment producer hook
are reused. New source/contract/seal identity is explicit in a parent admission,
not borrowed from their historical2579 authentication. No helper source changes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import types

BASE_SHA = '55f4511284ae09010ba9d79af2a747b70b1f9da3df6b4f713323340b64648b9d'
PAYMENT_SHA = '51580e8bd7c08025edd69588e70e62a2970cb6afb5f97f355d85ddf48460a8ac'
COMPOSITION_SHA = 'a9e7bc648f1a8e34d802695124a3faf72998a55fe278a0000cc5cf1b803a4ca5'
SOURCE_COMMIT = 'd0dce6181df83b03fe166abcbadef3c81de2e57e'
INVENTORY_SHA = 'c3f3f17a3747b27e2982793a796fff29fe622d3406b9d2d614a682bbaf34d068'
CONTRACT_SHA = '7918e7850e16b96edfa54a8854d3f3571e8101dfa6527cad78a06b4e15baea95'
BRIDGE = 'native-original-followup-after-stable-gate-v1'
CHANGED_KEYS = ('leg_velocity','wheel_velocity','horizontal_angular_velocity','dense_yaw_broad','dense_yaw_precision')
CLOCK_NAMES = ('task_actor_update','stage_actor_update','global_actor_update','actor_updates_consumed',
               'successful_updates_total','learning_lineage_updates','critic_warmup_remaining')


def require(condition, message):
    if not bool(condition): raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''): h.update(b)
    return h.hexdigest()


def same_callback(actual, expected):
    """Descriptors create fresh method objects; only the same owner/function qualifies."""
    return actual is expected or (isinstance(actual,types.MethodType)
        and isinstance(expected,types.MethodType) and actual.__self__ is expected.__self__
        and actual.__func__ is expected.__func__)


def load_bound(path, sha, name):
    path = Path(path).resolve(); raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'Bound artifact bytes differ: '+str(path))
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def read_admission(path, expected_sha):
    require(digest(path) == expected_sha, 'Parent admission SHA differs')
    a = json.loads(Path(path).read_text())
    required = {'schema','context','source_commit','source_root','source_manifest','source_manifest_sha256',
        'contract_relative','contract_sha256','checkpoint','checkpoint_sha256','checkpoint_contract_sha256',
        'checkpoint_seal_sha256','seed','block_index'}
    require(set(a) == required and a['schema'] == 'source021-native-teaching-admission-v1', 'Complete parent admission required')
    require(a['context'] in ('engineering_flat_prefix','sealed_recovery_resume'), 'Explicit observation context required')
    require(type(a['seed']) is int and a['seed'] >= 0 and type(a['block_index']) is int and a['block_index'] == 0, 'Native seed/block differs')
    rel = Path(a['contract_relative'])
    require(not rel.is_absolute() and '..' not in rel.parts, 'Contract path escape')
    require(len(a['source_commit']) == 40 and all(c in '0123456789abcdef' for c in a['source_commit']), 'Frozen commit label required')
    require(a['source_commit']==SOURCE_COMMIT and a['source_manifest_sha256']==INVENTORY_SHA
            and a['contract_sha256']==CONTRACT_SHA,'Frozen source021 admission differs')
    for key in [k for k in a if k.endswith('_sha256')]:
        require(type(a[key]) is str and len(a[key]) == 64 and all(c in '0123456789abcdef' for c in a[key]), 'Invalid admitted SHA '+key)
    return a


def validate_state(saved, seal, predecessor, config, a):
    """Full sealed learning state is authenticated, never loaded into an optimizer."""
    import torch
    require(set(saved) >= {'actor_state_dict','critic_state_dict','optimizer_state_dict','infos','iter'}, 'Full learning state required')
    info = saved['infos']
    require(info['contract_sha256'] == a['checkpoint_contract_sha256'], 'Parent infos contract differs')
    require(info['task_campaign'] == seal['task_campaign'] == predecessor['task_campaign'], 'Parent campaign differs')
    require(config['task_campaign']['domain'] == 'recovery', 'Recovery-only target required')
    expected_domain = 'flat' if a['context'] == 'engineering_flat_prefix' else 'recovery'
    require(predecessor['task_campaign']['domain'] == expected_domain, 'Admitted parent task differs')
    for name in ('asset_manifest_sha256','control_math_sha256'):
        require(info[name] == predecessor[name] == config[name], 'Frozen plant identity differs '+name)
    clocks = {k:info[k] for k in CLOCK_NAMES}
    require(all(type(x) is int and x >= 0 for x in clocks.values()), 'Full nonnegative native clocks required')
    for k in ('task_actor_update','stage_actor_update','global_actor_update','actor_updates_consumed','learning_lineage_updates'):
        require(seal[k] == info[k], 'Seal/infos clock differs '+k)
    require(seal['learning_updates'] == info['successful_updates_total'], 'Consumption clock differs')
    parameters = []
    for name, dims in [('actor',(35,256,128,64,6)),('critic',(81,256,128,64,1))]:
        expected = {'distribution.std_param':(6,)} if name == 'actor' else {}
        for i,(ins,outs) in enumerate(zip(dims,dims[1:])):
            expected[f'mlp.{i*2}.weight']=(outs,ins); expected[f'mlp.{i*2}.bias']=(outs,)
        state=saved[name+'_state_dict']; require(set(state)==set(expected),'Native parameter catalog differs '+name)
        require(all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and tuple(v.shape)==expected[k]
            and torch.isfinite(v).all() for k,v in state.items()), 'Native finite parameters differ '+name)
        parameters.extend(state.values())
    require((saved['actor_state_dict']['distribution.std_param']>0).all(),'Saved std must remain positive')
    opt=saved['optimizer_state_dict']
    require(len(opt['param_groups'])==1 and set(opt['state'])==set(range(17)), 'Complete Adam catalog differs')
    require(opt['param_groups'][0]['params']==list(range(17)) and opt['param_groups'][0]['lr']==config['learning_rate'], 'Saved Adam order/LR differs')
    for i,p in enumerate(parameters):
        s=opt['state'][i]; require(set(s)=={'step','exp_avg','exp_avg_sq'},'Saved Adam fields differ')
        require(all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in s.values()), 'Nonfinite Adam state')
        require(s['step'].numel()==1 and float(s['step'])>0 and s['exp_avg'].shape==s['exp_avg_sq'].shape==p.shape,'Adam shape/step differs')
    require(isinstance(info.get('rng_state'),torch.Tensor) and isinstance(info.get('env_rng_state'),torch.Tensor)
        and info.get('cuda_rng_states') and type(info.get('training_transitions')) is int
        and 'confirmed_jump_bins_m' in info,'Complete resume RNG/transition state required')
    return clocks


def authenticate(args, a, base):
    import torch
    root=Path(a['source_root']).resolve(); invpath=Path(a['source_manifest']).resolve(); pt=Path(a['checkpoint']).resolve()
    require(digest(invpath)==a['source_manifest_sha256'],'Source inventory SHA differs')
    files=json.loads(invpath.read_text())
    require(isinstance(files,dict) and files
            and all(type(v) is str and len(v)==64 for v in files.values()),'Archive-owned source inventory absent')
    require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files),'Source path escape')
    require({n:digest(root/n) for n in files}==files,'Actual source differs from frozen inventory')
    required_sources=['scripts/train_chassis.py','scripts/probe_scut_curriculum.py',
        *['src/wheeled_tasks/chassis/'+n+'.py' for n in ('env','rewards','recovery','recovery_training',
            'recovery_prefix_curriculum','recovery_quality_revision','root_kinematics','preflight')]]
    require(all(n in files for n in required_sources),'New teaching/native dependencies missing from source pin')
    contract=root/a['contract_relative']; require(digest(contract)==a['contract_sha256'],'Current worker SHA differs')
    for path,key in [(pt,'checkpoint_sha256'),(pt.with_suffix('.contract.json'),'checkpoint_contract_sha256'),
                     (pt.with_suffix('.checkpoint.json'),'checkpoint_seal_sha256')]:
        require(digest(path)==a[key],'Admitted immutable parent differs '+key)
    seal=json.loads(pt.with_suffix('.checkpoint.json').read_text())
    require(seal.get('sealed') is True and seal.get('schema')=='chassis-learning-checkpoint-v1'
        and seal['checkpoint_sha256']==a['checkpoint_sha256'] and seal['contract_sha256']==a['checkpoint_contract_sha256'],'Parent seal differs')
    sys.path[:0]=[str(root/'src'),str(root/'scripts')]
    from train_chassis import preflight
    from wheeled_tasks.chassis.recovery_quality_revision import canonical_worker, revision_binding
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.rewards import validate_recovery_hold_cost
    config,manifest=preflight(contract)
    require(config==canonical_worker('recovery',root),'Canonical v19 recovery worker differs')
    require(config['task_campaign']['revision_migration']==revision_binding(),'Explicit predecessor migration differs')
    if a['context'] == 'sealed_recovery_resume':
        require(a['checkpoint_contract_sha256'] in (a['contract_sha256'],revision_binding()['source_contract_sha256']), 'Unapproved recovery parent contract')
    require(config['enabled_stages']==['mixed'] and config['record_diagnostics'] is True
        and config.get('auto_reset',True) is True and not config.get('evaluation_exact_cases')
        and 'recovery_takeover' not in config,'Actual unprojected train semantics required')
    require(tuple(config[k] for k in ('physics_dt','policy_dt','actor_dim','critic_dim','action_dim'))==(.005,.02,35,81,6),'Native clocks/ABI differ')
    validate_recovery_hold_cost(config['recovery_hold_cost'])
    require(config['recovery_hold_cost']['enabled'] is True and config['recovery_training']['prefix_curriculum']['goal_bridge']==BRIDGE,'Both teaching fields must be enabled')
    control=load_chassis_control(root/config['control_math_source'])
    receipt=design_preflight(config,control,root,stage='mixed',role='train',seed=a['seed'],num_envs=4096,block_index=a['block_index'])
    predecessor=json.loads(pt.with_suffix('.contract.json').read_text())
    saved=torch.load(pt,map_location='cpu',weights_only=True); clocks=validate_state(saved,seal,predecessor,config,a)
    probe=load_bound(root/'scripts/probe_scut_curriculum.py',files['scripts/probe_scut_curriculum.py'],'source021_formal_native_probe')
    observation_clocks=dict(clocks)
    engineering_provenance=None
    if a['context'] == 'engineering_flat_prefix':
        require(args.arm == 'mean', 'Engineering flat context permits mean inference only')
        require(a['checkpoint_sha256']==probe.BASELINE_PT_SHA and a['checkpoint_contract_sha256']==probe.BASELINE_CONTRACT_SHA,
                'Engineering context requires exact formal-probe recommended_flat identity')
        native_args=types.SimpleNamespace(domain='recovery',contract=contract,policy_steps=600,checkpoint=pt)
        p_config,p_manifest,p_control,stage,seed,block,p_receipt,engineering_provenance=probe.authenticate(native_args)
        require(p_config==config and p_manifest==manifest and p_control==control and stage=='mixed'
                and seed==a['seed'] and block==a['block_index'] and p_receipt==receipt,
                'Formal frozen-probe independent admission differs')
        origin=probe.POSTURE_ORIGINS['recovery']
        require(origin==config['recovery_training']['prefix_curriculum']['origin_actor_update']==2294,
                'Engineering actor clock differs from frozen prefix origin')
        for k in ('task_actor_update','stage_actor_update','global_actor_update'):
            observation_clocks[k]=origin
    snapshot=base.snapshot_sources(root,config,files)
    identity=dict(source_commit_label=a['source_commit'],source_root=str(root),source_identity_scope='parent_inventory_pinned_bytes',
        source_files_sha256=snapshot,source_manifest=str(invpath),source_manifest_sha256=a['source_manifest_sha256'],
        archive_source_files=files,wrapper_sha256=BASE_SHA,contract_sha256=a['contract_sha256'],
        checkpoint=str(pt),checkpoint_sha256=a['checkpoint_sha256'],checkpoint_contract_sha256=a['checkpoint_contract_sha256'],
        checkpoint_seal_sha256=a['checkpoint_seal_sha256'],checkpoint_clocks=clocks,checkpoint_saved_iter_field=saved['iter'],
        observation_context=a['context'],observation_actor_clocks={k:observation_clocks[k] for k in ('task_actor_update','stage_actor_update','global_actor_update')},
        forced_engineering_actor_update=(2294 if a['context']=='engineering_flat_prefix' else None),
        forced_clock_is_not_resume_or_learning_progress=a['context']=='engineering_flat_prefix',
        actual_parent_seal_infos_not_modified=True,formal_probe_provenance=engineering_provenance,
        observation_rng_protocol='original55 saved parent CPU/environment/CUDA RNG then one native reset; sampling private stream uses frozen train seed',
        actor_factory='original55 load_native_actor, native saved Gaussian retained; no formal-probe OnPolicyRunner/optimizer factory',
        saved_std=saved['actor_state_dict']['distribution.std_param'].tolist(),
        actor_tensor_sha256={k:base.tensor_sha(v) for k,v in saved['actor_state_dict'].items()},
        seed=a['seed'],block_index=a['block_index'],design_preflight=receipt,qualification=None,PPO_updates=0)
    args.source_root=root;args.source_manifest=invpath;args.checkpoint=pt
    # Bind only the reusable observer loop's admitted input context, not its
    # native producers, commands, rewards, resets or private sample algorithm.
    base.SOURCE_COMMIT=a['source_commit'];base.CHECKPOINT_SHA=a['checkpoint_sha256'];base.CONTRACT_SHA=a['contract_sha256']
    base.CLOCKS=observation_clocks;base.SEED=a['seed'];base.BLOCK=a['block_index']
    args.formal_probe=probe
    return config,manifest,control,saved,identity


def check_cost_result(module, before, after, dq, omega, enabled, owned, held, settings, diagnostics):
    """Compare an observed producer return; never call that producer a second time."""
    import torch
    require(tuple(before)==tuple(after),'Native reward catalog/order changed')
    mask=enabled & owned & held
    require(torch.equal(diagnostics['mask'],mask),'Actual hold payer mask differs')
    errors={'leg':torch.where(mask[:,None],dq,0.)[:,[0,1,3,4]].abs().amax(-1)/settings['leg_gate_rad_s'],
        'wheel':torch.where(mask[:,None],dq,0.)[:,[2,5]].abs().amax(-1)*settings['wheel_radius_m']/settings['wheel_rim_gate_m_s'],
        'gyro':torch.where(mask[:,None],omega,0.).norm(dim=-1)/settings['gyro_gate_rad_s']}
    for key in before:
        require(torch.equal(after[key][~mask],before[key][~mask]),'Nonheld native density changed '+key)
        if key not in CHANGED_KEYS: require(torch.equal(after[key],before[key]),'Unrelated native density changed '+key)
    for axis,key in [('leg','leg_velocity'),('wheel','wheel_velocity'),('gyro','horizontal_angular_velocity')]:
        broad,fine=module.dense_error_terms(errors[axis],settings[axis])
        require(torch.equal(diagnostics[axis+'/gate_relative_error'],errors[axis]),'Native gate units differ '+axis)
        require(torch.equal(diagnostics[axis+'/broad_density'],broad) and torch.equal(diagnostics[axis+'/fine_density'],fine),'Native cost diagnostic differs '+axis)
        require(torch.equal(after[key][mask],(broad+fine)[mask]),'Actual hold density differs '+key)
    for key in ('dense_yaw_broad','dense_yaw_precision'):
        require(after[key][mask].eq(0).all(),'Native hold yaw double charge '+key)
    return mask


def check_goals(commands, enabled, released, original, held):
    import torch
    require(commands.shape==original.shape,'Original followup shape differs')
    # Release is committed at the end of outcome(); the first such interval
    # still owns its begin-policy zero command and hold cost. The next real
    # interval is the first original-motion exposure, as in the native FSM.
    motion=enabled & released & ~held
    require(torch.equal(commands[motion],original[motion]),'Released native command differs from original skill goal')
    require(commands[enabled & held,:2].eq(0).all(),'Begin-policy held goal escaped the original gate')
    return motion & original[:,:2].abs().gt(0).any(-1)


def make_observer(base, helper, composition, args, identity, report, binding, writer):
    import torch
    from wheeled_tasks.chassis import rewards as module
    # Replace historical metadata admission of the artifact composition only.
    # The byte-bound native loop, producer helper and controller remain intact.
    composition.native_context=lambda r,final=False:require(r.get('runtime_preflight_passed') is True and 'initial_physical_birth_sha256' in r,'Fresh native context absent')
    def observation_writer(path,value):
        if Path(path).name=='payment_summary.json':
            require(value['checkpoint_clocks']==base.CLOCKS,'Inherited observation clock ledger differs')
            value['checkpoint_clocks']=deepcopy(identity['checkpoint_clocks'])
            value['observation_actor_clocks']=deepcopy(identity['observation_actor_clocks'])
            value['observation_context']=identity['observation_context']
            value['forced_engineering_actor_update']=identity['forced_engineering_actor_update']
            value['forced_clock_is_not_resume_or_learning_progress']=identity['forced_clock_is_not_resume_or_learning_progress']
        return writer(path,value)
    parent=composition.observer_class(base,helper,args,identity,report,binding,observation_writer)
    class TeachingObserver(parent):
        def __init__(self,env,output):
            self.teaching_prior_step=self.teaching_step=self.cost_hook=self.cost_prior=None
            self.structural_owners=[]
            self.teaching_cleanup_done=False
            self.cost_calls=0;self.cost_active=None;self.total_hold=self.total_ordinary=self.total_excluded=0
            self.exposure=torch.zeros(2,3,3,dtype=torch.long,device=env.device)
            super().__init__(env,output)
            try:
                startup=report['native_startup']['runtime_preflight']['observed']
                require(startup['modules']['recovery_hold_cost'] is True and startup['recovery_hold_cost']==env.cfg['recovery_hold_cost'],'New actual native hold field disabled/different')
                expected_gate=module.validate_recovery_hold_gates(env.recovery_hold_cost_settings,env.recovery.fsm,env.recovery.geometry)
                require(startup['recovery_hold_cost_gate_binding']==expected_gate,'Native gate binding differs')
                require(env.recovery.pose_curriculum.settings['goal_bridge']==BRIDGE,'New actual native goal bridge absent')
                owners=[(env.recovery,'begin_policy'),(env.recovery,'torque'),(env,'apply_push'),(env,'_reset_jump_evidence')]
                if env.nonstandard_start is not None:
                    owners.extend([(env.nonstandard_start,'begin_policy'),(env.nonstandard_start,'apply_torque')])
                prior=[(owner,name,getattr(owner,name)) for owner,name in owners]
                try:
                    self.structural=args.formal_probe.StructuralAudit(env,'recovery')
                finally:
                    self.structural_owners=[(owner,name,getattr(owner,name),old) for owner,name,old in prior]
                specs=env.skills.specs
                self.original_goals=torch.tensor([specs[n].get('recovery_followup_command',[0.,0.,.305]) for n in env.scene_groups],device=env.device)
                require(torch.equal(env.recovery.followup_command[env.recovery.enabled],self.original_goals[env.recovery.enabled]),'Actual reset discarded original followup goal')
                self.cost_prior=module.apply_recovery_hold_costs
                def cost(components,dq,omega,enabled,owned,held,settings,*,diagnostics=None):
                    require(self.payment.active is not None and self.cost_active is None,'Hold producer outside/duplicate policy interval')
                    require(settings is env.recovery_hold_cost_settings and diagnostics is env.recovery_hold_cost_diagnostics,'Hold producer settings/diagnostic object differs')
                    before={k:v.detach().clone() for k,v in components.items()}
                    e,o,h=enabled.clone(),owned.clone(),held.clone()
                    result=self.cost_prior(components,dq,omega,enabled,owned,held,settings,diagnostics=diagnostics)
                    require(result is components,'Hold producer return dictionary changed')
                    mask=check_cost_result(module,before,result,dq,omega,e,o,h,settings,diagnostics)
                    self.cost_active=dict(mask=mask.clone(),enabled=e,owned=o,held=h,
                        diagnostic={k:v.detach().clone() for k,v in diagnostics.items()})
                    self.cost_calls+=1
                    return result
                self.cost_hook=cost;module.apply_recovery_hold_costs=cost
                self.teaching_prior_step=env.step
                def step(actions):
                    result=self.teaching_prior_step(actions)
                    self.structural.observe(self.payment.tick,result[1],result[2],result[3])
                    self.check_interval(result[3]);return result
                self.teaching_step=step;env.step=step
            except BaseException:
                self._before_close();raise

        def check_interval(self,extras):
            c=self.cost_active;require(c is not None,'New native hold producer missing')
            d=extras['diagnostics'];valid=extras['policy_learning_mask'];enabled=d['recovery_enabled'].bool()
            require(torch.equal(c['enabled'],enabled) and torch.equal(c['owned'],valid) and torch.equal(c['held'],self.hold_start),'Native full-interval payer ownership differs')
            mask=enabled & valid & self.hold_start;require(torch.equal(c['mask'],mask),'Native mask escaped actor interval')
            self.total_hold+=int(mask.sum());self.total_ordinary+=int((~enabled).sum());self.total_excluded+=int((enabled & ~valid).sum())
            values=c['diagnostic'];denominator=mask.sum().clamp_min(1)
            require(torch.equal(extras['log']['/HoldCost/actor_hold_population'],mask.sum()),'Native HoldCost population log differs')
            for axis in ('leg','wheel','gyro'):
                for name in ('gate_relative_error','broad_density','fine_density'):
                    require(torch.equal(extras['log']['/HoldCost/'+axis+'/'+name+'_mean'],values[axis+'/'+name].sum()/denominator),'Native pre-reset HoldCost log differs')
            release=d['recovery_motion_released'].bool()
            exposed=check_goals(d['commands'],enabled,release,self.original_goals,self.hold_start)
            levels=d['recovery_prefix_lesson_id']
            for lesson in range(3):
                for goal,axis in enumerate((0,1,1)):
                    direction=(self.original_goals[:,axis]>.001) if goal<2 else (self.original_goals[:,axis]<-.001)
                    selected=exposed & valid & (levels==lesson) & direction
                    self.exposure[0,lesson,goal]+=selected.sum()
                    self.exposure[1,lesson,goal]+=(selected & self.live).sum()
            self.cost_active=None

        def _restore(self):
            if self.teaching_cleanup_done:
                return super()._restore()
            errors=[]
            if self.teaching_step is not None:
                if self.env.step is self.teaching_step:self.env.step=self.teaching_prior_step
                elif self.env.step is not self.teaching_prior_step:errors.append('Teaching step ownership changed')
            if self.cost_hook is not None:
                if module.apply_recovery_hold_costs is self.cost_hook:module.apply_recovery_hold_costs=self.cost_prior
                elif module.apply_recovery_hold_costs is not self.cost_prior:errors.append('Hold cost hook ownership changed')
            for owner,name,installed,prior in reversed(self.structural_owners):
                actual=getattr(owner,name)
                if same_callback(installed,prior):
                    if not same_callback(actual,prior):errors.append('Formal probe hook ownership changed: '+name)
                elif same_callback(actual,installed):setattr(owner,name,prior)
                elif not same_callback(actual,prior):errors.append('Formal probe hook ownership changed: '+name)
            self.teaching_cleanup_done=True
            errors.extend(super()._restore())
            return errors

        def finish(self,output):
            try:
                return self._finish_teaching(output)
            except BaseException:
                # Payment finish may already have restored callbacks and marked
                # its close hook done. Persist our later failure before Kit can
                # exit0 and skip the outer Python exception handler.
                writer(output/'teaching_observer_failure.json',dict(status='teaching_finish_rejected_preclose',
                    complete=False,error=traceback.format_exc(),hold_producer_calls=self.cost_calls,
                    qualification=None,PPO_updates=0,app_close_returned=None))
                raise

        def _finish_teaching(self,output):
            require(self.cost_calls==600 and self.cost_active is None,'New hold producer coverage incomplete')
            posture=self.structural.finish_posture()
            structural=dict(native_scut_train_structure_checked=True,
                counts=dict(self.structural.counts),max_pin_gap_m=self.structural.max_gap,
                posture_native_readback=posture)
            original=super().finish(output)
            coverage=self.exposure.detach().cpu().tolist()
            teaching=dict(schema='native-source021-teaching-observation-v1',status='native_mechanisms_reconciled_preclose',
                complete=True,hold_producer_calls=self.cost_calls,policy_steps=600,physics_steps=2400,native_lane_intervals=2457600,
                native_hold_lane_intervals=self.total_hold,ordinary_unchanged_lane_intervals=self.total_ordinary,
                excluded_unchanged_lane_intervals=self.total_excluded,
                goal_exposure_clock='pre-autoreset actual 50Hz diagnostic command; whole-interval actor-owned',
                exposure_scope_order=['all_native_births','first_birth_only'],lesson_order=list(base.LESSONS[:3]),
                original_nonzero_goal_order=['forward_positive','yaw_positive','yaw_negative'],exposure_counts=coverage,
                L0_all_three_nonzero_followups_observed=all(x>0 for x in coverage[0][0]),
                goal_absence_is_coverage_gap_not_criterion_success=True,
                hold_mask_and_all_nonheld_components_exact=True,reward_catalog_order_preserved=True,
                positive_baselines_and_unrelated_components_preserved=True,
                native_once_only_events_reconciled=True,full_native_release_and_success_criteria_preserved=True,
                teaching_hooks_restored_preclose=True,qualification=None,PPO_updates=0)
            teaching['native_formal_probe_structural_audit']=structural
            writer(output/'teaching_summary.json',teaching)
            original.update(teaching_summary_sha256=digest(output/'teaching_summary.json'),native_teaching_complete=True)
            return original
    return TeachingObserver


def validate_terminal(report,payment,teaching):
    require(report['status']=='native600_completed_preclose' and report['native_teaching_complete'] is True,'Native terminal report incomplete')
    require(report['policy_steps_completed']==600 and report['physical_steps_completed']==2400,'Native complete step counts differ')
    require(payment['complete'] is True and payment['native_lane_intervals']==2457600 and payment['event_producer_calls']==600
        and payment['max_reconciliation_residual']==0,'Native payment reconciliation incomplete')
    require(teaching['complete'] is True and teaching['hold_producer_calls']==600
        and teaching['native_hold_lane_intervals']>0 and teaching['ordinary_unchanged_lane_intervals']>0
        and teaching['excluded_unchanged_lane_intervals']>0,'Native teaching mechanism coverage incomplete')
    require(teaching['native_formal_probe_structural_audit']['native_scut_train_structure_checked'] is True
        and teaching['native_formal_probe_structural_audit']['posture_native_readback']['actual_native_velocity_delta_observed'] is True,
        'Full train-role ordinary native engineering coverage incomplete')
    require(report['PPO_updates']==report['optimizer_steps']==0 and report['actor_tensors_and_saved_std_unchanged']
        and report['curriculum_clocks_unchanged'] and report['learning_transition_clock_unchanged'],'Native actor/clock ownership differs')
    require(report['payment_hooks_restored_before_app_close'] and teaching['teaching_hooks_restored_preclose'],'Native hooks not restored preclose')
    return dict(engineering_mechanisms_passed=True,
        actual_L0_followup_exposure_complete=teaching['L0_all_three_nonzero_followups_observed'],
        behavior_qualification=None,PPO_updates=0)


def review_output(output):
    """External CPU admission after the process closes; exit0 alone is irrelevant."""
    output=Path(output)
    require(not any((output/name).exists() for name in ('payment_observer_failure.json',
        'teaching_observer_failure.json','teaching_admission_failure.json','teaching_postclose_error.json')),
        'Native preclose failure sidecar present')
    report=json.loads((output/'report.json').read_text());identity=json.loads((output/'identity.json').read_text())
    payment=json.loads((output/'payment_summary.json').read_text());teaching=json.loads((output/'teaching_summary.json').read_text())
    require(digest(output/'identity.json')==report['identity_sha256']
        and digest(output/'payment_summary.json')==report['payment_summary_sha256']
        and digest(output/'teaching_summary.json')==report['teaching_summary_sha256'],'External terminal sidecar identity differs')
    require(report['source021_actual_observation_preclose'] is True and report['checkpoint_unchanged']
        and report['source_files_unchanged'] and report['native_effective_std_matches_saved'],'Actual native/source evidence incomplete')
    require(all(digest(path)==sha for path,sha in identity['observer_artifact_sha256'].items()),'External artifact bytes changed')
    require(payment['checkpoint_clocks']==identity['checkpoint_clocks']
        and payment['observation_actor_clocks']==identity['observation_actor_clocks']
        and payment['observation_context']==identity['observation_context'],'Actual parent and observation clocks conflated')
    admission=validate_terminal(report,payment,teaching)
    require(admission['actual_L0_followup_exposure_complete'] is True,
        'Mechanisms reconciled, but actual L0 forward and both yaw exposure remains a native coverage gap')
    require(identity['qualification'] is None and identity['PPO_updates']==0 and report['formal_updates_consumed']==0,
        'Observation incorrectly claims learning or qualification')
    return dict(schema='source021-native-teaching-independent-readback-v1',all_required_passed=True,
        native_counts={'lanes':4096,'policy_steps':600,'physics_steps':2400,'policy_lane_intervals':2457600},
        preclose_report_sha256=digest(output/'report.json'),identity_sha256=digest(output/'identity.json'),
        payment_summary_sha256=digest(output/'payment_summary.json'),teaching_summary_sha256=digest(output/'teaching_summary.json'),
        actual_goal_bridge_exposure_complete=True,observation_context=identity['observation_context'],
        qualification=None,PPO_updates=0,formal_updates_consumed=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--admission-sha',required=True)
    parser.add_argument('--arm',choices=('mean','stochastic'),required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args(argv)
    audit=Path(__file__).resolve().parent.parent
    base=load_bound(audit/'probe_prefix_mean_stochastic_019.py',BASE_SHA,'source021_bound_native55')
    helper=load_bound(audit/'observe_prefix_native_payments_019.py',PAYMENT_SHA,'source021_bound_payment51580')
    composition=load_bound(audit/'probe_prefix_mean_stochastic_payments_019.py',COMPOSITION_SHA,'source021_bound_composition')
    binding={str(p):digest(p) for p in [Path(__file__).resolve(),audit/'probe_prefix_mean_stochastic_019.py',
        audit/'observe_prefix_native_payments_019.py',audit/'probe_prefix_mean_stochastic_payments_019.py',args.admission.resolve()]}
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='source021-native-teaching-driver-v2',observer_revision=2,status='starting',arm=args.arm,qualification=None,
        acceptance='not_assessed',PPO_updates=0,optimizer_steps=0,optimizer_instances=0,formal_updates_consumed=0,
        policy_steps_requested=600,physics_steps_requested=2400,clock_advance=False,evaluation_projection=False,
        reward_controller_command_overrides=[],artifact_sha256=binding,started_at_utc=datetime.now(timezone.utc).isoformat())
    original_write=base.write_json
    try:
        a=read_admission(args.admission,args.admission_sha)
        config,manifest,control,saved,identity=authenticate(args,a,base)
        identity.update(admission_sha256=args.admission_sha,observer_artifact_sha256=binding,
            inherited_native_wrapper_sha256=BASE_SHA,actual_payment_helper_sha256=PAYMENT_SHA)
        original_write(args.output/'identity.json',identity);report['identity_sha256']=digest(args.output/'identity.json')
        report.update(observation_context=identity['observation_context'],
            forced_engineering_actor_update=identity['forced_engineering_actor_update'],
            saved_checkpoint_actor_clocks={k:identity['checkpoint_clocks'][k] for k in ('task_actor_update','stage_actor_update','global_actor_update')},
            observation_actor_clocks=identity['observation_actor_clocks'],formal_learning_parent_selected=False)
        def publish(path,value):
            try:
                require(all(digest(p)==sha for p,sha in binding.items()),'Bound observation artifacts changed')
                if Path(path).name=='report.json' and a['context']=='engineering_flat_prefix' and 'first_birth_protocol' in value:
                    value['first_birth_protocol']='constructor then native saved flat-parent RNG restoration, forced2294 engineering actor clock, and one native reset; not a recovery resume or learning progress'
                if Path(path).name=='report.json' and value.get('status')=='native600_completed_preclose':
                    require(not any((args.output/name).exists() for name in ('payment_observer_failure.json',
                        'teaching_observer_failure.json')),'Native observation cleanup failed')
                    payment=json.loads((args.output/'payment_summary.json').read_text());teaching=json.loads((args.output/'teaching_summary.json').read_text())
                    require(digest(args.output/'payment_summary.json')==value['payment_summary_sha256']
                        and digest(args.output/'teaching_summary.json')==value['teaching_summary_sha256'],'Terminal sidecar SHA differs')
                    value['teaching_admission']=validate_terminal(value,payment,teaching)
                    value['source021_actual_observation_preclose']=True
                return original_write(path,value)
            except BaseException:
                original_write(args.output/'teaching_admission_failure.json',dict(status='terminal_publication_rejected_preclose',
                    complete=False,error=traceback.format_exc(),qualification=None,PPO_updates=0,app_close_returned=None))
                raise
        base.write_json=publish
        base.NativeObserver=make_observer(base,helper,composition,args,identity,report,binding,original_write) if not args.preflight_only else base.NativeObserver
        if args.preflight_only:
            require(base.verify_inputs(args,identity) and base.snapshot_sources(args.source_root,config,identity['archive_source_files'])==identity['source_files_sha256'],'Immutable CPU admission changed')
            report.update(status='CPU_preflight_passed_scene_not_run',native_scene_run=False,policy_steps_completed=0,physical_steps_completed=0)
            publish(args.output/'report.json',report)
        else:
            publish(args.output/'report.json',report)
            base.run_native(args,config,manifest,control,saved,identity,report)
        return 0
    except BaseException:
        if report.get('status')!='native600_completed_preclose':
            report.update(status='failed',error=traceback.format_exc());original_write(args.output/'report.json',report)
        else:
            original_write(args.output/'teaching_postclose_error.json',dict(source_admission=False,error=traceback.format_exc()))
        return 1


if __name__=='__main__':raise SystemExit(main())
