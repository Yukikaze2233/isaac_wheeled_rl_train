#!/usr/bin/env python3
"""Reuse source022/55 native loop; one CAPTURE4->1 config, CPU default, zero PPO."""
from pathlib import Path
import argparse
import inspect
import sys
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'source_script_repair_native_candidate_022'
BASE_SHA = '40554571a0b3656aaed22a14e862c726704c51d550a1b07602d1d6d478de7ace'
BASELINE = BASE/'native_SOURCE_SCRIPT_001/native'


def bound_source():
    import importlib.util
    path=BASE/'run_source_script_native_022.py'
    spec=importlib.util.spec_from_file_location('capture_rate_frozen_source022',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_candidate(BASE_SHA)
    return module


def verify(expected):
    source=bound_source()
    source.require(source.sha(HERE/'candidate_manifest.json')==expected,'CAPTURE candidate manifest SHA differs')
    m=source.read(HERE/'candidate_manifest.json')
    source.require(m['schema']=='source022-CAPTURE-speed-diagnostic-candidate-v1','CAPTURE schema differs')
    for name,digest in m['files_sha256'].items():
        path=Path(name)
        source.require(not path.is_absolute() and '..' not in path.parts and source.sha(HERE/path)==digest,'Candidate file changed: '+name)
    for path,digest in m['external_sha256'].items():
        source.require(Path(path).is_absolute() and source.sha(path)==digest,'Bound evidence changed: '+path)
    return source,m


def authenticate_fsm(source,fsm,recovery,root):
    # This runs only after original native env/SourceObserver construction.
    name='wheeled_tasks.chassis.recovery';mod=sys.modules.get(name)
    source.require(mod is not None and type(fsm).__module__==name and mod.RecoveryFSM is type(fsm)
        and Path(inspect.getsourcefile(type(fsm))).resolve()==Path(root)/'src/wheeled_tasks/chassis/recovery.py'
        and source.sha(inspect.getsourcefile(type(fsm)))==source.RECOVERY_SHA,'Actual frozen native FSM differs')
    expected=fsm.DEFAULTS | {k:v for k,v in recovery.settings.items() if k in fsm.DEFAULTS}
    expected['dt']=recovery.reference_dt
    source.require(fsm.count==4096 and fsm.config==expected and fsm.config['capture_speed_rad_s']==4.,
        'Native instance config differs before diagnostic override')


def make_observer(parent,source,capture,source_helper,rate,writer,root):
    inherited=source.make_source_observer(parent,capture,source_helper,writer,root)
    class WithCaptureSpeed(inherited):
        def __init__(self,env,output):
            self.capture_override=None
            super().__init__(env,output)
            try:
                mechanical=source.load(source.MECHANICAL,'capture_rate_owned_hooks')
                self.capture_override=rate.CaptureSpeedOverride(env,mechanical.HookSet(),output,writer,mechanical.plain,
                    lambda fsm:authenticate_fsm(source,fsm,env.recovery,root))
                self.capture_override.apply()
            except BaseException:
                self._before_close();raise

        def _before_close(self):
            try:
                if self.capture_override is not None:
                    source.require(not self.capture_override.cleanup(),'CAPTURE cleanup ownership rejected')
            finally:
                super()._before_close()

        def finish(self,output):
            try:
                source.require(not self.capture_override.finish(),'CAPTURE preclose restoration rejected')
                # Remove outer update/step before all frozen inner owners restore.
                result=super().finish(output)
            except BaseException:
                if not (output/'capture_speed_observer_failure.json').exists():
                    writer(output/'capture_speed_observer_failure.json',dict(error=traceback.format_exc(),
                        status='capture_speed_failed_preclose',PPO_updates=0,qualification=None,app_close_returned=None))
                self._before_close();raise
            result.update(capture_speed_override_complete=True,
                capture_speed_override_summary_sha256=source.sha(output/'capture_speed_override_preclose.json'),
                effective_diagnostic_capture_speed_rad_s=1.,
                diagnostic_configuration_override=True,
                inherited_source_observer_control_flags_describe_only_its_own_hooks=True,
                native_startup_preflight_precedes_explicit_diagnostic_config_override=True)
            return result
    return WithCaptureSpeed


def review(args,source,m):
    output=args.output.resolve();native=output/'native'
    process,plan=source.read(args.process_receipt),source.read(output/'run_plan.json')
    source.require(process['process_exit_observed'] is True and process['process_exit_code']==0
        and process['process_present_after_wait'] is False and process['candidate_manifest_sha256']==args.candidate_sha
        and process['command']==plan['native_command'] and process['PPO_updates']==process['formal_updates_consumed']==0,
        'Actual completed exclusive diagnostic process/argv required')
    report,baseline=source.read(native/'report.json'),source.read(BASELINE/'report.json')
    teaching=source.load(source.TEACHING,'capture_rate_terminal_readback')
    integrity=teaching.validate_terminal(report,source.read(native/'payment_summary.json'),source.read(native/'teaching_summary.json'))
    source.require(source.read(native/'identity.json')==source.read(BASELINE/'identity.json'),'Original source/seal/actor/RNG identity differs')
    for key in ('initial_physical_birth_tensor_sha256','initial_physical_birth_sha256','initial_env_rng_sha256',
        'native_group_population','sampling_rng_initial_sha256','sampling_rng_final_sha256',
        'first_native_sampling','final_native_sampling','native_effective_std'):
        source.require(report[key]==baseline[key],'Saved initial birth/private sampler/std differs: '+key)
    source.require(report['checkpoint_unchanged'] and report['source_files_unchanged'],'Source/parent modified')
    override=source.read(native/'capture_speed_override_preclose.json')
    source.require(override['complete'] and override['config_restored'] and override['hooks_restored']
        and not override['errors'] and override['sole_changed_keys']==['capture_speed_rad_s']
        and override['before'] | {'capture_speed_rad_s':1.}==override['effective']
        and override['policy_calls_attempted']==override['policy_calls_completed']==600
        and override['reference_calls_attempted']==override['reference_calls_completed']==2400
        and report['capture_speed_override_complete'],'Single config/restoration/call clocks incomplete')
    parent=source.read(native/'source_script_observer_summary.json')
    mechanical=source.read(native/'mechanical_summary.json');layout=source.read(native/'mechanical_layout.json')
    source.require(parent['complete'] and parent['callbacks_restored_preclose']
        and mechanical['complete'] and mechanical['hooks_restored']
        and mechanical['policy_steps']==600 and mechanical['physics_steps']==mechanical['trace_cutoff_physics_steps']==2400
        and not mechanical['contact_count_at_capacity_indicators']
        and layout['lanes']==[881,1708,1877,2121,2216] and layout['solver']==[64,32] and layout['guard_m']==.003,
        'Original five-lane/full-contact/native coverage incomplete')
    for lane in layout['lanes']:
        target=parent['targets'][str(lane)]
        source.require(target['complete'] and target['hooks_restored']
            and target['extra_policy_decode_delay_FSM_reward_calls']==0
            and target['original_calls_completed']==dict(decode=600,delay=2400,pd=2400,fsm=2400,cost=600),
            'Original once-called target/payment observer incomplete')
    source.require(not any(native.rglob('*failure.json')),'Native failure sidecar present')
    traces=source.decoded(native/'mechanical_5ms.jsonl.gz')
    helper=source.load(BASE/'source_script_capture.py','capture_rate_contact_readback')
    cutoff=override['strict_preintervention_last_physical_step']
    source.require(type(cutoff) is int and 0<=cutoff<2400,'Global actual CAPTURE-sensitive cutoff absent')
    pre=source.compare_trace(BASELINE/'mechanical_5ms.jsonl.gz',traces,helper,tuple(layout['lanes']),cutoff)
    source.write(output/'pre_capture_global_contact_multiset_review.json',pre)
    source.require(pre['complete_record_comparison_passed'],'Global pre-CAPTURE prefix differs; preserve and reject integrity')
    argmax=source.decoded(native/'mechanical_all4096_argmax_5ms.jsonl.gz')
    source.require([r['physical_step'] for r in argmax]==list(range(1,2401)),'Whole native argmax coverage missing')
    inputs=[Path(args.process_receipt),output/'run_plan.json',HERE/'candidate_manifest.json',
        BASE/'native_SOURCE_SCRIPT_001/source_script_native_review.json',BASELINE/'mechanical_5ms.jsonl.gz',
        output/'pre_capture_global_contact_multiset_review.json',*native.rglob('*')]
    result=dict(schema='source022-CAPTURE-speed-native-diagnostic-readback-v1',
        status='single_capture_rate_native_integrity_complete_not_functionally_qualified',
        native_integrity=integrity,source_commit=m['source_commit'],historical_parent_actor=3076,
        birth_private_sampling_saved_std_exact=True,override=override,global_preintervention=pre,
        mechanical_capture=mechanical,actual_targets=parent['targets'],all4096_argmax_rows=2400,
        changed_trajectory_after_global_first_sensitive_is_descriptive=True,
        inherited_source_observer_control_flags_describe_only_its_own_hooks=True,
        effective_configuration_override_is_explicit=True,PPO_updates=0,formal_updates_consumed=0,qualification=None,
        App_close_returned=None,inputs_sha256={str(p):source.sha(p) for p in inputs if p.is_file()},
        limitations=['CAPTURE speed1 may delay failure or increase original8s timeout; preserve negative outcomes.',
            'ORBIT4 and all original gates/PD/assets/rewards/births remain; no claim all3 lanes are repaired.',
            'Firstbirth script survival is not RL handover/1s hold/release/followup/success.',
            'Own-run payment/event reconciliation is required; global cross-run F64 reductions are not trajectory identity.',
            'Contact record order alone is ignored within exact body/filter/tick; original ordered refusals remain historical.',
            'No actual massCOMfriction/self-contact pair/native overflow flag; no localization of old730mm stochastic peak.'])
    source.write(output/'capture_speed_review.json',result)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('cpu','native','review'),default='cpu')
    p.add_argument('--candidate-sha',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--process-receipt',type=Path);args=p.parse_args(argv)
    source,m=verify(args.candidate_sha);output=args.output.resolve()
    source.require(HERE in output.parents,'New candidate-only output required')
    if args.mode=='review':
        source.require(args.process_receipt is not None and output.is_dir(),'Completed output/process required')
        try:review(args,source,m)
        except BaseException:
            source.write(output/'capture_speed_rejection.json',dict(status='preserved_actual_rate_diagnostic_rejected',
                error=traceback.format_exc(),PPO_updates=0,qualification=None));return 2
        return 0
    source.require(args.process_receipt is None and not output.exists(),'New output only; no blind rerun')
    output.mkdir(parents=True)
    base_manifest=source.verify_candidate(BASE_SHA)
    rate=source.load(HERE/'capture_speed_override.py','capture_rate_owned_override')
    capture=source.load(source.CAPTURE,'capture_rate_frozen_capture')
    source_helper=source.load(BASE/'source_script_capture.py','capture_rate_frozen_source_helper')
    teaching=source.load(source.TEACHING,'capture_rate_private_teaching')
    old=source.read(source.PAIR/'admission.json')
    admission=old | dict(schema='source022-native-source-script-admission-v1',source_commit=base_manifest['source_commit'],
        source_root=base_manifest['source_root'],source_manifest=str(BASE/'source_inventory.json'),
        source_manifest_sha256=base_manifest['source_inventory_sha256'],
        contract_relative='contracts/v6_task_campaign_recovery_worker_v20.json',contract_sha256=source.WORKER_SHA)
    source.write(output/'admission.json',admission)
    resources=None
    if args.mode=='native':
        pair=source.load(source.EXPLORATION/'run_sealed_REC3076_pair_021.py','capture_rate_resource_check')
        resources=pair.local_GPU_empty()
    command=[source.PYTHON,'-B',str(Path(__file__).resolve()),'--mode','native','--candidate-sha',args.candidate_sha,'--output',str(output)]
    source.write(output/'run_plan.json',dict(schema='source022-CAPTURE-speed-run-plan-v1',mode=args.mode,
        native_command=command,candidate_manifest_sha256=args.candidate_sha,source_commit=base_manifest['source_commit'],
        checkpoint_sha256=source.PT_SHA,source_manifest_sha256=source.sha(BASE/'source_inventory.json'),
        inherited_source022_manifest_sha256=BASE_SHA,resource_check=resources,
        sole_configuration_override={'RecoveryFSM.config.capture_speed_rad_s':{'from':4.,'to':1.}},
        original_source_code_unchanged=True,old_v20_design_preflight_before_explicit_diagnostic_override=True,
        lanes=4096,policy_steps=600,physical_steps=2400,selected_lanes=list(source_helper.LANES),arm='mean',
        PPO_updates=0,formal_updates_consumed=0,qualification=None))
    def admitted(path,digest):
        source.require(source.sha(path)==digest and source.read(path)==admission,'Parent/source admission changed')
        return admission
    teaching.read_admission=admitted
    def auth(inner,a,base):
        inner.observer_candidate_sha=BASE_SHA
        return source.authenticate(inner,a,base,teaching)
    teaching.authenticate=auth
    original_factory=teaching.make_observer
    if args.mode=='native':
        def factory(base,producer,composition,inner,identity,report,binding,writer):
            parent=original_factory(base,producer,composition,inner,identity,report,binding,writer)
            return make_observer(parent,source,capture,source_helper,rate,writer,base_manifest['source_root'])
        teaching.make_observer=factory
    inherited=['--admission',str(output/'admission.json'),'--admission-sha',source.sha(output/'admission.json'),
        '--arm','mean','--output',str(output/'native')]
    if args.mode=='cpu':inherited.append('--preflight-only')
    code=teaching.main(inherited);verify(args.candidate_sha)
    if code:return code
    if args.mode=='cpu':
        report=source.read(output/'native/report.json')
        source.require(report['status']=='CPU_preflight_passed_scene_not_run' and not report['native_scene_run'],
            'Original App-free source022 preflight failed')
        source.write(output/'CPU_admission.json',dict(status='source022_REC3076_authenticated_CAPTURE_override_not_applied_no_scene',
            identity_sha256=source.sha(output/'native/identity.json'),candidate_manifest_sha256=args.candidate_sha,
            native_scene_run=False,configuration_override_applied=False,PPO_updates=0,qualification=None))
    # Native close can exit Python. Root owns exit observation and explicit review.
    return 0


if __name__=='__main__':
    raise SystemExit(main())
