#!/usr/bin/env python3
"""One isolated source022 reference-end handoff, inherited native loop, zero PPO."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'exploration_gate_review_021/source_script_repair_native_candidate_022'
BASE_SHA = '40554571a0b3656aaed22a14e862c726704c51d550a1b07602d1d6d478de7ace'
CORE_SHA = '3a20f2bdcadbeabc64324d2345c8076b8f986dd2e7fcde4420a8633fdaf6394d'
RUN_ROOT = Path('/tmp/robot_rl_immediate_handoff_022')

def load(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def verify():
    source = load(BASE/'run_source_script_native_022.py','root_immediate_bound_source')
    base = source.verify_candidate(BASE_SHA)
    source.require(source.sha(HERE/'candidate_manifest.json') == CORE_SHA,'Core manifest changed')
    core = source.read(HERE/'candidate_manifest.json')
    for name,digest in core['files_sha256'].items():
        source.require(source.sha(HERE/name)==digest,'Frozen core changed: '+name)
    for name,digest in core['external_sha256'].items():
        source.require(source.sha(name)==digest,'Frozen external core input changed: '+name)
    return source,base,core

def observer(parent,source,capture,helper,hook,writer,root):
    inherited = source.make_source_observer(parent,capture,helper,writer,root)
    class ImmediateObserver(inherited):
        def __init__(self,env,output):
            self.immediate = None
            super().__init__(env,output)
            try:
                mechanical = source.load(source.MECHANICAL,'immediate_private_hook_owners')
                def first_birth_cohort():
                    # Original target recorders stop observing after their
                    # first terminal. Never label a later reset as birth1.
                    return [(lane,1) for lane,entry in zip(helper.LANES,self.target_entries)
                            if entry.alive]
                self.immediate = hook.ImmediateHandoffHook(env,self,mechanical.HookSet(),output,
                    writer,mechanical.plain,arm='immediate',cohort=first_birth_cohort,expected_calls=2400)
                self.immediate.apply()
            except BaseException:
                self._before_close(); raise

        def _before_close(self):
            try:
                if self.immediate is not None:
                    source.require(not self.immediate.cleanup(),'Immediate cleanup ownership failed')
            finally:
                super()._before_close()

        def finish(self,output):
            try:
                source.require(not self.immediate.finish(),'Immediate preclose restoration failed')
                result=super().finish(output)
            except BaseException:
                if not (output/'immediate_observer_failure.json').exists():
                    writer(output/'immediate_observer_failure.json',{'error':traceback.format_exc(),
                        'PPO_updates':0,'qualification':None})
                self._before_close(); raise
            result.update(immediate_reference_end_override_complete=True,
                explicit_runtime_control_override=True,
                immediate_core_manifest_sha256=CORE_SHA,
                immediate_hook_summary_sha256=source.sha(output/'immediate_handoff_hook_preclose.json'),
                inherited_control_hook_flags_scope='inherited source observer only; immediate override is explicit',
                historical_same_baseline_actor=3076, qualification=None)
            return result
    return ImmediateObserver

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['cpu','native'],default='cpu')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review',type=Path)
    p.add_argument('--review-sha256')
    a=p.parse_args(); source,base,core=verify()
    out=a.output.resolve()
    source.require(out.parent==RUN_ROOT and not out.exists(),'Exclusive diagnostic run path required')
    source.require(out.name==('cpu001' if a.mode=='cpu' else 'native001'),'Diagnostic run name differs')
    if a.mode=='native':
        source.require(a.review is not None and source.sha(a.review)==a.review_sha256,'Actual independent review required')
        review=source.read(a.review)
        source.require(review['root_driver_sha256']==source.sha(__file__) and review['native_execution_admitted'] is True,
            'Independent driver admission absent')
        pair=source.load(source.EXPLORATION/'run_sealed_REC3076_pair_021.py','immediate_empty_GPU')
        resources=pair.local_GPU_empty()
    else:
        resources=None
    import shutil
    source.require(shutil.disk_usage(RUN_ROOT.parent).free>=1024**3,'Diagnostic output needs at least1GiB')
    out.mkdir(parents=True)
    teaching=source.load(source.TEACHING,'immediate_existing_teaching')
    capture=source.load(source.CAPTURE,'immediate_existing_capture')
    helper=source.load(BASE/'source_script_capture.py','immediate_existing_source_helper')
    hook=source.load(HERE/'immediate_handoff_hook.py','immediate_frozen_core')
    old=source.read(source.PAIR/'admission.json')
    admission=old | dict(schema='source022-native-source-script-admission-v1',source_commit=base['source_commit'],
        source_root=base['source_root'],source_manifest=str(BASE/'source_inventory.json'),
        source_manifest_sha256=base['source_inventory_sha256'],
        contract_relative='contracts/v6_task_campaign_recovery_worker_v20.json',contract_sha256=source.WORKER_SHA)
    source.write(out/'admission.json',admission)
    source.write(out/'run_plan.json',dict(mode=a.mode,source_commit=base['source_commit'],
        root_driver_sha256=source.sha(__file__),core_manifest_sha256=CORE_SHA,
        actual_argv=list(sys.argv),
        checkpoint_sha256=source.PT_SHA,historical_actor=3076,arm='mean',
        envs=4096,policy_steps=600,physical_steps=2400,resources=resources,
        explicit_runtime_control_override=a.mode=='native',PPO_updates=0,qualification=None))
    def admitted(path,digest):
        source.require(source.sha(path)==digest and source.read(path)==admission,'Admission changed')
        return admission
    teaching.read_admission=admitted
    def auth(inner,admitted_value,_base):
        inner.observer_candidate_sha=BASE_SHA
        return source.authenticate(inner,admitted_value,_base,teaching)
    teaching.authenticate=auth
    original_factory=teaching.make_observer
    if a.mode=='native':
        def factory(_base,producer,composition,inner,identity,report,binding,writer):
            parent=original_factory(_base,producer,composition,inner,identity,report,binding,writer)
            return observer(parent,source,capture,helper,hook,writer,base['source_root'])
        teaching.make_observer=factory
    inherited=['--admission',str(out/'admission.json'),'--admission-sha',source.sha(out/'admission.json'),
        '--arm','mean','--output',str(out/'native')]
    if a.mode=='cpu': inherited.append('--preflight-only')
    code=teaching.main(inherited); verify()
    return code

if __name__=='__main__':
    raise SystemExit(main())
