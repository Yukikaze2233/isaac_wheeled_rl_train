#!/usr/bin/env python3
"""Candidate-only composition of the immutable native600 mean driver.

Default CPU mode authenticates the real REC3076/source021 admission without App.
Native mode is explicit and checks local GPU0 ownership before the original App.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PAIR = PARENT/'REC3076_pair_001'
ORCHESTRATOR = PARENT/'run_sealed_REC3076_pair_021.py'
ORCHESTRATOR_SHA = '5d992e8f0945be677a2dbe82a8aaccb27d9af3f96542bf2d17995e1118bda801'
BASELINE_SHA = '5752df3fde8ad22fab3fe795371814cf4541acf99d0531bd3397d9eb34259dbc'
CONTACT_API = Path('/home/yukikaze/isaacsim60-venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.1.11+110.1.1.lx64.r.cp312.u7f4/omni/physics/tensors/api.py')
CONTACT_API_SHA = 'b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903'


def require(value, message):
    if not bool(value): raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_candidate(expected_sha):
    manifest=HERE/'candidate_manifest.json'
    require(sha(manifest)==expected_sha,'Candidate manifest SHA differs')
    value=read(manifest)
    require(value['schema']=='source021-mechanical-capture-artifact-candidate-v1','Candidate schema differs')
    for relative,digest in value['files_sha256'].items():
        path=Path(relative)
        require(not path.is_absolute() and '..' not in path.parts,'Candidate path escape')
        require(sha(HERE/path)==digest,'Candidate bytes changed: '+relative)
    for path,digest in value['external_sha256'].items():
        require(Path(path).is_absolute() and sha(path)==digest,'Bound external evidence changed: '+path)
    require(sha(ORCHESTRATOR)==ORCHESTRATOR_SHA,'Original pair driver changed')
    require(sha(PAIR/'mean/report.json')==BASELINE_SHA,'Actual original mean baseline changed')
    require(sha(CONTACT_API)==CONTACT_API_SHA,'Installed contact API changed')
    return value


def make_capture_observer(parent, env_module, helper, writer):
    class WithMechanicalCapture(parent):
        def __init__(self,env,output):
            self.mechanical=None
            super().__init__(env,output)
            try:
                import warp as wp
                self.mechanical=helper.MechanicalCapture(env,env_module,output,writer,wp.to_torch)
                self.mechanical.install()
            except BaseException:
                self._before_close();raise

        def _before_close(self):
            try:
                if self.mechanical is not None: self.mechanical.cleanup()
            finally: super()._before_close()

        def finish(self,output):
            try:
                capture=self.mechanical.finish()
                writer(output/'mechanical_summary.json',capture)
                result=super().finish(output)
            except BaseException:
                self._before_close();raise
            result.update(mechanical_summary_sha256=sha(output/'mechanical_summary.json'),
                mechanical_capture_complete=capture['complete'],mechanical_hooks_restored=True)
            return result
    return WithMechanicalCapture


def selected_first_births(path):
    with gzip.open(path,'rt') as stream:
        return {row['lane']:row for line in stream if (row:=json.loads(line))['lane'] in (1708,2216)}


def review_capture(output, pair_module):
    native=output/'native'
    integrity=pair_module.diagnostic_review(pair_module.bound_driver(),native)
    report=read(native/'report.json');baseline=read(PAIR/'mean/report.json')
    for key in ('initial_physical_birth_tensor_sha256','initial_physical_birth_sha256','initial_env_rng_sha256',
            'native_group_population','sampling_rng_initial_sha256','sampling_rng_final_sha256',
            'first_native_sampling','final_native_sampling','native_effective_std'):
        require(report[key]==baseline[key],'Observed baseline identity/birth/private RNG differs: '+key)
    capture=read(native/'mechanical_summary.json')
    require(not (native/'mechanical_capture_failure.json').exists(),'Mechanical capture failure sidecar exists')
    require(capture['complete'] and capture['physics_steps']==2400 and capture['policy_steps']==600
        and capture['hooks_restored'] and report['mechanical_capture_complete'],'Native mechanical capture incomplete')
    require(not capture['contact_count_at_capacity_indicators'],'Potential contact data capacity exhaustion; evidence scope incomplete')
    require(selected_first_births(native/'first_birth_rows.jsonl.gz')==selected_first_births(PAIR/'mean/first_birth_rows.jsonl.gz'),
        'Selected first guard signature changed; preserve trace but no reproduction admission')
    return dict(schema='source021-native-mechanical-capture-readback-v1',status='baseline_first_guards_reproduced_with_substep_capture',
        native_integrity=integrity,same_REC3076_birth_and_private_sampling=True,
        selected_first_guard_signatures_exact=True,capture=capture,
        files_sha256={str(p.relative_to(output)):sha(p) for p in native.glob('mechanical*')},
        PPO_updates=0,formal_updates_consumed=0,qualification=None,
        result_scope='Mechanical evidence only; neither solver nor self-righting functional qualification')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('cpu','native'),default='cpu')
    parser.add_argument('--candidate-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    require(HERE in output.parents and not output.exists(),'New candidate-only output directory required')
    candidate=verify_candidate(args.candidate_sha)
    pair=load(ORCHESTRATOR,'mechanical_frozen_pair')
    plan,admission=pair.load_plan(PAIR)
    require(read(PAIR/'CPU_preflight.json')['invocation_sha256']==sha(PAIR/'invocation.json'),'Original CPU identity/preflight absent')
    output.mkdir(parents=True)
    cpu_only=args.mode=='cpu'
    resources=None if cpu_only else pair.local_GPU_empty()
    run_plan=dict(schema='source021-mechanical-capture-run-plan-v1',mode=args.mode,
        candidate_sha256=args.candidate_sha,candidate_files=candidate['files_sha256'],
        inherited_pair_invocation_sha256=sha(PAIR/'invocation.json'),inherited_admission_sha256=plan['admission_sha256'],
        actual_REC3076_checkpoint_sha256=plan['checkpoint_sha256'],source_commit=plan['source_commit'],
        baseline_mean_report_sha256=BASELINE_SHA,contact_API_sha256=CONTACT_API_SHA,
        runtime_requested=dict(lanes=4096,policy_steps=600,physics_steps=2400,policy_dt=.02,physics_dt=.005,
            solver=[64,32],guard_m=.003,selected_lanes=[1708,2216],dense_first_physics_steps=260,arm='mean'),
        resource_check=resources,PPO_updates=0,formal_updates_consumed=0,qualification=None)
    write(output/'run_plan.json',run_plan)
    driver=pair.bound_driver()
    original_factory=driver.make_observer
    if not cpu_only:
        helper=load(HERE/'mechanical_capture.py','mechanical_capture021')
        def factory(base,producer,composition,native_args,identity,report,binding,writer):
            parent=original_factory(base,producer,composition,native_args,identity,report,binding,writer)
            from wheeled_tasks.chassis import env as env_module
            return make_capture_observer(parent,env_module,helper,writer)
        driver.make_observer=factory
    native_args=['--admission',str(PAIR/'admission.json'),'--admission-sha',plan['admission_sha256'],
        '--arm','mean','--output',str(output/'native')]
    if cpu_only: native_args.append('--preflight-only')
    code=driver.main(native_args)
    verify_candidate(args.candidate_sha)
    if code!=0:
        write(output/'capture_rejection.json',dict(status='inherited_native_driver_rejected',exit_code=code,qualification=None))
        return code
    if cpu_only:
        report=read(output/'native/report.json')
        require(report['status']=='CPU_preflight_passed_scene_not_run' and not report['native_scene_run'],'App-free preflight failed')
        write(output/'CPU_admission.json',dict(status='source_parent_candidate_CPU_authenticated_native_not_run',
            run_plan_sha256=sha(output/'run_plan.json'),identity_sha256=sha(output/'native/identity.json'),
            candidate_sha256=args.candidate_sha,native_scene_run=False,PPO_updates=0,qualification=None))
    else:
        try: write(output/'capture_review.json',review_capture(output,pair))
        except BaseException:
            write(output/'capture_rejection.json',dict(status='capture_or_baseline_readback_rejected',error=traceback.format_exc(),
                preserved_native_trace=True,qualification=None,PPO_updates=0));return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
