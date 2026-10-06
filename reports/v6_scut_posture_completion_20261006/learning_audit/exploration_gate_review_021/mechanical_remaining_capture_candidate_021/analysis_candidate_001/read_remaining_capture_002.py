"""Describe authentic mechanical data while preserving rejected whole reproduction."""
from __future__ import annotations
import argparse
import gzip
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CORE_SHA='6b2b3ba57c52e92f976e4a7afc54215430fb9ba778aa18fda645cde36e205bef'

def core():
    spec=importlib.util.spec_from_file_location('frozen_remaining_descriptive_core_001',HERE/'read_remaining_capture_001.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a=module.original_analysis()
    a.require(a.sha(HERE/'read_remaining_capture_001.py')==CORE_SHA,'Frozen descriptive core changed')
    return module,a

def require_admission_flags(admission, process):
    if not (admission['schema']=='source021-remaining-mechanical-analysis-admission-v1'
        and admission['status']=='completed_native_mechanical_data_admitted_for_descriptive_readonly_analysis'
        and admission['strict_whole_reproduction'] is False and admission['payment_aggregate_float_bits_exact'] is False
        and admission['qualification'] is None and admission['PPO_updates']==admission['formal_updates_consumed']==0
        and admission['actual_policy_steps']==600 and admission['actual_physics_steps']==2400
        and admission['selected_lanes']==[881,1877,2121] and admission['selected_lanes_5ms_complete'] is True
        and admission['full4096_firstbirth_events_global_record_states_float_bits_exact'] is True
        and admission['identity_teaching_override_float_bits_exact'] is True
        and admission['original_control_source_RNG_birth_clocks_report_values_exact'] is True
        and process['process_exit_observed'] is True and process['process_exit_code']==0
        and process['process_present_after_wait'] is False):
        raise ValueError('Mechanical-only admission absent or strict rejection/actual exit scope misrepresented')

def main(argv=None):
    c,a=core()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root',type=Path,required=True)
    parser.add_argument('--process-receipt',type=Path,required=True)
    parser.add_argument('--analysis-admission',type=Path,required=True)
    parser.add_argument('--admission-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);capture=args.capture_root.resolve();output=args.output.resolve()
    a.require(c.CANDIDATE in capture.parents and HERE in output.parents and not output.exists(),'New artifact-only analysis output required')
    a.require(a.sha(args.analysis_admission)==args.admission_sha,'Explicit readonly mechanical admission SHA differs')
    admission=a.read(args.analysis_admission);process=a.read(args.process_receipt)
    require_admission_flags(admission,process)
    inputs=dict(admission['actual_input_sha256'])
    inputs.update({str(p.resolve()):a.sha(p) for p in (args.analysis_admission,Path(__file__),HERE/'read_remaining_capture_001.py')})
    for path,digest in inputs.items():a.require(a.sha(path)==digest,'Diagnosis-bound evidence changed: '+path)
    diagnosis=a.read(admission['diagnosis_path'])
    a.require(a.sha(admission['diagnosis_path'])==admission['diagnosis_sha256']
        and diagnosis['strict_whole_reproduction'] is False,'Preserved diagnosis identity differs')
    rejection=a.read(admission['preserved_strict_rejection_path'])
    a.require(rejection['status']=='preserved_actual_measurement_rejected','Original strict rejection missing')
    identity=a.read(capture/'native/identity.json');report=a.read(capture/'native/report.json')
    a.require(identity['source_commit_label']=='d0dce6181df83b03fe166abcbadef3c81de2e57e'
        and identity['checkpoint_clocks']['task_actor_update']==3076,'Source/ownREC3076 parent differs')
    checkpoint=Path(identity['checkpoint'])
    for path,key in ((checkpoint,'checkpoint_sha256'),(checkpoint.with_suffix('.contract.json'),'checkpoint_contract_sha256'),
        (checkpoint.with_suffix('.checkpoint.json'),'checkpoint_seal_sha256'),(Path(identity['source_manifest']),'source_manifest_sha256')):
        digest=a.sha(path);a.require(digest==identity[key],'Authenticated parent/source differs: '+key);inputs[str(path)]=digest
    for path,digest in identity['source_files_sha256'].items():
        a.require(a.sha(path)==digest,'Frozen source changed: '+path);inputs[path]=digest
    a.require(report['status']=='native600_completed_preclose' and report['PPO_updates']==report['formal_updates_consumed']==0
        and report['qualification'] is None,'Actual native context differs')
    observed=report['native_startup']['runtime_preflight']['observed']
    a.require(observed['modules']['usb_transport'] is False and observed['signal_delay']['feedback_pd']=='fresh_physics_feedback',
        'Rawfeedback is not actual fresh nativePD feedback')
    gains={k:observed[k] for k in ('leg_kp','leg_kd','wheel_velocity_p')}
    summary=a.read(capture/'native/mechanical_summary.json');layout=a.read(capture/'native/mechanical_layout.json')
    a.require(layout['lanes']==list(c.LANES) and layout['profiles']==list(c.PROFILES)
        and summary['policy_steps']==600 and summary['physics_steps']==2400 and summary['hooks_restored'],
        'Actual mechanical selection/clock differs')
    raw=a.rows(capture/'native/mechanical_5ms.jsonl.gz');a.finite_tree(raw)
    first={r['lane']:r for r in a.rows(capture/'native/first_birth_rows.jsonl.gz') if r['lane'] in c.LANES}
    lanes=[];chronology=[]
    for lane,level,profile,group,origin in zip(c.LANES,c.LEVELS,c.PROFILES,c.GROUPS,layout['selected_env_origins_world_m']):
        birth=first[lane]
        a.require(birth['recovery_prefix_lesson_id']==level and birth['sampled_profile']==profile
            and birth['group']==group and birth['termination_reasons']==['closure_gap'],'Distinct actual L1/L2 profile differs')
        entry,converted=c.describe_lane(raw,layout,summary,lane,origin,gains);lanes.append(entry);chronology.extend(converted)
    for path,digest in inputs.items():a.require(a.sha(path)==digest,'Input changed while reading')
    output.mkdir(parents=True);series=output/'chronological_readback.jsonl.gz'
    with gzip.open(series,'wt') as stream:
        for row in sorted(chronology,key=lambda r:(r['physical_step'],r['lane'])):
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    result=dict(schema='source021-remaining-three-lane-descriptive-readback-v2',
        status='completed_actual_mechanical_trace_under_preserved_strict_reproduction_rejection',
        actual_inputs_sha256=inputs,analysis_admission_sha256=args.admission_sha,
        strict_whole_reproduction=False,payment_aggregate_float_bits_exact=False,
        selection=list(c.LANES),lanes=lanes,chronological_series=dict(path=str(series),sha256=a.sha(series),rows=len(chronology)),
        independent_PD_winding_reconstruction=False,qualified=False,qualification=None,PPO_updates=0,formal_updates_consumed=0,
        limitations=diagnosis['limitations']+['Native preclip commands are captured; continuousencoder winding remains unavailable in this trace.',
            'Netforce rows lack self-contact pair identity; terrainfiltered patch identity/overflow/randomizedmassCOMfriction limits remain.',
            'Postterminal trajectory is censored;6rad/s right-censored births are not per-substep safety evidence.'])
    a.write(output/'analysis.json',result)
    for path,digest in inputs.items():a.require(a.sha(path)==digest,'Input changed during publication')
    print(json.dumps(dict(status=result['status'],path=str(output/'analysis.json'),sha256=a.sha(output/'analysis.json'),
        qualification=None,strict_whole_reproduction=False)))

if __name__=='__main__':main()
