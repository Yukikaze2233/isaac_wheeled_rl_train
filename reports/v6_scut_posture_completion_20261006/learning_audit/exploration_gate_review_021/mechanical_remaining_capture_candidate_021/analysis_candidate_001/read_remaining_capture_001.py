#!/usr/bin/env python3
"""Read completed three-lane native data; no App, controller or simulation."""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parent
ORIGINAL=CANDIDATE.parent/'mechanical_capture_candidate_021'
ANALYSIS_SHA='5e8ac734d72e788bfa54d41c56000de4d4185cb725581ed292c402eec2c6c8a4'
CANDIDATE_SHA='af75f849a42e7447b35cbdb7c5d21628c27ae04a5b976a7c5d6e2897cbb698a2'
LANES=(881,1877,2121)
LEVELS=(2,2,1)
PROFILES=('front_down_90','step200_lip_near_shell','step200_upright_top')
GROUPS=('recovery_front_down_90_forward','recovery_step200_lip_near_shell_stand','recovery_step200_prone_belly_top_forward')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def original_analysis():
    module=load(ORIGINAL/'analysis_candidate_001.py','remaining_frozen003_analysis')
    module.require(module.sha(ORIGINAL/'analysis_candidate_001.py')==ANALYSIS_SHA,'Frozen003 analysis changed')
    module.LANES=LANES  # Private loaded reader instance only; frozen bytes unchanged.
    return module


def validate_clocks(selected):
    for ordinal,row in enumerate(selected):
        if row['physical_step']!=ordinal or row['state']['birth_id']!=1 or row['simulated_s']!=ordinal*.005:
            raise ValueError('Firstbirth physical5ms continuity differs')
        if ordinal:
            if row['policy_tick']!=(ordinal-1)//4+1 or row['substep']!=(ordinal-1)%4+1:
                raise ValueError('Policy50Hz/physical200Hz interval mapping differs')


def pin_norm_readback(state):
    if len(state['pin_endpoints_world_m'])!=len(state['pin_gaps_m']):
        raise ValueError('Native pin endpoint/count differs')
    result=[]
    for index,(pair,gap) in enumerate(zip(state['pin_endpoints_world_m'],state['pin_gaps_m'])):
        distance=math.dist(*pair)
        tolerance=8*2**-23*max(abs(gap),1e-6)
        if abs(distance-gap)>tolerance:raise ValueError('Captured endpoint norm differs from native pin gap')
        result.append(dict(pin_index=index,endpoint_distance_m_derived=distance,
            captured_native_gap_m=gap,absolute_roundoff_difference_m=abs(distance-gap),
            float32_norm_roundoff_tolerance_m=tolerance))
    if len(result)!=len(state['pin_gaps_m']):raise ValueError('Native pin endpoint/count differs')
    return result


def setter_readback(row,prior,layout,gains):
    control=row['applied_control']
    if control is None:return None
    names=layout['full_joint_names'];axes=layout['control_joint_names'];state=row['state']
    indices=[names.index(n) for n in axes]
    q,dq=control['feedback_full_q_prephysics'],control['feedback_full_dq_prephysics']
    if len(q)!=len(names) or len(dq)!=len(names):raise ValueError('Prephysics feedback/full axis count differs')
    if prior is not None:
        if q!=prior['state']['full_q'] or dq!=prior['state']['full_dq']:
            raise ValueError('Fresh physical feedback is not preceding postphysics sample')
    command=[]
    for axis,index in enumerate(indices):
        wheel=axis in (2,5);reference=control['script_targets_control6'][axis]
        previous=None if prior is None or prior['applied_control'] is None else prior['applied_control']['script_targets_control6'][axis]
        command.append(dict(control_axis=axis,joint_name=axes[axis],
            actual_fresh_raw_q_before_physics_rad=q[index],actual_fresh_dq_before_physics_rad_s=dq[index],
            native_script_reference=reference,reference_unit=('rad/s' if wheel else 'rad'),
            reference_rate_derived=(None if previous is None else (reference-previous)/.005),
            reference_rate_unit=('rad/s^2' if wheel else 'rad/s'),
            captured_native_preclip_command_nm=control['native_requested_motor_effort_nm_at_setter'][axis],
            preclip_command_scope=('native wheel PD plus support probe and release/blend gating' if wheel else
                'native continuous-encoder leg PD after release/blend gating; no independent winding reconstruction'),
            actual_native_instantaneous_symmetric_cap_nm=control['motor_bounds_nm'][axis],
            exact_setter_after_strength_transport_response_nm=control['full_joint_effort_target'][index],
            engine_applied_effort_after_physics_nm=state['applied_full_joint_effort'][index],
            native_motor_strength=control['motor_strength'][axis],
            native_gain=(dict(velocity_P=gains['wheel_velocity_p']) if wheel else dict(Kp=gains['leg_kp'],Kd=gains['leg_kd']))))
    return dict(axes=command,raw_feedback_scope='fresh physical feedback; authenticated usb_transport disabled',
        continuous_encoder_feedback_winding_directly_saved=False,
        PDcmd_source='captured native_requested_motor_effort_nm_at_setter; not a postphysics error reconstruction',
        native_setter_phase=control['phase'],native_setter_route=control['route'],
        side_reference_native_state=control['side_reference_audit'],
        release_finished=control['release_finished'],blend=control['blend'],
        original_transport_and_strength= {key:control[key] for key in ('motor_strength','spring_strength',
            'delay_lags_5ms','transport_delay_s','transport_age_s')})


def net_only_readback(row,layout):
    if row['contacts'] is None:return []
    return [dict(body_name=layout['body_names'][index],sensor_index=body['sensor_index'],
        net_force_all_contacts_without_pair_identity_n=body['net_force_all_contacts_n'])
        for index,body in enumerate(row['contacts']['sensor_body_rows'])
        if any(x!=0 for x in body['net_force_all_contacts_n'])]


def describe_lane(raw,layout,summary,lane,origin,gains):
    original=original_analysis()
    selected=[r for r in raw if r['state']['lane']==lane];validate_clocks(selected)
    entry,converted=original.lane_readback(raw,layout,summary,lane,origin)
    for index,(frame,row) in enumerate(zip(converted,selected)):
        frame['pin_endpoint_norm_check']=pin_norm_readback(row['state'])
        frame['actual_setter_command_feedback_caps']=setter_readback(row,None if index==0 else selected[index-1],layout,gains)
        frame['all_nonzero_unfiltered_net_contact_rows']=net_only_readback(row,layout)
    first=next((index for index,row in enumerate(converted) if row['pin_max_gap_m']>.003),None)
    if first is None:raise ValueError('Expected actual remaining closure lane has no captured3mm crossing')
    terminal=summary['native_first_terminals'][str(lane)]['policy_tick']
    if selected[-1]['physical_step']!=terminal*4:raise ValueError('Selected trace does not end at actual first terminal')
    entry.update(actual_lesson_level=LEVELS[LANES.index(lane)],actual_sampled_profile=PROFILES[LANES.index(lane)],
        behavior_group=GROUPS[LANES.index(lane)],
        first_guard_context_20ms_before_after=converted[max(0,first-4):min(len(converted),first+5)],
        actual_postguard_context_available_ms=(min(len(converted)-1,first+4)-first)*5,
        postguard20ms_censored_by_native_terminal=(len(converted)-1<first+4),
        first_native_terminal_last20ms=converted[max(0,len(converted)-5):],
        first_substep_crossing_need_not_be_terminal_crossing=True,
        native_terminal_policy_tick=terminal,
        source_equations_only=dict(leg='native Kp*(ref-continuous_encoder_q)-Kd*dq then release/clamp/blend',
            wheel='native velocity_P*(ref-dq)+probe then release/clamp/blend'),
        cause='Undetermined; actual pre/post/contact/reference series is descriptive evidence')
    entry.pop('crossing_context',None)
    return entry,converted


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root',type=Path,default=CANDIDATE/'native_remaining_001')
    parser.add_argument('--process-receipt',type=Path,required=True)
    parser.add_argument('--native-review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    capture=args.capture_root.resolve();output=args.output.resolve();a=original_analysis()
    a.require(CANDIDATE in capture.parents and HERE in output.parents and not output.exists(),'New candidate analysis output required')
    driver=load(CANDIDATE/'run_remaining_capture_021.py','remaining_frozen_driver_readonly')
    driver.verify_candidate(CANDIDATE_SHA)
    # No native execution: completed external exit and strict readback are prerequisites.
    process=a.read(args.process_receipt);review=a.read(args.native_review)
    a.require(process['process_exit_observed'] is True and process['process_exit_code']==0
        and process.get('process_present_after_wait') is False,'Actual exclusive diagnostic not exited successfully')
    a.require(review['status']=='same_SIDE_native_reproduced_measurement_complete_not_qualified'
        and review['original_control_source_RNG_births_clocks_guard_preserved'] is True,
        'Strict native SIDE reproduction review not complete')
    names=['run_plan.json','native/report.json','native/identity.json','native/mechanical_layout.json','native/mechanical_summary.json',
        'native/mechanical_5ms.jsonl.gz','native/mechanical_global_peaks.jsonl.gz','native/first_birth_rows.jsonl.gz',
        'native/events.jsonl.gz','native/payment_summary.json','native/teaching_summary.json']
    inputs={str(capture/n):a.sha(capture/n) for n in names}
    inputs.update({str(p.resolve()):a.sha(p) for p in (args.process_receipt,args.native_review,Path(__file__),
        CANDIDATE/'candidate_manifest.json',ORIGINAL/'analysis_candidate_001.py')})
    for path,digest in review['inputs_sha256'].items():a.require(a.sha(path)==digest,'Native readback-bound evidence changed')
    identity=a.read(capture/'native/identity.json');report=a.read(capture/'native/report.json')
    a.require(identity['source_commit_label']=='d0dce6181df83b03fe166abcbadef3c81de2e57e'
        and identity['checkpoint_clocks']['task_actor_update']==3076,'Source/ownREC3076 parent differs')
    checkpoint=Path(identity['checkpoint'])
    for path,key in ((checkpoint,'checkpoint_sha256'),(checkpoint.with_suffix('.contract.json'),'checkpoint_contract_sha256'),
        (checkpoint.with_suffix('.checkpoint.json'),'checkpoint_seal_sha256'),
        (Path(identity['source_manifest']),'source_manifest_sha256')):
        digest=a.sha(path);a.require(digest==identity[key],'Authenticated parent/source binding differs: '+key)
        inputs[str(path)]=digest
    for path,digest in identity['source_files_sha256'].items():
        a.require(a.sha(path)==digest,'Authenticated frozen source changed: '+path);inputs[path]=digest
    a.require(report['status']=='native600_completed_preclose' and report['PPO_updates']==0
        and report['formal_updates_consumed']==0 and report['qualification'] is None,'Native complete/zeroPPO scope differs')
    observed=report['native_startup']['runtime_preflight']['observed']
    a.require(observed['modules']['usb_transport'] is False and observed['signal_delay']['feedback_pd']=='fresh_physics_feedback',
        'Saved raw feedback is not certified fresh nativePD feedback')
    gains={key:observed[key] for key in ('leg_kp','leg_kd','wheel_velocity_p')}
    summary=a.read(capture/'native/mechanical_summary.json');layout=a.read(capture/'native/mechanical_layout.json')
    a.require(layout['lanes']==list(LANES) and layout['profiles']==list(PROFILES)
        and summary['policy_steps']==600 and summary['physics_steps']==2400 and summary['hooks_restored'],
        'Actual observed lane/profile/native clock differs')
    raw=a.rows(capture/'native/mechanical_5ms.jsonl.gz');a.finite_tree(raw)
    first={r['lane']:r for r in a.rows(capture/'native/first_birth_rows.jsonl.gz') if r['lane'] in LANES}
    lanes=[];chronology=[]
    for lane,level,profile,group,origin in zip(LANES,LEVELS,PROFILES,GROUPS,layout['selected_env_origins_world_m']):
        birth=first[lane]
        a.require(birth['recovery_prefix_lesson_id']==level and birth['sampled_profile']==profile
            and birth['group']==group and birth['termination_reasons']==['closure_gap'],
            'Actual distinct lesson/group/profile binding differs')
        entry,converted=describe_lane(raw,layout,summary,lane,origin,gains);lanes.append(entry);chronology.extend(converted)
    for path,digest in inputs.items():a.require(a.sha(path)==digest,'Input changed while reading')
    output.mkdir(parents=True);series=output/'chronological_readback.jsonl.gz'
    with gzip.open(series,'wt') as stream:
        for row in sorted(chronology,key=lambda r:(r['physical_step'],r['lane'])):
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    result=dict(schema='source021-remaining-three-lane-descriptive-readback-v1',status='completed_actual_trace_described_cause_undetermined',
        actual_inputs_sha256=inputs,selection=list(LANES),lanes=lanes,
        chronological_series=dict(path=str(series),sha256=a.sha(series),rows=len(chronology)),
        independent_PD_winding_reconstruction=False,qualified=False,qualification=None,PPO_updates=0,formal_updates_consumed=0,
        limitations=['Actual native preclip commands are recorded; postphysics residual is not its PD feedback error.',
            'World minus captured env origin is derived; all native coordinate values remain bound in rawtrace.',
            'Unfiltered net-only contact force has no pair identity; existing terrain patches do not disclose self-contact.',
            'Derived force*dt impulse is not independent impulse measurement; native overflow flag/randomized massCOMfriction remain unavailable.',
            'Substep firstguard and actual policy terminal differ; unavailable after-terminal context stays censored.',
            'Same6rad/s firstbirth right-censor is not evidence of per-substep safety;1.5 is not asserted as sole cause.'])
    a.write(output/'analysis.json',result)
    for path,digest in inputs.items():a.require(a.sha(path)==digest,'Input changed during publication')
    print(json.dumps(dict(status=result['status'],output=str(output/'analysis.json'),sha256=a.sha(output/'analysis.json'),qualification=None)))
    return 0


if __name__=='__main__':raise SystemExit(main())
