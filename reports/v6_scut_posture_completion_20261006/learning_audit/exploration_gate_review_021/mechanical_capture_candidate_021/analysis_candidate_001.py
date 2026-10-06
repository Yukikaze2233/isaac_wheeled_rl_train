#!/usr/bin/env python3
"""CPU-only readback of completed native003 mechanics; never run a simulation.

All chronological rows are actual captured first-birth data. Derived quantities
are explicitly labeled. This tool does not infer a cause or grant qualification.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
MANIFEST_SHA='34092556be7830a1f075b7d618c3010516fee94bfbcdf3d8b31444efd0b4abbf'
LANES=(1708,2216)
GUARD=.003


def require(value,message):
    if not bool(value):raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def rows(path):
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream]


def finite_tree(value):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite actual trace value')
    elif isinstance(value,dict):
        for item in value.values():finite_tree(item)
    elif isinstance(value,list):
        for item in value:finite_tree(item)


def local_points(points,origin):
    require(len(origin)==3 and all(len(point)==3 for point in points),'Contact/origin coordinate shape differs')
    return [[point[j]-origin[j] for j in range(3)] for point in points]


def contact_readback(contact,origin,body_names):
    if contact is None:return None
    result=[]
    for body_index,row in enumerate(contact['sensor_body_rows']):
        # Existing native view maps local lane then body; no self-pair inference.
        item=dict(sensor_index=row['sensor_index'],body_name=body_names[body_index],
            normal_matrix_force_n=row['matrix_force_n'],net_force_all_contacts_without_pair_identity_n=row['net_force_all_contacts_n'],
            normal_patches=[],friction_patches=[])
        for source,key in (('normal_patches','normal_patches'),('friction_patches','friction_patches')):
            for patch in row[source]:
                if patch['count']==0:continue
                copied=dict(patch)
                copied['point_local_m_derived_native_point_minus_env_origin']=local_points(patch['point_physx_m'],origin)
                item[key].append(copied)
        if item['normal_patches'] or item['friction_patches'] or any(any(x!=0 for x in v) for v in item['normal_matrix_force_n']):
            result.append(item)
    require(len(contact['sensor_body_rows'])==len(body_names),'Native sensor/body mapping differs')
    return dict(existing_view_index=contact['existing_view_index'],capacity_indicator=contact['capacity_indicator'],
        sensor_body_rows_with_observed_force_or_patch=result,
        omitted_rows='Only rows with zero patches and zero native normal matrix are omitted here; full raw trace retains every body')


def effort_readback(control,state,layout):
    if control is None:return None
    names=layout['full_joint_names'];control_names=layout['control_joint_names'];spring_names=layout['spring_joint_names']
    require(len(control_names)==6 and len(set(names))==len(names),'Native control/full-joint schema differs')
    indices=[names.index(name) for name in control_names]
    wheels=(2,5);reference=control['script_targets_control6'];request=control['native_requested_motor_effort_nm_at_setter'];bounds=control['motor_bounds_nm']
    require(all(len(v)==6 for v in (reference,request,bounds)),'Native six-axis request/reference schema differs')
    rotary=[]
    for axis,index in enumerate(indices):
        actual=state['full_dq'][index] if axis in wheels else state['full_q'][index]
        delta=reference[axis]-actual
        rotary.append(dict(control_axis=axis,joint_name=control_names[axis],
            script_reference=reference[axis],actual_postphysics=actual,
            reference_unit=('rad/s' if axis in wheels else 'rad'),
            raw_reference_minus_postphysics_derived=delta,
            periodic_position_error_rad_derived=(None if axis in wheels else math.atan2(math.sin(delta),math.cos(delta))),
            native_requested_at_setter_nm=request[axis],native_instantaneous_bound_nm=bounds[axis],
            abs_requested_minus_bound_nm_derived=abs(request[axis])-bounds[axis],
            exact_setter_effort_nm=control['full_joint_effort_target'][index],
            engine_applied_effort_nm=state['applied_full_joint_effort'][index]))
    springs=[]
    for name in spring_names:
        index=names.index(name)
        springs.append(dict(joint_name=name,q_m=state['full_q'][index],dq_m_s=state['full_dq'][index],
            exact_setter_axial_force_n=control['full_joint_effort_target'][index],engine_applied_axial_force_n=state['applied_full_joint_effort'][index]))
    return dict(rotary_axes=rotary,gas_prismatic_axes=springs,native_request_scope=control['native_request_scope'],
        setter_scope=control['target_scope'],strength_and_transport=dict(motor_strength=control['motor_strength'],
            spring_strength=control['spring_strength'],delay_lags_5ms=control['delay_lags_5ms'],
            transport_delay_s=control['transport_delay_s'],transport_age_s=control['transport_age_s']),
        interpretation='Reference/request drive the preceding interval; actual values are postphysics. Requested-minus-bound is a descriptive preclip difference, not proof of excessive applied torque')


def frame_readback(row,layout,origin):
    state=row['state'];gap=state['pin_gaps_m'];index=state['pin_argmax']
    require(index==max(range(len(gap)),key=lambda i:gap[i]) and gap[index]==state['pin_max_gap_m'],'Actual argmax/gap differs')
    return dict(physical_step=row['physical_step'],policy_tick=row.get('policy_tick'),substep=row.get('substep'),
        simulated_s=row['simulated_s'],stage=row['stage'],lane=state['lane'],birth_id=state['birth_id'],
        phase=state['phase'],route=state['route'],ownership={key:state[key] for key in
            ('pure_rl','scripted','learning_mask_so_far','policy_motion_hold_at_begin','release_finished','blend')},
        pin_argmax=index,pin_constraint=layout['closure_constraints'][index],pin_max_gap_m=gap[index],
        all_pin_gaps_m=gap,argmax_endpoints_world_m=state['pin_endpoints_world_m'][index],
        argmax_endpoints_local_m_derived=local_points(state['pin_endpoints_world_m'][index],origin),
        root_link_pose_world=state['root_link_pose_world'],root_link_velocity_world=state['root_link_velocity_world'],
        root_position_local_m_derived=[state['root_link_pose_world'][j]-origin[j] for j in range(3)],
        gyro_body=state['gyro_body'],gravity_body=state['gravity_body'],
        observed_support_sensor=state['observed_support_sensor'],observed_height_sensor_m=state['observed_height_sensor_m'],
        actual_body_link_poses_world=state['body_link_poses_world'],
        effort_reference_readback=effort_readback(row['applied_control'],state,layout),
        actual_terrain_contact_readback=contact_readback(row['contacts'],origin,layout['body_names']))


def lane_readback(raw,layout,summary,lane,origin):
    selected=[row for row in raw if row['state']['lane']==lane]
    require(len(selected)==summary['lane_row_counts'][str(lane)] and selected,'Captured lane row count differs')
    for ordinal,row in enumerate(selected):
        require(row['physical_step']==ordinal and row['state']['birth_id']==1,'First-birth substep continuity differs')
        require(row['simulated_s']==ordinal*.005,'Captured physical time differs')
    converted=[frame_readback(row,layout,origin) for row in selected]
    first=next((i for i,row in enumerate(converted) if row['pin_max_gap_m']>GUARD),None)
    transitions=[];prior=None
    for row in converted:
        signature=(row['phase'],row['route'],row['ownership']['pure_rl'],row['ownership']['scripted'])
        if signature!=prior:
            transitions.append({key:row[key] for key in ('physical_step','simulated_s','phase','route','ownership')});prior=signature
    original=summary['first_substep_guard_crossings'].get(str(lane))
    require((first is None)==(original is None),'Raw first3mm crossing differs from preclose summary')
    if first is not None:
        require(converted[first]['physical_step']==original['physical_step'] and converted[first]['pin_argmax']==original['pin_argmax']
            and converted[first]['pin_max_gap_m']==original['max_gap_m'],'Raw crossing identity differs')
    contact_onset=next((row for row in converted if row['actual_terrain_contact_readback'] is not None
        and row['actual_terrain_contact_readback']['sensor_body_rows_with_observed_force_or_patch']),None)
    entry=dict(lane=lane,actual_original_profile=layout['profiles'][list(LANES).index(lane)],
        native_env_origin_world_m=origin,captured_first_birth_physics_rows=len(converted)-1,
        phase_route_ownership_transitions=transitions,
        first_terrain_patch_or_matrix_force_observation=(None if contact_onset is None else
            {key:contact_onset[key] for key in ('physical_step','simulated_s','phase','route','actual_terrain_contact_readback')}),
        first_substep_above_original3mm=(None if first is None else converted[first]),
        crossing_context=([] if first is None else converted[max(0,first-5):min(len(converted),first+3)]),
        native_first_terminal=summary['native_first_terminals'].get(str(lane)),
        actual_pre_autoreset=summary['pre_autoreset_snapshots'].get(str(lane)),
        native_terminal_and_substep_crossing_are_distinct=True,
        cause='Not determined by this descriptive readback')
    return entry,converted


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root',type=Path,default=HERE/'native_capture_003')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exit-receipt',type=Path)
    args=parser.parse_args(argv)
    capture=args.capture_root.resolve();output=args.output.resolve()
    require(HERE in capture.parents and HERE in output.parents and not output.exists(),'New candidate-only analysis output required')
    spec=importlib.util.spec_from_file_location('frozen_capture_readonly',HERE/'run_capture_021.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.verify_candidate(MANIFEST_SHA)
    names=['run_plan.json','native/report.json','native/identity.json','native/mechanical_layout.json','native/mechanical_summary.json',
        'native/mechanical_5ms.jsonl.gz','native/mechanical_global_peaks.jsonl.gz','native/first_birth_rows.jsonl.gz',
        'native/events.jsonl.gz','native/payment_summary.json','native/teaching_summary.json']
    inputs={str(capture/name):sha(capture/name) for name in names}
    inputs[str(HERE/'candidate_manifest.json')]=sha(HERE/'candidate_manifest.json')
    inputs[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    identity=read(capture/'native/identity.json')
    require(identity['source_commit_label']=='d0dce6181df83b03fe166abcbadef3c81de2e57e'
        and identity['checkpoint_clocks']['task_actor_update']==3076,'Actual source/REC parent differs')
    checkpoint=Path(identity['checkpoint'])
    for path,key in ((checkpoint,'checkpoint_sha256'),(checkpoint.with_suffix('.contract.json'),'checkpoint_contract_sha256'),
        (checkpoint.with_suffix('.checkpoint.json'),'checkpoint_seal_sha256'),(Path(identity['source_manifest']),'source_manifest_sha256')):
        digest=sha(path);require(digest==identity[key],'Authenticated source/parent binding differs: '+key);inputs[str(path)]=digest
    for path,digest in identity['source_files_sha256'].items():
        require(sha(path)==digest,'Authenticated frozen source changed: '+path);inputs[path]=digest
    exit_receipt=None
    if args.exit_receipt:
        exit_receipt=read(args.exit_receipt);inputs[str(args.exit_receipt.resolve())]=sha(args.exit_receipt)
    summary=read(capture/'native/mechanical_summary.json');layout=read(capture/'native/mechanical_layout.json');report=read(capture/'native/report.json')
    require(read(capture/'run_plan.json')['candidate_sha256']==MANIFEST_SHA,'Actual run uses another candidate')
    require(summary['complete'] and summary['hooks_restored'] and summary['policy_steps']==600 and summary['physics_steps']==2400,
        'Actual native mechanical capture is not complete')
    require(not (capture/'native/mechanical_capture_failure.json').exists(),'Actual capture failure sidecar exists')
    require(layout['lanes']==list(LANES) and layout['solver']==[64,32] and layout['guard_m']==GUARD,
        'Original lanes/solver/guard changed')
    require(report['PPO_updates']==report['formal_updates_consumed']==0 and report['qualification'] is None,'Native learning/qualification claimed')
    require(report['status']=='native600_completed_preclose','Native parent observation did not complete')
    raw=rows(capture/'native/mechanical_5ms.jsonl.gz');peaks=rows(capture/'native/mechanical_global_peaks.jsonl.gz')
    finite_tree(raw);finite_tree(peaks);finite_tree(summary)
    require(peaks and max(row['gap_m'] for row in peaks)==summary['full_window_max_pin_gap_m'],'Actual global peak stream differs')
    require(all(peaks[i]['gap_m']<peaks[i+1]['gap_m'] and peaks[i]['physical_step']<=peaks[i+1]['physical_step']
        for i in range(len(peaks)-1)),'Global running peak chronology differs')
    lanes=[];chronology=[]
    for lane,origin in zip(LANES,layout['selected_env_origins_world_m']):
        entry,converted=lane_readback(raw,layout,summary,lane,origin);lanes.append(entry);chronology.extend(converted)
    for path,digest in inputs.items():require(sha(path)==digest,'Actual input changed while reading: '+path)
    output.mkdir(parents=True)
    trace_output=output/'chronological_readback.jsonl.gz'
    with gzip.open(trace_output,'wt') as stream:
        for row in sorted(chronology,key=lambda row:(row['physical_step'],row['lane'])):
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    maximum=max(peaks,key=lambda row:row['gap_m'])
    result=dict(schema='source021-mechanical-native003-descriptive-analysis-v1',actual_inputs_sha256=inputs,
        frozen_candidate_manifest_sha256=MANIFEST_SHA,native_PPO_updates=0,formal_updates_consumed=0,qualification=None,
        completed_actual_native_policy_steps=600,completed_actual_native_physics_steps=2400,
        actual_contact_count_at_capacity_indicators=summary['contact_count_at_capacity_indicators'],
        process_exit_receipt=exit_receipt,App_close_returned='unknown; raw process receipt does not prove Python close returned',
        lanes=lanes,global_window_actual_maximum=maximum,
        global_running_peak_positions=[{key:row[key] for key in ('physical_step','policy_tick','substep','simulated_s','lane','pin','birth_id','gap_m')} for row in peaks],
        chronological_readback=dict(path=str(trace_output),sha256=sha(trace_output),rows=len(chronology)),
        cause='Not inferred. Contact/reference/effort/pin coincidence is recorded without causal classification.',
        limitations=['No self-contact pair identity or native overflow flag','Actual randomized mass/COM/friction draw not captured',
            'Script reference is compared with postphysics joint state; all residuals are descriptive derived differences',
            'Terrain/contact local coordinates are native world values minus captured env origin',
            'Global peak is localized only for this actual captured mean run; old unlocalized stochastic730mm peak remains unassigned',
            'Mechanical/solver/self-righting functional qualification remains absent'])
    write(output/'analysis.json',result)
    for path,digest in inputs.items():require(sha(path)==digest,'Actual input changed during analysis publication: '+path)
    print(json.dumps(dict(status='actual_completed_native_capture_described_no_causal_claim',output=str(output/'analysis.json'),
        output_sha256=sha(output/'analysis.json'),qualification=None)))
    return 0


if __name__=='__main__':raise SystemExit(main())
