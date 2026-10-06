#!/usr/bin/env python3
"""Bounded CPU readback of already-completed source022 first-birth guards."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import math

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
CAPTURE = AUDIT / 'source_script_repair_native_candidate_022/native_SOURCE_SCRIPT_001'
LANES = (881, 1877, 2121)
PHASE = ('SELECT', 'FOLD', 'PLANT', 'ORBIT', 'THRUST', 'SIDE', 'CAPTURE', 'PREPARE', 'BLEND', 'RL', 'FAILED')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def stream(p):
    with gzip.open(p, 'rt') as f:
        for line in f:
            yield json.loads(line)


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    native = CAPTURE / 'native'
    process_path = CAPTURE.with_suffix('.process.json')
    paths = [process_path, CAPTURE / 'source_script_native_review.json', native / 'report.json',
        native / 'identity.json', native / 'mechanical_layout.json', native / 'mechanical_summary.json',
        native / 'mechanical_5ms.jsonl.gz', native / 'mechanical_global_peaks.jsonl.gz',
        native / 'first_birth_rows.jsonl.gz', native / 'source_change_summary.json',
        AUDIT / 'mechanical_remaining_capture_candidate_021/analysis_candidate_001/actual_analysis_002/first_guard_summary.json',
        AUDIT / 'mechanical_capture_candidate_021/analysis_candidate_001.py', Path(__file__)]
    identity = read(native / 'identity.json')
    paths.extend(Path(p) for p in identity['source_files_sha256'])
    paths += [Path(identity['checkpoint']), Path(identity['checkpoint']).with_suffix('.checkpoint.json'),
        Path(identity['checkpoint']).with_suffix('.contract.json')]
    before = {str(p.resolve()): sha(p) for p in paths}
    for p, digest in identity['source_files_sha256'].items():
        assert before[str(Path(p).resolve())] == digest
    process, review, report = read(process_path), read(CAPTURE / 'source_script_native_review.json'), read(native / 'report.json')
    assert process['process_exit_observed'] and process['process_exit_code'] == 0
    assert process['process_present_after_wait'] is False
    assert process['PPO_updates'] == process['formal_updates_consumed'] == 0
    assert review['complete_contact_multiset_mechanical_comparison_passed'] is True
    assert review['actual_source_commit'] == '723732adae37f12f8488f9537717664415a98d10'
    assert review['historical_parent_actor'] == 3076 and review['initial_birth_private_sampling_saved_std_exact']
    assert report['status'] == 'native600_completed_preclose'
    assert report['PPO_updates'] == report['formal_updates_consumed'] == 0
    layout, summary = read(native / 'mechanical_layout.json'), read(native / 'mechanical_summary.json')
    assert summary['complete'] and summary['hooks_restored']
    assert (summary['policy_steps'], summary['physics_steps']) == (600, 2400)
    assert layout['solver'] == [64, 32] and layout['guard_m'] == .003
    original_path = AUDIT / 'mechanical_capture_candidate_021/analysis_candidate_001.py'
    assert sha(original_path) == '5e8ac734d72e788bfa54d41c56000de4d4185cb725581ed292c402eec2c6c8a4'
    original = load(original_path, 'frozen_contact_readback_only')
    indices = [layout['full_joint_names'].index(n) for n in layout['control_joint_names']]
    previous, reduced = {}, {lane: [] for lane in LANES}
    for row in stream(native / 'mechanical_5ms.jsonl.gz'):
        state = row['state']; lane = state['lane']
        if lane not in LANES:
            continue
        step = row['physical_step']; prior = previous.get(lane)
        assert step == len(reduced[lane]) and state['birth_id'] == 1
        assert row['simulated_s'] == step * .005
        control = row['applied_control']
        if step:
            assert row['policy_tick'] == (step-1)//4+1 and row['substep'] == (step-1)%4+1
            assert control['feedback_full_q_prephysics'] == prior['state']['full_q']
            assert control['feedback_full_dq_prephysics'] == prior['state']['full_dq']
        origin = layout['selected_env_origins_world_m'][layout['lanes'].index(lane)]
        contacts = original.contact_readback(row['contacts'], origin, layout['body_names'])
        entry = dict(physical_step=step, simulated_s=row['simulated_s'],
            phase=state['phase'], phase_name=PHASE[state['phase']], route=state['route'],
            pin_gaps_mm=[1000*x for x in state['pin_gaps_m']],
            max_gap_mm=1000*state['pin_max_gap_m'], pin_index=state['pin_argmax'],
            pin_name=layout['closure_constraints'][state['pin_argmax']]['name'],
            gyro_body_rad_s=state['gyro_body'], gyro_norm_rad_s_derived=math.sqrt(sum(x*x for x in state['gyro_body'])),
            postphysics_dq_control_rad_s=[state['full_dq'][i] for i in indices],
            ownership={k: state[k] for k in ('pure_rl','scripted','learning_mask_so_far','blend','policy_motion_hold_at_begin')},
            sensor_cache_from_reference_input_physics_k_minus_1={k: state[k] for k in ('observed_support_sensor','observed_height_sensor_m')},
            actual_terrain_contact_readback=contacts,
            all_nonzero_unfiltered_net_force_rows=[] if row['contacts'] is None else
                [dict(body=layout['body_names'][i], net_force_n=b['net_force_all_contacts_n'])
                 for i,b in enumerate(row['contacts']['sensor_body_rows']) if any(x!=0 for x in b['net_force_all_contacts_n'])])
        if control:
            refs=control['script_targets_control6']
            rates=None if prior['applied_control'] is None else [(refs[i]-prior['applied_control']['script_targets_control6'][i])/.005 for i in range(6)]
            entry['native_control_same_interval'] = dict(
                script_reference_control6=refs, units=['rad','rad','rad/s','rad','rad','rad/s'],
                reference_finite_difference_per_s_derived=rates,
                fresh_dq_prephysics_rad_s=[control['feedback_full_dq_prephysics'][i] for i in indices],
                actual_preclip_request_nm=control['native_requested_motor_effort_nm_at_setter'],
                symmetric_bounds_nm=control['motor_bounds_nm'],
                actual_setter_nm=[control['full_joint_effort_target'][i] for i in indices],
                engine_effort_postphysics_nm=[state['applied_full_joint_effort'][i] for i in indices],
                phase_at_setter=control['phase'], route_at_setter=control['route'],
                continuous_encoder_winding_not_independently_reconstructed=True)
        reduced[lane].append(entry); previous[lane]=row
    births={b['lane']: b for b in stream(native/'first_birth_rows.jsonl.gz') if b['lane'] in LANES}
    old = read(paths[10]); old_by_lane={v['lane']:v for v in old['lanes']}
    lanes=[]
    for lane in LANES:
        rows=reduced[lane]; first=next(r for r in rows if r['max_gap_mm']>3)
        assert first['physical_step']==summary['first_substep_guard_crossings'][str(lane)]['physical_step']
        terminal=summary['native_first_terminals'][str(lane)]
        assert rows[-1]['physical_step']==terminal['policy_tick']*4
        assert births[lane]['termination_reasons']==['closure_gap']
        transitions=[]
        for row in rows:
            sig=(row['phase'], row['route'], row['ownership']['pure_rl'], row['ownership']['scripted'])
            if not transitions or sig!=transitions[-1]['signature']:
                transitions.append(dict(physical_step=row['physical_step'], simulated_s=row['simulated_s'], signature=sig,
                    phase_name=row['phase_name']))
        phase_stats=[]
        for phase in sorted(set(r['phase'] for r in rows[1:])):
            part=[r for r in rows[1:] if r['phase']==phase]
            phase_stats.append(dict(phase=phase, phase_name=PHASE[phase], samples=len(part),
                max_gap_mm=max(r['max_gap_mm'] for r in part),
                max_gyro_rad_s=max(r['gyro_norm_rad_s_derived'] for r in part),
                max_leg_dq_rad_s=max(abs(r['postphysics_dq_control_rad_s'][i]) for r in part for i in (0,1,3,4)),
                max_script_leg_reference_rate_rad_s_derived=max(abs(r['native_control_same_interval']['reference_finite_difference_per_s_derived'][i])
                    for r in part if r['native_control_same_interval']['reference_finite_difference_per_s_derived'] is not None for i in (0,1,3,4))))
        end=rows[-1]['physical_step']; start=first['physical_step']
        lanes.append(dict(lane=lane, actual_level=births[lane]['recovery_prefix_lesson_id'],
            actual_profile=births[lane]['sampled_profile'], group=births[lane]['group'],
            first_3mm=first, terminal=dict(policy_tick=terminal['policy_tick'], physical_step=end,
                fields={k:v for k,v in terminal['fields'].items() if k in (
                    'gap','done','terminated','timeout','success','recovery_phase','recovery_route','recovery_failure_code',
                    'recovery_reroute_count','recovery_pure_rl','recovery_motion_released','recovery_true_success')}),
            context_first_crossing=rows[max(0,start-4):min(end,start+4)+1],
            context_terminal=rows[max(0,end-4):],
            phase_route_transitions=transitions, phase_statistics=phase_stats,
            below_3mm_frames_after_first_crossing=sum(r['max_gap_mm']<=3 for r in rows[start+1:]),
            original_SIDEonly_descriptive_reference=dict(first_crossing_physical_step=old_by_lane[lane]['first_crossing_physical_step'],
                first_crossing_mm=old_by_lane[lane]['first_crossing_gap_mm'],
                terminal_policy_tick=old_by_lane[lane]['native_first_terminal_policy_tick'],
                strict_later_trajectory_identity_not_inferred_after_global_intervention=True)))
    peaks=list(stream(native/'mechanical_global_peaks.jsonl.gz')); maximum=max(peaks,key=lambda r:r['gap_m'])
    assert maximum['gap_m']==summary['full_window_max_pin_gap_m']
    assert before=={p:sha(p) for p in before}
    result=dict(schema='source022-remaining-firstbirth-closure-readonly-analysis-v1',
        status='actual_completed_native_described_no_causal_qualification', inputs_sha256=before,
        source_commit=review['actual_source_commit'], source_recovery_sha256=review['actual_source022_runtime']['source_recovery_sha256'],
        historical_parent_actor=3076, saved_birth_private_sampling_std_exact=True,
        source_global_strict_preintervention_cutoff=review['source_global_preintervention_cutoff'],
        whole_native_steps=dict(policy=600, physics=2400), selected_lanes=lanes,
        global_window_peak=maximum, App_close_returned=None, PPO_updates=0, formal_updates_consumed=0, qualification=None,
        clocks=dict(control_and_FSM_input='physics k-1', contacts_and_pin_and_gyro='physics k after simulation',
            policy_guard='post four substeps; transient crossings may precede actual terminal'),
        limitations=['Actual contact/dq/request/gap chronology does not separate target, contact and solver causal shares.',
            'Terrain patches identify filtered body/surface; all-contact net force has no self-contact pair identity.',
            'The sensor cache is FK/IMU/probe evidence from physics k-1, not the physics k terrain contact.',
            'Raw script references are different from once-decoded delayed RL PD targets during script ownership.',
            'No independent continuous-encoder winding, randomized mass/COM/friction or native overflow flag.',
            'Previously unlocalized stochastic730mm peak is not assigned to these lanes.'])
    output=HERE/'actual_remaining_022.json'
    with output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    assert before=={p:sha(p) for p in before}
    print(json.dumps(dict(output=str(output),sha256=sha(output),inputs_unchanged=True,
        lanes=[dict(lane=v['lane'],first_step=v['first_3mm']['physical_step'],phase=v['first_3mm']['phase_name'],
            first_gap_mm=v['first_3mm']['max_gap_mm'],terminal_step=v['terminal']['physical_step']) for v in lanes])))


if __name__=='__main__':
    main()
