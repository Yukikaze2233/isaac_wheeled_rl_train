"""Read the completed SIDE counterfactual; no native imports or simulation."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
NATIVE = AUDIT / 'mechanical_side_speed_candidate_021/native_SIDE_001/native'
FROZEN = Path('/tmp/scut-recovery-frozen-021-qv1ukmbr')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def stats(values):
    a = torch.tensor(values, dtype=torch.float64)
    return dict(n=len(values), mean=float(a.mean()), p50=float(a.quantile(.5)),
                p95=float(a.quantile(.95)), minimum=float(a.min()), maximum=float(a.max()))


def main():
    process_path = NATIVE.parent.parent / 'native_SIDE_001.process.json'
    process = read(process_path)
    assert process['process_exit_observed'] and process['process_exit_code'] == 0
    assert process['process_present_after_wait'] is False
    report = read(NATIVE / 'report.json')
    assert report['status'] == 'native600_completed_preclose'
    assert report['policy_steps_completed'] == 600 and report['physical_steps_completed'] == 2400
    assert report['arm'] == 'mean' and report['PPO_updates'] == report['formal_updates_consumed'] == 0
    inputs = [Path(__file__), process_path, NATIVE.parent / 'side_speed_review.json',
              NATIVE.parent.parent / 'independent_actual_review_001.json',
              NATIVE / 'report.json', NATIVE / 'identity.json', NATIVE / 'mechanical_layout.json',
              NATIVE / 'mechanical_5ms.jsonl.gz', NATIVE / 'first_birth_rows.jsonl.gz',
              AUDIT / 'immutable_REC3076/model.contract.json',
              *[FROZEN / 'src/wheeled_tasks/chassis' / name for name in
                ('recovery.py', 'recovery_training.py', 'env.py', 'v5_control.py')]]
    before = {str(path): sha(path) for path in inputs}
    assert sha(NATIVE.parent / 'side_speed_review.json') == '00d5ea58eec3ef47dba5c533f5e6002b08d57a61f23300a8a09313449c3c8d94'
    assert sha(NATIVE / 'identity.json') == 'bb7ac259cfef480df3581328b558257c4ee317499f8511b50e00ac65b028c553'
    contract = read(AUDIT / 'immutable_REC3076/model.contract.json')
    assert contract['usb_transport']['enabled'] is False
    assert report['native_startup']['signal_delay']['feedback_pd'] == 'fresh_physics_feedback'
    runtime = report['native_startup']['runtime_preflight']['observed']
    assert (runtime['leg_kp'], runtime['leg_kd'], runtime['wheel_velocity_p']) == (160., 2.5, .6)
    layout = read(NATIVE / 'mechanical_layout.json')
    ids = [layout['full_joint_names'].index(name) for name in layout['control_joint_names']]
    rows = {row['physical_step']: row for row in map(json.loads, gzip.open(NATIVE / 'mechanical_5ms.jsonl.gz', 'rt'))
            if row['state']['lane'] == 2216}
    native_final = next(row for row in map(json.loads, gzip.open(NATIVE / 'first_birth_rows.jsonl.gz', 'rt'))
                        if row['lane'] == 2216)
    assert native_final['partition'] == 'right_censored_unfinished'
    assert native_final['first_handover_reference_tick'] == 984 and not native_final['recovery_motion_released']
    failures, break_failures, measurements = Counter(), Counter(), defaultdict(list)
    frames, break_rows, runs = [], [], []
    dwell = peak = 0
    run_start = None
    for step in range(985, 2401):
        row, previous = rows[step], rows[step - 1]
        control, state = row['applied_control'], row['state']
        assert previous['state']['phase'] == state['phase'] == 9
        assert state['release_finished'] and state['pure_rl'] and state['blend'] == 1.
        assert state['learning_mask_so_far'] and state['policy_motion_hold_at_begin']
        assert control['blend'] == 1.
        q = torch.tensor(control['feedback_full_q_prephysics'], dtype=torch.float32)[ids]
        dq = torch.tensor(control['feedback_full_dq_prephysics'], dtype=torch.float32)[ids]
        gravity = torch.tensor(previous['state']['gravity_body'], dtype=torch.float32)
        gyro = torch.tensor(previous['state']['gyro_body'], dtype=torch.float32)
        height = torch.tensor(state['observed_height_sensor_m'], dtype=torch.float32)
        support = state['observed_support_sensor']
        gnorm = gravity.norm()
        tilt = torch.acos((-gravity[2] / gnorm).clamp(-1., 1.))
        gyro_norm = gyro.norm()
        wheel_rim = dq[[2, 5]].abs().max() * .06
        leg_speed = dq[[0, 1, 3, 4]].abs().max()
        # Reference k sees physical state k-1. Cached support/height in row k
        # come from that same original advance_reference, before physics k.
        # For a pre-existing pureRL lane, native body_clear is plausible & h>.27.
        gates = dict(valid=bool(torch.isfinite(q).all() & torch.isfinite(dq).all()
                               & torch.isfinite(gravity).all() & torch.isfinite(gyro).all()
                               & torch.isfinite(height) & (gnorm > .5)),
                     support=bool(support), body_clear=bool(support and height > .27),
                     tilt=bool(tilt < math.radians(8.)), gyro=bool(gyro_norm < .75),
                     height=bool((height > .27) & (height < .36)),
                     wheel_speed=bool(wheel_rim < .25), leg_speed=bool(leg_speed < 2.))
        failed = [key for key, passed in gates.items() if not passed]
        failures.update(failed)
        time = (step - 1) * .005
        if dwell > 0 and failed:
            break_rows.append(dict(reference_tick=step, observes_physics_step=step - 1,
                                   reference_time_s=time, preceding_dwell_ticks=dwell, failed_gates=failed))
            break_failures.update(failed)
            runs.append(dict(start_reference_tick=run_start, end_reference_tick=step - 1,
                             ticks=dwell, seconds=dwell * .005))
        if not failed and dwell == 0:
            run_start = step
        dwell = 0 if failed else dwell + 1
        peak = max(peak, dwell)
        frames.append(dict(time_s=time, reference_tick=step, failed_gates=failed, dwell_ticks=dwell))
        request = torch.tensor(control['native_requested_motor_effort_nm_at_setter'], dtype=torch.float32)
        # In pureRL, the request is the original unclipped V5 PD request.
        # These invert its inputs; they are NOT raw Gaussian means or the
        # retained script_targets_control6, and include target delay effects.
        effective_wheel_target = dq[[2, 5]] + request[[2, 5]] / .6
        effective_leg_error = (request[[0, 1, 3, 4]] + 2.5 * dq[[0, 1, 3, 4]]) / 160.
        for key, value in dict(gyro_norm_rad_s=gyro_norm, gyro_pitch_y_rad_s=gyro[1],
                              wheel_rim_peak_m_s=wheel_rim, leg_speed_peak_rad_s=leg_speed,
                              inferred_effective_wheel_target_rim_peak_m_s=effective_wheel_target.abs().max() * .06,
                              inferred_effective_leg_target_error_peak_rad=effective_leg_error.abs().max(),
                              leg_preclip_request_peak_nm=request[[0, 1, 3, 4]].abs().max(),
                              tilt_deg=tilt * (180. / math.pi), observed_height_m=height).items():
            measurements[key].append(float(value))
        wheel_forces = []
        for body in ('L_link3', 'R_link3'):
            contact = previous['contacts']['sensor_body_rows'][layout['body_names'].index(body)]
            force = torch.tensor(contact['matrix_force_n'], dtype=torch.float32).sum(0).norm()
            measurements[body + '_terrain_force_n'].append(float(force))
            wheel_forces.append(float(force))
        measurements['both_wheels_terrain_force_gt2n'].append(all(force > 2. for force in wheel_forces))
    assert len(frames) == 1416
    assert len(break_rows) == native_final['recovery_stable_break_count'] == 82
    assert peak == 16 and abs(native_final['recovery_stable_peak_seconds'] - peak * .005) < 1e-8
    assert dwell == 0 and native_final['recovery_stable_seconds'] == 0.
    bins = []
    for start, end in ((4.92, 6.), (6., 8.), (8., 10.), (10., 12.)):
        members = [frame for frame in frames if start <= frame['time_s'] < end]
        bins.append(dict(start_s=start, end_exclusive_s=end, frames=len(members),
                         failed_gate_frames_nonexclusive=dict(Counter(key for frame in members for key in frame['failed_gates'])),
                         maximum_running_dwell_ticks=max(frame['dwell_ticks'] for frame in members)))
    after = {str(path): sha(path) for path in inputs}
    assert before == after
    result = dict(schema='independent-2216-mean-hold-input-audit-v1',
                  status='source_bound_reconstruction_matches_native_counters',
                  inputs_before_sha256=before, inputs_after_sha256=after, all_inputs_unchanged=True,
                  lane=2216, birth_id=1, profile='double_step_first_riser_split', lesson='original_chain',
                  input_clock='Reference k before physics k sees refreshed physical state k-1; cached support/height in row k were produced by original reference k.',
                  actor_owned_hold_reference_ticks=[985, 2400], observed_physics_steps=[984, 2399], frames=1416,
                  native_counter_match=dict(breaks=82, peak_ticks=16, peak_seconds=.08, final_ticks=0),
                  reconstruction_scope='Per-tick native gate masks were not saved. Derived strict float32 predicates use actual input/cache snapshots and reproduce all three native dwell counters; this is not a new contact sensor.',
                  failed_gate_frames_nonexclusive={key: failures[key] for key in gates},
                  joint_pass_frames=sum(not frame['failed_gates'] for frame in frames),
                  breaks_failed_gates_nonexclusive=dict(break_failures), time_bins=bins,
                  longest_stable_runs=sorted(runs, key=lambda run: run['ticks'], reverse=True)[:8],
                  break_reference_events=break_rows, observed_physical_and_inferred_target_stats={key: stats(value) for key, value in measurements.items()},
                  inferred_effective_wheel_target_over_rim_gate_frames=sum(value >= .25 for value in measurements['inferred_effective_wheel_target_rim_peak_m_s']),
                  leg_request_over40nm_frames=sum(value > 40. for value in measurements['leg_preclip_request_peak_nm']),
                  interpretation=['This deterministic mean run fails the unchanged continuous1s gate despite actual RL ownership. Applied stochastic exploration is not necessary for this particular failure; saved sigma and entropy are not a causal proof.',
                                  'Support/bodyclear/height/valid never block this selected hold window. Both wheels have native terrain force>2N throughout the correctly aligned input window; sensor plausible support remains a separate FK/IMU proxy.',
                                  'Wheel speed, gyro and leg speed interrupt already-positive dwell; their counts overlap. Tilt blocks39 early samples but causes none of the82 positive-dwell breaks. Marginals must not be multiplied into a joint or1s success probability.',
                                  'The effective delayed wheel PD reference is frequently above the rim-speed hold threshold even under mean actions and zero motion command. This is an identified target/feedback coexistence, not proof that every such target inevitably violates the sensor gate or that delay/PD/reward alone is defective.',
                                  'Retained script_targets_control6 is not the RL mean target. Raw decoded mean actions/legs/wheels and exact delayed PD targets were not saved; the reported effective target is a source-bound inverse of the original unclipped pureRL PD request.',
                                  'The leg preclip request stays below40Nm; the recorded leg motion cannot be attributed to leg torque saturation from this evidence. Wheel envelope, filtering, ground response and spring loads remain separate physical effects.',
                                  'At12s there is no motion release/followup/success and the birth is unfinished/right-censored. No thresholds, reward, optimizer, saved std, source or contract are changed.'],
                  minimal_identification_proposal=dict(scope='Artifact-only read-only instrumentation of the existing original mean loop; no replacement controller or extra RNG/producer call.',
                                                      observe=['Save actual native stable_pass_masks/holding_rl/stable_ticks/break counter at the original reference return.',
                                                               'Save the once-decoded mean actions and legs/wheels, and exact pd_legs/pd_wheels returned by the original signal_delay.targets call before PD.',
                                                               'Save existing actor-owned leg/wheel/gyro hold-cost densities at the original50Hz payment producer, without recomputation or new reward keys.'],
                                                      DoD=['The direct gate sequence equals this correctly clocked reconstruction and native82/80ms counters.',
                                                           'Inverse-PD effective references match the actual targets within declared float32 arithmetic roundoff.',
                                                           'Report per-axis target/feedback and pitch timing with actual local payment units; retain the original1s and3mm criteria and all negative outcomes.'],
                                                      repair_decision='Do not add a duplicate velocity tax, reset sigma, relax gates or change PD from this single episode. Choose a bounded reward/curriculum or target-delay experiment only after actual native target/payment observations distinguish their contributions.'),
                  qualification=None, PPO_updates=0, production_edits=False, native_or_GPU_or_SSH_execution=False, new_tests=False)
    output = HERE / 'hold_gate_analysis_2216_001.json'
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output), native_counters=result['native_counter_match'],
                          failed_gate_frames=result['failed_gate_frames_nonexclusive'], input_count=len(inputs)), ensure_ascii=False))


if __name__ == '__main__':
    main()
