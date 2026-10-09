"""Read native feedback, controller predicates and four torque layers separately."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

base = Path(__file__).resolve().parent
directory = base / 'recovery_script_diagnostic008'
runtime = base.parents[2] / '.worktrees/serial-dispatch-source028-20261009'
sys.path.insert(0, str(runtime / 'src'))
from wheeled_tasks.chassis.recovery import RecoveryFSM, RecoveryPhase
import torch

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def longest_run(mask):
    peak = run = 0
    for value in mask:
        run = run + 1 if value else 0
        peak = max(peak, run)
    return peak

receipt = json.loads((directory / 'native/script_diagnostic.json').read_text())
assert receipt['status'] == 'native_script_diagnostic_completed_unscored'
path = directory / 'native/script_prefix_200hz.npz'
assert receipt['sidecar']['sha256'] == digest(path)
with np.load(path, allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files}
assert receipt['sidecar']['rows'] == 1600
assert set(receipt['original_method_calls'].values()) == {1600}
native = json.loads((directory / 'native/native_replay.json').read_text())
events = native['candidates'][0]['native_events']['replicas']
legs, wheels = list(RecoveryFSM.LEG_IDS), list(RecoveryFSM.WHEEL_IDS)
error_pd = arrays['target_after'][:, :, legs] - arrays['encoder_continuous_q_rad'][:, :, legs]
error_fsm = RecoveryFSM.paired_delta(torch.from_numpy(arrays['target_after'][:, :, legs]).reshape(-1, 4),
    torch.from_numpy(arrays['fsm_continuous_q_rad'][:, :, legs]).reshape(-1, 4)).numpy().reshape(1600, 5, 4)
rows = []
for index, lane in enumerate(receipt['sidecar']['lanes']):
    phase = arrays['phase_after'][:, index]
    active = arrays['release_finished'][:, index]
    window = (np.arange(1600) >= 1400) & active
    prepare = (arrays['phase_before'][:, index] == int(RecoveryPhase.PREPARE)) & active
    late = arrays['feedback_gyro_body_rad_s'][window, index]
    g = arrays['feedback_gravity_body'][window, index]
    tilt = np.degrees(np.arccos(np.clip(-g[:, 2], -1., 1.)))
    conditions = {key.removeprefix('condition_'): arrays[key][:, index] for key in arrays if key.startswith('condition_')}
    all_pass = np.logical_and.reduce(list(conditions.values()))
    requested = arrays['requested_after_ownership_nm'][:, index]
    bounds = arrays['bounds_at_script_nm'][:, index]
    issued = arrays['issued_motor_effort_nm'][:, index]
    applied = arrays['post_physics_applied_motor_nm'][:, index]
    pd_error = np.abs(error_pd[:, index]).max(axis=-1)
    fsm_error = np.abs(error_fsm[:, index]).max(axis=-1)
    row = dict(lane=lane, case=events[lane]['case'], native_terminal=events[lane],
        observed_first_episode_alive_all=bool(arrays['first_episode_alive'][:, index].all()),
        phase_reference_seconds={RecoveryPhase(int(p)).name: int((phase == p).sum()) * .005 for p in np.unique(phase)},
        ever_blend=bool((phase == int(RecoveryPhase.BLEND)).any()),
        maximum_ready_ticks=int(arrays['ready_ticks'][:, index].max()),
        late_window=dict(physics_intervals=int(window.sum()), feedback_time_range_s=[7., 7.995],
            tilt_deg_range=[float(tilt.min()), float(tilt.max())],
            gyro_norm_rad_s_range=[float(np.linalg.norm(late, axis=-1).min()), float(np.linalg.norm(late, axis=-1).max())],
            wheel_rim_speed_m_s_range=[float((np.abs(arrays['feedback_dq_rad_s'][window, index][:, wheels]).max(axis=-1) * .06).min()),
                                       float((np.abs(arrays['feedback_dq_rad_s'][window, index][:, wheels]).max(axis=-1) * .06).max())],
            PD_leg_error_abs_max_rad=float(pd_error[window].max()), FSM_paired_leg_error_abs_max_rad=float(fsm_error[window].max()),
            condition_fail_counts={key: int((~value[window]).sum()) for key, value in conditions.items()},
            all_sensor_pass_intervals=int(all_pass[window].sum()),
            native_ready_intervals=int(arrays['ready'][window, index].sum()),
            longest_all_sensor_pass_s=longest_run(all_pass[window]) * .005),
        prepare_intervals=int(prepare.sum()),
        prepare_all_sensor_pass_intervals=int((prepare & all_pass).sum()),
        prepare_good_sensor_and_paired_error_intervals=int((prepare & all_pass & (fsm_error < .12)).sum()),
        torque_since_release=dict(samples=int(active.sum()),
            requested_peak_abs_nm=np.abs(requested[active]).max(axis=0).tolist(),
            issued_peak_abs_nm=np.abs(issued[active]).max(axis=0).tolist(),
            actual_applied_peak_abs_nm=np.abs(applied[active]).max(axis=0).tolist(),
            requested_at_95pct_bound_fraction=(np.abs(requested[active]) >= .95 * bounds[active]).mean(axis=0).tolist(),
            requested_over_bound_fraction=(np.abs(requested[active]) > bounds[active]).mean(axis=0).tolist()),
        late_torque=dict(requested_peak_abs_nm=np.abs(requested[window]).max(axis=0).tolist(),
            issued_peak_abs_nm=np.abs(issued[window]).max(axis=0).tolist(),
            actual_applied_peak_abs_nm=np.abs(applied[window]).max(axis=0).tolist()),
        final_reference=dict(phase=RecoveryPhase(int(phase[-1])).name,
            target=arrays['target_after'][-1, index].tolist(), feedback_q_rad=arrays['encoder_continuous_q_rad'][-1,index].tolist(),
            feedback_dq_rad_s=arrays['feedback_dq_rad_s'][-1,index].tolist(),
            predicates={key: bool(value[-1]) for key,value in conditions.items()},
            native_ready=bool(arrays['ready'][-1,index]), ready_ticks=int(arrays['ready_ticks'][-1,index])))
    rows.append(row)
result = dict(status='completed_native_diagnostic_readonly_analysis', source_raw_sha256=digest(path),
    analyzer_sha256=digest(__file__), native_result_sha256=digest(directory / 'native/native_replay.json'),
    rows=rows, all_selected_never_blended=all(not row['ever_blend'] for row in rows),
    sampling_scope='1600 pre-reference states at0..7.995s; same intervals actual postphysics at.005..8.0s',
    remaining_available=False, true_contact_or_support_equivalence_proven=False,
    causal_repair_verified=False, qualification=None, PPO_updates=0, training_signals=0)
write(directory / 'analysis001.json', result)
print(json.dumps([dict(lane=row['lane'], phase=row['final_reference']['phase'],
    late_failed=row['late_window']['condition_fail_counts'], wheel_speed=row['late_window']['wheel_rim_speed_m_s_range'],
    leg_error=row['late_window']['PD_leg_error_abs_max_rad'], ready_peak=row['maximum_ready_ticks']) for row in rows]))
