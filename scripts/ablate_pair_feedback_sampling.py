"""Bounded, fixed-parameter CPU ablation of recorded host feedback updates.

Only left/right segments 79/89 are replayed. This reuses the existing Replay
model, initial state, actuator response and ideal-feedback baseline. A changed
host receive sequence latches the current simulated q/dq; an unchanged sequence
retains the previous simulated observation. No recorded q/dq is injected after
initialization. Host receive age is audited, never treated as motor sample age.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sampling_audit(replay, indices):
    arrays = replay.a
    seq = arrays['feedback_sequence'][indices].astype(np.int64)
    receive_ns = arrays['feedback_steady_ns'][indices].astype(np.int64)
    control_ns = arrays['control_steady_ns'][indices].astype(np.int64)
    diff = np.diff(seq, axis=0)
    same = diff == 0
    if np.any(seq <= 0) or np.any(diff < 0):
        raise ValueError('Missing or decreasing host receive sequence.')
    age_ms = (control_ns[:, None] - receive_ns) * 1e-6
    if np.any(age_ms < 0) or np.any(np.diff(receive_ns, axis=0) < 0):
        raise ValueError('Host receive timestamps violate ordering.')
    inconsistent = {}
    for key in ('q_api', 'dq_api', 'feedback_steady_ns'):
        changed = arrays[key][indices][1:] != arrays[key][indices][:-1]
        inconsistent[key] = (same & changed).sum(axis=0).tolist()
    if any(any(counts) for counts in inconsistent.values()):
        raise ValueError(f'Repeated sequence is inconsistent with telemetry: {inconsistent}')
    max_repeat = []
    for axis in range(2):
        starts = np.r_[0, np.flatnonzero(diff[:, axis] != 0) + 1, len(indices)]
        max_repeat.append(int(np.max(np.diff(starts)) - 1))
    return {
        'samples': len(indices),
        'same_sequence_count': same.sum(axis=0).tolist(),
        'same_sequence_fraction_of_transitions': same.mean(axis=0).tolist(),
        'max_consecutive_repeated_ticks': max_repeat,
        'sequence_increment_gt_one_count': (diff > 1).sum(axis=0).tolist(),
        'inconsistent_repeated_sequence_count': inconsistent,
        'host_receive_age_ms_percentiles': {
            str(p): np.percentile(age_ms, p, axis=0).tolist()
            for p in (0, 50, 99, 100)
        },
        'timestamp_semantics': 'host steady_clock in CAN receive callback; not motor sample time',
        'age_application': 'audit only; no reconstructed motor acquisition delay',
    }


def replay_sequence_hold(replay, segment, params):
    """Keep the baseline integrator unchanged; replace only PD observation q/dq."""
    ix, q0, v0, gap0 = replay.initial_state(segment)
    m, data = replay.model, replay.data
    m.dof_armature[replay.vi] = params['armature']
    m.dof_damping[replay.vi] = params['damping']
    m.dof_frictionloss[replay.vi] = params['frictionloss']
    mujoco.mj_resetData(m, data)
    data.qpos[:] = q0
    data.qvel[:] = v0
    mujoco.mj_forward(m, data)
    tau_s = np.broadcast_to(params['time_constant_s'], (2,))
    alpha = np.exp(-m.opt.timestep / np.maximum(tau_s, 1e-12))
    gain = np.broadcast_to(params.get('gain', 1.), (2,))
    effort = replay.api_to_model * replay.a['tau_cmd_api'][ix[0]]
    q = np.empty((len(ix), 2))
    dq = np.empty_like(q)
    torque = np.empty_like(q)
    sensed_q = np.empty_like(q)
    sensed_dq = np.empty_like(q)
    held_q, held_dq = q0[replay.qi].copy(), v0[replay.vi].copy()
    previous_seq = replay.a['feedback_sequence'][ix[0]].copy()
    max_gap = gap0
    for k, sample in enumerate(ix):
        q[k] = data.qpos[replay.qi]
        dq[k] = data.qvel[replay.vi]
        seq = replay.a['feedback_sequence'][sample]
        changed = seq != previous_seq
        held_q[changed] = q[k, changed]
        held_dq[changed] = dq[k, changed]
        previous_seq = seq.copy()
        sensed_q[k], sensed_dq[k] = held_q, held_dq
        request = np.clip(replay.kp * (replay.reference[sample] - held_q)
                          - replay.kd * held_dq, -replay.cap, replay.cap)
        torque[k] = request
        for _ in range(replay.substeps):
            effort = alpha * effort + (1 - alpha) * gain * request
            data.ctrl[replay.ui] = effort
            compression = replay.binding['compression_at_q_zero_m'] - data.qpos[replay.si]
            u = np.clip(compression / replay.curve['stroke_m'], 0., 1.)
            data.ctrl[replay.su] = np.polynomial.polynomial.polyval(
                u, replay.curve['monomial_coefficients_n'])
            mujoco.mj_step(m, data)
        if k % 10 == 0:
            gaps = (data.site_xpos[replay.site_pairs[:, 0]]
                    - data.site_xpos[replay.site_pairs[:, 1]])
            max_gap = max(max_gap, float(np.max(np.linalg.norm(gaps, axis=1))))
            if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel)) > 500:
                raise RuntimeError(f'Divergent replay in segment {segment} at sample {k}')
    trim = slice(min(1000, len(ix) // 4), None)
    err = q[trim] - replay.real_q[ix][trim]
    score = {
        'segment': segment, 'samples': len(ix),
        'axis_rmse_rad': np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
        'common_rmse_rad': float(np.sqrt(np.mean(err.mean(axis=1) ** 2))),
        'difference_rmse_rad': float(np.sqrt(np.mean(np.diff(err, axis=1) ** 2))),
        'velocity_rmse_rad_s': np.sqrt(np.mean(
            (dq[trim] - replay.real_dq[ix][trim]) ** 2, axis=0)).tolist(),
        'mean_error_rad': err.mean(axis=0).tolist(),
        'max_closure_gap_m': max_gap,
        'peak_request_nm': np.max(abs(torque), axis=0).tolist(),
        'warnings': {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number},
    }
    return score, {'indices': ix, 'q_sim': q, 'dq_sim': dq,
                   'tau_request': torque, 'q_observed': sensed_q, 'dq_observed': sensed_dq}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--replay-script', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.replay_script.resolve().parent))
    spec = importlib.util.spec_from_file_location('feedback_baseline_replay', args.replay_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    writer = SummaryWriter(str(args.output / 'tensorboard'))
    result = {
        'schema': 'pair_feedback_sequence_hold_ablation_v1',
        'script_sha256': sha256(Path(__file__)),
        'replay_script_sha256': sha256(args.replay_script),
        'bundle_manifest_sha256': sha256(args.bundle / 'manifest.json'),
        'bundle_physics_files_sha256': {
            name: sha256(args.bundle / name)
            for name in ('robot.xml', 'model_spec.json', 'fit_10mpa.json')
        },
        'physics_dt_s': .001, 'control_period_s': .001,
        'coordinate_sign': -1., 'gravity_mps2': [0., 0., -9.81],
        'parameters_refitted': False, 'training_release': False,
        'model_claim': 'conditional host sequence hold only; not identified FOC or transport latency',
        'validation_scope': 'four complete held-out segments in two existing bags; no new acquisition',
        'limitations': [
            'New sequence observes current simulated state, without unknown acquisition latency.',
            'Observed receive gaps are replayed exogenously; packet timing may change with state.',
            'Fitted actuator lag remains fixed and may already absorb some feedback effects.',
            'Host sequence/time and motor state are separate atomic reads; selected segments are checked for repeated-sequence consistency.',
        ],
        'cases': [],
    }
    for side, run_name in [('left', 'run_20260928T073044Z'), ('right', 'run_20260928T144606Z')]:
        run = args.experiment_root / run_name
        params_path = args.experiment_root / f'{side}-fit' / 'parameters.json'
        params = json.loads(params_path.read_text())
        if params.get('gain', 1.) != 1.:
            raise ValueError('Expected the original fixed six-parameter fit, gain=1.')
        replay = module.Replay(run, side=side, physics_dt=.001,
                               bundle=args.bundle, coordinate_sign=-1.)
        historical_path = args.experiment_root / f'{side}-final-asset-1ms' / 'report.json'
        historical = json.loads(historical_path.read_text())
        historical_bundle = args.experiment_root / 'asset-final'
        historical_manifest = json.loads((historical_bundle / 'manifest.json').read_text())
        current_manifest = json.loads((args.bundle / 'manifest.json').read_text())
        if sha256(historical_bundle / 'manifest.json') != historical['manifest_sha256']:
            raise AssertionError('Historical baseline bundle is not the recorded version.')
        for name, expected in result['bundle_physics_files_sha256'].items():
            if sha256(historical_bundle / name) != expected:
                raise AssertionError(f'Historical baseline physics differs: {name}')
        for key in ('tree_joint_names', 'control_joint_names', 'knee_inner_limits_deg'):
            if historical_manifest[key] != current_manifest[key]:
                raise AssertionError(f'Historical baseline geometry schema differs: {key}')
        for segment in (79, 89):
            start = time.monotonic()
            ix, *_ = replay.initial_state(segment)
            audit = sampling_audit(replay, ix)
            ideal, ideal_trace = replay.replay(segment, params)
            held, held_trace = replay_sequence_hold(replay, segment, params)
            prior = next(score for score in historical['scores'] if score['segment'] == segment)
            baseline_delta = np.array(ideal['axis_rmse_rad']) - prior['axis_rmse_rad']
            if np.max(abs(baseline_delta)) > 1e-8:
                raise AssertionError(f'Ideal baseline differs from frozen fit: {baseline_delta}')
            if not any(audit['same_sequence_count']):
                for key in ('q_sim', 'dq_sim', 'tau_request'):
                    np.testing.assert_array_equal(ideal_trace[key], held_trace[key])
            item = {
                'side': side, 'segment': segment, 'parameters': params,
                'params_sha256': sha256(params_path),
                'historical_final_asset_report_sha256': sha256(historical_path),
                'historical_bundle_manifest_sha256': historical['manifest_sha256'],
                'historical_physics_files_identical': True,
                'input_npz_sha256': sha256(run / 'response_1khz.npz'),
                'input_metadata_sha256': sha256(run / 'response_1khz.json'),
                'sampling_audit': audit, 'ideal_feedback': ideal,
                'recorded_sequence_hold': held,
                'ideal_vs_historical_axis_rmse_delta_rad': baseline_delta.tolist(),
                'hold_minus_ideal_axis_rmse_rad': (
                    np.array(held['axis_rmse_rad']) - ideal['axis_rmse_rad']).tolist(),
                'hold_vs_ideal_trajectory_rmse_rad': np.sqrt(np.mean(
                    (held_trace['q_sim'] - ideal_trace['q_sim']) ** 2, axis=0)).tolist(),
                'elapsed_s': time.monotonic() - start,
            }
            result['cases'].append(item)
            traces = {'indices': ix, 'q_real': replay.real_q[ix], 'dq_real': replay.real_dq[ix]}
            for mode, trace, score in [('ideal', ideal_trace, ideal), ('sequence_hold', held_trace, held)]:
                traces.update({f'{mode}_{key}': value for key, value in trace.items() if key != 'indices'})
                for axis, value in enumerate(score['axis_rmse_rad']):
                    writer.add_scalar(f'feedback_ablation/{side}/{mode}/axis{axis}_rmse_rad', value, segment)
                writer.add_scalar(f'feedback_ablation/{side}/{mode}/common_rmse_rad', score['common_rmse_rad'], segment)
            np.savez_compressed(args.output / f'{side}-segment{segment}-traces.npz', **traces)
            writer.flush()
            (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print(json.dumps({key: item[key] for key in (
                'side', 'segment', 'hold_minus_ideal_axis_rmse_rad', 'elapsed_s')}), flush=True)
    writer.close()


if __name__ == '__main__':
    main()
