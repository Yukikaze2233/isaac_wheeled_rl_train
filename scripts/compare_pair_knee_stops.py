"""Fixed-parameter, fixed-zero comparison of effective knee stops in paired bags.

Nominal 105 degrees and possible 106/107 degrees were confirmed by the user.
The effective stop inferred from segment 0 is only a coordinate-space scenario;
no independent physical stop or encoder bias is identified by this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from identify_pair_dynamics import Replay


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plateau_stop(replay):
    indices = np.flatnonzero(replay.a['segment_id'] == 0)[-500:]
    if len(indices) != 500:
        raise ValueError('Expected 500 calibration samples in the initial dwell.')
    unique, first, inverse = np.unique(replay.real_q[indices], axis=0,
                                      return_index=True, return_inverse=True)
    beta = []
    for index in indices[first]:
        q, _, _, _ = replay.state_at(index)
        beta.append(replay.knee_inner_deg(q[replay.knee_qi]))
    values = np.array(beta)[inverse]
    return {'source_segment': 0, 'last_samples': 500,
            'effective_inner_median_deg': float(np.median(values)),
            'inner_range_deg': [float(values.min()), float(values.max())],
            'scope': 'nominal encoder-coordinate plateau, not an independent physical calibration'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--side', choices=['left', 'right'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.experiment_root / ('run_20260928T073044Z' if args.side == 'left'
                                 else 'run_20260928T144606Z')
    parameter_file = args.experiment_root / f'{args.side}-fit/parameters.json'
    params = json.loads(parameter_file.read_text())
    calibration = Replay(run, side=args.side, physics_dt=.001,
                         bundle=args.bundle, coordinate_sign=-1.)
    plateau = plateau_stop(calibration)
    result = {'side': args.side, 'training_release': False,
              'source_user_confirmation': '2026-09-29: 105 degrees is calibration and nominal mechanical stop; actual errors may give 106 or 107',
              'scope': 'conditional effective-stop sensitivity, fixed PD/plant parameters/encoder offsets',
              'source_script_sha256': {n: digest(Path(__file__).parent/n)
                  for n in ['compare_pair_knee_stops.py', 'identify_pair_dynamics.py', 'audit_v5_motor_coupling.py']},
              'asset_manifest_sha256': digest(args.bundle/'manifest.json'),
              'input_npz_sha256': digest(run/'response_1khz.npz'),
              'parameters_sha256': digest(parameter_file), 'parameters': params,
              'calibration_plateau': plateau, 'physics_dt_s': .001,
              'initial_state_policy': 'recorded state, no clamping or settling before scoring',
              'initial_penetration_flag_deg': .1,
              'limitations': ['Stop and encoder bias are not separately identified.',
                  'MuJoCo limit compliance is an unchanged solver prior, not measured mechanical stiffness.',
                  'Fixed loss parameters were obtained with the historical 110-degree model.',
                  'A scenario initially exceeding its stop is diagnostically retained and flagged, never treated as a calibrated valid initial state.',
                  'Whole-segment holdouts remain from the same existing bag, not a new acquisition.'],
              'cases': []}
    writer = SummaryWriter(str(args.output/'events'))
    scenarios = [('historical_110', 110.), ('nominal_105', 105.),
                 ('effective_106', 106.), ('effective_107', 107.),
                 ('calibration_plateau', plateau['effective_inner_median_deg'])]
    for label, stop in scenarios:
        replay = Replay(run, side=args.side, physics_dt=.001, bundle=args.bundle,
                        coordinate_sign=-1., knee_upper_deg=stop)
        (args.output/f'{label}.xml').write_text(replay.xml)
        for mode, segments, max_samples in [('closed_loop', [37, 39, 41, 43, 79, 89], None),
                                           ('torque_input', [0, 37, 39, 41, 43, 79, 89], 1000)]:
            for segment in segments:
                score, trace = replay.replay(segment, params, max_samples=max_samples, mode=mode)
                score['initial_condition_conflict'] = score['knee_stop']['initial_upper_penetration_deg'] > .1
                result['cases'].append({'scenario': label, **score})
                prefix = f'stop_comparison/{args.side}/{label}/{mode}'
                for axis, value in enumerate(score['axis_rmse_rad']):
                    writer.add_scalar(f'{prefix}/axis{axis}_rmse_rad', value, segment)
                writer.add_scalar(f'{prefix}/stop_active_fraction', score['knee_stop']['limit_active_fraction'], segment)
                writer.add_scalar(f'{prefix}/peak_stop_constraint_nm', score['knee_stop']['peak_limit_constraint_nm'], segment)
                ix = trace['indices']
                if segment in (0, 79):
                    for k in range(0, len(ix), 10):
                        for axis in range(2):
                            curve = f'{prefix}/segment{segment}/axis{axis}'
                            writer.add_scalar(curve+'/q_sim_rad', trace['q_sim'][k, axis], k)
                            writer.add_scalar(curve+'/q_real_rad', replay.real_q[ix[k], axis], k)
                        writer.add_scalar(f'{prefix}/segment{segment}/beta_sim_deg', trace['knee_inner_deg_sim'][k], k)
                        writer.add_scalar(f'{prefix}/segment{segment}/limit_constraint_nm', trace['knee_limit_constraint_nm'][k], k)
                if label == 'historical_110' and mode == 'closed_loop':
                    previous = args.experiment_root/f'final-heldout-recheck-v1/{args.side}-candidate/report.json'
                    old = next(s for s in json.loads(previous.read_text())['scores'] if s['segment'] == segment)
                    np.testing.assert_allclose(score['axis_rmse_rad'], old['axis_rmse_rad'], atol=1e-10, rtol=0.)
                (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            writer.flush()
        print(json.dumps({'scenario_completed': label, 'stop_deg': stop, 'side': args.side}), flush=True)
    writer.close()


if __name__ == '__main__':
    main()
