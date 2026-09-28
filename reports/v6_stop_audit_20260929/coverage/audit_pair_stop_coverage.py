"""Audit nominal closed-chain knee-angle coverage; never calibrate a real stop.

Only recorded active coordinates and the sealed final-r3 geometry are consumed.
No fitted friction, inertia, torque gain, zero offset or force data is used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

FINAL_R3 = '9542c788afd6e1b7f0af5e396062f123e511f63579cd8807087068e82e401c5c'
FIT_SEGMENTS = [9, 11, 13, 15]
HOLDOUT_SEGMENTS = [37, 39, 41, 43, 79, 89]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


class ClosureMap:
    def __init__(self, bundle):
        self.manifest = read(bundle / 'manifest.json')
        if digest(bundle / 'manifest.json') != FINAL_R3:
            raise ValueError('This diagnosis is pinned to final_r3.')
        for name, expected in self.manifest['files_sha256'].items():
            if digest(bundle / name) != expected:
                raise ValueError(f'Sealed asset hash mismatch: {name}')
        self.spec = read(bundle / 'model_spec.json')
        sys.path.insert(0, str(bundle / 'tools'))
        from v6_mechanism import fk, point
        self.fk = fk
        source = bundle / 'tools/build_v6_closedchain.py'
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in ('closure_error', 'solve_pose')]
        if len(nodes) != 2:
            raise ValueError('Reviewed closure solver functions missing.')
        namespace = {'np': np, 'fk': fk, 'point': point, 'least_squares': least_squares}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        self.solve_pose, self.closure_error = namespace['solve_pose'], namespace['closure_error']
        self.solver_source_sha256 = digest(source)
        self.nominal = self.spec['nominal_joint_pos']
        self.active = {name: self.nominal[name] for name in self.manifest['control_joint_names']}

    def solve(self, side, target, initial=None, continue_from_nominal=False):
        names = [side + '_joint1', side * 2 + '_joint1']
        q = dict(self.nominal if initial is None else initial)
        start = np.array([q[name] for name in names])
        fractions = np.linspace(0., 1., 60) if continue_from_nominal else [1.]
        for fraction in fractions:
            fixed = {**self.active, **dict(zip(names, start + fraction * (target - start)))}
            q = self.solve_pose(self.spec, fixed, q)
        gap = float(np.linalg.norm(self.closure_error(self.spec, q).reshape(-1, 3), axis=1).max())
        if gap > 1e-8:
            raise ValueError(f'Closure gap exceeds bound: {gap}')
        return q, gap

    def beta(self, side, q):
        frames = self.fk(self.spec, q)
        joint = next(j for j in self.spec['joints'] if j['child'] == side + '_link2')
        axis = frames[side + '_link2'][:3, :3] @ joint['axis']
        axis = axis / np.linalg.norm(axis)
        hip, knee, wheel = (frames[f'{side}_link{i}'][:3, 3] for i in (1, 2, 3))
        thigh, shank = hip - knee, wheel - knee
        thigh -= axis * (axis @ thigh)
        shank -= axis * (axis @ shank)
        return float(np.degrees(np.arccos(np.clip(
            thigh @ shank / np.linalg.norm(thigh) / np.linalg.norm(shank), -1., 1.))))


def summary(beta):
    return {
        'samples': len(beta),
        **{key: float(value) for key, value in zip(
            ['min_deg', 'p1_deg', 'median_deg', 'p99_deg', 'max_deg'],
            np.percentile(beta, [0, 1, 50, 99, 100]))},
        'fraction_ge_deg': {str(x): float(np.mean(beta >= x)) for x in (104.5, 105, 106, 107)},
    }


def nominal_free_mask(beta, times):
    """Require beta<104 and more than 100ms both before/after beta>=104.5."""
    near = np.flatnonzero(beta >= 104.5)
    if not len(near):
        return beta < 104.
    bad_times = times[near]
    insertion = np.searchsorted(bad_times, times)
    previous = bad_times[np.maximum(insertion - 1, 0)]
    following = bad_times[np.minimum(insertion, len(bad_times) - 1)]
    distance = np.minimum(abs(times - previous), abs(times - following))
    return (beta < 104.) & (distance > .100)


def audit_run(mapping, run, side, output):
    metadata = read(run / 'response_1khz.json')
    cfg = metadata['controller']
    word = 'left' if side == 'L' else 'right'
    if cfg['side'] != word or cfg['model_sign'] != [1.] * 4:
        raise ValueError('Recorded side/sign differ from reviewed priors.')
    if cfg['model_offset'] != [-1.6, -2.93, 1.6, 2.93]:
        raise ValueError('Recorded offsets differ from reviewed priors.')
    raw = np.load(run / 'response_1khz.npz')
    source_rows = np.flatnonzero(raw['phase'] == 2)
    q = -(raw['q_ref_model'][source_rows] - raw['position_error_model'][source_rows])
    offset = np.array(cfg['model_offset'])[metadata['axes']]
    np.testing.assert_allclose(q, -(raw['q_api'][source_rows] + offset), rtol=0, atol=1e-12)
    delta = q[:, 1] - q[:, 0]
    segments = raw['segment_id'][source_rows]
    times = raw['control_steady_ns'][source_rows].astype(np.int64) * 1e-9
    if np.any(np.diff(times) <= 0):
        raise ValueError('Non-monotone control times.')

    # Report an early plateau estimate from direct FK, before constructing the table.
    plateau = np.flatnonzero(segments == 0)[-500:]
    choose = [plateau[np.argmin(delta[plateau])],
              plateau[np.argsort(delta[plateau])[len(plateau) // 2]],
              plateau[np.argmax(delta[plateau])]]
    direct_plateau = []
    state = None
    for sample in choose:
        state, _ = mapping.solve(side, q[sample], state, continue_from_nominal=state is None)
        direct_plateau.append(mapping.beta(side, state))
    plateau_unique, plateau_inverse = np.unique(q[plateau], axis=0, return_inverse=True)
    plateau_angles = []
    for target in plateau_unique:
        state, _ = mapping.solve(side, target, state)
        plateau_angles.append(mapping.beta(side, state))
    plateau_exact = np.array(plateau_angles)[plateau_inverse]
    plateau_reference = float(np.median(plateau_exact))
    print(json.dumps({'side': word, 'plateau_direct_beta_at_delta_min_median_max_deg': direct_plateau}), flush=True)

    grid = np.linspace(float(delta.min()) - 1e-8, float(delta.max()) + 1e-8, 257)
    hip = mapping.nominal[side + '_joint1']
    states, angles, gaps = [], [], []
    state = None
    for value in grid:
        state, gap = mapping.solve(side, np.array([hip, hip + value]), state,
                                   continue_from_nominal=state is None)
        states.append(state)
        angles.append(mapping.beta(side, state))
        gaps.append(gap)
    angles = np.array(angles)
    if not (np.all(np.diff(angles) > 0) or np.all(np.diff(angles) < 0)):
        raise ValueError('Non-monotone branch: a one-dimensional table is not admitted.')
    validation = []
    # Direct midpoint FK bounds interpolation error at the table's fixed common angle.
    for cell in np.linspace(0, len(grid) - 2, 33).astype(int):
        value = (grid[cell] + grid[cell + 1]) / 2
        state, gap = mapping.solve(side, np.array([hip, hip + value]), states[cell])
        exact = mapping.beta(side, state)
        validation.append({'kind': 'table_midpoint', 'delta_rad': float(value),
                           'direct_beta_deg': exact, 'closure_gap_m': gap,
                           'table_error_deg': float(np.interp(value, grid, angles) - exact)})
    # At actual whole-leg rotations, initialize from the same table assembly branch.
    # Both active axes are then prescribed at their recorded values; passive coordinates
    # begin on that branch. This verifies the one-dimensional reduction against full FK.
    actual_points = {0, len(q) - 1, int(np.argmin(delta)), int(np.argmax(delta)), *choose}
    for segment in np.unique(segments):
        ix = np.flatnonzero(segments == segment)
        actual_points.add(int(ix[len(ix) // 2]))
    for sample in sorted(actual_points):
        cell = int(np.clip(np.searchsorted(grid, delta[sample]) - 1, 0, len(grid) - 1))
        state, gap = mapping.solve(side, q[sample], states[cell])
        exact = mapping.beta(side, state)
        validation.append({'kind': 'recorded_active_pose', 'phase2_index': int(sample),
                           'segment': int(segments[sample]), 'active_q_rad': q[sample].tolist(),
                           'delta_rad': float(delta[sample]), 'direct_beta_deg': exact,
                           'closure_gap_m': gap,
                           'table_error_deg': float(np.interp(delta[sample], grid, angles) - exact)})
    max_error = max(abs(item['table_error_deg']) for item in validation)
    if max_error >= .005:
        raise ValueError(f'Table/direct FK error {max_error} >= 0.005 deg; no coverage emitted.')
    beta = np.interp(delta, grid, angles)
    free = nominal_free_mask(beta, times)
    segment_rows, windows = [], []
    for segment in np.unique(segments):
        ix = np.flatnonzero(segments == segment)
        row = {'segment': int(segment), **summary(beta[ix]),
               'fit_role': 'fit' if segment in FIT_SEGMENTS else 'holdout' if segment in HOLDOUT_SEGMENTS else 'other',
               'recorded_validation_role_values': np.unique(raw['segment_role'][source_rows[ix]]).tolist(),
               'nominal_conservative_candidate_fraction': float(free[ix].mean())}
        row['min_nominal_gap_to_seg0_plateau_deg'] = float(plateau_reference - beta[ix].max())
        segment_rows.append(row)
        cuts = np.flatnonzero(np.diff(np.r_[False, free[ix], False]))
        for lo, hi in cuts.reshape(-1, 2):
            selected = ix[lo:hi]
            duration = times[selected[-1]] - times[selected[0]] + .001
            if duration < .250:
                continue
            windows.append({'segment': int(segment), 'phase2_start_index': int(selected[0]),
                            'phase2_stop_index_exclusive': int(selected[-1] + 1),
                            'source_start_row': int(source_rows[selected[0]]),
                            'source_stop_row_exclusive': int(source_rows[selected[-1]] + 1),
                            'duration_s': float(duration), 'samples': len(selected),
                            'nominal_beta_min_deg': float(beta[selected].min()),
                            'nominal_beta_max_deg': float(beta[selected].max())})
    result = {
        'side': word, 'source_run': str(run),
        'npz_sha256': digest(run / 'response_1khz.npz'),
        'metadata_sha256': digest(run / 'response_1khz.json'),
        'controller_profile_sha256': metadata['profile_sha256'],
        'phase2_samples': len(q), 'all_phase2': summary(beta),
        'seg0_last500': {**summary(beta[plateau]),
                        'direct_fk_summary': summary(plateau_exact),
                        'direct_fk_unique_active_poses': len(plateau_unique),
                        'direct_beta_at_delta_min_median_max_deg': direct_plateau,
                        'active_q_median_rad': np.median(q[plateau], axis=0).tolist(),
                        'dq_api_rms_rad_s': np.sqrt(np.mean(raw['dq_api'][source_rows[plateau]] ** 2, axis=0)).tolist(),
                        'interpretation': 'conditional nominal-coordinate plateau, not physical stop calibration'},
        'table': {'delta_definition': 'q_aux_v6 - q_hip_v6; not knee angle',
                  'fixed_hip_rad': hip, 'delta_grid_rad': grid.tolist(), 'knee_inner_deg': angles.tolist(),
                  'knots': len(grid), 'max_table_knot_closure_gap_m': max(gaps),
                  'max_validation_error_deg': max_error, 'validation_bound_deg': .005,
                  'validation': validation},
        'segments': segment_rows,
        'gap_to_plateau_proxy': {
            'reference_beta_deg': plateau_reference,
            'reference': 'direct FK median of this side segment 0 last 500 samples',
            'definition': 'reference_beta_deg - maximum nominal beta in each segment',
            'selected_fit_and_holdout_min_gap_deg': min(
                row['min_nominal_gap_to_seg0_plateau_deg'] for row in segment_rows
                if row['fit_role'] != 'other'),
            'physical_contact_force_measured': False,
            'interpretation': 'conditional nominal-coordinate stop proxy; no force/contact proof',
        },
        'candidate_windows': windows,
        'candidate_rule': {'beta_strictly_below_deg': 104., 'near_stop_ge_deg': 104.5,
                           'distance_from_near_stop_s_strictly_greater': .100,
                           'exclusion_direction': 'both past and future in full phase2 recording',
                           'minimum_window_s': .250,
                           'label': 'nominal-coordinate conservative screening; not verified free motion'},
    }
    (output / f'{word}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output / f'{word}-nominal-coverage.npz',
                        source_rows=source_rows, segment_id=segments, active_difference_rad=delta,
                        nominal_beta_deg=beta, conservative_candidate_mask=free)
    print(json.dumps({'side': word, 'coverage': result['all_phase2'],
                      'seg0_last500': result['seg0_last500'], 'max_table_error_deg': max_error,
                      'candidate_windows': len(windows)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = ClosureMap(args.bundle)
    result = {'schema': 'v6_nominal_stop_coverage_v1', 'manifest_sha256': FINAL_R3,
              'script_sha256': digest(Path(__file__)), 'solver_source_sha256': mapping.solver_source_sha256,
              'mapping': 'q_v6 = -(q_ref_model - position_error_model); checked against -(q_api+recorded_offset)',
              'nominal_mechanical_stop_deg': 105., 'physical_stop_tolerance_not_estimated': True,
              'asset_limits_deg_unchanged': mapping.manifest['knee_inner_limits_deg'],
              'fitted_parameters_consumed': False, 'training_release': False,
              'fit_segments': FIT_SEGMENTS, 'holdout_segments': HOLDOUT_SEGMENTS,
              'scope': 'nominal geometry coverage only; no contact labels, no encoder-zero calibration',
              'sides': {}}
    for side, name in [('L', 'run_20260928T073044Z'), ('R', 'run_20260928T144606Z')]:
        item = audit_run(mapping, args.input_root / name, side, args.output)
        result['sides'][item['side']] = {key: item[key] for key in (
            'phase2_samples', 'all_phase2', 'seg0_last500', 'npz_sha256', 'metadata_sha256')}
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
