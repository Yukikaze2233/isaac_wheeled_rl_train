#!/usr/bin/env python3
"""Read-only FK audit of recorded encoder-zero priors; never calibrate hardware.

Uses the sealed V6 bundle's exact closure solver via AST extraction to avoid
importing unrelated mesh/export dependencies. Requires only NumPy and SciPy.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
FINAL_R3 = "9542c788afd6e1b7f0af5e396062f123e511f63579cd8807087068e82e401c5c"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(bundle, evidence):
    manifest = read(bundle / 'manifest.json')
    if digest(bundle / 'manifest.json') != FINAL_R3:
        raise ValueError('This diagnosis is pinned to final_r3; review before using another asset.')
    for name, expected in manifest['files_sha256'].items():
        if digest(bundle / name) != expected:
            raise ValueError(f'Sealed asset hash mismatch: {name}')
    sys.path.insert(0, str(bundle / 'tools'))
    from v6_mechanism import fk, point
    source = bundle / 'tools/build_v6_closedchain.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('closure_error', 'solve_pose')]
    if len(functions) != 2:
        raise ValueError('Reviewed closure solver functions missing')
    namespace = {'np': np, 'fk': fk, 'point': point, 'least_squares': least_squares}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    solve_pose, closure_error = namespace['solve_pose'], namespace['closure_error']
    spec = read(bundle / 'model_spec.json')
    curve = read(bundle / 'fit_10mpa.json')
    nominal, active = spec['nominal_joint_pos'], manifest['control_joint_names']

    def solve_active(target):
        q = dict(nominal)
        for fraction in np.linspace(0., 1., 60):
            prescribed = {n: nominal[n] + fraction * (target.get(n, nominal[n]) - nominal[n])
                          for n in active}
            q = solve_pose(spec, prescribed, q)
        return q

    def metrics(q, side):
        frames = fk(spec, q)
        joint = next(j for j in spec['joints'] if j['child'] == side + '_link2')
        axis = frames[side + '_link2'][:3, :3] @ joint['axis']
        hip, knee, wheel = (frames[f'{side}_link{i}'][:3, 3] for i in (1, 2, 3))
        thigh, shank = hip - knee, wheel - knee
        thigh -= axis * (axis @ thigh)
        shank -= axis * (axis @ shank)
        beta = np.arccos(np.clip(thigh @ shank / np.linalg.norm(thigh) / np.linalg.norm(shank), -1., 1.))
        direction = -thigh / np.linalg.norm(thigh)
        binding = spec['spring_binding'][side + '_spring_slide']
        compression = binding['compression_at_q_zero_m'] - q[side + '_spring_slide']
        return {'knee_inner_deg': float(np.degrees(beta)),
                'thigh_hip_to_knee_unit_base': direction.tolist(),
                'thigh_from_base_positive_z_deg': float(np.degrees(np.arccos(direction[2]))),
                'thigh_signed_xz_from_positive_z_deg': float(np.degrees(np.arctan2(direction[0], direction[2]))),
                'spring_pin_distance_m': binding['full_extension_pin_distance_m'] - compression,
                'spring_compression_m': compression,
                'spring_extension_from_minimum_m': binding['stroke_m'] - compression,
                'slider_q_m': q[side + '_spring_slide'],
                'closure_max_gap_m': float(np.linalg.norm(closure_error(spec, q).reshape(-1, 3), axis=1).max()),
                'active_q_rad': [q[side + '_joint1'], q[side * 2 + '_joint1']]}

    offsets = np.array([-1.6, -2.93, 1.6, 2.93])
    names = ['L_joint1', 'LL_joint1', 'R_joint1', 'RR_joint1']
    zero = solve_active(dict(zip(names, -offsets)))
    result = {'scope': 'offline_kinematics_and_conditional_load_projection_not_hardware_calibration',
              'bundle': str(bundle), 'manifest_sha256': FINAL_R3,
              'script_sha256': digest(Path(__file__)),
              'solver_source_sha256': digest(source),
              'mapping': {'old_model_offset_rad': offsets.tolist(), 'model_sign': [1., 1., 1., 1.],
                          'api_motor_reversed': [True] * 4, 'new_coordinate_sign': -1.,
                          'formula': 'q_v6 = -remainder(model_offset + q_api, 2*pi); q_api already reversed',
                          'active_order': names},
              'asset_knee_limits_deg_unchanged': manifest['knee_inner_limits_deg'],
              'zero_pose': {side: metrics(zero, side) for side in ('L', 'R')}, 'static_initial': {}}
    for index, (side, word) in enumerate((('L', 'left'), ('R', 'right'))):
        source_dir = evidence / f'{word}-final-asset-1ms'
        static_path, report_path = source_dir / 'static_load.json', source_dir / 'report.json'
        row = next(row for row in read(static_path) if row['segment'] == 0)
        replay = read(report_path)
        cfg = replay['source_run']['controller']
        if cfg['model_offset'] != offsets.tolist() or cfg['model_sign'] != [1.] * 4 or replay['coordinate_sign'] != -1.:
            raise ValueError('Recorded mapping does not match the stated zero prior')
        hip, auxiliary = names[2 * index:2 * index + 2]
        q = solve_active(dict(zip((hip, auxiliary), row['q_rad'])))
        measurement = metrics(q, side)
        # Projection of nominal gravity/gas loads; no fitted losses,
        # actuator gains, zero offsets, or force coefficients are consumed.
        beta_jacobian, gravity_hold, gas_hold = [], [], []
        epsilon = 1e-5
        force = np.polynomial.polynomial.polyval(
            measurement['spring_compression_m'] / curve['stroke_m'], curve['monomial_coefficients_n'])
        for name in (hip, auxiliary):
            angles, potentials, compressions = [], [], []
            for sign in (-1., 1.):
                prescribed = {n: q[n] for n in active}
                prescribed[name] += sign * epsilon
                perturbed = solve_pose(spec, prescribed, q)
                geometry = metrics(perturbed, side)
                angles.append(np.radians(geometry['knee_inner_deg']))
                compressions.append(geometry['spring_compression_m'])
                frames = fk(spec, perturbed)
                potentials.append(sum(9.81 * b['mass'] * point(frames[b['name']], b['com'])[2]
                                      for b in spec['bodies'] if b['name'].startswith(side)))
            beta_jacobian.append((angles[1] - angles[0]) / (2. * epsilon))
            gravity_hold.append((potentials[1] - potentials[0]) / (2. * epsilon))
            gas_hold.append(float(force * (compressions[1] - compressions[0]) / (2. * epsilon)))
        jacobian = np.array(beta_jacobian)
        predicted_hold = np.array(gravity_hold) + gas_hold
        needed = predicted_hold - row['measured_command_nm']
        multiplier = float(jacobian @ needed / (jacobian @ jacobian))
        projection = multiplier * jacobian
        # A geometric reference, not a proposed hardware-offset correction.
        knee_name = 'L_joint2' if side == 'L' else 'R_jonit2'
        target = {n: q[n] for n in active if n != auxiliary}
        target[knee_name] = (1. if side == 'L' else -1.) * (np.radians(105.) - (np.pi - 2.3573))
        pose_105 = solve_pose(spec, target, q)
        zero_target = {n: zero[n] for n in active if n != auxiliary}
        zero_target[knee_name] = target[knee_name]
        zero_105 = solve_pose(spec, zero_target, zero)
        assert abs(metrics(pose_105, side)['knee_inner_deg'] - 105.) < 1e-7
        assert abs(metrics(zero_105, side)['knee_inner_deg'] - 105.) < 1e-7
        result['zero_pose'][side]['aux_q_for_exact_model_105_deg_at_same_hip_rad'] = zero_105[auxiliary]
        result['zero_pose'][side]['geometric_aux_difference_to_105_rad'] = zero_105[auxiliary] - zero[auxiliary]
        measurement.update({
            'source_static_path': str(static_path), 'source_static_sha256': digest(static_path),
            'source_report_sha256': digest(report_path),
            'source_replay_manifest_sha256': replay['manifest_sha256'],
            'source_recorded_inner_deg': row['knee_inner_deg'],
            'source_report_angle_minus_recomputed_deg': row['knee_inner_deg'] - measurement['knee_inner_deg'],
            'knee_change_from_encoder_zero_deg': measurement['knee_inner_deg'] - result['zero_pose'][side]['knee_inner_deg'],
            'api_q_angle_equivalent_representative_rad': (-np.array(row['q_rad']) - offsets[2*index:2*index+2]).tolist(),
            'relative_input_minus_zero_rad': (q[auxiliary] - q[hip]) - (zero[auxiliary] - zero[hip]),
            'reported_velocity_rms_rad_s': row['velocity_rms_rad_s'],
            'reported_command_nm': row['measured_command_nm'],
            'nominal_gravity_hold_nm': row['gravity_hold_nm'], 'nominal_gas_hold_nm': row['gas_hold_nm'],
            'nominal_predicted_hold_nm': row['predicted_hold_nm'],
            'recomputed_r3_gravity_hold_nm': gravity_hold,
            'recomputed_r3_gas_hold_nm': gas_hold,
            'recomputed_r3_predicted_hold_nm': predicted_hold.tolist(),
            'r3_vs_source_hold_max_error_nm': float(np.max(abs(predicted_hold - row['predicted_hold_nm']))),
            'knee_jacobian_rad_per_motor_rad': beta_jacobian,
            'reaction_needed_under_nominal_load_nm': needed.tolist(),
            'projection_as_knee_stop_generalized_torque_nm': multiplier,
            'projected_motor_stop_reaction_nm': projection.tolist(),
            'reaction_not_explained_by_ideal_knee_stop_nm': (needed-projection).tolist(),
            'geometric_aux_difference_to_105_at_same_hip_rad': pose_105[auxiliary] - q[auxiliary],
            'model_105_spring_compression_m': metrics(pose_105, side)['spring_compression_m'],
            'not_a_fitted_or_measured_stop': True})
        result['static_initial'][side] = measurement
    assert max(row['closure_max_gap_m'] for row in result['zero_pose'].values()) < 1e-9
    assert max(row['closure_max_gap_m'] for row in result['static_initial'].values()) < 1e-9
    for row in [*result['zero_pose'].values(), *result['static_initial'].values()]:
        assert 0. <= row['spring_compression_m'] <= .080
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'model/纯底盘_v6_232mm/urdf')
    parser.add_argument('--evidence', type=Path, default=ROOT / 'reports/v6_identification_20260929/pair')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/v6_zero_pose_audit_20260929/audit.json')
    args = parser.parse_args()
    result = run(args.bundle.resolve(), args.evidence.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'zero_pose': result['zero_pose'],
                      'static_initial': result['static_initial']}, indent=2))


if __name__ == '__main__':
    main()
