#!/usr/bin/env python3
"""Verify hypothetical V6 stop limits without changing the sealed asset."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'model/纯底盘_v6_232mm/urdf'
EXPECTED = '9542c788afd6e1b7f0af5e396062f123e511f63579cd8807087068e82e401c5c'
ZERO = math.pi - 2.3573


def passive_range_for_inner_limits(lower_deg, upper_deg):
    """Model passive coordinates in radians; independent of motor API offsets."""
    if not 0. < lower_deg < upper_deg < 180.:
        raise ValueError('Require ordered physical inner angles in (0,180)')
    low, high = np.radians([lower_deg, upper_deg]) - ZERO
    return {'L_joint2': [float(low), float(high)],
            'R_jonit2': [float(-high), float(-low)]}


def main():
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = digest(BUNDLE / 'manifest.json')
    assert before == EXPECTED
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        assert digest(BUNDLE / name) == expected, name
    sys.path.insert(0, str(BUNDLE / 'tools'))
    from v6_mechanism import fk, point
    source = BUNDLE / 'tools/build_v6_closedchain.py'
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in ('closure_error', 'solve_pose')]
    assert len(nodes) == 2
    ns = {'np': np, 'fk': fk, 'point': point, 'least_squares': least_squares}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    spec = json.loads((BUNDLE / 'model_spec.json').read_text())
    knee_names = ('L_joint2', 'R_jonit2')
    original = {'spec_rad': {j['name']: [float(j['limit'][k]) for k in ('lower', 'upper')]
                            for j in spec['joints'] if j['name'] in knee_names}}
    for filename in ('robot.urdf', 'robot.xml'):
        root = ET.parse(BUNDLE / filename).getroot()
        original[filename] = {}
        for name in knee_names:
            joint = root.find(f'.//joint[@name="{name}"]')
            original[filename][name] = (
                [float(joint.find('limit').get(k)) for k in ('lower', 'upper')]
                if filename.endswith('urdf') else list(map(float, joint.get('range').split())))
    usd = (BUNDLE / 'robot.usda').read_text()
    original['usd_deg'] = {}
    for name in knee_names:
        body = usd.split(f'def PhysicsRevoluteJoint "{name}"', 1)[1].split('}', 1)[0]
        original['usd_deg'][name] = [float(re.search(rf'physics:{key}Limit = ([-.0-9e]+)', body)[1])
                                     for key in ('lower', 'upper')]
    for exported in ('spec_rad', 'robot.urdf', 'robot.xml'):
        assert original[exported] == passive_range_for_inner_limits(40., 110.)
    rows = []
    for upper in (105., 106., 107., 110.):
        ranges = passive_range_for_inner_limits(40., upper)
        samples = []
        for hip in (-.42, 1.6):
            nominal = spec['nominal_joint_pos']
            q = dict(nominal)
            for fraction in np.linspace(0., 1., 35):
                knee = nominal['L_joint2'] + fraction * (ranges['L_joint2'][1] - nominal['L_joint2'])
                current_hip = nominal['L_joint1'] + fraction * (hip - nominal['L_joint1'])
                q = ns['solve_pose'](spec, {'L_joint1': current_hip, 'R_joint1': -current_hip,
                    'L_joint2': knee, 'R_jonit2': -knee, 'L_joint3': 0., 'R_joint3': 0.}, q)
            frames = fk(spec, q)
            for side in ('L', 'R'):
                joint = next(j for j in spec['joints'] if j['child'] == side + '_link2')
                axis = frames[side + '_link2'][:3, :3] @ joint['axis']
                hip_pos, knee_pos, wheel = (frames[f'{side}_link{i}'][:3, 3] for i in (1, 2, 3))
                a, b = hip_pos-knee_pos, wheel-knee_pos
                a -= axis * (axis @ a)
                b -= axis * (axis @ b)
                beta = float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1., 1.))))
                gap = float(np.linalg.norm(ns['closure_error'](spec, q).reshape(-1, 3), axis=1).max())
                binding = spec['spring_binding'][side + '_spring_slide']
                compression = binding['compression_at_q_zero_m'] - q[side + '_spring_slide']
                assert abs(beta-upper) < 1e-8 and gap < 1e-9
                samples.append({'side': side, 'hip_rad': q[side+'_joint1'],
                    'auxiliary_rad': q[side*2+'_joint1'], 'fk_inner_deg': beta,
                    'closure_max_gap_m': gap, 'compression_m': compression,
                    'AB_m': binding['full_extension_pin_distance_m']-compression})
        rows.append({'inner_upper_deg': upper, 'passive_ranges_rad': ranges,
                     'usd_ranges_deg': {n: np.degrees(v).tolist() for n, v in ranges.items()},
                     'closedchain_samples': samples})
    assert digest(BUNDLE / 'manifest.json') == before
    result = {'scope': 'read_only_FK_no_asset_mutation_no_physics_rollout',
              'user_confirmation': '105 deg calibration and nominal mechanical maximum; actual error may reach 106/107 deg',
              'manifest_sha256': before, 'script_sha256': digest(Path(__file__)),
              'passive_zero_inner_deg': math.degrees(ZERO), 'original_exports': original, 'hypotheses': rows}
    output = Path(__file__).with_name('mapping.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'cases': len(rows), 'fk_samples': 16,
                      'max_gap_m': max(s['closure_max_gap_m'] for r in rows for s in r['closedchain_samples'])}))


if __name__ == '__main__':
    main()
