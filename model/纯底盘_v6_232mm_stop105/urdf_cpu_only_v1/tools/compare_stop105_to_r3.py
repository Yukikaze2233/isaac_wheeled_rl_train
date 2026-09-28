#!/usr/bin/env python3
"""Require the only physical delta from sealed r3 to be the two knee limits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

PARENT = '9542c788afd6e1b7f0af5e396062f123e511f63579cd8807087068e82e401c5c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(candidate, parent):
    require(sha(parent / 'manifest.json') == PARENT, 'Wrong frozen parent')
    manifests = {key: load(root / 'manifest.json') for key, root in [('parent', parent), ('candidate', candidate)]}
    for key, root in [('parent', parent), ('candidate', candidate)]:
        for name, expected in manifests[key]['files_sha256'].items():
            require(sha(root / name) == expected, f'{key} input hash mismatch: {name}')
    require(manifests['candidate']['parent_manifest_sha256'] == PARENT, 'Candidate parent mismatch')
    require(manifests['candidate']['knee_inner_limits_deg'] == [40., 105.], 'Candidate must be nominal105')
    old, new = (load(root / 'model_spec.json') for root in (parent, candidate))
    expected = copy.deepcopy(old)
    changes = []
    for joint in expected['joints']:
        if joint['name'] in ('L_joint2', 'R_jonit2'):
            bound = 'upper' if joint['name'] == 'L_joint2' else 'lower'
            replacement = '1.048303061004253' if bound == 'upper' else '-1.048303061004253'
            changes.append({'joint': joint['name'], 'field': bound,
                            'before_rad': float(joint['limit'][bound]), 'after_rad': float(replacement)})
            joint['limit'][bound] = replacement
    require(new == expected, 'Unexpected spec change beyond nominal105 passive knee bounds')
    physical_unchanged = []
    for path in sorted(parent.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(parent)
        if name.parts[0] in ('meshes', 'collisions'):
            require((candidate / name).exists() and sha(path) == sha(candidate / name), f'Mesh changed: {name}')
            physical_unchanged.append(str(name))
    for name in ('constraints.json', 'gas_spring_binding.json', 'fit_10mpa.json', 'installation.json',
                 'coordinate_mapping.json', 'inertial_sources.json', 'mesh_ownership.json',
                 'collision_filters.srdf', 'source_urdf_v5.0.urdf', 'source_urdf_v5.0.csv',
                 'v5_232mm_reference_spec.json', 'source_provenance.json'):
        require(sha(parent / name) == sha(candidate / name), f'Unexpected physical/provenance change: {name}')
        physical_unchanged.append(name)
    for name in ('robot.urdf', 'robot.xml'):
        reference = ET.parse(parent / name).getroot()
        current = ET.parse(candidate / name).getroot()
        for joint_name in ('L_joint2', 'R_jonit2'):
            old_joint = reference.find(f'.//joint[@name="{joint_name}"]')
            new_joint = current.find(f'.//joint[@name="{joint_name}"]')
            if name.endswith('urdf'):
                for key in ('lower', 'upper'):
                    new_joint.find('limit').set(key, old_joint.find('limit').get(key))
            else:
                new_joint.set('range', old_joint.get('range'))
        require(ET.tostring(reference) == ET.tostring(current), f'Unexpected XML physical change: {name}')
    reference, current = ((root / 'robot.usda').read_text() for root in (parent, candidate))
    for joint_name in ('L_joint2', 'R_jonit2'):
        pattern = rf'(def PhysicsRevoluteJoint "{joint_name}"\s*\{{)(.*?)(\n\s*\}})'
        old_block, new_block = re.search(pattern, reference, re.S), re.search(pattern, current, re.S)
        require(old_block is not None and new_block is not None, 'USD knee missing')
        modified = new_block[0]
        for key in ('lower', 'upper'):
            limit = rf'float physics:{key}Limit = [-.0-9e]+'
            old_value = re.search(limit, old_block[0])[0]
            modified = re.sub(limit, old_value, modified)
        current = current[:new_block.start()] + modified + current[new_block.end():]
    require(reference == current, 'Unexpected USD change beyond two knee bound attributes')
    return {'schema_version': 1, 'passed': True, 'parent_manifest_sha256': PARENT,
            'candidate_build_manifest_sha256': sha(candidate / 'manifest.json'),
            'source_sha256': sha(Path(__file__)), 'physical_changes': changes,
            'model_spec_exactly_equal_after_only_two_bound_substitutions': True,
            'urdf_mjcf_usd_equal_after_only_knee_range_substitutions': True,
            'unchanged_physical_files_sha256': {name: sha(candidate / name) for name in physical_unchanged},
            'mass_kg_unchanged': sum(b['mass'] for b in new['bodies']),
            'all_body_inertias_and_com_exactly_unchanged': old['bodies'] == new['bodies'],
            'all_coordinate_origins_axes_constraints_and_nominal_exactly_unchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.candidate, args.parent)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'unchanged_physical_files_sha256'}, indent=2))
