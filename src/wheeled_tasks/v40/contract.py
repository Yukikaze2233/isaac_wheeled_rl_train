"""Versioned V4.0 contract validation, usable without Torch or Isaac."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = REPO_ROOT / 'contracts/own_v40_v1.json'
CONTRACT_ID = 'own-v40-jointspace-h5-v1'
CONTRACT_V2_ID = 'own-v40-jointspace-h5-v2'
CONTRACT_IDS = {CONTRACT_ID, CONTRACT_V2_ID}
ORDER = ['L_joint1', 'L_joint2', 'L_joint3', 'R_joint1', 'R_jonit2', 'R_joint3']


def is_round2(contract):
    identity = contract['contract_id']
    if identity not in CONTRACT_IDS:
        raise ValueError('Unsupported V4 contract identity')
    return identity == CONTRACT_V2_ID


def _finite(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if positive and value <= 0:
        raise ValueError(f'{label} must be positive')
    return float(value)


def _numbers(values, size, label, positive=False):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f'{label} must contain {size} numbers')
    return [_finite(x, label, positive) for x in values]


def validate_contract(c):
    """Reject incompatible dimensions/units and internally inconsistent settings."""
    try:
        if type(c['schema_version']) is not int or c['schema_version'] != 1 or c['contract_id'] not in CONTRACT_IDS or c['robot_id'] != 'own_v40':
            raise ValueError('Unsupported V4 contract identity')
        if c['control_frame'] != 'Xforward_Yleft_Zup':
            raise ValueError('Policy requires canonical body axes, not CAD axes')
        if c['source_to_control_rotation'] != [[0, -1, 0], [1, 0, 0], [0, 0, 1]]:
            raise ValueError('Unexpected source-frame conversion')
        timing, obs, joints, action = c['timing'], c['observations'], c['joints'], c['actions']
        dt = _finite(timing['physics_dt'], 'physics_dt', True)
        policy_dt = _finite(timing['policy_dt'], 'policy_dt', True)
        decimation = timing['decimation']
        if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1 or not math.isclose(dt * decimation, policy_dt, abs_tol=1e-12):
            raise ValueError('Physics and policy clocks do not match decimation')
        if [obs['single_dim'], obs['history_length'], obs['actor_dim'], obs['critic_dim'], action['dimension']] != [25, 5, 125, 29, 6]:
            raise ValueError('V1 requires single25/history5/actor125/critic29/action6')
        if obs['history_order'] != 'oldest_to_newest_current_last' or obs['initial_history'] != 'repeat_first_valid_observation':
            raise ValueError('Unsupported history semantics')
        if obs['empirical_normalization'] is not False or obs['wheel_position_observed'] is not False:
            raise ValueError('Unsupported observation/normalization contract')
        for key in ('angular_velocity', 'joint_position', 'joint_velocity'):
            _finite(obs['scales'][key], key, True)
        _numbers(obs['scales']['command'], 3, 'command scales', True)
        _finite(obs['clip'], 'observation clip', True)
        if joints['action_order'] != ORDER or joints['leg_indices'] != [0, 1, 3, 4] or joints['wheel_indices'] != [2, 5] or joints['hip_indices'] != [0, 3] or joints['knee_indices'] != [1, 4]:
            raise ValueError('Joint semantic order mismatch')
        if set(joints['continuous']) != {'L_joint1', 'R_joint1', 'L_joint3', 'R_joint3'}:
            raise ValueError('Hips and wheels must stay continuous')
        if joints['knee_inner_limits_deg'] != [35, 80]:
            raise ValueError('User-confirmed knee inner angles must remain35..80deg')
        nominal = _numbers(joints['nominal_positions'], 6, 'nominal positions')
        margin = _finite(joints['knee_soft_margin'], 'knee soft margin', True)
        _finite(joints['hip_soft_deviation'], 'hip work deviation', True)
        for name, offset, sign in [('L_joint2', -2.404, 1), ('R_jonit2', -2.3573, -1)]:
            lo, hi = _numbers(joints['knee_hard_limits'][name], 2, name)
            expected = sorted((math.radians(beta) - math.pi - offset) / sign for beta in (35, 80))
            if any(abs(a - b) > 1e-10 for a, b in zip((lo, hi), expected)):
                raise ValueError(f'{name} hard limits disagree with mechanical-angle conversion')
            if not lo + margin < nominal[ORDER.index(name)] < hi - margin:
                raise ValueError(f'{name} nominal must be inside soft boundaries')
        _numbers(action['leg_position_scales'], 4, 'leg action scales', True)
        _finite(action['clip'], 'action clip', True)
        _finite(action['wheel_velocity_scale'], 'wheel action scale', True)
        leg, wheel = c['actuators']['leg'], c['actuators']['wheel']
        if leg['model'] != 'DM-J8009-2EC' or wheel['model'] != 'M3508' or wheel['gear_ratio'] != 11:
            raise ValueError('User-confirmed motor identities/ratio mismatch')
        for module in (leg, wheel):
            for name in ('kd', 'effort_limit'):
                _finite(module[name], name, True)
            if _finite(module['armature'], 'armature') < 0:
                raise ValueError('Armature cannot be negative')
        _finite(leg['kp'], 'leg kp', True)
        eta = _finite(wheel['gearbox_efficiency'], 'gearbox efficiency', True)
        if eta > 1 or wheel['curve_side'] != 'motor':
            raise ValueError('Wheel curve must be motor-side and efficiency <=1')
        speeds, torques = wheel['motor_speed_rad_s'], wheel['motor_torque_nm']
        if len(speeds) < 2 or len(speeds) != len(torques):
            raise ValueError('Motor curve needs matching samples')
        speeds = _numbers(speeds, len(speeds), 'motor speeds')
        torques = _numbers(torques, len(torques), 'motor torques')
        if speeds[0] != 0 or any(b <= a for a, b in zip(speeds, speeds[1:])) or any(t < 0 for t in torques) or torques[-1] != 0:
            raise ValueError('Motor curve needs increasing speeds, nonnegative torques, zero final torque')
        if not is_round2(c) and action['wheel_velocity_scale'] * action['clip'] > speeds[-1] / wheel['gear_ratio']:
            raise ValueError('Action scale exceeds declared motor speed domain')
        _finite(c['asset']['nominal_base_height'], 'nominal height', True)
        for stage in c['commands']['stages'].values():
            for name in ('vx', 'wz', 'height'):
                bounds = _numbers(stage[name], 2, name)
                if bounds[0] > bounds[1]:
                    raise ValueError('Reversed command interval')
            if not .20 <= stage['height'][0] <= stage['height'][1] <= .34:
                raise ValueError('V1 height commands exceed initial limited-knee research domain')
            if is_round2(c):
                probability = _finite(stage.get('standing_probability', 0.0), 'standing probability')
                if not 0.0 <= probability <= 1.0:
                    raise ValueError('Standing probability must be within [0,1]')
        _finite(c['commands']['resample_seconds'], 'resample interval', True)
        for key in ('sigma_velocity', 'sigma_yaw', 'sigma_height'):
            _finite(c['rewards'][key], key, True)
        for value in c['rewards']['weights'].values():
            _finite(value, 'reward weight')
        for key in ('contact_force_threshold', 'max_tilt_deg', 'min_base_height', 'knee_limit_tolerance', 'episode_seconds'):
            _finite(c['termination'][key], key, True)
        policy = c['policy']
        if policy['class_name'] != 'ActorCritic' or policy['empirical_normalization'] is not False:
            raise ValueError('V1 requires ordinary non-normalizing ActorCritic')
        for key in ('actor_hidden_dims', 'critic_hidden_dims'):
            dims = policy[key]
            if not isinstance(dims, list) or not 1 <= len(dims) <= 6 or any(type(x) is not int or not 1 <= x <= 2048 for x in dims):
                raise ValueError('Invalid policy hidden dimensions')
        if policy['activation'] not in {'elu', 'relu', 'tanh'}:
            raise ValueError('Unsupported V1 activation')
        if is_round2(c):
            if c['timing'] != {'physics_dt': .005, 'decimation': 2, 'policy_dt': .01}:
                raise ValueError('V2 sustained failure requires the declared 100Hz policy clock')
            if joints['soft_position_limit_factor'] != .97:
                raise ValueError('V2 finite-knee soft position factor must be .97')
            if obs['noise'] != {'angular_velocity': .2, 'gravity': .05, 'joint_position': .02, 'joint_velocity': 1.5}:
                raise ValueError('V2 physical uniform noise scales mismatch')
            if c['reset'] != {'root_velocity_range': [-.5, .5], 'evaluation_root_velocity': 0.0, 'evaluation_observation_noise': False}:
                raise ValueError('V2 reset/evaluation settings mismatch')
            if c['termination']['failure_gravity_z'] != -.1 or c['termination']['failure_seconds'] != 1.0:
                raise ValueError('V2 sustained failure settings mismatch')
        if 'round3' in c:
            from .round3 import validate_round3
            validate_round3(c)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f'Incomplete V4 contract: {error}') from error
    return c


def load_contract(path=None):
    def reject(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    c = json.loads(Path(path or DEFAULT_CONTRACT).read_text(encoding='utf-8'), parse_constant=reject)
    return validate_contract(c)


def contract_digest(c):
    validate_contract(c)
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _inside(base, relative):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError(f'Expected confined relative asset path: {relative}')
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f'Asset path escapes its root: {relative}')
    return path


def audit_asset(c, repo_root=REPO_ROOT):
    """Read-only integrity/shape audit; does not approve collision assumptions."""
    validate_contract(c)
    directory = _inside(Path(repo_root), c['asset']['directory'])
    manifest_path = _inside(directory, c['asset']['manifest'])
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    if m.get('schema_version') != 1 or m.get('robot_id') != 'own_v40' or m.get('control_frame') != c['control_frame'] or m.get('knee_inner_limits_deg') != [35, 80]:
        raise ValueError('Asset identity/frame/knee convention mismatch')
    hashes = m.get('files_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Asset file integrity manifest missing')
    for name, expected in hashes.items():
        path = _inside(directory, name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Asset file hash mismatch: {name}')
    for field in ('urdf', 'mjcf'):
        if m.get(field) != c['asset'][field] or m[field] not in hashes:
            raise ValueError(f'Unhashed/mismatched {field} asset')
    urdf = _inside(directory, m['urdf'])
    robot = ET.parse(urdf).getroot()
    links = robot.findall('link')
    expected_links = {'base_link', 'L_link1', 'L_link2', 'L_link3', 'R_link1', 'R_link2', 'R_link3'}
    if len(links) != 7 or {link.get('name') for link in links} != expected_links:
        raise ValueError('Asset must contain exactly the seven V4 bodies')
    mass = sum(float(link.find('inertial/mass').get('value')) for link in links)
    if not math.isclose(mass, 12.752, abs_tol=1e-8) or not math.isclose(mass, m.get('total_mass_kg', -1), abs_tol=1e-8):
        raise ValueError('Asset mass mismatch')
    for mesh in robot.findall('.//mesh'):
        name = mesh.get('filename', '')
        _inside(directory, name)
        if name not in hashes:
            raise ValueError(f'URDF references an unhashed/nonlocal mesh: {name}')
    joints = {j.get('name'): j for j in robot.findall('joint')}
    if len(robot.findall('joint')) != 6 or set(joints) != set(ORDER):
        raise ValueError('URDF joint names do not match policy order')
    for name in ORDER:
        j = joints[name]
        if name in c['joints']['continuous']:
            if j.get('type') != 'continuous':
                raise ValueError(f'{name} must remain continuous')
        else:
            actual = [float(j.find('limit').get(k)) for k in ('lower', 'upper')]
            expected = c['joints']['knee_hard_limits'][name]
            if j.get('type') != 'revolute' or any(abs(a - b) > 1e-10 for a, b in zip(actual, expected)):
                raise ValueError(f'{name} URDF hard limit mismatch')
    nominal = m.get('nominal_joint_pos', {})
    if any(name not in nominal or abs(nominal[name] - q) > 1e-10 for name, q in zip(ORDER, c['joints']['nominal_positions'])):
        raise ValueError('Asset and policy nominal positions disagree')
    if abs(m.get('nominal_base_height_m', -1) - c['asset']['nominal_base_height']) > 1e-10:
        raise ValueError('Asset and policy nominal heights disagree')
    result = {'manifest': m, 'manifest_path': manifest_path, 'directory': directory,
              'urdf_path': urdf, 'mjcf_path': _inside(directory, m['mjcf']),
              'asset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    if m.get('research_model') is not None:
        model = m['research_model']
        for kind in ('approval', 'raw_manifest'):
            file_key = 'approval_file' if kind == 'approval' else 'raw_manifest_file'
            hash_key = 'approval_sha256' if kind == 'approval' else 'raw_manifest_sha256'
            name = model.get(file_key)
            if not isinstance(name, str) or hashes.get(name) != model.get(hash_key):
                raise ValueError(f'Research {kind} is not bound to asset hashes')
            result['research_approval' if kind == 'approval' else 'raw_manifest'] = json.loads(_inside(directory, name).read_text())
        raw_hashes = result['raw_manifest'].get('files_sha256', {})
        expected_hashes = dict(raw_hashes)
        expected_hashes[model['raw_manifest_file']] = model['raw_manifest_sha256']
        expected_hashes[model['approval_file']] = model['approval_sha256']
        if hashes != expected_hashes:
            raise ValueError('Research model must preserve every raw artifact exactly')
        for key in ('urdf', 'mjcf', 'nominal_joint_pos', 'nominal_base_height_m', 'total_mass_kg', 'control_frame'):
            if m.get(key) != result['raw_manifest'].get(key):
                raise ValueError(f'Research model altered physical identity: {key}')
        validate_filter_policy(m, result['research_approval'], result['raw_manifest'])
    return result


RESEARCH_SCOPE = 'own-v40-equivalent-serial-internal-contact-v1'
FILTER_TRIPLES = {
    ('L_joint1', 'base_link', 'L_link1'), ('L_joint2', 'L_link1', 'L_link2'),
    ('L_joint3', 'L_link2', 'L_link3'), ('R_joint1', 'base_link', 'R_link1'),
    ('R_jonit2', 'R_link1', 'R_link2'), ('R_joint3', 'R_link2', 'R_link3'),
}


def validate_filter_policy(manifest, research_approval=None, raw_manifest=None):
    """Validate filter semantics; file/hash provenance is checked by audit_asset.

    Source material review is never relabelled as repaired. A separately bound
    explicit user decision can approve a narrow equivalent research model.
    """
    collision = manifest.get('collision_validation', {})
    if collision.get('passed') is not True:
        raise ValueError('Collision scope has not passed')
    pairs = manifest.get('adjacent_collision_filter_pairs', [])
    actual = {(p.get('joint'), p.get('body1'), p.get('body2')) for p in pairs}
    if len(pairs) != 6 or actual != FILTER_TRIPLES:
        raise ValueError('Explicit adjacent collision filter scope must match the six named joints')
    model = manifest.get('research_model')
    if model is None:
        for pair in pairs:
            if pair.get('geometry_review_supported') is not True or pair.get('policy_status') != 'nominal_proxy_filter_supported':
                raise ValueError(f'Unapproved adjacent collision filter: {pair.get("body1")}/{pair.get("body2")}')
        return pairs
    approval = research_approval
    if not isinstance(approval, dict) or not isinstance(raw_manifest, dict):
        raise ValueError('Research filter requires independently hash-audited approval and raw manifest')
    if model.get('scope_id') != RESEARCH_SCOPE or approval.get('scope_id') != RESEARCH_SCOPE or model.get('model') != 'equivalent_serial_open_chain_research':
        raise ValueError('Unsupported research contact scope')
    if collision.get('scope') != 'equivalent_serial_research_with_explicit_internal_joint_exclusions':
        raise ValueError('Research collision scope label mismatch')
    for key, expected in [('approved', True), ('global_self_collision_enabled', True),
                          ('nonadjacent_collisions_unchanged', True), ('source_geometry_modified', False),
                          ('hardware_deployment_approved', False), ('raw_material_review_stays_unresolved', True),
                          ('supersedes_prior_pause_for_this_research_scope_only', True)]:
        if approval.get(key) is not expected:
            raise ValueError(f'Invalid research approval flag: {key}')
    if approval.get('robot_id') != 'own_v40' or approval.get('schema_version') != 1 or approval.get('knee_inner_limits_deg') != [35, 80]:
        raise ValueError('Research approval robot/limit identity mismatch')
    source = approval.get('approval_source', {})
    if source.get('kind') != 'explicit_user_decision' or source.get('question_id') != 'v40-equivalent-research-scope' or source.get('selected_option') != '同意，先用串联等效研究模型训练':
        raise ValueError('Explicit research decision record missing')
    for key in ('raw_manifest_file', 'raw_manifest_sha256', 'approval_source', 'scope_id'):
        if approval.get(key) != model.get(key):
            raise ValueError(f'Research approval/model mismatch: {key}')
    if manifest.get('hardware_deployment_ready') is not False or model.get('hardware_deployment_approved') is not False:
        raise ValueError('Research approval cannot grant hardware deployment')
    if approval.get('adjacent_collision_filter_pairs') != pairs:
        raise ValueError('Research filter list does not match explicit approval')
    raw_pairs = {p['joint']: p for p in raw_manifest.get('adjacent_collision_filter_pairs', [])}
    for pair in pairs:
        raw = raw_pairs.get(pair['joint'], {})
        if pair.get('research_exclusion_approved') is not True or pair.get('policy_status') != 'user_approved_joint_internal_contact_exclusion':
            raise ValueError('Research exclusion not explicitly approved')
        if pair.get('geometry_review_supported') != raw.get('geometry_review_supported') or pair.get('raw_policy_status') != raw.get('policy_status'):
            raise ValueError('Research model must preserve raw material-review status')
    expected_checks = {'active_contacts_legal_for_research_scope', 'exact_six_named_internal_pairs_only',
                       'global_self_collision_and_floor_enabled', 'hard_knee_limits_and_continuous_hips_wheels',
                       'mass_and_frame_preserved', 'measured_wheel_cylinders_and_floor_tolerance',
                       'nominal_pose_unchanged', 'nonadjacent_positive_clearance',
                       'source_wheel_tangency_and_nonwheel_ground_clearance', 'static_frame_mass_evidence_valid'}
    details = collision.get('details', {})
    checks = details.get('checks', {})
    if set(checks) != expected_checks or any(value is not True for value in checks.values()):
        raise ValueError('Research static evidence checklist incomplete')
    if details.get('source_material_overlap_repaired') is not False or details.get('raw_failure_preserved') is not True:
        raise ValueError('Raw material failures must remain disclosed')
    return pairs


def validate_asset(c, allow_research=False, repo_root=REPO_ROOT):
    result = audit_asset(c, repo_root)
    collision = result['manifest'].get('collision_validation', {})
    if collision.get('passed') is not True:
        blockers = collision.get('details', {}).get('blockers', ['collision approval missing'])
        raise ValueError('Asset collision gate BLOCKED: ' + '; '.join(str(x) for x in blockers))
    validate_filter_policy(result['manifest'], result.get('research_approval'), result.get('raw_manifest'))
    if not allow_research:
        raise ValueError('This contract has unverified dynamics; explicit --research is required, not hardware approval')
    return result


def make_run_manifest(c, asset_manifest):
    """Accept the validated result from audit_asset/validate_asset, not guessed hashes."""
    if 'asset_manifest_sha256' not in asset_manifest or 'manifest' not in asset_manifest:
        raise ValueError('Pass the validated asset audit result to make_run_manifest')
    return {'schema_version': 1, 'contract_id': c['contract_id'], 'contract_sha256': contract_digest(c),
            'asset_manifest_sha256': asset_manifest['asset_manifest_sha256'],
            'actor_obs_dim': 125, 'critic_obs_dim': 29, 'action_dim': 6,
            'policy': dict(c['policy']), 'research_only': True, 'hardware_deployment_ready': False,
            'contact_model_scope': asset_manifest['manifest'].get('collision_validation', {}).get('scope', 'unspecified'),
            'raw_material_overlap_repaired': False,
            'leg_motor_mapping_verified': c['actuators'].get('leg_transmission', {}).get('verified', False)}
