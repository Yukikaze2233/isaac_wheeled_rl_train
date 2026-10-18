"""Shared CPU fixtures for current runtime tests."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from wheeled_tasks.chassis.recovery_training import RecoveryTraining
from wheeled_tasks.chassis.v5_control import V5Control
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def rig(tmp_path):
    cfg = json.loads((ROOT / 'configs/v7_formal_20261012/mixed_tracking_repair_063.json').read_text())
    bundle = ROOT / cfg['asset_directory']
    manifest = json.loads((bundle / 'manifest.json').read_text())
    spec = json.loads((bundle / 'model_spec.json').read_text())
    fit = json.loads((bundle / 'fit_10mpa.json').read_text())
    control = json.loads((ROOT / cfg['control_math_source']).read_text())
    v5 = V5Control(manifest, spec, fit, control, cfg['v5_control'], 'cpu')
    names = manifest['tree_joint_names']
    nominal = torch.tensor([[manifest['nominal_joint_pos'][name] for name in names]]).repeat(2, 1)
    ids = [names.index(name) for name in manifest['control_joint_names']]
    env = NS(num_envs=2, device='cpu', dt=0.001, policy_dt=0.02, tick=0, cfg=cfg, manifest=manifest, nominal=nominal, ids=ids, v5=v5, scene_groups=['recover', 'ordinary'], kinds=['flat', 'flat'], origins=torch.tensor([[10.0, 20.0, 0.0], [30.0, 40.0, 0.0]]), commands=torch.zeros(2, 3), raw_height=torch.zeros(2), command_target=torch.zeros(2, 2), command_acceleration=torch.zeros(2), actions=torch.zeros(2, 6), skills=NS(specs={'recover': {'recovery_pose': 'upright', 'recovery_followup_command': [0.3, 0.0, 0.305]}}), robot=NS(joint_names=names, data=NS(joint_pos=NS(torch=nominal.clone()), root_link_lin_vel_w=NS(torch=torch.zeros(2, 3)), root_link_pose_w=NS(torch=torch.tensor([[0.0, 0.0, 0.305, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)))))
    table = {'control_joint_names': manifest['control_joint_names'], 'asset_manifest_sha256': cfg['asset_manifest_sha256'], 'axis_signs': [1.0, 1.0, -1.0, -1.0], 'wheel_axis_signs': [1.0, -1.0], 'prepare_reference': {'geometry_control6_rad': v5.nominal.tolist(), 'pd_control6_rad': v5.nominal.tolist(), 'hold_torque_nm': [0.0] * 6, 'kp': 160.0}, 'targets': {name: {'control6_rad': v5.nominal.tolist()} for name in ('nominal', 'fold', 'thrust', 'support')}, 'profiles': {'upright': {'enabled': True, 'terrain': 'flat', 'root_position_m': [0.0, 0.0, 0.305], 'root_quaternion_xyzw': [0.0, 0.0, 0.0, 1.0], 'full_joint_positions': manifest['nominal_joint_pos'], 'initial_release_s': 0.2}}}
    (tmp_path / 'profiles.json').write_text(json.dumps(table))
    (tmp_path / 'lookup.json').write_bytes((ROOT / 'reports/v7_formal_training_20261012/recovery107_candidate/height_lookup_107.json').read_bytes())
    settings = {'enabled': True, 'profiles_file': 'profiles.json', 'height_lookup_file': 'lookup.json', **{prefix + '_sha256': hashlib.sha256((tmp_path / file).read_bytes()).hexdigest() for prefix, file in (('profiles', 'profiles.json'), ('height_lookup', 'lookup.json'))}}
    recovery = RecoveryTraining(env, tmp_path, settings)
    recovery.reset(torch.arange(2))
    return (env, recovery)
