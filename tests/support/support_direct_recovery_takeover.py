"""Shared CPU fixtures for current runtime tests."""
from copy import deepcopy
from pathlib import Path
import pytest
import torch
from wheeled_tasks.chassis.application.runtime.recovery_training import DIRECT_TAKEOVER_SETTINGS, RecoveryTraining, direct_capture_ready, validate_direct_takeover
from tests.support.support_recovery_training_integration import rig
ROOT = Path(__file__).resolve().parents[2]

def direct_settings(settings):
    return {**settings, 'direct_takeover': deepcopy(DIRECT_TAKEOVER_SETTINGS), 'reference_dt': 0.005, 'blend_seconds': 0.0, 'stable_seconds': 1.0, 'prepare_speed_rad_s': 6.0, 'orbit_speed_rad_s': 6.0, 'side_speed_rad_s': 6.0, 'complete_started_blend': False}

@pytest.fixture
def direct_rig(rig, tmp_path):
    env, previous = rig
    env.dt = 0.005
    env.physical_tick = 0
    env.robot.data.projected_gravity_b = type(env.robot.data.root_link_pose_w)(torch=torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1))
    env.cfg = deepcopy(env.cfg)
    env.cfg['asset_directory'] = str(ROOT / env.cfg['asset_directory'])
    recovery = RecoveryTraining(env, tmp_path, direct_settings(previous.settings))
    recovery.initial_release.zero_()
    q = recovery.capture_reference.repeat(2, 1)
    env.robot.data.joint_pos.torch[:, env.ids] = q
    recovery.reset(torch.arange(2))
    return (env, recovery, q)

def sensor_inputs(count=1):
    q = torch.tensor(DIRECT_TAKEOVER_SETTINGS['capture_reference_control6_rad']).repeat(count, 1)
    return dict(q=q, dq=torch.zeros(count, 6), gravity=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1), gyro=torch.zeros(count, 3), acceleration=torch.tensor([[0.0, 0.0, 9.81]]).repeat(count, 1), height=torch.full((count,), 0.25), support=torch.ones(count, dtype=torch.bool), sensor_valid=torch.ones(count, dtype=torch.bool), reference=q[0].clone(), shell_vertices=torch.tensor([[0.28, 0.25, -0.17], [-0.28, -0.25, -0.17]]))
