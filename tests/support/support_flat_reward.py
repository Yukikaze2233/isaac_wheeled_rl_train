"""Shared CPU fixtures for current runtime tests."""
import math
import torch
from wheeled_tasks.chassis.tasks.rewards.flat_reward import ScutFlatRewardFrame, WEIGHTS, scut_flat_reward

def frame(count=1, dtype=torch.float64):

    def zeros(*shape):
        return torch.zeros(count, *shape, dtype=dtype)
    gravity = zeros(3)
    gravity[:, 2] = -1.0
    return ScutFlatRewardFrame(root_linear_velocity_body_m_s=zeros(3), root_angular_velocity_body_rad_s=zeros(3), projected_gravity=gravity, pitch_rad=zeros(), root_height_m=zeros() + 0.3, target_height_m=zeros() + 0.3, command_vx_m_s=zeros(), command_yaw_rate_rad_s=zeros(), joint_velocity_rad_s=zeros(6), joint_acceleration_rad_s2=zeros(6), applied_torque_nm=zeros(6), raw_action=zeros(6), previous_raw_action=zeros(6), before_previous_raw_action=zeros(6), wheel_positions_body_m=zeros(2, 3), undesired_contact_forces_world_n=zeros(3, 2, 3), terminated=torch.zeros(count, dtype=torch.bool), numerical_safety_reset=torch.zeros(count, dtype=torch.bool))

def nontrivial_frame():
    f = frame()
    f.root_linear_velocity_body_m_s[0] = torch.tensor([0.4, -0.3, 0.2])
    f.root_angular_velocity_body_rad_s[0] = torch.tensor([0.1, -0.2, 0.7])
    f.projected_gravity[0] = torch.tensor([0.2, -0.1, -math.sqrt(0.95)])
    f.pitch_rad[0] = 0.3
    f.root_height_m[0], f.target_height_m[0] = (0.31, 0.25)
    f.command_vx_m_s[0], f.command_yaw_rate_rad_s[0] = (0.05, -0.4)
    f.joint_velocity_rad_s[0] = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    f.joint_acceleration_rad_s2[0] = torch.tensor([10.0, -20.0, 30.0, -40.0, 50.0, -60.0])
    f.applied_torque_nm[0] = torch.tensor([2.0, -3.0, 4.0, -5.0, 6.0, 7.0])
    f.raw_action[0] = torch.tensor([1.0, -2.0, 0.5, -0.5, 3.0, -4.0])
    f.previous_raw_action[0] = torch.tensor([0.2, -1.0, 0.3, 0.1, 1.0, -2.0])
    f.before_previous_raw_action[0] = torch.tensor([-0.1, 0.4, 0.2, -0.3, -0.5, 1.0])
    f.wheel_positions_body_m[0, 0, 0] = 0.08
    f.wheel_positions_body_m[0, 1, 0] = 0.01
    f.undesired_contact_forces_world_n[0, 0, 1] = torch.tensor([3.0, 4.0, 0.0])
    f.terminated[0] = True
    return f
_ev = -0.4 * math.cos(0.3)
_orientation_scale = 2.0 * math.exp(-0.05 ** 2 / 3.0) + 2.0
SOURCE_EXPECTATIONS = {'termination': (-200.0, 1.0), 'leg_joint_acc': (-5e-07, 3000.0), 'leg_joint_vel': (-0.005, 30.0), 'joint_torque': (-0.0001, 139.0), 'wheel_acc': (-1e-08, 6100.0), 'wheel_vel': (-1e-05, 61.0), 'wheel_power': (-0.0001, 30.0), 'lin_vel_z': (-0.5, 0.04), 'ang_vel_xy': (-0.05, 0.05), 'action_smoothness_leg': (-0.05, 1.42), 'action_rate': (-0.01, 10.04), 'action_smoothness_wheel': (-0.01, 1.25), 'flat_orientation_y_v': (-2.0, (_orientation_scale * 0.2) ** 2), 'flat_orientation_y_exp': (1.0, math.exp(-2.0)), 'flat_orientation_x_v': (-2.0, (_orientation_scale * -0.1) ** 2), 'flat_orientation_x_exp': (1.0, math.exp(-1.0)), 'track_lin_vel_xy': (1.0, math.exp(-_ev ** 2 / 0.5) * 0.8), 'track_lin_vel_xy_square': (-0.1, (0.25 * _ev) ** 2 * 0.8), 'track_ang_vel_z': (1.0, math.exp(-0.8 ** 2 / 0.25) * 0.8), 'track_ang_vel_z_square': (-0.1, (0.5 * -1.1) ** 2 * 0.8), 'stand_still_lin_vel': (-2.0, 0.7), 'track_height_exp_tight': (1.0, math.exp(-3.6)), 'track_height_square': (-1.0, 0.36), 'no_fork': (-1.0, 1.0), 'no_fork_square': (-1.0, 0.1225), 'undesired_contact': (-2.0, 1.0)}
