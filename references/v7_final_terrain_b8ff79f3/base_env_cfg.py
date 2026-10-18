# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

import agent_tasks.manager.mdp.isaaclab as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelCfg, ConstantNoiseCfg, UniformNoiseCfg, GaussianNoiseCfg
from dataclasses import MISSING, field
from collections import OrderedDict
import copy
import os
import torch


from agent_world.assets.wheelbipe25_v3 import Wheelbipe25_v3_CFG, Wheelbipe25_v3_IdealPD_CFG, Wheelbipe25_v3_DelayPD_CFG
from agent_world.assets.wheelbipe25_v3_reduce_spring import Wheelbipe25_v3_reduce_spring_CFG

@configclass
class EventCfg:
    # on start up
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.9, 1.2),
            "operation": "scale",
        },
    )
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_front1_link",".*_front2_link",".*_front3_link",".*_front4_link",
                                                             ".*_rear1_link",".*_rear2_link",".*_spring1_link",".*_spring2_link"]),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.04, 0.04), "z": (-0.02, 0.02)},
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    # on reset
    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (-0.5, 0.5), 
    #             "y": (-0.5, 0.5),
    #             "z": (0.0, 0.25),
    #             "roll": (-0.3, 0.3),
    #             "pitch": (-0.3, 0.3),
    #             "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-2., 2.),
    #             "y": (-2., 2.),
    #             "z": (-0.5, 0.5),
    #             "roll": (-0.25, 0.25),
    #             "pitch": (-0.25, 0.25),
    #             "yaw": (-0.25, 0.25),
    #         },
    #     },
    # )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_vel_b,
        mode="reset",
        params={
            "pose_range": {
                # "x": (-0.5, 0.5), 
                # "y": (-0.5, 0.5),
                "z": (0.0, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                # "z": (-0.25, 0.25),
                # "roll": (-0.25, 0.25),
                # "pitch": (-0.25, 0.25),
                # "yaw": (-0.25, 0.25),
            },
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # on interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            # "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}
            },
    )
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(5.0, 10.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force_range": (-25.0, 25.0),
    #         "torque_range": (-2.5, 2.5),
    #     },
    # )
    base_external_force_torque_xyz = EventTerm(
        func=mdp.apply_external_force_torque_xyz,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": ((-10.,10.),(-10.0, 10.0),(-10.,10.)),
            "torque_range": ((-1., 1.),(-1.,1.),(-1.,1.)),
        },
    )

@configclass
class Wheelbipe25v3BaseEnvCfg(DirectRLEnvCfg):
    ''' env configs '''
    # env
    decimation = 4 # 1/200 -> 4    1/500 -> 5
    episode_length_s = 20.0
    termination_roll_deg = 40.0   # 终止条件：|roll| 超过该角度则 reset
    termination_pitch_deg = 40.0  # 终止条件：|pitch| 超过该角度则 reset
    termination_duration_enabled: bool = False
    termination_duration_steps: int = 1
    action_scale = 0.25  # [N]
    action_space = 6
    observation_space = 28
    state_space = 32
    value_space = 1
    play: bool = False
    play_keep_done_reset: bool = False
    reset_heading_axis_aligned_only: bool = False
    reset_heading_axis_aligned_candidates_deg: tuple[float, ...] = (0.0, 90.0, 180.0, -90.0)
    reset_heading_target_terminate_enabled: bool = False
    reset_heading_target_terminate_threshold_deg: float = 2.0
    # frame stack
    num_obs_hist = 1
    num_privileged_obs_hist = 1
    num_single_obs = 28
    num_single_privileged_obs = 32
    task_flag_obs_enabled: bool = False
    task_flag_obs_dim: int = 0
    play_height_scanner_debug_vis: bool = False
    play_terrain_debug_vis: bool = False
    play_ang_vel_z_debug_vis: bool = True
    enable_state_machines: bool = True
    airborne_state_machine_cfg: dict = field(default_factory=lambda: {
        "enabled": False,
        "enter": {
            "wheel_radius": 0.05,
            "body_height_threshold": 0.3,
            "wheel_clearance_threshold": 0.03,
        },
        "target_height": {
            "bias": 0.0,
            "max": None,
        },
        "exit": {
            "wheel_contact_force_threshold": 1.0,
            "wheel_contact_duration_s": 0.1,
            "base_contact_force_threshold": 5.0,
            "base_contact_duration_s": 0.05,
            "max_duration_s": None,
        },
        "reward_scales": {
            "undesired_contact": 50.0,
            "flat_orientation_y": 1.0,
            "flat_orientation_x": 1.0,
            "wheel_motor_z_axis_align_exp": 1.0,
            "wheel_motor_z_axis_align_exp_tight": 1.0,
        },
        "reward_additions": {},
    })
    wheel_forward_scan_cfg: dict = field(default_factory=lambda: {
        "enabled": False,
        "scan": {
            "forward_offset": 0.10,
        },
        "detect": {
            "step_height_min": 0.15,
            "step_height_max": 0.25,
            "wall_height": 0.25,
        },
        "height_cmd": {
            "bias": 0.05,
            "hold_s": 2.0,
            "max": None,
        },
    })
    undesired_contact_force_threshold: float = 5.0
    desired_contact_force_threshold: float = 1.0
    # 概率帧遮蔽：[p_keep_1, p_keep_2, ..., p_keep_M]
    # 训练时每步对每个环境独立采样保留帧数，剩余历史帧置零
    # 例：[0.4, 0.1] 表示 40% 只保留最新1帧，10% 保留最新2帧，50% 保留全部帧
    # None 表示关闭（默认）
    frame_mask_probs: list | None = None

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # inner simulation time step 1/200-1/500
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # height scanner for body height estimation
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        ray_alignment = "yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.015, 0.015)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    dot_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot_cfg: ArticulationCfg = MISSING

    # data delays
    obs_delay_cfg = {
        'motor': [1, 5],   # 默认无延迟
        'imu': [1, 5],
    }
    obs_history_len = 7 # 4
    obs_default_time_lag = 3
    use_obs_delay = False
    
    act_delay_cfg = {
        'leg_actions': [1, 5],
        'wheel_actions': [1, 5],
    }
    use_act_delay = False


@configclass
class Wheelbipe25v3ReduceSpringFlatEnvCfg(Wheelbipe25v3BaseEnvCfg):
    ''' basic configs '''
    # play 模式下低频打印 undesired link 接触情况；<=0 关闭
    undesired_contact_debug_interval: int = 0
    undesired_contact_debug_force_threshold: float = 1.0
    undesired_contact_debug_max_envs: int = 4
    undesired_contact_debug_max_links: int = 6

    # robot
    robot_cfg: ArticulationCfg = Wheelbipe25_v3_reduce_spring_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    # legs joints that should be controlled
    legs_act_name = robot_cfg.actuators["legs_act"].joint_names_expr
    # legs joints that have no effort
    legs_inact_name = robot_cfg.actuators["legs_inact"].joint_names_expr
    # wheels joints
    wheel_name = robot_cfg.actuators["wheel"].joint_names_expr
    # action scale
    leg_action_scale = 0.5
    wheel_action_scale = 1.0
    # obs scale
    lin_vel_scale = 1.0
    height_scale = 5.0
    ang_vel_scale = 1.0
    joint_pos_scale = 1.0
    joint_vel_scale = 0.1
    wheel_vel_scale = 0.1
    joint_torque_scale = 0.1
    # lin_vel_scale = 1.0
    # height_scale = 1.0
    # ang_vel_scale = 1.0
    # joint_pos_scale = 1.0
    # joint_vel_scale = 1.0
    # wheel_vel_scale = 1.0
    # joint_torque_scale = 1.0

    ''' commands '''
    # motion commands
    commands = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.0, 3.0),
            lin_vel_y=(0., 0.),
            ang_vel_z=(0, 0),
            heading=(0, 0)
        ),
    )
    # height commands
    default_height_cmd = 0.25
    height_range = [0.22,0.35]
    use_leg_length_as_height = False
    # 高度来源:
    # 1) use_leg_length_as_height=True  -> 使用腿长
    # 2) use_leg_length_as_height=False 且 use_absolute_height=False -> 使用 raycast 估计地面后的相对高度
    # 3) use_absolute_height=True -> 直接使用世界坐标绝对高度，并关闭 raycast height_scanner
    use_absolute_height = False
    # Optional clamp on the observed height signal before obs/reward/trace consumers use it.
    height_obs_clip_enabled = False
    height_obs_clip_range = [None, None]
    # extra commands
    # use_lin_vel_x_constrain = True # constrain lin vel x between tar and cur
    # lin_vel_x_constrain = 1.

    ''' dof limits '''
    leg_front_rear_range = [0.,1.4] # rad
    soft_range_scale = 1. # [0,1]
    foot_bound_dist = 0.15 # m, 脚端边界距离，超过则进入惩罚区域
    no_fork_distance = 0.05 # m
    no_fork_z_distance = 0.05 # m
    upper_joint_limit = 3.14 # rad
    lower_joint_limit = -3.14 # rad
    max_wheel_torque = 10. # N/m
    mute_wheel_pos_obs = False

    ''' wheel control mode '''
    # True: 轮速控制（policy 输出目标角速度，对目标位置做 damping）
    # False: 力矩控制（policy 输出目标力矩，默认行为）
    use_wheel_vel_control: bool = True
    wheel_vel_action_scale: float = 10.0  # 轮速模式的 action scale（rad/s per unit）
    max_wheel_vel: float = 100.0           # 轮速模式最大轮速限制（rad/s）

    ''' action filter '''
    use_action_low_pass_filter: bool = False
    action_low_pass_prev_weight: float = 0.2
    action_low_pass_curr_weight: float = 0.8

    ''' reset and randomizations '''
    events: EventCfg = EventCfg()

    # randomly spesify leg lengths and leg angles while reseting 
    use_leg_random_start = True 
    links_length = [0.1,0.11814,0.215]
    alpha_offset = [7.48/180.*torch.pi,
                    torch.pi,
                    36.11/180.*torch.pi,
                    (180.+7.48-2*36.11)/180.*torch.pi,
                    (36.11)/180.*torch.pi,
                    36.11/180.*torch.pi]
    leg_length_range = [0.2,0.32]
    leg_angle_range = [-0.25*torch.pi,0.75*torch.pi]
    wheel_angle_range = [-2.*torch.pi,2.*torch.pi]
    use_predefined_leg_random_start = False
    predefined_reset_ground = dict(
        prob = 1.0,
        leg_height = [-0.05,0.15],
        leg_length = [0.16,0.35],
        start_reset_time=2.0,
        start_root_height=0.2,
    )

    # randomly spesify leg and wheel wheels while reseting
    use_joint_vel_random_start = True
    leg_joint_vel_range = [-torch.pi/2.,torch.pi/2.]
    wheel_joint_vel_range = [-50.,50.]
    
    spring_settings = dict(
        mode = 'constant', # 'constant','linear','curve'
        constant_force = 240., # 固定弹簧力
        random_force = [-20.,20.], # 随机弹簧力范围
        spring_offset = 0.058769, # 初始位置弹簧压缩行程mm
        linear_up = 788., # 线性模式下最大压缩弹簧力N
        linear_down = 450., # 线性模式下最小压缩弹簧力N
        linear_length = 0.075, # 线性模式下弹簧力从最小到最大变化的行程m
        damping = False,
        rand_stretch_damping_range = [300,800],
        rand_contract_damping_range = [300,800],
    )

    # random obs and act time lag
    # obs_delay_cfg = {
    #     'root_ang_vel_b':[0,4],
    #     'projected_gravity_b':[0,4],
    #     'joint_pos':[0,4],
    #     'joint_vel':[0,4],
    # }
    # act_delay_cfg = {
    #     'leg_actions':[0,6],
    #     'wheel_actions':[0,6],
    # }

    ''' rewards '''
    lin_vel_err_constraint = None
    lin_vel_xy_sigma = 0.25
    lin_vel_xy_soft_sigma = 1.0
    lin_vel_xy_tight_sigma = 0.01
    lin_vel_xy_torlarance_gap = 0.5
    lin_vel_xy_square_sigma = 0.25
    high_speed_pen_sigma = 1.0
    ang_vel_err_constraint = None
    ang_vel_z_sigma = 0.25
    amg_vel_z_soft_sigma = 1.0
    ang_vel_z_torlarance_gap = 1.0
    ang_vel_z_square_sigma = 0.25
    high_angVel_pen_sigma = 1.0
    height_err_constraint = None
    height_sigma = 0.01
    height_soft_sigma = 0.1
    height_tight_sigma = 0.001
    height_square_sigma = 5.
    height_l1_sigma = 2.
    height_tanh_sigma = 0.02
    height_torlarance_gap = 0.02
    base_height_bound = 0.2
    pen_base_too_low_sigma = 5.
    wheel_motor_z_axis_align_ref_y_offset = 0.217
    wheel_motor_z_axis_align_tolerance = 0.0
    wheel_motor_z_axis_align_sigma = 0.01
    wheel_motor_z_axis_align_tight_sigma = 0.001
    lin_vel_z_sigma = 0.25
    orientation_x_bias = 2.
    orientation_x_sigma = 1.
    orientation_x_A = 2.
    orientation_y_bias = 2.
    orientation_y_sigma = 1.
    orientation_y_A = 2.
    orientation_x_square_sigma = 4.
    orientation_y_square_sigma = 2.
    orientation_y_exp_sigma = 0.01
    orientation_x_exp_sigma = 0.01
    flat_pitch_l1_sigma = 1.0
    flat_roll_l1_sigma = 1.0
    flat_pitch_tanh_sigma = 0.05
    flat_roll_tanh_sigma = 0.05
    foot_bound_square_sigma = 2.
    foot_bound_exp_pen_sigma = 0.2
    foot_bound_exp_sigma = 0.02
    foot_bound_ssquare_sigma = 8.
    no_fork_square_sigma = 2.
    no_fork_exp_sigma = 0.0001
    no_fork_z_square_sigma = 2.
    no_fork_z_exp_sigma = 0.0001
    l_leg_ang_exp_sigma = 0.1
    r_leg_ang_exp_sigma = 0.1

    # 速度/角速度奖励门控：只有接近水平时 tracking 奖励才能高
    # gate = exp(- upright_err / vel_upright_gate_sigma), 其中 upright_err = sum(pgb_xy^2)
    vel_upright_gate_enabled: bool = False
    vel_upright_gate_sigma: float = 0.5
    # 速度/角速度高度门控：直接使用 height tracker 形式 gate=exp(-height_err/sigma)
    # vel_height_gate_tracker_sigma=None 时复用 height_sigma；
    # 参考：0.005(非常严格，稍低就大幅降速奖励) / 0.01(平衡) / 0.02(更宽松)
    vel_height_gate_enabled: bool = False
    vel_height_gate_tracker_sigma: float = 0.02
    # 高度跟随奖励门控：只有接近水平时高度 tracking 奖励才高
    height_upright_gate_enabled: bool = False
    height_upright_gate_sigma: float = 0.5
    stand_still_deadzone_enabled: bool = False
    stand_still_deadzone_threshold: float = 0.2
    rewards = OrderedDict(
        ### alive
        # termination = 0.,
        # epi_len = 0.,

        ### regularization
        # joint_acc = -2.5e-8,
        # leg_joint_acc = -2.5e-8,
        # wheel_acc = -2.5e-9,
        # joint_vel = -5.0e-7,
        # leg_joint_vel = -5.0e-7,
        # wheel_vel = -5.0e-7,
        joint_torque = -1e-5,
        wheel_power = -1e-4,
        lin_vel_z = -0.5,
        # lin_vel_z_exp = 0.5,
        ang_vel_xy = -0.05,
        action_rate = -0.01,
        action_smoothness = -0.05,

        ### tasks
        # flat_orientation = -1.,
        flat_orientation_y = -1.,
        flat_orientation_y_v = -1.,
        # flat_orientation_x = -5.,
        flat_orientation_x_v = -1.,

        track_lin_vel_xy = 1.0,
        track_lin_vel_xy_tight = 0.2,
        track_lin_vel_xy_huge_gap = -1.0,
        track_lin_vel_xy_square = -1.0,
        # penalty_high_speed = -1.0,

        track_ang_vel_z = 0.5,
        # track_ang_vel_z_soft = 0.3,
        # track_ang_vel_z_huge_gap = -1.0,
        track_ang_vel_z_square = -1.0,

        # track_height_square = -1.0,
        track_height_exp = 0.7,
        track_height_exp_soft = 0.0,
        track_height_exp_tight = 0.3,
        track_height_exp_both_wheels_contact = 0.3,
        wheel_motor_z_axis_align_exp = 0.1,
        wheel_motor_z_axis_align_exp_tight = 0.1,

        # stand_nice = -0.0,
        no_fork = -1.0,
        no_fork_z = -2.0,
        # no_fork_exp = -0.5,
        # no_fork_z_exp = -0.5,
        # no_fork_square = -1.0,
        # no_fork_z_square = -1.0,
        # undesired_contact = -10.,
        # desired_contact = -0.0,

        ### limits
        # actions_joint_limits = -0.0,
        # current_joint_limits = -1.0,

        ### standup (仅 enable_standup_training 时生效)
        # standup_height = 0.0,
        # standup_vel_z = -1.0,
        # standup_smoothness = -0.1,
        # standup_joint_torque = -1e-3,
        # standup_wheel_power = -1e-2,
        # standup_leg_joint_acc = -1e-8,
        # standup_wheel_vel = -1e-4,

    )

@configclass
class Wheelbipe25v3FlatEnvCfg(Wheelbipe25v3ReduceSpringFlatEnvCfg):
    events: EventCfg = EventCfg()
    # robot
    robot_cfg: ArticulationCfg = Wheelbipe25_v3_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # act as a constant force spring
    spring_name = robot_cfg.actuators["spring"].joint_names_expr






    

    




    




    

# ==================== RMUC Terrain Configuration ====================

