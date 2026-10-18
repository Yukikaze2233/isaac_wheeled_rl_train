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

from collections import OrderedDict
from dataclasses import field
import copy
import torch

import agent_tasks.manager.mdp.isaaclab as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelCfg, UniformNoiseCfg
from agent_world.assets.wheelbipe_V14 import Wheelbipe_V14_CFG, Wheelbipe_V14_M3508_CFG, Wheelbipe_V14_No_Gimbal_CFG
from agent_world.assets.wheelbipe_V14_2 import Wheelbipe_V14_2_CFG, Wheelbipe_V14_2_NG_CFG, Wheelbipe_V14_2_M3508_CFG
from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env_cfg import EventCfg, Wheelbipe25v3FlatEnvCfg
from agent_tasks.manager.mdp.terrain import TerrainCommandOverrideCfg
from agent_tasks.direct.wheelbipe.wheelbipe_V14.cfg_utils import *


@configclass
class EventCfgV14(EventCfg):
    """Event configuration for the Wheelbipe V14 direct RL environments."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            # "mass_distribution_params": (1.0, 1.4),
            "mass_distribution_params": (0.9, 1.3),
            "operation": "scale",
        },
    )
    # add_gimbal_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["gimbal_yaw_link", "gimbal_pitch_link"]),
    #         "mass_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #     },
    # )
    base_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    base_inertia = None
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    ".*_front1_link",
                    ".*_front2_link",
                    ".*_front3_link",
                    ".*_front4_link",
                    ".*_rear1_link",
                    ".*_rear2_link",
                    ".*_spring1_link",
                    ".*_spring2_link",
                    "gimbal_yaw_link",
                    "gimbal_pitch_link",
                ],
            ),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # add_leg_mass = None
    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    wheels_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    wheels_inertia = None
    # gimbal_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["gimbal_yaw_link", "gimbal_pitch_link"]),
    #         "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
    #     },
    # )
    # gimbal_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["gimbal_yaw_link", "gimbal_pitch_link"]),
    #         "inertia_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )
    # gimbal_inertia = None
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.04, 0.04), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    base_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "static_friction_range": (0.01, 0.1),
            "dynamic_friction_range": (0.01, 0.1),
            "restitution_range": (0.02, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    guide_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_guide_link"),
            "static_friction_range": (0.1, 0.7),
            "dynamic_friction_range": (0.1, 0.7),
            "restitution_range": (0.01, 0.1),
            "num_buckets": 8,
            "make_consistent": True,
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.02, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    # legs_act_joint_frictions = EventTerm(
    #     func=mdp.randomize_joint_parameters_v1,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_rear1_joint", ".*_front1_joint"]),
    #         "static_friction_distribution_params": (0.25, 1.0),
    #         "dynamic_friction_distribution_params": (0.15, 0.6),
    #         "viscous_friction_distribution_params": (0.05, 0.25),
    #         # "armatuleft_wheel_link material=static=1.1905 dynamic=0.9076 restitution=0.1467 contact=|F|=10.43 peak=33.36re_distribution_params": (0.001, 0.003),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )
    leg_front_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_front1_joint"]),
            # "static_friction_distribution_params": (2.0, 4.0),
            # "dynamic_friction_distribution_params": (1.0, 2.0),
            # "viscous_friction_distribution_params": (0.02, 0.1),
            # "armature_distribution_params": (0.001, 0.003),
            # "static_friction_distribution_params": (1.5, 2.0),
            # "dynamic_friction_distribution_params": (1.4, 1.8),
            # "viscous_friction_distribution_params": (0.01, 0.1),
            "static_friction_distribution_params": (0.25, 1.0),
            "dynamic_friction_distribution_params": (0.25, 1.0),
            "viscous_friction_distribution_params": (0.05, 0.2),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    leg_rear_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_rear1_joint"]),
            # "static_friction_distribution_params": (1.5, 3.0),
            # "dynamic_friction_distribution_params": (0.75, 1.5),
            # "viscous_friction_distribution_params": (0.01, 0.1),
            # "armature_distribution_params": (0.001, 0.003),
            # "static_friction_distribution_params": (0.9, 1.4),
            # "dynamic_friction_distribution_params": (0.8, 1.2),
            # "viscous_friction_distribution_params": (0.01, 0.1),
            "static_friction_distribution_params": (0.25, 1.0),
            "dynamic_friction_distribution_params": (0.25, 1.0),
            "viscous_friction_distribution_params": (0.05, 0.2),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    wheel_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
            "static_friction_distribution_params": (0.05, 0.25),
            # "static_friction_distribution_params": (0.025, 0.125),
            # "dynamic_friction_distribution_params": (0.025, 0.125),
            # "dynamic_friction_distribution_params": (0.05, 0.25),
            "viscous_friction_distribution_params": (0.0, 0.01),
            # "viscous_friction_distribution_params": (0.0, 0.005),
            # "armature_distribution_params": (0.00, 0.003),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # wheel_joint_frictions = None
    legs_inact_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_rear2_joint",
                    ".*_front2_joint",
                    ".*_front3_joint",
                    ".*_front4_joint",
                    ".*_spring1_joint",
                ],
            ),
            "static_friction_distribution_params": (0.05, 0.1),
            # "dynamic_friction_distribution_params": (0.025, 0.05),
            # "dynamic_friction_distribution_params": (0.05, 0.1),
            "viscous_friction_distribution_params": (0.01, 0.025),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # spring_frictions = EventTerm(
    #     func=mdp.randomize_joint_parameters_v1,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_spring2_joint"),
    #         "static_friction_distribution_params": (0.1, 1.0),
    #         "viscous_friction_distribution_params": (25., 75.),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )
    gimbal_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gimbal_yaw_joint", "gimbal_pitch_joint"]),
            "static_friction_distribution_params": (0.002, 0.01),
            # "dynamic_friction_distribution_params": (0.001, 0.005),
            # "dynamic_friction_distribution_params": (0.002, 0.01),
            "viscous_friction_distribution_params": (0.002, 0.01),
            # "armature_distribution_params": (0.0, 0.002),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # "stiffness_distribution_params": (0.75, 1.5),
            # "damping_distribution_params": (0.75, 1.5),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    spring_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_spring2_joint"),
            "stiffness_distribution_params": (0.01, 0.01),
            # "damping_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    leg_effort_noise = EventTerm(
        func=mdp.randomize_actuator_effort_output,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_front1_joint",".*_rear1_joint"]),
            # 输出力矩扰动：tau = clip(tau_nominal * scale + bias + N(0, noise_std))
            # 默认配置不改变行为，需要训练时可把 scale/bias/noise_std 的范围放开。
            "effort_scale_distribution_params": (0.8, 1.1),
            "effort_bias_distribution_params": (0.0, 0.0),
            "effort_noise_std_distribution_params": (0.0, 0.0),
            "distribution": "uniform",
        },
    )
    wheel_effort_noise = EventTerm(
        func=mdp.randomize_actuator_effort_output,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
            # 输出力矩扰动：tau = clip(tau_nominal * scale + bias + N(0, noise_std))
            # 默认配置不改变行为，需要训练时可把 scale/bias/noise_std 的范围放开。
            "effort_scale_distribution_params": (0.9, 1.1),
            "effort_bias_distribution_params": (0.0, 0.0),
            "effort_noise_std_distribution_params": (0.0, 0.0),
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_vel_b,
        mode="reset",
        params={
            "pose_range": {
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )
    # base_external_force_torque_xyz = None


@configclass
class EventCfgV14_Play(EventCfgV14):
    """Play-mode event configuration for Wheelbipe V14."""




@configclass
class CurriculumCfgV14:
    """Wheelbipe V14 command curriculum configuration."""

    track_height_progression = CurrTerm(
        func=mdp.RewardWeightProgression,
        params={
            "reward_key": "track_height_exp",
            "num_steps_per_env": 24,
            "window_size": 64,
            "min_stage_episodes": 64,
            "normalize_by_episode_length": True,
            # 最后一阶段再次达标后恢复默认 reward，即 reward_scale 回到 1.0。
            "restore_defaults_on_last_stage_threshold": True,
            "stages": [
                {
                    "reward_weights": {
                        # "flat_orientation_y": -1.0,
                        # "flat_orientation_y_v": -1.0,
                        # "flat_orientation_x": -1.0,
                        # "flat_orientation_x_v": -1.0,
                        "track_height_exp": 1.0,
                        "track_height_exp_tight": 1.0,
                        # "track_height_exp_soft": 2.0,
                        # "track_height_exp_both_wheels_contact": 5.0,
                        # "lin_vel_z": -0.1,
                        # "rear2_rear1_joint_pos_limits": -1.0,
                        # "rear2_rear1_joint_pos_limits_torque": -1.0,
                        # "rear2_rear1_joint_pos_limits_vel": -1.0,
                        # "termination": -100.0,
                    },
                    "reward_scale": {
                        # "action_smoothness_leg": 0.1,
                        # "leg_joint_acc": 0.1,
                        # "action_rate": 0.1,


                    },
                    "threshold": 0.4,
                    "min_episodes": 500,
                },
                {
                    "reward_weights": {
                        # "flat_orientation_y": -1.0,
                        # "flat_orientation_y_v": -1.0,
                        # "flat_orientation_x": -1.0,
                        # "flat_orientation_x_v": -1.0,
                        "track_height_exp": 0.8,
                        "track_height_exp_tight": 0.6,
                        # "track_height_exp_soft": 0.5,
                        # "track_height_exp_both_wheels_contact": 1.0,
                        # "lin_vel_z": -0.5,
                        # "termination": -100.0,
                        # "termination": -10.0,
                    },
                    "reward_scale": {
                        # "action_smoothness_leg": 0.5,
                        # "leg_joint_acc": 0.5,
                        # "action_rate": 0.5,
                    },
                    "threshold": 0.4,
                    "min_episodes": 500,
                },
                # {
                #     "reward_weights": {
                #         "track_height_exp": 0.4,
                #         "track_height_exp_tight": 1.0,
                #         "lin_vel_z": -1.0,
                #     },
                # },
            ],
        },
    )

    base_vertical_assist_force_progression = CurrTerm(
        func=mdp.BaseVerticalAssistForceProgression,
        params={
            "reward_key": "track_height_exp",
            "num_steps_per_env": 24,
            "window_size": 64,
            "min_stage_episodes": 64,
            "normalize_by_episode_length": True,
            "apply_on_compute": True,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "stages": [
                {
                    "force_z": 160.0,
                    "threshold": 0.4,
                    "min_episodes": 500,
                },
                {
                    "force_z": 80.0,
                    "threshold": 0.4,
                    "min_episodes": 500,
                },
                {
                    "force_z": 0.0,
                },
            ],
        },
    )


@configclass
class WheelbipeV14FlatEnvCfg(Wheelbipe25v3FlatEnvCfg):
    """Configuration for the Wheelbipe V14 direct RL environment with flat terrain."""

    # PPO runner 每轮采集步数，用于从 common_step_counter 外推训练 iteration。
    training_progress_steps_per_iteration = 24

    # Temporarily disable domain randomization for V14 training.
    events = EventCfgV14()
    # curriculum = CurriculumCfgV14()
    curriculum = None
    play_keep_done_reset = True
    # reset_heading_axis_aligned_only = True
    robot_cfg: ArticulationCfg = Wheelbipe_V14_2_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()
    # robot_cfg: ArticulationCfg = Wheelbipe_V14_No_Gimbal_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()
    # robot_cfg = Wheelbipe_V14_2_NG_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()

    legs_act_name = robot_cfg.actuators["legs_act"].joint_names_expr
    legs_inact_name = robot_cfg.actuators["legs_inact"].joint_names_expr
    wheel_name = robot_cfg.actuators["wheel"].joint_names_expr
    spring_name = robot_cfg.actuators["spring"].joint_names_expr
    gimbal_yaw_name = robot_cfg.actuators["gimbal_yaw"].joint_names_expr
    gimbal_pitch_name = robot_cfg.actuators["gimbal_pitch"].joint_names_expr
    use_gimbal = True
    # gimbal_yaw_name = None
    # gimbal_pitch_name = None

    gimbal_pitch_target_pos: float = -0.5
    gimbal_yaw_velocity_range: tuple[float, float] = (-torch.pi, torch.pi)
    ordered_leg_joint_names = V14_ORDERED_LEG_JOINT_NAMES
    ordered_leg_body_names = V14_ORDERED_LEG_BODY_NAMES
    # links_length = V14_LINKS_LENGTH
    # alpha_offset = V14_ALPHA_OFFSET

    mute_wheel_pos_obs = True
    default_height_cmd = 0.22

    obs_delay_cfg = {
        "root_ang_vel_b": [1, 4],
        "projected_gravity_b": [1, 4],
        "joint_pos": [1, 4],
        "joint_vel": [1, 4],
    }
    # obs_delay_cfg = {
    #     "root_ang_vel_b": [2, 5],
    #     "projected_gravity_b": [2, 5],
    #     "joint_pos": [2, 5],
    #     "joint_vel": [2, 5],
    # }
    obs_history_len = 10
    obs_default_time_lag = 1
    use_obs_delay = True

    act_delay_cfg = {
        "leg_actions": [1, 3],
        "wheel_actions": [1, 3],
    }
    # act_delay_cfg = {
    #     "leg_actions": [1, 4],
    #     "wheel_actions": [1, 4],
    # }
    use_act_delay = True

    # Temporarily disable observation noise for V14 training.
    ''' noise '''
    self_obs_noise_cfg = {
        'root_ang_vel_b': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.25, n_max=0.25)),
        'projected_gravity_b': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
        'joint_pos': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.025, n_max=0.025)),
        'leg_joint_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.5, n_max=0.5)),
        'wheel_joint_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-1.0, n_max=1.0)),
        'joint_torque': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.25, n_max=0.25)),
        'lin_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
        'height': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01)),
    }

    # self_act_noise_cfg = {
    #     'leg_actions': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.025, n_max=0.025)),
    #     'wheel_actions': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.5, n_max=0.5)),
    # }
    self_act_noise_cfg = None

    spring_settings = dict(
        mode = 'linear', # 'constant','linear','curve'
        constant_force = 240., # 固定弹簧力
        random_force = [-50.,50.], # 随机弹簧力范围
        # spring_offset = 0.03876, # 初始位置弹簧压缩行程m
        spring_offset = 0.06076,
        linear_up = 600.*1.0, # 线性模式下最大压缩弹簧力N
        linear_down = 400.*1.0, # 线性模式下最小压缩弹簧力N
        linear_length = 0.07, # 线性模式下弹簧力从最小到最大变化的行程m
        damping = False,
        rand_stretch_damping_range = [300,800],
        rand_contract_damping_range = [300,800],
    )

    use_leg_length_as_height = False
    height_range = [0.20, 0.42]
    terrain_command_overrides: dict[str, TerrainCommandOverrideCfg] = field(default_factory=dict)
    terrain_command_switch_hold_steps: int = 0
    use_absolute_height = True
    use_wheel_vel_control = True
    front_rear_joint_limit_rewards_enabled = False
    rear2_rear1_joint_limit_lower = -1.0 / 180.0 * torch.pi
    rear2_rear1_joint_limit_upper = 68.5 / 180.0 * torch.pi
    rear2_rear1_joint_limit_boundary_ratio = 0.03
    rear2_rear1_joint_limit_lower_boundary_ratio = 0.03
    rear2_rear1_joint_limit_upper_boundary_ratio = 0.03
    rear2_rear1_joint_limit_vel_threshold = 10.0

    links_length = [0.1134,0.135,0.210]
    alpha_offset = [-6.61/180.*torch.pi,
                    torch.pi,
                    29.7/180.*torch.pi,
                    (180.-6.61-2*29.7)/180.*torch.pi,
                    (29.7)/180.*torch.pi,
                    29.7/180.*torch.pi]
    leg_length_range = [0.13,0.32]
    leg_angle_range = [-0.5*torch.pi,0.75*torch.pi]
    use_predefined_leg_random_start = True
    predefined_reset_ground = copy.deepcopy(V14_PREDEFINED_RESET_GROUND)
    enable_state_machines = False
    termination_duration_enabled = True
    termination_duration_steps = 20
    reset_heading_target_terminate_enabled = False
    reset_heading_target_terminate_threshold_deg = 20.0
    ctrl_mode_obs_enabled = True
    ctrl_mode_obs_dim = 7
    ctrl_mode_obs_layout = (
        "normal",
        "stair",
        "slope",
        "recover",
        "jump",
        "height_target",
        "state_time",
    )
    # Backward-compatible aliases. Prefer ctrl_mode_obs_* for new configs.
    jump_takeoff_extra_obs_enabled = ctrl_mode_obs_enabled
    jump_takeoff_extra_obs_dim = ctrl_mode_obs_dim
    jump_takeoff_extra_obs_layout = ctrl_mode_obs_layout
    jump_takeoff_extra_obs_slope_terrain_names = (
        "slope_for_rm_low",
        "slope_for_rm_high",
        "inv_slope_for_rm_low",
        "inv_slope_for_rm_high",
        "stair_slope_for_rm_low",
        "stair_slope_for_rm_high",
        "inv_stair_slope_for_rm_low",
        "inv_stair_slope_for_rm_high",
        "cliff_inv_stair_slope_for_rm",
        "cliff_inv_stair_slope_tall_for_rm",
    )

    airborne_state_machine_cfg = {
        "enabled": False,
        "allowed_terrain_names": (),
        "not_allowed_terrain_names": (),
        "enter": {
            "wheel_radius": 0.06,
            "body_height_threshold": 0.3,
            "wheel_clearance_threshold": 0.08,
            "duration_s": 0.02,
        },
        "target_height": {
            # 关闭后 airborne 状态机不再修改高度奖励的目标高度，
            # 也不再改写高度奖励使用的参考高度计算口径。
            "enabled": False,
            "bias": 0.12,
            "max": 0.36,
        },
        "landing_trajectory": {
            "enabled": False,
            # 任一轮接触计时达到该阈值后，若还在下落，则初始化一次落地缓冲轨迹。
            "start_wheel_contact_duration_s": 0.02,
            # 固定时长结束时的参考高度；若触发起点低于 target_height + min_height_margin，则不规划。
            "target_height": 0.22,
            # 固定时长结束时的参考 z 速度，默认 0，且不会允许配置为正值。
            "end_vel_z": 0.0,
            "min_height_margin": 0.02,
            # 只有 z 方向速度小于 -min_down_vel 时才触发，避免轻微噪声启动轨迹。
            "min_down_vel": 0.2,
            # 固定二次轨迹总时长；到 duration_s 时刚好到达终点高度和终点速度。
            "duration_s": 0.3,
            # 可选加速度上限；若二次轨迹所需常加速度超过该值，则不启动本次轨迹。
            "max_abs_acc": 30.0,
        },
        "exit": {
            "wheel_contact_force_threshold": 20.0,
            "wheel_contact_height_threshold": 0.15,
            "wheel_contact_duration_s": 0.3,
            "base_contact_force_threshold": 5.0,
            "base_contact_duration_s": 0.25,
            "max_duration_s": 1.,
        },
        "reward_scales": {
            "undesired_contact": 25.0,
            # "flat_orientation_y_v": 2.0,
            # "flat_orientation_x_v": 2.0,
            "foot_bound_square": 1.0,
            # "track_height_exp_tight": 1.0,
            # "track_height_square": 0.0,
            "termination": 3,
        },
        "reward_full": {
        },
        "reward_additions": {
        },
        "terrain_command_resample": {
            "enabled": False,
            # 进入 airborne 时按概率决定是否启用本次 airborne 临时速度命令覆盖。
            # profiles 的 key 默认就是地形名；也可在 profile 内用 terrain_names 指定多个地形。
            "prob": 0.15,
            # True 时，lin_vel_x 重采样符号跟当前目标速度保持一致：
            # 当前目标为正则只采样 >=0，为负则只采样 <=0。
            "lin_vel_x_sign_from_current": True,
            "profiles": {
                "high_stair_for_rm": {
                    "lin_vel_x": [(-1.5, 1.5)],
                    # "lin_vel_x": [(-0., 0.)],
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                },
                "high_speed_stair_for_rm": {
                    "lin_vel_x": [(-1.5, 1.5)],
                    # "lin_vel_x": [(-0., 0.)],
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                },
                "low_speed_stair_for_rm": {
                    "lin_vel_x": [(-1.5, 1.5)],
                    # "lin_vel_x": [(-0., 0.)],
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                },
            },
        },
    }
    jump_takeoff_permission_cfg = {
        # 独立于 special_modes 的跳跃许可层。采样到许可的 env 才允许
        # jump_takeoff_state_machine 的 random/flag/manual trigger 生效。
        "enabled": False,
        # 每次 command resample 时，对所有 env 独立采样许可的比例。
        # 它不占用 special_mode 桶，因此可以和 spin/dash 等模式同时存在。
        "rel_envs": 0.0,
        "iteration_start": 0,
        "iteration_end": -1,
        "steps_per_iteration": 24,
        # None 表示只提供跳跃许可，不覆盖速度命令。
        # 配置后优先级高于 normal/special_mode，低于 terrain command override。
        "ranges": None,
        # None 表示不覆盖 height_cmd。配置后同样低于 terrain height override。
        "height_range": None,
    }
    wheel_forward_scan_cfg = {
        "enabled": False,
        "scan": {
            "forward_offset": 0.5,
        },
        "detect": {
            "step_height_min": 0.12,
            "step_height_max": 0.14,
            "wall_height": 0.14,
        },
        "height_cmd": {
            "bias": 0.16,
            "hold_s": 2.0,
            "max": 0.40,
        },
    }
    undesired_contact_force_threshold = 3.0
    desired_contact_force_threshold = 5.0

    use_frame_stack = False
    num_obs_hist = 1
    num_privileged_obs_hist = 1

    vel_upright_gate_enabled: bool = False
    vel_upright_gate_sigma: float = 0.1
    vel_orientation_y_gate_enabled: bool = False
    vel_orientation_y_gate_full_deg: float = 5.0
    vel_orientation_y_gate_zero_deg: float = 20.0
    vel_height_gate_enabled: bool = False
    vel_height_gate_mode: str = "linear_band"
    vel_height_gate_full_error: float = 0.05
    vel_height_gate_zero_error: float = 0.1
    vel_height_gate_tracker_sigma: float = 0.02
    height_upright_gate_enabled: bool = False
    height_upright_gate_sigma: float = 0.1
    stand_still_deadzone_enabled: bool = True
    stand_still_deadzone_threshold: float = 0.1
    wheel_motor_z_axis_align_ref_y_offset: float = 0.20855
    wheel_motor_z_axis_align_tolerance: float = 0.0
    wheel_motor_z_axis_align_sigma: float = 0.01
    wheel_motor_z_axis_align_tight_sigma: float = 0.001
    play_wheel_motor_z_axis_align_debug: bool = False
    play_wheel_motor_z_axis_align_debug_interval: int = 50
    play_wheel_motor_z_axis_align_debug_env_id: int = 0
    play_wheel_material_debug: bool = True
    play_wheel_material_debug_interval: int = 50
    play_wheel_material_debug_env_id: int = 0

    debug_value_diagnosis: bool = False
    debug_value_diagnosis_interval: int = 50
    debug_value_diagnosis_topk: int = 3
    debug_value_diagnosis_threshold_only: bool = False
    debug_value_diagnosis_thresholds: dict = {
        "reward_total_abs": 1.0,
        "reward_term_abs": 0.5,
        "state_joint_vel_wheel_abs": 120.0,
        "state_root_lin_vel_b_abs": 20.0,
        "state_root_ang_vel_b_abs": 20.0,
        "state_applied_torque_abs": 350.0,
        "obs_policy_abs": 100.0,
        "obs_critic_abs": 100.0,
    }
    obs_input_clip_cfg: dict = V14_BASIC_OBS_CLIP
    # Promote the stable observation scaling from experiment 001 into the default
    # V14 flat task so the policy is not dominated by high-magnitude velocity terms.
    obs_input_scale_enabled: bool = True
    obs_input_scale_streams: tuple[str, ...] = ("policy","critic")
    obs_input_scale_cfg: dict = V14_BASIC_OBS_SCALE
    joint_pos_obs_encoding: str = "raw"
    privileged_extra_obs_enabled: bool = True
    privileged_extra_obs_dim: int = 39
    num_single_privileged_obs = 71
    state_space = 71
    privileged_extra_joint_count: int = 6
    privileged_extra_wheel_count: int = 2
    privileged_extra_body_count: int = 1
    privileged_extra_inertia_body_count: int = 0
    privileged_extra_material_body_count: int = 2
    # privileged extra obs 的实体选择使用名字/正则配置，避免依赖 robot 内部 body/joint 顺序。
    privileged_extra_joint_names: tuple[str, ...] = (
        ".*_rear1_joint",
        ".*_front1_joint",
        ".*_wheel_joint",
    )
    privileged_extra_wheel_body_names: tuple[str, ...] = (
        "left_wheel_link",
        "right_wheel_link",
    )
    privileged_extra_body_names: tuple[str, ...] = (
        "base_link",
        # *V14_ORDERED_LEG_BODY_NAMES,
        # "left_wheel_link",
        # "right_wheel_link",
        # "gimbal_yaw_link",
        # "gimbal_pitch_link",
    )
    privileged_extra_inertia_body_names: tuple[str, ...] = (
        # "base_link",
        # "left_wheel_link",
        # "right_wheel_link",
        # "gimbal_yaw_link",
        # "gimbal_pitch_link",
    )
    privileged_extra_material_body_names: tuple[str, ...] = (
        "left_wheel_link",
        "right_wheel_link",
    )
    obs_input_clip_cfg = V14_EXTRA_OBS_CLIP
    obs_input_scale_cfg = V14_EXTRA_OBS_SCALE
    debug_obs_alert_threshold: float = 120.0
    debug_obs_alert_topk: int = 3
    debug_obs_alert_print_interval: int = 1

    # REWARD MAP V14_FLAT:
    # - 仅直接作用于 WheelbipeV14FlatEnvCfg 及继承后未重写 rewards 的 flat 类任务。
    # - Log 只会显示实际进入 reward_terms 且 cfg.rewards 中存在的键。
    orientation_x_bias = 2.
    orientation_x_sigma = 3.
    orientation_x_A = 2.
    orientation_y_bias = 2.
    orientation_y_sigma = 3.
    orientation_y_A = 2.
    orientation_x_square_sigma = 4.
    orientation_y_square_sigma = 2.
    flat_pitch_tanh_sigma = 0.1
    flat_roll_tanh_sigma = 0.05
    foot_bound_dist = 0.12
    foot_bound_square_sigma = 2.
    foot_bound_exp_pen_sigma = 0.2
    foot_bound_exp_sigma = 0.02
    foot_bound_ssquare_sigma = 8.
    lin_vel_xy_sigma = 0.5
    lin_vel_xy_tight_sigma = 0.1
    lin_vel_xy_soft_sigma = 1.5
    high_speed_pen_sigma = 1.0
    ang_vel_z_sigma = 0.25
    ang_vel_z_square_sigma = 0.5
    high_angVel_pen_sigma = 1.0
    height_sigma = 0.025
    height_square_sigma = 10.
    base_height_bound = 0.2
    pen_base_too_low_sigma = 5.
    orientation_y_exp_sigma = 0.02
    orientation_x_exp_sigma = 0.01
    lin_vel_err_constraint = 1.0
    ang_vel_err_constraint = 0.8
    height_err_constraint = 0.15
    no_fork_square_sigma = 5.
    rewards = OrderedDict(
        termination = -200.,
        leg_joint_acc=-5e-7,
        leg_joint_vel = -5.0e-3,
        leg_joint_pair_pos_diff=-0.0,
        joint_torque=-1e-4,
        wheel_acc=-1e-8,
        wheel_vel=-1e-5,
        wheel_power=-1e-4,
        wheel_air_spin=0.,
        lin_vel_z=-0.5,
        ang_vel_xy=-0.05,
        action_smoothness_leg=-0.05,
        action_rate = -0.01,
        action_smoothness_wheel=-0.01,
        flat_orientation_y=-0.0,
        flat_orientation_y_v=-2.0,
        flat_orientation_y_exp = 1.0,
        # flat_pitch_l1 = -1.0,
        # flat_pitch_tanh = 1.0,
        flat_orientation_x=-0.0,
        flat_orientation_x_v=-2.0,
        flat_orientation_x_exp = 1.0,
        # flat_roll_l1 = -1.0,
        # flat_roll_tanh = 1.0,
        track_lin_vel_xy=1.0,
        track_lin_vel_xy_tight=0.0,
        track_lin_vel_xy_square=-1.0,
        # track_lin_vel_xy_square=-0.1,
        track_ang_vel_z=1.0,
        track_ang_vel_z_square=-1.0,
        # track_ang_vel_z_square=-0.1,
        stand_still_lin_vel=-1.0,
        # stand_still=-2.0,
        stand_still=-0.0,
        track_height_exp=0.0,
        track_height_exp_soft=0.0,
        track_height_exp_tight=1.0,
        track_height_square=-1.0,
        track_height_exp_both_wheels_contact=0.0,
        no_fork = -1.0,
        no_fork_square = -1.0,
        no_fork_exp=-0.0,
        no_fork_z_exp=-0.0,
        undesired_contact=-2.0,
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_v14_flat_runtime_optimizations(self)
        # if getattr(self.terrain, "terrain_type", None) == "plane":
        #     self.terrain = copy.deepcopy(self.terrain)
        #     self.terrain.physics_material = None
        if bool(getattr(self, "use_leg_length_as_height", False)):
            self.use_absolute_height = False
            self.enable_state_machines = False
            self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
            self.airborne_state_machine_cfg["enabled"] = False
            self.wheel_forward_scan_cfg = copy.deepcopy(self.wheel_forward_scan_cfg)
            self.wheel_forward_scan_cfg["enabled"] = False

        self._apply_ctrl_mode_obs_cfg()

        if bool(getattr(self, "use_absolute_height", False)):
            self.height_scanner = None
            _disable_v14_wheel_height_scanners(self)
        else:
            _enable_v14_body_height_scanner(self)
            if bool(self.airborne_state_machine_cfg.get("enabled", False)) or bool(
                getattr(self, "wheel_forward_scan_cfg", {}).get("enabled", False)
            ) or bool(
                getattr(self, "stair_state_machine_cfg", {}).get("enabled", False)
            ) or bool(
                getattr(self, "jump_takeoff_state_machine_cfg", {}).get("enabled", False)
            ):
                _enable_v14_wheel_height_scanners(self)
            else:
                _disable_v14_wheel_height_scanners(self)

        if hasattr(self, "use_frame_stack") and self.use_frame_stack:
            self.observation_space = self.num_obs_hist * getattr(self, "num_single_obs", 28)
        self.state_space = self.num_privileged_obs_hist * getattr(self, "num_single_privileged_obs", 32)

        # self.commands = mdp.UniformVelocityCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(7.0, 15.0),
        #     rel_standing_envs=0.1,
        #     rel_heading_envs=0.5,
        #     heading_command=True,
        #     heading_control_stiffness=1.0,
        #     debug_vis=False,
        #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #         lin_vel_x=(-2.5, 2.5),
        #         lin_vel_y=(0.0, 0.0),
        #         ang_vel_z=(-torch.pi, torch.pi),
        #         heading=(-torch.pi, torch.pi),
        #     ),
        # )

        # ── SpecialModeUniformVelocityCommand ──
        # 各特殊模式按 rel_envs 占非站立环境的比例独立分配（互斥、无优先级）。
        # 每个 env 单独掷 U(0,1) 选桶，兼容逐 env 重采样；模式顺序每次随机打乱。
        # 模式可配置 iteration_start/iteration_end 按训练轮次启停；
        # iteration 由 env 根据 common_step_counter 外推，无需 runner 每轮回调。
        # 模式 ranges 的每个字段支持单区间 ``(low, high)`` 或多区间 ``[(l1,h1), (l2,h2)]``，
        # 多区间时按宽度比例随机选一段再均匀采样。
        self.commands = mdp.SpecialModeUniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 15.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=5.0,
            debug_vis=False,
            special_mode_min_episode_time=5.0,
            special_mode_require_stable=False,
            special_mode_stable_projected_gravity_xy_norm_max=0.5,
            special_mode_stable_root_lin_vel_b_abs_max=3.0,
            special_mode_stable_root_ang_vel_b_abs_max=10.0,
            ranges=mdp.SpecialModeUniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.7, 2.7),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-2.*torch.pi, 2.*torch.pi),
                heading=(-torch.pi, torch.pi),
            ),
            special_modes={
                # 模式0 — 纯自旋：20% 非站立环境
                "spin_low": mdp.SpecialModeEntryCfg(
                    rel_envs=0.15,
                    iteration_start=3000,
                    iteration_end=-1,      # 永不过期
                    disable_jump_takeoff=False,
                    debug_print=False,
                    ranges=mdp.SpecialModeEntryCfg.Ranges(
                        lin_vel_x=(-0.1, 0.1),
                        lin_vel_y=(0.0, 0.0),
                        ang_vel_z=[(2.*torch.pi, 3.25*torch.pi), (-3.25*torch.pi, -2.*torch.pi)],
                    ),
                ),
                "spin_mid": mdp.SpecialModeEntryCfg(
                    rel_envs=0.15,
                    iteration_start=4000,
                    iteration_end=-1,      # 永不过期
                    disable_jump_takeoff=False,
                    debug_print=False,
                    ranges=mdp.SpecialModeEntryCfg.Ranges(
                        lin_vel_x=(-0.1, 0.1),
                        lin_vel_y=(0.0, 0.0),
                        ang_vel_z=[(3.25*torch.pi, 4.5*torch.pi), (-4.5*torch.pi, -3.25*torch.pi)],
                    ),
                ),
                # "spin_high": mdp.SpecialModeEntryCfg(
                #     rel_envs=0.1,
                #     iteration_start=5000,
                #     iteration_end=-1,      # 永不过期
                #     debug_print=False,
                #     ranges=mdp.SpecialModeEntryCfg.Ranges(
                #         lin_vel_x=(-0.1, 0.1),
                #         lin_vel_y=(0.0, 0.0),
                #         ang_vel_z=[(4.5*torch.pi, 5.5*torch.pi), (-4.5*torch.pi, -5.5*torch.pi)],
                #     ),
                # ),
                # 模式1 — 高速前冲/后退：20% 非站立环境
                "dash": mdp.SpecialModeEntryCfg(
                    rel_envs=0.3,
                    iteration_start=2000,
                    iteration_end=-1,
                    disable_jump_takeoff=True,
                    debug_print=False,
                    ranges=mdp.SpecialModeEntryCfg.Ranges(
                        lin_vel_x=[(2.0, 3.0), (-3.0, -2.0)],  # 多区间：向前或向后高速
                        lin_vel_y=(0.0, 0.0),
                        ang_vel_z=[(-2.*torch.pi, 2.*torch.pi)],
                    ),
                ),
            },
        )
        self.height_command_special_modes_cfg = {
            "enabled": False,
            "min_episode_time": 0.0,
            "modes": {
                "height_sine": {
                    "rel_envs": 0.0,
                    "iteration_start": 0,
                    "iteration_end": -1,
                    "height_wave": mdp.HeightWaveCfg(
                        mean=0.3,
                        mean_range=(0.25,0.35),
                        amplitude=0.1,
                        amplitude_range=(0.05,0.1),
                        frequency_hz=1.0,
                        frequency_range_hz=(1.0,2.5),
                        phase=0.0,
                        random_phase=True,
                        clamp_range=(0.20, 0.40),
                    ),
                },
                "height_step": {
                    "rel_envs": 0.0,
                    "iteration_start": 0,
                    "iteration_end": -1,
                    "height_step": mdp.HeightStepCfg(
                        mean=0.3,
                        mean_range=(0.25, 0.35),
                        amplitude=0.05,
                        amplitude_range=(0.1, 0.2),
                        frequency_hz=1.0,
                        frequency_range_hz=(1.0, 2.5),
                        phase=0.0,
                        random_phase=True,
                        clamp_range=(0.20, 0.40),
                    ),
                },
            },
        }

        self.decimation = 4
        self.sim.dt = 1 / 200.0
        self.max_wheel_torque = 20.0

    def _apply_ctrl_mode_obs_cfg(self, enabled: bool | None = None):
        # ctrl_mode_obs 独立于状态机 cfg 生效（jump_takeoff_state_machine_cfg 已移除）。
        extra_obs_cfg: dict = {}
        use_extra_obs = (
            bool(
                getattr(
                    self,
                    "ctrl_mode_obs_enabled",
                    getattr(self, "jump_takeoff_extra_obs_enabled", False),
                )
            )
            if enabled is None
            else bool(enabled)
        )
        extra_obs_dim = int(
            getattr(self, "ctrl_mode_obs_dim", getattr(self, "jump_takeoff_extra_obs_dim", 7))
        )
        extra_obs_layout = tuple(
            getattr(
                self,
                "ctrl_mode_obs_layout",
                getattr(
                    self,
                    "jump_takeoff_extra_obs_layout",
                    (
                        "normal",
                        "stair",
                        "slope",
                        "recover",
                        "jump",
                        "height_target",
                        "state_time",
                    ),
                ),
            )
        )
        extra_obs_cfg.update(
            {
                "enabled": use_extra_obs,
                "dim": extra_obs_dim,
                "layout": extra_obs_layout,
            }
        )
        self.ctrl_mode_obs_enabled = use_extra_obs
        self.ctrl_mode_obs_dim = extra_obs_dim
        self.ctrl_mode_obs_layout = extra_obs_layout
        # Backward-compatible mirrors for older state-machine/config code.
        self.jump_takeoff_extra_obs_enabled = use_extra_obs
        self.jump_takeoff_extra_obs_dim = extra_obs_dim
        self.jump_takeoff_extra_obs_layout = extra_obs_layout
        if not use_extra_obs:
            return

        current_extra_dim = int(getattr(self, "_ctrl_mode_obs_applied_dim", 0))
        extra_obs_delta = max(extra_obs_dim - current_extra_dim, 0)
        self._ctrl_mode_obs_applied_dim = max(current_extra_dim, extra_obs_dim)
        self._jump_takeoff_extra_obs_applied_dim = self._ctrl_mode_obs_applied_dim
        if extra_obs_delta > 0:
            self.num_single_obs = int(getattr(self, "num_single_obs", 28)) + extra_obs_delta
            self.num_single_privileged_obs = (
                int(getattr(self, "num_single_privileged_obs", 32)) + extra_obs_delta
            )
            if isinstance(getattr(self, "observation_space", None), dict):
                observation_space = dict(self.observation_space)
                if "policy" in observation_space:
                    observation_space["policy"] = int(observation_space["policy"]) + extra_obs_delta
                if "critic" in observation_space:
                    observation_space["critic"] = int(observation_space["critic"]) + extra_obs_delta
                self.observation_space = observation_space
            else:
                self.observation_space = int(
                    getattr(self, "observation_space", self.num_single_obs)
                ) + extra_obs_delta
            if isinstance(getattr(self, "state_space", None), dict):
                state_space = dict(self.state_space)
                if "critic" in state_space:
                    state_space["critic"] = int(state_space["critic"]) + extra_obs_delta
                self.state_space = state_space
            elif getattr(self, "state_space", None):
                self.state_space = int(self.state_space) + extra_obs_delta
            else:
                self.state_space = (
                    int(getattr(self, "num_privileged_obs_hist", 1))
                    * self.num_single_privileged_obs
                )
        scale_cfg = dict(getattr(self, "obs_input_scale_cfg", {}))
        scale_cfg["ctrl_mode_obs"] = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]
        scale_cfg.pop("jump_takeoff_obs", None)
        self.obs_input_scale_cfg = scale_cfg

    def _apply_jump_takeoff_extra_obs_cfg(self, enabled: bool | None = None):
        """Backward-compatible alias; ctrl_mode_obs is the canonical name."""
        self._apply_ctrl_mode_obs_cfg(enabled=enabled)


@configclass
class WheelbipeV14FlatEnvCfg_v2(WheelbipeV14FlatEnvCfg):
    """Flat V14 with gimbal-heading PD and gimbal-frame spin/translation commands."""

    gimbal_heading_control_cfg = {
        "enabled": True,
        "target_mode": "sampled",
        "fixed_heading": 0.0,
        "heading_range": (-torch.pi, torch.pi),
        "kp": 20.0,
        "kd": 0.1,
        "kp_range": (20.0, 40.0),
        "kd_range": (0.05, 0.1),
        "randomize_gains": True,
        "gain_distribution": "uniform",
        "max_effort": 2.0,
        "apply_only_in_special_mode": False,
        "special_mode_name": "gimbal_spin_translate",
    }
    gimbal_spin_translate_cfg = {
        "enabled": True,
        "special_mode_name": "gimbal_spin_translate",
        # "lin_vel_yaw_speed_range": [(0.0, 0.04), (0.25, 0.5)],
        "lin_vel_yaw_speed_range": [(0.0, 0.75)],
        "lin_vel_yaw_speed_deadzone": 0.05,
        "lin_vel_yaw_heading_range": (-torch.pi, torch.pi),
        "lin_vel_yaw_height_range": (0.20, 0.40),
        "zero_heading_in_deadzone": False,
        "use_sampled_heading_obs": False,
        "require_heading_control": True,
        "project_to_body_command": False,
    }
    gimbal_spin_suppressed_reward_terms = (
        "track_lin_vel_xy",
        "track_lin_vel_xy_soft",
        "track_lin_vel_xy_tight",
        "track_lin_vel_xy_huge_gap",
        "track_lin_vel_xy_square",
        "stand_still_lin_vel",
    )
    ctrl_mode_obs_layout = (
        "normal_mode_flag",
        "gimbal_spin_translate_mode_flag",
        "gimbal_spin_speed_cmd_yaw",
        "gimbal_spin_sin_heading_cmd_yaw",
        "gimbal_spin_cos_heading_cmd_yaw",
        "gimbal_spin_sin_yaw_joint_angle",
        "gimbal_spin_cos_yaw_joint_angle",
    )
    # gimbal_spin_track_lin_vel_yaw_frame: reward exp(-||v_cmd_yaw - v_meas_yaw||^2 / sigma).
    gimbal_spin_lin_vel_yaw_sigma = 0.25
    # gimbal_spin_track_lin_speed: reward exp(-(speed_cmd - speed_meas)^2 / sigma).
    gimbal_spin_lin_speed_sigma = 0.25
    # gimbal_spin_track_lin_heading: reward exp(-heading_error^2 / sigma), gated by valid speed.
    gimbal_spin_lin_heading_sigma = 0.025
    # Minimum command speed required before applying heading-direction reward/penalty.
    gimbal_spin_heading_cmd_speed_min = 0.1
    # Minimum measured speed required before applying heading-direction reward/penalty.
    gimbal_spin_heading_meas_speed_min = 0.0
    # gimbal_spin_lin_vel_yaw_square: penalty sigma^2 * ||v_cmd_yaw - v_meas_yaw||^2.
    gimbal_spin_lin_vel_yaw_square_sigma = 0.5
    # gimbal_spin_lin_speed_overshoot: penalty max(speed_meas - speed_cmd, 0)^2 * sigma^2.
    gimbal_spin_lin_speed_overshoot_sigma = 0.5
    # gimbal_spin_heading_error_square: penalty sigma^2 * heading_error^2, gated by valid speed.
    gimbal_spin_heading_error_square_sigma = 4.
    # gimbal_spin_stand_still_lin_vel: L1 yaw-link-frame linear velocity penalty when speed_cmd is near zero.
    gimbal_spin_stand_still_speed_threshold = 0.05

    def __post_init__(self):
        super().__post_init__()
        self.robot_cfg = copy.deepcopy(self.robot_cfg)
        gimbal_yaw_actuator = self.robot_cfg.actuators.get("gimbal_yaw", None)
        if gimbal_yaw_actuator is not None:
            gimbal_yaw_actuator.effort_limit = float(self.gimbal_heading_control_cfg.get("max_effort", 5.0))
        self.commands.special_modes['spin_low'].iteration_start = 0
        self.commands.special_modes['spin_low'].rel_envs = 0.1
        self.commands.special_modes['spin_mid'].iteration_start = 0
        self.commands.special_modes['spin_mid'].rel_envs = 0.1
        self.commands.special_modes['dash'].iteration_start = 0
        self.commands.special_modes['dash'].rel_envs = 0.2
        heading_cfg = dict(self.gimbal_heading_control_cfg)
        kp_range = heading_cfg.get("kp_range", None) if bool(heading_cfg.get("randomize_gains", False)) else None
        kd_range = heading_cfg.get("kd_range", None) if bool(heading_cfg.get("randomize_gains", False)) else None
        self.events = copy.deepcopy(self.events)
        self.events.gimbal_heading_pd_gains = EventTerm(
            func=mdp.randomize_gimbal_heading_pd_gains,
            mode="startup",
            params={
                "kp_distribution_params": kp_range,
                "kd_distribution_params": kd_range,
                "distribution": heading_cfg.get("gain_distribution", "uniform"),
            },
        )
        self.commands = copy.deepcopy(self.commands)
        special_modes = getattr(self.commands, "special_modes", {}) or {}
        if not isinstance(special_modes, dict):
            special_modes = {
                f"mode_{idx}": mode_cfg
                for idx, mode_cfg in enumerate(tuple(special_modes))
            }
        else:
            special_modes = dict(special_modes)
        special_modes["gimbal_spin_translate"] = mdp.SpecialModeEntryCfg(
            rel_envs=0.2,
            iteration_start=0,
            iteration_end=-1,
            disable_jump_takeoff=True,
            debug_print=False,
            ranges=mdp.SpecialModeEntryCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=[
                    (2.4 * torch.pi, 3.6 * torch.pi),
                    (-3.6 * torch.pi, -2.4 * torch.pi),
                ],
            ),
        )
        self.commands.special_modes = special_modes
        self.ctrl_mode_obs_layout = tuple(self.ctrl_mode_obs_layout)
        self.jump_takeoff_extra_obs_layout = self.ctrl_mode_obs_layout
        scale_cfg = dict(getattr(self, "obs_input_scale_cfg", {}))
        scale_cfg["ctrl_mode_obs"] = [1.0] * 7
        self.obs_input_scale_cfg = scale_cfg
        self.stand_still_deadzone_enabled = True
        self.rewards["stand_still_lin_vel"] = -1.0
        # self.rewards["track_lin_vel_xy_square"] = -0.1
        # self.rewards["track_ang_vel_z_square"] = -0.1
        self.rewards["gimbal_spin_track_lin_vel_yaw_frame"] = 1.
        self.rewards["gimbal_spin_track_lin_speed"] = 1.
        self.rewards["gimbal_spin_track_lin_heading"] = 5.0
        self.rewards["gimbal_spin_lin_vel_yaw_square"] = -0.2
        self.rewards["gimbal_spin_lin_speed_overshoot"] = -0.
        self.rewards["gimbal_spin_heading_error_square"] = -0.2
        # self.rewards["gimbal_spin_track_lin_heading_v2"] = 5.0
        # self.rewards["gimbal_spin_heading_error_square_v2"] = -0.2
        self.rewards["gimbal_spin_stand_still_lin_vel"] = -1.0


@configclass
class WheelbipeV14FlatEnvCfg_v2_Play(WheelbipeV14FlatEnvCfg_v2):
    """Play config for V14 flat v2 gimbal spin/translate evaluation."""

    events = EventCfgV14_Play()
    curriculum = None
    use_frame_stack = False
    num_obs_hist = 1
    num_privileged_obs_hist = 1

    def __post_init__(self):
        super().__post_init__()
        self.play = True
        self.episode_length_s = 20.0
        self.height_range = [0.2, 0.3]
        self.play_gimbal_spin_translate_debug_vis = True
        self.play_gimbal_spin_translate_marker_height = 0.85
        self.play_gimbal_spin_translate_marker_radius = 0.12

        self.gimbal_heading_control_cfg = dict(self.gimbal_heading_control_cfg)
        self.gimbal_heading_control_cfg['target_mode'] = 'fixed'
        self.gimbal_spin_translate_cfg = dict(self.gimbal_spin_translate_cfg)
        self.gimbal_spin_translate_cfg["enabled"] = True
        self.gimbal_spin_translate_cfg["lin_vel_yaw_speed_range"] = (0.6, 0.6)
        self.gimbal_spin_translate_cfg["lin_vel_yaw_speed_deadzone"] = 0.05
        self.gimbal_spin_translate_cfg["lin_vel_yaw_heading_range"] = (0., 0.)
        self.gimbal_spin_translate_cfg["project_to_body_command"] = False

        self.commands = copy.deepcopy(self.commands)
        special_modes = getattr(self.commands, "special_modes", {}) or {}
        if not isinstance(special_modes, dict):
            special_modes = {
                f"mode_{idx}": mode_cfg
                for idx, mode_cfg in enumerate(tuple(special_modes))
            }
        else:
            special_modes = dict(special_modes)
        for mode_cfg in special_modes.values():
            mode_cfg.rel_envs = 0.0
            mode_cfg.iteration_start = 0
            mode_cfg.iteration_end = -1
        if "gimbal_spin_translate" in special_modes:
            special_modes["gimbal_spin_translate"].rel_envs = 1.0
        self.commands.special_modes = special_modes


def _apply_v14_airborne_landing_precontact_cfg(cfg) -> None:
    """Apply the airborne landing/pre-contact additions shared by flat-v1 and rough-v1."""

    cfg.airborne_state_machine_cfg = copy.deepcopy(cfg.airborne_state_machine_cfg)

    landing_trajectory = copy.deepcopy(
        cfg.airborne_state_machine_cfg.get("landing_trajectory", {})
    )
    landing_trajectory.update(
        {
            "enabled": False,
            "start_wheel_contact_duration_s": 0.02,
            "target_height": 0.24,
            "end_vel_z": 0.0,
            "min_height_margin": 0.02,
            "min_down_vel": 0.2,
            "duration_s": 0.3,
            "max_abs_acc": 30.0,
        }
    )
    cfg.airborne_state_machine_cfg["landing_trajectory"] = landing_trajectory

    reward_additions = dict(cfg.airborne_state_machine_cfg.get("reward_additions", {}))
    reward_additions["airborne_wheel_contact_force_over"] = {
        "type": "wheel_contact_force_over",
        "force_threshold": 300.0,
        "mode": "l1",
        "reduce": "sum",
    }
    reward_additions["airborne_landing_wheel_body_x_positive"] = {
        "type": "landing_wheel_body_x_positive",
        "target_x": 0.03,
        "sigma": 0.03,
        "command_x_min": 1.0,
        "start_wheel_contact_duration_s": 0.02,
        "contact_force_threshold": 1.0,
        "contact_mode": "any_wheel",
        "use_entry_command": True,
    }
    reward_additions["airborne_air_wheel_zero_torque_exp"] = {
        "type": "wheel_zero_torque_exp",
        "sigma": 1.5,
        "before_wheel_contact_duration_s": 0.02,
        "contact_mode": "any_wheel",
    }
    reward_additions["airborne_precontact_wheel_directional_speed"] = {
        "type": "wheel_directional_speed",
        "start": 0.0,
        "full": 10.0,
        "command_x_threshold": 1.0,
        "root_x_threshold": 1.0,
        "reduce": "min",
        "before_wheel_contact_duration_s": 0.02,
        "contact_mode": "any_wheel",
    }
    reward_additions["airborne_precontact_wheel_directional_speed_shortfall"] = {
        "type": "wheel_directional_speed_shortfall",
        "start": 0.0,
        "full": 10.0,
        "command_x_threshold": 1.0,
        "root_x_threshold": 1.0,
        "reduce": "min",
        "require_wheel_contact_timer_started": True,
        "before_wheel_contact_duration_s": 0.05,
        "contact_mode": "any_wheel",
    }
    reward_additions["airborne_landing_wheel_max_contact_force"] = {
        "type": "landing_wheel_max_contact_force",
        "force_start": 200.0,
        "force_full": 400.0,
        "start_wheel_contact_duration_s": 0.02,
        "contact_force_threshold": 1.0,
        "contact_mode": "any_wheel",
    }
    reward_additions["airborne_landing_traj_height"] = {
        "type": "landing_traj_height_exp",
        "sigma": 0.01,
    }
    reward_additions["airborne_landing_traj_vel_z"] = {
        "type": "landing_traj_vel_z_exp",
        "sigma": 0.25,
    }
    cfg.airborne_state_machine_cfg["reward_additions"] = reward_additions

    # 保持 Rough-v1 当前启用的 airborne 权重；其它候选项仅保留 reward_additions，按需再开。
    # cfg.rewards["airborne_wheel_contact_force_over"] = -0.1
    # cfg.rewards["airborne_landing_wheel_body_x_positive"] = 5.0
    cfg.rewards["airborne_air_wheel_zero_torque_exp"] = 20.0
    cfg.rewards["airborne_precontact_wheel_directional_speed"] = 10.0
    cfg.rewards["airborne_precontact_wheel_directional_speed_shortfall"] = -10.0
    # cfg.rewards["airborne_landing_wheel_max_contact_force"] = -10.0
    # cfg.rewards["airborne_landing_traj_height"] = 20.0
    # cfg.rewards["airborne_landing_traj_vel_z"] = 2.0

''' 腾空落地预训练 '''
@configclass
class WheelbipeV14FlatEnvCfg_v1(WheelbipeV14FlatEnvCfg):
    termination_duration_steps = 10
    ctrl_mode_obs_enabled = True
    ctrl_mode_obs_dim = 7
    ctrl_mode_obs_layout = (
        "normal",
        "stair",
        "slope",
        "recover",
        "jump",
        "height_target",
        "state_time",
    )
    jump_takeoff_extra_obs_enabled = False
    jump_takeoff_extra_obs_dim = 7
    jump_takeoff_extra_obs_layout = ctrl_mode_obs_layout
    height_obs_clip_enabled = True
    height_obs_clip_range = [0.05, 0.45]
    rear2_rear1_joint_limit_lower = -1.0 / 180.0 * torch.pi
    rear2_rear1_joint_limit_upper = 68.5 / 180.0 * torch.pi
    rear2_rear1_joint_limit_boundary_ratio = 0.03
    rear2_rear1_joint_limit_lower_boundary_ratio = 0.1
    rear2_rear1_joint_limit_upper_boundary_ratio = 0.05
    rear2_rear1_joint_limit_vel_threshold = 3*torch.pi
    predefined_reset_air = {
        "enabled": True,
        "modes": (
            {
                "name": "air_1",
                "prob": 0.3,
                "iteration_start": 0,
                "iteration_end": -1,
                "pose_range": {
                    "z": (0.12, 0.42),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-torch.pi, torch.pi),
                },
                "velocity_range": {
                    "x": (-2.5, 2.5),
                    "y": (-0.5, 0.5),
                    "z": (-0., 0.),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.5, 0.5),
                },
                "command_limits": {
                    "duration_s": 3.0,
                    "lin_vel_x": (-2.5, 2.5),
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-0.5 * torch.pi, 0.5 * torch.pi),
                    "height": (0.18, 0.43),
                },
                "leg_length_range": (0.15, 0.35),
                "leg_angle_range": (-0.25 * torch.pi, 0.25 * torch.pi),
            }
        ),
    }

    def __post_init__(self):
        super().__post_init__()
        self.use_absolute_height = False
        self.enable_state_machines = True
        self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
        self.airborne_state_machine_cfg["enabled"] = True
        _enable_v14_body_height_scanner(self)
        _enable_v14_wheel_height_scanners(self)
        self.commands.special_modes['spin_low'].iteration_start = 0
        self.commands.special_modes['spin_mid'].iteration_start = 0
        self.commands.special_modes['dash'].iteration_start = 0
        self.commands.special_modes['dash'].rel_envs = 0.2
        self.commands.special_modes["zero_cmd"] = mdp.SpecialModeEntryCfg(
            rel_envs=0.1,
            iteration_start=0,
            iteration_end=-1,
            disable_jump_takeoff=True,
            debug_print=False,
            ranges=mdp.SpecialModeEntryCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )
        self.predefined_reset_ground["modes"]["positive"]["prob"] = 0.2
        self.predefined_reset_ground["modes"]["negative"]["prob"] = 0.1
        reward_scales = dict(self.airborne_state_machine_cfg.get("reward_scales", {}))
        reward_scales["undesired_contact"] = 25.0
        reward_scales["flat_orientation_y_v"] = 0.0
        # reward_scales["flat_orientation_x_v"] = 0.0
        # reward_scales["flat_orientation_y_exp"] = 0.1
        reward_scales["termination"] = 6.0
        # reward_scales["lin_vel_z"] = 0.1
        # extra
        # reward_scales["leg_joint_acc"] = 0.1
        # reward_scales["leg_joint_vel"] = 0.1
        # reward_scales["track_height_exp_tight"] = 0.1
        reward_scales["track_height_square"] = 0.0
        reward_scales["foot_bound_square"] = 0.0
        self.airborne_state_machine_cfg["reward_scales"] = reward_scales
        reward_full = dict(self.airborne_state_machine_cfg.get("reward_full", {}))
        reward_full.update(
            {
                # "track_lin_vel_xy": 0.8,
                # "track_lin_vel_xy_square": 0.0,
                # "track_ang_vel_z": 0.8,
                # "track_ang_vel_z_square": 0.0,
                # "track_height_exp_tight": 0.8,
                # "track_height_square": 0.0,
            }
        )
        self.airborne_state_machine_cfg["reward_full"] = reward_full
        reward_additions = dict(self.airborne_state_machine_cfg.get("reward_additions", {}))
        reward_additions.update(
            {
                "airborne_undesired_contact_force": {
                    "type": "undesired_contact_force",
                    "force_threshold": self.undesired_contact_force_threshold,
                    "mode": "l1",
                },
                "airborne_landing_down_vel": {
                    "type": "negative_lin_vel_z_after_wheel_contact",
                    "start_duration_s": 0.02,
                    "use_world_frame": True,
                    "mode": "l2",
                    "square_sigma": 2.0,
                },
                "airborne_landing_down_vel_exp": {
                    "type": "negative_lin_vel_z_after_wheel_contact_exp",
                    "start_duration_s": 0.02,
                    "use_world_frame": True,
                    "sigma": 0.25,
                },
                "airborne_joint_pos_limits": {
                    "type": "rear2_rear1_joint_pos_limits",
                },
                "airborne_joint_pos_limits_vel_reg": {
                    "type": "rear2_rear1_joint_pos_limits_vel_reg",
                },
                "airborne_leg_length_min": {
                    "type": "leg_retraction",
                    "mode": "below_target_per_leg",
                    "target": 0.25,
                    "before_wheel_contact_duration_s": 0.02,
                    "contact_mode": "any_wheel",
                },
                "airborne_wheel_height_below_base": {
                    "type": "wheel_height_below_base_exp",
                    # target 表示轮子底部相对 base 低多少米，不是轮心高度。
                    "target": 0.34,
                    "sigma": 0.025,
                    "before_wheel_contact_duration_s": 0.02,
                    "contact_mode": "any_wheel",
                },
                "airborne_wheel_heading_x_centering": {
                    "type": "wheel_heading_x_centering",
                    "wheel_contact_duration_s": 0.02,
                    "base_contact_duration_s": 0.02,
                    "wheel_heading_z_max": -0.1,
                    "sigma": 0.02,
                },
                # "airborne_wheel_height_below_base_tight": {
                #     "type": "wheel_height_below_base_exp",
                #     # target 表示轮子底部相对 base 低多少米，不是轮心高度。
                #     "target": 0.22,
                #     "sigma": 0.005,
                #     "before_wheel_contact_duration_s": 0.02,
                #     "contact_mode": "any_wheel",
                # },
                "airborne_low_body_height": {
                    "type": "body_height_below",
                    "threshold": 0.3,
                    "mode": "l2",
                    "square_sigma": 5.0,
                },
                "airborne_body_height_below_binary": {
                    "type": "body_height_below",
                    "threshold": 0.25,
                    "mode": "binary",
                },
            }
        )
        self.airborne_state_machine_cfg["reward_additions"] = reward_additions
        # self.rewards['foot_bound_square'] = -1.0
        self.rewards['rear2_rear1_joint_pos_limits'] = 0.0
        self.rewards['rear2_rear1_joint_pos_limits_torque'] = 0.0
        self.rewards['rear2_rear1_joint_pos_limits_vel'] = 0.0
        # self.rewards['airborne_undesired_contact_force'] = -1.0
        # self.rewards['airborne_landing_down_vel'] = -1.0
        # self.rewards['airborne_landing_down_vel_exp'] = 1.0
        self.rewards['airborne_joint_pos_limits'] = -10.0
        # self.rewards['airborne_joint_pos_limits_vel_reg'] = -100.0
        # self.rewards["airborne_leg_length_min"] = -40.0
        # self.rewards["airborne_wheel_height_below_base"] = 40.0
        self.rewards["airborne_wheel_heading_x_centering"] = 10.0
        # self.rewards["airborne_wheel_height_below_base_tight"] = 10.0
        # self.rewards["airborne_low_body_height"] = -10.0
        # self.rewards["airborne_body_height_below_binary"] = -10.0

        self.rewards["action_rate"] = -0.002
        self.rewards["action_smoothness_leg"] = -0.005
        self.rewards["action_smoothness_wheel"] = -0.001
        self.rewards["leg_joint_acc"] = -1e-7
        self.rewards["leg_joint_vel"] = -1e-3
        self.rewards["wheel_acc"] = -2e-9
        self.rewards["wheel_vel"] = -2e-6

        _apply_v14_airborne_landing_precontact_cfg(self)


''' 小陀螺训练 '''
@configclass
class WheelbipeV14RoughEnvCfg(WheelbipeV14FlatEnvCfg_v2):
# class WheelbipeV14RoughEnvCfg(WheelbipeV14FlatEnvCfg):
    # play_keep_done_reset = True
    rough_terrain_generator_cfg = copy.deepcopy(mdp.RM_ROTATION_TERRAINS_CFG_99)
    rough_terrain_command_overrides_cfg = copy.deepcopy(V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1)
    # rough_height_offset_curriculum_cfg = {
    #     "enabled": True,
    #     "interval": 400,
    #     "max_iteration": 5000,
    #     "num_levels": 11,
    #     "steps_per_iteration": 24,
    #     "random_reset_up_to_current_level": False,
    #     "random_reset_after_max": True,
    #     "randomize_type_on_random_reset": True,
    # }
    rough_terrain_boundary_reset_cfg = {
        "enabled": True,
        "margin": 0.5,
        "use_inner_terrain_area": False,
    }
    predefined_reset_air = {
        "enabled": False,
        "modes": (
            {
                "name": "air_1",
                "prob": 0.2,
                "iteration_start": 0,
                "iteration_end": -1,
                "pose_range": {
                    "z": (0.05, 0.25),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-torch.pi, torch.pi),
                },
                "velocity_range": {
                    "x": (-2.0, 2.0),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.0, 0.0),
                },
                "leg_length_range": (0.20, 0.35),
                "leg_angle_range": (-0.25 * torch.pi, 0.25 * torch.pi),
            },
            {
                "name": "air_2",
                "prob": 0.2,
                "iteration_start": 2000,
                "iteration_end": -1,
                "pose_range": {
                    "z": (0.25, 0.35),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-torch.pi, torch.pi),
                },
                "velocity_range": {
                    "x": (-2.0, 2.0),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
                "leg_length_range": (0.20, 0.35),
                "leg_angle_range": (-0.25 * torch.pi, 0.25 * torch.pi),
            },
        ),
    }

    vel_orientation_y_gate_enabled: bool = False
    vel_height_gate_enabled: bool = False

    def __post_init__(self):
        super().__post_init__()
        # self.predefined_reset_ground['start_root_height'] = 0.25
        # self.predefined_reset_ground['prob'] = 0.3
        _apply_v14_rough_runtime_cfg(self)
        self.enable_state_machines = False
        self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
        self.airborne_state_machine_cfg["enabled"] = False
        self.wheel_forward_scan_cfg = copy.deepcopy(self.wheel_forward_scan_cfg)
        self.wheel_forward_scan_cfg["enabled"] = False
        # _disable_v14_wheel_height_scanners(self)
        # self.height_range = [0.2,0.4]
        self.commands.special_modes['spin_low'].iteration_start = 0
        # self.commands.special_modes['spin_low'].rel_envs = 0.3
        self.commands.special_modes['spin_mid'].iteration_start = 0
        # self.commands.special_modes['spin_mid'].rel_envs = 0.3
        self.commands.special_modes['dash'].iteration_start = 0
        # self.commands.special_modes['dash'].rel_envs = 0.
        # self.commands.special_modes["zero_cmd"] = mdp.SpecialModeEntryCfg(
        #     rel_envs=0.1,
        #     iteration_start=0,
        #     iteration_end=-1,
        #     disable_jump_takeoff=True,
        #     debug_print=False,
        #     ranges=mdp.SpecialModeEntryCfg.Ranges(
        #         lin_vel_x=(0.0, 0.0),
        #         lin_vel_y=(0.0, 0.0),
        #         ang_vel_z=(0.0, 0.0),
        #     ),
        # )
        self.rewards['wheel_power'] = -1e-5
        self.rewards['joint_torque'] = -1e-5
        self.stand_still_deadzone_enabled = True
        self.rewards["stand_still_lin_vel"] = -1.0

''' 跑场训练 '''
@configclass 
class WheelbipeV14RoughEnvCfg_v1(WheelbipeV14FlatEnvCfg_v1):
    rough_terrain_boundary_reset_cfg = {
        "enabled": True,
        "margin": 0.5,
        "use_inner_terrain_area": False,
    }
    def __post_init__(self):
        super().__post_init__()
        # self.predefined_reset_ground['prob'] = 0.3
        self.predefined_reset_air = copy.deepcopy(self.predefined_reset_air)
        self.predefined_reset_air["enabled"] = True
        _apply_v14_rough_runtime_cfg(self)
        # self.commands.resampling_time_range = (3.0, 7.0)
        self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
        terrain_command_resample = copy.deepcopy(
            self.airborne_state_machine_cfg.get("terrain_command_resample", {})
        )
        terrain_command_resample["enabled"] = True
        terrain_command_resample["lin_vel_x_sign_from_current"] = True
        self.airborne_state_machine_cfg["terrain_command_resample"] = terrain_command_resample
        self.rewards['wheel_power'] = -1e-5
        self.rewards['joint_torque'] = -1e-5
        self.rewards['track_lin_vel_xy'] = 1.25

@configclass
class WheelbipeV14FlatEnvCfg_Play(WheelbipeV14FlatEnvCfg):
    events = EventCfgV14_Play()
    curriculum = None
    use_frame_stack = False
    num_obs_hist = 1
    num_privileged_obs_hist = 1

    def __post_init__(self):
        super().__post_init__()
        # self.episode_length_s = 2.0
        # self.episode_length_s = 3.0
        # self.height_range = [0.25, 0.25]
        # self.predefined_reset_ground["modes"]["positive"]["prob"] = 0.
        # self.predefined_reset_ground["modes"]["negative"]["prob"] = 0.
        # self.predefined_reset_air = {
        #     "enabled": True,
        #     "modes": (
        #         {
        #             "name": "play_air",
        #             "prob": 1.0,
        #             "iteration_start": 0,
        #             "iteration_end": -1,
        #             "pose_range": {
        #                 "z": (0.35, 0.35),
        #                 "roll": (-0.05, 0.05),
        #                 "pitch": (-0.1, 0.1),
        #                 "yaw": (-torch.pi, torch.pi),
        #             },
        #             "velocity_range": {
        #                 "x": (2., 2.),
        #                 "y": (-0.25, 0.25),
        #                 "z": (0.0, 0.0),
        #                 "roll": (0.0, 0.0),
        #                 "pitch": (0.0, 0.0),
        #                 "yaw": (-0.0, 0.0),
        #             },
        #             "leg_length_range": (0.20, 0.30),
        #             "leg_angle_range": (-0.2 * torch.pi, 0.2 * torch.pi),
        #         },
        #     ),
        # }
        
        self.play = True
        self.play_height_scanner_debug_vis = True
        self.play_terrain_debug_vis = True
        self.play_wheel_motor_z_axis_align_debug = True
        self.play_wheel_motor_z_axis_align_debug_interval = 50
        self.play_wheel_motor_z_axis_align_debug_env_id = 0
        self.play_wheel_material_debug = False
        self.play_wheel_material_debug_interval = 50
        self.play_wheel_material_debug_env_id = 0

@configclass
class WheelbipeV14FlatDreamWaqEnvCfg(WheelbipeV14FlatEnvCfg):
    """V14 flat experiment: DreamWaQ CENet policy with proprioceptive history."""

    # Disable base Flat's 7D ctrl_mode_obs so obs matches the declared 28/71
    # dims (otherwise obs_history init 28D vs appended 35D -> torch.stack crash).
    ctrl_mode_obs_enabled = False
    # curriculum = CurriculumCfgV14()
    curriculum = None
    use_frame_stack = False
    num_obs_hist = V14_DREAMWAQ_POLICY_HIST
    num_privileged_obs_hist = 1
    n_state_est = V14_DREAMWAQ_ESTIMATED_STATE_DIM
    observation_space = {
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_DREAMWAQ_POLICY_HIST,
        "critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
    }
    state_space = V14_BASE_PRIVILEGED_OBS_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_BASE_PRIVILEGED_OBS_DIM
        self.observation_space = {
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_DREAMWAQ_POLICY_HIST,
            "critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
        }






@configclass
class WheelbipeV14FlatDreamWaqEnvCfg_Play(WheelbipeV14FlatEnvCfg_Play):
    """Play config matching the V14 flat DreamWaQ experiment."""

    ctrl_mode_obs_enabled = False
    use_frame_stack = False
    num_obs_hist = V14_DREAMWAQ_POLICY_HIST
    num_privileged_obs_hist = 1
    n_state_est = V14_DREAMWAQ_ESTIMATED_STATE_DIM
    observation_space = {
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_DREAMWAQ_POLICY_HIST,
        "critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
    }
    state_space = V14_BASE_PRIVILEGED_OBS_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_BASE_PRIVILEGED_OBS_DIM
        self.observation_space = {
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_DREAMWAQ_POLICY_HIST,
            "critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
        }


@configclass
class WheelbipeV14FlatHIMEnvCfg(WheelbipeV14FlatEnvCfg):
    """V14 flat experiment: HIMLoco hybrid internal model with policy history."""

    ctrl_mode_obs_enabled = False
    curriculum = CurriculumCfgV14()
    use_frame_stack = False
    num_obs_hist = V14_HIM_POLICY_HIST
    num_privileged_obs_hist = 1
    n_state_est = V14_HIM_ESTIMATED_STATE_DIM
    observation_space = {
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_HIM_POLICY_HIST,
        "critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
    }
    state_space = V14_BASE_PRIVILEGED_OBS_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_BASE_PRIVILEGED_OBS_DIM
        self.observation_space = {
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_HIM_POLICY_HIST,
            "critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
        }




@configclass
class WheelbipeV14FlatHIMEnvCfg_Play(WheelbipeV14FlatEnvCfg_Play):
    """Play config matching the V14 flat HIMLoco experiment."""

    ctrl_mode_obs_enabled = False
    use_frame_stack = False
    num_obs_hist = V14_HIM_POLICY_HIST
    num_privileged_obs_hist = 1
    n_state_est = V14_HIM_ESTIMATED_STATE_DIM
    observation_space = {
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_HIM_POLICY_HIST,
        "critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
    }
    state_space = V14_BASE_PRIVILEGED_OBS_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_BASE_PRIVILEGED_OBS_DIM
        self.observation_space = {
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_HIM_POLICY_HIST,
            "critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
        }


@configclass
class WheelbipeV14FlatNP3OBarlowEnvCfg(WheelbipeV14FlatEnvCfg):
    """V14 flat experiment: NP3O cost channels with Barlow Twins history actor."""

    ctrl_mode_obs_enabled = False
    # curriculum = CurriculumCfgV14()
    curriculum = None
    np3o_barlow_enabled = True
    use_frame_stack = False
    num_obs_hist = V14_NP3O_POLICY_HIST
    num_privileged_obs_hist = 1
    n_scan = 0
    n_state_est = V14_NP3O_EST_DIM
    n_priv_latent = V14_NP3O_EST_DIM
    num_costs = V14_NP3O_COST_DIM
    np3o_cost_d_values = [0.0, 0.0, 0.0, 0.0, 0.0]
    np3o_cost_k_initial = [1.0, 1.0, 1.0, 0.5, 0.5]
    np3o_tilt_limit_deg = 20.0
    np3o_body_height_min = 0.18
    np3o_body_height_max = 0.42
    np3o_ang_vel_xy_limit = 4.0
    np3o_torque_limit = 30.0
    np3o_joint_velocity_limit = 80.0
    np3o_cost_clip = 100.0
    vel_height_gate_enabled = True
    vel_orientation_y_gate_enabled = False
    observation_space = {
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST,
        "priv_latent": V14_NP3O_EST_DIM,
        "on_constraint": V14_NP3O_ON_CONSTRAINT_DIM,
    }
    state_space = V14_NP3O_ON_CONSTRAINT_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_NP3O_ON_CONSTRAINT_DIM
        self.observation_space = {
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST,
            "priv_latent": V14_NP3O_EST_DIM,
            "on_constraint": V14_NP3O_ON_CONSTRAINT_DIM,
        }
        # self.rewards = copy.deepcopy(self.rewards)
        # self.rewards.update(
        #     {
        #         "flat_orientation_y": -0.0,
        #         "flat_orientation_y_v": -1.0,
        #         "flat_orientation_x": -0.0,
        #         "flat_orientation_x_v": -0.5,
        #         "ang_vel_xy": -0.002,
        #         "lin_vel_z": -0.2,
        #         "joint_torque": -2.0e-5,
        #         "leg_joint_vel": -1.0e-3,
        #     }
        # )




        

@configclass
class WheelbipeV14FlatNP3OBarlowEnvCfg_Play(WheelbipeV14FlatEnvCfg_Play):
    """Play config matching the V14 flat NP3O + Barlow Twins experiment."""

    ctrl_mode_obs_enabled = False
    np3o_barlow_enabled = True
    use_frame_stack = False
    num_obs_hist = V14_NP3O_POLICY_HIST
    num_privileged_obs_hist = 1
    n_scan = 0
    n_state_est = V14_NP3O_EST_DIM
    n_priv_latent = V14_NP3O_EST_DIM
    num_costs = V14_NP3O_COST_DIM
    np3o_cost_d_values = [0.0, 0.0, 0.0, 0.0, 0.0]
    np3o_cost_k_initial = [1.0, 1.0, 1.0, 0.5, 0.5]
    np3o_tilt_limit_deg = 15.0
    np3o_body_height_min = 0.18
    np3o_body_height_max = 0.42
    np3o_ang_vel_xy_limit = 4.0
    np3o_torque_limit = 30.0
    np3o_joint_velocity_limit = 80.0
    np3o_cost_clip = 100.0
    observation_space = {
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST,
        "priv_latent": V14_NP3O_EST_DIM,
        "on_constraint": V14_NP3O_ON_CONSTRAINT_DIM,
    }
    state_space = V14_NP3O_ON_CONSTRAINT_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_NP3O_ON_CONSTRAINT_DIM
        self.observation_space = {
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST,
            "priv_latent": V14_NP3O_EST_DIM,
            "on_constraint": V14_NP3O_ON_CONSTRAINT_DIM,
        }

@configclass
class WheelbipeV14RoughEnvCfg_Play(WheelbipeV14RoughEnvCfg):
    rough_height_offset_curriculum_cfg = {
        **V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG,
        "enabled": False,
    }
    rough_terrain_boundary_reset_cfg = {
        "enabled": True,
        "margin": 0.5,
        "use_inner_terrain_area": True,
    }
    
    curriculum = None
    use_frame_stack = False
    num_obs_hist = 1
    num_privileged_obs_hist = 1
    episode_length_s = 5.0
    events = EventCfgV14_Play()

    def __post_init__(self):
        super().__post_init__()
        play_terrain_name = "cliff_inv_stair_slope_short_for_rm_play"
        self.predefined_reset_ground['prob'] = 0.0
        self.predefined_reset_air = {
            "enabled": True,
            "modes": (
                {
                    "name": "play_air",
                    "prob": 1.0,
                    "iteration_start": 0,
                    "iteration_end": -1,
                    "pose_range": {
                        "z": (0.25, 0.25),
                        "roll": (-0.05, 0.05),
                        "pitch": (-0.1, 0.1),
                        "yaw": (-torch.pi, torch.pi),
                    },
                    "velocity_range": {
                        "x": (1.9, 2.0),
                        "y": (-0.25, 0.25),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (-0.0, 0.0),
                    },
                    "leg_length_range": (0.20, 0.35),
                    "leg_angle_range": (-0.2 * torch.pi, 0.2 * torch.pi),
                },
            ),
        }
        self.episode_length_s = 5.
        _play_terrain_gen = copy.deepcopy(mdp.RM_ROUGH_TERRAINS_PLAY_CFG)
        if len(getattr(_play_terrain_gen, "sub_terrains", {}) or {}) == 0:
            _play_terrain_gen.sub_terrains = {
                play_terrain_name: mdp.CLIFF_INV_STAIR_SLOPE_SHORT_FOR_RM_PLAY,
            }
        _play_terrain_gen.num_rows = 10
        _play_terrain_gen.curriculum = True
        self.terrain_command_overrides = _filter_v14_terrain_command_overrides(
            V14_ROUGH_TERRAIN_COMMAND_OVERRIDES, _play_terrain_gen
        )
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            collision_group=-1,
            terrain_generator=_play_terrain_gen,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        # self.height_range = [0.25, 0.35]
        self.events.robot_joint_stiffness_and_damping = None
        self.commands = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5, 15),
            rel_standing_envs=0.0,
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(2.2, 2.2),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-torch.pi, torch.pi),
                heading=(-torch.pi, torch.pi),
            ),
        )
        self.play = False
        self.play_height_scanner_debug_vis = True
        self.play_terrain_debug_vis = True
        self.velocity_trace_cfg = {
            "enabled": True,
            "terrain_name": play_terrain_name,
            "agent_index": None,
            "lock_agent": True,
            "sample_dt": 0.02,
            "html_update_interval_s": 1.0,
            "max_rows": 20000,
            "unique_path": True,
            "csv_path": "logs/debug/rough_play_velocity_trace.csv",
            "html_path": "logs/debug/rough_play_velocity_trace.html",
        }

@configclass
class WheelbipeV14RoughEnvCfg_v1_Play(WheelbipeV14RoughEnvCfg_v1):
    ctrl_mode_obs_enabled = True
    def __post_init__(self):
        super().__post_init__()
        play_terrain_name = "cliff_inv_stair_slope_short_for_rm_play"
        self.episode_length_s = 5
        self.height_range = [0.25,0.25]
        self.play = True
        self.play_ang_vel_z_debug_vis = False
        self.play_height_scanner_debug_vis = True
        self.play_terrain_debug_vis = True
        self.velocity_trace_cfg = {
            "enabled": True,
            "terrain_name": play_terrain_name,
            "agent_index": None,
            "lock_agent": True,
            "sample_dt": 0.02,
            "html_update_interval_s": 1.0,
            "max_rows": 20000,
            "unique_path": True,
            "csv_path": "logs/debug/730/rough_v1_play_velocity_trace.csv",
            "html_path": "logs/debug/730/rough_v1_play_velocity_trace.html",
        }
        _play_terrain_gen = copy.deepcopy(mdp.RM_ROUGH_TERRAINS_PLAY_CFG)
        _play_terrain_gen.curriculum = True
        self.terrain_command_overrides = _filter_v14_terrain_command_overrides(
            V14_ROUGH_TERRAIN_COMMAND_OVERRIDES, _play_terrain_gen
        )
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            collision_group=-1,
            terrain_generator=_play_terrain_gen,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )


