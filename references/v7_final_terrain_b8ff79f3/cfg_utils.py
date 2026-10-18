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
import copy
import torch

import agent_tasks.manager.mdp.isaaclab as mdp
import isaaclab.sim as sim_utils
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from agent_tasks.manager.mdp.terrain import TerrainCommandOverrideCfg


V14_TASK_FLAG_DIM = 1
V14_BASE_POLICY_OBS_DIM = 28
V14_BASE_PRIVILEGED_OBS_DIM = 32
V14_PRIV_OBS_PLUS_EXTRA_DIM = 85 + 44
V14_PRIV_OBS_PLUS_PRIVILEGED_DIM = V14_BASE_PRIVILEGED_OBS_DIM + V14_PRIV_OBS_PLUS_EXTRA_DIM
V14_JOINT_SINCOS_EXTRA_DIM = 6
V14_JOINT_SINCOS_POLICY_OBS_DIM = V14_BASE_POLICY_OBS_DIM + V14_JOINT_SINCOS_EXTRA_DIM
V14_JOINT_SINCOS_PRIVILEGED_OBS_DIM = V14_BASE_PRIVILEGED_OBS_DIM + V14_JOINT_SINCOS_EXTRA_DIM
V14_JOINT_SINCOS_ACT_SINCOS_ACTION_DIM = 10
V14_JOINT_SINCOS_ACT_SINCOS_EXTRA_DIM = 4
V14_JOINT_SINCOS_ACT_SINCOS_POLICY_OBS_DIM = (
    V14_JOINT_SINCOS_POLICY_OBS_DIM + V14_JOINT_SINCOS_ACT_SINCOS_EXTRA_DIM
)
V14_JOINT_SINCOS_ACT_SINCOS_PRIVILEGED_OBS_DIM = (
    V14_JOINT_SINCOS_PRIVILEGED_OBS_DIM + V14_JOINT_SINCOS_ACT_SINCOS_EXTRA_DIM
)
V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG = {
    "enabled": False,
    "interval": 500,
    "max_iteration": 5000,
    "num_levels": 11,
    "steps_per_iteration": 24,
    "random_reset_up_to_current_level": False,
    "random_reset_after_max": True,
    "randomize_type_on_random_reset": True,
}
V14_DREAMWAQ_ESTIMATED_STATE_DIM = 4
V14_DREAMWAQ_POLICY_HIST = 5
V14_SEMA_ESTIMATED_STATE_DIM = 6
V14_SEMA_POLICY_HIST = 10
V14_RESIDUAL_ESTIMATED_STATE_DIM = 4
V14_RESIDUAL_POLICY_HIST = 5
V14_HIM_ESTIMATED_STATE_DIM = 4
V14_HIM_POLICY_HIST = 5
V14_NP3O_POLICY_HIST = 10
V14_NP3O_EST_DIM = 4
V14_NP3O_COST_DIM = 5
V14_NP3O_ON_CONSTRAINT_DIM = (
    V14_BASE_PRIVILEGED_OBS_DIM
    + V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST
)
V14_NP3O_ON_CONSTRAINT_EXTRA_PRIV_DIM = (
    V14_BASE_PRIVILEGED_OBS_DIM
    + V14_PRIV_OBS_PLUS_EXTRA_DIM
    + V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST
)
V14_BODY_HEIGHT_SCANNER_GRID_SIZE = (0.02, 0.02)
V14_BODY_HEIGHT_SCANNER_RESOLUTION = 0.01
V14_WHEEL_HEIGHT_SCANNER_GRID_SIZE = (0.015, 0.015)
V14_WHEEL_HEIGHT_SCANNER_RESOLUTION = 0.01

V14_BASIC_OBS_CLIP: dict = {
    "command": 100.0,
    "height_cmd": [0.0, 1.0],
    "root_ang_vel_b": 100.0,
    "projected_gravity_b": 100.0,
    "joint_pos": 100.0,
    "joint_vel_leg": 200.0,
    "joint_vel_wheel": 200.0,
    "actions": 100.0,
    "root_lin_vel_b": 100.0,
    "obs_height": [-10.0, 10.0],
}

V14_BASIC_OBS_SCALE: dict = {
    "command": {
        "lin_vel_x": 1.0,
        "lin_vel_y": 1.0,
        "ang_vel_z": 1.0,
    },
    "height_cmd": 5.0,
    "root_ang_vel_b": 0.5,
    "joint_vel_leg": 0.1,
    "joint_vel_wheel": 0.1,
    "obs_height": 5.0,
}

V14_EXTRA_OBS_CLIP = copy.deepcopy(V14_BASIC_OBS_CLIP)
V14_EXTRA_OBS_CLIP.update(
    {
        "spring_force": 1000.,
        # "root_pos_w": 100.0,
        # "root_lin_vel_w": 100.0,
        # "root_ang_vel_w": 200.0,
        "joint_torque": 100.0,
        "obs_delay_steps": 100.0,
        "act_delay_steps": 100.0,
        "joint_acc": 1000.0,
        "wheel_body_lin_vel": 100.0,
        "wheel_contact_force": 5000.0,
        "wheel_contact_state": 1.0,
        "joint_stiffness": 100.0,
        "joint_damping": 100.0,
        "joint_friction": 100.0,
        "body_mass": 100.0,
        "body_mass_scale": 10.0,
        "body_inertia_diag": 100.0,
        "body_material": 100.0,
        "body_com": 100.0,
    }
)

V14_EXTRA_OBS_SCALE = copy.deepcopy(V14_BASIC_OBS_SCALE)
V14_EXTRA_OBS_SCALE.update(
    {
        "spring_force": 0.01,
        # "root_pos_w": 0.1,
        # "root_lin_vel_w": 0.25,
        # "root_ang_vel_w": 0.25,
        "joint_torque": 0.05,
        "obs_delay_steps": 1.0,
        "act_delay_steps": 1.0,
        "joint_acc": 0.01,
        "wheel_body_lin_vel": 1.0,
        "wheel_contact_force": 0.01,
        "wheel_contact_state": 1.0,
        "joint_stiffness": 1.0,
        "joint_damping": 1.0,
        "joint_friction": 1.0,
        "body_mass": 0.1,
        "body_mass_scale": 1.0,
        "body_inertia_diag": 1.0,
        "body_material": 1.0,
        "body_com": 1.0,
    }
)

V14_ORDERED_LEG_JOINT_NAMES: tuple[str, ...] = (
    "left_front1_joint",
    "right_front1_joint",
    "left_rear1_joint",
    "right_rear1_joint",
    "left_front2_joint",
    "right_front2_joint",
    "left_front3_joint",
    "right_front3_joint",
    "left_front4_joint",
    "right_front4_joint",
    "left_rear2_joint",
    "right_rear2_joint",
)

V14_ORDERED_LEG_BODY_NAMES: tuple[str, ...] = (
    "left_front1_link",
    "right_front1_link",
    "left_rear1_link",
    "right_rear1_link",
    "left_front2_link",
    "right_front2_link",
    "left_front3_link",
    "right_front3_link",
    "left_front4_link",
    "right_front4_link",
    "left_rear2_link",
    "right_rear2_link",
)

V14_LINKS_LENGTH = [
    0.11340111711971801,
    0.13499721265641007,
    0.21,
]

_V14_ALPHA0_DEG = 9.430885159953315
_V14_ALPHA2_DEG = 37.874675469056214
V14_ALPHA_OFFSET = [
    _V14_ALPHA0_DEG / 180.0 * torch.pi,
    torch.pi,
    _V14_ALPHA2_DEG / 180.0 * torch.pi,
    (180.0 + _V14_ALPHA0_DEG - 2.0 * _V14_ALPHA2_DEG) / 180.0 * torch.pi,
    (20.0 + _V14_ALPHA2_DEG) / 180.0 * torch.pi,
    _V14_ALPHA2_DEG / 180.0 * torch.pi,
]

V14_PREDEFINED_RESET_GROUND = dict(
    modes={
        "positive": dict(
            prob=0.3,
            sign=1.0,
            leg_height=[-0.06, 0.12],
            leg_length=[0.14, 0.36],
        ),
        "negative": dict(
            prob=0.2,
            sign=-1.0,
            leg_height=[-0.06, 0.],
            leg_length=[0.14, 0.36],
        ),
    },
    # Backward-compatible fields used only when modes is absent.
    prob=0.2,
    leg_height=[-0.06, 0.12],
    leg_length=[0.14, 0.36],
    # Env-level reset settings shared by all ground reset leg modes.
    start_reset_time=1.5,
    start_root_height=0.25,
    zero_torque_time_s=0.2,
    command_ranges={
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    },
)

V14_LEGACY_OBS_INPUT_SCALE_CFG = {
    "command": 1.0,
    "height_cmd": 5.0,
    "root_ang_vel_b": 1.0,
    "joint_pos": 1.0,
    "joint_vel_leg": 0.1,
    "joint_vel_wheel": 0.1,
    "root_lin_vel_b": 1.0,
    "obs_height": 5.0,
    "root_lin_vel_w": 1.0,
    "root_ang_vel_w": 1.0,
    "wheel_body_lin_vel": 1.0,
    "joint_torque": 0.1,
}

V14_ROUGH_TERRAIN_COMMAND_OVERRIDES: dict[str, TerrainCommandOverrideCfg] = {
    # "cliff_inv_stair_slope_flat_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.2, 0.4)],
    #     lin_vel_x=[(-2.5, 2.5)],
    #     lin_vel_y=(0.0, 0.0),
    # ),
    # "cliff_inv_stair_slope_for_rm1": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(1.0, 2.5), (-2.5, -1.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.01, 0.01),
    #     reset_heading_axis_aligned_only=True,
    # ),
    # "cliff_inv_stair_slope_for_rm2": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(1.0, 2.5), (-2.5, -1.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.01, 0.01),
    #     reset_heading_axis_aligned_only=True,
    # ),
    # "cliff_inv_stair_slope_for_rm3": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(2.0, 2.5), (-2.5, -2.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.01, 0.01),
    #     reset_heading_axis_aligned_only=True,
    # ),
    # "cliff_inv_stair_slope_for_rm4": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(2.0, 2.5), (-2.5, -2.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.01, 0.01),
    #     reset_heading_axis_aligned_only=True,
    # ),
    "high_stair_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.34)],
        lin_vel_x=[(1.5, 2.7), (-2.7, -1.5)],
        lin_vel_y=(0.0, 0.0),
        ang_vel_z_heading=(-1, 1),
        ang_vel_z_non_heading=(-0.1, 0.1),
        reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        disable_special_mode=True,
    ),
    "high_speed_stair_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.34)],
        lin_vel_x=[(1.5, 2.7), (-2.7, -1.5)],
        lin_vel_y=(0.0, 0.0),
        # ang_vel_z_heading=(-1, 1),
        ang_vel_z_non_heading=(-0.1, 0.1),
        reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        disable_special_mode=True,
    ),
    "low_speed_stair_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.34)],
        # lin_vel_x=[(2.0, 3.0), (-3.0, -2.0)],
        lin_vel_x=[(0.5, 1.5), (-1.5, -0.5)],
        lin_vel_y=(0.0, 0.0),
        # ang_vel_z_heading=(-1, 1),
        ang_vel_z_non_heading=(-0.1, 0.1),
        reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        disable_special_mode=True,
    ),
    # "inv_pyramid_stair_slope_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.40)],
    #     lin_vel_x=(-2.5, 2.5),
    #     lin_vel_y=(0.0, 0.0),
    # ),
    # "plane_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.20, 0.40)],
    #     lin_vel_x=(-0.0, 0.0),
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=[(-12, -3), (3, 12)],
    #     reset_heading_axis_aligned_only=True,
    # ),
    # "stair_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.37, 0.40)],
    #     lin_vel_x=[(1.0, 2.5)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.5, 0.5),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    #     disable_special_mode=True,
    # ),
    # "inv_stair_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.37, 0.40)],
    #     lin_vel_x=[(1.0, 2.5)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.1, 0.1),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    #     disable_special_mode=True,
    # ),
    # 'tiny_step': TerrainCommandOverrideCfg(
    #     height_range=[(0.20, 0.30)],
    #     lin_vel_x=[(1.5, 2.5), (-2.5, -1.5)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-1.0, 1.0),
    # ),
    # "cliff_inv_stair_slope_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(2.0, 2.5), (-2.5, -2.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.1, 0.1),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    # ),
    # "cliff_inv_stair_slope_tall_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.28, 0.35)],
    #     lin_vel_x=[(2.0, 2.5), (-2.5, -2.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.1, 0.1),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    #     disable_special_mode=True,
    # ),
    "cliff_inv_stair_slope_short_for_rm_play": TerrainCommandOverrideCfg(
        height_range=[(0.3, 0.3)],
        lin_vel_x=[(2.5, 2.5)],
        lin_vel_y=(0.0, 0.0),
        # ang_vel_z_heading=(-0.1, 0.1),
        ang_vel_z_non_heading=(-0., 0.),
        reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        disable_predefined_reset_ground=True,
        disable_special_mode=True,
    ),
    "cliff_inv_stair_slope_short_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.32)],
        lin_vel_x=[(2.0, 2.7), (-2.7, -2.0)],
        lin_vel_y=(0.0, 0.0),
        # ang_vel_z_heading=(-1, 1),
        ang_vel_z_non_heading=(-0.1, 0.1),
        reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        # disable_predefined_reset_ground=True,
        disable_special_mode=True,
    ),
    # "cliff_inv_stair_slope_long_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.34)],
    #     lin_vel_x=[(1.5, 2.6), (-2.6, -1.5)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0., 0.),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    #     disable_predefined_reset_ground=True,
    #     disable_special_mode=True,
    # ),
    'slope_for_rm_low': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    'slope_for_rm_high': TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.32)],
        lin_vel_x=[(-2.5, 2.5)],
        lin_vel_y=(0.0, 0.0),
        # reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        # disable_special_mode=True,
    ),
    'inv_slope_for_rm_low': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    'inv_slope_for_rm_high': TerrainCommandOverrideCfg(
        height_range=[(0.25, 0.34)],
        lin_vel_x=[(-2.5, 2.5)],
        lin_vel_y=(0.0, 0.0),
        disable_predefined_reset_air = True,
        disable_special_mode=True,
    ),
    # 'stair_slope_for_rm_low': TerrainCommandOverrideCfg(
    #     height_range=[(0.22, 0.42)],
    # ),
    'stair_slope_for_rm_high': TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.32)],
        lin_vel_x=[(-2.5, 2.5)],
        lin_vel_y=(0.0, 0.0),
        # ang_vel_z_heading=(-1, 1),
        ang_vel_z_non_heading=(-torch.pi, torch.pi),
        disable_predefined_reset_air = True,
        # disable_special_mode=True,
    ),
    'inv_stair_slope_for_rm_low': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
        # height_range=[(0.24, 0.38)],
    ),
    'inv_stair_slope_for_rm_high': TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.32)],
        lin_vel_x=[(1.5, 2.7), (-2.7, -1.5)],
        lin_vel_y=(0.0, 0.0),
        ang_vel_z_non_heading=(-0.1, 0.1),
        reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        disable_special_mode=True,
    ),
    'random_uniform_for_rm': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    'tiny_step': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.32)],
        disable_predefined_reset_air = True,
        # disable_predefined_reset_ground=True,
    ),
    # "fort_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.24, 0.4)],
    # ),
}

V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1: dict[str, TerrainCommandOverrideCfg] = {
    "tiny_step_rot": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "inv_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "stair_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "stair_slope_for_rm_mid": TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.38)],
    ),
    "inv_stair_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
        # height_range=[(0.24, 0.38)],
    ),
    "plane_for_rm_rot": TerrainCommandOverrideCfg(
        height_range=[(0.20, 0.42)],
    ),
    "random_uniform_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "stair_for_rm_rot": TerrainCommandOverrideCfg(
        height_range=[(0.2, 0.42)],
        lin_vel_x=[(0.5, 2.0), (-2.0, -0.5)],
        lin_vel_y=(0.0, 0.0),
        # ang_vel_z_heading=(-1, 1),
        ang_vel_z_non_heading=(-1., 1.),
        reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        disable_special_mode=True,
    ),
}

V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_2: dict[str, TerrainCommandOverrideCfg] = {
    "tiny_step_rot": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "inv_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "stair_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "stair_slope_for_rm_mid": TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.4)],
    ),
    "inv_stair_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "plane_for_rm_rot": TerrainCommandOverrideCfg(
        height_range=[(0.20, 0.42)],
    ),
    "random_uniform_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],
    ),
    "fort_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.4)],
    ),
    "stair_for_rm_rot": TerrainCommandOverrideCfg(
        height_range=[(0.2, 0.42)],
        lin_vel_x=[(0.5, 2.0), (-2.0, -0.5)],
        lin_vel_y=(0.0, 0.0),
        # ang_vel_z_heading=(-1, 1),
        ang_vel_z_non_heading=(-1., 1.),
        reset_heading_axis_aligned_only=True,
        disable_predefined_reset_air = True,
        disable_special_mode=True,
    ),
}


def _enable_v14_rough_height_offset_curriculum(terrain_gen, num_levels: int) -> None:
    """Enable row-bucket scaling for cliff height offsets in V14 rough terrain."""

    terrain_gen.num_rows = int(num_levels)
    for sub_cfg in terrain_gen.sub_terrains.values():
        height_offset_range = getattr(sub_cfg, "height_offset_range", None)
        if height_offset_range is None:
            continue
        if not any(abs(float(value)) > 1.0e-9 for value in height_offset_range):
            continue
        if hasattr(sub_cfg, "height_offset_curriculum_scale_by_difficulty"):
            sub_cfg.height_offset_curriculum_scale_by_difficulty = True
            sub_cfg.height_offset_curriculum_num_levels = int(num_levels)


def _filter_v14_terrain_command_overrides(
    overrides: dict[str, TerrainCommandOverrideCfg],
    terrain_gen,
) -> dict[str, TerrainCommandOverrideCfg]:
    """Keep only command overrides matching the active V14 rough terrain keys."""

    terrain_keys = set(getattr(terrain_gen, "sub_terrains", {}).keys())
    return copy.deepcopy({key: value for key, value in overrides.items() if key in terrain_keys})


def _apply_v14_rough_runtime_cfg(cfg) -> None:
    """Apply the shared V14 rough-terrain configuration to a composed env cfg."""

    cfg.use_leg_length_as_height = False
    cfg.use_absolute_height = False
    cfg.enable_state_machines = True
    cfg.airborne_state_machine_cfg = copy.deepcopy(cfg.airborne_state_machine_cfg)
    cfg.airborne_state_machine_cfg["enabled"] = True
    cfg.wheel_forward_scan_cfg = copy.deepcopy(cfg.wheel_forward_scan_cfg)
    cfg.wheel_forward_scan_cfg["enabled"] = True
    cfg.curriculum = None

    cfg.termination_duration_enabled = True

    rough_height_offset_curriculum_cfg = dict(V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG)
    raw_rough_height_offset_curriculum_cfg = getattr(cfg, "rough_height_offset_curriculum_cfg", None)
    if raw_rough_height_offset_curriculum_cfg is not None:
        rough_height_offset_curriculum_cfg.update(raw_rough_height_offset_curriculum_cfg)
    cfg.rough_height_offset_curriculum_cfg = rough_height_offset_curriculum_cfg

    raw_rough_terrain_gen = getattr(cfg, "rough_terrain_generator_cfg", None)
    if raw_rough_terrain_gen is None:
        raw_rough_terrain = getattr(cfg, "terrain", None)
        raw_rough_terrain_gen = getattr(raw_rough_terrain, "terrain_generator", None)
    if raw_rough_terrain_gen is None:
        raw_rough_terrain_gen = mdp.RM_ROUGH_TERRAINS_CFG

    rough_terrain_gen = copy.deepcopy(raw_rough_terrain_gen)
    rough_terrain_gen.curriculum = True
    if cfg.rough_height_offset_curriculum_cfg["enabled"]:
        _enable_v14_rough_height_offset_curriculum(
            rough_terrain_gen, cfg.rough_height_offset_curriculum_cfg["num_levels"]
        )

    raw_terrain_command_overrides = getattr(
        cfg, "rough_terrain_command_overrides_cfg", V14_ROUGH_TERRAIN_COMMAND_OVERRIDES
    )
    cfg.terrain_command_overrides = _filter_v14_terrain_command_overrides(
        raw_terrain_command_overrides, rough_terrain_gen
    )
    raw_terrain_importer = copy.deepcopy(getattr(cfg, "rough_terrain_importer_cfg", None))
    if raw_terrain_importer is None:
        raw_terrain_importer = copy.deepcopy(getattr(cfg, "terrain", None))
    if raw_terrain_importer is not None:
        cfg.terrain = raw_terrain_importer
        cfg.terrain.terrain_type = "generator"
        cfg.terrain.terrain_generator = rough_terrain_gen
    else:
        cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            collision_group=-1,
            terrain_generator=rough_terrain_gen,
            # max_init_terrain_level=0,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )

    _enable_v14_body_height_scanner(cfg)
    if bool(cfg.airborne_state_machine_cfg.get("enabled", False)) or bool(
        getattr(cfg, "wheel_forward_scan_cfg", {}).get("enabled", False)
    ) or bool(
        getattr(cfg, "stair_state_machine_cfg", {}).get("enabled", False)
    ):
        _enable_v14_wheel_height_scanners(cfg)
    else:
        _disable_v14_wheel_height_scanners(cfg)

    cfg.debug_value_diagnosis = False
    cfg.debug_value_diagnosis_interval = 200
    cfg.debug_value_diagnosis_topk = 5


def _make_v14_body_height_scanner_cfg() -> RayCasterCfg:
    """Create the default V14 body raycaster cfg when a parent config disabled it."""

    return RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=V14_BODY_HEIGHT_SCANNER_RESOLUTION,
            size=V14_BODY_HEIGHT_SCANNER_GRID_SIZE,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def _enable_v14_task_flag_obs(cfg) -> None:
    """Append a reserved task flag slot next to command observations."""
    obs_space_is_dict = isinstance(getattr(cfg, "observation_space", None), dict)
    cfg.task_flag_obs_enabled = True
    cfg.task_flag_obs_dim = V14_TASK_FLAG_DIM
    cfg.num_single_obs = V14_BASE_POLICY_OBS_DIM + V14_TASK_FLAG_DIM
    cfg.num_single_privileged_obs = (
        getattr(cfg, "num_single_privileged_obs", V14_BASE_PRIVILEGED_OBS_DIM) + V14_TASK_FLAG_DIM
    )
    cfg.state_space = cfg.num_privileged_obs_hist * cfg.num_single_privileged_obs

    obs_clip_cfg = dict(getattr(cfg, "obs_input_clip_cfg", {}))
    obs_clip_cfg["task_flag"] = [-10.0, 10.0]
    cfg.obs_input_clip_cfg = obs_clip_cfg

    if obs_space_is_dict:
        cfg.observation_space = dict(cfg.observation_space)
        cfg.observation_space["policy"] = cfg.num_single_obs * cfg.num_obs_hist
        cfg.observation_space["policy_hist"] = cfg.num_single_obs * cfg.num_obs_hist
        cfg.observation_space["critic"] = cfg.num_single_privileged_obs
        cfg.observation_space["prev_critic"] = cfg.num_single_privileged_obs
        cfg.observation_space["critic_hist"] = cfg.num_single_privileged_obs * cfg.num_privileged_obs_hist
    else:
        if bool(getattr(cfg, "use_frame_stack", False)):
            cfg.observation_space = cfg.num_obs_hist * cfg.num_single_obs
        else:
            cfg.observation_space = cfg.num_single_obs


def _set_v14_observation_dims(cfg, policy_dim: int, privileged_dim: int) -> None:
    """Set policy and privileged observation dimensions for isolated experiments."""
    cfg.num_single_obs = int(policy_dim)
    cfg.num_single_privileged_obs = int(privileged_dim)
    cfg.state_space = cfg.num_privileged_obs_hist * cfg.num_single_privileged_obs

    if isinstance(getattr(cfg, "observation_space", None), dict):
        cfg.observation_space = dict(cfg.observation_space)
        cfg.observation_space["policy"] = cfg.num_single_obs * cfg.num_obs_hist
        cfg.observation_space["policy_hist"] = cfg.num_single_obs * cfg.num_obs_hist
        cfg.observation_space["critic"] = cfg.num_single_privileged_obs
        cfg.observation_space["prev_critic"] = cfg.num_single_privileged_obs
        cfg.observation_space["critic_hist"] = cfg.num_single_privileged_obs * cfg.num_privileged_obs_hist
    else:
        cfg.observation_space = cfg.num_obs_hist * cfg.num_single_obs if bool(
            getattr(cfg, "use_frame_stack", False)
        ) else cfg.num_single_obs


def _enable_v14_body_height_scanner(cfg) -> None:
    """Apply V14-specific body height scanner settings."""
    if getattr(cfg, "height_scanner", None) is None:
        cfg.height_scanner = _make_v14_body_height_scanner_cfg()
    cfg.height_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=V14_BODY_HEIGHT_SCANNER_RESOLUTION,
        size=V14_BODY_HEIGHT_SCANNER_GRID_SIZE,
    )
    cfg.height_scanner.ray_alignment = "yaw"
    cfg.height_scanner.debug_vis = False
    cfg.height_scanner.mesh_prim_paths = ["/World/ground"]


def _enable_v14_wheel_height_scanners(cfg) -> None:
    """Attach dedicated raycasters to the left/right wheel centers."""
    base_scanner = copy.deepcopy(cfg.height_scanner)
    base_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=V14_WHEEL_HEIGHT_SCANNER_RESOLUTION,
        size=V14_WHEEL_HEIGHT_SCANNER_GRID_SIZE,
    )
    base_scanner.ray_alignment = "yaw"
    base_scanner.debug_vis = False

    right_scanner = copy.deepcopy(base_scanner)
    right_scanner.prim_path = "/World/envs/env_.*/Robot/right_wheel_link"

    left_scanner = copy.deepcopy(base_scanner)
    left_scanner.prim_path = "/World/envs/env_.*/Robot/left_wheel_link"

    cfg.right_wheel_height_scanner = right_scanner
    cfg.left_wheel_height_scanner = left_scanner


def _disable_v14_wheel_height_scanners(cfg) -> None:
    """Disable dedicated wheel raycasters when scan-based helpers are unused."""
    cfg.right_wheel_height_scanner = None
    cfg.left_wheel_height_scanner = None


def _apply_v14_flat_runtime_optimizations(cfg) -> None:
    """Keep V14 plane tasks on the lightweight absolute-height path.

    Flat V14 does not need terrain-aware state machines or any raycast-based
    helpers. We force the plane variant onto absolute-height observations so
    scan sensors are not created accidentally through inherited defaults.
    """
    terrain_cfg = getattr(cfg, "terrain", None)
    if getattr(terrain_cfg, "terrain_type", None) != "plane":
        return

    cfg.use_leg_length_as_height = False
    cfg.use_absolute_height = True
    cfg.enable_state_machines = False
    cfg.airborne_state_machine_cfg = copy.deepcopy(cfg.airborne_state_machine_cfg)
    cfg.airborne_state_machine_cfg["enabled"] = False
    cfg.wheel_forward_scan_cfg = copy.deepcopy(cfg.wheel_forward_scan_cfg)
    cfg.wheel_forward_scan_cfg["enabled"] = False
    cfg.play_height_scanner_debug_vis = False


V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_REWARDS = OrderedDict(
    # REWARD MAP V14_ROUGH_NP3O_BARLOW_PLUS_PRIV:
    # - Only used by WheelbipeV14RoughNP3OBarlowPlusPrivEnvCfg.
    # - Uses the selected balance-gated setting as the default rough policy target.
    termination=-500.0,
    # leg_joint_acc=-2.5e-4,
    leg_joint_acc=-2.5e-6,
    # joint_vel=-5.0e-5,
    leg_joint_vel=-1.0e-3,
    joint_torque=-2.0e-5,
    rear2_rear1_joint_pos_limits=-3.0,
    rear2_rear1_joint_pos_limits_torque=-5.0,
    rear2_rear1_joint_pos_limits_vel=-5.0,
    wheel_power=-1.0e-3,
    wheel_air_spin=-1.0e-2,
    lin_vel_z=-0.2,
    ang_vel_xy=-0.002,
    action_smoothness_leg=-0.05,
    action_rate=-0.05,
    action_smoothness_wheel=-0.01,
    flat_orientation_y=-2.0,
    flat_orientation_y_v=-2.0,
    flat_orientation_x=-0.2,
    flat_orientation_x_v=-0.5,
    track_lin_vel_xy=5.0,
    track_lin_vel_xy_tight=2.5,
    track_lin_vel_xy_square=-5.0,
    track_ang_vel_z=1.0,
    track_ang_vel_z_square=-0.5,
    stand_still_lin_vel=-2.0,
    stand_still=-0.0,
    track_height_exp=0.4,
    track_height_exp_soft=0.2,
    # track_height_exp_tight=2.0,
    track_height_exp_tight=1.0,
    # track_height_exp_both_wheels_contact=0.2,
    track_height_exp_both_wheels_contact=0.0,
    # no_fork_exp=-5.0,
    # no_fork_z_exp=-5.0,
    no_fork_exp=-1.0,
    no_fork_z_exp=-0.0,
    undesired_contact=-5.0,
    # wheel_motor_z_axis_align_exp=1.0,
    # wheel_motor_z_axis_align_exp_tight=1.0,
)


V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_COSTS = {
    # COST MAP V14_ROUGH_NP3O_BARLOW_PLUS_PRIV:
    # - Only used by WheelbipeV14RoughNP3OBarlowPlusPrivEnvCfg.
    # - Cost channel order is defined by Wheelbipe25v3Env._get_np3o_costs:
    #   body_tilt, body_height, body_ang_vel_xy, torque_limit, joint_velocity_limit.
    "num_costs": V14_NP3O_COST_DIM,
    "np3o_cost_d_values": [0.0, 0.0, 0.0, 0.0, 0.0],
    "np3o_cost_k_initial": [1.0, 1.0, 1.0, 0.5, 0.5],
    "np3o_tilt_limit_deg": 30.0,
    "np3o_body_height_min": 0.18,
    "np3o_body_height_max": 0.42,
    "np3o_ang_vel_xy_limit": 8.0,
    "np3o_torque_limit": 30.0,
    "np3o_joint_velocity_limit": 80.0,
    "np3o_cost_clip": 100.0,
}


def _apply_v14_rough_np3o_barlow_plus_priv_cost_cfg(cfg) -> None:
    """Apply the isolated NP3O cost config for the V14 rough NP3O Barlow PlusPriv task."""

    for key, value in V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_COSTS.items():
        setattr(cfg, key, copy.deepcopy(value))


__all__ = (
    "V14_TASK_FLAG_DIM",
    "V14_BASE_POLICY_OBS_DIM",
    "V14_BASE_PRIVILEGED_OBS_DIM",
    "V14_PRIV_OBS_PLUS_EXTRA_DIM",
    "V14_PRIV_OBS_PLUS_PRIVILEGED_DIM",
    "V14_JOINT_SINCOS_EXTRA_DIM",
    "V14_JOINT_SINCOS_POLICY_OBS_DIM",
    "V14_JOINT_SINCOS_PRIVILEGED_OBS_DIM",
    "V14_JOINT_SINCOS_ACT_SINCOS_ACTION_DIM",
    "V14_JOINT_SINCOS_ACT_SINCOS_EXTRA_DIM",
    "V14_JOINT_SINCOS_ACT_SINCOS_POLICY_OBS_DIM",
    "V14_JOINT_SINCOS_ACT_SINCOS_PRIVILEGED_OBS_DIM",
    "V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG",
    "V14_DREAMWAQ_ESTIMATED_STATE_DIM",
    "V14_DREAMWAQ_POLICY_HIST",
    "V14_SEMA_ESTIMATED_STATE_DIM",
    "V14_SEMA_POLICY_HIST",
    "V14_RESIDUAL_ESTIMATED_STATE_DIM",
    "V14_RESIDUAL_POLICY_HIST",
    "V14_HIM_ESTIMATED_STATE_DIM",
    "V14_HIM_POLICY_HIST",
    "V14_NP3O_POLICY_HIST",
    "V14_NP3O_EST_DIM",
    "V14_NP3O_COST_DIM",
    "V14_NP3O_ON_CONSTRAINT_DIM",
    "V14_NP3O_ON_CONSTRAINT_EXTRA_PRIV_DIM",
    "V14_BASIC_OBS_CLIP",
    "V14_BASIC_OBS_SCALE",
    "V14_EXTRA_OBS_CLIP",
    "V14_EXTRA_OBS_SCALE",
    "V14_BODY_HEIGHT_SCANNER_GRID_SIZE",
    "V14_BODY_HEIGHT_SCANNER_RESOLUTION",
    "V14_WHEEL_HEIGHT_SCANNER_GRID_SIZE",
    "V14_WHEEL_HEIGHT_SCANNER_RESOLUTION",
    "V14_ORDERED_LEG_JOINT_NAMES",
    "V14_ORDERED_LEG_BODY_NAMES",
    "V14_LINKS_LENGTH",
    "_V14_ALPHA0_DEG",
    "_V14_ALPHA2_DEG",
    "V14_ALPHA_OFFSET",
    "V14_PREDEFINED_RESET_GROUND",
    "V14_LEGACY_OBS_INPUT_SCALE_CFG",
    "V14_ROUGH_TERRAIN_COMMAND_OVERRIDES",
    # "V14_ROTATION_TERRAIN_COMMAND_OVERRIDES",
    "V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1",
    "V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_2",
    "_enable_v14_rough_height_offset_curriculum",
    "_filter_v14_terrain_command_overrides",
    "_apply_v14_rough_runtime_cfg",
    "_enable_v14_task_flag_obs",
    "_set_v14_observation_dims",
    "_enable_v14_body_height_scanner",
    "_enable_v14_wheel_height_scanners",
    "_disable_v14_wheel_height_scanners",
    "_apply_v14_flat_runtime_optimizations",
    "V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_REWARDS",
    "V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_COSTS",
    "_apply_v14_rough_np3o_barlow_plus_priv_cost_cfg",
)
