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

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg

from agent_world import AssetPath
from agent_world.actuators import M3508ActuatorCfg, DiffVelPDActuatorCfg

DM8009_ARMATURE = 1.95e-04*9.0*9.0


Wheelbipe_V14_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/wheelbipeV14_2_1/wheelbipeV14_2.usd",
        activate_contact_sensors=True,
        copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=6,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_rear1_joint": 0.0,
            ".*_rear2_joint": 0.0,
            ".*_front1_joint": 0.0,
            ".*_front2_joint": 0.0,
            ".*_front3_joint": 0.0,
            ".*_front4_joint": 0.0,
            ".*_spring1_joint": 0.0,
            ".*_spring2_joint": 0.0,
            "gimbal_yaw_joint": 0.0,
            "gimbal_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # "legs_act": ImplicitActuatorCfg(
        "legs_act": IdealPDActuatorCfg(
        # "legs_act": DiffVelPDActuatorCfg(
            # diff_dt=0.005,
            joint_names_expr=[
                ".*_rear1_joint",
                ".*_front1_joint",
            ],
            stiffness={
                ".*_rear1_joint": 60.0,
                ".*_front1_joint": 60.0,
            },
            damping={
                ".*_rear1_joint": 2.0,
                ".*_front1_joint": 2.0,
            },
            effort_limit={
                ".*_rear1_joint": 40.0,
                ".*_front1_joint": 40.0,
            },
            velocity_limit={
                ".*_rear1_joint": 17,
                ".*_front1_joint": 17,
            },
            # armature=0.001,
            armature=DM8009_ARMATURE,
            # armature=DM8009_ARMATURE*0.1,
            # armature=0.,
        ),
        # "legs_inact": ImplicitActuatorCfg(
        "legs_inact": IdealPDActuatorCfg(
        # "legs_inact": DiffVelPDActuatorCfg(
            # diff_dt=0.005,
            joint_names_expr=[
                ".*_rear2_joint",
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
                ".*_guide_joint",
            ],
            stiffness=0.0,
            damping=0.01,
            armature=0.0001,
            # armature=0,
            effort_limit=50.0,
            velocity_limit=300.0,
        ),
        "wheel": IdealPDActuatorCfg(
        # "wheel": DiffVelPDActuatorCfg(
            # diff_dt=0.005,
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=0.2,
            effort_limit=5.0,
            velocity_limit=60.0,
            # armature=0.001,
            armature=0,
        ),
        # "spring": ImplicitActuatorCfg(
        "spring": IdealPDActuatorCfg(
        # "spring": DiffVelPDActuatorCfg(
            # diff_dt=0.005,
            joint_names_expr=[".*_spring2_joint"],
            stiffness=0.0,
            # damping=1000.,
            # damping=500.,
            damping=50.,
            effort_limit=1000.0,
            velocity_limit=50.0,
            armature=0.0001,
            # armature=0,
        ),
        "gimbal_yaw": IdealPDActuatorCfg(
            joint_names_expr=["gimbal_yaw_joint"],
            stiffness=0.0,
            damping=0.5,
            effort_limit=2.0,
            velocity_limit=30.0,
            armature=0.0001,
        ),
        # "gimbal_pitch": ImplicitActuatorCfg(
        "gimbal_pitch": IdealPDActuatorCfg(
            joint_names_expr=["gimbal_pitch_joint"],
            stiffness=20.0,
            damping=0.5,
            effort_limit=10.0,
            velocity_limit=30.0,
            armature=0.0001,
        ),
    },
)

Wheelbipe_V14_2_NG_CFG = Wheelbipe_V14_2_CFG.copy()
Wheelbipe_V14_2_NG_CFG.spawn = copy.deepcopy(Wheelbipe_V14_2_CFG.spawn)
Wheelbipe_V14_2_NG_CFG.actuators = copy.deepcopy(Wheelbipe_V14_2_CFG.actuators)
Wheelbipe_V14_2_NG_CFG.spawn.usd_path = f"{AssetPath}/usd_files/wheelbipeV14_2/wheelbipeV14_2_ng.usd"
Wheelbipe_V14_2_NG_CFG.actuators["legs_inact"].joint_names_expr=[
                                                                    ".*_rear2_joint",
                                                                    ".*_front2_joint",
                                                                    ".*_front3_joint",
                                                                    ".*_front4_joint",
                                                                    ".*_spring1_joint",
                                                                ]

Wheelbipe_V14_2_M3508_CFG = Wheelbipe_V14_2_CFG.copy()
Wheelbipe_V14_2_M3508_CFG.actuators = copy.deepcopy(Wheelbipe_V14_2_CFG.actuators)
Wheelbipe_V14_2_M3508_CFG.actuators["wheel"] = M3508ActuatorCfg(
    joint_names_expr=[".*_wheel_joint"],
    stiffness=0.0,
    damping=0.2,
    curve_path="experiments/v14_flat/005_3508_motor/m3508_c620_current_closed_loop_raw_motor_shaft_curve_dense.csv",
    gear_ratio=268.0 / 17.0,
    gearbox_efficiency=1.0,
    output_stall_torque=5.5,
    effort_limit=5.5,
    velocity_limit=63.3734,
    armature=0.001,
)
