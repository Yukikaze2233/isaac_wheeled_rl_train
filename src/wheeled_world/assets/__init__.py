"""Articulation asset config for the reference wheeled biped.

The USD itself is NOT shipped (mirroring the registry). Point
WHEELED_RL_ASSETS_DIR at a folder containing:

    <WHEELED_RL_ASSETS_DIR>/wheeled_biped/wheeled_biped.usd

Convert it from our URDF inside Isaac Sim (see ../../assets/README.md).
Joint names below define the naming convention the env relies on — either
match them in the URDF or adjust both sides together.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

AssetPath = os.environ.get("WHEELED_RL_ASSETS_DIR", os.path.join(os.getcwd(), "assets"))
USD_PATH = f"{AssetPath}/wheeled_biped/wheeled_biped.usd"

# Rotor inertia reflected through the gearbox (J_rotor * ratio^2) — a core
# sim2real ingredient. Replace with our motor data (front/rear share one motor
# type in the reference design; per-motor dicts are supported by Isaac Lab).
LEG_ARMATURE = 1.95e-4 * 9.0 * 9.0  # DM8009-class actuator

WheeledBipedCFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            # Closed-chain legs need solver headroom; in-field testing found sim rate
            # (200Hz here) matters more than iteration counts.
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=6,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_front1_joint": 0.0,
            ".*_rear1_joint": 0.0,
            ".*_front2_joint": 0.0,
            ".*_rear2_joint": 0.0,
            ".*_front3_joint": 0.0,
            ".*_front4_joint": 0.0,
            ".*_spring1_joint": 0.0,
            ".*_spring2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Actively driven leg joints (position-controlled by the policy).
        # OFFICIAL ORDER (pretrained env.yaml legs_act): rear first, so the
        # 6D action dims are [left_rear, right_rear, left_front, right_front].
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=[".*_rear1_joint", ".*_front1_joint"],
            stiffness=60.0,
            damping=2.0,
            effort_limit=40.0,
            velocity_limit=17.0,
            armature=LEG_ARMATURE,
        ),
        # Passive closed-chain companions (pure damping).
        "legs_inact": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_front2_joint",
                ".*_rear2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
            ],
            stiffness=0.0,
            damping=0.01,
            effort_limit=50.0,
            velocity_limit=300.0,
            armature=0.0001,
        ),
        # Wheels: velocity-controlled, actuator provides viscous damping only.
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=0.2,
            effort_limit=5.0,
            velocity_limit=60.0,
            armature=0.0,
        ),
        # Gas spring prismatic joints: force comes from the env every step.
        "spring": IdealPDActuatorCfg(
            joint_names_expr=[".*_spring2_joint"],
            stiffness=0.0,
            damping=50.0,
            effort_limit=1000.0,
            velocity_limit=50.0,
            armature=0.0001,
        ),
    },
)
