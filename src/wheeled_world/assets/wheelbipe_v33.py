"""Isaac Lab asset config for the own serial-leg wheeled biped (urdf_V3.3_fix).

Loads the sanitized training URDF directly (Isaac Lab URDF importer), with the
verified contract values:

- legs (L_joint1/L_joint2/R_joint1/R_joint2): position PD 60/2, |tau| <= 40 Nm
- wheels (L_joint3/R_joint3): velocity servo (stiffness 0), damping 1.0
  Nm/(rad/s) (V3.3 trained value; KV=2.0 variant exists — pick after training
  results), |tau| <= 5 Nm
- no gas springs (serial legs)

Default standing pose: base z = 0.48 m, joints at the IK defaults from
assets/urdf_v33/defaults.json (L hip +0.8702, L knee +1.1311,
R hip -0.8702, R knee -1.0844).

Usage in env_cfg:  robot = WheeledBipedV33CFG.replace(prim_path=...)
"""
import json
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(_HERE, "urdf_v33", "urdf_V3.3_rl.urdf")
_DEFAULTS = json.load(open(os.path.join(_HERE, "urdf_v33", "defaults.json")))

WheeledBipedV33CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        urdf_path=URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
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
        pos=(0.0, 0.0, _DEFAULTS["base_height_default"]),
        joint_pos={
            "L_joint1": _DEFAULTS["default_joint_pos"]["L_joint1"],
            "L_joint2": _DEFAULTS["default_joint_pos"]["L_joint2"],
            "R_joint1": _DEFAULTS["default_joint_pos"]["R_joint1"],
            "R_joint2": _DEFAULTS["default_joint_pos"]["R_joint2"],
            "L_joint3": 0.0,
            "R_joint3": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=["L_joint1", "L_joint2", "R_joint1", "R_joint2"],
            stiffness=60.0,
            damping=2.0,
            effort_limit=40.0,
            velocity_limit=17.0,
            armature=0.0,  # TODO identification (rotor inertia * ratio^2)
        ),
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=["L_joint3", "R_joint3"],
            stiffness=0.0,
            damping=1.0,   # velocity servo gain Nm/(rad/s), V3.3 contract
            effort_limit=5.0,
            velocity_limit=150.0,
            armature=0.0,
        ),
    },
)
