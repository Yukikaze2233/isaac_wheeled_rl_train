"""Flat-terrain wheeled-biped env config.

All hyperparameters live here so experiments stay diffable. Observation and
action semantics MUST stay identical to the deploy repo CONTRACT.md — change
both together or not at all.
"""
from collections import OrderedDict

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from wheeled_world.assets import WheeledBipedCFG


@configclass
class WheeledBipedFlatEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------ #
    # timing: 200 Hz physics x decimation 4 -> 50 Hz policy (flat task)               #
    # ------------------------------------------------------------------ #
    decimation: int = 4
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0)
    robot: ArticulationCfg = None  # set in __post_init__
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        history_length=3,
        track_air_time=False,
    )

    # ------------------------------------------------------------------ #
    # spaces: obs 35 (policy) / 43 (critic = 35+3+1+4 DR readout) / act 6  #
    # ------------------------------------------------------------------ #
    observation_space = 35
    privileged_extra_obs: int = 4  # lin_vel(3) + true_height(1)
    action_space = 6

    # observation scales — deploy CONTRACT.md section 2 must match
    scale_ang_vel: float = 0.5
    scale_height_cmd: float = 5.0
    scale_joint_vel: float = 0.1
    clip_obs: float = 100.0
    mute_wheel_pos_obs: bool = True

    # action decode: legs -> position target offset, wheels -> velocity target.
    # Authoritative official values (pretrained env.yaml + env.py):
    #   use_wheel_vel_control=True overrides wheel_action_scale to
    #   wheel_vel_action_scale=10.0 and max_wheel_vel to 100*1.5=150 rad/s.
    leg_action_scale: float = 0.5
    wheel_action_scale: float = 10.0
    max_wheel_vel: float = 150.0
    wheel_torque_limit: float = 5.0

    # joint/body name patterns (official V14 contract: legs rear-first so the
    # 6D action dims are [left_rear, right_rear, left_front, right_front, L_wheel, R_wheel])
    leg_joint_patterns: tuple = (".*_rear1_joint", ".*_front1_joint")
    wheel_joint_patterns: tuple = (".*_wheel_joint",)
    spring_joint_patterns: tuple = (".*_spring2_joint",)
    base_body_patterns: tuple = ("base_link",)
    leg_body_patterns: tuple = (
        ".*_front1_link", ".*_rear1_link", ".*_front2_link", ".*_rear2_link",
        ".*_front3_link", ".*_front4_link", ".*_spring1_link", ".*_spring2_link",
        "gimbal_yaw_link", "gimbal_pitch_link",
    )
    wheel_body_patterns: tuple = (".*_wheel_link",)

    # ------------------------------------------------------------------ #
    # episode / height command                                            #
    # ------------------------------------------------------------------ #
    episode_length_s: float = 20.0
    default_height_cmd: float = 0.22
    # absolute-height semantics (use_absolute_height=True): the height command
    # is the base origin z in the world frame. Official training range 0.20-0.42.
    height_range: tuple[float, float] = (0.20, 0.42)
    resampling_time_range: tuple[float, float] = (5.0, 15.0)

    # ------------------------------------------------------------------ #
    # gas spring model (prismatic joint + per-step force)                 #
    # ------------------------------------------------------------------ #
    spring_settings: dict = dict(
        force_up=600.0,        # N at full compression (official linear_up)
        force_down=400.0,      # N at free length (official linear_down)
        linear_length=0.07,    # m of travel between the two
        spring_offset=0.06076, # m initial compression (official spring_offset)
        rand_force=50.0,       # ± N constant offset per env (official random_force [-50, 50])
    )

    # ------------------------------------------------------------------ #
    # pipeline delays (in 50 Hz control steps) and observation noise      #
    # ------------------------------------------------------------------ #
    use_obs_delay: bool = True
    obs_delay_range: tuple[int, int] = (1, 4)   # 20-80 ms
    use_act_delay: bool = True
    act_delay_range: tuple[int, int] = (1, 3)   # 20-60 ms

    obs_noise: dict = dict(
        ang_vel=0.25,
        gravity=0.05,
        joint_pos=0.025,
        leg_vel=0.5,
        wheel_vel=1.0,
    )

    # ------------------------------------------------------------------ #
    # domain randomization (the full DR set core subset)                 #
    # ------------------------------------------------------------------ #
    dr_mass_base: tuple[float, float] = (0.9, 1.3)    # startup, scale
    dr_mass_leg: tuple[float, float] = (0.9, 1.1)     # startup, scale
    dr_gains: tuple[float, float] = (0.75, 1.25)      # reset, scale Kp/Kd
    dr_wheel_friction_add: tuple[float, float] = (0.05, 0.25)
    dr_leg_friction_add: tuple[float, float] = (0.05, 0.2)
    dr_reset_pose: dict = dict(roll=(-0.15, 0.15), pitch=(-0.15, 0.15), yaw=(-3.14, 3.14))
    # extras (the full DR set full set)
    dr_base_com: dict = dict(x=(-0.04, 0.04), y=(-0.02, 0.02), z=(-0.02, 0.02))  # startup
    dr_wheel_material: dict = dict(static_friction=(0.5, 1.2), dynamic_friction=(0.4, 1.0),
                                   restitution=(0.02, 0.2))                       # startup
    dr_action_noise: dict = dict(scale=(1.0, 1.0), bias=(0.0, 0.0), noise_std=(0.0, 0.0),
                                 )  # decoded-target disturbance, opened for training

    # ------------------------------------------------------------------ #
    # privileged DR readout: critic also sees the sampled DR parameters    #
    # (gains scale, wheel/leg friction, mass scale) — 71D-pattern readout             #
    # ------------------------------------------------------------------ #
    privileged_dr_readout: bool = True
    privileged_dr_dims: int = 4

    # ------------------------------------------------------------------ #
    # state machines (airborne/jump + stair) — 7D mode flags become live   #
    # ------------------------------------------------------------------ #
    enable_state_machines: bool = True
    airborne_state_machine_cfg: dict = dict(
        body_height_threshold=0.30, min_down_vel=0.2,
        target_height=0.30, landing_duration_s=0.3,
    )
    # curriculum: reward-weight stages + base assist force decay
    enable_curriculum: bool = False
    curriculum_stages: list = None  # set in __post_init__ (mdp.curriculums.Stage list)

    # ------------------------------------------------------------------ #
    # command sampler: base ranges + special-mode buckets (implicit curriculum)       #
    # ------------------------------------------------------------------ #
    base_cmd_ranges: dict = dict(vx=(-1.5, 1.5), yaw_rate=(-3.14, 3.14))
    rel_standing_envs: float = 0.1
    rel_heading_envs: float = 0.5
    special_modes: dict = {
        # name: (rel_envs, vx_ranges, yaw_rate_ranges, iteration_start)
        "spin_low": (0.15, ((-0.1, 0.1),), ((2 * 3.14159, 3.25 * 3.14159), (-3.25 * 3.14159, -2 * 3.14159)), 3000),
        "spin_mid": (0.15, ((-0.1, 0.1),), ((3.25 * 3.14159, 4.5 * 3.14159), (-4.5 * 3.14159, -3.25 * 3.14159)), 4000),
        "dash": (0.30, ((2.0, 3.0), (-3.0, -2.0)), ((-6.28, 6.28),), 2000),
    }
    steps_per_iteration: int = 24

    # ------------------------------------------------------------------ #
    # rewards: scale table (sign carries penalty/reward) + kernel sigmas  #
    # ------------------------------------------------------------------ #
    rewards: OrderedDict = OrderedDict(
        track_lin_vel_xy_exp=1.0,
        track_ang_vel_z_exp=1.0,
        track_height_exp=1.0,
        lin_vel_z=-0.5,
        ang_vel_xy=-0.05,
        joint_torque=-1e-4,
        joint_acc=-5e-7,
        action_rate=-0.01,
        undesired_contact=-2.0,
        termination=-200.0,
    )
    sigma_lin_vel: float = 0.5
    sigma_ang_vel: float = 0.25
    sigma_height: float = 0.025
    undesired_contact_force: float = 5.0
    fall_gravity_z_threshold: float = -0.5   # projected_gravity_z above this = tipped over
    min_base_height: float = 0.1

    # ------------------------------------------------------------------ #
    # rough terrain: height-field generator (stairs / inv-stairs / slopes) #
    # ------------------------------------------------------------------ #
    use_rough_terrain: bool = False
    terrain: object = None  # TerrainImporterCfg; set in __post_init__ for rough

    def __post_init__(self):
        if self.robot is None:
            self.robot = WheeledBipedCFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.sim.render_interval = self.decimation
        if self.use_rough_terrain and self.terrain is None:
            from wheeled_tasks.manager.mdp.terrain import make_rm_rough_terrain_cfg
            self.terrain = make_rm_rough_terrain_cfg()


@configclass
class WheeledBipedRoughEnvCfg(WheeledBipedFlatEnvCfg):
    """Rough-terrain variant: stairs / inv-stairs / slopes drive the stair FSM."""

    use_rough_terrain: bool = True

    def __post_init__(self):
        super().__post_init__()
        # rough tasks benefit from the curriculum: stage reward weights as the
        # height tracking stabilizes before the terrain difficulty bites
        self.enable_curriculum = True
