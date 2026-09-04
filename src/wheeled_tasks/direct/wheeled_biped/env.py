"""Wheeled-biped flat locomotion env (Isaac Lab DirectRLEnv, self-implemented).

Feature set mirrors the the flat task task:
- frozen 35D policy / 39D critic / 6D action contract (deploy CONTRACT.md)
- gas spring modeled as prismatic joint + per-step force with per-env offset
- observation & action pipeline delays (per-env resampled lag)
- observation noise, startup/reset domain randomization
- special-mode velocity commands (spin / dash buckets, iteration-gated)
- dense exp-kernel tracking rewards + penalty table + termination penalty

Not implemented (extension points): terrain/rough tasks, airborne & jump
state machines, gimbal modes, privileged DR-parameter readout.
"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from wheeled_tasks.manager.mdp import (
    DelayBuffer,
    SpecialModeEntryCfg,
    SpecialModeUniformVelocityCommand,
    SpecialModeUniformVelocityCommandCfg,
)
from wheeled_tasks.manager.mdp import events as dr
from wheeled_tasks.direct.wheeled_biped.state_machines import StateMachineManager


class WheeledBipedEnv(DirectRLEnv):
    cfg: "WheeledBipedFlatEnvCfg"

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        c = self.cfg
        # ---- joint discovery by name (naming convention in assets cfg) ----
        self._leg_ids = self.robot.find_joints([".*_front1_joint", ".*_rear1_joint"], preserve_order=True)[0]
        self._wheel_ids = self.robot.find_joints(".*_wheel_joint", preserve_order=True)[0]
        self._spring_ids = self.robot.find_joints(".*_spring2_joint", preserve_order=True)[0]
        self._actuate_ids = torch.cat([self._leg_ids, self._wheel_ids])
        self._default_leg_pos = self.robot.data.default_joint_pos[:, self._leg_ids]

        # ---- per-env buffers ----
        n = self.num_envs
        self._raw_actions = torch.zeros(n, c.action_space, device=self.device)
        self._last_actions = torch.zeros(n, c.action_space, device=self.device)
        self._spring_rand = torch.zeros(n, 2, device=self.device)
        self._height_cmd = torch.full((n,), c.default_height_cmd, device=self.device)
        # pipeline delays, expressed in 50 Hz control steps
        self._obs_imu = DelayBuffer(n, 6, c.obs_delay_range[1], self.device)   # ang_vel + gravity
        self._obs_pos = DelayBuffer(n, 6, c.obs_delay_range[1], self.device)   # 4 legs + 2 wheel slots
        self._obs_vel = DelayBuffer(n, 6, c.obs_delay_range[1], self.device)
        self._act_delay = DelayBuffer(n, c.action_space, c.act_delay_range[1], self.device)

        # ---- command sampler with special-mode buckets ----
        modes = {
            name: SpecialModeEntryCfg(rel_envs=rel, ranges={"vx": vx, "yaw_rate": wz}, iteration_start=it0)
            for name, (rel, vx, wz, it0) in c.special_modes.items()
        }
        cmd_cfg = SpecialModeUniformVelocityCommandCfg(
            base_ranges=c.base_cmd_ranges,
            resampling_time_range=c.resampling_time_range,
            rel_standing_envs=c.rel_standing_envs,
            rel_heading_envs=c.rel_heading_envs,
            special_modes=modes,
        )
        self.command_generator = SpecialModeUniformVelocityCommand(cmd_cfg, n, self.device)
        self.command_generator.resample_all()

        # ---- startup domain randomization: masses (once, all envs) ----
        all_ids = torch.arange(n, device=self.device)
        base_bodies = ["base_link"]
        leg_bodies = [".*_front1_link", ".*_rear1_link", ".*_front2_link", ".*_rear2_link",
                      ".*_front3_link", ".*_front4_link", ".*_spring1_link", ".*_spring2_link"]
        mass_scale = dr.randomize_body_mass(self.robot, all_ids, base_bodies, c.dr_mass_base)
        dr.randomize_body_mass(self.robot, all_ids, leg_bodies, c.dr_mass_leg)
        dr.randomize_body_com(self.robot, all_ids, base_bodies, c.dr_base_com)
        dr.randomize_body_material(self.robot, all_ids, ["*_wheel_geom" if False else ".*_wheel_link"],
                                   c.dr_wheel_material["static_friction"], c.dr_wheel_material["dynamic_friction"],
                                   c.dr_wheel_material["restitution"])

        # ---- privileged DR readout: gains(1) leg-fric(1) wheel-fric(1) mass(1) ----
        self._dr_readout = torch.zeros(n, c.privileged_dr_dims, device=self.device)
        self._dr_readout[:, 3] = mass_scale.squeeze(-1)

        # ---- state machines: 7D mode flags become live ----
        if c.enable_state_machines:
            sm_cfg = dict(c.airborne_state_machine_cfg)
            sm_cfg.setdefault("control_dt", self.step_dt)
            self.state_machine = StateMachineManager(n, self.device, sm_cfg)
        else:
            self.state_machine = None

        # ---- decoded-action disturbance decoded-action disturbance ----
        def _noise_param(key: str) -> torch.Tensor:
            lo, hi = c.dr_action_noise[key]
            return lo + (hi - lo) * torch.rand((n, 1), device=self.device)

        self._act_scale, self._act_bias, self._act_std = (
            _noise_param("scale"), _noise_param("bias"), _noise_param("noise_std"))
        # DR-on-reset gate (min_step_count_between_reset=720): PhysX writes
        # for friction/gains fire at most once per window, not per env reset
        self._dr_gate_steps = 720
        self._dr_last_rewrite = -self._dr_gate_steps

        # ---- curriculum: reward-weight stages + base assist force ----
        self.curriculum = None
        self._reward_weights = dict(c.rewards)
        self._assist_force = torch.zeros(n, device=self.device)
        self._track_h_window_sum = torch.zeros((), device=self.device)
        self._track_h_window_count = 0
        self._episodes_since_stage = 0
        if c.enable_curriculum:
            from wheeled_tasks.manager.mdp.curriculums import RewardWeightProgression
            self.curriculum = RewardWeightProgression(list(c.curriculum_stages or []))

    # ------------------------------------------------------------------ #
    # scene                                                                #
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        if self.cfg.use_rough_terrain and self.cfg.terrain is not None:
            # rough height-field terrain: TerrainImporter before cloning so
            # scene.env_origins picks up the per-env patch origins
            from isaaclab.terrains import TerrainImporter
            self.scene.terrain = TerrainImporter(self.cfg.terrain)
        else:
            sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_paths=True)
        sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg())

    # ------------------------------------------------------------------ #
    # action pipeline (50 Hz entry, applied at every 200 Hz physics step)  #
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        c = self.cfg
        self._last_actions.copy_(self._raw_actions)
        self._raw_actions = actions.clone()

        # decode: legs = default + scale * a ; wheels = clamp(scale * a)
        leg_targets = self._default_leg_pos + c.leg_action_scale * actions[:, :4]
        wheel_targets = torch.clamp(c.wheel_action_scale * actions[:, 4:6], -c.max_wheel_vel, c.max_wheel_vel)
        targets = torch.cat([leg_targets, wheel_targets], dim=-1)
        # decoded-action disturbance decoded-command-side disturbance
        targets = targets * self._act_scale + self._act_bias + self._act_std * torch.randn_like(targets)
        if c.use_act_delay:
            targets = self._act_delay.compute(targets)
        self._leg_targets, self._wheel_targets = targets[:, :4], targets[:, 4:6]

        # curriculum assist force on the base (pair with reward stages — see base.py note)
        if bool((self._assist_force > 0).any()):
            up = torch.zeros_like(self.robot.data.root_pos_w)
            up[:, 2] = self._assist_force
            self.robot.set_external_force_and_torque(forces=up, torques=torch.zeros_like(up),
                                                     body_ids=self.robot.find_bodies("base_link")[0])

    def _apply_action(self) -> None:
        c = self.cfg
        self.robot.set_joint_position_target(self._leg_targets, joint_ids=self._leg_ids)
        self.robot.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_ids)
        # gas spring: F = F_down + (F_up - F_down)/L * compression + per-env rand
        spring_pos = self.robot.data.joint_pos[:, self._spring_ids]
        compression = torch.clamp(c.spring_settings["spring_offset"] - spring_pos, min=0.0)
        force = (c.spring_settings["force_down"]
                 + (c.spring_settings["force_up"] - c.spring_settings["force_down"])
                 / c.spring_settings["linear_length"] * compression
                 + self._spring_rand)
        self.robot.set_joint_effort_target(force, joint_ids=self._spring_ids)

    # ------------------------------------------------------------------ #
    # observations: 35D policy / 39D critic (see deploy CONTRACT.md)       #
    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict:
        c = self.cfg
        n = self.num_envs

        # command resample timers; special modes gated by training iteration
        iteration = self.common_step_counter // c.steps_per_iteration
        resampled = self.command_generator.update(self.step_dt, iteration)
        if resampled.numel() > 0:
            lo, hi = c.height_range
            self._height_cmd[resampled] = lo + (hi - lo) * torch.rand(resampled.numel(), device=self.device)

        # state machines: live 7D flags + airborne height bias
        root_z = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        base_force = torch.norm(self.contact_sensor.data.net_forces_w_history[:, -1, 0, :], dim=-1)
        stair_flag = slope_flag = None
        if self.cfg.use_rough_terrain:
            terrain = getattr(self.scene, "terrain", None)
            if terrain is not None and getattr(terrain, "terrain_type_names", None):
                from wheeled_tasks.manager.mdp.terrain import terrain_flags_from_types
                stair_flag, slope_flag = terrain_flags_from_types(
                    terrain.terrain_type_names, terrain.terrain_types, n, self.device)
        if self.state_machine is not None:
            self.state_machine.update(root_z, self.robot.data.root_lin_vel_b[:, 2],
                                      base_force > 1.0, terrain_flag=stair_flag, slope_flag=slope_flag)
            flags = self.state_machine.mode_flags()
            height_cmd_eff = self._height_cmd + self.state_machine.height_bias()
            in_air = self.state_machine.airborne.in_air
        else:
            flags = torch.zeros(n, 7, device=self.device)
            flags[:, 0] = 1.0
            height_cmd_eff = self._height_cmd
            in_air = torch.zeros(n, dtype=torch.bool, device=self.device)

        # raw sensor streams
        ang_vel = self.robot.data.root_ang_vel_b
        grav = self.robot.data.projected_gravity_b
        joint_pos = (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self._actuate_ids]
        joint_vel = self.robot.data.joint_vel[:, self._actuate_ids]

        # per-stream delay, then uniform noise, then wheel-slot mute
        if c.use_obs_delay:
            imu = self._obs_imu.compute(torch.cat([ang_vel, grav], dim=-1))
            joint_pos = self._obs_pos.compute(joint_pos)
            joint_vel = self._obs_vel.compute(joint_vel)
            ang_vel, grav = imu[:, :3], imu[:, 3:]

        def noise(x: torch.Tensor, std: float) -> torch.Tensor:
            return x + std * (2 * torch.rand_like(x) - 1)

        ang_vel = noise(ang_vel, c.obs_noise["ang_vel"])
        grav = noise(grav, c.obs_noise["gravity"])
        pos_all = noise(joint_pos, c.obs_noise["joint_pos"])
        vel_all = torch.cat([
            noise(joint_vel[:, :4], c.obs_noise["leg_vel"]),
            noise(joint_vel[:, 4:], c.obs_noise["wheel_vel"]),
        ], dim=-1)
        if c.mute_wheel_pos_obs:
            pos_all[:, 4:] = 0.0

        policy = torch.cat([
            self.command_generator.command,               # 3
            height_cmd_eff.unsqueeze(-1) * c.scale_height_cmd,  # 1 (state-machine bias included)
            ang_vel * c.scale_ang_vel,                    # 3
            grav,                                         # 3
            pos_all[:, :4],                               # 4 leg joint pos
            pos_all[:, 4:],                               # 2 wheel slots (muted)
            vel_all * c.scale_joint_vel,                  # 6 joint vel
            self._last_actions,                           # 6
            flags,                                        # 7 (live from state machines)
        ], dim=-1)
        policy = torch.nan_to_num(torch.clamp(policy, -c.clip_obs, c.clip_obs))

        # asymmetric critic: true lin-vel + true height + sampled DR params
        # (71D-pattern: critic sees the randomization the policy must infer)
        height = self.robot.data.root_pos_w[:, 2:3] - self.scene.env_origins[:, 2:3]
        critic = torch.cat([policy, self.robot.data.root_lin_vel_b, height], dim=-1)
        if c.privileged_dr_readout:
            critic = torch.cat([critic, self._dr_readout], dim=-1)
        critic = torch.nan_to_num(torch.clamp(critic, -c.clip_obs, c.clip_obs))
        return {"policy": policy, "critic": critic}

    # ------------------------------------------------------------------ #
    # rewards (scale table in cfg; kernels: exp tracking + L2 penalties)   #
    # ------------------------------------------------------------------ #
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        r = self._reward_weights  # runtime weights (curriculum may restage them)
        cmd = self.command_generator.command
        lin_vel = self.robot.data.root_lin_vel_b
        ang_vel = self.robot.data.root_ang_vel_b
        root_z = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

        track_lin = torch.exp(-torch.sum(torch.square(cmd[:, :2] - lin_vel[:, :2]), dim=-1) / c.sigma_lin_vel ** 2)
        track_ang = torch.exp(-torch.square(cmd[:, 2] - ang_vel[:, 2]) / c.sigma_ang_vel ** 2)
        track_h = torch.exp(-torch.square(self._height_cmd - root_z) / c.sigma_height ** 2)

        torque = self.robot.data.applied_torque[:, self._actuate_ids]
        acc = self.robot.data.joint_acc[:, self._actuate_ids]

        total = (r["track_lin_vel_xy_exp"] * track_lin
                 + r["track_ang_vel_z_exp"] * track_ang
                 + r["track_height_exp"] * track_h
                 + r["lin_vel_z"] * torch.square(lin_vel[:, 2])
                 + r["ang_vel_xy"] * torch.sum(torch.square(ang_vel[:, :2]), dim=-1)
                 + r["joint_torque"] * torch.sum(torch.square(torque), dim=-1)
                 + r["joint_acc"] * torch.sum(torch.square(acc), dim=-1)
                 + r["action_rate"] * torch.sum(torch.square(self._raw_actions - self._last_actions), dim=-1))

        base_force = torch.norm(self.contact_sensor.data.net_forces_w_history[:, -1, 0, :], dim=-1)
        total = total + r["undesired_contact"] * (base_force > c.undesired_contact_force).float()
        total = total + r["termination"] * self._terminated().float()

        # airborne landing window: dense trajectory-following terms (reward-shaping rule:
        # reward the whole trajectory, never just the apex)
        if self.state_machine is not None and bool(self.state_machine.airborne.in_air.any()):
            in_air = self.state_machine.airborne.in_air.float()
            landing_h = torch.exp(-torch.square(root_z - self.state_machine.airborne.target_height)
                                  / c.sigma_height ** 2) * in_air
            landing_v = torch.exp(-torch.square(self.robot.data.root_lin_vel_b[:, 2]) / c.sigma_ang_vel ** 2) * in_air
            total = total + 1.0 * landing_h + 0.5 * landing_v

        # curriculum: window the height-tracking term, poll stage progression.
        # Accumulate on-device; the float() sync fires only once per window
        # (every window_size steps), not every control step.
        if self.curriculum is not None:
            self._track_h_window_sum += track_h.mean()
            self._track_h_window_count += 1
            if self._track_h_window_count >= self.curriculum.window_size:
                self.curriculum.track(
                    float(self._track_h_window_sum / self._track_h_window_count),
                    episodes_finished=0)
                self._track_h_window_sum.zero_()
                self._track_h_window_count = 0
                _, effects = self.curriculum.step()
                self._reward_weights.update(effects["reward_weights"])
                self._assist_force.fill_(effects["force_z"])
        return total

    # ------------------------------------------------------------------ #
    # dones                                                                #
    # ------------------------------------------------------------------ #
    def _terminated(self) -> torch.Tensor:
        c = self.cfg
        grav_z = self.robot.data.projected_gravity_b[:, 2]
        root_z = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        base_force = torch.norm(self.contact_sensor.data.net_forces_w_history[:, -1, 0, :], dim=-1)
        return (grav_z > c.fall_gravity_z_threshold) | (root_z < c.min_base_height) | (base_force > c.undesired_contact_force)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._terminated(), time_out

    # ------------------------------------------------------------------ #
    # resets: randomize root state, delays, reset-mode DR                  #
    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        c = self.cfg
        super()._reset_idx(env_ids)

        dr.randomize_root_state(self.robot, env_ids, pose_range=c.dr_reset_pose, velocity_range={})
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )

        lo, hi = c.height_range
        self._height_cmd[env_ids] = lo + (hi - lo) * torch.rand(env_ids.numel(), device=self.device)
        self._spring_rand[env_ids] = (2 * torch.rand(env_ids.numel(), 2, device=self.device) - 1) * c.spring_settings["rand_force"]

        if c.use_act_delay:
            self._act_delay.reset(env_ids)
            self._act_delay.resample_uniform(*c.act_delay_range, env_ids=env_ids)
        if c.use_obs_delay:
            for buf in (self._obs_imu, self._obs_pos, self._obs_vel):
                buf.reset(env_ids)
                buf.resample_uniform(*c.obs_delay_range, env_ids=env_ids)

        # reset-mode DR, gated to once per 720 control steps (defaults-restore semantics);
        # sampled values feed the privileged critic readout honestly
        if self.common_step_counter - self._dr_last_rewrite >= self._dr_gate_steps:
            self._dr_last_rewrite = self.common_step_counter
            gains_scale = dr.randomize_actuator_gains(self.robot, env_ids, c.dr_gains)
            wheel_fric = dr.randomize_joint_friction(self.robot, env_ids, [".*_wheel_joint"],
                                                     c.dr_wheel_friction_add)
            leg_fric = dr.randomize_joint_friction(self.robot, env_ids, [".*_front1_joint", ".*_rear1_joint"],
                                                   c.dr_leg_friction_add)
            if c.privileged_dr_readout:
                self._dr_readout[env_ids, 0] = gains_scale.squeeze(-1)
                self._dr_readout[env_ids, 1] = leg_fric.mean(-1)
                self._dr_readout[env_ids, 2] = wheel_fric.mean(-1)

        # resample the decoded-action disturbance params
        for slot, key in ((self._act_scale, "scale"), (self._act_bias, "bias"), (self._act_std, "noise_std")):
            lo, hi = c.dr_action_noise[key]
            slot[env_ids] = lo + (hi - lo) * torch.rand((env_ids.numel(), 1), device=self.device)

        if env_ids.numel() == self.num_envs:
            self.command_generator.resample_all()
