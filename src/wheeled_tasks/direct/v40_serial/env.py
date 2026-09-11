"""Seven-body canonical V40 task, driven by the shared pure-tensor contract.

No coordinate re-rotation, custom PPO, hidden curriculum, or legacy task code.
ContactSensor forces are diagnostic rigid-body net forces, NOT proof of ground
support; pair-specific ground contact/cooking needs verification on real Isaac.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
import re
from copy import deepcopy

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from wheeled_tasks.v40.core import (
    HistoryStack, build_observation, build_critic, compute_reward_terms, compute_torques,
    contract_digest, decode_targets, load_contract, validate_asset,
    is_round2, NoisyHistoryStack, SustainedFailure,
)
from wheeled_world.assets.v40 import (
    make_v40_articulation, apply_approved_collision_filters,
    normalize_v40_usd_joint_limits, validate_v40_physx_joint_limits,
)
from .env_cfg import V40EnvCfg

BODY_NAMES = ["base_link", "L_link1", "L_link2", "L_link3", "R_link1", "R_link2", "R_link3"]
WHEEL_BODY_NAMES = ["L_link3", "R_link3"]
NON_WHEEL_BODY_NAMES = [name for name in BODY_NAMES if name not in WHEEL_BODY_NAMES]


class V40Env(DirectRLEnv):
    cfg: V40EnvCfg

    def __init__(self, cfg: V40EnvCfg, render_mode: str | None = None, **kwargs):
        self.contract = load_contract(cfg.contract_path)
        validated = validate_asset(self.contract, allow_research=cfg.allow_research)
        if validated["manifest"].get("collision_validation", {}).get("passed") is not True:
            raise ValueError("static collision_validation.passed must be true before any V40 simulation")
        self.asset_manifest = validated["manifest"]
        self.research_approval = validated.get("research_approval")
        self.raw_manifest = validated.get("raw_manifest")
        bounds = self.asset_manifest.get("base_visual_bounds_m")
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(not isinstance(row, list) or len(row) != 3 for row in bounds)
                or any(type(value) not in (int, float) or not math.isfinite(value) for row in bounds for value in row)
                or any(bounds[0][i] >= bounds[1][i] for i in range(3))):
            raise ValueError("finite ordered base-link visual bounds are required for conservative clearance")
        if cfg.scene.replicate_physics or cfg.scene.clone_in_fabric:
            raise ValueError("V40 reviewed filtered pairs require independent, inspectable USD clones")
        self.contract_sha256 = contract_digest(self.contract)
        self.asset_manifest_sha256 = validated["asset_manifest_sha256"]
        if cfg.stage not in self.contract["commands"]["stages"]:
            raise ValueError(f"unknown V40 stage: {cfg.stage}")
        timing = self.contract["timing"]
        cfg.sim.dt = timing["physics_dt"]
        cfg.decimation = timing["decimation"]
        cfg.sim.render_interval = cfg.decimation
        cfg.episode_length_s = self.contract["termination"]["episode_seconds"]
        cfg.observation_space = self.contract["observations"]["actor_dim"]
        cfg.state_space = self.contract["observations"]["critic_dim"]
        cfg.action_space = self.contract["actions"]["dimension"]
        cfg.is_finite_horizon = False
        self.joint_names = list(self.contract["joints"]["action_order"])
        legs = set(self.contract["joints"]["leg_indices"])
        motor_profiles = [self.contract["actuators"]["leg" if i in legs else "wheel"] for i in range(6)]
        cfg.robot_cfg = make_v40_articulation(
            urdf_path=validated["urdf_path"], joint_names=self.joint_names,
            nominal_positions=self.contract["joints"]["nominal_positions"],
            effort_limits=[profile["effort_limit"] for profile in motor_profiles],
            armatures=[profile["armature"] for profile in motor_profiles],
            nominal_base_height=self.contract["asset"]["nominal_base_height"],
            asset_manifest_sha256=self.asset_manifest_sha256,
            usd_cache_dir=cfg.usd_cache_dir,
        )
        body_expression = "(" + "|".join(re.escape(name) for name in BODY_NAMES) + ")"
        cfg.contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/" + body_expression,
            update_period=0.0, history_length=cfg.decimation, debug_vis=False,
        )
        super().__init__(cfg, render_mode, **kwargs)
        # DirectRLEnv initializes physics inside super().__init__. Fail before any
        # reset/rollout if the composed USD semantics did not reach every solver clone.
        self.joint_limit_physx_report = self.check_physics_joint_limits()
        self._base_visual_corners = torch.tensor(
            [(x, y, z) for x in (bounds[0][0], bounds[1][0])
             for y in (bounds[0][1], bounds[1][1]) for z in (bounds[0][2], bounds[1][2])],
            device=self.device, dtype=torch.float32,
        )
        self._joint_ids = self._named_indices(self.robot.joint_names, self.joint_names, "joint")
        self._wheel_body_ids = self._named_indices(self.contact_sensor.body_names, WHEEL_BODY_NAMES, "wheel body")
        self._non_wheel_body_ids = self._named_indices(self.contact_sensor.body_names, NON_WHEEL_BODY_NAMES, "non-wheel body")
        self._named_indices(self.robot.body_names, BODY_NAMES, "robot body")
        if len(self.robot.joint_names) != 6 or len(self.robot.body_names) != 7 or len(self.contact_sensor.body_names) != 7:
            raise RuntimeError("V40 importer/sensor must expose exactly six joints and seven bodies")
        joints = self.contract["joints"]
        self._leg_ids = torch.tensor(joints["leg_indices"], dtype=torch.long, device=self.device)
        self._knee_ids = torch.tensor(joints["knee_indices"], dtype=torch.long, device=self.device)
        self._nominal = torch.tensor(joints["nominal_positions"], device=self.device)
        self._knee_limits = torch.tensor(
            [joints["knee_hard_limits"][self.joint_names[i]] for i in joints["knee_indices"]], device=self.device,
        )
        self.actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.leg_targets = self._nominal[self._leg_ids].repeat(self.num_envs, 1)
        self.wheel_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_period_ticks = max(1, math.ceil(
            self.contract["commands"]["resample_seconds"] / self.contract["timing"]["policy_dt"],
        ))
        self._command_ticks_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._commands_due = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._invalid_actions = torch.zeros_like(self._commands_due)
        self._finite_state = torch.ones_like(self._commands_due)
        # Opt-in evaluation only. None preserves training sampling and logging behavior.
        self._evaluation_command_override = None
        self._evaluation_command_pending = False
        self._evaluation_snapshot = None
        self._last_reward_tick = -1
        self._episode_sums: dict[str, torch.Tensor] = {}
        history_type = NoisyHistoryStack if is_round2(self.contract) else HistoryStack
        self.history = history_type(
            self.num_envs, self.device,
            length=self.contract["observations"]["history_length"], dim=self.contract["observations"]["single_dim"],
        )
        if is_round2(self.contract):
            self._sustained_failure = SustainedFailure(self.num_envs, self.device)
        self._sample_commands(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def set_evaluation_command(self, command: tuple[float, float, float] | None) -> None:
        """Configure next reset's fixed (vx m/s, wz rad/s, height m); then MUST reset.

        No history is pushed here. None disables evaluation and restores the sampler
        on the next reset. This is not a mid-episode command/ramp interface.
        """
        if command is not None:
            from wheeled_tasks.v40.contract import is_round2
            from wheeled_algo.v40_metrics import InvalidTrajectory, validate_command, vector
            if is_round2(self.contract):
                command = tuple(vector(command, 3, "command"))
                stage = self.contract["commands"]["stages"][self.cfg.stage]
                for value, key in zip(command, ("vx", "wz", "height")):
                    low, high = stage[key]
                    if not low <= value <= high:
                        raise InvalidTrajectory(f"command {key} outside trained V2 stage")
            else:
                command = validate_command(self.contract, self.cfg.stage, command)
        self._evaluation_command_override = command
        self._evaluation_command_pending = True

    def _make_evaluation_snapshot(self, terminated, timeout, reasons, contact, clearance, *, kind):
        """IsaacLab 2.3.0 source-audited fields; fail on unavailable fields, never zero-fill.

        applied_torque is explicit-actuator clipped effort sent into simulation,
        not solver reaction torque, motor-side current, or a physical measurement.
        """
        from wheeled_tasks.v40.contract import is_round2
        data = self.robot.data
        wheel_ids = self._named_indices(self.robot.body_names, WHEEL_BODY_NAMES, "wheel link")
        q, dq = self._joint_state()
        snapshot = {
            "schema_version": 1, "sample_kind": kind,
            "policy_tick": int(self.common_step_counter),
            "physics_steps": int(self._sim_step_counter),
            "time_s": int(self.common_step_counter) * self.step_dt,
            "sim_time_s": int(self._sim_step_counter) * self.physics_dt,
            "policy_dt_s": self.step_dt, "physics_dt_s": self.physics_dt,
            "contact_history_samples": int(self.contact_sensor.data.net_forces_w_history.shape[1]),
            "episode_step": self.episode_length_buf,
            "episode_time_s": self.episode_length_buf * self.step_dt,
            "command": self.commands,
            "root_link_pos_w_m": data.root_link_pos_w,
            "root_link_quat_wxyz": data.root_link_quat_w,
            # These aliases are COM velocity expressed in the root link frame in 2.3.0.
            "root_com_lin_vel_b_m_s": data.root_lin_vel_b,
            "root_com_ang_vel_b_rad_s": data.root_ang_vel_b,
            "projected_gravity_b": data.projected_gravity_b,
            "height_m": data.root_link_pos_w[:, 2] - self.scene.env_origins[:, 2],
            "joint_pos_rad": q, "joint_vel_rad_s": dq,
            "sim_joint_effort_nm": data.applied_torque[:, self._joint_ids],
            # Explicit link/actor frame origins at wheel axes; NOT body_com_pos_w.
            "wheel_axis_midpoint_w_m": data.body_link_pos_w[:, wheel_ids].mean(dim=1),
            "wheel_net_force_max_n": contact[:, self._wheel_body_ids],
            "non_wheel_net_force_max_n": contact[:, self._non_wheel_body_ids].amax(dim=1),
            "base_visual_clearance_lower_bound_m": clearance,
            "terminated": terminated, "timeout": timeout, "termination_flags": reasons,
        }
        if is_round2(self.contract):
            snapshot["evaluation_settings"] = {
                "contract_id": self.contract["contract_id"],
                "observation_noise": False,
                "root_reset_velocity": self.contract["reset"]["evaluation_root_velocity"],
                "tilt_flag_semantics": "projected_gravity_z_gt_minus_0.1_for_more_than_1s",
            }
        return deepcopy(snapshot)  # detach ownership before reward / DirectRLEnv auto-reset

    def get_evaluation_snapshot(self) -> dict:
        """Return an owned copy of the last PRE-reset policy tick; never read live state."""
        if self._evaluation_snapshot is None:
            raise RuntimeError("no evaluation policy snapshot; enable override and step first")
        return deepcopy(self._evaluation_snapshot)

    def capture_evaluation_initial_snapshot(self) -> dict:
        """Read the reset-time wheel-axis anchor once, without stepping or pushing history."""
        if (self._evaluation_command_override is None or self._evaluation_command_pending
                or bool(self.episode_length_buf.any())):
            raise RuntimeError("initial snapshot requires evaluation command followed by reset")
        flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._make_evaluation_snapshot(
            flags, flags, {}, self._contact_magnitudes(), self._base_visual_clearance(), kind="initial",
        )

    def _named_indices(self, actual: list[str], requested: list[str], kind: str) -> torch.Tensor:
        if any(actual.count(name) != 1 for name in requested):
            raise RuntimeError(f"missing/ambiguous {kind} names: expected {requested}, received {actual}")
        return torch.tensor([actual.index(name) for name in requested], device=self.device, dtype=torch.long)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact"] = self.contact_sensor
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=True)
        # Use the stage owned by DirectRLEnv's use_stage context (also for in-memory stages).
        stage = self.sim.get_initial_stage()
        self.joint_limit_usd_report = normalize_v40_usd_joint_limits(
            stage, self.scene.env_prim_paths, self.contract["joints"],
        )
        self.collision_filter_report = apply_approved_collision_filters(
            stage, self.scene.env_prim_paths, self.asset_manifest,
            research_approval=self.research_approval, raw_manifest=self.raw_manifest,
        )
        # Independent clones need explicit cross-env filtering on GPU as well as CPU.
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def check_physics_joint_limits(self) -> dict:
        """Fresh readback shared by startup and bounded simulator diagnostics."""
        return validate_v40_physx_joint_limits(self.robot, self.contract["joints"], num_envs=self.num_envs)

    def _joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.data.joint_pos[:, self._joint_ids], self.robot.data.joint_vel[:, self._joint_ids]

    def _base_height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    def _base_visual_clearance(self) -> torch.Tensor:
        """Lower bound from eight rotated base-link AABB corners, NOT COM/mesh distance."""
        w, x, y, z = self.robot.data.root_quat_w.unbind(-1)
        corners = self._base_visual_corners
        # World-z row of the canonical base-link quaternion rotation matrix.
        rotated_z = (2 * (x * z - w * y))[:, None] * corners[None, :, 0]
        rotated_z += (2 * (y * z + w * x))[:, None] * corners[None, :, 1]
        rotated_z += (1 - 2 * (x.square() + y.square()))[:, None] * corners[None, :, 2]
        return self._base_height() + rotated_z.amin(dim=1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 6):
            raise ValueError("expected N x 6 actions")
        self.previous_actions.copy_(self.actions)
        self._invalid_actions = ~torch.isfinite(actions).all(dim=-1)
        safe_actions = torch.where(self._invalid_actions[:, None], torch.zeros_like(actions), actions)
        joint_pos, _ = self._joint_state()
        self.leg_targets, self.wheel_targets, clipped = decode_targets(safe_actions, joint_pos, self.contract)
        self.actions.copy_(clipped)  # o_(t+1) sees the just-executed a_t, never a_(t-1).

    def _apply_action(self) -> None:
        # DirectRLEnv calls this EACH physics substep: feedback must not be frozen at policy rate.
        joint_pos, joint_vel = self._joint_state()
        finite = (torch.isfinite(joint_pos).all(-1) & torch.isfinite(joint_vel).all(-1)
                  & ~self._invalid_actions)
        self._invalid_actions |= ~finite
        self.torques.zero_()
        if finite.any():
            self.torques[finite] = compute_torques(
                joint_pos[finite], joint_vel[finite], self.leg_targets[finite], self.wheel_targets[finite], self.contract,
            )
        bad_effort = ~torch.isfinite(self.torques).all(-1)
        self._invalid_actions |= bad_effort
        self.torques[bad_effort] = 0.0
        self.robot.set_joint_effort_target(self.torques, joint_ids=self._joint_ids)

    def _contact_magnitudes(self) -> torch.Tensor:
        # N x history x bodies x xyz. Use both substeps, not only the final instant.
        history = self.contact_sensor.data.net_forces_w_history
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        from wheeled_tasks.v40.contract import is_round2
        joint_pos, joint_vel = self._joint_state()
        body = self.robot.data
        contact = self._contact_magnitudes()
        clearance = self._base_visual_clearance()
        self._finite_state = (
            torch.isfinite(body.root_state_w).all(-1)
            & torch.isfinite(joint_pos).all(-1) & torch.isfinite(joint_vel).all(-1)
            & torch.isfinite(contact).all(-1) & torch.isfinite(self.torques).all(-1)
            & torch.isfinite(clearance)
            & ~self._invalid_actions
        )
        limits = self.contract["termination"]
        non_wheel_contact = (contact[:, self._non_wheel_body_ids] > limits["contact_force_threshold"]).any(-1)
        knee_q = joint_pos[:, self._knee_ids]
        tol = limits["knee_limit_tolerance"]
        knee_out = ((knee_q < self._knee_limits[:, 0] - tol) | (knee_q > self._knee_limits[:, 1] + tol)).any(-1)
        # projected_gravity_b is already in canonical body axes; upright is [0,0,-1].
        tilt = -body.projected_gravity_b[:, 2] < math.cos(math.radians(limits["max_tilt_deg"]))
        low = self._base_height() < limits["min_base_height"]
        base_bounds_ground = clearance <= 0.0
        terminated = ~self._finite_state | non_wheel_contact | knee_out | tilt | low | base_bounds_ground
        diagnostics = {"non_wheel_contact": non_wheel_contact, "knee_limit": knee_out,
                       "tilt": tilt, "low_height": low, "base_visual_bounds_ground": base_bounds_ground}
        reasons = {"nonfinite": ~self._finite_state, **diagnostics}
        if is_round2(self.contract):
            self._finite_state &= torch.isfinite(body.projected_gravity_b).all(-1)
            sustained = self._sustained_failure.update(
                body.projected_gravity_b[:, 2], int(self.common_step_counter), self.contract,
            )
            reasons = {name: torch.zeros_like(tilt) for name in diagnostics}
            reasons.update(nonfinite=~self._finite_state, tilt=sustained)
            terminated = ~self._finite_state | sustained
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if getattr(self, "_evaluation_command_override", None) is not None:
            tick = int(self.common_step_counter)
            if self._evaluation_snapshot is None or self._evaluation_snapshot["policy_tick"] != tick:
                self._evaluation_snapshot = self._make_evaluation_snapshot(
                    terminated, time_out,
                    reasons,
                    contact, clearance, kind="pre_reset",
                )
        # Runner log buffers retain dict references: never mutate an older step's log.
        self.extras["log"] = {}
        log = self.extras["log"]
        for name, flag in {**reasons, "timeout": time_out}.items():
            log[f"Termination/{name}"] = flag.float().mean()
        if is_round2(self.contract):
            for name, flag in diagnostics.items():
                log[f"Diagnostic/{name}"] = flag.float().mean()
            log["Diagnostic/failure_gravity"] = (body.projected_gravity_b[:, 2] > limits["failure_gravity_z"]).float().mean()
            log["Diagnostic/failure_ticks"] = self._sustained_failure.count.float().mean()
        log["Geometry/base_visual_clearance_lower_bound_m"] = torch.nan_to_num(
            clearance, nan=0.0, posinf=0.0, neginf=0.0,
        ).mean()
        for side, body_id in zip(("left", "right"), self._wheel_body_ids):
            force = torch.nan_to_num(contact[:, body_id], nan=0.0, posinf=0.0, neginf=0.0)
            log[f"Contact/{side}_wheel_net_force_n"] = force.mean()
            log[f"Contact/{side}_wheel_contact_candidate"] = (force > limits["contact_force_threshold"]).float().mean()
        log["Contact/non_wheel_force_n"] = torch.nan_to_num(contact[:, self._non_wheel_body_ids], nan=0.0, posinf=0.0, neginf=0.0).amax(-1).mean()
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        from wheeled_tasks.v40.contract import is_round2
        joint_pos, _ = self._joint_state()
        data = self.robot.data
        valid = self._finite_state
        # Nonfinite terminal rows get no nonterminal rates; do not poison PPO with NaNs.
        terms = compute_reward_terms(
            data.root_lin_vel_b[valid], data.root_ang_vel_b[valid], data.projected_gravity_b[valid],
            self._base_height()[valid], self.commands[valid], self.actions[valid], self.previous_actions[valid],
            self.torques[valid], joint_pos[valid], self.contract,
        )
        total = torch.zeros(self.num_envs, device=self.device)
        log = self.extras.setdefault("log", {})
        for name, step_reward in terms.items():
            if not torch.isfinite(step_reward).all():
                raise RuntimeError(f"non-finite V40 reward term: {name}")
            value = torch.zeros_like(total)
            # core already applies both the contract weight and policy_dt exactly once.
            value[valid] = step_reward
            total += value
            log[f"Reward/{name}"] = value.mean()
            self._episode_sums.setdefault(name, torch.zeros_like(total)).add_(value)
        terminal = self.reset_terminated.float() * self.contract["rewards"]["termination_penalty"]
        total += terminal  # A terminal event penalty is not a time-integrated rate.
        log["Reward/termination"] = terminal.mean()
        self._episode_sums.setdefault("termination", torch.zeros_like(total)).add_(terminal)
        for name, value in {
            "vx_m_s": data.root_lin_vel_b[:, 0], "wz_rad_s": data.root_ang_vel_b[:, 2],
            "height_m": self._base_height(),
            "vx_abs_error": (data.root_lin_vel_b[:, 0] - self.commands[:, 0]).abs(),
            "wz_abs_error": (data.root_ang_vel_b[:, 2] - self.commands[:, 1]).abs(),
            "height_abs_error": (self._base_height() - self.commands[:, 2]).abs(),
            "planar_speed_m_s": torch.linalg.vector_norm(data.root_lin_vel_b[:, :2], dim=-1),
        }.items():
            log[f"Tracking/{name}"] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).mean()
        if is_round2(self.contract):
            log["Command/standing_fraction"] = (self.commands[:, :2] == 0.0).all(-1).float().mean()
        # Reward and tracking above use the command that produced this action.
        # Sampling is deferred until _get_observations, after terminal resets.
        tick = int(self.common_step_counter)
        if tick != self._last_reward_tick:
            self._command_ticks_left -= 1
            self._commands_due |= self._command_ticks_left <= 0
            self._last_reward_tick = tick
        return total

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        from wheeled_tasks.v40.contract import is_round2
        if env_ids.numel() == 0:
            return
        override = getattr(self, "_evaluation_command_override", None)
        if override is not None:
            self.commands[env_ids] = torch.tensor(override, device=self.device, dtype=self.commands.dtype)
            self._command_ticks_left[env_ids] = self._command_period_ticks
            self._commands_due[env_ids] = False
            return
        stage = self.contract["commands"]["stages"][self.cfg.stage]
        for column, key in enumerate(("vx", "wz", "height")):
            low, high = stage[key]
            self.commands[env_ids, column] = low + (high - low) * torch.rand(len(env_ids), device=self.device)
        probability = stage.get("standing_probability", 0.0) if is_round2(self.contract) else 0.0
        if probability > 0.0:
            standing = torch.rand(len(env_ids), device=self.device) < probability
            # Match the upstream standing velocity bucket; retain the sampled height.
            self.commands[env_ids[standing], :2] = 0.0
        self._command_ticks_left[env_ids] = self._command_period_ticks
        self._commands_due[env_ids] = False

    def _get_observations(self) -> dict[str, torch.Tensor]:
        from wheeled_tasks.v40.contract import is_round2
        if getattr(self, "_evaluation_command_pending", False):
            raise RuntimeError("reset required after set_evaluation_command; no same-tick history rewrite")
        self._sample_commands(self._commands_due.nonzero(as_tuple=False).flatten())
        joint_pos, joint_vel = self._joint_state()
        data = self.robot.data
        obs25 = build_observation(
            data.root_ang_vel_b, data.projected_gravity_b, self.commands,
            joint_pos, joint_vel, self.actions, self.contract,
        )
        # PPO's pending transition must retain o_t while the history advances to o_(t+1).
        if is_round2(self.contract):
            policy = self.history.update_actor(
                obs25, tick=int(self.common_step_counter), contract=self.contract,
                enabled=self._evaluation_command_override is None,
            )
        else:
            policy = self.history.update(obs25, tick=int(self.common_step_counter))
        critic = build_critic(obs25, data.root_lin_vel_b, self._base_height())
        return {"policy": policy, "critic": critic}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        from wheeled_tasks.v40.contract import is_round2
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if self._episode_sums:
            episode_log = self.extras.setdefault("log", {})
            duration = (self.episode_length_buf[env_ids].float() * self.step_dt).clamp_min(self.step_dt)
            for name, sums in self._episode_sums.items():
                episode_log[f"Episode_Reward/{name}_per_second"] = (sums[env_ids] / duration).mean()
                sums[env_ids] = 0.0
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._nominal
        joint_vel = torch.zeros_like(joint_pos)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        root[:, 7:] = 0.0  # Preserve the default root quaternion, not a hardcoded replacement.
        if is_round2(self.contract) and self._evaluation_command_override is None:
            low, high = self.contract["reset"]["root_velocity_range"]
            root[:, 7:] = low + (high - low) * torch.rand_like(root[:, 7:])
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.leg_targets[env_ids] = self._nominal[self._leg_ids]
        self.wheel_targets[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        self._command_ticks_left[env_ids] = 0
        self._commands_due[env_ids] = False
        self._invalid_actions[env_ids] = False
        self._finite_state[env_ids] = True
        self.history.reset(env_ids)
        if is_round2(self.contract):
            self._sustained_failure.reset(env_ids)
        self._sample_commands(env_ids)
        self._evaluation_command_pending = False
        # Do NOT clear the pre-reset snapshot here: DirectRLEnv auto-resets before returning.
        self.robot.set_joint_effort_target(self.torques[env_ids], joint_ids=self._joint_ids, env_ids=env_ids)
