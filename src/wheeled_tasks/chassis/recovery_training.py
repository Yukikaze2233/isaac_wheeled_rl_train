"""Physical recovery prelude and RL takeover, without post-reset state writes.

Only reset profiles and outcome grading use simulator truth. The reference FSM
gets encoder/FK and simulated IMU signals; it never receives terrain or contacts.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch

from .mechanism_observer import ClosedChainHeightModel
from .continuous_encoders import ContinuousEncoderAngles
from .recovery import RecoveryFSM, RecoveryPhase
from .recovery_observer import SimulatedImu, WheelSupportProbe
from .step_trigger import estimate_height_if_grounded


def read_frozen(root, settings, prefix):
    path = root / settings[prefix + "_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != settings[prefix + "_sha256"]:
        raise ValueError(f"Recovery {prefix} differs from frozen contract")
    return json.loads(path.read_text())


class RecoveryTraining:
    PREMATURE_RELEASE_FAILURE = 7

    def __init__(self, env, root, settings):
        self.env, self.settings = env, dict(settings)
        table = read_frozen(root, settings, "profiles")
        self.geometry = ClosedChainHeightModel(read_frozen(root, settings, "height_lookup"), env.device)
        relative_limits = torch.stack([side["delta_rad"][[0, -1]] for side in self.geometry.sides])
        self.encoders = ContinuousEncoderAngles(env.num_envs, env.device, relative_limits)
        self.reference_dt = settings.get("reference_dt", .005)
        if env.dt != .001 or env.policy_dt != .02 or self.reference_dt != .005:
            raise ValueError("Recovery requires 1 kHz PD, 200 Hz references and 50 Hz policy")
        self.reference_stride = 5
        self.reference_tick = 0
        cfg = {key: settings[key] for key in RecoveryFSM.DEFAULTS if key in settings}
        cfg["dt"] = self.reference_dt
        targets = [table["targets"][name]["control6_rad"] for name in ("nominal", "fold", "thrust", "support")]
        # A geometric rest pose alone cannot hold the gas-spring/gravity load
        # under finite PD stiffness. Keep its model-derived command correction
        # separate from the physical poses used for episode initialization.
        reference = table["prepare_reference"]
        geometry_q, command_q, load = (torch.as_tensor(reference[key], dtype=torch.float32, device=env.device)
            for key in ("geometry_control6_rad", "pd_control6_rad", "hold_torque_nm"))
        stiffness = env.v5.settings["leg_kp"]
        if (reference["kp"] != stiffness or any(t.shape != (6,) or not torch.isfinite(t).all()
                for t in (geometry_q, command_q, load))
                or not torch.allclose(command_q, geometry_q + load / stiffness, atol=1e-6, rtol=0.)
                or load.abs().max() > 40. or not command_q[[2, 5]].eq(0.).all()
                or not self.geometry.wheel_positions(geometry_q[None])[1].all()):
            raise ValueError("Recovery prepare reference must authenticate geometry plus static load / native PD stiffness")
        targets[0] = targets[3] = command_q
        self.fsm = RecoveryFSM(env.num_envs, env.device, cfg, *targets, table["axis_signs"])
        self.wheel_balance = {"enabled": False, "pitch_kp_nm_rad": 8., "pitch_kd_nm_s_rad": 1.5,
            "wheel_damping_nm_s_rad": .2, "prepare_tilt_limit_deg": 55., "capture_tilt_limit_deg": 85.}
        balance = settings.get("wheel_balance", {})
        if set(balance) - self.wheel_balance.keys():
            raise ValueError("Unknown recovery wheel-balance configuration")
        self.wheel_balance.update(balance)
        if (type(self.wheel_balance["enabled"]) is not bool
                or any(not math.isfinite(value) or value <= 0. for key, value in self.wheel_balance.items() if key != "enabled")
                or not 0. < self.wheel_balance["prepare_tilt_limit_deg"] <= self.wheel_balance["capture_tilt_limit_deg"] < 90.):
            raise ValueError("Invalid recovery wheel-balance gains or support angles")
        # A collision cylinder's axis is unoriented: both meshes may use +Y
        # while the right motor's joint axis is -Y. The signed motor axes come
        # from the frozen profile builder's model-spec FK, not collision axes.
        self.wheel_axis_signs = torch.as_tensor(table.get("wheel_axis_signs", []), dtype=torch.float32, device=env.device)
        if (self.wheel_axis_signs.shape != (2,) or not self.wheel_axis_signs.abs().eq(1.).all()
                or not torch.allclose(self.geometry.wheel_axles_b[:, 1].abs(),
                                      torch.ones(2, device=env.device), atol=.001, rtol=0.)):
            raise ValueError("Recovery requires two frozen signed wheel-motor axes and near-Y collision axes")
        if not math.isfinite(env.v5.wheel_prior["kd"]) or env.v5.wheel_prior["kd"] <= 0.:
            raise ValueError("Recovery wheel targets require a positive native velocity-loop gain")
        if table["control_joint_names"] != env.manifest["control_joint_names"]:
            raise ValueError("Recovery control order differs from the asset")
        expected_manifest = table.get("asset_manifest_sha256")
        if expected_manifest and expected_manifest != env.cfg["asset_manifest_sha256"]:
            raise ValueError("Recovery profiles belong to a different mechanical asset")
        self.imu = SimulatedImu(env.device, env.num_envs, self.reference_dt)
        self.probe = WheelSupportProbe(env.device, env.num_envs, self.reference_dt)
        n, d = env.num_envs, env.device
        self.enabled = torch.zeros(n, dtype=torch.bool, device=d)
        self.root_pose = torch.zeros(n, 7, device=d)
        self.joint_pose = env.nominal.clone()
        self.initial_release = torch.zeros(n, device=d)
        self.followup_command = torch.zeros(n, 3, device=d)
        self.followup_command[:, 2] = .305
        self.required_platform_height = torch.zeros(n, device=d)
        self.min_ending_x = torch.full((n,), -math.inf, device=d)
        self.profile_names = []
        for index, group in enumerate(env.scene_groups):
            spec = env.skills.specs.get(group, {})
            name = spec.get("recovery_pose")
            self.profile_names.append(name)
            if name is None:
                continue
            profile = table["profiles"][name]
            if not profile["enabled"]:
                raise ValueError(f"Unvalidated recovery initial geometry: {name}")
            if profile.get("terrain", "flat") != env.kinds[index]:
                raise ValueError(f"Recovery initial terrain differs: {name}")
            self.enabled[index] = True
            self.root_pose[index] = torch.tensor(profile["root_position_m"] + profile["root_quaternion_xyzw"], device=d)
            q = profile["full_joint_positions"]
            self.joint_pose[index] = torch.tensor([q[key] for key in env.robot.joint_names], device=d)
            self.initial_release[index] = profile.get("initial_release_s", 0.)
            self.required_platform_height[index] = profile.get("required_platform_height_m", 0.)
            self.min_ending_x[index] = profile.get("min_ending_x", -math.inf)
            self.followup_command[index] = torch.tensor(spec.get("recovery_followup_command", [0., 0., .305]), device=d)
        self.has_tasks = bool(self.enabled.any())
        self.elapsed = torch.zeros(n, device=d)
        self.release_finished = ~self.enabled
        self.true_stable_seconds = torch.zeros(n, device=d)
        self.true_stable_peak = torch.zeros(n, device=d)
        self.followup_seconds = torch.zeros(n, device=d)
        self.ever_blended = torch.zeros(n, dtype=torch.bool, device=d)
        self.handover_count = torch.zeros(n, dtype=torch.long, device=d)
        self.true_success = torch.zeros(n, dtype=torch.bool, device=d)
        self.motion_release_observed = torch.zeros(n, dtype=torch.bool, device=d)
        self.premature_release = torch.zeros(n, dtype=torch.bool, device=d)
        self.wheel_probe_torque = torch.zeros(n, 2, device=d)
        self.observed_height = torch.zeros(n, device=d)
        self.observed_support = torch.zeros(n, dtype=torch.bool, device=d)
        self.learning_mask = torch.ones(n, dtype=torch.bool, device=d)
        self.wheel_balance_active = torch.zeros(n, dtype=torch.bool, device=d)
        self.goal_platform = self.required_platform_height == 0.

    @property
    def pure_rl(self):
        return (~self.enabled) | (self.release_finished & self.fsm.pure_rl & ~self.fsm.failed & ~self.premature_release)

    @property
    def prelude(self):
        return self.enabled & ~self.pure_rl

    @property
    def failed(self):
        return (self.fsm.failed | self.premature_release
                | (self.prelude & (self.elapsed >= self.settings.get("max_script_seconds", 8.))))

    def failure_reasons(self):
        """Attribute terminal outcomes independently of the PPO interval mask."""
        other = self.failed & ~self.premature_release
        handed_over = self.handover_count > 0
        return {"recovery_script": other & ~handed_over,
                "recovery_actor_failure": other & handed_over,
                "recovery_premature_release": self.premature_release.clone()}

    def initial_state(self, ids, root, joints, velocity):
        active = self.enabled[ids]
        chosen = ids[active]
        root[active] = self.root_pose[chosen]
        root[active, :3] += self.env.origins[chosen]
        joints[active] = self.joint_pose[chosen]
        velocity[active] = 0.

    def reset(self, ids):
        env = self.env
        transport = getattr(env, "usb_transport", None)
        q = env.robot.data.joint_pos.torch[:, env.ids] if transport is None else transport.feedback[:, :6]
        self.encoders.reset(ids, q)
        self.fsm.reset(ids, self.encoders.q, self.enabled & (self.initial_release == 0.))
        for name in ("elapsed", "true_stable_seconds", "true_stable_peak", "followup_seconds",
                     "ever_blended", "handover_count", "true_success", "wheel_probe_torque", "motion_release_observed",
                     "wheel_balance_active", "premature_release"):
            getattr(self, name)[ids] = 0
        self.release_finished[ids] = ~self.enabled[ids] | (self.initial_release[ids] == 0.)
        self.goal_platform[ids] = self.required_platform_height[ids] == 0.
        self.imu.previous_velocity[ids] = env.robot.data.root_link_lin_vel_w.torch[ids]
        for name in ("previous_velocity", "previous_gyro", "previous_pulse", "evidence",
                     "bad_samples", "quiet_time", "lost_time"):
            getattr(self.probe, name)[ids] = 0
        self.probe.age[ids] = math.inf
        self.override_commands()

    def override_commands(self):
        env, active = self.env, self.enabled
        env.commands[active] = self.followup_command[active]
        hold = active & (~self.release_finished | ~self.fsm.motion_released)
        env.commands[hold, :2] = 0.
        env.raw_height[active] = self.followup_command[active, 2]
        env.command_target[active] = env.commands[active, :2]
        env.command_acceleration[active] = 0.

    def begin_policy(self):
        self.learning_mask.copy_(self.pure_rl)

    def bootstrap_mask(self, terminated):
        """Evaluate after reset, but never bootstrap a physical terminal step."""
        return self.pure_rl & ~terminated

    def observe_feedback(self, q):
        """Consume the one CONTROL encoder stream once per 1 kHz feedback tick."""
        return self.encoders.update(q)

    def project(self, q):
        """Read the same continuous feedback winding without advancing it."""
        return self.encoders.project(q)

    def project_feedback(self, q):
        """Shared read-only encoder projection for other sensor consumers."""
        return self.project(q)

    def advance_reference(self, q, dq, gravity, gyro):
        env = self.env
        continuous_q = self.project(q)
        released_now = self.enabled & ~self.release_finished & (self.elapsed >= self.initial_release)
        if bool(released_now.any()):
            ids = released_now.nonzero().flatten()
            # The provider has followed zero-torque motion throughout release.
            # Keep accumulated hip turns while resolving any initial aux alias.
            self.encoders.realign(ids)
            continuous_q = self.project(q)
            self.fsm.reset(ids, continuous_q, self.enabled)
        self.release_finished |= released_now
        acceleration = self.imu.sample(env.robot.data.root_link_lin_vel_w.torch,
                                       env.robot.data.root_link_pose_w.torch[:, 3:])
        points, valid = self.geometry.wheel_positions(continuous_q)
        height, valid_height = estimate_height_if_grounded(points, gravity, self.geometry.wheel_radius_m,
            grounded=valid, sensor_valid=valid, wheel_axle_b=self.geometry.wheel_axles_b)
        self.observed_height = height.amin(-1)
        # A conditional FK height is not a contact measurement. Acceleration and
        # wheel response provide additional evidence; true contact grades later.
        plausible = (valid_height.all(-1) & (height.amin(-1) > .12)
                     & ((height[:, 0] - height[:, 1]).abs() < .035)
                     & (acceleration.norm(dim=-1) > 6.) & (acceleration.norm(dim=-1) < 16.)
                     & (gyro.norm(dim=-1) < 8.))
        confirmed = self.probe.observe(dq[:, [2, 5]], gyro, plausible)
        self.observed_support = plausible
        body_clear = (confirmed | self.fsm.blending | self.fsm.pure_rl) & plausible & (height.amin(-1) > .27)
        before = self.fsm.pure_rl.clone()
        previous_phase = self.fsm.phase.clone()
        # Release is simulated with zero active motor effort and retained gas
        # springs. Its finite duration is included in the prelude timeout.
        self.fsm.update(continuous_q, dq, gravity, gyro, self.observed_height, plausible, body_clear)
        self._update_wheel_targets(gravity, gyro, dq, plausible, previous_phase)
        self.handover_count += self.enabled & ~before & self.fsm.pure_rl
        self.ever_blended |= self.enabled & self.fsm.blending
        self.wheel_probe_torque = self.probe.command(
            self.reference_tick,
            self.enabled & self.fsm.scripted & self.release_finished & plausible, confirmed)
        self.reference_tick += 1

    def _update_wheel_targets(self, gravity, gyro, dq, support, previous_phase):
        """200 Hz sensor-only pitch stabilization through the existing wheel P loop.

        At the reference sample, P(target-speed) equals the ActivationBench
        pitch/gyro effort minus wheel damping. PD continues at 1 kHz between
        samples. The last scripted target is retained through torque blending.
        """
        wheels, settings = list(self.env.v5.WHEELS), self.wheel_balance
        tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        capture = self.fsm.phase == int(RecoveryPhase.CAPTURE)
        preparing = self.fsm.phase == int(RecoveryPhase.PREPARE)
        entered_blend = self.fsm.blending & (previous_phase == int(RecoveryPhase.PREPARE))
        limit = torch.where(capture, tilt.new_tensor(math.radians(settings["capture_tilt_limit_deg"])),
                            tilt.new_tensor(math.radians(settings["prepare_tilt_limit_deg"])))
        eligible = (self.enabled & self.release_finished & support & (tilt < limit)
                    & (preparing | capture | entered_blend) & settings["enabled"])
        kp = self.env.v5.wheel_prior["kd"]
        pitch_effort = settings["pitch_kp_nm_rad"] * pitch + settings["pitch_kd_nm_s_rad"] * gyro[:, 1]
        target = (pitch_effort[:, None] * self.wheel_axis_signs / kp
                  + (1. - settings["wheel_damping_nm_s_rad"] / kp) * dq[:, wheels])
        retained = self.fsm.blending & self.wheel_balance_active & support & (tilt < limit)
        desired = torch.where(retained[:, None], self.fsm.script_targets[:, wheels], 0.)
        desired = torch.where(eligible[:, None], target, desired)
        self.wheel_balance_active = eligible | retained
        self.fsm.script_targets[:, wheels] = desired

    def torque(self, q, dq, rl_torque):
        env = self.env
        active = self.enabled
        leg_ids, wheel_ids = list(env.v5.LEGS), list(env.v5.WHEELS)
        script_request = torch.zeros_like(q)
        # FK, reference FSM and PD all consume the same accumulated winding.
        # Never reselect the target's shortest arc: orbit lag must not reverse.
        continuous_q = self.project(q)[:, leg_ids]
        script_request[:, leg_ids] = (env.v5.settings["leg_kp"] * (self.fsm.script_targets[:, leg_ids] - continuous_q)
                                      - env.v5.settings["leg_kd"] * dq[:, leg_ids])
        script_request[:, wheel_ids] = (env.v5.wheel_prior["kd"] * (self.fsm.script_targets[:, wheel_ids] - dq[:, wheel_ids])
                                        + self.wheel_probe_torque)
        script_request *= self.release_finished[:, None]
        bound = env.v5.current_motor_bounds
        script_torque = script_request.clamp(min=-bound, max=bound)
        fraction = self.fsm.blend[:, None]
        result = (1. - fraction) * script_torque + fraction * rl_torque
        motor_enabled = self.release_finished & ~self.failed
        result[~motor_enabled] = 0.
        env.v5.requested_motor_effort[active] = ((1. - fraction) * script_request
            + fraction * env.v5.requested_motor_effort)[active]
        env.v5.requested_motor_effort[active & ~motor_enabled] = 0.
        self.elapsed += active * env.dt
        # Ownership of the elapsed interval survives an actor-caused terminal
        # disarm. FAILED supplies zero effort; it does not transfer control to
        # a script. Any release/script/blend tick still invalidates the whole
        # interval, and begin_policy prevents failed episodes gaining credit.
        actor_or_terminal = self.fsm.pure_rl | self.fsm.failed
        self.learning_mask &= ~active | (self.release_finished & actor_or_terminal)
        return torch.where(active[:, None], result, rl_torque)

    def effective_action_history(self):
        env = self.env
        script = torch.zeros_like(env.actions)
        legs = list(env.v5.LEGS)
        delta = self.fsm.script_targets[:, legs] - env.v5.nominal[legs]
        script[:, legs] = torch.atan2(delta.sin(), delta.cos()) / env.v5.settings["leg_position_scale"]
        wheels = list(env.v5.WHEELS)
        script[:, wheels] = self.fsm.script_targets[:, wheels] / env.v5.settings["wheel_velocity_scale"]
        fraction = self.fsm.blend[:, None]
        history = (1. - fraction) * script + fraction * env.actions
        history[~self.release_finished | self.failed] = 0.
        env.actions[self.prelude] = history[self.prelude]

    def outcome(self, contact, undesired, gravity, omega, height, velocity, *, goal_platform=None):
        env = self.env
        required = self.required_platform_height > 0.
        if goal_platform is not None and goal_platform.shape != required.shape:
            raise ValueError("Recovery platform evidence must match environment count")
        self.goal_platform = ~required | (torch.zeros_like(required) if goal_platform is None else goal_platform.bool())
        actual_stable = (self.enabled & self.pure_rl & contact.all(-1) & ~undesired
            & self.goal_platform
            & (gravity[:, 2] < -math.cos(math.radians(12.)))
            & (omega[:, :2].norm(dim=-1) < .75)
            & ((height - self.followup_command[:, 2]).abs() <= .05))
        stand = self.followup_command[:, :2].abs().amax(-1) < .01
        qualified_before = self.true_stable_peak >= 1. - 1e-5
        # The interval ending at the first observed release still belongs to
        # the hold. Later movement cannot manufacture missing hold evidence.
        stable_hold = actual_stable & ~self.motion_release_observed & (velocity[:, :2].norm(dim=-1) < .25)
        self.true_stable_seconds = torch.where(stable_hold,
            self.true_stable_seconds + env.policy_dt, 0.)
        self.true_stable_peak = torch.maximum(self.true_stable_peak, self.true_stable_seconds)
        released = self.enabled & self.pure_rl & self.fsm.motion_released
        first_release = released & ~self.motion_release_observed
        self.premature_release |= first_release & (self.true_stable_peak < 1. - 1e-5)
        self.motion_release_observed |= released
        tracking = ((velocity[:, 0] - self.followup_command[:, 0]).abs() <= .12)
        tracking &= (omega[:, 2] - self.followup_command[:, 1]).abs() <= .2
        tracking &= ~stand | (velocity[:, :2].norm(dim=-1) < .10)
        completing = (actual_stable & self.fsm.motion_released & tracking
                       & qualified_before)
        self.followup_seconds = torch.where(completing, self.followup_seconds + env.policy_dt, 0.)
        self.true_success = self.enabled & (self.followup_seconds >= 1. - 1e-5) & ~self.failed
        return self.true_success

    def diagnostics(self):
        failure_code = self.fsm.failure_code.clone()
        failure_code[self.failed & (failure_code == 0)] = 6
        failure_code[self.premature_release] = self.PREMATURE_RELEASE_FAILURE
        leg_error = self.fsm.script_targets[:, list(self.env.v5.LEGS)] - self.encoders.q[:, list(self.env.v5.LEGS)]
        return {**self.fsm.diagnostics(), "recovery_enabled": self.enabled.clone(),
            "recovery_failed": self.failed.clone(), "recovery_pure_rl": self.pure_rl.clone(),
            "recovery_failure_code": failure_code,
            "recovery_premature_release": self.premature_release.clone(),
            "recovery_motion_released": (self.enabled & self.release_finished & self.fsm.motion_released).clone(),
            "recovery_rl_stable_s": self.true_stable_peak.clone(),
            "recovery_followup_s": self.followup_seconds.clone(),
            "recovery_true_success": self.true_success.clone(),
            "recovery_platform_retained": self.goal_platform.clone(),
            "recovery_handover_count": self.handover_count.clone(),
            "recovery_ever_blended": self.ever_blended.clone(),
            "recovery_script_seconds": torch.minimum(self.elapsed, self.initial_release + self.fsm.age_ticks * self.reference_dt),
            "recovery_observed_height": self.observed_height.clone(),
            "recovery_observed_support": self.observed_support.clone(),
            "recovery_continuous_q": self.encoders.q.clone(),
            "recovery_leg_target_error_rad": leg_error.clone(),
            "recovery_wheel_balance_active": self.wheel_balance_active.clone(),
            "recovery_wheel_velocity_target": self.fsm.script_targets[:, list(self.env.v5.WHEELS)].clone()}
