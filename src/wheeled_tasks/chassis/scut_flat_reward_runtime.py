"""Physical observers for the explicit flat baseline flat / rough baseline rough rewards.

No simulator is imported here. ChassisEnv supplies refreshed native data at
every 5 ms step and the incoming actor output before control reordering or
clipping. This module does not choose commands, certify contacts, or train.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from .scut_flat_reward import (CONTACT_HISTORY_SAMPLES, REWARD_VERSION,
                               ROUGH_REWARD_VERSION, SOURCE_COMMIT, WEIGHTS,
                               ROUGH_WEIGHTS, ScutFlatRewardFrame,
                               ScutFlatRewardResult,
                               scut_flat_reward, scut_rough_reward)
from .scut_observation import POLICY_FROM_CONTROL
from .scut_reward_selection import validate_command_decoupling


ROOT_VELOCITY_REFERENCE = "root_rigid_body_com_in_root_link_frame"
WHEEL_POSITION_REFERENCE = "wheel_link_origins_in_root_link_frame"


def root_angles_xyzw(quaternion):
    """Wrapped XYZ Euler roll/pitch for the local SDK's xyzw convention."""
    x, y, z, w = quaternion.unbind(-1)
    roll = torch.atan2(2. * (w * x + y * z), 1. - 2. * (x.square() + y.square()))
    pitch = torch.asin((2. * (w * y - z * x)).clamp(-1., 1.))
    return torch.atan2(roll.sin(), roll.cos()), torch.atan2(pitch.sin(), pitch.cos())


def wheel_link_positions_body(root_pose, wheel_link_positions_world):
    """Inverse rotation about the root LINK origin; no wheel collision offset."""
    relative = wheel_link_positions_world - root_pose[:, None, :3]
    xyz, w = root_pose[:, None, 3:6], root_pose[:, None, 6:7]
    cross = torch.linalg.cross(xyz.expand_as(relative), relative)
    return relative - 2. * w * cross + 2. * torch.linalg.cross(xyz.expand_as(relative), cross)


class ScutFlatRewardRuntime:
    def __init__(self, count, device, body_names, mapping, *, reward_profile="scut0719_flat_v1", physics_dt_s=.005,
                 flat_rehearsal_mask=None, flat_rehearsal_reward_profile="scut0719_flat_v1",
                 command_decoupling=None, tracking_reward_repair=None):
        self.tracking_reward_repair = None
        if tracking_reward_repair is not None:
            from .tracking_reward_repair import validate
            self.tracking_reward_repair = validate(tracking_reward_repair)
        if physics_dt_s not in (.005, .001):
            raise ValueError("reference contact observer supports 200Hz training or 1kHz replay")
        self.physics_dt_s = physics_dt_s
        if reward_profile not in ("scut0719_flat_v1", "scut0730_rough_v1"):
            raise ValueError("Unknown reference reward runtime profile")
        self.is_rough = reward_profile == "scut0730_rough_v1"
        self.reward_profile = reward_profile
        self.height_reference = ("root_link_z_minus_base_ground_scan" if self.is_rough
                                 else "root_link_z_minus_env_origin")
        if mapping["root_velocity_reference"] != ROOT_VELOCITY_REFERENCE:
            raise ValueError("reference reward requires explicit root rigid-body COM velocity")
        if mapping["wheel_position_reference"] != WHEEL_POSITION_REFERENCE:
            raise ValueError("reference reward requires wheel link origins in the root link frame")
        if mapping["root_height_reference"] != self.height_reference:
            raise ValueError("reference reward height datum differs from its selected archive")
        if len(set(body_names)) != len(body_names):
            raise ValueError("Native body names must be unique")
        self.body_names = tuple(body_names)
        self.undesired_ids = self._body_ids(mapping["undesired_link_names"])
        self.terminal_ids = self._body_ids(mapping["terminal_link_names"])
        self.raw_action = torch.zeros(count, 6, device=device)
        self.command_decoupling = validate_command_decoupling(command_decoupling)
        if self.command_decoupling and self.command_decoupling["enabled"]:
            coefficient = torch.tensor(self.command_decoupling["coefficient_per_rad2_s2"],
                                       dtype=self.raw_action.dtype)
            if not bool(torch.isfinite(coefficient)) or coefficient <= 0:
                raise ValueError("V7 yaw cost coefficient must fit the reward runtime precision")
        self.flat_rehearsal_mask = None
        if flat_rehearsal_reward_profile not in ("scut0719_flat_v1", "scut0730_rough_v1"):
            raise ValueError("Source flat rehearsal requires an explicit 0719 or 0730 reward profile")
        if flat_rehearsal_mask is None and flat_rehearsal_reward_profile != "scut0719_flat_v1":
            raise ValueError("A rehearsal reward profile requires an explicit rehearsal mask")
        self.flat_rehearsal_reward_profile = flat_rehearsal_reward_profile
        if flat_rehearsal_mask is not None:
            if not self.is_rough:
                raise ValueError("Flat rehearsal reward selection requires the rough runtime")
            if (not isinstance(flat_rehearsal_mask, torch.Tensor)
                    or flat_rehearsal_mask.shape != (count,) or flat_rehearsal_mask.dtype != torch.bool
                    or flat_rehearsal_mask.device != self.raw_action.device):
                raise ValueError("Flat rehearsal reward selection requires bool[N] on the runtime device")
            # Fixed lane ownership must not change when the caller reuses its
            # allocation tensor. Reset clears physical history, not ownership.
            self.flat_rehearsal_mask = flat_rehearsal_mask.detach().clone()
        self.previous_raw_action = torch.zeros_like(self.raw_action)
        self.before_previous_raw_action = torch.zeros_like(self.raw_action)
        self.joint_velocity = torch.zeros_like(self.raw_action)
        self.joint_acceleration = torch.zeros_like(self.raw_action)
        self.applied_torque = torch.zeros_like(self.raw_action)
        self.net_contact = torch.zeros(count, len(body_names), 3, device=device)
        self.contact_history = torch.zeros(count, round(CONTACT_HISTORY_SAMPLES * .005 / physics_dt_s),
                                           len(self.undesired_ids), 3, device=device)
        self.terminal_counter = torch.zeros(count, dtype=torch.long, device=device)
        self.raw_terminal = torch.zeros(count, dtype=torch.bool, device=device)
        self.confirmation_policy_ticks = 20
        self.physics_samples = 0

    def _body_ids(self, names):
        if not names or len(set(names)) != len(names):
            raise ValueError("reference body sets must be explicit, nonempty and unique")
        missing = set(names) - set(self.body_names)
        if missing:
            raise ValueError(f"SCUT contact bodies missing from native articulation: {sorted(missing)}")
        return [self.body_names.index(name) for name in names]

    def reset(self, ids):
        for value in (self.raw_action, self.previous_raw_action, self.before_previous_raw_action,
                      self.joint_velocity, self.joint_acceleration, self.applied_torque,
                      self.net_contact, self.contact_history, self.terminal_counter, self.raw_terminal):
            value[ids] = 0

    def begin_policy(self, incoming_actor_actions):
        if incoming_actor_actions.shape != self.raw_action.shape:
            raise ValueError("reference raw actor actions require [environments,6]")
        self.before_previous_raw_action.copy_(self.previous_raw_action)
        self.previous_raw_action.copy_(self.raw_action)
        self.raw_action.copy_(incoming_actor_actions)

    def observe_physics(self, velocity_control, acceleration_control, applied_torque_control):
        """Native six active axes, AFTER the 5 ms update; preserve physical units."""
        for target, value in ((self.joint_velocity, velocity_control),
                              (self.joint_acceleration, acceleration_control),
                              (self.applied_torque, applied_torque_control)):
            if value.shape != target.shape:
                raise ValueError("reference native active-axis data require [environments,6]")
            target.copy_(value[:, POLICY_FROM_CONTROL])

    def finish_contacts(self):
        """Called once per physics step, after ALL native contact groups refresh."""
        self.contact_history[:, :-1] = self.contact_history[:, 1:].clone()
        self.contact_history[:, -1] = self.net_contact[:, self.undesired_ids]
        self.physics_samples += 1

    def termination(self, root_velocity_body, angular_velocity_body, root_quaternion_xyzw,
                    *, scripted_interval=None):
        """Confirm contact/attitude for configured policy ticks; numeric immediate."""
        roll, pitch = root_angles_xyzw(root_quaternion_xyzw)
        contact = (self.net_contact[:, self.terminal_ids].norm(dim=-1) > 1.).any(-1)
        attitude = (roll.abs() > math.radians(40.)) | (pitch.abs() > math.radians(40.))
        numerical = torch.zeros_like(self.raw_terminal)
        for value, limit in ((self.joint_velocity, 500.), (root_velocity_body, 100.),
                             (angular_velocity_body, 200.)):
            numerical |= ~torch.isfinite(value).all(-1) | (value.abs().amax(-1) > limit)
        # The source removes immediate failures before updating its delayed
        # terminal counter, even when contact or attitude is also bad.
        self.raw_terminal.copy_((contact | attitude) & ~numerical)
        if scripted_interval is not None:
            if (scripted_interval.shape != self.raw_terminal.shape or scripted_interval.dtype != torch.bool
                    or scripted_interval.device != self.raw_terminal.device):
                raise ValueError("Recovery terminal exemption requires bool[N] on the source device")
            # Clear confirmation history, not just its final result. Otherwise
            # a script's legal shell contact terminates its first RL interval.
            self.raw_terminal &= ~scripted_interval
        self.terminal_counter.copy_(torch.where(self.raw_terminal, self.terminal_counter + 1, 0))
        confirmed = self.terminal_counter >= self.confirmation_policy_ticks
        return confirmed | numerical, numerical, confirmed

    def reward_height(self, root_pose_world, origins, reward_ground_height_world=None):
        """Keep reward-relative rough height separate from actor/critic data."""
        if self.is_rough:
            ground = reward_ground_height_world
            if (not isinstance(ground, torch.Tensor) or ground.shape != root_pose_world[:, 2].shape
                    or ground.device != root_pose_world.device or ground.dtype != root_pose_world.dtype
                    or not bool(torch.isfinite(ground).all())):
                raise ValueError("Rough reward requires finite base-scan ground z in world coordinates")
            return root_pose_world[:, 2] - ground
        if reward_ground_height_world is not None:
            raise ValueError("Flat reward must not silently adopt a rough ground-height datum")
        return root_pose_world[:, 2] - origins[:, 2]

    def payment_contact_history(self):
        """Keep the source 3-bin frame ABI and its any-force-over-threshold test.

        At 1kHz each 5ms bin retains the actual largest-norm vector per body.
        Thus the source ANY > 3N test covers all 15 physical samples, without
        averaging away a short collision or changing the source reward core.
        """
        if self.physics_dt_s == .005:
            return self.contact_history
        bins = self.contact_history.unflatten(1, (CONTACT_HISTORY_SAMPLES, 5))
        indices = bins.norm(dim=-1).argmax(dim=2, keepdim=True)
        return bins.gather(2, indices[..., None].expand(*indices.shape, 3)).squeeze(2)

    def pay(self, *, root_velocity_body, angular_velocity_body, projected_gravity,
            root_pose_world, wheel_link_positions_world, origins, commands,
            terminated, numerical_safety_reset, reward_ground_height_world=None,
            flat_reward_height_m=None):
        """Pay one physical interval with explicit per-lane archive selection.

        Historical 0719 rehearsal requires root LINK z minus each rehearsal
        plane's origin z as a separate [N] input. Uniform 0730 payment instead
        uses the same rough scan datum on every lane and rejects that override.
        Non-rehearsal rows of the 0719 input are ignored. Physical
        NaN/Inf retain the source's per-payment sanitization/numeric-row rules.
        """
        if self.flat_rehearsal_mask is None:
            if flat_reward_height_m is not None:
                raise ValueError("Flat rehearsal height requires an explicit rehearsal mask")
        elif self.flat_rehearsal_reward_profile == "scut0730_rough_v1":
            if flat_reward_height_m is not None:
                raise ValueError("Unified 0730 rehearsal requires the shared scan height, not a flat height override")
        elif (not isinstance(flat_reward_height_m, torch.Tensor)
                or flat_reward_height_m.shape != self.flat_rehearsal_mask.shape
                or flat_reward_height_m.dtype != root_pose_world.dtype
                or flat_reward_height_m.device != root_pose_world.device):
            raise ValueError("Flat rehearsal reward requires explicit height [N] with the physical dtype/device")
        frame = self.physical_frame(root_velocity_body=root_velocity_body,
            angular_velocity_body=angular_velocity_body, projected_gravity=projected_gravity,
            root_pose_world=root_pose_world, wheel_link_positions_world=wheel_link_positions_world,
            origins=origins, commands=commands, terminated=terminated,
            numerical_safety_reset=numerical_safety_reset,
            reward_ground_height_world=reward_ground_height_world)
        result = self._pay_profile(frame, rough=self.is_rough)
        if (self.flat_rehearsal_mask is None
                or self.flat_rehearsal_reward_profile == "scut0730_rough_v1"):
            return self._pay_command_decoupling(result, frame)
        flat = self._pay_profile(replace(frame, root_height_m=flat_reward_height_m), rough=False)
        select = self.flat_rehearsal_mask
        selected = ScutFlatRewardResult(
            raw_terms={name: torch.where(select, flat.raw_terms[name], value)
                       for name, value in result.raw_terms.items()},
            payments={name: torch.where(select, flat.payments[name], value)
                      for name, value in result.payments.items()},
            total=torch.where(select, flat.total, result.total))
        return self._pay_command_decoupling(selected, frame, flat_reward_height_m)

    def _pay_profile(self, frame, *, rough):
        if self.tracking_reward_repair is not None:
            from .tracking_reward_repair import pay
            return pay(frame, rough=rough)
        return (scut_rough_reward if rough else scut_flat_reward)(frame)

    def physical_frame(self, *, root_velocity_body, angular_velocity_body, projected_gravity,
                       root_pose_world, wheel_link_positions_world, origins, commands,
                       terminated, numerical_safety_reset, reward_ground_height_world=None):
        """Expose the refreshed physical inputs without paying a second reward.

        Tensor references are live; callers consume the frame in the current
        policy interval before advancing action/contact history or commands.
        """
        _, pitch = root_angles_xyzw(root_pose_world[:, 3:7])
        return ScutFlatRewardFrame(
            root_linear_velocity_body_m_s=root_velocity_body,
            root_angular_velocity_body_rad_s=angular_velocity_body,
            projected_gravity=projected_gravity, pitch_rad=pitch,
            root_height_m=self.reward_height(root_pose_world, origins, reward_ground_height_world),
            target_height_m=commands[:, 2], command_vx_m_s=commands[:, 0],
            command_yaw_rate_rad_s=commands[:, 1],
            joint_velocity_rad_s=self.joint_velocity,
            joint_acceleration_rad_s2=self.joint_acceleration,
            applied_torque_nm=self.applied_torque,
            raw_action=self.raw_action, previous_raw_action=self.previous_raw_action,
            before_previous_raw_action=self.before_previous_raw_action,
            wheel_positions_body_m=wheel_link_positions_body(root_pose_world, wheel_link_positions_world),
            undesired_contact_forces_world_n=self.payment_contact_history(),
            terminated=terminated, numerical_safety_reset=numerical_safety_reset)

    def _pay_command_decoupling(self, result, frame, flat_reward_height_m=None):
        """Optional V7 cost; preserve the base payments and their lane datum.

        Only an exactly zero actor-visible yaw command selects this cost.
        Nonzero turns remain commanded even inside the base reward deadzone.
        The forward threshold and selected height gate remain unchanged. This
        uses current yaw rate without command rewrites, cumulative heading or
        PPO filtering. The coefficient requires a separately reviewed contract;
        there is no enabled default.
        """
        if not self.command_decoupling or not self.command_decoupling["enabled"]:
            return result
        pure_translation = ((frame.command_vx_m_s.abs() >= .1)
                            & (frame.command_yaw_rate_rad_s == 0.))
        if not self.is_rough:
            height_gate = ((.1 - (frame.root_height_m - frame.target_height_m).abs()) / .05).clamp(0., 1.)
        elif (self.flat_rehearsal_mask is not None
              and self.flat_rehearsal_reward_profile == "scut0719_flat_v1"):
            flat_gate = ((.1 - (flat_reward_height_m - frame.target_height_m).abs()) / .05).clamp(0., 1.)
            height_gate = torch.where(self.flat_rehearsal_mask, flat_gate, 1.)
        else:
            height_gate = torch.ones_like(frame.command_vx_m_s)
        raw = torch.where(pure_translation,
                          height_gate * frame.root_angular_velocity_body_rad_s[:, 2].square(), 0.)
        payment = torch.nan_to_num(
            -self.command_decoupling["coefficient_per_rad2_s2"] * raw * .02,
            nan=0., posinf=0., neginf=0.)
        payment = torch.where(frame.numerical_safety_reset, 0., payment)
        return ScutFlatRewardResult(
            raw_terms={**result.raw_terms, "pure_translation_yaw_square": raw},
            payments={**result.payments, "pure_translation_yaw_square": payment},
            total=result.total + payment)

    def summary(self):
        result = {"version": ROUGH_REWARD_VERSION if self.is_rough else REWARD_VERSION,
                "source_commit": SOURCE_COMMIT,
                "physical_input_adapter": "scut0730-rough-native-v1" if self.is_rough else "scut0719-flat-native-v1",
                "nonzero_weights": dict(ROUGH_WEIGHTS if self.is_rough else WEIGHTS),
                "policy_dt_s": .02,
                "payment_units": "already_dt_scaled_payment_per_policy_step",
                "root_velocity_reference": ROOT_VELOCITY_REFERENCE,
                "root_height_reference": self.height_reference,
                "wheel_position_reference": WHEEL_POSITION_REFERENCE,
                "undesired_link_names": [self.body_names[i] for i in self.undesired_ids],
                "terminal_link_names": [self.body_names[i] for i in self.terminal_ids],
                "contact_input": "native_net_contact_force_vectors",
                "contact_history_samples": self.contact_history.shape[1], "contact_sample_dt_s": self.physics_dt_s,
                "contact_payment_bins": "3x5ms; replay retains largest-norm actual vector per body per bin",
                "undesired_force_threshold_n": 3., "terminal_force_threshold_n": 1.,
                "attitude_terminal_angle_deg": 40., "confirmation_policy_ticks": self.confirmation_policy_ticks,
                "confirmation_seconds": self.confirmation_policy_ticks * .02,
                "concurrent_terminal_precedence": "termination",
                "local_adaptations": ["fixed_environment_origin_height_datum",
                    "explicit_chassis_contact_body_sets", "environment_owned_mechanical_guards",
                    "environment_owned_boundary_policy", "failure_over_timeout_bootstrap",
                    "passive_hold_intervals_excluded_from_ppo"],
                "acceleration_input": f"sdk_joint_acc_after_each_{self.physics_dt_s * 1000:g}ms_update",
                "applied_torque_input": f"native_applied_torque_after_each_{self.physics_dt_s * 1000:g}ms_update",
                "raw_action_history": "incoming_actor_before_control_mapping_clip_or_hold",
                "payment_dt_multiplications": 1, "legacy_payments": False,
                "failure_tail_settlement": False, "success_events": False,
                "source_command_sampler_verified": False, "native_qualified": False}
        if self.is_rough:
            result.update(velocity_height_gate_enabled=False,
                base_ground_scan="yaw_aligned_0.02m_square_0.01m_resolution_finite_mean_origin_z_fallback",
                critic_height_reference="root_link_world_z",
                local_adaptations=["explicit_chassis_contact_body_sets_no_gimbal",
                    "existing_mechanical_joint_guards", "whole_rough_map_boundary_timeout",
                    "failure_over_timeout_bootstrap", "passive_hold_intervals_excluded_from_ppo"])
        if self.flat_rehearsal_mask is not None:
            unified = self.flat_rehearsal_reward_profile == "scut0730_rough_v1"
            result["flat_rehearsal_reward"] = {
                "reward_profile": self.flat_rehearsal_reward_profile,
                "version": ROUGH_REWARD_VERSION if unified else REWARD_VERSION,
                "lanes": int(self.flat_rehearsal_mask.sum()),
                "selection": ("same_0730_payment_for_every_lane_mask_only_identifies_command_population"
                              if unified else "fixed_bool_mask_per_raw_term_and_already_dt_scaled_payment"),
                "root_height_reference": (self.height_reference if unified else
                                          "explicit_root_link_z_minus_rehearsal_plane_origin_z"),
                "velocity_height_gate_enabled": not unified,
                "nonzero_weights": dict(ROUGH_WEIGHTS if unified else WEIGHTS),
                "ordinary_rough_reward_profile": self.reward_profile,
                "shared_physical_action_contact_and_terminal_frame": True}
        if self.command_decoupling is not None:
            result["formal_command_decoupling"] = dict(self.command_decoupling)
            if self.command_decoupling["enabled"]:
                result["local_adaptations"].append("pure_translation_yaw_square_cost")
                result["additional_payment_weights"] = {
                    "pure_translation_yaw_square": -self.command_decoupling["coefficient_per_rad2_s2"]}
                result["command_decoupling_scope"] = {
                    "command_gate": "abs(vx)>=0.1 and actor-visible yaw==0; nonzero turns excluded",
                    "error_units": "body_yaw_rate_rad_s",
                    "height_gate": "same selected per-lane base reward height gate",
                    "base_payment_terms_unchanged": len(WEIGHTS),
                    "additional_payment_terms": 1,
                    "policy_dt_s": .02,
                    "deployment_actor_interface_changed": False}
        if self.tracking_reward_repair is not None:
            from .tracking_reward_repair import KEY, LINEAR_SQUARE_WEIGHT
            result[KEY] = self.tracking_reward_repair
            result["nonzero_weights"]["track_lin_vel_xy_square"] = LINEAR_SQUARE_WEIGHT
            result["tracking_commands"] = "actor_visible_targets_without_reward_deadzone"
            result["standing_command_gate"] = "vx_exactly_zero"
            result["local_adaptations"].append("exact_command_targets_and_linear_square_weight")
            if "flat_rehearsal_reward" in result:
                result["flat_rehearsal_reward"]["nonzero_weights"]["track_lin_vel_xy_square"] = LINEAR_SQUARE_WEIGHT
        return result
