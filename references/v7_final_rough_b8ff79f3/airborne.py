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

from __future__ import annotations

import torch
from isaaclab.utils.math import wrap_to_pi

from .base import WheelbipeStateMachineBase


class AirborneStateMachine(WheelbipeStateMachineBase):
    """Airborne-aware reward and target-height adjustments."""

    name = "airborne"

    def _cfg(self, env) -> dict:
        return getattr(env.cfg, "airborne_state_machine_cfg", {})

    def _height_reward_override_enabled(self, cfg: dict) -> bool:
        if not bool(cfg.get("enabled", False)):
            return False
        target_height_cfg = cfg.get("target_height", {})
        return bool(target_height_cfg.get("enabled", True))

    def _as_name_tuple(self, names) -> tuple[str, ...]:
        if names is None:
            return ()
        if isinstance(names, str):
            return (names,) if names else ()
        return tuple(name for name in names if name)

    def _get_allowed_terrain_mask(self, env, cfg: dict) -> torch.Tensor:
        allowed_names = self._as_name_tuple(cfg.get("allowed_terrain_names", ()))
        allowed_mask = env.get_terrain_name_mask(allowed_names)
        not_allowed_names = self._as_name_tuple(cfg.get("not_allowed_terrain_names", ()))
        if len(not_allowed_names) > 0:
            allowed_mask &= ~env.get_terrain_name_mask(not_allowed_names)
        return allowed_mask

    def _ensure_command_override_buffers(self, env) -> None:
        if hasattr(env, "airborne_command_override_active"):
            return
        env.airborne_command_override_active = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env.airborne_command_override_values = torch.zeros(
            env.num_envs, 3, dtype=torch.float, device=env.device
        )
        env.airborne_command_override_field_mask = torch.zeros(
            env.num_envs, 3, dtype=torch.bool, device=env.device
        )

    def _ensure_entry_command_buffers(self, env) -> None:
        if hasattr(env, "airborne_entry_command"):
            return
        env.airborne_entry_command = torch.zeros(
            env.num_envs, 3, dtype=torch.float, device=env.device
        )

    def _clear_entry_command(self, env, env_mask: torch.Tensor | None = None) -> None:
        if not hasattr(env, "airborne_entry_command"):
            return
        if env_mask is None:
            env.airborne_entry_command.zero_()
            return
        if torch.any(env_mask):
            env.airborne_entry_command[env_mask] = 0.0

    def _clear_command_override(self, env, env_mask: torch.Tensor | None = None) -> None:
        if not hasattr(env, "airborne_command_override_active"):
            return
        if env_mask is None:
            env.airborne_command_override_active.zero_()
            env.airborne_command_override_values.zero_()
            env.airborne_command_override_field_mask.zero_()
            return
        if torch.any(env_mask):
            env.airborne_command_override_active[env_mask] = False
            env.airborne_command_override_values[env_mask] = 0.0
            env.airborne_command_override_field_mask[env_mask] = False

    def _ensure_landing_trajectory_buffers(self, env) -> None:
        if hasattr(env, "airborne_landing_traj_active"):
            return
        env.airborne_landing_traj_active = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env.airborne_landing_traj_time = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )
        env.airborne_landing_traj_h0 = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )
        env.airborne_landing_traj_v0 = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )
        env.airborne_landing_traj_acc = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )
        env.airborne_landing_traj_quad = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )
        env.airborne_landing_traj_target_height = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )
        env.airborne_landing_traj_duration = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )

    def _clear_landing_trajectory(self, env, env_mask: torch.Tensor | None = None) -> None:
        if not hasattr(env, "airborne_landing_traj_active"):
            return
        if env_mask is None:
            env.airborne_landing_traj_active.zero_()
            env.airborne_landing_traj_time.zero_()
            env.airborne_landing_traj_h0.zero_()
            env.airborne_landing_traj_v0.zero_()
            env.airborne_landing_traj_acc.zero_()
            env.airborne_landing_traj_quad.zero_()
            env.airborne_landing_traj_target_height.zero_()
            env.airborne_landing_traj_duration.zero_()
            return
        if torch.any(env_mask):
            env.airborne_landing_traj_active[env_mask] = False
            env.airborne_landing_traj_time[env_mask] = 0.0
            env.airborne_landing_traj_h0[env_mask] = 0.0
            env.airborne_landing_traj_v0[env_mask] = 0.0
            env.airborne_landing_traj_acc[env_mask] = 0.0
            env.airborne_landing_traj_quad[env_mask] = 0.0
            env.airborne_landing_traj_target_height[env_mask] = 0.0
            env.airborne_landing_traj_duration[env_mask] = 0.0

    def _landing_trajectory_cfg(self, cfg: dict) -> dict:
        landing_cfg = cfg.get("landing_trajectory", {})
        return landing_cfg if isinstance(landing_cfg, dict) else {}

    def _landing_trajectory_enabled(self, cfg: dict) -> bool:
        return bool(cfg.get("enabled", False)) and bool(
            self._landing_trajectory_cfg(cfg).get("enabled", False)
        )

    def _get_body_relative_ground_height(self, env) -> torch.Tensor:
        return env.robot.data.root_pos_w[:, 2] - env.ground_z_est

    def _get_landing_trajectory_reference(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_landing_trajectory_buffers(env)
        duration = torch.clamp(env.airborne_landing_traj_duration, min=env.step_dt)
        t = torch.minimum(torch.clamp(env.airborne_landing_traj_time, min=0.0), duration)
        h_ref = (
            env.airborne_landing_traj_h0
            + env.airborne_landing_traj_v0 * t
            + 0.5 * env.airborne_landing_traj_acc * torch.square(t)
        )
        v_ref = env.airborne_landing_traj_v0 + env.airborne_landing_traj_acc * t
        return h_ref, v_ref

    def _update_landing_trajectory(self, env, cfg: dict, update_mask: torch.Tensor) -> None:
        landing_cfg = self._landing_trajectory_cfg(cfg)
        self._ensure_landing_trajectory_buffers(env)
        if not bool(landing_cfg.get("enabled", False)):
            self._clear_landing_trajectory(env)
            return

        active = env.airborne_landing_traj_active
        active_update = update_mask & active & env.height_reward_airborne_state
        env.airborne_landing_traj_time.copy_(
            torch.where(
                active_update,
                env.airborne_landing_traj_time + env.step_dt,
                env.airborne_landing_traj_time,
            )
        )
        duration_done = active & (
            env.airborne_landing_traj_time >= env.airborne_landing_traj_duration
        )
        self._clear_landing_trajectory(env, update_mask & (~env.height_reward_airborne_state))
        self._clear_landing_trajectory(env, update_mask & duration_done)

        wheel_exit_time = getattr(env, "airborne_wheel_contact_exit_time", None)
        if wheel_exit_time is None:
            return
        start_contact_duration_s = max(
            float(landing_cfg.get("start_wheel_contact_duration_s", 0.02)),
            0.0,
        )
        contact_started = torch.any(wheel_exit_time >= start_contact_duration_s, dim=1)
        min_down_vel = max(float(landing_cfg.get("min_down_vel", 0.05)), 0.0)
        lin_vel_z = env.robot.data.root_lin_vel_w[:, 2]
        start_height = self._get_body_relative_ground_height(env)
        target_height = float(landing_cfg.get("target_height", landing_cfg.get("final_height", 0.24)))
        end_vel_z = min(float(landing_cfg.get("end_vel_z", 0.0)), 0.0)
        min_height_margin = max(float(landing_cfg.get("min_height_margin", 0.02)), 0.0)
        duration_s = max(float(landing_cfg.get("duration_s", 0.25)), env.step_dt)
        displacement = target_height - start_height
        ref_v0 = 2.0 * displacement / duration_s - end_vel_z
        acceleration = (end_vel_z - ref_v0) / duration_s
        max_abs_acc = landing_cfg.get("max_abs_acc", None)
        if max_abs_acc is not None:
            max_abs_acc = max(float(max_abs_acc), 0.0)
            acc_ok = (
                torch.abs(acceleration) <= max_abs_acc
                if max_abs_acc > 0.0
                else torch.ones_like(update_mask, dtype=torch.bool)
            )
        else:
            acc_ok = torch.ones_like(update_mask, dtype=torch.bool)
        can_start = (
            update_mask
            & env.height_reward_airborne_state
            & (~env.airborne_landing_traj_active)
            & (~duration_done)
            & contact_started
            & (lin_vel_z < -min_down_vel)
            & (start_height > target_height + min_height_margin)
            & acc_ok
        )
        if not torch.any(can_start):
            return

        env.airborne_landing_traj_active[can_start] = True
        env.airborne_landing_traj_time[can_start] = 0.0
        env.airborne_landing_traj_h0[can_start] = start_height[can_start]
        env.airborne_landing_traj_v0[can_start] = ref_v0[can_start]
        env.airborne_landing_traj_acc[can_start] = acceleration[can_start]
        env.airborne_landing_traj_quad[can_start] = 0.0
        env.airborne_landing_traj_target_height[can_start] = target_height
        env.airborne_landing_traj_duration[can_start] = duration_s

    def _sample_range_spec(self, env, value, count: int) -> torch.Tensor:
        sampler = getattr(env, "_sample_permission_range_spec", None)
        if callable(sampler):
            return sampler(value, count)
        if count <= 0:
            return torch.zeros(0, dtype=torch.float, device=env.device)
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(
            isinstance(v, (float, int)) for v in value
        ):
            low, high = float(value[0]), float(value[1])
            if low == high:
                return torch.full((count,), low, dtype=torch.float, device=env.device)
            return torch.empty(count, dtype=torch.float, device=env.device).uniform_(
                min(low, high), max(low, high)
            )

        ranges = []
        for item in value:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(f"Invalid airborne command range entry: {item!r}")
            ranges.append((float(item[0]), float(item[1])))
        segment_ids = torch.randint(len(ranges), (count,), device=env.device)
        samples = torch.empty(count, dtype=torch.float, device=env.device)
        for segment_idx, (low, high) in enumerate(ranges):
            mask = segment_ids == segment_idx
            if not torch.any(mask):
                continue
            if low == high:
                samples[mask] = low
            else:
                samples[mask] = torch.empty(
                    int(mask.sum().item()), dtype=torch.float, device=env.device
                ).uniform_(min(low, high), max(low, high))
        return samples

    def _filter_range_spec_by_sign(self, value, sign: float):
        if sign > 0.0:
            low_limit, high_limit = 0.0, float("inf")
        elif sign < 0.0:
            low_limit, high_limit = -float("inf"), 0.0
        else:
            low_limit, high_limit = 0.0, 0.0

        if isinstance(value, (tuple, list)) and len(value) == 2 and all(
            isinstance(v, (float, int)) for v in value
        ):
            ranges = [(float(value[0]), float(value[1]))]
            return_single = True
        else:
            ranges = [(float(item[0]), float(item[1])) for item in value]
            return_single = False

        filtered = []
        for low, high in ranges:
            low, high = min(low, high), max(low, high)
            clipped_low = max(low, low_limit)
            clipped_high = min(high, high_limit)
            if clipped_low <= clipped_high:
                filtered.append((clipped_low, clipped_high))
        if len(filtered) == 0:
            filtered = [(0.0, 0.0)]
        if return_single and len(filtered) == 1:
            return filtered[0]
        return filtered

    def _sample_lin_vel_x_with_optional_sign_constraint(
        self,
        env,
        range_spec,
        env_ids: torch.Tensor,
        *,
        constrain_sign: bool,
    ) -> torch.Tensor:
        if not constrain_sign:
            return self._sample_range_spec(env, range_spec, int(env_ids.numel()))

        samples = torch.zeros(env_ids.numel(), dtype=torch.float, device=env.device)
        current_lin_vel_x = env.command[env_ids, 0]
        for sign in (-1.0, 0.0, 1.0):
            if sign < 0.0:
                sign_mask = current_lin_vel_x < 0.0
            elif sign > 0.0:
                sign_mask = current_lin_vel_x > 0.0
            else:
                sign_mask = current_lin_vel_x == 0.0
            if not torch.any(sign_mask):
                continue
            filtered_spec = self._filter_range_spec_by_sign(range_spec, sign)
            samples[sign_mask] = self._sample_range_spec(
                env,
                filtered_spec,
                int(sign_mask.sum().item()),
            )
        return samples

    def _sample_command_override_on_enter(
        self,
        env,
        cfg: dict,
        enter_event: torch.Tensor,
    ) -> None:
        command_cfg = cfg.get("terrain_command_resample", {})
        if not bool(command_cfg.get("enabled", False)):
            return
        profiles = command_cfg.get("profiles", {})
        if not isinstance(profiles, dict) or len(profiles) == 0:
            return
        if not torch.any(enter_event):
            return

        self._ensure_command_override_buffers(env)
        self._clear_command_override(env, enter_event)

        global_prob = min(max(float(command_cfg.get("prob", 1.0)), 0.0), 1.0)
        global_constrain_lin_vel_x_sign = bool(
            command_cfg.get("lin_vel_x_sign_from_current", False)
        )
        command_fields = (
            ("lin_vel_x", 0),
            ("lin_vel_y", 1),
            ("ang_vel_z", 2),
        )
        remaining_enter = enter_event.clone()
        for profile_name, profile_cfg in profiles.items():
            if not isinstance(profile_cfg, dict):
                continue
            terrain_names = self._as_name_tuple(
                profile_cfg.get("terrain_names", profile_name)
            )
            if len(terrain_names) == 0:
                continue
            profile_mask = remaining_enter & env.get_terrain_name_mask(terrain_names)
            if not torch.any(profile_mask):
                continue

            prob = min(max(float(profile_cfg.get("prob", global_prob)), 0.0), 1.0)
            if prob <= 0.0:
                remaining_enter &= ~profile_mask
                continue
            if prob < 1.0:
                profile_mask = profile_mask & (
                    torch.rand(env.num_envs, dtype=torch.float, device=env.device) < prob
                )
            if torch.any(profile_mask):
                count = int(profile_mask.sum().item())
                for field_name, command_idx in command_fields:
                    range_spec = profile_cfg.get(field_name, None)
                    if range_spec is None:
                        continue
                    profile_env_ids = profile_mask.nonzero(as_tuple=False).squeeze(-1)
                    if field_name == "lin_vel_x":
                        constrain_sign = bool(
                            profile_cfg.get(
                                "lin_vel_x_sign_from_current",
                                global_constrain_lin_vel_x_sign,
                            )
                        )
                        sampled_values = self._sample_lin_vel_x_with_optional_sign_constraint(
                            env,
                            range_spec,
                            profile_env_ids,
                            constrain_sign=constrain_sign,
                        )
                    else:
                        sampled_values = self._sample_range_spec(env, range_spec, count)
                    env.airborne_command_override_values[profile_mask, command_idx] = sampled_values
                    env.airborne_command_override_field_mask[profile_mask, command_idx] = True
                active = torch.any(env.airborne_command_override_field_mask, dim=1)
                env.airborne_command_override_active.copy_(active)

            # 一个 env 只匹配第一个 profile，避免多个地形名配置叠加。
            remaining_enter &= ~(
                enter_event & env.get_terrain_name_mask(terrain_names)
            )

    def on_command_updated(self, env) -> None:
        self.apply_command_overrides(env)

    def apply_command_overrides(self, env) -> None:
        cfg = self._cfg(env)
        command_cfg = cfg.get("terrain_command_resample", {})
        if not bool(cfg.get("enabled", False)) or not bool(command_cfg.get("enabled", False)):
            self._clear_command_override(env)
            return
        if not hasattr(env, "airborne_command_override_active"):
            return

        active_mask = env.airborne_command_override_active & env.height_reward_airborne_state
        if not torch.any(active_mask):
            return

        for command_idx in range(3):
            field_mask = active_mask & env.airborne_command_override_field_mask[:, command_idx]
            if torch.any(field_mask):
                env.command[field_mask, command_idx] = env.airborne_command_override_values[
                    field_mask, command_idx
                ]

    def _zero_exit_timers(self, env) -> None:
        enter_timer = getattr(env, "airborne_enter_time", None)
        if enter_timer is not None:
            enter_timer.zero_()
        wheel_timer = getattr(env, "airborne_wheel_contact_exit_time", None)
        if wheel_timer is not None:
            wheel_timer.zero_()
        base_timer = getattr(env, "airborne_base_contact_exit_time", None)
        if base_timer is not None:
            base_timer.zero_()
        current_duration = getattr(env, "airborne_current_duration", None)
        if current_duration is not None:
            current_duration.zero_()
        max_duration = getattr(env, "airborne_max_duration", None)
        if max_duration is not None:
            max_duration.zero_()

    def _get_base_contact_indices(self, env) -> list[int]:
        indices = getattr(env, "_airborne_base_contact_link_idx", None)
        if indices is None:
            indices = list(getattr(env, "_undesired_contact_link_idx", []))
            env._airborne_base_contact_link_idx = indices
        return indices

    def _get_current_contact_forces(self, env) -> torch.Tensor | None:
        contact_data = getattr(getattr(env, "contact_sensor", None), "data", None)
        if contact_data is None:
            return None
        return getattr(contact_data, "net_forces_w", None)

    def _shape_penalty(
        self,
        value: torch.Tensor,
        mode: str,
        *,
        square_sigma: float = 1.0,
    ) -> torch.Tensor:
        mode = mode.lower()
        if mode == "binary":
            return (value > 0.0).float()
        if mode == "l1":
            return value
        if mode == "l2":
            return square_sigma * torch.square(value)
        raise ValueError(f"Unsupported airborne reward penalty mode: {mode}")

    def _get_wheel_radius(self, env) -> float:
        return float(
            self._cfg(env).get("enter", {}).get(
                "wheel_radius",
                env._get_height_measure_wheel_radius()
                if getattr(env, "_get_height_measure_wheel_radius", None) is not None
                else 0.06,
            )
        )

    def _get_reward_window_mask(self, env, addition_cfg: dict) -> torch.Tensor:
        active_mask = env.height_reward_airborne_state
        max_contact_duration_s = addition_cfg.get("before_wheel_contact_duration_s", None)
        min_contact_duration_s = addition_cfg.get("after_wheel_contact_duration_s", None)
        require_contact_started = bool(addition_cfg.get("require_wheel_contact_timer_started", False))
        if max_contact_duration_s is None and min_contact_duration_s is None and not require_contact_started:
            return active_mask

        wheel_exit_time = getattr(env, "airborne_wheel_contact_exit_time", None)
        if wheel_exit_time is None:
            if require_contact_started or min_contact_duration_s is not None:
                return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            return active_mask

        contact_mode = addition_cfg.get("contact_mode", "any_wheel")
        if contact_mode not in ("any_wheel", "both_wheels"):
            raise ValueError(f"Unsupported airborne reward contact_mode: {contact_mode}")

        def _reduce_contact_mask(contact_mask: torch.Tensor) -> torch.Tensor:
            if contact_mode == "any_wheel":
                return torch.any(contact_mask, dim=1)
            return torch.all(contact_mask, dim=1)

        if require_contact_started:
            active_mask = active_mask & _reduce_contact_mask(wheel_exit_time > 0.0)

        if min_contact_duration_s is not None:
            min_contact_duration_s = max(float(min_contact_duration_s), 0.0)
            active_mask = active_mask & _reduce_contact_mask(wheel_exit_time >= min_contact_duration_s)

        if max_contact_duration_s is not None:
            max_contact_duration_s = max(float(max_contact_duration_s), 0.0)
            stop_mask = _reduce_contact_mask(wheel_exit_time >= max_contact_duration_s)
            active_mask = active_mask & (~stop_mask)

        return active_mask

    def _get_wheel_contact_started_mask(
        self,
        env,
        addition_cfg: dict,
    ) -> torch.Tensor:
        active_mask = env.height_reward_airborne_state
        wheel_exit_time = getattr(env, "airborne_wheel_contact_exit_time", None)
        if wheel_exit_time is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        start_duration_s = max(
            float(
                addition_cfg.get(
                    "start_wheel_contact_duration_s",
                    addition_cfg.get("wheel_contact_duration_s", 0.02),
                )
            ),
            0.0,
        )
        contact_reached = wheel_exit_time >= start_duration_s
        contact_mode = addition_cfg.get("contact_mode", "any_wheel")
        if contact_mode == "any_wheel":
            contact_started = torch.any(contact_reached, dim=1)
        elif contact_mode == "both_wheels":
            contact_started = torch.all(contact_reached, dim=1)
        else:
            raise ValueError(f"Unsupported airborne reward contact_mode: {contact_mode}")
        return active_mask & contact_started

    def _get_wheel_contact_force_mask(
        self,
        env,
        addition_cfg: dict,
    ) -> torch.Tensor:
        contact_force_threshold = addition_cfg.get("contact_force_threshold", None)
        if contact_force_threshold is None:
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        contact_forces_w = self._get_current_contact_forces(env)
        if contact_forces_w is None or len(getattr(env, "_desired_contact_link_idx", [])) == 0:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        wheel_forces = torch.norm(contact_forces_w[:, env._desired_contact_link_idx], dim=-1)
        contact_reached = wheel_forces > float(contact_force_threshold)
        contact_mode = addition_cfg.get("contact_mode", "any_wheel")
        if contact_mode == "any_wheel":
            return torch.any(contact_reached, dim=1)
        if contact_mode == "both_wheels":
            return torch.all(contact_reached, dim=1)
        raise ValueError(f"Unsupported airborne reward contact_mode: {contact_mode}")

    def _get_reward_command_lin_vel_x(self, env, addition_cfg: dict) -> torch.Tensor:
        if bool(addition_cfg.get("use_entry_command", False)):
            entry_command = getattr(env, "airborne_entry_command", None)
            if entry_command is not None:
                return entry_command[:, 0]
        return env.command[:, 0]

    def _get_directional_wheel_speed_score(
        self,
        env,
        addition_cfg: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        wheel_idx = getattr(env, "_wheel_idx", [])
        if len(wheel_idx) == 0:
            return zero, zero

        reward_mask = self._get_reward_window_mask(env, addition_cfg)
        command_x = self._get_reward_command_lin_vel_x(env, addition_cfg)
        root_vel_x = env.robot.data.root_lin_vel_b[:, 0]
        command_threshold = max(float(addition_cfg.get("command_x_threshold", 1.0)), 0.0)
        root_vel_threshold = max(float(addition_cfg.get("root_x_threshold", 1.0)), 0.0)

        positive_mask = (command_x > command_threshold) & (root_vel_x > root_vel_threshold)
        negative_mask = (command_x < -command_threshold) & (root_vel_x < -root_vel_threshold)
        direction_gate = positive_mask | negative_mask
        direction = torch.zeros_like(command_x)
        direction[positive_mask] = 1.0
        direction[negative_mask] = -1.0

        speed_start = float(addition_cfg.get("start", 0.0))
        speed_full = float(addition_cfg.get("full", 10.0))
        denom = max(speed_full - speed_start, 1.0e-6)
        wheel_vel = env.joint_vel[:, wheel_idx]
        directional_speed = wheel_vel * direction.unsqueeze(-1)
        wheel_scores = torch.clamp((directional_speed - speed_start) / denom, 0.0, 1.0)

        reduce = addition_cfg.get("reduce", "min")
        if reduce == "min":
            score = wheel_scores.min(dim=1).values
        elif reduce == "mean":
            score = wheel_scores.mean(dim=1)
        elif reduce == "max":
            score = wheel_scores.max(dim=1).values
        else:
            raise ValueError(f"Unsupported directional wheel speed reduce mode: {reduce}")

        gate_f = (reward_mask & direction_gate).float()
        return gate_f * score, gate_f

    def _get_leg_lengths(self, env) -> torch.Tensor:
        _, _, wheel_pos_heading_b, _, _ = env._get_root_quat_inv_and_wheel_pos_b()
        return torch.norm(wheel_pos_heading_b, dim=-1)

    def _get_wheel_bottom_relative_to_base_z(self, env) -> torch.Tensor:
        wheel_bottom_z = env.robot.data.body_pos_w[:, env._wheel_link_idx, 2] - self._get_wheel_radius(env)
        return wheel_bottom_z - env.robot.data.root_pos_w[:, 2].unsqueeze(-1)

    def _get_contact_timer_below_mask(
        self,
        env,
        addition_cfg: dict,
    ) -> torch.Tensor:
        active_mask = env.height_reward_airborne_state.clone()
        wheel_exit_time = getattr(env, "airborne_wheel_contact_exit_time", None)
        if wheel_exit_time is not None:
            wheel_duration_s = max(float(addition_cfg.get("wheel_contact_duration_s", 0.02)), 0.0)
            active_mask = active_mask & torch.all(wheel_exit_time < wheel_duration_s, dim=1)

        base_exit_time = getattr(env, "airborne_base_contact_exit_time", None)
        if base_exit_time is not None:
            base_duration_s = max(float(addition_cfg.get("base_contact_duration_s", 0.02)), 0.0)
            active_mask = active_mask & (base_exit_time < base_duration_s)
        return active_mask

    def _get_rear2_rear1_joint_limit_terms(
        self, env
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        rear2_joint_idx = getattr(env, "_rear2_joint_idx", [])
        if len(rear2_joint_idx) == 0:
            return zeros, zeros

        lower = float(getattr(env.cfg, "rear2_rear1_joint_limit_lower", -3.0 / 180.0 * torch.pi))
        upper = float(getattr(env.cfg, "rear2_rear1_joint_limit_upper", 60.0 / 180.0 * torch.pi))
        lower, upper = min(lower, upper), max(lower, upper)
        if upper <= lower:
            return zeros, zeros

        default_boundary_ratio = float(getattr(env.cfg, "rear2_rear1_joint_limit_boundary_ratio", 0.03))
        lower_boundary_ratio = float(
            getattr(
                env.cfg,
                "rear2_rear1_joint_limit_lower_boundary_ratio",
                default_boundary_ratio,
            )
        )
        upper_boundary_ratio = float(
            getattr(
                env.cfg,
                "rear2_rear1_joint_limit_upper_boundary_ratio",
                default_boundary_ratio,
            )
        )
        lower_boundary_ratio = min(max(lower_boundary_ratio, 0.0), 0.49)
        upper_boundary_ratio = min(max(upper_boundary_ratio, 0.0), 0.49)
        limit_span = upper - lower
        lower_margin = limit_span * lower_boundary_ratio
        upper_margin = limit_span * upper_boundary_ratio
        soft_lower = lower + lower_margin
        soft_upper = upper - upper_margin

        rear2_pos = wrap_to_pi(env.joint_pos[:, rear2_joint_idx])
        lower_over = torch.clamp(soft_lower - rear2_pos, min=0.0)
        upper_over = torch.clamp(rear2_pos - soft_upper, min=0.0)
        pos_penalty = torch.sum(lower_over + upper_over, dim=-1)

        vel_threshold = max(
            float(getattr(env.cfg, "rear2_rear1_joint_limit_vel_threshold", 10.0)),
            0.0,
        )
        rear2_vel = env.joint_vel[:, rear2_joint_idx]
        lower_vel_violation = (rear2_pos < soft_lower) & (rear2_vel < -vel_threshold)
        upper_vel_violation = (rear2_pos > soft_upper) & (rear2_vel > vel_threshold)
        vel_penalty = torch.sum(
            (lower_vel_violation | upper_vel_violation).float(),
            dim=-1,
        )

        return pos_penalty, vel_penalty

    def on_observation_step(self, env) -> None:
        # 开关关闭时，显式清空所有 airborne 运行时状态，保证后续 reward hook 退化为普通高度奖励。
        cfg = self._cfg(env)
        if not bool(cfg.get("enabled", False)):
            env.height_reward_airborne_state.zero_()
            force_enter_request = getattr(env, "airborne_force_enter_request", None)
            if force_enter_request is not None:
                force_enter_request.zero_()
            last_update_step = getattr(env, "airborne_state_last_update_step", None)
            if last_update_step is not None:
                last_update_step.fill_(-1)
            self._zero_exit_timers(env)
            self._clear_command_override(env)
            self._clear_landing_trajectory(env)
            self._clear_entry_command(env)
            return

        # 新的进入事件会进入 airborne。退出由轮子/base_link 接触力连续满足阈值来决定。
        current_step = int(getattr(env, "common_step_counter", 0))
        last_update_step = getattr(env, "airborne_state_last_update_step", None)
        if last_update_step is not None:
            update_mask = last_update_step != current_step
            if not torch.any(update_mask):
                return
        else:
            update_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        prev_airborne_state = env.height_reward_airborne_state.clone()

        # 进入 airborne 的判据：
        # 1. 机身相对地面足够高；
        # 2. 两个轮心都高于“轮半径 + 离地阈值”，避免只因地面估计抖动误判。
        enter_cfg = cfg.get("enter", {})
        body_relative_ground_height = self._get_body_relative_ground_height(env)
        wheel_relative_ground_heights = env._get_wheel_relative_ground_heights_raw()
        wheel_radius = float(enter_cfg["wheel_radius"])
        enter_body_threshold = float(enter_cfg["body_height_threshold"])
        enter_wheel_threshold = float(enter_cfg["wheel_clearance_threshold"])
        enter_wheel_center_threshold = wheel_radius + enter_wheel_threshold

        terrain_allowed_mask = self._get_allowed_terrain_mask(env, cfg)
        stair_active_mask = getattr(
            env,
            "wheel_forward_stair_state",
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
        natural_enter_mask = (
            terrain_allowed_mask
            & (~stair_active_mask)
            & (body_relative_ground_height > enter_body_threshold)
            & torch.all(
                wheel_relative_ground_heights > enter_wheel_center_threshold, dim=1
            )
        )
        enter_duration_s = max(float(enter_cfg.get("duration_s", 0.0)), 0.0)
        enter_time = getattr(env, "airborne_enter_time", None)
        if enter_time is None:
            enter_time = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            env.airborne_enter_time = enter_time
        enter_time.copy_(
            torch.where(
                update_mask,
                torch.where(
                    natural_enter_mask & (~prev_airborne_state),
                    enter_time + env.step_dt,
                    torch.zeros_like(enter_time),
                ),
                enter_time,
            )
        )
        natural_enter_ready = natural_enter_mask & (enter_time >= enter_duration_s)
        force_enter_request = getattr(env, "airborne_force_enter_request", None)
        if force_enter_request is None:
            force_enter_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        else:
            force_enter_mask = terrain_allowed_mask & (~stair_active_mask) & force_enter_request
        enter_mask = natural_enter_ready | force_enter_mask

        airborne_enter_event = update_mask & enter_mask & (~prev_airborne_state)
        if torch.any(airborne_enter_event):
            enter_time[airborne_enter_event] = 0.0
            self._ensure_entry_command_buffers(env)
            env.airborne_entry_command[airborne_enter_event] = env.command[
                airborne_enter_event, :3
            ]
        if force_enter_request is not None:
            force_enter_request[update_mask] = False
        contact_forces_w = self._get_current_contact_forces(env)

        wheel_contact_stable = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        wheel_height_low = torch.zeros_like(wheel_contact_stable)
        wheel_contact_countable = wheel_contact_stable
        if contact_forces_w is not None and len(getattr(env, "_desired_contact_link_idx", [])) >= 2:
            wheel_force = torch.norm(contact_forces_w[:, env._desired_contact_link_idx], dim=-1)
            exit_cfg = cfg.get("exit", {})
            wheel_force_threshold = float(exit_cfg["wheel_contact_force_threshold"])
            wheel_contact_stable = wheel_force[:, :2] > wheel_force_threshold
            enter_cfg = cfg.get("enter", {})
            default_height_threshold = float(enter_cfg["wheel_radius"]) + float(
                enter_cfg["wheel_clearance_threshold"]
            )
            wheel_contact_height_threshold = float(
                exit_cfg.get("wheel_contact_height_threshold", default_height_threshold)
            )
            wheel_height_low = env._get_wheel_relative_ground_heights_raw() < wheel_contact_height_threshold
            wheel_contact_countable = wheel_contact_stable & wheel_height_low

        base_contact_stable = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        base_contact_indices = self._get_base_contact_indices(env)
        if contact_forces_w is not None and len(base_contact_indices) > 0:
            base_force = torch.norm(contact_forces_w[:, base_contact_indices], dim=-1)
            exit_cfg = cfg.get("exit", {})
            base_force_threshold = float(exit_cfg["base_contact_force_threshold"])
            base_contact_stable = torch.any(base_force > base_force_threshold, dim=1)

        wheel_exit_time = getattr(env, "airborne_wheel_contact_exit_time", None)
        base_exit_time = getattr(env, "airborne_base_contact_exit_time", None)
        if wheel_exit_time is None or base_exit_time is None:
            return
        if torch.any(airborne_enter_event):
            wheel_exit_time[airborne_enter_event] = 0.0
            base_exit_time[airborne_enter_event] = 0.0

        active_for_exit = update_mask & prev_airborne_state & (~airborne_enter_event)
        wheel_exit_time.copy_(
            torch.where(
                update_mask.unsqueeze(-1),
                torch.where(
                    active_for_exit.unsqueeze(-1),
                    torch.where(
                        wheel_contact_countable,
                        wheel_exit_time + env.step_dt,
                        torch.where(
                            wheel_height_low,
                            wheel_exit_time,
                            torch.zeros_like(wheel_exit_time),
                        ),
                    ),
                    wheel_exit_time,
                ),
                wheel_exit_time,
            )
        )
        base_exit_time.copy_(
            torch.where(
                update_mask,
                torch.where(
                    active_for_exit & base_contact_stable,
                    base_exit_time + env.step_dt,
                    base_exit_time,
                ),
                base_exit_time,
            )
        )
        exit_cfg = cfg.get("exit", {})
        wheel_duration = max(float(exit_cfg["wheel_contact_duration_s"]), 0.0)
        base_duration = max(float(exit_cfg["base_contact_duration_s"]), 0.0)
        current_duration = getattr(env, "airborne_current_duration", None)
        max_airborne_duration_s = exit_cfg.get("max_duration_s", None)
        if current_duration is not None and max_airborne_duration_s is not None:
            max_duration_exit = active_for_exit & (
                current_duration + env.step_dt >= max(float(max_airborne_duration_s), 0.0)
            )
        else:
            max_duration_exit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        exit_mask = prev_airborne_state & (
            torch.any(wheel_exit_time >= wheel_duration, dim=1)
            | (base_exit_time >= base_duration)
            | max_duration_exit
            | stair_active_mask
        )

        next_airborne_state = torch.where(
            update_mask,
            (prev_airborne_state | airborne_enter_event) & (~exit_mask),
            prev_airborne_state,
        )
        env.height_reward_airborne_state.copy_(next_airborne_state)
        self._sample_command_override_on_enter(env, cfg, airborne_enter_event)
        clear_override_mask = update_mask & (~next_airborne_state)
        self._clear_command_override(env, clear_override_mask)
        self._clear_entry_command(env, clear_override_mask)
        self._update_landing_trajectory(env, cfg, update_mask)
        max_duration = getattr(env, "airborne_max_duration", None)
        if current_duration is not None and max_duration is not None:
            current_duration.copy_(
                torch.where(
                    update_mask,
                    torch.where(
                        next_airborne_state,
                        current_duration + env.step_dt,
                        torch.zeros_like(current_duration),
                    ),
                    current_duration,
                )
            )
            max_duration.copy_(torch.maximum(max_duration, current_duration))
        if last_update_step is not None:
            last_update_step[update_mask] = current_step

    def get_height_reward_reference_height(
        self,
        env,
        relative_obs_height: torch.Tensor,
        wheel_height_w: torch.Tensor,
    ) -> torch.Tensor:
        self.on_observation_step(env)
        cfg = self._cfg(env)
        if not self._height_reward_override_enabled(cfg):
            return relative_obs_height
        landing_active = getattr(env, "airborne_landing_traj_active", None)
        if landing_active is None:
            landing_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # 普通状态下使用“机身相对地面高度”；airborne 状态下改用轮位置构造的高度参考。
        # 腿长模式本身就是轮心口径；非腿长模式要加回轮半径，保持 root-to-ground 高度口径。
        body_minus_highest_wheel_z = env.robot.data.root_pos_w[:, 2] - torch.amax(
            wheel_height_w, dim=1
        )
        if getattr(env, "_use_leg_length_height", None) is not None and env._use_leg_length_height():
            airborne_reference_height = body_minus_highest_wheel_z
        else:
            wheel_radius = float(self._cfg(env).get("enter", {})["wheel_radius"])
            airborne_reference_height = body_minus_highest_wheel_z + wheel_radius
        return torch.where(
            env.height_reward_airborne_state & (~landing_active),
            airborne_reference_height,
            relative_obs_height,
        )

    def get_height_reward_target_height(
        self, env, target_height: torch.Tensor
    ) -> torch.Tensor:
        cfg = self._cfg(env)
        if not bool(cfg.get("enabled", False)):
            return target_height
        landing_active = getattr(env, "airborne_landing_traj_active", None)
        if landing_active is None:
            landing_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if torch.any(landing_active):
            landing_h_ref, _ = self._get_landing_trajectory_reference(env)
            target_height = torch.where(landing_active, landing_h_ref, target_height)
        if not self._height_reward_override_enabled(cfg):
            return target_height

        # airborne 时可临时抬高高度目标，引导机器人保持更合适的腾空姿态。
        target_height_cfg = cfg.get("target_height", {})
        airborne_bias = float(target_height_cfg["bias"])
        airborne_cmd_max = target_height_cfg["max"]
        if airborne_cmd_max is not None:
            max_height_target = float(airborne_cmd_max)
        else:
            height_range = getattr(env.cfg, "height_range", None)
            if height_range is not None and len(height_range) > 0:
                max_height_target = float(height_range[-1])
            else:
                max_height_target = float("inf")
        if getattr(env, "_use_leg_length_height", None) is not None and env._use_leg_length_height():
            wheel_radius = float(cfg.get("enter", {})["wheel_radius"])
            if max_height_target != float("inf"):
                # 腿长高度模式的 target 不含轮半径，因此上限也要对应减去轮半径。
                max_height_target = max(max_height_target - wheel_radius, 0.0)

        airborne_target_height = torch.clamp(
            target_height + airborne_bias,
            max=max_height_target,
        )
        return torch.where(
            env.height_reward_airborne_state & (~landing_active),
            airborne_target_height,
            target_height,
        )

    def apply_reward_term_scales(
        self, env, reward_terms: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        airborne_mask_f = env.height_reward_airborne_state.float()
        reward_full = self._cfg(env).get("reward_full", {})
        for term_name, full_value in reward_full.items():
            if term_name not in reward_terms:
                continue
            reward_terms[term_name] = torch.where(
                env.height_reward_airborne_state,
                torch.full_like(reward_terms[term_name], float(full_value)),
                reward_terms[term_name],
            )

        reward_scales = self._cfg(env).get("reward_scales", {})
        for term_name, boost_scale in reward_scales.items():
            if term_name not in reward_terms:
                continue
            if term_name in reward_full:
                continue
            scale = 1.0 + airborne_mask_f * (float(boost_scale) - 1.0)
            reward_terms[term_name] = reward_terms[term_name] * scale

        reward_additions = self._cfg(env).get("reward_additions", {})
        reward_weights = getattr(env.cfg, "rewards", {})
        for term_name, addition_cfg in reward_additions.items():
            # reward_additions are materialized before cfg.rewards filters terms.
            # Skip disabled additions here so expensive contact/kinematics terms do
            # not run every step when their final reward weight is zero or absent.
            reward_weight = reward_weights.get(term_name, None)
            if reward_weight is None or float(reward_weight) == 0.0:
                continue
            if isinstance(addition_cfg, dict):
                addition_type = addition_cfg.get("type", "constant")
                if addition_type == "constant":
                    added_reward = airborne_mask_f * float(addition_cfg["value"])
                elif addition_type == "body_height_below":
                    threshold = float(addition_cfg["threshold"])
                    mode = addition_cfg.get("mode", "l1")
                    square_sigma = float(addition_cfg.get("square_sigma", 1.0))
                    body_relative_ground_height = env.robot.data.root_pos_w[:, 2] - env.ground_z_est
                    height_deficit = torch.clamp(
                        threshold - body_relative_ground_height,
                        min=0.0,
                    )
                    if mode == "l1":
                        added_reward = airborne_mask_f * height_deficit
                    elif mode == "l2":
                        added_reward = airborne_mask_f * square_sigma * torch.square(height_deficit)
                    elif mode == "binary":
                        added_reward = airborne_mask_f * (height_deficit > 0.0).float()
                    else:
                        raise ValueError(f"Unsupported body_height_below reward mode: {mode}")
                elif addition_type == "undesired_contact_force":
                    net_contact_forces = getattr(
                        getattr(getattr(env, "contact_sensor", None), "data", None),
                        "net_forces_w_history",
                        None,
                    )
                    if net_contact_forces is None or len(getattr(env, "_undesired_contact_link_idx", [])) == 0:
                        added_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
                    else:
                        force_threshold = float(
                            addition_cfg.get(
                                "force_threshold",
                                getattr(env.cfg, "undesired_contact_force_threshold", 5.0),
                            )
                        )
                        mode = addition_cfg.get("mode", "l2")
                        square_sigma = float(addition_cfg.get("square_sigma", 1.0))
                        contact_forces = torch.norm(
                            net_contact_forces[:, :, env._undesired_contact_link_idx].flatten(start_dim=1),
                            dim=-1,
                        )
                        force_over = torch.clamp(
                            contact_forces - force_threshold,
                            min=0.0,
                        )
                        added_reward = airborne_mask_f * self._shape_penalty(
                            force_over, mode, square_sigma=square_sigma
                        )
                elif addition_type == "wheel_contact_force_over":
                    net_contact_forces = getattr(
                        getattr(getattr(env, "contact_sensor", None), "data", None),
                        "net_forces_w_history",
                        None,
                    )
                    if net_contact_forces is None or len(getattr(env, "_desired_contact_link_idx", [])) == 0:
                        added_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
                    else:
                        force_threshold = float(
                            addition_cfg.get(
                                "force_threshold",
                                getattr(env.cfg, "desired_contact_force_threshold", 5.0),
                            )
                        )
                        mode = addition_cfg.get("mode", "l1")
                        square_sigma = float(addition_cfg.get("square_sigma", 1.0))
                        reduce = addition_cfg.get("reduce", "sum")
                        wheel_forces = torch.amax(
                            torch.norm(net_contact_forces[:, :, env._desired_contact_link_idx], dim=-1),
                            dim=1,
                        )
                        force_over = torch.clamp(wheel_forces - force_threshold, min=0.0)
                        wheel_penalty = self._shape_penalty(
                            force_over, mode, square_sigma=square_sigma
                        )
                        if reduce == "mean":
                            added_reward = airborne_mask_f * wheel_penalty.mean(dim=1)
                        elif reduce == "max":
                            added_reward = airborne_mask_f * wheel_penalty.max(dim=1).values
                        elif reduce == "sum":
                            added_reward = airborne_mask_f * wheel_penalty.sum(dim=1)
                        else:
                            raise ValueError(f"Unsupported wheel_contact_force_over reduce mode: {reduce}")
                elif addition_type == "landing_wheel_max_contact_force":
                    contact_forces_w = self._get_current_contact_forces(env)
                    if contact_forces_w is None or len(getattr(env, "_desired_contact_link_idx", [])) == 0:
                        added_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
                    else:
                        reward_mask = self._get_wheel_contact_started_mask(env, addition_cfg)
                        reward_mask &= self._get_wheel_contact_force_mask(env, addition_cfg)
                        force_start = float(addition_cfg.get("force_start", 200.0))
                        force_full = float(addition_cfg.get("force_full", 400.0))
                        denom = max(force_full - force_start, 1.0e-6)
                        wheel_forces = torch.norm(
                            contact_forces_w[:, env._desired_contact_link_idx], dim=-1
                        )
                        max_wheel_force = wheel_forces.max(dim=1).values
                        added_reward = reward_mask.float() * torch.clamp(
                            (max_wheel_force - force_start) / denom,
                            0.0,
                            1.0,
                        )
                elif addition_type == "landing_wheel_body_x_positive":
                    reward_mask = self._get_wheel_contact_started_mask(env, addition_cfg)
                    reward_mask &= self._get_wheel_contact_force_mask(env, addition_cfg)
                    command_x_min = float(addition_cfg.get("command_x_min", 1.0))
                    command_x = self._get_reward_command_lin_vel_x(env, addition_cfg)
                    reward_mask &= command_x > command_x_min
                    target_x = float(addition_cfg.get("target_x", 0.03))
                    sigma = max(float(addition_cfg.get("sigma", target_x)), 1.0e-6)
                    _, _, wheel_pos_heading_b, _, _ = env._get_root_quat_inv_and_wheel_pos_b()
                    wheel_x_min = wheel_pos_heading_b[:, :, 0].min(dim=1).values
                    deficit = torch.clamp(target_x - wheel_x_min, min=0.0)
                    added_reward = reward_mask.float() * torch.exp(-deficit / sigma)
                elif addition_type == "wheel_zero_torque_exp":
                    wheel_idx = getattr(env, "_wheel_idx", [])
                    if len(wheel_idx) == 0:
                        added_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
                    else:
                        reward_mask_f = self._get_reward_window_mask(env, addition_cfg).float()
                        sigma = max(float(addition_cfg.get("sigma", 1.5)), 1.0e-6)
                        wheel_torque = env.robot.data.applied_torque[:, wheel_idx]
                        max_wheel_torque = torch.abs(wheel_torque).max(dim=1).values
                        added_reward = reward_mask_f * torch.exp(
                            -torch.square(max_wheel_torque / sigma)
                        )
                elif addition_type == "wheel_directional_speed":
                    speed_score, _ = self._get_directional_wheel_speed_score(env, addition_cfg)
                    added_reward = speed_score
                elif addition_type == "wheel_directional_speed_shortfall":
                    speed_score, gate_f = self._get_directional_wheel_speed_score(env, addition_cfg)
                    added_reward = gate_f * (1.0 - speed_score)
                elif addition_type == "landing_traj_height_exp":
                    self._ensure_landing_trajectory_buffers(env)
                    active_f = env.airborne_landing_traj_active.float()
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    h_ref, _ = self._get_landing_trajectory_reference(env)
                    height = self._get_body_relative_ground_height(env)
                    added_reward = active_f * torch.exp(-torch.square(height - h_ref) / sigma)
                elif addition_type == "landing_traj_vel_z_exp":
                    self._ensure_landing_trajectory_buffers(env)
                    active_f = env.airborne_landing_traj_active.float()
                    sigma = max(float(addition_cfg.get("sigma", 0.25)), 1.0e-6)
                    _, v_ref = self._get_landing_trajectory_reference(env)
                    vel_z = env.robot.data.root_lin_vel_w[:, 2]
                    added_reward = active_f * torch.exp(-torch.square(vel_z - v_ref) / sigma)
                elif addition_type == "negative_lin_vel_z_after_wheel_contact":
                    wheel_exit_time = getattr(env, "airborne_wheel_contact_exit_time", None)
                    if wheel_exit_time is None:
                        added_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
                    else:
                        start_duration_s = max(float(addition_cfg.get("start_duration_s", 0.02)), 0.0)
                        use_world_frame = bool(addition_cfg.get("use_world_frame", True))
                        mode = addition_cfg.get("mode", "l2")
                        square_sigma = float(addition_cfg.get("square_sigma", 1.0))
                        wheel_contact_started = torch.any(wheel_exit_time >= start_duration_s, dim=1)
                        if use_world_frame:
                            lin_vel_z = env.robot.data.root_lin_vel_w[:, 2]
                        else:
                            lin_vel_z = env.robot.data.root_lin_vel_b[:, 2]
                        down_vel = torch.clamp(-lin_vel_z, min=0.0)
                        added_reward = (
                            airborne_mask_f
                            * wheel_contact_started.float()
                            * self._shape_penalty(down_vel, mode, square_sigma=square_sigma)
                        )
                elif addition_type == "negative_lin_vel_z_after_wheel_contact_exp":
                    wheel_exit_time = getattr(env, "airborne_wheel_contact_exit_time", None)
                    if wheel_exit_time is None:
                        added_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
                    else:
                        start_duration_s = max(float(addition_cfg.get("start_duration_s", 0.02)), 0.0)
                        use_world_frame = bool(addition_cfg.get("use_world_frame", True))
                        sigma = max(float(addition_cfg.get("sigma", 1.0)), 1.0e-6)
                        wheel_contact_started = torch.any(wheel_exit_time >= start_duration_s, dim=1)
                        if use_world_frame:
                            lin_vel_z = env.robot.data.root_lin_vel_w[:, 2]
                        else:
                            lin_vel_z = env.robot.data.root_lin_vel_b[:, 2]
                        added_reward = (
                            airborne_mask_f
                            * wheel_contact_started.float()
                            * torch.exp(-torch.square(lin_vel_z) / sigma)
                        )
                elif addition_type == "leg_retraction":
                    reward_mask_f = self._get_reward_window_mask(env, addition_cfg).float()
                    leg_lengths = self._get_leg_lengths(env)
                    mean_leg_length = leg_lengths.mean(dim=1)
                    target = float(addition_cfg.get("target", 0.22))
                    mode = addition_cfg.get("mode", "above_target")
                    if mode == "raw":
                        added_reward = reward_mask_f * mean_leg_length
                    elif mode == "above_target":
                        added_reward = reward_mask_f * torch.clamp(mean_leg_length - target, min=0.0)
                    elif mode == "above_target_per_leg":
                        added_reward = reward_mask_f * torch.clamp(leg_lengths - target, min=0.0).mean(dim=1)
                    elif mode == "below_target":
                        added_reward = reward_mask_f * torch.clamp(target - mean_leg_length, min=0.0)
                    elif mode == "below_target_per_leg":
                        added_reward = reward_mask_f * torch.clamp(target - leg_lengths, min=0.0).mean(dim=1)
                    elif mode == "exp":
                        sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                        added_reward = reward_mask_f * torch.exp(
                            -torch.square(mean_leg_length - target) / sigma
                        )
                    elif mode == "exp_per_leg":
                        sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                        added_reward = reward_mask_f * torch.exp(
                            -torch.square(leg_lengths - target) / sigma
                        ).mean(dim=1)
                    else:
                        raise ValueError(f"Unsupported airborne leg_retraction reward mode: {mode}")
                elif addition_type == "wheel_height_below_base_exp":
                    reward_mask_f = self._get_reward_window_mask(env, addition_cfg).float()
                    target = float(addition_cfg.get("target", 0.22))
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    rel_wheel_bottom_z = self._get_wheel_bottom_relative_to_base_z(env)
                    target_rel_wheel_bottom_z = -target
                    added_reward = reward_mask_f * torch.exp(
                        -torch.square(rel_wheel_bottom_z - target_rel_wheel_bottom_z) / sigma
                    ).mean(dim=1)
                elif addition_type == "wheel_heading_x_centering":
                    reward_mask_f = self._get_contact_timer_below_mask(env, addition_cfg).float()
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    _, _, wheel_pos_heading_b, _, _ = env._get_root_quat_inv_and_wheel_pos_b()
                    wheel_heading_z_max = addition_cfg.get("wheel_heading_z_max", None)
                    if wheel_heading_z_max is not None:
                        wheel_z_below = torch.all(
                            wheel_pos_heading_b[:, :, 2] < float(wheel_heading_z_max),
                            dim=1,
                        )
                        reward_mask_f = reward_mask_f * wheel_z_below.float()
                    wheel_x = wheel_pos_heading_b[:, :, 0]
                    added_reward = reward_mask_f * torch.exp(-torch.square(wheel_x) / sigma).mean(dim=1)
                elif addition_type == "wheel_bottom_slip_exp":
                    reward_mask_f = self._get_contact_timer_below_mask(env, addition_cfg).float()
                    sigma = max(float(addition_cfg.get("sigma", 0.25)), 1.0e-6)
                    wheel_radius = float(addition_cfg.get("wheel_radius", self._get_wheel_radius(env)))
                    wheel_lin_vel_w = env.robot.data.body_lin_vel_w[:, env._wheel_link_idx]
                    wheel_ang_vel_w = env.robot.data.body_ang_vel_w[:, env._wheel_link_idx]
                    bottom_offset_w = torch.zeros_like(wheel_lin_vel_w)
                    bottom_offset_w[..., 2] = -wheel_radius
                    wheel_bottom_vel_w = wheel_lin_vel_w + torch.cross(
                        wheel_ang_vel_w, bottom_offset_w, dim=-1
                    )
                    slip_speed = torch.linalg.norm(wheel_bottom_vel_w[..., :2], dim=-1)
                    wheel_reward = torch.exp(-torch.square(slip_speed) / sigma)

                    wheel_height_threshold = addition_cfg.get("wheel_height_threshold", None)
                    if wheel_height_threshold is not None:
                        wheel_heights = env._get_wheel_relative_ground_heights_raw()
                        near_ground = wheel_heights < float(wheel_height_threshold)
                        near_ground_f = near_ground.float()
                        near_ground_count = torch.clamp(near_ground_f.sum(dim=1), min=1.0)
                        added_reward = reward_mask_f * (
                            (wheel_reward * near_ground_f).sum(dim=1) / near_ground_count
                        ) * torch.any(near_ground, dim=1).float()
                    else:
                        added_reward = reward_mask_f * wheel_reward.mean(dim=1)
                elif addition_type in (
                    "rear2_rear1_joint_pos_limits",
                    "rear2_rear1_joint_pos_limits_vel_reg",
                ):
                    pos_penalty, vel_penalty = self._get_rear2_rear1_joint_limit_terms(env)
                    if addition_type == "rear2_rear1_joint_pos_limits":
                        added_reward = airborne_mask_f * pos_penalty
                    else:
                        added_reward = airborne_mask_f * vel_penalty
                else:
                    raise ValueError(f"Unsupported airborne reward addition type: {addition_type}")
            else:
                added_reward = airborne_mask_f * float(addition_cfg)
            if term_name in reward_terms:
                reward_terms[term_name] = reward_terms[term_name] + added_reward
            else:
                reward_terms[term_name] = added_reward

        return reward_terms

    def append_reset_logs(
        self, env, extras: dict[str, float], env_ids: torch.Tensor
    ) -> None:
        # reset 时记录该批 env 当前 airborne/退出接触计时占比，便于判断这个状态机是否频繁触发。
        extras["Episode/HeightReward/AirborneRatio"] = (
            env.height_reward_airborne_state[env_ids].float().mean().item()
        )
        extras["Episode/HeightReward/AirborneWheelExitContactRatio"] = (
            torch.any(env.airborne_wheel_contact_exit_time[env_ids] > 0.0, dim=1)
            .float()
            .mean()
            .item()
        )
        extras["Episode/HeightReward/AirborneBaseExitContactRatio"] = (
            (env.airborne_base_contact_exit_time[env_ids] > 0.0).float().mean().item()
        )
        landing_traj_active = getattr(env, "airborne_landing_traj_active", None)
        if landing_traj_active is not None:
            extras["Episode/HeightReward/AirborneLandingTrajRatio"] = (
                landing_traj_active[env_ids].float().mean().item()
            )
        airborne_max_duration = getattr(env, "airborne_max_duration", None)
        if airborne_max_duration is not None:
            episode_airborne_max_duration = airborne_max_duration[env_ids]
            extras["Episode/HeightReward/AirborneMaxDurationMean"] = (
                episode_airborne_max_duration.mean().item()
            )
            extras["Episode/HeightReward/AirborneMaxDurationMax"] = (
                episode_airborne_max_duration.max().item()
            )

    def apply_visual_marker_state(
        self,
        env,
        marker_indices: torch.Tensor,
        priorities: torch.Tensor,
        state_name_to_index: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        airborne_state_index = state_name_to_index.get("airborne")
        if airborne_state_index is None:
            return marker_indices, priorities

        # play 可视化里 airborne 的优先级低于 step_up/wall，但高于 neutral。
        airborne_mask = env.height_reward_airborne_state
        airborne_priority = 10
        update_mask = airborne_mask & (airborne_priority >= priorities)
        if torch.any(update_mask):
            marker_indices = marker_indices.clone()
            priorities = priorities.clone()
            marker_indices[update_mask] = airborne_state_index
            priorities[update_mask] = airborne_priority
        return marker_indices, priorities

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        # 每个 env reset 时只清自己的状态，避免影响同批并行环境里未 reset 的 episode。
        env.height_reward_airborne_state[env_ids] = False
        force_enter_request = getattr(env, "airborne_force_enter_request", None)
        if force_enter_request is not None:
            force_enter_request[env_ids] = False
        env.airborne_wheel_contact_exit_time[env_ids] = 0.0
        env.airborne_base_contact_exit_time[env_ids] = 0.0
        enter_time = getattr(env, "airborne_enter_time", None)
        if enter_time is not None:
            enter_time[env_ids] = 0.0
        current_duration = getattr(env, "airborne_current_duration", None)
        if current_duration is not None:
            current_duration[env_ids] = 0.0
        max_duration = getattr(env, "airborne_max_duration", None)
        if max_duration is not None:
            max_duration[env_ids] = 0.0
        last_update_step = getattr(env, "airborne_state_last_update_step", None)
        if last_update_step is not None:
            last_update_step[env_ids] = -1
        if hasattr(env, "airborne_command_override_active"):
            env.airborne_command_override_active[env_ids] = False
            env.airborne_command_override_values[env_ids] = 0.0
            env.airborne_command_override_field_mask[env_ids] = False
        self._clear_landing_trajectory(env, env_ids)
