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

import colorsys
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import *
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.buffers import DelayBuffer, CircularBuffer, TimestampedBuffer

from agent_tasks.direct.wheelbipe.wheelbipe_V13.env_cfg import WheelbipeV13FlatEnvCfg
from agent_tasks.manager.mdp.terrain import TerrainCommandManager
from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env import Wheelbipe25V3Env


class WheelbipeV13Env(Wheelbipe25V3Env):
    cfg: WheelbipeV13FlatEnvCfg
    _skip_builtin_terrain_debug_marker = True
    
    def __init__(self, cfg: WheelbipeV13FlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # V13 keeps wheel body indices for kinematics/height, but contact-force related
        # indices must stay in the contact-sensor index space.
        self._wheel_link_idx,_ = self.robot.find_bodies(".*_wheel_link")
        self._undesired_contact_link_idx = self._find_contact_sensor_indices([
            "base_link", ".*_rear1_link", ".*_rear2_link",
            ".*_front1_link", ".*_front2_link", ".*_front3_link", ".*_front4_link",
            ".*_guide_link",
        ])
        self._desired_contact_link_idx = self._find_contact_sensor_indices([".*_wheel_link"])
        self._reset_contact_link_idx = self._find_contact_sensor_indices(["base_link", ".*_guide_link"])

        self._terrain_command_manager = TerrainCommandManager(
            terrain=self.terrain,
            cfg=self.cfg,
            device=self.device,
            num_envs=self.num_envs,
        )
        self._terrain_type_marker: VisualizationMarkers | None = None
        self._terrain_type_marker_height_offset = 0.7
        self._initialize_terrain_command_state()
        self._setup_terrain_type_marker()
        self._update_terrain_type_marker()

    def _is_terrain_type_marker_enabled(self) -> bool:
        return bool(getattr(self.cfg, "play", False)) and bool(
            getattr(self.cfg, "play_terrain_debug_vis", False)
        )

    def _setup_terrain_type_marker(self) -> None:
        manager = self._get_terrain_command_manager()
        if manager is None or not self._is_terrain_type_marker_enabled():
            self._terrain_type_marker = None
            return

        marker_defs: dict[str, sim_utils.SphereCfg] = {
            "unknown": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.45, 0.45, 0.45),
                    emissive_color=(0.08, 0.08, 0.08),
                    metallic=0.0,
                    roughness=0.45,
                ),
            )
        }

        terrain_color_map: list[str] = []
        num_keys = max(len(manager.terrain_keys), 1)
        for idx, terrain_key in enumerate(manager.terrain_keys):
            hue = float(idx) / float(num_keys)
            color = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            marker_defs[f"terrain_{idx}"] = sim_utils.SphereCfg(
                radius=0.085,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    emissive_color=tuple(channel * 0.35 for channel in color),
                    metallic=0.0,
                    roughness=0.18,
                ),
            )
            terrain_color_map.append(f"{idx}:{terrain_key}")

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/terrain_type_marker",
            markers=marker_defs,
        )
        self._terrain_type_marker = VisualizationMarkers(marker_cfg)
        self._terrain_type_marker_height_offset = 1.45
        self._terrain_type_marker.set_visibility(True)
        print("[TerrainMarker] key->color-index:", ", ".join(terrain_color_map))

    def _update_terrain_type_marker(self) -> None:
        marker = getattr(self, "_terrain_type_marker", None)
        if marker is None:
            return
        if not self._is_terrain_type_marker_enabled():
            marker.set_visibility(False)
            return

        manager = self._get_terrain_command_manager()
        if manager is None:
            marker.set_visibility(False)
            return

        marker_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        terrain_key_indices = manager.get_current_terrain_key_indices()
        valid_mask = terrain_key_indices >= 0
        if torch.any(valid_mask):
            marker_indices[valid_mask] = terrain_key_indices[valid_mask] + 1

        positions = self.robot.data.root_pos_w.clone()
        positions[:, 2] += self._terrain_type_marker_height_offset
        marker.set_visibility(True)
        marker.visualize(
            translations=positions.detach().cpu(),
            marker_indices=marker_indices.detach().cpu(),
        )

    def _get_terrain_command_manager(self) -> TerrainCommandManager | None:
        manager = getattr(self, "_terrain_command_manager", None)
        if manager is None or not manager.enabled:
            return None
        return manager

    def _initialize_terrain_command_state(self) -> None:
        manager = self._get_terrain_command_manager()
        if manager is None:
            return

        env_ids = self.robot._ALL_INDICES
        self._update_axis_aligned_reset_heading_mask(env_ids, use_env_origins=True)
        self._force_resample_commands(env_ids)
        self.height_cmd[env_ids] = self._sample_height_command(env_ids, use_env_origins=True)
        self._sync_heading_command_target_to_reset_heading(env_ids)
        manager.resample_command_overrides(env_ids)
        self._disable_terrain_special_modes(env_ids)
        axis_aligned_mask = self._get_axis_aligned_reset_heading_mask()
        is_heading_env = getattr(self.command_generator, "is_heading_env", None)
        manager.apply_command_overrides(self.command, is_heading_env=is_heading_env)
        self._restore_heading_closed_loop_yaw_command()
        if is_heading_env is not None:
            non_heading_mask = (~is_heading_env) & axis_aligned_mask
            if torch.any(non_heading_mask):
                self.command[non_heading_mask, 2] = 0.0
        self._sync_command_generator_command()

    def _sync_command_generator_command(self) -> None:
        """Keep IsaacLab command debug-vis in sync with the actual overridden command."""
        command_generator = getattr(self, "command_generator", None)
        if command_generator is None:
            return
        generator_command = getattr(command_generator, "command", None)
        if generator_command is None:
            return
        if generator_command.shape != self.command.shape:
            return
        generator_command.copy_(self.command)

    def _disable_terrain_special_modes(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ) -> torch.Tensor:
        """Disable special-mode command samples for terrain profiles that request it."""
        manager = self._get_terrain_command_manager()
        command_generator = getattr(self, "command_generator", None)
        if manager is None or command_generator is None:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        disable_special_modes = getattr(command_generator, "disable_special_modes", None)
        if disable_special_modes is None:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return env_ids_t

        disable_mask = manager.get_disable_special_mode_mask(env_ids_t)
        if not torch.any(disable_mask):
            return torch.zeros(0, dtype=torch.long, device=self.device)

        disabled_env_ids = env_ids_t[disable_mask]
        affected_ids = disable_special_modes(disabled_env_ids)
        if affected_ids.numel() > 0:
            self.command[affected_ids] = command_generator.command[affected_ids]
            self.jump_takeoff_request[affected_ids] = False
        return affected_ids

    def _restore_heading_closed_loop_yaw_command(self) -> None:
        """Keep heading envs on the command generator's closed-loop yaw-rate."""
        command_generator = getattr(self, "command_generator", None)
        if command_generator is None:
            return

        is_heading_env = getattr(command_generator, "is_heading_env", None)
        generator_command = getattr(command_generator, "command", None)
        if is_heading_env is None or generator_command is None:
            return
        if generator_command.shape != self.command.shape:
            return

        heading_mask = is_heading_env.to(device=self.device, dtype=torch.bool)
        if torch.any(heading_mask):
            self.command[heading_mask, 2] = generator_command[heading_mask, 2]

    def _update_axis_aligned_reset_heading_mask(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        use_env_origins: bool = False,
    ) -> None:
        manager = self._get_terrain_command_manager()
        if manager is None:
            super()._update_axis_aligned_reset_heading_mask(env_ids, use_env_origins=use_env_origins)
            return

        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        manager.sync_envs(
            env_ids_t,
            root_pos_w=self.robot.data.root_pos_w,
            use_env_origins=use_env_origins,
        )
        self._axis_aligned_reset_heading_mask[env_ids_t] = manager.get_reset_heading_mask(env_ids_t)
        if use_env_origins:
            manager.resample_command_overrides(env_ids_t)

    def _get_non_heading_axis_aligned_zero_mask(
        self, is_heading_env: torch.Tensor | None
    ) -> torch.Tensor:
        zero_mask = super()._get_non_heading_axis_aligned_zero_mask(is_heading_env)
        if not torch.any(zero_mask):
            return zero_mask

        manager = self._get_terrain_command_manager()
        if manager is None:
            return zero_mask
        non_heading_override_mask = manager.get_command_override_mask("ang_vel_z_non_heading")
        return zero_mask & (~non_heading_override_mask)

    def _get_predefined_reset_air_disabled_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        manager = self._get_terrain_command_manager()
        if manager is None:
            return super()._get_predefined_reset_air_disabled_mask(env_ids)

        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        manager.sync_envs(
            env_ids_t,
            root_pos_w=self.robot.data.root_pos_w,
            use_env_origins=True,
        )
        return manager.get_disable_predefined_reset_air_mask(env_ids_t)

    def _get_predefined_reset_ground_disabled_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        manager = self._get_terrain_command_manager()
        if manager is None:
            return super()._get_predefined_reset_ground_disabled_mask(env_ids)

        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        manager.sync_envs(
            env_ids_t,
            root_pos_w=self.robot.data.root_pos_w,
            use_env_origins=True,
        )
        return manager.get_disable_predefined_reset_ground_mask(env_ids_t)

    def _get_jump_takeoff_disabled_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        disabled = super()._get_jump_takeoff_disabled_mask(env_ids)
        manager = self._get_terrain_command_manager()
        if manager is None:
            return disabled

        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return disabled
        manager.sync_envs(
            env_ids_t,
            root_pos_w=self.robot.data.root_pos_w,
            use_env_origins=True,
        )
        return disabled | manager.get_disable_jump_takeoff_mask(env_ids_t)

    def _sample_height_command(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        use_env_origins: bool = False,
    ) -> torch.Tensor:
        manager = self._get_terrain_command_manager()
        if manager is None:
            return super()._sample_height_command(env_ids, use_env_origins=use_env_origins)

        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.float, device=self.device)
        manager.sync_envs(
            env_ids_t,
            root_pos_w=self.robot.data.root_pos_w,
            use_env_origins=use_env_origins,
        )
        return manager.sample_height(env_ids_t)

    def _resample_custom_cmd(self, command_counter: torch.tensor):
        resample_env_ids = super()._resample_custom_cmd(command_counter)
        manager = self._get_terrain_command_manager()
        if manager is None or resample_env_ids is None or len(resample_env_ids) == 0:
            return resample_env_ids

        manager.sync_envs(resample_env_ids, root_pos_w=self.robot.data.root_pos_w)
        manager.resample_command_overrides(resample_env_ids)
        self._disable_terrain_special_modes(resample_env_ids)
        return resample_env_ids

    def _apply_terrain_command_overrides_for_current_step(self, *, resample_on_switch: bool) -> None:
        manager = self._get_terrain_command_manager()
        if manager is None:
            return

        changed_env_ids = manager.sync_envs(
            self.robot._ALL_INDICES, root_pos_w=self.robot.data.root_pos_w
        )
        if resample_on_switch and changed_env_ids.numel() > 0:
            self._force_resample_commands(changed_env_ids)
            self._sync_heading_command_target_to_reset_heading(changed_env_ids)
            self.height_cmd[changed_env_ids] = self._sample_height_command(changed_env_ids)
            manager.resample_command_overrides(changed_env_ids)

        self._disable_terrain_special_modes(self.robot._ALL_INDICES)
        manager.apply_command_overrides(
            self.command,
            is_heading_env=getattr(self.command_generator, "is_heading_env", None),
        )
        self._restore_heading_closed_loop_yaw_command()

    def _before_state_machine_command_updated(self) -> None:
        super()._before_state_machine_command_updated()
        self._apply_terrain_command_overrides_for_current_step(resample_on_switch=True)

    def _after_state_machine_command_updated(self) -> None:
        super()._after_state_machine_command_updated()
        self._apply_terrain_command_overrides_for_current_step(resample_on_switch=False)

    def _on_command_updated(self) -> None:
        super()._on_command_updated()
        self._sync_command_generator_command()
        self._update_terrain_type_marker()
