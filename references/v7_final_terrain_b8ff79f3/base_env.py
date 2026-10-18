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
from collections.abc import Mapping, Sequence
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import CurriculumManager
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import *
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.buffers import DelayBuffer, CircularBuffer, TimestampedBuffer
from isaaclab.utils.warp import raycast_mesh

from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env_cfg import Wheelbipe25v3ReduceSpringFlatEnvCfg, Wheelbipe25v3FlatEnvCfg
from agent_tasks.direct.wheelbipe.state_machines import WheelbipeStateMachineManager
from agent_tasks.manager.mdp.terrain import TerrainTaskManager


class Wheelbipe25V3Env(DirectRLEnv):
    cfg: Wheelbipe25v3FlatEnvCfg | Wheelbipe25v3ReduceSpringFlatEnvCfg
    _skip_builtin_terrain_debug_marker = False

    def _get_vel_height_gate_full_error(self) -> float | torch.Tensor:
        return getattr(self.cfg, "vel_height_gate_full_error", 0.05)

    def _get_vel_height_gate_zero_error(self) -> float | torch.Tensor:
        return getattr(self.cfg, "vel_height_gate_zero_error", 0.10)

    def _get_vel_height_gate_enabled(self) -> bool | torch.Tensor:
        return bool(getattr(self.cfg, "vel_height_gate_enabled", False))

    def _get_vel_orientation_x_gate_enabled(self) -> bool | torch.Tensor:
        return bool(getattr(self.cfg, "vel_orientation_x_gate_enabled", False))

    def _get_vel_orientation_x_gate_full_deg(self) -> float | torch.Tensor:
        return getattr(self.cfg, "vel_orientation_x_gate_full_deg", 5.0)

    def _get_vel_orientation_x_gate_zero_deg(self) -> float | torch.Tensor:
        return getattr(self.cfg, "vel_orientation_x_gate_zero_deg", 20.0)

    def _get_vel_orientation_y_gate_enabled(self) -> bool | torch.Tensor:
        return bool(getattr(self.cfg, "vel_orientation_y_gate_enabled", False))

    def _get_vel_orientation_y_gate_full_deg(self) -> float | torch.Tensor:
        return getattr(self.cfg, "vel_orientation_y_gate_full_deg", 5.0)

    def _get_vel_orientation_y_gate_zero_deg(self) -> float | torch.Tensor:
        return getattr(self.cfg, "vel_orientation_y_gate_zero_deg", 20.0)

    def _as_reward_gate_tensor(self, value: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        '''return a value tensor whose shape is the same as the reference'''
        if torch.is_tensor(value):
            return torch.nan_to_num(value.to(device=self.device, dtype=torch.float), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.full_like(reference, float(value))

    def _as_reward_gate_bool_tensor(self, value: bool | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.bool)
        return torch.full_like(reference, bool(value), dtype=torch.bool)

    def _is_play_ang_vel_z_debug_vis_enabled(self) -> bool:
        """Whether the play-mode yaw-rate arrow visualization should be shown."""
        return bool(getattr(self.cfg, "play", False)) and bool(getattr(self.cfg, "play_ang_vel_z_debug_vis", True))

    def _setup_play_ang_vel_z_marker(self) -> None:
        """Create play-mode arrows for commanded and measured yaw-rate visualization."""
        self._play_ang_vel_z_cmd_marker = None
        self._play_ang_vel_z_actual_marker = None
        if not self._is_play_ang_vel_z_debug_vis_enabled():
            return

        cmd_marker_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/play_ang_vel_z_cmd")
        actual_marker_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/play_ang_vel_z_actual")
        cmd_marker_cfg.markers["arrow"].scale = (0.8, 0.25, 0.25)
        actual_marker_cfg.markers["arrow"].scale = (0.7, 0.20, 0.20)

        self._play_ang_vel_z_cmd_marker = VisualizationMarkers(cmd_marker_cfg)
        self._play_ang_vel_z_actual_marker = VisualizationMarkers(actual_marker_cfg)
        self._play_ang_vel_z_marker_height_offset = 0.78
        self._play_ang_vel_z_cmd_marker_z_offset = 0.035
        self._play_ang_vel_z_actual_marker_z_offset = 0.0
        self._play_ang_vel_z_arrow_scale_factor = 1.8
        self._play_ang_vel_z_cmd_marker.set_visibility(True)
        self._play_ang_vel_z_actual_marker.set_visibility(True)

    def _set_play_ang_vel_z_marker_visibility(self, visible: bool) -> None:
        """Set visibility for both yaw-rate play markers."""
        for attr_name in ("_play_ang_vel_z_cmd_marker", "_play_ang_vel_z_actual_marker"):
            marker = getattr(self, attr_name, None)
            if marker is not None:
                marker.set_visibility(visible)

    def _resolve_yaw_rate_to_arrow(self, yaw_rate: torch.Tensor, marker: VisualizationMarkers) -> tuple[torch.Tensor, torch.Tensor]:
        """Map yaw-rate sign/magnitude to a local-y arrow in the robot base frame."""
        default_scale = marker.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device, dtype=yaw_rate.dtype).repeat(yaw_rate.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(yaw_rate) * self._play_ang_vel_z_arrow_scale_factor

        heading_angle = torch.sign(yaw_rate) * (math.pi * 0.5)
        zeros = torch.zeros_like(heading_angle)
        arrow_quat_b = quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat_w = quat_mul(self.robot.data.root_quat_w, arrow_quat_b)
        return arrow_scale, arrow_quat_w

    def _get_play_ang_vel_z_marker_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearly overlapping marker positions above each robot."""
        base_positions = self.robot.data.root_pos_w.clone()
        base_positions[:, 2] += self._play_ang_vel_z_marker_height_offset

        cmd_positions = base_positions.clone()
        actual_positions = base_positions.clone()
        cmd_positions[:, 2] += self._play_ang_vel_z_cmd_marker_z_offset
        actual_positions[:, 2] += self._play_ang_vel_z_actual_marker_z_offset
        return cmd_positions, actual_positions

    def _update_play_ang_vel_z_marker(self) -> None:
        """Visualize commanded and measured yaw-rate as local-y arrows in play mode."""
        cmd_marker = getattr(self, "_play_ang_vel_z_cmd_marker", None)
        actual_marker = getattr(self, "_play_ang_vel_z_actual_marker", None)
        if cmd_marker is None or actual_marker is None:
            return
        if not self._is_play_ang_vel_z_debug_vis_enabled():
            self._set_play_ang_vel_z_marker_visibility(False)
            return

        yaw_rate_cmd = self.command[:, 2]
        yaw_rate_actual = self.robot.data.root_ang_vel_b[:, 2]
        cmd_positions, actual_positions = self._get_play_ang_vel_z_marker_positions()
        cmd_scales, cmd_quats = self._resolve_yaw_rate_to_arrow(yaw_rate_cmd, cmd_marker)
        actual_scales, actual_quats = self._resolve_yaw_rate_to_arrow(yaw_rate_actual, actual_marker)

        self._set_play_ang_vel_z_marker_visibility(True)
        cmd_marker.visualize(
            translations=cmd_positions.detach().cpu(),
            orientations=cmd_quats.detach().cpu(),
            scales=cmd_scales.detach().cpu(),
        )
        actual_marker.visualize(
            translations=actual_positions.detach().cpu(),
            orientations=actual_quats.detach().cpu(),
            scales=actual_scales.detach().cpu(),
        )

    def _setup_state_machine_marker(self) -> None:
        """Create play-mode state-machine markers through the runtime manager."""
        self.state_machine_manager.setup_visual_marker(self)

    def _update_state_machine_marker(self) -> None:
        """Update play-mode state-machine markers through the runtime manager."""
        self.state_machine_manager.update_visual_marker(self)

    def _setup_wheel_forward_scan_marker(self) -> None:
        """Create play-mode markers for dynamic wheel-forward terrain hit points."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/wheel_forward_scan_marker",
            markers={
                "right_hit": sim_utils.SphereCfg(
                    radius=0.028,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.8, 1.0),
                        emissive_color=(0.0, 0.12, 0.18),
                        metallic=0.0,
                        roughness=0.2,
                    ),
                ),
                "left_hit": sim_utils.SphereCfg(
                    radius=0.028,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 1.0, 0.45),
                        emissive_color=(0.0, 0.16, 0.05),
                        metallic=0.0,
                        roughness=0.2,
                    ),
                ),
            },
        )
        self._wheel_forward_scan_marker = VisualizationMarkers(marker_cfg)
        self._wheel_forward_scan_marker.set_visibility(False)

    def _is_wheel_forward_scan_enabled(self) -> bool:
        """Whether wheel-forward terrain probing is enabled."""
        return bool(self._get_wheel_forward_scan_cfg().get("enabled", False))

    def _get_wheel_forward_scan_cfg(self) -> dict:
        """Return wheel-forward scan config."""
        return getattr(self.cfg, "wheel_forward_scan_cfg", {"enabled": False})

    def _get_dynamic_wheel_forward_scan_points(
        self,
        *,
        forward_offset: float,
        query_cache_attr: str,
        hit_cache_attr: str,
        direction_cache_attr: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return dynamic query/hit points for two wheel-forward terrain probes."""
        if not self._is_wheel_forward_scan_enabled():
            return None, None
        if not self._use_raycast_height():
            return None, None
        cached_query_points = getattr(self, query_cache_attr, None)
        cached_hit_points = getattr(self, hit_cache_attr, None)
        if cached_query_points is not None and cached_hit_points is not None:
            return cached_query_points, cached_hit_points

        mesh_scanner = self.height_scanner or self.right_wheel_height_scanner or self.left_wheel_height_scanner
        if mesh_scanner is None:
            return None, None
        if len(self._right_wheel_link_idx) == 0 or len(self._left_wheel_link_idx) == 0:
            return None, None

        wheel_indices = [self._right_wheel_link_idx[0], self._left_wheel_link_idx[0]]
        wheel_pos_w = self.robot.data.body_pos_w[:, wheel_indices].clone()

        forward_dir_w = quat_apply_yaw(
            self.robot.data.root_quat_w,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
        )
        command = getattr(self, "command", None)
        if command is None or command.shape[0] != self.num_envs or command.shape[1] == 0:
            scan_direction_sign = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        else:
            scan_direction_sign = torch.where(
                command[:, 0] < 0.0,
                torch.full((self.num_envs,), -1.0, dtype=torch.float, device=self.device),
                torch.ones(self.num_envs, dtype=torch.float, device=self.device),
            )
        setattr(self, direction_cache_attr, scan_direction_sign)
        scan_dir_w = forward_dir_w * scan_direction_sign.unsqueeze(-1)
        forward_query_xy = wheel_pos_w + forward_offset * scan_dir_w.unsqueeze(1)

        ray_starts_w = forward_query_xy.clone()
        ray_starts_w[:, :, 2] += 5.0
        ray_directions_w = torch.zeros_like(ray_starts_w)
        ray_directions_w[:, :, 2] = -1.0

        ray_hits_w = raycast_mesh(
            ray_starts_w.view(-1, 3),
            ray_directions_w.view(-1, 3),
            max_dist=mesh_scanner.cfg.max_distance,
            mesh=mesh_scanner.meshes[mesh_scanner.cfg.mesh_prim_paths[0]],
        )[0].view(self.num_envs, 2, 3)

        valid_mask = torch.isfinite(ray_hits_w).all(dim=-1)
        if not torch.all(valid_mask):
            fallback_hits = forward_query_xy.clone()
            fallback_hits[:, :, 2] = self.ground_z_est.unsqueeze(1)
            ray_hits_w = torch.where(valid_mask.unsqueeze(-1), ray_hits_w, fallback_hits)

        setattr(self, query_cache_attr, ray_starts_w)
        setattr(self, hit_cache_attr, ray_hits_w)
        return ray_starts_w, ray_hits_w

    def _get_wheel_forward_scan_points(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return dynamic query/hit points for the regular wheel-forward terrain probes."""
        scan_cfg = self._get_wheel_forward_scan_cfg().get("scan", {})
        return self._get_dynamic_wheel_forward_scan_points(
            forward_offset=float(scan_cfg.get("forward_offset", 0.10)),
            query_cache_attr="_cached_wheel_forward_query_points",
            hit_cache_attr="_cached_wheel_forward_hit_points",
            direction_cache_attr="_cached_wheel_forward_scan_direction_sign",
        )

    def _get_wheel_forward_stair_scan_points(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return dynamic query/hit points for short-range wheel-forward stair probes."""
        stair_cfg = getattr(self.cfg, "stair_state_machine_cfg", {})
        if not bool(stair_cfg.get("enabled", False)):
            return None, None
        scan_cfg = stair_cfg.get("scan", {})
        return self._get_dynamic_wheel_forward_scan_points(
            forward_offset=float(scan_cfg.get("forward_offset", 0.20)),
            query_cache_attr="_cached_wheel_forward_stair_query_points",
            hit_cache_attr="_cached_wheel_forward_stair_hit_points",
            direction_cache_attr="_cached_wheel_forward_stair_scan_direction_sign",
        )

    def _update_wheel_forward_scan_marker(self) -> None:
        """Visualize wheel-forward terrain hit points in play mode."""
        marker = getattr(self, "_wheel_forward_scan_marker", None)
        if marker is None:
            return
        if not self._is_wheel_forward_scan_enabled():
            marker.set_visibility(False)
            return
        if not bool(getattr(self.cfg, "play", False)) or not bool(getattr(self.cfg, "play_height_scanner_debug_vis", False)):
            marker.set_visibility(False)
            return

        query_points_w, hit_points_w = self._get_wheel_forward_scan_points()
        if query_points_w is None or hit_points_w is None:
            marker.set_visibility(False)
            return

        marker.set_visibility(True)
        translations = torch.cat(
            [
                hit_points_w[:, 0],
                hit_points_w[:, 1],
            ],
            dim=0,
        )
        marker_indices = torch.cat(
            [
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.ones(self.num_envs, dtype=torch.long, device=self.device),
            ],
            dim=0,
        )
        marker.visualize(
            translations=translations.detach().cpu(),
            marker_indices=marker_indices.detach().cpu(),
        )

    def _is_builtin_terrain_debug_marker_enabled(self) -> bool:
        """Whether the generic terrain marker should be shown in play mode."""
        return (
            not bool(getattr(self, "_skip_builtin_terrain_debug_marker", False))
            and bool(getattr(self.cfg, "play", False))
            and bool(getattr(self.cfg, "play_terrain_debug_vis", False))
        )

    def _build_builtin_terrain_task_manager(self) -> TerrainTaskManager | None:
        """Create the terrain-to-marker mapper for rough curriculum terrains."""
        terrain_cfg = getattr(self.cfg, "terrain", None)
        terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
        sub_terrains = getattr(terrain_generator, "sub_terrains", None)
        if sub_terrains is None:
            return None

        sub_terrain_keys = list(sub_terrains.keys())
        if len(sub_terrain_keys) == 0:
            return None

        manager = TerrainTaskManager(
            terrain_importer=self.terrain,
            sub_terrain_keys=sub_terrain_keys,
            device=self.device,
        )
        return manager if manager.enabled else None

    def _get_terrain_task_manager(self) -> TerrainTaskManager | None:
        """Return the cached rough-terrain name mapper, building it lazily when needed."""
        if getattr(self, "_terrain_task_manager_initialized", False):
            return getattr(self, "_terrain_task_manager", None)
        self._terrain_task_manager = self._build_builtin_terrain_task_manager()
        self._terrain_task_manager_initialized = True
        return self._terrain_task_manager

    def get_terrain_name_mask(self, terrain_names: Sequence[str] | str | None) -> torch.Tensor:
        """Return a bool mask for envs whose current terrain key is in ``terrain_names``.

        An empty name list means unrestricted. A non-empty list on an unmapped
        terrain returns all False so gated features do not trigger accidentally.
        """
        if terrain_names is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if isinstance(terrain_names, str):
            terrain_names = (terrain_names,)
        terrain_names = tuple(name for name in terrain_names if name)
        if len(terrain_names) == 0:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        manager = self._get_terrain_task_manager()
        if manager is None or not manager.enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        masks = manager.get_task_masks(self.robot.data.root_pos_w)
        allowed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        unknown_names: list[str] = []
        for name in terrain_names:
            terrain_mask = masks.get(name)
            if terrain_mask is None:
                unknown_names.append(name)
                continue
            allowed_mask |= terrain_mask
        if unknown_names:
            warned = getattr(self, "_warned_unknown_stair_terrain_names", set())
            new_unknown = [name for name in unknown_names if name not in warned]
            if new_unknown:
                print(
                    "[WARNING] Unknown stair_state_machine_cfg terrain names: "
                    + ", ".join(new_unknown)
                )
                warned.update(new_unknown)
                self._warned_unknown_stair_terrain_names = warned
        return allowed_mask

    def _setup_builtin_terrain_debug_marker(self) -> None:
        """Create play-mode markers for rough-terrain type visualization."""
        self._terrain_debug_marker = None

        if not self._is_builtin_terrain_debug_marker_enabled():
            return

        manager = self._get_terrain_task_manager()
        if manager is None:
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
        terrain_names = list(manager.name_to_idx.keys())
        num_keys = max(len(terrain_names), 1)
        for idx, terrain_name in enumerate(terrain_names):
            color = colorsys.hsv_to_rgb(float(idx) / float(num_keys), 0.8, 1.0)
            marker_defs[f"terrain_{idx}"] = sim_utils.SphereCfg(
                radius=0.085,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    emissive_color=tuple(channel * 0.35 for channel in color),
                    metallic=0.0,
                    roughness=0.18,
                ),
            )
            terrain_color_map.append(f"{idx}:{terrain_name}")

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/terrain_type_marker",
            markers=marker_defs,
        )
        self._terrain_task_manager = manager
        self._terrain_debug_marker = VisualizationMarkers(marker_cfg)
        self._terrain_debug_marker_height_offset = 1.45
        self._terrain_debug_marker.set_visibility(True)
        print("[TerrainMarker] key->color-index:", ", ".join(terrain_color_map))

    def _update_builtin_terrain_debug_marker(self) -> None:
        """Update the play-mode terrain-type marker above each robot."""
        marker = getattr(self, "_terrain_debug_marker", None)
        if marker is None:
            return
        if not self._is_builtin_terrain_debug_marker_enabled():
            marker.set_visibility(False)
            return

        manager = getattr(self, "_terrain_task_manager", None)
        if manager is None or not manager.enabled:
            marker.set_visibility(False)
            return

        marker_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        terrain_ids = manager.get_task_ids(self.robot.data.root_pos_w)
        valid_mask = terrain_ids >= 0
        if torch.any(valid_mask):
            marker_indices[valid_mask] = terrain_ids[valid_mask] + 1

        positions = self.robot.data.root_pos_w.clone()
        positions[:, 2] += self._terrain_debug_marker_height_offset
        marker.set_visibility(True)
        marker.visualize(
            translations=positions.detach().cpu(),
            marker_indices=marker_indices.detach().cpu(),
        )

    def _use_absolute_height(self) -> bool:
        return bool(getattr(self.cfg, "use_absolute_height", False))

    def _use_leg_length_height(self) -> bool:
        return bool(getattr(self.cfg, "use_leg_length_as_height", False))

    def _use_raycast_height(self) -> bool:
        return (not self._use_leg_length_height()) and (not self._use_absolute_height())

    def _get_height_measure_wheel_radius(self) -> float:
        """Return the wheel radius used to align leg-length height with body height commands."""
        airborne_cfg = getattr(self.cfg, "airborne_state_machine_cfg", {})
        enter_cfg = airborne_cfg.get("enter", {})
        if "wheel_radius" in enter_cfg:
            return float(enter_cfg["wheel_radius"])

        wheel_radius = getattr(self, "wheel_radius", None)
        if wheel_radius is not None:
            return float(wheel_radius)
        return 0.05

    def _get_leg_length_height(self, wheel_pos_b: torch.Tensor) -> torch.Tensor:
        """Return the average body-to-wheel-joint-center length."""
        return torch.norm(wheel_pos_b, dim=-1).mean(dim=1)

    def _apply_height_obs_clip(self, height: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self.cfg, "height_obs_clip_enabled", False)):
            return height
        clip_range = getattr(self.cfg, "height_obs_clip_range", (None, None))
        if clip_range is None:
            return height
        if len(clip_range) != 2:
            raise RuntimeError("height_obs_clip_range must be [lower, upper]")
        lower, upper = clip_range
        lower_f = None if lower is None else float(lower)
        upper_f = None if upper is None else float(upper)
        if lower_f is None and upper_f is None:
            return height
        if lower_f is not None and upper_f is not None and upper_f < lower_f:
            raise RuntimeError("height_obs_clip_range upper must be >= lower")
        return torch.clamp(height, min=lower_f, max=upper_f)

    def _get_observed_height(self, wheel_pos_b: torch.Tensor | None = None) -> torch.Tensor:
        """Return the height signal used by obs/reward.

        - leg-length mode: average leg length to the wheel joint center
        - absolute mode: root absolute z in world frame
        - default mode: root z, then rewards convert it to relative height using raycast-estimated ground_z_est
        - optional height_obs_clip_range clamps this signal before downstream consumers use it
        """
        if self._use_leg_length_height():
            if wheel_pos_b is None:
                root_quat_inv = quat_inv(self.robot.data.root_quat_w)
                wheel_pos_b = quat_apply(
                    root_quat_inv.unsqueeze(1).expand(-1, self._wheel_link_count, -1),
                    self.robot.data.body_pos_w[:, self._wheel_link_idx]
                    - self.robot.data.root_pos_w.unsqueeze(1).expand(-1, self._wheel_link_count, -1),
                )
                wheel_pos_b[:, :, 1].fill_(0.0)
            height = self._get_leg_length_height(wheel_pos_b)
        else:
            height = self.robot.data.root_pos_w[:, 2]
        return self._apply_height_obs_clip(height)

    def _get_wheel_motor_z_axis_align_error_sq(self) -> torch.Tensor:
        """Return the squared leg-axis misalignment against the world gravity direction.

        Although the reward key keeps its legacy name for compatibility, the geometry is:
        - start point: the two body-fixed leg-root reference points at y = +/- ref_y_offset
        - end point: the corresponding wheel-link world position (leg endpoint proxy)
        - target: the start->end leg vector should align with gravity direction

        The returned error is the squared norm of the component perpendicular to gravity,
        computed on normalized leg vectors so it measures direction rather than leg length.
        """
        if len(getattr(self, "_left_wheel_link_idx", [])) == 0 or len(getattr(self, "_right_wheel_link_idx", [])) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        ref_y_offset = float(
            getattr(self.cfg, "wheel_motor_z_axis_align_ref_y_offset", 0.217)
        )
        tolerance = float(
            getattr(
                self.cfg,
                "wheel_motor_z_axis_align_tolerance",
                getattr(self.cfg, "wheel_body_x_zero_tolerance", 0.0),
            )
        )

        left_ref_b = self.robot.data.root_pos_w.new_tensor([0.0, ref_y_offset, 0.0]).unsqueeze(0).expand(self.num_envs, -1)
        right_ref_b = self.robot.data.root_pos_w.new_tensor([0.0, -ref_y_offset, 0.0]).unsqueeze(0).expand(self.num_envs, -1)
        root_quat_w = self.robot.data.root_quat_w
        root_pos_w = self.robot.data.root_pos_w
        left_ref_w = root_pos_w + quat_apply(root_quat_w, left_ref_b)
        right_ref_w = root_pos_w + quat_apply(root_quat_w, right_ref_b)

        left_end_w = self.robot.data.body_pos_w[:, self._left_wheel_link_idx[0]]
        right_end_w = self.robot.data.body_pos_w[:, self._right_wheel_link_idx[0]]

        gravity_vec = left_ref_w.new_tensor(
            getattr(getattr(self.cfg, "sim", None), "gravity", (0.0, 0.0, -9.81))
        )
        gravity_dir = gravity_vec / torch.clamp(torch.norm(gravity_vec), min=1.0e-6)

        left_leg_vec = left_end_w - left_ref_w
        right_leg_vec = right_end_w - right_ref_w
        left_leg_dir = left_leg_vec / torch.clamp(torch.norm(left_leg_vec, dim=-1, keepdim=True), min=1.0e-6)
        right_leg_dir = right_leg_vec / torch.clamp(torch.norm(right_leg_vec, dim=-1, keepdim=True), min=1.0e-6)

        left_cos = torch.sum(left_leg_dir * gravity_dir.unsqueeze(0), dim=-1).clamp(-1.0, 1.0)
        right_cos = torch.sum(right_leg_dir * gravity_dir.unsqueeze(0), dim=-1).clamp(-1.0, 1.0)

        left_align_err_sq = torch.clamp(1.0 - left_cos * left_cos, min=0.0)
        right_align_err_sq = torch.clamp(1.0 - right_cos * right_cos, min=0.0)
        mean_align_err_sq = 0.5 * (left_align_err_sq + right_align_err_sq)

        if tolerance > 0.0:
            tol_sq = tolerance * tolerance
            mean_align_err_sq = torch.clamp(mean_align_err_sq - tol_sq, min=0.0)
        self._maybe_print_wheel_motor_z_axis_align_debug(
            left_leg_vec,
            right_leg_vec,
            left_cos,
            right_cos,
            mean_align_err_sq,
        )
        return mean_align_err_sq

    def _maybe_print_wheel_motor_z_axis_align_debug(
        self,
        left_ref_to_wheel_w: torch.Tensor,
        right_ref_to_wheel_w: torch.Tensor,
        left_cos_gravity: torch.Tensor,
        right_cos_gravity: torch.Tensor,
        align_err_sq: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.cfg, "play", False)):
            return
        if not bool(getattr(self.cfg, "play_wheel_motor_z_axis_align_debug", False)):
            return
        interval = int(getattr(self.cfg, "play_wheel_motor_z_axis_align_debug_interval", 50))
        if interval <= 0:
            return

        self._wheel_motor_z_axis_align_debug_counter += 1
        if self._wheel_motor_z_axis_align_debug_counter % interval != 0:
            return

        env_id = int(getattr(self.cfg, "play_wheel_motor_z_axis_align_debug_env_id", 0))
        env_id = max(0, min(env_id, self.num_envs - 1))

        def fmt_vec(vec: torch.Tensor) -> str:
            values = vec.detach().cpu().tolist()
            return "[" + ", ".join(f"{float(value):+.4f}" for value in values) + "]"

        left_vec = left_ref_to_wheel_w[env_id]
        right_vec = right_ref_to_wheel_w[env_id]
        left_len = torch.norm(left_vec).detach().cpu().item()
        right_len = torch.norm(right_vec).detach().cpu().item()
        step = int(getattr(self, "common_step_counter", self._wheel_motor_z_axis_align_debug_counter))
        print(
            "[WheelMotorZAxisAlignDebug] "
            f"step={step} env={env_id} "
            f"left_ref_to_wheel_w={fmt_vec(left_vec)} "
            f"right_ref_to_wheel_w={fmt_vec(right_vec)} "
            f"left_len={left_len:.4f} right_len={right_len:.4f} "
            f"left_cos_gravity={left_cos_gravity[env_id].item():+.4f} "
            f"right_cos_gravity={right_cos_gravity[env_id].item():+.4f} "
            f"err_sq={align_err_sq[env_id].item():.6f}"
        )

    def _get_body_material_for_link(self, body_name: str, env_id: int) -> torch.Tensor | None:
        """Return [static_friction, dynamic_friction, restitution] for one body link."""
        material = self._get_body_material_tensor()
        if material is None:
            return None

        self._build_material_mapping()
        mat_indices = self._body_to_material_indices.get(body_name, [])
        if not mat_indices:
            return None

        valid_indices = [int(idx) for idx in mat_indices if 0 <= int(idx) < material.shape[1]]
        if not valid_indices:
            return None

        values = material[env_id, valid_indices]
        if values.ndim == 2:
            values = values.mean(dim=0)
        return values

    def _get_wheel_contact_force_for_link(self, body_name: str, env_id: int) -> tuple[torch.Tensor | None, float | None]:
        """Return world-frame contact force [fx, fy, fz] and history peak norm for one wheel link."""
        contact_data = getattr(self.contact_sensor, "data", None)
        if contact_data is None:
            return None, None

        body_names = getattr(self.contact_sensor, "body_names", None)
        if not body_names or body_name not in body_names:
            return None, None

        sensor_idx = body_names.index(body_name)

        net_contact_forces_history = getattr(contact_data, "net_forces_w_history", None)
        if net_contact_forces_history is not None and sensor_idx < net_contact_forces_history.shape[2]:
            force_history = net_contact_forces_history[env_id, :, sensor_idx]
            force_w = force_history[0]
            peak_norm = float(torch.norm(force_history, dim=-1).max().item())
            return force_w, peak_norm

        net_forces_w = getattr(contact_data, "net_forces_w", None)
        if net_forces_w is not None and sensor_idx < net_forces_w.shape[1]:
            force_w = net_forces_w[env_id, sensor_idx]
            peak_norm = float(torch.norm(force_w).item())
            return force_w, peak_norm

        return None, None

    def _maybe_print_wheel_material_debug(self) -> None:
        """Print left/right wheel contact material and contact forces."""
        if not bool(getattr(self.cfg, "play_wheel_material_debug", False)):
            return
        interval = int(getattr(self.cfg, "play_wheel_material_debug_interval", 50))
        if interval <= 0:
            return

        self._wheel_material_debug_counter += 1
        if self._wheel_material_debug_counter % interval != 0:
            return

        env_id = int(getattr(self.cfg, "play_wheel_material_debug_env_id", 0))
        env_id = max(0, min(env_id, self.num_envs - 1))

        left_mat = self._get_body_material_for_link("left_wheel_link", env_id)
        right_mat = self._get_body_material_for_link("right_wheel_link", env_id)
        left_force_w, left_force_peak = self._get_wheel_contact_force_for_link("left_wheel_link", env_id)
        right_force_w, right_force_peak = self._get_wheel_contact_force_for_link("right_wheel_link", env_id)

        def fmt_mat(mat: torch.Tensor | None) -> str:
            if mat is None:
                return "N/A"
            values = mat.detach().cpu().tolist()
            if len(values) < 3:
                return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"
            return (
                f"static={float(values[0]):.4f} "
                f"dynamic={float(values[1]):.4f} "
                f"restitution={float(values[2]):.4f}"
            )

        def fmt_force(force_w: torch.Tensor | None, peak_norm: float | None) -> str:
            if force_w is None:
                return "N/A"
            values = force_w.detach().cpu().tolist()
            norm = float(torch.norm(force_w).item())
            peak_str = f" peak={peak_norm:.2f}" if peak_norm is not None else ""
            return (
                f"|F|={norm:.2f}{peak_str} "
                f"F_w=[{float(values[0]):+.2f}, {float(values[1]):+.2f}, {float(values[2]):+.2f}]"
            )

        step = int(getattr(self, "common_step_counter", self._wheel_material_debug_counter))
        print(
            "[WheelMaterialDebug] "
            f"step={step} env={env_id} "
            f"left_wheel_link material={fmt_mat(left_mat)} contact={fmt_force(left_force_w, left_force_peak)} "
            f"right_wheel_link material={fmt_mat(right_mat)} contact={fmt_force(right_force_w, right_force_peak)}"
        )

    def _get_scanner_ground_height(self, scanner: RayCaster | None, fallback_z: torch.Tensor) -> torch.Tensor:
        """Return per-env ground height estimated by a raycaster."""
        if scanner is None:
            return fallback_z

        ray_hits = getattr(scanner.data, "ray_hits_w", None)
        if ray_hits is None or ray_hits.ndim != 3 or ray_hits.shape[1] == 0:
            return fallback_z

        ray_hits_z = ray_hits[:, :, 2]
        valid_mask = torch.isfinite(ray_hits_z)
        env_any_valid = torch.any(valid_mask, dim=1)
        sum_z = torch.sum(ray_hits_z.masked_fill(~valid_mask, 0.0), dim=1)
        count_z = torch.sum(valid_mask.float(), dim=1)
        mean_z = sum_z / torch.clamp(count_z, min=1.0)
        return torch.where(env_any_valid, mean_z, fallback_z)

    def _get_wheel_body_height(self, body_indices: list[int]) -> torch.Tensor:
        """Return wheel-center world height for one wheel body."""
        if len(body_indices) == 0:
            return self.robot.data.root_pos_w[:, 2]
        return self.robot.data.body_pos_w[:, body_indices[0], 2]

    def _get_wheel_relative_ground_heights_raw(self) -> torch.Tensor:
        """Return [right_wheel, left_wheel] height above local ground."""
        if self._cached_wheel_relative_heights is not None:
            return self._cached_wheel_relative_heights

        current_ground_z = self._get_current_wheel_ground_z_raw()
        right_ground_z = current_ground_z[:, 0]
        left_ground_z = current_ground_z[:, 1]
        right_wheel_z = self._get_wheel_body_height(self._right_wheel_link_idx)
        left_wheel_z = self._get_wheel_body_height(self._left_wheel_link_idx)

        self._cached_wheel_relative_heights = torch.stack(
            [right_wheel_z - right_ground_z, left_wheel_z - left_ground_z], dim=-1
        )
        return self._cached_wheel_relative_heights

    def _get_current_wheel_ground_z_raw(self) -> torch.Tensor:
        """Return [right_wheel, left_wheel] ground height from wheel scanners."""
        fallback_ground_z = self.ground_z_est
        return torch.stack(
            [
                self._get_scanner_ground_height(self.right_wheel_height_scanner, fallback_ground_z),
                self._get_scanner_ground_height(self.left_wheel_height_scanner, fallback_ground_z),
            ],
            dim=-1,
        )

    def _update_height_reward_airborne_state(self) -> None:
        """Update runtime state machines that depend on the latest observations."""
        self.state_machine_manager.on_observation_step(self)

    def _get_wheel_forward_spatial_height_diffs_raw(self) -> torch.Tensor:
        """Return per-wheel height deltas between the forward probe and the current wheel ground scan."""
        if self._cached_wheel_forward_spatial_height_diffs is not None:
            return self._cached_wheel_forward_spatial_height_diffs
        query_points_w, hit_points_w = self._get_wheel_forward_scan_points()
        if query_points_w is None or hit_points_w is None:
            self._cached_wheel_forward_spatial_height_diffs = torch.zeros(
                self.num_envs, 0, dtype=torch.float, device=self.device
            )
            return self._cached_wheel_forward_spatial_height_diffs

        current_ground_z = self._get_current_wheel_ground_z_raw()
        forward_ground_z = hit_points_w[:, :, 2]
        self._cached_wheel_forward_spatial_height_diffs = forward_ground_z - current_ground_z
        return self._cached_wheel_forward_spatial_height_diffs

    def _get_wheel_forward_temporal_height_diffs_raw(self) -> torch.Tensor:
        """Return per-wheel forward-scan height deltas against the previous control step."""
        if self._cached_wheel_forward_temporal_height_diffs is not None:
            return self._cached_wheel_forward_temporal_height_diffs
        query_points_w, hit_points_w = self._get_wheel_forward_scan_points()
        if query_points_w is None or hit_points_w is None:
            self._cached_wheel_forward_temporal_height_diffs = torch.zeros(
                self.num_envs, 0, dtype=torch.float, device=self.device
            )
            return self._cached_wheel_forward_temporal_height_diffs

        current_forward_ground_z = hit_points_w[:, :, 2]
        if not hasattr(self, "wheel_forward_prev_ground_z"):
            self._cached_wheel_forward_temporal_height_diffs = torch.zeros_like(current_forward_ground_z)
            return self._cached_wheel_forward_temporal_height_diffs

        current_direction_sign = getattr(self, "_cached_wheel_forward_scan_direction_sign", None)
        if current_direction_sign is None:
            current_direction_sign = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self, "wheel_forward_prev_direction_sign"):
            self.wheel_forward_prev_direction_sign = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        direction_changed = current_direction_sign != self.wheel_forward_prev_direction_sign
        prev_valid = self.wheel_forward_prev_ground_z_valid.unsqueeze(-1)
        temporal_diffs = torch.where(
            prev_valid & (~direction_changed).unsqueeze(-1),
            current_forward_ground_z - self.wheel_forward_prev_ground_z,
            torch.zeros_like(current_forward_ground_z),
        )
        self.wheel_forward_prev_ground_z.copy_(current_forward_ground_z)
        self.wheel_forward_prev_direction_sign.copy_(current_direction_sign)
        self.wheel_forward_prev_ground_z_valid.fill_(True)
        self._cached_wheel_forward_temporal_height_diffs = temporal_diffs
        return self._cached_wheel_forward_temporal_height_diffs

    def _get_wheel_forward_stair_temporal_height_diffs_raw(self) -> torch.Tensor:
        """Return per-wheel short-range stair-scan height deltas against the previous control step."""
        if self._cached_wheel_forward_stair_temporal_height_diffs is not None:
            return self._cached_wheel_forward_stair_temporal_height_diffs
        query_points_w, hit_points_w = self._get_wheel_forward_stair_scan_points()
        if query_points_w is None or hit_points_w is None:
            self._cached_wheel_forward_stair_temporal_height_diffs = torch.zeros(
                self.num_envs, 0, dtype=torch.float, device=self.device
            )
            return self._cached_wheel_forward_stair_temporal_height_diffs

        current_forward_ground_z = hit_points_w[:, :, 2]
        if not hasattr(self, "wheel_forward_stair_prev_ground_z"):
            self._cached_wheel_forward_stair_temporal_height_diffs = torch.zeros_like(
                current_forward_ground_z
            )
            return self._cached_wheel_forward_stair_temporal_height_diffs

        current_direction_sign = getattr(self, "_cached_wheel_forward_stair_scan_direction_sign", None)
        if current_direction_sign is None:
            current_direction_sign = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self, "wheel_forward_stair_prev_direction_sign"):
            self.wheel_forward_stair_prev_direction_sign = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        direction_changed = current_direction_sign != self.wheel_forward_stair_prev_direction_sign
        prev_valid = self.wheel_forward_stair_prev_ground_z_valid.unsqueeze(-1)
        temporal_diffs = torch.where(
            prev_valid & (~direction_changed).unsqueeze(-1),
            current_forward_ground_z - self.wheel_forward_stair_prev_ground_z,
            torch.zeros_like(current_forward_ground_z),
        )
        self.wheel_forward_stair_prev_ground_z.copy_(current_forward_ground_z)
        self.wheel_forward_stair_prev_direction_sign.copy_(current_direction_sign)
        self.wheel_forward_stair_prev_ground_z_valid.fill_(True)
        self._cached_wheel_forward_stair_temporal_height_diffs = temporal_diffs
        return self._cached_wheel_forward_stair_temporal_height_diffs

    def _get_height_reward_reference_height(
        self,
        relative_obs_height: torch.Tensor,
        wheel_height_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return the height signal used by track_height rewards."""
        return self.state_machine_manager.get_height_reward_reference_height(
            self, relative_obs_height, wheel_height_w
        )

    def _init_special_height_wave_state(self) -> None:
        """Initialize per-env state for dynamic height special modes."""
        self.special_height_wave_mode_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.special_height_wave_phase = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.special_height_wave_start_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.special_height_wave_mean = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.special_height_wave_amplitude = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.special_height_wave_frequency_hz = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.height_command_special_mode_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.height_command_special_phase = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.height_command_special_start_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.height_command_special_mean = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.height_command_special_amplitude = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.height_command_special_frequency_hz = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _cfg_value(self, cfg, key: str, default=None):
        if isinstance(cfg, Mapping):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _height_command_special_modes_cfg(self) -> Mapping:
        cfg = getattr(self.cfg, "height_command_special_modes_cfg", {}) or {}
        return cfg if isinstance(cfg, Mapping) else {}

    def _get_height_command_profile_cfg(self, mode_cfg):
        step_cfg = self._cfg_value(mode_cfg, "height_step", None)
        if step_cfg is not None:
            return "step", step_cfg
        wave_cfg = self._cfg_value(mode_cfg, "height_wave", None)
        if wave_cfg is not None:
            return "wave", wave_cfg
        return None, None

    def _height_command_special_mode_entries(self) -> list[tuple[int, str, object]]:
        cfg = self._height_command_special_modes_cfg()
        if not bool(cfg.get("enabled", False)):
            return []
        raw_modes = cfg.get("modes", ())
        if raw_modes is None:
            return []
        if isinstance(raw_modes, Mapping):
            items = tuple(raw_modes.items())
        else:
            items = tuple((f"mode_{idx}", mode) for idx, mode in enumerate(raw_modes))
        entries: list[tuple[int, str, object]] = []
        for mode_idx, (name, mode_cfg) in enumerate(items):
            _, profile_cfg = self._get_height_command_profile_cfg(mode_cfg)
            if profile_cfg is None:
                continue
            entries.append((mode_idx, str(name), mode_cfg))
        return entries

    def _is_height_command_special_mode_active(self, mode_cfg) -> bool:
        rel_envs = float(self._cfg_value(mode_cfg, "rel_envs", 0.0))
        if rel_envs <= 0.0:
            return False
        iteration = self._get_reset_training_iteration()
        start = int(self._cfg_value(mode_cfg, "iteration_start", 0))
        end = int(self._cfg_value(mode_cfg, "iteration_end", -1))
        return iteration >= start and (end < 0 or iteration < end)

    def _sample_height_command_special_phase(self, mode_cfg, count: int) -> torch.Tensor:
        _, profile_cfg = self._get_height_command_profile_cfg(mode_cfg)
        phase = float(self._cfg_value(profile_cfg, "phase", 0.0))
        if not bool(self._cfg_value(profile_cfg, "random_phase", False)):
            return torch.full((count,), phase, dtype=torch.float, device=self.device)
        phase_range = self._cfg_value(profile_cfg, "phase_range", None)
        if phase_range is None:
            phase_low, phase_high = 0.0, 2.0 * math.pi
        else:
            phase_low = float(phase_range[0])
            phase_high = float(phase_range[1])
        sampled = torch.empty(count, dtype=torch.float, device=self.device).uniform_(
            min(phase_low, phase_high), max(phase_low, phase_high)
        )
        return phase + sampled

    def _sample_height_wave_param(
        self,
        wave_cfg,
        value_key: str,
        range_key: str,
        default: float,
        count: int,
    ) -> torch.Tensor:
        value_range = self._cfg_value(wave_cfg, range_key, None)
        if value_range is None:
            value = float(self._cfg_value(wave_cfg, value_key, default))
            return torch.full((count,), value, dtype=torch.float, device=self.device)
        low = float(value_range[0])
        high = float(value_range[1])
        if low == high:
            return torch.full((count,), low, dtype=torch.float, device=self.device)
        return torch.empty(count, dtype=torch.float, device=self.device).uniform_(
            min(low, high), max(low, high)
        )

    def _resample_height_command_special_modes(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ) -> None:
        """Independently sample height-only special modes for selected envs."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        if not hasattr(self, "height_command_special_mode_id"):
            self._init_special_height_wave_state()

        self.height_command_special_mode_id[env_ids_t] = -1
        entries = [
            (mode_idx, name, mode_cfg)
            for mode_idx, name, mode_cfg in self._height_command_special_mode_entries()
            if self._is_height_command_special_mode_active(mode_cfg)
        ]
        if not entries:
            return

        candidate_ids = env_ids_t
        cfg = self._height_command_special_modes_cfg()
        min_episode_time = float(cfg.get("min_episode_time", 0.0))
        if min_episode_time > 0.0:
            episode_time = self.episode_length_buf[candidate_ids].to(dtype=torch.float) * self.step_dt
            candidate_ids = candidate_ids[episode_time >= min_episode_time]
        if candidate_ids.numel() == 0:
            return

        order = torch.randperm(len(entries), device=self.device).tolist()
        shuffled = [entries[i] for i in order]
        r = torch.rand(candidate_ids.numel(), device=self.device)
        cumulative = 0.0
        for mode_idx, _name, mode_cfg in shuffled:
            rel_envs = float(self._cfg_value(mode_cfg, "rel_envs", 0.0))
            low = cumulative
            high = cumulative + rel_envs
            slot_mask = (r >= low) & (r < high)
            if torch.any(slot_mask):
                mode_env_ids = candidate_ids[slot_mask]
                episode_time = self.episode_length_buf[mode_env_ids].to(dtype=torch.float) * self.step_dt
                self.height_command_special_mode_id[mode_env_ids] = mode_idx
                self.height_command_special_start_time[mode_env_ids] = episode_time
                self.height_command_special_phase[mode_env_ids] = self._sample_height_command_special_phase(
                    mode_cfg, int(mode_env_ids.numel())
                )
                _, profile_cfg = self._get_height_command_profile_cfg(mode_cfg)
                count = int(mode_env_ids.numel())
                self.height_command_special_mean[mode_env_ids] = self._sample_height_wave_param(
                    profile_cfg, "mean", "mean_range", getattr(self.cfg, "default_height_cmd", 0.0), count
                )
                self.height_command_special_amplitude[mode_env_ids] = self._sample_height_wave_param(
                    profile_cfg, "amplitude", "amplitude_range", 0.0, count
                )
                self.height_command_special_frequency_hz[mode_env_ids] = self._sample_height_wave_param(
                    profile_cfg, "frequency_hz", "frequency_range_hz", 0.0, count
                )
            cumulative = high

    def _apply_height_command_special_modes(self, height_cmd: torch.Tensor) -> torch.Tensor:
        """Apply height-only special modes without changing velocity commands."""
        if not hasattr(self, "height_command_special_mode_id"):
            self._init_special_height_wave_state()
        if not torch.any(self.height_command_special_mode_id >= 0):
            return height_cmd

        output = height_cmd.clone()
        for mode_idx, _name, mode_cfg in self._height_command_special_mode_entries():
            mode_mask = self.height_command_special_mode_id == mode_idx
            if not torch.any(mode_mask):
                continue
            mode_env_ids = mode_mask.nonzero(as_tuple=False).flatten()
            profile_type, profile_cfg = self._get_height_command_profile_cfg(mode_cfg)
            elapsed = (
                self.episode_length_buf[mode_env_ids].to(dtype=torch.float) * self.step_dt
                - self.height_command_special_start_time[mode_env_ids]
            )
            elapsed = torch.clamp(elapsed, min=0.0)
            mean = self.height_command_special_mean[mode_env_ids]
            amplitude = self.height_command_special_amplitude[mode_env_ids]
            frequency_hz = self.height_command_special_frequency_hz[mode_env_ids]
            phase = self.height_command_special_phase[mode_env_ids]
            phase_arg = (2.0 * math.pi * frequency_hz) * elapsed + phase
            if profile_type == "step":
                signal = torch.where(
                    torch.sin(phase_arg) >= 0.0,
                    torch.ones_like(phase_arg),
                    -torch.ones_like(phase_arg),
                )
            else:
                signal = torch.sin(phase_arg)
            wave = mean + amplitude * signal
            clamp_range = self._cfg_value(profile_cfg, "clamp_range", None)
            if clamp_range is not None:
                clamp_low = float(clamp_range[0])
                clamp_high = float(clamp_range[1])
                wave = torch.clamp(wave, min=min(clamp_low, clamp_high), max=max(clamp_low, clamp_high))
            output[mode_env_ids] = wave
        return output

    def _latch_special_height_wave(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Latch phase and start time for envs currently assigned to height-wave modes."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        if not hasattr(self, "special_height_wave_mode_id"):
            self._init_special_height_wave_state()

        self.special_height_wave_mode_id[env_ids_t] = -1
        command_generator = getattr(self, "command_generator", None)
        special_mode_id = getattr(command_generator, "special_mode_id", None)
        if command_generator is None or special_mode_id is None:
            return

        modes = tuple(getattr(getattr(command_generator, "cfg", None), "special_modes", ()) or ())
        if len(modes) == 0:
            return

        mode_ids = special_mode_id[env_ids_t]
        for mode_idx, mode_cfg in enumerate(modes):
            profile_type, profile_cfg = self._get_height_command_profile_cfg(mode_cfg)
            if profile_cfg is None:
                continue
            mode_mask = mode_ids == mode_idx
            if not torch.any(mode_mask):
                continue

            mode_env_ids = env_ids_t[mode_mask]
            episode_time = self.episode_length_buf[mode_env_ids].to(dtype=torch.float) * self.step_dt
            self.special_height_wave_mode_id[mode_env_ids] = mode_idx
            self.special_height_wave_start_time[mode_env_ids] = episode_time

            phase = float(self._cfg_value(profile_cfg, "phase", 0.0))
            if bool(self._cfg_value(profile_cfg, "random_phase", False)):
                phase_range = self._cfg_value(profile_cfg, "phase_range", None)
                if phase_range is None:
                    phase_low, phase_high = 0.0, 2.0 * math.pi
                else:
                    phase_low = float(phase_range[0])
                    phase_high = float(phase_range[1])
                low = min(phase_low, phase_high)
                high = max(phase_low, phase_high)
                sampled_phase = torch.empty(
                    mode_env_ids.numel(), dtype=torch.float, device=self.device
                ).uniform_(low, high)
                self.special_height_wave_phase[mode_env_ids] = phase + sampled_phase
            else:
                self.special_height_wave_phase[mode_env_ids] = phase
            count = int(mode_env_ids.numel())
            self.special_height_wave_mean[mode_env_ids] = self._sample_height_wave_param(
                profile_cfg, "mean", "mean_range", getattr(self.cfg, "default_height_cmd", 0.0), count
            )
            self.special_height_wave_amplitude[mode_env_ids] = self._sample_height_wave_param(
                profile_cfg, "amplitude", "amplitude_range", 0.0, count
            )
            self.special_height_wave_frequency_hz[mode_env_ids] = self._sample_height_wave_param(
                profile_cfg, "frequency_hz", "frequency_range_hz", 0.0, count
            )

    def _apply_special_height_wave(self, height_cmd: torch.Tensor) -> torch.Tensor:
        """Apply dynamic height commands for envs assigned to height-wave special modes."""
        command_generator = getattr(self, "command_generator", None)
        special_mode_id = getattr(command_generator, "special_mode_id", None)
        if command_generator is None or special_mode_id is None:
            return height_cmd

        modes = tuple(getattr(getattr(command_generator, "cfg", None), "special_modes", ()) or ())
        if len(modes) == 0:
            return height_cmd
        if not hasattr(self, "special_height_wave_mode_id"):
            self._init_special_height_wave_state()

        output = height_cmd.clone()
        for mode_idx, mode_cfg in enumerate(modes):
            profile_type, profile_cfg = self._get_height_command_profile_cfg(mode_cfg)
            if profile_cfg is None:
                continue

            mode_mask = special_mode_id == mode_idx
            if not torch.any(mode_mask):
                continue

            stale_mask = mode_mask & (self.special_height_wave_mode_id != mode_idx)
            if torch.any(stale_mask):
                self._latch_special_height_wave(stale_mask.nonzero(as_tuple=False).flatten())

            mode_env_ids = mode_mask.nonzero(as_tuple=False).flatten()
            elapsed = (
                self.episode_length_buf[mode_env_ids].to(dtype=torch.float) * self.step_dt
                - self.special_height_wave_start_time[mode_env_ids]
            )
            elapsed = torch.clamp(elapsed, min=0.0)

            mean = self.special_height_wave_mean[mode_env_ids]
            amplitude = self.special_height_wave_amplitude[mode_env_ids]
            frequency_hz = self.special_height_wave_frequency_hz[mode_env_ids]
            phase = self.special_height_wave_phase[mode_env_ids]
            phase_arg = (2.0 * math.pi * frequency_hz) * elapsed + phase
            if profile_type == "step":
                signal = torch.where(
                    torch.sin(phase_arg) >= 0.0,
                    torch.ones_like(phase_arg),
                    -torch.ones_like(phase_arg),
                )
            else:
                signal = torch.sin(phase_arg)
            wave = mean + amplitude * signal

            clamp_range = self._cfg_value(profile_cfg, "clamp_range", None)
            if clamp_range is not None:
                clamp_low = float(clamp_range[0])
                clamp_high = float(clamp_range[1])
                wave = torch.clamp(
                    wave,
                    min=min(clamp_low, clamp_high),
                    max=max(clamp_low, clamp_high),
                )

            output[mode_env_ids] = wave
        return output

    def _get_effective_height_cmd(self) -> torch.Tensor:
        """Return the runtime height command used by height rewards."""
        base_height_cmd = self._apply_special_height_wave(self.height_cmd)
        base_height_cmd = self._apply_height_command_special_modes(base_height_cmd)
        height_cmd = self.state_machine_manager.get_effective_height_cmd(self, base_height_cmd)
        return self._apply_predefined_reset_air_height_limits(height_cmd)

    def _get_observation_height_cmd(self) -> torch.Tensor:
        """Return the height command exposed in policy and critic observations."""
        return self._get_effective_height_cmd()

    def _as_env_ids_tensor(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """Normalize env ids into a 1-D tensor on the environment device."""
        if env_ids is None:
            return self.robot._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_axis_aligned_reset_heading_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the per-env episode-latched axis-aligned reset mask."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        if not hasattr(self, "_axis_aligned_reset_heading_mask"):
            return torch.full(
                (env_ids_t.numel(),),
                fill_value=bool(getattr(self.cfg, "reset_heading_axis_aligned_only", False)),
                dtype=torch.bool,
                device=self.device,
            )
        return self._axis_aligned_reset_heading_mask[env_ids_t]

    def _get_predefined_reset_air_disabled_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return whether predefined air reset modes are disabled for each env."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        return torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)

    def _get_predefined_reset_ground_disabled_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return whether predefined ground reset modes are disabled for each env."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        return torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)

    def _update_axis_aligned_reset_heading_mask(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        use_env_origins: bool = False,
    ) -> None:
        """Latch the axis-aligned reset-heading rule for the current episode."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        if not hasattr(self, "_axis_aligned_reset_heading_mask"):
            self._axis_aligned_reset_heading_mask = torch.full(
                (self.num_envs,),
                fill_value=bool(getattr(self.cfg, "reset_heading_axis_aligned_only", False)),
                dtype=torch.bool,
                device=self.device,
            )
        self._axis_aligned_reset_heading_mask[env_ids_t] = bool(
            getattr(self.cfg, "reset_heading_axis_aligned_only", False)
        )

    def _sample_height_command(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        use_env_origins: bool = False,
    ) -> torch.Tensor:
        """Sample height commands for the provided environments."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.float, device=self.device)

        height_range = getattr(self.cfg, "height_range", None)
        if height_range is None or len(height_range) < 2:
            default_height = float(getattr(self.cfg, "default_height_cmd", 0.0))
            height_cmd = torch.full((env_ids_t.numel(),), default_height, dtype=torch.float, device=self.device)
            height_cmd = self._apply_special_mode_height_ranges(env_ids_t, height_cmd)
            return self._apply_jump_takeoff_permission_height_range(env_ids_t, height_cmd)

        h_min = float(height_range[0])
        h_max = float(height_range[1])
        if h_min == h_max:
            height_cmd = torch.full((env_ids_t.numel(),), h_min, dtype=torch.float, device=self.device)
        else:
            height_cmd = torch.empty(env_ids_t.numel(), dtype=torch.float, device=self.device).uniform_(h_min, h_max)
        height_cmd = self._apply_special_mode_height_ranges(env_ids_t, height_cmd)
        return self._apply_jump_takeoff_permission_height_range(env_ids_t, height_cmd)

    def _get_jump_takeoff_permission_cfg(self) -> dict:
        return dict(getattr(self.cfg, "jump_takeoff_permission_cfg", {}) or {})

    def _is_jump_takeoff_permission_enabled(self) -> bool:
        return bool(self._get_jump_takeoff_permission_cfg().get("enabled", False))

    def _normalize_permission_range_spec(self, value) -> tuple[tuple[float, float], ...]:
        command_generator = getattr(self, "command_generator", None)
        normalize = getattr(command_generator, "_normalize_range_spec", None)
        if callable(normalize):
            return normalize(value)
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(
            isinstance(v, (float, int)) for v in value
        ):
            return ((float(value[0]), float(value[1])),)
        ranges = []
        for item in value:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(f"Invalid jump_takeoff_permission range entry: {item!r}")
            ranges.append((float(item[0]), float(item[1])))
        return tuple(ranges)

    def _sample_permission_range_spec(self, value, count: int) -> torch.Tensor:
        range_spec = self._normalize_permission_range_spec(value)
        command_generator = getattr(self, "command_generator", None)
        sample = getattr(command_generator, "_sample_range_spec", None)
        if callable(sample):
            return sample(range_spec, count, self.device)
        if count <= 0:
            return torch.zeros(0, dtype=torch.float, device=self.device)
        if len(range_spec) == 1:
            low, high = range_spec[0]
            if low == high:
                return torch.full((count,), low, dtype=torch.float, device=self.device)
            return torch.empty(count, dtype=torch.float, device=self.device).uniform_(low, high)
        segment_ids = torch.randint(len(range_spec), (count,), device=self.device)
        samples = torch.empty(count, dtype=torch.float, device=self.device)
        for segment_idx, (low, high) in enumerate(range_spec):
            mask = segment_ids == segment_idx
            if not torch.any(mask):
                continue
            if low == high:
                samples[mask] = low
            else:
                samples[mask] = torch.empty(int(mask.sum().item()), dtype=torch.float, device=self.device).uniform_(
                    low, high
                )
        return samples

    def _is_jump_takeoff_permission_iteration_active(self, cfg: dict) -> bool:
        start = int(cfg.get("iteration_start", 0))
        end = int(cfg.get("iteration_end", -1))
        get_iteration = getattr(self, "_get_training_iteration", None)
        if callable(get_iteration):
            iteration = int(get_iteration())
        else:
            steps_per_iteration = max(
                int(cfg.get("steps_per_iteration", getattr(self.cfg, "training_progress_steps_per_iteration", 24))),
                1,
            )
            iteration = max(int(getattr(self, "_training_iteration", 0)), 0)
            iteration += max(int(getattr(self, "common_step_counter", 0)), 0) // steps_per_iteration
        return iteration >= start and (end < 0 or iteration < end)

    def _resample_jump_takeoff_permission(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0 or not hasattr(self, "jump_takeoff_permission_mask"):
            return

        self.jump_takeoff_permission_mask[env_ids_t] = False
        self.jump_takeoff_permission_lin_vel_x[env_ids_t] = torch.nan
        self.jump_takeoff_permission_lin_vel_y[env_ids_t] = torch.nan
        self.jump_takeoff_permission_ang_vel_z[env_ids_t] = torch.nan

        cfg = self._get_jump_takeoff_permission_cfg()
        if not bool(cfg.get("enabled", False)) or not self._is_jump_takeoff_permission_iteration_active(cfg):
            return

        rel_envs = max(min(float(cfg.get("rel_envs", 0.0)), 1.0), 0.0)
        if rel_envs <= 0.0:
            return
        selected = torch.rand(env_ids_t.numel(), dtype=torch.float, device=self.device) < rel_envs
        if not torch.any(selected):
            return

        selected_env_ids = env_ids_t[selected]
        self.jump_takeoff_permission_mask[selected_env_ids] = True

        ranges = cfg.get("ranges", None)
        if ranges is None:
            return
        if hasattr(ranges, "lin_vel_x"):
            lin_vel_x_range = ranges.lin_vel_x
            lin_vel_y_range = ranges.lin_vel_y
            ang_vel_z_range = ranges.ang_vel_z
        else:
            lin_vel_x_range = ranges.get("lin_vel_x", None)
            lin_vel_y_range = ranges.get("lin_vel_y", None)
            ang_vel_z_range = ranges.get("ang_vel_z", None)
        count = int(selected_env_ids.numel())
        if lin_vel_x_range is not None:
            self.jump_takeoff_permission_lin_vel_x[selected_env_ids] = self._sample_permission_range_spec(
                lin_vel_x_range, count
            )
        if lin_vel_y_range is not None:
            self.jump_takeoff_permission_lin_vel_y[selected_env_ids] = self._sample_permission_range_spec(
                lin_vel_y_range, count
            )
        if ang_vel_z_range is not None:
            self.jump_takeoff_permission_ang_vel_z[selected_env_ids] = self._sample_permission_range_spec(
                ang_vel_z_range, count
            )

    def _apply_jump_takeoff_permission_height_range(
        self, env_ids: torch.Tensor, height_cmd: torch.Tensor
    ) -> torch.Tensor:
        if env_ids.numel() == 0 or not self._is_jump_takeoff_permission_enabled():
            return height_cmd
        height_range = self._get_jump_takeoff_permission_cfg().get("height_range", None)
        if height_range is None or not hasattr(self, "jump_takeoff_permission_mask"):
            return height_cmd
        mask = self.jump_takeoff_permission_mask[env_ids]
        if not torch.any(mask):
            return height_cmd
        height_cmd = height_cmd.clone()
        height_cmd[mask] = self._sample_permission_range_spec(height_range, int(mask.sum().item()))
        return height_cmd

    def _apply_jump_takeoff_permission_command(self) -> None:
        if not self._is_jump_takeoff_permission_enabled() or not hasattr(self, "jump_takeoff_permission_mask"):
            return
        mask = self.jump_takeoff_permission_mask
        if not torch.any(mask):
            return
        field_specs = (
            (0, self.jump_takeoff_permission_lin_vel_x),
            (1, self.jump_takeoff_permission_lin_vel_y),
            (2, self.jump_takeoff_permission_ang_vel_z),
        )
        for command_idx, values in field_specs:
            field_mask = mask & torch.isfinite(values)
            if torch.any(field_mask):
                self.command[field_mask, command_idx] = values[field_mask]

    def _clear_predefined_reset_ground_command_override(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0 or not hasattr(self, "predefined_reset_ground_command_override_until_time"):
            return
        self.predefined_reset_ground_command_override_until_time[env_ids_t] = 0.0
        self.predefined_reset_ground_restore_command[env_ids_t] = torch.nan
        self.predefined_reset_ground_sampled_command[env_ids_t] = torch.nan

    def _set_predefined_reset_ground_command_override(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        command_ranges: Mapping | None,
        duration_s: float,
    ) -> None:
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        self._clear_predefined_reset_ground_command_override(env_ids_t)
        if not isinstance(command_ranges, Mapping) or duration_s <= 0.0:
            return

        self.predefined_reset_ground_restore_command[env_ids_t] = self.command_generator.command[env_ids_t].detach().clone()
        count = int(env_ids_t.numel())
        for command_idx, key in (
            (0, "lin_vel_x"),
            (1, "lin_vel_y"),
            (2, "ang_vel_z"),
        ):
            range_spec = command_ranges.get(key, None)
            if range_spec is None:
                continue
            self.predefined_reset_ground_sampled_command[env_ids_t, command_idx] = (
                self._sample_permission_range_spec(range_spec, count)
            )

        episode_time = self.episode_length_buf[env_ids_t].to(dtype=torch.float) * self.step_dt
        self.predefined_reset_ground_command_override_until_time[env_ids_t] = episode_time + duration_s

    def _apply_predefined_reset_ground_command_override(self) -> None:
        if not hasattr(self, "predefined_reset_ground_command_override_until_time"):
            return
        current_time = self.episode_length_buf.to(dtype=torch.float) * self.step_dt
        pending_mask = self.predefined_reset_ground_command_override_until_time > 0.0
        active_mask = pending_mask & (current_time < self.predefined_reset_ground_command_override_until_time)
        expired_mask = pending_mask & ~active_mask
        if torch.any(expired_mask):
            expired_ids = expired_mask.nonzero(as_tuple=False).squeeze(-1)
            restore_command = self.predefined_reset_ground_restore_command[expired_ids]
            restore_valid = torch.isfinite(restore_command).all(dim=-1)
            restore_ids = expired_ids[restore_valid]
            if restore_ids.numel() > 0:
                restored_values = self.predefined_reset_ground_restore_command[restore_ids]
                self.command[restore_ids] = restored_values
                vel_command_b = getattr(self.command_generator, "vel_command_b", None)
                if vel_command_b is not None:
                    vel_command_b[restore_ids] = restored_values
            self._clear_predefined_reset_ground_command_override(expired_ids)
        if not torch.any(active_mask):
            return
        command = self.command.clone()
        for command_idx in range(3):
            field_mask = active_mask & torch.isfinite(self.predefined_reset_ground_sampled_command[:, command_idx])
            if torch.any(field_mask):
                command[field_mask, command_idx] = self.predefined_reset_ground_sampled_command[
                    field_mask, command_idx
                ]
        self.command = command

    def _get_special_mode_disable_jump_takeoff_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        command_generator = getattr(self, "command_generator", None)
        special_mode_id = getattr(command_generator, "special_mode_id", None)
        if command_generator is None or special_mode_id is None:
            return torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)

        modes = tuple(getattr(getattr(command_generator, "cfg", None), "special_modes", ()) or ())
        if len(modes) == 0:
            return torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)

        mode_ids = special_mode_id[env_ids_t]
        disabled = torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)
        for mode_idx, mode_cfg in enumerate(modes):
            if not bool(getattr(mode_cfg, "disable_jump_takeoff", False)):
                continue
            disabled |= mode_ids == mode_idx
        return disabled

    def _get_jump_takeoff_disabled_mask(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._get_special_mode_disable_jump_takeoff_mask(env_ids)

    def _apply_special_mode_height_ranges(
        self, env_ids: torch.Tensor, height_cmd: torch.Tensor
    ) -> torch.Tensor:
        """Override sampled height commands for envs assigned to special command modes."""
        command_generator = getattr(self, "command_generator", None)
        special_mode_id = getattr(command_generator, "special_mode_id", None)
        if command_generator is None or special_mode_id is None or env_ids.numel() == 0:
            return height_cmd

        modes = tuple(getattr(getattr(command_generator, "cfg", None), "special_modes", ()) or ())
        if len(modes) == 0:
            return height_cmd

        mode_ids = special_mode_id[env_ids]
        height_cmd = height_cmd.clone()
        normalize = getattr(command_generator, "_normalize_range_spec", None)
        sample = getattr(command_generator, "_sample_range_spec", None)
        for mode_idx, mode_cfg in enumerate(modes):
            height_range = getattr(mode_cfg, "height_range", None)
            if height_range is None:
                continue
            mask = mode_ids == mode_idx
            if not torch.any(mask):
                continue
            if callable(normalize) and callable(sample):
                height_cmd[mask] = sample(
                    normalize(height_range),
                    int(mask.sum().item()),
                    self.device,
                )
            else:
                h_min = float(height_range[0])
                h_max = float(height_range[1])
                height_cmd[mask] = torch.empty(
                    int(mask.sum().item()), dtype=torch.float, device=self.device
                ).uniform_(min(h_min, h_max), max(h_min, h_max))
        return height_cmd

    def _request_special_mode_jump_takeoff(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Request jump takeoff for envs whose current special mode enables it."""
        if env_ids is None or not hasattr(self, "jump_takeoff_request"):
            return
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return

        command_generator = getattr(self, "command_generator", None)
        special_mode_id = getattr(command_generator, "special_mode_id", None)
        if command_generator is None or special_mode_id is None:
            self.jump_takeoff_request[env_ids_t] = False
            return

        modes = tuple(getattr(getattr(command_generator, "cfg", None), "special_modes", ()) or ())
        mode_ids = special_mode_id[env_ids_t]
        request_mask = torch.zeros(env_ids_t.numel(), dtype=torch.bool, device=self.device)
        for mode_idx, mode_cfg in enumerate(modes):
            if not bool(getattr(mode_cfg, "jump_takeoff_enabled", False)):
                continue
            request_mask |= mode_ids == mode_idx

        self.jump_takeoff_request[env_ids_t[~request_mask]] = False
        if torch.any(request_mask):
            self.request_jump_takeoff(env_ids_t[request_mask])

    def _force_resample_commands(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """Resample command-generator outputs immediately for selected environments."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return env_ids_t
        self.command_generator.reset(env_ids_t)
        self._resample_jump_takeoff_permission(env_ids_t)
        self.command_counter = self.command_generator.command_counter.clone()
        self.command = self.command_generator.command.clone()
        self._latch_special_height_wave(env_ids_t)
        self._resample_height_command_special_modes(env_ids_t)
        self._apply_jump_takeoff_permission_command()
        return env_ids_t

    def _get_non_heading_axis_aligned_zero_mask(
        self, is_heading_env: torch.Tensor | None
    ) -> torch.Tensor:
        """Return envs whose yaw command should be zeroed by axis-aligned reset policy."""
        axis_aligned_mask = self._get_axis_aligned_reset_heading_mask()
        if not torch.any(axis_aligned_mask) or is_heading_env is None:
            return torch.zeros_like(axis_aligned_mask)
        return (~is_heading_env) & axis_aligned_mask

    def _apply_axis_aligned_reset_heading(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Snap reset yaw to one of the configured axis-aligned headings."""
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        enabled_mask = self._get_axis_aligned_reset_heading_mask(env_ids_t)
        if enabled_mask.numel() == 0 or not torch.any(enabled_mask):
            return
        env_ids_t = env_ids_t[enabled_mask]

        yaw_candidates_deg = tuple(getattr(self.cfg, "reset_heading_axis_aligned_candidates_deg", (0.0, 90.0, 180.0, -90.0)))
        if len(yaw_candidates_deg) == 0:
            return

        root_state = self.robot.data.root_state_w[env_ids_t].clone()
        roll, pitch, _ = euler_xyz_from_quat(root_state[:, 3:7])
        yaw_candidates = torch.tensor(yaw_candidates_deg, dtype=torch.float, device=self.device) * (torch.pi / 180.0)
        sampled_idx = torch.randint(0, yaw_candidates.numel(), (root_state.shape[0],), device=self.device)
        sampled_yaw = yaw_candidates[sampled_idx]
        root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, sampled_yaw)
        self.robot.write_root_state_to_sim(root_state, env_ids_t)

    def _record_reset_heading_target(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Cache the reset yaw as the per-episode target heading."""
        if env_ids is None or len(env_ids) == 0:
            return
        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
        self.reset_heading_target[env_ids_t] = self.robot.data.heading_w[env_ids_t]

    def _sync_heading_command_target_to_reset_heading(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Align heading-command targets with the reset yaw for selected envs."""
        if env_ids is None or len(env_ids) == 0:
            return

        env_ids_t = self._as_env_ids_tensor(env_ids)
        axis_aligned_mask = self._get_axis_aligned_reset_heading_mask(env_ids_t)
        if not torch.any(axis_aligned_mask):
            return
        env_ids_t = env_ids_t[axis_aligned_mask]

        command_generator = getattr(self, "command_generator", None)
        heading_target = getattr(command_generator, "heading_target", None)
        if heading_target is None:
            return

        heading_target[env_ids_t] = self.reset_heading_target[env_ids_t]

        if bool(getattr(getattr(command_generator, "cfg", None), "heading_command", False)):
            update_command = getattr(command_generator, "_update_command", None)
            if callable(update_command):
                update_command()

    def _get_height_reward_target_height(self) -> torch.Tensor:
        """Return the target height used by track_height rewards."""
        target_height = self._get_effective_height_cmd()
        airborne_cfg = getattr(self.cfg, "airborne_state_machine_cfg", {})
        wheel_radius = float(airborne_cfg.get("enter", {}).get("wheel_radius", 0.05))
        if self._use_leg_length_height():
            target_height = torch.clamp(target_height - wheel_radius, min=0.0)
        target_height = self.state_machine_manager.get_height_reward_target_height(
            self, target_height
        )
        if self._use_leg_length_height():
            target_height = torch.clamp(target_height, min=0.0)
        return target_height

    def _get_task_flag_obs_raw(self) -> torch.Tensor:
        """Return task flag placeholders appended next to command observations."""
        task_flag_dim = int(getattr(self.cfg, "task_flag_obs_dim", 0))
        if not bool(getattr(self.cfg, "task_flag_obs_enabled", False)) or task_flag_dim <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        return torch.zeros(self.num_envs, task_flag_dim, dtype=torch.float, device=self.device)

    def _get_policy_task_flag_obs_raw(self, task_flag_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Return the task flag block visible to policy observations."""
        return self._get_task_flag_obs_raw() if task_flag_obs is None else task_flag_obs

    def _get_policy_extra_obs_blocks(self) -> dict[str, torch.Tensor]:
        """Return named policy observation blocks appended after actions."""
        extra_obs = self._get_ctrl_mode_obs_raw()
        if extra_obs.shape[-1] == 0:
            return {}
        return {"ctrl_mode_obs": extra_obs}

    def _get_critic_extra_obs_blocks(self, root_quat_inv: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named critic observation blocks appended before privileged extras."""
        extra_obs = self._get_ctrl_mode_obs_raw()
        if extra_obs.shape[-1] == 0:
            return {}
        return {"ctrl_mode_obs": extra_obs}

    def _get_ctrl_mode_obs_raw(self) -> torch.Tensor:
        enabled = bool(
            getattr(
                self.cfg,
                "ctrl_mode_obs_enabled",
                getattr(self.cfg, "jump_takeoff_extra_obs_enabled", False),
            )
        )
        if not enabled:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        if not hasattr(self, "jump_takeoff_phase"):
            return torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device)

        phase = self.jump_takeoff_phase
        jump_state = phase != 0
        airborne_state = getattr(
            self,
            "height_reward_airborne_state",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        stair_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        slope_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        recover_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        normal_state = ~(jump_state | recover_state | stair_state | slope_state)

        one_hot = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device)
        one_hot[:, 0] = normal_state.float()
        one_hot[:, 1] = stair_state.float()
        one_hot[:, 2] = slope_state.float()
        one_hot[:, 3] = recover_state.float()
        one_hot[:, 4] = jump_state.float()

        jump_height_target = torch.where(
            jump_state,
            getattr(self, "jump_takeoff_ref_target_height", self.jump_takeoff_height_cmd),
            torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        )
        airborne_duration = getattr(
            self,
            "airborne_current_duration",
            torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        )
        if bool(getattr(self.cfg, "jump_takeoff_state_machine_cfg", {}).get("trajectory", {}).get("enabled", False)):
            jump_aux = getattr(self, "jump_takeoff_ref_phase", self.jump_takeoff_phase_time)
        else:
            jump_aux = self.jump_takeoff_phase_time
        jump_phase_time = torch.where(phase != 0, jump_aux, airborne_duration)
        jump_phase_time = torch.where(
            jump_state,
            jump_phase_time,
            torch.zeros_like(jump_phase_time),
        )
        return torch.cat(
            [
                one_hot,
                jump_height_target.unsqueeze(-1),
                jump_phase_time.unsqueeze(-1),
            ],
            dim=-1,
        )

    def _get_jump_takeoff_extra_obs_raw(self) -> torch.Tensor:
        """Backward-compatible alias for older configs/tools."""
        return self._get_ctrl_mode_obs_raw()

    def _invalidate_step_caches(self) -> None:
        """Invalidate tensors that are identical within one control step."""
        self._cached_ground_height_valid = False
        self._cached_wheel_kinematics_valid = False
        self._cached_wheel_relative_heights = None
        self._cached_root_quat_inv = None
        self._cached_wheel_pos_b = None
        self._cached_wheel_pos_heading_b = None
        self._cached_wheel_lin_vel_b = None
        self._cached_wheel_lin_vel_heading_b = None
        self._cached_wheel_forward_query_points = None
        self._cached_wheel_forward_hit_points = None
        self._cached_wheel_forward_scan_direction_sign = None
        self._cached_wheel_forward_spatial_height_diffs = None
        self._cached_wheel_forward_temporal_height_diffs = None
        self._cached_wheel_forward_stair_query_points = None
        self._cached_wheel_forward_stair_hit_points = None
        self._cached_wheel_forward_stair_scan_direction_sign = None
        self._cached_wheel_forward_stair_temporal_height_diffs = None

    def _get_root_quat_inv_and_wheel_pos_b(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute wheel kinematics in body and yaw-level body-following frames once per step.

        ``wheel_pos_b`` uses the full body frame and keeps the legacy y=0
        planar projection. ``wheel_pos_heading_b`` uses a frame whose origin
        and yaw follow the base, while roll/pitch are removed so its xy plane
        is parallel to the world horizontal plane. The velocity tensors follow
        the same frame distinction.
        """
        if self._cached_wheel_kinematics_valid:
            return (
                self._cached_root_quat_inv,
                self._cached_wheel_pos_b,
                self._cached_wheel_pos_heading_b,
                self._cached_wheel_lin_vel_b,
                self._cached_wheel_lin_vel_heading_b,
            )

        root_quat_inv = quat_inv(self.robot.data.root_quat_w)
        wheel_rel_pos_w = (
            self.robot.data.body_pos_w[:, self._wheel_link_idx]
            - self.robot.data.root_pos_w.unsqueeze(1).expand(-1, self._wheel_link_count, -1)
        )
        wheel_pos_b = quat_apply(
            root_quat_inv.unsqueeze(1).expand(-1, self._wheel_link_count, -1),
            wheel_rel_pos_w,
        )
        wheel_pos_b[:, :, 1].fill_(0.0)
        wheel_rel_lin_vel_w = (
            self.robot.data.body_lin_vel_w[:, self._wheel_link_idx]
            - self.robot.data.root_lin_vel_w.unsqueeze(1).expand(-1, self._wheel_link_count, -1)
        )
        wheel_lin_vel_b = quat_apply(
            root_quat_inv.unsqueeze(1).expand(-1, self._wheel_link_count, -1),
            wheel_rel_lin_vel_w,
        )
        wheel_lin_vel_b[:, :, 1].fill_(0.0)
        zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        heading_quat_inv = quat_inv(quat_from_euler_xyz(zeros, zeros, yaw))
        wheel_pos_heading_b = quat_apply(
            heading_quat_inv.unsqueeze(1).expand(-1, self._wheel_link_count, -1),
            wheel_rel_pos_w,
        )
        wheel_pos_heading_b[:, :, 1].fill_(0.0)
        wheel_lin_vel_heading_b = quat_apply(
            heading_quat_inv.unsqueeze(1).expand(-1, self._wheel_link_count, -1),
            wheel_rel_lin_vel_w,
        )
        wheel_lin_vel_heading_b[:, :, 1].fill_(0.0)
        self._cached_root_quat_inv = root_quat_inv
        self._cached_wheel_pos_b = wheel_pos_b
        self._cached_wheel_pos_heading_b = wheel_pos_heading_b
        self._cached_wheel_lin_vel_b = wheel_lin_vel_b
        self._cached_wheel_lin_vel_heading_b = wheel_lin_vel_heading_b
        self._cached_wheel_kinematics_valid = True
        return root_quat_inv, wheel_pos_b, wheel_pos_heading_b, wheel_lin_vel_b, wheel_lin_vel_heading_b

    def _build_state_machine_manager(self) -> WheelbipeStateMachineManager:
        """Create the shared wheelbipe runtime state-machine stack.

        Subclasses/tasks can override this to append or replace machines while
        keeping the common hook surface unchanged.
        """
        return WheelbipeStateMachineManager.from_env(self)

    def request_jump_takeoff(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Set the jump-takeoff request flag for selected environments."""
        if not hasattr(self, "jump_takeoff_request"):
            return
        if env_ids is None:
            self.jump_takeoff_request[:] = True
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.jump_takeoff_request[env_ids] = True

    def _get_left_right_leg_joint_pair_indices(self) -> tuple[list[int], list[int]]:
        """Return matching left/right leg joint indices for symmetry rewards."""
        ordered_names = tuple(getattr(self.cfg, "ordered_leg_joint_names", ()))
        if not ordered_names:
            ordered_names = tuple(getattr(self, "_legs_act_idx_name", ()))

        left_by_key: dict[str, str] = {}
        right_by_key: dict[str, str] = {}
        for name in ordered_names:
            if name.startswith("left_"):
                left_by_key[name[len("left_") :]] = name
            elif name.startswith("right_"):
                right_by_key[name[len("right_") :]] = name

        left_indices: list[int] = []
        right_indices: list[int] = []
        active_leg_idx = {int(idx) for idx in getattr(self, "_legs_act_idx", [])}
        for key in sorted(set(left_by_key) & set(right_by_key)):
            left_idx, _ = self.robot.find_joints(left_by_key[key])
            right_idx, _ = self.robot.find_joints(right_by_key[key])
            if left_idx and right_idx and int(left_idx[0]) in active_leg_idx and int(right_idx[0]) in active_leg_idx:
                left_indices.append(int(left_idx[0]))
                right_indices.append(int(right_idx[0]))
        return left_indices, right_indices

    def __init__(self, cfg: Wheelbipe25v3FlatEnvCfg | Wheelbipe25v3ReduceSpringFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # initial actions buffer
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self._before_previous_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        # try:
        #     self.obs = torch.zeros(
        #         self.num_envs, self.cfg.observation_space, device=self.device
        #     )
        # except:
        #     self.obs = torch.zeros(
        #         self.num_envs, self.cfg.observation_space['policy'], device=self.device
        #     )
        self.obs = torch.zeros(
            self.num_envs, self.cfg.num_single_obs, device=self.device
        )
        self.prev_obs = torch.zeros(
            self.num_envs, self.cfg.num_single_obs, device=self.device
        )
        if self.cfg.state_space:
            try:
                self.priv = torch.zeros(
                    self.num_envs, self.cfg.state_space, device=self.device
                )
                self.prev_priv = torch.zeros(
                    self.num_envs, self.cfg.state_space, device=self.device
                )
            except:
                    self.priv = torch.zeros(
                        self.num_envs, self.cfg.state_space['critic'], device=self.device
                    )
                    self.prev_priv = torch.zeros(
                        self.num_envs, self.cfg.state_space['critic'], device=self.device
                    )
        num_costs = int(getattr(self.cfg, "num_costs", 0) or 0)
        self.cost_k_values = torch.ones(num_costs, dtype=torch.float, device=self.device)
        cost_k_initial = getattr(self.cfg, "np3o_cost_k_initial", None)
        if cost_k_initial is not None and num_costs > 0:
            self.cost_k_values = torch.as_tensor(cost_k_initial, dtype=torch.float, device=self.device)
            if self.cost_k_values.numel() == 1:
                self.cost_k_values = self.cost_k_values.repeat(num_costs)
        cost_d_values = getattr(self.cfg, "np3o_cost_d_values", None)
        if cost_d_values is None:
            cost_d_values = [0.0] * num_costs
        self.cost_d_values_tensor = torch.as_tensor(cost_d_values, dtype=torch.float, device=self.device)
        if self.cost_d_values_tensor.numel() == 1 and num_costs > 1:
            self.cost_d_values_tensor = self.cost_d_values_tensor.repeat(num_costs)
        self.use_action_low_pass_filter = bool(getattr(self.cfg, "use_action_low_pass_filter", False))
        self.action_low_pass_prev_weight = float(getattr(self.cfg, "action_low_pass_prev_weight", 0.2))
        self.action_low_pass_curr_weight = float(getattr(self.cfg, "action_low_pass_curr_weight", 0.8))
        if self.use_action_low_pass_filter:
            print(
                "[ActionFilter] 一阶低通滤波已启用: "
                f"prev={self.action_low_pass_prev_weight}, curr={self.action_low_pass_curr_weight}"
            )
        # self._prev_vel = torch.zeros(self.num_envs, 3, device=self.device)
        # self._prev_height = torch.zeros(self.num_envs, 1, device=self.device)

        self.leg_action_scale = self.cfg.leg_action_scale
        # 轮速控制模式开关
        self.use_wheel_vel_control = getattr(self.cfg, 'use_wheel_vel_control', False)
        if self.use_wheel_vel_control:
            self.wheel_action_scale = getattr(self.cfg, 'wheel_vel_action_scale', 10.0)
            self.max_wheel_vel = getattr(self.cfg, 'max_wheel_vel', 60.0)*1.5
            print(f'[WheelCtrl] 轮速控制模式已启用: scale={self.wheel_action_scale}, max_vel={self.max_wheel_vel} rad/s')
        else:
            self.wheel_action_scale = self.cfg.wheel_action_scale
            print(f'[WheelCtrl] 力矩控制模式（默认）: scale={self.wheel_action_scale}')
        self.leg_actions = None
        # self.leg_actions = torch.zeros(self.num_envs,4,dtype=torch.float,device=self.device)
        self.wheel_actions = None

        # initial joint idx
        self._legs_act_idx, self._legs_act_idx_name = self.robot.find_joints(self.cfg.legs_act_name)
        self._legs_inact_idx, _ = self.robot.find_joints(self.cfg.legs_inact_name)
        self._wheel_idx, _ =  self.robot.find_joints(self.cfg.wheel_name)
        self._actuate_idx = self._legs_act_idx + self._wheel_idx
        self._legs_front_idx, _ = self.robot.find_joints(".*_front1_joint")
        self._legs_rear_idx, _ = self.robot.find_joints(".*_rear1_joint")
        self._front1_joint_idx, _ = self.robot.find_joints(".*_front1_joint")
        self._rear1_joint_idx, _ = self.robot.find_joints(".*_rear1_joint")
        self._front2_joint_idx, _ = self.robot.find_joints(".*_front2_joint")
        self._front3_joint_idx, _ = self.robot.find_joints(".*_front3_joint")
        self._front4_joint_idx, _ = self.robot.find_joints(".*_front4_joint")
        self._rear2_joint_idx, _ = self.robot.find_joints(".*_rear2_joint")
        self.reorder_reset_joint_idx = self._front1_joint_idx+self._rear1_joint_idx+self._front2_joint_idx+self._front3_joint_idx+self._front4_joint_idx+self._rear2_joint_idx
        self._deviation_joint_idx = self._front1_joint_idx+self._rear1_joint_idx
        self._left_right_leg_joint_pair_idx = self._get_left_right_leg_joint_pair_indices()
        self._left_front1_leg_act_local_idx = None
        self._left_rear1_leg_act_local_idx = None
        for local_idx, joint_name in enumerate(self._legs_act_idx_name):
            if "left_" in joint_name and "front1_joint" in joint_name:
                self._left_front1_leg_act_local_idx = local_idx
            elif "left_" in joint_name and "rear1_joint" in joint_name:
                self._left_rear1_leg_act_local_idx = local_idx
        self._previous_applied_torque = torch.zeros_like(self.robot.data.applied_torque)
        self._before_previous_applied_torque = torch.zeros_like(self.robot.data.applied_torque)

        # 弹簧系统开关（NS版本设为 False）
        self.use_spring = getattr(self.cfg, 'use_spring', True)  # 默认启用弹簧（向后兼容）

        # prismatic spring simulate
        if self.use_spring:
            try:
                self._spring_idx, _ = self.robot.find_joints(self.cfg.spring_name)
            except:
                self._spring_idx = None
            # virtual spring simulate（calculate force according to tf）
            try:
                self._upper_spring_link_idx, _ = self.robot.find_bodies(".*_spring1_virtual_link")
                self._lower_spring_link_idx, _ = self.robot.find_bodies(".*_spring_force_link")
            except:
                self._upper_spring_link_idx = None
                self._lower_spring_link_idx = None
        else:
            # NS版本：完全跳过弹簧相关初始化
            self._spring_idx = None
            self._upper_spring_link_idx = None
            self._lower_spring_link_idx = None
            print("[NS Version] Spring system disabled.")

        try:
            self._left_wheel_link_idx, _ = self.robot.find_bodies("left_wheel_link")
        except Exception:
            self._left_wheel_link_idx = []
        try:
            self._right_wheel_link_idx, _ = self.robot.find_bodies("right_wheel_link")
        except Exception:
            self._right_wheel_link_idx = []

        # initial link idx
        self._wheel_link_idx,_ = self.robot.find_bodies(".*_wheel_link")
        try:
            self._guide_link_idx, _ = self.robot.find_bodies(".*_guide_link")
        except Exception:
            self._guide_link_idx = []
        self._undesired_contact_link_idx = self._find_contact_sensor_indices([
            "base_link", ".*_rear1_link", ".*_rear2_link",
            ".*_front1_link", ".*_front2_link", ".*_front3_link", ".*_front4_link"
        ])
        self._desired_contact_link_idx = self._find_contact_sensor_indices([".*_wheel_link"])
        self._reset_contact_link_idx = self._find_contact_sensor_indices(["base_link"])

        # initial material index
        # material shape mapping & cached randomize params (built once)
        self._body_to_material_indices: dict[str, list[int]] = {}
        self._material_body_names: list[str] = []

        self._randomed_friction_link_idx = self._get_material_indices([
            ".*_wheel_link"
        ])

        # initial static index
        self._init_static_index_layouts()

        # X/Y linear velocity and yaw angular velocity commands
        self.command_generator = self.cfg.commands.class_type(cfg=self.cfg.commands, env=self)
        self.command_counter = self.command_generator.command_counter.clone()
        self.command = self.command_generator.command.clone()

        # height command
        self.height_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.height_cmd.fill_(self.cfg.default_height_cmd)
        self._init_special_height_wave_state()

        # initial obs buffers
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_zero_torque = torch.zeros_like(self.joint_pos)
        self.default_joint_pos = self.robot.data.default_joint_pos
        self.finish_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.finish_init_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.start_reset = self.finish_init.clone()
        self.predefined_reset_ground_zero_torque_until_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.predefined_reset_ground_command_override_until_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.predefined_reset_ground_restore_command = torch.full(
            (self.num_envs, 3), torch.nan, dtype=torch.float, device=self.device
        )
        self.predefined_reset_ground_sampled_command = torch.full(
            (self.num_envs, 3), torch.nan, dtype=torch.float, device=self.device
        )
        # 弹簧力初始化（仅当启用弹簧时）
        if self.use_spring and self._spring_idx is not None:
            self.spring_force = torch.zeros(self.num_envs,len(self._spring_idx),dtype=torch.float,device=self.device)
            self.spring_force.fill_(self.cfg.spring_settings['constant_force'])
            self.spring_force_rand = self.spring_force.clone()
        else:
            self.spring_force = None
            self.spring_force_rand = None
        if self.cfg.spring_settings['damping'] and self._spring_idx is not None:
            self.spring_stretch_damping = torch.zeros(self.num_envs,len(self._spring_idx),dtype=torch.float,device=self.device)
            self.spring_contract_damping = torch.zeros(self.num_envs,len(self._spring_idx),dtype=torch.float,device=self.device)
            self.spring_current_damping = torch.zeros(self.num_envs,len(self._spring_idx),dtype=torch.float,device=self.device)
        else:
            self.spring_stretch_damping = None
            self.spring_contract_damping = None
            self.spring_current_damping = None

        # ground height estimation
        self.ground_z_est = torch.zeros(self.num_envs, device=self.device)
        self.wheel_relative_ground_heights = torch.zeros(self.num_envs, 2, device=self.device)
        self.reset_heading_target = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._axis_aligned_reset_heading_mask = torch.full(
            (self.num_envs,),
            fill_value=bool(getattr(self.cfg, "reset_heading_axis_aligned_only", False)),
            dtype=torch.bool,
            device=self.device,
        )
        self.height_reward_airborne_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.airborne_force_enter_request = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.airborne_wheel_contact_exit_time = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.airborne_base_contact_exit_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.airborne_state_last_update_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.airborne_current_duration = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.airborne_max_duration = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_request = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_takeoff_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.jump_takeoff_phase_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_cooldown_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_height_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_base_height_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_ref_vel_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_ref_release_vel_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_push_max_vel_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_ref_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_ref_target_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_ref_peak_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_ref_duration = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_trigger_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_takeoff_exit_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_takeoff_push_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_takeoff_tuck_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_takeoff_assist_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_takeoff_episode_trigger_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_episode_exit_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_episode_assist_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_episode_max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_episode_max_vel_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_episode_target_peak_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_takeoff_permission_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_takeoff_permission_lin_vel_x = torch.full(
            (self.num_envs,), torch.nan, dtype=torch.float, device=self.device
        )
        self.jump_takeoff_permission_lin_vel_y = torch.full(
            (self.num_envs,), torch.nan, dtype=torch.float, device=self.device
        )
        self.jump_takeoff_permission_ang_vel_z = torch.full(
            (self.num_envs,), torch.nan, dtype=torch.float, device=self.device
        )
        self.predefined_reset_air_command_limit_remaining_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.predefined_reset_air_command_limit_mode_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.predefined_reset_air_command_limit_last_update_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.predefined_reset_air_command_limit_lin_vel_x = torch.full(
            (self.num_envs, 2), torch.nan, dtype=torch.float, device=self.device
        )
        self.predefined_reset_air_command_limit_lin_vel_y = torch.full(
            (self.num_envs, 2), torch.nan, dtype=torch.float, device=self.device
        )
        self.predefined_reset_air_command_limit_ang_vel_z = torch.full(
            (self.num_envs, 2), torch.nan, dtype=torch.float, device=self.device
        )
        self.predefined_reset_air_command_limit_height = torch.full(
            (self.num_envs, 2), torch.nan, dtype=torch.float, device=self.device
        )
        self.wheel_forward_height_cmd_hold_remaining_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_height_cmd_hold_value = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_wall_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wheel_forward_prev_ground_z = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.wheel_forward_prev_ground_z_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_prev_direction_sign = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_step_detect_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_state = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_reference_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_height_cmd = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_success_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_state_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_wall_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_failure_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_detect_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_success_exit_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_failure_exit_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_timeout_exit_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_fast_timer = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_fast_timer_started = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_fast_success_time_event = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_fast_success_started_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_prev_body_relative_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_body_relative_height_progress = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_prev_ground_z = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.wheel_forward_stair_prev_ground_z_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wheel_forward_stair_prev_direction_sign = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._cached_ground_height_valid = False
        self._cached_wheel_kinematics_valid = False
        self._cached_wheel_relative_heights: torch.Tensor | None = None
        self._cached_root_quat_inv: torch.Tensor | None = None
        self._cached_wheel_pos_b: torch.Tensor | None = None
        self._cached_wheel_pos_heading_b: torch.Tensor | None = None
        self._cached_wheel_lin_vel_b: torch.Tensor | None = None
        self._cached_wheel_lin_vel_heading_b: torch.Tensor | None = None
        self._cached_wheel_forward_query_points: torch.Tensor | None = None
        self._cached_wheel_forward_hit_points: torch.Tensor | None = None
        self._cached_wheel_forward_scan_direction_sign: torch.Tensor | None = None
        self._cached_wheel_forward_spatial_height_diffs: torch.Tensor | None = None
        self._cached_wheel_forward_temporal_height_diffs: torch.Tensor | None = None
        self._cached_wheel_forward_stair_query_points: torch.Tensor | None = None
        self._cached_wheel_forward_stair_hit_points: torch.Tensor | None = None
        self._cached_wheel_forward_stair_scan_direction_sign: torch.Tensor | None = None
        self._cached_wheel_forward_stair_temporal_height_diffs: torch.Tensor | None = None
        self._undesired_contact_debug_counter = 0
        self._undesired_contact_debug_interval = int(getattr(self.cfg, "undesired_contact_debug_interval", 0))
        self._undesired_contact_debug_force_threshold = float(
            getattr(
                self.cfg,
                "undesired_contact_debug_force_threshold",
                getattr(self.cfg, "undesired_contact_force_threshold", 5.0),
            )
        )
        self._undesired_contact_debug_max_envs = int(getattr(self.cfg, "undesired_contact_debug_max_envs", 4))
        self._undesired_contact_debug_max_links = int(getattr(self.cfg, "undesired_contact_debug_max_links", 6))
        self._wheel_motor_z_axis_align_debug_counter = 0
        self._wheel_material_debug_counter = 0

        # regist delay buffer (Sensor-level: motor/imu)
        self.use_obs_delay = False
        if hasattr(self.cfg, 'use_obs_delay') and self.cfg.use_obs_delay and self.cfg.obs_delay_cfg:
            try:
                self.obs_delay = dict()
                # 校验缓冲区长度是否足够
                max_obs_delay = max([v[1] for v in self.cfg.obs_delay_cfg.values()])
                if self.cfg.obs_history_len < max_obs_delay:
                    raise ValueError(
                        f"obs_history_len ({self.cfg.obs_history_len}) 必须 >= 最大延迟步数 ({max_obs_delay})。"
                        f"请在配置中将 obs_history_len 设为至少 {max_obs_delay}。"
                    )

                for k, v in self.cfg.obs_delay_cfg.items():
                    self.obs_delay[k] = DelayBuffer(self.cfg.obs_history_len,self.num_envs,self.device)
                    self.obs_delay[k].set_time_lag(self.cfg.obs_default_time_lag)

                self.use_obs_delay = True
                print(f'[Sim2Real] 观测延迟已启用 (物理步级) - 分组: {list(self.obs_delay.keys())}')
                print(f'[Sim2Real]   - 延迟范围: {self.cfg.obs_delay_cfg}')
                print(f'[Sim2Real]   - 缓冲区长度: {self.cfg.obs_history_len}')
                print(f'[Sim2Real]   - 初始化默认延迟: {self.cfg.obs_default_time_lag} 步 ({self.cfg.obs_default_time_lag * self.cfg.sim.dt * 1000:.1f}ms)')
                print(f'[Sim2Real]   - 物理步时间: {self.cfg.sim.dt*1000:.1f}ms (1步 ≈ {self.cfg.sim.dt*1000:.1f}ms)')
            except Exception as e:
                print(f'[警告] 观测延迟初始化失败: {e}')
                self.use_obs_delay = False

        self.use_act_delay = False
        if hasattr(self.cfg, 'use_act_delay') and self.cfg.use_act_delay and self.cfg.act_delay_cfg:
            try:
                self.act_delay = dict()
                # 使用 act_delay_cfg 的最大值作为缓冲区长度
                max_act_delay = max([v[1] for v in self.cfg.act_delay_cfg.values()])
                act_history_len = max_act_delay + 2  # 预留 2 步的余量

                for k, v in self.cfg.act_delay_cfg.items():
                    self.act_delay[k] = DelayBuffer(act_history_len, self.num_envs, self.device)

                self.use_act_delay = True
                print(f'[Sim2Real] 执行延迟已启用 (物理步级) - 分组: {list(self.act_delay.keys())}')
                print(f'[Sim2Real]   - 延迟范围: {self.cfg.act_delay_cfg}')
                print(f'[Sim2Real]   - 缓冲区长度: {act_history_len}')
            except Exception as e:
                print(f'[警告] 执行延迟初始化失败: {e}')
                self.use_act_delay = False

        if self.use_act_delay:
            for k, v in self.act_delay.items():
                lo, hi = self.cfg.act_delay_cfg[k]
                if lo < hi:
                    v.set_time_lag(torch.randint(lo, hi, (self.num_envs,), dtype=torch.int, device=self.device))
        if self.use_obs_delay:
                for k, v in self.obs_delay.items():
                    lo, hi = self.cfg.obs_delay_cfg[k]
                    if lo < hi:
                        v.set_time_lag(torch.randint(lo, hi, (self.num_envs,), dtype=torch.int, device=self.device))

        # regist observation noise model
        self.use_self_obs_noise = False
        try:
            if self.cfg.self_obs_noise_cfg:
                self.self_obs_noise = dict()
                for k, v in self.cfg.self_obs_noise_cfg.items():
                    self.self_obs_noise[k] = v.class_type(
                        v, num_envs=self.num_envs, device=self.device
                    )
                self.use_self_obs_noise = True
        except:
            print('No observations noise!')

        # regist action noise model
        self.use_self_act_noise = False
        try:
            if self.cfg.self_act_noise_cfg:
                self.self_act_noise = dict()
                for k, v in self.cfg.self_act_noise_cfg.items():
                    self.self_act_noise[k] = v.class_type(
                        v, num_envs=self.num_envs, device=self.device
                    )
                self.use_self_act_noise = True
        except:
            print('No actions noise!')

        # ========== 帧堆叠缓冲区初始化 ==========
        # 使用 deque 存储历史观测
        if hasattr(self.cfg, 'num_obs_hist'):
            self.obs_history = deque(maxlen=self.cfg.num_obs_hist)
            # 初始化填充零观测
            for _ in range(self.cfg.num_obs_hist):
                self.obs_history.append(torch.zeros(
                    self.num_envs, self.cfg.num_single_obs, dtype=torch.float, device=self.device))
        else:
            self.obs_history = None
        if hasattr(self.cfg, 'num_privileged_obs_hist'):
            self.critic_history = deque(maxlen=self.cfg.num_privileged_obs_hist)
            self._critic_history_frame_dim = int(getattr(self.cfg, "num_single_privileged_obs", 0) or 0)
            obs_space_cfg = getattr(self.cfg, "observation_space", None)
            if isinstance(obs_space_cfg, dict) and "critic_hist" in obs_space_cfg:
                hist_len = max(int(getattr(self.cfg, "num_privileged_obs_hist", 1) or 1), 1)
                self._critic_history_frame_dim = int(obs_space_cfg["critic_hist"]) // hist_len
            # 初始化填充零观测
            for _ in range(self.cfg.num_privileged_obs_hist):
                self.critic_history.append(torch.zeros(
                    self.num_envs, self._critic_history_frame_dim, dtype=torch.float, device=self.device))
        else:
            self.critic_history = None
            self._critic_history_frame_dim = 0

        # ========== 概率帧遮蔽（Frame Mask Dropout）==========
        # frame_mask_probs: [p_keep_1, p_keep_2, ..., p_keep_N]
        # 训练时按概率对每个环境“每次 reset 采样一次”保留最新 k 帧，其余历史帧置零
        _raw_probs = getattr(self.cfg, 'frame_mask_probs', None)
        if _raw_probs is not None:
            self.frame_mask_probs = list(_raw_probs)
        else:
            self.frame_mask_probs = None

        # 每个环境缓存一个“保留帧数”，在 reset 时更新，episode 生命周期内保持不变
        self._frame_mask_num_keep: torch.Tensor | None = None
        if self.frame_mask_probs is not None and self.obs_history is not None:
            self._frame_mask_num_keep = torch.full(
                (self.num_envs,),
                fill_value=int(self.obs_history.maxlen),
                dtype=torch.int64,
                device=self.device,
            )
            # 初始化时采样一次（等价于所有 env 一次 reset）
            self._resample_frame_mask_num_keep(self.robot._ALL_INDICES)

        # value 爆炸诊断开关（默认关闭；V13 rough 可在 cfg 中开启）
        self._value_debug_enabled = bool(getattr(self.cfg, "debug_value_diagnosis", False))
        self._value_debug_print_interval = int(getattr(self.cfg, "debug_value_diagnosis_interval", 200))
        self._value_debug_topk = int(getattr(self.cfg, "debug_value_diagnosis_topk", 5))
        self._value_debug_threshold_only = bool(getattr(self.cfg, "debug_value_diagnosis_threshold_only", False))
        self._value_debug_thresholds = dict(getattr(self.cfg, "debug_value_diagnosis_thresholds", {}))
        self._value_debug_step = 0
        # 观测输入限幅配置（键名见 _clip_obs_component）
        self._obs_input_clip_cfg = dict(getattr(self.cfg, "obs_input_clip_cfg", {}))
        self._obs_input_scale_enabled = bool(getattr(self.cfg, "obs_input_scale_enabled", False))
        self._obs_input_scale_cfg = dict(getattr(self.cfg, "obs_input_scale_cfg", {}))
        self._obs_input_scale_streams = tuple(
            getattr(self.cfg, "obs_input_scale_streams", ("policy",))
        )
        # 原始观测预警：仅当某分量绝对值超过阈值时打印，平时不输出
        self._obs_alert_threshold = float(getattr(self.cfg, "debug_obs_alert_threshold", 100.0))
        self._obs_alert_topk = int(getattr(self.cfg, "debug_obs_alert_topk", 3))
        self._obs_alert_print_interval = int(getattr(self.cfg, "debug_obs_alert_print_interval", 1))
        self._obs_alert_print_counter = 0

        # ========== 观测异常检测（用于触发 reset）==========
        # 在 _get_observations() 中缓存 raw obs 统计，再在 _get_dones() 里触发 terminate。
        self._obs_raw_policy_max_abs = torch.zeros(self.num_envs, device=self.device)
        self._obs_raw_policy_has_nonfinite = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obs_raw_critic_max_abs = torch.zeros(self.num_envs, device=self.device)
        self._obs_raw_critic_has_nonfinite = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 当前 step 内由物理状态异常触发的 reset 掩码。
        # 用于在 _get_rewards() 中屏蔽坏 env 的奖励，避免污染 PPO rollout。
        self._numerical_safety_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._termination_duration_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._termination_duration_raw_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 默认启用：当 raw obs 超阈值/非有限时直接 terminate -> reset
        self._terminate_on_obs_outlier = bool(getattr(self.cfg, "terminate_on_obs_outlier", True))
        self._terminate_obs_abs = float(getattr(self.cfg, "terminate_obs_abs", 0.0))
        self._terminate_obs_print_interval = int(getattr(self.cfg, "terminate_obs_print_interval", 50))
        self._terminate_obs_print_counter = 0

        # initial obs
        self.obs_update_flag = 1
        self._update_obs()

        # Logging
        self._episode_sums = dict()

        # Curriculum manager (optional)
        self.curriculum_manager: CurriculumManager | None = None
        if hasattr(self.cfg, "curriculum") and self.cfg.curriculum:
            try:
                self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
                print("[INFO] Curriculum Manager: ", self.curriculum_manager)
            except Exception as exc:
                print(f"[WARNING] 无法初始化课程管理器：{exc}")
                self.curriculum_manager = None

        self.state_machine_manager = self._build_state_machine_manager()

        self._wheel_forward_scan_marker: VisualizationMarkers | None = None
        self._play_ang_vel_z_cmd_marker: VisualizationMarkers | None = None
        self._play_ang_vel_z_actual_marker: VisualizationMarkers | None = None
        self._terrain_task_manager: TerrainTaskManager | None = None
        self._terrain_task_manager_initialized = False
        self._terrain_debug_marker: VisualizationMarkers | None = None
        self._terrain_debug_marker_height_offset = 1.45
        if bool(getattr(self.cfg, "play", False)):
            self._setup_state_machine_marker()
            if self._is_wheel_forward_scan_enabled():
                self._setup_wheel_forward_scan_marker()
            self._setup_play_ang_vel_z_marker()
            self._setup_builtin_terrain_debug_marker()

        # initial static obs params
        self.static_priv_obs = self._get_static_priv_obs()

    def _find_contact_sensor_indices(self, body_names_expr: str | Sequence[str]) -> list[int]:
        """根据正则表达式在接触传感器的 body_names 中寻找对应的索引 (解决 robot 索引不匹配问题)"""
        import re
        if isinstance(body_names_expr, str):
            body_names_expr = [body_names_expr]

        indices = []
        # 将通配符 .* 转换为正则
        for expr in body_names_expr:
            pattern = re.compile(expr)
            for i, name in enumerate(self.contact_sensor.body_names):
                if pattern.match(name):
                    indices.append(i)
        return sorted(list(set(indices)))
    
    def _build_material_mapping(self):
        """建立 body名称 到 material_properties (23维) 的索引映射"""
        if self._body_to_material_indices:  # 已构建
            return

        physics_sim_view = self.robot._physics_sim_view
        self._body_to_material_indices.clear()
        self._material_body_names.clear()

        shape_idx = 0
        for i, link_path in enumerate(self.robot.root_physx_view.link_paths[0]):
            body_name = self.robot.body_names[i]
            link_view = physics_sim_view.create_rigid_body_view(link_path)
            num_shapes = int(link_view.max_shapes)

            indices = list(range(shape_idx, shape_idx + num_shapes))
            self._body_to_material_indices[body_name] = indices
            self._material_body_names.append(body_name)

            shape_idx += num_shapes

        mat_shape = self.robot.root_physx_view.get_material_properties().shape[1]
        if shape_idx != mat_shape:
            print(f"[WARNING] Shape count mismatch: got {shape_idx}, material dim={mat_shape}")

    def _get_material_indices(self, body_patterns: str | list[str]) -> list[int]:
        """通过 body 名称或正则表达式，返回其在 material_properties (23维) 中的所有索引"""
        self._build_material_mapping()
        if isinstance(body_patterns, str):
            body_patterns = [body_patterns]

        import re
        material_indices: list[int] = []
        for pattern_str in body_patterns:
            pattern = re.compile(pattern_str.replace(".*", ".*"))  # 支持 .* 通配符
            for body_name, mat_idx in self._body_to_material_indices.items():
                if pattern.search(body_name):
                    material_indices.extend(mat_idx)
        return sorted(list(set(material_indices)))

    def _init_static_index_layouts(self) -> None:
        """Cache index/count layouts that only depend on cfg and robot topology."""
        self._leg_action_dim = len(self._legs_act_idx)
        self._wheel_action_dim = len(self._wheel_idx)
        self._actuated_joint_count = len(self._actuate_idx)
        self._wheel_link_count = len(self._wheel_link_idx)
        self._init_privileged_extra_obs_layout()

    def _init_privileged_extra_obs_layout(self) -> None:
        self._privileged_extra_obs_enabled = bool(
            getattr(self.cfg, "privileged_extra_obs_enabled", False)
        )
        self._legacy_privileged_extra_obs_enabled = self._compute_uses_legacy_privileged_extra_obs()
        self._privileged_extra_obs_dim = int(getattr(self.cfg, "privileged_extra_obs_dim", 0) or 0)
        self._privileged_extra_joint_count = int(
            getattr(self.cfg, "privileged_extra_joint_count", self.cfg.action_space)
        )
        self._privileged_extra_body_count = int(getattr(self.cfg, "privileged_extra_body_count", 0))
        self._privileged_extra_inertia_body_count = int(
            getattr(self.cfg, "privileged_extra_inertia_body_count", 0)
        )
        self._privileged_extra_material_body_count = int(
            getattr(self.cfg, "privileged_extra_material_body_count", 0)
        )
        self._privileged_extra_wheel_count = int(getattr(self.cfg, "privileged_extra_wheel_count", 2))

        joint_names_cfg = getattr(self.cfg, "privileged_extra_joint_names", None)
        wheel_body_names_cfg = getattr(self.cfg, "privileged_extra_wheel_body_names", None)
        body_names_cfg = getattr(self.cfg, "privileged_extra_body_names", None)
        inertia_body_names_cfg = getattr(self.cfg, "privileged_extra_inertia_body_names", None)
        material_body_names_cfg = getattr(self.cfg, "privileged_extra_material_body_names", None)

        self._privileged_extra_actuate_idx = (
            self._resolve_name_patterns_to_indices(joint_names_cfg, self.robot.joint_names)
            if joint_names_cfg
            else list(self._actuate_idx)
        )
        if joint_names_cfg:
            self._privileged_extra_joint_count = len(self._privileged_extra_actuate_idx)

        self._privileged_extra_wheel_idx = (
            self._resolve_name_patterns_to_indices(wheel_body_names_cfg, self.robot.body_names)
            if wheel_body_names_cfg
            else list(self._wheel_link_idx)
        )
        if wheel_body_names_cfg:
            self._privileged_extra_wheel_count = len(self._privileged_extra_wheel_idx)

        self._privileged_extra_body_indices = self._get_privileged_extra_body_indices()
        self._privileged_extra_inertia_body_indices = self._get_privileged_extra_inertia_body_indices()
        if body_names_cfg:
            self._privileged_extra_body_count = len(self._privileged_extra_body_indices)
        if inertia_body_names_cfg:
            self._privileged_extra_inertia_body_count = len(self._privileged_extra_inertia_body_indices)

        if material_body_names_cfg:
            self._privileged_extra_material_indices = self._get_material_indices(material_body_names_cfg)
            self._privileged_extra_material_body_count = len(self._privileged_extra_material_indices)
        else:
            self._privileged_extra_material_indices = (
                list(self._randomed_friction_link_idx)
                if self._randomed_friction_link_idx
                else list(self._privileged_extra_body_indices)
            )

        body_lin_vel_w = getattr(self.robot.data, "body_lin_vel_w", None)
        robot_body_count = body_lin_vel_w.shape[1] if body_lin_vel_w is not None and body_lin_vel_w.ndim >= 2 else 0
        self._privileged_extra_valid_wheel_idx = [
            int(idx)
            for idx in self._privileged_extra_wheel_idx
            if 0 <= int(idx) < robot_body_count
        ][: self._privileged_extra_wheel_count]

    def _set_play_height_scanner_debug_vis(self) -> None:
        """Enable all height-related raycaster debug visualization in play mode."""
        if not bool(getattr(self.cfg, "play", False)):
            return
        if not bool(getattr(self.cfg, "play_height_scanner_debug_vis", False)):
            return

        for attr_name in (
            "height_scanner",
            "right_wheel_height_scanner",
            "left_wheel_height_scanner",
        ):
            scanner = getattr(self, attr_name, None)
            if scanner is None:
                continue
            try:
                scanner.set_debug_vis(True)
            except Exception as exc:
                print(f"[WARNING] 无法开启 {attr_name} 可视化：{exc}")

    def _setup_scene(self):
        # add robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.height_scanner = None
        if self._use_raycast_height():
            self.height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self.height_scanner
        self.right_wheel_height_scanner = None
        right_wheel_height_scanner_cfg = getattr(self.cfg, "right_wheel_height_scanner", None)
        if right_wheel_height_scanner_cfg is not None:
            self.right_wheel_height_scanner = RayCaster(right_wheel_height_scanner_cfg)
            self.scene.sensors["right_wheel_height_scanner"] = self.right_wheel_height_scanner
        self.left_wheel_height_scanner = None
        left_wheel_height_scanner_cfg = getattr(self.cfg, "left_wheel_height_scanner", None)
        if left_wheel_height_scanner_cfg is not None:
            self.left_wheel_height_scanner = RayCaster(left_wheel_height_scanner_cfg)
            self.scene.sensors["left_wheel_height_scanner"] = self.left_wheel_height_scanner
        self.dot_scanner = None
        dot_scanner_cfg = getattr(self.cfg, "dot_scanner", None)
        if dot_scanner_cfg is not None and bool(getattr(self.cfg, "enable_scan_dot", False)):
            self.dot_scanner = RayCaster(dot_scanner_cfg)
            self.scene.sensors["dot_scanner"] = self.dot_scanner
        self._set_play_height_scanner_debug_vis()

        # add terrains
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # self.leg_actions = self.robot.data.default_joint_pos[:,self._legs_act_idx]

    def _low_pass_action_filter(self, actions: torch.Tensor) -> torch.Tensor:
        return (
            self.last_actions * self.action_low_pass_prev_weight
            + actions * self.action_low_pass_curr_weight
        )

    def _get_leg_policy_action_dim(self) -> int:
        encoding = str(getattr(self.cfg, "leg_action_encoding", "raw")).lower()
        if encoding in ("raw", "none", ""):
            return self._leg_action_dim
        if encoding == "sincos_abs":
            return 2 * self._leg_action_dim
        raise RuntimeError(f"Unsupported leg_action_encoding: {encoding}")

    def _split_policy_actions(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leg_action_dim = self._get_leg_policy_action_dim()
        expected_action_dim = leg_action_dim + self._wheel_action_dim
        if actions.shape[-1] != expected_action_dim:
            raise RuntimeError(
                f"Action dim mismatch for leg_action_encoding="
                f"{getattr(self.cfg, 'leg_action_encoding', 'raw')}: "
                f"got {actions.shape[-1]}, expected {expected_action_dim}"
            )
        return actions[:, :leg_action_dim], actions[:, leg_action_dim:]

    def _decode_leg_policy_actions(self, leg_policy_actions: torch.Tensor) -> torch.Tensor:
        '''解析腿部动作编码，返回实际的腿部关节位置命令'''
        encoding = str(getattr(self.cfg, "leg_action_encoding", "raw")).lower()
        default_leg_pos = self.robot.data.default_joint_pos[:, self._legs_act_idx]
        if encoding in ("raw", "none", ""):
            return self.leg_action_scale * leg_policy_actions + default_leg_pos
        if encoding == "sincos_abs":
            leg_sin = torch.nan_to_num(
                leg_policy_actions[:, : self._leg_action_dim], nan=0.0, posinf=0.0, neginf=0.0
            )
            leg_cos = torch.nan_to_num(
                leg_policy_actions[:, self._leg_action_dim :], nan=0.0, posinf=0.0, neginf=0.0
            )
            if bool(getattr(self.cfg, "leg_action_sincos_default_prior", False)):
                vector_scale = float(getattr(self.cfg, "leg_action_sincos_vector_scale", 1.0))
                leg_sin = torch.sin(default_leg_pos) + vector_scale * leg_sin
                leg_cos = torch.cos(default_leg_pos) + vector_scale * leg_cos
            decoded_leg_pos = torch.atan2(leg_sin, leg_cos)
            action_norm = torch.sqrt(torch.square(leg_sin) + torch.square(leg_cos))
            zero_norm_eps = float(getattr(self.cfg, "leg_action_sincos_zero_norm_epsilon", 1.0e-6))
            decoded_leg_pos = torch.where(action_norm > zero_norm_eps, decoded_leg_pos, default_leg_pos)
            if bool(getattr(self.cfg, "leg_action_sincos_unwrap_to_current", False)):
                current_leg_pos = self.joint_pos[:, self._legs_act_idx]
                decoded_leg_pos = current_leg_pos + wrap_to_pi(decoded_leg_pos - current_leg_pos)
            if not bool(getattr(self.cfg, "leg_action_sincos_clamp_after_unwrap", True)):
                return decoded_leg_pos
            return torch.clamp(
                decoded_leg_pos,
                self.cfg.lower_joint_limit,
                self.cfg.upper_joint_limit,
            )
        raise RuntimeError(f"Unsupported leg_action_encoding: {encoding}")

    def _get_policy_action_slices(self) -> tuple[slice, slice]:
        leg_action_dim = self._get_leg_policy_action_dim()
        return slice(0, leg_action_dim), slice(leg_action_dim, leg_action_dim + self._wheel_action_dim)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._invalidate_step_caches() # 清理缓存buffer（这些缓存是为了跨函数获取同一个物理量）
        # always init is finished
        self.finish_init.fill_(True)
        self.start_reset.fill_(True)

        # 起身训练阶段更新
        # if getattr(self.cfg, 'enable_standup_training', False):
        #     landing_steps = self.cfg.landing_phase_steps
        #     standup_steps = self.cfg.standup_phase_steps
        #     ep_len = self.episode_length_buf
        #     self._standup_phase.fill_(2)
        #     self._standup_phase[ep_len < landing_steps] = 0
        #     self._standup_phase[(ep_len >= landing_steps) & (ep_len < landing_steps + standup_steps)] = 1

        # update disable actions ids
        # self.extras["disable_actions"] = (~self.finish_init).nonzero(as_tuple=False).squeeze(-1)

        # cal actions
        self._actions = actions.clone()
        if self.use_action_low_pass_filter:
            self._actions = self._low_pass_action_filter(self._actions)
        self.last_actions.copy_(self._actions)
        leg_policy_actions, wheel_policy_actions = self._split_policy_actions(self._actions)
        self.leg_actions = self._decode_leg_policy_actions(leg_policy_actions)
        # self.leg_actions = self.leg_actions + self.leg_action_scale * self._actions[:, : self._leg_action_dim]
        self.wheel_actions = self.wheel_action_scale * wheel_policy_actions

    def _apply_action(self) -> None:
        # dealing action limits
        upper_joint_limit = torch.zeros(self.num_envs, self._leg_action_dim, dtype=torch.float, device=self.device)
        lower_joint_limit = upper_joint_limit.clone()
        upper_joint_limit = self.cfg.upper_joint_limit
        lower_joint_limit = self.cfg.lower_joint_limit
        max_wheel_torque = self.cfg.max_wheel_torque

        # initial enable/disable env ids
        # 起身训练时：落地阶段(phase=0)也执行零力矩，与 init 未完成同逻辑
        episode_time = self.episode_length_buf.to(dtype=torch.float) * self.step_dt
        predefined_reset_ground_zero_torque_mask = episode_time < self.predefined_reset_ground_zero_torque_until_time
        enable_mask = self.finish_init.clone() & ~predefined_reset_ground_zero_torque_mask
        # if getattr(self.cfg, 'enable_standup_training', False):
        #     enable_mask = enable_mask & (self._standup_phase != 0)
        enable_env_ids = enable_mask.nonzero(as_tuple=False).squeeze(-1)
        disable_env_ids = (~enable_mask).nonzero(as_tuple=False).squeeze(-1)

        # if self.use_act_delay:
        #     # 每步重采样 act lag：帧间延迟不稳定，policy 无法利用稳定帧差做微分
        #     for k, v in self.act_delay.items():
        #         lo, hi = self.cfg.act_delay_cfg[k]
        #         if lo < hi:
        #             v.set_time_lag(torch.randint(lo, hi, (self.num_envs,), dtype=torch.int, device=self.device))

        # apply actions
        if self.leg_actions is not None:
            leg_actions_deal = self.leg_actions.clone()
            # 原始action添加噪声
            if self.use_self_act_noise and "leg_actions" in self.self_act_noise:
                leg_actions_deal = self.self_act_noise["leg_actions"](self.leg_actions)
            # action 裁剪
            leg_actions_deal = torch.clamp(
                leg_actions_deal,
                lower_joint_limit,
                upper_joint_limit
            )
            if self.use_act_delay:
                leg_actions_deal = self.act_delay['leg_actions'].compute(leg_actions_deal)
            self.robot.set_joint_position_target(leg_actions_deal[enable_env_ids], joint_ids=self._legs_act_idx, env_ids = enable_env_ids)

        if self.wheel_actions is not None:
            wheel_actions_deal = self.wheel_actions.clone()
            # 原始action添加噪声
            if self.use_self_act_noise and "wheel_actions" in self.self_act_noise:
                wheel_actions_deal = self.self_act_noise["wheel_actions"](wheel_actions_deal)
            if self.use_wheel_vel_control:
                # 轮速控制模式：policy 输出目标角速度，夹紧到 ±max_wheel_vel
                wheel_actions_deal = torch.clamp(
                    wheel_actions_deal,
                    -self.max_wheel_vel,
                    self.max_wheel_vel,
                )
                if self.use_act_delay:
                    wheel_actions_deal = self.act_delay['wheel_actions'].compute(wheel_actions_deal)
                # 设置速度目标
                self.robot.set_joint_velocity_target(wheel_actions_deal[enable_env_ids], joint_ids=self._wheel_idx, env_ids=enable_env_ids)
                # 对目标位置做 damping：将当前轮位置作为位置目标，
                # 使执行器在速度为 0 时产生阻尼保持效果（actuator stiffness > 0 时生效）
                current_wheel_pos = self.joint_pos[enable_env_ids][:, self._wheel_idx]
                self.robot.set_joint_position_target(current_wheel_pos, joint_ids=self._wheel_idx, env_ids=enable_env_ids)
            else:
                # 力矩控制模式（默认）：直接输出力矩，夹紧到 ±max_wheel_torque
                wheel_actions_deal = torch.clamp(
                    wheel_actions_deal,
                    -max_wheel_torque,
                    max_wheel_torque,
                )
                if self.use_act_delay:
                    wheel_actions_deal = self.act_delay['wheel_actions'].compute(wheel_actions_deal)
                self.robot.set_joint_effort_target(wheel_actions_deal[enable_env_ids], joint_ids=self._wheel_idx, env_ids=enable_env_ids)

        if len(disable_env_ids) > 0:
            disable_joint_pos = self.joint_pos[disable_env_ids]
            disable_joint_vel = self.joint_vel[disable_env_ids]
            disable_joint_effort = self.joint_zero_torque[disable_env_ids]
            self.robot.set_joint_position_target(disable_joint_pos[:,self._legs_act_idx], joint_ids=self._legs_act_idx, env_ids = disable_env_ids)
            self.robot.set_joint_velocity_target(disable_joint_vel[:,self._legs_act_idx], joint_ids=self._legs_act_idx, env_ids = disable_env_ids)
            self.robot.set_joint_effort_target(disable_joint_effort[:,self._legs_act_idx], joint_ids=self._legs_act_idx, env_ids = disable_env_ids)
            self.robot.set_joint_position_target(disable_joint_pos[:,self._wheel_idx], joint_ids=self._wheel_idx, env_ids = disable_env_ids)
            self.robot.set_joint_velocity_target(disable_joint_vel[:,self._wheel_idx], joint_ids=self._wheel_idx, env_ids = disable_env_ids)
            self.robot.set_joint_effort_target(disable_joint_effort[:,self._wheel_idx], joint_ids=self._wheel_idx, env_ids = disable_env_ids)

        # 应用弹簧力（仅当启用弹簧时）
        if self.use_spring:
            self._apply_spring()

        self._update_obs()

        # print(self.joint_vel[0,self._wheel_idx])
        # print(self.robot.data.joint_vel_limits[0,self._wheel_idx])
        # print(self.robot.data.root_lin_vel_b[:,0])
        # print(self.robot.data.root_ang_vel_b[:, 2])
        # for n in self._actuate_idx:
        #     print(self.robot.joint_names[n])

    def _update_obs(self, is_obs=False):

        if self.obs_update_flag == 0:
            # orin obs
            self.obs_root_ang_vel_b = self.robot.data.root_ang_vel_b.clone()
            self.obs_projected_gravity_b = self.robot.data.projected_gravity_b.clone()
            self.obs_joint_pos = (self.joint_pos - self.default_joint_pos)[:,self._actuate_idx]
            self.obs_joint_vel = self.joint_vel[:,self._actuate_idx].clone()

            if self.use_obs_delay:
            #     # 每步重采样 obs lag：帧间延迟不稳定，policy 无法利用稳定帧差做微分
            #     for k, v in self.obs_delay.items():
            #         lo, hi = self.cfg.obs_delay_cfg[k]
            #         if lo < hi:
            #             v.set_time_lag(torch.randint(lo, hi, (self.num_envs,), dtype=torch.int, device=self.device))
                # compute delay obs
                self.obs_root_ang_vel_b = self.obs_delay['root_ang_vel_b'].compute(self.obs_root_ang_vel_b)
                self.obs_projected_gravity_b = self.obs_delay['projected_gravity_b'].compute(self.obs_projected_gravity_b)
                self.obs_joint_pos = self.obs_delay['joint_pos'].compute(self.obs_joint_pos)
                self.obs_joint_vel = self.obs_delay['joint_vel'].compute(self.obs_joint_vel)

                # ========== 电机延迟 (motor) ==========
                # 将 joint_pos 和 joint_vel 合并后一起延迟，然后再拆分
                if 'motor' in self.obs_delay:
                    obs_motor = torch.cat([self.obs_joint_pos, self.obs_joint_vel], dim=-1)
                    obs_motor_delayed = self.obs_delay['motor'].compute(obs_motor)
                    # 拆分回 joint_pos 和 joint_vel
                    motor_dim = self.obs_joint_pos.shape[-1]
                    self.obs_joint_pos = obs_motor_delayed[:, :motor_dim]
                    self.obs_joint_vel = obs_motor_delayed[:, motor_dim:]

                # ========== IMU延迟 (imu) ==========
                # 将 ang_vel 和 projected_gravity 合并后一起延迟，然后再拆分
                if 'imu' in self.obs_delay:
                    obs_imu = torch.cat([self.obs_root_ang_vel_b, self.obs_projected_gravity_b], dim=-1)
                    obs_imu_delayed = self.obs_delay['imu'].compute(obs_imu)
                    # 拆分回 ang_vel 和 projected_gravity
                    self.obs_root_ang_vel_b = obs_imu_delayed[:, :3]
                    self.obs_projected_gravity_b = obs_imu_delayed[:, 3:]

        if is_obs:
            self.obs_update_flag = 1
        else:
            self.obs_update_flag = 0

    def _debug_print_undesired_contacts(self, net_contact_forces: torch.Tensor | None) -> None:
        """在 play 模式下低频打印 undesired link 接触情况。"""
        if not getattr(self.cfg, "play", False):
            return
        if self._undesired_contact_debug_interval <= 0:
            return
        if net_contact_forces is None or len(self._undesired_contact_link_idx) == 0:
            return

        self._undesired_contact_debug_counter += 1
        if self._undesired_contact_debug_counter % self._undesired_contact_debug_interval != 0:
            return

        undesired_forces = torch.norm(net_contact_forces[:, :, self._undesired_contact_link_idx], dim=-1)
        max_forces = torch.amax(undesired_forces, dim=1)
        threshold = self._undesired_contact_debug_force_threshold
        active_mask = torch.any(max_forces > threshold, dim=-1)
        if not torch.any(active_mask):
            print(
                f"[UndesiredContact][step={self._undesired_contact_debug_counter}] "
                f"no undesired contacts above {threshold:.2f}"
            )
            return

        active_env_ids = torch.where(active_mask)[0]
        sample_env_ids = active_env_ids[: min(self._undesired_contact_debug_max_envs, active_env_ids.numel())]
        print(
            f"[UndesiredContact][step={self._undesired_contact_debug_counter}] "
            f"active_envs={active_env_ids.numel()} threshold={threshold:.2f}"
        )

        for env_id in sample_env_ids.detach().cpu().tolist():
            env_force_vals = max_forces[env_id]
            active_link_local = torch.where(env_force_vals > threshold)[0]
            if active_link_local.numel() == 0:
                continue
            active_force_vals = env_force_vals[active_link_local]
            sort_order = torch.argsort(active_force_vals, descending=True)
            top_local = active_link_local[sort_order[: self._undesired_contact_debug_max_links]]
            link_msgs: list[str] = []
            for local_idx in top_local.detach().cpu().tolist():
                contact_idx = self._undesired_contact_link_idx[local_idx]
                link_name = self.contact_sensor.body_names[contact_idx]
                link_force = env_force_vals[local_idx].item()
                link_msgs.append(f"{link_name}:{link_force:.2f}")
            print(f"[UndesiredContact][env={env_id}] links=[{', '.join(link_msgs)}]")

    def _get_wheel_contact_force_peaks(
        self, net_contact_forces: torch.Tensor | None
    ) -> torch.Tensor:
        """Return per-wheel max contact-force norm over the contact history window."""
        if net_contact_forces is None or len(self._desired_contact_link_idx) == 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        return torch.amax(
            torch.norm(net_contact_forces[:, :, self._desired_contact_link_idx], dim=-1),
            dim=1,
        )

    def _get_wheel_air_spin_reward(
        self,
        wheel_contact_force_peaks: torch.Tensor,
        contact_force_threshold: float,
    ) -> torch.Tensor:
        """Return wheel-speed squared only for wheels without contact."""
        wheel_vel = self.joint_vel[:, self._wheel_idx]
        if wheel_contact_force_peaks.shape[1] == 0 or wheel_vel.shape[1] == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        wheel_count = min(wheel_contact_force_peaks.shape[1], wheel_vel.shape[1])
        no_contact = wheel_contact_force_peaks[:, :wheel_count] <= contact_force_threshold
        wheel_vel_sq = torch.square(
            torch.nan_to_num(wheel_vel[:, :wheel_count], nan=0.0, posinf=0.0, neginf=0.0)
        )
        return torch.sum(wheel_vel_sq * no_contact.float(), dim=1)

    def _get_np3o_costs(self, obs_height: torch.Tensor) -> torch.Tensor:
        num_costs = int(getattr(self.cfg, "num_costs", 0) or 0)
        if num_costs <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)

        projected_gravity = self.robot.data.projected_gravity_b
        tilt_limit = math.sin(math.radians(float(getattr(self.cfg, "np3o_tilt_limit_deg", 15.0))))
        body_tilt_cost = torch.square(torch.relu(torch.linalg.norm(projected_gravity[:, :2], dim=-1) - tilt_limit))

        height = obs_height.squeeze(-1)
        min_height = float(getattr(self.cfg, "np3o_body_height_min", 0.18))
        max_height = float(getattr(self.cfg, "np3o_body_height_max", 0.42))
        body_height_cost = torch.square(torch.relu(min_height - height)) + torch.square(torch.relu(height - max_height))

        ang_vel_limit = float(getattr(self.cfg, "np3o_ang_vel_xy_limit", 4.0))
        body_ang_vel_xy_cost = torch.square(
            torch.relu(torch.linalg.norm(self.robot.data.root_ang_vel_b[:, :2], dim=-1) / max(ang_vel_limit, 1.0e-6) - 1.0)
        )

        torque_limit = float(getattr(self.cfg, "np3o_torque_limit", 30.0))
        torque = self.robot.data.applied_torque[:, self._actuate_idx]
        torque_limit_cost = torch.mean(torch.square(torch.relu(torch.abs(torque) / max(torque_limit, 1.0e-6) - 1.0)), dim=-1)

        joint_vel_limit = float(getattr(self.cfg, "np3o_joint_velocity_limit", 80.0))
        joint_velocity = self.joint_vel[:, self._actuate_idx]
        joint_velocity_limit_cost = torch.mean(
            torch.square(torch.relu(torch.abs(joint_velocity) / max(joint_vel_limit, 1.0e-6) - 1.0)),
            dim=-1,
        )

        costs = torch.stack(
            [
                body_tilt_cost,
                body_height_cost,
                body_ang_vel_xy_cost,
                torque_limit_cost,
                joint_velocity_limit_cost,
            ],
            dim=-1,
        )
        if costs.shape[-1] != num_costs:
            costs = costs[:, :num_costs]
        cost_clip = float(getattr(self.cfg, "np3o_cost_clip", 100.0))
        costs = torch.nan_to_num(costs, nan=0.0, posinf=cost_clip, neginf=0.0)
        return torch.clamp(costs, min=0.0, max=cost_clip)

    def _get_observations(self) -> dict:
        # print(self.robot.data.root_ang_vel_b.clone()[0,2])
        self._maybe_print_wheel_material_debug()
        # prepare observations
        # 更新地面高度估计
        self._update_ground_height_estimate()
        self._update_obs(True)
        self._before_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        observations = dict()
        joint_pos_obs = (self.joint_pos - self.default_joint_pos)[:,self._actuate_idx]
        joint_vel_obs = self.joint_vel[:,self._actuate_idx]
        self.command = self.command_generator.command.clone()
        # print(self.command_generator.command.clone()[0])
        # 子类钩子：在命令刷新后、obs 构建前，允许子类对 command / height_cmd 做修改
        self._on_command_updated()
        self._apply_predefined_reset_air_command_limits()
        # 起身训练：整个落地+起身阶段速度命令和角速度命令强制为0
        # if getattr(self.cfg, 'enable_standup_training', False):
        #     in_standup_phase = (self._standup_phase == 0) | (self._standup_phase == 1)
        #     self.command[in_standup_phase, :3] = 0.0
        # if self.cfg.use_lin_vel_x_constrain:
        #     self.command[:,0] = self._value_constrain(self.command[:,0],self.robot.data.root_lin_vel_b[:,0],self.cfg.lin_vel_x_constrain)
        height_cmd = self._get_observation_height_cmd().clone().unsqueeze(-1)
        self._advance_predefined_reset_air_command_limit_timers()
        root_quat_inv, wheel_pos_b, wheel_pos_heading_b, _, _ = (
            self._get_root_quat_inv_and_wheel_pos_b()
        )
        obs_height = self._get_observed_height(wheel_pos_b).unsqueeze(-1)
        self.wheel_relative_ground_heights = self._get_wheel_relative_ground_heights_raw()
        
        self._update_height_reward_airborne_state()
        self._update_state_machine_marker()
        if self._is_wheel_forward_scan_enabled():
            self._update_wheel_forward_scan_marker()
        self._update_play_ang_vel_z_marker()
        self._update_builtin_terrain_debug_marker()
        task_flag_obs = self._get_task_flag_obs_raw()
        policy_task_flag_obs = self._get_policy_task_flag_obs_raw(task_flag_obs)
        # 观测高度应以估计地面为参考
        # relative_obs_height = obs_height - self.ground_z_est.unsqueeze(-1)

        # construct observations
        if self.cfg.mute_wheel_pos_obs:
            joint_pos_obs[:, self._leg_action_dim :] = 0.
            self.obs_joint_pos[:, self._leg_action_dim :] = 0.

        # update observations (必须在本步 append 前完成，Sim2Sim 需 append 当前帧)
        leg_dim = self._leg_action_dim
        if self.use_self_obs_noise:
            raw_obs_ang_vel = self.self_obs_noise['root_ang_vel_b'](self.obs_root_ang_vel_b)
            raw_obs_proj_g = self.self_obs_noise['projected_gravity_b'](self.obs_projected_gravity_b)
            raw_obs_joint_pos = self.self_obs_noise['joint_pos'](self.obs_joint_pos)
            raw_obs_joint_vel_leg = self.self_obs_noise['leg_joint_vel'](self.obs_joint_vel[:, :leg_dim])
            raw_obs_joint_vel_wheel = self.self_obs_noise['wheel_joint_vel'](self.obs_joint_vel[:, leg_dim:])
        else:
            raw_obs_ang_vel = self.obs_root_ang_vel_b
            raw_obs_proj_g = self.obs_projected_gravity_b
            raw_obs_joint_pos = self.obs_joint_pos
            raw_obs_joint_vel_leg = self.obs_joint_vel[:, :leg_dim]
            raw_obs_joint_vel_wheel = self.obs_joint_vel[:, leg_dim:]
        policy_joint_pos_obs = self._encode_joint_pos_obs(raw_obs_joint_pos)
        policy_extra_obs_blocks = self._get_policy_extra_obs_blocks()

        raw_obs_blocks = {
            "command": self.command,
            "task_flag": policy_task_flag_obs,
            "height_cmd": height_cmd,
            "root_ang_vel_b": raw_obs_ang_vel,
            "projected_gravity_b": raw_obs_proj_g,
            "joint_pos": policy_joint_pos_obs,
            "joint_vel_leg": raw_obs_joint_vel_leg,
            "joint_vel_wheel": raw_obs_joint_vel_wheel,
            "actions": self._actions,
        }
        raw_obs_blocks.update(policy_extra_obs_blocks)
        self._debug_obs_alert(raw_obs_blocks, stream_name="policy_raw")
        self._obs_raw_policy_max_abs, self._obs_raw_policy_has_nonfinite = self._per_env_max_abs_from_blocks(
            raw_obs_blocks, num_envs=self.num_envs, device=self.device
        )
        policy_extra_obs = [
            self._scale_obs_component(key, self._clip_obs_component(key, value))
            for key, value in policy_extra_obs_blocks.items()
        ]

        clip_obs_joint_vel = torch.cat(
            [
                self._scale_obs_component(
                    "joint_vel_leg",
                    self._clip_obs_component("joint_vel_leg", raw_obs_joint_vel_leg),
                ),
                self._scale_obs_component(
                    "joint_vel_wheel",
                    self._clip_obs_component("joint_vel_wheel", raw_obs_joint_vel_wheel),
                ),
            ],
            dim=-1,
        )
        self.obs = torch.cat([
            self._scale_obs_component("command", self._clip_obs_component("command", self.command)),
            self._scale_obs_component("task_flag", self._clip_obs_component("task_flag", policy_task_flag_obs)),
            self._scale_obs_component("height_cmd", self._clip_obs_component("height_cmd", height_cmd)),
            self._scale_obs_component(
                "root_ang_vel_b",
                self._clip_obs_component("root_ang_vel_b", raw_obs_ang_vel),
            ),
            self._scale_obs_component(
                "projected_gravity_b",
                self._clip_obs_component("projected_gravity_b", raw_obs_proj_g),
            ),
            self._scale_obs_component("joint_pos", self._clip_obs_component("joint_pos", policy_joint_pos_obs)),
            clip_obs_joint_vel,
            self._scale_obs_component("actions", self._clip_obs_component("actions", self._actions)),
            *policy_extra_obs,
        ], dim=-1)
        self.obs = torch.nan_to_num(self.obs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.state_space:
            raw_priv_joint_vel_leg = joint_vel_obs[:, :leg_dim]
            raw_priv_joint_vel_wheel = joint_vel_obs[:, leg_dim:]
            priv_joint_pos_obs = self._encode_joint_pos_obs(joint_pos_obs)
            critic_extra_obs_blocks = self._get_critic_extra_obs_blocks(root_quat_inv)
            raw_priv_blocks = {
                "command": self.command,
                "task_flag": task_flag_obs,
                "height_cmd": height_cmd,
                "root_ang_vel_b": self.robot.data.root_ang_vel_b,
                "projected_gravity_b": self.robot.data.projected_gravity_b,
                "joint_pos": priv_joint_pos_obs,
                "joint_vel_leg": raw_priv_joint_vel_leg,
                "joint_vel_wheel": raw_priv_joint_vel_wheel,
                "actions": self._actions,
                "root_lin_vel_b": self.robot.data.root_lin_vel_b,
                "obs_height": obs_height,
            }
            raw_priv_blocks.update(critic_extra_obs_blocks)
            self._debug_obs_alert(raw_priv_blocks, stream_name="critic_raw")
            self._obs_raw_critic_max_abs, self._obs_raw_critic_has_nonfinite = self._per_env_max_abs_from_blocks(
                raw_priv_blocks, num_envs=self.num_envs, device=self.device
            )
            critic_extra_obs = [
                self._scale_obs_component(
                    key,
                    self._clip_obs_component(key, value),
                    stream_name="critic",
                )
                for key, value in critic_extra_obs_blocks.items()
            ]
            clip_priv_joint_vel = torch.cat(
                [
                    self._scale_obs_component(
                        "joint_vel_leg",
                        self._clip_obs_component("joint_vel_leg", raw_priv_joint_vel_leg),
                        stream_name="critic",
                    ),
                    self._scale_obs_component(
                        "joint_vel_wheel",
                        self._clip_obs_component("joint_vel_wheel", raw_priv_joint_vel_wheel),
                        stream_name="critic",
                    ),
                ],
                dim=-1,
            )
            self.priv = torch.cat([
                self._scale_obs_component("command", self._clip_obs_component("command", self.command), stream_name="critic"),
                self._scale_obs_component("task_flag", self._clip_obs_component("task_flag", task_flag_obs), stream_name="critic"),
                self._scale_obs_component("height_cmd", self._clip_obs_component("height_cmd", height_cmd), stream_name="critic"),
                self._scale_obs_component(
                    "root_ang_vel_b",
                    self._clip_obs_component("root_ang_vel_b", self.robot.data.root_ang_vel_b),
                    stream_name="critic",
                ),
                self._scale_obs_component(
                    "projected_gravity_b",
                    self._clip_obs_component("projected_gravity_b", self.robot.data.projected_gravity_b),
                    stream_name="critic",
                ),
                self._scale_obs_component("joint_pos", self._clip_obs_component("joint_pos", priv_joint_pos_obs), stream_name="critic"),
                clip_priv_joint_vel,
                self._scale_obs_component("actions", self._clip_obs_component("actions", self._actions), stream_name="critic"),
            ], dim=-1)
            self.priv_latent = torch.cat([
                self._scale_obs_component(
                    "root_lin_vel_b",
                    self._clip_obs_component("root_lin_vel_b", self.robot.data.root_lin_vel_b),
                    stream_name="critic",
                ),
                self._scale_obs_component("obs_height", self._clip_obs_component("obs_height", obs_height), stream_name="critic"),
                *critic_extra_obs,
                self._get_privileged_extra_obs(root_quat_inv),
            ], dim=-1)
            self.priv = torch.nan_to_num(self.priv, nan=0.0, posinf=0.0, neginf=0.0)
            self.priv_latent = torch.nan_to_num(self.priv_latent, nan=0.0, posinf=0.0, neginf=0.0)
            self.priv = torch.cat([
                self.priv,
                self.priv_latent,
            ],dim=-1)
            observations['critic'] = self.priv.clone()
            observations['prev_critic'] = self.prev_priv.clone()
            self.prev_priv = self.priv.clone()

        # ========== 历史帧 (History Frame) - Sim2Sim 兼容 ==========
        # Sim2Sim 索引: [0:D)=最旧, [D:2D)=..., [(N-1)D:ND)=当前帧(最新)
        # 先 append 当前帧，再 stack，顺序为 [oldest, ..., newest]
        if self.obs_history is not None:
            self.obs_history.append(self.obs.clone())
            obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # (B, T, K)
            observations['policy_hist'] = obs_buf_all.reshape(self.num_envs, -1)  # (B, T*K)

        if self.critic_history is not None and self.cfg.state_space:
            self.critic_history.append(self._get_critic_history_frame().clone())
            critic_buf_all = torch.stack([self.critic_history[i] for i in range(self.critic_history.maxlen)], dim=1)  # (B, T, K)
            observations['critic_hist'] = critic_buf_all.reshape(self.num_envs, -1)
            # 若 state_space 配置为堆叠维度，则 critic 使用多帧堆叠（与 env_jump.py 对齐）
            stacked_dim = self.critic_history.maxlen * self._critic_history_frame_dim
            if self.cfg.state_space == stacked_dim:
                observations['critic'] = critic_buf_all.reshape(self.num_envs, -1)

        obs_space_cfg = getattr(self.cfg, "observation_space", None)
        if isinstance(obs_space_cfg, dict):
            if obs_space_cfg.get("lin_vel", 0) > 0:
                observations["lin_vel"] = self.robot.data.root_lin_vel_b.clone() * self.cfg.lin_vel_scale
            if obs_space_cfg.get("height", 0) > 0:
                observations["height"] = obs_height.clone() * self.cfg.height_scale
        # Sim2Sim: policy = N 帧堆叠 [最旧...最新]，维度 N*D（非 [当前|历史] 的 (N+1)*D）
        if hasattr(self.cfg, 'use_frame_stack') and self.cfg.use_frame_stack and self.obs_history is not None:
            if self.frame_mask_probs is not None and not getattr(self.cfg, 'play', False):
                obs_for_policy = self._apply_frame_mask(obs_buf_all)
            else:
                obs_for_policy = obs_buf_all
            observations['policy'] = obs_for_policy.reshape(self.num_envs, -1)
        else:
            observations['policy'] = self.obs.clone()
        if isinstance(obs_space_cfg, dict) and "prev_policy" in obs_space_cfg:
            observations["prev_policy"] = self.prev_obs.clone()
        self.prev_obs = self.obs.clone()
        if isinstance(obs_space_cfg, dict):
            if "priv_latent" in obs_space_cfg:
                # observations["priv_latent"] = self._pad_flat_features(
                #     self.priv_latent,
                #     int(obs_space_cfg["priv_latent"]),
                # )
                observations["priv_latent"] = self.priv_latent
            if "on_constraint" in obs_space_cfg:
                policy_hist = observations.get("policy_hist")
                if policy_hist is None:
                    hist_frames = int(getattr(self.cfg, "num_obs_hist", 1) or 1)
                    policy_hist = observations["policy"].repeat(1, hist_frames)
                on_constraint = torch.cat(
                    [
                        observations["policy"],
                        self.priv_latent,
                        policy_hist,
                    ],
                    dim=-1,
                )
                # observations["on_constraint"] = self._pad_flat_features(
                #     torch.nan_to_num(on_constraint, nan=0.0, posinf=0.0, neginf=0.0),
                #     int(obs_space_cfg["on_constraint"]),
                # )
                observations["on_constraint"] = on_constraint
                self.extras["costs"] = self._get_legacy_costs(obs_height)
        if bool(getattr(self.cfg, "np3o_barlow_enabled", False)):
            self.extras["costs"] = self._get_np3o_costs(obs_height)
        self._debug_print_observation_stats(observations)
        # print(observations['policy'][0,28:35])
        # print(observations['policy'][0,3:4])
        # print(observations['critic'][0,31:32])
        # print(self.joint_pos[:,self._spring_idx])
        return observations

    def get_privilaged_obs(self) -> torch.Tensor:
        self._update_ground_height_estimate()
        height_cmd = self._get_observation_height_cmd().clone().unsqueeze(-1)
        task_flag_obs = self._get_task_flag_obs_raw()
        joint_pos_obs = (self.joint_pos - self.default_joint_pos)[:,self._actuate_idx]
        joint_vel_obs = self.joint_vel[:,self._actuate_idx]
        root_quat_inv, wheel_pos_b, wheel_pos_heading_b, _, _ = (
            self._get_root_quat_inv_and_wheel_pos_b()
        )
        obs_height = self._get_observed_height(wheel_pos_b).unsqueeze(-1)
        self.wheel_relative_ground_heights = self._get_wheel_relative_ground_heights_raw()
        leg_dim = self._leg_action_dim
        priv_joint_pos_obs = self._encode_joint_pos_obs(joint_pos_obs)
        critic_extra_obs_blocks = self._get_critic_extra_obs_blocks(root_quat_inv)
        critic_extra_obs = [
            self._scale_obs_component(
                key,
                self._clip_obs_component(key, value),
                stream_name="critic",
            )
            for key, value in critic_extra_obs_blocks.items()
        ]
        clip_priv_joint_vel = torch.cat(
            [
                self._scale_obs_component(
                    "joint_vel_leg",
                    self._clip_obs_component("joint_vel_leg", joint_vel_obs[:, :leg_dim]),
                    stream_name="critic",
                ),
                self._scale_obs_component(
                    "joint_vel_wheel",
                    self._clip_obs_component("joint_vel_wheel", joint_vel_obs[:, leg_dim:]),
                    stream_name="critic",
                ),
            ],
            dim=-1,
        )
        priv_obs = torch.cat([
                self._scale_obs_component("command", self._clip_obs_component("command", self.command), stream_name="critic"),
                self._scale_obs_component("task_flag", self._clip_obs_component("task_flag", task_flag_obs), stream_name="critic"),
                self._scale_obs_component("height_cmd", self._clip_obs_component("height_cmd", height_cmd), stream_name="critic"),
                self._scale_obs_component(
                    "root_ang_vel_b",
                    self._clip_obs_component("root_ang_vel_b", self.robot.data.root_ang_vel_b),
                    stream_name="critic",
                ),
                self._scale_obs_component(
                    "projected_gravity_b",
                    self._clip_obs_component("projected_gravity_b", self.robot.data.projected_gravity_b),
                    stream_name="critic",
                ),
                self._scale_obs_component("joint_pos", self._clip_obs_component("joint_pos", priv_joint_pos_obs), stream_name="critic"),
                clip_priv_joint_vel,
                self._scale_obs_component("actions", self._clip_obs_component("actions", self._actions), stream_name="critic"),
                self._scale_obs_component(
                    "root_lin_vel_b",
                    self._clip_obs_component("root_lin_vel_b", self.robot.data.root_lin_vel_b),
                    stream_name="critic",
                ),
                self._scale_obs_component("obs_height", self._clip_obs_component("obs_height", obs_height), stream_name="critic"),
                *critic_extra_obs,
                self._get_privileged_extra_obs(root_quat_inv),
            ], dim=-1)
        return torch.nan_to_num(priv_obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_rear2_rear1_joint_limit_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        rear2_joint_idx = self._rear2_joint_idx
        if len(rear2_joint_idx) == 0:
            return zeros, zeros, zeros

        lower = float(getattr(self.cfg, "rear2_rear1_joint_limit_lower", 45.0 / 180.0 * torch.pi))
        upper = float(getattr(self.cfg, "rear2_rear1_joint_limit_upper", -3.0 / 180.0 * torch.pi))
        if upper <= lower:
            return zeros, zeros, zeros

        boundary_ratio = float(getattr(self.cfg, "rear2_rear1_joint_limit_boundary_ratio", 0.03))
        boundary_ratio = min(max(boundary_ratio, 0.0), 0.49)
        limit_span = upper - lower
        margin = limit_span * boundary_ratio
        norm = max(margin, 1.0e-6)
        soft_lower = lower + margin
        soft_upper = upper - margin

        rear2_pos = wrap_to_pi(self.joint_pos[:, rear2_joint_idx])
        lower_over = torch.clamp(soft_lower - rear2_pos, min=0.0)
        upper_over = torch.clamp(rear2_pos - soft_upper, min=0.0)
        pos_penalty = torch.sum(torch.square((lower_over + upper_over) / norm), dim=-1)

        lower_band = torch.clamp((soft_lower - rear2_pos) / norm, min=0.0)
        upper_band = torch.clamp((rear2_pos - soft_upper) / norm, min=0.0)

        rear2_torque = self.robot.data.applied_torque[:, rear2_joint_idx]
        outward_torque = torch.clamp(-rear2_torque, min=0.0) * lower_band
        outward_torque += torch.clamp(rear2_torque, min=0.0) * upper_band
        torque_penalty = torch.sum(torch.square(outward_torque), dim=-1)

        rear2_vel = self.joint_vel[:, rear2_joint_idx]
        outward_vel = torch.clamp(-rear2_vel, min=0.0) * lower_band
        outward_vel += torch.clamp(rear2_vel, min=0.0) * upper_band
        vel_penalty = torch.sum(torch.square(outward_vel), dim=-1)

        return pos_penalty, torque_penalty, vel_penalty

    def _get_rewards(self) -> torch.Tensor:
        self._update_ground_height_estimate()
        # prepare
        root_quat_inv, wheel_pos_b, wheel_pos_heading_b, wheel_lin_vel_b, wheel_lin_vel_heading_b = (
            self._get_root_quat_inv_and_wheel_pos_b()
        )
        left_leg_length = torch.norm(wheel_pos_heading_b[:,0],dim=-1)
        right_leg_length = torch.norm(wheel_pos_heading_b[:,1],dim=-1)
        left_leg_angle = torch.atan2(-wheel_pos_heading_b[:,0,0],-wheel_pos_heading_b[:,0,2])
        right_leg_angle = torch.atan2(-wheel_pos_heading_b[:,1,0],-wheel_pos_heading_b[:,1,2])
        left_wheel_pos_b_normal = wheel_pos_heading_b[:,0]/left_leg_length.unsqueeze(-1)
        right_wheel_pos_b_normal = wheel_pos_heading_b[:,1]/right_leg_length.unsqueeze(-1)
        left_leg_length_dot = torch.sum(wheel_lin_vel_heading_b[:,0]*left_wheel_pos_b_normal,dim=-1)
        right_leg_length_dot = torch.sum(wheel_lin_vel_heading_b[:,1]*right_wheel_pos_b_normal,dim=-1)
        # leg_ang_vel = self.joint_vel[:,self._front1_joint_idx] - self.joint_vel[:,self._rear1_joint_idx]

        wheel_power = self.robot.data.applied_torque[:,self._wheel_idx]*self.joint_vel[:,self._wheel_idx]
        comsume_wheel_power = torch.clamp(wheel_power,min=0)

        # both_height_cmd = self.height_cmd.unsqueeze(-1).repeat((1,2))
        # deviation_joints = self._inverse_kinematics(both_height_cmd,torch.zeros_like(both_height_cmd,dtype=torch.float,device=self.device)).transpose(-2,-1).reshape(self.num_envs,-1)[:,:len(self._deviation_joint_idx)]

        # 起身训练：整个落地+起身阶段速度命令和角速度命令强制为0（reward 计算用）
        # if getattr(self.cfg, 'enable_standup_training', False):
        #     in_standup_phase = (self._standup_phase == 0) | (self._standup_phase == 1)
        #     self.command[in_standup_phase, :3] = 0.0

        # alive
        rew_termination = self.reset_terminated
        rew_epi_len = self.episode_length_buf

        # regularization
        joint_acc = self.robot.data.joint_acc
        rew_joint_acc = torch.sum(torch.square(joint_acc[:,self._actuate_idx]), dim=1)
        rew_leg_joint_acc = torch.sum(torch.square(joint_acc[:,self._legs_act_idx]), dim=1)
        rew_leg_joint_com_acc = torch.sum(torch.square(0.5*(joint_acc[:,self._legs_front_idx]+joint_acc[:,self._legs_rear_idx])), dim=1)
        rew_leg_joint_diff_acc = torch.sum(torch.square(0.5*(joint_acc[:,self._legs_front_idx]-joint_acc[:,self._legs_rear_idx])), dim=1)
        rew_wheel_acc = torch.sum(torch.square(joint_acc[:,self._wheel_idx]), dim=1)
        rew_wheel_com_acc = torch.square(0.5*(joint_acc[:,self._wheel_idx[0]]+joint_acc[:,self._wheel_idx[1]]))
        rew_wheel_diff_acc = torch.square(0.5*(joint_acc[:,self._wheel_idx[0]]-joint_acc[:,self._wheel_idx[1]]))
        rew_joint_vel = torch.sum(torch.square(self.joint_vel[:,self._actuate_idx]), dim=1)
        rew_leg_joint_vel = torch.sum(torch.square(self.joint_vel[:,self._legs_act_idx]), dim=1)
        rew_leg_joint_com_vel = torch.sum(torch.square(0.5*(self.joint_vel[:,self._legs_front_idx]+self.joint_vel[:,self._legs_rear_idx])), dim=1)
        rew_leg_joint_diff_vel = torch.sum(torch.square(0.5*(self.joint_vel[:,self._legs_front_idx]-self.joint_vel[:,self._legs_rear_idx])), dim=1)
        left_leg_pair_idx, right_leg_pair_idx = self._left_right_leg_joint_pair_idx
        if left_leg_pair_idx and right_leg_pair_idx:
            leg_pair_pos_diff = wrap_to_pi(self.joint_pos[:, left_leg_pair_idx] - self.joint_pos[:, right_leg_pair_idx])
            rew_leg_joint_pair_pos_diff = torch.sum(torch.square(leg_pair_pos_diff), dim=-1)
        else:
            rew_leg_joint_pair_pos_diff = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        rew_wheel_vel = torch.sum(torch.square(self.joint_vel[:,self._wheel_idx]), dim=-1)
        rew_wheel_com_vel = torch.square(0.5*(self.joint_vel[:,self._wheel_idx[0]]+self.joint_vel[:,self._wheel_idx[1]]))
        rew_wheel_diff_vel = torch.square(0.5*(self.joint_vel[:,self._wheel_idx[0]]-self.joint_vel[:,self._wheel_idx[1]]))
        applied_torque = self.robot.data.applied_torque
        rew_joint_torque = torch.sum(torch.square(applied_torque[:,self._actuate_idx]), dim=1)
        torque_rate = applied_torque - self._previous_applied_torque
        torque_second_diff = (
            applied_torque
            - 2.0 * self._previous_applied_torque
            + self._before_previous_applied_torque
        )
        rew_torque_rate = torch.sum(torch.square(torque_rate[:, self._actuate_idx]), dim=-1)
        rew_torque_smoothness_leg = torch.sum(
            torch.square(torque_second_diff[:, self._legs_act_idx]), dim=-1
        )
        rew_torque_smoothness_wheel = torch.sum(
            torch.square(torque_second_diff[:, self._wheel_idx]), dim=-1
        )
        self._before_previous_applied_torque.copy_(self._previous_applied_torque)
        self._previous_applied_torque.copy_(applied_torque)
        wheel_power = applied_torque[:,self._wheel_idx]*self.joint_vel[:,self._wheel_idx]
        comsume_wheel_power = torch.clamp(wheel_power,min=0)
        rew_wheel_power = torch.sum(comsume_wheel_power, dim=-1)
        # if self.cfg.use_leg_length_as_height:
        #     rew_lin_vel_z = torch.square(left_leg_length_dot+right_leg_length_dot)
        # else:
        rew_leg_len_vel = torch.square(left_leg_length_dot) + torch.square(right_leg_length_dot)
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_lin_vel_z_exp = torch.exp(-rew_lin_vel_z / self.cfg.lin_vel_z_sigma)
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        rew_action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_second_diff = self._actions - 2 * self._previous_actions + self._before_previous_actions
        rew_action_smoothness = torch.sum(torch.square(action_second_diff), dim=-1)
        leg_action_slice, wheel_action_slice = self._get_policy_action_slices()
        rew_action_rate_leg = torch.sum(torch.square(self._actions[:, leg_action_slice] - self._previous_actions[:, leg_action_slice]), dim=1)
        rew_action_rate_wheel = torch.sum(torch.square(self._actions[:, wheel_action_slice] - self._previous_actions[:, wheel_action_slice]), dim=1)
        rew_action_smoothness_leg = torch.sum(torch.square(action_second_diff[:, leg_action_slice]), dim=-1)
        rew_action_smoothness_wheel = torch.sum(torch.square(action_second_diff[:, wheel_action_slice]), dim=-1)
        # rew_joint_deviation_l2 = torch.sum(torch.square(self.joint_pos[:,self._deviation_joint_idx]-deviation_joints),dim=-1)
        # rew_joint_deviation_l1 = torch.sum(torch.abs(self.joint_pos[:,self._deviation_joint_idx]-deviation_joints),dim=-1)

        # print(self.robot.data.root_lin_vel_b[:, 0])
        # print(self.robot.data.applied_torque[:,self._wheel_idx])

        # tasks
        reward_command = self.command
        tracking_command = reward_command
        stand_still_lin_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        stand_still_yaw_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if bool(getattr(self.cfg, "stand_still_deadzone_enabled", False)):
            stand_still_threshold = max(
                float(getattr(self.cfg, "stand_still_deadzone_threshold", 0.2)),
                0.0,
            )
            stand_still_lin_mask = torch.abs(reward_command[:, 0]) < stand_still_threshold
            stand_still_yaw_mask = torch.abs(reward_command[:, 2]) < stand_still_threshold
            tracking_command = reward_command.clone()
            tracking_command[stand_still_lin_mask, 0] = 0.0
            tracking_command[stand_still_yaw_mask, 2] = 0.0

        pgb = self.robot.data.projected_gravity_b[:, :2]
        # pgb[:,0] = pgb[:,0]-0.02
        # rew_flat_orientation = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        rew_flat_orientation = torch.sum(torch.square(pgb), dim=-1)
        rew_flat_orientation_y = torch.square(self.cfg.orientation_y_square_sigma*pgb[:,0])
        rew_flat_orientation_x = torch.square(self.cfg.orientation_x_square_sigma*pgb[:,1])
        rew_flat_orientation_y_exp = torch.exp(-torch.square(pgb[:,0]) / self.cfg.orientation_y_exp_sigma)
        rew_flat_orientation_x_exp = torch.exp(-torch.square(pgb[:,1]) / self.cfg.orientation_x_exp_sigma)
        lin_vel_x_cmd = torch.square(reward_command[:,0])
        rew_flat_orientation_y_v = torch.square(
            (
                self.cfg.orientation_y_A * torch.exp(-lin_vel_x_cmd / self.cfg.orientation_y_sigma)
                + self.cfg.orientation_y_bias
            )
            * pgb[:, 0]
        )
        rew_flat_orientation_x_v = torch.square((self.cfg.orientation_x_A * torch.exp(-lin_vel_x_cmd/self.cfg.orientation_x_sigma) + self.cfg.orientation_x_bias)*pgb[:,1])
        # rew_flat_orientation = torch.square(self.robot.data.projected_gravity_b[:, 1])

        # ========== 姿态门控：只有接近水平时速度/角速度 tracking 奖励才高 ==========
        # upright_err: projected_gravity_b 的水平分量平方和，越小越水平
        upright_err = rew_flat_orientation
        if bool(getattr(self.cfg, "vel_upright_gate_enabled", False)):
            sigma = float(getattr(self.cfg, "vel_upright_gate_sigma", 0.02))
            sigma = max(sigma, 1e-6)
            vel_upright_gate = torch.exp(-upright_err / sigma)
        else:
            vel_upright_gate = 1.0

        vel_orientation_x_gate_enabled = self._get_vel_orientation_x_gate_enabled()
        if torch.is_tensor(vel_orientation_x_gate_enabled) or bool(vel_orientation_x_gate_enabled):
            orientation_x_angle = torch.asin(torch.clamp(torch.abs(pgb[:, 1]), max=1.0))
            full_angle_deg = torch.clamp(
                self._as_reward_gate_tensor(self._get_vel_orientation_x_gate_full_deg(), orientation_x_angle),
                min=0.0,
            )
            zero_angle_deg = torch.maximum(
                torch.clamp(
                    self._as_reward_gate_tensor(self._get_vel_orientation_x_gate_zero_deg(), orientation_x_angle),
                    min=0.0,
                ),
                full_angle_deg,
            )
            full_angle = torch.deg2rad(full_angle_deg)
            zero_angle = torch.deg2rad(zero_angle_deg)
            angle_band = zero_angle - full_angle
            vel_orientation_x_gate = torch.where(
                angle_band <= 0.0,
                (orientation_x_angle <= full_angle).float(),
                torch.clamp(
                    (zero_angle - orientation_x_angle) / torch.clamp(angle_band, min=1.0e-6),
                    min=0.0,
                    max=1.0,
                ),
            )
            if torch.is_tensor(vel_orientation_x_gate_enabled):
                vel_orientation_x_gate = torch.where(
                    self._as_reward_gate_bool_tensor(vel_orientation_x_gate_enabled, orientation_x_angle),
                    vel_orientation_x_gate,
                    torch.ones_like(vel_orientation_x_gate),
                )
        else:
            vel_orientation_x_gate = 1.0

        vel_orientation_y_gate_enabled = self._get_vel_orientation_y_gate_enabled()
        if torch.is_tensor(vel_orientation_y_gate_enabled) or bool(vel_orientation_y_gate_enabled):
            orientation_y_angle = torch.asin(torch.clamp(torch.abs(pgb[:, 0]), max=1.0))
            full_angle_deg = torch.clamp(
                self._as_reward_gate_tensor(self._get_vel_orientation_y_gate_full_deg(), orientation_y_angle),
                min=0.0,
            )
            zero_angle_deg = torch.maximum(
                torch.clamp(
                    self._as_reward_gate_tensor(self._get_vel_orientation_y_gate_zero_deg(), orientation_y_angle),
                    min=0.0,
                ),
                full_angle_deg,
            )
            full_angle = torch.deg2rad(full_angle_deg)
            zero_angle = torch.deg2rad(zero_angle_deg)
            angle_band = zero_angle - full_angle
            vel_orientation_y_gate = torch.where(
                angle_band <= 0.0,
                (orientation_y_angle <= full_angle).float(),
                torch.clamp(
                    (zero_angle - orientation_y_angle) / torch.clamp(angle_band, min=1.0e-6),
                    min=0.0,
                    max=1.0,
                ),
            )
            if torch.is_tensor(vel_orientation_y_gate_enabled):
                vel_orientation_y_gate = torch.where(
                    self._as_reward_gate_bool_tensor(vel_orientation_y_gate_enabled, orientation_y_angle),
                    vel_orientation_y_gate,
                    torch.ones_like(vel_orientation_y_gate),
                )
        else:
            vel_orientation_y_gate = 1.0

        if bool(getattr(self.cfg, "height_upright_gate_enabled", False)):
            height_sigma = float(getattr(self.cfg, "height_upright_gate_sigma", 0.02))
            height_sigma = max(height_sigma, 1e-6)
            height_upright_gate = torch.exp(-upright_err / height_sigma)
        else:
            height_upright_gate = 1.0

        # 相对地面高度（默认模式）/ 绝对或腿长高度模式（按当前 cfg 保持原定义）
        obs_height = self._get_observed_height(wheel_pos_b)
        if self._use_absolute_height() or self._use_leg_length_height():
            relative_obs_height = obs_height
        else:
            relative_obs_height = obs_height - self.ground_z_est
        wheel_relative_ground_heights = self._get_wheel_relative_ground_heights_raw()
        self.wheel_relative_ground_heights = wheel_relative_ground_heights
        wheel_height_w = self.robot.data.body_pos_w[:, self._wheel_link_idx, 2]
        height_reward_ref = self._get_height_reward_reference_height(relative_obs_height, wheel_height_w)
        # print(height_reward_ref)
        # 普通速度 tracking 仍按“水平前进速度”计算：
        # 使用机体系前向 x 速度，再通过机身 pitch 投影到水平面，避免直接拿机体系 x 导致冲坡时高估水平速度。
        root_rpy = euler_xyz_from_quat(self.robot.data.root_quat_w)
        pitch = wrap_to_pi(root_rpy[1])
        roll = wrap_to_pi(root_rpy[0])
        rew_flat_pitch_l1 = torch.abs(pitch*self.cfg.flat_pitch_l1_sigma)
        rew_flat_pitch_tanh = 1 - torch.tanh(torch.abs(pitch)/self.cfg.flat_pitch_tanh_sigma)
        rew_flat_roll_l1 = torch.abs(roll*self.cfg.flat_roll_l1_sigma)
        rew_flat_roll_tanh = 1 - torch.tanh(torch.abs(roll)/self.cfg.flat_roll_tanh_sigma)

        forward_lin_vel_horizontal = self.robot.data.root_lin_vel_b[:, 0] * torch.cos(pitch)
        lin_vel_err = tracking_command[:, 0] - forward_lin_vel_horizontal
        if self.cfg.lin_vel_err_constraint is not None:
            track_lin_vel_err = torch.square(torch.clamp(lin_vel_err,
                                                         min=-self.cfg.lin_vel_err_constraint,
                                                         max=self.cfg.lin_vel_err_constraint))
        else:
            track_lin_vel_err = torch.square(lin_vel_err)
        rew_track_lin_vel_xy = torch.exp(-track_lin_vel_err / self.cfg.lin_vel_xy_sigma)
        rew_track_lin_vel_xy_soft = torch.exp(-track_lin_vel_err / self.cfg.lin_vel_xy_soft_sigma)
        rew_track_lin_vel_xy_tight = torch.exp(-track_lin_vel_err / self.cfg.lin_vel_xy_tight_sigma)
        rew_track_lin_vel_xy_huge_gap = track_lin_vel_err>self.cfg.lin_vel_xy_torlarance_gap
        rew_track_lin_vel_xy_square = torch.square(
            lin_vel_err * self.cfg.lin_vel_xy_square_sigma
        )
        lin_vel_abs_err = torch.abs(tracking_command[:, 0]) - torch.abs(forward_lin_vel_horizontal)
        rew_pen_high_speed = torch.square(torch.clamp(lin_vel_abs_err, max=0.)*self.cfg.high_speed_pen_sigma)

        ang_vel_err = tracking_command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        if self.cfg.ang_vel_err_constraint is not None:
            track_yaw_rate_err = torch.square(torch.clamp(ang_vel_err,
                                                         min=-self.cfg.ang_vel_err_constraint,
                                                         max=self.cfg.ang_vel_err_constraint))
        else:
            track_yaw_rate_err = torch.square(ang_vel_err)
        rew_track_ang_vel_z = torch.exp(-track_yaw_rate_err / self.cfg.ang_vel_z_sigma)
        rew_track_ang_vel_z_soft = torch.exp(-track_yaw_rate_err / self.cfg.amg_vel_z_soft_sigma)
        rew_track_ang_vel_z_huge_gap = track_yaw_rate_err>self.cfg.ang_vel_z_torlarance_gap
        rew_track_ang_vel_z_square = torch.square(ang_vel_err*self.cfg.ang_vel_z_square_sigma)
        ang_vel_abs_err = torch.abs(tracking_command[:, 2]) - torch.abs(self.robot.data.root_ang_vel_b[:, 2])
        rew_pen_high_angVel = torch.square(torch.clamp(ang_vel_abs_err, max=0.)*self.cfg.high_angVel_pen_sigma)
        
        rew_stand_still = (
            torch.sum(torch.square(self.robot.data.root_lin_vel_b[:, :2]), dim=1)
            * stand_still_lin_mask.float()
            + torch.square(self.robot.data.root_ang_vel_b[:, 2]) * stand_still_yaw_mask.float()
        )
        rew_stand_still_lin_vel = (
            torch.sum(torch.abs(self.robot.data.root_lin_vel_b[:, :2]), dim=1)
            * stand_still_lin_mask.float()
        )

        # 约束加减速足端位置
        # foot_bound_square = torch.square(wheel_pos_heading_b[:,:,0])
        # rew_foot_bound = torch.sum(torch.abs(wheel_pos_heading_b[:,:,0])>self.cfg.foot_bound_dist,dim=1)
        # rew_foot_bound_square = torch.sum(torch.square(wheel_pos_heading_b[:,:,0]*self.cfg.foot_bound_square_sigma),dim=1)
        # rew_foot_bound_exp_pen = torch.sum(1-torch.exp(-foot_bound_square/self.cfg.foot_bound_exp_pen_sigma),dim=1)
        # rew_foot_bound_exp = torch.sum(torch.exp(-foot_bound_square/self.cfg.foot_bound_sigma),dim=1)
        # rew_foot_bound_ssquare = torch.sum(torch.square(foot_bound_square*self.cfg.foot_bound_ssquare_sigma),dim=1)

        rew_stand_nice = torch.sum(torch.square(wheel_pos_heading_b[:,:,0]),dim=-1)
        rew_no_fork_raw = torch.abs(wheel_pos_b[:,0,0]-wheel_pos_b[:,1,0])>self.cfg.no_fork_distance
        rew_no_fork = rew_no_fork_raw.float()
        # exp 平滑的 no_fork 约束：仅在超过阈值后逐渐加大惩罚（与 no_fork 的硬阈值互补）
        no_fork_dist = torch.abs(wheel_pos_b[:,0,0]-wheel_pos_b[:,1,0])
        no_fork_over = torch.clamp(no_fork_dist - self.cfg.no_fork_distance, min=0.0)
        rew_no_fork_exp = 1.0 - torch.exp(-no_fork_over / self.cfg.no_fork_exp_sigma)
        no_fork_z_distance = float(getattr(self.cfg, "no_fork_z_distance", self.cfg.no_fork_distance))
        rew_no_fork_z_raw = torch.abs(wheel_pos_b[:,0,2]-wheel_pos_b[:,1,2]) > no_fork_z_distance
        rew_no_fork_z = rew_no_fork_z_raw.float()
        no_fork_z_dist = torch.abs(wheel_pos_b[:,0,2]-wheel_pos_b[:,1,2])
        no_fork_z_over = torch.clamp(no_fork_z_dist - no_fork_z_distance, min=0.0)
        no_fork_z_exp_sigma = float(getattr(self.cfg, "no_fork_z_exp_sigma", self.cfg.no_fork_exp_sigma))
        rew_no_fork_z_exp = 1.0 - torch.exp(-no_fork_z_over / no_fork_z_exp_sigma)
        # 起身训练：落地阶段 no_fork 惩罚缩小
        # if getattr(self.cfg, 'enable_standup_training', False):
        #     landing_mask = (self._standup_phase == 0).float()
        #     no_fork_landing_scale = getattr(self.cfg, 'no_fork_landing_scale', 0.1)
        #     rew_no_fork = rew_no_fork_raw.float() * (1.0 - landing_mask * (1.0 - no_fork_landing_scale))
        # else:
        #     rew_no_fork = rew_no_fork_raw
        rew_no_fork_square = torch.square((wheel_pos_b[:,0,0]-wheel_pos_b[:,1,0])*self.cfg.no_fork_square_sigma)
        no_fork_z_square_sigma = float(getattr(self.cfg, "no_fork_z_square_sigma", self.cfg.no_fork_square_sigma))
        rew_no_fork_z_square = torch.square((wheel_pos_b[:,0,2]-wheel_pos_b[:,1,2]) * no_fork_z_square_sigma)
        # 兼容旧 reward key 名称：实际约束的是左右腿向量与世界重力方向对齐。
        wheel_motor_z_axis_align_err_sq = self._get_wheel_motor_z_axis_align_error_sq()
        wheel_motor_z_axis_align_sigma = max(
            float(
                getattr(
                    self.cfg,
                    "wheel_motor_z_axis_align_sigma",
                    getattr(self.cfg, "wheel_body_x_zero_sigma", 0.02),
                )
            ),
            1e-6,
        )
        wheel_motor_z_axis_align_tight_sigma = max(
            float(
                getattr(
                    self.cfg,
                    "wheel_motor_z_axis_align_tight_sigma",
                    getattr(self.cfg, "wheel_body_x_zero_tight_sigma", 0.002),
                )
            ),
            1e-6,
        )
        rew_wheel_motor_z_axis_align_exp = torch.exp(
            -wheel_motor_z_axis_align_err_sq / wheel_motor_z_axis_align_sigma
        )
        rew_wheel_motor_z_axis_align_exp_tight = torch.exp(
            -wheel_motor_z_axis_align_err_sq / wheel_motor_z_axis_align_tight_sigma
        )
        # rew_leg_end_vel = torch.sum(torch.square(wheel_lin_vel_b[:,:,0]),dim=-1)
        # rew_leg_angle_l2 = torch.sum(torch.square(torch.stack([left_leg_angle,right_leg_angle],dim=-1)),dim=-1)
        # rew_leg_angle_l1 = torch.sum(torch.abs(torch.stack([left_leg_angle,right_leg_angle],dim=-1)),dim=-1)
        # rew_leg_ang_vel_l2 = torch.sum(torch.square(leg_ang_vel),dim=-1)
        # rew_leg_ang_vel_l1 = torch.sum(torch.abs(leg_ang_vel),dim=-1)
        rew_l_leg_ang_exp = torch.exp(-torch.square(left_leg_angle)/self.cfg.l_leg_ang_exp_sigma)
        rew_r_leg_ang_exp = torch.exp(-torch.square(right_leg_angle)/self.cfg.r_leg_ang_exp_sigma)

        height_reward_target = self._get_height_reward_target_height()
        height_err = height_reward_ref - height_reward_target
        if self.cfg.height_err_constraint is not None:
            track_height_err = torch.square(torch.clamp(height_err,
                                                         min=-self.cfg.height_err_constraint,
                                                         max=self.cfg.height_err_constraint))
        else:
            track_height_err = torch.square(height_err)
        rew_track_height_square = torch.square(self.cfg.height_square_sigma * (height_err))
        rew_track_height_l1 = torch.abs(self.cfg.height_l1_sigma * (height_err))
        rew_track_height_exp = torch.exp(-track_height_err / self.cfg.height_sigma)
        rew_track_height_exp_soft = torch.exp(-track_height_err / self.cfg.height_soft_sigma)
        rew_track_height_exp_tight = torch.exp(-track_height_err / self.cfg.height_tight_sigma)
        rew_track_height_tanh = 1 - torch.tanh(torch.abs(height_err)/self.cfg.height_tanh_sigma)
        rew_pen_base_too_low = torch.square(torch.clamp((self.cfg.base_height_bound-height_reward_ref), min=0.)*self.cfg.pen_base_too_low_sigma)

        # 速度高度门控：高度没跟住时降低速度/角速度 tracking 奖励，避免 policy 为了追速度牺牲高度。
        vel_height_gate_enabled = self._get_vel_height_gate_enabled()
        # enabled 可以是全局 bool，也可以是 per-env bool tensor；tensor 用于不同任务/状态单独开关。
        if torch.is_tensor(vel_height_gate_enabled) or bool(vel_height_gate_enabled):
            vel_height_gate_mode = str(getattr(self.cfg, "vel_height_gate_mode", "exp")).lower()
            if vel_height_gate_mode in ("linear_band", "band", "piecewise_linear"):
                height_abs_err = torch.abs(height_reward_ref - height_reward_target)
                # full_error 内门控为 1；zero_error 外门控为 0；中间线性衰减。
                full_error = torch.clamp(
                    self._as_reward_gate_tensor(self._get_vel_height_gate_full_error(), height_abs_err),
                    min=0.0,
                )
                zero_error = torch.maximum(
                    torch.clamp(
                        self._as_reward_gate_tensor(self._get_vel_height_gate_zero_error(), height_abs_err),
                        min=0.0,
                    ),
                    full_error,
                )
                height_band = zero_error - full_error
                vel_height_gate = torch.where(
                    # 若 full_error == zero_error，则退化为硬阈值，避免除以 0。
                    height_band <= 0.0,
                    (height_abs_err <= full_error).float(),
                    torch.clamp(
                        (zero_error - height_abs_err) / torch.clamp(height_band, min=1.0e-6),
                        min=0.0,
                        max=1.0,
                    ),
                )
            else:
                # 指数模式：默认复用高度 tracking reward，也可单独配置 tracker sigma。
                vel_gate_sigma_cfg = getattr(self.cfg, "vel_height_gate_tracker_sigma", None)
                if vel_gate_sigma_cfg is None:
                    vel_height_gate = rew_track_height_exp
                else:
                    vel_gate_sigma = max(float(vel_gate_sigma_cfg), 1e-6)
                    vel_height_gate = torch.exp(-track_height_err / vel_gate_sigma)
            if torch.is_tensor(vel_height_gate_enabled):
                # 对未启用高度门控的 env，门控置 1，相当于不影响速度 tracking 奖励。
                vel_height_gate = torch.where(
                    self._as_reward_gate_bool_tensor(vel_height_gate_enabled, height_reward_ref),
                    vel_height_gate,
                    torch.ones_like(vel_height_gate),
                )
        else:
            vel_height_gate = 1.0

        # 叠加门控
        vel_reward_gate = vel_upright_gate * vel_orientation_x_gate * vel_orientation_y_gate * vel_height_gate
        rew_track_lin_vel_xy = rew_track_lin_vel_xy * vel_reward_gate
        rew_track_lin_vel_xy_soft = rew_track_lin_vel_xy_soft * vel_reward_gate
        rew_track_lin_vel_xy_tight = rew_track_lin_vel_xy_tight * vel_reward_gate
        rew_track_lin_vel_xy_square = rew_track_lin_vel_xy_square * vel_reward_gate
        rew_track_ang_vel_z = rew_track_ang_vel_z * vel_reward_gate
        rew_track_ang_vel_z_soft = rew_track_ang_vel_z_soft * vel_reward_gate
        rew_track_ang_vel_z_square = rew_track_ang_vel_z_square * vel_reward_gate
        rew_track_lin_vel_xy_huge_gap = rew_track_lin_vel_xy_huge_gap.float()
        rew_track_ang_vel_z_huge_gap = rew_track_ang_vel_z_huge_gap.float()

        rew_track_height_exp = rew_track_height_exp * height_upright_gate
        rew_track_height_exp_soft = rew_track_height_exp_soft * height_upright_gate
        rew_track_height_exp_tight = rew_track_height_exp_tight * height_upright_gate
        rew_track_height_huge_gap = height_reward_target - obs_height > self.cfg.height_torlarance_gap
        # print(torch.abs(wheel_pos_b[:,0,0]-wheel_pos_b[:,1,0]))
        # print('tar',self.height_cmd)
        # print('cur',obs_height)

        # undesired contacts
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        undesired_contact_force_threshold = float(
            getattr(self.cfg, "undesired_contact_force_threshold", 5.0)
        )
        desired_contact_force_threshold = float(
            getattr(self.cfg, "desired_contact_force_threshold", 1.0)
        )
        wheel_contact_force_peaks = self._get_wheel_contact_force_peaks(net_contact_forces)
        if wheel_contact_force_peaks.shape[1] >= 2:
            both_wheels_contact = torch.all(
                wheel_contact_force_peaks > desired_contact_force_threshold, dim=1
            )
        else:
            both_wheels_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rew_wheel_air_spin = self._get_wheel_air_spin_reward(
            wheel_contact_force_peaks,
            desired_contact_force_threshold,
        )
        rew_track_height_exp_both_wheels_contact = (
            rew_track_height_exp_soft * both_wheels_contact.float()
        )

        # 计算总的有害接触奖励 (用于训练)
        if net_contact_forces is not None:
            harmful_contact_mask = (
                torch.norm(net_contact_forces[:, :, self._undesired_contact_link_idx], dim=-1)
                > undesired_contact_force_threshold
            )
            is_contact = torch.max(torch.max(harmful_contact_mask, dim=1)[0], dim=1)[0]
            rew_undesired_contact = is_contact.float()
        else:
            rew_undesired_contact = torch.zeros(self.num_envs, device=self.device)

        self._debug_print_undesired_contacts(net_contact_forces)

        # [内置] 实时终端打印 (已移除)

        # desired contacts (车轮触地奖励/惩罚)
        if wheel_contact_force_peaks.shape[1] > 0:
            # 检查车轮是否在历史窗口内有接触
            is_not_contact = wheel_contact_force_peaks < desired_contact_force_threshold
            rew_desired_contact = torch.sum(is_not_contact.float(), dim=1)
        else:
            rew_desired_contact = torch.zeros(self.num_envs, device=self.device)

        # dof limits
        if bool(getattr(self.cfg, "front_rear_joint_limit_rewards_enabled", True)):
            limits_actions_joint_error = [(self.leg_actions[:,0]-self.leg_actions[:,1]).unsqueeze(-1),
                                  (self.leg_actions[:,2]-self.leg_actions[:,3]).unsqueeze(-1)]
            limits_actions_joint_error = torch.cat(limits_actions_joint_error,dim=-1)
            upper_error = limits_actions_joint_error > self.cfg.leg_front_rear_range[1]
            lower_error = limits_actions_joint_error < self.cfg.leg_front_rear_range[0]
            rew_actions_joint_limits = torch.sum(upper_error+lower_error,dim=-1)

            limits_joint_error = self.joint_pos[:,self._legs_front_idx] - self.joint_pos[:,self._legs_rear_idx]
            soft_range = [0.,0.]
            scale_range = abs(self.cfg.leg_front_rear_range[1]-self.cfg.leg_front_rear_range[0])*(1-self.cfg.soft_range_scale)
            soft_range[0] = self.cfg.leg_front_rear_range[0] + scale_range
            soft_range[1] = self.cfg.leg_front_rear_range[1] - scale_range
            upper_error = limits_joint_error > soft_range[1]
            lower_error = limits_joint_error < soft_range[0]
            rew_current_joint_limits = torch.sum(upper_error+lower_error,dim=-1)
        else:
            rew_actions_joint_limits = torch.zeros(self.num_envs, device=self.device)
            rew_current_joint_limits = torch.zeros(self.num_envs, device=self.device)

        (
            rew_rear2_rear1_joint_pos_limits,
            rew_rear2_rear1_joint_pos_limits_torque,
            rew_rear2_rear1_joint_pos_limits_vel,
        ) = self._get_rear2_rear1_joint_limit_terms()

        # 起身训练奖励（仅 phase=1 起身阶段非零）
        # in_standup = (self._standup_phase == 1) if getattr(self.cfg, 'enable_standup_training', False) else False
        # if getattr(self.cfg, 'enable_standup_training', False):
        #     standup_mask = in_standup.float()
        #     rew_standup_height = obs_height * standup_mask
        #     rew_standup_vel_z = torch.clamp(self.robot.data.root_lin_vel_b[:, 2], min=0.0) * standup_mask
        #     rew_standup_smoothness = rew_action_smoothness * standup_mask
        #     rew_standup_joint_torque = rew_joint_torque * standup_mask
        #     rew_standup_wheel_power = rew_wheel_power * standup_mask
        #     rew_standup_leg_joint_acc = rew_leg_joint_acc * standup_mask
        #     rew_standup_wheel_vel = rew_wheel_vel * standup_mask
        # else:
        #     rew_standup_height = torch.zeros(self.num_envs, device=self.device)
        #     rew_standup_vel_z = torch.zeros(self.num_envs, device=self.device)
        #     rew_standup_smoothness = torch.zeros(self.num_envs, device=self.device)
        #     rew_standup_joint_torque = torch.zeros(self.num_envs, device=self.device)
        #     rew_standup_wheel_power = torch.zeros(self.num_envs, device=self.device)
        #     rew_standup_leg_joint_acc = torch.zeros(self.num_envs, device=self.device)
        #     rew_standup_wheel_vel = torch.zeros(self.num_envs, device=self.device)

        reward_terms = {k[4:]: v for k, v in locals().items() if k.startswith("rew_")}
        reward_terms = self._postprocess_reward_terms(reward_terms)

        # gather rewards
        rewards = {k: v * self.step_dt for k, v in reward_terms.items()}
        # 用 .get() 防止 cfg.rewards 里存在但对应 rew_* 变量被注释掉的键（如 standup 相关）
        _zero_rew = torch.zeros(self.num_envs, device=self.device)
        rewards = {k: w * rewards.get(k, _zero_rew) for k, w in self.cfg.rewards.items()}

        # 数值保护：
        # 1. 将每个 reward term 内部产生的 NaN/Inf 直接归零，防止污染 rollout。
        # 2. 对本 step 已判定物理爆炸的 env，整条 reward 置零，避免坏状态参与 PPO 更新。
        bad_reward_envs = self._numerical_safety_reset_buf
        if bad_reward_envs.any():
            bad_reward_envs_f = bad_reward_envs.float()
            keep_reward_envs_f = (~bad_reward_envs).float()
        for key, value in rewards.items():
            sanitized_value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            if bad_reward_envs.any():
                sanitized_value = sanitized_value * keep_reward_envs_f
            rewards[key] = sanitized_value
        self._last_reward_terms = {key: value.detach() for key, value in rewards.items()}

        if bool(getattr(self.cfg, "reward_groups_enabled", False)):
            group_names = tuple(getattr(self.cfg, "reward_group_names", ("task", "safety", "energy")))
            group_index = {name: idx for idx, name in enumerate(group_names)}
            term_groups = dict(getattr(self.cfg, "reward_term_groups", {}))
            reward_groups = torch.zeros(
                self.num_envs,
                len(group_names),
                dtype=torch.float,
                device=self.device,
            )
            default_group = group_index.get("task", 0)
            for key, value in rewards.items():
                group_name = term_groups.get(key, "task")
                group_id = group_index.get(group_name, default_group)
                reward_groups[:, group_id] += value
            self.extras["reward_groups"] = torch.nan_to_num(
                reward_groups,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # Stepping command
        sync_cmd_iteration = getattr(self, "_sync_command_generator_training_iteration", None)
        if sync_cmd_iteration is not None:
            sync_cmd_iteration()
        self.command_generator.compute(self.step_dt)
        self._resample_custom_cmd(self.command_generator.command_counter)
        self.command = self.command_generator.command.clone()
        self._on_command_updated()
        self._apply_predefined_reset_air_command_limits()

        # Logging rewards
        for key, value in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[key] += value
        rewards_tensor = torch.stack([rewards[k] for k in self.cfg.rewards.keys()], dim=-1)
        total_reward = torch.sum(rewards_tensor, dim=-1)
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_total_reward = total_reward.detach()
        self._debug_print_reward_and_state_stats(rewards, total_reward)
        return total_reward

    def _postprocess_reward_terms(self, reward_terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Hook for task-specific reward shaping before weights and logs are applied."""
        return self.state_machine_manager.apply_reward_term_scales(self, reward_terms)

    def _debug_print_observation_stats(self, observations: dict) -> None:
        """按固定间隔打印观测统计，用于定位 value 爆炸输入源。"""
        if not self._value_debug_enabled:
            return
        if self._value_debug_print_interval <= 0:
            return
        if self._value_debug_step % self._value_debug_print_interval != 0:
            return
        policy_obs = observations.get("policy", self.obs)
        if self._value_debug_threshold_only:
            policy_max = self._tensor_max_abs(policy_obs)
            critic_max = self._tensor_max_abs(observations["critic"]) if "critic" in observations else 0.0
            obs_policy_th = float(self._value_debug_thresholds.get("obs_policy_abs", 0.0))
            obs_critic_th = float(self._value_debug_thresholds.get("obs_critic_abs", 0.0))
            policy_hit = obs_policy_th > 0.0 and policy_max > obs_policy_th
            critic_hit = ("critic" in observations) and (obs_critic_th > 0.0 and critic_max > obs_critic_th)
            if not (policy_hit or critic_hit):
                return
        self._print_tensor_stats("obs/policy", policy_obs)
        if "critic" in observations:
            self._print_tensor_stats("obs/critic", observations["critic"])

    def _debug_print_reward_and_state_stats(self, reward_terms: dict[str, torch.Tensor], total_reward: torch.Tensor) -> None:
        """按固定间隔打印奖励和关键状态统计。"""
        if not self._value_debug_enabled:
            return
        self._value_debug_step += 1
        if self._value_debug_print_interval <= 0:
            return
        if self._value_debug_step % self._value_debug_print_interval != 0:
            return
        if self._value_debug_threshold_only and not self._should_print_value_debug(reward_terms, total_reward):
            return

        topk = max(1, self._value_debug_topk)
        finite_mask = torch.isfinite(total_reward)
        finite_ratio = finite_mask.float().mean().item()
        if finite_mask.any():
            total_finite = total_reward[finite_mask]
            total_max = total_finite.max().item()
            total_min = total_finite.min().item()
            total_p99 = torch.quantile(total_finite, 0.99).item()
        else:
            total_max = float("nan")
            total_min = float("nan")
            total_p99 = float("nan")
        print(
            f"[ValueDebug][step={self._value_debug_step}] reward_total "
            f"max={total_max:.6f} min={total_min:.6f} p99={total_p99:.6f} finite_ratio={finite_ratio:.4f}"
        )

        for term_name in self.cfg.rewards.keys():
            term_tensor = reward_terms.get(term_name, None)
            if term_tensor is None:
                continue
            abs_val = torch.abs(term_tensor)
            finite_mask = torch.isfinite(abs_val)
            if finite_mask.any():
                abs_val = torch.where(finite_mask, abs_val, torch.full_like(abs_val, -1.0))
            k = min(topk, abs_val.numel())
            if k <= 0:
                continue
            top_vals, top_ids = torch.topk(abs_val, k=k, largest=True)
            ids_str = ",".join([str(i) for i in top_ids.tolist()])
            vals_str = ",".join([f"{v:.6f}" for v in top_vals.tolist()])
            max_abs = top_vals[0].item()
            print(
                f"[ValueDebug][step={self._value_debug_step}] reward_term={term_name} "
                f"max_abs={max_abs:.6f} top{topk}_env_ids=[{ids_str}] top{topk}_abs=[{vals_str}]"
            )

        # 关键状态
        wheel_vel = self.joint_vel[:, self._wheel_idx]
        root_lin_vel = self.robot.data.root_lin_vel_b
        root_ang_vel = self.robot.data.root_ang_vel_b
        applied_torque = self.robot.data.applied_torque
        self._print_tensor_stats("state/joint_vel_wheel", wheel_vel)
        self._print_tensor_stats("state/root_lin_vel_b", root_lin_vel)
        self._print_tensor_stats("state/root_ang_vel_b", root_ang_vel)
        self._print_tensor_stats("state/applied_torque", applied_torque)

    def _print_tensor_stats(self, name: str, tensor: torch.Tensor) -> None:
        """打印 max(abs)、mean、std、finite_ratio。"""
        flat = tensor.reshape(-1)
        finite_mask = torch.isfinite(flat)
        finite_ratio = finite_mask.float().mean().item()
        if finite_mask.any():
            finite_vals = flat[finite_mask]
            max_abs = torch.abs(finite_vals).max().item()
            mean = finite_vals.mean().item()
            std = finite_vals.std(unbiased=False).item()
        else:
            max_abs = float("nan")
            mean = float("nan")
            std = float("nan")
        print(
            f"[ValueDebug][step={self._value_debug_step}] {name} "
            f"max_abs={max_abs:.6f} mean={mean:.6f} std={std:.6f} finite_ratio={finite_ratio:.4f}"
        )

    def _tensor_max_abs(self, tensor: torch.Tensor) -> float:
        """返回张量有限值的绝对值最大值。"""
        flat = tensor.reshape(-1)
        finite_mask = torch.isfinite(flat)
        if not finite_mask.any():
            return float("nan")
        return torch.abs(flat[finite_mask]).max().item()

    def _should_print_value_debug(self, reward_terms: dict[str, torch.Tensor], total_reward: torch.Tensor) -> bool:
        """是否触发 ValueDebug 打印（超阈值触发）。"""
        thresholds = self._value_debug_thresholds
        # reward 类阈值
        reward_total_th = float(thresholds.get("reward_total_abs", 0.0))
        reward_term_th = float(thresholds.get("reward_term_abs", 0.0))
        if reward_total_th > 0.0 and self._tensor_max_abs(total_reward) > reward_total_th:
            return True
        if reward_term_th > 0.0:
            for term_name in self.cfg.rewards.keys():
                term_tensor = reward_terms.get(term_name, None)
                if term_tensor is not None and self._tensor_max_abs(term_tensor) > reward_term_th:
                    return True

        # state 类阈值
        state_checks = (
            ("state_joint_vel_wheel_abs", self.joint_vel[:, self._wheel_idx]),
            ("state_root_lin_vel_b_abs", self.robot.data.root_lin_vel_b),
            ("state_root_ang_vel_b_abs", self.robot.data.root_ang_vel_b),
            ("state_applied_torque_abs", self.robot.data.applied_torque),
        )
        for key, tensor in state_checks:
            th = float(thresholds.get(key, 0.0))
            if th > 0.0 and self._tensor_max_abs(tensor) > th:
                return True

        # obs 类阈值（与 _debug_print_observation_stats 保持一致）
        obs_policy_th = float(thresholds.get("obs_policy_abs", 0.0))
        obs_critic_th = float(thresholds.get("obs_critic_abs", 0.0))
        if obs_policy_th > 0.0 and self._tensor_max_abs(self.obs) > obs_policy_th:
            return True
        if obs_critic_th > 0.0 and self.cfg.state_space is not None and self._tensor_max_abs(self.priv) > obs_critic_th:
            return True
        return False

    def _apply_frame_mask(self, obs_buf: torch.Tensor) -> torch.Tensor:
        """按缓存的保留帧数遮蔽历史帧，仅保留最新的 k 帧，其余置零。

        frame_mask_probs = [p_keep_1, p_keep_2, ..., p_keep_M] (M <= T)
        - 概率 p_keep_j 表示该环境只保留最新 j 帧，前 T-j 帧置零
        - 剩余概率 1 - sum(p) 均匀分配给未指定的帧数（最后补到 T 帧全保留）
        - 仅在训练期间（cfg.play=False）调用
        """
        B, T, K = obs_buf.shape
        if self._frame_mask_num_keep is None:
            return obs_buf
        # 这里默认 obs_buf 是全体环境 (B == num_envs)。若不是，则退化为不遮蔽，避免错位。
        if B != self.num_envs:
            return obs_buf

        num_keep = self._frame_mask_num_keep.clamp(1, T)  # (B,)

        # 构建帧级掩码：只保留最新 num_keep 帧，[oldest ... newest]
        frame_idx = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
        keep_from = (T - num_keep).unsqueeze(1)                       # (B, 1)
        frame_mask = (frame_idx >= keep_from).float().unsqueeze(-1)   # (B, T, 1)
        return obs_buf * frame_mask

    def _resample_frame_mask_num_keep(self, env_ids: torch.Tensor | list[int] | None) -> None:
        """在 reset 时为指定环境采样一次保留帧数（episode 内固定）。"""
        if self.frame_mask_probs is None or self._frame_mask_num_keep is None or self.obs_history is None:
            return
        if getattr(self.cfg, "play", False):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        T = int(self.obs_history.maxlen)
        probs = list(self.frame_mask_probs)
        total = sum(probs)
        remaining_len = T - len(probs)
        if remaining_len > 0:
            remaining_prob = max(0.0, 1.0 - total)
            probs = probs + [remaining_prob / remaining_len] * remaining_len
        probs_tensor = torch.tensor(probs[:T], dtype=torch.float32, device=self.device)
        cum = probs_tensor.cumsum(0)  # (T,)

        # 采样每个环境保留的帧数：rand < cum[j] 意味着 num_keep = j+1
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        rand = torch.rand((env_ids_t.shape[0],), device=self.device)  # (E,)
        num_keep = (rand.unsqueeze(1) >= cum.unsqueeze(0)).sum(dim=1) + 1  # (E,)
        num_keep = num_keep.clamp(1, T)
        self._frame_mask_num_keep[env_ids_t] = num_keep

    def _clip_obs_component(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """按组件名对观测分量做限幅；支持标量(abs)或[min,max]。"""
        clip_cfg = self._obs_input_clip_cfg.get(key, None)
        if clip_cfg is None:
            return tensor
        if isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) == 2:
            low = float(clip_cfg[0])
            high = float(clip_cfg[1])
            return torch.clamp(tensor, low, high)
        limit = abs(float(clip_cfg))
        return torch.clamp(tensor, -limit, limit)

    def _scale_obs_component(
        self,
        key: str,
        tensor: torch.Tensor,
        *,
        stream_name: str = "policy",
    ) -> torch.Tensor:
        """Apply optional post-clip scaling to selected observation components."""
        if (
            not self._obs_input_scale_enabled
            or stream_name not in self._obs_input_scale_streams
        ):
            return tensor
        scale_cfg = self._obs_input_scale_cfg.get(key, None)
        if scale_cfg is None:
            return tensor
        if isinstance(scale_cfg, (list, tuple)):
            scale = torch.as_tensor(scale_cfg, dtype=tensor.dtype, device=tensor.device)
            if tensor.ndim > 1 and scale.numel() != tensor.shape[-1]:
                raise RuntimeError(
                    f"obs_input_scale_cfg['{key}'] has {scale.numel()} values, "
                    f"but component dim is {tensor.shape[-1]}"
                )
            return tensor * scale
        if isinstance(scale_cfg, Mapping):
            if key != "command":
                raise RuntimeError(f"obs_input_scale_cfg['{key}'] dict scales are only supported for command")
            scale_values = [
                self._get_scale_alias(scale_cfg, ("lin_vel_x")),
                self._get_scale_alias(scale_cfg, ("lin_vel_y")),
                self._get_scale_alias(scale_cfg, ("ang_vel_z")),
            ]
            scale = torch.as_tensor(scale_values, dtype=tensor.dtype, device=tensor.device)
            if tensor.ndim > 1 and scale.numel() != tensor.shape[-1]:
                raise RuntimeError(
                    f"obs_input_scale_cfg['{key}'] has {scale.numel()} command values, "
                    f"but component dim is {tensor.shape[-1]}"
                )
            return tensor * scale
        return tensor * float(scale_cfg)

    @staticmethod
    def _get_scale_alias(scale_cfg: Mapping, aliases: tuple[str, ...], default: float = 1.0) -> float:
        if isinstance(aliases, str):
            aliases = (aliases,)
        for alias in aliases:
            if alias in scale_cfg:
                return float(scale_cfg[alias])
        return default

    def _encode_joint_pos_obs(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Encode joint-position observations according to the experiment cfg."""
        encoding = str(getattr(self.cfg, "joint_pos_obs_encoding", "raw")).lower()
        if encoding in ("raw", "none", ""):
            return joint_pos
        if encoding == "sincos":
            return torch.cat([torch.sin(joint_pos), torch.cos(joint_pos)], dim=-1)
        raise RuntimeError(f"Unsupported joint_pos_obs_encoding: {encoding}")

    def _clip_scale_critic_component(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        return self._scale_obs_component(
            key,
            self._clip_obs_component(key, tensor),
            stream_name="critic",
        )

    def _pad_flat_features(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        flat = tensor.reshape(tensor.shape[0], -1)
        target_dim = int(target_dim)
        if flat.shape[-1] == target_dim:
            return flat
        if flat.shape[-1] > target_dim:
            return flat[:, :target_dim]
        pad = torch.zeros(
            flat.shape[0],
            target_dim - flat.shape[-1],
            dtype=flat.dtype,
            device=flat.device,
        )
        return torch.cat([flat, pad], dim=-1)

    def _select_entity_features(
        self,
        tensor: torch.Tensor | None,
        indices: Sequence[int],
        target_count: int,
        *,
        feature_dim: int | None = None,
    ) -> torch.Tensor:
        target_count = int(target_count)
        if target_count <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        if tensor is None:
            width = target_count * int(feature_dim or 1)
            return torch.zeros(self.num_envs, width, dtype=torch.float, device=self.device)

        tensor = torch.as_tensor(tensor, dtype=torch.float, device=self.device)
        if tensor.ndim < 2:
            return self._pad_flat_features(tensor, target_count * int(feature_dim or 1))

        valid_indices = [int(idx) for idx in indices if 0 <= int(idx) < tensor.shape[1]][:target_count]
        if valid_indices:
            selected = tensor[:, valid_indices]
        else:
            selected_shape = (self.num_envs, 0) + tuple(tensor.shape[2:])
            selected = torch.zeros(selected_shape, dtype=tensor.dtype, device=tensor.device)

        inferred_feature_dim = int(torch.tensor(selected.shape[2:]).prod().item()) if selected.ndim > 2 else 1
        width = target_count * int(feature_dim or inferred_feature_dim)
        return self._pad_flat_features(selected, width)

    def _resolve_name_patterns_to_indices(
        self,
        patterns: str | Sequence[str] | None,
        available_names: Sequence[str],
    ) -> list[int]:
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = (patterns,)

        import re

        indices: list[int] = []
        seen: set[int] = set()
        for pattern_str in patterns:
            if not pattern_str:
                continue
            pattern = re.compile(pattern_str)
            for idx, name in enumerate(available_names):
                if idx in seen:
                    continue
                if pattern.fullmatch(name) or pattern.search(name):
                    indices.append(idx)
                    seen.add(idx)
        return indices

    def _get_privileged_extra_body_indices(self) -> list[int]:
        cached = getattr(self, "_privileged_extra_body_indices", None)
        if cached is not None:
            return cached

        body_names = list(getattr(self.robot, "body_names", []))
        body_names_cfg = getattr(self.cfg, "privileged_extra_body_names", None)
        if body_names_cfg:
            indices = self._resolve_name_patterns_to_indices(body_names_cfg, body_names)
        else:
            name_to_idx = {name: idx for idx, name in enumerate(body_names)}
            requested_names = ["base_link"]
            requested_names.extend(getattr(self.cfg, "ordered_leg_body_names", ()))
            requested_names.extend(["left_wheel_link", "right_wheel_link", "gimbal_yaw_link", "gimbal_pitch_link"])

            indices: list[int] = []
            seen: set[int] = set()
            for name in requested_names:
                idx = name_to_idx.get(name)
                if idx is not None and idx not in seen:
                    indices.append(idx)
                    seen.add(idx)

        self._privileged_extra_body_indices = indices
        return indices

    def _get_privileged_extra_inertia_body_indices(self) -> list[int]:
        cached = getattr(self, "_privileged_extra_inertia_body_indices", None)
        if cached is not None:
            return cached

        body_names = list(getattr(self.robot, "body_names", []))
        inertia_body_names_cfg = getattr(self.cfg, "privileged_extra_inertia_body_names", None)
        if inertia_body_names_cfg:
            indices = self._resolve_name_patterns_to_indices(inertia_body_names_cfg, body_names)
        else:
            name_to_idx = {name: idx for idx, name in enumerate(body_names)}
            requested_names = ["base_link", "left_wheel_link", "right_wheel_link", "gimbal_yaw_link", "gimbal_pitch_link"]
            indices = [name_to_idx[name] for name in requested_names if name in name_to_idx]
        self._privileged_extra_inertia_body_indices = indices
        return indices

    def _get_robot_physx_view(self):
        return getattr(self.robot, "root_physx_view", None) or getattr(self.robot, "root_physics_view", None)

    def _get_body_masses_tensor(self) -> torch.Tensor | None:
        return torch.as_tensor(self.robot.root_physx_view.get_masses(), dtype=torch.float, device=self.device)

    def _get_body_mass_scale_tensor(self) -> torch.Tensor | None:
        masses = self._get_body_masses_tensor()
        default_masses = getattr(self.robot.data, "default_mass", None)
        if masses is None or default_masses is None:
            return None
        default_masses = torch.as_tensor(default_masses, dtype=torch.float, device=self.device)
        if default_masses.shape != masses.shape:
            return None
        mass_scale = masses / torch.clamp(default_masses, min=1.0e-6)
        return torch.nan_to_num(mass_scale, nan=1.0, posinf=1.0, neginf=1.0)

    def _get_delay_time_lag_obs(
        self,
        delay_attr: str,
        cfg_attr: str,
        enabled_attr: str,
    ) -> torch.Tensor:
        if not bool(getattr(self, enabled_attr, False)):
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        delay_buffers = getattr(self, delay_attr, None)
        delay_cfg = getattr(self.cfg, cfg_attr, None)
        if not delay_buffers or not delay_cfg:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)

        blocks = []
        for key in delay_cfg.keys():
            delay_buffer = delay_buffers.get(key)
            if delay_buffer is None:
                blocks.append(torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device))
                continue
            time_lags = torch.as_tensor(delay_buffer.time_lags, dtype=torch.float, device=self.device)
            blocks.append(time_lags.reshape(self.num_envs, 1))
        return torch.cat(blocks, dim=-1) if blocks else torch.zeros(
            self.num_envs, 0, dtype=torch.float, device=self.device
        )

    def _get_body_inertias_tensor(self) -> torch.Tensor | None:
        return torch.as_tensor(self.robot.root_physx_view.get_inertias(), dtype=torch.float, device=self.device)

    def _get_body_material_tensor(self) -> torch.Tensor | None:
        return torch.as_tensor(self.robot.root_physx_view.get_material_properties(), dtype=torch.float, device=self.device)

    def _get_scan_dot_obs(self) -> torch.Tensor:
        scan_dim = int(getattr(self.cfg, "n_scan", 0) or 0)
        if scan_dim <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        scanner = getattr(self, "dot_scanner", None)
        if scanner is None or not bool(getattr(self.cfg, "enable_scan_dot", False)):
            return torch.zeros(self.num_envs, scan_dim, dtype=torch.float, device=self.device)

        data = getattr(scanner, "data", None)
        ray_hits = getattr(data, "ray_hits_w", None)
        pos_w = getattr(data, "pos_w", None)
        if ray_hits is None or pos_w is None:
            return torch.zeros(self.num_envs, scan_dim, dtype=torch.float, device=self.device)

        scan = torch.clamp(pos_w[:, 2].unsqueeze(1) - ray_hits[..., 2], min=-1.0, max=1.0)
        scan = self._pad_flat_features(scan, scan_dim) * self.cfg.height_scale
        return torch.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_legacy_spring_state_obs(self) -> torch.Tensor:
        # state_dim = int(getattr(self.cfg, "n_state_est", 0) or 0)
        # spring_dim = max(state_dim - 4, 0)
        spring_dim = 2
        if spring_dim <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        spring_force = self.spring_force
        if spring_force is None:
            spring_force = torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        center = float(getattr(self.cfg, "constant_spring_force", 250.0))
        spring_obs = (spring_force - center) / 100.0
        return self._pad_flat_features(spring_obs, spring_dim)

    def _get_legacy_contact_force_obs(self) -> torch.Tensor:
        net_contact_forces = getattr(getattr(self.contact_sensor, "data", None), "net_forces_w_history", None)
        if net_contact_forces is None or len(self._desired_contact_link_idx) == 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        contact_force = net_contact_forces[:, 0, self._desired_contact_link_idx].reshape(self.num_envs, -1)
        return torch.nan_to_num((contact_force - 100.0) / 100.0, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_legacy_randomize_params_obs(self, target_dim: int) -> torch.Tensor:
        target_dim = int(target_dim)
        if target_dim <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        if not bool(getattr(self.cfg, "use_randomize_params_obs", False)):
            return torch.zeros(self.num_envs, target_dim, dtype=torch.float, device=self.device)

        eps = 1.0e-8
        default_stiffness = self.robot.data.default_joint_stiffness[:, self._actuate_idx].clamp_min(eps)
        default_damping = self.robot.data.default_joint_damping[:, self._actuate_idx].clamp_min(eps)
        joint_stiffness_scale = self.robot.data.joint_stiffness[:, self._actuate_idx] / default_stiffness
        joint_damping_scale = self.robot.data.joint_damping[:, self._actuate_idx] / default_damping
        joint_friction = getattr(self.robot.data, "joint_friction_coeff", None)
        if joint_friction is None:
            joint_friction = torch.zeros_like(joint_stiffness_scale)
        else:
            joint_friction = joint_friction[:, self._actuate_idx]

        masses = self._get_body_masses_tensor()
        if masses is None:
            masses = torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        body_com = getattr(self.robot.data, "body_com_pos_b", None)
        if body_com is None:
            body_com = torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        material = self._get_body_material_tensor()
        if material is None:
            material = torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)

        blocks = [
            self._get_legacy_contact_force_obs(),
            joint_stiffness_scale,
            joint_damping_scale,
            joint_friction,
            masses,
            body_com,
            material,
        ]
        randomize_obs = torch.cat([block.reshape(self.num_envs, -1) for block in blocks], dim=-1)
        randomize_obs = torch.nan_to_num(randomize_obs, nan=0.0, posinf=0.0, neginf=0.0)
        return self._pad_flat_features(randomize_obs, target_dim)

    def _get_legacy_priv_latent_obs(self, obs_height: torch.Tensor) -> torch.Tensor:
        state_dim = int(getattr(self.cfg, "n_state_est", 0) or 0)
        state_obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b * self.cfg.lin_vel_scale,
                obs_height * self.cfg.height_scale,
                self._get_legacy_spring_state_obs(),
            ],
            dim=-1,
        )
        if state_dim > 0:
            state_obs = self._pad_flat_features(state_obs, state_dim)
        else:
            state_obs = torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        priv_dim = int(getattr(self.cfg, "n_priv_latent", 0) or 0)
        return torch.cat([state_obs, self._get_legacy_randomize_params_obs(priv_dim)], dim=-1)

    def _compute_uses_legacy_privileged_extra_obs(self) -> bool:
        return (
            bool(getattr(self.cfg, "enable_scan_dot", False))
            or bool(getattr(self.cfg, "use_randomize_params_obs", False))
            or int(getattr(self.cfg, "n_state_est", 0) or 0) > 4
            or int(getattr(self.cfg, "n_priv_latent", 0) or 0) > 0
        )

    def _uses_legacy_privileged_extra_obs(self) -> bool:
        return bool(
            getattr(
                self,
                "_legacy_privileged_extra_obs_enabled",
                self._compute_uses_legacy_privileged_extra_obs(),
            )
        )

    def _get_legacy_privileged_extra_obs(self) -> torch.Tensor:
        extra_obs = torch.cat(
            [
                self._get_legacy_spring_state_obs(),
                self._get_legacy_randomize_params_obs(int(getattr(self.cfg, "n_priv_latent", 0) or 0)),
                self._get_scan_dot_obs(),
            ],
            dim=-1,
        )
        return torch.nan_to_num(extra_obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_critic_history_frame(self) -> torch.Tensor:
        target_dim = int(getattr(self, "_critic_history_frame_dim", 0) or 0)
        if target_dim <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        if self.priv.shape[-1] == target_dim:
            return self.priv

        est_dim = int(getattr(self.cfg, "n_state_est", 0) or 0)
        policy_dim = int(getattr(self.cfg, "num_single_obs", self.obs.shape[-1]) or self.obs.shape[-1])
        if est_dim > 0 and self.priv.shape[-1] - est_dim == target_dim:
            return torch.cat([self.priv[:, :policy_dim], self.priv[:, policy_dim + est_dim :]], dim=-1)
        return self._pad_flat_features(self.priv, target_dim)

    def _get_legacy_costs(self, obs_height: torch.Tensor) -> torch.Tensor:
        cfg_costs = getattr(self.cfg, "costs", None)
        if not isinstance(cfg_costs, dict) or len(cfg_costs) == 0:
            return self._get_np3o_costs(obs_height)

        costs: dict[str, torch.Tensor] = {}
        joint_vel = self.joint_vel[:, self._actuate_idx]
        joint_vel_limits = getattr(self.robot.data, "joint_vel_limits", None)
        if joint_vel_limits is not None:
            joint_vel_limits = joint_vel_limits[:, self._actuate_idx]
            costs["dof_vel_limit"] = torch.sum(torch.relu(torch.abs(joint_vel) - joint_vel_limits), dim=-1)
        else:
            costs["dof_vel_limit"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        torque = self.robot.data.applied_torque[:, self._actuate_idx]
        effort_limits = getattr(self.robot.data, "joint_effort_limits", None)
        if effort_limits is not None:
            effort_limits = effort_limits[:, self._actuate_idx]
            costs["torque_limit"] = torch.sum(torch.relu(torch.abs(torque) - effort_limits), dim=-1)
        else:
            costs["torque_limit"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        weighted = []
        for key, weight in cfg_costs.items():
            value = costs.get(key, torch.zeros(self.num_envs, dtype=torch.float, device=self.device))
            weighted.append(float(weight) * value * self.step_dt)
        if not weighted:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
        return torch.stack(weighted, dim=-1)

    def _get_dynamic_priv_obs(self, root_quat_inv: torch.Tensor) -> torch.Tensor:
        joint_count = self._privileged_extra_joint_count
        actuate_idx = self._privileged_extra_actuate_idx
        wheel_count = self._privileged_extra_wheel_count

        joint_stiffness = self._select_entity_features(
            getattr(self.robot.data, "joint_stiffness", None), actuate_idx, joint_count
        )
        joint_damping = self._select_entity_features(
            getattr(self.robot.data, "joint_damping", None), actuate_idx, joint_count
        )

        # spring_force = self.spring_force - 250.

        applied_torque = self._select_entity_features(
            self.robot.data.applied_torque, actuate_idx, joint_count
        )
        obs_delay_steps = self._get_delay_time_lag_obs("obs_delay", "obs_delay_cfg", "use_obs_delay")
        act_delay_steps = self._get_delay_time_lag_obs("act_delay", "act_delay_cfg", "use_act_delay")
        # joint_acc = self._select_entity_features(
        #     self.robot.data.joint_acc, actuate_idx, joint_count
        # )
        _, _, _, cached_wheel_lin_vel_b, _ = self._get_root_quat_inv_and_wheel_pos_b()
        wheel_lin_vel_b = torch.zeros(self.num_envs, wheel_count, 3, dtype=torch.float, device=self.device)
        valid_wheel_idx = self._privileged_extra_valid_wheel_idx
        if valid_wheel_idx:
            wheel_link_idx = [int(idx) for idx in self._wheel_link_idx]
            local_wheel_idx = [wheel_link_idx.index(int(idx)) for idx in valid_wheel_idx if int(idx) in wheel_link_idx]
            transformed = cached_wheel_lin_vel_b[:, local_wheel_idx]
            wheel_lin_vel_b[:, : len(local_wheel_idx)] = transformed
                
        wheel_contact_force = torch.zeros(self.num_envs, wheel_count, dtype=torch.float, device=self.device)
        wheel_contact_state = torch.zeros(self.num_envs, wheel_count, dtype=torch.float, device=self.device)
        net_contact_forces = getattr(getattr(self.contact_sensor, "data", None), "net_forces_w_history", None)
        contact_peaks = self._get_wheel_contact_force_peaks(net_contact_forces)
        if contact_peaks.numel() > 0:
            padded_contact_peaks = self._pad_flat_features(contact_peaks, wheel_count)
            wheel_contact_force = padded_contact_peaks - 100.
            desired_contact_force_threshold = float(
                getattr(self.cfg, "desired_contact_force_threshold", 1.0)
            )
            wheel_contact_state = (padded_contact_peaks > desired_contact_force_threshold).float()
        
        # body_com = self.robot.data.body_com_pos_b[:, self._privileged_extra_body_indices, :]

        blocks = [
            self._clip_scale_critic_component("joint_stiffness", joint_stiffness),
            self._clip_scale_critic_component("joint_damping", joint_damping),
            # self._clip_scale_critic_component("spring_force", spring_force),
            # self._clip_scale_critic_component("root_pos_w", self.robot.data.root_pos_w),
            # self._clip_scale_critic_component("root_lin_vel_w", self.robot.data.root_lin_vel_w),
            # self._clip_scale_critic_component("root_ang_vel_w", self.robot.data.root_ang_vel_w),
            self._clip_scale_critic_component("joint_torque", applied_torque),
            self._clip_scale_critic_component("obs_delay_steps", obs_delay_steps),
            # self._clip_scale_critic_component("joint_acc", joint_acc),
            self._clip_scale_critic_component("act_delay_steps", act_delay_steps),
            self._clip_scale_critic_component("wheel_body_lin_vel", wheel_lin_vel_b.reshape(self.num_envs, -1)),
            # self._clip_scale_critic_component("wheel_contact_force", wheel_contact_force),
            self._clip_scale_critic_component("wheel_contact_state", wheel_contact_state),
            # self._clip_scale_critic_component("body_com", body_com.reshape(self.num_envs, -1)),
        ]
        dynamic_obs = torch.cat(blocks, dim=-1)
        return dynamic_obs

    def _get_static_priv_obs(self) -> torch.Tensor:
        joint_count = self._privileged_extra_joint_count
        body_count = self._privileged_extra_body_count
        inertia_body_count = self._privileged_extra_inertia_body_count
        material_body_count = self._privileged_extra_material_body_count
        actuate_idx = self._privileged_extra_actuate_idx
        body_indices = self._privileged_extra_body_indices
        inertia_body_indices = self._privileged_extra_inertia_body_indices
        material_indices = self._privileged_extra_material_indices

        # joint_stiffness = self._select_entity_features(
        #     getattr(self.robot.data, "joint_stiffness", None), actuate_idx, joint_count
        # )
        # joint_damping = self._select_entity_features(
        #     getattr(self.robot.data, "joint_damping", None), actuate_idx, joint_count
        # )
        # joint_friction = self._select_entity_features(
        #     getattr(self.robot.data, "joint_friction_coeff", None), actuate_idx, joint_count
        # )

        body_masses = self._select_entity_features(
            self._get_body_masses_tensor(), body_indices, body_count
        )
        body_mass_scale = self._select_entity_features(
            self._get_body_mass_scale_tensor(), body_indices, body_count
        )

        # inertias = self._get_body_inertias_tensor()
        # if inertias is not None and inertias.ndim >= 3 and inertias.shape[-1] >= 9:
        #     inertias = inertias[..., [0, 4, 8]]
        # body_inertia_diag = self._select_entity_features(
        #     inertias, inertia_body_indices, inertia_body_count, feature_dim=3
        # )

        material = self._get_body_material_tensor()
        body_material = self._select_entity_features(
            material, material_indices, material_body_count, feature_dim=3
        )

        blocks = [
            # self._clip_scale_critic_component("joint_stiffness", joint_stiffness),
            # self._clip_scale_critic_component("joint_damping", joint_damping),
            # self._clip_scale_critic_component("joint_friction", joint_friction),
            # self._clip_scale_critic_component("body_mass", body_masses),
            self._clip_scale_critic_component("body_mass_scale", body_mass_scale),
            # self._clip_scale_critic_component("body_inertia_diag", body_inertia_diag),
            self._clip_scale_critic_component("body_material", body_material),
        ]
        static_obs = torch.cat(blocks, dim=-1)
        return static_obs

    def _get_privileged_extra_obs(self, root_quat_inv: torch.Tensor) -> torch.Tensor:
        # if self._uses_legacy_privileged_extra_obs():
        #     return self._get_legacy_privileged_extra_obs()
        if not self._privileged_extra_obs_enabled:
            return torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)

        dynamic_obs = self._get_dynamic_priv_obs(root_quat_inv)

        extra_obs = torch.cat([dynamic_obs, self.static_priv_obs], dim=-1)
        expected_dim = self._privileged_extra_obs_dim or extra_obs.shape[-1]
        extra_obs = self._pad_flat_features(extra_obs, expected_dim)
        return torch.nan_to_num(extra_obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _debug_obs_alert(self, blocks: dict[str, torch.Tensor], stream_name: str) -> None:
        """当原始观测任意分量超过阈值时，打印细粒度告警日志。"""
        threshold = self._obs_alert_threshold
        if threshold <= 0.0:
            return
        # 初始化阶段（step<=10）动作尚未稳定，跳过告警避免噪声
        if self._value_debug_step <= 10:
            return

        triggered = []
        for name, tensor in blocks.items():
            if tensor.numel() == 0:
                continue
            abs_tensor = torch.abs(tensor)
            max_abs = abs_tensor.max().item()
            if max_abs <= threshold:
                continue
            flat_abs = abs_tensor.reshape(-1)
            flat_raw = tensor.reshape(-1)
            max_flat_idx = int(torch.argmax(flat_abs).item())
            feat_dim = tensor.shape[-1] if tensor.ndim > 1 else 1
            env_id = max_flat_idx // feat_dim if tensor.ndim > 1 else max_flat_idx
            feat_id = max_flat_idx % feat_dim if tensor.ndim > 1 else 0
            triggered.append((name, max_abs, env_id, feat_id, flat_abs, flat_raw, feat_dim))

        if not triggered:
            return

        self._obs_alert_print_counter += 1
        if self._obs_alert_print_interval > 1 and (self._obs_alert_print_counter % self._obs_alert_print_interval != 0):
            return

        print(
            f"[ObsAlert][step={self._value_debug_step}][{stream_name}] "
            f"raw_input_exceeds_threshold={threshold:.3f}"
        )
        topk = max(1, self._obs_alert_topk)
        for name, max_abs, env_id, feat_id, flat_abs, flat_raw, feat_dim in triggered:
            k = min(topk, flat_abs.numel())
            top_vals, top_ids = torch.topk(flat_abs, k=k, largest=True)
            details = []
            for i in range(k):
                flat_idx = int(top_ids[i].item())
                detail_env = flat_idx // feat_dim if feat_dim > 1 else flat_idx
                detail_feat = flat_idx % feat_dim if feat_dim > 1 else 0
                raw_val = flat_raw[flat_idx].item()
                details.append(f"(env={detail_env},dim={detail_feat},raw={raw_val:.6f},abs={top_vals[i].item():.6f})")
            print(
                f"[ObsAlert][step={self._value_debug_step}][{stream_name}] "
                f"component={name} max_abs={max_abs:.6f} peak=(env={env_id},dim={feat_id}) top{topk}={details}"
            )

    @staticmethod
    def _per_env_max_abs_from_blocks(blocks: dict[str, torch.Tensor], num_envs: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (per_env_max_abs, per_env_has_nonfinite)。blocks 的 tensor 形状应以 env 维为第 0 维。"""
        per_env_max = torch.zeros(num_envs, device=device)
        per_env_nonfinite = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for tensor in blocks.values():
            if tensor is None or tensor.numel() == 0:
                continue
            if tensor.ndim == 1:
                per_env_nonfinite |= ~torch.isfinite(tensor)
                per_env_max = torch.maximum(per_env_max, torch.abs(tensor))
            else:
                per_env_nonfinite |= ~torch.isfinite(tensor).all(dim=-1)
                per_env_max = torch.maximum(per_env_max, torch.amax(torch.abs(tensor), dim=-1))
        return per_env_max, per_env_nonfinite

    def _apply_termination_duration(
        self,
        terminate: torch.Tensor,
        *,
        counter_attr: str = "_termination_duration_counter",
        raw_attr: str = "_termination_duration_raw_buf",
    ) -> torch.Tensor:
        """Require a terminate condition to persist for N consecutive control steps."""
        terminate = terminate.to(dtype=torch.bool, device=self.device)
        steps = max(int(getattr(self.cfg, "termination_duration_steps", 1)), 1)
        enabled = bool(getattr(self.cfg, "termination_duration_enabled", False)) and steps > 1

        raw_buf = getattr(self, raw_attr, None)
        if raw_buf is None or raw_buf.shape != terminate.shape:
            raw_buf = torch.zeros_like(terminate, dtype=torch.bool)
            setattr(self, raw_attr, raw_buf)
        raw_buf.copy_(terminate)

        counter = getattr(self, counter_attr, None)
        if counter is None or counter.shape != terminate.shape:
            counter = torch.zeros(terminate.shape, dtype=torch.int32, device=self.device)
            setattr(self, counter_attr, counter)
        if not enabled:
            counter.zero_()
            return terminate

        counter_next = torch.where(
            terminate,
            torch.clamp(counter + 1, max=steps),
            torch.zeros_like(counter),
        )
        setattr(self, counter_attr, counter_next)
        return counter_next >= steps

    def _clear_termination_duration_buffers(
        self,
        env_ids: torch.Tensor,
        *,
        counter_attr: str = "_termination_duration_counter",
        raw_attr: str = "_termination_duration_raw_buf",
    ) -> None:
        counter = getattr(self, counter_attr, None)
        if counter is not None:
            counter[env_ids] = 0
        raw_buf = getattr(self, raw_attr, None)
        if raw_buf is not None:
            raw_buf[env_ids] = False

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # contact termination
        reset_contact_force = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._reset_contact_link_idx], dim=-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminate = torch.any(reset_contact_force > 1.0, dim=-1)

        # ------------------------------------------------------------------
        # Numerical safety termination
        # - NaN/Inf in key states -> terminate immediately
        # - Large outliers in velocities -> terminate immediately
        # This prevents a single "bad" env from polluting rollout with NaNs.
        # ------------------------------------------------------------------
        joint_vel_act = self.joint_vel[:, self._actuate_idx]
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b

        nan_terminate = (
            ~torch.isfinite(joint_vel_act).all(dim=-1)
            | ~torch.isfinite(root_lin_vel_b).all(dim=-1)
            | ~torch.isfinite(root_ang_vel_b).all(dim=-1)
        )

        joint_vel_abs_th = float(getattr(self.cfg, "terminate_joint_vel_abs", 500))
        root_ang_vel_abs_th = float(getattr(self.cfg, "terminate_root_ang_vel_abs", 200.0))
        root_lin_vel_abs_th = float(getattr(self.cfg, "terminate_root_lin_vel_abs", 100.0))

        joint_vel_abs = torch.amax(torch.abs(joint_vel_act), dim=-1)
        root_ang_vel_abs = torch.amax(torch.abs(root_ang_vel_b), dim=-1)
        root_lin_vel_abs = torch.amax(torch.abs(root_lin_vel_b), dim=-1)

        outlier_terminate = (
            (joint_vel_abs > joint_vel_abs_th)
            | (root_ang_vel_abs > root_ang_vel_abs_th)
            | (root_lin_vel_abs > root_lin_vel_abs_th)
        )
        self._numerical_safety_reset_buf.copy_(nan_terminate | outlier_terminate)
        # 数值安全类终止（NaN/Inf + Outlier）必须立即生效，不受 termination_duration 限制
        immediate_terminate = nan_terminate | outlier_terminate
        # 3. 精准打印逻辑
        if bool(getattr(self.cfg, "debug_numerical_safety_print", False)) and outlier_terminate.any():
            # 获取触发异常的环境索引 (env_ids)
            outlier_indices = torch.where(outlier_terminate)[0]

            print(f"\n[!!!] 检测到 {len(outlier_indices)} 个环境数值爆炸:")
            for idx in outlier_indices:
                env_id = idx.item()
                print(f"--- Env ID: {env_id} ---")
                print(f"  关节速度最大值: {joint_vel_abs[env_id]:.4f} (阈值: {joint_vel_abs_th})")
                print(f"  机身角速度最大值: {root_ang_vel_abs[env_id]:.4f} (阈值: {root_ang_vel_abs_th})")
                print(f"  机身线速度最大值: {root_lin_vel_abs[env_id]:.4f} (阈值: {root_lin_vel_abs_th})")
            print("########################################################################")

        terminate |= immediate_terminate

        # ------------------------------------------------------------------
        # Observation safety termination
        # - raw obs / raw privileged obs 非有限或超过阈值 -> terminate
        # 说明：raw obs 在 _get_observations() 里缓存，确保包含 actions 等未 clip 的输入。
        # ------------------------------------------------------------------
        if self._terminate_on_obs_outlier and self._value_debug_step > 10:
            obs_abs_th = self._terminate_obs_abs if self._terminate_obs_abs > 0.0 else float(getattr(self.cfg, "debug_obs_alert_threshold", 0.0))
            obs_outlier_terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            if obs_abs_th > 0.0:
                obs_outlier_terminate |= self._obs_raw_policy_max_abs > obs_abs_th
                if self.cfg.state_space is not None:
                    obs_outlier_terminate |= self._obs_raw_critic_max_abs > obs_abs_th

            obs_outlier_terminate |= self._obs_raw_policy_has_nonfinite
            if self.cfg.state_space is not None:
                obs_outlier_terminate |= self._obs_raw_critic_has_nonfinite

            if obs_outlier_terminate.any():
                self._terminate_obs_print_counter += 1
                if self._terminate_obs_print_interval <= 1 or (self._terminate_obs_print_counter % self._terminate_obs_print_interval == 0):
                    bad_ids = torch.where(obs_outlier_terminate)[0]
                    print(f"\n[ObsReset] 检测到 {len(bad_ids)} 个环境 raw obs 异常，将触发 reset. th={obs_abs_th}")
                    print(f"[ObsReset] env_ids={bad_ids.detach().cpu().tolist()}")

            immediate_terminate |= obs_outlier_terminate
            terminate |= obs_outlier_terminate

        # orientation termination
        root_rpy = euler_xyz_from_quat(self.robot.data.root_quat_w)
        roll_limit_rad = torch.deg2rad(
            torch.tensor(float(getattr(self.cfg, "termination_roll_deg", 40.0)), device=self.device)
        )
        pitch_limit_rad = torch.deg2rad(
            torch.tensor(float(getattr(self.cfg, "termination_pitch_deg", 40.0)), device=self.device)
        )
        terminate |= torch.abs(wrap_to_pi(root_rpy[0])) > roll_limit_rad
        terminate |= torch.abs(wrap_to_pi(root_rpy[1])) > pitch_limit_rad
        if bool(getattr(self.cfg, "reset_heading_target_terminate_enabled", False)):
            reset_heading_target_threshold_rad = torch.deg2rad(
                torch.tensor(
                    float(getattr(self.cfg, "reset_heading_target_terminate_threshold_deg", 2.0)),
                    device=self.device,
                )
            )
            reset_heading_error = torch.abs(wrap_to_pi(self.robot.data.heading_w - self.reset_heading_target))
            terminate |= reset_heading_error > reset_heading_target_threshold_rad
        # 根据不同状态机覆盖终止需求
        terminate, time_out = self.state_machine_manager.apply_done_masks(
            self, terminate, time_out
        )
        # 可以设置终止连续条件，持续N步满足才真正终止
        # 注意：数值安全类终止（NaN/Inf/Outlier/观测异常）不受 duration 限制，立即生效，
        # 以避免 NaN 环境在延迟窗口内持续污染仿真和 PPO rollout。
        delayed_terminate = terminate & ~immediate_terminate
        delayed_terminate = self._apply_termination_duration(delayed_terminate)
        terminate = delayed_terminate | immediate_terminate

        # others
        if self.cfg.play is True and not bool(getattr(self.cfg, "play_keep_done_reset", False)):
            terminate = torch.zeros_like(terminate)

        return terminate, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # extra infos
        self.extras['termination_ids'] = env_ids
        self.extras['termination_privileged_obs'] = self.get_privilaged_obs()[env_ids]

        # 重置课程管理器
        if self.curriculum_manager is not None and len(env_ids) > 0:
            self.curriculum_manager.compute(env_ids)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._before_previous_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self._previous_applied_torque[env_ids] = 0.0
        self._before_previous_applied_torque[env_ids] = 0.0
        self.predefined_reset_ground_zero_torque_until_time[env_ids] = 0.0
        self._clear_predefined_reset_ground_command_override(env_ids)
        self.obs[env_ids] = 0.0
        self.prev_obs[env_ids] = 0.0
        # 清除终止持续条件的计数器和原始缓冲区
        self._clear_termination_duration_buffers(env_ids)
        # 数值不稳定状态清除
        if hasattr(self, "_numerical_safety_reset_buf"):
            self._numerical_safety_reset_buf[env_ids] = False
        if self.cfg.state_space:
            self.priv[env_ids] = 0.0
            self.prev_priv[env_ids] = 0.0

        # ========== 重置帧堆叠历史缓冲区 ==========
        # 对于重置的环境，清空其历史观测（填充零）
        if self.obs_history is not None:
            for i in range(self.obs_history.maxlen):
                    self.obs_history[i][env_ids] = 0.0
        if self.critic_history is not None:
            for i in range(self.critic_history.maxlen):
                self.critic_history[i][env_ids] = 0.0

        # ========== 帧屏蔽：reset 时采样一次 ==========
        self._resample_frame_mask_num_keep(env_ids)

        # Sample new commands
        self.command_generator.reset(env_ids)
        self._update_axis_aligned_reset_heading_mask(env_ids, use_env_origins=True)
        self.height_cmd[env_ids] = self._sample_height_command(env_ids, use_env_origins=True)
        self._latch_special_height_wave(env_ids)
        self._resample_height_command_special_modes(env_ids)
        self._clear_predefined_reset_air_command_limits(env_ids)

        # Reset robot obs
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        # self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._custom_reset_random(env_ids)
        self._apply_axis_aligned_reset_heading(env_ids) # 特定场景需要指定reset航向
        self._record_reset_heading_target(env_ids)
        self._sync_heading_command_target_to_reset_heading(env_ids)

        # Resample time lag
        if self.use_act_delay:
            for k, v in self.act_delay.items():
                resample_time_lag = torch.randint(
                    *(self.cfg.act_delay_cfg[k]), (len(env_ids),), dtype=torch.int, device=self.device
                )
                v.set_time_lag(resample_time_lag, env_ids)
        if self.use_obs_delay:
            for k, v in self.obs_delay.items():
                resample_time_lag = torch.randint(
                    *(self.cfg.obs_delay_cfg[k]), (len(env_ids),), dtype=torch.int, device=self.device
                )
                v.set_time_lag(resample_time_lag, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode/Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode/Reset/terminate"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode/Reset/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.state_machine_manager.append_reset_logs(self, extras, env_ids)
        self.extras["log"].update(extras)

        # 重置状态机
        self.state_machine_manager.on_reset(self, env_ids)
        self._update_state_machine_marker()

        # 轮前向扫描
        if self._is_wheel_forward_scan_enabled():
            self._update_wheel_forward_scan_marker()

        self._update_builtin_terrain_debug_marker()

        if self.curriculum_manager is not None and len(env_ids) > 0:
            curriculum_log = self.curriculum_manager.reset(env_ids)
            self.extras["log"].update(curriculum_log)

        self._invalidate_step_caches()


    def _get_randomize_params(self):
        params_dict = dict()
        params_dict["joint_stiffness"] = self.robot.data.joint_stiffness
        params_dict["joint_damping"] = self.robot.data.joint_damping
        params_dict["joint_armature"] = self.robot.data.joint_armature
        params_dict["joint_friction_coeff"] = self.robot.data.joint_friction_coeff
        params_dict["root_com_pos_w"] = self.robot.data.root_com_pos_w
        params_dict["root_com_quat_w"] = self.robot.data.root_com_quat_w
        params_dict["body_com_pos_w"] = self.robot.data.body_com_pos_w
        params_dict["body_com_quat_w"] = self.robot.data.body_com_quat_w
        params_dict["body_inertias"] = self.robot.root_physics_view.get_inertias() # [env,bodies,9]
        params_dict["body_masses"] = self.robot.root_physics_view.get_masses() # [env,bodies,1]
        params_dict["body_properties"] = self.robot.root_physics_view.get_material_porperties() # [env,bodies,3], static/dynamic friction, restitution
        return params_dict

    def _range_pair_as_float(self, range_pair, default=(0.0, 0.0)) -> tuple[float, float]:
        if range_pair is None:
            return float(default[0]), float(default[1])
        return float(range_pair[0]), float(range_pair[1])

    def _apply_root_state_uniform_vel_b(self, env_ids, pose_range, velocity_range):
        if len(env_ids) == 0:
            return
        if not isinstance(pose_range, Mapping):
            pose_range = {}
        if not isinstance(velocity_range, Mapping):
            velocity_range = {}

        root_states = self.robot.data.default_root_state[env_ids].clone()
        pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        pose_ranges = torch.tensor(
            [self._range_pair_as_float(pose_range.get(key), (0.0, 0.0)) for key in pose_keys],
            dtype=torch.float,
            device=self.device,
        )
        pose_samples = sample_uniform(
            pose_ranges[:, 0],
            pose_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )

        positions = root_states[:, 0:3] + self.terrain.env_origins[env_ids] + pose_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
        orientations = quat_mul(root_states[:, 3:7], orientations_delta)

        yaw_only_samples = pose_samples[:, 3:6].clone()
        yaw_only_samples[:, 0] = 0.0
        yaw_only_samples[:, 1] = 0.0
        orientations_vel_delta = quat_from_euler_xyz(
            yaw_only_samples[:, 0], yaw_only_samples[:, 1], yaw_only_samples[:, 2]
        )
        orientations_vel = quat_mul(root_states[:, 3:7], orientations_vel_delta)

        vel_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        vel_ranges = torch.tensor(
            [self._range_pair_as_float(velocity_range.get(key), (0.0, 0.0)) for key in vel_keys],
            dtype=torch.float,
            device=self.device,
        )
        vel_samples = sample_uniform(
            vel_ranges[:, 0],
            vel_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        vel_samples[:, :3] = quat_apply(orientations_vel, vel_samples[:, :3])
        velocities = root_states[:, 7:13] + vel_samples

        self.robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    def _get_reset_training_iteration(self) -> int:
        get_iteration = getattr(self, "_get_training_iteration", None)
        if callable(get_iteration):
            return max(int(get_iteration()), 0)
        return max(int(getattr(self, "_training_iteration", 0)), 0)

    def _get_active_predefined_reset_air_modes(self) -> list[Mapping]:
        predefined_reset_air = getattr(self.cfg, "predefined_reset_air", {})
        if not isinstance(predefined_reset_air, Mapping) or not bool(predefined_reset_air.get("enabled", False)):
            return []

        raw_modes = predefined_reset_air.get("modes", ())
        if raw_modes is None:
            return []
        if isinstance(raw_modes, Mapping):
            raw_modes = (raw_modes,)

        iteration = self._get_reset_training_iteration()
        active_modes: list[Mapping] = []
        for mode in raw_modes:
            if not isinstance(mode, Mapping):
                continue
            start = int(mode.get("iteration_start", 0))
            end = int(mode.get("iteration_end", -1))
            if iteration >= start and (end < 0 or iteration < end):
                active_modes.append(mode)
        return active_modes

    def _clear_predefined_reset_air_command_limits(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids = self._as_env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self.predefined_reset_air_command_limit_remaining_time[env_ids] = 0.0
        self.predefined_reset_air_command_limit_mode_id[env_ids] = -1
        self.predefined_reset_air_command_limit_last_update_step[env_ids] = -1
        self.predefined_reset_air_command_limit_lin_vel_x[env_ids] = torch.nan
        self.predefined_reset_air_command_limit_lin_vel_y[env_ids] = torch.nan
        self.predefined_reset_air_command_limit_ang_vel_z[env_ids] = torch.nan
        self.predefined_reset_air_command_limit_height[env_ids] = torch.nan

    @staticmethod
    def _parse_predefined_reset_air_limit_pair(command_limits: Mapping, key: str) -> tuple[float, float] | None:
        value = command_limits.get(key, None)
        if value is None:
            return None
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(f"predefined_reset_air command_limits['{key}'] must be a pair. Got: {value!r}")
        low = float(value[0])
        high = float(value[1])
        return min(low, high), max(low, high)

    def _set_predefined_reset_air_command_limits(
        self,
        env_ids: torch.Tensor,
        command_limits: Mapping | None,
        *,
        mode_idx: int,
    ) -> None:
        if env_ids.numel() == 0:
            return
        if not isinstance(command_limits, Mapping):
            self._clear_predefined_reset_air_command_limits(env_ids)
            return

        duration_s = max(float(command_limits.get("duration_s", 0.0)), 0.0)
        if duration_s <= 0.0:
            self._clear_predefined_reset_air_command_limits(env_ids)
            return

        self._clear_predefined_reset_air_command_limits(env_ids)
        self.predefined_reset_air_command_limit_remaining_time[env_ids] = duration_s
        self.predefined_reset_air_command_limit_mode_id[env_ids] = int(mode_idx)
        self.predefined_reset_air_command_limit_last_update_step[env_ids] = -1

        for key, buffer in (
            ("lin_vel_x", self.predefined_reset_air_command_limit_lin_vel_x),
            ("lin_vel_y", self.predefined_reset_air_command_limit_lin_vel_y),
            ("ang_vel_z", self.predefined_reset_air_command_limit_ang_vel_z),
            ("height", self.predefined_reset_air_command_limit_height),
        ):
            pair = self._parse_predefined_reset_air_limit_pair(command_limits, key)
            if pair is not None:
                buffer[env_ids, 0] = pair[0]
                buffer[env_ids, 1] = pair[1]

    def _apply_predefined_reset_air_limit_pair(
        self,
        values: torch.Tensor,
        limit_buffer: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        finite_limits = torch.isfinite(limit_buffer).all(dim=-1)
        mask = active_mask & finite_limits
        if not torch.any(mask):
            return values
        clipped = values.clone()
        clipped[mask] = torch.clamp(
            clipped[mask],
            min=limit_buffer[mask, 0],
            max=limit_buffer[mask, 1],
        )
        return clipped

    def _apply_predefined_reset_air_command_limits(self) -> None:
        if not hasattr(self, "predefined_reset_air_command_limit_remaining_time"):
            return
        active_mask = self.predefined_reset_air_command_limit_remaining_time > 0.0
        if not torch.any(active_mask):
            return

        self.command[:, 0] = self._apply_predefined_reset_air_limit_pair(
            self.command[:, 0],
            self.predefined_reset_air_command_limit_lin_vel_x,
            active_mask,
        )
        self.command[:, 1] = self._apply_predefined_reset_air_limit_pair(
            self.command[:, 1],
            self.predefined_reset_air_command_limit_lin_vel_y,
            active_mask,
        )
        self.command[:, 2] = self._apply_predefined_reset_air_limit_pair(
            self.command[:, 2],
            self.predefined_reset_air_command_limit_ang_vel_z,
            active_mask,
        )

    def _advance_predefined_reset_air_command_limit_timers(self) -> None:
        if not hasattr(self, "predefined_reset_air_command_limit_remaining_time"):
            return
        active_mask = self.predefined_reset_air_command_limit_remaining_time > 0.0
        if not torch.any(active_mask):
            return
        step = int(getattr(self, "common_step_counter", 0))
        update_mask = active_mask & (self.predefined_reset_air_command_limit_last_update_step != step)
        if not torch.any(update_mask):
            return
        self.predefined_reset_air_command_limit_remaining_time[update_mask] = torch.clamp(
            self.predefined_reset_air_command_limit_remaining_time[update_mask] - self.step_dt,
            min=0.0,
        )
        self.predefined_reset_air_command_limit_last_update_step[update_mask] = step

    def _apply_predefined_reset_air_height_limits(self, height_cmd: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "predefined_reset_air_command_limit_remaining_time"):
            return height_cmd
        active_mask = self.predefined_reset_air_command_limit_remaining_time > 0.0
        return self._apply_predefined_reset_air_limit_pair(
            height_cmd,
            self.predefined_reset_air_command_limit_height,
            active_mask,
        )

    def _custom_reset_random(self, env_ids):
        '''定制的reset随机化'''
        env_count = len(env_ids)
        # spring force randomize（仅当启用弹簧时）
        if self.use_spring and (self.cfg.spring_settings['random_force'] is not None) and self._spring_idx is not None:
            self.spring_force_rand[env_ids] = torch.empty(
                (env_count, len(self._spring_idx)), dtype=torch.float, device=self.device
            ).uniform_(*self.cfg.spring_settings['random_force'])
        if self.use_spring and self.cfg.spring_settings['damping'] and self._spring_idx is not None:
            self.spring_stretch_damping[env_ids] = torch.empty(
                (env_count, len(self._spring_idx)), dtype=torch.float, device=self.device
            ).uniform_(*self.cfg.spring_settings['rand_stretch_damping_range'])
            self.spring_contract_damping[env_ids] = torch.empty(
                (env_count, len(self._spring_idx)), dtype=torch.float, device=self.device
            ).uniform_(*self.cfg.spring_settings['rand_contract_damping_range'])

        # leg random start
        if self.cfg.use_leg_random_start:

            random_length = torch.empty(
                (env_count, 2), dtype=torch.float, device=self.device
            ).uniform_(*self.cfg.leg_length_range)
            random_angle = torch.empty(
                (env_count, 2), dtype=torch.float, device=self.device
            ).uniform_(*self.cfg.leg_angle_range)

            predefined_reset_ground = getattr(self.cfg, "predefined_reset_ground", {})
            if not isinstance(predefined_reset_ground, Mapping):
                predefined_reset_ground = {}
            use_predefined_reset = bool(self.cfg.use_predefined_leg_random_start)
            air_modes = self._get_active_predefined_reset_air_modes()
            air_probs = [max(float(mode.get("prob", 0.0)), 0.0) for mode in air_modes]
            ground_modes = []
            if use_predefined_reset:
                raw_ground_modes = predefined_reset_ground.get("modes", None)
                if isinstance(raw_ground_modes, Mapping):
                    ground_modes = [mode for mode in raw_ground_modes.values() if isinstance(mode, Mapping)]
                elif isinstance(raw_ground_modes, (tuple, list)):
                    ground_modes = [mode for mode in raw_ground_modes if isinstance(mode, Mapping)]
                elif max(float(predefined_reset_ground.get("prob", 0.0)), 0.0) > 0.0:
                    # Backward-compatible single positive-angle ground reset mode.
                    ground_modes = [
                        {
                            "prob": predefined_reset_ground.get("prob", 0.0),
                            "sign": 1.0,
                            "leg_height": predefined_reset_ground.get("leg_height", self.cfg.leg_length_range),
                            "leg_length": predefined_reset_ground.get("leg_length", self.cfg.leg_length_range),
                        }
                    ]
            ground_probs = [max(float(mode.get("prob", 0.0)), 0.0) for mode in ground_modes]

            air_mode_lrs_ids = [
                torch.zeros(0, dtype=torch.long, device=self.device) for _ in air_modes
            ]
            ground_lrs_ids = torch.zeros(0, dtype=torch.long, device=self.device)
            ground_prob = sum(ground_probs)
            mode_sampling_enabled = any(prob > 0.0 for prob in air_probs) or ground_prob > 0.0
            if mode_sampling_enabled:
                mode_samples = torch.empty(env_count, device=self.device).uniform_(0.0, 1.0)
                air_enabled_mask = ~self._get_predefined_reset_air_disabled_mask(env_ids)
                cursor = 0.0
                for i, prob in enumerate(air_probs):
                    next_cursor = min(cursor + prob, 1.0)
                    if next_cursor > cursor:
                        air_mode_lrs_ids[i] = (
                            (mode_samples >= cursor) & (mode_samples < next_cursor) & air_enabled_mask
                        ).nonzero(as_tuple=False).flatten()
                    cursor = next_cursor
                    if cursor >= 1.0:
                        break
                if cursor < 1.0:
                    next_cursor = min(cursor + ground_prob, 1.0)
                    if next_cursor > cursor:
                        ground_enabled_mask = ~self._get_predefined_reset_ground_disabled_mask(env_ids)
                        ground_lrs_ids = (
                            (mode_samples >= cursor) & (mode_samples < next_cursor) & ground_enabled_mask
                        ).nonzero(as_tuple=False).flatten()

            for mode_idx, (mode, air_lrs_ids) in enumerate(zip(air_modes, air_mode_lrs_ids)):
                if air_lrs_ids.numel() == 0:
                    continue
                air_lrs_env_ids = env_ids[air_lrs_ids]
                air_length_range = mode.get("leg_length_range", self.cfg.leg_length_range)
                air_angle_range = mode.get("leg_angle_range", self.cfg.leg_angle_range)
                random_length[air_lrs_ids] = torch.empty(
                    (len(air_lrs_ids), 2), dtype=torch.float, device=self.device
                ).uniform_(*air_length_range)
                random_angle[air_lrs_ids] = torch.empty(
                    (len(air_lrs_ids), 2), dtype=torch.float, device=self.device
                ).uniform_(*air_angle_range)
                self._apply_root_state_uniform_vel_b(
                    air_lrs_env_ids,
                    mode.get("pose_range", {}),
                    mode.get("velocity_range", {}),
                )
                self._set_predefined_reset_air_command_limits(
                    air_lrs_env_ids,
                    mode.get("command_limits", None),
                    mode_idx=mode_idx,
                )

            if ground_lrs_ids.numel() > 0 and ground_prob > 0.0:
                ground_lrs_env_ids = env_ids[ground_lrs_ids]
                ground_count = len(ground_lrs_ids)
                pdf_random_length = torch.empty((ground_count, 2), dtype=torch.float, device=self.device)
                pdf_random_angle = torch.empty((ground_count, 2), dtype=torch.float, device=self.device)
                leg_mode_samples = torch.empty((ground_count, 2), dtype=torch.float, device=self.device).uniform_(
                    0.0, ground_prob
                )
                leg_mode_assigned = torch.zeros((ground_count, 2), dtype=torch.bool, device=self.device)
                cursor = 0.0
                for ground_mode, prob in zip(ground_modes, ground_probs):
                    if prob <= 0.0:
                        continue
                    next_cursor = cursor + prob
                    leg_mask = (leg_mode_samples >= cursor) & (leg_mode_samples < next_cursor)
                    leg_mask &= ~leg_mode_assigned
                    cursor = next_cursor
                    if not torch.any(leg_mask):
                        continue
                    leg_height_range = ground_mode.get(
                        "leg_height", predefined_reset_ground.get("leg_height", self.cfg.leg_length_range)
                    )
                    leg_length_range = ground_mode.get(
                        "leg_length", predefined_reset_ground.get("leg_length", self.cfg.leg_length_range)
                    )
                    leg_count = int(torch.count_nonzero(leg_mask).item())
                    mode_random_height = torch.empty(leg_count, dtype=torch.float, device=self.device).uniform_(
                        *leg_height_range
                    )
                    mode_random_length = torch.empty(leg_count, dtype=torch.float, device=self.device).uniform_(
                        *leg_length_range
                    )
                    height_length_ratio = torch.clamp(
                        mode_random_height / torch.clamp(mode_random_length, min=1.0e-6),
                        min=-1.0 + 1.0e-6,
                        max=1.0 - 1.0e-6,
                    )
                    mode_random_angle = float(ground_mode.get("sign", 1.0)) * torch.arccos(height_length_ratio)
                    pdf_random_length[leg_mask] = mode_random_length
                    pdf_random_angle[leg_mask] = mode_random_angle
                    leg_mode_assigned |= leg_mask
                random_length[ground_lrs_ids] = pdf_random_length
                random_angle[ground_lrs_ids] = pdf_random_angle
                pdf_root_state = self.robot.data.default_root_state[ground_lrs_env_ids]
                pdf_root_state[:,2] = float(predefined_reset_ground.get("start_root_height", 0.0))
                pdf_root_state[:, :3] += self.terrain.env_origins[ground_lrs_env_ids]
                self.robot.write_root_state_to_sim(pdf_root_state, ground_lrs_env_ids)
                self.finish_init_time[ground_lrs_env_ids] = float(predefined_reset_ground.get("start_reset_time", 0.0))
                self._set_predefined_reset_ground_command_override(
                    ground_lrs_env_ids,
                    predefined_reset_ground.get("command_ranges", None),
                    float(predefined_reset_ground.get("start_reset_time", 0.0)),
                )
                zero_torque_time_s = float(predefined_reset_ground.get("zero_torque_time_s", 0.0))
                if zero_torque_time_s > 0.0:
                    episode_time = self.episode_length_buf[ground_lrs_env_ids].to(dtype=torch.float) * self.step_dt
                    self.predefined_reset_ground_zero_torque_until_time[ground_lrs_env_ids] = (
                        episode_time + zero_torque_time_s
                    )
            if mode_sampling_enabled:
                normal_lrs_mask = torch.ones(env_count, dtype=torch.bool, device=self.device)
                for air_lrs_ids in air_mode_lrs_ids:
                    normal_lrs_mask[air_lrs_ids] = False
                normal_lrs_mask[ground_lrs_ids] = False
                normal_lrs_env_ids = env_ids[normal_lrs_mask.nonzero(as_tuple=False).flatten()]
                self.finish_init_time[normal_lrs_env_ids] = 0

            random_joint_cfgs = (
                self._inverse_kinematics(random_length, random_angle)
                .transpose(-2, -1)
                .reshape(env_count, -1)
            )

            random_wheel_cfgs = torch.empty(
                (env_count, 2), dtype=torch.float, device=self.device
            ).uniform_(*self.cfg.wheel_angle_range)
            self.robot.write_joint_state_to_sim(random_joint_cfgs, torch.zeros_like(random_joint_cfgs), self.reorder_reset_joint_idx, env_ids)
            self.robot.write_joint_state_to_sim(random_wheel_cfgs, torch.zeros_like(random_wheel_cfgs), self._wheel_idx, env_ids)
            # joint vel random start
            if self.cfg.use_joint_vel_random_start:
                random_vel_leg = torch.empty(
                    (env_count, 4), dtype=torch.float, device=self.device
                ).uniform_(*self.cfg.leg_joint_vel_range)
                random_vel_wheel = torch.empty(
                    (env_count, 2), dtype=torch.float, device=self.device
                ).uniform_(*self.cfg.wheel_joint_vel_range)
                self.robot.write_joint_velocity_to_sim(random_vel_leg, self._legs_act_idx, env_ids)
                self.robot.write_joint_velocity_to_sim(random_vel_wheel, self._wheel_idx, env_ids)

    def _apply_spring(self):
        # prismatic spring simulate
        if self._spring_idx is not None:
            settings = self.cfg.spring_settings
            if settings['mode'] == 'constant':
                self.spring_force = settings['constant_force']+self.spring_force_rand
            elif settings['mode'] == 'linear':
                spring_pos = self.joint_pos[:,self._spring_idx]
                spring_length = torch.clamp(settings['spring_offset'] - spring_pos,min=0)
                self.spring_force = settings['linear_down'] + \
                                    (settings['linear_up']-settings['linear_down']) / settings['linear_length'] * spring_length + \
                                    self.spring_force_rand
            if self.cfg.spring_settings['damping']:
                joint_vel = self.joint_vel[:,self._spring_idx]
                stretch_damping_force = self.spring_stretch_damping * -torch.clamp(joint_vel, min=0.) # 伸展damping
                contract_damping_force = self.spring_contract_damping * -torch.clamp(joint_vel, max=0.) # 压缩damping
                total_damping_force = stretch_damping_force + contract_damping_force
                total_spring_force = self.spring_force + total_damping_force
                self.robot.set_joint_effort_target(total_spring_force, joint_ids=self._spring_idx)
            else:
                self.robot.set_joint_effort_target(self.spring_force, joint_ids=self._spring_idx)
            # print("spring force:", self.spring_force)
        # # virtual spring simulate
        # elif self._upper_spring_link_idx is not None and self._lower_spring_link_idx is not None:
        #     upper_pos_w = self.robot.data.body_pos_w[:,self._upper_spring_link_idx]
        #     lower_pos_w = self.robot.data.body_pos_w[:,self._lower_spring_link_idx]
        #     force_vec_w = lower_pos_w - upper_pos_w
        #     # force_vec_w_normal = force_vec_w / torch.norm(force_vec_w,p=2,dim=-1,keepdim=True)
        #     lower_quat_w = self.robot.data.body_quat_w[:,self._lower_spring_link_idx]
        #     force_vec_b = quat_apply_inverse(lower_quat_w, force_vec_w)
        #     force_vec_b_normal = force_vec_b / torch.norm(force_vec_b,p=2,dim=-1,keepdim=True)
        #     self.robot.set_external_force_and_torque(forces=self.spring_force*force_vec_b_normal,
        #                                              torques=torch.zeros_like(force_vec_b_normal,dtype=torch.float,device=self.device),
        #                                              body_ids=self._lower_spring_link_idx)
        # print(self.spring_force)

    def _resample_custom_cmd(self, command_counter: torch.tensor):
        # resample height command
        resample_env_ids = (command_counter > self.command_counter).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample_jump_takeoff_permission(resample_env_ids)
            self.height_cmd[resample_env_ids] = self._sample_height_command(resample_env_ids)
            self._latch_special_height_wave(resample_env_ids)
            self._resample_height_command_special_modes(resample_env_ids)
            self._request_special_mode_jump_takeoff(resample_env_ids)
            # 在启用了“按轴对齐 reset 朝向”的 env 中，把 command generator 的 heading target 同步成 reset 后的机器人朝向
            self._sync_heading_command_target_to_reset_heading(resample_env_ids)
        self.command_counter = command_counter.clone()
        return resample_env_ids

    def _inverse_kinematics(self, leg_length, leg_angle):
        links_length = torch.tensor(self.cfg.links_length,dtype=torch.float,device=self.device)
        links_length_pow = torch.pow(links_length,2)
        leg_length_pow = torch.pow(leg_length,2)

        alpha_offset = torch.tensor(self.cfg.alpha_offset,dtype=torch.float,device=self.device)
        alpha_buffer = torch.zeros(leg_length.shape[0],leg_length.shape[1],6,dtype=torch.float,device=self.device)
        solve_triangle = torch.arccos((links_length_pow[0]*leg_length_pow+
                                       links_length_pow[2]*(links_length_pow[0]-links_length_pow[1]))/
                                      (2*links_length[2]*links_length_pow[0]*leg_length))

        alpha_buffer[...,0] = leg_angle+0.5*torch.pi-solve_triangle
        alpha_buffer[...,1] = leg_angle+0.5*torch.pi+solve_triangle
        alpha_buffer[...,2] = torch.arccos((links_length_pow[0]+
                                         links_length_pow[1]-
                                         links_length_pow[0]*leg_length_pow/links_length_pow[2])/
                                         (2*links_length[0]*links_length[1]))
        alpha_buffer[...,3] = 2*torch.pi-(alpha_buffer[...,1]-alpha_buffer[...,0])-2*alpha_buffer[...,2]
        alpha_buffer[...,4] = alpha_buffer[...,2]
        alpha_buffer[...,5] = alpha_buffer[...,2]

        alpha_buffer[...,0] = alpha_buffer[...,0]-alpha_offset[0]
        alpha_buffer[...,1] = alpha_buffer[...,1]-alpha_offset[1]
        alpha_buffer[...,2] = -(alpha_buffer[...,2]-alpha_offset[2])
        alpha_buffer[...,3] = -(alpha_buffer[...,3]-alpha_offset[3])
        alpha_buffer[...,4] = -(alpha_buffer[...,4]-alpha_offset[4])
        alpha_buffer[...,5] = alpha_buffer[...,5]-alpha_offset[5]
        return alpha_buffer

    def _on_command_updated(self) -> None:
        """Hook for command post-processing."""
        self._apply_jump_takeoff_permission_command()
        is_heading_env = getattr(self.command_generator, "is_heading_env", None)
        non_heading_mask = self._get_non_heading_axis_aligned_zero_mask(is_heading_env)
        if torch.any(non_heading_mask):
            self.command[non_heading_mask, 2] = 0.0

        self._before_state_machine_command_updated()
        self.state_machine_manager.on_command_updated(self)
        self._apply_predefined_reset_ground_command_override()
        self._after_state_machine_command_updated()
        self.state_machine_manager.apply_command_overrides(self)

    def _before_state_machine_command_updated(self) -> None:
        """Hook for command overrides that state machines should observe."""
        pass

    def _after_state_machine_command_updated(self) -> None:
        """Hook for final command overrides after state-machine updates."""
        pass

    def _value_constrain(self,tar,cur,c_value):
        up_constrain_idx = torch.nonzero((tar-cur)>abs(c_value))
        low_constrain_idx = torch.nonzero((tar-cur)<-abs(c_value))
        tar[up_constrain_idx] = cur[up_constrain_idx] + abs(c_value)
        tar[low_constrain_idx] = cur[low_constrain_idx] - abs(c_value)
        return tar

    def _init_episode_length(self, init_s: float | int | torch.Tensor) -> torch.Tensor:
        """将秒数转换为 control 步数，支持 float 或 tensor 输入。"""
        dt_dec = self.cfg.sim.dt * self.cfg.decimation
        if isinstance(init_s, torch.Tensor):
            return torch.floor(init_s / dt_dec).to(device=self.device, dtype=torch.long)
        return torch.floor(torch.tensor(init_s / dt_dec, device=self.device)).long()

    def _update_ground_height_estimate(self):
        """使用RayCaster估计地面高度。"""
        if self._cached_ground_height_valid:
            return
        # 使用 height_scanner 获取地面高度
        if self.height_scanner is None or self._use_absolute_height() or self._use_leg_length_height():
            self.ground_z_est.zero_()
            self._cached_ground_height_valid = True
            return

        ray_hits = getattr(self.height_scanner.data, "ray_hits_w", None)
        if ray_hits is None or ray_hits.ndim != 3 or ray_hits.shape[1] == 0:
            self.ground_z_est = self.terrain.env_origins[:, 2]
            self._cached_ground_height_valid = True
            return

        ray_hits_z = ray_hits[:, :, 2]
        valid_mask = torch.isfinite(ray_hits_z)
        fallback_z = self.terrain.env_origins[:, 2]
        env_any_valid = torch.any(valid_mask, dim=1)
        sum_z = torch.sum(ray_hits_z.masked_fill(~valid_mask, 0.0), dim=1)
        count_z = torch.sum(valid_mask.float(), dim=1)
        mean_z = sum_z / torch.clamp(count_z, min=1.0)
        self.ground_z_est = torch.where(env_any_valid, mean_z, fallback_z)
        self._cached_ground_height_valid = True
        # print('z_est:',self.ground_z_est)
        # print('z_abs:',self.robot.data.root_pos_w[:,2])
