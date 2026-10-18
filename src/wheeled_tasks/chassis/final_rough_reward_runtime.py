"""Refreshed native-input composition for explicit V7 final rough selection.

Physics/contact history belongs to the existing source observer. This owner
adds the wheel LINK scan, once-per-policy airborne state and passive-knee
geometry before one 31-term payment. It neither selects a course nor samples
airborne commands. Terrain-specific entry sampling is a separate owner.
"""
import torch

from .final_rough_reward import CHASSIS_VERSION, WEIGHTS, V7FinalRoughPhysicalFrame, v7_final_rough_reward
from .relative_knee_geometry import RelativeKneeGeometry
from .scut_flat_reward_runtime import ScutFlatRewardRuntime
from .source_airborne_state import AirbornePhysicalObservation, SourceAirborneState
from .wheel_ground_scan import wheel_ground_scan


REWARD_PROFILE = "v7_final_rough_v1"
RUNTIME_VERSION = "v7-final-rough-native-v1"


def wheel_link_positions_heading(root_pose_world, wheel_positions_world):
    """Yaw-level relative LINK positions; source centering suppresses lateral Y."""
    relative = wheel_positions_world - root_pose_world[:, None, :3]
    x, y, z, w = root_pose_world[:, 3:7].unbind(-1)
    yaw = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y.square() + z.square()))
    cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
    return torch.stack((cosine * relative[..., 0] + sine * relative[..., 1],
                        torch.zeros_like(relative[..., 1]), relative[..., 2]), dim=-1)


class V7FinalRoughRewardRuntime(ScutFlatRewardRuntime):
    def __init__(self, count, device, body_names, mapping, *, model_spec, manifest,
                 native_joint_names, wheel_body_names=("L_link3", "R_link3"), physics_dt_s=.005,
                 flat_rehearsal_mask=None, flat_rehearsal_reward_profile=REWARD_PROFILE,
                 command_decoupling=None, terrain_source="scut0730"):
        from .scut_reward_selection import V7_FINAL_UNDESIRED_BODY_NAMES, V7_FINAL_TERRAIN_SOURCE
        if terrain_source not in ("scut0730", V7_FINAL_TERRAIN_SOURCE):
            raise ValueError("V7 final rough requires a named implemented terrain source")
        self.terrain_source = terrain_source
        if (set(mapping["undesired_link_names"]) != set(V7_FINAL_UNDESIRED_BODY_NAMES)
                or list(mapping["terminal_link_names"]) != ["base_link"]):
            raise ValueError("V7 final rough requires the mapped13-body contact set and base-only terminal set")
        if tuple(wheel_body_names) != ("L_link3", "R_link3"):
            raise ValueError("V7 final rough requires explicit left/right wheel LINK pairing")
        if flat_rehearsal_reward_profile != REWARD_PROFILE:
            raise ValueError("V7 final rough rehearsal uses the same31 payments on every lane")
        if command_decoupling is not None and command_decoupling.get("enabled"):
            raise ValueError("V7 final rough has no additional yaw-cost payment adapter")
        self.knee_geometry = RelativeKneeGeometry(model_spec, manifest, native_joint_names)
        # Reuse only the physical observer, not its26-term payment dispatch.
        super().__init__(count, device, body_names, mapping, reward_profile="scut0730_rough_v1",
            physics_dt_s=physics_dt_s, flat_rehearsal_mask=flat_rehearsal_mask,
            flat_rehearsal_reward_profile=("scut0730_rough_v1" if flat_rehearsal_mask is not None
                                          else "scut0719_flat_v1"),
            command_decoupling=command_decoupling)
        self.reward_profile = REWARD_PROFILE
        self.flat_rehearsal_reward_profile = REWARD_PROFILE
        self.confirmation_policy_ticks = 10
        self.wheel_ids = self._body_ids(wheel_body_names)
        self.airborne = SourceAirborneState(count, self.raw_action.device)
        self.last_airborne_transition = None
        self.wheel_scan_hit_count = torch.full((count, 2), -1, dtype=torch.long, device=self.raw_action.device)
        self._last_policy_metrics = {}

    def reset(self, ids):
        super().reset(ids)
        self.airborne.reset(ids)
        self.wheel_scan_hit_count[ids] = -1

    def _native_tensor(self, value, shape, name, *, boolean=False):
        dtype = torch.bool if boolean else self.raw_action.dtype
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != dtype or value.device != self.raw_action.device):
            raise ValueError(f"V7 final rough {name} requires native shape{shape}, dtype and device")

    def reward_height(self, root_pose_world, origins, reward_ground_height_world=None):
        self._native_tensor(reward_ground_height_world, (len(self.raw_action),), "base ground scan")
        # Validity is classified in pay before ray queries. Preserve invalid
        # numerical rows rather than repairing their pose or ground height.
        return root_pose_world[:, 2] - reward_ground_height_world

    def pay(self, *, root_velocity_body, angular_velocity_body, projected_gravity,
            root_pose_world, wheel_link_positions_world, origins, commands,
            terminated, numerical_safety_reset, reward_ground_height_world,
            native_joint_positions, wheel_link_pose_world, policy_tick, raycast, stair_active,
            flat_reward_height_m=None, scripted_interval=None):
        """Consume the current physics interval, before command updates/reset.

        Already classified numerical-reset lanes do not enter the scan/state
        observer. Their absent scan is NaN/count=-1 and their final payments
        are zero. This does not remove PPO samples or classify finite gaps.
        Unexpected invalid physical mappings still fail before payment.

        An explicit whole-interval script/blend mask clears only the natural
        airborne phase/cache. Physical scans and contact history remain real.
        The first fully autonomous interval observes its current physical state
        from fresh phase timers; it can enter natural flight immediately.
        """
        n = len(self.raw_action)
        if type(policy_tick) is not int or policy_tick < 0:
            raise ValueError("V7 final rough requires a nonnegative policy counter")
        self._native_tensor(root_velocity_body, (n, 3), "root COM velocity")
        self._native_tensor(angular_velocity_body, (n, 3), "root angular velocity")
        self._native_tensor(projected_gravity, (n, 3), "projected gravity")
        self._native_tensor(origins, (n, 3), "environment origins")
        self._native_tensor(terminated, (n,), "termination mask", boolean=True)
        self._native_tensor(root_pose_world, (n, 7), "root LINK pose")
        self._native_tensor(wheel_link_pose_world, (n, 2, 7), "wheel LINK poses")
        self._native_tensor(wheel_link_positions_world, (n, 2, 3), "wheel LINK positions")
        self._native_tensor(native_joint_positions, (n, len(self.knee_geometry.native_joint_names)), "full joint positions")
        self._native_tensor(commands, (n, 3), "body commands")
        self._native_tensor(reward_ground_height_world, (n,), "base ground scan")
        self._native_tensor(numerical_safety_reset, (n,), "numerical-reset mask", boolean=True)
        self._native_tensor(stair_active, (n,), "stair state", boolean=True)
        if scripted_interval is not None:
            self._native_tensor(scripted_interval, (n,), "scripted interval", boolean=True)
        if flat_reward_height_m is not None:
            raise ValueError("V7 final rough has no separate flat payment datum")
        valid = ~numerical_safety_reset
        finite = (torch.isfinite(root_pose_world).all(-1)
            & torch.isfinite(root_velocity_body).all(-1)
            & torch.isfinite(angular_velocity_body).all(-1)
            & torch.isfinite(projected_gravity).all(-1)
            & torch.isfinite(origins).all(-1)
            & torch.isfinite(wheel_link_pose_world).flatten(1).all(-1)
            & torch.isfinite(native_joint_positions).all(-1)
            & torch.isfinite(reward_ground_height_world)
            & torch.isfinite(commands).all(-1)
            & torch.isfinite(self.net_contact).flatten(1).all(-1))
        quaternion_ok = ((root_pose_world[:, 3:7].square().sum(-1) - 1.).abs() <= 1e-3
            ) & ((wheel_link_pose_world[..., 3:7].square().sum(-1) - 1.).abs() <= 1e-3).all(-1)
        paired = (wheel_link_positions_world == wheel_link_pose_world[..., :3]).flatten(1).all(-1)
        if bool((valid & ~(finite & quaternion_ok & paired)).any()):
            raise RuntimeError("Invalid V7 final rough physical mapping outside numerical reset lanes")
        if not callable(raycast):
            raise TypeError("V7 final rough requires the bound collision raycast")
        wheel_heights = self.raw_action.new_full((n, 2), float("nan"))
        counts = torch.full((n, 2), -1, dtype=torch.long, device=self.raw_action.device)
        if bool(valid.any()):
            _, counts_valid, heights_valid = wheel_ground_scan(wheel_link_pose_world[valid],
                reward_ground_height_world[valid], raycast)
            counts[valid], wheel_heights[valid] = counts_valid, heights_valid
        self.wheel_scan_hit_count.copy_(counts)
        source_commands = torch.stack((commands[:, 0], torch.zeros_like(commands[:, 0]), commands[:, 1]), -1)
        observation_valid = valid
        if scripted_interval is not None:
            # Reset only phase state, after validating the physical frame.
            # Numeric rows retain their existing exclusion/reset ownership.
            scripted_ids = (valid & scripted_interval).nonzero(as_tuple=False).flatten()
            self.airborne.reset(scripted_ids)
            self.airborne.entry_command3[scripted_ids] = 0.
            observation_valid = valid & ~scripted_interval
        transition = self.airborne.observe(AirbornePhysicalObservation(
            body_relative_ground_height_m=root_pose_world[:, 2] - reward_ground_height_world,
            wheel_relative_ground_heights_m=wheel_heights,
            wheel_contact_forces_world_n=self.net_contact[:, self.wheel_ids],
            exit_body_contact_forces_world_n=self.net_contact[:, self.undesired_ids],
            commands3=source_commands, terrain_allowed=torch.ones_like(valid), stair_active=stair_active),
            policy_tick=policy_tick, observation_valid=observation_valid)
        self.last_airborne_transition = transition
        base = self.physical_frame(root_velocity_body=root_velocity_body,
            angular_velocity_body=angular_velocity_body, projected_gravity=projected_gravity,
            root_pose_world=root_pose_world, wheel_link_positions_world=wheel_link_positions_world,
            origins=origins, commands=commands, terminated=terminated,
            numerical_safety_reset=numerical_safety_reset,
            reward_ground_height_world=reward_ground_height_world)
        state = transition.state
        result = v7_final_rough_reward(V7FinalRoughPhysicalFrame(base=base,
            airborne=state["airborne"], root_height_world_m=root_pose_world[:, 2],
            ground_height_world_m=reward_ground_height_world,
            wheel_contact_elapsed_s=state["wheel_contact_elapsed_s"],
            base_contact_elapsed_s=state["base_contact_elapsed_s"],
            knee_inner_angles_rad=self.knee_geometry.inner_angles_rad(native_joint_positions),
            wheel_positions_heading_m=wheel_link_positions_heading(root_pose_world, wheel_link_positions_world)),
            knee_inner_limits_rad=self.knee_geometry.inner_limits_rad)
        denominator = valid.sum().clamp_min(1)
        self._last_policy_metrics = {
            "/airborne/physical_state_fraction": (state["airborne"] & valid).sum() / denominator,
            "/airborne/entered_fraction": transition.entered.sum() / denominator,
            "/airborne/exited_fraction": transition.exited.sum() / denominator,
            "/airborne/numerical_observation_excluded_fraction": numerical_safety_reset.float().mean(),
            "/terrain/wheel_scan_valid_fraction": torch.where(valid[:, None], counts, 0.).sum() / (8. * denominator),
        }
        if scripted_interval is not None:
            self._last_policy_metrics["/airborne/scripted_observation_excluded_fraction"] = (
                valid & scripted_interval).float().mean()
        return result

    def policy_metrics(self):
        return dict(self._last_policy_metrics)

    def summary(self):
        result = super().summary()
        result.update(version=CHASSIS_VERSION, physical_input_adapter=RUNTIME_VERSION,
            reward_profile=REWARD_PROFILE, nonzero_weights=dict(WEIGHTS),
            nonzero_term_count=31, root_height_reference="root_link_z_minus_base_ground_scan_then_clip",
            relative_height_clip_m=[.05, .45], confirmation_policy_ticks=10, confirmation_seconds=.2,
            normal_termination_payment=-4., airborne_termination_payment=-24.,
            knee_coordinate="actual_passive_thigh_shank_inner_angle_rad",
            knee_inner_limits_rad=list(self.knee_geometry.inner_limits_rad),
            knee_joint_names=[self.knee_geometry.native_joint_names[i] for i in self.knee_geometry.joint_indices],
            airborne_contact_reference="same_mapped13_body_net_vectors_as_undesired_payment",
            airborne_command_sampling_enabled=False, airborne_command_sampling_reason="shared0730_terrain_keys_match_no_source_entry_profile",
            stair_state_machine_enabled=False, stepup_command_is_separate=True,
            wheel_ground_scan_reference="wheel_LINK_origin_and_attachment_yaw_not_tire_midplane",
            numerical_observation_exclusion="scan_and_state_only_no_PPO_sample_filter",
            scripted_airborne_observation="optional_whole_interval_mask_clears_phase_timers_requests_and_entry_cache",
            scripted_physical_observation="retain_scans_contacts_and_existing_numeric_safety",
            autonomous_airborne_handoff="first_complete_RL_interval_observes_current_physics_with_fresh_phase_state",
            local_adaptations=["passive_knee_inner_angle_and_asset_limits", "relative_height_before_clip",
                "explicit13_chassis_contact_bodies_no_gimbal", "shared0730_terrain_geometry",
                "existing_mechanical_guards_and_boundary_policy"],
            source_final_environment_equivalent=False, native_qualified=False)
        if self.terrain_source == "v7_final_b8ff79f3":
            result.update(terrain_source=self.terrain_source,
                airborne_command_sampling_reason="terrain_entry_command_owner_not_yet_connected",
                source_birth_mixture_enabled=False, birth_adapter="explicit_nominal_four_axis_headings",
                stepup_detector_enabled=False)
            result["local_adaptations"][3] = "final_source_geometry_with_nominal_birth_and_fixed_rehearsal"
        if self.flat_rehearsal_mask is not None:
            result["flat_rehearsal_reward"] = {
                "reward_profile": REWARD_PROFILE, "version": CHASSIS_VERSION,
                "lanes": int(self.flat_rehearsal_mask.sum()),
                "selection": "same31_payments_for_every_lane_mask_only_identifies_command_population",
                "root_height_reference": result["root_height_reference"],
                "velocity_height_gate_enabled": False}
        return result
