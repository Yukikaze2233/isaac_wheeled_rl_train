"""Physical recovery prelude and RL takeover, without post-reset state writes.

Only reset profiles and outcome grading use simulator truth. The reference FSM
gets encoder/FK and simulated IMU signals; it never receives terrain or contacts.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from wheeled_tasks.chassis.source_layout import module_source_path

import torch

from wheeled_tasks.chassis.observations.mechanism_observer import ClosedChainHeightModel
from wheeled_tasks.chassis.observations.continuous_encoders import ContinuousEncoderAngles
from wheeled_tasks.chassis.tasks.recovery.recovery import (
    DIRECT_TAKEOVER_VERSION, LOW_POSE_TAKEOVER_VERSION, SIDE_TRANSITION_TAKEOVER_VERSION,
    REPLANT_TAKEOVER_VERSION, RecoveryFSM, RecoveryPhase,
)
from wheeled_tasks.chassis.observations.recovery_observer import SimulatedImu, WheelSupportProbe
from wheeled_tasks.chassis.tasks.terrain.step_trigger import estimate_height_if_grounded
from wheeled_tasks.chassis.application.runtime.training_schedule import stage_actor_updates
from wheeled_tasks.chassis.simulation.root_kinematics import root_link_linear_velocity


def read_frozen(root, settings, prefix):
    path = root / settings[prefix + "_file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != settings[prefix + "_sha256"]:
        raise ValueError(f"Recovery {prefix} differs from frozen contract")
    return json.loads(raw)


DIRECT_TAKEOVER_SETTINGS = {
    "version": DIRECT_TAKEOVER_VERSION, "script_sample_dt_s": .001,
    "feedback_sample_dt_s": .005, "minimum_sensor_history_s": .06,
    "capture_height_m": [.20, .29], "capture_max_tilt_rad": math.pi / 6,
    "upright_max_tilt_rad": .2, "upright_max_gyro_rad_s": 2., "other_max_gyro_rad_s": 1.,
    "capture_max_leg_error_rad": .45, "capture_max_leg_velocity_rad_s": 4.,
    "capture_max_wheel_velocity_rad_s": 10., "specific_support_acceleration_min_m_s2": 6.,
    "shell_clearance_tolerance_m": .003, "feedback_max_age_s": .02,
    "acceleration_max_age_s": .03, "sensor_max_skew_s": .01,
    "capture_reference_control6_rad": [-.2521074033177618, -.11790932322695527, 0.,
                                       .2518308496732414, .11764252669871, 0.],
    "capture_hinge_max_rad": [.3784673759566423, .37845928134324935],
    "hinge_min_rad": [-.140638855, -.140638855],
}


def validate_direct_takeover(settings):
    """Select a fixed capture version; no stage, geometry or native admission."""
    if "direct_takeover" not in settings:
        return False
    selected = settings["direct_takeover"]
    version = selected.get("version") if isinstance(selected, Mapping) else None
    expected = {**DIRECT_TAKEOVER_SETTINGS, "version": version}
    if (version not in (DIRECT_TAKEOVER_VERSION, LOW_POSE_TAKEOVER_VERSION,
                        SIDE_TRANSITION_TAKEOVER_VERSION, REPLANT_TAKEOVER_VERSION)
            or json.dumps(selected, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False)):
        raise ValueError("Direct recovery requires the explicit V7 capture settings")
    expected = {"reference_dt": .005, "blend_seconds": 0., "stable_seconds": 1.,
                "prepare_speed_rad_s": 6., "orbit_speed_rad_s": 6., "side_speed_rad_s": 6.,
                "complete_started_blend": False}
    if any(type(settings.get(key)) is not type(value) or settings.get(key) != value
           for key, value in expected.items()):
        raise ValueError("Direct recovery requires its 200 Hz feedback and zero-blend script configuration")
    return True


def direct_capture_ready(q, dq, gravity, gyro, acceleration, height, support, sensor_valid,
                         reference, shell_vertices):
    """Deployment capture predicates in CONTROL order, using sensor/FK inputs."""
    cfg = DIRECT_TAKEOVER_SETTINGS
    unit_gravity = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    finite = (torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
              & torch.isfinite(gravity).all(-1) & (gravity.norm(dim=-1) > .5)
              & torch.isfinite(gyro).all(-1) & torch.isfinite(acceleration).all(-1)
              & torch.isfinite(height))
    tilt_ok = -unit_gravity[:, 2] >= math.cos(cfg["capture_max_tilt_rad"])
    upright = -unit_gravity[:, 2] >= math.cos(cfg["upright_max_tilt_rad"])
    gyro_max = torch.where(upright, height.new_tensor(cfg["upright_max_gyro_rad_s"]),
                          height.new_tensor(cfg["other_max_gyro_rad_s"]))
    delta = reference[list(RecoveryFSM.LEG_IDS)] - q[:, RecoveryFSM.LEG_IDS]
    wrapped_error = torch.atan2(delta.sin(), delta.cos()).abs().amax(-1)
    hinge = torch.stack((q[:, 1] - q[:, 0], q[:, 3] - q[:, 4]), -1)
    hinge_ok = ((hinge >= hinge.new_tensor(cfg["hinge_min_rad"]))
                & (hinge <= hinge.new_tensor(cfg["capture_hinge_max_rad"]))).all(-1)
    shell_depth = (unit_gravity @ shell_vertices.T).amax(-1)
    outward_rate = unit_gravity[:, 0] * gyro[:, 1] - unit_gravity[:, 1] * gyro[:, 0]
    outward_ok = ((-unit_gravity[:, 2] >= math.cos(math.pi / 6))
                  | (outward_rate <= .05 * unit_gravity[:, :2].norm(dim=-1)))
    return (finite & sensor_valid & support & tilt_ok & outward_ok & hinge_ok
            & (height > cfg["capture_height_m"][0]) & (height < cfg["capture_height_m"][1])
            & (gyro.norm(dim=-1) <= gyro_max)
            & (dq[:, RecoveryFSM.LEG_IDS].abs().amax(-1) <= cfg["capture_max_leg_velocity_rad_s"])
            & (dq[:, RecoveryFSM.WHEEL_IDS].abs().amax(-1) <= cfg["capture_max_wheel_velocity_rad_s"])
            & (wrapped_error <= cfg["capture_max_leg_error_rad"])
            & ((acceleration * -unit_gravity).sum(-1) >= cfg["specific_support_acceleration_min_m_s2"])
            & (height + cfg["shell_clearance_tolerance_m"] >= shell_depth))


def _direct_shell_geometry(env, root):
    """Use the authenticated current chassis meshes, never simulator contacts."""
    from wheeled_tasks.chassis.simulation.runtime.flat_birth_placement import PhysicalFlatPlacement
    directory = Path(root) / env.cfg["asset_directory"]
    spec = json.loads((directory / "model_spec.json").read_bytes())
    placement = PhysicalFlatPlacement(spec, env.manifest, directory, clearance_m=0.,
                                      device=env.device, dtype=torch.float32)
    if placement.manifest_sha256 != env.cfg["asset_manifest_sha256"]:
        raise ValueError("Direct recovery shell differs from the selected chassis")
    shapes = [shape for shape in placement._shapes if shape[0] == "base_link"]
    if not shapes or any(shape[2] != "mesh" for shape in shapes):
        raise ValueError("Direct recovery requires the bound base collision meshes")
    vertices = torch.cat([points @ origin[:3, :3].T + origin[:3, 3]
                          for _, _, _, origin, points in shapes])
    return vertices, {"asset_manifest_sha256": placement.manifest_sha256,
                      "model_spec_sha256": placement.model_spec_sha256,
                      "vertex_count": len(vertices),
                      "support": "maximum body-gravity projection of actual base collision vertices"}


class RecoveryTraining:
    PREMATURE_RELEASE_FAILURE = 7
    TRANSITION_SCHEMA = "v6-recovery-physical-transition-v1"
    EVENT_NAMES = ("script_ready", "rl_entered", "legacy_physical_hold_qualified",
                   "legacy_followup_complete")

    def __init__(self, env, root, settings, *, sampled_birth_recipes=None):
        # A selected reset owner may be constructed while PPO is sampling.
        # Its mutable FSM/episode buffers must remain writable outside that scope.
        if sampled_birth_recipes is not None and torch.is_inference_mode_enabled():
            with torch.inference_mode(False):
                self.__init__(env, root, settings, sampled_birth_recipes=sampled_birth_recipes)
            return
        self.env, self.settings = env, dict(settings)
        self.direct_takeover = validate_direct_takeover(settings)
        self.source_flat_handoff = "source_flat_handoff" in settings
        if self.source_flat_handoff:
            from wheeled_tasks.chassis.application.configuration.flat_episode_selection import validate_scut_flat_recovery
            admission = validate_scut_flat_recovery(env.cfg, root)
            source_specs = {group: {"recovery_pose": name}
                            for group, name in admission["profiles_by_scene"].items()}
        table = read_frozen(root, settings, "profiles")
        lookup = read_frozen(root, settings, "height_lookup")
        self.geometry = ClosedChainHeightModel(lookup, env.device)
        relative_limits = torch.stack([side["delta_rad"][[0, -1]] for side in self.geometry.sides])
        self.encoders = ContinuousEncoderAngles(env.num_envs, env.device, relative_limits)
        self.reference_dt = settings.get("reference_dt", .005)
        if env.dt not in (.001, .005) or env.policy_dt != .02 or self.reference_dt != .005:
            raise ValueError("Recovery requires declared 1 kHz or 200 Hz PD, 200 Hz references and 50 Hz policy")
        if self.direct_takeover and env.dt != .005:
            raise ValueError("Direct candidate requires 200 Hz physics with held 1 ms script samples")
        self.reference_stride = round(self.reference_dt / env.dt)
        self.script_substeps = 5 if self.direct_takeover else 1
        self.reference_tick = 0
        cfg = {key: settings[key] for key in RecoveryFSM.DEFAULTS if key in settings}
        cfg["dt"] = .001 if self.direct_takeover else self.reference_dt
        if self.direct_takeover:
            cfg["takeover_version"] = settings["direct_takeover"]["version"]
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
        if self.direct_takeover:
            capture = torch.tensor(DIRECT_TAKEOVER_SETTINGS["capture_reference_control6_rad"],
                                   dtype=torch.float32, device=env.device)
            if not self.geometry.wheel_positions(capture[None])[1].all():
                raise ValueError("Direct capture reference lies outside the selected geometry")
            # Confirm the 50-degree target and 60-degree gate in this asset's
            # nonlinear relative-angle map, independently of the motor zero.
            for side, hip, auxiliary, sign, bound in zip(("left", "right"), (0, 3), (1, 4), (1., -1.),
                    DIRECT_TAKEOVER_SETTINGS["capture_hinge_max_rad"]):
                rows = lookup["sides"][side]
                x = torch.tensor(rows["delta_rad"], dtype=torch.float64)
                y = torch.tensor(rows["passive_inner_deg"], dtype=torch.float64)
                query = torch.tensor([float(capture[auxiliary] - capture[hip]), sign * bound], dtype=torch.float64)
                upper = torch.searchsorted(x, query).clamp(1, len(x) - 1)
                amount = (query - x[upper - 1]) / (x[upper] - x[upper - 1])
                angles = y[upper - 1] + amount * (y[upper] - y[upper - 1])
                if not torch.allclose(angles, angles.new_tensor([50., 60.]), atol=.002, rtol=0.):
                    raise ValueError("Direct capture knee reference/gate differs from bound geometry")
            self.capture_reference = capture
            targets[0] = capture
            if cfg["takeover_version"] == DIRECT_TAKEOVER_VERSION:
                targets[3] = capture
            self.shell_vertices, self.shell_identity = _direct_shell_geometry(env, root)
        self.fsm = RecoveryFSM(env.num_envs, env.device, cfg, *targets, table["axis_signs"],
            shell_vertices=self.shell_vertices if self.direct_takeover else None)
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
        self.initial_root_velocity = torch.zeros(n, 6, dtype=env.nominal.dtype, device=d)
        self.initial_joint_velocity = torch.zeros_like(env.nominal)
        self.initial_release = torch.zeros(n, device=d)
        self.followup_command = torch.zeros(n, 3, device=d)
        self.followup_command[:, 2] = .305
        self.required_platform_height = torch.zeros(n, device=d)
        self.min_ending_x = torch.full((n,), -math.inf, device=d)
        self.profile_names = []
        for index, group in enumerate(env.scene_groups):
            spec = (source_specs if self.source_flat_handoff else env.skills.specs).get(group, {})
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
            self._load_initial_profile(index, profile, name)
            self.initial_release[index] = profile.get("initial_release_s", 0.)
            self.required_platform_height[index] = profile.get("required_platform_height_m", 0.)
            self.min_ending_x[index] = profile.get("min_ending_x", -math.inf)
            self.followup_command[index] = torch.tensor(spec.get("recovery_followup_command", [0., 0., .305]), device=d)
        self.has_tasks = bool(self.enabled.any())
        if self.source_flat_handoff and (not self.has_tasks or bool(self.enabled.all())):
            raise ValueError("Native source recovery allocation requires both recovery and ordinary lanes")
        self.pose_curriculum = None
        if settings.get("prefix_curriculum", {}).get("enabled") and not env.cfg.get("evaluation_exact_cases"):
            from wheeled_tasks.chassis.tasks.curricula.recovery_prefix_curriculum import RecoveryPrefixCurriculum
            self.pose_curriculum = RecoveryPrefixCurriculum(env, table, settings["prefix_curriculum"])
        elif settings.get("pose_curriculum", {}).get("enabled") and not env.cfg.get("evaluation_exact_cases"):
            from wheeled_tasks.chassis.tasks.curricula.recovery_curriculum import RecoveryResetCurriculum
            self.pose_curriculum = RecoveryResetCurriculum(env, table, settings["pose_curriculum"])
        self.elapsed = torch.zeros(n, device=d)
        self.release_finished = ~self.enabled
        self._motor_enabled_for_substep = self.release_finished.clone()
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
        from wheeled_tasks.chassis.application.configuration.flat_episode_selection import validate_recovery_episode_window
        self.episode_window = validate_recovery_episode_window(env.cfg)
        if self.episode_window is not None:
            self.window_policy_ticks = torch.zeros(n, dtype=torch.long, device=d)
            self.window_first_actor_tick = torch.full_like(self.window_policy_ticks, -1)
            self.window_eligible_ticks = torch.zeros_like(self.window_policy_ticks)
            self._window_last_policy_tick = None
        if self.direct_takeover:
            from wheeled_tasks.chassis.observations.recovery_sensor_cache import RecoveryAccelerationCache
            from wheeled_tasks.chassis.observations.recovery_sensor_guard import RecoverySensorGuard
            self.acceleration_cache = RecoveryAccelerationCache(n, d, env.dt,
                DIRECT_TAKEOVER_SETTINGS["acceleration_max_age_s"])
            self.sensor_guard = RecoverySensorGuard(n, d,
                feedback_max_age_s=DIRECT_TAKEOVER_SETTINGS["feedback_max_age_s"],
                acceleration_max_age_s=DIRECT_TAKEOVER_SETTINGS["acceleration_max_age_s"],
                sensor_max_skew_s=DIRECT_TAKEOVER_SETTINGS["sensor_max_skew_s"])
            self.sensor_sample_sequences = torch.zeros(n, 8, dtype=torch.long, device=d)
            self.sensor_sample_stamps_s = torch.zeros(n, 8, dtype=torch.float64, device=d)
            self.sensor_observed_stamps_s = torch.full_like(self.sensor_sample_stamps_s, torch.nan)
            self.sensor_status = None
            self.sensor_history_start_s = torch.full((n,), torch.nan, dtype=torch.float64, device=d)
            self.sensor_history_last_s = torch.full_like(self.sensor_history_start_s, torch.nan)
            self.valid_sensor_history_s = torch.zeros_like(self.sensor_history_start_s)
            self.capture_ready = torch.zeros(n, dtype=torch.bool, device=d)
        self.policy_motion_hold = torch.zeros(n, dtype=torch.bool, device=d)
        self.policy_command_override_active = torch.zeros(n, dtype=torch.bool, device=d)
        self.wheel_balance_active = torch.zeros(n, dtype=torch.bool, device=d)
        self.goal_platform = self.required_platform_height == 0.
        # These are observations of the existing controller/outcome predicates,
        # never new transition gates or V7 skill qualification. Lesson/profile
        # identity is latched only on the same physical episode reset.
        self.episode_ids = torch.full((n,), -1, dtype=torch.long, device=d)
        self.episode_lesson_ids = torch.full_like(self.episode_ids, -1)
        profile_names = self._register_sampled_birth_recipes(sampled_birth_recipes, tuple(table["profiles"]))
        profile_index = {name: index for index, name in enumerate(profile_names)}
        self._original_profile_ids = torch.tensor([profile_index.get(name, -1)
            for name in self.profile_names], dtype=torch.long, device=d)
        self.episode_profile_ids = self._original_profile_ids.clone()
        self._event_latched = {name: torch.zeros_like(self.enabled) for name in self.EVENT_NAMES}
        self._policy_events = {name: torch.zeros_like(self.enabled) for name in self.EVENT_NAMES}
        self._policy_occurrences = {name: torch.full_like(self.episode_ids, -1) for name in self.EVENT_NAMES}
        self._event_policy_open = False
        source = Path(__file__)
        self._source_identity = {
            "schema": "v6-recovery-physical-event-source-v1",
            "profiles_file": settings["profiles_file"], "profiles_sha256": settings["profiles_sha256"],
            "height_lookup_file": settings["height_lookup_file"],
            "height_lookup_sha256": settings["height_lookup_sha256"],
            "asset_manifest_sha256": env.cfg.get("asset_manifest_sha256"),
            "model_spec_sha256": table.get("model_spec_sha256"),
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (source, module_source_path("recovery.py"),
                    module_source_path("recovery_curriculum.py"),
                    module_source_path("recovery_prefix_curriculum.py"))},
            "source_hash_scope": "producer_fsm_and_reset_samplers_not_transitive_execution_tree",
            "control_joint_names": list(table["control_joint_names"]),
            "profile_names": list(profile_names),
            "lesson_names": (["prefix_stand", "prefix_release", "original_chain"]
                if settings.get("prefix_curriculum", {}).get("enabled") else []),
            "physics_dt_s": env.dt, "policy_dt_s": env.policy_dt, "reference_dt_s": self.reference_dt,
            **({"takeover_version": settings["direct_takeover"]["version"], "script_sample_dt_s": self.fsm.dt,
                "feedback_model": "one real 5ms sample held for five 1ms reference updates; not 1kHz physical equivalence",
                "sensor_provenance": "seven groups share cached motor/IMU feedback; raw acceleration is reconstructed at its timestamp without a second uplink delay",
                "sensor_guard": "eight stamp/sequence checks; shared-source simulation, not independent CAN packets",
                "shell_geometry": self.shell_identity,
                "outcome_scope": "direct release has no legacy 1s pre-motion hold; followup remains diagnostic"}
               if self.direct_takeover else {}),
            "occurrence_clock": "completed_recovery_reference_ticks",
            "fsm_settings": deepcopy(self.fsm.config),
            "effective_recovery_settings": {**deepcopy(self.settings),
                "reference_dt": self.reference_dt,
                "max_script_seconds": self.settings.get("max_script_seconds", 8.),
                "complete_started_blend": self.settings.get("complete_started_blend", False)},
            **({"command_owner": "source_flat_sampler_after_physical_stability",
                "followup_target": "published_source_command_each_policy_interval"}
                if self.source_flat_handoff else {}),
            **({"episode_window": deepcopy(self.episode_window)}
                if self.episode_window is not None else {}),
            "wheel_balance": deepcopy(self.wheel_balance),
            "active_reset_sampler": ({"class": type(self.pose_curriculum).__name__,
                "settings": deepcopy(self.pose_curriculum.settings)}
                if self.pose_curriculum is not None else None),
            "initial_scene_selection": {"scene_groups": list(env.scene_groups),
                "terrain_kinds": list(env.kinds), "original_profile_names": list(self.profile_names),
                "enabled": self.enabled.detach().cpu().tolist(),
                "followup_command": self.followup_command.detach().cpu().tolist(),
                "initial_root_velocity_com_world6": self.initial_root_velocity.detach().cpu().tolist(),
                "initial_joint_velocity_native_order": self.initial_joint_velocity.detach().cpu().tolist(),
                "native_joint_names": list(env.robot.joint_names),
                "initial_release_s": self.initial_release.detach().cpu().tolist(),
                "required_platform_height_m": self.required_platform_height.detach().cpu().tolist(),
                # JSON has no infinity. Null explicitly represents the existing
                # unconstrained negative-infinity minimum, never a finite zero.
                "min_ending_x_m": [None if value == -math.inf else value
                    for value in self.min_ending_x.detach().cpu().tolist()],
                "min_ending_x_null_semantics": "negative_infinity_no_minimum"},
            "predicate_scope": "legacy_physical_outcome_diagnostics_not_v7_qualification",
        }
        if self._sampled_birth_recipe_names:
            self._source_identity["sampled_birth_recipes"] = {
                "recipe_names_in_local_id_order": list(self._sampled_birth_recipe_names),
                "metadata": json.loads(self._sampled_birth_recipes_json),
                "profile_labels": ["sampled/" + name for name in self._sampled_birth_recipe_names],
                "label_semantics": "recipe_families_not_fixed_initial_poses",
                "request_semantics": "requested_reset_not_native_readback",
                "source_leg_ground_birth": "source_leg_sampling_label_not_floor_or_motor_off_state"}
        self.source_identity_sha256 = hashlib.sha256(json.dumps(self._source_identity,
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def _register_sampled_birth_recipes(self, recipes, profile_names):
        self._sampled_birth_recipe_names = ()
        self._sampled_birth_recipes_json = "{}"
        if recipes is None:
            return profile_names
        if (not isinstance(recipes, Mapping) or not recipes or self.pose_curriculum is not None
                or any(not isinstance(name, str) or not name.strip()
                       or not isinstance(metadata, Mapping) or not metadata
                       for name, metadata in recipes.items())):
            raise ValueError("Sampled recovery recipes require named nonempty JSON metadata without a pose curriculum")
        try:
            frozen = json.dumps(dict(recipes), sort_keys=True, allow_nan=False)
            normalized = json.loads(frozen)
        except (TypeError, ValueError) as error:
            raise ValueError("Sampled recovery recipe metadata must be finite JSON") from error
        if normalized != dict(recipes):
            raise ValueError("Sampled recovery recipe metadata must use JSON objects and values")
        names = tuple(recipes)
        labels = tuple("sampled/" + name for name in names)
        if set(labels) & set(profile_names):
            raise ValueError("Sampled recovery recipe labels collide with fixed profiles")
        self._sampled_birth_recipe_names = names
        self._sampled_birth_recipes_json = frozen
        self._sampled_birth_profile_offset = len(profile_names)
        with torch.inference_mode(False):
            self._pending_birth_profiles = torch.full_like(self.episode_ids, -1)
            self._pending_birth_expected_ids = torch.full_like(self.episode_ids, -1)
        self._pending_birth_requests, self._episode_birth_requests = {}, {}
        return profile_names + labels

    def _sampled_birth_request_record(self, birth, converted, order):
        """Clone requested vectors, including source leg labels, before native writes."""
        count, device = len(birth.env_ids), birth.env_ids.device
        record = dict(root_pose=converted[0], joint_position=converted[1][:, order],
                      root_velocity=converted[2], joint_velocity=converted[3][:, order])
        fields = {"interpolation_fraction": (), "yaw_offset_rad": (),
            "requested_root_position_m": (3,), "requested_root_quaternion_xyzw": (4,),
            "proposal_counts": (), "accepted_counts": (), "rejection_counts": (),
            "placement_raise_m": (), "minimum_shape_clearance_m": ()}
        record.update({name: getattr(birth, name) for name in fields})
        for name in ("ground_birth", "leg_mode", "virtual_leg_length_m", "virtual_leg_angle_rad",
                     "wheel_angle_rad", "control_velocity_rad_s"):
            key = "source_leg/" + name
            record[key] = getattr(birth.leg_requests, name)
            fields[key] = (() if name == "ground_birth" else (6,) if name == "control_velocity_rad_s" else (2,))
        for name, value in birth.rejection_reasons.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Recovery rejection reasons require nonempty names")
            record["rejection/" + name], fields["rejection/" + name] = value, ()
        for name, shape in fields.items():
            value = record[name]
            if (not isinstance(value, torch.Tensor) or value.shape != (count, *shape)
                    or value.device != device or not bool(torch.isfinite(value).all())):
                raise ValueError("Recovery birth diagnostics require finite complete request tensors")
        for name in ("proposal_counts", "accepted_counts", "rejection_counts"):
            if record[name].dtype != torch.long or bool((record[name] < 0).any()):
                raise ValueError("Recovery proposal accounting requires nonnegative long counts")
        if (not bool((record["accepted_counts"] == 1).all())
                or not torch.equal(record["proposal_counts"], record["accepted_counts"] + record["rejection_counts"])):
            raise ValueError("Recovery proposal accounting must match the accepted request")
        with torch.inference_mode(False):
            return {name: value.detach().clone() for name, value in record.items()}

    def _pending_sampled_birth_ids(self, ids):
        if not getattr(self, "_sampled_birth_recipe_names", ()):
            return None
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long
                or ids.device != self.episode_ids.device or ids.unique().numel() != ids.numel()
                or bool(((ids < 0) | (ids >= self.env.num_envs)).any())):
            raise ValueError("Recovery reset IDs require unique local long indices")
        sampled = ids[self._pending_birth_profiles[ids] >= 0]
        if not torch.equal(self._pending_birth_expected_ids[sampled], self.episode_ids[sampled] + 1):
            raise ValueError("Pending recovery birth belongs to a different next episode")
        return sampled

    def sampled_birth_requests(self, ids=None):
        """Read current-episode requests only; never a native state or PPO input."""
        if not getattr(self, "_sampled_birth_recipe_names", ()):
            return None
        if ids is None:
            ids = torch.arange(self.env.num_envs, device=self.episode_ids.device)
        selected = ids[self.episode_profile_ids[ids] >= self._sampled_birth_profile_offset]
        if not len(selected):
            return None
        with torch.inference_mode(False):
            return {"scope": "requested_reset_not_native_readback", "env_ids": selected.clone(),
                "source_identity_sha256": self.source_identity_sha256,
                "episode_ids": self.episode_ids[selected].clone(),
                "episode_profile_ids": self.episode_profile_ids[selected].clone(),
                "recipe_ids": (self.episode_profile_ids[selected] - self._sampled_birth_profile_offset).clone(),
                "recipe_names_in_local_id_order": list(self._sampled_birth_recipe_names),
                "recipe_metadata": json.loads(self._sampled_birth_recipes_json),
                "joint_names": list(self.env.robot.joint_names), "root_velocity_frame": "com_world",
                "source_leg_ground_birth_semantics": "source_leg_sampling_label_not_floor_or_motor_off_state",
                "requests": {name: value[selected].clone() for name, value in self._episode_birth_requests.items()}}

    def _load_initial_profile(self, index, profile, name):
        """Read a frozen reset state in native joint order, without simulator writes."""
        names = self.env.robot.joint_names
        positions = profile.get("full_joint_positions")
        velocities = profile.get("full_joint_velocities", dict.fromkeys(names, 0.))
        if (len(set(names)) != len(names)
                or any(not isinstance(values, Mapping) or set(values) != set(names)
                       for values in (positions, velocities))):
            raise ValueError(f"Recovery profile requires every native joint exactly once: {name}")
        try:
            tensors = [torch.as_tensor(values, dtype=self.joint_pose.dtype,
                                       device=self.joint_pose.device) for values in (
                profile.get("root_position_m"), profile.get("root_quaternion_xyzw"),
                [positions[key] for key in names], [velocities[key] for key in names],
                profile.get("root_linear_velocity_m_s", [0., 0., 0.]),
                profile.get("root_angular_velocity_rad_s", [0., 0., 0.]))]
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"Invalid recovery profile state values: {name}") from error
        shapes = ((3,), (4,), (len(names),), (len(names),), (3,), (3,))
        if any(value.shape != shape or not bool(torch.isfinite(value).all())
               for value, shape in zip(tensors, shapes)):
            raise ValueError(f"Recovery profile state must have finite complete vectors: {name}")
        position, quaternion, joint_position, joint_velocity, linear, angular = tensors
        root_velocity = torch.cat((linear, angular))
        frame = profile.get("root_velocity_frame")
        # Native reset velocity is root-COM linear velocity and angular velocity,
        # both in world axes. Zero legacy profiles need no coordinate declaration.
        if ((frame is not None and frame != "com_world")
                or (bool(root_velocity.ne(0.).any()) and frame != "com_world")):
            raise ValueError(f"Nonzero recovery root velocity requires root_velocity_frame=com_world: {name}")
        self.root_pose[index] = torch.cat((position, quaternion))
        self.joint_pose[index] = joint_position
        self.initial_root_velocity[index] = root_velocity
        self.initial_joint_velocity[index] = joint_velocity

    @property
    def pure_rl(self):
        return (~self.enabled) | (self.release_finished & self.fsm.pure_rl & ~self.fsm.failed & ~self.premature_release)

    @property
    def prelude(self):
        return self.enabled & ~self.pure_rl

    @property
    def failed(self):
        timed_script = self.prelude
        if self.settings.get("complete_started_blend", False):
            timed_script = timed_script & ~self.fsm.blending
        if getattr(self, "episode_window", None) is not None:
            # The full first actor interval must start before the prelude
            # deadline; a final scripted substep cannot claim that interval.
            expired = (self.enabled & (self.window_first_actor_tick < 0)
                & (self.window_policy_ticks >= round(self.episode_window["prelude_seconds"] / self.env.policy_dt)))
        else:
            expired = timed_script & (self.elapsed >= self.settings.get("max_script_seconds", 8.))
        return (self.fsm.failed | self.premature_release
                | expired)

    def failure_reasons(self):
        """Attribute terminal outcomes independently of the PPO interval mask."""
        other = self.failed & ~self.premature_release
        handed_over = self.handover_count > 0
        if self.episode_window is not None:
            handed_over = self.window_first_actor_tick >= 0
        return {"recovery_script": other & ~handed_over,
                "recovery_actor_failure": other & handed_over,
                "recovery_premature_release": self.premature_release.clone()}

    def _apply_sampled_birth(self, ids, root, joints, velocity, joint_velocity, birth, recipe_ids):
        """Consume a reset proposal only; the caller owns its admission and journal.

        Sampled world poses already include placement and environment offsets.
        Static profile buffers remain unchanged for subsequent anchor resets.
        This interface does not select a distribution or enable a native course.
        """
        from wheeled_tasks.chassis.tasks.recovery.recovery_birth_sampling import RecoveryBirthResult
        if not getattr(self, "_sampled_birth_recipe_names", ()):
            raise ValueError("Sampled recovery births require registered recipe identities")
        if self.pose_curriculum is not None:
            raise ValueError("Sampled recovery births cannot replace an active pose curriculum")
        if not isinstance(birth, RecoveryBirthResult):
            raise ValueError("Recovery reset requires a complete RecoveryBirthResult")
        active = self.enabled[ids]
        chosen = ids[active]
        if (birth.env_ids.dtype != ids.dtype or birth.env_ids.device != ids.device
                or not torch.equal(birth.env_ids, chosen)):
            raise ValueError("Sampled recovery IDs must exactly match this reset's recovery lanes")
        native_names = tuple(self.env.robot.joint_names)
        names = birth.tree_joint_names
        if (not isinstance(names, tuple) or len(names) != len(native_names)
                or len(set(names)) != len(names) or set(names) != set(native_names)
                or birth.asset_manifest_sha256 != self.env.cfg["asset_manifest_sha256"]):
            raise ValueError("Sampled recovery births must match the selected named mechanical asset")
        count = len(chosen)
        if (not isinstance(recipe_ids, torch.Tensor) or recipe_ids.shape != (count,)
                or recipe_ids.dtype != torch.long or recipe_ids.device != ids.device
                or bool(((recipe_ids < 0) | (recipe_ids >= len(self._sampled_birth_recipe_names))).any())):
            raise ValueError("Sampled recovery births require local long recipe IDs for every enabled lane")
        if (birth.valid.shape != (count,) or birth.valid.dtype != torch.bool
                or birth.valid.device != ids.device or not bool(birth.valid.all())):
            raise ValueError("Unaccepted recovery births cannot enter reset outputs")
        outputs = ((root, (len(ids), 7)), (joints, (len(ids), len(native_names))),
                   (velocity, (len(ids), 6)), (joint_velocity, (len(ids), len(native_names))))
        for output, shape in outputs:
            if (not isinstance(output, torch.Tensor) or output.shape != shape
                    or output.device != ids.device or not output.is_floating_point()):
                raise ValueError("Sampled recovery reset requires complete floating reset outputs")
        fields = ((birth.root_pose, (count, 7), root),
                  (birth.joint_position, (count, len(names)), joints),
                  (birth.root_velocity, (count, 6), velocity),
                  (birth.joint_velocity, (count, len(names)), joint_velocity))
        converted = []
        for value, shape, output in fields:
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.device != ids.device or not value.is_floating_point()
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Sampled recovery state requires finite complete vectors")
            converted.append(value.to(output))
            if not bool(torch.isfinite(converted[-1]).all()):
                raise ValueError("Sampled recovery state overflows the native reset dtype")
        if not torch.allclose(birth.root_pose[:, 3:].norm(dim=-1),
                              torch.ones(count, device=ids.device, dtype=birth.root_pose.dtype),
                              atol=1e-6, rtol=0.):
            raise ValueError("Sampled recovery root orientation must be unit XYZW")
        order = [names.index(name) for name in native_names]
        record = self._sampled_birth_request_record(birth, converted, order)
        # Allocate ordinary mutable history even when the caller is sampling.
        with torch.inference_mode(False):
            for name, value in record.items():
                if name not in self._pending_birth_requests:
                    shape = (self.env.num_envs, *value.shape[1:])
                    self._pending_birth_requests[name] = value.new_zeros(shape)
                    self._episode_birth_requests[name] = value.new_zeros(shape)
                elif (self._pending_birth_requests[name].shape[1:] != value.shape[1:]
                        or self._pending_birth_requests[name].dtype != value.dtype):
                    raise ValueError("Recovery request diagnostic layout changed within this owner")
        # Validate every field before touching any caller-owned output tensor.
        root[active], joints[active] = converted[0], converted[1][:, order]
        velocity[active], joint_velocity[active] = converted[2], converted[3][:, order]
        for value in self._pending_birth_requests.values():
            value[chosen] = 0
        for name, value in record.items():
            self._pending_birth_requests[name][chosen] = value
        self._pending_birth_profiles[chosen] = recipe_ids + self._sampled_birth_profile_offset
        self._pending_birth_expected_ids[chosen] = self.episode_ids[chosen] + 1

    def initial_state(self, ids, root, joints, velocity, joint_velocity=None, *,
                      birth_result=None, birth_recipe_ids=None):
        self._pending_sampled_birth_ids(ids)
        if birth_result is not None:
            self._apply_sampled_birth(ids, root, joints, velocity, joint_velocity, birth_result, birth_recipe_ids)
            return
        if birth_recipe_ids is not None:
            raise ValueError("Recovery recipe IDs require their sampled birth result")
        if self.pose_curriculum is not None:
            actor_clock = stage_actor_updates(self.env)
            if (self.env.cfg.get("integrated_runtime_candidate") is not None
                    and not self.env.cfg.get("evaluation_exact_cases")):
                from wheeled_tasks.chassis.application.compatibility.integrated_runtime import component_actor_updates
                actor_clock = component_actor_updates(self.env, "recovery")
            bank = self.pose_curriculum.reset(ids, actor_clock)
            selected = bank >= 0
            lanes, choices = ids[selected], bank[selected]
            for field in ("root_pose", "joint_pose", "initial_release", "required_platform_height", "min_ending_x"):
                getattr(self, field)[lanes] = getattr(self.pose_curriculum, field + "_bank")[choices]
            # These existing banks explicitly admit zero initial velocities only.
            self.initial_root_velocity[lanes] = 0.
            self.initial_joint_velocity[lanes] = 0.
            if self.settings.get("prefix_curriculum", {}).get("enabled"):
                self.pose_curriculum.apply_lesson(self, lanes)
        active = self.enabled[ids]
        chosen = ids[active]
        if joint_velocity is None:
            if bool(self.initial_joint_velocity[chosen].ne(0.).any()):
                raise ValueError("Nonzero recovery joint velocity requires a joint_velocity reset output")
        elif (not isinstance(joint_velocity, torch.Tensor)
                or joint_velocity.shape != joints.shape or joint_velocity.dtype != joints.dtype
                or joint_velocity.device != joints.device):
            raise ValueError("Recovery joint_velocity output must match the joint position tensor")
        root[active] = self.root_pose[chosen]
        root[active, :3] += self.env.origins[chosen]
        joints[active] = self.joint_pose[chosen]
        velocity[active] = self.initial_root_velocity[chosen]
        if joint_velocity is not None:
            joint_velocity[active] = self.initial_joint_velocity[chosen]
        if getattr(self, "_sampled_birth_recipe_names", ()):
            self._pending_birth_profiles[chosen] = -1
            self._pending_birth_expected_ids[chosen] = -1

    def reset(self, ids):
        # Check the staged identity before the FSM or episode state can change.
        sampled = self._pending_sampled_birth_ids(ids)
        env = self.env
        transport = getattr(env, "usb_transport", None)
        q = env.robot.data.joint_pos.torch[:, env.ids] if transport is None else transport.feedback[:, :6]
        self.encoders.reset(ids, q)
        self.fsm.reset(ids, self.encoders.q, self.enabled & (self.initial_release == 0.))
        for name in ("elapsed", "true_stable_seconds", "true_stable_peak", "followup_seconds",
                     "ever_blended", "handover_count", "true_success", "wheel_probe_torque", "motion_release_observed",
                     "wheel_balance_active", "premature_release"):
            getattr(self, name)[ids] = 0
        if self.episode_window is not None:
            self.window_policy_ticks[ids] = self.window_eligible_ticks[ids] = 0
            self.window_first_actor_tick[ids] = -1
        self.release_finished[ids] = ~self.enabled[ids] | (self.initial_release[ids] == 0.)
        self._motor_enabled_for_substep[ids] = self.release_finished[ids]
        self.goal_platform[ids] = self.required_platform_height[ids] == 0.
        self.imu.previous_velocity[ids] = root_link_linear_velocity(env)[ids]
        if self.direct_takeover:
            acceleration = -env.robot.data.projected_gravity_b.torch * 9.81
            self.acceleration_cache.reset(ids, acceleration, env.physical_tick)
            self.sensor_guard.reset(ids)
            self.sensor_sample_sequences[ids] = 0
            self.sensor_sample_stamps_s[ids] = 0.
            self.sensor_observed_stamps_s[ids] = torch.nan
            if self.sensor_status is not None:
                from wheeled_tasks.chassis.observations.recovery_sensor_guard import RecoverySensorIssue
                self.sensor_status.valid[ids] = False
                self.sensor_status.issue[ids] = int(RecoverySensorIssue.MISSING)
                self.sensor_status.invalid_mask[ids] = 0xff
                for name in ("maximum_motor_age_ms", "imu_age_ms", "acceleration_age_ms"):
                    getattr(self.sensor_status, name)[ids] = -1.
                self.sensor_status.channel_age_s[ids] = -.001
            self.sensor_history_start_s[ids] = torch.nan
            self.sensor_history_last_s[ids] = torch.nan
            self.valid_sensor_history_s[ids] = 0.
            self.capture_ready[ids] = False
        for name in ("previous_velocity", "previous_gyro", "previous_pulse", "evidence",
                     "bad_samples", "quiet_time", "lost_time"):
            getattr(self.probe, name)[ids] = 0
        self.probe.age[ids] = math.inf
        self.episode_ids[ids] += 1
        self.episode_lesson_ids[ids] = -1
        self.episode_profile_ids[ids] = self._original_profile_ids[ids]
        if sampled is not None:
            self.episode_profile_ids[sampled] = self._pending_birth_profiles[sampled]
            for name, value in self._pending_birth_requests.items():
                self._episode_birth_requests[name][sampled] = value[sampled]
            self._pending_birth_profiles[ids] = -1
            self._pending_birth_expected_ids[ids] = -1
        if self.pose_curriculum is not None:
            self.episode_profile_ids[ids] = self.pose_curriculum.profile_ids[ids]
            if self.settings.get("prefix_curriculum", {}).get("enabled"):
                self.episode_lesson_ids[ids] = self.pose_curriculum.level_ids[ids]
        for name in self.EVENT_NAMES:
            self._event_latched[name][ids] = False
            self._policy_events[name][ids] = False
            self._policy_occurrences[name][ids] = -1
        self.override_commands()

    def override_commands(self):
        env, active = self.env, self.enabled
        hold = active & (~self.release_finished | ~self.fsm.motion_released)
        # The source sampler keeps advancing underneath the physical prelude.
        # After stable release its currently published command owns this lane;
        # no fixed followup, resample, RNG reset or command-clock restart.
        selected = hold if self.source_flat_handoff else active
        env.commands[selected] = self.followup_command[selected]
        env.commands[hold, :2] = 0.
        env.raw_height[selected] = self.followup_command[selected, 2]
        env.command_target[active] = env.commands[active, :2]
        env.command_acceleration[active] = 0.

    def begin_policy(self):
        self.learning_mask.copy_(self.pure_rl)
        self.policy_motion_hold.copy_(self.enabled & ~self.fsm.motion_released)
        # Paid interval ownership must survive an in-interval motion release.
        # This is readout only; override_commands remains the command owner.
        self.policy_command_override_active.copy_(
            self.enabled & (~self.release_finished | ~self.fsm.motion_released))
        for name in self.EVENT_NAMES:
            self._policy_events[name].zero_()
            self._policy_occurrences[name].fill_(-1)
        self._event_policy_open = True

    def advance_episode_window(self, learning_mask):
        """Close one physical policy interval, independently of optimizer credit."""
        if self.episode_window is None:
            raise ValueError("Extended recovery timing was not selected")
        if (learning_mask.shape != self.enabled.shape or learning_mask.dtype != torch.bool
                or learning_mask.device != self.enabled.device):
            raise ValueError("Recovery window needs the final bool[N] learning ownership mask")
        if self._window_last_policy_tick == self.env.tick:
            raise RuntimeError("Recovery episode window counted a policy interval twice")
        self._window_last_policy_tick = self.env.tick
        eligible = self.enabled & learning_mask
        first = eligible & (self.window_first_actor_tick < 0)
        self.window_first_actor_tick[first] = self.window_policy_ticks[first]
        self.window_policy_ticks += self.enabled
        self.window_eligible_ticks += eligible

    def episode_window_timeouts(self, ordinary_timeouts):
        """Recovery owns its post-handoff horizon; other lanes keep their limit."""
        elapsed = self.window_policy_ticks - self.window_first_actor_tick
        completed = ((self.window_first_actor_tick >= 0)
            & (elapsed >= round(self.episode_window["post_handoff_seconds"] / self.env.policy_dt)))
        return torch.where(self.enabled, completed, ordinary_timeouts)

    def _record_physical_event(self, name, occurred, *, occurrence_tick=None):
        pulse = occurred & self.enabled & ~self._event_latched[name]
        self._event_latched[name] |= pulse
        self._policy_events[name] |= pulse
        tick = self.reference_tick if occurrence_tick is None else occurrence_tick
        self._policy_occurrences[name] = torch.where(pulse, tick, self._policy_occurrences[name])

    def source_identity(self):
        """Construction-bound source/profile metadata, not an admission receipt."""
        return deepcopy(self._source_identity)

    def policy_transition(self, done, terminated, timeouts):
        """Snapshot old episode evidence before auto-reset, regardless of logging.

        Pulses include script/blend substeps for the physical funnel. Consumers
        must separately intersect the whole-interval mask and actual optimized
        population for PPO exposure. Readout never advances a counter or FSM.
        The environment appends independently observed returned episode IDs.
        """
        if not self._event_policy_open:
            raise ValueError("Recovery transition requires begin_policy before the elapsed interval")
        for name, value in (("done", done), ("terminated", terminated), ("timeouts", timeouts)):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
                    or value.shape != self.enabled.shape or value.device != self.enabled.device):
                raise ValueError(f"Recovery {name} requires bool[N] on the producer device")
        return {"schema": self.TRANSITION_SCHEMA,
            "source_identity_sha256": self.source_identity_sha256,
            "enabled": self.enabled.clone(), "episode_ids": self.episode_ids.clone(),
            "episode_lesson_ids": self.episode_lesson_ids.clone(),
            "episode_profile_ids": self.episode_profile_ids.clone(),
            "events": {name: value.clone() for name, value in self._policy_events.items()},
            "occurrence_ids": {name: value.clone() for name, value in self._policy_occurrences.items()},
            "policy_learning_mask": self.learning_mask.clone(), "phase": self.fsm.phase.clone(),
            "done": done.clone(), "terminated": terminated.clone(), "timeouts": timeouts.clone(),
            "failure_reasons": self.failure_reasons()}

    def bootstrap_mask(self, terminated):
        """Evaluate after reset, but never bootstrap a physical terminal step."""
        return self.pure_rl & ~terminated

    def observe_feedback(self, q):
        """Consume the one CONTROL encoder stream once per 1 kHz feedback tick."""
        continuous = self.encoders.update(q)
        return torch.where(self.enabled[:, None], continuous, q) if self.source_flat_handoff else continuous

    def project(self, q):
        """Read the same continuous feedback winding without advancing it."""
        return self.encoders.project(q)

    def project_feedback(self, q):
        """Shared read-only encoder projection for other sensor consumers."""
        projected = self.project(q)
        # Decode and final PD must use the same winding. Source ordinary lanes
        # keep their raw native angles, matching observe_feedback.
        return torch.where(self.enabled[:, None], projected, q) if self.source_flat_handoff else projected

    def _direct_acceleration(self):
        env = self.env
        tick = env.physical_tick
        if type(tick) is not int or tick < 0 or tick < self.acceleration_cache.last_physical_tick:
            raise ValueError("Direct recovery requires the monotonic actual physical tick")
        if tick != self.acceleration_cache.last_physical_tick:
            acceleration = self.imu.sample(root_link_linear_velocity(env),
                env.robot.data.root_link_pose_w.torch[:, 3:])
            self.acceleration_cache.append(acceleration, tick)
        transport = getattr(env, "_pc_feedback_transport", None)
        requested = (torch.full_like(self.valid_sensor_history_s, tick * env.dt)
                     if transport is None else transport.sensor_timestamp_s)
        return self.acceleration_cache.read(requested)

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
        if self.direct_takeover:
            acceleration, acceleration_stamp = self._direct_acceleration()
        else:
            acceleration = self.imu.sample(root_link_linear_velocity(env),
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
        kwargs = {}
        if self.direct_takeover:
            transport = getattr(env, "_pc_feedback_transport", None)
            now = env.physical_tick * env.dt
            stamp = (torch.full_like(self.valid_sensor_history_s, now) if transport is None
                     else transport.sensor_timestamp_s)
            stamps = torch.cat((stamp[:, None].expand(-1, 7), acceleration_stamp[:, None]), -1)
            self.sensor_observed_stamps_s.copy_(stamps)
            # Virtual reconstructed-frame identity, with the deployment clock's
            # ns resolution. Failed snapshots do not replace accepted metadata.
            changed = (stamps * 1e9).round() != (self.sensor_sample_stamps_s * 1e9).round()
            sequences = self.sensor_sample_sequences + changed.long()
            status = self.sensor_guard.update(now, sequences, stamps)
            self.sensor_sample_sequences[status.valid] = sequences[status.valid]
            self.sensor_sample_stamps_s[status.valid] = stamps[status.valid]
            self.sensor_status = status
            sensor_valid = (torch.isfinite(continuous_q).all(-1) & torch.isfinite(dq).all(-1)
                & torch.isfinite(gravity).all(-1) & (gravity.norm(dim=-1) > .5)
                & torch.isfinite(gyro).all(-1) & torch.isfinite(acceleration).all(-1)
                & status.valid)
            oldest_stamp = stamps.amin(-1)
            first = sensor_valid & ~torch.isfinite(self.sensor_history_start_s)
            self.sensor_history_start_s[first] = oldest_stamp[first]
            self.sensor_history_start_s[~sensor_valid] = torch.nan
            self.sensor_history_last_s.copy_(torch.where(sensor_valid, oldest_stamp, torch.nan))
            history_s = ((oldest_stamp * 1e9).round()
                         - (self.sensor_history_start_s * 1e9).round()) * 1e-9
            self.valid_sensor_history_s.copy_(torch.where(sensor_valid,
                history_s.clamp_min(0.), 0.))
            self.capture_ready.copy_(direct_capture_ready(continuous_q, dq, gravity, gyro, acceleration,
                self.observed_height, plausible, sensor_valid, self.capture_reference, self.shell_vertices))
            kwargs = {"capture_ready": self.capture_ready, "valid_sensor_history_s": self.valid_sensor_history_s}
        for _ in range(self.script_substeps):
            self.fsm.update(continuous_q, dq, gravity, gyro, self.observed_height, plausible, body_clear, **kwargs)
        entered = self.release_finished & ~before & self.fsm.pure_rl
        ready_event = (entered if self.direct_takeover else self.release_finished & self.fsm.blending
                       & (previous_phase != int(RecoveryPhase.BLEND)))
        self._record_physical_event("script_ready", ready_event, occurrence_tick=self.reference_tick + 1)
        self._record_physical_event("rl_entered", self.release_finished & ~before & self.fsm.pure_rl,
            occurrence_tick=self.reference_tick + 1)
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
        pitch/gyro effort minus wheel damping. The caller owns the declared PD
        cadence. The last scripted target is retained through torque blending.
        """
        wheels, settings = list(self.env.v5.WHEELS), self.wheel_balance
        tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        capture = self.fsm.phase == int(RecoveryPhase.CAPTURE)
        preparing = self.fsm.phase == int(RecoveryPhase.PREPARE)
        push_off = (self.fsm.low_pose_capture & self.fsm.push_off_required
                    & (self.fsm.phase == int(RecoveryPhase.THRUST)))
        entered_blend = self.fsm.blending & (previous_phase == int(RecoveryPhase.PREPARE))
        limit = torch.where(capture, tilt.new_tensor(math.radians(settings["capture_tilt_limit_deg"])),
                            tilt.new_tensor(math.radians(settings["prepare_tilt_limit_deg"])))
        eligible = (self.enabled & self.release_finished & support & (tilt < limit)
                    & (preparing | capture | push_off | entered_blend) & settings["enabled"])
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
        # Retain this interval's decision across transport/response and the
        # elapsed increment below; the next interval may have another owner.
        self._motor_enabled_for_substep.copy_(~active | motor_enabled)
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

    def gate_active_effort(self, effort):
        """Remove retained drive output without clearing transport/response state."""
        return torch.where(self._motor_enabled_for_substep[:, None], effort, 0.)

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
        target = env.commands if self.source_flat_handoff else self.followup_command
        required = self.required_platform_height > 0.
        if goal_platform is not None and goal_platform.shape != required.shape:
            raise ValueError("Recovery platform evidence must match environment count")
        self.goal_platform = ~required | (torch.zeros_like(required) if goal_platform is None else goal_platform.bool())
        actual_stable = (self.enabled & self.pure_rl & contact.all(-1) & ~undesired
            & self.goal_platform
            & (gravity[:, 2] < -math.cos(math.radians(12.)))
            & (omega[:, :2].norm(dim=-1) < .75)
            & ((height - target[:, 2]).abs() <= .05))
        stand = target[:, :2].abs().amax(-1) < .01
        qualified_before = self.true_stable_peak >= 1. - 1e-5
        # The interval ending at the first observed release still belongs to
        # the hold. Later movement cannot manufacture missing hold evidence.
        stable_hold = actual_stable & ~self.motion_release_observed & (velocity[:, :2].norm(dim=-1) < .25)
        self.true_stable_seconds = torch.where(stable_hold,
            self.true_stable_seconds + env.policy_dt, 0.)
        self.true_stable_peak = torch.maximum(self.true_stable_peak, self.true_stable_seconds)
        self._record_physical_event("legacy_physical_hold_qualified",
            self.true_stable_peak >= 1. - 1e-5)
        released = self.enabled & self.pure_rl & self.fsm.motion_released
        first_release = released & ~self.motion_release_observed
        if not self.direct_takeover:
            self.premature_release |= first_release & (self.true_stable_peak < 1. - 1e-5)
        self.motion_release_observed |= released
        tracking = ((velocity[:, 0] - target[:, 0]).abs() <= .12)
        tracking &= (omega[:, 2] - target[:, 1]).abs() <= .2
        tracking &= ~stand | (velocity[:, :2].norm(dim=-1) < .10)
        completing = (actual_stable & self.fsm.motion_released & tracking
                       & (qualified_before | self.direct_takeover))
        self.followup_seconds = torch.where(completing, self.followup_seconds + env.policy_dt, 0.)
        self.true_success = self.enabled & (self.followup_seconds >= 1. - 1e-5) & ~self.failed
        self._record_physical_event("legacy_followup_complete", self.true_success)
        return self.true_success

    def diagnostics(self):
        failure_code = self.fsm.failure_code.clone()
        failure_code[self.failed & (failure_code == 0)] = 6
        failure_code[self.premature_release] = self.PREMATURE_RELEASE_FAILURE
        leg_error = self.fsm.script_targets[:, list(self.env.v5.LEGS)] - self.encoders.q[:, list(self.env.v5.LEGS)]
        pose = self.pose_curriculum.diagnostics() if self.pose_curriculum is not None else {}
        sensor = {}
        if self.direct_takeover and self.sensor_status is not None:
            sensor = {"recovery_sensor_time_valid": self.sensor_status.valid.clone(),
                "recovery_sensor_issue": self.sensor_status.issue.clone(),
                "recovery_sensor_invalid_mask": self.sensor_status.invalid_mask.clone(),
                "recovery_sensor_motor_age_ms": self.sensor_status.maximum_motor_age_ms.clone(),
                "recovery_sensor_imu_age_ms": self.sensor_status.imu_age_ms.clone(),
                "recovery_sensor_acceleration_age_ms": self.sensor_status.acceleration_age_ms.clone(),
                "recovery_acceleration_stamp_s": self.sensor_observed_stamps_s[:, 7].clone(),
                "recovery_accepted_acceleration_stamp_s": self.sensor_sample_stamps_s[:, 7].clone()}
        window = {}
        if self.episode_window is not None:
            started = self.window_first_actor_tick >= 0
            elapsed = torch.where(started,
                self.window_policy_ticks - self.window_first_actor_tick, 0)
            window = {
                "recovery_episode_elapsed_s": self.window_policy_ticks * self.env.policy_dt,
                "recovery_first_whole_actor_start_s": torch.where(started,
                    self.window_first_actor_tick * self.env.policy_dt, -1.),
                "recovery_post_handoff_elapsed_s": elapsed * self.env.policy_dt,
                "recovery_post_handoff_eligible_s": self.window_eligible_ticks * self.env.policy_dt,
                "recovery_post_handoff_full_window": (self.enabled & ~self.failed
                    & (self.window_eligible_ticks >= round(self.episode_window["post_handoff_seconds"] / self.env.policy_dt)))}
        return {**self.fsm.diagnostics(), **pose, **window,
            **sensor,
            **({"recovery_valid_sensor_history_s": self.valid_sensor_history_s.clone(),
                "recovery_direct_capture_ready": self.capture_ready.clone()} if self.direct_takeover else {}),
            "recovery_enabled": self.enabled.clone(),
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
            "recovery_script_seconds": torch.minimum(self.elapsed, self.initial_release + self.fsm.age_ticks * self.fsm.dt),
            "recovery_observed_height": self.observed_height.clone(),
            "recovery_observed_support": self.observed_support.clone(),
            "recovery_continuous_q": self.encoders.q.clone(),
            "recovery_leg_target_error_rad": leg_error.clone(),
            "recovery_wheel_balance_active": self.wheel_balance_active.clone(),
            "recovery_wheel_velocity_target": self.fsm.script_targets[:, list(self.env.v5.WHEELS)].clone()}
