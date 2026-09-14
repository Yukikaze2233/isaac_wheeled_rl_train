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
from isaaclab.sensors import ContactSensorCfg
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
from .contact_sensor import V40ContactSensor

BODY_NAMES = ["base_link", "L_link1", "L_link2", "L_link3", "R_link1", "R_link2", "R_link3"]
WHEEL_BODY_NAMES = ["L_link3", "R_link3"]
NON_WHEEL_BODY_NAMES = [name for name in BODY_NAMES if name not in WHEEL_BODY_NAMES]


class V40Env(DirectRLEnv):
    cfg: V40EnvCfg

    def __init__(self, cfg: V40EnvCfg, render_mode: str | None = None, **kwargs):
        self.contract = load_contract(cfg.contract_path)
        if cfg.ground_usd_path is not None:
            from wheeled_tasks.v40.round4 import verify_ground_usd
            self.ground_asset_report = verify_ground_usd(cfg.ground_usd_path)
        if self.contract.get('round3', {}).get('stage') == 'B1':
            from wheeled_tasks.v40.round3 import B1MaterialBuckets
            self._b1_materials = B1MaterialBuckets(
                self.contract['round3']['material_randomization'], cfg.scene.num_envs,
                restored_report=cfg.round3_material_restore,
            )
        elif cfg.round3_material_restore is not None:
            raise ValueError('saved randomized material mapping is only supported for B1')
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
            usd_seed=cfg.usd_seed,
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
        # Lab 3 Warp actuator kernels require int32 joint indices.
        self._joint_ids = self._named_indices(self.robot.joint_names, self.joint_names, "joint").to(torch.int32)
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
        if 'round3' in self.contract:
            from wheeled_tasks.v40.round3 import Round3Commands
            self._round3_commands = Round3Commands(self.contract, cfg.stage, self.num_envs, self.device)
        if type(cfg.wheel_slip_diagnostics) is not bool or (cfg.wheel_slip_diagnostics and self.num_envs > 64):
            raise ValueError('wheel slip diagnostics are opt-in and limited to 64 environments')
        if getattr(self, '_b1_materials', None) is not None:
            self.round3_material_report = self.check_round3_materials()
        if 'round4' in self.contract:
            from wheeled_tasks.v40.round4 import PushCurriculum, FullManeuverCommands, effective_dynamic_from_readback
            self._push_curriculum = PushCurriculum(self.contract, self.num_envs, self.device)
            self._push_curriculum.reset(torch.arange(self.num_envs, device=self.device))
            self.first_push_report = None
            self._full_commands = FullManeuverCommands(
                self.contract, cfg.stage, self.num_envs, self.device,
                effective_dynamic_from_readback(self.round3_material_report, self.num_envs),
            )
            self._round3_commands = self._full_commands
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
        """Owned Sim6 state diagnostics, independent of task termination decisions.

        applied_torque is explicit-actuator clipped effort sent into simulation,
        not solver reaction torque, motor-side current, or a physical measurement.
        Reading a snapshot never advances the sustained-failure counter. Initial
        snapshots inspect post-reset state and never replace the cached pre-reset tick.
        """
        from wheeled_tasks.v40.contract import is_round2
        data = self.robot.data
        wheel_ids = self._named_indices(self.robot.body_names, WHEEL_BODY_NAMES, "wheel link")
        q, dq = self._joint_state()
        limits = self.contract["termination"]
        gravity = data.projected_gravity_b.torch
        # Recompute from the captured state: _finite_state may describe the previous
        # transition or have been cleared by reset. Do not infer diagnostics from reasons.
        finite = (
            torch.isfinite(data.root_link_pose_w.torch).all(-1)
            & torch.isfinite(data.root_com_vel_w.torch).all(-1)
            & torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
            & torch.isfinite(contact).all(-1) & torch.isfinite(self.torques).all(-1)
            & torch.isfinite(clearance) & torch.isfinite(gravity).all(-1)
            & ~self._invalid_actions
        )
        knee_q = q[:, self._knee_ids]
        tolerance = limits["knee_limit_tolerance"]
        failure_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        failure_gravity = torch.zeros_like(terminated)
        sustained_failure = torch.zeros_like(terminated)
        if is_round2(self.contract):
            failure_ticks = self._sustained_failure.count
            failure_gravity = gravity[:, 2] > limits["failure_gravity_z"]
            sustained_failure = failure_ticks > round(limits["failure_seconds"] / self.step_dt)
        diagnostic_flags = {
            "nonfinite": ~finite,
            "non_wheel_contact": (contact[:, self._non_wheel_body_ids]
                                  > limits["contact_force_threshold"]).any(-1),
            "knee_limit": ((knee_q < self._knee_limits[:, 0] - tolerance)
                           | (knee_q > self._knee_limits[:, 1] + tolerance)).any(-1),
            "instantaneous_tilt": -gravity[:, 2] < math.cos(math.radians(limits["max_tilt_deg"])),
            "low_height": self._base_height() < limits["min_base_height"],
            "base_visual_bounds_ground": clearance <= 0.0,
            "failure_gravity": failure_gravity,
            "sustained_failure": sustained_failure,
        }
        snapshot = {
            "schema_version": 1, "sample_kind": kind,
            "state_phase": "post_reset" if kind == "initial" else "pre_reset",
            "diagnostic_flags_version": 1,
            "diagnostic_flags": diagnostic_flags,
            "sustained_failure_ticks": failure_ticks,
            "contact_semantics": "rigid_body_net_force_history_peak_not_ground_pair",
            "policy_tick": int(self.common_step_counter),
            "physics_steps": int(self._sim_step_counter),
            "time_s": int(self.common_step_counter) * self.step_dt,
            "sim_time_s": int(self._sim_step_counter) * self.physics_dt,
            "policy_dt_s": self.step_dt, "physics_dt_s": self.physics_dt,
            "contact_history_samples": int(self.contact_sensor.data.net_forces_w_history.torch.shape[1]),
            "episode_step": self.episode_length_buf,
            "episode_time_s": self.episode_length_buf * self.step_dt,
            "command": self.commands,
            "root_link_pos_w_m": data.root_link_pos_w.torch,
            "root_link_quat_wxyz": data.root_link_quat_w.torch[:, [3, 0, 1, 2]],
            # These aliases are COM velocity expressed in the root link frame in 2.3.0.
            "root_com_lin_vel_b_m_s": data.root_com_lin_vel_b.torch,
            "root_com_ang_vel_b_rad_s": data.root_com_ang_vel_b.torch,
            "projected_gravity_b": data.projected_gravity_b.torch,
            "height_m": data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2],
            "joint_pos_rad": q, "joint_vel_rad_s": dq,
            "sim_joint_effort_nm": data.applied_torque.torch[:, self._joint_ids],
            # Explicit link/actor frame origins at wheel axes; NOT body_com_pos_w.
            "wheel_axis_midpoint_w_m": data.body_link_pos_w.torch[:, wheel_ids].mean(dim=1),
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
                "diagnostic_tilt_flag": "instantaneous_tilt_uses_contract_max_tilt_deg",
                "diagnostic_sustained_failure_flag": "actual_counter_exceeds_contract_failure_duration",
            }
        if getattr(self.cfg, "wheel_slip_diagnostics", False):
            from wheeled_tasks.v40.round3 import wheel_slip_proxy
            snapshot["wheel_diagnostics"] = wheel_slip_proxy(
                data.body_link_pos_w.torch[:, wheel_ids], data.body_link_quat_w.torch[:, wheel_ids],
                data.body_com_pos_w.torch[:, wheel_ids], data.body_com_vel_w.torch[:, wheel_ids],
                contact[:, self._wheel_body_ids], self.scene.env_origins[:, 2],
            )
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
        self.scene.articulations["robot"] = self.robot
        ground_cfg = GroundPlaneCfg()
        if 'round3' in self.contract:
            ground_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(**self.contract['round3']['physics_material'])
        if self.cfg.ground_usd_path is not None:
            from wheeled_tasks.v40.round4 import verify_ground_usd
            self.ground_asset_report = verify_ground_usd(self.cfg.ground_usd_path)
            ground_cfg.usd_path = self.ground_asset_report['path']
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
        self.scene.clone_environments(copy_from_source=True)
        # Use the stage owned by DirectRLEnv's use_stage context (also for in-memory stages).
        stage = self.sim.stage
        if 'round3' in self.contract:
            self._bind_round3_wheel_material(stage)
        self.joint_limit_usd_report = normalize_v40_usd_joint_limits(
            stage, self.scene.env_prim_paths, self.contract["joints"],
        )
        self.collision_filter_report = apply_approved_collision_filters(
            stage, self.scene.env_prim_paths, self.asset_manifest,
            research_approval=self.research_approval, raw_manifest=self.raw_manifest,
        )
        self.contact_sensor = V40ContactSensor(
            self.cfg.contact_sensor_cfg, stage=stage,
            env_prim_paths=self.scene.env_prim_paths, body_names=BODY_NAMES,
        )
        self.scene.sensors["contact"] = self.contact_sensor
        # Independent clones need explicit cross-env filtering on GPU as well as CPU.
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _bind_round3_wheel_material(self, stage):
        """Explicit physics-purpose binding on the two wheel colliders per clone."""
        if getattr(self, '_b1_materials', None) is not None:
            return self._bind_b1_wheel_material(stage)
        from pxr import UsdPhysics
        material_path = '/World/v40Round3WheelMaterial'
        material = sim_utils.RigidBodyMaterialCfg(**self.contract['round3']['physics_material'])
        material.func(material_path, material)
        paths = []
        for prim in stage.Traverse():
            if (prim.GetName() in ('L_link3_cylinder', 'R_link3_cylinder')
                    and str(prim.GetPath()).startswith('/World/envs/')
                    and prim.HasAPI(UsdPhysics.CollisionAPI)):
                sim_utils.bind_physics_material(prim.GetPath(), material_path, stage=stage)
                paths.append(str(prim.GetPath()))
        if len(paths) != 2 * self.cfg.scene.num_envs:
            raise RuntimeError('expected exactly two named wheel collision shapes per environment')
        self._round3_wheel_collision_paths = paths

    def _bind_b1_wheel_material(self, stage):
        """Author immutable bucket materials, avoiding shared-material last-write wins."""
        from pxr import UsdPhysics
        buckets = self._b1_materials
        nominal_path = '/World/v40Round3WheelMaterial'
        base = self.contract['round3']['physics_material']
        material = sim_utils.RigidBodyMaterialCfg(**base)
        material.func(nominal_path, material)
        bucket_paths = []
        for index, values in enumerate(buckets.mapping['wheel_buckets']):
            path = f'/World/v40B1Materials/bucket_{index:02d}'
            material = sim_utils.RigidBodyMaterialCfg(**{
                **base, **dict(zip(('static_friction', 'dynamic_friction', 'restitution'), values)),
            })
            material.func(path, material)
            bucket_paths.append(path)
        env_lookup = {path: index for index, path in enumerate(self.scene.env_prim_paths)}
        bindings = {}
        seen = set()
        for prim in stage.Traverse():
            if prim.GetName() not in ('L_link3_cylinder', 'R_link3_cylinder') or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            env_path = '/'.join(str(prim.GetPath()).split('/')[:4])
            if env_path not in env_lookup:
                continue
            index = env_lookup[env_path]
            if (index, prim.GetName()) in seen:
                raise RuntimeError('B1 duplicate wheel collider in environment')
            seen.add((index, prim.GetName()))
            target = (nominal_path if buckets.mapping['nominal_mask'][index]
                      else bucket_paths[buckets.mapping['env_bucket_ids'][index]])
            sim_utils.bind_physics_material(prim.GetPath(), target, stage=stage)
            bindings[str(prim.GetPath())] = {'env_id': index, 'material': target}
        if len(bindings) != 2 * self.cfg.scene.num_envs:
            raise RuntimeError('B1 requires both named wheel colliders in every environment')
        self._b1_material_bindings = bindings
        self._round3_wheel_collision_paths = list(bindings)

    def _check_b1_materials(self):
        """Read parsed material identities and actual wheel-shape coefficients."""
        from pxr import PhysxSchema, UsdPhysics, UsdShade
        from omni.physx.scripts.ifaces import get_physxunittests_interface
        import warp as wp

        stage, buckets = self.sim.stage, self._b1_materials
        ground_paths = [str(p.GetPath()) for p in stage.Traverse()
                        if str(p.GetPath()).startswith('/World/ground/') and p.GetTypeName() == 'Plane']
        if len(ground_paths) != 1:
            raise RuntimeError('B1 expected one ground Plane collider')
        query = get_physxunittests_interface()
        bindings = []
        for path in [*self._round3_wheel_collision_paths, *ground_paths]:
            binding = self._b1_material_bindings.get(path)
            expected = buckets.wheel_coefficients(binding['env_id']) if binding else [.5, .5, 0.]
            expected_path = binding['material'] if binding else '/World/ground/physicsMaterial'
            material, rel = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')
            if not material or not rel or str(material.GetPath()) != expected_path:
                raise RuntimeError(f'B1 material binding mismatch: {path}')
            api, physx = UsdPhysics.MaterialAPI(material.GetPrim()), PhysxSchema.PhysxMaterialAPI(material.GetPrim())
            values = [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(), api.GetRestitutionAttr().Get()]
            modes = [physx.GetFrictionCombineModeAttr().Get(), physx.GetRestitutionCombineModeAttr().Get()]
            parsed = list(query.get_materials_paths(path))
            if (any(abs(a-b) > 1e-6 for a, b in zip(values, expected)) or modes != ['average', 'average']
                    or parsed != [expected_path]):
                raise RuntimeError(f'B1 parsed shape/coefficients/combine mismatch: {path}')
            bindings.append({'collider': path, 'material': expected_path, 'physx_shape_material_paths': parsed,
                             'usd_coefficients': values, 'usd_combine_modes': modes})
        wheel_paths = [path for path in self.contact_sensor._v40_body_paths if path.rsplit('/', 1)[-1] in WHEEL_BODY_NAMES]
        env_lookup = {path: index for index, path in enumerate(self.scene.env_prim_paths)}
        view = self.contact_sensor._physics_sim_view.create_rigid_body_view(wheel_paths)
        if view.count != 2 * self.num_envs or view.max_shapes != 1 or list(view.prim_paths) != wheel_paths:
            raise RuntimeError('B1 wheel shape/order mismatch')
        actual = wp.to_torch(view.get_material_properties()).clone()
        env_ids = [env_lookup['/'.join(path.split('/')[:4])] for path in wheel_paths]
        expected = actual.new_tensor([buckets.wheel_coefficients(i) for i in env_ids]).unsqueeze(1)
        if (actual.shape != (2 * self.num_envs, 1, 3) or not torch.isfinite(actual).all()
                or not torch.allclose(actual, expected, rtol=0, atol=1e-6)):
            raise RuntimeError('B1 actual PhysX wheel materials differ from per-env mapping')
        if (actual[..., 1] > actual[..., 0]).any():
            raise RuntimeError('B1 PhysX dynamic friction exceeds static')
        nonwheel_indices = [i for i, path in enumerate(self.contact_sensor._v40_body_paths)
                            if path.rsplit('/', 1)[-1] not in WHEEL_BODY_NAMES]
        # Non-wheel bodies have multiple shapes; report only the first actual shape,
        # not padding entries in the max_shapes tensor. Only wheel USD bindings changed.
        nonwheel = wp.to_torch(self.contact_sensor._body_physx_view.get_material_properties())[nonwheel_indices, 0].clone()
        if not torch.allclose(nonwheel, nonwheel.new_tensor([.5, .5, 0.]).expand_as(nonwheel), atol=1e-6, rtol=0):
            raise RuntimeError('B1 non-wheel first-shape material differs from nominal')
        return {
            'schema_version': 1, 'stage': 'B1', 'passed': True, 'sampling': 'startup_only_no_episode_resampling',
            'expected_mapping': deepcopy(buckets.mapping), 'mapping_sha256': buckets.mapping_sha256,
            'bindings': bindings, 'wheel_body_paths': wheel_paths, 'wheel_env_ids': env_ids,
            'wheel_physx_coefficients': actual.cpu().tolist(),
            'nonwheel_first_shape_physx_coefficients': nonwheel.cpu().tolist(),
            'coefficient_order': ['static_friction', 'dynamic_friction', 'restitution'],
            'ground_scope': 'parsed PhysX material identity + USD coefficients; no static-ground tensor coefficients',
            'combine_scope': 'USD average modes plus parsed shape binding; coefficient tensor has no combine modes',
        }

    def check_physics_joint_limits(self) -> dict:
        """Fresh readback shared by startup and bounded simulator diagnostics."""
        return validate_v40_physx_joint_limits(self.robot, self.contract["joints"], num_envs=self.num_envs)

    def check_round3_materials(self) -> dict:
        """Opt-in USD + parsed PhysX shape binding and wheel material tensor readback.

        Static ground material identity comes from the PhysX diagnostic interface;
        tensor coefficients are read for dynamic wheel shapes, not invented for ground.
        """
        if getattr(self, '_b1_materials', None) is not None:
            return self._check_b1_materials()
        from pxr import PhysxSchema, UsdPhysics, UsdShade
        from omni.physx.scripts.ifaces import get_physxunittests_interface
        import warp as wp

        if 'round3' not in self.contract:
            raise ValueError('explicit material readback requires the round3 contract')
        stage = self.sim.stage
        ground_paths = [str(p.GetPath()) for p in stage.Traverse()
                        if str(p.GetPath()).startswith('/World/ground/') and p.GetTypeName() == 'Plane']
        if len(ground_paths) != 1:
            raise RuntimeError('expected one ground Plane collider')
        # This pinned SDK exposes parsed collider materials on PhysXUnitTests,
        # not PhysXSceneQuery. It reads engine state without editing the scene.
        query = get_physxunittests_interface()
        bindings = []
        for path in [*self._round3_wheel_collision_paths, *ground_paths]:
            prim = stage.GetPrimAtPath(path)
            material, rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            if not material or not rel:
                raise RuntimeError(f'missing physics material binding: {path}')
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            physx_api = PhysxSchema.PhysxMaterialAPI(material.GetPrim())
            values = [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(),
                      api.GetRestitutionAttr().Get()]
            modes = [physx_api.GetFrictionCombineModeAttr().Get(), physx_api.GetRestitutionCombineModeAttr().Get()]
            parsed_paths = list(query.get_materials_paths(path))
            if values != [.5, .5, 0.] or modes != ['average', 'average'] or str(material.GetPath()) not in parsed_paths:
                raise RuntimeError(f'USD/PhysX material identity mismatch: {path}, {parsed_paths}, {values}, {modes}')
            bindings.append({'collider': path, 'usd_material': str(material.GetPath()),
                             'physx_shape_material_paths': parsed_paths, 'usd_coefficients': values,
                             'usd_combine_modes': modes})
        wheel_paths = [p for p in self.contact_sensor._v40_body_paths
                       if p.rsplit('/', 1)[-1] in WHEEL_BODY_NAMES]
        view = self.contact_sensor._physics_sim_view.create_rigid_body_view(wheel_paths)
        if view.count != 2 * self.num_envs or view.max_shapes != 1 or list(view.prim_paths) != wheel_paths:
            raise RuntimeError('wheel material view shape/order mismatch')
        values = wp.to_torch(view.get_material_properties()).clone()
        if values.shape != (2 * self.num_envs, 1, 3) or not torch.isfinite(values).all():
            raise RuntimeError('invalid PhysX wheel material tensor')
        if not torch.allclose(values, values.new_tensor([.5, .5, 0.]).expand_as(values), atol=1e-6, rtol=0):
            raise RuntimeError('PhysX wheel material coefficients differ from explicit nominal')
        return {'passed': True, 'bindings': bindings, 'wheel_body_paths': wheel_paths,
                'wheel_physx_coefficients': values.cpu().tolist(),
                'coefficient_order': ['static_friction', 'dynamic_friction', 'restitution'],
                'ground_scope': 'parsed PhysX shape material identity + USD coefficients; no static-ground tensor coefficients',
                'combine_scope': 'explicit USD modes + parsed shape material identity; tensor API exposes coefficients only'}

    def _joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.data.joint_pos.torch[:, self._joint_ids], self.robot.data.joint_vel.torch[:, self._joint_ids]

    def set_training_iteration(self, update: int) -> None:
        """Set completed PPO update count (first completed update=1), never env ticks."""
        if getattr(self, '_push_curriculum', None) is None:
            raise ValueError('training-iteration push curriculum requires a Round4 profile')
        self._push_curriculum.set_training_iteration(update)
        if getattr(self, '_full_commands', None) is not None:
            self._full_commands.set_training_iteration(update)

    @property
    def training_curriculum_state(self):
        """JSON progress/timing inspection; not a claim of bitwise trajectory resume."""
        curriculum = getattr(self, '_push_curriculum', None)
        if curriculum is None:
            return None
        state = curriculum.state()
        if getattr(self, '_full_commands', None) is not None:
            state['command_curriculum'] = self._full_commands.state()
        return state

    def _read_root_impulse_state(self, env_ids):
        """Fresh PhysX backend reads, selected on device before any host reporting."""
        velocity = self.robot.root_view.get_root_velocities()
        pose = self.robot.root_view.get_root_transforms()
        if not isinstance(velocity, torch.Tensor):
            import warp as wp
            velocity = wp.to_torch(velocity)
            pose = wp.to_torch(pose)
        if velocity.shape != (self.num_envs, 6) or pose.shape != (self.num_envs, 7):
            raise RuntimeError('unexpected PhysX root COM velocity / link pose layout')
        return velocity[env_ids].clone(), pose[env_ids].clone()

    def apply_velocity_impulse(self, delta_xy: torch.Tensor, env_ids) -> dict:
        """Add world-XY root COM velocity [m/s] to selected envs; preserve z/omega/pose.

        Shared by random training events and explicit held-out evaluation. Call at
        the observation boundary; this does not resample commands, clear history,
        alter termination, or overwrite the saved pre-reset transition snapshot.
        """
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim == 1 and ids.numel() == 0:
            ids = ids.to(torch.long)
        if (ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64)
                or (ids < 0).any() or (ids >= self.num_envs).any() or ids.unique().numel() != ids.numel()):
            raise ValueError('impulse env_ids must be unique in-range integer indices')
        if (not isinstance(delta_xy, torch.Tensor) or delta_xy.shape != (len(ids), 2)
                or delta_xy.dtype != torch.float32 or delta_xy.device != torch.device(self.device)
                or not torch.isfinite(delta_xy).all()):
            raise ValueError('impulse delta must be finite float32 [len(env_ids),2] on the env device')
        if ids.numel() == 0:
            return {'events_count': 0}
        before, pose_before = self._read_root_impulse_state(ids)
        if not torch.isfinite(before).all() or not torch.isfinite(pose_before).all():
            raise ValueError('cannot add an impulse to nonfinite root state')
        requested = before.clone()
        requested[:, :2] += delta_xy
        if not torch.isfinite(requested).all():
            raise ValueError('impulse would overflow root velocity')
        # Lab 3 PhysX articulation.py:680–733: partial COM-world [K,6], int32 ids.
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=requested, env_ids=ids.to(torch.int32), full_data=False)
        # This pinned Lab3 setter invalidates root state but not its cached body-frame
        # linear projection. Reward already read that projection in the same tick;
        # invalidate it so the NEXT critic observation sees the applied impulse.
        self.robot.data._root_com_lin_vel_b.timestamp = -1.0
        after, pose_after = self._read_root_impulse_state(ids)
        finite = bool(torch.isfinite(after).all() and torch.isfinite(pose_after).all())
        velocity_ok = finite and torch.allclose(after, requested, atol=1e-5, rtol=0)
        angular_ok = finite and torch.equal(after[:, 3:], before[:, 3:])
        vertical_ok = finite and torch.equal(after[:, 2], before[:, 2])
        pose_ok = finite and torch.allclose(pose_after, pose_before, atol=1e-7, rtol=0)
        passed = bool(velocity_ok and angular_ok and vertical_ok and pose_ok)
        if getattr(self, 'first_push_report', None) is None:
            self.first_push_report = {
                'readback_passed': passed, 'policy_tick': int(self.common_step_counter),
                'env_ids': ids.cpu().tolist(), 'requested_delta_xy_m_s': delta_xy.cpu().tolist(),
                'before_com_velocity_w_m_s_rad_s': before.cpu().tolist(),
                'after_com_velocity_w_m_s_rad_s': after.cpu().tolist() if finite else None,
                'angular_unchanged': bool(angular_ok), 'vertical_unchanged': bool(vertical_ok),
                'pose_unchanged': bool(pose_ok), 'readback_finite': finite,
                'source': 'PhysX root_view fresh read around write_root_com_velocity_to_sim_index',
            }
        if not passed:
            raise RuntimeError('root COM impulse readback did not preserve requested velocity/angular/pose invariants')
        return {'events_count': len(ids), 'env_ids': ids, 'requested_delta_xy_m_s': delta_xy.clone(),
                'realized_delta_xy_m_s': after[:, :2] - before[:, :2],
                'readback_error_m_s': after[:, :2] - requested[:, :2]}

    def _apply_scheduled_pushes(self):
        """Interval-event boundary: old reward/reset finished, next observation not built."""
        curriculum = self._push_curriculum
        event = curriculum.sample(int(self.common_step_counter), self.episode_length_buf,
                                  automatic=self._evaluation_command_override is None)
        if event is None:
            return  # Repeated same-tick observation must not erase a recorded event.
        ids, delta = event
        log = self.extras.setdefault('log', {})
        log.update({'Push/events_count': 0, 'Push/max_delta_v': curriculum.max_delta_v,
                    'Push/completed_ppo_updates': curriculum.training_iteration,
                    'Push/requested_delta_v_max': 0., 'Push/realized_delta_v_max': 0.,
                    'Push/readback_error_max': 0.})
        if ids.numel():
            report = self.apply_velocity_impulse(delta, ids)
            log['Push/events_count'] = report['events_count']
            log['Push/requested_delta_v_max'] = torch.linalg.vector_norm(delta, dim=-1).max()
            log['Push/realized_delta_v_max'] = torch.linalg.vector_norm(report['realized_delta_xy_m_s'], dim=-1).max()
            log['Push/readback_error_max'] = report['readback_error_m_s'].abs().max()

    def _base_height(self) -> torch.Tensor:
        return self.robot.data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]

    def _base_visual_clearance(self) -> torch.Tensor:
        """Lower bound from eight rotated base-link AABB corners, NOT COM/mesh distance."""
        # Lab 3 uses XYZW; the clearance formula and exported snapshot stay unchanged.
        x, y, z, w = self.robot.data.root_link_quat_w.torch.unbind(-1)
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
        self.robot.set_joint_effort_target_index(target=self.torques, joint_ids=self._joint_ids)

    def _contact_magnitudes(self) -> torch.Tensor:
        # N x history x bodies x xyz. Use both substeps, not only the final instant.
        history = self.contact_sensor.data.net_forces_w_history.torch
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        from wheeled_tasks.v40.contract import is_round2
        joint_pos, joint_vel = self._joint_state()
        body = self.robot.data
        contact = self._contact_magnitudes()
        clearance = self._base_visual_clearance()
        self._finite_state = (
            torch.isfinite(body.root_link_pose_w.torch).all(-1)
            & torch.isfinite(body.root_com_vel_w.torch).all(-1)
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
        tilt = -body.projected_gravity_b.torch[:, 2] < math.cos(math.radians(limits["max_tilt_deg"]))
        low = self._base_height() < limits["min_base_height"]
        base_bounds_ground = clearance <= 0.0
        terminated = ~self._finite_state | non_wheel_contact | knee_out | tilt | low | base_bounds_ground
        diagnostics = {"non_wheel_contact": non_wheel_contact, "knee_limit": knee_out,
                       "tilt": tilt, "low_height": low, "base_visual_bounds_ground": base_bounds_ground}
        reasons = {"nonfinite": ~self._finite_state, **diagnostics}
        if is_round2(self.contract):
            self._finite_state &= torch.isfinite(body.projected_gravity_b.torch).all(-1)
            sustained = self._sustained_failure.update(
                body.projected_gravity_b.torch[:, 2], int(self.common_step_counter), self.contract,
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
            log["Diagnostic/failure_gravity"] = (body.projected_gravity_b.torch[:, 2] > limits["failure_gravity_z"]).float().mean()
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
        joint_pos, joint_vel = self._joint_state()
        data = self.robot.data
        valid = self._finite_state
        # Nonfinite terminal rows get no nonterminal rates; do not poison PPO with NaNs.
        terms = compute_reward_terms(
            data.root_com_lin_vel_b.torch[valid], data.root_com_ang_vel_b.torch[valid], data.projected_gravity_b.torch[valid],
            self._base_height()[valid], self.commands[valid], self.actions[valid], self.previous_actions[valid],
            self.torques[valid], joint_pos[valid], self.contract,
            **({'joint_vel6': joint_vel[valid]} if 'round4' in self.contract else {}),
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
            "vx_m_s": data.root_com_lin_vel_b.torch[:, 0], "wz_rad_s": data.root_com_ang_vel_b.torch[:, 2],
            "height_m": self._base_height(),
            "vx_abs_error": (data.root_com_lin_vel_b.torch[:, 0] - self.commands[:, 0]).abs(),
            "wz_abs_error": (data.root_com_ang_vel_b.torch[:, 2] - self.commands[:, 1]).abs(),
            "height_abs_error": (self._base_height() - self.commands[:, 2]).abs(),
            "planar_speed_m_s": torch.linalg.vector_norm(data.root_com_lin_vel_b.torch[:, :2], dim=-1),
        }.items():
            log[f"Tracking/{name}"] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).mean()
        if is_round2(self.contract):
            log["Command/standing_fraction"] = (self.commands[:, :2] == 0.0).all(-1).float().mean()
        if getattr(self, '_round3_commands', None) is not None:
            sampler = self._round3_commands
            log['Command/spin_bucket_fraction'] = (sampler.velocity_bucket == 1).float().mean()
            log['Command/height_low_endpoint_fraction'] = (self.commands[:, 2] == sampler.bounds['height'][0]).float().mean()
            log['Command/height_high_endpoint_fraction'] = (self.commands[:, 2] == sampler.bounds['height'][1]).float().mean()
        if getattr(self, '_full_commands', None) is not None and self._evaluation_command_override is None:
            sampler = self._full_commands
            log['Command/straight_bucket_fraction'] = (sampler.velocity_bucket == 2).float().mean()
            log['Command/combined_bucket_fraction'] = (sampler.velocity_bucket == 3).float().mean()
            log['Command/cap_vx_m_s'], log['Command/cap_wz_rad_s'] = sampler.caps
            log['Command/requested_vx_abs_max'] = sampler.requested_velocity[:, 0].abs().max()
            log['Command/requested_wz_abs_max'] = sampler.requested_velocity[:, 1].abs().max()
            log['Command/executed_vx_abs_max'] = self.commands[:, 0].abs().max()
            log['Command/executed_wz_abs_max'] = self.commands[:, 1].abs().max()
            log['Command/traction_clipped_fraction'] = sampler.traction_clipped.float().mean()
            log['Command/wheel_speed_clipped_fraction'] = sampler.wheel_speed_clipped.float().mean()
        # Reward and tracking above use the command that produced this action.
        # Sampling is deferred until _get_observations, after terminal resets.
        tick = int(self.common_step_counter)
        if tick != self._last_reward_tick:
            self._command_ticks_left -= 1
            self._commands_due |= self._command_ticks_left <= 0
            if getattr(self, '_round3_commands', None) is not None:
                self._round3_commands.advance()
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
        if getattr(self, '_round3_commands', None) is not None:
            self._round3_commands.sample_velocity(self.commands, env_ids)
            self._round3_commands.sample_height(self.commands, env_ids)
            self._command_ticks_left[env_ids] = self._command_period_ticks
            self._commands_due[env_ids] = False
            return
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
        if (getattr(self, '_round3_commands', None) is not None
                and self._evaluation_command_override is None):
            self._round3_commands.sample_height(self.commands)
        if getattr(self, '_push_curriculum', None) is not None:
            self._apply_scheduled_pushes()
        joint_pos, joint_vel = self._joint_state()
        data = self.robot.data
        obs25 = build_observation(
            data.root_com_ang_vel_b.torch, data.projected_gravity_b.torch, self.commands,
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
        critic = build_critic(obs25, data.root_com_lin_vel_b.torch, self._base_height())
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
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._nominal
        joint_vel = torch.zeros_like(joint_pos)
        root = torch.cat((self.robot.data.default_root_pose.torch[env_ids],
                          self.robot.data.default_root_vel.torch[env_ids]), dim=-1)
        root[:, :3] += self.scene.env_origins[env_ids]
        root[:, 7:] = 0.0  # Preserve the default root quaternion, not a hardcoded replacement.
        if is_round2(self.contract) and self._evaluation_command_override is None:
            low, high = self.contract["reset"]["root_velocity_range"]
            root[:, 7:] = low + (high - low) * torch.rand_like(root[:, 7:])
        self.robot.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
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
        if getattr(self, '_round3_commands', None) is not None:
            self._round3_commands.reset(env_ids)
        if getattr(self, '_push_curriculum', None) is not None:
            self._push_curriculum.reset(env_ids, automatic=self._evaluation_command_override is None)
        self._sample_commands(env_ids)
        self._evaluation_command_pending = False
        # Do NOT clear the pre-reset snapshot here: DirectRLEnv auto-resets before returning.
        self.robot.set_joint_effort_target_index(target=self.torques[env_ids], joint_ids=self._joint_ids, env_ids=env_ids)
