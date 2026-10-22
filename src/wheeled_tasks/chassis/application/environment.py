"""Isaac Lab 3 / PhysX multi-scene environment for the 15-body closed chain."""
from __future__ import annotations

from wheeled_tasks.chassis.source_layout import module_source_path

import math
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import warp as wp
from tensordict import TensorDict
from pxr import PhysxSchema, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab_physx.physics import PhysxManager, PhysxCfg

from wheeled_tasks.chassis.control.actuation import HistoryStack
from wheeled_tasks.chassis.application.configuration.preflight import GAS_SPRING_MODEL_KINDS, SubstepControlAudit, design_preflight, validate_runtime_state
from wheeled_tasks.chassis.control.actuator_response import ActuatorResponse
from wheeled_tasks.chassis.simulation.root_kinematics import StaticRootLinkKinematics, root_link_linear_velocity, update_robot_data
from wheeled_tasks.chassis.application.runtime.training_schedule import stage_actor_updates
from wheeled_tasks.chassis.application.configuration.small_rough_course import FINAL_MOTION_VERSION, MOTION_VERSIONS
from wheeled_tasks.chassis.application.compatibility.terrain_runout import append_route_runout, apply_route_boundary, route_runout_selection, validate_runout
from wheeled_tasks.chassis.tasks.episodes import FallConfirmation, Phase, PhaseTracker, Surface, corridor_mesh, choose_terrains, choose_scene_groups, episode_outcome, phase_reward_masks, terrain_surfaces


class ChassisEnv:
    """Stock RSL VecEnv protocol with explicit, terrain-filtered contact identity."""

    def __init__(self, config, manifest, control, root: Path, *, stage_name, num_envs,
                 device="cuda:0", level=0.0, seed=617, coverage=False, run_role=None, block_index=0,
                 integrated_design_preflight=None, evaluation_setup=None, replay_setup=None, physics_probe=False):
        from wheeled_tasks.chassis.application.configuration.training_profile import validate_profile_selection
        validate_profile_selection(config)
        from wheeled_tasks.chassis.application.configuration.reward_selection import validate_reward_selection
        self.reward_selection = validate_reward_selection(config, stage_name=stage_name)
        self.scut_reward = None
        self._configure_reward_episode_measurement(config)
        self.evaluation_surface = "training"
        self.replay_setup = replay_setup
        if replay_setup is not None:
            from wheeled_tasks.chassis.application.evaluation.ordinary_replay import OrdinaryReplaySetup
            if type(replay_setup) is not OrdinaryReplaySetup or evaluation_setup is not None:
                raise TypeError("Ordinary replay requires its separate typed setup")
            replay_setup.validate_constructor(config, num_envs, seed, run_role)
            self.evaluation_surface = replay_setup.evaluation_surface
        if evaluation_setup is not None:
            from wheeled_tasks.chassis.application.evaluation.flat_evaluation_setup import FlatEvaluationSetup
            if type(evaluation_setup) is not FlatEvaluationSetup:
                raise TypeError("Source flat evaluation requires its typed setup")
            evaluation_setup.validate_constructor(config, num_envs, seed, run_role)
            self.evaluation_surface = evaluation_setup.evaluation_surface
        if ("formal_dynamic_nonsteady_birth" in config and run_role not in (None, "train")
                and evaluation_setup is None and replay_setup is None):
            raise ValueError("Dynamic birth measurements require their typed evaluation or replay setup")
        if run_role == "source_rough_plane_regression" and self.evaluation_surface != "plane_regression_v1":
            raise ValueError("Rough plane regression requires its validated typed surface overlay")
        self.rehearsal_spec = None
        self.source_flat_rehearsal_mask = None
        self._dynamic_birth_nominal_mask = None
        self._dynamic_birth_source_mask = None
        if run_role == "source_scripted_recovery_evaluation" and evaluation_setup is None:
            raise ValueError("Scripted recovery evaluation requires its validated typed protocol")
        from wheeled_tasks.chassis.application.configuration.flat_episode_selection import require_native_flat_admission
        require_native_flat_admission(config, manifest, physics_probe=physics_probe)
        self.flat_episode = None
        self.flat_events = None
        self.v7_stepup = None
        self.v7_stepup_settings = None
        self.curriculum_controller = None
        self.curriculum_runtime = None
        if "v7_curriculum_runtime" in config:
            from wheeled_tasks.chassis.application.configuration.curriculum_selection import load_curriculum_definition
            from wheeled_tasks.chassis.tasks.curricula.autonomous_curriculum import AutonomousCurriculum
            if run_role not in (None, "train") or evaluation_setup is not None:
                raise ValueError("V7 training exposure cannot own a fixed evaluation scene")
            definition = load_curriculum_definition(config, root)
            self.curriculum_controller = AutonomousCurriculum(definition,
                training_seeds=[seed], external_budget=True)
        if config.get("integrated_runtime_candidate", {}).get("static_recipe_preflight"):
            raise ValueError("Static joint recipe cannot construct a native environment")
        if (config.get("scut_heading_command", {}).get("enabled")
                and not config.get("evaluation_exact_cases") and run_role not in (None, "train")):
            raise ValueError("reference heading command is restricted to explicit training")
        if integrated_design_preflight is not None and (not config.get("integrated_production")
                or config.get("evaluation_exact_cases") or run_role not in (None, "train")):
            raise ValueError("Joint design admission is restricted to explicit training")
        if config.get("integrated_production") and not config.get("evaluation_exact_cases") and integrated_design_preflight is None:
            raise ValueError("Joint production environment requires authenticated design admission")
        if "feedback_transport" in config:
            from wheeled_tasks.chassis.control.transport.pc_communication import validate_pc_communication
            validate_pc_communication(config)
        self.cfg = dict(config)
        self.cfg.update(stage=stage_name, num_envs=num_envs, terrain_level=level, seed=seed)
        self.integrated_runtime_enabled = bool(config.get("integrated_runtime_candidate")) and not config.get("evaluation_exact_cases")
        self.route_runout_settings = validate_runout(config)
        role = run_role or ("full" if config.get("evaluation_exact_cases") else "train")
        self.design_preflight = design_preflight(config, control, root, stage=stage_name,
            role=role, seed=seed, num_envs=num_envs, block_index=block_index,
            evaluation_setup=evaluation_setup, replay_setup=replay_setup, physics_probe=physics_probe)
        if integrated_design_preflight is not None:
            from wheeled_tasks.chassis.application.compatibility.integrated_training import validate_design_hook
            self.design_preflight = validate_design_hook(config, root, self.design_preflight,
                integrated_design_preflight, stage=stage_name, num_envs=num_envs,
                seed=seed, block_index=block_index)
        self.control_audit = SubstepControlAudit(
            replay_setup.physics_dt if replay_setup is not None else config["physics_dt"], config["policy_dt"])
        self.stage_cfg = next(s for s in config["stages"] if s["name"] == stage_name)
        self.num_envs, self.num_actions, self.device = num_envs, 6, device
        self.control, self.manifest = control, manifest
        self.is_v5 = manifest["model_kind"] in GAS_SPRING_MODEL_KINDS
        if not self.is_v5:
            raise ValueError("Only the closed-chain chassis control model is supported")
        self.scut35 = config.get("actor_observation_source") in ("scut35_encoders_imu_commands", "encoders_imu_command_reference36")
        self.critic78 = config.get("critic_observation_version") == "v6_scut78_v1"
        if self.critic78 and (not self.scut35 or config.get("manual_context35") is not True
                              or config.get("critic_dim") != 78 or config.get("actor_frame_dim") != 35):
            raise ValueError("78D critic requires the frozen clean35 command-context interface")
        if self.reward_selection.get("scut_source_selected", self.reward_selection.get("scut_flat_selected", False)):
            if (not self.is_v5 or config.get("actor_observation_source") != "scut35_encoders_imu_commands"
                    or config.get("actor_frame_dim") != 35 or not config.get("monitor_applied_effort")):
                raise ValueError("flat baseline reward requires native V5 active axes and the existing actor35 interface")
            if config.get("auto_reset", True) is not True or config.get("evaluation_exact_cases"):
                raise ValueError("flat baseline reward requires auto-reset; the old fixed-case evaluator has no adapter")
        self.physical_tick = 0
        self.references = None
        self.step_jump = self.step_climb = self.step_sensors = None
        self.recovery = None
        self.jump_actuation_measurement = None
        measurement = config.get("jump_actuation_measurement")
        if measurement is not None:
            if (not isinstance(measurement, dict) or set(measurement) != {"version", "enabled"}
                    or measurement["version"] != "jump_actuation_measurement_v1"
                    or type(measurement["enabled"]) is not bool):
                raise ValueError("Jump actuation measurement requires an explicit version and boolean")
            if measurement["enabled"] and (not self.is_v5 or not config.get("monitor_applied_effort")
                    or config["physics_dt"] != .005 or config["policy_dt"] != .02):
                raise ValueError("Jump measurement requires native gas-spring efforts at 200/50 Hz")
            if measurement["enabled"] and self.reward_selection.get("scut_source_selected", self.reward_selection.get("scut_flat_selected", False)):
                raise ValueError("Source flat commands do not expose jump request identity")
        self._configure_jump_physics_trace(config)
        self.nonstandard_start = None
        self.ordinary_posture = None
        self.body_count = len(manifest["rigid_body_names"])
        self.joint_count = len(manifest["tree_joint_names"])
        self.v5 = None
        self.spring_pressure_element = None
        self._spring_pressure_frame = None
        self.height_margin = None
        if self.is_v5:
            from wheeled_tasks.chassis.control.pd import V5Control
            directory = root / config["asset_directory"]
            self.model_spec = json.loads((directory / "model_spec.json").read_text())
            fit = json.loads((directory / "fit_10mpa.json").read_text())
            self.v5 = V5Control(manifest, self.model_spec, fit, control, config["v5_control"], device)
            if config.get("height_workspace"):
                reference = config["height_workspace"]
                if (reference["model_spec_sha256"] != manifest["files_sha256"]["model_spec.json"]
                        or not math.isclose(reference["knee_working_margin_rad"], config["knee_working_margin_rad"])):
                    raise ValueError("Commanded-height margin reference differs from the mechanical model")
                from wheeled_tasks.chassis.tasks.rewards.rewards import CommandedHeightMargin
                self.height_margin = CommandedHeightMargin(reference, device)
        self.dt = replay_setup.physics_dt if replay_setup is not None else config["physics_dt"]
        self.policy_dt = config["policy_dt"]
        self.passive_landing = None
        if config.get("landing_tracking"):
            from wheeled_tasks.chassis.tasks.rewards.rewards import PassiveLandingWindow, validate_landing_tracking
            if validate_landing_tracking(config["landing_tracking"]) == "finite_task_cost_v1":
                self.passive_landing = PassiveLandingWindow(num_envs, device, self.policy_dt,
                                                            config["landing_tracking"])
        self.jump_landing = None
        self.jump_cycle_reset_on_accepted = config.get("jump_cycle_reset_on_accepted", False)
        if type(self.jump_cycle_reset_on_accepted) is not bool:
            raise ValueError("Jump accepted-cycle reset must be an explicit boolean")
        if config.get("jump_landing", {}).get("enabled"):
            from wheeled_tasks.chassis.tasks.rewards.jump_landing import JumpLandingWindow
            self.jump_landing = JumpLandingWindow(num_envs, device, self.policy_dt, config["jump_landing"],
                step_jump_stop_target=config.get("step_jump_landing_stop_target", False))
        self.landing_reward_mask = torch.zeros(num_envs, device=device)
        self.actuator_response = ActuatorResponse(num_envs, device, self.dt,
            config.get("actuator_response"))
        if self.actuator_response.enabled and (not self.is_v5 or not config.get("monitor_applied_effort")):
            raise ValueError("Identified response requires explicit six-axis V5 effort monitoring")
        self.decimation = self.control_audit.decimation
        self.max_episode_length = round(config["episode_seconds"] / self.policy_dt)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.level = level
        physics_settings = {}
        if "physx_solver_type" in config:
            physics_settings["solver_type"] = config["physx_solver_type"]
        if config.get("physx_external_forces_every_iteration") is True:
            physics_settings["enable_external_forces_every_iteration"] = True
        if "physx_friction_offset_threshold_m" in config:
            physics_settings["friction_offset_threshold"] = config["physx_friction_offset_threshold_m"]
        if "gpu_found_lost_aggregate_pairs_capacity" in config:
            physics_settings["gpu_found_lost_aggregate_pairs_capacity"] = config[
                "gpu_found_lost_aggregate_pairs_capacity"]
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=device, render_interval=self.decimation,
            physics=PhysxCfg(**physics_settings) if physics_settings else None,
            enable_scene_query_support=config.get("enable_scene_queries", False)))
        import carb.settings
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        if config.get("dense_tracking", {}).get("enabled"):
            from wheeled_tasks.chassis.tasks.rewards.rewards import validate_dense_tracking
            validate_dense_tracking(config["dense_tracking"])
        if "locomotion_posture" in config.get("dense_tracking", {}):
            if (not config["dense_tracking"].get("enabled")
                    or config.get("reward_profile") != "scut_v14_flat_v5"
                    or not config.get("command_reference")
                    or not config.get("reference_reward", {}).get("visible_reference_only")):
                raise ValueError("Locomotion posture floor requires active dense tracking and visible reference rewards")
        self.recovery_takeover_settings = None
        if config.get("recovery_takeover", {}).get("enabled"):
            from wheeled_tasks.chassis.tasks.rewards.rewards import validate_recovery_takeover
            self.recovery_takeover_settings = dict(config["recovery_takeover"], dense_tracking=config["dense_tracking"])
            validate_recovery_takeover(self.recovery_takeover_settings)
            if not config.get("recovery_training", {}).get("enabled"):
                raise ValueError("Recovery takeover shaping requires the physical recovery prelude")
        self.recovery_hold_cost_settings = None
        self.recovery_hold_cost_diagnostics = {}
        if "recovery_hold_cost" in config:
            from wheeled_tasks.chassis.tasks.rewards.rewards import validate_recovery_hold_cost
            validate_recovery_hold_cost(config["recovery_hold_cost"])
            if config["recovery_hold_cost"]["enabled"] and (not config.get("recovery_training", {}).get("enabled")
                    or not config.get("dense_tracking", {}).get("enabled")
                    or config.get("reward_profile") != "scut_v14_flat_v5"
                    or self.recovery_takeover_settings is not None):
                raise ValueError("Recovery hold cost requires native recovery and dense rewards without legacy takeover")
            self.recovery_hold_cost_settings = dict(config["recovery_hold_cost"])
        self.training_transitions = 0
        self.stage_actor_update = 0
        if config.get("integrated_runtime_candidate") and not config.get("evaluation_exact_cases"):
            self.task_actor_update = config["integrated_runtime_candidate"]["clocks_at_source"]["domain_actor_updates"]
        self.global_actor_update = config.get("global_actor_update_offset", 0)
        if config.get("evaluation_exact_cases"):
            cases = config["evaluation"]["cases"]
            if num_envs % len(cases):
                raise ValueError("Fixed evaluation requires equal integer replicas per case")
            scenes = [(case["name"], case.get("terrain", "flat")) for case in cases for _ in range(num_envs // len(cases))]
            groups, kinds = zip(*scenes)
        elif config.get("scene_groups"):
            scenes = choose_scene_groups(config["scene_groups"], num_envs)
            groups, kinds = zip(*scenes)
        else:
            kinds = choose_terrains(self.stage_cfg, num_envs, config["base_scene_fraction"], coverage)
            groups = ["foundation"] * num_envs
        if "formal_small_rough_mix" in config and evaluation_setup is None and replay_setup is None:
            from wheeled_tasks.chassis.application.configuration.small_rough_course import allocate
            self.rehearsal_spec = allocate(config, root, num_envs, groups)
        elif "formal_flat_rehearsal" in config and evaluation_setup is None and replay_setup is None:
            from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import load_fixed_flat_rehearsal
            ordinary = None
            if config.get("reward_profile") == "scut0719_flat_v1":
                ordinary_groups = {scene["name"] for scene in config["scene_groups"]
                                   if "recovery_pose" not in scene}
                ordinary = [i for i, group in enumerate(groups) if group in ordinary_groups]
            scripted_plane_ids = None
            if (config.get("formal_terrain", {}).get("source") == "v7_final_b8ff79f3"
                    and config.get("recovery_training", {}).get("source_flat_handoff")):
                scripted_groups = {scene["name"] for scene in config["scene_groups"]
                                   if "recovery_pose" in scene}
                scripted_plane_ids = [i for i, group in enumerate(groups) if group in scripted_groups]
            self.rehearsal_spec = load_fixed_flat_rehearsal(config, root, num_envs,
                **({"ordinary_lane_ids": ordinary} if ordinary is not None else {}),
                **({"scripted_plane_lane_ids": scripted_plane_ids}
                    if scripted_plane_ids is not None else {}))
        runout_groups = ([route_runout_selection(config, group, kind)
                          for group, kind in zip(groups, kinds)]
                         if self.route_runout_settings is not None else None)
        if runout_groups is not None and not any(runout_groups):
            raise ValueError("Enabled terrain route runout has no selected training rows")
        layout_indices, layout_count = list(range(num_envs)), num_envs
        if config.get("evaluation_exact_cases") and config["evaluation"].get("stable_case_layout"):
            from wheeled_tasks.chassis.application.evaluation.evaluation import canonical_evaluation_layout
            layout_indices, layout_count = canonical_evaluation_layout(config["evaluation"], num_envs)
        grid = math.ceil(math.sqrt(layout_count))
        floor_width = config.get("flat_floor_width_m", 4.)
        if not 4. <= floor_width <= 8.:
            raise ValueError("Flat collider width must stay within the validated 4-8m tile range")
        half_length = config.get("flat_half_length_m", 44.)
        corridor_spacing = 2 * half_length + 12.
        corridor_columns = max(1, math.ceil(math.sqrt(layout_count * (floor_width + 1.) / corridor_spacing)))
        corridor_rows = math.ceil(layout_count / corridor_columns)
        origins, self.surfaces = [], []
        self.contact_domain = None
        if config.get("contact_domain", {}).get("enabled"):
            from wheeled_tasks.chassis.simulation.randomization.contact_domain import ContactDomain
            profiles = None
            if config.get("evaluation_exact_cases"):
                cases = {c["name"]: c for c in config["evaluation"]["cases"]}
                profiles = [cases[name].get("contact_profile") for name in groups]
            self.contact_domain = ContactDomain(num_envs, config["contact_domain"], seed, profiles)
        from wheeled_tasks.chassis.simulation.runtime.privileged_runtime import ContactMaterialPolicy
        material_policy = ContactMaterialPolicy.from_config(config,
            legacy_contact_domain_active=self.contact_domain is not None)
        combine = material_policy.friction_combine
        authored_material_paths = []
        self.shared_terrain = None
        if config.get("formal_terrain") is not None:
            if not config.get("formal_v7_flat"):
                raise ValueError("Shared reference terrain requires the formal V7 adapter")
            from wheeled_tasks.chassis.simulation.terrain.formal_terrain import create_shared_terrain
            ground_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1., dynamic_friction=1., restitution=0.,
                friction_combine_mode=combine,
                restitution_combine_mode=material_policy.restitution_combine)
            from wheeled_tasks.chassis.application.configuration.small_rough_course import KEY as mixed_key, selection as mixed_selection
            mixed = mixed_selection(config) if mixed_key in config else None
            # Inference may omit rehearsal allocation while retaining this
            # contract's independently validated collision geometry.
            terrain_seed = mixed.get("geometry_seed", seed) if mixed is not None else seed
            self.shared_terrain = create_shared_terrain(self.sim.stage,
                ({"type": "plane", "source": "scut0719", "env_spacing_m": 4.}
                    if self.evaluation_surface == "plane_regression_v1" else config["formal_terrain"]),
                num_envs, device, terrain_seed, ground_material,
                **({"rehearsal_collider": config["formal_flat_rehearsal_collider"]}
                    if "formal_flat_rehearsal_collider" in config else {}),
                **({"max_init_terrain_level": config["formal_rough_initial_level"]}
                    if "formal_rough_initial_level" in config
                    and self.evaluation_surface != "plane_regression_v1" else {}),
                **({"rehearsal_origins_by_env": self.rehearsal_spec["origins_by_env"]}
                    if self.rehearsal_spec is not None and self.rehearsal_spec["origins_by_env"] is not None else {}))
            if self.rehearsal_spec is not None and "small_rough_mix" in self.rehearsal_spec:
                self.shared_terrain.distribute_mixed_rough_origins()
            authored_material_paths.extend(self.shared_terrain.material_paths)
        for i, kind in enumerate(kinds):
            # A single line of 4096 corridors reaches tens of kilometres and
            # loses millimetre precision in float32 world-space constraints.
            slot = layout_indices[i]
            origin = (((slot % corridor_columns) - (corridor_columns - 1) / 2) * corridor_spacing,
                      ((slot // corridor_columns) - (corridor_rows - 1) / 2) * (floor_width + 1.), 0.) if config.get("evaluation_long_corridors") else ((slot % grid) * 10., (slot // grid) * 10., 0.)
            if self.shared_terrain is not None:
                origin = self.shared_terrain.origins_m[i]
            origins.append(origin)
            path = f"/World/envs/env_{i}"
            # Rough Tensor.tolist() origins are lists; USD translate needs a vector-compatible tuple.
            UsdGeom.Xform.Define(self.sim.stage, path).AddTranslateOp().Set(tuple(float(value) for value in origin))
            if self.shared_terrain is not None:
                # Legacy task metadata only. Shared terrain owns native
                # collision geometry, height queries and any outer boundary.
                self.surfaces.append([Surface(-1.0e6, 1.0e6, friction=1.)])
                continue
            UsdGeom.Xform.Define(self.sim.stage, path + "/Terrain")
            tiers = config.get("terrain_difficulty_tiers", [1.])
            limits = config.get("skill_specs", {}).get(groups[i], {}).get("terrain_limits", config["terrain_limits"])
            terrain_index = i
            if config.get("evaluation_exact_cases"):
                case = next(c for c in config["evaluation"]["cases"] if c["name"] == groups[i])
                limits = case.get("terrain_limits", limits)
                terrain_index = case.get("terrain_seed", 0) + i % (num_envs // len(config["evaluation"]["cases"]))
            terrain_level = config.get("skill_specs", {}).get(groups[i], {}).get("terrain_level", tiers[(i // len(tiers)) % len(tiers)])
            if config.get("evaluation_exact_cases"):
                terrain_level = case.get("terrain_level", case.get("skill", {}).get("terrain_level", terrain_level))
                if config.get("fixed_evaluation_terrain"):
                    terrain_level = 1.
            surfaces = terrain_surfaces(kind, limits, level * terrain_level, terrain_index)
            if (config.get("release_stop", {}).get("enabled")
                    and config.get("skill_specs", {}).get(groups[i], {}).get("kind") == "release_stop"
                    and kind != "flat"):
                from wheeled_tasks.chassis.tasks.recovery.release_stop import release_stop_surfaces
                surfaces = release_stop_surfaces(kind, limits, level * terrain_level,
                    terrain_index, config["release_stop"])
            if config.get("evaluation_long_corridors") and kind in ("flat", "jump"):
                # Preserve the validated collider aspect ratio. A single 80m
                # thin cuboid changed wheel contact in the GPU PhysX probe.
                surfaces = ([Surface(-half_length, half_length)] if config.get("flat_triangle_mesh")
                            else [Surface(x - 4., x + 4.) for x in range(-40, 41, 8)])
            if runout_groups is not None and runout_groups[i]:
                surfaces = append_route_runout(surfaces, self.route_runout_settings)
            surfaces = material_policy.nominal_surfaces(surfaces, kind=kind)
            self.surfaces.append(surfaces)
            if self.contact_domain is not None:
                surfaces = self.contact_domain.surfaces(surfaces, i)
                self.surfaces[-1] = surfaces
            if config.get("flat_triangle_mesh") and kind in ("flat", "jump"):
                mesh = UsdGeom.Mesh.Define(self.sim.stage, path + "/Terrain/surface_0/geometry/mesh")
                points, indices = corridor_mesh(half_length, floor_width)
                mesh.CreatePointsAttr(points)
                mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
                mesh.CreateFaceVertexIndicesAttr(indices)
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateDoubleSidedAttr(True)
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
                material_path = path + "/Terrain/Material"
                mu = material_policy.ground_friction(surfaces[0].friction, triangle_mesh=True)
                material_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=mu,
                    dynamic_friction=mu, restitution=0., friction_combine_mode=combine,
                    restitution_combine_mode=material_policy.restitution_combine)
                material_cfg.func(material_path, material_cfg)
                authored_material_paths.append(material_path)
                sim_utils.bind_physics_material(str(mesh.GetPath()), material_path)
                continue
            for j, surface in enumerate(surfaces):
                bottom = None
                if config.get("solid_step_platforms") and kind in ("low_step", "step_up", "step_down", "platform", "stairs", "stairs_down"):
                    bottom = min(s.z0 for s in surfaces) - .1
                size, position, quat = surface.box(width=floor_width if kind in ("flat", "jump") else 4., solid_bottom=bottom)
                if config.get("playback_open_ground"):
                    size = (size[0], 80., size[2])
                mu = material_policy.ground_friction(surface.friction)
                cfg = sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu,
                        dynamic_friction=mu, restitution=0., friction_combine_mode=combine,
                        restitution_combine_mode=material_policy.restitution_combine))
                # Surface.box uses USD's scalar-first quaternion; Isaac Lab 3
                # spawners take scalar-last quaternions.
                cfg.func(path + f"/Terrain/surface_{j}", cfg, translation=position,
                         orientation=(*quat[1:], quat[0]))
                authored_material_paths.append(path + f"/Terrain/surface_{j}/geometry/" + cfg.physics_material_path)
        actuators = {"effort": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0., damping=0.002,
            effort_limit=100., effort_limit_sim=100., velocity_limit_sim=1e9, armature=0., friction=0.)}
        if self.is_v5:
            springs = manifest["spring_joint_names"]
            other = [n for n in manifest["tree_joint_names"] if n not in springs]
            actuators = {
                "rotary": IdealPDActuatorCfg(joint_names_expr=other, stiffness=0., damping=.002,
                    effort_limit=100., effort_limit_sim=100., velocity_limit_sim=1e9, armature=0., friction=0.),
                "gas": IdealPDActuatorCfg(joint_names_expr=springs, stiffness=0., damping=.002,
                    effort_limit=1000., effort_limit_sim=1000., velocity_limit_sim=100., armature=0., friction=0.),
            }
            if config.get("monitor_applied_effort", False):
                active = manifest["control_joint_names"]
                passive = [n for n in other if n not in active]
                actuators = {
                    "legs": IdealPDActuatorCfg(joint_names_expr=[active[i] for i in (0, 1, 3, 4)],
                        stiffness=0., damping=0., effort_limit=40., effort_limit_sim=40., velocity_limit_sim=1e9),
                    "wheels": IdealPDActuatorCfg(joint_names_expr=[active[2], active[5]],
                        stiffness=0., damping=0., effort_limit=control["actuators"]["wheel"]["effort_limit"],
                        effort_limit_sim=control["actuators"]["wheel"]["effort_limit"], velocity_limit_sim=1e9),
                    "passive": IdealPDActuatorCfg(joint_names_expr=passive, stiffness=0., damping=.002,
                        effort_limit=100., effort_limit_sim=100., velocity_limit_sim=1e9),
                    "gas": actuators["gas"],
                }
                if self.actuator_response.enabled:
                    from isaaclab.utils.version import get_isaac_sim_version
                    if get_isaac_sim_version().major < 5:
                        raise RuntimeError("Identified effort-based friction requires Isaac Sim >=5")
                    native = self.actuator_response.native_parameters()
                    for group in ("legs", "wheels"):
                        names = actuators[group].joint_names_expr
                        actuators[group].armature = {name: native["armature"][name] for name in names}
                        actuators[group].friction = {name: native["frictionloss"][name] for name in names}
                        actuators[group].dynamic_friction = {name: native["frictionloss"][name] for name in names}
                        actuators[group].viscous_friction = {name: native["damping"][name] for name in names}
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(root / config["asset_directory"] / manifest["usd"]),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                    solver_position_iteration_count=config["position_iterations"],
                    solver_velocity_iteration_count=config["velocity_iterations"])),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.),
                joint_pos=manifest["nominal_joint_pos"], joint_vel={".*": 0.}),
            actuators=actuators,
            soft_joint_pos_limit_factor=1.)
        self.robot = Articulation(robot_cfg)
        robot_mu = material_policy.robot_friction
        material = sim_utils.RigidBodyMaterialCfg(static_friction=robot_mu, dynamic_friction=robot_mu,
            restitution=0., friction_combine_mode=combine,
            restitution_combine_mode=material_policy.restitution_combine)
        material.func("/World/RobotMaterial", material)
        authored_material_paths.append("/World/RobotMaterial")
        loop_count = 0
        formal_solver_roots = 0
        for prim in self.sim.stage.Traverse():
            if "/Robot/" not in str(prim.GetPath()):
                continue
            if config.get("formal_v7_flat") and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                actual = tuple(prim.GetAttribute("physxArticulation:" + name).Get() for name in
                    ("solverPositionIterationCount", "solverVelocityIterationCount"))
                expected = (config["position_iterations"], config["velocity_iterations"])
                if actual != expected:
                    raise RuntimeError(f"Formal solver properties differ at {prim.GetPath()}: {actual} != {expected}")
                formal_solver_roots += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                sim_utils.bind_physics_material(str(prim.GetPath()), "/World/RobotMaterial")
            if prim.IsA(UsdPhysics.SphericalJoint):
                joint = UsdPhysics.SphericalJoint(prim)
                if not joint.GetExcludeFromArticulationAttr().Get():
                    raise RuntimeError("Loop joint accidentally included in articulation tree")
                loop_count += 1
        if loop_count != (6 if self.is_v5 else 4) * num_envs:
            raise RuntimeError("Unexpected number of real loop constraints")
        if config.get("formal_v7_flat") and formal_solver_roots != num_envs:
            raise RuntimeError("Formal solver readback did not cover every articulation root")
        if self.shared_terrain is not None:
            from wheeled_tasks.chassis.simulation.terrain.formal_terrain import isolate_shared_terrain_robots
            collision_isolation_receipt = isolate_shared_terrain_robots(self.sim.stage, self.sim.cfg.physics_prim_path,
                [f"/World/envs/env_{i}" for i in range(num_envs)],
                self.shared_terrain.global_prim_paths)
        if material_policy.source_wheel_buckets:
            from wheeled_tasks.chassis.simulation.runtime.privileged_runtime import verify_authored_combine_modes
            material_authoring_receipt = verify_authored_combine_modes(self.sim.stage,
                authored_material_paths, material_policy)
        else:
            material_authoring_receipt = None
        self.contact_generation_distance = None
        if "formal_contact_generation_distance" in config:
            from wheeled_tasks.chassis.simulation.terrain.contact_generation_distance import ContactGenerationDistance, selected as contact_distance_selected
            contact_distance_selected(config)
            if self.shared_terrain is None:
                raise ValueError("Base contact distance requires an explicit shared terrain")
            self.contact_generation_distance = ContactGenerationDistance(
                self.sim.stage, self.shared_terrain.collision_paths, terrain=self.shared_terrain,
                retain_flat_parent_pairs="formal_small_rough_mix" in config,
                version=config["formal_contact_generation_distance"])
        self.sim.reset()
        if self.contact_domain is not None or material_policy.source_wheel_buckets:
            properties = wp.to_torch(self.robot.root_view.get_material_properties())
            if not torch.allclose(properties[..., :2], torch.ones_like(properties[..., :2]), atol=1e-6):
                raise RuntimeError("Unit robot friction material was not imported by PhysX")
        if set(self.robot.joint_names) != set(manifest["tree_joint_names"]) or set(self.robot.body_names) != set(manifest["rigid_body_names"]):
            raise RuntimeError("Unexpected solver topology")
        if self.contact_generation_distance is not None:
            self.contact_generation_distance.apply_to_robot(self.robot, PhysxManager.get_physics_sim_view())
        paths = list(self.robot.root_view.prim_paths)
        order = [int(p.split("/env_")[1].split("/")[0]) for p in paths]
        if sorted(order) != list(range(num_envs)):
            raise RuntimeError("Ambiguous PhysX clone order")
        self.origins = torch.tensor([origins[i] for i in order], device=device)
        if self.rehearsal_spec is not None:
            from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import reorder_fixed_flat_rehearsal
            self.rehearsal_spec = reorder_fixed_flat_rehearsal(self.rehearsal_spec, order)
        self.clone_indices = order
        self.domain_draw = (self.contact_domain.domain_draw[order].to(device)
                            if self.contact_domain is not None else None)
        self.kinds = [kinds[i] for i in order]
        self.scene_groups = [groups[i] for i in order]
        self.route_runout_mask = (torch.tensor([runout_groups[i] for i in order],
                                              dtype=torch.bool, device=device)
                                  if runout_groups is not None else None)
        self.surfaces = [self.surfaces[i] for i in order]
        max_surfaces = max(map(len, self.surfaces))
        surface_data = torch.zeros(num_envs, max_surfaces, 5, device=device)
        surface_valid = torch.zeros(num_envs, max_surfaces, dtype=torch.bool, device=device)
        for i, surfaces in enumerate(self.surfaces):
            surface_data[i, :len(surfaces)] = torch.tensor(
                [[s.x0, s.x1, s.z0, s.slope, s.cross_slope] for s in surfaces], device=device)
            surface_valid[i, :len(surfaces)] = True
        self.surface_data, self.surface_valid = surface_data, surface_valid
        self.surface_mu = torch.zeros(num_envs, max_surfaces, device=device)
        for i, surfaces in enumerate(self.surfaces):
            self.surface_mu[i, :len(surfaces)] = torch.tensor([s.friction for s in surfaces], device=device)
        self.platform_delta = torch.tensor(
            [s[-1].height(2.) - s[0].height(-1.5) for s in self.surfaces], device=device)
        self.env_paths = [p.split("/Robot")[0] for p in paths]
        body_paths = [p + "/Robot/" + n for p in self.env_paths for n in self.robot.body_names]
        terrain_filters = [list(self.shared_terrain.collision_paths) if self.shared_terrain is not None
                           else [p + f"/Terrain/surface_{j}/geometry/mesh" for j in range(len(self.surfaces[i]))]
                           for i, p in enumerate(self.env_paths)]
        filters = [terrain_filters[i] for i in range(num_envs) for _ in self.robot.body_names]
        view = PhysxManager.get_physics_sim_view()
        # Each filter must name one body; filter-list lengths must match within a view.
        self.contact_views = []
        for surface_count in sorted({len(paths) for paths in terrain_filters}):
            ids = [i for i, paths in enumerate(terrain_filters) if len(paths) == surface_count]
            group_paths = [self.env_paths[i] + "/Robot/" + n for i in ids for n in self.robot.body_names]
            group_filters = [terrain_filters[i] for i in ids for _ in self.robot.body_names]
            contact_view = view.create_rigid_contact_view(group_paths, filter_patterns=group_filters,
                max_contact_data_count=64 * len(group_paths))
            if contact_view.filter_count != surface_count:
                raise RuntimeError("Terrain contact filter count mismatch")
            self._validate_jump_trace_contact_view(contact_view, group_paths, group_filters)
            self.contact_views.append((torch.tensor(ids, device=device), contact_view))
        self.body_view = view.create_rigid_body_view(body_paths)
        if list(self.body_view.prim_paths) != body_paths:
            raise RuntimeError("Contact body order mismatch")
        masses = wp.to_torch(self.robot.root_view.get_masses())
        self.body_mass = masses.to(device=device).clone()
        self.nominal_body_mass = self.body_mass.clone()
        if not torch.allclose(masses.sum(-1), masses.new_full((num_envs,), manifest["total_mass_kg"]), atol=1e-5, rtol=0):
            raise RuntimeError("Mass mismatch after cloning")
        self.dynamics_randomization = None
        dynamics_cfg = config.get("dynamics_randomization", {})
        if dynamics_cfg.get("enabled", False):
            from wheeled_tasks.chassis.simulation.randomization.dynamics import RigidBodyRandomization
            profiles = None
            if config.get("evaluation_exact_cases"):
                cases_by_name = {case["name"]: case for case in config["evaluation"]["cases"]}
                profiles = [cases_by_name[name].get("dynamics_profile") for name in self.scene_groups]
            self.dynamics_randomization = RigidBodyRandomization(
                self.body_mass, wp.to_torch(self.robot.root_view.get_inertias()).to(device).clone(),
                wp.to_torch(self.robot.root_view.get_coms()).to(device).clone(),
                self.robot.body_names, dynamics_cfg, seed + 8291, profiles,
                enabled_mask=(self.domain_draw < dynamics_cfg["enabled_fraction"] if self.domain_draw is not None else None))
            self.dynamics_randomization.apply(self.robot)
            actual = wp.to_torch(self.robot.root_view.get_masses()).to(device)
            if not torch.allclose(actual, self.dynamics_randomization.masses, atol=1e-5, rtol=1e-6):
                raise RuntimeError("Randomized body masses were not applied to PhysX")
            actual_inertias = wp.to_torch(self.robot.root_view.get_inertias()).to(device)
            actual_coms = wp.to_torch(self.robot.root_view.get_coms()).to(device)
            for name, actual_property, expected in (
                    ("inertia", actual_inertias, self.dynamics_randomization.inertias),
                    ("COM", actual_coms[:, :, :3], self.dynamics_randomization.coms[:, :, :3])):
                if not torch.allclose(actual_property, expected, atol=1e-6, rtol=1e-5):
                    error = (actual_property - expected).abs()
                    index = torch.unravel_index(error.argmax(), error.shape)
                    raise RuntimeError(f"Randomized {name} readback mismatch: max_error={float(error.max())}, "
                        f"index={tuple(int(i) for i in index)}, actual={float(actual_property[index])}, "
                        f"expected={float(expected[index])}")
            self.body_mass.copy_(actual)
        self.root_kinematics = StaticRootLinkKinematics(self.robot, num_envs, device)
        self.startup_report = {"body_count": self.body_count, "tree_dofs": self.joint_count, "loop_constraints": loop_count,
            "num_envs": num_envs, "terrain_families": self.kinds,
            "mass_kg_each": self.body_mass.sum(-1).cpu().tolist(), "contact_filters": filters,
            "self_collision_enabled": False, "solver_order": order}
        if self.reward_episode_measurement_enabled:
            self.startup_report["reward_episode_measurement"] = {
                "version": "reward_episode_measurement_v1", "enabled": True,
                "scope": "actual_environment_payments_current_worker_measurement_only",
                "payment_units": "already_dt_scaled_payment_per_policy_step",
                "policy_dt_s": self.policy_dt, "reward_profile": config["reward_profile"],
                "reward_changed": False, "native_exact_resume_qualified": False,
                "PPO_return_target": False}
        self.startup_report["contact_material_policy"] = material_policy.summary()
        if self.contact_generation_distance is not None:
            self.startup_report["contact_generation_distance"] = self.contact_generation_distance.receipt
        if self.shared_terrain is not None:
            self.startup_report["formal_terrain"] = self.shared_terrain.summary()
            self.startup_report["formal_terrain"]["collision_isolation_readback"] = collision_isolation_receipt
        if self.rehearsal_spec is not None:
            from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import rehearsal_summary
            self.startup_report["fixed_flat_rehearsal"] = rehearsal_summary(self.rehearsal_spec)
        if self.evaluation_surface == "plane_regression_v1":
            self.startup_report["evaluation_surface_override"] = {
                "evaluation_surface": self.evaluation_surface, "run_role": run_role,
                "training_terrain": dict(config["formal_terrain"]),
                "actual_terrain": {"type": "plane", "source": "scut0719", "env_spacing_m": 4.},
                "training_contract_rewritten": False, "command_owner": "ScutFlatCommands_fixed_requests",
                "birth_placement": "PhysicalFlatPlacement", "rough_birth_support_enabled": False,
                "height_scan": "nine_analytic_forward_rays_on_world_z0_plane",
                "retained_reward_profile": config["reward_profile"],
                "retained_critic_height_reference": config["critic_height_reference"]}
        if config.get("formal_closure_monitor_only"):
            self.startup_report["closure_monitor"] = {
                "gap_telemetry_enabled": True, "gap_reset_enabled": False,
                "gap_fatal_threshold_m": None, "finite_state_checks_enabled": True}
        elif config.get("closure_gap_termination_m"):
            self.startup_report["closure_monitor"] = {
                "gap_telemetry_enabled": True,
                "gap_reset_enabled": not config.get("formal_closure_observation_only", False),
                "gap_reset_threshold_m": config["closure_gap_termination_m"],
                "comparison": "strictly_greater_than",
                "measurement": "per_lane_peak_over_5ms_substeps",
                "reset_boundary": "20ms_policy_end",
                "finite_state_checks_enabled": True}
        if material_authoring_receipt is not None:
            self.startup_report["contact_material_policy"]["authored_readback"] = material_authoring_receipt
        if self.route_runout_settings is not None:
            self.startup_report["terrain_route_runout"] = {
                **self.route_runout_settings,
                "selected_environment_count": sum(runout_groups),
                "selected_groups": sorted({group for group, selected in zip(groups, runout_groups) if selected}),
                "negative_boundary_x_m": -3.6, "lateral_boundary_y_m": 1.7,
                "corridor_spacing_m": corridor_spacing,
                "corridor_y_spacing_m": floor_width + 1.}
        if self.actuator_response.enabled:
            response = self.actuator_response.cfg
            response_ids = [self.robot.joint_names.index(name) for name in response["joint_names"]]
            friction = wp.to_torch(self.robot.root_view.get_dof_friction_properties())[:, response_ids]
            native = {"armature": wp.to_torch(self.robot.root_view.get_dof_armatures())[:, response_ids],
                "frictionloss": friction[..., 0], "dynamic_friction": friction[..., 1],
                "damping": friction[..., 2]}
            if list(manifest["control_joint_names"]) != response["joint_names"]:
                raise RuntimeError("Identified response axis order differs from the actual asset")
            for field, actual in native.items():
                source = "frictionloss" if field == "dynamic_friction" else field
                expected = actual.new_tensor(response[source])
                if not torch.allclose(actual, expected.expand_as(actual), atol=1e-7, rtol=1e-6):
                    raise RuntimeError(f"Native actuator response readback differs: {field}")
            self.startup_report["actuator_response"] = {
                "config": response, "native_properties_verified": True,
                "native_first_environment": {k: v[0].cpu().tolist() for k, v in native.items()},
                "functional_probe": self.actuator_response.runtime_probe()}
        if self.dynamics_randomization is not None:
            self.startup_report["nominal_mass_kg"] = manifest["total_mass_kg"]
            self.startup_report["dynamics_randomization"] = self.dynamics_randomization.summary()
        if self.contact_domain is not None:
            self.startup_report["contact_domain"] = self.contact_domain.summary()
        scene = self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)
        if config.get("formal_v7_flat"):
            scene_settings = {name: scene.GetAttribute("physxScene:" + name).Get() for name in
                ("solverType", "timeStepsPerSecond", "enableExternalForcesEveryIteration")}
            expected = {"solverType": "PGS" if config.get("physx_solver_type", 1) == 0 else "TGS",
                "timeStepsPerSecond": round(1. / self.dt),
                "enableExternalForcesEveryIteration": config.get("physx_external_forces_every_iteration", False)}
            if "gpu_found_lost_aggregate_pairs_capacity" in config:
                attribute = "gpuFoundLostAggregatePairsCapacity"
                scene_settings[attribute] = scene.GetAttribute("physxScene:" + attribute).Get()
                expected[attribute] = config["gpu_found_lost_aggregate_pairs_capacity"]
            if scene_settings != expected:
                raise RuntimeError(f"Formal physics scene properties differ: {scene_settings} != {expected}")
            self.startup_report["solver_configuration_readback"] = {
                "scope": "authored_USD_properties_after_simulation_initialization; not_a_closure_accuracy_pass",
                "articulation_roots_checked": formal_solver_roots,
                "position_iterations": config["position_iterations"],
                "velocity_iterations": config["velocity_iterations"], **scene_settings}
            if ("physx_friction_offset_threshold_m" in config
                    or "formal_flat_rehearsal_collider" in config):
                distance = scene.GetAttribute("physxScene:frictionOffsetThreshold").Get()
                configured_distance = (config["physx_friction_offset_threshold_m"]
                    if "physx_friction_offset_threshold_m" in config
                    else self.sim.cfg.physics.friction_offset_threshold)
                if (type(distance) not in (int, float) or not math.isclose(distance,
                        configured_distance, rel_tol=1e-6, abs_tol=1e-9)):
                    raise RuntimeError("Formal friction activation distance differs from the selected contact repair")
                self.startup_report["solver_configuration_readback"]["frictionOffsetThreshold"] = distance
            if "physx_solver_type" in config:
                self.startup_report["solver_configuration_readback"]["physx_solver_type"] = config["physx_solver_type"]
        self.startup_report.update(physics_device=str(device),
            gpu_dynamics_enabled=bool(scene.GetAttribute("physxScene:enableGPUDynamics").Get()),
            state_tensor_device=str(self.robot.data.root_link_pose_w.torch.device))
        self.startup_report["scene_group_counts"] = {name: self.scene_groups.count(name) for name in set(self.scene_groups)}
        self.sampling_metrics = {}
        if config.get("behavior_pool_membership") and not config.get("evaluation_exact_cases"):
            membership = config["behavior_pool_membership"]
            counts = {name: sum(membership[group] == name for group in self.scene_groups)
                      for name in config["behavior_pool_fractions"]}
            self.startup_report["behavior_pool_counts"] = counts
            self.sampling_metrics = {f"/sampling/{name}_fraction": count / num_envs for name, count in counts.items()}
        shared_collision_paths = set(self.shared_terrain.collision_paths) if self.shared_terrain is not None else set()
        self.startup_report["terrain_collision_paths"] = [str(p.GetPath()) for p in self.sim.stage.Traverse()
            if ("/Terrain/" in str(p.GetPath()) or str(p.GetPath()).startswith("/World/ground/")
                or str(p.GetPath()) in shared_collision_paths)
            and p.HasAPI(UsdPhysics.CollisionAPI)]
        self.ids = [self.robot.joint_names.index(n) for n in manifest["control_joint_names"]]
        self.wheel_ids = [self.robot.body_names.index(n) for n in ("L_link3", "R_link3")]
        self.wheel_offsets = torch.zeros(2, 3, device=device)
        if self.is_v5:
            bodies = {b["name"]: b for b in self.model_spec["bodies"]}
            self.wheel_offsets = torch.tensor([[bodies[n]["collisions"][0]["origin"][i][3] for i in range(3)]
                                               for n in ("L_link3", "R_link3")], device=device)
        self.nonwheel_ids = [i for i in range(self.body_count) if i not in self.wheel_ids]
        grace = config.get("step_contact_grace")
        self.step_contact_time = torch.zeros(num_envs, device=device)
        self.step_contact_allowed = torch.zeros(num_envs, self.body_count, dtype=torch.bool, device=device)
        if grace:
            allowed = [self.robot.body_names.index(name) for name in grace["bodies"]]
            if self.robot.body_names.index("base_link") in allowed or not 0 < grace["seconds"] <= .5:
                raise ValueError("Step contact grace requires bounded non-base support")
            high_steps = torch.tensor([kind == "step_up" and float(delta) >= .15 - 1e-5
                                      for kind, delta in zip(self.kinds, self.platform_delta)], device=device)
            self.step_contact_allowed[:, allowed] = high_steps[:, None]
        constraints = self.model_spec["constraints"] if self.is_v5 else manifest["closed_chain_constraints"]
        self._closure_body_ids = torch.tensor([[self.robot.body_names.index(c[f"body{end}"])
                                               for end in (0, 1)] for c in constraints], device=device)
        self._closure_points = torch.tensor([[c[f"local_pos{end}_m"] for end in (0, 1)]
                                             for c in constraints], device=device)
        self.knee_ids = [self.robot.joint_names.index(n) for n in ("L_joint2", "R_jonit2")]
        if self.is_v5:
            self.spring_ids = [self.robot.joint_names.index(n) for n in manifest["spring_joint_names"]]
            self.knee_bounds = {j["name"]: [float(j["limit"]["lower"]), float(j["limit"]["upper"])]
                                for j in self.model_spec["joints"] if j["name"] in ("L_joint2", "R_jonit2")}
            limits = wp.to_torch(self.robot.root_view.get_dof_limits())
            for j in self.model_spec["joints"]:
                if j["type"] in ("revolute", "prismatic"):
                    index = self.robot.joint_names.index(j["name"])
                    expected = limits.new_tensor([float(j["limit"]["lower"]), float(j["limit"]["upper"])])
                    if not torch.allclose(limits[:, index], expected.expand(num_envs, 2), atol=1e-6, rtol=0):
                        raise RuntimeError(f"V5 solver limits differ: {j['name']}")
            self.startup_report["gas_spring_force_enabled"] = True
            self.startup_report["active_joint_order"] = manifest["control_joint_names"]
            if config.get("gas_pressure_application") == "main_link_mounts_v1":
                from wheeled_tasks.chassis.simulation.spring_force_element import SpringForceElement
                self.spring_pressure_element = SpringForceElement(self.model_spec, manifest,
                    fit, self.robot.body_names, device)
                self._spring_pressure_body_ids = torch.tensor(
                    self.spring_pressure_element.body_ids, dtype=torch.int32, device=device)
                self._spring_pressure_generated_tick = torch.zeros(num_envs, 1,
                    dtype=torch.int64, device=device)
                self.startup_report["gas_pressure_application"] = {
                    "version": "main_link_mounts_v1",
                    "main_link_body_order": self.spring_pressure_element.main_link_body_names,
                    "pressure_slider_effort_target_n": 0., "slider_damping_ns_m": .002,
                    "wrench_frame": "world_force_and_moment_about_actual_body_COM",
                    "pressure_source": "unchanged_10MPa_cubic_and_spring_strength",
                    "native_stability_qualified": False}
        else:
            self.knee_bounds = control["joints"]["knee_hard_limits"]
        self.nominal = torch.tensor([[manifest["nominal_joint_pos"][n] for n in self.robot.joint_names]], device=device).repeat(num_envs, 1)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        if self.reward_episode_measurement_enabled:
            # Independent of optional recovery/flat/jump producers. Only a real
            # reset below advances this current-worker lane identity.
            self.reward_episode_ids = torch.full_like(self.episode_length_buf, -1)
        self.episode_limits = torch.full_like(self.episode_length_buf, self.max_episode_length)
        if config.get("evaluation_exact_cases"):
            cases_by_name = {c["name"]: c for c in config["evaluation"]["cases"]}
            self.episode_limits[:] = torch.tensor([round(cases_by_name[g].get("episode_seconds", config["episode_seconds"]) / self.policy_dt)
                                                  for g in self.scene_groups], device=device)
        self.commands = torch.zeros(num_envs, 3, device=device)
        self.raw_height = torch.zeros(num_envs, device=device)
        self.command_acceleration = torch.zeros(num_envs, device=device)
        self.step_assist = None
        if config.get("step_assist", {}).get("enabled"):
            from wheeled_tasks.chassis.tasks.terrain.step_assist import StepHeightAssist
            self.step_assist = StepHeightAssist(num_envs, device, self.policy_dt, config["step_assist"])
            cases_by_name = {case["name"]: case for case in config["evaluation"]["cases"]}
            self.step_assist_enabled = torch.tensor([
                cases_by_name[group].get("step_assist_enabled", True) if config.get("evaluation_exact_cases") else True
                for group in self.scene_groups], device=device)
        self.command_target = torch.zeros(num_envs, 2, device=device)
        self.mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.targets = torch.zeros(num_envs, 4, device=device)
        self.actions = torch.zeros(num_envs, 6, device=device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.before_previous_actions = torch.zeros_like(self.actions)
        self.previous_motor_velocity = torch.zeros_like(self.actions)
        self.torque = torch.zeros_like(self.actions)
        self.contact_force = torch.zeros(num_envs, self.body_count, 3, device=device)
        self.contact_peak = torch.zeros(num_envs, device=device)
        self.wheel_force_peak_n = torch.zeros(num_envs, 2, device=device) if self.jump_landing is not None else None
        self.phase = PhaseTracker(num_envs, device, self.policy_dt)
        # Acceptance keeps the original one-frame criterion, independently of training recovery time.
        fall_seconds = 0. if config.get("evaluation_exact_cases") else config.get("fall_confirmation_seconds", 0.)
        self.fall_confirmation = FallConfirmation(num_envs, device, self.policy_dt, fall_seconds)
        self.history = HistoryStack(num_envs, device, length=config["history_length"], dim=config["actor_frame_dim"])
        self.tick = 0
        self.completed_updates = 0
        self.max_closure_gap = 0.
        self.max_closure_gap_all_finite = 0.
        self.max_closure_gap_numerical_reset = 0.
        self.phase_counts = torch.zeros(5, dtype=torch.long, device=device)
        self.termination_counts = dict(fall=0, nonwheel_contact=0, knee=0, boundary=0, takeoff_timeout=0)
        if self.step_assist is not None:
            self.termination_counts["blocked"] = 0
        if config.get("closure_gap_termination_m"):
            self.termination_counts["closure_gap"] = 0
        if self.is_v5:
            self.termination_counts["spring_travel"] = 0
        if config.get("task_semantics"):
            self.termination_counts.update(jump_request_timeout=0, jump_outcome_timeout=0)
        self.spring_reserve_frames = 0
        self.success_count = 0
        self.timeout_count = 0
        self.success_hold = torch.zeros(num_envs, device=device)
        self.sequence_success_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.successful_jump_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.jump_requested = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.command_clock = torch.zeros(num_envs, device=device)
        self.height_clock = torch.zeros_like(self.command_clock)
        self.push_clock = torch.zeros_like(self.command_clock)
        self.push_enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.route = torch.tensor([k not in ("flat", "jump") for k in self.kinds], device=device)
        self.has_route_tasks = any(k not in ("flat", "jump") for k in self.kinds)
        self.has_jump_tasks = "jump" in self.kinds
        self.torque_monitor = None
        if self.is_v5 and config.get("monitor_applied_effort", False):
            from wheeled_tasks.chassis.simulation.telemetry.torque_monitor import TorqueMonitor
            self.torque_monitor = TorqueMonitor(self.scene_groups, manifest["control_joint_names"], device,
                control["actuators"]["wheel"]["effort_limit"],
                mount_pressure=self.spring_pressure_element is not None)
        self.full_tasks, self.perturbations = None, None
        self.performance_curriculum, self.stationary_anchor = None, None
        self.stationary_group = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.skills = None
        if config.get("skill_specs"):
            from wheeled_tasks.chassis.application.runtime.skill_commands import SkillCommands
            self.skills = SkillCommands(self, root=root)
        if config.get("stationary_tracking"):
            from wheeled_tasks.chassis.tasks.rewards.rewards import StationaryAnchor
            settings = config["stationary_tracking"]
            self.stationary_anchor = StationaryAnchor(num_envs, device, settings["position_weight"], settings["position_band_m"])
            self.stationary_group[:] = torch.tensor([
                self.skills.specs.get(group, {}).get("kind") in ("stand", "height")
                for group in self.scene_groups], device=device)
        if config.get("performance_curriculum") and not config.get("evaluation_exact_cases"):
            from wheeled_tasks.chassis.tasks.curricula.performance_curriculum import PerformanceCurriculum
            curriculum_type = PerformanceCurriculum
            if config["performance_curriculum"].get("kind") == "adaptive_commands":
                from wheeled_tasks.chassis.application.runtime.adaptive_commands import AdaptiveCommandCurriculum
                curriculum_type = AdaptiveCommandCurriculum
            self.performance_curriculum = curriculum_type(
                self.scene_groups, self.skills.specs, config["performance_curriculum"], device)
        if config.get("task_semantics"):
            from wheeled_tasks.chassis.tasks.curricula.full_tasks import FullTaskSemantics
            semantics = dict(config["task_semantics"])
            if self.skills is not None:
                overrides = [self.skills.specs.get(group, {}).get("semantics", {}) for group in self.scene_groups]
                for key in {key for values in overrides for key in values}:
                    semantics[key] = torch.tensor([values.get(key, config["task_semantics"].get(key, 0.))
                                                   for values in overrides], device=device)
            self.full_tasks = FullTaskSemantics(num_envs, device, self.policy_dt, semantics)
        if config.get("command_reference"):
            from wheeled_tasks.chassis.tasks.commands.references import CommandReference
            if not self.scut35 or self.body_count != 19 or self.full_tasks is None:
                raise ValueError("Command-reference control requires the declared V5 task interface")
            self.references = CommandReference(num_envs, device, config["command_reference"])
        if config.get("step_jump_task", {}).get("enabled"):
            from wheeled_tasks.chassis.tasks.terrain.step_jump import StepJumpTask
            from wheeled_tasks.chassis.tasks.terrain.step_climb import StepClimbTask
            from wheeled_tasks.chassis.tasks.terrain.step_sensors import StepTaskSensors
            self.step_jump = StepJumpTask(self.scene_groups, self.skills.specs, device, self.policy_dt)
            self.step_climb = StepClimbTask(self.scene_groups, self.skills.specs, device, self.policy_dt)
            self.has_jump_tasks |= self.step_jump.has_tasks
            if self.step_jump.has_tasks:
                self.step_sensors = StepTaskSensors(num_envs, device, self.policy_dt,
                    config["step_jump_task"], root, seed, evaluation=config.get("evaluation_exact_cases", False))
            self._init_body_support_contacts()
            self.startup_report["step_tasks"] = {
                "step_jump_count": int(self.step_jump.enabled.sum()),
                "body_climb_count": int(self.step_climb.enabled.sum()),
                "normal_climb_request": "forward_velocity_only",
                "jump_request_inputs": "preset_height_wheel_odometry_imu_single_beam",
                "contact_evidence": "native_physx_base_link_contact_points"}
        if config.get("signal_perturbations", {}).get("enabled", False):
            from wheeled_tasks.chassis.simulation.randomization.robustness import V5SignalPerturbations
            self.perturbations = V5SignalPerturbations(num_envs, device,
                {**config["signal_perturbations"], "frame_dim": config["actor_frame_dim"]}, self.generator)
        self.actor_sensor_noise = None
        if "actor_sensor_noise" in config:
            from wheeled_tasks.chassis.observations.actor_sensor_noise import ActorSensorNoise
            self.actor_sensor_noise = ActorSensorNoise(num_envs, device, config, self.generator)
            self.startup_report["actor_sensor_noise"] = {
                "settings": dict(self.actor_sensor_noise.cfg), "actor_only": True,
                "critic_noise": False, "feedback_pd_noise": False,
                "commands_and_action_history_noise": False}
        self.policy_signal_delay = None
        if "policy_signal_delay" in config:
            from wheeled_tasks.chassis.control.transport.policy_signal_delay import PolicySignalDelay
            self.policy_signal_delay = PolicySignalDelay(num_envs, device, config, self.generator)
            self.startup_report["policy_signal_delay"] = {
                "settings": dict(self.policy_signal_delay.cfg),
                "pc_feedback_extra_delay": False, "frame_stack": False,
                "observation_group_order": ["position", "velocity", "gyro", "gravity"],
                "critic_observation_age_order": ["gyro", "gravity", "position", "velocity"],
                "target_group_order": ["legs", "wheels"],
                "action_age_reference": "selected target sample through local downlink, before motor response",
                "policy_origin_reference": "independent original leg/wheel proposal generation timestamps"}
        self.signal_delay = None
        if config.get("signal_delay", {}).get("enabled"):
            if (not self.scut35 or not self.is_v5 or config.get("usb_transport", {}).get("enabled")
                    or self.perturbations is not None and self.perturbations.max_lag):
                raise ValueError("Physical signal delay requires 35D and no second policy-tick delay")
            from wheeled_tasks.chassis.control.transport.signal_delay import PhysicalSignalDelay
            self.signal_delay = PhysicalSignalDelay(num_envs, device, config["signal_delay"], self.dt, self.generator)
            self.startup_report["signal_delay"] = dict(config["signal_delay"],
                physics_dt_s=self.dt, policy_dt_s=self.policy_dt,
                quantum_ms=self.dt * 1000., physics_steps_observed=0)
        self.usb_transport = None
        if config.get("usb_transport", {}).get("enabled", False):
            if not self.scut35 or not self.is_v5:
                raise ValueError("USB transport requires the V5 35D control interface")
            if self.perturbations is not None and self.perturbations.max_lag:
                raise ValueError("USB transport cannot silently stack with policy-tick delay")
            from wheeled_tasks.chassis.control.transport.usb_transport import SubstepUsbTransport
            self.usb_transport = SubstepUsbTransport(num_envs, device, self.dt, config["usb_transport"], seed)
            self.startup_report["usb_transport"] = {
                "enabled": True, "model": "substep_rtt_sensor_interpolation_and_torque_impulse",
                "payload_size": config["usb_transport"]["payload_size"],
                "jitter": "iid_per_physics_step_quantile_surrogate", "directional_split": "research_assumption"}
        self.feedback_transport = None
        self.round_trip_transport = None
        if "round_trip_transport" in config:
            from wheeled_tasks.chassis.control.transport.round_trip_transport import RoundTripDelayBudget
            self.round_trip_transport = RoundTripDelayBudget(
                num_envs, device, config["round_trip_transport"], seed)
        communication_draw = self.domain_draw
        if "feedback_transport" in config:
            from wheeled_tasks.chassis.control.transport.feedback_transport import FractionalFeedbackTransport, shared_activation_draw
            communication_draw = shared_activation_draw(num_envs, device, seed, self.domain_draw)
            if config["feedback_transport"]["enabled"]:
                if not self.scut35 or not self.is_v5:
                    raise ValueError("Returned PC feedback requires the six-axis 35D interface")
                self.feedback_transport = FractionalFeedbackTransport(num_envs, device, self.dt,
                    config["feedback_transport"], seed, domain_draw=communication_draw,
                    round_trip_budget=self.round_trip_transport)
            self.startup_report["feedback_transport"] = dict(config["feedback_transport"],
                model=(config["feedback_transport"]["version"] if self.feedback_transport is not None else None),
                feedback_consumers="policy_pc_pd_recovery", physics_dt_s=self.dt,
                policy_dt_s=self.policy_dt)
        self.route_goal = torch.tensor([1.8 if kind in ("stairs", "stairs_down", "slope", "slope_up", "slope_down", "rough", "cross_slope", "material") else config.get("task_semantics", {}).get("route_goal_x_m", .65)
                                        for kind in self.kinds], device=device)
        self.motor_strength = torch.ones(num_envs, 6, device=device)
        self.command_transport = None
        if config.get("command_transport", {}).get("enabled"):
            if self.usb_transport is not None:
                raise ValueError("Downlink transport must not double-count legacy USB RTT")
            from wheeled_tasks.chassis.control.transport.command_transport import UsbCommandTransport
            profiles = None
            if config.get("evaluation_exact_cases"):
                from wheeled_tasks.chassis.control.transport.pc_communication import command_profiles
                profiles = command_profiles(config, self.scene_groups)
            if self.round_trip_transport is not None:
                from wheeled_tasks.chassis.control.transport.round_trip_transport import EffectiveCommandDelay
                if profiles is not None and any(profile is not None for profile in profiles):
                    raise ValueError("Shared communication delay cannot use independent command profiles")
                self.command_transport = EffectiveCommandDelay(num_envs, device, self.dt,
                    config["command_transport"], self.round_trip_transport)
            else:
                self.command_transport = UsbCommandTransport(num_envs, device, self.dt,
                    config["command_transport"], seed, profiles, domain_draw=communication_draw)
            self.startup_report["command_transport"] = dict(config["command_transport"])
            self.startup_report["command_transport"].update(physics_dt_s=self.dt,
                transport_dt_s=self.command_transport.transport_dt,
                transport_ticks_per_physics_step=self.command_transport.substeps,
                application=("effective_latency_interval_torque_average" if self.round_trip_transport is not None
                    else "phase_aware_interval_torque_average" if self.command_transport.phase_aware
                    else "interval_held_torque_average" if self.command_transport.substeps > 1 else "one_tick_hold"))
        if self.round_trip_transport is not None and self.actuator_response.enabled:
            self.startup_report["actuator_response"]["transport_integration"] = {
                "model": "piecewise_delayed_input_exact_one_pole_interval_mean",
                "native_effort": "mean over the real 5ms interval",
                "filter_state": "endpoint retained separately for the next interval",
                "pd_age_scope": "transported PD requests before actuator dynamics",
                "legacy_apply_torque": "unchanged outside the shared round-trip path"}
        self.spring_strength = torch.ones(num_envs, 1, device=device)
        from wheeled_tasks.chassis.simulation.randomization.dynamics import validate_episode_spring_force_randomization
        self.episode_spring_strength_settings = validate_episode_spring_force_randomization(config)
        if self.episode_spring_strength_settings is not None:
            self.startup_report["episode_spring_force_randomization"] = dict(
                self.episode_spring_strength_settings,
                scope="pressure_force_multiplier_not_damping_or_a_measured_uncertainty_interval")
        # Terrain/reset origins are immutable within this environment. Cache
        # them once instead of issuing O(num_envs) tiny CUDA writes per reset.
        offsets = []
        for group, kind in zip(self.scene_groups, self.kinds):
            x = 0. if config.get("centered_locomotion_resets") and kind in ("flat", "jump") else -1.5
            if (config.get("release_stop", {}).get("enabled")
                    and config.get("skill_specs", {}).get(group, {}).get("kind") == "release_stop"):
                x = 0.
            if config.get("scene_groups"):
                if group in ("stand", "translate", "rotate") and kind in ("slope", "material", "rough"):
                    x = 0.
                elif kind == "platform":
                    x = 1.
                elif group in ("step_up", "step_down"):
                    x = -.8
            offsets.append(x)
        self.reset_positions = self.origins.clone()
        self.reset_positions[:, 0] += torch.tensor(offsets, device=device)
        if self.rehearsal_spec is not None:
            selected = torch.tensor([case >= 0 for case in self.rehearsal_spec["case_index"]], device=device)
            if "flat_lane_mask" in self.rehearsal_spec:
                selected = torch.tensor(self.rehearsal_spec["flat_lane_mask"], device=device)
            if "scripted_plane_lane_mask" in self.rehearsal_spec:
                selected |= torch.tensor(self.rehearsal_spec["scripted_plane_lane_mask"], device=device)
            self.reset_positions[selected, :2] = self.origins[selected, :2]
        self._configure_rough_birth_anchor()
        self.reset_positions[:, 2] += self.ground_height(self.reset_positions[:, :2] - self.origins[:, :2]) + .32
        if self.step_jump is not None:
            selected = self.step_jump.enabled | self.step_climb.enabled
            self.reset_positions[selected, 0] = self.origins[selected, 0] - .8
            self.reset_positions[selected, 2] = self.origins[selected, 2] + .32
        self.jump_assist = None
        if config.get("jump_assist", {}).get("enabled"):
            from wheeled_tasks.chassis.tasks.terrain.step_assist import JumpAssist
            self.jump_assist = JumpAssist(num_envs, device, config["jump_assist"])
            if (bool(config.get("integrated_runtime_candidate")) and not config.get("evaluation_exact_cases")):
                owners, settings = self._component_configuration("jump_assist")
                self.jump_assist.configure_components(owners, settings)
            self.assist_body_ids = torch.tensor([self.robot.body_names.index("base_link")], device=device)
            self.assist_force = torch.zeros(num_envs, 1, 3, device=device)
        if config.get("recovery_training", {}).get("enabled"):
            from wheeled_tasks.chassis.application.runtime.recovery_training import RecoveryTraining
            self.recovery = RecoveryTraining(self, root, config["recovery_training"])
            if (self.recovery_hold_cost_settings is not None
                    and self.recovery_hold_cost_settings["enabled"]):
                from wheeled_tasks.chassis.tasks.rewards.rewards import validate_recovery_hold_gates
                self.recovery_hold_cost_gate_binding = validate_recovery_hold_gates(
                    self.recovery_hold_cost_settings, self.recovery.fsm, self.recovery.geometry)
            for reason in self.recovery.failure_reasons():
                self.termination_counts[reason] = 0
            self.startup_report["recovery_training"] = {
                "enabled": True, "count": int(self.recovery.enabled.sum()),
                "reference_dt": self.recovery.reference_dt, "pd_dt": self.dt,
                "blend_seconds": self.recovery.fsm.config["blend_seconds"],
                "state_writes": "episode_initial_reset_only",
                "learning": "exclude_script_and_blend_transitions",
                "physical_event_source": self.recovery.source_identity(),
                "physical_event_source_sha256": self.recovery.source_identity_sha256,
                "v7_skill_qualification": "not_assessed"}
        if "nonstandard_start" in config:
            from wheeled_tasks.chassis.tasks.recovery.nonstandard_start import NonstandardStart
            self.nonstandard_start = NonstandardStart(num_envs, device, self.dt,
                config["nonstandard_start"], root, self.robot.joint_names,
                evaluation=bool(config.get("evaluation_exact_cases")),
                asset_manifest_sha256=config["asset_manifest_sha256"])
            if self.nonstandard_start.enabled and not config.get("command_slew"):
                raise ValueError("Nonstandard birth capture requires the existing command slew")
            eligible = torch.tensor([kind != "jump" for kind in self.kinds], device=device)
            if self.skills is not None:
                # World-frame spin references also own reward and observation
                # targets; a body-command-only birth cap cannot replace them.
                eligible &= ~self.skills.spin
            if self.recovery is not None:
                eligible &= ~self.recovery.enabled
            if self.step_jump is not None:
                eligible &= ~(self.step_jump.enabled | self.step_climb.enabled)
            campaign = config.get("task_campaign", {})
            if (campaign.get("revision_migration", {}).get("schema") == "task-worker-revision-v17"
                    and campaign.get("domain") in ("jump", "stepjump")):
                # Request-sequence rows can have a flat terrain or forward
                # spec. Only the 41 declared ordinary rehearsal rows may use
                # passive births; active requests own their complete evidence.
                ordinary_pools = {"stand_anchors", "stand_continuous", "height",
                    "basic_translation", "basic_transitions", "basic_yaw"}
                membership = config["behavior_pool_membership"]
                eligible &= torch.tensor([membership[group] in ordinary_pools
                    for group in self.scene_groups], device=device)
                if bool((eligible & (self.mode == 4)).any()):
                    raise ValueError("Nonstandard jump rehearsal must exclude every active request lane")
                self.startup_report["nonstandard_start_eligibility"] = {
                    "scope": "v17_declared_ordinary_rehearsal_only",
                    "eligible_count": int(eligible.sum()),
                    "ordinary_pools": sorted(ordinary_pools),
                    "request_pool_count": int(torch.tensor([membership[group] not in ordinary_pools
                        for group in self.scene_groups], device=device).sum()),
                    "active_request_eligible_count": 0}
            if (bool(config.get("integrated_runtime_candidate")) and not config.get("evaluation_exact_cases")):
                owners, settings = self._component_configuration("nonstandard_start")
                self.nonstandard_start.configure_components(owners, settings)
                eligible &= self.nonstandard_start.component_eligible
            self.nonstandard_start_eligible = eligible
            self.startup_report["nonstandard_start"] = self.nonstandard_start.runtime_state(self.policy_dt)
        if "ordinary_posture_curriculum" in config:
            from wheeled_tasks.chassis.tasks.terrain.ordinary_posture_curriculum import OrdinaryPostureCurriculum, ordinary_eligibility
            posture_evaluation = bool(config.get("evaluation_exact_cases"))
            if config["ordinary_posture_curriculum"]["enabled"] and not posture_evaluation:
                eligible = ordinary_eligibility(config, self.scene_groups, self.kinds, device,
                    recovery=self.recovery.enabled if self.recovery is not None else None,
                    spin=self.skills.spin if self.skills is not None else None)
            else:
                eligible = torch.zeros(num_envs, device=device, dtype=torch.bool)
            self.ordinary_posture = OrdinaryPostureCurriculum(num_envs, device,
                config["ordinary_posture_curriculum"], eligible,
                evaluation=posture_evaluation)
            if (bool(config.get("integrated_runtime_candidate")) and not config.get("evaluation_exact_cases")):
                owners, settings = self._component_configuration("ordinary_posture_curriculum")
                self.ordinary_posture.configure_components(owners, settings)
                eligible = self.ordinary_posture.eligible
            if self.ordinary_posture.enabled:
                if self.nonstandard_start is not None:
                    self.nonstandard_start_eligible &= eligible
                self.ordinary_posture.bind_geometry(root / config["asset_directory"], manifest,
                    self.robot.joint_names, self.nonstandard_start)
        if "release_stop" in config:
            self.release_stop_terrain = torch.tensor([
                kind != "flat" and config.get("skill_specs", {}).get(group, {}).get("kind") == "release_stop"
                for group, kind in zip(self.scene_groups, self.kinds)], device=device, dtype=torch.bool)
        if self.skills is not None and self.skills.heading is not None:
            self.skills.heading.validate_owners(self.skills.spin,
                self.recovery.enabled if self.recovery is not None else None,
                self.step_jump.enabled if self.step_jump is not None else None,
                self.step_climb.enabled if self.step_climb is not None else None)
            self.startup_report["scut_heading_command"] = self.skills.heading.summary()
        if self.reward_selection.get("scut_source_selected", self.reward_selection.get("scut_flat_selected", False)):
            from wheeled_tasks.chassis.simulation.runtime.flat_reward_runtime import ScutFlatRewardRuntime
            if self.rehearsal_spec is not None:
                from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import SOURCE_VERSION
                if "flat_lane_mask" in self.rehearsal_spec:
                    self.source_flat_rehearsal_mask = torch.tensor(
                        self.rehearsal_spec["flat_lane_mask"], dtype=torch.bool, device=device)
                elif self.rehearsal_spec["settings"]["version"] == SOURCE_VERSION:
                    self.source_flat_rehearsal_mask = torch.tensor(
                        [index >= 0 for index in self.rehearsal_spec["case_index"]],
                        dtype=torch.bool, device=device)
            if config["reward_profile"] == "v7_final_rough_v1":
                from wheeled_tasks.chassis.simulation.runtime.final_rough_reward_runtime import V7FinalRoughRewardRuntime
                self.scut_reward = V7FinalRoughRewardRuntime(num_envs, device, self.robot.body_names,
                    config["scut_reward_mapping"], model_spec=self.model_spec, manifest=manifest,
                    terrain_source=config["formal_terrain"]["source"],
                    exact_commands=(config.get("formal_small_rough_mix", {}).get("version")
                        == FINAL_MOTION_VERSION),
                    native_joint_names=self.robot.joint_names,
                    wheel_body_names=tuple(self.robot.body_names[index] for index in self.wheel_ids),
                    physics_dt_s=self.dt,
                    flat_rehearsal_reward_profile=(self.rehearsal_spec["settings"]["reward_profile"]
                        if self.source_flat_rehearsal_mask is not None else "v7_final_rough_v1"),
                    **({"command_decoupling": config["formal_command_decoupling"]}
                       if "formal_command_decoupling" in config else {}),
                    **({"flat_rehearsal_mask": self.source_flat_rehearsal_mask}
                       if self.source_flat_rehearsal_mask is not None else {}))
            else:
                self.scut_reward = ScutFlatRewardRuntime(num_envs, device, self.robot.body_names,
                    config["scut_reward_mapping"], reward_profile=config["reward_profile"],
                    physics_dt_s=self.dt,
                    tracking_reward_repair=config.get("formal_tracking_reward_repair"),
                    **({"command_decoupling": config["formal_command_decoupling"]}
                       if "formal_command_decoupling" in config else {}),
                    **({"flat_rehearsal_mask": self.source_flat_rehearsal_mask,
                        "flat_rehearsal_reward_profile": self.rehearsal_spec["settings"]["reward_profile"]}
                       if self.source_flat_rehearsal_mask is not None else {}))
            self.startup_report["scut_flat_reward"] = self.scut_reward.summary()
            self.closure_gap_peak = torch.zeros(num_envs, device=device)
        self.rough_birth_support = None
        if "formal_stepup_command" in config:
            from wheeled_tasks.chassis.tasks.commands.stepup_command import StepUpCommand, validate_stepup_command
            self.v7_stepup_settings = validate_stepup_command(config["formal_stepup_command"])
            if self.v7_stepup_settings and self.v7_stepup_settings["enabled"]:
                self.v7_stepup = StepUpCommand(num_envs, device, self.v7_stepup_settings)
            self.startup_report["v7_stepup_command"] = {
                "settings": self.v7_stepup_settings, "event_source": "external_step_wall_events",
                "hold_clock": "one integer20ms policy tick; no observation-read advancement",
                "detector_connected": False, "step_skill_qualified": False}
        if "flat_episode_runtime" in config:
            from wheeled_tasks.chassis.application.runtime.flat_episode_runtime import build_flat_episode_runtime
            self.flat_episode = build_flat_episode_runtime(config, manifest, self.model_spec,
                root, num_envs, device, **({"curriculum": self.curriculum_controller}
                    if self.curriculum_controller is not None else {}),
                **({"evaluation_setup": evaluation_setup}
                    if evaluation_setup is not None and self.evaluation_surface == "plane_regression_v1" else {}),
                **({"replay_setup": replay_setup} if replay_setup is not None else {}),
                **({"rough_terrain": self.shared_terrain, "environment_origins": self.origins}
                    if config.get("formal_terrain", {}).get("type") == "rough"
                    and self.evaluation_surface != "plane_regression_v1" else {}))
            if getattr(self.flat_episode, "final_source_birth_enabled", False):
                self.startup_report["final_rough_birth_adapter"] = self.flat_episode.final_birth_summary()
            if (config.get("formal_terrain", {}).get("type") == "rough"
                    and self.evaluation_surface != "plane_regression_v1"):
                from wheeled_tasks.chassis.simulation.terrain.rough_support import RoughBirthSupport
                self.rough_birth_support = RoughBirthSupport(self.flat_episode.placement, self.shared_terrain)
                self.startup_report["rough_birth_support"] = self.rough_birth_support.summary()
            if config.get("formal_terrain", {}).get("type") == "rough":
                self.startup_report["rough_base_height_scan"] = {
                    "size_m": [.02, .02], "spacing_m": .01, "ray_count": 9,
                    "alignment": "yaw", "ray_origin_above_root_m": 5.,
                    "reduction": "finite_hit_world_z_mean", "all_miss_fallback": "environment_origin_world_z",
                    "critic_height_reference": "root_link_world_z", "actor_frame_changed": False}
            self.flat_tree_to_native = [self.flat_episode.tree_joint_names.index(name)
                                       for name in self.robot.joint_names]
            if self.rehearsal_spec is not None:
                if "small_rough_mix" in self.rehearsal_spec:
                    from wheeled_tasks.chassis.application.configuration.small_rough_course import install
                    source_clock = install(self.rehearsal_spec, self.flat_episode.commands,
                        recovery_mask=self.recovery.enabled if self.recovery is not None else None)
                else:
                    from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import install_fixed_flat_rehearsal
                    source_clock = install_fixed_flat_rehearsal(self.rehearsal_spec, self.flat_episode.commands,
                        recovery_mask=self.recovery.enabled if self.recovery is not None else None,
                        **({"source_flat_iteration_offset": config.get("global_actor_update_offset")}
                            if "scripted_plane_lane_mask" in self.rehearsal_spec else {}))
                if source_clock is not None:
                    self.startup_report["rough_recovery_source_command_clock"] = source_clock
            if "formal_dynamic_nonsteady_birth" in config:
                self._configure_dynamic_birth(training=role == "train" and evaluation_setup is None
                                              and replay_setup is None)
                self.startup_report["formal_dynamic_nonsteady_birth"] = self._dynamic_birth_runtime_state()
            if "flat_interval_events" in config:
                from wheeled_tasks.chassis.simulation.randomization.flat_interval_events import FlatIntervalEvents
                self.flat_events = FlatIntervalEvents(num_envs, device, self.generator)
                self.flat_event_base_id = self.robot.body_names.index("base_link")
                self.startup_report["flat_interval_events"] = {
                    "version": config["flat_interval_events"]["version"], "interval_s": [5., 10.],
                    "push_world_COM_velocity_increment_xy_m_s": [-.25, .25],
                    "base_force_body_n_each_axis": [-10., 10.], "base_torque_body_nm_each_axis": [-1., 1.],
                    "clock": "continuous_policy_ticks_across_episode_resets",
                    "application": "post_payment_post_reset_before_returned_observations",
                    "wrench": "permanent_body_local_until_replaced_or_reset",
                    "qualification": "pending_native_experiment"}
        self.pc_parameters = self.critic_material = self.critic_contact = None
        if self.critic78:
            from wheeled_tasks.chassis.control.pc_parameters import PcParameters
            from wheeled_tasks.chassis.simulation.runtime.privileged_runtime import WheelMaterialCache, WheelContactHistory
            from wheeled_tasks.chassis.observations.critic_observation import critic_joint_indices
            if self.usb_transport is not None:
                raise ValueError("78D age provenance does not support legacy aggregate RTT transport")
            if self.signal_delay is not None and self._pc_feedback_transport is not None:
                raise ValueError("78D sensor age cannot double-count two feedback transports")
            self.pc_parameters = PcParameters(num_envs, device, config["v5_control"],
                self.v5.wheel_prior["kd"], config.get("pc_relative_priors", {"enabled": False}), self.generator)
            self.v5.set_pc_parameters(self.pc_parameters.kp, self.pc_parameters.kd, self.pc_parameters.strength)
            self.critic_material = WheelMaterialCache(self.robot, view, device)
            if config.get("wheel_material_priors", {}).get("enabled"):
                from wheeled_tasks.chassis.simulation.runtime.privileged_runtime import apply_startup_wheel_material_buckets
                if config.get("friction_combine_mode") != "multiply":
                    raise ValueError("Wheel priors require explicit multiply friction combination")
                buckets = apply_startup_wheel_material_buckets(self.critic_material, seed=seed + 6149,
                    legacy_ground_randomization_enabled=self.contact_domain is not None,
                    version=config["wheel_material_priors"]["version"])
                self.startup_report["wheel_material_priors"] = {
                    "version": buckets["version"], "num_buckets": 64,
                    "native_readback_verified": True,
                    "bucket_ids": buckets["bucket_ids"].tolist(),
                    "coverage": {key: value for key, value in buckets.items()
                        if key not in ("version", "buckets", "bucket_ids")}}
            self.critic_contact = WheelContactHistory(num_envs, device, physics_dt_s=self.dt)
            self.critic_net_contact_force = torch.zeros(num_envs, self.body_count, 3, device=device)
            self.critic_axis_from_control = critic_joint_indices(self.v5.ACTIVE)
            self.critic_applied_torque = torch.zeros(num_envs, 6, device=device)
            self.critic_applied_at_s = torch.zeros(num_envs, device=device, dtype=torch.float64)
            self.critic_pd_origin_s = torch.zeros(num_envs, 2, device=device, dtype=torch.float64)
            if self.policy_signal_delay is not None:
                self.critic_target_sample_s = torch.zeros_like(self.critic_pd_origin_s)
            self.policy_generated_s = torch.zeros(num_envs, device=device, dtype=torch.float64)
            self.closure_gap_peak = torch.zeros(num_envs, device=device)
            self.startup_report["critic78"] = {"version": "v6_scut78_v1",
                "root_velocity_reference": "base_link_origin_body_frame",
                "root_height_reference": ("root_link_world_z"
                    if config.get("critic_height_reference") == "root_link_world_z"
                    else "root_world_z_minus_environment_origin_z"),
                "motor_order": list(__import__("wheeled_tasks.chassis.observations.critic_observation", fromlist=["JOINT_NAMES"]).JOINT_NAMES),
                "contact_history": {"samples": len(self.critic_contact.history), "dt_s": self.dt, "threshold_n": 5.,
                    "force_source": "native_net_contact_forces_all_partners"},
                "action_age_origin": ("selected_policy_target_sample_time" if self.policy_signal_delay is not None
                    else "original_pc_pd_computation_time"),
                "action_age_effective_time": "physical_interval_end_minus_duration_mean_origin",
                "age_tick_s": .005, "wheel_material": "native_shape_readback"}
        if evaluation_setup is not None:
            evaluation_setup.apply_before_first_reset(self)
        if replay_setup is not None:
            replay_setup.apply_before_first_reset(self)
        self._initialize_jump_actuation_measurement()
        self._initialize_jump_physics_trace()
        self.reset(torch.arange(num_envs, device=device))
        if self.round_trip_transport is not None:
            self.startup_report["round_trip_transport"] = {
                "configuration": dict(config["round_trip_transport"]),
                "feedback_and_PD_share_cached_sensor_timestamp": True,
                "command_delay_includes_CAN": True,
                "additional_CAN_phase_s": 0.,
                "path_split_is_measured": False,
                "metrics": {key: float(value) for key, value in self.round_trip_transport.metrics().items()}}
        if self.critic78:
            self._critic78_validate_once = True
            self.get_observations()
            self._critic78_validate_once = False
        self.startup_report["control_timing"] = {
            "physics_dt": self.dt, "policy_dt": self.policy_dt, "decimation": self.decimation,
            "pc_control_dt": self.dt, "target_update": "once_per_policy_step",
            "feedback_control": "after_each_physical_state_refresh",
            "runtime_substep_verification": "pending_first_step"}
        if self.design_preflight is not None:
            actual = {"physics_dt": self.sim.cfg.dt, "policy_dt": self.policy_dt,
                "decimation": self.decimation, "pc_control_dt": self.dt,
                "seed": self.generator.initial_seed(),
                "leg_kp": self.v5.settings["leg_kp"], "leg_kd": self.v5.settings["leg_kd"],
                "wheel_velocity_p": self.v5.wheel_prior["kd"],
                "modules": {"usb_transport": self.usb_transport is not None,
                    "command_transport": self.command_transport is not None,
                    "signal_perturbations": self.perturbations is not None,
                    "dynamics_randomization": self.dynamics_randomization is not None,
                    "contact_domain": self.contact_domain is not None,
                    "step_assist": self.step_assist is not None},
                "precision_tracking": self.references is not None and bool(config["precision_tracking"])}
            if config.get("reward_profile") == "v7_final_rough_v1":
                from wheeled_tasks.chassis.simulation.runtime.final_rough_reward_runtime import V7FinalRoughRewardRuntime
                actual["modules"]["v7_final_rough_reward"] = isinstance(self.scut_reward, V7FinalRoughRewardRuntime)
                actual["v7_final_rough_reward"] = (self.scut_reward.summary()
                    if actual["modules"]["v7_final_rough_reward"] else None)
            if config.get("new_asset_curriculum", {}).get("enabled"):
                actual["modules"].update(jump_assist=self.jump_assist is not None,
                    training_schedule=self.skills is not None and self.skills.schedule is not None)
                actual.update(dense_tracking=bool(config.get("dense_tracking", {}).get("enabled")),
                    command_reference=self.references is not None, precision_tracking=False)
            if "actuator_response" in config:
                actual["modules"]["actuator_response"] = self.actuator_response.enabled
                actual["actuator_response"] = self.actuator_response.cfg
            if "signal_delay" in config:
                actual["modules"]["signal_delay"] = self.signal_delay is not None
                actual["signal_delay"] = dict(config["signal_delay"])
            if "actor_sensor_noise" in config:
                actual["modules"]["actor_sensor_noise"] = self.actor_sensor_noise is not None
                actual["actor_sensor_noise"] = dict(self.actor_sensor_noise.cfg)
            if "policy_signal_delay" in config:
                actual["modules"]["policy_signal_delay"] = self.policy_signal_delay is not None
                actual["policy_signal_delay"] = dict(self.policy_signal_delay.cfg)
            if "episode_spring_force_randomization" in config:
                actual["modules"]["episode_spring_force_randomization"] = self.episode_spring_strength_settings is not None
                actual["episode_spring_force_randomization"] = self.episode_spring_strength_settings
            if "formal_command_decoupling" in config:
                actual["formal_command_decoupling"] = self.scut_reward.command_decoupling
                actual["modules"]["formal_command_decoupling"] = (
                    self.scut_reward.command_decoupling is not None
                    and self.scut_reward.command_decoupling["enabled"])
            if "formal_stepup_command" in config:
                actual["formal_stepup_command"] = self.v7_stepup_settings
                actual["modules"]["formal_stepup_command"] = self.v7_stepup is not None
            if "formal_dynamic_nonsteady_birth" in config:
                actual["formal_dynamic_nonsteady_birth"] = self._dynamic_birth_runtime_state()
                actual["modules"]["formal_dynamic_nonsteady_birth"] = (
                    actual["formal_dynamic_nonsteady_birth"]["enabled"])
            if "rough_birth_anchor" in self.startup_report:
                actual["rough_birth_anchor"] = dict(self.startup_report["rough_birth_anchor"])
            if "feedback_transport" in config:
                actual["modules"]["feedback_transport"] = self.feedback_transport is not None
                actual["feedback_transport"] = dict(config["feedback_transport"])
                actual["feedback_transport_model"] = (config["feedback_transport"]["version"]
                    if self.feedback_transport is not None else None)
                actual["feedback_consumers"] = "policy_pc_pd_recovery"
            if self.round_trip_transport is not None:
                actual["modules"]["round_trip_transport"] = True
                actual["round_trip_transport"] = dict(config["round_trip_transport"])
            quality = {key: dict(config["dense_tracking"][key])
                for key in ("stand_still_lin_vel", "locomotion_posture")
                if key in config.get("dense_tracking", {})}
            if quality:
                actual["modules"].update(stand_still_lin_vel="stand_still_lin_vel" in quality,
                    locomotion_posture_floor="locomotion_posture" in quality and self.references is not None)
                actual["locomotion_quality_rewards"] = quality
                self.startup_report["locomotion_quality_rewards"] = quality
            if config.get("command_transport", {}).get("delivery_model"):
                actual["command_transport_model"] = (self.command_transport.delivery_model
                    if self.command_transport is not None else None)
            if config.get("step_jump_task", {}).get("enabled"):
                actual["modules"].update(step_jump_task=self.step_jump is not None,
                    step_climb_task=self.step_climb is not None)
            if config.get("recovery_training", {}).get("enabled"):
                actual["modules"]["recovery_training"] = self.recovery is not None
                actual["recovery_pose_curriculum"] = (dict(self.recovery.pose_curriculum.settings)
                    if self.recovery.pose_curriculum is not None else None)
                actual["recovery_complete_started_blend"] = self.recovery.fsm.config["complete_started_blend"]
                if self.recovery.source_flat_handoff:
                    actual["source_flat_handoff"] = self.recovery.settings["source_flat_handoff"]
                if self.recovery.episode_window is not None:
                    actual["recovery_episode_window"] = dict(self.recovery.episode_window)
                if "prefix_curriculum" in config["recovery_training"]:
                    from wheeled_tasks.chassis.tasks.curricula.recovery_prefix_curriculum import runtime_expectation
                    prefix = self.recovery.pose_curriculum
                    enabled = prefix is not None and not config.get("evaluation_exact_cases", False)
                    actual["modules"]["recovery_prefix_curriculum"] = enabled
                    actual["recovery_prefix_curriculum"] = (prefix.runtime_state() if enabled else
                        runtime_expectation(config["recovery_training"]["prefix_curriculum"], enabled=False))
                    self.startup_report["recovery_prefix_curriculum"] = actual["recovery_prefix_curriculum"]
            if self.recovery_hold_cost_settings is not None:
                actual["modules"]["recovery_hold_cost"] = self.recovery_hold_cost_settings["enabled"]
                actual["recovery_hold_cost"] = self.recovery_hold_cost_settings
                self.startup_report["recovery_hold_cost"] = dict(self.recovery_hold_cost_settings)
                if self.recovery_hold_cost_settings["enabled"]:
                    actual["recovery_hold_cost_gate_binding"] = self.recovery_hold_cost_gate_binding
                    self.startup_report["recovery_hold_cost_gate_binding"] = self.recovery_hold_cost_gate_binding
            if config.get("task_campaign", {}).get("revision_migration", {}).get("schema") == "task-worker-revision-v20":
                from wheeled_tasks.chassis.application.compatibility.recovery_script_revision import native_runtime_binding
                actual["recovery_script_revision"] = native_runtime_binding(self.recovery.fsm, config)
                self.startup_report["recovery_script_revision"] = actual["recovery_script_revision"]
            if "recovery_takeover" in config:
                actual["recovery_takeover"] = dict(config["recovery_takeover"])
            if (config.get("landing_tracking") or {}).get("passive_semantics") == "finite_task_cost_v1":
                actual["finite_passive_landing"] = {
                    "enabled": self.passive_landing is not None,
                    "window_seconds": self.passive_landing.duration if self.passive_landing is not None else None,
                    "policy_dt": self.passive_landing.dt if self.passive_landing is not None else None}
            if "start_stop_start_delay_seconds" in config.get("new_asset_curriculum", {}):
                actual["start_stop_timing"] = self.skills.start_stop_timing_state()
            if "locomotion_difficulty" in config:
                actual["locomotion_difficulty"] = self.skills.difficulty_runtime_state()
                self.startup_report["locomotion_difficulty_probe"] = self.skills.verify_difficulty_samples()
            if "jump_landing" in config:
                actual["modules"]["jump_landing"] = self.jump_landing is not None
                actual["jump_landing"] = self.jump_landing.runtime_state()
            if "jump_cycle_reset_on_accepted" in config:
                actual["jump_cycle_reset_on_accepted"] = self.jump_cycle_reset_on_accepted
            if "nonstandard_start" in config:
                actual["modules"]["nonstandard_start"] = self.nonstandard_start.enabled
                actual["nonstandard_start"] = (self.nonstandard_start.runtime_state(self.policy_dt)
                    if self.nonstandard_start is not None else None)
            if "ordinary_posture_curriculum" in config:
                actual["modules"]["ordinary_posture_curriculum"] = self.ordinary_posture.enabled
                actual["ordinary_posture_curriculum"] = self.ordinary_posture.runtime_state()
                actual["ordinary_posture_curriculum"]["runtime_observed"] = self.ordinary_posture.observed(
                    (self._curriculum_clock("ordinary_posture_curriculum") if self.cfg.get("integrated_runtime_candidate") and not self.cfg.get("evaluation_exact_cases") else stage_actor_updates(self)))
                self.startup_report["ordinary_posture_curriculum"] = actual["ordinary_posture_curriculum"]
            if "terrain_route_runout" in config:
                from wheeled_tasks.chassis.application.compatibility.terrain_runout_runtime import observe_runtime
                actual["modules"]["terrain_route_runout"] = self.route_runout_settings is not None
                actual["terrain_route_runout"] = observe_runtime(self)
                self.startup_report["terrain_route_runout"] = actual["terrain_route_runout"]
            if "release_stop" in config:
                actual["modules"]["release_stop"] = self.skills.release_stop.enabled
                actual["release_stop"] = self.skills.release_stop.runtime_state()
                self.startup_report["release_stop"] = actual["release_stop"]
            self.startup_report["design_preflight"] = self.design_preflight
            if self.integrated_runtime_enabled:
                from wheeled_tasks.chassis.application.compatibility.integrated_preflight import observe_component_clocks, validate_integrated_runtime_state
                actual["integrated_curriculum_observed"] = observe_component_clocks(self)
                self.startup_report["runtime_preflight"] = validate_integrated_runtime_state(
                    self.design_preflight["runtime_expected"], actual, config=self.cfg,
                    num_envs=self.num_envs, actor_update=stage_actor_updates(self))
            else:
                self.startup_report["runtime_preflight"] = validate_runtime_state(
                    self.design_preflight["runtime_expected"], actual)

    def _configure_rough_birth_anchor(self):
        """Keep continuous rough births on their own tile's central platform."""
        from wheeled_tasks.chassis.application.configuration.small_rough_course import ROLES, ROUGH_ANCHOR_KEY, ROUGH_ANCHOR_VERSION
        if ROUGH_ANCHOR_KEY not in self.cfg:
            return
        if (self.cfg[ROUGH_ANCHOR_KEY] != ROUGH_ANCHOR_VERSION
                or self.rehearsal_spec is None
                or "small_rough_mix" not in self.rehearsal_spec):
            raise ValueError("Centered rough births require the actual mixed-course allocation")
        roles = torch.tensor(self.rehearsal_spec["population_index"], device=self.origins.device)
        selected = roles == ROLES.index("source_rough")
        rough = torch.tensor([kind == "rough" for kind in self.kinds], device=self.origins.device)
        if not selected.any() or not torch.equal(selected, rough):
            raise ValueError("Centered rough births differ from actual native rough ownership")
        self.reset_positions[selected, :2] = self.origins[selected, :2]
        delta = self.reset_positions[selected, :2] - self.origins[selected, :2]
        self.startup_report["rough_birth_anchor"] = {
            "version": ROUGH_ANCHOR_VERSION, "rough_lanes": int(selected.sum()),
            "x_offset_m": 0., "y_offset_m": 0., "maximum_xy_error_m": float(delta.abs().max())}

    def _configure_dynamic_birth(self, *, training):
        """Bind birth ownership after fixed requests and native lane reordering."""
        from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import validate_dynamic_nonsteady_birth
        if not validate_dynamic_nonsteady_birth(self.cfg) or self.flat_episode is None:
            raise ValueError("Dynamic births require the constructed flat episode owner")
        fixed = self.flat_episode.commands.fixed_request_mask
        if (not isinstance(fixed, torch.Tensor) or fixed.dtype != torch.bool
                or fixed.shape != (self.num_envs,) or fixed.device != self.flat_episode.device):
            raise ValueError("Dynamic births require valid actual fixed lane membership")
        recovery = getattr(self, "recovery", None)
        scripted = (recovery.enabled if recovery is not None else torch.zeros_like(fixed))
        if (not isinstance(scripted, torch.Tensor) or scripted.dtype != torch.bool
                or scripted.shape != fixed.shape or scripted.device != fixed.device):
            raise ValueError("Dynamic births require valid actual recovery lane membership")
        if "formal_small_rough_mix" in self.cfg:
            from wheeled_tasks.chassis.application.configuration.small_rough_course import ROLES, ROUGH_BIRTH_KEY
            if not training or self.rehearsal_spec is None:
                self._dynamic_birth_nominal_mask = torch.ones_like(fixed)
            else:
                roles = torch.tensor(self.rehearsal_spec["population_index"], device=fixed.device)
                if not torch.equal(scripted, roles == ROLES.index("recovery")):
                    raise ValueError("Mixed dynamic births differ from actual recovery membership")
                self._dynamic_birth_nominal_mask = roles != ROLES.index("dynamic_flat")
                if ROUGH_BIRTH_KEY in self.cfg:
                    self._dynamic_birth_nominal_mask &= roles != ROLES.index("source_rough")
            self._dynamic_birth_source_mask = ~self._dynamic_birth_nominal_mask & ~scripted
            return
        mix = None
        if "formal_scripted_recovery_mix" in self.cfg:
            from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import validate_scripted_flat_mix
            mix = validate_scripted_flat_mix(self.cfg)
            expected = torch.tensor([group in mix["profiles_by_scene"] for group in self.scene_groups],
                                    dtype=torch.bool, device=fixed.device)
            if (recovery is None or not recovery.source_flat_handoff
                    or not torch.equal(scripted, expected) or bool((fixed & scripted).any())):
                raise ValueError("Scripted mix requires disjoint actual fixed and native recovery lanes")
        elif bool(scripted.any()):
            raise ValueError("Scripted dynamic births require the explicit scripted mix selector")
        if training:
            allocated = torch.tensor([case >= 0 for case in self.rehearsal_spec["case_index"]],
                                     dtype=torch.bool, device=self.device)
            if not torch.equal(fixed, allocated) or int(fixed.sum()) * 4 != self.num_envs:
                raise ValueError("Dynamic births require the actual fixed quarter of rehearsal lanes")
            self._dynamic_birth_nominal_mask = fixed.clone()
        else:
            # Existing independent measurements retain their nominal starts.
            # They do not qualify recovery from nonstandard births.
            self._dynamic_birth_nominal_mask = torch.ones_like(fixed)
        self._dynamic_birth_source_mask = ~self._dynamic_birth_nominal_mask & ~scripted

    def _dynamic_birth_runtime_state(self):
        from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import DYNAMIC_NONSTEADY_BIRTH_VERSION
        mask = self._dynamic_birth_nominal_mask
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != (self.num_envs,) or mask.device != self.flat_episode.device):
            raise ValueError("Dynamic birth ownership has no valid runtime lane binding")
        source = self._dynamic_birth_source_mask
        if (not isinstance(source, torch.Tensor) or source.dtype != torch.bool
                or source.shape != mask.shape or source.device != mask.device):
            raise ValueError("Dynamic birth source ownership has no valid runtime lane binding")
        recovery = getattr(self, "recovery", None)
        scripted = recovery.enabled if recovery is not None else torch.zeros_like(mask)
        if (not isinstance(scripted, torch.Tensor) or scripted.dtype != torch.bool
                or scripted.shape != mask.shape or scripted.device != mask.device
                or not torch.equal(source, ~mask & ~scripted)):
            raise ValueError("Dynamic birth source ownership differs from actual native membership")
        nominal_count = int(mask.sum())
        source_count = self.num_envs - nominal_count
        if "formal_small_rough_mix" in self.cfg:
            from wheeled_tasks.chassis.application.configuration.small_rough_course import ROLES, ROUGH_BIRTH_KEY
            result = {"version": DYNAMIC_NONSTEADY_BIRTH_VERSION, "enabled": bool(source.any()),
                    "nominal_lanes": int((mask & ~scripted).sum()), "source_lanes": int(source.sum()),
                    "scripted_lanes": int(scripted.sum()),
                    "scope": "dynamic flat only; fixed, temporal and rough births remain nominal"}
            if ROUGH_BIRTH_KEY in self.cfg:
                flat_count = rough_count = 0
                if bool(source.any()):
                    roles = torch.tensor(self.rehearsal_spec["population_index"], device=source.device)
                    flat_count = int((source & (roles == ROLES.index("dynamic_flat"))).sum())
                    rough_count = int((source & (roles == ROLES.index("source_rough"))).sum())
                result.update(flat_source_lanes=flat_count, rough_source_lanes=rough_count,
                    scope="continuous flat and rough source births; fixed and temporal remain nominal")
            return result
        if "formal_scripted_recovery_mix" in self.cfg:
            nominal_count = int((mask & ~scripted).sum())
            source_count = int(source.sum())
            return {"version": DYNAMIC_NONSTEADY_BIRTH_VERSION, "enabled": source_count > 0,
                    "nominal_lanes": nominal_count, "source_lanes": source_count,
                    "measurement_nominal_override": source_count == 0,
                    "scripted_recovery_mix": self.cfg["formal_scripted_recovery_mix"],
                    "fixed_nominal_lanes": nominal_count, "ordinary_dynamic_lanes": source_count,
                    "scripted_lanes": int(scripted.sum())}
        return {"version": DYNAMIC_NONSTEADY_BIRTH_VERSION, "enabled": source_count > 0,
                "nominal_lanes": nominal_count, "source_lanes": source_count,
                "measurement_nominal_override": source_count == 0}

    def _has_integrated_runtime(self):
        return bool(self.cfg.get("integrated_runtime_candidate")) and not self.cfg.get("evaluation_exact_cases")

    @property
    def _pc_feedback_transport(self):
        return getattr(self, "feedback_transport", None) or self.usb_transport

    def _component_configuration(self, field):
        """Startup-only binding; components share existing device state and geometry."""
        from wheeled_tasks.chassis.application.compatibility.integrated_runtime import row_settings
        audit = self.cfg["integrated_runtime_candidate"]
        key = {"ordinary_posture_curriculum": "posture_component", "nonstandard_start": "nonstandard_component",
               "jump_assist": "request_component"}[field]
        owners = [audit["row_ownership"][group][key] for group in self.scene_groups]
        settings = {}
        for group in set(self.scene_groups):
            owner = audit["row_ownership"][group][key]
            if owner is not None:
                value, _ = row_settings(self, group, field)
                if value is not None:
                    settings[owner] = value
        owners = [owner if owner in settings else None for owner in owners]
        return owners, settings

    def _curriculum_clock(self, field):
        if not getattr(self, "integrated_runtime_enabled", False):
            return stage_actor_updates(self)
        from wheeled_tasks.chassis.application.compatibility.integrated_runtime import component_actor_updates
        module = getattr(self, {"ordinary_posture_curriculum": "ordinary_posture",
                  "nonstandard_start": "nonstandard_start", "jump_assist": "jump_assist"}[field])
        return {name: component_actor_updates(self, name) for name in module.component_settings}

    def random(self, count):
        return torch.rand(count, device=self.device, generator=self.generator)

    def _init_body_support_contacts(self):
        """Only normal climbing may use chassis support, proved by real patches."""
        self.base_support_views = []
        self.base_support_id = self.robot.body_names.index("base_link")
        if not self.step_climb.has_tasks:
            return
        view = PhysxManager.get_physics_sim_view()
        for size in sorted({len(s) for s in self.surfaces}):
            members = [i for i, surfaces in enumerate(self.surfaces)
                       if len(surfaces) == size and bool(self.step_climb.enabled[i])]
            if not members:
                continue
            paths = [self.env_paths[i] + "/Robot/base_link" for i in members]
            filters = [[self.env_paths[i] + f"/Terrain/surface_{j}/geometry/mesh"
                        for j in range(size)] for i in members]
            sensor = view.create_rigid_contact_view(paths, filter_patterns=filters,
                max_contact_data_count=32 * size * len(paths))
            if sensor.filter_count != size or sensor.sensor_count != len(paths):
                raise RuntimeError("Base support sensor/filter layout mismatch")
            self.base_support_views.append((torch.tensor(members, device=self.device), sensor))
        patches = 32 * self.surface_data.shape[1]
        self.base_support_points = torch.zeros(self.num_envs, patches, 3, device=self.device)
        self.base_support_forces = torch.zeros_like(self.base_support_points)
        self.base_patch_offsets = torch.arange(32, device=self.device)

    def _body_support_contacts(self):
        self.base_support_points.zero_()
        self.base_support_forces.zero_()
        for ids, sensor in self.base_support_views:
            force, point, normal, _, counts, starts = map(wp.to_torch, sensor.get_contact_data(self.dt))
            counts, starts = counts.long(), starts.long()
            if bool((counts > 32).any()):
                raise RuntimeError("Base support contact patch capacity exceeded")
            valid = self.base_patch_offsets < counts[..., None]
            index = (starts[..., None] + self.base_patch_offsets).clamp(0, len(force) - 1)
            points = point[index].reshape(len(ids), -1, 3) - self.origins[ids, None, :]
            forces = torch.where(valid[..., None], normal[index] * force[index], 0.).reshape(len(ids), -1, 3)
            self.base_support_points[ids, :points.shape[1]] = points
            self.base_support_forces[ids, :points.shape[1]] = forces
        return self.base_support_points, self.base_support_forces

    def _initialize_jump_actuation_measurement(self):
        """Bind raw physics evidence without adding a skill gate or payment."""
        if not self.cfg.get("jump_actuation_measurement", {}).get("enabled"):
            return
        import hashlib
        from wheeled_tasks.chassis.simulation.telemetry.jump_actuation_evidence import JumpActuationEvidence
        sources = {name: hashlib.sha256(module_source_path(name).read_bytes()).hexdigest()
            for name in ("env.py", "jump_actuation_evidence.py", "step_jump.py",
                         "training_schedule.py", "v5_control.py", "root_kinematics.py")}
        identity = {
            "configuration": self.cfg,
            "sources_sha256": sources,
            "control_joint_names": self.manifest["control_joint_names"],
            "spring_joint_names": self.manifest["spring_joint_names"],
            "wheel_body_names": [self.robot.body_names[index] for index in self.wheel_ids],
            "contact_source": "native_net_contact_forces_all_partners_after_physics_refresh",
            "effort_source": "ArticulationData.applied_torque_explicit_actuator_buffer_not_force_sensor",
            "spring_work_meaning": "generalized_actuator_output_including_damping_not_stored_potential_energy",
            "wheel_contact_threshold_n": 2.,
            "nonwheel_contact_threshold_n": 5.,
            "contact_threshold_purpose": "raw_measurement_labels_not_skill_acceptance",
            "COM_mass_weights": "cached_native_mass_readback_after_startup_randomization",
            "base_pose_quaternion_order": "xyzw",
            "external_intervention_observation": "not_observed_by_this_producer",
            "configured_jump_assist_enabled": self.jump_assist is not None,
            "counting_scope": "current_worker_only_not_resumed",
            "native_physical_exact_resume_qualified": False,
        }
        self.jump_actuation_measurement = JumpActuationEvidence(
            self.num_envs, self.device, physics_dt=self.dt,
            control_joint_names=self.manifest["control_joint_names"],
            spring_joint_names=self.manifest["spring_joint_names"], source_identity=identity)
        self.jump_net_contact_force = torch.zeros(self.num_envs, self.body_count, 3, device=self.device)
        self.startup_report["jump_actuation_measurement"] = self.jump_actuation_measurement.summary()

    def _configure_jump_physics_trace(self, config):
        """Opt-in raw evidence only; this cannot enable a skill or native role."""
        self.jump_physics_trace = None
        self.jump_physics_trace_enabled = False
        if "jump_physics_trace" not in config:
            return
        settings = config["jump_physics_trace"]
        if (not isinstance(settings, dict) or set(settings) != {"version", "enabled"}
                or settings["version"] != "jump_physics_trace_v1"
                or type(settings["enabled"]) is not bool):
            raise ValueError("Jump physics trace requires an explicit version and boolean")
        if settings["enabled"]:
            if (not self.is_v5 or not config.get("monitor_applied_effort")
                    or config["physics_dt"] != .005 or config["policy_dt"] != .02
                    or config.get("jump_actuation_measurement") != {
                        "version": "jump_actuation_measurement_v1", "enabled": True}):
                raise ValueError("Jump physics trace requires the existing native 200/50 Hz actuation measurement")
            if self.reward_selection.get("scut_source_selected", self.reward_selection.get("scut_flat_selected", False)) or "flat_episode_runtime" in config:
                raise ValueError("Jump physics trace has no source-flat episode adapter")
            self.jump_physics_trace_enabled = True

    def _validate_jump_trace_contact_view(self, contact_view, group_paths, group_filters):
        """Authenticate every grouped native contact view before physical reads."""
        if (contact_view.sensor_count != len(group_paths)
                or list(contact_view.sensor_paths) != group_paths):
            raise RuntimeError("Native contact sensor identity/order mismatch")
        actual_filters = [list(paths) for paths in contact_view.filter_paths]
        if actual_filters != group_filters:
            raise RuntimeError("Native terrain contact filter identity/order mismatch")

    def _initialize_jump_physics_trace(self):
        if not self.jump_physics_trace_enabled:
            return
        import hashlib
        import inspect
        from wheeled_tasks.chassis.simulation.telemetry.jump_physics_trace import JumpPhysicsTrace
        sources = {str(module_source_path(name)): hashlib.sha256(module_source_path(name).read_bytes()).hexdigest()
            for name in ("env.py", "jump_physics_trace.py", "jump_actuation_evidence.py",
                         "root_kinematics.py", "training_schedule.py", "skill_commands.py")}
        sdk_sources = {}
        for component in (type(self.robot), type(self.robot.instantaneous_wrench_composer),
                          *(type(view) for _, view in self.contact_views)):
            source = inspect.getsourcefile(component)
            if source is None:
                raise RuntimeError("Jump trace requires readable SDK dispatch source identity")
            path = Path(source).resolve()
            sdk_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups = [{"environment_ids": ids.detach().cpu().tolist(),
                   "sensor_paths": list(view.sensor_paths),
                   "filter_paths": [list(paths) for paths in view.filter_paths],
                   "shared_contact_data_capacity": view.max_contact_data_count}
                  for ids, view in self.contact_views]
        identity = {"configuration": self.cfg, "sources_sha256": sources,
            "SDK_sources_sha256": sdk_sources, "contact_view_groups": groups,
            "contact_counts_and_starts": "raw_indices_into_each_shared_contact_view_buffer_not_lane_local",
            "net_contact": "all_partners_net_vector_not_proof_of_contact_absence",
            "surface_contact": "own_terrain_filter_pairs_only_no_unfiltered_patch_count",
            "composer_inputs": "public_raw_buffers_before_native_dispatch_not_force_sensor",
            "velocity_write_source": "legacy_push_native_root_COM_world_read_write_read",
            "coverage": "current_ChassisEnv_known_callsites_source_flat_not_supported",
            "counting_scope": "current_worker_only_no_exact_resume",
            "qualification": "measurement_only_no_active_takeoff_skill_pass_or_PPO_credit"}
        self.jump_physics_trace = JumpPhysicsTrace(self.num_envs, self.device,
            body_names=self.robot.body_names, surface_counts=[len(s) for s in self.surfaces],
            physics_dt=self.dt, policy_dt=self.policy_dt, source_identity=identity)
        shape = (self.num_envs, self.body_count, max(map(len, self.surfaces)))
        self.jump_trace_surface_force = torch.zeros(*shape, 3, device=self.device)
        self.jump_trace_contact_counts = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.jump_trace_contact_starts = torch.zeros_like(self.jump_trace_contact_counts)
        self.jump_trace_contact_capacity = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for ids, view in self.contact_views:
            self.jump_trace_contact_capacity[ids] = view.max_contact_data_count
        self.startup_report["jump_physics_trace"] = self.jump_physics_trace.summary()

    def _prepare_jump_physics_trace(self):
        """Copy raw composer inputs without composing or changing native state."""
        trace = getattr(self, "jump_physics_trace", None)
        if trace is None:
            return
        fields = ("global_force_w", "global_torque_w", "global_force_at_com_w",
                  "local_force_b", "local_torque_b")
        inputs = {}
        for name in ("instantaneous", "permanent"):
            composer = getattr(self.robot, name + "_wrench_composer")
            inputs[name] = {"active": composer.active,
                **{field: wp.to_torch(getattr(composer, field)).clone() for field in fields}}
        trace.prepare_physics(physical_tick=self.physical_tick, composer_inputs=inputs)

    def _record_jump_trace_contacts(self, ids, contact_view, matrix):
        """Preserve raw filtered pairs/patch indices in the actual view order."""
        surfaces = contact_view.filter_count
        self.jump_trace_surface_force[ids, :, :surfaces] = matrix.reshape(len(ids), self.body_count, surfaces, 3)
        contact_data = contact_view.get_contact_data(dt=self.dt)
        counts, starts = (wp.to_torch(value).long() for value in contact_data[-2:])
        self.jump_trace_contact_counts[ids, :, :surfaces] = counts.reshape(len(ids), self.body_count, surfaces)
        self.jump_trace_contact_starts[ids, :, :surfaces] = starts.reshape(len(ids), self.body_count, surfaces)

    def _finish_jump_physics_trace(self):
        trace = getattr(self, "jump_physics_trace", None)
        if trace is None:
            return
        data, mass = self.robot.data, self.body_mass
        trace.finish_physics(physical_tick=self.physical_tick,
            body_net_force_n=self.jump_net_contact_force,
            surface_force_n=self.jump_trace_surface_force,
            surface_contact_counts=self.jump_trace_contact_counts,
            surface_contact_starts=self.jump_trace_contact_starts,
            contact_data_capacity=self.jump_trace_contact_capacity,
            com_position_world_m=(data.body_com_pose_w.torch[:, :, :3] * mass[:, :, None]).sum(1) / mass.sum(-1)[:, None],
            com_velocity_world_m_s=(data.body_com_lin_vel_w.torch * mass[:, :, None]).sum(1) / mass.sum(-1)[:, None],
            base_link_pose_world=data.root_link_pose_w.torch,
            wheel_centers_world_m=self.wheel_centers())

    def _observe_jump_actuation(self):
        measurement = getattr(self, "jump_actuation_measurement", None)
        if measurement is None:
            return
        data = self.robot.data
        mass = self.body_mass
        com_position = (data.body_com_pose_w.torch[:, :, :3] * mass[:, :, None]).sum(1) / mass.sum(-1)[:, None]
        com_velocity = (data.body_com_lin_vel_w.torch * mass[:, :, None]).sum(1) / mass.sum(-1)[:, None]
        forces = self.jump_net_contact_force.norm(dim=-1)
        measurement.observe_physics(physical_tick=self.physical_tick,
            applied_motor_nm=data.applied_torque.torch[:, self.ids],
            motor_velocity_rad_s=data.joint_vel.torch[:, self.ids],
            applied_spring_n=data.applied_torque.torch[:, self.spring_ids],
            spring_velocity_m_s=data.joint_vel.torch[:, self.spring_ids],
            base_link_pose_world=data.root_link_pose_w.torch,
            wheel_center_world=self.wheel_centers(),
            com_position_world=com_position, com_velocity_world=com_velocity,
            wheel_contacts=forces[:, self.wheel_ids] > 2.,
            nonwheel_contact=(forces[:, self.nonwheel_ids] > 5.).any(-1),
            contact_measurement_valid=torch.isfinite(self.jump_net_contact_force).all(dim=-1).all(dim=-1),
            legacy_phase=self.phase.phase, request_active=self.jump_requested)

    def _update_step_jump(self):
        if self.step_sensors is None:
            return
        _, omega, gravity, _, q, dq, local = self.state()
        if self._pc_feedback_transport is not None:
            q, dq = self._pc_feedback_transport.feedback[:, :6], self._pc_feedback_transport.feedback[:, 6:12]
            omega, gravity = (self._pc_feedback_transport.feedback[:, 12:15],
                self._pc_feedback_transport.feedback[:, 15:18])
        if self.recovery is not None:
            # Projection shares the PC feedback history without advancing it.
            q = self.recovery.project_feedback(q)
        time = self.episode_length_buf * self.policy_dt
        sensors = self.step_sensors
        wheel_speed = sensors.update(time, q, dq, omega, gravity,
            self.robot.data.root_link_pose_w.torch, root_link_linear_velocity(self),
            local, self.surface_data, self.surface_valid)
        settings = self.cfg["command_reference"]
        event = self.step_jump.control(time, wheel_speed, sensors.support, sensors.force_norm,
            sensors.range, sensors.range_valid, sensors.range_age,
            preload_seconds=settings["preload_seconds"], preload_depth=settings["preload_depth_m"],
            release_offset=settings["release_offset_m"], beam_forward_distance=sensors.forward_distance)
        accepted = event.nonzero(as_tuple=False).flatten()
        if len(accepted):
            measurement = getattr(self, "jump_actuation_measurement", None)
            if measurement is not None:
                measurement.accept_request(accepted, source="step_jump",
                    request_kind="step_jump",
                    source_request_ids=self.step_jump.request_count[accepted],
                    route_stage_ids=self.step_jump.index[accepted], physical_tick=self.physical_tick)
            self._reset_jump_evidence(accepted, local)

    def _reset_jump_evidence(self, ids, local):
        if self.jump_cycle_reset_on_accepted:
            self.full_tasks.reset_accepted(ids, local)
            if self.jump_landing is not None:
                self.jump_landing.reset_accepted(ids)
        else:
            self.full_tasks.reset(ids, local)
        self.phase.reset(ids)
        self.success_hold[ids] = 0.
        self.sequence_success_latched[ids] = False

    def ground_height(self, xy):
        if getattr(self, "shared_terrain", None) is not None:
            return self.shared_terrain.height(xy + self.origins[:, :2]) - self.origins[:, 2]
        # Same piecewise analytic surfaces used to create static collision boxes.
        x = xy[:, :1]
        s = self.surface_data
        inside = (x >= s[:, :, 0]) & (x <= s[:, :, 1]) & self.surface_valid
        heights = s[:, :, 2] + (x - s[:, :, 0]) * s[:, :, 3] + xy[:, 1:2] * s[:, :, 4]
        z = heights.masked_fill(~inside, -torch.inf).amax(-1)
        return torch.where(inside.any(-1), z, 0.)

    def wheel_centers(self):
        pose = self.robot.data.body_link_pose_w.torch[:, self.wheel_ids]
        offset = self.wheel_offsets.expand(self.num_envs, -1, -1)
        cross = torch.linalg.cross(pose[:, :, 3:6], offset)
        return pose[:, :, :3] + offset + 2 * (pose[:, :, 6:] * cross + torch.linalg.cross(pose[:, :, 3:6], cross))

    def update_height_assist(self, ids=None, *, advance=True):
        if self.step_assist is None:
            return
        root = self.robot.data.root_link_pose_w.torch
        x, y, z, w = root[:, 3:].unbind(-1)
        heading = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)), -1)
        direction = F.normalize(heading, dim=-1) * self.commands[:, :1].sign()
        xy = self.wheel_centers()[:, :, :2] - self.origins[:, None, :2]
        xy += direction[:, None, :] * self.cfg["step_assist"]["lookahead_m"]
        heights = torch.stack([self.ground_height(xy[:, side]) for side in range(2)], -1)
        valid = torch.stack([(((xy[:, side, :1] >= self.surface_data[:, :, 0])
            & (xy[:, side, :1] <= self.surface_data[:, :, 1]) & self.surface_valid).any(-1)
            & (xy[:, side, 1].abs() <= 2.)) for side in range(2)], -1)
        eligible = (self.mode != 4) & (self.commands[:, 0].abs() > .01)
        selected = self.step_assist.all_ids if ids is None else ids
        self.commands[selected, 2] = self.step_assist.update(self.raw_height, heights, valid,
            direction, eligible, self.step_assist_enabled & (self.mode != 4), ids=ids, advance=advance)

    def update_command_reference(self, ids=None):
        if self.references is None:
            return
        elapsed = self.episode_length_buf * self.policy_dt
        jumping = self.jump_requested & (self.mode == 4)
        apex = torch.as_tensor(self.full_tasks.cfg.get("jump_apex_delta_m", .06), device=self.device).expand(self.num_envs)
        jump_elapsed = elapsed - self.cfg["task_semantics"]["jump_request_seconds"]
        manual_mode = self.skills.manual_step_mode
        manual_elapsed = elapsed - self.skills.manual_step_start
        if self.skills.schedule.requests_enabled:
            schedule = self.skills.schedule
            jump_elapsed = torch.where(jumping, elapsed - schedule.request_start, 0.)
            manual_mode = manual_mode * schedule.request_active.long()
            manual_elapsed = torch.where(schedule.request_active, elapsed - schedule.request_start, 0.)
        landing_delta = None
        if self.step_jump is not None and self.step_jump.has_tasks:
            selected = self.step_jump.enabled
            jump_elapsed = torch.where(selected, elapsed - self.step_jump.request_start, jump_elapsed)
            apex = torch.where(selected, self.step_jump.apex, apex)
            landing_delta = self.step_jump.landing_delta
        effective = self.references.update(self.raw_height, self.command_acceleration,
            manual_mode, manual_elapsed,
            self.skills.manual_step_duration, jumping,
            jump_elapsed, apex, landing_delta=landing_delta)
        if ids is None:
            self.commands[:, 2] = effective
        else:
            # During auto-reset the other lanes already contain raw commands.
            # Reapplying their bias here would contaminate the next raw snapshot.
            self.commands[ids, 2] = effective[ids]

    def state(self):
        data = self.robot.data
        local = data.root_link_pose_w.torch[:, :3] - self.origins
        support = self.ground_height(local[:, :2])
        if self.cfg.get("height_reference") == "wheel_support_mean":
            xy = self.wheel_centers()[:, :, :2] - self.origins[:, None, :2]
            support = torch.stack([self.ground_height(xy[:, side]) for side in range(2)], -1).mean(-1)
        return (data.root_com_lin_vel_b.torch, data.root_com_ang_vel_b.torch,
                data.projected_gravity_b.torch, local[:, 2] - support,
                data.joint_pos.torch[:, self.ids], data.joint_vel.torch[:, self.ids], local)

    def _transport_sensors(self):
        data = self.robot.data
        return torch.cat((data.joint_pos.torch[:, self.ids], data.joint_vel.torch[:, self.ids],
                          data.root_com_ang_vel_b.torch, data.projected_gravity_b.torch), -1)

    def _configure_reward_episode_measurement(self, config):
        """Select measurement without selecting a new reward or skill adapter."""
        self.reward_episode_measurement_enabled = False
        if "reward_episode_measurement" not in config:
            return
        settings = config["reward_episode_measurement"]
        if (not isinstance(settings, dict) or set(settings) != {"version", "enabled"}
                or settings["version"] != "reward_episode_measurement_v1"
                or type(settings["enabled"]) is not bool):
            raise ValueError("Reward episode measurement requires the explicit v1 schema")
        if config.get("reward_profile") not in ("scut0719_flat_v1", "scut_v14_flat_v5", "scut0730_rough_v1",
                                               "v7_final_rough_v1"):
            raise ValueError("Reward episode measurement requires an explicit reference reward profile")
        if config.get("policy_dt") != .02:
            raise ValueError("Reward episode measurement requires the existing 50 Hz policy contract")
        self.reward_episode_measurement_enabled = settings["enabled"]

    def _capture_reward_episode_payment(self, extras, payments, done, terminated,
                                        timeouts, success, boundary):
        """Snapshot actual paid values and ownership before any auto-reset.

        These are physical environment payments, not timeout-augmented RSL
        storage rewards, value targets, or claims of successful optimization.
        No payment is reconstructed from a later component dictionary.
        """
        extras["final_reward_payments"] = {name: value.clone() for name, value in payments.items()}
        extras["reward_episode_transition"] = {
            "schema": "reward_episode_transition_v1",
            "payment_units": "already_dt_scaled_payment_per_policy_step",
            "policy_dt_s": self.policy_dt,
            "episode_ids": self.reward_episode_ids.clone(),
            "episode_ticks": self.episode_length_buf.clone(),
            "policy_interval_id": self.tick,
            "done": done.clone(), "terminated": terminated.clone(),
            "timeouts": timeouts.clone(), "success": success.clone(),
            "boundary": boundary.clone(),
            "policy_learning_mask": extras.get("policy_learning_mask", torch.ones_like(done)).clone()}

    def _finish_reward_episode_payment(self, extras):
        """Attach returned identities without changing the paid terminal frame."""
        payload = extras["reward_episode_transition"]
        payload["returned_episode_ids"] = self.reward_episode_ids.clone()
        payload["reset_mask"] = payload["returned_episode_ids"] != payload["episode_ids"]

    def reset(self, ids):
        count = len(ids)
        # Wrenches are recomputed from returned physical poses on the next
        # interval; a previous episode's command is never a reset observation.
        self._spring_pressure_frame = None
        if getattr(self, "reward_episode_measurement_enabled", False):
            self.reward_episode_ids[ids] += 1
        if self.scut_reward is not None:
            self.scut_reward.reset(ids)
            self.closure_gap_peak[ids] = 0.
        if self.critic78:
            if self.physical_tick % self.decimation:
                raise RuntimeError("PC DR reset must occur at an integer policy boundary")
            self.pc_parameters.reset(ids, global_policy_tick=self.physical_tick // self.decimation)
            self.critic_contact.reset(ids)
            self.critic_net_contact_force[ids] = 0.
            self.critic_applied_torque[ids] = 0.
            self.critic_applied_at_s[ids] = self.physical_tick * self.dt
            self.critic_pd_origin_s[ids] = self.physical_tick * self.dt
            if getattr(self, "policy_signal_delay", None) is not None:
                self.critic_target_sample_s[ids] = self.physical_tick * self.dt
            self.policy_generated_s[ids] = self.physical_tick * self.dt
            self.closure_gap_peak[ids] = 0.
        flat = getattr(self, "flat_episode", None)
        flat_reset = None
        scripted_reset_ids = None
        if flat is not None:
            from wheeled_tasks.chassis.application.runtime.training_schedule import actor_update_clock
            recovery = getattr(self, "recovery", None)
            scripted = getattr(recovery, "source_flat_handoff", False)
            source_mask = ~recovery.enabled[ids] if scripted else torch.ones(count, dtype=torch.bool, device=self.device)
            source_ids = ids[source_mask]
            if len(source_ids):
                birth_kwargs = {}
                if getattr(flat, "final_source_birth_enabled", False):
                    birth_kwargs["iteration"] = self._flat_command_iteration()
                nominal_mask = getattr(self, "_dynamic_birth_nominal_mask", None)
                if nominal_mask is not None or getattr(self, "cfg", {}).get("formal_foundation_nominal_birth"):
                    birth_kwargs["nominal_q"] = self.nominal[source_ids][:, [self.robot.joint_names.index(n)
                        for n in flat.tree_joint_names]]
                if nominal_mask is not None:
                    birth_kwargs["nominal_mask"] = nominal_mask[source_ids]
                flat_reset = flat.prepare_reset(source_ids, self.generator, origins=self.origins[source_ids],
                    reset_xy_world=self.reset_positions[source_ids, :2],
                    ground_z_world=self.origins[source_ids, 2] + self.ground_height(
                        self.reset_positions[:, :2] - self.origins[:, :2])[source_ids],
                    **birth_kwargs)
                final_terrain_birth = (getattr(self, "cfg", {}).get("formal_terrain", {}).get("source")
                                       == "v7_final_b8ff79f3"
                                       and getattr(self, "evaluation_surface", "training") != "plane_regression_v1")
                if final_terrain_birth:
                    from wheeled_tasks.chassis.tasks.recovery.final_rough_birth import apply_nominal_axis_heading
                    axis_kwargs = ({"nominal_mask": nominal_mask[source_ids]}
                                   if nominal_mask is not None else {})
                    if getattr(flat, "final_source_birth_enabled", False):
                        axis_kwargs = {"nominal_mask": flat_reset.requests.birth_kind == -2}
                    apply_nominal_axis_heading(flat_reset, flat.commands, self.generator, **axis_kwargs)
                if getattr(self, "rough_birth_support", None) is not None:
                    self.rough_birth_support.adjust(flat_reset,
                        **({"actual_root_pose_orientation": True} if final_terrain_birth else {}))
            if scripted:
                scripted_reset_ids = ids[~source_mask]
                root = self.nominal.new_zeros(count, 7)
                reset_velocity = self.nominal.new_zeros(count, 6)
                joints = self.nominal[ids].clone()
                joint_velocity = torch.zeros_like(joints)
                if flat_reset is not None:
                    root[source_mask] = flat_reset.root_pose.to(root)
                    reset_velocity[source_mask] = flat_reset.root_velocity.to(reset_velocity)
                    joints[source_mask] = flat_reset.joint_position[:, self.flat_tree_to_native].to(joints)
                    joint_velocity[source_mask] = flat_reset.joint_velocity[:, self.flat_tree_to_native].to(joint_velocity)
                recovery.initial_state(ids, root, joints, reset_velocity)
            else:
                root = flat_reset.root_pose.to(dtype=self.nominal.dtype)
                reset_velocity = flat_reset.root_velocity.to(dtype=self.nominal.dtype)
                joints = flat_reset.joint_position[:, self.flat_tree_to_native].to(dtype=self.nominal.dtype)
                joint_velocity = flat_reset.joint_velocity[:, self.flat_tree_to_native].to(dtype=self.nominal.dtype)
            trace = getattr(self, "cfg", {})
            if (trace.get("formal_physics_trace_directory") and self.physical_tick == 0
                    and (trace.get("formal_physics_trace_substeps")
                         or trace.get("formal_dynamic_nonsteady_birth"))):
                ground_birth = torch.zeros(count, dtype=torch.bool, device=self.device)
                if flat_reset is not None:
                    ground_birth[source_mask] = flat_reset.requests.ground_birth
                torch.save({"env_ids": ids.detach().cpu(), "joint_names": self.robot.joint_names,
                    "q": joints.detach().cpu(), "dq": joint_velocity.detach().cpu(),
                    "root_pose": root.detach().cpu(), "root_velocity": reset_velocity.detach().cpu(),
                    "ground_birth": ground_birth.detach().cpu(),
                    # commit_reset increments these IDs after the native write.
                    # A restored worker can reset again before its first step.
                    "episode_id": (flat.episode_id[ids] + 1).detach().cpu(),
                    "episode_physics_ticks": torch.zeros_like(ids).detach().cpu(),
                    **({"scripted_recovery_birth": recovery.enabled[ids].detach().cpu()} if scripted else {}),
                    "scope": "requested native write, not a simulator readback"},
                    Path(self.cfg["formal_physics_trace_directory"]) / "birth_requested.pt")
            posture = None
        else:
            root = torch.zeros(count, 7, device=self.device)
            root[:, :3], root[:, 6] = self.reset_positions[ids], 1.
            reset_velocity = torch.zeros(count, 6, device=self.device)
            joint_velocity = torch.zeros(count, self.joint_count, device=self.device)
            if self.skills is not None:
                self.skills.reset_pose(ids, root, reset_velocity)
            if self.step_jump is not None:
                selected = (self.step_jump.enabled | self.step_climb.enabled)[ids]
                root[selected, 0] = self.origins[ids[selected], 0] - .8
            joints = self.nominal[ids].clone()
            if self.recovery is not None:
                self.recovery.initial_state(ids, root, joints, reset_velocity, joint_velocity=joint_velocity)
            posture = getattr(self, "ordinary_posture", None)
            if posture is not None and posture.enabled:
                posture.reset(ids, (self._curriculum_clock("ordinary_posture_curriculum") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self)), generator=self.generator)
                posture.initial_orientation(ids, root)
            if self.nonstandard_start is not None:
                from wheeled_tasks.chassis.tasks.recovery.nonstandard_start import support_height_for_footprint
                self.nonstandard_start.reset(ids, (self._curriculum_clock("nonstandard_start") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self)),
                    eligible=self.nonstandard_start_eligible, generator=self.generator)
                support = support_height_for_footprint(self.surface_data[ids], self.surface_valid[ids],
                    root[:, :2] - self.origins[ids, :2], self.nonstandard_start.collision_footprints(ids))
                self.nonstandard_start.initial_state(ids, root, joints, reset_velocity,
                    support + self.origins[ids, 2], preserve_orientation=posture is not None and posture.enabled)
            if posture is not None and posture.enabled:
                cancelled_bank = posture.align_support(ids, root, joints, self.origins, self.surface_data, self.surface_valid,
                    profile_ids=self.nonstandard_start.profile_ids if self.nonstandard_start is not None else None)
                if self.nonstandard_start is not None:
                    self.nonstandard_start.cancel_reset(cancelled_bank)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=reset_velocity, env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=joints, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=ids)
        self.robot.reset(ids)
        if getattr(self, "flat_events", None) is not None:
            self.flat_events.reset(ids)
        if getattr(self, "v7_stepup", None) is not None:
            self.v7_stepup.reset(ids)
        update_robot_data(self, self.dt)
        if self.step_jump is not None:
            self.step_jump.reset(ids)
            self.step_climb.reset(ids)
        if self.step_sensors is not None:
            self.step_sensors.reset(ids, root_link_linear_velocity(self))
        self.actuator_response.reset(ids)
        if self.usb_transport is not None:
            self.usb_transport.reset(ids, self._transport_sensors())
        if self.feedback_transport is not None:
            if self.round_trip_transport is not None:
                self.round_trip_transport.reset(ids)
            self.feedback_transport.reset(ids, self._transport_sensors())
        if self.command_transport is not None:
            self.command_transport.reset(ids)
        if getattr(self, "policy_signal_delay", None) is not None:
            self.policy_signal_delay.reset(ids, self._pc_feedback_transport.feedback,
                self._pc_feedback_transport.sensor_timestamp_s)
        self.actions[ids] = 0
        self.previous_actions[ids] = 0
        self.before_previous_actions[ids] = 0
        self.previous_motor_velocity[ids] = 0
        self.contact_force[ids] = 0
        self.episode_length_buf[ids] = 0
        self.success_hold[ids] = 0
        self.sequence_success_latched[ids] = False
        self.successful_jump_count[ids] = 0
        self.step_contact_time[ids] = 0
        if self.step_assist is not None:
            self.step_assist.reset(ids)
        if self.jump_assist is not None:
            self.jump_assist.reset(ids, (self._curriculum_clock("jump_assist") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self)), self.random(count),
                                   evaluation=self.cfg.get("evaluation_exact_cases", False))
        self.jump_requested[ids] = False
        self.phase.reset(ids)
        self.fall_confirmation.reset(ids)
        if self.passive_landing is not None:
            self.passive_landing.reset(ids)
        if self.jump_landing is not None:
            self.jump_landing.reset(ids)
        self.landing_reward_mask[ids] = 0.
        self.command_acceleration[ids] = 0.
        if self.references is not None:
            self.references.reset(ids)
        if self.stationary_anchor is not None:
            self.stationary_anchor.reset(ids)
        if self.performance_curriculum is not None:
            self.performance_curriculum.reset_episodes(ids)
        if self.skills is not None:
            self.skills.reset_phase(ids)
        self.history.reset(ids)
        if getattr(self, "actor_sensor_noise", None) is not None:
            self.actor_sensor_noise.reset(ids)
        if self.signal_delay is not None:
            self.signal_delay.reset(ids, self._transport_sensors())
        if self.perturbations is not None:
            self.perturbations.reset(ids)
            if self.domain_draw is not None:
                self.perturbations.set_enabled(ids, self.perturbations.enabled[ids] & (self.domain_draw[ids] < .5))
            if self.usb_transport is not None:
                self.perturbations.set_enabled(ids, self.usb_transport.enabled[ids])
            self.motor_strength[ids] = .85 + .15 * torch.rand(count, 6, generator=self.generator, device=self.device)
            self.spring_strength[ids] = .9 + .2 * torch.rand(count, 1, generator=self.generator, device=self.device)
            clean = ids[~self.perturbations.enabled[ids]]
            self.motor_strength[clean] = 1.
            if self.critic78 and self.cfg.get("pc_relative_priors", {}).get("enabled"):
                self.motor_strength[ids] = 1.
            self.spring_strength[clean] = 1.
        if getattr(self, "episode_spring_strength_settings", None) is not None:
            low, high = self.episode_spring_strength_settings["scale_range"]
            self.motor_strength[ids] = 1.
            self.spring_strength[ids] = low + (high - low) * torch.rand(
                count, 1, device=self.device, generator=self.generator)
        self.command_clock[ids] = 0
        self.height_clock[ids] = 0
        if flat is not None:
            # Source event DR needs its own admitted adapter. Never inherit the
            # old phase-gated push sampler through this command/reset path.
            self.push_clock[ids] = torch.inf
            self.push_enabled[ids] = False
        else:
            self.push_clock[ids] = 5 + 2 * self.random(count)
            self.push_enabled[ids] = self.random(count) < 0.5
        if posture is not None and posture.enabled:
            ordinary = ids[posture.eligible[ids]]
            self.push_clock[ordinary] = posture.sample_intervals(ordinary, generator=self.generator)
            self.push_enabled[ordinary] = True
        if flat is not None:
            iteration = self._flat_command_iteration()
            if flat_reset is not None:
                flat.commit_reset(flat_reset, self._flat_heading()[source_ids], self.generator, iteration)
            if scripted_reset_ids is not None and len(scripted_reset_ids):
                flat.commit_scripted_reset(scripted_reset_ids, self._flat_heading()[scripted_reset_ids], self.generator, iteration)
            flat.publish(self.commands, self.mode, self.targets, self.jump_requested)
        else:
            self.resample_commands(ids, reset_height=True)
        self.raw_height[ids] = self.commands[ids, 2]
        if getattr(self, "v7_stepup", None) is not None:
            self.commands[:, 2] = self.v7_stepup.effective_height(self.commands[:, 2])
        if self.recovery is not None:
            self.recovery.reset(ids)
        if self.full_tasks is not None:
            self.full_tasks.reset(ids, self.robot.data.root_link_pose_w.torch[:, :3] - self.origins)
        measurement = getattr(self, "jump_actuation_measurement", None)
        if measurement is not None:
            measurement.reset(ids, physical_tick=self.physical_tick)
        if getattr(self, "jump_physics_trace", None) is not None:
            self.jump_physics_trace.reset(ids, physical_tick=self.physical_tick)
        self.update_targets()
        if flat is not None:
            air_limits = getattr(flat.commands, "apply_air_birth_command_limits", None)
            if air_limits is not None:
                air_limits(self.commands, ids=ids)
        if flat is None:
            self.update_height_assist(ids, advance=False)
            self.update_command_reference(ids)

    def resample_commands(self, ids, *, reset_height=False):
        if getattr(self, "flat_episode", None) is not None:
            raise RuntimeError("Source flat commands have one owner; forced legacy resampling is forbidden")
        n = len(ids)
        if (self.step_assist is not None or self.references is not None) and not reset_height:
            self.commands[ids, 2] = self.raw_height[ids]
        previous_velocity_commands = self.commands[ids, :2].clone()
        if self.skills is not None:
            # SkillCommands samples entire groups on the device. Its values
            # replace the legacy per-environment sampler completely.
            self.skills.sample(ids, reset_height=reset_height)
            if self.step_assist is not None or self.references is not None:
                self.raw_height[ids] = self.commands[ids, 2]
            if self.cfg.get("command_slew"):
                target_ids = ids
                release = getattr(self.skills, "release_stop", None)
                if not reset_height and release is not None:
                    # This finite profile already changed only its request.
                    # Preserve that request when an explicit resample occurs.
                    target_ids = ids[~release.active[ids]]
                self.command_target[target_ids] = self.commands[target_ids, :2]
                self.commands[ids, :2] = 0. if reset_height else previous_velocity_commands
            return
        pick = self.random(n)
        # 30% stand, 20% turn, 30% straight, 20% combined, on the foundation samples.
        curriculum = self.cfg.get("command_curriculum")
        fraction = min(1., self.training_transitions / curriculum["ramp_transitions"]) if curriculum else 1.
        vx_max = ((1 - fraction) * curriculum["initial_vx"] + fraction * self.stage_cfg["vx_max"]) if curriculum else self.stage_cfg["vx_max"]
        yaw_max = ((1 - fraction) * curriculum["initial_yaw"] + fraction * self.stage_cfg["yaw_max"]) if curriculum else self.stage_cfg["yaw_max"]
        vx = (2 * self.random(n) - 1) * vx_max
        yaw = (2 * self.random(n) - 1) * yaw_max
        vx[pick < 0.5] = 0
        yaw[(pick < 0.3) | ((pick >= 0.5) & (pick < 0.8))] = 0
        # Conservative rolling-speed and lateral-acceleration command envelope.
        lateral = (vx * yaw).abs()
        yaw *= torch.minimum(torch.ones_like(yaw), 0.7 * 0.3 * 9.81 / lateral.clamp_min(1e-6))
        speed = vx.abs() + 0.4373 * yaw.abs() / 2
        scale = (4.35 / speed.clamp_min(1e-6)).clamp_max(1)
        self.commands[ids, 0] = vx * scale
        self.commands[ids, 1] = yaw * scale
        self.mode[ids] = (pick >= 0.3).long()
        if reset_height:
            low, high = self.cfg["height_range_m"]
            h = self.random(n)
            self.commands[ids, 2] = low + (high - low) * h
            self.commands[ids[h < 0.15], 2] = low
            self.commands[ids[h > 0.85], 2] = high
            self.height_clock[ids] = 5 + 3 * self.random(n)
        for idx in ids.tolist():
            kind = self.kinds[idx]
            group = self.scene_groups[idx]
            if group == "stand":
                self.commands[idx, :2] = 0
                self.mode[idx] = 0
            elif group == "translate":
                sign = 1 if float(self.random(1)[0]) >= .5 else -1
                self.commands[idx, 0] = sign * vx_max * (.2 + .8 * self.random(1)[0])
                self.commands[idx, 1] = 0
                self.mode[idx] = 1
            elif group == "rotate":
                cap = yaw_max if kind == "flat" else min(yaw_max, 2.)
                sign = 1 if float(self.random(1)[0]) >= .5 else -1
                self.commands[idx, 0] = 0
                self.commands[idx, 1] = sign * cap * (.2 + .8 * self.random(1)[0])
                self.mode[idx] = 1
            elif group == "combined":
                vx_value = (2 * self.random(1)[0] - 1) * vx_max
                yaw_value = (2 * self.random(1)[0] - 1) * yaw_max
                yaw_value *= (2.06 / (vx_value * yaw_value).abs().clamp_min(1e-6)).clamp_max(1.)
                scale_value = (4.35 / (vx_value.abs() + .4373 * yaw_value.abs() / 2).clamp_min(1e-6)).clamp_max(1.)
                self.commands[idx, :2] = torch.stack((vx_value * scale_value, yaw_value * scale_value))
                self.mode[idx] = 1
            elif self.route[idx] or kind == "jump":
                self.commands[idx, :2] = self.commands.new_tensor([0.4, 0.])
                self.mode[idx] = {"step_up": 2, "stairs": 2, "step_down": 3, "jump": 4}.get(kind, 1)
                if self.full_tasks is not None:
                    if kind == "jump":
                        self.commands[idx, :2] = 0.
                    elif kind == "low_step":
                        self.mode[idx] = 2
        if self.skills is not None:
            self.skills.sample(ids, reset_height=reset_height)
        if self.full_tasks is not None:
            self.height_clock[ids[self.mode[ids] >= 2]] = 1e9
        if self.cfg.get("command_slew"):
            self.command_target[ids] = self.commands[ids, :2]
            self.commands[ids, :2] = 0. if reset_height else previous_velocity_commands
        if self.skills is None:
            self.command_clock[ids] = 3 + 2 * self.random(n)
        if self.step_assist is not None or self.references is not None:
            self.raw_height[ids] = self.commands[ids, 2]

    def update_targets(self):
        if getattr(self, "flat_episode", None) is not None:
            self.targets.zero_()
            self.jump_requested.zero_()
            self.mode.fill_(1)
            return
        local = self.robot.data.root_link_pose_w.torch[:, :3] - self.origins
        task = self.mode >= 2
        self.targets.zero_()
        self.targets[:, 0] = torch.where(task, -local[:, 0], 0.)
        self.targets[:, 2] = torch.where(task, 1.8 - local[:, 0], 0.)
        self.targets[:, 1] = torch.where(task, self.platform_delta, 0.)
        near = (self.targets[:, 0] < self.cfg.get("jump_trigger_distance_m", .35)) & (self.targets[:, 0] > -0.1)
        request = (((self.mode == 2) & near) | ((self.mode == 4) & (self.episode_length_buf >= 100)))
        self.jump_requested |= request
        self.targets[:, 3] = self.jump_requested.float()
        if self.full_tasks is not None:
            jumping = self.mode == 4
            self.jump_requested &= jumping
            self.jump_requested |= jumping & (self.episode_length_buf * self.policy_dt >= self.full_tasks.cfg.get("jump_request_seconds", .8))
            self.targets[:, 0] = torch.where(jumping, 0., self.targets[:, 0])
            self.targets[:, 1] = torch.where(jumping, self.full_tasks.cfg.get("jump_apex_delta_m", .06), self.targets[:, 1])
            self.targets[:, 2] = torch.where(jumping, 0., self.route_goal - local[:, 0]) * task
            self.targets[:, 3] = self.jump_requested.float()
        if self.skills is not None:
            self.skills.observation_targets()
            schedule = self.skills.schedule
            if schedule.requests_enabled:
                elapsed = self.episode_length_buf * self.policy_dt
                route_request = self.skills.manual_step_mode != 0
                initial = ((self.mode == 4) & (elapsed >= self.cfg["task_semantics"]["jump_request_seconds"])) | (
                    route_request & (elapsed >= self.skills.manual_step_start))
                eligible = (self.mode == 4) | route_request
                if self.step_jump is not None:
                    eligible &= ~self.step_jump.enabled
                    initial &= ~self.step_jump.enabled
                self.jump_requested = schedule.update_requests(elapsed, self.phase.phase, eligible, initial)
                self.targets[:, 3] = self.jump_requested.float()
                measurement = getattr(self, "jump_actuation_measurement", None)
                if measurement is not None:
                    for request_kind, kind_mask in (
                            ("schedule_mode4_jump", self.mode == 4),
                            ("schedule_manual_step_reference", self.mode != 4)):
                        accepted = ((schedule.request_event == 1) & kind_mask).nonzero(as_tuple=False).flatten()
                        if len(accepted):
                            measurement.accept_request(accepted, source="schedule", request_kind=request_kind,
                                source_request_ids=schedule.accepted_request_count[accepted],
                                physical_tick=self.physical_tick)
                accepted_jump = ((schedule.request_event == 1) & (self.mode == 4)).nonzero(as_tuple=False).flatten()
                # A second jump needs a new release/apex record, not the first
                # jump's successful history. Continuous motor angles stay intact.
                if len(accepted_jump):
                    if self.jump_cycle_reset_on_accepted:
                        self._reset_jump_evidence(accepted_jump, local)
                    else:
                        for name in ("clearance_peak", "height_peak", "clear_air_time", "clear_air_time_peak",
                                     "release_velocity", "com_release_height", "com_release_speed", "com_rise",
                                     "com_start_height", "com_displacement"):
                            getattr(self.full_tasks, name)[accepted_jump] = 0.
                        for name in ("release_recorded", "com_released", "com_release_pending", "com_start_valid"):
                            getattr(self.full_tasks, name)[accepted_jump] = False
                        self.full_tasks.origin_xy[accepted_jump] = local[accepted_jump, :2]
                        self.phase.flew[accepted_jump] = False
                        self.success_hold[accepted_jump] = 0.
                        self.sequence_success_latched[accepted_jump] = False
                cancel_grounded = ((schedule.request_event == 3) & (self.phase.phase == Phase.TAKEOFF)
                                   & (self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.).all(-1))
                self.phase.reset(cancel_grounded.nonzero(as_tuple=False).flatten())
        if self.step_jump is not None and self.step_jump.has_tasks:
            self._update_step_jump()
            selected = self.step_jump.enabled
            self.jump_requested[selected] = self.step_jump.active[selected]
            self.targets[selected, 1] = self.step_jump.apex[selected]
            self.targets[selected, 3] = self.step_jump.active[selected].float()

    def get_observations(self, *, critic_only=False):
        velocity, omega, gravity, height, q, dq, _ = self.state()
        if self.scut35:
            from wheeled_tasks.chassis.observations.actor import build_scut35
            requested = self.jump_requested & (self.mode == 4)
            # The source flat profile excludes legacy jump semantics entirely.
            # Its unchanged35D interface has an ordinary, zero-clock context.
            elapsed = (torch.zeros_like(self.commands[:, 0]) if self.scut_reward is not None
                       else self.episode_length_buf * self.policy_dt
                       - self.cfg["task_semantics"]["jump_request_seconds"])
            lateral = self.targets[:, 1] * self.skills.spin if self.skills is not None else None
            frame = build_scut35(omega, gravity, self.commands, q, dq, self.actions,
                self.v5.nominal, requested, self.targets[:, 1] * (self.mode == 4), elapsed, lateral)
            if self.references is not None:
                from wheeled_tasks.chassis.observations.actor import build_manual35, build_reference36
                builder = build_manual35 if self.cfg.get("manual_context35") else build_reference36
                frame = builder(omega, gravity, self.commands, q, dq, self.actions,
                    self.v5.nominal, requested, self.targets[:, 1] * (self.mode == 4), self.references, lateral)
        else:
            frame = self.v5.proprioception(omega, gravity, self.commands, q, dq, self.actions)
            if self.is_v5:
                compression, speed = self.v5.spring_state(self.robot.data.joint_pos.torch[:, self.spring_ids],
                                                         self.robot.data.joint_vel.torch[:, self.spring_ids])
                frame = torch.cat((frame, compression / .08, speed / 1.2), -1)
            contacts = (self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2).float()
            target = self.targets * frame.new_tensor([0.5, 5., 0.5, 1.])
            frame = torch.cat((frame, F.one_hot(self.mode, 5).float(), target.clamp(-10, 10), contacts,
                               F.one_hot(self.phase.phase, 5).float(), self.phase.time.clamp_max(5)[:, None]), -1)
        if self.critic78:
            critic = self._critic78_frame(frame)
        else:
            critic_q = self.robot.data.joint_pos.torch
            if self.cfg.get("zero_critic_wheel_positions"):
                critic_q = critic_q.clone()
                critic_q[:, [self.ids[2], self.ids[5]]] = 0.
            critic = torch.cat((frame, velocity, height[:, None],
                                self.contact_force[:, self.wheel_ids].norm(dim=-1) / 125.,
                                critic_q, self.robot.data.joint_vel.torch * 0.1), -1)
            if self.is_v5:
                wheel_xy = self.robot.data.body_link_pose_w.torch[:, self.wheel_ids, :2] - self.origins[:, None, :2]
                support = torch.stack([self.ground_height(wheel_xy[:, side]) for side in range(2)], -1)
                mu = []
                for side in range(2):
                    x = wheel_xy[:, side, :1]
                    mask = (x >= self.surface_data[:, :, 0]) & (x <= self.surface_data[:, :, 1]) & self.surface_valid
                    index = mask.long().argmax(-1)
                    mu.append(self.surface_mu.gather(1, index[:, None])[:, 0])
                critic = torch.cat((critic, torch.stack(mu, -1), support), -1)
            if self.references is not None and not self.cfg.get("manual_context35"):
                com_offset = (self.dynamics_randomization.com_offset if self.dynamics_randomization is not None
                              else torch.zeros(self.num_envs, 3, device=self.device))
                com_vz = (self.robot.data.body_com_lin_vel_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                relative_xy = self.robot.data.root_link_pose_w.torch[:, :2] - self.origins[:, :2] - self.full_tasks.origin_xy
                critic = torch.cat((critic, self.body_mass / self.nominal_body_mass, com_offset * 10.,
                    relative_xy.clamp(-5., 5.) * .2, com_vz[:, None], self.full_tasks.com_rise[:, None] * 5.,
                    F.one_hot(self.phase.phase, 5).float(), self.full_tasks.com_displacement[:, None] * 5.), -1)
        if frame.shape[-1] != self.cfg["actor_frame_dim"] or critic.shape[-1] != self.cfg["critic_dim"]:
            raise RuntimeError("New task observation layout mismatch")
        if critic_only:
            # Terminal value inputs must not populate actor history/noise
            # caches before reset and the next interval's external events.
            return critic
        actor_noise = getattr(self, "actor_sensor_noise", None)
        if actor_noise is not None:
            self._actor_sensor_noise_clean_frame = frame.clone()
        if self._pc_feedback_transport is not None:
            sensors = self._pc_feedback_transport.feedback
            frame = build_scut35(sensors[:, 12:15], sensors[:, 15:18], self.commands,
                sensors[:, :6], sensors[:, 6:12], self.actions, self.v5.nominal,
                requested, self.targets[:, 1] * (self.mode == 4), elapsed, lateral)
            if self.references is not None:
                frame = builder(sensors[:, 12:15], sensors[:, 15:18], self.commands,
                    sensors[:, :6], sensors[:, 6:12], self.actions, self.v5.nominal,
                    requested, self.targets[:, 1] * (self.mode == 4), self.references, lateral)
        if self.signal_delay is not None:
            from wheeled_tasks.chassis.observations.actor import build_sensor18
            sensors = self.signal_delay.feedback
            frame[:, 4:22] = build_sensor18(sensors[:, 12:15], sensors[:, 15:18],
                sensors[:, :6], sensors[:, 6:12], self.v5.nominal)
        policy_delay = getattr(self, "policy_signal_delay", None)
        if policy_delay is not None:
            from wheeled_tasks.chassis.observations.actor import build_sensor18
            sensors = policy_delay.feedback
            frame[:, 4:22] = build_sensor18(sensors[:, 12:15], sensors[:, 15:18],
                sensors[:, :6], sensors[:, 6:12], self.v5.nominal)
        if actor_noise is not None:
            from wheeled_tasks.chassis.observations.actor import build_sensor18
            if policy_delay is not None:
                sensors = policy_delay.feedback
            elif self.signal_delay is not None:
                sensors = self.signal_delay.feedback
            elif self._pc_feedback_transport is not None:
                sensors = self._pc_feedback_transport.feedback
            else:
                sensors = torch.cat((q, dq, omega, gravity), -1)
            noisy = actor_noise.apply(sensors, self.tick)
            frame[:, 4:22] = build_sensor18(noisy[:, 12:15], noisy[:, 15:18],
                noisy[:, :6], noisy[:, 6:12], self.v5.nominal)
        if self.perturbations is not None:
            frame = self.perturbations.observation(frame, self.tick)
        return TensorDict({"policy": self.history.update(frame, self.tick), "critic": critic}, batch_size=[self.num_envs])

    def _critic78_frame(self, clean_frame):
        from wheeled_tasks.chassis.observations.critic_observation import build_scut78, pd_action_age_ticks
        from wheeled_tasks.chassis.simulation.runtime.privileged_runtime import wheel_relative_velocity_body
        order = self.critic_axis_from_control
        age = torch.zeros(self.num_envs, 4, device=self.device)
        policy_delay = getattr(self, "policy_signal_delay", None)
        if policy_delay is not None:
            age = (policy_delay.observation_age_s[:, [2, 3, 0, 1]] / self.dt).float()
        elif self._pc_feedback_transport is not None:
            age = (self._pc_feedback_transport.age_s / self.dt).float()[:, None].expand(-1, 4)
        elif self.signal_delay is not None:
            available = (self.signal_delay.sensor_buffer.current_length - 1).clamp_min(0)
            age = torch.minimum(self.signal_delay.lags[:, [2, 3, 0, 1]], available[:, None]).float()
        base_id = self.robot.body_names.index("base_link")
        root_height = self.robot.data.root_link_pose_w.torch[:, 2:3]
        if getattr(self, "cfg", {}).get("critic_height_reference") != "root_link_world_z":
            root_height = root_height - self.origins[:, 2:3]
        action_age = (pd_action_age_ticks(self.critic_applied_at_s, self.critic_pd_origin_s)
            if policy_delay is None else
            ((self.critic_applied_at_s[:, None] - self.critic_target_sample_s) / self.dt).float())
        return build_scut78(clean_common28=clean_frame[:, :28],
            root_linear_velocity_b=root_link_linear_velocity(self, frame="body"),
            root_height=root_height,
            command_context7=clean_frame[:, 28:35],
            pc_kp=self.pc_parameters.kp[:, order], pc_kd=self.pc_parameters.kd[:, order],
            applied_torque=self.critic_applied_torque[:, order], observation_age_ticks=age,
            action_age_ticks=action_age,
            wheel_relative_velocity_b=wheel_relative_velocity_body(self.robot.data, self.wheel_ids,
                root_link_linear_velocity(self, frame="world")),
            wheel_contact=self.critic_contact.contacts(),
            base_mass_ratio=self.body_mass[:, base_id:base_id+1] / self.nominal_body_mass[:, base_id:base_id+1],
            wheel_material=self.critic_material.values,
            validate=getattr(self, "_critic78_validate_once", False))

    def closure_gap(self):
        pose = self.robot.data.body_link_pose_w.torch[:, self._closure_body_ids]
        local = self._closure_points.expand(self.num_envs, -1, -1, -1)
        cross = torch.linalg.cross(pose[..., 3:6], local)
        world = pose[..., :3] + local + 2 * (pose[..., 6:] * cross
                + torch.linalg.cross(pose[..., 3:6], cross))
        return (world[:, :, 0] - world[:, :, 1]).norm(dim=-1).amax(-1)

    def step(self, actions):
        if self.cfg.get("formal_physics_trace_directory") and self.tick == 0:
            self._save_formal_physics_state("before_first_step")
        self.control_audit.begin_policy()
        if (self.skills is not None and self.skills.large_stair_curricula
                and self.skills.large_stair_actor_clock != getattr(self, "large_stair_assessment_clock", None)):
            # Arm at the physical transition's start, so the first real riser
            # crossing is retained without crediting pretraining coordinates.
            self.step_climb.set_assessment_mask(self.skills.large_stair_assessment,
                self.wheel_centers() - self.origins[:, None, :])
            self.large_stair_assessment_clock = self.skills.large_stair_actor_clock
        if self.signal_delay is not None:
            if getattr(self, "integrated_runtime_enabled", False):
                from wheeled_tasks.chassis.application.compatibility.integrated_runtime import component_actor_updates
                delay_clock = component_actor_updates(self, "speed")
            else:
                delay_clock = stage_actor_updates(self)
            self.signal_delay.begin_policy(delay_clock)
        if self.jump_assist is not None and getattr(self, "integrated_runtime_enabled", False):
            self.jump_assist_actor_clock = self.jump_assist.actor_clock_tensor((self._curriculum_clock("jump_assist") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self)))
        if self.recovery is not None:
            self.recovery.begin_policy()
        measurement = getattr(self, "jump_actuation_measurement", None)
        if measurement is not None:
            measurement.begin_policy()
        if getattr(self, "jump_physics_trace", None) is not None:
            self.jump_physics_trace.begin_policy()
        if self.nonstandard_start is not None:
            self.nonstandard_start.begin_policy()
        flat = getattr(self, "flat_episode", None)
        if flat is not None:
            flat.begin_policy()
        if self.scut_reward is not None:
            self.scut_reward.begin_policy(actions)
        if self.scut35:
            from wheeled_tasks.chassis.observations.actor import CONTROL_FROM_POLICY
            actions = actions[:, CONTROL_FROM_POLICY]
        self.before_previous_actions.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        q = self.robot.data.joint_pos.torch[:, self.ids]
        if self._pc_feedback_transport is not None:
            q = self._pc_feedback_transport.feedback[:, :6]
        if self.recovery is not None:
            q = self.recovery.project_feedback(q)
        applied_actions = self.perturbations.action(actions, self.tick) if self.perturbations is not None else actions
        legs, wheels, clipped = self.v5.decode(applied_actions, q)
        self.control_audit.targets_decoded()
        if self.perturbations is not None:
            clipped = actions.clamp(-self.v5.action_bounds, self.v5.action_bounds)
        self.actions.copy_(clipped)
        if self.critic78:
            self.policy_generated_s.fill_(self.physical_tick * self.dt)
        if self.critic78 or self.scut_reward is not None:
            self.closure_gap_peak.zero_()
        self.contact_peak.zero_()
        if self.wheel_force_peak_n is not None:
            self.wheel_force_peak_n.zero_()
        for substep in range(self.decimation):
            q_now, dq_now = self.robot.data.joint_pos.torch[:, self.ids], self.robot.data.joint_vel.torch[:, self.ids]
            feedback_q, feedback_dq = q_now, dq_now
            if self._pc_feedback_transport is not None:
                feedback_q, feedback_dq = (self._pc_feedback_transport.feedback[:, :6],
                    self._pc_feedback_transport.feedback[:, 6:12])
            if self.recovery is not None:
                control_q = self.recovery.observe_feedback(feedback_q)
            else:
                control_q = feedback_q
            self.control_audit.feedback_control()
            pd_legs, pd_wheels = (self.signal_delay.targets(legs, wheels)
                                 if self.signal_delay is not None else (legs, wheels))
            policy_delay = getattr(self, "policy_signal_delay", None)
            if policy_delay is not None:
                pd_legs, pd_wheels = policy_delay.targets(legs, wheels,
                    self.policy_generated_s, self.physical_tick)
            self.torque = self.v5.motor_efforts(control_q, feedback_dq, pd_legs, pd_wheels)
            if self.recovery is not None and self.recovery.has_tasks:
                if substep % self.recovery.reference_stride == 0:
                    gravity = self.robot.data.projected_gravity_b.torch
                    gyro = self.robot.data.root_com_ang_vel_b.torch
                    if self._pc_feedback_transport is not None:
                        gyro, gravity = (self._pc_feedback_transport.feedback[:, 12:15],
                            self._pc_feedback_transport.feedback[:, 15:18])
                    self.recovery.advance_reference(feedback_q, feedback_dq, gravity, gyro)
                self.torque = self.recovery.torque(feedback_q, feedback_dq, self.torque)
            self.torque *= self.motor_strength
            if self.skills is not None:
                self.torque *= self.skills.schedule.motor_scale
            if self.nonstandard_start is not None:
                # Feed zero into drive history during the passive interval.
                # The final gate below also removes retained/filter output.
                self.torque = torch.where(self.nonstandard_start.holding[:, None], 0., self.torque)
            if flat is not None:
                self.torque = flat.gate_active_effort(self.torque)
            response_integrated = False
            if self.command_transport is not None:
                from wheeled_tasks.chassis.control.actuation import motor_torque_limit
                bounds = (self.v5.current_motor_bounds[:, list(self.v5.WHEELS)]
                    if self._pc_feedback_transport is None else
                    motor_torque_limit(dq_now[:, list(self.v5.WHEELS)], self.v5.wheel_prior))
                self.v5.current_motor_bounds[:, list(self.v5.WHEELS)] = bounds
                response_integrated = (getattr(self, "round_trip_transport", None) is not None
                                       and self.actuator_response.enabled)
                response_options = ({"actuator_response": self.actuator_response,
                    "effort_bounds": self.v5.current_motor_bounds} if response_integrated else {})
                if getattr(self, "critic78", False):
                    feedback_time = (self._pc_feedback_transport.sensor_timestamp_s
                        if self._pc_feedback_transport is not None else self.physical_tick * self.dt)
                    target_origins = ({"target_sample_s": policy_delay.target_sample_s,
                        "target_policy_generated_s": policy_delay.target_policy_generated_s}
                        if policy_delay is not None else {})
                    self.torque = self.command_transport.apply_torque(self.torque,
                        pd_generated_s=self.physical_tick * self.dt,
                        policy_generated_s=self.policy_generated_s, feedback_sample_s=feedback_time,
                        **response_options, **target_origins)
                else:
                    self.torque = self.command_transport.apply_torque(self.torque, **response_options)
                # Without delayed feedback, motor_efforts just computed the
                # envelope at this exact physical velocity. Transport changes
                # torque only; reuse that bound instead of repeating GPU work.
                self.torque[:, list(self.v5.WHEELS)] = self.torque[:, list(self.v5.WHEELS)].clamp(-bounds, bounds)
            if self.usb_transport is not None:
                from wheeled_tasks.chassis.control.actuation import motor_torque_limit
                self.torque = self.usb_transport.apply_torque(self.torque)
                # Physical drive limits still depend on actual, not delayed, shaft speed.
                bound = motor_torque_limit(dq_now[:, list(self.v5.WHEELS)], self.v5.wheel_prior)
                wheels_tau = self.torque[:, list(self.v5.WHEELS)]
                self.torque[:, list(self.v5.WHEELS)] = torch.maximum(torch.minimum(wheels_tau, bound), -bound)
                self.v5.current_motor_bounds[:, list(self.v5.WHEELS)] = bound
            if self.actuator_response.enabled and not response_integrated:
                self.torque = self.actuator_response.apply_torque(self.torque)
                # The filtered effort still obeys the instantaneous drive envelope.
                # This also clips the retained motor state, preventing windup.
                bound = self.v5.current_motor_bounds
                self.torque.clamp_(min=-bound, max=bound)
            if self.nonstandard_start is not None:
                self.torque = self.nonstandard_start.apply_torque(self.torque)
            if flat is not None:
                self.torque = flat.gate_active_effort(self.torque)
            if self.recovery is not None and self.recovery.has_tasks:
                self.torque = self.recovery.gate_active_effort(self.torque)
            effort = torch.zeros_like(self.nominal)
            effort[:, self.ids] = self.torque
            if self.is_v5 and self.spring_pressure_element is None:
                effort[:, self.spring_ids] = self.v5.spring_efforts(self.robot.data.joint_pos.torch[:, self.spring_ids]) * self.spring_strength
            if self.spring_pressure_element is not None:
                self._apply_mount_spring_pressure()
            if self.jump_assist is not None:
                supported = (self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.).any(-1)
                pushing = (self.references.phase == self.references.PUSH) & (self.mode == 4)
                self.assist_force[:, 0, 2] = self.jump_assist.force(self.body_mass.sum(-1), self.targets[:, 1],
                    pushing, supported, self.phase.phase == Phase.FLIGHT, self.dt,
                    self.jump_assist_actor_clock if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self),
                    evaluation=self.cfg.get("evaluation_exact_cases", False))
                self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                    forces=self.assist_force, body_ids=self.assist_body_ids, is_global=True)
            physics_trace_frame = None
            if self.physical_tick + 1 in getattr(self, "cfg", {}).get("formal_physics_trace_substeps", ()):
                physics_trace_frame = self._formal_physics_control_frame(
                    pd_legs, pd_wheels, control_q, feedback_dq, effort)
            numerical_capture = getattr(self, "_numerical_failure_capture", None)
            if numerical_capture is not None:
                numerical_capture.remember_before_physics(self.physical_tick, self.tick,
                    self._numerical_failure_state(pd_legs, pd_wheels, control_q, feedback_dq, effort))
            self.robot.set_joint_effort_target_index(target=effort)
            if getattr(self, "jump_physics_trace", None) is not None:
                self._prepare_jump_physics_trace()
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            update_robot_data(self, self.dt)
            self.physical_tick += 1
            if flat is not None:
                replay = getattr(self, "replay_setup", None)
                if replay is not None:
                    replay.advance_source_clock(flat, self.physical_tick)
                else:
                    flat.advance_physics(self.dt)
            if self.scut_reward is not None:
                self.scut_reward.observe_physics(
                    self.robot.data.joint_vel.torch[:, self.ids],
                    self.robot.data.joint_acc.torch[:, self.ids],
                    self.robot.data.applied_torque.torch[:, self.ids])
            if self.critic78:
                self.critic_applied_torque.copy_(self.robot.data.applied_torque.torch[:, self.ids])
                self.critic_applied_at_s.fill_(self.physical_tick * self.dt)
                if self.command_transport is None:
                    self.critic_pd_origin_s.fill_((self.physical_tick - 1) * self.dt)
                else:
                    origin = self.command_transport.applied_origins_s[:, :1]
                    self.critic_pd_origin_s.copy_(origin.expand(-1, 2))
                if getattr(self, "policy_signal_delay", None) is not None:
                    self.critic_target_sample_s.copy_(self.command_transport.applied_target_sample_s)
            physical_closure_gap = None
            if self.critic78 or self.scut_reward is not None:
                # Record every5ms on-device; reduce after the policy interval.
                physical_closure_gap = self.closure_gap()
                self.closure_gap_peak.copy_(torch.maximum(self.closure_gap_peak, physical_closure_gap))
            if self.nonstandard_start is not None:
                self.nonstandard_start.advance()
            self.control_audit.physics_updated(self.dt)
            if self.signal_delay is not None:
                self.signal_delay.observe(self._transport_sensors())
            if self.usb_transport is not None:
                self.usb_transport.observe(self._transport_sensors())
            if self.feedback_transport is not None:
                self.feedback_transport.observe(self._transport_sensors())
            if getattr(self, "policy_signal_delay", None) is not None:
                self.policy_signal_delay.observe(self._pc_feedback_transport.feedback,
                    self._pc_feedback_transport.sensor_timestamp_s, self.physical_tick)
            if self.torque_monitor is not None:
                applied = self.robot.data.applied_torque.torch
                velocity = self.robot.data.joint_vel.torch
                compression, _ = self.v5.spring_state(self.robot.data.joint_pos.torch[:, self.spring_ids], velocity[:, self.spring_ids])
                self.torque_monitor.observe(applied[:, self.ids], velocity[:, self.ids],
                    applied[:, self.spring_ids], velocity[:, self.spring_ids], compression,
                    self.v5.requested_motor_effort, self.v5.current_motor_bounds,
                    pressure_force=(self._spring_pressure_frame.pressure_force_n
                        if self.spring_pressure_element is not None else None))
            for ids, contact_view in self.contact_views:
                matrix = wp.to_torch(contact_view.get_contact_force_matrix(dt=self.dt))
                self.contact_force[ids] = matrix.reshape(len(ids), self.body_count, -1, 3).sum(2)
                if getattr(self, "jump_physics_trace", None) is not None:
                    self._record_jump_trace_contacts(ids, contact_view, matrix)
                if self.scut_reward is not None or self.critic78 or getattr(self, "jump_actuation_measurement", None) is not None:
                    net = wp.to_torch(contact_view.get_net_contact_forces(dt=self.dt))
                    net = net.reshape(len(ids), self.body_count, 3)
                    if self.scut_reward is not None:
                        self.scut_reward.net_contact[ids] = net
                    if self.critic78:
                        self.critic_net_contact_force[ids] = net
                    if getattr(self, "jump_actuation_measurement", None) is not None:
                        self.jump_net_contact_force[ids] = net
            if self.scut_reward is not None:
                self.scut_reward.finish_contacts()
            if getattr(self, "jump_actuation_measurement", None) is not None:
                self._observe_jump_actuation()
            if getattr(self, "jump_physics_trace", None) is not None:
                self._finish_jump_physics_trace()
            if self.references is not None and self.has_jump_tasks:
                com_height = (self.robot.data.body_com_pose_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                com_vz = (self.robot.data.body_com_lin_vel_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                self.full_tasks.observe_com(self.mode, self.phase.phase, com_height, com_vz,
                    requested=self.references.jumping, contacts=self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.)
            wheel_force_n = self.contact_force[:, self.wheel_ids].norm(dim=-1)
            if self.critic78:
                self.critic_contact.observe(self.critic_net_contact_force[:, self.wheel_ids].norm(dim=-1))
            self.contact_peak = torch.maximum(self.contact_peak, wheel_force_n.sum(-1))
            if self.wheel_force_peak_n is not None:
                self.wheel_force_peak_n.copy_(torch.maximum(self.wheel_force_peak_n, wheel_force_n))
            if numerical_capture is not None:
                numerical_capture.observe_after_physics(self.physical_tick, self.tick,
                    (self.robot.data.joint_vel.torch[:, self.ids],
                     self.robot.data.root_com_lin_vel_b.torch, self.robot.data.root_com_ang_vel_b.torch),
                    lambda: self._numerical_failure_state(pd_legs, pd_wheels, control_q, feedback_dq, effort),
                    closure_gap=physical_closure_gap,
                    joint_position=self.robot.data.joint_pos.torch,
                    joint_velocity=self.robot.data.joint_vel.torch,
                    root_pose=self.robot.data.root_link_pose_w.torch)
            if physics_trace_frame is not None:
                self._save_formal_physics_state(f"physics_substep_{self.physical_tick:04d}",
                    control_frame=physics_trace_frame)
        self.startup_report["control_timing"]["runtime_substep_verification"] = self.control_audit.end_policy()
        if self.signal_delay is not None:
            self.startup_report["signal_delay"]["physics_steps_observed"] = self.signal_delay.physics_steps
        if self.recovery is not None and self.recovery.has_tasks:
            self.recovery.effective_action_history()
        if self.nonstandard_start is not None:
            self.actions.copy_(self.nonstandard_start.effective_action_history(self.actions))
        self.tick += 1
        self.episode_length_buf += 1
        if self.scut_reward is not None:
            return self._finish_scut_flat_step()
        measure_payments = getattr(self, "reward_episode_measurement_enabled", False)
        final_payments = {} if measure_payments else None
        velocity, omega, gravity, height, q, dq, local = self.state()
        if self.cfg.get("reward_velocity_reference") == "base_link_origin":
            velocity = root_link_linear_velocity(self, frame="body")
        vertical_velocity = -(velocity * gravity).sum(-1)
        finite = torch.stack([torch.isfinite(t).all()
                              for t in (velocity, omega, height, q, dq, self.contact_force)]).all()
        if not bool(finite):
            raise RuntimeError("Nonfinite physical transition")
        gap = self.closure_gap_peak if self.critic78 else self.closure_gap()
        self.max_closure_gap = max(self.max_closure_gap, float(gap.max()))
        if (not self.cfg.get("formal_closure_monitor_only") and self.max_closure_gap > 0.003
                and not self.cfg.get("closure_gap_termination_m")):
            raise RuntimeError(f"Closed-chain numerical gap exceeds 3 mm: {self.max_closure_gap}")
        contact = self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.
        touching = self.contact_force.norm(dim=-1) > 5.
        allowed_support = touching & self.step_contact_allowed
        near_step = self.route & (local[:, 0] > -.4) & (local[:, 0] < self.route_goal)
        self.step_contact_time = torch.where(allowed_support.any(-1) & near_step,
                                             self.step_contact_time + self.policy_dt, 0.)
        grace_active = near_step & (self.step_contact_time <= self.cfg.get("step_contact_grace", {}).get("seconds", 0.))
        undesired = touching & ~(self.step_contact_allowed & grace_active[:, None])
        fall_height = height
        if self.step_climb is not None and self.step_climb.has_tasks:
            points, forces = self._body_support_contacts()
            legal = self.step_climb.legal_body_contacts(points, forces)
            invalid = ((forces.norm(dim=-1) > 5.) & ~legal).any(-1)
            supported_base = legal.any(-1) & ~invalid
            undesired[:, self.base_support_id] &= ~supported_base
            # Wheel-plane height is not a falling test while the chassis itself
            # rests on the platform. Keep attitude and all mechanical checks.
            fall_height = torch.where(supported_base, height.clamp_min(.15), height)
        undesired = undesired[:, self.nonwheel_ids].any(-1)
        stable = (gravity[:, 2] < -0.985) & ((height - self.commands[:, 2]).abs() < 0.01)
        self.phase.update(contact, self.targets[:, 3].bool() & ~self.phase.flew, stable,
                          recovery_takeoff=self.references is not None)
        self.phase_counts += torch.bincount(self.phase.phase, minlength=5)
        masks = phase_reward_masks(self.phase.phase)
        grounded = masks["ground"].float()
        flight = masks["flight"].float()
        support_tracking = grounded + (self.phase.phase == Phase.RECOVERY).float() * self.cfg.get("track_height_during_recovery", False)
        reward_support_tracking = support_tracking
        if self.passive_landing is not None:
            # Ordinary tracking stays on through passive contact/recovery. A
            # transient physical phase must not hide the ordinary height cost.
            passive_contact = ((self.phase.phase == Phase.LANDING) | (self.phase.phase == Phase.RECOVERY)) & (self.mode != 4)
            reward_support_tracking = torch.where(passive_contact, 1., support_tracking)
        use_locomotion_rewards = self.cfg.get("reward_profile") == "scut_v14_flat_v5"
        spin_reward = self.skills is not None and self.skills.has_spin
        if not use_locomotion_rewards or spin_reward:
            velocity_reward = 2 * torch.exp(-((velocity[:, 0] - self.commands[:, 0]) / 0.5).square())
            if self.skills is not None:
                velocity_reward = self.skills.velocity_reward(velocity, velocity_reward)
        poses = self.robot.data.body_link_pose_w.torch
        extension = local[:, 2] + self.origins[:, 2] - poses[:, self.wheel_ids, 2].mean(-1)
        if not use_locomotion_rewards:
            quiet = grounded * (self.commands[:, :2].abs() < 0.05).all(-1)
            takeoff_speed = (2 * 9.81 * (self.targets[:, 1].clamp_min(0) + 0.04)).sqrt()
            old_takeoff_term = masks["takeoff"] if self.full_tasks is None else torch.zeros_like(masks["takeoff"])
            reward = (velocity_reward
                + torch.exp(-((omega[:, 2] - self.commands[:, 1]) / 0.5).square())
                + 2 * reward_support_tracking * torch.exp(-((height - self.commands[:, 2]) / 0.03).square())
                - 4 * gravity[:, :2].square().sum(-1) - 0.05 * omega[:, :2].square().sum(-1)
                - 0.5 * grounded * velocity[:, 2].square()
                + self.cfg.get("quiet_velocity_weight", -2.) * quiet * velocity[:, :2].abs().sum(-1)
                + self.cfg.get("quiet_wheel_weight", -.02) * quiet * ((0.06 * dq[:, [2, 5]].abs()
                    - self.cfg.get("quiet_wheel_deadband_m_s", .018)).clamp_min(0) / 0.1).square().mean(-1)
                - 0.01 * (self.actions - self.previous_actions).square().sum(-1)
                - 0.02 * (self.torque / self.torque.new_tensor([40, 40, 3.84, 40, 40, 3.84])).square().sum(-1)
                + 2 * old_takeoff_term * torch.exp(-((velocity[:, 2] - takeoff_speed) / 0.5).square())
                - 0.05 * masks["landing"] * ((self.contact_peak / 125 - 2).clamp_min(0)).square()) * self.policy_dt
            reward += flight * torch.exp(-((extension - .20) / .05).square()) * self.policy_dt
            reward += self.cfg.get("height_l1_weight", 0.) * reward_support_tracking * (height - self.commands[:, 2]).abs() * self.policy_dt
        components = None
        if use_locomotion_rewards:
            from wheeled_tasks.chassis.tasks.rewards.rewards import height_tracking_terms, reward_terms
            root = self.robot.data.root_link_pose_w.torch
            delta = self.wheel_centers() - root[:, None, :3]
            inverse_xyz = -root[:, None, 3:6].expand(-1, 2, -1)
            cross = torch.linalg.cross(inverse_xyz, delta)
            wheel_b = delta + 2 * (root[:, None, 6:] * cross + torch.linalg.cross(inverse_xyz, cross))
            ordinary = ~self.skills.spin if self.skills is not None else torch.ones_like(height, dtype=torch.bool)
            nonwheel = undesired
            components = reward_terms(velocity, omega, gravity, height, self.commands, dq,
                (dq - self.previous_motor_velocity) / self.policy_dt,
                self.robot.data.applied_torque.torch[:, self.ids], self.actions, self.previous_actions,
                self.before_previous_actions, wheel_b, reward_support_tracking, ordinary, nonwheel,
                self.performance_curriculum.env_height_width if self.performance_curriculum is not None else None)
            if self.performance_curriculum is not None:
                scales = self.performance_curriculum.env_height_scale
                components["track_height"] *= scales
                components["height_square"] *= scales
            if self.stationary_anchor is not None:
                requested = self.stationary_group & (self.mode == 0) & (self.command_target.abs().amax(-1) < .01)
                supported = reward_support_tracking.bool() & contact.all(-1)
                components["stationary_anchor"] = self.stationary_anchor.reward(local[:, :2], requested, supported)
                components["stand_translation"] *= torch.where(requested,
                    self.cfg["stationary_tracking"]["velocity_scale"], 1.)
            if self.cfg.get("zero_command_velocity_scale"):
                requested = (self.commands[:, :2].abs() < .01).all(-1) & (self.mode != 4) & ordinary
                components["stand_translation"] *= torch.where(requested, self.cfg["zero_command_velocity_scale"], 1.)
            if self.cfg.get("height_tracking") and not self.cfg.get("dense_tracking", {}).get("enabled"):
                settings = self.cfg["height_tracking"]
                moving_height = self.skills.height_motion
                active = reward_support_tracking * moving_height
                error = height - self.commands[:, 2]
                vz_error = vertical_velocity - self.skills.height_velocity_reference
                components.update(height_tracking_terms(error, vz_error, reward_support_tracking, moving_height, settings))
                components["vertical_velocity"] = torch.where(moving_height, -.5 * active * vz_error.square(),
                                                              components["vertical_velocity"])
            if self.cfg.get("landing_tracking"):
                settings = self.cfg["landing_tracking"]
                landing = ((self.phase.phase == Phase.LANDING) | (self.phase.phase == Phase.RECOVERY)).float()
                if settings.get("passive_semantics") == "finite_task_cost_v1":
                    from wheeled_tasks.chassis.tasks.rewards.rewards import passive_landing_costs
                    eligible = (self.skills.landing if self.skills is not None else torch.zeros_like(height, dtype=torch.bool)) & (self.mode != 4)
                    self.landing_reward_mask = self.passive_landing.update(self.phase.phase, eligible)
                    h_ref, v_ref = self.full_tasks.landing_reference(self.commands[:, 2], self.passive_landing.elapsed)
                    components.update(passive_landing_costs(height - h_ref, vertical_velocity - v_ref,
                                                            self.landing_reward_mask, settings))
                else:
                    # Preserve archived contracts and their frozen comparisons.
                    h_ref, v_ref = self.full_tasks.landing_reference(self.commands[:, 2], self.phase.contact_time)
                    passive_landing = landing * (self.mode != 4)
                    self.landing_reward_mask = passive_landing
                    components["landing_height_tracking"] = passive_landing * settings["height_weight"] * torch.exp(-((height - h_ref) / .035).square())
                    components["landing_velocity_tracking"] = passive_landing * settings["velocity_weight"] * torch.exp(-((vertical_velocity - v_ref) / .35).square())
                weight = self.body_mass.sum(-1) * 9.81
                components["landing_impact"] = -landing * settings["impact_weight"] * (self.contact_peak / weight - 2.).clamp_min(0.).square()
            applied_wheel = self.robot.data.applied_torque.torch[:, [self.ids[2], self.ids[5]]]
            air_task = (self.mode == 4) | (self.skills.landing if self.skills is not None else False)
            air_mask = flight * air_task * (self.phase.air_time < 1.)
            components["air_wheel_centering"] = 10. * air_mask * torch.exp(-wheel_b[:, :, 0].square().mean(-1) / .02)
            components["air_low_wheel_torque"] = air_mask * torch.exp(-(applied_wheel / 3.84).square().mean(-1))
            # Active jump retraction is already included in dense_jump_reward.
            components["air_leg_retraction"] = air_mask * (self.mode != 4) * torch.exp(-((extension - .20) / .05).square())
            for term in ("motor_torque", "wheel_power"):
                components[term] *= self.cfg.get("scut_effort_reward_scale", 1.)
            if self.references is not None:
                from wheeled_tasks.chassis.tasks.rewards.rewards import reference_motion_terms
                components.pop("lin_vel_square")
                components.update(reference_motion_terms(velocity, gravity, self.commands, reward_support_tracking,
                    ordinary, self.references, self.cfg["reference_reward"],
                    locomotion_posture=self.cfg.get("dense_tracking", {}).get("locomotion_posture")))
                for term in ("track_height", "height_square"):
                    components[term] *= ~self.references.jumping
                push = self.references.phase == self.references.PUSH
                scale = torch.where(push, self.cfg["reference_reward"]["push_soft_cost_scale"], 1.)
                for term in ("motor_torque", "wheel_power", "action_rate", "leg_action_smoothness", "wheel_action_smoothness"):
                    components[term] *= scale
                if self.cfg.get("manual_context35"):
                    from wheeled_tasks.chassis.tasks.rewards.rewards import apply_manual_tracking
                    apply_manual_tracking(components, velocity, omega, gravity, self.commands,
                        reward_support_tracking, ordinary, self.references, self.cfg["reference_reward"])
                if self.cfg.get("precision_tracking") and not self.cfg.get("dense_tracking", {}).get("enabled"):
                    from wheeled_tasks.chassis.tasks.rewards.rewards import precision_tracking_terms
                    components.update(precision_tracking_terms(velocity, gravity, height, self.commands,
                        reward_support_tracking, ordinary, self.references, wheel_b, self.cfg["precision_tracking"]))
            if self.cfg.get("dense_tracking", {}).get("enabled"):
                from wheeled_tasks.chassis.tasks.rewards.rewards import apply_dense_tracking
                spin_error = ((self.skills.reference_filtered - self.skills.reference_target).norm(dim=-1)
                              if spin_reward else None)
                apply_dense_tracking(components, velocity, omega, gravity, height, self.commands,
                    reward_support_tracking, ordinary, self.references, self.cfg["dense_tracking"],
                    spin_velocity_error=spin_error)
            if (self.recovery_hold_cost_settings is not None
                    and self.recovery_hold_cost_settings["enabled"]):
                from wheeled_tasks.chassis.tasks.rewards.rewards import apply_recovery_hold_costs
                apply_recovery_hold_costs(components, dq, omega, self.recovery.enabled,
                    self.recovery.learning_mask, self.recovery.policy_motion_hold,
                    self.recovery_hold_cost_settings, diagnostics=self.recovery_hold_cost_diagnostics)
            if self.recovery_takeover_settings is not None:
                from wheeled_tasks.chassis.tasks.rewards.rewards import apply_recovery_takeover
                apply_recovery_takeover(components, gravity, omega, height, velocity, self.commands, dq,
                    self.recovery.enabled & self.recovery.learning_mask,
                    self.recovery.policy_motion_hold, self.recovery_takeover_settings)
            if self.step_climb is not None and self.step_climb.has_tasks:
                from wheeled_tasks.chassis.tasks.rewards.rewards import apply_climb_height_tracking
                scale = self.step_climb.height_tracking_scale(self.wheel_centers() - self.origins[:, None, :])
                apply_climb_height_tracking(components, scale)
            reward = torch.stack(list(components.values())).sum(0) * self.policy_dt
            if measure_payments:
                final_payments.update({"legacy/base/" + name: value * self.policy_dt
                    for name, value in components.items()})
            if spin_reward and not self.cfg.get("dense_tracking", {}).get("enabled"):
                reward += self.skills.spin * velocity_reward * self.policy_dt
                if measure_payments:
                    final_payments["legacy/spin_extra"] = self.skills.spin * velocity_reward * self.policy_dt
        self.previous_motor_velocity.copy_(dq)
        success_now = (self.mode >= 2) & (local[:, 0] > 1.8) & contact.all(-1) & stable
        success_now &= (self.mode == 3) | self.phase.flew
        if self.full_tasks is not None and (self.has_route_tasks or self.has_jump_tasks):
            centers = self.wheel_centers() - self.origins[:, None, :]
            support_heights = torch.stack([self.ground_height(centers[:, side, :2]) for side in range(2)], -1)
            clearance = centers[:, :, 2] - .06 - support_heights
            vertical_velocity = wp.to_torch(self.robot.root_view.get_root_velocities())[:, 2]
            armed = self.references.jumping if self.references is not None else None
            self.full_tasks.observe(self.mode, height, clearance, self.phase.phase, vertical_velocity, requested=armed)
            if self.has_jump_tasks and self.cfg.get("task_semantics", {}).get("jump_com_rise_m") is not None:
                com_height = (self.robot.data.body_com_pose_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                com_vz = (self.robot.data.body_com_lin_vel_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                self.full_tasks.observe_com(self.mode, self.phase.phase, com_height, com_vz, requested=armed,
                                            contacts=contact if self.references is not None else None)
            if self.has_jump_tasks:
                if self.references is not None:
                    from wheeled_tasks.chassis.tasks.rewards.rewards import reference_jump_terms
                    terms = reference_jump_terms(self.references, self.full_tasks, self.phase.phase, contact,
                        height, com_vz, velocity, gravity, extension, local, self.commands,
                        landing_contact_time=self.phase.contact_time)
                    if self.jump_landing is not None:
                        apex = self.references.release_speed.square() / (2 * 9.81)
                        apex = torch.as_tensor(apex, device=self.device).expand(self.num_envs)
                        self.jump_landing.update(self.references.jumping, self.phase.phase, contact,
                            self.full_tasks.com_released, self.full_tasks.clear_air_time_peak,
                            self.raw_height, self.commands, apex, height=height, com_vz=com_vz,
                            landing_delta=self.references.landing_delta, origin_xy=local[:, :2],
                            com_rise=self.full_tasks.com_rise,
                            step_jump=self.step_jump.enabled if self.step_jump is not None else None)
                        # The six legacy components depend on a nominal command
                        # phase that may never overlap real touchdown. Replace
                        # only landing payment; preload/release/air terms stay.
                        for name in ("jump_landing_height", "jump_recovery_height",
                                     "jump_landing_vertical_velocity", "jump_landing_velocity",
                                     "jump_landing_attitude", "jump_landing_position"):
                            terms[name] = torch.zeros_like(height)
                        terms.update(self.jump_landing.reward_terms(height, com_vz, velocity,
                            omega, gravity, dq[:, (2, 5)],
                            self.robot.data.applied_torque.torch[:, self.ids][:, (2, 5)],
                            self.wheel_force_peak_n, self.body_mass.sum(-1) * 9.81,
                            position_xy=local[:, :2], wheel_clearance=clearance))
                    jump_reward = torch.stack(list(terms.values())).sum(0) * self.policy_dt
                    if measure_payments:
                        final_payments.update({"legacy/jump/" + name: value * self.policy_dt
                            for name, value in terms.items()})
                    if components is not None:
                        components.update(terms)
                else:
                    jump_reward = self.full_tasks.dense_jump_reward(self.mode, self.phase, height, vertical_velocity, self.commands[:, 2], extension)
                    if measure_payments:
                        final_payments["legacy/dense_jump"] = jump_reward.clone()
                reward += jump_reward
                if components is not None and self.references is None:
                    components["dense_jump"] = jump_reward / self.policy_dt
            progress_reward = self.full_tasks.route_progress_reward(self.mode, local)
            reward += progress_reward
            if measure_payments:
                final_payments["legacy/route_progress"] = progress_reward.clone()
            if components is not None:
                components["route_progress"] = progress_reward / self.policy_dt
            success_now = self.full_tasks.completion(self.mode, local, centers[:, :, 0], self.route_goal,
                                                     contact, stable, self.phase.phase, self.commands[:, 2], velocity[:, :2].norm(dim=-1))
        if self.skills is not None:
            success_now |= self.skills.completion(contact, stable)
        if self.step_jump is not None and self.step_jump.has_tasks:
            centers = self.wheel_centers() - self.origins[:, None, :]
            self.step_jump.observe(centers, contact, com_vz, velocity[:, :2].norm(dim=-1),
                gravity, stable, self.jump_requested)
            terms = self.step_jump.reward_terms(centers, gravity)
            reward += torch.stack(list(terms.values())).sum(0) * self.policy_dt
            if measure_payments:
                final_payments.update({"legacy/step_jump/" + name: value * self.policy_dt
                    for name, value in terms.items()})
            if components is not None:
                components.update(terms)
            success_now[self.step_jump.enabled] = self.step_jump.success[self.step_jump.enabled]
        if self.step_climb is not None and self.step_climb.has_tasks:
            centers = self.wheel_centers() - self.origins[:, None, :]
            # Climbing need not maintain a flat-ground nominal body height
            # while straddling a riser or resting its chassis on the top face.
            climb_stable = (gravity[:, 2] < -math.cos(math.radians(20))) & (omega[:, :2].norm(dim=-1) < 1.)
            self.step_climb.observe(centers, contact, climb_stable,
                body_bottom_points=self.base_support_points, body_contact_forces=self.base_support_forces)
            terms = self.step_climb.reward_terms(centers, gravity)
            reward += torch.stack(list(terms.values())).sum(0) * self.policy_dt
            if measure_payments:
                final_payments.update({"legacy/step_climb/" + name: value * self.policy_dt
                    for name, value in terms.items()})
            if components is not None:
                components.update(terms)
            success_now[self.step_climb.assessing] = self.step_climb.success[self.step_climb.assessing]
        if self.recovery is not None:
            centers = self.wheel_centers() - self.origins[:, None, :]
            surfaces = torch.stack([self.ground_height(centers[:, side, :2]) for side in range(2)], -1)
            platform = self.recovery.required_platform_height[:, None]
            retained = ((surfaces - platform).abs().amax(-1) < .001)
            retained &= ((centers[:, :, 2] - self.recovery.geometry.wheel_radius_m - platform).abs().amax(-1) < .025)
            retained &= centers[:, :, 0].amin(-1) >= self.recovery.min_ending_x
            recovery_success = self.recovery.outcome(contact, undesired, gravity, omega, height, velocity,
                                                     goal_platform=retained)
            success_now[self.recovery.enabled] = recovery_success[self.recovery.enabled]
        self.success_hold = torch.where(success_now, self.success_hold + self.policy_dt, 0.)
        success = self.success_hold >= self.cfg.get("task_semantics", {}).get("success_hold_seconds", 1.)
        if self.recovery is not None:
            success[self.recovery.enabled] = recovery_success[self.recovery.enabled]
        reasons = {
            "fall": self.fall_confirmation.update(gravity[:, 2], fall_height, masks["flight"]),
            "nonwheel_contact": undesired & (self.episode_length_buf * self.policy_dt > .2),
            "knee": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "boundary": (local[:, 0].abs() > 3.6) | (local[:, 1].abs() > 1.7),
            "takeoff_timeout": (self.phase.phase == Phase.TAKEOFF) & (self.phase.time > 1.),
        }
        if self.cfg.get("closure_gap_termination_m") and not self.cfg.get("formal_closure_monitor_only"):
            reasons["closure_gap"] = gap > self.cfg["closure_gap_termination_m"]
        if self.cfg.get("evaluation_long_corridors"):
            x_limit = local.new_tensor([self.cfg.get("flat_half_length_m", 39.) - 4.
                                       if kind in ("flat", "jump") else 3.6 for kind in self.kinds])
            y_limit = local.new_tensor([self.cfg.get("flat_floor_width_m", 4.) / 2 - .3
                                       if kind in ("flat", "jump") else 1.7 for kind in self.kinds])
            if self.cfg.get("playback_open_ground"):
                y_limit = 35.
            release = getattr(self.skills, "release_stop", None)
            if release is not None:
                extended = release.active & self.release_stop_terrain
                x_limit[extended] = release.settings["terrain_half_length_m"] - .4
            if self.route_runout_mask is not None:
                x_limit = apply_route_boundary(local[:, 0], x_limit,
                    self.route_runout_mask, self.route_runout_settings)
            reasons["boundary"] = (local[:, 0].abs() > x_limit) | (local[:, 1].abs() > y_limit)
        if getattr(self, "shared_terrain", None) is not None:
            reasons["boundary"] = self.shared_terrain.boundary(local + self.origins)
        if self.full_tasks is not None:
            elapsed = self.episode_length_buf * self.policy_dt
            jumping = self.mode == 4
            reasons["jump_request_timeout"] = jumping & (elapsed > 2.) & ~self.phase.flew
            reasons["jump_outcome_timeout"] = jumping & (elapsed > 6.) & ~success
            if self.references is not None:
                r = self.references
                reasons["takeoff_timeout"] = (r.jumping & (r.elapsed > r.push_end + self.cfg["command_reference"]["takeoff_margin_seconds"])
                    & ~self.full_tasks.com_released)
                reasons["jump_request_timeout"].zero_()
                reasons["jump_outcome_timeout"] = r.jumping & (r.elapsed > 6.) & ~success
                if self.jump_landing is not None:
                    reasons["jump_outcome_timeout"] = self.jump_landing.outcome_timeout & ~success
        for name, index in zip(("L_joint2", "R_jonit2"), self.knee_ids):
            lo, hi = self.knee_bounds[name]
            position = self.robot.data.joint_pos.torch[:, index]
            reasons["knee"] |= (position < lo - 0.03) | (position > hi + 0.03)
        raw_margin_risk = None
        if self.is_v5:
            compression, _ = self.v5.spring_state(self.robot.data.joint_pos.torch[:, self.spring_ids],
                                                 self.robot.data.joint_vel.torch[:, self.spring_ids])
            reasons["spring_travel"] = ((compression < -.001) | (compression > self.v5.stroke + .001)).any(-1)
            self.spring_reserve_frames += int((compression > .072).any(-1).sum())
            if "working_margin_weight" in self.cfg:
                # One normalized working-margin cost avoids double-penalizing
                # a knee stop and its mechanically coupled spring compression.
                risk = self.v5.working_margin_risk(self.robot.data.joint_pos.torch[:, self.knee_ids],
                    self.robot.data.joint_pos.torch[:, self.spring_ids], self.cfg["knee_working_margin_rad"])
                raw_margin_risk = risk
                if self.height_margin is not None:
                    risk = self.height_margin.excess(risk, self.commands[:, 2])
                margin_density = self.cfg["working_margin_weight"] * support_tracking * risk.square().mean(-1)
                reward += margin_density * self.policy_dt
                if measure_payments:
                    final_payments["legacy/working_margin"] = margin_density * self.policy_dt
                if components is not None:
                    components["working_margin"] = margin_density
            else:
                reward -= 2 * ((compression / self.v5.stroke - .9).clamp_min(0)).square().sum(-1) * self.policy_dt
                if measure_payments:
                    final_payments["legacy/spring_reserve"] = -2 * ((compression / self.v5.stroke - .9).clamp_min(0)).square().sum(-1) * self.policy_dt
        # Leaving a finite terrain tile is a collection truncation, not a fall.
        if self.step_assist is not None:
            reasons["blocked"] = self.step_assist.wall.clone()
        if self.recovery is not None:
            # Contact with the chassis and inverted attitude belong to the
            # scripted prelude. Mechanical limits still apply throughout.
            prelude = self.recovery.prelude
            for name in ("fall", "nonwheel_contact", "takeoff_timeout", "jump_request_timeout", "jump_outcome_timeout"):
                if name in reasons:
                    reasons[name] &= ~prelude
            self.fall_confirmation.reset(prelude.nonzero().flatten())
            reasons.update(self.recovery.failure_reasons())
        terminated, timeouts, success = episode_outcome(reasons, success, self.episode_length_buf >= self.episode_limits)
        success_event = success & ~self.sequence_success_latched
        self.sequence_success_latched |= success
        self.successful_jump_count += (success_event & (self.mode == 4)).long()
        from wheeled_tasks.chassis.tasks.rewards.rewards import episode_event_terms
        event_terms = episode_event_terms(success_event, terminated, self.cfg)
        reward += event_terms["success_event"]
        sequence_pending = self.skills.sequence_pending() if self.skills is not None else torch.zeros_like(success)
        done = terminated | timeouts | (success & ~sequence_pending)
        if self.performance_curriculum is not None:
            reference_error = (self.skills.reference_filtered - self.skills.reference_target).norm(dim=-1)
            reference_error = torch.where(self.skills.spin, reference_error, velocity[:, 0] - self.commands[:, 0])
            self.performance_curriculum.observe(height - self.commands[:, 2], velocity[:, 0] - self.commands[:, 0],
                omega[:, 2] - self.commands[:, 1], support_tracking.bool() & contact.all(-1), done,
                terminated | reasons["boundary"], self.training_transitions / self.cfg["curriculum_reference_batch"],
                reference_velocity_error=reference_error, commands=self.commands)
        reward += event_terms["failure_event"]
        if measure_payments:
            final_payments["event/success"] = event_terms["success_event"].clone()
            final_payments["event/failure"] = event_terms["failure_event"].clone()
        # Keep integer reductions and the original counter order; one host
        # transfer avoids a separate device synchronization for each count.
        counts = torch.stack([*(mask.sum() for mask in reasons.values()),
                              success_event.sum(), timeouts.sum()]).cpu().tolist()
        for name, count in zip(reasons, counts[:-2]):
            self.termination_counts[name] += count
        self.success_count += counts[-2]
        self.timeout_count += counts[-1]
        extras = {"time_outs": timeouts, "log": {"/task/pin_gap_m": gap.mean(),
            "/task/height_error_m": (height - self.commands[:, 2]).abs().mean(),
            "/task/success": success.float().mean(), "/task/flight": flight.mean(),
            "/task/wheel_contact": contact.float().mean()}}
        extras["log"].update({"/reward/" + name: value.mean() for name, value in event_terms.items()})
        extras["log"].update(self.sampling_metrics)
        if self.jump_landing is not None:
            extras["log"].update({"/jump_contact/" + name: value.float().mean()
                for name, value in self.jump_landing.diagnostics().items()})
        if self.skills is not None:
            extras["log"].update(self.skills.difficulty_metrics())
            extras["log"].update(self.skills.large_stair_metrics())
        if self.recovery is not None:
            extras["policy_learning_mask"] = self.recovery.learning_mask.clone()
            extras["log"].update(self._recovery_metrics(done, success, terminated, timeouts))
        if self.jump_assist is not None:
            extras["log"].update(self.jump_assist.metrics())
            unassisted = ~self.jump_assist.assisted
            extras["log"]["/task/unassisted_success"] = (success & unassisted).float().mean()
            extras["log"]["/task/unassisted_flight"] = (masks["flight"] & unassisted).float().mean()
        if self.skills is not None and self.skills.schedule.enabled:
            extras["log"].update({"/curriculum/stage_actor_updates": float(self.stage_actor_update),
                "/curriculum/task_actor_updates": float(stage_actor_updates(self)),
                "/curriculum/global_actor_updates": float(self.global_actor_update),
                "/curriculum/unilateral_degradation_fraction": self.skills.schedule.degraded.float().mean(),
                "/curriculum/edge_reset_fraction": self.skills.schedule.edge_reset.float().mean(),
                "/curriculum/request_stress_fraction": self.skills.schedule.request_stress.float().mean()})
        if self.usb_transport is not None:
            extras["log"].update(self.usb_transport.metrics())
        if self.command_transport is not None:
            extras["log"].update(self.command_transport.metrics())
        if self.feedback_transport is not None:
            extras["log"].update(self.feedback_transport.metrics())
        if self.signal_delay is not None:
            extras["log"].update(self.signal_delay.metrics())
        if self.round_trip_transport is not None:
            extras["log"].update(self.round_trip_transport.metrics())
        if getattr(self, "actor_sensor_noise", None) is not None:
            extras["log"].update(self.actor_sensor_noise.metrics())
        if getattr(self, "policy_signal_delay", None) is not None:
            extras["log"].update(self.policy_signal_delay.metrics())
        if getattr(self, "episode_spring_strength_settings", None) is not None:
            extras["log"].update({"/domain/spring_force_scale_mean": self.spring_strength.mean(),
                "/domain/spring_force_scale_min": self.spring_strength.min(),
                "/domain/spring_force_scale_max": self.spring_strength.max()})
        if self.dynamics_randomization is not None:
            extras["log"].update(self.dynamics_randomization.metrics())
        if self.references is not None:
            extras["log"].update(self.references.metrics())
        if self.step_assist is not None:
            extras["log"].update(self.step_assist.metrics())
            extras["log"]["/commands/height_assist_bias_mean_m"] = (self.commands[:, 2] - self.raw_height).mean()
        if self.nonstandard_start is not None:
            extras["policy_learning_mask"] = (extras.get("policy_learning_mask",
                torch.ones_like(self.nonstandard_start.learning_mask)) & self.nonstandard_start.learning_mask)
            extras["log"].update(self.nonstandard_start.metrics((self._curriculum_clock("nonstandard_start") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self))))
        if getattr(self, "ordinary_posture", None) is not None:
            extras["log"].update(self.ordinary_posture.metrics((self._curriculum_clock("ordinary_posture_curriculum") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self))))
        if self.cfg.get("fall_confirmation_seconds", 0.) > 0:
            extras["log"].update({
                "/termination/fall_candidate_fraction": self.fall_confirmation.candidate.float().mean(),
                "/termination/fall_confirmed_fraction": reasons["fall"].float().mean(),
                "/termination/nonwheel_contact_fraction": reasons["nonwheel_contact"].float().mean()})
        if components is not None:
            extras["log"].update({"/reward/" + name: value.mean() * self.policy_dt for name, value in components.items()})
            # Expose the computed components to any rollout audit. A new training
            # entry point must not need a second producer-side feature whitelist.
            extras["native_reward_components"] = components
            if self.cfg.get("diagnostic_logging"):
                self._accumulate_diagnostic_rewards(components, support_tracking, ordinary)
        for task in (self.step_jump, self.step_climb):
            if task is not None and task.has_tasks:
                extras["log"].update({"/task/" + name: value.float().mean()
                    for name, value in task.diagnostics().items() if value.dtype == torch.bool or "count" in name})
        if raw_margin_risk is not None:
            extras["log"]["/mechanics/working_margin_raw_max"] = raw_margin_risk.max()
        if self.performance_curriculum is not None:
            extras["log"].update({
                "/curriculum/applied_height_scale_mean": self.performance_curriculum.env_height_scale.mean(),
                "/curriculum/applied_height_width_m_mean": self.performance_curriculum.env_height_width.mean()})
        if self.cfg.get("height_tracking"):
            extras["log"].update({"/task/height_command_min_m": self.commands[:, 2].min(),
                                   "/task/height_command_max_m": self.commands[:, 2].max()})
        if self.cfg.get("special_mode_activation"):
            settings = self.cfg["special_mode_activation"]["spin_translate"]
            self.skills.stable.copy_(contact.all(-1) & support_tracking.bool()
                & (gravity[:, :2].norm(dim=-1) <= settings["gravity_xy_max"])
                & ((height - self.commands[:, 2]).abs() <= settings["height_error_m"]))
            extras["log"]["/commands/spin_translate_active_fraction"] = self.skills.spin.float().mean()
        if self.cfg.get("record_diagnostics", False):
            # Capture before auto-reset mutates commands, episode lengths and robot state.
            extras["diagnostics"] = {"velocity": velocity.clone(), "omega": omega.clone(),
                "gravity": gravity.clone(), "height": height.clone(), "position": local.clone(),
                "commands": self.commands.clone(), "episode_ticks": self.episode_length_buf.clone(),
                "motor_position": q.clone(), "motor_velocity": dq.clone(),
                "motor_effort": self.robot.data.applied_torque.torch[:, self.ids].clone(),
                "reward": reward.clone(), "gap": gap.clone(), "done": done.clone(),
                "terminated": terminated.clone(), "success": success.clone(),
                "reasons": {name: mask.clone() for name, mask in reasons.items()}}
            extras["diagnostics"]["successful_jump_count"] = self.successful_jump_count.clone()
            extras["diagnostics"].update(physical_phase=self.phase.phase.clone(),
                physical_phase_time_s=self.phase.time.clone(),
                landing_reward_mask=self.landing_reward_mask.clone())
            if self.passive_landing is not None:
                extras["diagnostics"]["landing_window_elapsed_s"] = self.passive_landing.elapsed.clone()
                extras["diagnostics"]["landing_task_eligible"] = (
                    (self.skills.landing if self.skills is not None else torch.zeros_like(height, dtype=torch.bool)) & (self.mode != 4)).clone()
            if self.jump_landing is not None:
                extras["diagnostics"].update(self.jump_landing.diagnostics())
            extras["diagnostics"]["planar_speed_world"] = root_link_linear_velocity(self)[:, :2].norm(dim=-1).clone()
            if self.cfg.get("diagnostic_trace"):
                extras["diagnostics"].update(leg_target_position=legs.clone(),
                    requested_motor_effort=self.v5.requested_motor_effort.clone(),
                    motor_effort_bounds=self.v5.current_motor_bounds.clone())
            if self.full_tasks is not None:
                extras["diagnostics"].update(jump_clearance_peak=self.full_tasks.clearance_peak.clone(),
                    jump_air_time_peak=self.full_tasks.clear_air_time_peak.clone(), jump_height_peak=self.full_tasks.height_peak.clone(),
                    jump_release_velocity=self.full_tasks.release_velocity.clone(),
                    jump_com_rise=self.full_tasks.com_rise.clone(), jump_com_release_speed=self.full_tasks.com_release_speed.clone())
            if self.references is not None:
                measured_com_vz = (self.robot.data.body_com_lin_vel_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                extras["diagnostics"].update(reference_phase=self.references.phase.clone(),
                    reference_height=self.references.height.clone(), reference_vz=self.references.vertical_velocity.clone(),
                    reference_ax=self.references.acceleration.clone(), physical_phase=self.phase.phase.clone(),
                    jump_requested=self.references.jumping.clone(), raw_height_command=self.raw_height.clone(),
                    whole_com_vz=measured_com_vz, wheel_target_velocity=wheels.clone(),
                    motor_velocity=dq.clone(), actions=self.actions.clone(),
                    reference_height_error=torch.where(self.references.jumping,
                        self.full_tasks.com_displacement - self.references.height_delta,
                        height - self.references.height), com_displacement=self.full_tasks.com_displacement.clone())
            if self.skills is not None:
                extras["diagnostics"].update(self.skills.diagnostics(velocity,
                    omega=omega, height_error=height - self.commands[:, 2], gravity=gravity))
            for task in (self.step_jump, self.step_climb):
                if task is not None and task.has_tasks:
                    extras["diagnostics"].update(task.diagnostics())
            if self.step_sensors is not None:
                extras["diagnostics"].update(self.step_sensors.diagnostics())
            if self.recovery is not None:
                extras["diagnostics"].update(self.recovery.diagnostics())
            if self.nonstandard_start is not None:
                extras["diagnostics"].update(self.nonstandard_start.diagnostics())
            if self.step_assist is not None:
                extras["diagnostics"].update(raw_height_command=self.raw_height.clone(),
                    step_assist_active=self.step_assist.active.clone(), step_detected=self.step_assist.triggered.clone())
            if self.cfg.get("height_tracking"):
                extras["diagnostics"]["height_velocity_error"] = (
                    vertical_velocity - self.skills.height_velocity_reference) * self.skills.height_motion
        if self.step_assist is not None or self.references is not None:
            self.commands[:, 2] = self.raw_height
        if self.jump_landing is not None:
            self.jump_landing.close(done, success)
        if getattr(self, "jump_actuation_measurement", None) is not None:
            payload = self.jump_actuation_measurement.policy_transition(done, terminated, timeouts)
            payload["policy_interval_id"] = self.tick
            # Plain jumps have no scripted birth owner: absent masks retain
            # PPO's existing all-unmasked behavior without changing its inputs.
            payload["policy_learning_mask"] = extras.get("policy_learning_mask", torch.ones_like(done)).clone()
            payload["policy_learning_mask_source"] = ("explicit_environment_mask"
                if "policy_learning_mask" in extras else "PPO_default_all_unmasked")
            extras["jump_actuation_transition"] = payload
            extras["log"].update(self.jump_actuation_measurement.metrics())
        if getattr(self, "jump_physics_trace", None) is not None:
            extras["jump_physics_trace"] = self.jump_physics_trace.policy_transition(
                done, terminated, timeouts, jump_transition=extras["jump_actuation_transition"])
            extras["log"].update(self.jump_physics_trace.metrics())
        if self.recovery is not None and self.recovery.has_tasks:
            payload = self.recovery.policy_transition(done, terminated, timeouts)
            # The final mask also includes nonstandard-start holds. A producer's
            # recovery-only mask cannot substitute for the mask sent to PPO.
            payload["policy_learning_mask"] = extras["policy_learning_mask"].clone()
            payload["policy_interval_id"] = self.tick
            extras["recovery_transition"] = payload
        if measure_payments:
            self._capture_reward_episode_payment(extras, final_payments, done, terminated,
                timeouts, success, reasons["boundary"])
        self._advance_policy_state(done, terminated, extras)
        if measure_payments:
            self._finish_reward_episode_payment(extras)
        if "recovery_transition" in extras:
            payload = extras["recovery_transition"]
            payload["returned_episode_ids"] = self.recovery.episode_ids.clone()
            payload["reset_mask"] = payload["returned_episode_ids"] != payload["episode_ids"]
        if "jump_actuation_transition" in extras:
            payload = extras["jump_actuation_transition"]
            payload["returned_episode_ids"] = self.jump_actuation_measurement.episode_ids.clone()
            payload["reset_mask"] = payload["returned_episode_ids"] != payload["episode_ids"]
        if "jump_physics_trace" in extras:
            payload = extras["jump_physics_trace"]
            payload["returned_episode_ids"] = self.jump_physics_trace.episode_ids.clone()
            payload["reset_mask"] = payload["returned_episode_ids"] != payload["episode_ids"]
        return self.get_observations(), reward, done, extras

    def _finish_scut_flat_step(self):
        """Exclusive final payments; no legacy density, events or tail settlement."""
        source = self.scut_reward
        data = self.robot.data
        velocity = data.root_com_lin_vel_b.torch
        omega = data.root_com_ang_vel_b.torch
        gravity = data.projected_gravity_b.torch
        root_pose = data.root_link_pose_w.torch
        local = root_pose[:, :3] - self.origins
        height = local[:, 2]
        recovery = getattr(self, "recovery", None)
        scripted = getattr(recovery, "source_flat_handoff", False)
        extended_recovery_window = scripted and getattr(recovery, "episode_window", None) is not None
        if extended_recovery_window:
            recovery.advance_episode_window(recovery.learning_mask & self.flat_episode.learning_mask)
        scripted_interval = recovery.enabled & ~recovery.learning_mask if scripted else None
        terminated, numerical, confirmed = source.termination(velocity, omega, root_pose[:, 3:7],
            **({"scripted_interval": scripted_interval} if scripted else {}))
        numerical_capture = getattr(self, "_numerical_failure_capture", None)
        if numerical_capture is not None:
            numerical_capture.observe_original_policy_end(self.physical_tick, self.tick, numerical)
        gap = self.closure_gap_peak
        # These pre-existing local mechanical guards are NOT source numeric
        # reset thresholds. Source numeric failures may reset a single lane;
        # other corrupt model/sensor inputs abort rather than fabricate data.
        valid_model = (torch.isfinite(root_pose).all(-1)
                       & torch.isfinite(data.joint_pos.torch).all(-1)
                       & torch.isfinite(gravity).all(-1)
                       & torch.isfinite(source.net_contact).flatten(1).all(-1)
                       & torch.isfinite(gap))
        if bool((~valid_model & ~numerical).any()):
            raise RuntimeError("Nonfinite reference physical mapping outside the source numerical reset mask")
        reward_ground_height_world = None
        if self.cfg.get("reward_profile") in ("scut0730_rough_v1", "v7_final_rough_v1"):
            from wheeled_tasks.chassis.simulation.terrain.rough_support import base_ground_scan
            terrain = self.shared_terrain
            reward_raycast = ((lambda starts, directions: terrain.raycast(starts, directions)[0])
                if getattr(self, "evaluation_surface", "training") == "plane_regression_v1" else
                (lambda starts, directions: terrain.raycast(starts, directions, terrain.warp_mesh)[0]))
            reward_ground_height_world, rough_scan_hit_count = base_ground_scan(
                root_pose, self.origins[:, 2], reward_raycast)
            height = source.reward_height(root_pose, self.origins, reward_ground_height_world)
        finite_gap = torch.where(torch.isfinite(gap), gap, 0.)
        peak, all_finite_peak, numerical_peak = torch.stack((
            torch.where(numerical, 0., gap).max(),
            finite_gap.max(),
            torch.where(numerical, finite_gap, 0.).max(),
        )).cpu().tolist()
        self.max_closure_gap = max(self.max_closure_gap, peak)
        # Keep the old guard statistic; report excluded reset lanes separately.
        self.max_closure_gap_all_finite = max(self.max_closure_gap_all_finite, all_finite_peak)
        self.max_closure_gap_numerical_reset = max(self.max_closure_gap_numerical_reset, numerical_peak)
        gap_reset_enabled = (bool(self.cfg.get("closure_gap_termination_m"))
                             and not self.cfg.get("formal_closure_monitor_only")
                             and not self.cfg.get("formal_closure_observation_only"))
        # Capture the first exceedance even in monitor-only mode. Repeated
        # lane resets must not copy the full scene to CPU on every policy step.
        capture_threshold = self.cfg.get("closure_gap_termination_m") or .03
        gap_failure = peak > capture_threshold and not getattr(self, "_closure_failure_saved", False)
        if self.cfg.get("formal_physics_trace_directory") and (
                self.tick == 1 or gap_failure or self.tick in self.cfg.get("formal_physics_trace_ticks", ())):
            name = ("closure_failure" if gap_failure else
                    "after_first_step" if self.tick == 1 else f"physics_step_{self.tick:04d}")
            self._save_formal_physics_state(name)
            if gap_failure:
                self._closure_failure_saved = True
        if (self.cfg.get("formal_v7_flat") and not self.cfg.get("formal_closure_monitor_only")
                and not gap_reset_enabled and peak > .03):
            raise RuntimeError(f"Formal V7 severe closed-chain gap: {peak} m; state saved before reset")
        if (not self.cfg.get("formal_closure_monitor_only") and self.max_closure_gap > .003
                and not self.cfg.get("closure_gap_termination_m")):
            raise RuntimeError(f"Closed-chain numerical gap exceeds 3 mm: {self.max_closure_gap}")
        reasons = {"source_attitude_or_contact": confirmed.clone(),
                   "numerical_safety": numerical.clone(),
                   "knee": torch.zeros_like(terminated),
                   "boundary": (local[:, 0].abs() > 3.6) | (local[:, 1].abs() > 1.7)}
        if scripted:
            contact = self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.
            undesired = (self.contact_force[:, self.nonwheel_ids].norm(dim=-1) > 5.).any(-1)
            recovery.outcome(contact, undesired, gravity, omega, height, velocity)
            reasons.update(recovery.failure_reasons())
        if self.cfg.get("closure_gap_termination_m") and not self.cfg.get("formal_closure_monitor_only"):
            reasons["closure_gap"] = gap > self.cfg["closure_gap_termination_m"]
            if self.cfg.get("formal_closure_observation_only"):
                # Physics-only diagnosis observes drift without censoring it
                # at 3 mm. The 30 mm severe stop above remains in force.
                reasons["closure_gap"].zero_()
        if self.cfg.get("evaluation_long_corridors"):
            x_limit = self.cfg.get("flat_half_length_m", 39.) - 4.
            y_limit = self.cfg.get("flat_floor_width_m", 4.) / 2. - .3
            reasons["boundary"] = (local[:, 0].abs() > x_limit) | (local[:, 1].abs() > y_limit)
        if getattr(self, "shared_terrain", None) is not None:
            reasons["boundary"] = self.shared_terrain.boundary(root_pose[:, :3])
        probe = getattr(self, "_formal_plane_contact_probe", None)
        if probe is not None:
            ids = probe["env_ids"]
            outside = torch.stack((local[ids[0], 0].abs() > 44., local[ids[1], 1].abs() > 4.))
            wheel_force = self.contact_force[ids][:, self.wheel_ids].norm(dim=-1).amax(-1)
            probe["outside_frames"] += outside.long()
            probe["outside_peak_wheel_force_n"] = torch.maximum(
                probe["outside_peak_wheel_force_n"], torch.where(outside, wheel_force, 0.))
            probe["boundary_seen"] |= reasons["boundary"][ids]
            probe["nonfinite_seen"] |= ~valid_model[ids] | numerical[ids]
        for name, index in zip(("L_joint2", "R_jonit2"), self.knee_ids):
            low, high = self.knee_bounds[name]
            position = data.joint_pos.torch[:, index]
            reasons["knee"] |= (position < low - .03) | (position > high + .03)
        compression, _ = self.v5.spring_state(data.joint_pos.torch[:, self.spring_ids],
                                              data.joint_vel.torch[:, self.spring_ids])
        reasons["spring_travel"] = ((compression < -.001) | (compression > self.v5.stroke + .001)).any(-1)
        self.spring_reserve_frames += int((compression > .072).any(-1).sum())
        stepup = getattr(self, "v7_stepup", None)
        if stepup is not None:
            protected = torch.zeros_like(terminated)
            for name, mask in reasons.items():
                if name not in ("source_attitude_or_contact", "boundary"):
                    protected |= mask
            wall_truncation = stepup.wall_event & ~protected
            reasons["source_attitude_or_contact"] &= ~wall_truncation
            terminated &= ~wall_truncation
            reasons["wall_blocked"] = wall_truncation
            # Reuse the collection-truncation flag, without introducing a
            # finite track boundary. Numeric/mechanical failures take priority.
            reasons["boundary"] |= wall_truncation
        for name, mask in reasons.items():
            if name not in ("boundary", "wall_blocked"):
                terminated = terminated | mask
        # Legacy contracts keep the source's -1 convention. The explicit new
        # window counts complete physical intervals and owns recovery's limit.
        horizon = self.episode_length_buf >= self.episode_limits - (0 if extended_recovery_window else 1)
        if extended_recovery_window:
            horizon = recovery.episode_window_timeouts(horizon)
        timeouts = (horizon | reasons["boundary"]) & ~terminated
        done = terminated | timeouts
        result = source.pay(root_velocity_body=velocity, angular_velocity_body=omega,
            projected_gravity=gravity, root_pose_world=root_pose,
            wheel_link_positions_world=data.body_link_pose_w.torch[:, self.wheel_ids, :3],
            origins=self.origins, commands=self.commands,
            terminated=terminated, numerical_safety_reset=numerical,
            **({"flat_reward_height_m": local[:, 2]}
                if (getattr(self, "source_flat_rehearsal_mask", None) is not None
                    and source.flat_rehearsal_reward_profile == "scut0719_flat_v1") else {}),
            **({"reward_ground_height_world": reward_ground_height_world}
                if reward_ground_height_world is not None else {}),
            **({"native_joint_positions": data.joint_pos.torch,
                "wheel_link_pose_world": data.body_link_pose_w.torch[:, self.wheel_ids],
                "policy_tick": self.tick, "raycast": reward_raycast,
                # The final source has no active Stair state owner. A height
                # hold or a legacy phase label is not that state.
                "stair_active": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                **({"scripted_interval": scripted_interval} if scripted else {})}
                if self.cfg.get("reward_profile") == "v7_final_rough_v1" else {}))
        reward = result.total
        # Phase is diagnostic/ordinary-push context only; it never gates the
        # selected payments or claims that a passive flight was a learned jump.
        contact = self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.
        stable = (gravity[:, 2] < -.985) & ((height - self.commands[:, 2]).abs() < .01)
        self.phase.update(contact, torch.zeros_like(done), stable)
        self.phase_counts += torch.bincount(self.phase.phase, minlength=5)
        counts = torch.stack([*(mask.sum() for mask in reasons.values()), timeouts.sum()]).cpu().tolist()
        for name, count in zip(reasons, counts[:-1]):
            self.termination_counts[name] = self.termination_counts.get(name, 0) + count
        self.timeout_count += counts[-1]
        extras = {"time_outs": timeouts, "native_reward_components": result.payments,
                  "native_reward_component_units": "already_dt_scaled_payment_per_policy_step",
                  "log": {"/reward/" + name: value.mean() for name, value in result.payments.items()}}
        if numerical_capture is not None and getattr(numerical_capture, "energy", None) is not None:
            extras["log"].update(numerical_capture.energy.metrics())
        if self.cfg.get("reward_profile") == "v7_final_rough_v1":
            extras["log"].update(source.policy_metrics())
        measurement_valid = ~numerical & torch.isfinite(data.applied_torque.torch[:, self.ids]).all(-1)
        flat = getattr(self, "flat_episode", None)
        if flat is not None:
            flat.commands.observe_motion_interval(velocity, omega, done, measurement_valid)
        valid_count = measurement_valid.sum().clamp_min(1)
        extras["log"].update({"/reward/total": reward.mean(),
            "/task/pin_gap_m": torch.where(measurement_valid, gap, 0.).sum() / valid_count,
            "/task/height_error_m": torch.where(measurement_valid,
                (height - self.commands[:, 2]).abs(), 0.).sum() / valid_count,
            "/task/wheel_contact": torch.where(measurement_valid,
                contact.float().mean(-1), 0.).sum() / valid_count,
            "/diagnostics/measurement_valid_fraction": measurement_valid.float().mean(),
            "/diagnostics/closure/all_finite_peak_since_start_m": self.max_closure_gap_all_finite,
            "/diagnostics/closure/numerical_reset_peak_since_start_m": self.max_closure_gap_numerical_reset,
            "/termination/source_candidate_fraction": source.raw_terminal.float().mean(),
            "/termination/source_confirmed_fraction": confirmed.float().mean(),
            "/termination/numerical_reset_fraction": numerical.float().mean()})
        extras["log"].update(self.sampling_metrics)
        if self._spring_pressure_frame is not None:
            frame = self._spring_pressure_frame
            extras["log"].update({
                "/spring/mount_pressure_force_mean_n": frame.pressure_force_n.mean(),
                "/spring/mount_pressure_force_max_n": frame.pressure_force_n.max(),
                "/spring/mount_distance_mean_m": frame.mount_distance_m.mean(),
                "/spring/mount_compression_mean_m": frame.compression_m.mean(),
                "/spring/mount_geometry_valid_fraction": frame.valid.float().mean()})
        source_flat_mask = getattr(self, "source_flat_rehearsal_mask", None)
        if source_flat_mask is not None:
            rehearsal_height = (height if source.flat_rehearsal_reward_profile in ("scut0730_rough_v1", "v7_final_rough_v1")
                                else local[:, 2])
            for group, group_mask, measured_height in (
                    ("source_flat0719", source_flat_mask, rehearsal_height),
                    ("source_rough0730", ~source_flat_mask, height)):
                valid = measurement_valid & group_mask
                denominator = valid.sum().clamp_min(1)
                error = measured_height - self.commands[:, 2]
                extras["log"].update({
                    f"/rehearsal/{group}/height_mae_m": torch.where(valid, error.abs(), 0.).sum() / denominator,
                    f"/rehearsal/{group}/height_bias_m": torch.where(valid, error, 0.).sum() / denominator,
                    f"/rehearsal/{group}/valid_frames": valid.sum(),
                })
        if reward_ground_height_world is not None:
            extras["log"].update({
                "/terrain/base_scan_valid_fraction": (rough_scan_hit_count / 9.).mean(),
                "/terrain/base_ground_height_world_m": reward_ground_height_world.mean()})
        for module in (self.command_transport, self.feedback_transport, self.round_trip_transport, self.signal_delay,
                       self.usb_transport, self.dynamics_randomization,
                       getattr(self, "actor_sensor_noise", None), getattr(self, "policy_signal_delay", None)):
            if module is not None:
                extras["log"].update(module.metrics())
        if getattr(self, "episode_spring_strength_settings", None) is not None:
            extras["log"].update({"/domain/spring_force_scale_mean": self.spring_strength.mean(),
                "/domain/spring_force_scale_min": self.spring_strength.min(),
                "/domain/spring_force_scale_max": self.spring_strength.max()})
        if self.nonstandard_start is not None:
            # Capture ownership BEFORE auto-reset; source reward history still
            # contains RAW actor outputs during the excluded passive interval.
            extras["policy_learning_mask"] = self.nonstandard_start.learning_mask.clone()
            extras["log"].update(self.nonstandard_start.metrics(stage_actor_updates(self)))
        flat = getattr(self, "flat_episode", None)
        if flat is not None:
            extras["policy_learning_mask"] = flat.learning_mask.clone()
            if scripted:
                extras["policy_learning_mask"] &= recovery.learning_mask
            extras["log"].update(flat.metrics(**({"source_birth_mask": ~recovery.enabled} if scripted else {})))
            if getattr(self, "curriculum_controller", None) is not None:
                # Capture identities BEFORE autoreset; this is a physical
                # opportunity pulse, not a tracking/landing capability pass.
                complete = done & (self.episode_length_buf >= self.episode_limits - 1) & ~terminated & ~reasons["boundary"]
                extras["curriculum_transition"] = {
                    "node_indices": flat.commands.episode_node_indices.clone(),
                    "category_indices": flat.commands.category_indices.clone(),
                    "episode_ids": flat.episode_id.clone(),
                    "episode_complete": complete.clone(),
                    "key_phase_events": {"source_horizon_complete": complete.clone()},
                    "key_phase_occurrence_ids": {"source_horizon_complete": torch.where(
                        complete, self.episode_length_buf, -1)}}
        if self.ordinary_posture is not None:
            extras["log"].update(self.ordinary_posture.metrics(stage_actor_updates(self)))
        if scripted:
            owned = recovery.diagnostics()
            extras["log"].update({"/recovery/" + name: value[recovery.enabled].float().mean()
                for name, value in owned.items()})
            payload = recovery.policy_transition(done, terminated, timeouts)
            payload["policy_learning_mask"] = extras["policy_learning_mask"].clone()
            payload["policy_interval_id"] = self.tick
            extras["recovery_transition"] = payload
        if self.cfg.get("record_diagnostics", False):
            extras["diagnostics"] = {"velocity": velocity.clone(), "omega": omega.clone(),
                "gravity": gravity.clone(), "height": height.clone(), "position": local.clone(),
                "commands": self.commands.clone(), "episode_ticks": self.episode_length_buf.clone(),
                "motor_position": data.joint_pos.torch[:, self.ids].clone(),
                "motor_velocity": data.joint_vel.torch[:, self.ids].clone(),
                "motor_effort": data.applied_torque.torch[:, self.ids].clone(),
                "reward": reward.clone(), "gap": gap.clone(), "done": done.clone(),
                "terminated": terminated.clone(), "success": torch.zeros_like(done),
                "reasons": {name: mask.clone() for name, mask in reasons.items()},
                "raw_actor_actions": source.raw_action.clone(),
                "raw_previous_actor_actions": source.previous_raw_action.clone(),
                "raw_before_previous_actor_actions": source.before_previous_raw_action.clone(),
                "native_joint_acceleration_policy_order": source.joint_acceleration.clone(),
                "native_applied_torque_policy_order": source.applied_torque.clone(),
                "native_undesired_contact_history_n": source.contact_history.clone(),
                "source_terminal_counter_policy_ticks": source.terminal_counter.clone(),
                "measurement_valid_mask": measurement_valid.clone(),
                "planar_speed_world": root_link_linear_velocity(self)[:, :2].norm(dim=-1).clone()}
            if reward_ground_height_world is not None:
                extras["diagnostics"].update({
                    "height_reference": "root_link_world_z_minus_base_ground_scan",
                    "root_height_world": root_pose[:, 2].clone(),
                    "reward_ground_height_world": reward_ground_height_world.clone(),
                    "rough_base_scan_valid_hit_count": rough_scan_hit_count.clone()})
            if self.nonstandard_start is not None:
                extras["diagnostics"].update(self.nonstandard_start.diagnostics())
            if flat is not None:
                owned = flat.diagnostics()
                if scripted:
                    owned["scripted_recovery_birth"] = recovery.enabled.clone()
                    extras["diagnostics"].update(recovery.diagnostics())
                    extras["diagnostics"]["recovery_interval_learning_mask"] = recovery.learning_mask.clone()
                    extras["diagnostics"]["recovery_policy_motion_hold"] = recovery.policy_motion_hold.clone()
                    extras["diagnostics"]["recovery_policy_command_override_active"] = recovery.policy_command_override_active.clone()
                base = torch.stack((owned["base_request_command3"][:, 0],
                                    owned["base_request_command3"][:, 2], owned["height"]), -1)
                extras["diagnostics"].update({"source_flat_episode": owned,
                    "episode_id": owned["episode_id"], "episode_tick": self.episode_length_buf.clone(),
                    "root_link_position_world": root_pose[:, :3].clone(),
                    "root_yaw_rad": self._flat_heading().clone(),
                    "root_COM_velocity_body": velocity.clone(), "angular_velocity_body": omega.clone(),
                    "effective_command": self.commands.clone(), "base_request": base,
                    "command_override_active": (owned["paid_override_active"] | recovery.policy_command_override_active
                        if scripted else owned["paid_override_active"]),
                    "timeout": timeouts.clone(), "boundary": reasons["boundary"].clone(),
                    "policy_learning_mask": extras["policy_learning_mask"].clone(),
                    "payments": {name: value.clone() for name, value in result.payments.items()}})
                if self.cfg.get("formal_small_rough_mix", {}).get("version") in MOTION_VERSIONS:
                    # Origins and terminal labels must precede automatic reset.
                    extras["diagnostics"]["rough_birth_origins_world"] = self.origins.clone()
                if stepup is not None:
                    extras["diagnostics"]["v7_stepup_command"] = stepup.diagnostics()
                    extras["diagnostics"]["command_override_active"] |= stepup.active
                if hasattr(data, "root_com_lin_vel_w"):
                    from wheeled_tasks.chassis.observations.com_motion_measurement import ComMotionMeasurement
                    world_velocity = data.root_com_lin_vel_w.torch
                    extras["diagnostics"]["root_COM_velocity_world"] = world_velocity.clone()
                    motion = getattr(self, "_com_motion_measurement", None)
                    if motion is None:
                        motion = self._com_motion_measurement = ComMotionMeasurement(world_velocity)
                    extras["log"].update(motion.observe(world_velocity, owned["episode_id"], measurement_valid))
                if getattr(self, "flat_events", None) is not None:
                    extras["diagnostics"]["source_interval_events"] = self.flat_events.diagnostics()
        if flat is not None:
            if getattr(self, "reward_episode_measurement_enabled", False):
                self._capture_reward_episode_payment(extras, result.payments, done, terminated,
                    timeouts, torch.zeros_like(done), reasons["boundary"])
            if extended_recovery_window and bool(timeouts.any()):
                extras["_capture_timeout_observation"] = True
            self._advance_flat_episode(done, terminated, extras)
            if getattr(self, "curriculum_controller", None) is not None:
                extras["curriculum_transition"]["returned_episode_ids"] = flat.episode_id.clone()
        else:
            if getattr(self, "reward_episode_measurement_enabled", False):
                self._capture_reward_episode_payment(extras, result.payments, done, terminated,
                    timeouts, torch.zeros_like(done), reasons["boundary"])
            self._advance_policy_state(done, terminated, extras)
        if getattr(self, "reward_episode_measurement_enabled", False):
            self._finish_reward_episode_payment(extras)
        if scripted:
            payload["returned_episode_ids"] = recovery.episode_ids.clone()
            payload["reset_mask"] = payload["returned_episode_ids"] != payload["episode_ids"]
        return self.get_observations(), reward, done, extras

    def _flat_heading(self):
        """Root LINK xyzw yaw, preserving invalid measurements in diagnostics."""
        x, y, z, w = self.robot.data.root_link_pose_w.torch[:, 3:7].unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))

    def prepare_formal_plane_contact_probe(self):
        """Scene-validation-only opt-in: exercise real contact outside the old floor."""
        if (self.tick != 0 or self.num_envs < 2 or self.scut_reward is None
                or self.cfg.get("formal_terrain", {}).get("type") != "plane"
                or getattr(self, "shared_terrain", None) is None):
            raise ValueError("Plane contact probe requires an unstepped formal plane with at least two robots")
        ids = torch.arange(2, device=self.device)
        offsets = self.origins.new_tensor([[50., 0.], [0., 6.]])
        self.reset_positions[ids, :2] = self.origins[ids, :2] + offsets
        self.reset(ids)
        self._formal_plane_contact_probe = {
            "env_ids": ids, "requested_local_xy_m": offsets,
            "outside_frames": torch.zeros(2, dtype=torch.long, device=self.device),
            "outside_peak_wheel_force_n": torch.zeros(2, device=self.device),
            "boundary_seen": torch.zeros(2, dtype=torch.bool, device=self.device),
            "nonfinite_seen": torch.zeros(2, dtype=torch.bool, device=self.device)}

    def formal_plane_contact_probe_result(self):
        probe = getattr(self, "_formal_plane_contact_probe", None)
        if probe is None:
            return None
        observed = ((probe["outside_frames"] > 0) & (probe["outside_peak_wheel_force_n"] > 0.)
                    & ~probe["boundary_seen"] & ~probe["nonfinite_seen"])
        return {"scope": "scene_validation_only_real_native_contact_outside_previous_88x8m_floor",
                "passed": bool(observed.all()),
                **{name: value.detach().cpu().tolist() for name, value in probe.items()}}

    def enable_numerical_failure_capture(self, *, stop_on_first_event=False,
            stop_on_nonfinite=False, closure_observation_threshold_m=.03):
        """Opt in to first-event or nonfinite-only 5ms worker stops."""
        if (not self.cfg.get("formal_v7_flat") or self.scut_reward is None or not self.critic78
                or self.flat_episode is None or self.dt != .005 or self.policy_dt != .02
                or self.physical_tick != 0 or getattr(self, "_numerical_failure_capture", None) is not None):
            raise ValueError("Numerical capture requires unstepped formal source 200/50 Hz")
        from wheeled_tasks.chassis.simulation.telemetry.numerical_failure_capture import NumericalFailureCapture
        import hashlib
        directory = self.cfg.get("formal_physics_trace_directory")
        if not directory:
            raise ValueError("Numerical capture requires the existing formal trace directory")
        constraints = (self.model_spec["constraints"] if self.is_v5
            else self.manifest["closed_chain_constraints"])
        knee_names = [self.robot.joint_names[index] for index in self.knee_ids]
        metadata = {"joint_names": list(self.robot.joint_names), "body_names": list(self.robot.body_names),
            "active_joint_ids_control_order": list(self.ids), "spring_joint_ids": list(self.spring_ids),
            "knee_joint_ids": list(self.knee_ids), "knee_joint_names": knee_names,
            "knee_joint_bounds_rad": {name: list(self.knee_bounds[name]) for name in knee_names},
            "closure_constraint_names": [constraint["name"] for constraint in constraints],
            "closure_body_ids": self._closure_body_ids.detach().cpu().tolist(),
            "closure_local_points_m": self._closure_points.detach().cpu().tolist(),
            "wheel_body_ids": list(self.wheel_ids), "physics_dt_s": self.dt, "policy_dt_s": self.policy_dt,
            "configuration": dict(self.cfg), "asset_files_sha256": dict(self.manifest["files_sha256"]),
            "sources_sha256": {name: hashlib.sha256(module_source_path(name).read_bytes()).hexdigest()
                for name in ("env.py", "numerical_failure_capture.py")},
            "native_applied_effort_semantics": "SDK explicit-actuator applied_torque buffer, not a force sensor",
            "native_contact_readback_semantics": {
                "source": "RigidContactView.get_net_contact_forces(dt=physics_dt_s)",
                "components": "net normal resultants per body; tangential friction is a separate API",
                "total_contact_impulse_measured": False},
            "raw_actor_order": "policy order; active/control arrays use active_joint_ids_control_order",
            "prior_control_semantics": "targets/feedback/sent effort for next_physical_tick; native state/contact at physical_tick",
            "current_control_semantics": "targets/feedback/sent effort for interval ending at dump_physical_tick",
            "root_velocity_semantics": "root COM linear/angular velocity in body frame",
            "omitted_to_limit_capture_cost": ["joint_acc"]}
        if getattr(self, "flat_events", None) is not None:
            from wheeled_tasks.chassis.simulation.randomization.flat_interval_events import TERM_ORDER
            metadata["source_interval_events"] = {
                "event_count_columns": list(TERM_ORDER),
                "counter_scope": "cumulative per lane; articulation resets clear held wrenches only",
                "wrench_frame": "base_link body frame; torque about its COM",
                "measurement": "cached commands verified against the composer when written, not measured forces",
                "timing": "events applied at policy boundaries before the next physical substep"}
            metadata["sources_sha256"]["flat_interval_events.py"] = hashlib.sha256(
                module_source_path("flat_interval_events.py").read_bytes()).hexdigest()
        if self.cfg.get("v5_control", {}).get("leg_winding_version") == "paired_branch107_v1":
            metadata["policy_input_scope"] = (
                "Actual35D observation consumed by PPO.act, Gaussian mean/std in policy order; "
                "cloned before env.step and retained for the control interval, not a clean-input proxy")
        if getattr(self, "spring_pressure_element", None) is not None:
            metadata["gas_pressure_application"] = dict(self.startup_report["gas_pressure_application"])
            metadata["pressure_body_ids"] = list(self.spring_pressure_element.body_ids)
            metadata["pressure_command_semantics"] = (
                "Pre-step world wrenches about actual COM for generated_physical_tick+1; "
                "slider target contains no pressure and native slider effort is damping only")
        energy = None
        if self.cfg.get("formal_small_rough_mix", {}).get("version") in MOTION_VERSIONS:
            from wheeled_tasks.chassis.simulation.telemetry.finite_energy_readback import FiniteEnergyReadback
            joint_types = {joint["name"]: joint["type"] for joint in self.model_spec["joints"]}
            gravity = tuple(self.sim.cfg.gravity)
            if gravity[:2] != (0., 0.) or gravity[2] >= 0:
                raise ValueError("Current rigid energy readback requires the configured downward gravity")
            event_columns = metadata.get("source_interval_events", {}).get("event_count_columns", [])
            energy = FiniteEnergyReadback(
                wp.to_torch(self.robot.root_view.get_masses()).to(self.device),
                wp.to_torch(self.robot.root_view.get_inertias()).to(self.device), physics_dt_s=self.dt,
                angular_joints=[joint_types[name] != "prismatic" for name in self.robot.joint_names],
                active_joint_ids=self.ids, base_body_id=self.robot.body_names.index("base_link"),
                wheel_body_ids=self.wheel_ids, pressure_body_ids=metadata.get("pressure_body_ids", ()),
                configured_armature=self.cfg.get("actuator_response", {}).get("armature"),
                velocity_push_column=event_columns.index("push_robot") if "push_robot" in event_columns else None,
                external_wrench_enabled="base_external_force_torque_xyz" in event_columns,
                gravity_m_s2=-gravity[2])
            metadata["finite_energy"] = {
                "native_properties": "startup existing root_view get_masses/get_inertias; no setters",
                "inertia_reference": "body-link axes about actual COM, verified by startup property readback",
                "gravity_m_s2": -gravity[2], "gravity_source": "SimulationContext.cfg.gravity",
                "work_scope": "explicit actuation, mount pressure and held source wrench estimates; contact/constraint/passive work incomplete",
                "contact_peak_snapshot": "selected native terrain normal/friction CSR rows at the same post-physics peak; no state writes",
                "peak_selection": "largest finite rigid increment, explicit-work residual and wheel force; no added thresholds"}
            metadata["sources_sha256"]["finite_energy_readback.py"] = hashlib.sha256(
                module_source_path("finite_energy_readback.py").read_bytes()).hexdigest()
        self._numerical_failure_capture = NumericalFailureCapture(directory, self.num_envs, metadata,
            stop_on_first_event=stop_on_first_event, stop_on_nonfinite=stop_on_nonfinite,
            closure_observation_threshold_m=closure_observation_threshold_m, energy_readback=energy,
            energy_contact_snapshot=self._formal_native_contact_snapshot if energy is not None else None)
        self.startup_report["numerical_failure_capture"] = self._numerical_failure_capture.summary

    def _numerical_failure_state(self, legs, wheels, feedback_q, feedback_dq, effort):
        """Raw device views; the observer owns detached copies before any in-place reset."""
        data, flat = self.robot.data, self.flat_episode
        result = {"q": data.joint_pos.torch, "dq": data.joint_vel.torch,
            "native_applied_effort": data.applied_torque.torch,
            "root_link_pose_world": data.root_link_pose_w.torch,
            "root_com_velocity_body": data.root_com_lin_vel_b.torch,
            "root_angular_velocity_body": data.root_com_ang_vel_b.torch,
            "body_link_pose_world": data.body_link_pose_w.torch, "body_com_pose_world": data.body_com_pose_w.torch,
            "body_com_linear_velocity_world": data.body_com_lin_vel_w.torch,
            "body_com_angular_velocity_world": data.body_com_ang_vel_w.torch,
            "world_xy": data.root_link_pose_w.torch[:, :2], "origins": self.origins,
            "terrain_filtered_contact_force": self.contact_force,
            "all_partner_net_contact_force": self.critic_net_contact_force,
            "spring_q": data.joint_pos.torch[:, self.spring_ids], "spring_dq": data.joint_vel.torch[:, self.spring_ids],
            "spring_native_effort": data.applied_torque.torch[:, self.spring_ids],
            "spring_sent_effort": effort[:, self.spring_ids],
            "leg_position_target": legs, "wheel_velocity_target": wheels,
            "feedback_q": feedback_q, "feedback_dq": feedback_dq, "sent_effort": effort,
            "active_effort_after_filter_and_gate": self.torque,
            "requested_motor_effort": self.v5.requested_motor_effort,
            "motor_effort_bounds": self.v5.current_motor_bounds,
            "raw_actor_action": self.scut_reward.raw_action, "clipped_control_action": self.actions,
            "previous_control_action": self.previous_actions, "commands": self.commands,
            "episode_id": flat.episode_id, "episode_source_physics_ticks": flat.commands.episode_physics_ticks,
            "episode_policy_ticks": self.episode_length_buf, "holding": flat.holding,
            **getattr(self, "_formal_actor_input", {})}
        events = getattr(self, "flat_events", None)
        if events is not None:
            # Preserve disturbance provenance in the same ring as the impact;
            # post-reset caches cannot explain a previously captured energy jump.
            result.update(source_interval_event_counts=events.event_count.transpose(0, 1),
                source_held_force_body_n=events.force_body_n.squeeze(1),
                source_held_torque_body_nm=events.torque_body_nm.squeeze(1))
        result.update(self._mount_pressure_control_fields())
        if getattr(self, "round_trip_transport", None) is not None and self.actuator_response.enabled:
            result["motor_response_endpoint_state"] = self.actuator_response.state
        return result

    def _apply_mount_spring_pressure(self):
        """Apply one passive pressure command before the sole physical write."""
        data = self.robot.data
        frame = self.spring_pressure_element.evaluate(data.body_link_pose_w.torch,
            data.body_com_pose_w.torch[..., :3], self.spring_strength)
        if not bool(frame.valid.all()):
            raise FloatingPointError("Invalid main-link spring pressure geometry before physics")
        self._spring_pressure_frame = frame
        self._spring_pressure_generated_tick.fill_(self.physical_tick)
        self._spring_pressure_strength = self.spring_strength.clone()
        self._spring_pressure_com_positions = data.body_com_pose_w.torch[..., :3].index_select(
            1, self._spring_pressure_body_ids)
        # Moment is already about the actual COM. Supplying mount positions
        # as well would add the lever-arm moment twice.
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=frame.forces_world.contiguous(),
            torques=frame.torques_about_com_world.contiguous(),
            body_ids=self._spring_pressure_body_ids, is_global=True)

    def _mount_pressure_control_fields(self):
        frame = getattr(self, "_spring_pressure_frame", None)
        if frame is None:
            return {}
        return {"pressure_generated_physical_tick": self._spring_pressure_generated_tick,
            "pressure_strength": self._spring_pressure_strength,
            "pressure_body_com_positions_world": self._spring_pressure_com_positions,
            "pressure_mount_distance_m": frame.mount_distance_m,
            "pressure_compression_m": frame.compression_m,
            "pressure_force_n": frame.pressure_force_n,
            "pressure_mount_positions_world": frame.mount_positions_world,
            "pressure_forces_world": frame.forces_world,
            "pressure_torques_about_com_world": frame.torques_about_com_world}

    def _formal_physics_control_frame(self, legs, wheels, feedback_q, feedback_dq, effort):
        """Clone before stepping: tick 40 is still held even though post-hold clears."""
        flat = self.flat_episode
        response_state = ({"motor_response_endpoint_state": self.actuator_response.state}
            if getattr(self, "round_trip_transport", None) is not None and self.actuator_response.enabled else {})
        policy_delay = getattr(self, "policy_signal_delay", None)
        if policy_delay is not None:
            response_state.update(policy_target_sample_s=policy_delay.target_sample_s,
                policy_target_generated_s=policy_delay.target_policy_generated_s,
                unlagged_policy_generated_s=self.policy_generated_s,
                local_feedback_sensor_timestamp_s=self._pc_feedback_transport.sensor_timestamp_s,
                applied_target_sample_s=self.command_transport.applied_target_sample_s,
                applied_target_policy_generated_s=self.command_transport.applied_target_policy_generated_s,
                applied_pd_origins_s=self.command_transport.applied_origins_s)
        return {key: value.detach().clone() for key, value in {
            "leg_position_target": legs, "wheel_velocity_target": wheels,
            "feedback_q": feedback_q, "feedback_dq": feedback_dq,
            "sent_effort": effort, "holding_before": flat.holding,
            "episode_id_before": flat.episode_id,
            "episode_physics_ticks_before": flat.commands.episode_physics_ticks,
            **self._mount_pressure_control_fields(), **response_state}.items()}

    def _compact_formal_contact_data(self, raw, fields, *, sensor_indices=None):
        """Copy valid CSR entries only; never copy shared capacity buffers to CPU."""
        if len(raw) != len(fields) + 2:
            raise ValueError("Unexpected native contact tuple length")
        buffers = {key: wp.to_torch(value) for key, value in zip(fields, raw)}
        counts_raw, starts_raw = (wp.to_torch(value) for value in raw[-2:])
        if sensor_indices is not None:
            if (not isinstance(sensor_indices, torch.Tensor) or sensor_indices.dtype != torch.long
                    or sensor_indices.ndim != 1 or sensor_indices.device != counts_raw.device
                    or sensor_indices.unique().numel() != sensor_indices.numel()
                    or bool(((sensor_indices < 0) | (sensor_indices >= counts_raw.shape[0])).any())):
                raise ValueError("Contact selection requires unique native sensor row indices")
            counts_raw = counts_raw.index_select(0, sensor_indices)
            starts_raw = starts_raw.index_select(0, sensor_indices)
        # The contact and friction getters share these two native buffers.
        # Finish this detached CPU record before calling the next getter.
        result = {"counts": counts_raw.detach().cpu().clone(),
            "starts": starts_raw.detach().cpu().clone()}
        if sensor_indices is not None:
            result["source_sensor_indices"] = sensor_indices.detach().cpu().clone()
        counts, starts = counts_raw.long(), starts_raw.long()
        capacity = next(iter(buffers.values())).shape[0]
        result["buffer_capacity"] = capacity
        if (counts.ndim != 2 or counts.shape != starts.shape
                or any(value.shape[0] != capacity for value in buffers.values())):
            return {**result, "status": "invalid", "error": "Native contact buffer shapes differ"}
        active = counts > 0
        total = int(counts.sum())
        result["valid_data_count"] = total
        if (bool((counts < 0).any()) or total > capacity
                or bool((active & ((starts < 0) | (starts > capacity - counts))).any())):
            return {**result, "status": "invalid", "error": "Native contact counts/starts exceed capacity"}
        flat_counts = counts.flatten()
        compact_starts = flat_counts.cumsum(0) - flat_counts
        pairs = torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), flat_counts)
        indices = (starts.flatten()[pairs] + torch.arange(total, device=counts.device)
            - compact_starts[pairs])
        result.update(status="ok", compact_starts=compact_starts.reshape(counts.shape).cpu().clone(),
            buffer_indices=indices.cpu().clone(),
            fields={key: value.index_select(0, indices).detach().cpu().clone()
                for key, value in buffers.items()})
        return result

    def _formal_native_contact_snapshot(self, *, environment_ids=None):
        """Read existing terrain-filtered views without modifying native state."""
        if environment_ids is not None and (
                not isinstance(environment_ids, torch.Tensor) or environment_ids.ndim != 1
                or environment_ids.dtype != torch.long or environment_ids.device != torch.device(self.device)
                or not environment_ids.numel() or environment_ids.unique().numel() != environment_ids.numel()
                or bool(((environment_ids < 0) | (environment_ids >= self.num_envs)).any())):
            raise ValueError("Contact peak evidence requires unique actual environment ids")
        result = {"dt_s": self.dt, "state_timing": "after_physics_contact_refresh_before_reset",
            "force_units": "N_from_native_impulse_divided_by_dt", "position_and_separation_units": "m",
            "frame": "raw_native_world_points_normals_and_friction_vectors",
            "pair_scope": "existing_sensor_body_vs_terrain_filter_pairs_not_unfiltered_contacts",
            "index_semantics": "counts/starts_are_original_shared_buffer_CSR; compact_starts_index_saved_fields; buffer_indices_map_to_original_buffer",
            "sensor_order": "environment_ids_order_then_robot_body_names_order",
            "body_names": list(self.robot.body_names), "groups": []}
        for ids, view in self.contact_views:
            sensor_indices = None
            if environment_ids is not None:
                selected = torch.isin(ids, environment_ids).nonzero(as_tuple=False).flatten()
                if not len(selected):
                    continue
                sensor_indices = (selected[:, None] * len(self.robot.body_names)
                    + torch.arange(len(self.robot.body_names), device=ids.device)).flatten()
                ids = ids[selected]
            group = {"environment_ids": ids.detach().cpu().clone()}
            result["groups"].append(group)
            try:
                group.update(sensor_paths=list(view.sensor_paths),
                    filter_paths=[list(paths) for paths in view.filter_paths],
                    sensor_count=view.sensor_count, filter_count=view.filter_count,
                    shared_capacity=view.max_contact_data_count)
                if sensor_indices is not None:
                    rows = sensor_indices.cpu().tolist()
                    group["sensor_paths"] = [group["sensor_paths"][i] for i in rows]
                    group["filter_paths"] = [group["filter_paths"][i] for i in rows]
                    group["selected_sensor_count"] = len(rows)
            except Exception as error:
                group["mapping_error"] = repr(error)
            for key, getter, fields in (
                    ("contact", "get_contact_data", ("normal_force_N", "point_world_m", "normal_world", "separation_m")),
                    ("friction", "get_friction_data", ("force_world_N", "point_world_m"))):
                try:
                    # Do not fetch both tuples together: their CSR buffers alias.
                    raw = getattr(view, getter)(dt=self.dt)
                    compact = (self._compact_formal_contact_data(raw, fields) if sensor_indices is None
                        else self._compact_formal_contact_data(raw, fields, sensor_indices=sensor_indices))
                    group[key] = {"api": getter, **compact}
                except AttributeError as error:
                    group[key] = {"status": "unsupported", "api": getter, "error": repr(error)}
                except Exception as error:
                    group[key] = {"status": "error", "api": getter, "error": repr(error)}
        return result

    def _formal_native_collision_properties(self):
        """Read base offsets from the existing articulation view, without setters."""
        view = self.robot.root_view
        layout = self.critic_material.layout
        base = self.robot.body_names.index("base_link")
        start = sum(layout.shapes_per_body[:base])
        end = start + layout.shapes_per_body[base]
        paths = list(view.prim_paths)
        result = {"articulation_paths": paths, "environment_ids": torch.arange(len(paths)),
            "body_name": "base_link", "native_shape_range": [start, end],
            "shape_order": "same_native_articulation_shape_order_as_native_material_properties",
            "units": "m", "base": {},
            "ground": {"status": "unsupported", "reason": "Terrain is only an existing contact filter; no ground rigid view exists and no view was created"},
            "max_depenetration_velocity": {"status": "unsupported",
                "reason": "Tensor API exposes no native max-depenetration velocity getter; authored values are not native readback"}}
        for key, getter in (("contact_offsets_m", "get_contact_offsets"), ("rest_offsets_m", "get_rest_offsets")):
            try:
                raw = wp.to_torch(getattr(view, getter)())
                if raw.ndim != 2 or raw.shape[0] != len(paths) or raw.shape[1] != layout.total_shapes:
                    raise ValueError("Native offsets differ from existing articulation shape layout")
                result["base"][key] = {"status": "ok", "api": getter,
                    "values": raw[:, start:end].detach().cpu().clone()}
            except AttributeError as error:
                result["base"][key] = {"status": "unsupported", "api": getter, "error": repr(error)}
            except Exception as error:
                result["base"][key] = {"status": "error", "api": getter, "error": repr(error)}
        return result

    def _save_formal_physics_state(self, name, *, control_frame=None):
        data = self.robot.data
        values = {"joint_names": self.robot.joint_names, "body_names": self.robot.body_names,
            "tick": self.tick, "physical_tick": self.physical_tick}
        for key, value in {"q": data.joint_pos.torch, "dq": data.joint_vel.torch,
                "root_pose": data.root_link_pose_w.torch, "body_pose": data.body_link_pose_w.torch,
                "origins": self.origins, "gap": self.closure_gap(),
                "gap_peak": self.closure_gap_peak, "contact": self.critic_net_contact_force,
                "applied_torque": self.critic_applied_torque}.items():
            values[key] = value.detach().cpu().clone()
        if self.cfg.get("formal_dynamic_nonsteady_birth"):
            flat = self.flat_episode
            for key, value in {"episode_id": flat.episode_id,
                    "episode_physics_ticks": flat.commands.episode_physics_ticks,
                    "ground_birth": flat.commands.ground_birth,
                    "holding_after": flat.holding}.items():
                values[key] = value.detach().cpu().clone()
        if name == "before_first_step" and getattr(self, "spring_pressure_element", None) is not None:
            # One existing diagnostic snapshot, after startup DR and full restore.
            # Keep actual inertials available without adding per-step readbacks.
            view = self.robot.root_view
            values["native_properties"] = {key: wp.to_torch(getter()).detach().cpu().clone()
                for key, getter in {"body_mass": view.get_masses,
                    "body_inertia_link": view.get_inertias,
                    "body_com_local_xyzw": view.get_coms,
                    "joint_max_velocity": view.get_dof_max_velocities,
                    "joint_drive_stiffness": view.get_dof_stiffnesses,
                    "joint_drive_damping": view.get_dof_dampings}.items()}
            values["native_properties_semantics"] = (
                "Actual native order; inertia about COM in link axes; root and body poses xyzw. "
                "Startup-only mass DR is retained across this worker's resets.")
            if "physx_solver_type" in self.cfg:
                layout = self.critic_material.layout
                values["native_material_properties"] = self.critic_material.native_materials()
                values["native_material_layout"] = {
                    "body_names": list(self.robot.body_names),
                    "shapes_per_body": list(layout.shapes_per_body),
                    "total_shapes": layout.total_shapes,
                    "wheel_body_ids": list(layout.body_ids),
                    "wheel_shape_ranges": [list(pair) for pair in layout.shape_ranges],
                    "columns": ["static_friction", "dynamic_friction", "restitution"]}
        if name == "before_first_step" and self.cfg.get("formal_physics_trace_substeps"):
            try:
                values["native_collision_properties"] = self._formal_native_collision_properties()
            except Exception as error:
                values["native_collision_properties"] = {"status": "error", "error": repr(error)}
        if name == "closure_failure":
            threshold = self.cfg.get("closure_gap_termination_m") or .03
            values["closure_event"] = {
                "threshold_m": threshold,
                "env_ids": (values["gap_peak"] > threshold).nonzero().flatten().tolist(),
                "measurement": "peak_over_5ms_substeps_in_current_policy_interval",
                "state_timing": "20ms_policy_end_before_reset_not_peak_substep"}
        if self.cfg.get("formal_physics_trace_substeps"):
            flat = self.flat_episode
            # COM velocity plus link pose resolves anchor-point velocity;
            # SDK joint_acc is a cached finite difference of joint velocity,
            # not an independent native acceleration observation.
            for key, value in {"joint_acc": data.joint_acc.torch,
                    "native_applied_torque": data.applied_torque.torch,
                    "body_com_pose": data.body_com_pose_w.torch,
                    "body_com_linear_velocity": data.body_com_lin_vel_w.torch,
                    "body_com_angular_velocity": data.body_com_ang_vel_w.torch,
                    "closure_body_ids": self._closure_body_ids,
                    "closure_local_points": self._closure_points,
                    "active_joint_ids": self.ids, "episode_id": flat.episode_id,
                    "episode_physics_ticks": flat.commands.episode_physics_ticks,
                    "ground_birth": flat.commands.ground_birth, "holding_after": flat.holding}.items():
                values[key] = torch.as_tensor(value).detach().cpu().clone()
            values["joint_acc_provenance"] = {
                "source": "SDK_cached_joint_velocity_finite_difference",
                "independent_native_acceleration": False}
            values["control_frame"] = (None if control_frame is None else
                {key: value.detach().cpu().clone() for key, value in control_frame.items()})
            if (129 <= self.physical_tick <= 132
                    and name == f"physics_substep_{self.physical_tick:04d}"):
                try:
                    values["native_contact_data"] = self._formal_native_contact_snapshot()
                except Exception as error:
                    values["native_contact_data"] = {"status": "error", "error": repr(error)}
        torch.save(values, Path(self.cfg["formal_physics_trace_directory"]) / (name + ".pt"))

    def flat_evaluation_snapshot(self, reset_ids):
        """Read returned native fields for the independent flat evaluator."""
        if "flat_evaluation_setup" not in self.startup_report:
            raise RuntimeError("Returned evaluation snapshots require the typed first-reset setup")
        from wheeled_tasks.chassis.application.evaluation.flat_evaluation_loop import reset_snapshot
        return reset_snapshot(self, reset_ids)

    def _flat_command_iteration(self):
        from wheeled_tasks.chassis.application.runtime.training_schedule import actor_update_clock
        # Only the explicit source-rehearsal owner adds its inherited flat
        # offset. Sending the global clock here would count that offset twice.
        if (getattr(self, "source_flat_rehearsal_mask", None) is not None
                or getattr(self, "rehearsal_spec", None) is not None
                and self.rehearsal_spec["settings"]["version"] == "v7_fixed17_plane_rehearsal_v2"):
            return stage_actor_updates(self)
        iteration = actor_update_clock(self)
        return min(iteration, self.cfg.get("flat_special_mode_iteration_cap", iteration))

    def submit_stepup_events(self, step_events, wall_events, *, episode_ids):
        """Queue explicit events for the next returned command update.

        The producer must identify the currently observed episodes. This API
        does not implement or qualify a scanner, and does not change commands
        after the caller has already generated an action from an observation.
        """
        stepup = getattr(self, "v7_stepup", None)
        if stepup is None or self.flat_episode is None:
            raise RuntimeError("V7 StepUp is not explicitly enabled")
        current = self.flat_episode.episode_id
        if (not isinstance(episode_ids, torch.Tensor) or episode_ids.dtype != current.dtype
                or episode_ids.device != current.device or episode_ids.shape != current.shape
                or not torch.equal(episode_ids, current)):
            raise ValueError("V7 StepUp events require current episode identities; stale events are rejected")
        stepup.submit(step_events, wall_events)

    def _advance_flat_episode(self, done, terminated, extras):
        flat = self.flat_episode
        heading = self._flat_heading()
        # An invalid terminal pose will be reset below. Its discarded command
        # cache may use zero heading; the raw diagnostic is never repaired and
        # invalid nonterminal headings still fail the sampler's validation.
        heading = torch.where(terminated & ~torch.isfinite(heading), 0., heading)
        flat.advance(self.policy_dt, heading,
            self._flat_command_iteration(),
            self.generator, **({"world_positions": self.robot.data.root_link_pose_w.torch[:, :3]}
                if self.cfg.get("formal_terrain", {}).get("type") == "rough"
                and getattr(self, "evaluation_surface", "training") != "plane_regression_v1" else {}))
        if extras.pop("_capture_timeout_observation", False):
            # Publish the command transition before reading the final physical
            # state. Autoreset below must never supply a timeout's critic input.
            flat.publish(self.commands, self.mode, self.targets, self.jump_requested)
            self.raw_height.copy_(self.commands[:, 2])
            self.recovery.override_commands()
            extras["policy_terminal_critic_observation"] = self.get_observations(critic_only=True).clone()
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.reset(ids)
        events = getattr(self, "flat_events", None)
        if events is not None:
            batch = events.apply_to_robot(self.robot, self.flat_event_base_id, self.generator,
                root_kinematics=getattr(self, "root_kinematics", None))
            extras["log"].update(events.metrics())
        flat.publish(self.commands, self.mode, self.targets, self.jump_requested)
        self.raw_height.copy_(self.commands[:, 2])
        stepup = getattr(self, "v7_stepup", None)
        if stepup is not None:
            eligible = ~flat.holding
            recovery = getattr(self, "recovery", None)
            if getattr(recovery, "source_flat_handoff", False):
                eligible &= ~recovery.policy_motion_hold
            self.commands[:, 2] = stepup.advance(self.raw_height, eligible, policy_tick=self.tick)
            extras["log"].update(stepup.metrics())
        extras["policy_bootstrap_mask"] = flat.bootstrap_mask(terminated).clone()
        recovery = getattr(self, "recovery", None)
        if getattr(recovery, "source_flat_handoff", False):
            recovery.override_commands()
            extras["policy_bootstrap_mask"] &= recovery.bootstrap_mask(terminated)
        air_limits = getattr(flat.commands, "apply_air_birth_command_limits", None)
        if air_limits is not None:
            air_limits(self.commands)
        if "diagnostics" in extras:
            extras["diagnostics"]["policy_bootstrap_mask"] = extras["policy_bootstrap_mask"].clone()
            # These are the RETURNED state identities, separate from the paid
            # pre-reset transition above. A newborn lane has tick0, not tick1.
            extras["returned_episode_state"] = {"episode_id": flat.episode_id.clone(),
                "episode_tick": self.episode_length_buf.clone(), "reset_mask": done.clone(),
                "source_flat_episode": flat.diagnostics()}
            if stepup is not None:
                extras["returned_episode_state"]["v7_stepup_command"] = stepup.diagnostics()
            if getattr(recovery, "source_flat_handoff", False):
                extras["returned_episode_state"]["source_flat_episode"]["scripted_recovery_birth"] = recovery.enabled.clone()
                extras["returned_episode_state"]["recovery_bootstrap_mask"] = recovery.bootstrap_mask(terminated).clone()
            if events is not None:
                extras["returned_episode_state"]["source_interval_events"] = events.diagnostics()
                extras["returned_episode_state"]["event_batch"] = {
                    "push_ids": batch.push_ids.clone(), "push_increment_world": batch.push_increment_world.clone(),
                    "push_velocity_world": batch.push_velocity_world.clone(), "wrench_ids": batch.wrench_ids.clone(),
                    "force_body_n": batch.force_body_n.clone(), "torque_body_nm": batch.torque_body_nm.clone()}

    def _advance_policy_state(self, done, terminated, extras):
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids) and self.cfg.get("auto_reset", True):
            self.reset(ids)
        self.command_clock -= self.policy_dt
        self.height_clock -= self.policy_dt
        self.push_clock -= self.policy_dt
        change = (self.command_clock <= 0) & ~done
        if not self.cfg.get("new_asset_curriculum", {}).get("enabled") or self.cfg.get("evaluation_exact_cases"):
            change &= self.mode < 2
        if change.any():
            self.resample_commands(change.nonzero(as_tuple=False).flatten())
        change_height = (self.height_clock <= 0) & ~done
        if change_height.any():
            n = int(change_height.sum())
            lo, hi = self.cfg["height_range_m"]
            self.commands[change_height, 2] = lo + (hi - lo) * self.random(n)
            self.height_clock[change_height] = 5 + 3 * self.random(n)
        push_phase = torch.ones_like(done) if self.cfg.get("evaluation_exact_cases") else (self.phase.phase == Phase.GROUND)
        pushed = (self.push_clock <= 0) & self.push_enabled & push_phase & ~done
        posture = getattr(self, "ordinary_posture", None)
        if posture is not None and posture.enabled:
            actor_owned = self.recovery.pure_rl if self.recovery is not None else torch.ones_like(done)
            passive_hold = (self.nonstandard_start.holding if self.nonstandard_start is not None
                else torch.zeros_like(done))
            ordinary = posture.push_mask(self.push_clock <= 0, self.phase.phase == Phase.GROUND,
                actor_owned, passive_hold, done) & (self.mode != 4)
            pushed = (pushed & ~posture.eligible) | ordinary
        if pushed.any():
            ids = pushed.nonzero(as_tuple=False).flatten()
            self.apply_push(ids)
        if self.cfg.get("command_slew"):
            heading = getattr(self.skills, "heading", None)
            heading_previous_planar = self.commands[heading.ids, :2].clone() if heading is not None else None
            if self.skills is not None:
                self.skills.update()
            if self.step_jump is not None and self.step_jump.has_tasks:
                self._update_step_jump()
                self.command_target[self.step_jump.enabled, 0] = self.step_jump.command_vx[self.step_jump.enabled]
                self.command_target[self.step_jump.enabled, 1] = 0.
            rates = self.commands.new_tensor([self.cfg["command_slew"]["vx_m_s2"], self.cfg["command_slew"]["yaw_rad_s2"]])
            if self.references is not None:
                rates = torch.stack((self.skills.vx_slew, torch.full_like(self.skills.vx_slew, self.cfg["command_slew"]["yaw_rad_s2"])), -1)
            target = self.command_target
            if self.references is not None:
                target = target.clone()
                manual_mode = self.skills.manual_step_mode
                manual_elapsed = self.episode_length_buf * self.policy_dt - self.skills.manual_step_start
                if self.skills.schedule.requests_enabled:
                    schedule = self.skills.schedule
                    manual_mode = manual_mode * schedule.request_active.long()
                    manual_elapsed = torch.where(schedule.request_active,
                        self.episode_length_buf * self.policy_dt - schedule.request_start, 0.)
                preparing = self.references.step_preparing(manual_mode, manual_elapsed)
                target[preparing, 0] = 0.
            if self.nonstandard_start is not None:
                target = self.nonstandard_start.cap_commands(target)
            change = (target - self.commands[:, :2]).clamp(-rates * self.policy_dt, rates * self.policy_dt)
            self.commands[:, :2] += change
            self.command_acceleration = change[:, 0] / self.policy_dt
            if self.skills is not None:
                # Rotating a fixed world-frame request is a coordinate change,
                # not a physical acceleration request to be rate-limited twice.
                self.commands[self.skills.spin, 0] = self.command_target[self.skills.spin, 0]
                self.command_acceleration[self.skills.spin] = 0.
            if self.nonstandard_start is not None:
                self.commands.copy_(self.nonstandard_start.cap_commands(self.commands))
            if heading is not None:
                heading.finish(self.commands, heading_previous_planar, self.command_acceleration, self.cfg["motion_limits"])
        if self.step_assist is not None:
            self.raw_height.copy_(self.commands[:, 2])
            self.update_height_assist()
        if self.recovery is not None:
            self.recovery.override_commands()
            extras["policy_bootstrap_mask"] = self.recovery.bootstrap_mask(terminated)
        if self.nonstandard_start is not None:
            extras["policy_bootstrap_mask"] = (extras.get("policy_bootstrap_mask", ~terminated)
                & self.nonstandard_start.bootstrap_mask(terminated))
        self.update_targets()
        if self.references is not None:
            self.raw_height.copy_(self.commands[:, 2])
            self.update_command_reference()

    def apply_push(self, ids):
        velocity_w = wp.to_torch(self.robot.root_view.get_root_velocities())[ids].clone()
        trace = getattr(self, "jump_physics_trace", None)
        trace_before = velocity_w.clone() if trace is not None else None
        posture = getattr(self, "ordinary_posture", None)
        ordinary = ids[posture.eligible[ids]] if posture is not None and posture.enabled else None
        before = velocity_w.clone() if ordinary is not None and len(ordinary) else None
        angle = self.random(len(ids)) * math.tau
        cap = self.skills.push_speed[ids] if self.skills is not None else self.stage_cfg["push_max"]
        if posture is not None and posture.enabled:
            _, _, ordinary_cap = posture.amplitudes((self._curriculum_clock("ordinary_posture_curriculum") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self)))
            cap = torch.as_tensor(cap, device=self.device).expand(len(ids)).clone()
            cap[posture.eligible[ids]] = (ordinary_cap[ids[posture.eligible[ids]]]
                if torch.is_tensor(ordinary_cap) else ordinary_cap)
        amplitude = self.random(len(ids)) * cap
        velocity_w[:, :2] += torch.stack((angle.cos(), angle.sin()), -1) * amplitude[:, None]
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity_w, env_ids=ids, full_data=False)
        if trace is not None:
            trace.record_velocity_write(ids, physical_tick=self.physical_tick, source="legacy_push",
                before_world=trace_before, commanded_world=velocity_w,
                readback_world=wp.to_torch(self.robot.root_view.get_root_velocities())[ids].clone())
        self.push_clock[ids] = 3 + 2 * self.random(len(ids))
        if posture is not None and posture.enabled:
            self.push_clock[ordinary] = posture.sample_intervals(ordinary, generator=self.generator)
            if len(ordinary):
                after = wp.to_torch(self.robot.root_view.get_root_velocities())[ids]
                if not torch.allclose(after[posture.eligible[ids]], velocity_w[posture.eligible[ids]],
                        atol=1e-5, rtol=1e-5):
                    raise RuntimeError("Ordinary posture velocity push did not match native readback")
                posture.record_push(ids, before, after)

    def _recovery_metrics(self, done, success, terminated, timeouts):
        """Record native outcomes before auto-reset; lessons cannot credit L3."""
        recovery, curriculum = self.recovery, self.recovery.pose_curriculum
        prefix = (curriculum is not None and self.cfg["recovery_training"].get(
            "prefix_curriculum", {}).get("enabled", False))
        result = {"/recovery/script_fraction": recovery.prelude.float().mean(),
            "/recovery/eligible_policy_fraction": recovery.learning_mask.float().mean()}
        if getattr(recovery, "direct_takeover", False) and recovery.sensor_status is not None:
            sensor = recovery.sensor_status
            result.update({"/recovery/sensor_time_valid_fraction": sensor.valid.float().mean(),
                "/recovery/sensor_motor_age_max_ms": sensor.maximum_motor_age_ms.max(),
                "/recovery/sensor_imu_age_max_ms": sensor.imu_age_ms.max(),
                "/recovery/sensor_acceleration_age_max_ms": sensor.acceleration_age_ms.max()})
        if prefix:
            curriculum.record_outcomes(done, success, terminated, timeouts)
            result.update(curriculum.lesson_metrics(recovery.true_success))
        else:
            # Old fixed evaluation retains its original population semantics.
            result["/recovery/full_chain_success"] = recovery.true_success.float().mean()
        if curriculum is not None:
            if getattr(self, "integrated_runtime_enabled", False):
                from wheeled_tasks.chassis.application.compatibility.integrated_runtime import component_actor_updates
                clock = component_actor_updates(self, "recovery")
            else:
                clock = stage_actor_updates(self)
            result.update(curriculum.metrics(clock))
        if self.recovery_takeover_settings is not None or prefix:
            diagnostic = recovery.fsm.diagnostics()
            holding = recovery.enabled & diagnostic["recovery_holding_rl"]
            denominator = holding.sum().clamp_min(1)
            for name in ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed"):
                passed = diagnostic["recovery_stable_" + name + "_pass"]
                result["/recovery/hold_gate/" + name + "_pass_fraction"] = (holding & passed).sum() / denominator
            result["/recovery/hold_population"] = holding.sum()
            result["/recovery/stable_peak_seconds_mean"] = diagnostic["recovery_stable_peak_seconds"].mean()
            result["/recovery/stable_break_count_mean"] = diagnostic["recovery_stable_break_count"].float().mean()
        if self.recovery_hold_cost_diagnostics:
            values = self.recovery_hold_cost_diagnostics
            denominator = values["mask"].sum().clamp_min(1)
            result["/HoldCost/actor_hold_population"] = values["mask"].sum()
            for axis in ("leg", "wheel", "gyro"):
                for name in ("gate_relative_error", "broad_density", "fine_density"):
                    result["/HoldCost/" + axis + "/" + name + "_mean"] = values[axis + "/" + name].sum() / denominator
        return result

    def _accumulate_diagnostic_rewards(self, components, support, ordinary):
        from wheeled_tasks.chassis.application.training.diagnostic_curriculum import DiagnosticRewardAccumulator
        reference = self.references
        normal = ordinary.bool() & ~reference.jumping & (reference.terrain_mode == 0)
        stationary = normal & (self.commands[:, :2].abs() < .01).all(-1)
        masks = {"height_precision": support.bool() & ~reference.jumping,
                 "velocity_precision": normal & ~stationary,
                 "stand_precision": stationary & support.bool(),
                 "stationary_speed_precision": stationary & support.bool()}
        cfg = self.cfg["precision_tracking"]
        weights = {"height_precision": cfg["height_weight"], "velocity_precision": cfg["velocity_weight"],
                   "stand_precision": cfg["stationary_weight"],
                   "stationary_speed_precision": self.cfg["reference_reward"]["stationary_weight"]}
        if not hasattr(self, "diagnostic_reward_stats"):
            self.diagnostic_reward_stats = DiagnosticRewardAccumulator(
                self.num_envs, self.device, self.skills.batches, weights)
        self.diagnostic_reward_stats.observe(components, masks)

    def diagnostic_reward_metrics(self):
        return self.diagnostic_reward_stats.drain(self.policy_dt)

    def summary(self):
        net_max = max(float(wp.to_torch(v.get_net_contact_forces(dt=self.dt)).abs().max()) for _, v in self.contact_views)
        return {"policy_ticks": self.tick, "transitions": self.tick * self.num_envs,
            "max_closure_gap_m": self.max_closure_gap, "phase_samples": self.phase_counts.cpu().tolist(),
            **({"max_closure_gap_scope": "returned policy intervals excluding numerical-reset lanes",
                "max_closure_gap_all_finite_m": self.max_closure_gap_all_finite,
                "max_closure_gap_numerical_reset_m": self.max_closure_gap_numerical_reset,
                "closure_peak_scope": "finite values across every physics substep of returned policy intervals in this worker; aborted calls remain in physical captures"}
               if self.scut_reward is not None else {}),
            "spring_reserve_frames": self.spring_reserve_frames,
            "termination_counts": self.termination_counts, "successes": self.success_count,
            "timeouts": self.timeout_count, "terrain_families": self.kinds,
            "scene_group_counts": self.startup_report["scene_group_counts"],
            "final_relative_height_m": self.state()[3].cpu().tolist(),
            "final_filtered_wheel_force_n": self.contact_force[:, self.wheel_ids].norm(dim=-1).cpu().tolist(),
            "final_unfiltered_force_max_component_n": net_max}

    def close(self):
        self.sim.stop()
