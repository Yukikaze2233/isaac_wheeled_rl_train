"""Isaac Lab 3 / PhysX multi-scene environment for the 15-body closed chain."""
from __future__ import annotations

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
from isaaclab_physx.physics import PhysxManager

from wheeled_tasks.v40.core import HistoryStack, build_observation, compute_torques, decode_targets
from .preflight import GAS_SPRING_MODEL_KINDS, SubstepControlAudit, design_preflight, validate_runtime_state
from .actuator_response import ActuatorResponse
from .root_kinematics import StaticRootLinkKinematics, root_link_linear_velocity, update_robot_data
from .training_schedule import stage_actor_updates
from .terrain_runout import append_route_runout, apply_route_boundary, route_runout_selection, validate_runout
from .task import FallConfirmation, Phase, PhaseTracker, Surface, corridor_mesh, choose_terrains, choose_scene_groups, episode_outcome, phase_reward_masks, terrain_surfaces


class ChassisEnv:
    """Stock RSL VecEnv protocol with explicit, terrain-filtered contact identity."""

    def __init__(self, config, manifest, control, root: Path, *, stage_name, num_envs,
                 device="cuda:0", level=0.0, seed=617, coverage=False, run_role=None, block_index=0,
                 integrated_design_preflight=None):
        if config.get("integrated_runtime_candidate", {}).get("static_recipe_preflight"):
            raise ValueError("Static joint recipe cannot construct a native environment")
        if (config.get("scut_heading_command", {}).get("enabled")
                and not config.get("evaluation_exact_cases") and run_role not in (None, "train")):
            raise ValueError("SCUT heading command is restricted to explicit training")
        if integrated_design_preflight is not None and (not config.get("integrated_production")
                or config.get("evaluation_exact_cases") or run_role not in (None, "train")):
            raise ValueError("Joint design admission is restricted to explicit training")
        if config.get("integrated_production") and not config.get("evaluation_exact_cases") and integrated_design_preflight is None:
            raise ValueError("Joint production environment requires authenticated design admission")
        self.cfg = dict(config)
        self.cfg.update(stage=stage_name, num_envs=num_envs, terrain_level=level, seed=seed)
        self.integrated_runtime_enabled = bool(config.get("integrated_runtime_candidate")) and not config.get("evaluation_exact_cases")
        self.route_runout_settings = validate_runout(config)
        role = run_role or ("full" if config.get("evaluation_exact_cases") else "train")
        self.design_preflight = design_preflight(config, control, root, stage=stage_name,
            role=role, seed=seed, num_envs=num_envs, block_index=block_index)
        if integrated_design_preflight is not None:
            from .integrated_training import validate_design_hook
            self.design_preflight = validate_design_hook(config, root, self.design_preflight,
                integrated_design_preflight, stage=stage_name, num_envs=num_envs,
                seed=seed, block_index=block_index)
        self.control_audit = SubstepControlAudit(config["physics_dt"], config["policy_dt"])
        self.stage_cfg = next(s for s in config["stages"] if s["name"] == stage_name)
        self.num_envs, self.num_actions, self.device = num_envs, 6, device
        self.control, self.manifest = control, manifest
        self.is_v5 = manifest["model_kind"] in GAS_SPRING_MODEL_KINDS
        self.scut35 = config.get("actor_observation_source") in ("scut35_encoders_imu_commands", "encoders_imu_command_reference36")
        self.references = None
        self.step_jump = self.step_climb = self.step_sensors = None
        self.recovery = None
        self.nonstandard_start = None
        self.ordinary_posture = None
        self.body_count = len(manifest["rigid_body_names"])
        self.joint_count = len(manifest["tree_joint_names"])
        self.v5 = None
        self.height_margin = None
        if self.is_v5:
            from .v5_control import V5Control
            directory = root / config["asset_directory"]
            self.model_spec = json.loads((directory / "model_spec.json").read_text())
            fit = json.loads((directory / "fit_10mpa.json").read_text())
            self.v5 = V5Control(manifest, self.model_spec, fit, control, config["v5_control"], device)
            if config.get("height_workspace"):
                reference = config["height_workspace"]
                if (reference["model_spec_sha256"] != manifest["files_sha256"]["model_spec.json"]
                        or not math.isclose(reference["knee_working_margin_rad"], config["knee_working_margin_rad"])):
                    raise ValueError("Commanded-height margin reference differs from the mechanical model")
                from .rewards import CommandedHeightMargin
                self.height_margin = CommandedHeightMargin(reference, device)
        self.dt, self.policy_dt = config["physics_dt"], config["policy_dt"]
        self.passive_landing = None
        if config.get("landing_tracking"):
            from .rewards import PassiveLandingWindow, validate_landing_tracking
            if validate_landing_tracking(config["landing_tracking"]) == "finite_task_cost_v1":
                self.passive_landing = PassiveLandingWindow(num_envs, device, self.policy_dt,
                                                            config["landing_tracking"])
        self.jump_landing = None
        self.jump_cycle_reset_on_accepted = config.get("jump_cycle_reset_on_accepted", False)
        if type(self.jump_cycle_reset_on_accepted) is not bool:
            raise ValueError("Jump accepted-cycle reset must be an explicit boolean")
        if config.get("jump_landing", {}).get("enabled"):
            from .jump_landing import JumpLandingWindow
            self.jump_landing = JumpLandingWindow(num_envs, device, self.policy_dt, config["jump_landing"])
        self.landing_reward_mask = torch.zeros(num_envs, device=device)
        self.actuator_response = ActuatorResponse(num_envs, device, self.dt,
            config.get("actuator_response"))
        if self.actuator_response.enabled and (not self.is_v5 or not config.get("monitor_applied_effort")):
            raise ValueError("Identified response requires explicit six-axis V5 effort monitoring")
        self.decimation = self.control_audit.decimation
        self.max_episode_length = round(config["episode_seconds"] / self.policy_dt)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.level = level
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=device, render_interval=self.decimation,
            enable_scene_query_support=config.get("enable_scene_queries", False)))
        import carb.settings
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        if config.get("dense_tracking", {}).get("enabled"):
            from .rewards import validate_dense_tracking
            validate_dense_tracking(config["dense_tracking"])
        if "locomotion_posture" in config.get("dense_tracking", {}):
            if (not config["dense_tracking"].get("enabled")
                    or config.get("reward_profile") != "scut_v14_flat_v5"
                    or not config.get("command_reference")
                    or not config.get("reference_reward", {}).get("visible_reference_only")):
                raise ValueError("Locomotion posture floor requires active dense tracking and visible reference rewards")
        self.recovery_takeover_settings = None
        if config.get("recovery_takeover", {}).get("enabled"):
            from .rewards import validate_recovery_takeover
            self.recovery_takeover_settings = dict(config["recovery_takeover"], dense_tracking=config["dense_tracking"])
            validate_recovery_takeover(self.recovery_takeover_settings)
            if not config.get("recovery_training", {}).get("enabled"):
                raise ValueError("Recovery takeover shaping requires the physical recovery prelude")
        self.recovery_hold_cost_settings = None
        self.recovery_hold_cost_diagnostics = {}
        if "recovery_hold_cost" in config:
            from .rewards import validate_recovery_hold_cost
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
        runout_groups = ([route_runout_selection(config, group, kind)
                          for group, kind in zip(groups, kinds)]
                         if self.route_runout_settings is not None else None)
        if runout_groups is not None and not any(runout_groups):
            raise ValueError("Enabled terrain route runout has no selected training rows")
        layout_indices, layout_count = list(range(num_envs)), num_envs
        if config.get("evaluation_exact_cases") and config["evaluation"].get("stable_case_layout"):
            from .evaluation import canonical_evaluation_layout
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
            from .contact_domain import ContactDomain
            profiles = None
            if config.get("evaluation_exact_cases"):
                cases = {c["name"]: c for c in config["evaluation"]["cases"]}
                profiles = [cases[name].get("contact_profile") for name in groups]
            self.contact_domain = ContactDomain(num_envs, config["contact_domain"], seed, profiles)
        combine = "multiply" if self.contact_domain is not None else "average"
        for i, kind in enumerate(kinds):
            # A single line of 4096 corridors reaches tens of kilometres and
            # loses millimetre precision in float32 world-space constraints.
            slot = layout_indices[i]
            origin = (((slot % corridor_columns) - (corridor_columns - 1) / 2) * corridor_spacing,
                      ((slot // corridor_columns) - (corridor_rows - 1) / 2) * (floor_width + 1.), 0.) if config.get("evaluation_long_corridors") else ((slot % grid) * 10., (slot // grid) * 10., 0.)
            origins.append(origin)
            path = f"/World/envs/env_{i}"
            UsdGeom.Xform.Define(self.sim.stage, path).AddTranslateOp().Set(origin)
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
                from .release_stop import release_stop_surfaces
                surfaces = release_stop_surfaces(kind, limits, level * terrain_level,
                    terrain_index, config["release_stop"])
            if config.get("evaluation_long_corridors") and kind in ("flat", "jump"):
                # Preserve the validated collider aspect ratio. A single 80m
                # thin cuboid changed wheel contact in the GPU PhysX probe.
                surfaces = ([Surface(-half_length, half_length)] if config.get("flat_triangle_mesh")
                            else [Surface(x - 4., x + 4.) for x in range(-40, 41, 8)])
            if runout_groups is not None and runout_groups[i]:
                surfaces = append_route_runout(surfaces, self.route_runout_settings)
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
                material_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=surfaces[0].friction,
                    dynamic_friction=surfaces[0].friction, restitution=0., friction_combine_mode=combine)
                material_cfg.func(material_path, material_cfg)
                sim_utils.bind_physics_material(str(mesh.GetPath()), material_path)
                continue
            for j, surface in enumerate(surfaces):
                bottom = None
                if config.get("solid_step_platforms") and kind in ("low_step", "step_up", "step_down", "platform", "stairs", "stairs_down"):
                    bottom = min(s.z0 for s in surfaces) - .1
                size, position, quat = surface.box(width=floor_width if kind in ("flat", "jump") else 4., solid_bottom=bottom)
                if config.get("playback_open_ground"):
                    size = (size[0], 80., size[2])
                # Both robot and default material are 0.5; average combine yields the requested mu.
                mu = surface.friction if self.contact_domain is not None else 2 * surface.friction - 0.5
                cfg = sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu,
                        dynamic_friction=mu, restitution=0., friction_combine_mode=combine))
                # Surface.box uses USD's scalar-first quaternion; Isaac Lab 3
                # spawners take scalar-last quaternions.
                cfg.func(path + f"/Terrain/surface_{j}", cfg, translation=position,
                         orientation=(*quat[1:], quat[0]))
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
        robot_mu = 1. if self.contact_domain is not None else .5
        material = sim_utils.RigidBodyMaterialCfg(static_friction=robot_mu, dynamic_friction=robot_mu,
            restitution=0., friction_combine_mode=combine)
        material.func("/World/RobotMaterial", material)
        loop_count = 0
        for prim in self.sim.stage.Traverse():
            if "/Robot/" not in str(prim.GetPath()):
                continue
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
        self.sim.reset()
        if self.contact_domain is not None:
            properties = wp.to_torch(self.robot.root_view.get_material_properties())
            if not torch.allclose(properties[..., :2], torch.ones_like(properties[..., :2]), atol=1e-6):
                raise RuntimeError("Unit robot friction material was not imported by PhysX")
        if set(self.robot.joint_names) != set(manifest["tree_joint_names"]) or set(self.robot.body_names) != set(manifest["rigid_body_names"]):
            raise RuntimeError("Unexpected solver topology")
        paths = list(self.robot.root_view.prim_paths)
        order = [int(p.split("/env_")[1].split("/")[0]) for p in paths]
        if sorted(order) != list(range(num_envs)):
            raise RuntimeError("Ambiguous PhysX clone order")
        self.origins = torch.tensor([origins[i] for i in order], device=device)
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
        filters = [[p + f"/Terrain/surface_{j}/geometry/mesh" for j in range(len(self.surfaces[i]))]
                   for i, p in enumerate(self.env_paths) for _ in self.robot.body_names]
        view = PhysxManager.get_physics_sim_view()
        # Each filter must name one body; filter-list lengths must match within a view.
        self.contact_views = []
        for surface_count in sorted({len(s) for s in self.surfaces}):
            ids = [i for i, s in enumerate(self.surfaces) if len(s) == surface_count]
            group_paths = [self.env_paths[i] + "/Robot/" + n for i in ids for n in self.robot.body_names]
            group_filters = [[self.env_paths[i] + f"/Terrain/surface_{j}/geometry/mesh" for j in range(surface_count)]
                             for i in ids for _ in self.robot.body_names]
            contact_view = view.create_rigid_contact_view(group_paths, filter_patterns=group_filters,
                max_contact_data_count=64 * len(group_paths))
            if contact_view.filter_count != surface_count:
                raise RuntimeError("Terrain contact filter count mismatch")
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
            from .dynamics import RigidBodyRandomization
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
        self.startup_report["terrain_collision_paths"] = [str(p.GetPath()) for p in self.sim.stage.Traverse()
            if "/Terrain/" in str(p.GetPath()) and p.HasAPI(UsdPhysics.CollisionAPI)]
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
        else:
            self.knee_bounds = control["joints"]["knee_hard_limits"]
        self.nominal = torch.tensor([[manifest["nominal_joint_pos"][n] for n in self.robot.joint_names]], device=device).repeat(num_envs, 1)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
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
            from .step_assist import StepHeightAssist
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
            from .torque_monitor import TorqueMonitor
            self.torque_monitor = TorqueMonitor(self.scene_groups, manifest["control_joint_names"], device,
                                                control["actuators"]["wheel"]["effort_limit"])
        self.full_tasks, self.perturbations = None, None
        self.performance_curriculum, self.stationary_anchor = None, None
        self.stationary_group = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.skills = None
        if config.get("skill_specs"):
            from .skill_commands import SkillCommands
            self.skills = SkillCommands(self, root=root)
        if config.get("stationary_tracking"):
            from .rewards import StationaryAnchor
            settings = config["stationary_tracking"]
            self.stationary_anchor = StationaryAnchor(num_envs, device, settings["position_weight"], settings["position_band_m"])
            self.stationary_group[:] = torch.tensor([
                self.skills.specs.get(group, {}).get("kind") in ("stand", "height")
                for group in self.scene_groups], device=device)
        if config.get("performance_curriculum") and not config.get("evaluation_exact_cases"):
            from .performance_curriculum import PerformanceCurriculum
            curriculum_type = PerformanceCurriculum
            if config["performance_curriculum"].get("kind") == "adaptive_commands":
                from .adaptive_commands import AdaptiveCommandCurriculum
                curriculum_type = AdaptiveCommandCurriculum
            self.performance_curriculum = curriculum_type(
                self.scene_groups, self.skills.specs, config["performance_curriculum"], device)
        if config.get("task_semantics"):
            from .full_tasks import FullTaskSemantics
            semantics = dict(config["task_semantics"])
            if self.skills is not None:
                overrides = [self.skills.specs.get(group, {}).get("semantics", {}) for group in self.scene_groups]
                for key in {key for values in overrides for key in values}:
                    semantics[key] = torch.tensor([values.get(key, config["task_semantics"].get(key, 0.))
                                                   for values in overrides], device=device)
            self.full_tasks = FullTaskSemantics(num_envs, device, self.policy_dt, semantics)
        if config.get("command_reference"):
            from .references import CommandReference
            if not self.scut35 or self.body_count != 19 or self.full_tasks is None:
                raise ValueError("Command-reference control requires the declared V5 task interface")
            self.references = CommandReference(num_envs, device, config["command_reference"])
        if config.get("step_jump_task", {}).get("enabled"):
            from .step_jump import StepJumpTask
            from .step_climb import StepClimbTask
            from .step_sensors import StepTaskSensors
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
            from .robustness import V5SignalPerturbations
            self.perturbations = V5SignalPerturbations(num_envs, device,
                {**config["signal_perturbations"], "frame_dim": config["actor_frame_dim"]}, self.generator)
        self.signal_delay = None
        if config.get("signal_delay", {}).get("enabled"):
            if (not self.scut35 or not self.is_v5 or config.get("usb_transport", {}).get("enabled")
                    or self.perturbations is not None and self.perturbations.max_lag):
                raise ValueError("Physical signal delay requires SCUT35 and no second policy-tick delay")
            from .signal_delay import PhysicalSignalDelay
            self.signal_delay = PhysicalSignalDelay(num_envs, device, config["signal_delay"], self.dt, self.generator)
            self.startup_report["signal_delay"] = dict(config["signal_delay"],
                physics_dt_s=self.dt, policy_dt_s=self.policy_dt,
                quantum_ms=self.dt * 1000., physics_steps_observed=0)
        self.usb_transport = None
        if config.get("usb_transport", {}).get("enabled", False):
            if not self.scut35 or not self.is_v5:
                raise ValueError("USB transport requires the V5 SCUT35 control interface")
            if self.perturbations is not None and self.perturbations.max_lag:
                raise ValueError("USB transport cannot silently stack with policy-tick delay")
            from .usb_transport import SubstepUsbTransport
            self.usb_transport = SubstepUsbTransport(num_envs, device, self.dt, config["usb_transport"], seed)
            self.startup_report["usb_transport"] = {
                "enabled": True, "model": "substep_rtt_sensor_interpolation_and_torque_impulse",
                "payload_size": config["usb_transport"]["payload_size"],
                "jitter": "iid_per_physics_step_quantile_surrogate", "directional_split": "research_assumption"}
        self.route_goal = torch.tensor([1.8 if kind in ("stairs", "stairs_down", "slope", "slope_up", "slope_down", "rough", "cross_slope", "material") else config.get("task_semantics", {}).get("route_goal_x_m", .65)
                                        for kind in self.kinds], device=device)
        self.motor_strength = torch.ones(num_envs, 6, device=device)
        self.command_transport = None
        if config.get("command_transport", {}).get("enabled"):
            if self.usb_transport is not None:
                raise ValueError("Downlink transport must not double-count legacy USB RTT")
            from .command_transport import UsbCommandTransport
            profiles = None
            if config.get("evaluation_exact_cases"):
                cases = {c["name"]: c for c in config["evaluation"]["cases"]}
                profiles = [cases[name].get("communication_profile") for name in self.scene_groups]
            self.command_transport = UsbCommandTransport(num_envs, device, self.dt,
                config["command_transport"], seed, profiles, domain_draw=self.domain_draw)
            self.startup_report["command_transport"] = dict(config["command_transport"])
            self.startup_report["command_transport"].update(physics_dt_s=self.dt,
                transport_dt_s=self.command_transport.transport_dt,
                transport_ticks_per_physics_step=self.command_transport.substeps,
                application=("phase_aware_interval_torque_average" if self.command_transport.phase_aware
                    else "interval_held_torque_average" if self.command_transport.substeps > 1 else "one_tick_hold"))
        self.spring_strength = torch.ones(num_envs, 1, device=device)
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
        self.reset_positions[:, 2] += self.ground_height(self.reset_positions[:, :2] - self.origins[:, :2]) + .32
        if self.step_jump is not None:
            selected = self.step_jump.enabled | self.step_climb.enabled
            self.reset_positions[selected, 0] = self.origins[selected, 0] - .8
            self.reset_positions[selected, 2] = self.origins[selected, 2] + .32
        self.jump_assist = None
        if config.get("jump_assist", {}).get("enabled"):
            from .step_assist import JumpAssist
            self.jump_assist = JumpAssist(num_envs, device, config["jump_assist"])
            if (bool(config.get("integrated_runtime_candidate")) and not config.get("evaluation_exact_cases")):
                owners, settings = self._component_configuration("jump_assist")
                self.jump_assist.configure_components(owners, settings)
            self.assist_body_ids = torch.tensor([self.robot.body_names.index("base_link")], device=device)
            self.assist_force = torch.zeros(num_envs, 1, 3, device=device)
        if config.get("recovery_training", {}).get("enabled"):
            from .recovery_training import RecoveryTraining
            self.recovery = RecoveryTraining(self, root, config["recovery_training"])
            if (self.recovery_hold_cost_settings is not None
                    and self.recovery_hold_cost_settings["enabled"]):
                from .rewards import validate_recovery_hold_gates
                self.recovery_hold_cost_gate_binding = validate_recovery_hold_gates(
                    self.recovery_hold_cost_settings, self.recovery.fsm, self.recovery.geometry)
            for reason in self.recovery.failure_reasons():
                self.termination_counts[reason] = 0
            self.startup_report["recovery_training"] = {
                "enabled": True, "count": int(self.recovery.enabled.sum()),
                "reference_dt": .005, "pd_dt": self.dt, "blend_seconds": .2,
                "state_writes": "episode_initial_reset_only",
                "learning": "exclude_script_and_blend_transitions"}
        if "nonstandard_start" in config:
            from .nonstandard_start import NonstandardStart
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
            from .ordinary_posture_curriculum import OrdinaryPostureCurriculum, ordinary_eligibility
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
        self.reset(torch.arange(num_envs, device=device))
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
                if "prefix_curriculum" in config["recovery_training"]:
                    from .recovery_prefix_curriculum import runtime_expectation
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
                from .recovery_script_revision import native_runtime_binding
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
                from .terrain_runout_runtime import observe_runtime
                actual["modules"]["terrain_route_runout"] = self.route_runout_settings is not None
                actual["terrain_route_runout"] = observe_runtime(self)
                self.startup_report["terrain_route_runout"] = actual["terrain_route_runout"]
            if "release_stop" in config:
                actual["modules"]["release_stop"] = self.skills.release_stop.enabled
                actual["release_stop"] = self.skills.release_stop.runtime_state()
                self.startup_report["release_stop"] = actual["release_stop"]
            self.startup_report["design_preflight"] = self.design_preflight
            if self.integrated_runtime_enabled:
                from .integrated_preflight import observe_component_clocks, validate_integrated_runtime_state
                actual["integrated_curriculum_observed"] = observe_component_clocks(self)
                self.startup_report["runtime_preflight"] = validate_integrated_runtime_state(
                    self.design_preflight["runtime_expected"], actual, config=self.cfg,
                    num_envs=self.num_envs, actor_update=stage_actor_updates(self))
            else:
                self.startup_report["runtime_preflight"] = validate_runtime_state(
                    self.design_preflight["runtime_expected"], actual)

    def _has_integrated_runtime(self):
        return bool(self.cfg.get("integrated_runtime_candidate")) and not self.cfg.get("evaluation_exact_cases")

    def _component_configuration(self, field):
        """Startup-only binding; components share existing device state and geometry."""
        from .integrated_runtime import row_settings
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
        from .integrated_runtime import component_actor_updates
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

    def _update_step_jump(self):
        if self.step_sensors is None:
            return
        _, omega, gravity, _, q, dq, local = self.state()
        if self.recovery is not None:
            # Share the chronological encoder stream used by the 1 kHz loop.
            # Project this observation without advancing its history a second
            # time; USB-delayed and ideal encoders must never be mixed.
            if self.usb_transport is not None:
                q, dq = self.usb_transport.feedback[:, :6], self.usb_transport.feedback[:, 6:12]
                omega, gravity = self.usb_transport.feedback[:, 12:15], self.usb_transport.feedback[:, 15:18]
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

    def reset(self, ids):
        count = len(ids)
        root = torch.zeros(count, 7, device=self.device)
        root[:, :3], root[:, 6] = self.reset_positions[ids], 1.
        reset_velocity = torch.zeros(count, 6, device=self.device)
        if self.skills is not None:
            self.skills.reset_pose(ids, root, reset_velocity)
        if self.step_jump is not None:
            selected = (self.step_jump.enabled | self.step_climb.enabled)[ids]
            root[selected, 0] = self.origins[ids[selected], 0] - .8
        joints = self.nominal[ids].clone()
        if self.recovery is not None:
            self.recovery.initial_state(ids, root, joints, reset_velocity)
        posture = getattr(self, "ordinary_posture", None)
        if posture is not None and posture.enabled:
            posture.reset(ids, (self._curriculum_clock("ordinary_posture_curriculum") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self)), generator=self.generator)
            posture.initial_orientation(ids, root)
        if self.nonstandard_start is not None:
            from .nonstandard_start import support_height_for_footprint
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
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(count, self.joint_count, device=self.device), env_ids=ids)
        self.robot.reset(ids)
        update_robot_data(self, self.dt)
        if self.step_jump is not None:
            self.step_jump.reset(ids)
            self.step_climb.reset(ids)
        if self.step_sensors is not None:
            self.step_sensors.reset(ids, root_link_linear_velocity(self))
        self.actuator_response.reset(ids)
        if self.usb_transport is not None:
            self.usb_transport.reset(ids, self._transport_sensors())
        if self.command_transport is not None:
            self.command_transport.reset(ids)
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
            self.spring_strength[clean] = 1.
        self.command_clock[ids] = 0
        self.height_clock[ids] = 0
        self.push_clock[ids] = 5 + 2 * self.random(count)
        self.push_enabled[ids] = self.random(count) < 0.5
        if posture is not None and posture.enabled:
            ordinary = ids[posture.eligible[ids]]
            self.push_clock[ordinary] = posture.sample_intervals(ordinary, generator=self.generator)
            self.push_enabled[ordinary] = True
        self.resample_commands(ids, reset_height=True)
        self.raw_height[ids] = self.commands[ids, 2]
        if self.recovery is not None:
            self.recovery.reset(ids)
        if self.full_tasks is not None:
            self.full_tasks.reset(ids, self.robot.data.root_link_pose_w.torch[:, :3] - self.origins)
        self.update_targets()
        self.update_height_assist(ids, advance=False)
        self.update_command_reference(ids)

    def resample_commands(self, ids, *, reset_height=False):
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

    def get_observations(self):
        velocity, omega, gravity, height, q, dq, _ = self.state()
        if self.scut35:
            from .scut_observation import build_scut35
            requested = self.jump_requested & (self.mode == 4)
            elapsed = self.episode_length_buf * self.policy_dt - self.cfg["task_semantics"]["jump_request_seconds"]
            lateral = self.targets[:, 1] * self.skills.spin if self.skills is not None else None
            frame = build_scut35(omega, gravity, self.commands, q, dq, self.actions,
                self.v5.nominal, requested, self.targets[:, 1] * (self.mode == 4), elapsed, lateral)
            if self.references is not None:
                from .scut_observation import build_manual35, build_reference36
                builder = build_manual35 if self.cfg.get("manual_context35") else build_reference36
                frame = builder(omega, gravity, self.commands, q, dq, self.actions,
                    self.v5.nominal, requested, self.targets[:, 1] * (self.mode == 4), self.references, lateral)
        else:
            frame = (self.v5.proprioception(omega, gravity, self.commands, q, dq, self.actions) if self.is_v5 else
                     build_observation(omega, gravity, self.commands, q, dq, self.actions, self.control))
            if self.is_v5:
                compression, speed = self.v5.spring_state(self.robot.data.joint_pos.torch[:, self.spring_ids],
                                                         self.robot.data.joint_vel.torch[:, self.spring_ids])
                frame = torch.cat((frame, compression / .08, speed / 1.2), -1)
            contacts = (self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2).float()
            target = self.targets * frame.new_tensor([0.5, 5., 0.5, 1.])
            frame = torch.cat((frame, F.one_hot(self.mode, 5).float(), target.clamp(-10, 10), contacts,
                               F.one_hot(self.phase.phase, 5).float(), self.phase.time.clamp_max(5)[:, None]), -1)
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
        if self.usb_transport is not None:
            sensors = self.usb_transport.feedback
            frame = build_scut35(sensors[:, 12:15], sensors[:, 15:18], self.commands,
                sensors[:, :6], sensors[:, 6:12], self.actions, self.v5.nominal,
                requested, self.targets[:, 1] * (self.mode == 4), elapsed, lateral)
            if self.references is not None:
                frame = builder(sensors[:, 12:15], sensors[:, 15:18], self.commands,
                    sensors[:, :6], sensors[:, 6:12], self.actions, self.v5.nominal,
                    requested, self.targets[:, 1] * (self.mode == 4), self.references, lateral)
        if self.signal_delay is not None:
            from .scut_observation import build_sensor18
            sensors = self.signal_delay.feedback
            frame[:, 4:22] = build_sensor18(sensors[:, 12:15], sensors[:, 15:18],
                sensors[:, :6], sensors[:, 6:12], self.v5.nominal)
        if self.perturbations is not None:
            frame = self.perturbations.observation(frame, self.tick)
        return TensorDict({"policy": self.history.update(frame, self.tick), "critic": critic}, batch_size=[self.num_envs])

    def closure_gap(self):
        pose = self.robot.data.body_link_pose_w.torch[:, self._closure_body_ids]
        local = self._closure_points.expand(self.num_envs, -1, -1, -1)
        cross = torch.linalg.cross(pose[..., 3:6], local)
        world = pose[..., :3] + local + 2 * (pose[..., 6:] * cross
                + torch.linalg.cross(pose[..., 3:6], cross))
        return (world[:, :, 0] - world[:, :, 1]).norm(dim=-1).amax(-1)

    def step(self, actions):
        self.control_audit.begin_policy()
        if self.signal_delay is not None:
            if getattr(self, "integrated_runtime_enabled", False):
                from .integrated_runtime import component_actor_updates
                delay_clock = component_actor_updates(self, "speed")
            else:
                delay_clock = stage_actor_updates(self)
            self.signal_delay.begin_policy(delay_clock)
        if self.jump_assist is not None and getattr(self, "integrated_runtime_enabled", False):
            self.jump_assist_actor_clock = self.jump_assist.actor_clock_tensor((self._curriculum_clock("jump_assist") if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self)))
        if self.recovery is not None:
            self.recovery.begin_policy()
        if self.nonstandard_start is not None:
            self.nonstandard_start.begin_policy()
        if self.scut35:
            from .scut_observation import CONTROL_FROM_POLICY
            actions = actions[:, CONTROL_FROM_POLICY]
        self.before_previous_actions.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        q = self.robot.data.joint_pos.torch[:, self.ids]
        if self.usb_transport is not None:
            q = self.usb_transport.feedback[:, :6]
        if self.recovery is not None:
            q = self.recovery.project_feedback(q)
        applied_actions = self.perturbations.action(actions, self.tick) if self.perturbations is not None else actions
        legs, wheels, clipped = self.v5.decode(applied_actions, q) if self.is_v5 else decode_targets(applied_actions, q, self.control)
        self.control_audit.targets_decoded()
        if self.perturbations is not None:
            clipped = actions.clamp(-self.v5.action_bounds, self.v5.action_bounds)
        self.actions.copy_(clipped)
        self.contact_peak.zero_()
        if self.wheel_force_peak_n is not None:
            self.wheel_force_peak_n.zero_()
        for substep in range(self.decimation):
            q_now, dq_now = self.robot.data.joint_pos.torch[:, self.ids], self.robot.data.joint_vel.torch[:, self.ids]
            feedback_q, feedback_dq = q_now, dq_now
            if self.usb_transport is not None:
                feedback_q, feedback_dq = self.usb_transport.feedback[:, :6], self.usb_transport.feedback[:, 6:12]
            if self.recovery is not None:
                control_q = self.recovery.observe_feedback(feedback_q)
            else:
                control_q = feedback_q
            self.control_audit.feedback_control()
            pd_legs, pd_wheels = (self.signal_delay.targets(legs, wheels)
                                 if self.signal_delay is not None else (legs, wheels))
            self.torque = (self.v5.motor_efforts(control_q, feedback_dq, pd_legs, pd_wheels) if self.is_v5 else
                           compute_torques(q_now, dq_now, legs, wheels, self.control))
            if self.recovery is not None and self.recovery.has_tasks:
                if substep % self.recovery.reference_stride == 0:
                    gravity = self.robot.data.projected_gravity_b.torch
                    gyro = self.robot.data.root_com_ang_vel_b.torch
                    if self.usb_transport is not None:
                        gyro, gravity = self.usb_transport.feedback[:, 12:15], self.usb_transport.feedback[:, 15:18]
                    self.recovery.advance_reference(feedback_q, feedback_dq, gravity, gyro)
                self.torque = self.recovery.torque(feedback_q, feedback_dq, self.torque)
            self.torque *= self.motor_strength
            if self.skills is not None:
                self.torque *= self.skills.schedule.motor_scale
            if self.nonstandard_start is not None:
                # Feed zero into drive history during the passive interval.
                # The final gate below also removes retained/filter output.
                self.torque = torch.where(self.nonstandard_start.holding[:, None], 0., self.torque)
            if self.command_transport is not None:
                from wheeled_tasks.v40.core import motor_torque_limit
                self.torque = self.command_transport.apply_torque(self.torque)
                # Without delayed feedback, motor_efforts just computed the
                # envelope at this exact physical velocity. Transport changes
                # torque only; reuse that bound instead of repeating GPU work.
                bounds = (self.v5.current_motor_bounds[:, list(self.v5.WHEELS)]
                    if self.usb_transport is None else
                    motor_torque_limit(dq_now[:, list(self.v5.WHEELS)], self.v5.wheel_prior))
                self.torque[:, list(self.v5.WHEELS)] = self.torque[:, list(self.v5.WHEELS)].clamp(-bounds, bounds)
            if self.usb_transport is not None:
                from wheeled_tasks.v40.core import motor_torque_limit
                self.torque = self.usb_transport.apply_torque(self.torque)
                # Physical drive limits still depend on actual, not delayed, shaft speed.
                bound = motor_torque_limit(dq_now[:, list(self.v5.WHEELS)], self.v5.wheel_prior)
                wheels_tau = self.torque[:, list(self.v5.WHEELS)]
                self.torque[:, list(self.v5.WHEELS)] = torch.maximum(torch.minimum(wheels_tau, bound), -bound)
                self.v5.current_motor_bounds[:, list(self.v5.WHEELS)] = bound
            if self.actuator_response.enabled:
                self.torque = self.actuator_response.apply_torque(self.torque)
                # The filtered effort still obeys the instantaneous drive envelope.
                # This also clips the retained motor state, preventing windup.
                bound = self.v5.current_motor_bounds
                self.torque.clamp_(min=-bound, max=bound)
            if self.nonstandard_start is not None:
                self.torque = self.nonstandard_start.apply_torque(self.torque)
            effort = torch.zeros_like(self.nominal)
            effort[:, self.ids] = self.torque
            if self.is_v5:
                effort[:, self.spring_ids] = self.v5.spring_efforts(self.robot.data.joint_pos.torch[:, self.spring_ids]) * self.spring_strength
            if self.jump_assist is not None:
                supported = (self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.).any(-1)
                pushing = (self.references.phase == self.references.PUSH) & (self.mode == 4)
                self.assist_force[:, 0, 2] = self.jump_assist.force(self.body_mass.sum(-1), self.targets[:, 1],
                    pushing, supported, self.phase.phase == Phase.FLIGHT, self.dt,
                    self.jump_assist_actor_clock if getattr(self, "integrated_runtime_enabled", False) else stage_actor_updates(self),
                    evaluation=self.cfg.get("evaluation_exact_cases", False))
                self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                    forces=self.assist_force, body_ids=self.assist_body_ids, is_global=True)
            self.robot.set_joint_effort_target_index(target=effort)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            update_robot_data(self, self.dt)
            if self.nonstandard_start is not None:
                self.nonstandard_start.advance()
            self.control_audit.physics_updated(self.dt)
            if self.signal_delay is not None:
                self.signal_delay.observe(self._transport_sensors())
            if self.usb_transport is not None:
                self.usb_transport.observe(self._transport_sensors())
            if self.torque_monitor is not None:
                applied = self.robot.data.applied_torque.torch
                velocity = self.robot.data.joint_vel.torch
                compression, _ = self.v5.spring_state(self.robot.data.joint_pos.torch[:, self.spring_ids], velocity[:, self.spring_ids])
                self.torque_monitor.observe(applied[:, self.ids], velocity[:, self.ids],
                    applied[:, self.spring_ids], velocity[:, self.spring_ids], compression,
                    self.v5.requested_motor_effort, self.v5.current_motor_bounds)
            for ids, contact_view in self.contact_views:
                matrix = wp.to_torch(contact_view.get_contact_force_matrix(dt=self.dt))
                self.contact_force[ids] = matrix.reshape(len(ids), self.body_count, -1, 3).sum(2)
            if self.references is not None and self.has_jump_tasks:
                com_height = (self.robot.data.body_com_pose_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                com_vz = (self.robot.data.body_com_lin_vel_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                self.full_tasks.observe_com(self.mode, self.phase.phase, com_height, com_vz,
                    requested=self.references.jumping, contacts=self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.)
            wheel_force_n = self.contact_force[:, self.wheel_ids].norm(dim=-1)
            self.contact_peak = torch.maximum(self.contact_peak, wheel_force_n.sum(-1))
            if self.wheel_force_peak_n is not None:
                self.wheel_force_peak_n.copy_(torch.maximum(self.wheel_force_peak_n, wheel_force_n))
        self.startup_report["control_timing"]["runtime_substep_verification"] = self.control_audit.end_policy()
        if self.signal_delay is not None:
            self.startup_report["signal_delay"]["physics_steps_observed"] = self.signal_delay.physics_steps
        if self.recovery is not None and self.recovery.has_tasks:
            self.recovery.effective_action_history()
        if self.nonstandard_start is not None:
            self.actions.copy_(self.nonstandard_start.effective_action_history(self.actions))
        self.tick += 1
        self.episode_length_buf += 1
        velocity, omega, gravity, height, q, dq, local = self.state()
        if self.cfg.get("reward_velocity_reference") == "base_link_origin":
            velocity = root_link_linear_velocity(self, frame="body")
        vertical_velocity = -(velocity * gravity).sum(-1)
        finite = torch.stack([torch.isfinite(t).all()
                              for t in (velocity, omega, height, q, dq, self.contact_force)]).all()
        if not bool(finite):
            raise RuntimeError("Nonfinite physical transition")
        gap = self.closure_gap()
        self.max_closure_gap = max(self.max_closure_gap, float(gap.max()))
        if self.max_closure_gap > 0.003 and not self.cfg.get("closure_gap_termination_m"):
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
            from .rewards import height_tracking_terms, reward_terms
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
                    from .rewards import passive_landing_costs
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
                from .rewards import reference_motion_terms
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
                    from .rewards import apply_manual_tracking
                    apply_manual_tracking(components, velocity, omega, gravity, self.commands,
                        reward_support_tracking, ordinary, self.references, self.cfg["reference_reward"])
                if self.cfg.get("precision_tracking") and not self.cfg.get("dense_tracking", {}).get("enabled"):
                    from .rewards import precision_tracking_terms
                    components.update(precision_tracking_terms(velocity, gravity, height, self.commands,
                        reward_support_tracking, ordinary, self.references, wheel_b, self.cfg["precision_tracking"]))
            if self.cfg.get("dense_tracking", {}).get("enabled"):
                from .rewards import apply_dense_tracking
                spin_error = ((self.skills.reference_filtered - self.skills.reference_target).norm(dim=-1)
                              if spin_reward else None)
                apply_dense_tracking(components, velocity, omega, gravity, height, self.commands,
                    reward_support_tracking, ordinary, self.references, self.cfg["dense_tracking"],
                    spin_velocity_error=spin_error)
            if (self.recovery_hold_cost_settings is not None
                    and self.recovery_hold_cost_settings["enabled"]):
                from .rewards import apply_recovery_hold_costs
                apply_recovery_hold_costs(components, dq, omega, self.recovery.enabled,
                    self.recovery.learning_mask, self.recovery.policy_motion_hold,
                    self.recovery_hold_cost_settings, diagnostics=self.recovery_hold_cost_diagnostics)
            if self.recovery_takeover_settings is not None:
                from .rewards import apply_recovery_takeover
                apply_recovery_takeover(components, gravity, omega, height, velocity, self.commands, dq,
                    self.recovery.enabled & self.recovery.learning_mask,
                    self.recovery.policy_motion_hold, self.recovery_takeover_settings)
            if self.step_climb is not None and self.step_climb.has_tasks:
                from .rewards import apply_climb_height_tracking
                scale = self.step_climb.height_tracking_scale(self.wheel_centers() - self.origins[:, None, :])
                apply_climb_height_tracking(components, scale)
            reward = torch.stack(list(components.values())).sum(0) * self.policy_dt
            if spin_reward and not self.cfg.get("dense_tracking", {}).get("enabled"):
                reward += self.skills.spin * velocity_reward * self.policy_dt
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
                    from .rewards import reference_jump_terms
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
                            com_rise=self.full_tasks.com_rise)
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
                    if components is not None:
                        components.update(terms)
                else:
                    jump_reward = self.full_tasks.dense_jump_reward(self.mode, self.phase, height, vertical_velocity, self.commands[:, 2], extension)
                reward += jump_reward
                if components is not None and self.references is None:
                    components["dense_jump"] = jump_reward / self.policy_dt
            progress_reward = self.full_tasks.route_progress_reward(self.mode, local)
            reward += progress_reward
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
            if components is not None:
                components.update(terms)
            success_now[self.step_climb.enabled] = self.step_climb.success[self.step_climb.enabled]
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
        if self.cfg.get("closure_gap_termination_m"):
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
                if components is not None:
                    components["working_margin"] = margin_density
            else:
                reward -= 2 * ((compression / self.v5.stroke - .9).clamp_min(0)).square().sum(-1) * self.policy_dt
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
        from .rewards import episode_event_terms
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
        if self.signal_delay is not None:
            extras["log"].update(self.signal_delay.metrics())
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
        return self.get_observations(), reward, done, extras

    def apply_push(self, ids):
        velocity_w = wp.to_torch(self.robot.root_view.get_root_velocities())[ids].clone()
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
        if prefix:
            curriculum.record_outcomes(done, success, terminated, timeouts)
            result.update(curriculum.lesson_metrics(recovery.true_success))
        else:
            # Old fixed evaluation retains its original population semantics.
            result["/recovery/full_chain_success"] = recovery.true_success.float().mean()
        if curriculum is not None:
            if getattr(self, "integrated_runtime_enabled", False):
                from .integrated_runtime import component_actor_updates
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
        from .diagnostic_curriculum import DiagnosticRewardAccumulator
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
            "spring_reserve_frames": self.spring_reserve_frames,
            "termination_counts": self.termination_counts, "successes": self.success_count,
            "timeouts": self.timeout_count, "terrain_families": self.kinds,
            "scene_group_counts": self.startup_report["scene_group_counts"],
            "final_relative_height_m": self.state()[3].cpu().tolist(),
            "final_filtered_wheel_force_n": self.contact_force[:, self.wheel_ids].norm(dim=-1).cpu().tolist(),
            "final_unfiltered_force_max_component_n": net_max}

    def close(self):
        self.sim.stop()
