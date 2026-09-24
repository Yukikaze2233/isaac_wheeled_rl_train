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
from .task import FallConfirmation, Phase, PhaseTracker, Surface, corridor_mesh, choose_terrains, choose_scene_groups, phase_reward_masks, terrain_surfaces


class ChassisEnv:
    """Stock RSL VecEnv protocol with explicit, terrain-filtered contact identity."""

    def __init__(self, config, manifest, control, root: Path, *, stage_name, num_envs,
                 device="cuda:0", level=0.0, seed=617, coverage=False):
        self.cfg = dict(config)
        self.cfg.update(stage=stage_name, num_envs=num_envs, terrain_level=level, seed=seed)
        self.stage_cfg = next(s for s in config["stages"] if s["name"] == stage_name)
        self.num_envs, self.num_actions, self.device = num_envs, 6, device
        self.control, self.manifest = control, manifest
        self.is_v5 = manifest["model_kind"] == "v5_gas_spring_closedchain_research"
        self.scut35 = config.get("actor_observation_source") == "scut35_encoders_imu_commands"
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
        self.decimation = round(self.policy_dt / self.dt)
        self.max_episode_length = round(config["episode_seconds"] / self.policy_dt)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.level = level
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=device, render_interval=self.decimation))
        import carb.settings
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.training_transitions = 0
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
        grid = math.ceil(math.sqrt(num_envs))
        floor_width = config.get("flat_floor_width_m", 4.)
        if not 4. <= floor_width <= 8.:
            raise ValueError("Flat collider width must stay within the validated 4-8m tile range")
        half_length = config.get("flat_half_length_m", 44.)
        corridor_spacing = 2 * half_length + 12.
        corridor_columns = max(1, math.ceil(math.sqrt(num_envs * (floor_width + 1.) / corridor_spacing)))
        corridor_rows = math.ceil(num_envs / corridor_columns)
        origins, self.surfaces = [], []
        for i, kind in enumerate(kinds):
            # A single line of 4096 corridors reaches tens of kilometres and
            # loses millimetre precision in float32 world-space constraints.
            origin = (((i % corridor_columns) - (corridor_columns - 1) / 2) * corridor_spacing,
                      ((i // corridor_columns) - (corridor_rows - 1) / 2) * (floor_width + 1.), 0.) if config.get("evaluation_long_corridors") else ((i % grid) * 10., (i // grid) * 10., 0.)
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
            surfaces = terrain_surfaces(kind, limits, level * tiers[(i // len(tiers)) % len(tiers)], terrain_index)
            if config.get("evaluation_long_corridors") and kind in ("flat", "jump"):
                # Preserve the validated collider aspect ratio. A single 80m
                # thin cuboid changed wheel contact in the GPU PhysX probe.
                surfaces = ([Surface(-half_length, half_length)] if config.get("flat_triangle_mesh")
                            else [Surface(x - 4., x + 4.) for x in range(-40, 41, 8)])
            self.surfaces.append(surfaces)
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
                material_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=.5, dynamic_friction=.5,
                    restitution=0., friction_combine_mode="average")
                material_cfg.func(material_path, material_cfg)
                sim_utils.bind_physics_material(str(mesh.GetPath()), material_path)
                continue
            for j, surface in enumerate(surfaces):
                size, position, quat = surface.box(width=floor_width if kind in ("flat", "jump") else 4.)
                if config.get("playback_open_ground"):
                    size = (size[0], 80., size[2])
                # Both robot and default material are 0.5; average combine yields the requested mu.
                mu = 2 * surface.friction - 0.5
                cfg = sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu,
                        dynamic_friction=mu, restitution=0., friction_combine_mode="average"))
                cfg.func(path + f"/Terrain/surface_{j}", cfg, translation=position, orientation=quat)
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
        material = sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5,
            restitution=0., friction_combine_mode="average")
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
        if set(self.robot.joint_names) != set(manifest["tree_joint_names"]) or set(self.robot.body_names) != set(manifest["rigid_body_names"]):
            raise RuntimeError("Unexpected solver topology")
        paths = list(self.robot.root_view.prim_paths)
        order = [int(p.split("/env_")[1].split("/")[0]) for p in paths]
        if sorted(order) != list(range(num_envs)):
            raise RuntimeError("Ambiguous PhysX clone order")
        self.origins = torch.tensor([origins[i] for i in order], device=device)
        self.clone_indices = order
        self.kinds = [kinds[i] for i in order]
        self.scene_groups = [groups[i] for i in order]
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
        if not torch.allclose(masses.sum(-1), masses.new_full((num_envs,), manifest["total_mass_kg"]), atol=1e-5, rtol=0):
            raise RuntimeError("Mass mismatch after cloning")
        self.startup_report = {"body_count": self.body_count, "tree_dofs": self.joint_count, "loop_constraints": loop_count,
            "num_envs": num_envs, "terrain_families": self.kinds,
            "mass_kg_each": masses.sum(-1).cpu().tolist(), "contact_filters": filters,
            "self_collision_enabled": False, "solver_order": order}
        scene = self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)
        self.startup_report.update(physics_device=str(device),
            gpu_dynamics_enabled=bool(scene.GetAttribute("physxScene:enableGPUDynamics").Get()),
            state_tensor_device=str(self.robot.data.root_link_pose_w.torch.device))
        self.startup_report["scene_group_counts"] = {name: self.scene_groups.count(name) for name in set(self.scene_groups)}
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
            self.skills = SkillCommands(self)
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
        if config.get("signal_perturbations", {}).get("enabled", False):
            from .robustness import V5SignalPerturbations
            self.perturbations = V5SignalPerturbations(num_envs, device,
                {**config["signal_perturbations"], "frame_dim": config["actor_frame_dim"]}, self.generator)
        self.route_goal = torch.tensor([1.8 if kind in ("stairs", "stairs_down", "slope", "slope_up", "slope_down", "rough", "cross_slope", "material") else config.get("task_semantics", {}).get("route_goal_x_m", .65)
                                        for kind in self.kinds], device=device)
        self.motor_strength = torch.ones(num_envs, 6, device=device)
        self.spring_strength = torch.ones(num_envs, 1, device=device)
        # Terrain/reset origins are immutable within this environment. Cache
        # them once instead of issuing O(num_envs) tiny CUDA writes per reset.
        offsets = []
        for group, kind in zip(self.scene_groups, self.kinds):
            x = 0. if config.get("centered_locomotion_resets") and kind in ("flat", "jump") else -1.5
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
        self.reset(torch.arange(num_envs, device=device))

    def random(self, count):
        return torch.rand(count, device=self.device, generator=self.generator)

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

    def reset(self, ids):
        count = len(ids)
        root = torch.zeros(count, 7, device=self.device)
        root[:, :3], root[:, 6] = self.reset_positions[ids], 1.
        reset_velocity = torch.zeros(count, 6, device=self.device)
        if self.skills is not None:
            self.skills.reset_pose(ids, root, reset_velocity)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=reset_velocity, env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=self.nominal[ids], env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(count, self.joint_count, device=self.device), env_ids=ids)
        self.robot.reset(ids)
        self.robot.update(self.dt)
        self.actions[ids] = 0
        self.previous_actions[ids] = 0
        self.before_previous_actions[ids] = 0
        self.previous_motor_velocity[ids] = 0
        self.contact_force[ids] = 0
        self.episode_length_buf[ids] = 0
        self.success_hold[ids] = 0
        self.jump_requested[ids] = False
        self.phase.reset(ids)
        self.fall_confirmation.reset(ids)
        if self.stationary_anchor is not None:
            self.stationary_anchor.reset(ids)
        if self.performance_curriculum is not None:
            self.performance_curriculum.reset_episodes(ids)
        if self.skills is not None:
            self.skills.reset_phase(ids)
        self.history.reset(ids)
        if self.perturbations is not None:
            self.perturbations.reset(ids)
            self.motor_strength[ids] = .85 + .15 * torch.rand(count, 6, generator=self.generator, device=self.device)
            self.spring_strength[ids] = .9 + .2 * torch.rand(count, 1, generator=self.generator, device=self.device)
        self.command_clock[ids] = 0
        self.height_clock[ids] = 0
        self.push_clock[ids] = 5 + 2 * self.random(count)
        self.push_enabled[ids] = self.random(count) < 0.5
        self.resample_commands(ids, reset_height=True)
        if self.full_tasks is not None:
            self.full_tasks.reset(ids, self.robot.data.root_link_pose_w.torch[:, :3] - self.origins)
        self.update_targets()

    def resample_commands(self, ids, *, reset_height=False):
        n = len(ids)
        previous_velocity_commands = self.commands[ids, :2].clone()
        if self.skills is not None:
            # SkillCommands samples entire groups on the device. Its values
            # replace the legacy per-environment sampler completely.
            self.skills.sample(ids, reset_height=reset_height)
            if self.cfg.get("command_slew"):
                self.command_target[ids] = self.commands[ids, :2]
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
            self.skills.sample(ids)
        if self.full_tasks is not None:
            self.height_clock[ids[self.mode[ids] >= 2]] = 1e9
        if self.cfg.get("command_slew"):
            self.command_target[ids] = self.commands[ids, :2]
            self.commands[ids, :2] = 0. if reset_height else previous_velocity_commands
        if self.skills is None:
            self.command_clock[ids] = 3 + 2 * self.random(n)

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

    def get_observations(self):
        velocity, omega, gravity, height, q, dq, _ = self.state()
        if self.scut35:
            from .scut_observation import build_scut35
            requested = self.jump_requested & (self.mode == 4)
            elapsed = self.episode_length_buf * self.policy_dt - self.cfg["task_semantics"]["jump_request_seconds"]
            lateral = self.targets[:, 1] * self.skills.spin if self.skills is not None else None
            frame = build_scut35(omega, gravity, self.commands, q, dq, self.actions,
                self.v5.nominal, requested, self.targets[:, 1] * (self.mode == 4), elapsed, lateral)
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
        if frame.shape[-1] != self.cfg["actor_frame_dim"] or critic.shape[-1] != self.cfg["critic_dim"]:
            raise RuntimeError("New task observation layout mismatch")
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
        if self.scut35:
            from .scut_observation import CONTROL_FROM_POLICY
            actions = actions[:, CONTROL_FROM_POLICY]
        self.before_previous_actions.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        q = self.robot.data.joint_pos.torch[:, self.ids]
        applied_actions = self.perturbations.action(actions, self.tick) if self.perturbations is not None else actions
        legs, wheels, clipped = self.v5.decode(applied_actions, q) if self.is_v5 else decode_targets(applied_actions, q, self.control)
        if self.perturbations is not None:
            clipped = actions.clamp(-self.v5.action_bounds, self.v5.action_bounds)
        self.actions.copy_(clipped)
        self.contact_peak.zero_()
        for _ in range(self.decimation):
            q_now, dq_now = self.robot.data.joint_pos.torch[:, self.ids], self.robot.data.joint_vel.torch[:, self.ids]
            self.torque = (self.v5.motor_efforts(q_now, dq_now, legs, wheels) if self.is_v5 else
                           compute_torques(q_now, dq_now, legs, wheels, self.control))
            self.torque *= self.motor_strength
            effort = torch.zeros_like(self.nominal)
            effort[:, self.ids] = self.torque
            if self.is_v5:
                effort[:, self.spring_ids] = self.v5.spring_efforts(self.robot.data.joint_pos.torch[:, self.spring_ids]) * self.spring_strength
            self.robot.set_joint_effort_target_index(target=effort)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.dt)
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
            self.contact_peak = torch.maximum(self.contact_peak, self.contact_force[:, self.wheel_ids].norm(dim=-1).sum(-1))
        self.tick += 1
        self.episode_length_buf += 1
        velocity, omega, gravity, height, q, dq, local = self.state()
        if self.cfg.get("reward_velocity_reference") == "base_link_origin":
            velocity = self.robot.data.root_link_lin_vel_b.torch
        vertical_velocity = -(velocity * gravity).sum(-1)
        if not all(bool(torch.isfinite(t).all()) for t in (velocity, omega, height, q, dq, self.contact_force)):
            raise RuntimeError("Nonfinite physical transition")
        gap = self.closure_gap()
        self.max_closure_gap = max(self.max_closure_gap, float(gap.max()))
        if self.max_closure_gap > 0.003 and not self.cfg.get("closure_gap_termination_m"):
            raise RuntimeError(f"Closed-chain numerical gap exceeds 3 mm: {self.max_closure_gap}")
        contact = self.contact_force[:, self.wheel_ids].norm(dim=-1) > 2.
        stable = (gravity[:, 2] < -0.985) & ((height - self.commands[:, 2]).abs() < 0.01)
        self.phase.update(contact, self.targets[:, 3].bool() & ~self.phase.flew, stable)
        self.phase_counts += torch.bincount(self.phase.phase, minlength=5)
        masks = phase_reward_masks(self.phase.phase)
        grounded = masks["ground"].float()
        flight = masks["flight"].float()
        support_tracking = grounded + (self.phase.phase == Phase.RECOVERY).float() * self.cfg.get("track_height_during_recovery", False)
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
                + 2 * support_tracking * torch.exp(-((height - self.commands[:, 2]) / 0.03).square())
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
            reward += self.cfg.get("height_l1_weight", 0.) * support_tracking * (height - self.commands[:, 2]).abs() * self.policy_dt
        components = None
        if use_locomotion_rewards:
            from .rewards import height_tracking_terms, reward_terms
            root = self.robot.data.root_link_pose_w.torch
            delta = self.wheel_centers() - root[:, None, :3]
            inverse_xyz = -root[:, None, 3:6].expand(-1, 2, -1)
            cross = torch.linalg.cross(inverse_xyz, delta)
            wheel_b = delta + 2 * (root[:, None, 6:] * cross + torch.linalg.cross(inverse_xyz, cross))
            ordinary = ~self.skills.spin if self.skills is not None else torch.ones_like(height, dtype=torch.bool)
            nonwheel = self.contact_force[:, self.nonwheel_ids].norm(dim=-1).amax(-1) > 5.
            components = reward_terms(velocity, omega, gravity, height, self.commands, dq,
                (dq - self.previous_motor_velocity) / self.policy_dt,
                self.robot.data.applied_torque.torch[:, self.ids], self.actions, self.previous_actions,
                self.before_previous_actions, wheel_b, support_tracking, ordinary, nonwheel,
                self.performance_curriculum.env_height_width if self.performance_curriculum is not None else None)
            if self.performance_curriculum is not None:
                scales = self.performance_curriculum.env_height_scale
                components["track_height"] *= scales
                components["height_square"] *= scales
            if self.stationary_anchor is not None:
                requested = self.stationary_group & (self.mode == 0) & (self.command_target.abs().amax(-1) < .01)
                supported = support_tracking.bool() & contact.all(-1)
                components["stationary_anchor"] = self.stationary_anchor.reward(local[:, :2], requested, supported)
                components["stand_translation"] *= torch.where(requested,
                    self.cfg["stationary_tracking"]["velocity_scale"], 1.)
            if self.cfg.get("height_tracking"):
                settings = self.cfg["height_tracking"]
                moving_height = self.skills.height_motion
                active = support_tracking * moving_height
                error = height - self.commands[:, 2]
                vz_error = vertical_velocity - self.skills.height_velocity_reference
                components.update(height_tracking_terms(error, vz_error, support_tracking, moving_height, settings))
                components["vertical_velocity"] = torch.where(moving_height, -.5 * active * vz_error.square(),
                                                              components["vertical_velocity"])
            if self.cfg.get("landing_tracking"):
                settings = self.cfg["landing_tracking"]
                landing = ((self.phase.phase == Phase.LANDING) | (self.phase.phase == Phase.RECOVERY)).float()
                h_ref, v_ref = self.full_tasks.landing_reference(self.commands[:, 2], self.phase.contact_time)
                passive_landing = landing * (self.mode != 4)
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
            reward = torch.stack(list(components.values())).sum(0) * self.policy_dt
            if spin_reward:
                reward += self.skills.spin * velocity_reward * self.policy_dt
        self.previous_motor_velocity.copy_(dq)
        success_now = (self.mode >= 2) & (local[:, 0] > 1.8) & contact.all(-1) & stable
        success_now &= (self.mode == 3) | self.phase.flew
        if self.full_tasks is not None and (self.has_route_tasks or self.has_jump_tasks):
            centers = self.wheel_centers() - self.origins[:, None, :]
            support_heights = torch.stack([self.ground_height(centers[:, side, :2]) for side in range(2)], -1)
            clearance = centers[:, :, 2] - .06 - support_heights
            vertical_velocity = wp.to_torch(self.robot.root_view.get_root_velocities())[:, 2]
            self.full_tasks.observe(self.mode, height, clearance, self.phase.phase, vertical_velocity)
            if self.has_jump_tasks and self.cfg.get("task_semantics", {}).get("jump_com_rise_m") is not None:
                com_height = (self.robot.data.body_com_pose_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                com_vz = (self.robot.data.body_com_lin_vel_w.torch[:, :, 2] * self.body_mass).sum(-1) / self.body_mass.sum(-1)
                self.full_tasks.observe_com(self.mode, self.phase.phase, com_height, com_vz)
            if self.has_jump_tasks:
                jump_reward = self.full_tasks.dense_jump_reward(self.mode, self.phase, height, vertical_velocity, self.commands[:, 2], extension)
                reward += jump_reward
                if components is not None:
                    components["dense_jump"] = jump_reward / self.policy_dt
            progress_reward = self.full_tasks.route_progress_reward(self.mode, local)
            reward += progress_reward
            if components is not None:
                components["route_progress"] = progress_reward / self.policy_dt
            success_now = self.full_tasks.completion(self.mode, local, centers[:, :, 0], self.route_goal,
                                                     contact, stable, self.phase.phase, self.commands[:, 2], velocity[:, :2].norm(dim=-1))
        if self.skills is not None:
            success_now |= self.skills.completion(contact, stable)
        self.success_hold = torch.where(success_now, self.success_hold + self.policy_dt, 0.)
        success = self.success_hold >= self.cfg.get("task_semantics", {}).get("success_hold_seconds", 1.)
        reasons = {
            "fall": self.fall_confirmation.update(gravity[:, 2], height, masks["flight"]),
            "nonwheel_contact": (self.contact_force[:, self.nonwheel_ids].norm(dim=-1).amax(-1) > 5) & (self.episode_length_buf * self.policy_dt > .2),
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
            reasons["boundary"] = (local[:, 0].abs() > x_limit) | (local[:, 1].abs() > y_limit)
        if self.full_tasks is not None:
            elapsed = self.episode_length_buf * self.policy_dt
            jumping = self.mode == 4
            reasons["jump_request_timeout"] = jumping & (elapsed > 2.) & ~self.phase.flew
            reasons["jump_outcome_timeout"] = jumping & (elapsed > 6.) & ~success
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
        terminated = torch.stack([v for k, v in reasons.items() if k != "boundary"]).any(0)
        success &= ~terminated
        reward += success.float() * 5.
        timeouts = ((self.episode_length_buf >= self.episode_limits) | reasons["boundary"]) & ~terminated & ~success
        done = terminated | timeouts | success
        if self.performance_curriculum is not None:
            reference_error = (self.skills.reference_filtered - self.skills.reference_target).norm(dim=-1)
            reference_error = torch.where(self.skills.spin, reference_error, velocity[:, 0] - self.commands[:, 0])
            self.performance_curriculum.observe(height - self.commands[:, 2], velocity[:, 0] - self.commands[:, 0],
                omega[:, 2] - self.commands[:, 1], support_tracking.bool() & contact.all(-1), done,
                terminated | reasons["boundary"], self.training_transitions / self.cfg["curriculum_reference_batch"],
                reference_velocity_error=reference_error, commands=self.commands)
        reward -= terminated.float() * self.cfg.get("termination_event_cost", 1.)
        for name, mask in reasons.items():
            self.termination_counts[name] += int(mask.sum())
        self.success_count += int(success.sum())
        self.timeout_count += int(timeouts.sum())
        extras = {"time_outs": timeouts, "log": {"/task/pin_gap_m": gap.mean(),
            "/task/height_error_m": (height - self.commands[:, 2]).abs().mean(),
            "/task/success": success.float().mean(), "/task/flight": flight.mean(),
            "/task/wheel_contact": contact.float().mean()}}
        if self.cfg.get("fall_confirmation_seconds", 0.) > 0:
            extras["log"].update({
                "/termination/fall_candidate_fraction": self.fall_confirmation.candidate.float().mean(),
                "/termination/fall_confirmed_fraction": reasons["fall"].float().mean(),
                "/termination/nonwheel_contact_fraction": reasons["nonwheel_contact"].float().mean()})
        if components is not None:
            extras["log"].update({"/reward/" + name: value.mean() * self.policy_dt for name, value in components.items()})
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
                "motor_effort": self.robot.data.applied_torque.torch[:, self.ids].clone(),
                "reward": reward.clone(), "gap": gap.clone(), "done": done.clone(),
                "terminated": terminated.clone(), "success": success.clone(),
                "reasons": {name: mask.clone() for name, mask in reasons.items()}}
            if self.full_tasks is not None:
                extras["diagnostics"].update(jump_clearance_peak=self.full_tasks.clearance_peak.clone(),
                    jump_air_time_peak=self.full_tasks.clear_air_time_peak.clone(), jump_height_peak=self.full_tasks.height_peak.clone(),
                    jump_release_velocity=self.full_tasks.release_velocity.clone(),
                    jump_com_rise=self.full_tasks.com_rise.clone(), jump_com_release_speed=self.full_tasks.com_release_speed.clone())
            if self.skills is not None:
                extras["diagnostics"].update(self.skills.diagnostics(velocity))
            if self.cfg.get("height_tracking"):
                extras["diagnostics"]["height_velocity_error"] = (
                    vertical_velocity - self.skills.height_velocity_reference) * self.skills.height_motion
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids) and self.cfg.get("auto_reset", True):
            self.reset(ids)
        self.command_clock -= self.policy_dt
        self.height_clock -= self.policy_dt
        self.push_clock -= self.policy_dt
        change = (self.command_clock <= 0) & ~done & (self.mode < 2)
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
        if pushed.any():
            ids = pushed.nonzero(as_tuple=False).flatten()
            self.apply_push(ids)
        if self.cfg.get("command_slew"):
            if self.skills is not None:
                self.skills.update()
            rates = self.commands.new_tensor([self.cfg["command_slew"]["vx_m_s2"], self.cfg["command_slew"]["yaw_rad_s2"]])
            change = (self.command_target - self.commands[:, :2]).clamp(-rates * self.policy_dt, rates * self.policy_dt)
            self.commands[:, :2] += change
            if self.skills is not None:
                # Rotating a fixed world-frame request is a coordinate change,
                # not a physical acceleration request to be rate-limited twice.
                self.commands[self.skills.spin, 0] = self.command_target[self.skills.spin, 0]
        self.update_targets()
        return self.get_observations(), reward, done, extras

    def apply_push(self, ids):
        velocity_w = wp.to_torch(self.robot.root_view.get_root_velocities())[ids].clone()
        angle = self.random(len(ids)) * math.tau
        cap = self.skills.push_speed[ids] if self.skills is not None else self.stage_cfg["push_max"]
        amplitude = self.random(len(ids)) * cap
        velocity_w[:, :2] += torch.stack((angle.cos(), angle.sin()), -1) * amplitude[:, None]
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity_w, env_ids=ids, full_data=False)
        self.push_clock[ids] = 3 + 2 * self.random(len(ids))

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
