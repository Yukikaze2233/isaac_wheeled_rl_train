#!/usr/bin/env python3
"""Native PhysX spring/linkage bench: fixed base, force-driven sliders, passive loops.

Only reset writes joint state. The offline lookup supplies active motor targets;
no passive coordinates or rigid-body poses are overwritten during simulation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

# Kit spawns helper processes during startup; avoid a pre-existing OpenBLAS thread pool at fork.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TITLE = "V5 FULL MECHANISM | PHYSX FIXED-BASE | 10 MPa SPRINGS"


def read_bundle(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["model_kind"] != "v5_gas_spring_closedchain_research":
        raise ValueError("Expected the V5 19-body spring model")
    for name, expected in manifest["files_sha256"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle dependency mismatch: {name}")
    spec = json.loads((path / "model_spec.json").read_text())
    fit = json.loads((path / "fit_10mpa.json").read_text())
    # Solve target poses offline, before starting the engine; use only the six active coordinates at runtime.
    sys.path.insert(0, str(path / "tools"))
    from build_v5_closedchain import POSE_COORDINATES, solve_pose
    from v5_mechanism import fk, point
    angles = np.linspace(55., 78., 47)
    rows, gravity_rows, spring_jacobians = [], [], []
    names = [j["name"] for j in spec["joints"]]
    active = [names.index(n) for n in manifest["control_joint_names"]]
    passive = [i for i in range(len(names)) if i not in active]
    spring_ids = [names.index(n) for n in manifest["spring_joint_names"]]

    def geometry_values(q):
        frames = fk(spec, q)
        residual = np.concatenate([point(frames[c["body0"]], c["local_pos0_m"])
                                   - point(frames[c["body1"]], c["local_pos1_m"]) for c in spec["constraints"]])
        potential = sum(b["mass"] * 9.81 * point(frames[b["name"]], b["com"])[2] for b in spec["bodies"])
        return residual, potential

    q = spec["nominal_joint_pos"]
    for angle in angles:
        knee = math.radians(angle) - (math.pi - 2.3573)
        prescribed = dict(zip(POSE_COORDINATES, [.42, knee, 0., -.42, -knee, 0.]))
        q = solve_pose(spec, prescribed, q)
        rows.append([q[n] for n in manifest["control_joint_names"]])
        jacobian, gradient = [], []
        for name in names:
            plus, minus = dict(q), dict(q)
            plus[name] += 1e-6
            minus[name] -= 1e-6
            rp, vp = geometry_values(plus)
            rm, vm = geometry_values(minus)
            jacobian.append((rp - rm) / 2e-6)
            gradient.append((vp - vm) / 2e-6)
        jacobian = np.column_stack(jacobian)
        mapping = np.zeros((len(names), 6))
        mapping[active] = np.eye(6)
        mapping[passive] = np.linalg.lstsq(jacobian[:, passive], -jacobian[:, active], rcond=1e-8)[0]
        if np.max(np.abs(jacobian @ mapping)) > 1e-7:
            raise RuntimeError("Invalid closed-chain virtual-work mapping")
        gravity_rows.append(np.array(gradient) @ mapping)
        spring_jacobians.append(mapping[spring_ids])
    return manifest, spec, fit, angles, np.array(rows), np.array(gravity_rows), np.array(spring_jacobians)


class SpringBench:
    def __init__(self, args, manifest, spec, fit, angles, targets, gravity_rows, spring_jacobians):
        import torch
        import warp as wp
        from pxr import UsdPhysics
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        self.args, self.manifest, self.spec, self.fit = args, manifest, spec, fit
        self.angles, self.targets = angles, targets
        self.gravity_rows, self.spring_jacobians = gravity_rows, spring_jacobians
        self.dt, self.time = args.dt, 0.
        self.target_angle = 67.85497
        self.auto, self.gas, self.paused = True, True, False
        self.reset_requested = False
        self.max_gap = 0.
        self.max_pin_length_error = 0.
        self.q_min = None
        self.q_max = None
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=self.dt, render_interval=20))
        floor = sim_utils.CuboidCfg(size=(3., 3., .05), collision_props=sim_utils.CollisionPropertiesCfg())
        floor.func("/World/Ground", floor, translation=(0., 0., -.025))
        active = manifest["control_joint_names"]
        springs = manifest["spring_joint_names"]
        passive = [n for n in manifest["tree_joint_names"] if n not in active + springs]
        cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(args.bundle / "robot.usda"),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=8),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., .15),
                joint_pos=manifest["nominal_joint_pos"], joint_vel={".*": 0.}),
            actuators={
                "active": IdealPDActuatorCfg(joint_names_expr=active, stiffness=0., damping=0.,
                    effort_limit=40., effort_limit_sim=40., velocity_limit_sim=1000., armature=0.),
                "passive": IdealPDActuatorCfg(joint_names_expr=passive, stiffness=0., damping=.002,
                    effort_limit=100., effort_limit_sim=100., velocity_limit_sim=1000., armature=0.),
                "gas": IdealPDActuatorCfg(joint_names_expr=springs, stiffness=0., damping=.002,
                    effort_limit=1000., effort_limit_sim=1000., velocity_limit_sim=100., armature=0.),
            }, soft_joint_pos_limit_factor=1.)
        self.robot = Articulation(cfg)
        self.stage = self.sim.stage
        loops = [p for p in self.stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
        if len(loops) != 6 or not all(UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get() for p in loops):
            raise RuntimeError("Expected six external loop constraints")
        self.sim.reset()
        if set(self.robot.body_names) != set(manifest["rigid_body_names"]) or set(self.robot.joint_names) != set(manifest["tree_joint_names"]):
            raise RuntimeError("PhysX body/joint topology mismatch")
        self.active_ids = [self.robot.joint_names.index(n) for n in active]
        self.spring_ids = [self.robot.joint_names.index(n) for n in springs]
        self.s0 = torch.tensor([spec["spring_binding"][n]["compression_at_q_zero_m"] for n in springs], device=args.device)
        limits = wp.to_torch(self.robot.root_view.get_dof_limits()).cpu().numpy()[0]
        self.solver_report = {"bodies": self.robot.body_names, "joints": self.robot.joint_names,
            "mass_kg": float(wp.to_torch(self.robot.root_view.get_masses()).sum()), "loops": len(loops),
            "fixed_base_fixture": True, "dt": self.dt, "position_iterations": 32, "velocity_iterations": 8}
        for j in spec["joints"]:
            if j["type"] in ("prismatic", "revolute"):
                expected = np.array([float(j["limit"]["lower"]), float(j["limit"]["upper"])])
                if not np.allclose(limits[self.robot.joint_names.index(j["name"])] , expected, atol=1e-6, rtol=0):
                    raise RuntimeError(f"PhysX slider/knee limit mismatch: {j['name']}")
        self.nominal = torch.tensor([[manifest["nominal_joint_pos"][n] for n in self.robot.joint_names]], device=args.device)
        self.efforts = torch.zeros_like(self.nominal)
        self.reset()

    def reset(self):
        import torch
        self.robot.write_joint_position_to_sim_index(position=self.nominal)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(self.nominal))
        self.robot.reset()
        self.robot.update(self.dt)
        self.time = 0.
        self.target_angle = 67.85497
        self.reset_requested = False

    def step(self):
        import torch
        if self.reset_requested:
            self.reset()
        target = torch.tensor([[np.interp(self.target_angle, self.angles, self.targets[:, i]) for i in range(6)]],
                              device=self.args.device, dtype=torch.float32)
        q, dq = self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch
        error = target - q[:, self.active_ids]
        error = torch.atan2(error.sin(), error.cos())
        motor = 60. * error - 2. * dq[:, self.active_ids]
        compression = self.s0 - q[:, self.spring_ids]
        # Do not silently extrapolate the fitted catalogue force in this interactive bench.
        if bool((compression < -.0005).any()) or bool((compression > .0725).any()):
            raise RuntimeError(f"Spring left the preview's fitted working domain: {compression.tolist()}")
        u = compression.clamp(0., .072) / self.fit["stroke_m"]
        a, b, c, d = self.fit["monomial_coefficients_n"]
        force = (a + u * (b + u * (c + u * d))) * float(self.gas)
        # A suspended leg has no chassis load. Counter its spring preload using
        # an offline virtual-work table so the inspection servo can traverse the stroke.
        gravity = torch.tensor([np.interp(self.target_angle, self.angles, self.gravity_rows[:, i]) for i in range(6)],
                               device=self.args.device, dtype=torch.float32)
        jacobian = torch.tensor([[np.interp(self.target_angle, self.angles, self.spring_jacobians[:, s, i])
                                  for i in range(6)] for s in range(2)], device=self.args.device, dtype=torch.float32)
        motor = (motor + gravity - force @ jacobian).clamp(-40., 40.)
        motor[:, [2, 5]] = (-.2 * dq[:, [self.active_ids[2], self.active_ids[5]]]).clamp(-3.84, 3.84)
        self.efforts.zero_()
        self.efforts[:, self.active_ids] = motor
        self.efforts[:, self.spring_ids] = force
        self.robot.set_joint_effort_target_index(target=self.efforts)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(self.dt)
        self.time += self.dt

    def sample(self):
        pose = self.robot.data.body_link_pose_w.torch[0].cpu().numpy()
        q = self.robot.data.joint_pos.torch[0].cpu().numpy()
        dq = self.robot.data.joint_vel.torch[0].cpu().numpy()
        if not np.isfinite(np.r_[pose.ravel(), q, dq]).all():
            raise RuntimeError("Nonfinite PhysX state")
        self.q_min = q.copy() if self.q_min is None else np.minimum(self.q_min, q)
        self.q_max = q.copy() if self.q_max is None else np.maximum(self.q_max, q)

        def world(body, local):
            p = pose[self.robot.body_names.index(body)]
            v = np.asarray(local)
            uv = np.cross(p[3:6], v)
            return p[:3] + v + 2 * (p[6] * uv + np.cross(p[3:6], uv))

        gaps = [float(np.linalg.norm(world(c["body0"], c["local_pos0_m"]) - world(c["body1"], c["local_pos1_m"])))
                for c in self.spec["constraints"]]
        self.max_gap = max(self.max_gap, max(gaps))
        if max(gaps) > .003:
            raise RuntimeError(f"Loop gap exceeded 3 mm: {gaps}")
        springs = []
        for side, name, index in zip(("L", "R"), self.manifest["spring_joint_names"], self.spring_ids):
            upper = world(side * 3 + "_link1", [0, 0, 0])
            lower = world(side * 3 + "_link2", [0, 0, 0])
            pin_distance = float(np.linalg.norm(upper - lower))
            binding = self.spec["spring_binding"][name]
            s = binding["compression_at_q_zero_m"] - float(q[index])
            length_error = float(abs(pin_distance - (binding["source_mount_distance_m"] + float(q[index]))))
            self.max_pin_length_error = max(self.max_pin_length_error, length_error)
            u = np.clip(s / self.fit["stroke_m"], 0., .9)
            f = float(np.polynomial.polynomial.polyval(u, self.fit["monomial_coefficients_n"])) if self.gas else 0.
            normal_point = world(side + "_link1", [0, 0, 1]) - world(side + "_link1", [0, 0, 0])
            springs.append({"side": side, "upper_pin": upper.tolist(), "lower_pin": lower.tolist(),
                "pin_axis": normal_point.tolist(), "pin_distance_m": pin_distance,
                "compression_m": s, "force_n": f, "slider_q_m": float(q[index]), "pin_length_error_m": length_error})
        return {"time_s": self.time, "target_knee_deg": self.target_angle,
            "actual_knee_deg": [44.93665895381104 + math.degrees(float(q[self.robot.joint_names.index("L_joint2")])),
                                44.93665895381104 - math.degrees(float(q[self.robot.joint_names.index("R_jonit2")]))],
            "loop_gaps_m": gaps, "springs": springs,
            "motor_effort_nm": self.efforts[0, self.active_ids].cpu().tolist(),
            "joint_positions": q.tolist(), "joint_velocities": dq.tolist()}


class BenchView:
    COLORS = {"base": (.42, .47, .52), "thigh": (.34, .64, .39), "shank": (.47, .53, .61),
              "crank": (.60, .28, .72), "coupler": (.67, .64, .55), "cylinder": (.04, .55, .92),
              "rod": (1., .55, .07), "wheel": (.08, .10, .12)}

    def __init__(self, bench, output):
        import carb.windowing
        import omni.appwindow
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
        self.bench, self.output, self.captures = bench, output, []
        stage = bench.stage
        native = omni.appwindow.get_default_app_window()
        carb.windowing.acquire_windowing_interface().set_window_title(native.get_window(), TITLE)
        self.visibility = {}
        for name in bench.robot.body_names:
            kind = ("base" if name == "base_link" else "rod" if name.startswith(("LLL", "RRR")) and name.endswith("1")
                    else "cylinder" if name.startswith(("LLL", "RRR")) else "crank" if name in ("LL_link1", "RR_link1")
                    else "thigh" if name in ("L_link1", "R_link1") else "shank" if name in ("L_link2", "R_link2")
                    else "wheel" if name in ("L_link3", "R_link3") else "coupler")
            visual = stage.GetPrimAtPath("/World/Robot/" + name + "/Visual")
            self.visibility[name] = UsdGeom.Imageable(visual)
            material = UsdShade.Material.Define(stage, "/World/Materials/" + kind)
            shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.COLORS[kind]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.45)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(visual).Bind(material)
        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(600.)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(1500.)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-35., -25., 30.))
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateClippingRangeAttr(Gf.Vec2f(.003, 50.))
        camera.CreateFocalLengthAttr(45.)
        self.camera_op = camera.AddTransformOp()
        self.viewport = get_active_viewport()
        self.viewport.camera_path = str(camera.GetPath())
        self.viewport.resolution = (1600, 1000)
        self.lines, self.pin_ops = {}, {}
        for side in ("L", "R"):
            line = UsdGeom.BasisCurves.Define(stage, "/World/Annotations/" + side + "_spring_force")
            line.CreateTypeAttr("linear")
            line.CreateWidthsAttr([.0018])
            line.SetWidthsInterpolation("constant")
            line.CreateDisplayColorAttr([Gf.Vec3f(1., .8, .15)])
            self.lines[side] = line
            for end, color in (("upper", self.COLORS["rod"]), ("lower", self.COLORS["cylinder"])):
                sphere = UsdGeom.Sphere.Define(stage, f"/World/Annotations/{side}_{end}_pin")
                sphere.CreateRadiusAttr(.006)
                sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                self.pin_ops[(side, end)] = sphere.AddTranslateOp()
        self.panel = ui.Window("V5 spring installation / real PhysX", width=440, height=620)
        with self.panel.frame:
            with ui.VStack(spacing=7):
                ui.Label("FIXED-BASE PHYSX BENCH / NO POLICY", style={"color": 0xFF80DFFF})
                ui.Label("19 bodies | 16 hinges + 2 sliders | 6 loop closures")
                ui.Label("BLUE cylinder: hinged to the SHANK\nORANGE rod: closes back to the THIGH", word_wrap=True)
                ui.Label("Yellow arrows: equal/opposite extension forces", word_wrap=True)
                ui.Label("Motor target: knee inner angle [55, 78] deg")
                self.slider = ui.FloatSlider(min=55., max=78.)
                self.slider.model.set_value(bench.target_angle)
                with ui.HStack(height=24):
                    self.auto = ui.CheckBox(width=24)
                    self.auto.model.set_value(True)
                    ui.Label("Slow motor-driven cycle (12 simulation seconds)")
                with ui.HStack(height=24):
                    self.gas = ui.CheckBox(width=24)
                    self.gas.model.set_value(True)
                    ui.Label("Apply the 10 MPa nonlinear gas-spring force")
                with ui.HStack(height=24):
                    self.pause = ui.CheckBox(width=24)
                    ui.Label("Pause simulation")
                with ui.HStack(height=24):
                    self.base = ui.CheckBox(width=24)
                    self.base.model.set_value(False)
                    ui.Label("Show base mesh")
                    self.right = ui.CheckBox(width=24)
                    self.right.model.set_value(False)
                    ui.Label("Show right leg")
                with ui.HStack(height=28):
                    ui.Button("Whole mechanism", clicked_fn=lambda: self.camera(False))
                    ui.Button("Left spring closeup", clicked_fn=lambda: self.camera(True))
                with ui.HStack(height=28):
                    ui.Button("Reset nominal", clicked_fn=lambda: setattr(bench, "reset_requested", True))
                    ui.Button("Capture", clicked_fn=self.capture)
                self.status = ui.Label("Starting...", word_wrap=True, height=150)
                ui.Label("Motor PD 60/2 + offline gravity/spring feedforward.\n40 Nm motor / 1000 N slider limits; no passive pose writes.", word_wrap=True)
                ui.Label("Research inertia / CAD-derived mounting reference.\nSelf-collision disabled; base attached to world for inspection.", word_wrap=True)
        self.camera(bench.args.view == "left")

    def camera(self, closeup):
        from pxr import Gf
        eye = Gf.Vec3d(.03, 1.05, .43) if closeup else Gf.Vec3d(.9, 1.7, 1.02)
        target = Gf.Vec3d(.015, .185, .35) if closeup else Gf.Vec3d(0., 0., .32)
        self.camera_op.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0., 0., 1.)).GetInverse())
        self.base.model.set_value(not closeup)
        self.right.model.set_value(not closeup)

    def capture(self, name=None):
        from omni.kit.viewport.utility import capture_viewport_to_file
        file = self.output / (name or ("viewport_" + time.strftime("%H%M%S") + ".png"))
        self.captures.append(capture_viewport_to_file(self.viewport, str(file), is_hdr=False))

    def update(self, sample):
        from pxr import Gf, UsdGeom, Vt
        for name, visual in self.visibility.items():
            visible = (name != "base_link" or self.base.model.as_bool) and (not name.startswith("R") or self.right.model.as_bool)
            visual.MakeVisible() if visible else visual.MakeInvisible()
        for spring in sample["springs"]:
            side = spring["side"]
            upper, lower = np.array(spring["upper_pin"]), np.array(spring["lower_pin"])
            normal = np.array(spring["pin_axis"])
            direction = (upper - lower) / np.linalg.norm(upper - lower)
            offset = normal * .012
            points = [lower + offset, upper + offset]
            for pin, sign in ((upper, 1), (lower, -1)):
                tail = pin + offset
                tip = tail + sign * direction * .035
                points.extend([tail, tip, tip, tip - sign * direction * .008 + normal * .005,
                               tip, tip - sign * direction * .008 - normal * .005,
                               pin - normal * .02, pin + normal * .02])
            line = self.lines[side]
            line.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
            line.CreateCurveVertexCountsAttr([2] * (len(points) // 2))
            visible = side == "L" or self.right.model.as_bool
            for end, pos in (("upper", upper), ("lower", lower)):
                self.pin_ops[(side, end)].Set(Gf.Vec3d(*(pos + offset)))
                imageable = UsdGeom.Imageable(self.pin_ops[(side, end)].GetAttr().GetPrim())
                imageable.MakeVisible() if visible else imageable.MakeInvisible()
            line.MakeVisible() if visible else line.MakeInvisible()
        left, right = sample["springs"]
        self.status.text = (f"Sim {sample['time_s']:.2f} s | target {sample['target_knee_deg']:.1f} deg\n"
            f"Actual knee L/R: {sample['actual_knee_deg'][0]:.1f} / {sample['actual_knee_deg'][1]:.1f} deg\n"
            f"Compression L/R: {left['compression_m']*1000:.2f} / {right['compression_m']*1000:.2f} mm\n"
            f"Gas force L/R: {left['force_n']:.1f} / {right['force_n']:.1f} N\n"
            f"Pin distance L/R: {left['pin_distance_m']*1000:.2f} / {right['pin_distance_m']*1000:.2f} mm\n"
            f"Maximum loop gap: {max(sample['loop_gaps_m'])*1000:.4f} mm\n"
            "A (orange): thigh mount; B (blue): shank mount")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model/纯底盘_v5/urdf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--seconds", type=float, default=12., help="Bounded simulated duration for headless checks")
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--view", choices=("whole", "left"), default="whole")
    parser.add_argument("--swing-deg", type=float, default=8., help="Motor-driven automatic knee-angle amplitude")
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    if not 0 < args.dt <= .005 or not 0 < args.seconds <= 120 or not 0 < args.swing_deg <= 10:
        parser.error("Use bounded positive timestep/duration")
    manifest, spec, fit, angles, targets, gravity_rows, spring_jacobians = read_bundle(args.bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "title": TITLE,
        "asset_manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "fixed_base_fixture": True,
        "passive_pose_writes_after_reset": 0, "kinematic_solver_calls_during_physics": 0,
        "headless": args.headless, "keep_open_until_window_closed": not args.headless,
        "samples": [], "status": "starting"}
    launcher, bench = None, None
    try:
        from isaaclab.app import AppLauncher
        config = {"headless": args.headless, "device": args.device, "enable_cameras": False,
                  "kit_args": "--/app/window/width=1600 --/app/window/height=1000"}
        if not args.headless:
            config["visualizer"] = ["kit"]
        launcher = AppLauncher(config)
        import carb.settings
        import torch
        torch.set_num_threads(4)
        carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
        bench = SpringBench(args, manifest, spec, fit, angles, targets, gravity_rows, spring_jacobians)
        # Set after SimulationContext construction as well: only the renderer is synchronized here.
        carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
        report["solver"] = bench.solver_report
        view = None if args.headless else BenchView(bench, args.output)
        report["status"] = "running"
        print("V5_PREVIEW_READY", json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
        frames, last_save = 0, -10.
        while launcher.app.is_running() and (not args.headless or bench.time < args.seconds):
            wall = time.monotonic()
            if view:
                bench.auto, bench.gas, bench.paused = view.auto.model.as_bool, view.gas.model.as_bool, view.pause.model.as_bool
                bench.target_angle = view.slider.model.as_float
            if bench.auto:
                bench.target_angle = 67.85497 + args.swing_deg * math.sin(bench.time * math.tau / 12.)
                if view:
                    view.slider.model.set_value(bench.target_angle)
            if not bench.paused:
                for _ in range(20):
                    bench.step()
            sample = bench.sample()
            if frames % 10 == 0:
                report["samples"].append(sample)
                if len(report["samples"]) > 3000:
                    report["samples"] = report["samples"][-3000:]
            if view:
                view.update(sample)
                bench.sim.render()
                if frames == 60:
                    view.capture("viewport_initial.png")
            if time.monotonic() - last_save > 2:
                runtime = {"status": "running", "pid": os.getpid(), "title": TITLE,
                           "max_loop_gap_m": bench.max_gap, "current": sample,
                           "paused": bench.paused, "automatic_cycle": bench.auto, "gas_enabled": bench.gas,
                           "fabric_update_enabled": carb.settings.get_settings().get("/physics/fabricUpdateTransformations")}
                (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
                last_save = time.monotonic()
            frames += 1
            if view:
                time.sleep(max(0., 1 / 30 - (time.monotonic() - wall)))
        report["status"] = "completed" if args.headless else "window_closed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if bench is not None:
            report.update(max_loop_gap_m=bench.max_gap, max_pin_length_error_m=bench.max_pin_length_error,
                          simulated_seconds=bench.time,
                          joint_range=(bench.q_max - bench.q_min).tolist() if bench.q_min is not None else None)
            bench.sim.stop()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("V5_PREVIEW_DONE", report["status"], flush=True)
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
