#!/usr/bin/env python3
"""PhysX matched-pose load test: wheels bear weight; a vertical guide fixes attitude.

The no-spring case removes the four spring/adapter bodies and their joints.
Feedforward is computed offline; measured results use actual actuator efforts
and terrain contact forces. No state is overwritten between trial resets.
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

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_v5_closedchain import POSE_COORDINATES, closure_error, solve_pose
from v5_mechanism import fk, point

SPRING_BODIES = {"LLL_link1", "LLL_link2", "RRR_link1", "RRR_link2"}


def prepare_trial(spec, manifest, fit, angle, root_roll=None):
    knee = math.radians(angle) - (math.pi - 2.3573)
    bodies = spec["bodies"]
    mass = sum(b["mass"] for b in bodies)
    root = np.eye(4)

    def pose(hip):
        return solve_pose(spec, dict(zip(POSE_COORDINATES, [hip, knee, 0., -hip, -knee, 0.])),
                          spec["nominal_joint_pos"])

    def centers(q, selected=bodies):
        frames = fk(spec, q, root)
        com = sum(b["mass"] * point(frames[b["name"]], b["com"]) for b in selected) / sum(b["mass"] for b in selected)
        wheels = np.array([point(frames[name], np.array(next(b for b in bodies if b["name"] == name)["collisions"][0]["origin"])[:3, 3])
                           for name in ("L_link3", "R_link3")])
        return com, wheels

    hip = brentq(lambda h: centers(pose(h))[0][0] - centers(pose(h))[1][:, 0].mean(), -.4, 1.4)
    q = pose(hip)
    com, wheels = centers(q)
    if root_roll is None:
        delta = wheels[0] - wheels[1]
        root_roll = math.atan2(-delta[2], delta[1])
    cosine, sine = math.cos(root_roll), math.sin(root_roll)
    root[:3, :3] = [[1., 0., 0.], [0., cosine, -sine], [0., sine, cosine]]
    com, wheels = centers(q)
    names = [j["name"] for j in spec["joints"]]
    active = [names.index(n) for n in manifest["control_joint_names"]]
    passive = [i for i in range(len(names)) if i not in active]
    jacobian, gradients, wheel_jacobian = [], [], []
    selections = [[b for b in bodies if b["name"] not in SPRING_BODIES], bodies]

    def values(q):
        frames = fk(spec, q, root)
        potential = [sum(b["mass"] * 9.81 * point(frames[b["name"]], b["com"])[2] for b in selected)
                     for selected in selections]
        return closure_error(spec, q), np.array(potential), centers(q)[1][:, 2]

    for name in names:
        plus, minus = dict(q), dict(q)
        plus[name] += 1e-6
        minus[name] -= 1e-6
        rp, vp, wp = values(plus)
        rm, vm, wm = values(minus)
        jacobian.append((rp - rm) / 2e-6)
        gradients.append((vp - vm) / 2e-6)
        wheel_jacobian.append((wp - wm) / 2e-6)
    jacobian = np.column_stack(jacobian)
    mapping = np.zeros((len(names), 6))
    mapping[active] = np.eye(6)
    mapping[passive] = np.linalg.lstsq(jacobian[:, passive], -jacobian[:, active], rcond=1e-8)[0]
    if np.abs(jacobian @ mapping).max() > 1e-7:
        raise ValueError("Invalid closed-chain virtual-work mapping")
    gradients = np.array(gradients).T @ mapping
    wheel_jacobian = np.array(wheel_jacobian).T @ mapping
    spring_mapping = mapping[[names.index(n) for n in manifest["spring_joint_names"]]]
    compression = np.array([spec["spring_binding"][n]["compression_at_q_zero_m"] - q[n]
                            for n in manifest["spring_joint_names"]])
    if not np.all((compression >= 0.) & (compression <= .072)):
        raise ValueError("Requested pose is outside the fitted spring working domain")
    forces = np.polynomial.polynomial.polyval(compression / .08, fit["monomial_coefficients_n"])
    feedforwards, loads = [], []
    for index, selected in enumerate(selections):
        weight = sum(b["mass"] for b in selected) * 9.81
        current_com, _ = centers(q, selected)
        left = weight * (current_com[1] - wheels[1, 1]) / (wheels[0, 1] - wheels[1, 1])
        vertical = np.array([left, weight - left])
        if (vertical <= 0).any():
            raise ValueError("COM outside lateral wheel support span")
        feedforwards.append(gradients[index] - vertical @ wheel_jacobian - index * forces @ spring_mapping)
        loads.append(vertical)
    radius = next(b for b in bodies if b["name"] == "L_link3")["collisions"][0]["radius"]
    return {"knee_deg": angle, "hip_raw_rad": hip, "joint_positions": q,
            "base_height_m": float(radius - wheels[:, 2].mean()),
            "compression_m": compression.tolist(), "spring_force_n": forces.tolist(),
            "feedforward_nm": np.array(feedforwards).tolist(), "expected_wheel_loads_n": np.array(loads).tolist(),
            "full_mass_kg": mass, "spring_mapping": spring_mapping.tolist(), "root_roll_rad": root_roll,
            "wheel_height_difference_m": float(wheels[0, 2] - wheels[1, 2])}


class LoadRig:
    def __init__(self, args, sim, manifest, spec, fit, gas):
        import torch
        from pxr import Gf, PhysxSchema, UsdPhysics
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        self.args, self.sim, self.manifest, self.spec, self.fit, self.gas = args, sim, manifest, spec, fit, gas
        self.name = "with_spring" if gas else "without_spring"
        self.path = "/World/" + self.name
        self.y = .65 if gas else -.65
        active = manifest["control_joint_names"]
        self.joint_names = [j["name"] for j in spec["joints"] if gas or j["child"] not in SPRING_BODIES]
        springs = manifest["spring_joint_names"] if gas else []
        passive = [n for n in self.joint_names if n not in active + springs]
        actuators = {
            "active": IdealPDActuatorCfg(joint_names_expr=active, stiffness=0., damping=0.,
                                         effort_limit=40., effort_limit_sim=40., velocity_limit_sim=1000.),
            "passive": IdealPDActuatorCfg(joint_names_expr=passive, stiffness=0., damping=.002,
                                          effort_limit=100., effort_limit_sim=100., velocity_limit_sim=1000.),
        }
        if gas:
            actuators["gas"] = IdealPDActuatorCfg(joint_names_expr=springs, stiffness=0., damping=.002,
                                                effort_limit=1000., effort_limit_sim=1000., velocity_limit_sim=100.)
        cfg = ArticulationCfg(
            prim_path=self.path,
            spawn=sim_utils.UsdFileCfg(usd_path=str(args.bundle / "robot.usda"), activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False,
                    enabled_self_collisions=False, solver_position_iteration_count=args.position_iterations,
                    solver_velocity_iteration_count=args.velocity_iterations)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., self.y, 0.),
                joint_pos={n: manifest["nominal_joint_pos"][n] for n in self.joint_names}, joint_vel={".*": 0.}),
            actuators=actuators, soft_joint_pos_limit_factor=1.)
        self.robot = Articulation(cfg)
        if not gas:
            for prim in list(sim.stage.Traverse()):
                if not str(prim.GetPath()).startswith(self.path + "/"):
                    continue
                if prim.IsA(UsdPhysics.Joint):
                    joint = UsdPhysics.Joint(prim)
                    targets = joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
                    if any(p.name in SPRING_BODIES for p in targets):
                        prim.SetActive(False)
            for name in SPRING_BODIES:
                sim.stage.GetPrimAtPath(self.path + "/" + name).SetActive(False)
        guide = UsdPhysics.PrismaticJoint.Define(sim.stage, self.path + "_vertical_guide")
        guide.CreateBody1Rel().SetTargets([self.path + "/base_link"])
        guide.CreateAxisAttr("Z")
        guide.CreateLocalPos0Attr(Gf.Vec3f(0., self.y, 0.))
        guide.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        guide.CreateLocalRot0Attr(Gf.Quatf(1.))
        # Compensate the small CAD frame roll while keeping the guide axis world-Z.
        guide.CreateLocalRot1Attr(Gf.Quatf(math.cos(args.root_roll / 2),
                                         Gf.Vec3f(-math.sin(args.root_roll / 2), 0., 0.)))
        guide.CreateLowerLimitAttr(-10.)
        guide.CreateUpperLimitAttr(10.)
        guide.CreateExcludeFromArticulationAttr(True)
        for prim in sim.stage.Traverse():
            if str(prim.GetPath()).startswith(self.path + "/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
        self.device = args.device

    def initialize(self):
        import torch
        import warp as wp
        from isaaclab_physx.physics import PhysxManager
        self.ids = [self.robot.joint_names.index(n) for n in self.manifest["control_joint_names"]]
        self.knees = [self.robot.joint_names.index(n) for n in ("L_joint2", "R_jonit2")]
        self.spring_ids = [self.robot.joint_names.index(n) for n in self.manifest["spring_joint_names"]] if self.gas else []
        self.wheel_ids = [self.robot.body_names.index(n) for n in ("L_link3", "R_link3")]
        if set(self.robot.joint_names) != set(self.joint_names):
            raise ValueError("Unexpected rig articulation topology")
        self.masses = wp.to_torch(self.robot.root_view.get_masses())[0].cpu().numpy()
        self.mass = float(self.masses.sum())
        view = PhysxManager.get_physics_sim_view()
        paths = [self.path + "/" + n for n in self.robot.body_names]
        self.contact = view.create_rigid_contact_view(paths,
            filter_patterns=[["/World/Ground/geometry/mesh"] for _ in paths], max_contact_data_count=4096)
        self.effort = torch.zeros(1, len(self.robot.joint_names), device=self.device)
        self.s0 = torch.tensor([self.spec["spring_binding"][n]["compression_at_q_zero_m"]
                                for n in self.manifest["spring_joint_names"]], device=self.device)
        self.constraints = [c for c in self.spec["constraints"] if self.gas or c["body1"] not in SPRING_BODIES]

    def reset(self, trial):
        import torch
        self.trial = trial
        q = torch.tensor([[trial["joint_positions"][n] for n in self.robot.joint_names]], device=self.device)
        roll = trial["root_roll_rad"]
        root = torch.tensor([[0., self.y, trial["base_height_m"] + .0005,
                              math.sin(roll / 2), 0., 0., math.cos(roll / 2)]], device=self.device)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.device))
        self.robot.write_joint_position_to_sim_index(position=q)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        self.robot.reset()
        self.robot.update(self.args.dt)
        self.target = q[:, self.ids]
        self.ff = torch.tensor(trial["feedforward_nm"][int(self.gas)], device=self.device)
        self.mapping = torch.tensor(trial["spring_mapping"], device=self.device)
        self.nominal_force = torch.tensor(trial["spring_force_n"], device=self.device)
        self.integral = torch.zeros_like(self.target)
        self.filtered_velocity = torch.zeros_like(self.target)
        self.samples, self.all_torques = [], []

    def command(self):
        import torch
        q, dq = self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch
        delta = self.target - q[:, self.ids]
        error = torch.atan2(delta.sin(), delta.cos())
        self.integral = (self.integral + error * self.args.dt).clamp(-.05, .05)
        # Suppress constraint-solver velocity jitter in this static load servo.
        alpha = self.args.dt / (.02 + self.args.dt)
        self.filtered_velocity += alpha * (dq[:, self.ids] - self.filtered_velocity)
        motor = self.ff + 120. * error + 80. * self.integral - 4. * self.filtered_velocity
        self.effort.zero_()
        if self.gas:
            compression = self.s0 - q[:, self.spring_ids]
            if not bool(torch.isfinite(compression).all()) or bool(((compression < -.0005) | (compression > .0725)).any()):
                raise RuntimeError("Spring left fitted working domain")
            u = compression.clamp(0., .072) / .08
            a, b, c, d = self.fit["monomial_coefficients_n"]
            force = a + u * (b + u * (c + u * d))
            self.effort[:, self.spring_ids] = force
            motor -= (force - self.nominal_force) @ self.mapping
        # Free-rolling wheels isolate vertical support loads from a wheel-speed
        # servo's braking torque. This bench is not a wheel-motor sizing test.
        motor[:, [2, 5]] = 0.
        self.effort[:, self.ids] = motor.clamp(-40., 40.)
        self.robot.set_joint_effort_target_index(target=self.effort)
        self.robot.write_data_to_sim()

    def sample(self, seconds, *, record=True):
        import warp as wp
        pose = self.robot.data.body_link_pose_w.torch[0].cpu().numpy()
        q = self.robot.data.joint_pos.torch[0].cpu().numpy()
        dq = self.robot.data.joint_vel.torch[0].cpu().numpy()
        forces = wp.to_torch(self.contact.get_contact_force_matrix(dt=self.args.dt)).cpu().numpy().sum(axis=1)
        if not np.isfinite(np.r_[pose.ravel(), q, dq, forces.ravel()]).all():
            raise RuntimeError("Nonfinite rig state")

        def world(body, local):
            p = pose[self.robot.body_names.index(body)]
            v = np.asarray(local)
            uv = np.cross(p[3:6], v)
            return p[:3] + v + 2 * (p[6] * uv + np.cross(p[3:6], uv))

        gap = max(np.linalg.norm(world(c["body0"], c["local_pos0_m"]) - world(c["body1"], c["local_pos1_m"])) for c in self.constraints)
        nonwheel = [i for i in range(len(forces)) if i not in self.wheel_ids]
        result = {"time_s": seconds, "knee_deg": [44.93665895381104 + math.degrees(float(q[self.knees[0]])),
                                                  44.93665895381104 - math.degrees(float(q[self.knees[1]]))],
                  "base_height_m": float(pose[self.robot.body_names.index("base_link"), 2]),
                  "wheel_vertical_forces_n": forces[self.wheel_ids, 2].tolist(),
                  "nonwheel_contact_force_n": float(np.linalg.norm(forces[nonwheel], axis=1).sum()),
                  "max_loop_gap_m": float(gap), "max_leg_speed_rad_s": float(np.abs(dq[self.ids][[0, 1, 3, 4]]).max()),
                  "active_joint_positions_rad": q[self.ids].tolist(),
                  "filtered_leg_speed_rad_s": float(self.filtered_velocity[0, [0, 1, 3, 4]].abs().max()),
                  "motor_effort_nm": self.robot.data.applied_torque.torch[0, self.ids].cpu().tolist()}
        if self.gas:
            result["compression_m"] = (self.s0.cpu().numpy() - q[self.spring_ids]).tolist()
            result["spring_applied_force_n"] = self.robot.data.applied_torque.torch[0, self.spring_ids].cpu().tolist()
        if record:
            self.samples.append(result)
        return result

    def summarize(self):
        samples = [s for s in self.samples if s["time_s"] >= self.args.settle]
        torque = np.array(self.all_torques, dtype=np.float64)
        steady = torque[round(self.args.settle / self.args.dt):]
        knees = np.array([s["knee_deg"] for s in samples])
        contact = np.array([s["wheel_vertical_forces_n"] for s in samples])
        ratio = float(contact.sum(axis=1).mean() / (self.mass * 9.81))
        positions = np.array([s["active_joint_positions_rad"] for s in samples])[:, [0, 1, 3, 4]]
        position_speeds = np.diff(positions, axis=0) / np.diff([s["time_s"] for s in samples])[:, None]
        strict_static_checks = {
            "settled_filtered_leg_speed": max(s["filtered_leg_speed_rad_s"] for s in samples) < .02,
            "settled_position_derived_leg_speed": bool(np.abs(position_speeds).max() < .02),
        }
        # A quasi-static load estimate requires a narrow position envelope and
        # repeatable torque, not mathematically zero solver velocities. Preserve
        # the stricter speed failures separately rather than concealing them.
        leg_steady = steady[:, [0, 1, 3, 4]]
        checks = {
            "wheels_bear_full_weight": abs(ratio - 1.) < .01,
            "both_wheels_supported": bool((contact.min(axis=0) > 10.).all()),
            "no_nonwheel_support": max(s["nonwheel_contact_force_n"] for s in samples) < .1,
            "posture_error_below_half_degree": bool(np.abs(knees - self.trial["knee_deg"]).max() < .5),
            "active_joint_envelope_below_quarter_degree": bool(np.rad2deg(np.ptp(positions, axis=0)).max() < .25),
            "active_joint_target_error_below_half_degree": bool(np.rad2deg(np.abs(positions - self.target.cpu().numpy()[:, [0, 1, 3, 4]])).max() < .5),
            "leg_torque_std_below_point_two_nm": bool(leg_steady.std(axis=0).max() < .2),
            "knee_peak_to_peak_below_point_one_degree": bool(np.ptp(knees, axis=0).max() < .1),
            "loop_gap_below_point_one_mm": max(s["max_loop_gap_m"] for s in samples) < .0001,
            "no_motor_saturation": bool(np.abs(steady[:, [0, 1, 3, 4]]).max() < 39.9),
        }
        result = {"case": self.name, "mass_kg": self.mass, "body_count": len(self.robot.body_names),
                  "tree_dofs": len(self.robot.joint_names), "spherical_loops": len(self.constraints),
                  "knee_target_deg": self.trial["knee_deg"], "knee_mean_deg": knees.mean(axis=0).tolist(),
                  "base_height_mean_m": float(np.mean([s["base_height_m"] for s in samples])),
                  "wheel_force_mean_n": contact.mean(axis=0).tolist(), "wheel_weight_ratio": ratio,
                  "steady_window_s": [self.args.settle, self.args.seconds],
                  "checks": checks, "passed": all(checks.values()), "motors": {},
                  "strict_static_checks": strict_static_checks,
                  "strict_static_passed": all(strict_static_checks.values()),
                  "max_steady_loop_gap_m": max(s["max_loop_gap_m"] for s in samples)}
        result["max_active_joint_envelope_deg"] = float(np.rad2deg(np.ptp(positions, axis=0)).max())
        result["max_steady_leg_speed_rad_s"] = max(s["max_leg_speed_rad_s"] for s in samples)
        result["max_position_derived_leg_speed_rad_s"] = float(np.abs(position_speeds).max())
        result["max_filtered_leg_speed_rad_s"] = max(s["filtered_leg_speed_rad_s"] for s in samples)
        result["max_steady_knee_error_deg"] = float(np.abs(knees - self.trial["knee_deg"]).max())
        for i, name in enumerate(self.manifest["control_joint_names"]):
            result["motors"][name] = {"mean_signed_nm": float(steady[:, i].mean()),
                "mean_abs_nm": float(np.abs(steady[:, i]).mean()), "rms_nm": float(np.sqrt((steady[:, i] ** 2).mean())),
                "std_nm": float(steady[:, i].std()),
                "steady_peak_abs_nm": float(np.abs(steady[:, i]).max()),
                "whole_trial_peak_abs_nm": float(np.abs(torque[:, i]).max())}
        if self.gas:
            result["spring_force_mean_n"] = np.mean([s["spring_applied_force_n"] for s in samples], axis=0).tolist()
            result["compression_mean_m"] = np.mean([s["compression_m"] for s in samples], axis=0).tolist()
        return result


def live_text(rigs, samples, title):
    return title + "\nFree-Z guide; wheel contacts support weight. Output-shaft Nm.\n\n" + "\n\n".join(
        rig.name + f" ({rig.mass:.3f} kg), knee " + "/".join(f"{v:.2f}" for v in sample["knee_deg"]) + " deg\n"
        + "\n".join(f"{n}: {v:+.3f} Nm" for n, v in zip(rig.manifest["control_joint_names"], sample["motor_effort_nm"]))
        + ("\nSpring L/R: " + "/".join(f"{v:.1f}" for v in sample["spring_applied_force_n"]) + " N" if rig.gas else "")
        for rig, sample in zip(rigs, samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model/纯底盘_v5/urdf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", nargs="+", type=float, default=[55., 67.85497, 78.])
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--settle", type=float, default=7.)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--solver", default="tgs", choices=("tgs", "pgs"))
    parser.add_argument("--position-iterations", type=int, default=32)
    parser.add_argument("--velocity-iterations", type=int, default=8)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not (0 < args.dt <= .002 and 0 < args.settle <= args.seconds - .5 and args.seconds <= 60 and all(50 <= a <= 80 for a in args.angles)
            and 1 <= args.position_iterations <= 255 and 1 <= args.velocity_iterations <= 255):
        parser.error("Invalid bounded trial settings")
    args.bundle = args.bundle.resolve()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((args.bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    spec = json.loads((args.bundle / "model_spec.json").read_text())
    fit = json.loads((args.bundle / "fit_10mpa.json").read_text())
    args.root_roll = prepare_trial(spec, manifest, fit, 67.85497)["root_roll_rad"]
    trials = [prepare_trial(spec, manifest, fit, angle, args.root_roll) for angle in args.angles]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "benchmark_source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "prepared_trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(trials, indent=2))
        return 0
    report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fixture": "unactuated world-Z prismatic guide: free vertical translation; fixed attitude and XY",
              "measurement": "PhysX applied actuator effort, every physics step; float64 offline accumulation",
              "pressure_mpa": 10., "dt": args.dt, "results": [], "state_writes_between_resets": 0,
              "root_roll_rad": args.root_roll, "controller": {"kp": 120., "kd": 4., "ki": 80.,
                    "integral_effort_limit_nm": 4., "velocity_filter_time_constant_s": .02},
              "external_forces_every_iteration": True,
              "solver": args.solver, "position_iterations": args.position_iterations, "velocity_iterations": args.velocity_iterations,
              "wheel_control": "zero motor effort, free rolling; no wheel speed servo",
              "scope": "matched-pose static wheel-supported load test, not free-balancing policy or dynamic motor rating"}
    launcher = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": args.headless, "device": args.device,
                                **({"visualizer": ["kit"]} if not args.headless else {})})
        import carb.settings
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg
        torch.set_num_threads(4)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=args.dt, render_interval=20,
            physics=PhysxCfg(solver_type=int(args.solver == "tgs"), enable_external_forces_every_iteration=True)))
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
        floor = sim_utils.CuboidCfg(size=(6., 6., .1), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8, restitution=0.))
        floor.func("/World/Ground", floor, translation=(0., 0., -.05))
        rigs = [LoadRig(args, sim, manifest, spec, fit, gas) for gas in (False, True)]
        sim.reset()
        for rig in rigs:
            rig.initialize()
        report["status"] = "running"
        label = None
        if not args.headless:
            import omni.ui as ui
            light = sim_utils.DomeLightCfg(intensity=1200.)
            light.func("/World/Light", light)
            sim.set_camera_view((1.5, -2., 1.1), (0., 0., .2))
            import carb.windowing
            import omni.appwindow
            native = omni.appwindow.get_default_app_window()
            carb.windowing.acquire_windowing_interface().set_window_title(native.get_window(), "V5 SPRING LOAD A/B | ISAAC PHYSX")
            window = ui.Window("V5 LOAD A/B | WHEELS BEAR WEIGHT", width=590, height=460)
            with window.frame:
                label = ui.Label("Starting matched-pose load comparison", word_wrap=True)
        for trial_index, trial in enumerate(trials):
            for rig in rigs:
                rig.reset(trial)
            for step in range(round(args.seconds / args.dt)):
                for rig in rigs:
                    rig.command()
                sim.step(render=False)
                for rig in rigs:
                    rig.robot.update(args.dt)
                    rig.all_torques.append(rig.robot.data.applied_torque.torch[0, rig.ids].cpu().tolist())
                    if step % 10 == 0:
                        rig.sample((step + 1) * args.dt)
                if label is not None and step % 20 == 0:
                    label.text = live_text(rigs, [r.samples[-1] for r in rigs],
                        f"Knee target: {trial['knee_deg']:.2f} deg | trial {trial_index + 1}/{len(trials)}")
                    sim.render()
            results = [rig.summarize() for rig in rigs]
            report["results"].extend(results)
            for rig in rigs:
                stem = f"{trial_index:02d}_{rig.name}"
                (args.output / f"{stem}_samples.json").write_text(json.dumps(rig.samples) + "\n")
                np.save(args.output / f"{stem}_torques.npy", np.asarray(rig.all_torques, dtype=np.float64))
            print("V5_LOAD_TRIAL", json.dumps(results), flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "quasistatic_load_passed" if all(r["passed"] for r in report["results"]) else "validation_failed"
        report["strict_static_passed"] = all(r["strict_static_passed"] for r in report["results"])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if not args.headless:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            capture_viewport_to_file(get_active_viewport(), str(args.output / "viewport.png"))
            last_save = 0.
            hold_time = args.seconds
            while launcher.app.is_running():
                wall = time.monotonic()
                for _ in range(20):
                    for rig in rigs:
                        rig.command()
                    sim.step(render=False)
                    for rig in rigs:
                        rig.robot.update(args.dt)
                hold_time += 20 * args.dt
                samples = [r.sample(hold_time, record=False) for r in rigs]
                label.text = live_text(rigs, samples, "Comparison saved. Live final-pose hold; close window to stop.")
                if time.monotonic() - last_save > 2:
                    (args.output / "runtime.json").write_text(json.dumps({"pid": os.getpid(),
                        "updated_at": datetime.now(timezone.utc).isoformat(), "samples": samples}, indent=2) + "\n")
                    last_save = time.monotonic()
                sim.render()
                time.sleep(max(0., .02 - (time.monotonic() - wall)))
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] == "quasistatic_load_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
