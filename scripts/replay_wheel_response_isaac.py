#!/usr/bin/env python3
"""Replay airborne wheel bags on the V6 USD with fixed wheel carriers.

The nonwheel coordinates are locked at a closed-chain nominal pose, matching the
quiet-carrier approximation used by the wheel fitter. This checks native wheel
inertia, armature and effort-based friction, not full installed-leg dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from replay_identification_isaac import BUNDLE, DT, point_world


def project_native_inertia(tensor, com_pose_xyzw, mass, axis):
    """Project the engine mass-frame tensor onto a link-frame revolute axis."""
    com_pose = np.asarray(com_pose_xyzw)
    rotation_pose = np.r_[np.zeros(3), com_pose[3:]]
    rotation = np.column_stack([point_world(rotation_pose, basis) for basis in np.eye(3)])
    inertia = rotation @ np.asarray(tensor).reshape(3, 3) @ rotation.T
    axis = np.asarray(axis) / np.linalg.norm(axis)
    com = com_pose[:3]
    return float(axis @ inertia @ axis + mass * (com @ com - (axis @ com) ** 2))


def load_inputs(run, segments, kp):
    with np.load(run / "wheel_1khz.npz") as source:
        running = source["phase"] == 2
        data = {key: source[key][running] for key in ("segment_id", "control_steady_ns",
            "q_api", "dq_api", "tau_frame_api", "wheel_velocity_target_api")}
    plan = {row["id"]: row for row in json.loads((run / "trajectory-plan.json").read_text())["segments"]}
    arrays = {}
    for segment in segments:
        ix = np.flatnonzero(data["segment_id"] == segment)
        if len(ix) < 2 or np.any(np.diff(ix) != 1) or segment not in plan:
            raise ValueError(f"Missing, split or singleton wheel segment {segment}")
        times = (data["control_steady_ns"][ix].astype(np.int64)
                 - int(data["control_steady_ns"][ix[0]])) * 1e-9
        delta = np.diff(times)
        if delta.min() <= 0 or delta.max() > .002 or np.max(abs(times - np.arange(len(ix)) * DT)) > DT:
            raise ValueError(f"Wheel segment {segment} is not a continuous nominal 1kHz grid")
        trace = {key: data[key][ix] for key in ("q_api", "dq_api", "tau_frame_api", "wheel_velocity_target_api")}
        if not all(np.isfinite(value[:, 4:6] if value.shape[1] == 6 else value).all()
                   for value in trace.values()):
            raise ValueError(f"Nonfinite wheel data in segment {segment}")
        # Check the controller from the actual recorded frames, not a filename.
        target = trace["wheel_velocity_target_api"]
        request = np.zeros_like(target) if plan[segment]["coast"] else np.clip(
            kp * (target - trace["dq_api"][:, 4:6]), -4.5, 4.5)
        # Frame-current quantization and short source/submission alignment noise
        # are reported. A gross mismatch rejects a wrong P/coordinate schema.
        frame_error = abs(trace["tau_frame_api"][:, 4:6] - request)
        if np.quantile(frame_error, .99) > .05:
            raise ValueError(f"Recorded torque does not match wheel P={kp} in segment {segment}")
        arrays[segment] = {**trace, "indices": ix, "time_s": times,
            "description": plan[segment], "max_grid_error_s": float(np.max(abs(times - np.arange(len(ix)) * DT))),
            "frame_pd_error_p99_nm": float(np.quantile(frame_error, .99))}
    return arrays


def wheel_parameters(path):
    if path is None:
        return {"armature": np.zeros(2), "coulomb": np.zeros(2), "viscous": np.zeros(2)}, None
    report = json.loads(path.read_text())
    if report.get("coordinate") != "wheel_output" or not report.get("ratio_15_8_already_applied"):
        raise ValueError("Expected an explicitly wheel-output fit with ratio already applied")
    if "axes" in report:
        rows = {}
        for side, joint in (("left", "L_joint3"), ("right", "R_joint3")):
            axis = report["axes"][joint]
            rows[side] = {"fit": {"effective_inertia_kg_m2": axis["effective_total_inertia_kg_m2"],
                "coulomb_positive_nm": axis["coulomb_positive_nm"],
                "coulomb_negative_nm": axis["coulomb_negative_nm"],
                "viscous_nm_per_rad_s": axis["viscous_nm_per_rad_s"]},
                "cad_axial_inertia_kg_m2": axis["cad_inertia_kg_m2"],
                "candidate_added_armature_kg_m2": axis["additional_armature_kg_m2"]}
    else:
        rows = {row["side"]: row for row in report["wheels"]}
    ordered = [rows[name] for name in ("left", "right")]
    values = {
        "armature": np.array([r["candidate_added_armature_kg_m2"] for r in ordered]),
        "coulomb": np.array([.5 * (r["fit"]["coulomb_positive_nm"] + r["fit"]["coulomb_negative_nm"]) for r in ordered]),
        "viscous": np.array([r["fit"]["viscous_nm_per_rad_s"] for r in ordered]),
    }
    if not all(np.isfinite(v).all() and (v >= 0).all() for v in values.values()):
        raise ValueError("Wheel response parameters must be finite and nonnegative")
    for row in ordered:
        if not np.isclose(row["fit"]["effective_inertia_kg_m2"], row["cad_axial_inertia_kg_m2"]
                          + row["candidate_added_armature_kg_m2"], atol=1e-10, rtol=0):
            raise ValueError("Total wheel inertia must equal existing CAD plus added armature")
    return values, report


class WheelBench:
    def __init__(self, args, manifest, spec, params):
        import torch
        import warp as wp
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.utils.version import get_isaac_sim_version
        from pxr import UsdGeom, UsdPhysics

        if get_isaac_sim_version().major < 5:
            raise RuntimeError("Wheel effort-based friction requires Isaac Sim >=5")
        self.args, self.params, self.torch = args, params, torch
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            device=args.device, dt=DT, render_interval=20, gravity=(0., 0., -9.81)))
        self.count = getattr(args, "instances", 1)
        if self.count > 1:
            for env in range(self.count):
                UsdGeom.Xform.Define(self.sim.stage, f"/World/envs/env_{env}").AddTranslateOp().Set(
                    (2. * (env % 4), 2. * (env // 4), 0.))
        wheels = ["L_joint3", "R_joint3"]
        others = [n for n in manifest["tree_joint_names"] if n not in wheels]
        actuators = {"carriers": IdealPDActuatorCfg(joint_names_expr=others,
            stiffness=0., damping=0., armature=0., friction=0., dynamic_friction=0., viscous_friction=0.,
            effort_limit=1000., effort_limit_sim=1000., velocity_limit_sim=1e9)}
        for axis, name in enumerate(wheels):
            actuators[f"wheel_{axis}"] = IdealPDActuatorCfg(joint_names_expr=[name],
                stiffness=0., damping=0., armature=float(params["armature"][axis]),
                friction=float(params["coulomb"][axis]), dynamic_friction=float(params["coulomb"][axis]),
                viscous_friction=float(params["viscous"][axis]), effort_limit=4.5, effort_limit_sim=4.5,
                velocity_limit_sim=1000.)
        cfg = ArticulationCfg(prim_path="/World/envs/env_.*/Robot" if self.count > 1 else "/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(args.bundle / manifest["usd"]),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=8)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.),
                joint_pos=manifest["nominal_joint_pos"], joint_vel={".*": 0.}),
            actuators=actuators, soft_joint_pos_limit_factor=1.)
        self.robot = Articulation(cfg)
        # Continuous USD joints must have finite limits BEFORE PhysX creates
        # the articulation. Runtime setDofLimits cannot enable absent limits.
        # Keep the original degrees of freedom and closed-chain topology, with
        # narrow finite ranges serving as the declared quiet-carrier fixture.
        carrier_bounds = {}
        for prim in self.sim.stage.Traverse():
            name = prim.GetName()
            if name not in others:
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint, scale, epsilon = UsdPhysics.RevoluteJoint(prim), 180 / np.pi, 1e-6
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                joint, scale, epsilon = UsdPhysics.PrismaticJoint(prim), 1., 1e-8
            else:
                continue
            nominal = manifest["nominal_joint_pos"][name]
            carrier_bounds[name] = [nominal - epsilon, nominal + epsilon]
            joint.CreateLowerLimitAttr().Set(float((nominal - epsilon) * scale))
            joint.CreateUpperLimitAttr().Set(float((nominal + epsilon) * scale))
        if set(carrier_bounds) != set(others):
            raise RuntimeError("Could not bind every carrier joint to a finite USD limit before initialization")
        self.sim.reset()
        if not self.robot.is_fixed_base or set(self.robot.joint_names) != set(manifest["tree_joint_names"]):
            raise RuntimeError("Wheel fixture topology does not match the fixed sealed asset")
        self.order = ([int(path.split("/env_")[1].split("/")[0]) for path in self.robot.root_view.prim_paths]
                      if self.count > 1 else [0])
        if sorted(self.order) != list(range(self.count)):
            raise RuntimeError("Ambiguous wheel instance ordering")
        self.ids = [self.robot.joint_names.index(name) for name in wheels]
        self.carrier_ids = [self.robot.joint_names.index(name) for name in others]
        self.nominal = torch.tensor([[manifest["nominal_joint_pos"][n] for n in self.robot.joint_names]],
                                    device=args.device, dtype=torch.float32).repeat(self.count, 1)
        self.efforts = torch.zeros_like(self.nominal)
        lock = np.asarray([carrier_bounds[name] for name in others])
        armature = wp.to_torch(self.robot.root_view.get_dof_armatures()).cpu().numpy()[0, self.ids]
        friction = wp.to_torch(self.robot.root_view.get_dof_friction_properties()).cpu().numpy()[0, self.ids]
        all_limits = wp.to_torch(self.robot.root_view.get_dof_limits()).cpu().numpy()[:, self.carrier_ids]
        limits = all_limits[0]
        if not np.allclose(all_limits, lock[None], atol=1e-7, rtol=0.):
            raise RuntimeError("Wheel-carrier lock readback mismatch")
        if not np.allclose(armature, params["armature"], atol=1e-8):
            raise RuntimeError("Native wheel armature readback mismatch")
        expected_friction = np.column_stack([params["coulomb"], params["coulomb"], params["viscous"]])
        if not np.allclose(friction, expected_friction, atol=1e-7):
            raise RuntimeError("Native wheel friction readback mismatch")
        self.readback = {"isaac_sim_version": str(get_isaac_sim_version()), "joint_names": wheels,
            "instances": self.count, "clone_order": self.order, "carrier_limits_all_instances_verified": True,
            "armature_kg_m2": armature.tolist(), "friction_static_dynamic_viscous": friction.tolist(),
            "locked_joint_names": others, "locked_limits": limits.tolist(),
            "body_names": self.robot.body_names,
            "body_masses_kg": wp.to_torch(self.robot.root_view.get_masses()).cpu().numpy()[0].tolist(),
            "body_inertia_tensors": wp.to_torch(self.robot.root_view.get_inertias()).cpu().numpy()[0].tolist(),
            "body_com_poses_xyzw": wp.to_torch(self.robot.root_view.get_coms()).cpu().numpy()[0].tolist()}
        wheel_inertias = []
        for name in wheels:
            joint = next(j for j in spec["joints"] if j["name"] == name)
            body = self.robot.body_names.index(joint["child"])
            inertia = project_native_inertia(self.readback["body_inertia_tensors"][body],
                self.readback["body_com_poses_xyzw"][body], self.readback["body_masses_kg"][body], joint["axis"])
            source = next(b for b in spec["bodies"] if b["name"] == joint["child"])
            expected = project_native_inertia(source["inertia"], [*source["com"], 0., 0., 0., 1.],
                                              source["mass"], joint["axis"])
            if not np.isclose(inertia, expected, atol=1e-8, rtol=1e-4):
                raise RuntimeError(f"Runtime wheel axial inertia differs from sealed CAD: {name}: {inertia} vs {expected}")
            wheel_inertias.append(inertia)
        self.readback["cad_axial_inertia_kg_m2"] = wheel_inertias
        self.readback["total_axial_inertia_with_armature_kg_m2"] = (np.asarray(wheel_inertias) + armature).tolist()

    def configure_batch_response(self, cases):
        """Keep one fixture implementation and vary only native per-env values."""
        import warp as wp
        self.cases = [cases[index] for index in self.order]
        tensor = lambda key: self.torch.as_tensor(np.stack([row["params"][key] for row in self.cases]),
            device=self.args.device, dtype=self.efforts.dtype)
        armature, coulomb, viscous = (tensor(key) for key in ("armature", "coulomb", "viscous"))
        self.robot.write_joint_armature_to_sim_index(armature=armature, joint_ids=self.ids)
        self.robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=coulomb,
            joint_dynamic_friction_coeff=coulomb, joint_viscous_friction_coeff=viscous, joint_ids=self.ids)
        actual_armature = wp.to_torch(self.robot.root_view.get_dof_armatures()).cpu().numpy()[:, self.ids]
        actual_friction = wp.to_torch(self.robot.root_view.get_dof_friction_properties()).cpu().numpy()[:, self.ids]
        expected = np.stack([coulomb.cpu().numpy(), coulomb.cpu().numpy(), viscous.cpu().numpy()], axis=-1)
        if not np.allclose(actual_armature, armature.cpu().numpy(), atol=1e-8) or not np.allclose(actual_friction, expected, atol=1e-7):
            raise RuntimeError("Batched wheel native parameters did not reach the engine")
        self.readback["all_armature_kg_m2"] = actual_armature.tolist()
        self.readback["all_friction_static_dynamic_viscous"] = actual_friction.tolist()
        self.readback["all_total_axial_inertia_with_armature_kg_m2"] = (
            actual_armature + np.asarray(self.readback["cad_axial_inertia_kg_m2"])).tolist()

    def initialize(self, q, dq):
        position, velocity = self.nominal.clone(), self.torch.zeros_like(self.nominal)
        position[:, self.ids] = self.torch.as_tensor(q, device=self.args.device, dtype=position.dtype)
        velocity[:, self.ids] = self.torch.as_tensor(dq, device=self.args.device, dtype=velocity.dtype)
        self.robot.write_joint_position_to_sim_index(position=position)
        self.robot.write_joint_velocity_to_sim_index(velocity=velocity)
        self.robot.reset()
        self.robot.update(DT)

    def step(self, effort):
        self.efforts.zero_()
        self.efforts[:, self.ids] = self.torch.as_tensor(effort, device=self.args.device, dtype=self.efforts.dtype)
        self.robot.set_joint_effort_target_index(target=self.efforts)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(DT)

    def sample(self):
        q = self.robot.data.joint_pos.torch[0].cpu().numpy()
        dq = self.robot.data.joint_vel.torch[0].cpu().numpy()
        drift = float(abs(q[self.carrier_ids] - self.nominal[0, self.carrier_ids].cpu().numpy()).max())
        if not np.isfinite(np.r_[q, dq]).all() or abs(dq).max() > 500:
            raise RuntimeError("Wheel replay diverged")
        return q[self.ids].copy(), dq[self.ids].copy(), drift

    def run_trace(self, target, recorded, initial_q, initial_dq, coast, mode):
        """Keep the 1kHz loop on device; transfer complete traces only once."""
        torch = self.torch
        self.initialize(initial_q, initial_dq)
        reference = torch.as_tensor(target, device=self.args.device, dtype=self.efforts.dtype)
        input_torque = torch.as_tensor(recorded, device=self.args.device, dtype=self.efforts.dtype)
        trace = torch.empty((len(target), 7), device=self.args.device, dtype=self.efforts.dtype)
        zero = torch.zeros(2, device=self.args.device, dtype=self.efforts.dtype)
        carrier_nominal = self.nominal[0, self.carrier_ids]
        started = time.monotonic()
        for step in range(len(target)):
            q, dq = self.robot.data.joint_pos.torch[0], self.robot.data.joint_vel.torch[0]
            position, velocity = q[self.ids], dq[self.ids]
            drift = (q[self.carrier_ids] - carrier_nominal).abs().max()
            torque = (zero if coast else (self.args.kp * (reference[step] - velocity)).clamp(-4.5, 4.5))
            if mode == "torque_input":
                torque = input_torque[step]
            trace[step] = torch.cat([position, velocity, torque, drift[None]])
            self.step(torque)
            if (step + 1) % 2000 == 0:
                if not bool(torch.isfinite(trace[step]).all()) or bool((velocity.abs() > 500).any()):
                    raise RuntimeError(f"Wheel replay diverged at sample {step}")
                print(json.dumps({"wheel_physics_steps": step + 1, "total_steps": len(target),
                    "elapsed_s": time.monotonic() - started}), flush=True)
        result = trace.cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Wheel trace contains nonfinite values")
        return result[:, :2], result[:, 2:4], result[:, 4:6], result[:, 6]

    def run_batch_trace(self, cases):
        self.configure_batch_response(cases)
        torch, cases = self.torch, self.cases
        target, recorded, sizes = pad_wheel_cases(cases)
        tensor = lambda array: torch.as_tensor(array, device=self.args.device, dtype=self.efforts.dtype)
        reference, input_torque = tensor(target), tensor(recorded)
        coast = torch.as_tensor([row["coast"] for row in cases], device=self.args.device)[:, None]
        self.initialize(np.stack([row["q_real"][0] for row in cases]),
                        np.stack([row["dq_real"][0] for row in cases]))
        trace = torch.empty((*target.shape[:2], 7), device=self.args.device, dtype=self.efforts.dtype)
        carrier_nominal = self.nominal[:, self.carrier_ids]
        started = time.monotonic()
        for step in range(len(target)):
            q, dq = self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch
            position, velocity = q[:, self.ids], dq[:, self.ids]
            drift = (q[:, self.carrier_ids] - carrier_nominal).abs().amax(dim=1)
            torque = (self.args.kp * (reference[step] - velocity)).clamp(-4.5, 4.5)
            torque = torch.where(coast, 0., torque) if self.args.mode == "closed_loop" else input_torque[step]
            trace[step] = torch.cat([position, velocity, torque, drift[:, None]], dim=1)
            self.step(torque)
            if (step + 1) % 2000 == 0:
                if not bool(torch.isfinite(trace[step]).all()) or bool((velocity.abs() > 500).any()):
                    raise RuntimeError(f"Wheel batch diverged at sample {step}")
                print(json.dumps({"wheel_physics_steps": step + 1, "total_steps": len(target),
                    "independent_cases": len(cases), "elapsed_s": time.monotonic() - started}), flush=True)
        result = trace.cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite wheel batch trace")
        return [(row, result[:size, env]) for env, (row, size) in enumerate(zip(cases, sizes))]


def pad_wheel_cases(cases):
    sizes = [len(row["target"]) for row in cases]
    reference = np.empty((max(sizes), len(cases), 2), dtype=np.float32)
    torque = np.empty_like(reference)
    for env, (row, size) in enumerate(zip(cases, sizes)):
        reference[:size, env], reference[size:, env] = row["target"], row["target"][-1]
        torque[:size, env], torque[size:, env] = row["recorded"], row["recorded"][-1]
    return reference, torque, sizes


def write_wheel_case(output, writer, row, values):
    segment, variant, data = row["segment"], row["variant"], row["data"]
    position, velocity, torque, drift = values[:, :2], values[:, 2:4], values[:, 4:6], values[:, 6]
    n, real_q, real_dq, target, recorded = len(values), row["q_real"], row["dq_real"], row["target"], row["recorded"]
    if drift.max() > 1e-4:
        raise RuntimeError(f"Fixed wheel carrier drifted beyond 1e-4 rad/m: {drift.max()}")
    destination = output / variant
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / f"segment_{segment}.npz", q_sim=position, dq_sim=velocity,
        tau_request=torque, q_real=real_q, dq_real=real_dq, target=target, tau_real_nominal=recorded,
        time_s=np.arange(n) * DT, carrier_drift=drift, indices=data["indices"][:n])
    for step in range(0, n, 10):
        for axis, side in enumerate(("left", "right")):
            prefix = f"isaac_wheel/{variant}/segment_{segment}/{side}"
            for key, value in (("speed_real_rad_s", real_dq[step, axis]), ("speed_sim_rad_s", velocity[step, axis]),
                ("speed_target_rad_s", target[step, axis]), ("torque_nominal_real_nm", recorded[step, axis]),
                ("torque_request_nm", torque[step, axis])):
                writer.add_scalar(prefix + "/" + key, value, step)
    return {"segment": segment, "variant": variant, "label": data["description"]["label"], "samples": n,
        "velocity_rmse_rad_s": np.sqrt(np.mean((velocity - real_dq) ** 2, axis=0)).tolist(),
        "carrier_max_drift_rad_or_m": float(drift.max()), "max_grid_error_s": data["max_grid_error_s"],
        "frame_pd_error_p99_nm": data["frame_pd_error_p99_nm"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parameter_group = parser.add_mutually_exclusive_group()
    parameter_group.add_argument("--params", type=Path, help="wheel_response_fit.json; omitted gives current zero-loss prior")
    parameter_group.add_argument("--compare-params", type=Path,
        help="Run nominal and this candidate for every selected segment in parallel, same WheelBench physics")
    parser.add_argument("--segments", nargs="+", type=int, required=True)
    parser.add_argument("--axis-sign", nargs=2, type=float, choices=(-1., 1.), required=True,
                        help="API-to-bench left/right signs, explicit experimental convention")
    parser.add_argument("--kp", type=float, default=.6)
    parser.add_argument("--mode", choices=("closed_loop", "torque_input"), default="closed_loop")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kp <= 0 or not np.isfinite(args.kp) or (args.max_samples is not None and args.max_samples < 2):
        parser.error("kp must be finite/positive and max-samples at least two")
    inputs = load_inputs(args.run, args.segments, args.kp)
    params, fit = wheel_parameters(args.compare_params or args.params)
    args.instances = 2 * len(inputs) if args.compare_params else 1
    # Reuse the sealed manifest validation and closure audit without starting a simulator.
    from audit_v5_motor_coupling import MotorCouplingAudit
    audit = MotorCouplingAudit(args.bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "prepared", "training_release": False,
        "scope": "V6 CAD wheel inertia + native armature/friction; fixed-carrier approximation",
        "kp": args.kp, "control_hz": 1000, "target_hz": 50, "effort_limit_nm": 4.5,
        "axis_sign": args.axis_sign, "sign_scope": "explicit bench convention, not independent hardware-axis calibration",
        "independent_instances": args.instances, "parallel_nominal_candidate": bool(args.compare_params),
        "trace_collection": "full 1kHz GPU buffer; copy and TensorBoard serialization after each complete segment",
        "ratio_15_8_already_applied": True, "mode": args.mode,
        "friction_mapping": "native static=dynamic=mean(C+,C-), viscous=b; no added actuator lag",
        "candidate_parameters": {k: v.tolist() for k, v in params.items()}, "fit_provenance": fit,
        "manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((args.run / "wheel_1khz.npz").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_injection": "once per segment; no measured states injected during replay", "scores": [],
        "limitations": ["Carrier joints are fixed at nominal CAD pose, not replayed leg motion.",
            "Native static friction equals moving Coulomb candidate; low-speed presliding remains unmodeled.",
            "Wheel-output torque is nominal current conversion, not a load-cell measurement.",
            "Native friction semantics are effort-based on Sim>=5; solver equivalence still needs response checks."]}
    path, writer, launcher = args.output / "report.json", None, None
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": args.device, "enable_cameras": False})
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output / "events"))
        bench = WheelBench(args, audit.manifest, audit.spec,
                           wheel_parameters(None)[0] if args.compare_params else params)
        report["native_readback"] = bench.readback
        sign = np.asarray(args.axis_sign)
        if args.compare_params:
            cases = []
            for variant, response in (("nominal", wheel_parameters(None)[0]), ("candidate", params)):
                for segment, data in inputs.items():
                    n = min(len(data["indices"]), args.max_samples or len(data["indices"]))
                    cases.append({"segment": segment, "variant": variant, "params": response,
                        "data": data, "coast": data["description"]["coast"],
                        "q_real": data["q_api"][:n, 4:6] * sign,
                        "dq_real": data["dq_api"][:n, 4:6] * sign,
                        "target": data["wheel_velocity_target_api"][:n] * sign,
                        "recorded": data["tau_frame_api"][:n, 4:6] * sign})
            started = time.monotonic()
            for row, values in bench.run_batch_trace(cases):
                score = write_wheel_case(args.output, writer, row, values)
                report["scores"].append(score)
                print(json.dumps(score), flush=True)
            report.update(status="completed", runtime_s=time.monotonic() - started,
                state_initializations=len(cases), tail_padding="Short cases hold tail but padded samples never score/save")
            return
        for segment, data in inputs.items():
            started = time.monotonic()
            n = min(len(data["indices"]), args.max_samples or len(data["indices"]))
            real_q, real_dq = data["q_api"][:n, 4:6] * sign, data["dq_api"][:n, 4:6] * sign
            target, recorded = data["wheel_velocity_target_api"][:n] * sign, data["tau_frame_api"][:n, 4:6] * sign
            position, velocity, torque, drift = bench.run_trace(
                target, recorded, real_q[0], real_dq[0], data["description"]["coast"], args.mode)
            for step in range(0, n, 10):
                for axis, side in enumerate(("left", "right")):
                    prefix = f"isaac_wheel/segment_{segment}/{side}"
                    for key, value in (("speed_real_rad_s", real_dq[step, axis]),
                        ("speed_sim_rad_s", velocity[step, axis]), ("speed_target_rad_s", target[step, axis]),
                        ("torque_nominal_real_nm", recorded[step, axis]), ("torque_request_nm", torque[step, axis])):
                        writer.add_scalar(prefix + "/" + key, value, step)
            score = {"segment": segment, "label": data["description"]["label"], "samples": n,
                "velocity_rmse_rad_s": np.sqrt(np.mean((velocity - real_dq) ** 2, axis=0)).tolist(),
                "carrier_max_drift_rad_or_m": float(drift.max()),
                "max_grid_error_s": data["max_grid_error_s"],
                "frame_pd_error_p99_nm": data["frame_pd_error_p99_nm"], "runtime_s": time.monotonic() - started}
            if drift.max() > 1e-4:
                raise RuntimeError(f"Fixed wheel carrier drifted beyond 1e-4 rad/m: {drift.max()}")
            np.savez_compressed(args.output / f"segment_{segment}.npz", q_sim=position, dq_sim=velocity,
                tau_request=torque, q_real=real_q, dq_real=real_dq, target=target, tau_real_nominal=recorded,
                time_s=np.arange(n) * DT, carrier_drift=drift, indices=data["indices"][:n])
            report["scores"].append(score)
            report["status"] = "running"
            writer.flush()
            print(json.dumps(score), flush=True)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if writer is not None:
            writer.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
