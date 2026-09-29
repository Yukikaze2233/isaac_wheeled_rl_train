#!/usr/bin/env python3
"""Parallel nominal/candidate replay of every requested installed-leg segment.

Each (variant, segment) owns a separate fixed-base articulation, initialized once
from its own measured start. One batched PhysX step advances all instances. No
physics result is sampled onto a reduced grid; only TensorBoard display is 100Hz.
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

from replay_identification_isaac import BUNDLE, DT, parameters, map_initial_state


def case_plan(replay, segments, candidate, max_samples=None):
    rows = []
    for variant, params in (("nominal", parameters({})), ("candidate", candidate)):
        for segment in segments:
            indices, q0, v0, gap = replay.initial_state(segment)
            indices = indices[:max_samples] if max_samples is not None else indices
            rows.append({"variant": variant, "segment": segment, "indices": indices,
                         "q0": q0, "v0": v0, "initial_gap": gap, "params": params})
    return rows


def padded_inputs(replay, rows):
    """Only completed environments receive held tail padding; metrics trim it."""
    length = max(len(row["indices"]) for row in rows)
    reference = np.empty((length, len(rows), 2), dtype=np.float32)
    torque = np.empty_like(reference)
    for env, row in enumerate(rows):
        ix = row["indices"]
        reference[:len(ix), env] = replay.reference[ix]
        reference[len(ix):, env] = replay.reference[ix[-1]]
        torque[:len(ix), env] = replay.real_torque[ix]
        torque[len(ix):, env] = replay.real_torque[ix[-1]]
    return reference, torque


class BatchBench:
    def __init__(self, args, replay, cases):
        import torch
        import warp as wp
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.utils.version import get_isaac_sim_version
        from pxr import UsdGeom, UsdPhysics

        if get_isaac_sim_version().major < 5:
            raise RuntimeError("Native effort friction requires Sim>=5")
        self.torch, self.replay, self.args = torch, replay, args
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            device=args.device, dt=DT, render_interval=20, gravity=tuple(args.gravity)))
        for env in range(len(cases)):
            UsdGeom.Xform.Define(self.sim.stage, f"/World/envs/env_{env}").AddTranslateOp().Set(
                (2. * (env % 4), 2. * (env // 4), 0.))
        manifest = replay.audit.manifest
        active, springs = manifest["control_joint_names"], manifest["spring_joint_names"]
        passive = [n for n in manifest["tree_joint_names"] if n not in active + springs]
        def actuator(names, damping=0., speed=1e9):
            return IdealPDActuatorCfg(joint_names_expr=names, stiffness=0., damping=damping,
                effort_limit=1000., effort_limit_sim=1000., velocity_limit_sim=speed,
                armature=0., friction=0., dynamic_friction=0., viscous_friction=0.)
        cfg = ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(args.bundle / manifest["usd"]),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False, solver_position_iteration_count=32,
                    solver_velocity_iteration_count=8)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.),
                joint_pos=manifest["nominal_joint_pos"], joint_vel={".*": 0.}),
            actuators={"active": actuator(active), "passive": actuator(passive, .002),
                       "gas": actuator(springs, .002, 100.)}, soft_joint_pos_limit_factor=1.)
        self.robot = Articulation(cfg)
        loops = [p for p in self.sim.stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
        if len(loops) != 6 * len(cases) or not all(UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get() for p in loops):
            raise RuntimeError("Incorrect batched external loop count")
        self.sim.reset()
        if not self.robot.is_fixed_base or set(self.robot.joint_names) != set(manifest["tree_joint_names"]):
            raise RuntimeError("Batched articulation topology mismatch")
        order = [int(path.split("/env_")[1].split("/")[0]) for path in self.robot.root_view.prim_paths]
        if sorted(order) != list(range(len(cases))):
            raise RuntimeError("Ambiguous engine instance order")
        self.cases = [cases[i] for i in order]
        self.ids = [self.robot.joint_names.index(n) for n in replay.joints]
        self.spring_ids = [self.robot.joint_names.index(n) for n in springs]
        self.knee_id = self.robot.joint_names.index(replay.knee)
        self.tensor = lambda value: torch.as_tensor(value, device=args.device, dtype=torch.float32)
        self.params = {key: self.tensor(np.stack([row["params"][key] for row in self.cases]))
                       for key in ("armature", "damping", "frictionloss", "time_constant_s", "gain")}
        self.robot.write_joint_armature_to_sim_index(armature=self.params["armature"], joint_ids=self.ids)
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=self.params["frictionloss"], joint_dynamic_friction_coeff=self.params["frictionloss"],
            joint_viscous_friction_coeff=self.params["damping"], joint_ids=self.ids)
        armature = wp.to_torch(self.robot.root_view.get_dof_armatures()).cpu().numpy()[:, self.ids]
        friction = wp.to_torch(self.robot.root_view.get_dof_friction_properties()).cpu().numpy()[:, self.ids]
        if not np.allclose(armature, self.params["armature"].cpu().numpy(), atol=1e-7):
            raise RuntimeError("Batch native armature mismatch")
        expected = np.stack([self.params["frictionloss"].cpu().numpy()] * 2
                            + [self.params["damping"].cpu().numpy()], axis=-1)
        if not np.allclose(friction, expected, atol=1e-7):
            raise RuntimeError("Batch native friction mismatch")
        limits = wp.to_torch(self.robot.root_view.get_dof_limits()).cpu().numpy()[:, self.knee_id]
        if not np.allclose(limits, replay.model.jnt_range[replay.knee_id], atol=1e-6, rtol=0):
            raise RuntimeError("Batch passive knee limit mismatch")
        self.native = {"isaac_sim_version": str(get_isaac_sim_version()), "clone_order": order,
            "selected_joint_names": replay.joints, "armature": armature.tolist(),
            "friction_static_dynamic_viscous": friction.tolist(), "knee_limits_rad": limits.tolist()}
        initial = [map_initial_state(replay, row["q0"], row["v0"], self.robot.joint_names) for row in self.cases]
        self.robot.write_joint_position_to_sim_index(position=self.tensor(np.stack([q for q, _ in initial])))
        self.robot.write_joint_velocity_to_sim_index(velocity=self.tensor(np.stack([v for _, v in initial])))
        self.robot.reset()
        self.robot.update(DT)
        self.reference, self.recorded = (self.tensor(x) for x in padded_inputs(replay, self.cases))
        self.motor = self.tensor(np.stack([replay.api_to_model * replay.a["tau_cmd_api"][row["indices"][0]]
                                          for row in self.cases]))
        self.alpha = torch.exp(-DT / self.params["time_constant_s"].clamp_min(1e-12))
        self.kp, self.kd, self.cap = (self.tensor(x) for x in (replay.kp, replay.kd, replay.cap))
        self.s0 = self.tensor([replay.spec["spring_binding"][n]["compression_at_q_zero_m"] for n in springs])
        self.efforts = torch.zeros((len(cases), len(self.robot.joint_names)), device=args.device)
        constraints = [c for c in replay.spec["constraints"] if c["body0"].startswith(replay.letter)]
        self.body_ids = [[self.robot.body_names.index(c["body0"]), self.robot.body_names.index(c["body1"])]
                         for c in constraints]
        self.points = self.tensor([[c["local_pos0_m"], c["local_pos1_m"]] for c in constraints])

    def run(self):
        torch, replay = self.torch, self.replay
        length, count = self.reference.shape[:2]
        # Per sample: q(2),dq(2),request(2),motor(2),applied(2),beta,gap,
        # compression(2),spring force(2). No per-step CPU synchronization.
        trace = torch.empty((length, count, 16), device=self.args.device)
        sign = 1. if replay.side == "left" else -1.
        stop_q = replay.original_knee_range[1 if replay.side == "left" else 0]
        coeff = replay.curve["monomial_coefficients_n"]
        started = time.monotonic()
        for step in range(length):
            q, dq = self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch
            selected_q, selected_dq = q[:, self.ids], dq[:, self.ids]
            request = (self.kp * (self.reference[step] - selected_q) - self.kd * selected_dq).clamp(-self.cap, self.cap)
            if self.args.mode == "torque_input":
                request = self.recorded[step]
            self.motor = self.alpha * self.motor + (1 - self.alpha) * self.params["gain"] * request
            compression = self.s0 - q[:, self.spring_ids]
            u = (compression / replay.curve["stroke_m"]).clamp(0., 1.)
            spring = coeff[0] + u * (coeff[1] + u * (coeff[2] + u * coeff[3]))
            beta = replay.inner_limits_deg[1] + (180 / np.pi) * sign * (q[:, self.knee_id] - stop_q)
            pose = self.robot.data.body_link_pose_w.torch[:, self.body_ids]
            local = self.points.expand(count, -1, -1, -1)
            cross = torch.linalg.cross(pose[..., 3:6], local)
            world = pose[..., :3] + local + 2 * (pose[..., 6:] * cross + torch.linalg.cross(pose[..., 3:6], cross))
            gap = (world[:, :, 0] - world[:, :, 1]).norm(dim=-1).amax(-1)
            self.efforts.zero_()
            self.efforts[:, self.ids] = self.motor
            self.efforts[:, self.spring_ids] = spring
            self.robot.set_joint_effort_target_index(target=self.efforts)
            self.robot.write_data_to_sim()
            applied = self.robot.data.applied_torque.torch[:, self.ids]
            trace[step] = torch.cat([selected_q, selected_dq, request, self.motor, applied,
                                    beta[:, None], gap[:, None], compression, spring], dim=1)
            self.sim.step(render=False)
            self.robot.update(DT)
            if (step + 1) % 2000 == 0:
                if not bool(torch.isfinite(trace[step]).all()) or bool((selected_dq.abs() > 500).any()):
                    raise RuntimeError(f"Batched replay diverged at sample {step}")
                print(json.dumps({"completed_physics_steps": step + 1, "total_steps": length,
                    "independent_cases": count, "elapsed_s": time.monotonic() - started}), flush=True)
        result = trace.cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite batched trace")
        return result


def save_case(output, writer, replay, row, values):
    ix, segment, variant = row["indices"], row["segment"], row["variant"]
    values = values[:len(ix)]
    q, dq, beta, gap = values[:, :2], values[:, 2:4], values[:, 10], values[:, 11]
    trim = slice(min(1000, len(ix) // 4), None)
    lower, upper = replay.inner_limits_deg
    error = q[trim] - replay.real_q[ix][trim]
    score = {"variant": variant, "segment": segment, "samples": len(ix),
        "axis_rmse_rad": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "velocity_rmse_rad_s": np.sqrt(np.mean((dq[trim] - replay.real_dq[ix][trim]) ** 2, axis=0)).tolist(),
        "max_closure_gap_m": float(gap.max()), "initial_closure_gap_m": row["initial_gap"],
        "peak_request_nm": abs(values[:, 4:6]).max(axis=0).tolist(),
        "simulated_inner_range_deg": [float(beta.min()), float(beta.max())],
        "knee_stop": {"proximity_margin_deg": .1, "inner_limits_deg": [float(lower), float(upper)],
            "upper_proximity_fraction": float(np.mean(beta >= upper - .1)),
            "lower_proximity_fraction": float(np.mean(beta <= lower + .1)),
            "max_upper_violation_deg": float(max(beta.max() - upper, 0.)),
            "scope": "position proximity including penetration, not native reaction-force measurement"}}
    destination = output / variant
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / f"segment_{segment}.npz", indices=ix, time_s=np.arange(len(ix)) * DT,
        q_sim=q, dq_sim=dq, tau_request=values[:, 4:6], tau_motor=values[:, 6:8], tau_applied=values[:, 8:10],
        knee_inner_deg_sim=beta, closure_gap_m=gap, spring_compression_m=values[:, 12:14],
        spring_force_n=values[:, 14:16], q_real=replay.real_q[ix], dq_real=replay.real_dq[ix],
        q_reference=replay.reference[ix], tau_real_nominal=replay.real_torque[ix])
    for step in range(0, len(ix), 10):
        prefix = f"isaac_batch/{replay.side}/{variant}/segment_{segment}"
        writer.add_scalar(prefix + "/knee_inner_deg", beta[step], step)
        writer.add_scalar(prefix + "/closure_gap_m", gap[step], step)
        for axis in range(2):
            for name, value in (("q_sim_rad", q[step, axis]), ("q_real_rad", replay.real_q[ix[step], axis]),
                ("q_reference_rad", replay.reference[ix[step], axis]), ("dq_sim_rad_s", dq[step, axis]),
                ("dq_real_rad_s", replay.real_dq[ix[step], axis]), ("tau_motor_nm", values[step, 6 + axis])):
                writer.add_scalar(prefix + f"/axis_{axis}/" + name, value, step)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--segments", type=int, nargs="+", default=[37, 39, 41, 43, 79, 89])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--mode", choices=("closed_loop", "torque_input"), default="closed_loop")
    parser.add_argument("--coordinate-sign", type=float, choices=(-1., 1.), default=-1.)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0., 0., -9.81])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.segments)) != len(args.segments) or (args.max_samples is not None and args.max_samples < 2):
        parser.error("Unique segments and max-samples>=2 required")
    from identify_pair_dynamics import Replay
    replay = Replay(args.run, side=args.side, physics_dt=DT, bundle=args.bundle,
                    coordinate_sign=args.coordinate_sign, gravity=args.gravity)
    if not np.allclose(replay.kp, 160.) or not np.allclose(replay.kd, 2.5):
        raise ValueError("Expected recorded 1kHz PD160/2.5 and held 50Hz references")
    candidate = parameters(json.loads(args.params.read_text()))
    rows = case_plan(replay, args.segments, candidate, args.max_samples)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "prepared", "training_release": False, "physics_dt_s": DT,
        "pc_control_hz": 1000, "reference_hz": 50, "side": args.side, "mode": args.mode,
        "pd_kp": replay.kp.tolist(), "pd_kd": replay.kd.tolist(), "torque_cap_nm": replay.cap.tolist(),
        "candidate": {k: v.tolist() for k, v in candidate.items()},
        "nominal": {k: v.tolist() for k, v in parameters({}).items()},
        "manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_npz_sha256": hashlib.sha256((args.run / "response_1khz.npz").read_bytes()).hexdigest(),
        "source_run": str(args.run), "input_timing_audits": replay.time_audits,
        "state_initialization": "each independent (variant,segment) initialized once, then free integration",
        "tail_padding": "shorter cases continue under last held reference; padded samples never scored or saved",
        "native_friction": "Sim>=5 static=dynamic=frictionloss Nm, viscous=damping Nm*s/rad; no explicit duplicate",
        "scores": []}
    path, launcher, writer = args.output / "report.json", None, None
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": args.device, "enable_cameras": False})
        bench = BatchBench(args, replay, rows)
        report["native_readback"] = bench.native
        report["status"] = "running"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        values = bench.run()
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output / "events"))
        for env, row in enumerate(bench.cases):
            score = save_case(args.output, writer, replay, row, values[:, env])
            report["scores"].append(score)
            print(json.dumps(score), flush=True)
        report.update(status="completed", runtime_s=time.monotonic() - started,
                      state_initializations=len(rows), independent_cases=len(rows))
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
