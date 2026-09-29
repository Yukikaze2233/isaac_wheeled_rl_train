#!/usr/bin/env python3
"""Fixed-chassis Isaac replay of installed-leg bag references.

The kinematic solve initializes each requested segment once. Thereafter PhysX
evolves the entire closed linkage under gravity, gas force and recorded PC PD.
This tool measures cross-engine agreement; it never releases fitted parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/纯底盘_v6_232mm_stop105/urdf"
DT = .001


def parameters(raw):
    """Accept the MuJoCo fitter schema or explicit named hip/knee fields."""
    raw = dict(raw)
    for pair, names in (("damping", ("damping_hip", "damping_knee")),
                        ("frictionloss", ("friction_hip", "friction_knee"))):
        if any(name in raw for name in names):
            if pair in raw or not all(name in raw for name in names):
                raise ValueError(f"Provide {pair} or both named axes, without duplication")
            raw[pair] = [raw.pop(name) for name in names]
    if "lag" in raw:
        if "time_constant_s" in raw:
            raise ValueError("Provide lag or time_constant_s, not both")
        raw["time_constant_s"] = raw.pop("lag")
    defaults = {"armature": 0., "damping": 0., "frictionloss": 0.,
                "time_constant_s": 0., "gain": 1.}
    if unknown := set(raw) - set(defaults):
        raise ValueError(f"Unknown response parameters: {sorted(unknown)}")
    result = {}
    for name, default in defaults.items():
        try:
            value = np.broadcast_to(np.asarray(raw.get(name, default), dtype=float), (2,)).copy()
        except ValueError as error:
            raise ValueError(f"{name} must be scalar or a hip/knee pair") from error
        if not np.isfinite(value).all() or np.any(value < 0) or (name == "gain" and np.any(value == 0)):
            raise ValueError(f"{name} must be finite and nonnegative (gain positive)")
        result[name] = value
    return result


def response_effort(previous, request, velocity, params, friction_velocity):
    """Exact first-order drive update, plus separately declared passive losses.

    Smooth Coulomb friction is an optional explicit approximation. The default
    simulator path uses Sim >=5 native effort-based static/dynamic friction;
    in that path the returned loss is diagnostic only and is not also applied.
    """
    if not np.isfinite(friction_velocity) or friction_velocity <= 0:
        raise ValueError("friction_velocity must be finite and positive")
    alpha = np.exp(-DT / np.maximum(params["time_constant_s"], 1e-12))
    motor = alpha * previous + (1 - alpha) * params["gain"] * request
    loss = params["damping"] * velocity + params["frictionloss"] * np.tanh(velocity / friction_velocity)
    return motor, loss


def map_initial_state(replay, q0, v0, joint_names):
    """Name-bind selected-side closed-chain initial q/dq into the full USD."""
    q = np.array([replay.spec["nominal_joint_pos"][name] for name in joint_names], dtype=float)
    dq = np.zeros(len(joint_names))
    for index in range(replay.model.njnt):
        joint = replay.model.joint(index)
        if joint.name not in joint_names:
            raise ValueError(f"USD does not contain recorded mechanism joint {joint.name}")
        output = joint_names.index(joint.name)
        q[output] = q0[joint.qposadr[0]]
        dq[output] = v0[joint.dofadr[0]]
    return q, dq


def point_world(pose_xyzw, local):
    pose, local = np.asarray(pose_xyzw), np.asarray(local)
    uv = np.cross(pose[3:6], local)
    return pose[:3] + local + 2 * (pose[6] * uv + np.cross(pose[3:6], uv))


class IsaacReplay:
    def __init__(self, args, replay, params):
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.utils.version import get_isaac_sim_version
        from pxr import UsdPhysics

        self.args, self.replay, self.params = args, replay, params
        self.torch = torch
        self.version = str(get_isaac_sim_version())
        if args.friction_model == "native" and get_isaac_sim_version().major < 5:
            raise RuntimeError("Native effort-based friction requires Isaac Sim >=5")
        self.spec, self.manifest = replay.spec, replay.audit.manifest
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            device=args.device, dt=DT, render_interval=20, gravity=tuple(args.gravity)))
        active, springs = self.manifest["control_joint_names"], self.manifest["spring_joint_names"]
        passive = [name for name in self.manifest["tree_joint_names"] if name not in active + springs]
        # Motor actuator feedback gains are zero: PC PD is evaluated in step().
        # Losses use native friction properties or the explicit smooth option,
        # never both. Passive hinge/slider damping retains the asset prior.
        actuators = {
            "other_active": IdealPDActuatorCfg(joint_names_expr=[n for n in active if n not in replay.joints],
                stiffness=0., damping=0., effort_limit=1000., effort_limit_sim=1000.,
                velocity_limit_sim=1e9, armature=0., friction=0., dynamic_friction=0., viscous_friction=0.),
            "passive": IdealPDActuatorCfg(joint_names_expr=passive, stiffness=0., damping=.002,
                effort_limit=1000., effort_limit_sim=1000., velocity_limit_sim=1e9, armature=0.,
                friction=0., dynamic_friction=0., viscous_friction=0.),
            "gas": IdealPDActuatorCfg(joint_names_expr=springs, stiffness=0., damping=.002,
                effort_limit=1000., effort_limit_sim=1000., velocity_limit_sim=100., armature=0.,
                friction=0., dynamic_friction=0., viscous_friction=0.),
        }
        for axis, name in enumerate(replay.joints):
            native_friction = float(params["frictionloss"][axis]) if args.friction_model == "native" else 0.
            native_viscous = float(params["damping"][axis]) if args.friction_model == "native" else 0.
            actuators[f"selected_{axis}"] = IdealPDActuatorCfg(joint_names_expr=[name],
                stiffness=0., damping=0., effort_limit=1000., effort_limit_sim=1000.,
                velocity_limit_sim=1e9, armature=float(params["armature"][axis]),
                friction=native_friction, dynamic_friction=native_friction, viscous_friction=native_viscous)
        cfg = ArticulationCfg(prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(args.bundle / self.manifest["usd"]),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False, solver_position_iteration_count=args.position_iterations,
                    solver_velocity_iteration_count=args.velocity_iterations)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.),
                joint_pos=self.manifest["nominal_joint_pos"], joint_vel={".*": 0.}),
            actuators=actuators, soft_joint_pos_limit_factor=1.)
        self.robot = Articulation(cfg)
        loops = [prim for prim in self.sim.stage.Traverse() if prim.IsA(UsdPhysics.SphericalJoint)]
        if len(loops) != 6 or not all(UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get() for p in loops):
            raise RuntimeError("Expected six external closed-chain constraints")
        self.sim.reset()
        if set(self.robot.joint_names) != set(self.manifest["tree_joint_names"]):
            raise RuntimeError("PhysX joint topology differs from sealed asset")
        if set(self.robot.body_names) != set(self.manifest["rigid_body_names"]):
            raise RuntimeError("PhysX body topology differs from sealed asset")
        if not self.robot.is_fixed_base:
            raise RuntimeError("Rig root is not fixed")
        self.ids = [self.robot.joint_names.index(n) for n in replay.joints]
        import warp as wp
        # Direct engine readback, not merely the requested actuator config.
        native_friction = wp.to_torch(self.robot.root_view.get_dof_friction_properties()).cpu().numpy()[0]
        self.native_properties = {
            "joint_names": self.robot.joint_names,
            "armature": wp.to_torch(self.robot.root_view.get_dof_armatures()).cpu().numpy()[0].tolist(),
            "friction_properties": native_friction.tolist(),
            "selected_joint_names": replay.joints,
            "selected_static_effort_nm": native_friction[self.ids, 0].tolist(),
            "selected_dynamic_effort_nm": native_friction[self.ids, 1].tolist(),
            "selected_viscous_nm_s_rad": native_friction[self.ids, 2].tolist(),
        }
        if not np.allclose(np.asarray(self.native_properties["armature"])[self.ids], params["armature"], atol=1e-7):
            raise RuntimeError("Native armature readback differs from candidate")
        for field, expected in (("selected_static_effort_nm", params["frictionloss"]),
                ("selected_dynamic_effort_nm", params["frictionloss"]),
                ("selected_viscous_nm_s_rad", params["damping"])):
            if not np.allclose(self.native_properties[field], expected if args.friction_model == "native" else 0., atol=1e-7):
                raise RuntimeError(f"Native response property mismatch: {field}")
        self.spring_ids = [self.robot.joint_names.index(n) for n in springs]
        self.knee_id = self.robot.joint_names.index(replay.knee)
        native_limits = wp.to_torch(self.robot.root_view.get_dof_limits()).cpu().numpy()[0, self.knee_id]
        self.native_properties["passive_knee_limits_rad"] = native_limits.tolist()
        if not np.allclose(native_limits, replay.model.jnt_range[replay.knee_id], atol=1e-6, rtol=0.):
            raise RuntimeError("Native knee stops differ from the sealed replay asset")
        self.s0 = np.array([self.spec["spring_binding"][n]["compression_at_q_zero_m"] for n in springs])
        self.constraints = [c for c in self.spec["constraints"] if c["body0"].startswith(replay.letter)]
        self.efforts = torch.zeros((1, len(self.robot.joint_names)), device=args.device)
        self.state_writes = 0

    def initialize(self, q0, v0):
        q, dq = map_initial_state(self.replay, q0, v0, self.robot.joint_names)
        self.robot.write_joint_position_to_sim_index(position=self.torch.tensor(
            q[None], device=self.args.device, dtype=self.torch.float32))
        self.robot.write_joint_velocity_to_sim_index(velocity=self.torch.tensor(
            dq[None], device=self.args.device, dtype=self.torch.float32))
        self.robot.reset()
        self.robot.update(DT)
        self.state_writes += 1

    def snapshot(self):
        q = self.robot.data.joint_pos.torch[0].cpu().numpy().copy()
        dq = self.robot.data.joint_vel.torch[0].cpu().numpy().copy()
        pose = self.robot.data.body_link_pose_w.torch[0].cpu().numpy()
        if not np.isfinite(np.r_[q, dq, pose.ravel()]).all() or np.abs(dq).max() > 500:
            raise RuntimeError("Divergent PhysX replay")
        gaps = [np.linalg.norm(
            point_world(pose[self.robot.body_names.index(c["body0"])], c["local_pos0_m"])
            - point_world(pose[self.robot.body_names.index(c["body1"])], c["local_pos1_m"]))
            for c in self.constraints]
        beta = self.replay.knee_inner_deg(q[self.knee_id])
        return q, dq, float(beta), float(max(gaps))

    def step(self, motor, loss, q):
        compression = self.s0 - q[self.spring_ids]
        # Match identify_pair_dynamics.py's explicit catalogue extrapolation
        # policy (clamp normalized compression at 0..1), not preview's 0..0.9.
        u = np.clip(compression / self.replay.curve["stroke_m"], 0., 1.)
        spring = np.polynomial.polynomial.polyval(u, self.replay.curve["monomial_coefficients_n"])
        self.efforts.zero_()
        self.efforts[:, self.ids] = self.torch.as_tensor(
            motor - loss, device=self.args.device, dtype=self.efforts.dtype)
        self.efforts[:, self.spring_ids] = self.torch.as_tensor(
            spring, device=self.args.device, dtype=self.efforts.dtype)
        self.robot.set_joint_effort_target_index(target=self.efforts)
        self.robot.write_data_to_sim()
        applied = self.robot.data.applied_torque.torch[0, self.ids].cpu().numpy().copy()
        self.sim.step(render=False)
        self.robot.update(DT)
        return applied, compression, spring


def run_segment(bench, segment, writer):
    replay, args, params = bench.replay, bench.args, bench.params
    ix, q0, v0, gap0 = replay.initial_state(segment)
    if args.max_samples is not None:
        ix = ix[:args.max_samples]
    bench.initialize(q0, v0)
    count = len(ix)
    trace = {name: np.empty((count, 2)) for name in (
        "q_sim", "dq_sim", "tau_request", "tau_motor", "tau_passive_loss", "tau_applied",
        "spring_compression_m", "spring_force_n")}
    trace.update({name: np.empty(count) for name in ("knee_inner_deg_sim", "closure_gap_m")})
    motor = replay.api_to_model * replay.a["tau_cmd_api"][ix[0]]
    for step, sample in enumerate(ix):
        q, dq, beta, gap = bench.snapshot()
        selected_q, selected_dq = q[bench.ids], dq[bench.ids]
        request = (np.clip(replay.kp * (replay.reference[sample] - selected_q)
                           - replay.kd * selected_dq, -replay.cap, replay.cap)
                   if args.mode == "closed_loop" else replay.real_torque[sample])
        motor, loss = response_effort(motor, request, selected_dq, params, args.friction_velocity)
        if args.friction_model == "native":
            loss = np.zeros(2)  # Native solver supplies all candidate friction/damping once.
        applied, compression, spring = bench.step(motor, loss, q)
        for key, value in (("q_sim", selected_q), ("dq_sim", selected_dq),
                ("tau_request", request), ("tau_motor", motor), ("tau_passive_loss", loss),
                ("tau_applied", applied), ("knee_inner_deg_sim", beta), ("closure_gap_m", gap),
                ("spring_compression_m", compression), ("spring_force_n", spring)):
            trace[key][step] = value
        if step % args.log_every == 0:
            prefix = f"isaac_replay/{replay.side}/segment_{segment}"
            writer.add_scalar(prefix + "/time_s", step * DT, step)
            writer.add_scalar(prefix + "/knee_inner_deg", beta, step)
            writer.add_scalar(prefix + "/closure_gap_m", gap, step)
            for axis in range(2):
                p = prefix + f"/axis_{axis}"
                for name, value in (("q_real_rad", replay.real_q[sample, axis]),
                    ("q_reference_rad", replay.reference[sample, axis]), ("q_sim_rad", selected_q[axis]),
                    ("dq_real_rad_s", replay.real_dq[sample, axis]), ("dq_sim_rad_s", selected_dq[axis]),
                    ("tau_real_nominal_nm", replay.real_torque[sample, axis]),
                    ("tau_request_nm", request[axis]), ("tau_motor_nm", motor[axis]),
                    ("tau_passive_loss_nm", loss[axis]), ("tau_applied_nm", applied[axis])):
                    writer.add_scalar(p + "/" + name, value, step)
    trim = slice(min(1000, count // 4), None)
    error = trace["q_sim"][trim] - replay.real_q[ix][trim]
    score = {"segment": int(segment), "samples": count, "initial_closure_gap_m": gap0,
        "axis_rmse_rad": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "velocity_rmse_rad_s": np.sqrt(np.mean((trace["dq_sim"][trim] - replay.real_dq[ix][trim]) ** 2, axis=0)).tolist(),
        "max_closure_gap_m": float(trace["closure_gap_m"].max()),
        "simulated_inner_range_deg": [float(trace["knee_inner_deg_sim"].min()), float(trace["knee_inner_deg_sim"].max())],
        "peak_request_nm": np.abs(trace["tau_request"]).max(axis=0).tolist()}
    lower, upper = replay.inner_limits_deg
    beta = trace["knee_inner_deg_sim"]
    score["knee_stop"] = {
        "inner_limits_deg": [float(lower), float(upper)], "proximity_margin_deg": .1,
        "initial_upper_violation_deg": float(max(beta[0] - upper, 0.)),
        "upper_proximity_fraction": float(np.mean(beta >= upper - .1)),
        "lower_proximity_fraction": float(np.mean(beta <= lower + .1)),
        "max_upper_violation_deg": float(max(beta.max() - upper, 0.)),
        "max_lower_violation_deg": float(max(lower - beta.min(), 0.)),
        "scope": "position-based proximity proxy including penetration; not measured/native reaction-force contact",
    }
    trace.update(indices=ix, time_s=np.arange(count) * DT,
                 q_real=replay.real_q[ix], dq_real=replay.real_dq[ix],
                 q_reference=replay.reference[ix], tau_real_nominal=replay.real_torque[ix])
    return score, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--params", type=Path, help="Fitter parameters; omitted means current unfit nominal prior")
    parser.add_argument("--segments", nargs="+", type=int, default=[37, 39, 41, 43, 79, 89])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("closed_loop", "torque_input"), default="closed_loop")
    parser.add_argument("--coordinate-sign", type=float, choices=(-1., 1.), default=-1.)
    parser.add_argument("--gravity", type=float, nargs=3, default=[0., 0., -9.81])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--friction-velocity", type=float, default=.01,
                        help="rad/s smooth-Coulomb scale; no exact static friction model")
    parser.add_argument("--friction-model", choices=("native", "smooth"), default="native",
                        help="Native Sim>=5 effort friction, or explicit smooth-Coulomb diagnostic")
    parser.add_argument("--position-iterations", type=int, default=32)
    parser.add_argument("--velocity-iterations", type=int, default=8)
    args = parser.parse_args()
    if args.log_every < 1 or (args.max_samples is not None and args.max_samples < 2):
        parser.error("log-every must be positive; max-samples must be at least two")
    if len(set(args.segments)) != len(args.segments):
        parser.error("segments must be unique")
    params = parameters(json.loads(args.params.read_text()) if args.params else {})
    response_effort(np.zeros(2), np.zeros(2), np.zeros(2), params, args.friction_velocity)
    # MuJoCo is used only for the verified closure initialization and file parser.
    # Perform all numerical input checks before starting Kit or consuming GPU.
    from identify_pair_dynamics import Replay
    replay = Replay(args.run, side=args.side, physics_dt=DT, bundle=args.bundle,
                    coordinate_sign=args.coordinate_sign, gravity=args.gravity)
    if not np.allclose(replay.kp, 160.) or not np.allclose(replay.kd, 2.5):
        raise ValueError("This cross-engine bench expects recorded PD160/2.5")
    for segment in args.segments:
        replay.initial_state(segment)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "prepared", "training_release": False,
        "engine": "Isaac PhysX", "fixed_base": True, "floor_present": False,
        "physics_dt_s": DT, "pc_control_dt_s": DT, "reference_frequency_hz": 50,
        "reference_source": "recorded held q_ref_model; no resampling or interpolation",
        "pd_kp": replay.kp.tolist(), "pd_kd": replay.kd.tolist(), "torque_cap_nm": replay.cap.tolist(),
        "parameters": {name: value.tolist() for name, value in params.items()},
        "coordinate_sign": args.coordinate_sign, "gravity": args.gravity, "mode": args.mode,
        "friction_model": {"kind": args.friction_model, "velocity_scale_rad_s": args.friction_velocity,
            "exact_mujoco_frictionloss_equivalence": False, "static_stiction_modeled": args.friction_model == "native",
            "native_mapping": "static=dynamic=frictionloss Nm; viscous=damping Nm*s/rad; Sim>=5 only",
            "tau_passive_loss_trace": "explicit loss only; zero when native solver handles loss",
            "tau_applied_trace": "external submitted effort, excludes native solver friction and constraint forces"},
        "state_injection": "once at each requested segment initial condition; no writes during integration",
        "opposite_leg": "present and unpowered; fixed chassis decouples its mechanics",
        "source_run": str(args.run.resolve()), "source_metadata": replay.meta,
        "input_npz_sha256": hashlib.sha256((args.run / "response_1khz.npz").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
        "nominal_grid_time_audit": replay.time_audits,
        "limitations": ["Effective-response candidates are conditional, not measured physical truth.",
            "Native PhysX and MuJoCo constraint regularization differ; equal parameters require response comparison.",
            "No feedback quantization, unknown device sample delay or stop compliance is inferred here.",
            "Same-recording held-out segments are not independent new acquisitions."], "scores": []}
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    launcher = writer = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": args.device, "enable_cameras": False})
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output / "events"))
        bench = IsaacReplay(args, replay, params)
        report.update(isaac_sim_version=bench.version, native_properties=bench.native_properties)
        for segment in args.segments:
            started = time.monotonic()
            score, trace = run_segment(bench, segment, writer)
            score["runtime_s"] = time.monotonic() - started
            report["scores"].append(score)
            report["state_initializations"] = bench.state_writes
            np.savez_compressed(args.output / f"segment_{segment}.npz", **trace)
            writer.flush()
            print(json.dumps(score, allow_nan=False), flush=True)
            report["status"] = "running"
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
