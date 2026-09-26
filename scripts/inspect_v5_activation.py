#!/usr/bin/env python3
"""Physical zero-torque / prepare / ONNX takeover experiments with optional GUI.

This is an activation research bench, not a deployment recovery controller.
Only trial reset writes robot state. Passive gas springs remain enabled.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "docs/examples"))


POSES = {
    "upright": (0., 0., 0.),
    "upright_release": (0., 0., .2),
    "crouched": (0., 0., 0.),
    "crouched_release": (0., 0., .2),
    "pitch_forward_45": (0., 45., .2),
    "pitch_backward_45": (0., -45., .2),
    "front_down_90": (0., 90., 1.),
    "back_down_90": (0., -90., 1.),
    "left_side_90": (90., 0., 1.),
    "right_side_90": (-90., 0., 1.),
}
METHODS = ("direct_rl", "prepare_rl", "fold_plant_rl", "passive", "rollover_positive", "rollover_negative", "recovery_auto", "side_swing", "stand_rl")
PHASES = ("ZERO", "FOLD", "PLANT", "PREPARE", "BLEND", "RL", "FAILED", "ORBIT", "THRUST", "SIDE_SWING", "WAIT_GROUND", "CAPTURE")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ActivationBench:
    def __init__(self, args, contract, manifest, prior):
        import numpy as np
        import torch
        import trimesh
        import onnxruntime as ort
        from scipy.spatial.transform import Rotation
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.env import ChassisEnv
        from analyze_v5_spring_limits import balanced_standing_pose
        from v5_mechanism import fk

        self.args, self.torch, self.np = args, torch, np
        self.bundle = args.bundle.resolve()
        self.spec = json.loads((self.bundle / "model_spec.json").read_text())
        self.rows = [(pose, method, rep) for pose in args.poses for method in args.methods for rep in range(args.repeats)]
        cases = [{"name": f"{p}__{m}__{r}", "command": [0., 0., .305], "terrain": "flat",
                  "skill": {"kind": "constant", "command": [0., 0., .305], "mode": 0}}
                 for p, m, r in self.rows]
        cfg = deepcopy(contract)
        cfg["physics_dt"] = contract["physics_dt"] / args.physics_substeps
        cfg.update(asset_directory=str(self.bundle), asset_manifest_sha256=digest(self.bundle / "manifest.json"),
                   auto_reset=False, record_diagnostics=False,
                   monitor_applied_effort=True, flat_triangle_mesh=False, flat_half_length_m=44., flat_floor_width_m=8.,
                   evaluation_exact_cases=True, evaluation_long_corridors=True,
                   skill_specs={c["name"]: c["skill"] for c in cases},
                   evaluation={"cases": cases, "seed": 190619},
                   signal_perturbations={"enabled": False}, usb_transport={"enabled": False})
        cfg.pop("performance_curriculum", None)
        self.env = ChassisEnv(cfg, manifest, prior, ROOT, stage_name=cfg["enabled_stages"][0],
                              num_envs=len(cases), device=args.device, seed=190619)
        self.count, self.dt = len(cases), contract["physics_dt"]
        self.policy_steps = round(cfg["policy_dt"] / self.dt)
        self.nominal = self.env.v5.nominal
        self.legs = list(self.env.v5.LEGS)
        self.wheels = list(self.env.v5.WHEELS)
        fold = balanced_standing_pose(self.spec, 50.)
        self.fold = self.nominal.new_tensor([fold["joint_positions"][n] for n in manifest["control_joint_names"]])
        thrust_angle = args.thrust_knee_deg or min(75., manifest["knee_inner_limits_deg"][1] - 3.)
        if not manifest["knee_inner_limits_deg"][0] < thrust_angle < manifest["knee_inner_limits_deg"][1]:
            raise ValueError("Thrust knee angle is outside the asset's mechanical range")
        thrust = balanced_standing_pose(self.spec, thrust_angle)
        self.thrust = self.nominal.new_tensor([thrust["joint_positions"][n] for n in manifest["control_joint_names"]])
        rollover_angle = thrust_angle if args.rollover_knee_deg is None else args.rollover_knee_deg
        if not manifest["knee_inner_limits_deg"][0] < rollover_angle < manifest["knee_inner_limits_deg"][1]:
            raise ValueError("Rollover knee angle is outside the asset's mechanical range")
        rollover = balanced_standing_pose(self.spec, rollover_angle)
        self.rollover_thrust = self.nominal.new_tensor([rollover["joint_positions"][n] for n in manifest["control_joint_names"]])
        self.stand_target = self.nominal.clone()
        self.upright_target = self.nominal.clone()
        self.support_target = self.thrust.clone()
        self.upright_support_target = self.thrust.clone()
        self.capture_support_target = self.thrust.clone()
        if args.balanced_prepare:
            self.stand_target, self.upright_target = self._balanced_prepare_targets(.305)
            if args.gravity_aligned_prepare:
                self.support_target, self.upright_support_target = self._balanced_prepare_targets(args.support_height)
                self.capture_support_target, _ = self._balanced_prepare_targets(
                    args.capture_support_height or args.support_height)
        joints = {j["name"]: j for j in self.spec["joints"]}
        axes = [np.asarray(joints[manifest["control_joint_names"][i]]["origin"])[:3, :3]
                @ np.asarray(joints[manifest["control_joint_names"][i]]["axis"]) for i in self.legs]
        if any(abs(axis[1]) < .99 for axis in axes):
            raise ValueError("Plant-angle hypothesis requires the measured sagittal root axes")
        self.axis_sign = self.nominal.new_tensor([axis[1] for axis in axes])
        nominal_frames = fk(self.spec, manifest["nominal_joint_pos"])
        wheel_axes = [nominal_frames[joints[manifest["control_joint_names"][i]]["child"]][:3, :3]
                      @ np.asarray(joints[manifest["control_joint_names"][i]]["axis"]) for i in self.wheels]
        self.wheel_axis_sign = self.nominal.new_tensor([axis[1] for axis in wheel_axes])
        wheel_bodies = {b["name"]: b for b in self.spec["bodies"]}
        self.wheel_radius = self.nominal.new_tensor([wheel_bodies[n]["collisions"][0]["radius"]
                                                     for n in ("L_link3", "R_link3")])
        self.wheel_axle_local = self.nominal.new_tensor(np.asarray([
            np.asarray(wheel_bodies[n]["collisions"][0]["origin"])[:3, 2] for n in ("L_link3", "R_link3")]))
        self.spring_maps = []
        if args.spring_compensation:
            from build_v5_closedchain import POSE_COORDINATES, solve_pose
            samples = []
            pose = self.spec["nominal_joint_pos"]
            for angle in np.linspace(max(36., manifest["knee_inner_limits_deg"][0]), manifest["knee_inner_limits_deg"][1], 45):
                knee = math.radians(angle) - (math.pi - 2.3573)
                pose = solve_pose(self.spec, dict(zip(POSE_COORDINATES, [.42, knee, 0., -.42, -knee, 0.])), pose)
                samples.append([pose["LL_joint1"] - pose["L_joint1"], pose["L_spring_slide"],
                                pose["RR_joint1"] - pose["R_joint1"], pose["R_spring_slide"]])
            samples = np.asarray(samples)
            for side in range(2):
                difference, extension = samples[:, 2 * side], samples[:, 2 * side + 1]
                derivative = np.gradient(extension, difference)
                order = np.argsort(difference)
                self.spring_maps.append((self.nominal.new_tensor(difference[order]),
                                         self.nominal.new_tensor(derivative[order])))

        # Support-map height keeps the initial collision hulls above the floor.
        meshes = {c["file"]: trimesh.load(self.bundle / c["file"], force="mesh", process=False).vertices
                  for b in self.spec["bodies"] for c in b["collisions"] if c["type"] == "mesh"}
        root_poses, joint_poses = [], []
        for pose, _, rep in self.rows:
            roll, pitch, _ = POSES[pose]
            rng = np.random.default_rng(190619 + rep)
            # Identical initial perturbations for every controller of the same pose.
            rotation = Rotation.from_euler("xyz", np.deg2rad([roll, pitch, 0.]) + rng.normal(0., .003, 3))
            q = fold["joint_positions"] if pose.startswith("crouched") else manifest["nominal_joint_pos"]
            frames = fk(self.spec, q)
            minimum = math.inf
            for body in self.spec["bodies"]:
                for collision in body["collisions"]:
                    transform = frames[body["name"]] @ np.asarray(collision["origin"])
                    rot = rotation.as_matrix() @ transform[:3, :3]
                    pos = rotation.apply(transform[:3, 3])
                    if collision["type"] == "mesh":
                        low = (meshes[collision["file"]] @ rot[2]).min() + pos[2]
                    else:
                        extent = collision["radius"] * math.sqrt(max(0., 1 - rot[2, 2] ** 2))
                        extent += collision["length"] * .5 * abs(rot[2, 2])
                        low = pos[2] - extent
                    minimum = min(minimum, float(low))
            root_poses.append([0., 0., -minimum + .003, *rotation.as_quat()])
            joint_poses.append([q[n] for n in self.env.robot.joint_names])
        self.initial_root = self.nominal.new_tensor(root_poses)
        self.initial_root[:, :3] += self.env.origins
        self.initial_joint = self.nominal.new_tensor(joint_poses)
        self.release = self.nominal.new_tensor([args.fallen_release_seconds if p.endswith("_90") else POSES[p][2]
                                                for p, _, _ in self.rows])
        self.requested_method = torch.tensor([METHODS.index(m) for _, m, _ in self.rows], device=args.device)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        self.session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
        self.sensor_only = args.sensor_only
        if self.sensor_only:
            from wheeled_tasks.chassis.recovery_observer import MotorEncoderAlignment, RecoveryObserver, SimulatedImu
            reference = json.loads((ROOT / "docs/evidence/v5_self_righting_reference_20260926.json").read_text())
            self.observer = RecoveryObserver(self.spec, reference, args.device, self.count, self.dt)
            self.simulated_imu = SimulatedImu(args.device, self.count, self.dt, noise=args.sensor_noise)
            zero_pose = balanced_standing_pose(self.spec, 110.)["joint_positions"]
            self.simulated_encoder_zero = [zero_pose[name] for name in
                                           ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")]
            self.encoder_alignment = MotorEncoderAlignment(
                self.simulated_encoder_zero, [1., 1., 1., 1.], args.device, self.count, self.dt,
                position_resolution=.0004 if args.sensor_noise else None,
                velocity_resolution=.022 if args.sensor_noise else None)
        self.writer = SummaryWriter(str(args.output / "tensorboard"))
        self.run_index = -1
        self.reset()

    def _balanced_prepare_targets(self, height):
        from scipy.optimize import brentq
        from analyze_v5_spring_limits import balanced_standing_pose
        from compare_v5_spring_load import prepare_trial

        manifest = self.env.manifest
        upper = manifest["knee_inner_limits_deg"][1] - 1.
        angle = brentq(lambda a: balanced_standing_pose(self.spec, a)["base_frame_height_m"] - height, 50., upper)
        fit = json.loads((self.bundle / "fit_10mpa.json").read_text())
        standing = prepare_trial(self.spec, manifest, fit, angle, allow_reserve_extrapolation=True)
        pose = self.nominal.new_tensor([standing["joint_positions"][n] for n in manifest["control_joint_names"]])
        feedforward = self.nominal.new_tensor(standing["feedforward_nm"][1])[self.legs]
        targets = []
        for stiffness in (self.args.prepare_kp, self.args.stand_kp):
            target = pose.clone()
            target[self.legs] += feedforward / stiffness
            targets.append(target)
        return targets

    def reset(self):
        torch, env = self.torch, self.env
        ids = torch.arange(self.count, device=env.device)
        env.robot.write_root_link_pose_to_sim_index(root_pose=self.initial_root, env_ids=ids)
        env.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(self.count, 6, device=env.device), env_ids=ids)
        env.robot.write_joint_position_to_sim_index(position=self.initial_joint, env_ids=ids)
        env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(self.initial_joint), env_ids=ids)
        env.robot.reset(ids)
        env.robot.update(self.dt)
        if self.sensor_only:
            self.observer.reset()
            self.simulated_imu.reset(env.robot.data.root_com_lin_vel_w.torch)
            self.encoder_alignment.reset()
        self.last_raw_q = env.robot.data.joint_pos.torch[:, env.ids].clone()
        self.continuous_q = self.last_raw_q.clone()
        gravity = env.robot.data.projected_gravity_b.torch
        self.last_pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        self.continuous_pitch = self.last_pitch.clone()
        self.time, self.ticks = 0., 0
        self.method = self.requested_method.clone()
        self.phase = torch.zeros(self.count, dtype=torch.long, device=env.device)
        self.phase_start = torch.zeros(self.count, device=env.device)
        self.previous_action = torch.zeros(self.count, 6, device=env.device)
        self.desired = env.robot.data.joint_pos.torch[:, env.ids][:, self.legs].clone()
        self.rl_legs, self.rl_wheels = self.desired.clone(), torch.zeros(self.count, 2, device=env.device)
        self.handover_time = torch.full((self.count,), -1., device=env.device)
        self.enable_tilt = torch.zeros(self.count, device=env.device)
        self.enable_height = torch.zeros(self.count, device=env.device)
        self.good_duration = torch.zeros(self.count, device=env.device)
        self.ready_duration = torch.zeros_like(self.good_duration)
        self.plant_contact_duration = torch.zeros_like(self.good_duration)
        self.landing_duration = torch.zeros_like(self.good_duration)
        self.inverted_duration = torch.zeros_like(self.good_duration)
        self.recovery_reroutes = torch.zeros(self.count, dtype=torch.long, device=env.device)
        self.orbit_start = self.desired.clone()
        self.orbit_angle = torch.zeros_like(self.good_duration)
        self.plant_pitch = torch.zeros_like(self.good_duration)
        self.plant_orbit = torch.zeros_like(self.good_duration)
        self.side_start = self.desired.clone()
        self.side_direction = torch.ones_like(self.desired)
        self.side_converted_time = torch.full_like(self.good_duration, -1.)
        self.side_started_lateral = torch.zeros_like(self.good_duration, dtype=torch.bool)
        self.side_attempts = torch.zeros(self.count, dtype=torch.long, device=env.device)
        self.max_gap = torch.zeros_like(self.good_duration)
        self.peak_torque = torch.zeros(self.count, 6, device=env.device)
        self.mech_failed = torch.zeros(self.count, dtype=torch.bool, device=env.device)
        self.failure_reason = [None] * self.count
        self.success = torch.zeros_like(self.mech_failed)
        self.traces = []
        self.diagnostics = []
        self.sensor_traces = []
        self.run_index += 1

    @staticmethod
    def unwrap_angle(previous, previous_raw, current_raw):
        import torch
        difference = current_raw - previous_raw
        return previous + torch.atan2(difference.sin(), difference.cos())

    def update_motor_angles(self, raw):
        self.continuous_q.copy_(self.unwrap_angle(self.continuous_q, self.last_raw_q, raw))
        self.last_raw_q.copy_(raw)
        return self.continuous_q

    @staticmethod
    def recovery_handover_ready(gravity, omega, height, wheel_normal, velocity, body_force):
        return ((gravity[:, 2] < -math.cos(math.radians(8.)))
                & (omega.norm(dim=-1) < .75) & (height > .27) & (height < .36)
                & (wheel_normal.amin(-1) > 5.) & (body_force < 5.)
                & (velocity[:, :2].norm(dim=-1) < .25) & (velocity[:, 2].abs() < .15))

    @staticmethod
    def select_automatic_method(gravity, height):
        import torch
        tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        method = torch.where(tilt < math.radians(70.), 2, torch.where(pitch >= 0., 4, 5))
        near_upright = (tilt < math.radians(15.)) & (height > .20)
        method[near_upright] = METHODS.index("stand_rl")
        side = (tilt >= math.radians(70.)) & (gravity[:, 1].abs() > .7)
        method[side] = METHODS.index("side_swing")
        return method

    @staticmethod
    def support_alignment_ready(phase, tilt, wheel_normal, capture_tilt_limit_deg=65.):
        limit = tilt.new_full(tilt.shape, math.radians(65.))
        limit[phase == 11] = math.radians(capture_tilt_limit_deg)
        return (((phase == 3) | (phase == 4) | (phase == 11)) & (tilt < limit)
                & (wheel_normal.amin(-1) > 2.))

    @staticmethod
    def sensor_alignment_ready(phase, tilt, contact_candidate, capture_tilt_limit_deg=65.):
        limit = tilt.new_full(tilt.shape, math.radians(65.))
        limit[phase == 11] = math.radians(capture_tilt_limit_deg)
        return (((phase == 3) | (phase == 4) | (phase == 11)) & contact_candidate & (tilt < limit))

    @staticmethod
    def landing_ready(omega, leg_velocity, wheel_normal, body_force):
        supported = (wheel_normal.amin(-1) > 3.) | (body_force > 10.)
        return supported & (omega.norm(dim=-1) < 1.5) & (leg_velocity.abs().amax(-1) < 2.)

    @staticmethod
    def leg_reference_delta(goal, reference):
        # The hip selects the winding for both root axes. The auxiliary root
        # must follow it even when its own shortest arc crosses pi; averaging
        # the axes can also choose the opposite hip sweep for an extended leg.
        pairs = (goal - reference).reshape(-1, 2, 2)
        turns = (pairs[..., :1] / math.tau).round()
        return (pairs - turns * math.tau).reshape_as(reference)

    @staticmethod
    def pitch_reaction_effort(omega_y, axis_sign, damping):
        # The reaction on the body is -sum(axis * torque). A common torque on
        # each leg's root pair damps pitch without adding a relative-knee torque.
        return damping * omega_y[:, None] / axis_sign

    @staticmethod
    def inverted_prepare_reroute(phase, method, tilt, omega_norm, inverted_duration, reroutes, elapsed, budget):
        return (((phase == 3) | (phase == 11))
                & ((method == METHODS.index("stand_rl")) | (method == METHODS.index("fold_plant_rl")))
                & (tilt > math.radians(100.)) & (omega_norm < 3.)
                & (inverted_duration >= .15) & (reroutes == 0)
                & (elapsed < budget - 3.))

    def step(self):
        import warp as wp
        from wheeled_tasks.chassis.scut_observation import build_scut35, CONTROL_FROM_POLICY
        torch, env = self.torch, self.env
        if self.sensor_only:
            data = env.robot.data
            omega = data.root_com_ang_vel_b.torch
            gravity = data.projected_gravity_b.torch
            gravity, omega = self.simulated_imu.measure_attitude(gravity, omega)
            q, dq, _ = self.encoder_alignment.sample(
                data.joint_pos.torch[:, env.ids], data.joint_vel.torch[:, env.ids])
            orientation = data.root_link_pose_w.torch[:, 3:7]
            acceleration = self.simulated_imu.sample(data.root_com_lin_vel_w.torch, orientation)
            q = self.update_motor_angles(q)
            observed = self.observer.update(q, dq, gravity, omega, orientation, acceleration)
            height = observed.height_if_wheels_grounded
            velocity = torch.stack((observed.wheel_speed_estimate,
                                    torch.zeros_like(height), observed.height_rate), dim=-1)
        else:
            velocity, omega, gravity, height, q, dq, local = env.state()
            q = self.update_motor_angles(q)
        tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
        if self.sensor_only:
            inverted = (((self.phase == 3) | (self.phase == 11))
                        & (tilt > math.radians(100.)) & (omega.norm(dim=-1) < 3.))
            self.inverted_duration = torch.where(inverted, self.inverted_duration + self.dt, 0.)
            reroute = self.inverted_prepare_reroute(
                self.phase, self.method, tilt, omega.norm(dim=-1), self.inverted_duration,
                self.recovery_reroutes, self.time - self.release, self.args.recovery_budget)
            self.method[reroute] = self.select_automatic_method(gravity[reroute], height[reroute])
            self.phase[reroute], self.phase_start[reroute] = 1, self.time
            self.desired[reroute] = q[reroute][:, self.legs]
            self.ready_duration[reroute] = 0.
            self.recovery_reroutes[reroute] += 1
        enabling = (self.phase == 0) & (self.time >= self.release) & (self.method != 3)
        automatic = enabling & (self.requested_method == 6)
        self.method[automatic] = self.select_automatic_method(gravity[automatic], height[automatic])
        standing_method = self.method == METHODS.index("stand_rl")
        orbit_method = (self.method == 4) | (self.method == 5) | (self.method == 7)
        model_prepare = orbit_method | standing_method
        candidate = (self.method == 2) | model_prepare
        self.enable_tilt[enabling] = tilt[enabling]
        self.enable_height[enabling] = height[enabling]
        self.desired[enabling] = q[enabling][:, self.legs]
        self.phase[enabling & (self.method == 0)] = 5
        self.phase[enabling & (self.method == 1)] = 3
        self.phase[enabling & candidate] = 1
        already_upright = enabling & candidate & (tilt < math.radians(15.)) & (height > .27)
        already_upright |= enabling & standing_method & (tilt < math.radians(15.)) & (height > .20)
        self.phase[already_upright] = 3
        self.phase_start[enabling] = self.time
        self.handover_time[enabling & (self.method == 0)] = self.time
        # Abruptly braking freely rotating legs transfers their momentum to
        # the airborne body. Wait for a settled contact before starting PD.
        if self.sensor_only:
            airborne_start = (enabling & standing_method & ~observed.supported
                              & (acceleration.norm(dim=-1) < 9.))
        else:
            airborne_start = (enabling & standing_method & (self.release > 0.)
                              & (env.contact_force[:, env.wheel_ids, 2].amin(-1) < 2.))
        self.phase[airborne_start] = 10
        waiting = self.phase == 10
        if self.sensor_only:
            landing = (waiting & (observed.contact_candidate | observed.body_contact_suspected)
                       & (omega.norm(dim=-1) < 1.) & (dq[:, self.legs].abs().amax(-1) < 2.)
                       & (dq[:, self.wheels].abs().amax(-1) < 10.)
                       & (acceleration.norm(dim=-1) > 7.)
                       & (acceleration.norm(dim=-1) < 12.))
        else:
            landing = waiting & self.landing_ready(
                omega, dq[:, self.legs], env.contact_force[:, env.wheel_ids, 2],
                env.contact_force[:, env.nonwheel_ids].norm(dim=-1).amax(-1))
        self.landing_duration = torch.where(landing, self.landing_duration + self.dt,
                                            torch.zeros_like(self.landing_duration))
        landed = waiting & (self.landing_duration >= (.12 if self.sensor_only else .04))
        self.phase[landed], self.phase_start[landed] = 3, self.time
        self.desired[landed] = q[landed][:, self.legs]
        if self.sensor_only:
            shell_low = landed & (height < .22)
            self.method[shell_low] = METHODS.index("fold_plant_rl")
            self.phase[shell_low] = 1

        goal = self.nominal[self.legs].expand(self.count, -1).clone()
        goal[orbit_method] = self.stand_target[self.legs]
        goal[standing_method] = self.upright_target[self.legs]
        folding = self.phase == 1
        goal[folding] = self.fold[self.legs]
        # Experimental sagittal support placement, not copied SCUT target angles.
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        self.continuous_pitch.copy_(self.unwrap_angle(self.continuous_pitch, self.last_pitch, pitch))
        self.last_pitch.copy_(pitch)
        if self.args.gravity_aligned_prepare:
            # World-aligned support assumes a ground reaction. In flight the
            # same feedback spins the body through the leg reaction torque.
            if self.sensor_only:
                alignment_contact = ((model_prepare & observed.contact_candidate)
                                     | ((self.method == 2) & observed.alignment_candidate))
                aligning = self.sensor_alignment_ready(
                    self.phase, tilt, alignment_contact, self.args.capture_support_deg)
            else:
                aligning = model_prepare & self.support_alignment_ready(
                    self.phase, tilt, env.contact_force[:, env.wheel_ids, 2], self.args.capture_support_deg)
            extension = ((tilt[aligning] - math.radians(12.)) / math.radians(43.)).clamp(0., 1.)
            stand = torch.where(standing_method[:, None], self.upright_target[self.legs], self.stand_target[self.legs])
            extended = torch.where(standing_method[:, None], self.upright_support_target[self.legs], self.support_target[self.legs])
            extended = torch.where((self.phase == 11)[:, None], self.capture_support_target[self.legs], extended)
            target = stand[aligning] + extension[:, None] * (extended[aligning] - stand[aligning])
            goal[aligning] = target - pitch[aligning, None] / self.axis_sign
        sagittal = gravity[:, 1].abs() < .5
        planting = self.phase == 2
        # A positive body pitch needs a forward support point: r_x * F_z gives
        # a negative pitch moment. Positive leg rotation about Y moves it back.
        offset = (-pitch - math.radians(self.args.plant_angle_deg) * pitch.sign()).clamp(-2.2, 2.2)
        goal[planting] = self.fold[self.legs]
        goal[planting & sagittal] += offset[planting & sagittal, None] / self.axis_sign
        orbiting = self.phase == 7
        direction = torch.where(self.method == 4, 1., -1.)
        self.orbit_angle[orbiting] += self.args.orbit_speed * self.dt
        goal[orbiting] = (self.orbit_start[orbiting]
                         + (direction[orbiting] * self.orbit_angle[orbiting])[:, None] / self.axis_sign)
        thrusting = self.phase == 8
        pitch_change = self.continuous_pitch - self.plant_pitch
        planted_offset = self.plant_orbit - pitch_change
        goal[thrusting] = self.rollover_thrust[self.legs] + planted_offset[thrusting, None] / self.axis_sign
        side_swinging = self.phase == 9
        side_duration = math.radians(self.args.side_angle_deg) / self.args.side_speed
        side_elapsed = self.time - self.phase_start
        stroke = torch.where(side_elapsed < side_duration,
                             self.args.side_speed * side_elapsed,
                             math.radians(self.args.side_angle_deg) - self.args.side_speed * (side_elapsed - side_duration - .08).clamp_min(0.))
        stroke = stroke.clamp(0., math.radians(self.args.side_angle_deg))
        goal[side_swinging] = self.side_start[side_swinging] + self.side_direction[side_swinging] * stroke[side_swinging, None] / self.axis_sign
        delta = self.leg_reference_delta(goal, self.desired)
        # An intentional revolution must not be folded back to a shortest angle.
        directed = orbiting | thrusting | side_swinging
        delta[directed] = goal[directed] - self.desired[directed]
        speed = torch.where(self.phase == 3, self.release.new_tensor(self.args.push_speed),
                            self.release.new_tensor(self.args.prepare_speed))
        speed[orbiting] = self.args.orbit_speed
        speed[thrusting] = self.args.push_speed
        speed[thrusting | (self.phase == 11)] = self.args.rollover_speed or self.args.push_speed
        if self.args.capture_speed is not None:
            speed[self.phase == 11] = self.args.capture_speed
        speed[side_swinging] = self.args.side_speed
        speed[standing_method] = self.args.stand_speed
        reference_step = torch.maximum(torch.minimum(delta, speed[:, None] * self.dt), -speed[:, None] * self.dt)
        preparing_reference = standing_method & ((self.phase == 3) | (self.phase == 4))
        scale = (speed * self.dt / delta.abs().amax(-1).clamp_min(1e-6)).clamp_max(1.)
        # A shared interpolation fraction preserves the intended knee change.
        reference_step[preparing_reference] = delta[preparing_reference] * scale[preparing_reference, None]
        self.desired += reference_step
        reference_ready = delta.abs().amax(-1) < .02
        elapsed = self.time - self.phase_start
        folded = folding & reference_ready & (elapsed >= .3)
        self.phase[folded & ~orbit_method], self.phase_start[folded & ~orbit_method] = 2, self.time
        tracked = torch.atan2((self.desired - q[:, self.legs]).sin(), (self.desired - q[:, self.legs]).cos()).abs().amax(-1) < .2
        orbit_starting = folded & orbit_method & (self.method != 7) & tracked
        choose_direction = orbit_starting & (self.requested_method == 6)
        self.method[choose_direction] = torch.where(pitch[choose_direction] >= 0., 4, 5)
        self.phase[orbit_starting], self.phase_start[orbit_starting] = 7, self.time
        self.orbit_start[orbit_starting] = self.desired[orbit_starting]
        self.orbit_angle[orbit_starting] = 0.
        side_starting = folded & (self.method == 7) & tracked
        self.phase[side_starting], self.phase_start[side_starting] = 9, self.time
        self.side_start[side_starting] = self.desired[side_starting]
        self.side_started_lateral[side_starting] = gravity[side_starting, 1].abs() > .7
        self.side_attempts[side_starting] += 1
        if self.sensor_only:
            lower_left = observed.wheel_height_difference < 0
            ambiguous_side = observed.wheel_height_difference.abs() < .01
        else:
            wheel_heights = env.wheel_centers()[:, :, 2]
            lower_left = wheel_heights[:, 0] < wheel_heights[:, 1]
            ambiguous_side = (wheel_heights[:, 0] - wheel_heights[:, 1]).abs() < .01
        if self.args.side_first_leg != "auto":
            lower_left[ambiguous_side] = self.args.side_first_leg == "left"
        for leg, columns in ((0, [0, 1]), (1, [2, 3])):
            upper = ~lower_left if leg == 0 else lower_left
            sign = torch.where(upper, 1., -1.) if self.args.side_pattern == "opposed" else torch.ones_like(pitch)
            if self.args.side_pattern == "upper_only":
                sign = upper.float()
            if self.args.side_pattern == "lower_only":
                sign = (~upper).float()
            for column in columns:
                self.side_direction[side_starting, column] = sign[side_starting] * self.args.side_direction
            lengthen = side_starting & (upper if self.args.side_long_leg == "upper" else ~upper)
            if self.args.side_long_leg != "none":
                for column in columns:
                    self.side_start[lengthen, column] += self.thrust[self.legs[column]] - self.fold[self.legs[column]]
        plane_changed = torch.where(self.side_started_lateral, gravity[:, 1].abs() < .65,
                                    (gravity[:, 1].abs() > .7) | (tilt < math.radians(70.)))
        side_converted = side_swinging & (side_elapsed > .05) & plane_changed
        self.side_converted_time[side_converted] = self.time
        self.method[side_converted] = torch.where(pitch[side_converted] >= 0., 4, 5)
        self.phase[side_converted], self.phase_start[side_converted] = 1, self.time
        converted_to_side = side_converted & (gravity[:, 1].abs() > .7) & (self.side_attempts < 2)
        self.method[converted_to_side] = 7
        side_exhausted = side_swinging & ~side_converted & (side_elapsed > 2 * side_duration + .4)
        self.phase[side_exhausted] = 6
        for i in side_exhausted.nonzero(as_tuple=False).flatten().tolist():
            self.failure_reason[i] = "side_swing_no_reorientation"
        loaded = (observed.contact_candidate if self.sensor_only else
                  env.contact_force[:, env.wheel_ids].norm(dim=-1).amin(-1) > 3.)
        self.plant_contact_duration = torch.where(planting & loaded, self.plant_contact_duration + self.dt,
                                                  torch.zeros_like(self.plant_contact_duration))
        planted = planting & (delta.abs().amax(-1) < .15) & (self.plant_contact_duration >= .06)
        self.phase[planted], self.phase_start[planted] = 3, self.time
        no_support = planting & ~planted & (elapsed >= 2.5)
        self.phase[no_support] = 6
        for i in no_support.nonzero(as_tuple=False).flatten().tolist():
            self.failure_reason[i] = "plant_timeout_no_support"
        orbit_contact = (orbiting & loaded & (tilt < math.radians(145.)) & (height > .12)
                          & (self.orbit_angle > .2))
        if self.sensor_only:
            orbit_contact &= self.orbit_angle > 3.7
        if self.args.thrust_on_contact:
            self.plant_orbit[orbit_contact] = direction[orbit_contact] * self.orbit_angle[orbit_contact]
            self.plant_pitch[orbit_contact] = self.continuous_pitch[orbit_contact]
            self.phase[orbit_contact], self.phase_start[orbit_contact] = 8, self.time
        captured = orbiting & (tilt < math.radians(30.)) & (height > .22) & loaded
        self.phase[captured], self.phase_start[captured] = 3, self.time
        thrust_captured = (thrusting & (tilt < math.radians(self.args.capture_tilt_deg)) & (height > .24)
                           & (omega.norm(dim=-1) < 8.))
        self.phase[thrust_captured], self.phase_start[thrust_captured] = 11, self.time
        orbit_exhausted = (self.phase == 7) & ~captured & (self.orbit_angle >= math.tau * self.args.orbit_turns)
        self.phase[orbit_exhausted] = 6
        for i in orbit_exhausted.nonzero(as_tuple=False).flatten().tolist():
            self.failure_reason[i] = "orbit_completed_without_capture"
        preparing = (self.phase == 3) | (self.phase == 11)
        measured_error = torch.atan2((self.desired - q[:, self.legs]).sin(), (self.desired - q[:, self.legs]).cos()).abs().amax(-1)
        support = (observed.supported if self.sensor_only else
                   env.contact_force[:, env.wheel_ids].norm(dim=-1).amin(-1) > 2.)
        ordinary_ready = preparing & reference_ready & (measured_error < .12) & (tilt < math.radians(20))
        if self.sensor_only:
            orbit_ready = preparing & observed.settled
            ordinary_ready &= observed.settled
        else:
            orbit_ready = preparing & self.recovery_handover_ready(
                gravity, omega, height, env.contact_force[:, env.wheel_ids, 2], velocity,
                env.contact_force[:, env.nonwheel_ids].norm(dim=-1).amax(-1))
        ready = (torch.where(model_prepare, orbit_ready, ordinary_ready)
                 & (omega.norm(dim=-1) < 1.5) & support & (height > .27))
        self.ready_duration = torch.where(ready, self.ready_duration + self.dt, torch.zeros_like(self.ready_duration))
        script_ready = (reference_ready & (elapsed >= .2) if self.args.ungated_script_handover
                        else self.ready_duration >= .1)
        handover = preparing & (((self.method == 1) & reference_ready & (elapsed >= .2))
                                | ((self.method == 2) & script_ready)
                                | (model_prepare & (self.ready_duration >= .1)))
        self.phase[handover], self.phase_start[handover] = 4, self.time
        self.handover_time[handover] = self.time
        self.previous_action[handover] = 0.
        blending = self.phase == 4
        blend = ((self.time - self.phase_start) / .2).clamp(0., 1.)
        self.phase[blending & (blend >= 1.)] = 5
        timed_out = (((self.phase >= 1) & (self.phase <= 3)) | (self.phase >= 7)) & (
            self.time - self.release > self.args.recovery_budget)
        self.phase[timed_out] = 6
        for i in timed_out.nonzero(as_tuple=False).flatten().tolist():
            self.failure_reason[i] = "prepare_timeout"
        lost_upright = candidate & ((self.phase == 4) | (self.phase == 5)) & (tilt > math.radians(45.))
        self.phase[lost_upright] = 6
        for i in lost_upright.nonzero(as_tuple=False).flatten().tolist():
            self.failure_reason[i] = "lost_upright_during_takeover"

        if self.ticks % self.policy_steps == 0:
            commands = q.new_tensor([0., 0., .305]).expand(self.count, -1)
            zeros = q.new_zeros(self.count)
            observation = build_scut35(omega, gravity, commands, q, dq, self.previous_action,
                                       self.nominal, zeros.bool(), zeros, zeros)
            raw = self.session.run(None, {"obs": observation.cpu().numpy()})[0]
            action = torch.as_tensor(raw, device=env.device)[:, CONTROL_FROM_POLICY]
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError("Nonfinite ONNX action")
            self.rl_legs, self.rl_wheels, clipped = env.v5.decode(action, q)
            active = (self.phase == 4) | (self.phase == 5)
            self.previous_action[active] = clipped[active]
        tau_rl = env.v5.motor_efforts(q, dq, self.rl_legs, self.rl_wheels)
        tau_prepare = env.v5.motor_efforts(q, dq, self.desired, torch.zeros_like(self.rl_wheels))
        tau_prepare[:, self.legs] = (self.args.prepare_kp * (self.desired - q[:, self.legs])
                                    - self.args.prepare_kd * dq[:, self.legs]).clamp(-40., 40.)
        stand_effort = (self.args.stand_kp * (self.desired - q[:, self.legs])
                        - self.args.stand_kd * dq[:, self.legs]).clamp(-40., 40.)
        for column, joint in enumerate(self.legs):
            tau_prepare[standing_method, joint] = stand_effort[standing_method, column]
        capture_effort = self.pitch_reaction_effort(omega[:, 1], self.axis_sign, self.args.capture_pitch_damping)
        for column, joint in enumerate(self.legs):
            tau_prepare[self.phase == 11, joint] += capture_effort[self.phase == 11, column]
        tau_prepare[:, self.legs] = tau_prepare[:, self.legs].clamp(-40., 40.)
        if self.spring_maps:
            spring_positions = (observed.spring_positions if self.sensor_only else
                                env.robot.data.joint_pos.torch[:, env.spring_ids])
            force = env.v5.spring_efforts(spring_positions)
            for side, (hip, driven) in enumerate(((0, 1), (3, 4))):
                knots, derivative = self.spring_maps[side]
                relative = torch.atan2((q[:, driven] - q[:, hip]).sin(), (q[:, driven] - q[:, hip]).cos()).contiguous()
                upper = torch.searchsorted(knots, relative).clamp(1, len(knots) - 1)
                lower = upper - 1
                alpha = ((relative - knots[lower]) / (knots[upper] - knots[lower])).clamp(0., 1.)
                compensation = (derivative[lower] + alpha * (derivative[upper] - derivative[lower])) * force[:, side]
                compensation = torch.where(orbit_method, compensation, 0.)
                compensation = torch.where(self.phase == 8, 0., compensation)
                if self.args.balanced_prepare:
                    compensation = torch.where((self.phase == 3) | (self.phase == 4) | (self.phase == 11), 0., compensation)
                tau_prepare[:, hip] += compensation
                tau_prepare[:, driven] -= compensation
            tau_prepare[:, self.legs] = tau_prepare[:, self.legs].clamp(-40., 40.)
        tau_prepare[:, self.wheels] = 0.
        balance_angle = torch.where(model_prepare, tilt.new_tensor(math.radians(70.)), tilt.new_tensor(math.radians(55.)))
        balance_angle[self.phase == 11] = math.radians(self.args.capture_support_deg)
        balancing = candidate & (((self.phase >= 2) & (self.phase <= 4)) | (self.phase == 11)) & (tilt < balance_angle)
        pitch_effort = self.args.balance_wheel_kp * pitch + 1.5 * omega[:, 1]
        wheel_torque = pitch_effort[:, None] * self.wheel_axis_sign - .2 * dq[:, self.wheels]
        thrust_braking = self.phase == 8
        wheel_world_omega_y = (observed.wheel_world_omega_y if self.sensor_only else
                               env.robot.data.body_link_ang_vel_w.torch[:, env.wheel_ids, 1])
        wheel_torque[thrust_braking] = -.2 * wheel_world_omega_y[thrust_braking] * self.wheel_axis_sign
        if self.args.brake_on_body_contact:
            body_loaded = (observed.body_contact_suspected if self.sensor_only else
                           env.contact_force[:, env.nonwheel_ids].norm(dim=-1).amax(-1) > 10.)
            # Standing must roll the support point under the body. Pinning the
            # wheels while the shell still touches forces a large body pitch.
            braking = candidate & body_loaded & ~standing_method
            wheel_torque[braking] = -.4 * wheel_world_omega_y[braking] * self.wheel_axis_sign
        bounds = env.v5.current_motor_bounds[:, self.wheels]
        if self.args.traction_cap:
            if self.sensor_only:
                bounds = torch.minimum(bounds, bounds.new_full(bounds.shape, 1.5))
            else:
                normal = env.contact_force[:, env.wheel_ids, 2].clamp_min(0.)
                bounds = torch.minimum(bounds, .8 * .5 * normal * self.wheel_radius)
        wheel_torque = torch.maximum(torch.minimum(wheel_torque, bounds), -bounds)
        wheel_script = balancing | thrust_braking
        if self.sensor_only:
            wheel_script &= observed.contact_candidate
        tau_prepare[wheel_script, 2] = wheel_torque[wheel_script, 0]
        tau_prepare[wheel_script, 5] = wheel_torque[wheel_script, 1]
        tau = torch.zeros_like(tau_rl)
        scripted = ((self.phase >= 1) & (self.phase <= 3)) | ((self.phase >= 7) & (self.phase <= 9)) | (self.phase == 11)
        tau[scripted] = tau_prepare[scripted]
        blending = self.phase == 4
        tau[blending] = ((1 - blend[blending, None]) * tau_prepare[blending]
                         + blend[blending, None] * tau_rl[blending])
        tau[self.phase == 5] = tau_rl[self.phase == 5]
        if self.sensor_only:
            probing = (((self.phase == 3) | (self.phase == 11)) & observed.geometry_valid
                       & (height > .21) & (height < .42)
                       & (observed.wheel_height_difference.abs() < .03)
                       & (tilt < math.radians(65.)) & (omega.norm(dim=-1) < 1.5)
                       & (dq[:, self.legs].abs().amax(-1) < 2.)
                       & (dq[:, self.wheels].abs().amax(-1) < 6.))
            pulse = self.observer.probe.command(self.ticks, probing, observed.probe_confirmed)
            for index, joint in enumerate(self.wheels):
                active = pulse[:, index] != 0
                tau[active, joint] = pulse[active, index]
        tau[self.mech_failed] = 0.
        self.peak_torque = torch.maximum(self.peak_torque, tau.abs())
        effort = torch.zeros_like(env.nominal)
        effort[:, env.ids] = tau
        compression, _ = env.v5.spring_state(env.robot.data.joint_pos.torch[:, env.spring_ids],
                                            env.robot.data.joint_vel.torch[:, env.spring_ids])
        mean_force = torch.zeros_like(env.contact_force)
        gap = torch.zeros(self.count, device=env.device)
        for _ in range(self.args.physics_substeps):
            effort[:, env.spring_ids] = env.v5.spring_efforts(env.robot.data.joint_pos.torch[:, env.spring_ids])
            env.robot.set_joint_effort_target_index(target=effort)
            env.robot.write_data_to_sim()
            env.sim.step(render=False)
            env.robot.update(env.dt)
            for ids, contact_view in env.contact_views:
                matrix = wp.to_torch(contact_view.get_contact_force_matrix(dt=env.dt))
                mean_force[ids] += matrix.reshape(len(ids), env.body_count, -1, 3).sum(2)
            gap = torch.maximum(gap, env.closure_gap())
        env.contact_force.copy_(mean_force / self.args.physics_substeps)
        compression, _ = env.v5.spring_state(env.robot.data.joint_pos.torch[:, env.spring_ids],
                                            env.robot.data.joint_vel.torch[:, env.spring_ids])
        self.max_gap = torch.maximum(self.max_gap, gap)
        knee = env.robot.data.joint_pos.torch[:, env.knee_ids]
        bounds = env.v5.knee_bounds
        bad_knee = ((knee < bounds[:, 0] - .03) | (knee > bounds[:, 1] + .03)).any(-1)
        bad_spring = ((compression < -.001) | (compression > env.v5.stroke + .001)).any(-1)
        position = env.robot.data.root_link_pose_w.torch[:, :3] - env.origins
        outside = (position[:, 0].abs() > 43.) | (position[:, 1].abs() > 3.5)
        for reason, mask in (("closure_gap", gap > .003), ("knee_limit", bad_knee), ("spring_limit", bad_spring),
                             ("left_test_area", outside)):
            for i in (mask & ~self.mech_failed).nonzero(as_tuple=False).flatten().tolist():
                self.failure_reason[i] = reason
            self.mech_failed |= mask
        self.phase[self.mech_failed] = 6
        self.time += self.dt
        self.ticks += 1
        if self.ticks % self.policy_steps == 0:
            velocity, omega, gravity, height, q, dq, local = env.state()
            q = self.update_motor_angles(q)
            tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
            support = env.contact_force[:, env.wheel_ids].norm(dim=-1).amin(-1) > 2.
            body_clear = env.contact_force[:, env.nonwheel_ids].norm(dim=-1).amax(-1) < 5.
            good = ((self.phase == 5) & ~self.mech_failed & (tilt < math.radians(10.))
                    & ((height - .305).abs() < .02) & (velocity[:, :2].norm(dim=-1) < .2)
                    & (omega.norm(dim=-1) < .5) & support & body_clear)
            self.good_duration = torch.where(good, self.good_duration + env.policy_dt, torch.zeros_like(self.good_duration))
            self.success |= self.good_duration >= 1.
            self.last = torch.cat((height[:, None], tilt[:, None], omega.norm(dim=-1, keepdim=True),
                                   local, self.phase[:, None], tau, compression), -1).cpu().numpy()
            self.traces.append(self.last)
            wheel_force = env.contact_force[:, env.wheel_ids].norm(dim=-1)
            body_force = env.contact_force[:, env.nonwheel_ids].norm(dim=-1).amax(-1)
            measured_error = torch.atan2((self.desired - q[:, self.legs]).sin(),
                                         (self.desired - q[:, self.legs]).cos()).abs().amax(-1)
            diagnostic = torch.cat((q[:, self.legs], dq[:, self.legs], self.desired, gravity, omega,
                                    velocity, wheel_force, body_force[:, None], knee,
                                    measured_error[:, None], delta.abs().amax(-1, keepdim=True),
                                    self.ready_duration[:, None], self.plant_contact_duration[:, None]), -1)
            self.diagnostics.append(diagnostic.cpu().numpy())
            if self.sensor_only:
                self.sensor_traces.append(torch.cat((
                    observed.height_if_wheels_grounded[:, None],
                    observed.height_rate[:, None],
                    observed.wheel_height_difference[:, None],
                    acceleration.norm(dim=-1, keepdim=True),
                    observed.wheel_world_omega_y,
                    observed.alignment_candidate[:, None].float(),
                    observed.contact_candidate[:, None].float(),
                    observed.supported[:, None].float(),
                    observed.settled[:, None].float(),
                    observed.impact[:, None].float(),
                    observed.probe_confirmed[:, None].float(),
                ), -1).cpu().numpy())
            wheel_pose = env.robot.data.body_link_pose_w.torch[:, env.wheel_ids]
            wheel_omega = env.robot.data.body_link_ang_vel_w.torch[:, env.wheel_ids]
            center = env.wheel_centers()
            center_velocity = (env.robot.data.body_link_lin_vel_w.torch[:, env.wheel_ids]
                               + torch.linalg.cross(wheel_omega, center - wheel_pose[..., :3]))
            qv, qw = wheel_pose[..., 3:6], wheel_pose[..., 6:7]
            axle = self.wheel_axle_local.expand(self.count, -1, -1)
            cross = torch.linalg.cross(qv, axle)
            axle = axle + 2 * (qw * cross + torch.linalg.cross(qv, cross))
            normal = torch.zeros_like(axle)
            normal[..., 2] = 1.
            direction = torch.linalg.cross(axle, normal)
            rolling_valid = direction.norm(dim=-1) > .7
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            radial = normal - (normal * axle).sum(-1, keepdim=True) * axle
            contact_offset = -self.wheel_radius[None, :, None] * radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            contact_velocity = center_velocity + torch.linalg.cross(wheel_omega, contact_offset)
            rolling = self.wheel_radius * (wheel_omega * axle).sum(-1)
            motion = (center_velocity * direction).sum(-1)
            slip = (contact_velocity * direction).sum(-1)
            normal_force = env.contact_force[:, env.wheel_ids, 2].clamp_min(0.)
            wheel_diagnostic = torch.cat((normal_force, motion, rolling, slip, rolling_valid.float(), dq[:, self.wheels]), -1)
            self.diagnostics[-1] = self.np.concatenate((self.diagnostics[-1], wheel_diagnostic.cpu().numpy()), -1)
            if self.ticks % (self.policy_steps * 5) == 0:
                for i, (pose, method, rep) in enumerate(self.rows):
                    if rep:
                        continue
                    for name, value in (("height_m", height[i]), ("tilt_deg", torch.rad2deg(tilt[i])),
                                        ("phase", self.phase[i]), ("torque_peak_nm", tau[i].abs().max()),
                                        ("omega_norm_rad_s", omega[i].norm()),
                                        ("xy_speed_m_s", velocity[i, :2].norm()),
                                        ("wheel_normal_L_n", normal_force[i, 0]),
                                        ("wheel_normal_R_n", normal_force[i, 1]),
                                        ("nonwheel_force_peak_n", body_force[i])):
                        self.writer.add_scalar(f"Activation/{pose}/{method}/{name}", float(value),
                                               self.run_index * 100000 + self.ticks)

    def result(self, i):
        pose, method, rep = self.rows[i]
        return {"pose": pose, "method": method, "replica": rep,
                          "selected_controller": METHODS[int(self.method[i])],
                          **({"recovery_reroutes": int(self.recovery_reroutes[i])} if self.sensor_only else {}),
                         "side_converted_s": float(self.side_converted_time[i]),
                         "success": bool(self.success[i] and self.good_duration[i] >= 1. and not self.mech_failed[i]),
                         "reached_stable_rl": bool(self.success[i]),
                         "final_stable_seconds": float(self.good_duration[i]), "phase": PHASES[int(self.phase[i])],
                         "handover_s": float(self.handover_time[i]), "enable_tilt_deg": math.degrees(float(self.enable_tilt[i])),
                         "enable_height_m": float(self.enable_height[i]), "final_height_m": float(self.last[i, 0]),
                         "final_tilt_deg": math.degrees(float(self.last[i, 1])), "failure": self.failure_reason[i],
                         "max_closure_gap_m": float(self.max_gap[i]), "peak_torque_nm": self.peak_torque[i].tolist()}

    def report(self):
        return [self.result(i) for i in range(self.count)]


class ActivationView:
    def __init__(self, bench):
        import carb.windowing
        import omni.appwindow
        import omni.ui as ui
        from pxr import UsdLux
        self.bench, self.paused, self.restart = bench, False, False
        self.last_outcome_pause = None
        native = omni.appwindow.get_default_app_window()
        carb.windowing.acquire_windowing_interface().set_window_title(native.get_window(),
            "V5 ACTIVATION | ZERO - PREPARE - RL | 12486 ONNX")
        UsdLux.DomeLight.Define(bench.env.sim.stage, "/World/ActivationLight").CreateIntensityAttr(900.)
        self.window = ui.Window("ACTIVATION STUDY | 12486 ONNX", width=590, height=570)
        labels = [f"{p} / release {float(bench.release[i]):.1f}s / {'CONTROL: ' if m in METHODS[:2] else ''}{m} / {r}"
                  for i, (p, m, r) in enumerate(bench.rows)]
        with self.window.frame:
            with ui.VStack(spacing=8):
                limits = bench.env.manifest["knee_inner_limits_deg"]
                spring_mm = bench.spec["spring_binding"]["L_spring_slide"]["full_extension_pin_distance_m"] * 1000
                ui.Label(f"Zero torque -> scripted prepare -> RL\nKnee: {limits[0]:g}-{limits[1]:g} deg; spring pin length: {spring_mm:.1f} mm\nOnly trial resets write pose; gas springs stay active.", word_wrap=True)
                default = next((i for i, row in enumerate(bench.rows) if row[:2] == ("front_down_90", "rollover_positive")), 0)
                self.selection = ui.ComboBox(default, *labels)
                self.info = ui.Label("Starting", height=370, word_wrap=True)
                with ui.HStack(height=24):
                    self.pause_on_outcome = ui.CheckBox()
                    self.pause_on_outcome.model.set_value(True)
                    ui.Label("Pause at selected success/failure; replay to watch again")
                with ui.HStack(height=32):
                    ui.Button("Run / pause", clicked_fn=self.toggle)
                    ui.Button("Replay all initial poses", clicked_fn=self.replay)
        self.selected = -1
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().resolution = (960, 640)

    def toggle(self):
        self.paused = not self.paused

    def replay(self):
        self.restart, self.paused = True, False

    def update(self):
        index = self.selection.model.get_item_value_model().as_int
        pose, method, _ = self.bench.rows[index]
        if hasattr(self.bench, "last"):
            row = self.bench.result(index)
            outcome_key = (self.bench.run_index, index)
            if (self.pause_on_outcome.model.as_bool and self.last_outcome_pause != outcome_key
                    and (row["phase"] == "FAILED" or row["success"])):
                self.paused = True
                self.last_outcome_pause = outcome_key
            self.info.text = (f"Time: {self.bench.time:.2f} s; active: {max(0., self.bench.time - float(self.bench.release[index])):.2f} s\nPose: {pose}\nMethod: {method}\n"
                              f"Phase: {row['phase']}\nHeight: {row['final_height_m']:.3f} m\n"
                              f"Tilt: {row['final_tilt_deg']:.1f} deg\nHandover: {row['handover_s']:.2f} s\n"
                              f"Stable RL reached: {row['success']}\nFailure: {row['failure']}\n"
                              "Research comparison; arbitrary fallen recovery is not assumed.")
            if self.bench.diagnostics:
                d = self.bench.diagnostics[-1][index]
                slip = [f"{d[36 + j]:.2f}" if d[38 + j] and d[30 + j] > 2 else "air/sidewall" for j in range(2)]
                self.info.text += (f"\nWheel loads: {d[30]:.1f} / {d[31]:.1f} N"
                                   f"\nWheel rolling: {d[34]:.2f} / {d[35]:.2f} m/s"
                                   f"\nWheel center: {d[32]:.2f} / {d[33]:.2f} m/s"
                                   f"\nSlip: {slip[0]} / {slip[1]} m/s"
                                   f"\nNonwheel support peak: {d[23]:.1f} N"
                                   f"\nJoint error: {d[26]:.3f} rad"
                                   f"\nLeg torque: {[round(float(x), 1) for x in self.bench.last[index, [7, 8, 10, 11]]]} Nm"
                                   f"\nLeg velocity: {[round(float(x), 1) for x in d[4:8]]} rad/s"
                                   f"\nSelected controller: {row['selected_controller']}")
        position = self.bench.env.robot.data.root_link_pose_w.torch[index, :3].cpu().numpy()
        self.bench.env.sim.set_camera_view(position + [.9, 1.3, .65], position)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--allow-mechanical-variant", action="store_true",
                        help="Authenticate the original ONNX bundle, then explicitly evaluate a same-ABI mechanical variant")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--sensor-only", action="store_true",
                        help="Recover using only simulated IMU and motor encoders; physics truth is reserved for scoring")
    parser.add_argument("--sensor-noise", action="store_true",
                        help="Quantize encoder feedback and perturb the simulated IMU; requires --sensor-only")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--seconds", type=float, default=14.)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--poses", nargs="+", choices=list(POSES), default=list(POSES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--prepare-speed", type=float, default=1.)
    parser.add_argument("--push-speed", type=float, default=4.)
    parser.add_argument("--prepare-kp", type=float, default=60.)
    parser.add_argument("--prepare-kd", type=float, default=2.)
    parser.add_argument("--stand-speed", type=float, default=1.)
    parser.add_argument("--stand-kp", type=float, default=80.)
    parser.add_argument("--stand-kd", type=float, default=2.)
    parser.add_argument("--recovery-budget", type=float, default=8., help="Active recovery seconds after the zero-torque release")
    parser.add_argument("--brake-on-body-contact", action="store_true")
    parser.add_argument("--plant-angle-deg", type=float, default=20.)
    parser.add_argument("--balance-wheel-kp", type=float, default=8.)
    parser.add_argument("--fallen-release-seconds", type=float, default=1.)
    parser.add_argument("--spring-compensation", action="store_true")
    parser.add_argument("--traction-cap", action="store_true")
    parser.add_argument("--orbit-speed", type=float, default=4.)
    parser.add_argument("--orbit-turns", type=float, default=1.)
    parser.add_argument("--thrust-on-contact", action="store_true")
    parser.add_argument("--physics-substeps", type=int, choices=(1, 2, 4), default=1,
                        help="Refine physics while keeping PC feedback at 200Hz and policy at 50Hz")
    parser.add_argument("--capture-tilt-deg", type=float, default=65., help="Scripted balance capture, not the stricter RL handover gate")
    parser.add_argument("--balanced-prepare", action="store_true",
                        help="Use the model's balanced 0.305m posture with static torque encoded as a PD target bias")
    parser.add_argument("--gravity-aligned-prepare", action="store_true",
                        help="After rollover, align the support leg with world gravity while unloading the body")
    parser.add_argument("--thrust-knee-deg", type=float)
    parser.add_argument("--rollover-knee-deg", type=float, default=70.,
                        help="Sagittal thrust knee target, separate from the side recovery support leg")
    parser.add_argument("--rollover-speed", type=float, default=4.,
                        help="Sagittal thrust reference speed")
    parser.add_argument("--capture-support-deg", type=float, default=85.)
    parser.add_argument("--capture-support-height", type=float, default=.35,
                        help="Maximum capture support height, separate from other preparation profiles")
    parser.add_argument("--capture-speed", type=float, default=4.,
                        help="Landing-foot placement speed after thrust, independent of thrust speed")
    parser.add_argument("--capture-pitch-damping", type=float, default=2.,
                        help="Per-root-axis pitch reaction damping in Nm per rad/s during capture")
    parser.add_argument("--support-height", type=float, default=.34)
    parser.add_argument("--side-pattern", choices=("same", "opposed", "upper_only", "lower_only"), default="opposed")
    parser.add_argument("--side-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--side-long-leg", choices=("none", "upper", "lower"), default="upper")
    parser.add_argument("--side-speed", type=float, default=6.)
    parser.add_argument("--side-angle-deg", type=float, default=180.)
    parser.add_argument("--side-first-leg", choices=("auto", "left", "right"), default="auto",
                        help="Deterministic first support leg when an inverted pose has nearly equal wheel heights")
    parser.add_argument("--ungated-script-handover", action="store_true",
                        help="Diagnostic ablation: enter RL when the scripted reference finishes")
    args = parser.parse_args()
    if args.sensor_noise and not args.sensor_only:
        parser.error("--sensor-noise requires --sensor-only")
    if (not 3 <= args.seconds <= 30 or not 1 <= args.repeats <= 8 or not 0 < args.prepare_speed <= 2
            or not 0 < args.push_speed <= 6 or not 0 <= args.balance_wheel_kp <= 20
            or not -35 <= args.plant_angle_deg <= 35 or not 0 <= args.fallen_release_seconds <= 2
            or not 0 < args.orbit_speed <= 6 or not .25 <= args.orbit_turns <= 1.5
            or not 25 <= args.capture_tilt_deg <= 80 or not 20 <= args.prepare_kp <= 160
            or not .5 <= args.prepare_kd <= 8 or not 1 <= args.recovery_budget <= 10
            or not 0 < args.stand_speed <= 6 or not 20 <= args.stand_kp <= 160 or not .5 <= args.stand_kd <= 8
            or (args.rollover_speed is not None and not 0 < args.rollover_speed <= 6)
            or not 65 <= args.capture_support_deg <= 89
            or (args.capture_speed is not None and not 0 < args.capture_speed <= 6)
            or not 0 <= args.capture_pitch_damping <= 4
            or not .28 <= args.capture_support_height <= .43
            or not 1 <= args.side_speed <= 12 or not 60 <= args.side_angle_deg <= 270):
        parser.error("Use bounded duration, repetitions and prepare speed")
    from v5_policy_io import verify_bundle
    contract_path = Path(str(args.onnx) + ".contract.json")
    source_manifest_path = args.onnx.parent / "manifest.json" if args.allow_mechanical_variant else args.bundle / "manifest.json"
    contract, manifest, prior, _ = verify_bundle(args.onnx, contract_path, source_manifest_path,
                                               ROOT / "contracts/own_v40_v2.json")
    source_asset_sha = digest(source_manifest_path)
    if args.allow_mechanical_variant:
        from wheeled_tasks.chassis.evaluation import validate_cross_asset_actor
        target_manifest = json.loads((args.bundle / "manifest.json").read_text())
        validate_cross_asset_actor(contract, contract, manifest, target_manifest)
        manifest = target_manifest
    for name, expected in manifest["files_sha256"].items():
        if digest(args.bundle / name) != expected:
            raise ValueError(f"Playback asset dependency mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
              "policy_sha256": digest(args.onnx), "asset_manifest_sha256": digest(args.bundle / "manifest.json"),
              "source_asset_manifest_sha256": source_asset_sha,
              "knee_inner_limits_deg": manifest["knee_inner_limits_deg"],
              "script_sha256": digest(__file__), "source_kind": "SCUT-inspired_activation_research_not_trained_recovery",
              "control_observation": "imu_encoders" if args.sensor_only else "physics_truth",
              "prepare_speed_rad_s": args.prepare_speed, "plant_angle_deg": args.plant_angle_deg,
              "prepare_kp": args.prepare_kp, "prepare_kd": args.prepare_kd,
              "active_recovery_budget_s": args.recovery_budget, "brake_on_body_contact": args.brake_on_body_contact,
              "push_speed_rad_s": args.push_speed,
              "balance_wheel_kp_nm_rad": args.balance_wheel_kp,
              "fallen_release_seconds": args.fallen_release_seconds,
              "ungated_script_handover": args.ungated_script_handover,
              "spring_compensation": args.spring_compensation, "traction_cap": args.traction_cap,
              "orbit_speed_rad_s": args.orbit_speed, "orbit_turns_max": args.orbit_turns,
              "thrust_on_contact": args.thrust_on_contact,
               "script_capture_tilt_deg": args.capture_tilt_deg, "rollover_rl_handover_tilt_deg": 8.,
               "ordinary_script_handover_tilt_deg": 20., "control_groups_are_ungated": True,
               "model_prepare_handover": {"max_tilt_deg": 8., "max_omega_rad_s": .75,
                   "max_xy_speed_m_s": .25, "max_vertical_speed_m_s": .15,
                   "min_wheel_normal_n": 5., "max_nonwheel_force_n": 5., "dwell_s": .1},
               "capture_profile": {"rollover_knee_deg": args.rollover_knee_deg,
                   "thrust_speed_rad_s": args.rollover_speed, "placement_speed_rad_s": args.capture_speed,
                   "support_tilt_limit_deg": args.capture_support_deg,
                   "support_height_m": args.capture_support_height,
                   "pitch_damping_nm_per_rad_s_per_root": args.capture_pitch_damping},
               "phase_names": list(PHASES),
              "balanced_prepare": args.balanced_prepare,
              "gravity_aligned_prepare": args.gravity_aligned_prepare,
              "thrust_knee_deg": args.thrust_knee_deg, "support_height_m": args.support_height,
              "policy_hz": 1 / contract["policy_dt"], "feedback_hz": 1 / contract["physics_dt"],
              "physics_hz": args.physics_substeps / contract["physics_dt"],
              "state_writes_after_trial_reset": 0, "passive_gas_springs_enabled": True,
              "self_collision_enabled": manifest["self_collision_enabled"]}
    report["arguments"] = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    dependencies = ["src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/chassis/v5_control.py",
                    "src/wheeled_tasks/chassis/scut_observation.py", "src/wheeled_tasks/chassis/task.py",
                    "tools/v5_mechanism.py", "tools/analyze_v5_spring_limits.py", "scripts/compare_v5_spring_load.py"]
    report["source_dependencies_sha256"] = {name: digest(ROOT / name) for name in dependencies}
    report["trace_columns"] = ["height", "tilt", "omega_norm", "x", "y", "z", "phase",
                                *["tau_" + str(i) for i in range(6)], "compression_L", "compression_R"]
    report["diagnostic_columns"] = [*["q_" + str(i) for i in range(4)], *["dq_" + str(i) for i in range(4)],
                                     *["desired_" + str(i) for i in range(4)], "gravity_x", "gravity_y", "gravity_z",
                                     "omega_x", "omega_y", "omega_z", "velocity_x", "velocity_y", "velocity_z",
                                     "wheel_force_L", "wheel_force_R", "nonwheel_force_max", "knee_L", "knee_R",
                                     "joint_error_max", "reference_error_max", "ready_duration", "plant_contact_duration"]
    report["diagnostic_columns"] += ["wheel_normal_L", "wheel_normal_R", "wheel_center_speed_L", "wheel_center_speed_R",
                                      "wheel_rolling_speed_L", "wheel_rolling_speed_R", "wheel_slip_L", "wheel_slip_R",
                                      "rolling_valid_L", "rolling_valid_R", "wheel_dq_L", "wheel_dq_R"]
    if args.sensor_only:
        report["sensor_columns"] = ["conditional_height", "conditional_height_rate", "wheel_height_difference",
                                    "imu_specific_force_norm", "wheel_world_omega_y_L", "wheel_world_omega_y_R",
                                    "alignment_candidate", "contact_candidate", "supported", "settled", "impact",
                                    "probe_confirmed"]
    launcher, bench = None, None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": not args.gui, "device": args.device, "enable_cameras": False,
                                **({"visualizer": ["kit"]} if args.gui else {})})
        import torch
        import carb.settings
        torch.set_num_threads(4)
        bench = ActivationBench(args, contract, manifest, prior)
        if args.sensor_only:
            report["simulated_dm_zero_model_pose"] = bench.simulated_encoder_zero
            report["simulated_dm_signs"] = bench.encoder_alignment.sign.tolist()
        carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
        view = ActivationView(bench) if args.gui else None
        report["status"] = "running"
        saved = False
        last_view = -1.
        while launcher.app.is_running():
            started = time.monotonic()
            if bench.env.sim.is_stopped():
                break
            if view and view.restart:
                bench.reset()
                view.restart, saved = False, False
                report["status"] = "running"
                report.pop("trials", None)
            if bench.time < args.seconds and (view is None or not view.paused):
                for _ in range(bench.policy_steps):
                    bench.step()
            if bench.time >= args.seconds and not saved:
                report.update(status="evaluated", simulated_seconds=bench.time, trials=bench.report())
                (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                sensor_trace = ({"sensor": bench.np.asarray(bench.sensor_traces)} if args.sensor_only else {})
                bench.np.savez_compressed(args.output / f"traces_{bench.run_index:03d}.npz",
                                         values=bench.np.asarray(bench.traces), diagnostics=bench.np.asarray(bench.diagnostics),
                                         **sensor_trace)
                print("ACTIVATION_EVALUATED", json.dumps(report["trials"]), flush=True)
                saved = True
                if view:
                    view.paused = True
                else:
                    break
            if view:
                if time.monotonic() - last_view >= .05:
                    view.update()
                    bench.env.sim.render()
                    last_view = time.monotonic()
                runtime = {"pid": os.getpid(), "time_s": bench.time, "paused": view.paused,
                           "selected": view.selection.model.get_item_value_model().as_int}
                if hasattr(bench, "last"):
                    runtime["selected_trial"] = bench.result(runtime["selected"])
                (args.output / "runtime.json").write_text(json.dumps(runtime) + "\n")
                time.sleep(max(0., bench.env.policy_dt - (time.monotonic() - started)))
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if bench is not None:
            if report["status"] == "running" and hasattr(bench, "last"):
                report.update(status="window_closed_before_horizon", simulated_seconds=bench.time,
                              trials=bench.report(), full_horizon_evaluated=False)
                sensor_trace = ({"sensor": bench.np.asarray(bench.sensor_traces)} if args.sensor_only else {})
                bench.np.savez_compressed(args.output / f"traces_{bench.run_index:03d}_partial.npz",
                                          values=bench.np.asarray(bench.traces), diagnostics=bench.np.asarray(bench.diagnostics),
                                          **sensor_trace)
            bench.writer.close()
            bench.env.close()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if launcher is not None:
            launcher.app.close()
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
