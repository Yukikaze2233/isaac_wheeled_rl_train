#!/usr/bin/env python3
"""Native free-base V5 policy playback with real motor/spring effort overlays."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TITLE = "V5 GROUNDED POLICY | SIX MOTOR TORQUES + GAS SPRINGS"


def begin_manual_jump(env, contract, height):
    """Start the existing trained task clock without changing physical state."""
    profile = next((spec for spec in contract.get("skill_specs", {}).values()
        if spec.get("kind") == "jump" and math.isclose(spec["semantics"]["jump_apex_delta_m"], height)), None)
    if profile is None:
        raise ValueError(f"No trained standing-jump command profile for {height:g} m")
    settings = {**contract["task_semantics"], **profile["semantics"]}
    env.full_tasks.cfg.update(settings)
    env.cfg["task_semantics"].update(settings)
    ids = env.episode_length_buf.new_tensor([0])
    env.phase.reset(ids)
    env.fall_confirmation.reset(ids)
    env.full_tasks.reset(ids, env.robot.data.root_link_pose_w.torch[:, :3] - env.origins)
    env.success_hold[0] = 0.
    env.jump_requested[0] = False
    env.episode_length_buf[0] = round(settings["jump_request_seconds"] / env.policy_dt)
    env.mode[0] = 4
    env.update_targets()


class PolicyView:
    def __init__(self, env, args):
        import carb.windowing
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
        from preview_v5_springs import BenchView
        self.env, self.args = env, args
        self.paused = False
        self.reset_requested = False
        self.jump_requested = None
        self.current_height = .305
        self.height_range = tuple(env.cfg.get("height_range_m", (.29, .32)))
        self.policy_label = args.onnx.parent.name if args.onnx else args.checkpoint.parent.name
        from wheeled_tasks.chassis.teleop import KeyboardCommand
        # Interactive command response is separate from the training curriculum's slew.
        self.teleop = KeyboardCommand(args.keyboard_vx_acceleration, args.keyboard_yaw_acceleration,
                                      height_range=self.height_range)
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.keyboard_subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.on_key)
        self.probe = None
        self.visibility = {}
        stage = env.sim.stage
        self.lines = UsdGeom.BasisCurves.Define(stage, "/World/PolicyForceOverlay")
        self.lines.CreateTypeAttr("linear")
        self.lines.CreateWrapAttr("nonperiodic")
        self.lines.CreateWidthsAttr([.002])
        self.lines.SetWidthsInterpolation("constant")
        self.lines.CreateDisplayColorPrimvar("uniform")
        root = env.env_paths[0] + "/Robot"
        native = omni.appwindow.get_default_app_window()
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().resolution = (1280, 720)
        title = f"{self.policy_label} | ONNX | WASD | LOCAL ISAAC SIM" if args.onnx else TITLE
        carb.windowing.acquire_windowing_interface().set_window_title(native.get_window(), title)
        for name in env.robot.body_names:
            kind = ("base" if name == "base_link" else "rod" if name.startswith(("LLL", "RRR")) and name.endswith("1")
                    else "cylinder" if name.startswith(("LLL", "RRR")) else "crank" if name in ("LL_link1", "RR_link1")
                    else "thigh" if name in ("L_link1", "R_link1") else "shank" if name in ("L_link2", "R_link2")
                    else "wheel" if name in ("L_link3", "R_link3") else "coupler")
            visual = stage.GetPrimAtPath(root + "/" + name + "/Visual")
            self.visibility[name] = UsdGeom.Imageable(visual)
            material = UsdShade.Material.Define(stage, "/World/PolicyMaterials/" + kind)
            shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*BenchView.COLORS[kind]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.55)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(visual).Bind(material)
        UsdLux.DomeLight.Define(stage, "/World/PolicyDome").CreateIntensityAttr(800.)
        sun = UsdLux.DistantLight.Define(stage, "/World/PolicySun")
        sun.CreateIntensityAttr(1800.)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-35., -25., 30.))
        self.window = ui.Window("GROUND POLICY / ACTUAL EFFORT", width=580, height=860)
        with self.window.frame:
            with ui.VStack(spacing=7):
                ui.Label("W/S forward/back | A/D yaw | Q/E or T/G height\nJ jump 6cm | K jump 10cm | SPACE stop | P pause | R reset\nClick the viewport for keyboard focus.", word_wrap=True)
                self.info = ui.Label("Initializing", height=310, word_wrap=True)
                ui.Label(f"Contract height command range: {self.height_range[0]:.2f}-{self.height_range[1]:.2f} m (candidate)")
                self.height = ui.FloatSlider(min=self.height_range[0], max=self.height_range[1])
                self.height.model.set_value(.305)
                with ui.HStack(height=22):
                    self.auto = ui.CheckBox()
                    self.auto.model.set_value(not bool(args.onnx))
                    ui.Label("Automatic slow height sweep near 0.305 m")
                with ui.HStack(height=22):
                    self.keyboard_enabled = ui.CheckBox()
                    self.keyboard_enabled.model.set_value(bool(args.onnx))
                    ui.Label("Keyboard commands")
                    self.follow_camera = ui.CheckBox()
                    self.follow_camera.model.set_value(True)
                    ui.Label("Follow robot")
                ui.Label("Forward velocity [m/s]")
                self.vx = ui.FloatSlider(min=-3., max=3.)
                self.vx.model.set_value(0.)
                ui.Label("Yaw velocity [rad/s]")
                self.yaw = ui.FloatSlider(min=-4 * math.pi, max=4 * math.pi)
                self.yaw.model.set_value(0.)
                ui.Label("Keyboard speed limits [m/s] / [rad/s]")
                with ui.HStack(height=24):
                    self.speed_limit = ui.FloatSlider(min=.1, max=3.)
                    self.speed_limit.model.set_value(.5)
                    self.yaw_limit = ui.FloatSlider(min=.1, max=4 * math.pi)
                    self.yaw_limit.model.set_value(1.)
                ui.Label("Keyboard command ramp [m/s^2] / [rad/s^2]")
                with ui.HStack(height=24):
                    self.vx_acceleration = ui.FloatSlider(min=.1, max=20.)
                    self.vx_acceleration.model.set_value(args.keyboard_vx_acceleration)
                    self.yaw_acceleration = ui.FloatSlider(min=.1, max=60.)
                    self.yaw_acceleration.model.set_value(args.keyboard_yaw_acceleration)
                with ui.HStack(height=28):
                    ui.Button("Whole robot", clicked_fn=lambda: self.set_view("whole"))
                    ui.Button("Left mechanism", clicked_fn=lambda: self.set_view("left"))
                if not args.onnx:
                    with ui.HStack(height=28):
                        ui.Button("Probe lower range", clicked_fn=lambda: self.start_probe(.205))
                        ui.Button("Probe upper range", clicked_fn=lambda: self.start_probe(.36))
                with ui.HStack(height=28):
                    ui.Button("Pause / resume", clicked_fn=self.toggle_pause)
                    ui.Button("Reset to 0.305 m", clicked_fn=self.request_reset)
                with ui.HStack(height=28):
                    ui.Button("J: request 6cm jump", clicked_fn=lambda: self.request_jump(.06))
                    ui.Button("K: request 10cm jump", clicked_fn=lambda: self.request_jump(.10))
                limits = env.manifest["knee_inner_limits_deg"]
                spring_mm = env.model_spec["spring_binding"]["L_spring_slide"]["full_extension_pin_distance_m"] * 1000
                ui.Label(f"Knee {limits[0]:g}-{limits[1]:g} deg; spring pin length {spring_mm:.1f} mm.\nCandidate policy; failure pauses in place.", word_wrap=True)
        self.set_view(args.view)

    def on_key(self, event, *_args):
        name, kind = self.teleop.event_fields(event)
        if name is None:
            return True
        if kind == "KEY_RELEASE":
            self.teleop.key(name, False)
        elif kind in ("KEY_PRESS", "KEY_REPEAT"):
            if name in self.teleop.MOTION_KEYS:
                self.teleop.key(name, True)
                self.auto.model.set_value(False)
            elif kind == "KEY_REPEAT":
                return False
            elif name == "SPACE":
                self.teleop.stop()
                self.vx.model.set_value(0.)
                self.yaw.model.set_value(0.)
            elif name == "P":
                self.toggle_pause()
            elif name == "R":
                self.request_reset()
            elif name in ("J", "K"):
                self.request_jump(.06 if name == "J" else .10)
        return name not in self.teleop.MOTION_KEYS | {"SPACE", "P", "R", "J", "K"}

    def request_jump(self, height):
        if not self.paused:
            self.jump_requested = height
            self.auto.model.set_value(False)

    def close(self):
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.keyboard_subscription)

    def set_view(self, mode):
        self.mode = mode
        for name, visual in self.visibility.items():
            if mode == "left" and not name.startswith("L"):
                visual.MakeInvisible()
            else:
                visual.MakeVisible()
        eye = (.5, 1.15, .48) if mode == "left" else (1.0, 1.2, .85)
        self.eye_offset = np.array(eye) - np.array([0., 0., .3])
        self.env.sim.set_camera_view(eye, (0., 0., .23))

    def toggle_pause(self):
        self.paused = not self.paused

    def request_reset(self):
        self.reset_requested = True
        self.jump_requested = None
        self.paused = False
        self.teleop.stop()
        self.teleop.height = .305
        self.probe = None
        self.height.model.set_value(.305)
        self.auto.model.set_value(False)
        self.vx.model.set_value(0.)
        self.yaw.model.set_value(0.)

    def start_probe(self, height):
        self.probe = height
        self.auto.model.set_value(False)
        self.vx.model.set_value(0.)
        self.yaw.model.set_value(0.)

    def command(self, seconds, dt):
        if self.keyboard_enabled.model.as_bool:
            self.teleop.vx_limit = self.speed_limit.model.as_float
            self.teleop.yaw_limit = self.yaw_limit.model.as_float
            self.teleop.vx_acceleration = self.vx_acceleration.model.as_float
            self.teleop.yaw_acceleration = self.yaw_acceleration.model.as_float
            command = self.teleop.advance(dt)
            self.vx.model.set_value(command[0])
            self.yaw.model.set_value(command[1])
            self.height.model.set_value(command[2])
            self.current_height = command[2]
            return command
        if self.probe is not None:
            self.height.model.set_value(self.current_height + np.clip(self.probe - self.current_height, -.005 * dt, .005 * dt))
        elif self.auto.model.as_bool:
            self.height.model.set_value(.305 + .015 * math.sin(seconds * math.tau / 16.))
        goal = self.height.model.as_float
        self.current_height += float(np.clip(goal - self.current_height, -.01 * dt, .01 * dt))
        return [self.vx.model.as_float, self.yaw.model.as_float, self.current_height]

    def update(self, sample, poses, spec):
        from pxr import Gf
        starts, ends, colors = [], [], []

        def rotate(quat, vector):
            uv = np.cross(quat[:3], vector)
            return vector + 2 * (quat[3] * uv + np.cross(quat[:3], uv))

        def arrow(start, end, color):
            start, end = np.asarray(start), np.asarray(end)
            direction = end - start
            if np.linalg.norm(direction) < 1e-7:
                return
            direction /= np.linalg.norm(direction)
            normal = np.cross(direction, [0., 0., 1.])
            if np.linalg.norm(normal) < .1:
                normal = np.cross(direction, [0., 1., 0.])
            normal /= np.linalg.norm(normal)
            for a, b in ((start, end), (end, end - .009 * direction + .004 * normal),
                         (end, end - .009 * direction - .004 * normal)):
                starts.append(tuple(map(float, a)))
                ends.append(tuple(map(float, b)))
                colors.append(color)

        joints = {j["name"]: j for j in spec["joints"]}
        for name, torque in zip(self.env.manifest["control_joint_names"], sample["motor_effort_nm"]):
            if self.mode == "left" and name.startswith("R"):
                continue
            joint = joints[name]
            pose = poses[joint["child"]]
            axis = rotate(pose[3:], np.asarray(joint["axis"]))
            radial = np.cross(axis, [0., 0., 1.])
            if np.linalg.norm(radial) < 1e-6:
                radial = np.cross(axis, [1., 0., 0.])
            radial /= np.linalg.norm(radial)
            tangent = np.cross(axis, radial)
            bound = 3.84 if name.endswith("joint3") else 40.
            sweep = math.copysign(.35 + 2.5 * min(abs(torque) / bound, 1.), torque)
            points = [pose[:3] + .035 * (math.cos(a) * radial + math.sin(a) * tangent) for a in np.linspace(0., sweep, 18)]
            color = (.25, 1., .45, 1.) if torque >= 0 else (1., .25, .2, 1.)
            for a, b in zip(points[:-2], points[1:-1]):
                starts.append(tuple(map(float, a)))
                ends.append(tuple(map(float, b)))
                colors.append(color)
            arrow(points[-2], points[-1], color)
        for side, force in zip(("L", "R"), sample["spring_force_n"]):
            if self.mode == "left" and side == "R":
                continue
            upper, lower = poses[side * 3 + "_link1"][:3], poses[side * 3 + "_link2"][:3]
            axis = (upper - lower) / np.linalg.norm(upper - lower)
            arrow(upper, upper + axis * force / 8000., (1., .85, .1, 1.))
            arrow(lower, lower - axis * force / 8000., (1., .85, .1, 1.))
        if starts:
            self.lines.GetCurveVertexCountsAttr().Set([2] * len(starts))
            self.lines.GetPointsAttr().Set([Gf.Vec3f(*p) for pair in zip(starts, ends) for p in pair])
            self.lines.GetDisplayColorPrimvar().Set([Gf.Vec3f(*c[:3]) for c in colors])
        command = sample["command"]
        if self.follow_camera.model.as_bool:
            position = poses["base_link"][:3]
            self.env.sim.set_camera_view(tuple(position + self.eye_offset), tuple(position + [0., 0., -.08]))
        self.info.text = (
            f"{self.policy_label}\nInference: {sample.get('inference_backend', 'torch')} | {sample.get('inference_ms', 0.):.3f} ms\n"
            f"STATUS: {sample['status'] if sample['status'].startswith('PAUSED:') else 'USER PAUSED' if self.paused else sample['status']} | simulated {sample['time_s']:.2f} s\n"
            f"Commands vx/yaw: {command[0]:+.2f} m/s / {command[1]:+.2f} rad/s\n"
            f"Measured vx/yaw: {sample['velocity_m_s'][0]:+.2f} m/s / {sample['omega_rad_s'][2]:+.2f} rad/s\n"
            f"Height target/actual: {command[2]:.3f} / {sample['height_m']:.3f} m\n"
            f"Jump: {sample.get('jump_phase', 'GROUND')} | request {sample.get('jump_active', False)} | "
            f"clearance {sample.get('jump_clearance_peak_m', 0.) * 100:.2f} cm | "
            f"air {sample.get('jump_air_time_peak_s', 0.):.3f} s\n"
            f"Knee L/R: {sample['knee_deg'][0]:.2f} / {sample['knee_deg'][1]:.2f} deg\n"
            f"Spring compression: {sample['compression_mm'][0]:.2f} / {sample['compression_mm'][1]:.2f} mm\n"
            f"Spring force: {sample['spring_force_n'][0]:.1f} / {sample['spring_force_n'][1]:.1f} N\n"
            f"Wheel contact forces: {sample['wheel_force_n'][0]:.1f} / {sample['wheel_force_n'][1]:.1f} N\n"
            + "\n".join(f"{name:12s} {torque:+7.3f} Nm" for name, torque in zip(self.env.manifest["control_joint_names"], sample["motor_effort_nm"]))
            + f"\nLoop gap max: {sample['gap_mm']:.4f} mm\n"
            + ("SPRING CURVE EXTRAPOLATION: compression exceeds fitted 72 mm" if max(sample["compression_mm"]) > 72. else "Spring fitted working band: compression <= 72 mm"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    policy_source = parser.add_mutually_exclusive_group(required=True)
    policy_source.add_argument("--checkpoint", type=Path)
    policy_source.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--asset-directory", type=Path, help="Relocate the exact manifest-pinned asset")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--view", choices=("whole", "left"), default="whole")
    parser.add_argument("--floor-boxes", action="store_true", help="Equivalent flat box tiles for CPU contact compatibility")
    parser.add_argument("--keyboard-vx-acceleration", type=float, default=6., help="Interactive velocity ramp in m/s^2")
    parser.add_argument("--keyboard-yaw-acceleration", type=float, default=16., help="Interactive yaw ramp in rad/s^2")
    parser.add_argument("--jump-at", type=float, help="One scheduled manual jump request for a bounded playback probe")
    parser.add_argument("--jump-height", type=float, choices=(.06, .10), default=.06)
    args = parser.parse_args()
    policy_file = args.onnx or args.checkpoint
    if not policy_file.is_file() or not 0 < args.seconds <= 300:
        parser.error("Existing policy and bounded positive duration required")
    if not (.1 <= args.keyboard_vx_acceleration <= 20. and .1 <= args.keyboard_yaw_acceleration <= 60.):
        parser.error("Keyboard ramp must fit the positive GUI adjustment range")
    if args.contract is None:
        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
        if args.onnx:
            sidecar = Path(str(args.onnx) + ".contract.json")
            args.contract = sidecar if sidecar.exists() else args.onnx.parent / "contract.json"
        else:
            args.contract = checkpoint_contract_path(args.checkpoint)
    from train_chassis import digest, preflight
    if args.jump_at is not None and (not math.isfinite(args.jump_at) or args.jump_at < 0):
        parser.error("jump-at must be finite and nonnegative")
    contract, manifest = preflight(args.contract, asset_directory=args.asset_directory)
    config = deepcopy(contract)
    config.update(scene_groups=[{"name": "live", "fraction": 1., "terrain": ["jump"]}],
                  record_diagnostics=True, auto_reset=False, monitor_applied_effort=True,
                  evaluation_long_corridors=True, playback_open_ground=False, evaluation_exact_cases=True,
                  skill_specs={"live": {"kind": "constant", "command": [0., 0., .305], "mode": 0}},
                   evaluation={"cases": [{"name": "live", "terrain": "jump", "command": [0., 0., .305],
                                         "skill": {"kind": "constant", "command": [0., 0., .305], "mode": 0}}], "seed": 190619})
    if args.floor_boxes:
        config.update(flat_triangle_mesh=False, flat_half_length_m=44.)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
        "policy_file": str(policy_file.resolve()), "policy_sha256": digest(policy_file),
        "asset_manifest_sha256": contract["asset_manifest_sha256"],
        "asset_directory": config["asset_directory"], "manual_jump_events": [],
        "jump_request_scope": "trained request/apex/clock inputs only; no force or velocity assistance",
        "inference_backend": "onnxruntime_cpu" if args.onnx else "torch",
        "physics_device": args.device, "playback_floor": "box_tiles" if args.floor_boxes else "training_mesh",
        "controller": "deterministic learned actor controlling all six motors", "fixed_base": False,
        "external_guide": False, "state_writes_between_resets": 0, "samples": [],
        "keyboard_command_ramp": {"vx_m_s2": args.keyboard_vx_acceleration,
                                  "yaw_rad_s2": args.keyboard_yaw_acceleration}}
    launcher, env, view, writer = None, None, None, None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": args.headless, "device": args.device, "enable_cameras": False,
            **({"visualizer": ["kit"]} if not args.headless else {})})
        import carb.settings
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.v40.core import load_contract
        torch.set_num_threads(4)
        writer = SummaryWriter(str(args.output))
        env = FixedCaseEnv(config, manifest, load_contract(ROOT / config["control_math_source"]), ROOT,
            stage_name=config["enabled_stages"][0], num_envs=1, device=args.device, seed=190619, level=0.)
        env.max_episode_length = 2**40
        env.episode_limits.fill_(env.max_episode_length)
        carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
        carb.settings.get_settings().set_bool("/app/file/ignoreUnsavedOnExit", True)
        if args.onnx:
            import onnxruntime as ort
            metadata = json.loads(Path(str(args.onnx) + ".json").read_text())
            if (metadata["onnx_sha256"] != digest(args.onnx)
                    or metadata["asset_manifest_sha256"] != contract["asset_manifest_sha256"]
                    or metadata["contract_sha256"] != digest(args.contract)):
                raise ValueError("ONNX, model or operating contract identity mismatch")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
            if (len(session.get_inputs()) != 1 or session.get_inputs()[0].name != "obs"
                    or session.get_inputs()[0].shape[-1] != contract["actor_dim"]
                    or session.get_inputs()[0].type != "tensor(float)"
                    or len(session.get_outputs()) != 1 or session.get_outputs()[0].shape[-1] != 6):
                raise ValueError("ONNX policy interface mismatch")
            report["onnx_metadata"] = metadata
        else:
            from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
            from rsl_rl.runners import OnPolicyRunner
            from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
            cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
            cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
            cfg.device = args.device
            runner = OnPolicyRunner(env, deepcopy(cfg.to_dict()), log_dir=None, device=args.device)
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if checkpoint["infos"]["asset_manifest_sha256"] != contract["asset_manifest_sha256"]:
                raise ValueError("Policy and model identities disagree")
            runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
            actor = runner.alg.actor.as_onnx(verbose=False).to(args.device).eval()
            report["checkpoint_updates"] = checkpoint["infos"]["successful_updates_total"]
        view = None if args.headless else PolicyView(env, args)
        observations = env.reset_suite()
        sim_time, ticks, last_save, last_render, last_overlay = 0., 0, 0., 0., 0.
        sample, poses = None, None
        jump_active, scheduled_jump_sent = False, False
        report["status"] = "running"
        while launcher.app.is_running() and (not args.headless or sim_time < args.seconds):
            wall = time.monotonic()
            if env.sim.is_stopped():
                report["status"] = "timeline_stopped"
                break
            if view and view.reset_requested:
                observations = env.reset_suite()
                view.current_height = .305
                view.reset_requested = False
                jump_active = False
                report["status"] = "running"
            if not view or (not view.paused and env.sim.is_playing()):
                command = view.command(sim_time, env.policy_dt) if view else [0., 0., .305]
                proposed = env.commands.new_tensor([command])
                if config.get("motion_limits"):
                    from wheeled_tasks.chassis.motion_limits import project_commands
                    proposed = project_commands(proposed, config["motion_limits"])
                command = proposed[0].tolist()
                env.commands[0] = proposed[0]
                env.command_target[0] = env.command_target.new_tensor(command[:2])
                requested_jump = view.jump_requested if view else None
                if args.jump_at is not None and not scheduled_jump_sent and sim_time >= args.jump_at:
                    requested_jump, scheduled_jump_sent = args.jump_height, True
                if view:
                    view.jump_requested = None
                if requested_jump is not None and not jump_active:
                    begin_manual_jump(env, contract, requested_jump)
                    jump_active = True
                    report["manual_jump_events"].append({"time_s": sim_time, "height_m": requested_jump})
                env.mode[0] = 4 if jump_active else int(abs(command[0]) + abs(command[1]) > .01)
                # Refresh a single-frame policy's command immediately. A cached
                # observation from the last step would add an unintended delay.
                if config["history_length"] == 1:
                    env.history.last_tick = None
                    env.update_targets()
                    observations = env.get_observations()
                jump_context = observations["policy"][0, 32:35].cpu().tolist()
                with torch.no_grad():
                    inference_started = time.perf_counter()
                    if args.onnx:
                        output = session.run(None, {"obs": observations["policy"].cpu().numpy().astype(np.float32)})[0]
                        if output.shape != (1, 6) or not np.isfinite(output).all():
                            raise RuntimeError("Invalid ONNX action")
                        actions = torch.as_tensor(output, device=args.device)
                    else:
                        actions = actor(observations["policy"])
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    observations, _, done, extras = env.step(actions)
                sim_time += env.policy_dt
                ticks += 1
                diagnostic = extras["diagnostics"]
                positions = env.robot.data.joint_pos.torch[0]
                pose_array = env.robot.data.body_link_pose_w.torch[0].cpu().numpy()
                poses = dict(zip(env.robot.body_names, pose_array))
                sample = {"time_s": sim_time, "status": "running", "command": command,
                    "pressed_keys": sorted(view.teleop.keys) if view else [],
                    "keyboard_enabled": bool(view and view.keyboard_enabled.model.as_bool),
                    "keyboard_command_ramp": {"vx_m_s2": view.teleop.vx_acceleration,
                                              "yaw_rad_s2": view.teleop.yaw_acceleration} if view else None,
                    "inference_backend": report["inference_backend"], "inference_ms": inference_ms,
                    "height_m": float(diagnostic["height"][0]),
                    "velocity_m_s": diagnostic["velocity"][0].cpu().tolist(),
                    "omega_rad_s": diagnostic["omega"][0].cpu().tolist(),
                    "position_m": diagnostic["position"][0].cpu().tolist(),
                    "knee_deg": [44.93665895381104 + math.degrees(float(positions[env.knee_ids[0]])),
                                  44.93665895381104 - math.degrees(float(positions[env.knee_ids[1]]))],
                    "compression_mm": ((env.v5.s0 - positions[env.spring_ids]) * 1000).cpu().tolist(),
                    "spring_force_n": env.robot.data.applied_torque.torch[0, env.spring_ids].cpu().tolist(),
                    "motor_effort_nm": diagnostic["motor_effort"][0].cpu().tolist(),
                    "wheel_force_n": env.contact_force[0, env.wheel_ids].norm(dim=-1).cpu().tolist(),
                    "gap_mm": float(diagnostic["gap"][0]) * 1000}
                from wheeled_tasks.chassis.task import Phase
                sample.update(jump_active=jump_active, jump_phase=Phase(int(env.phase.phase[0])).name,
                    actor_jump_context=jump_context,
                    jump_clearance_peak_m=float(env.full_tasks.clearance_peak[0]),
                    jump_air_time_peak_s=float(env.full_tasks.clear_air_time_peak[0]),
                    jump_com_rise_m=float(env.full_tasks.com_rise[0]),
                    task_success=bool(diagnostic["success"][0]))
                if bool(done[0]):
                    reasons = [name for name, mask in diagnostic["reasons"].items() if bool(mask[0])]
                    if bool(diagnostic["success"][0]):
                        reasons = ["jump_success" if jump_active else "task_success"]
                    sample["status"] = "PAUSED: " + ", ".join(reasons)
                    report["status"] = "paused_at_terminal_condition"
                    if view:
                        view.paused = True
                    else:
                        report["samples"].append(sample)
                        break
                if ticks % 10 == 0:
                    report["samples"].append(sample)
                    report["samples"] = report["samples"][-6000:]
                    for tag, value in {
                        "height_command_m": command[2], "height_m": sample["height_m"],
                        "vx_command_m_s": command[0], "vx_m_s": sample["velocity_m_s"][0],
                        "yaw_command_rad_s": command[1], "yaw_rad_s": sample["omega_rad_s"][2],
                        "inference_ms": inference_ms, "closure_gap_mm": sample["gap_mm"],
                        "jump_clearance_peak_m": sample["jump_clearance_peak_m"],
                        "jump_air_time_peak_s": sample["jump_air_time_peak_s"],
                        "jump_com_rise_m": sample["jump_com_rise_m"],
                    }.items():
                        writer.add_scalar("Playback/" + tag, value, ticks)
            if view and wall - last_overlay >= .1 and sample is not None:
                view.update(sample, poses, env.model_spec)
                last_overlay = wall
            if view and wall - last_render >= 1 / 30.:
                if env.sim.is_playing():
                    env.sim.render()
                else:
                    launcher.app.update()
                last_render = wall
                if ticks >= 300 and not (args.output / "viewport.png").exists():
                    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                    capture_viewport_to_file(get_active_viewport(), str(args.output / "viewport.png"))
            if time.monotonic() - last_save > .5 and sample is not None:
                temporary = args.output / "runtime.tmp"
                temporary.write_text(json.dumps({"pid": os.getpid(),
                    "inference_backend": report["inference_backend"], "paused": bool(view and view.paused),
                    "timeline_playing": env.sim.is_playing(),
                    "updated_at": datetime.now(timezone.utc).isoformat(), "current": sample}, indent=2) + "\n")
                temporary.replace(args.output / "runtime.json")
                last_save = time.monotonic()
            if view:
                time.sleep(max(0., env.policy_dt - (time.monotonic() - wall)))
        if report["status"] == "running":
            report["status"] = "completed" if args.headless else "window_closed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if view is not None:
            view.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 1 if report["status"] in ("failed", "paused_at_terminal_condition") else 0


if __name__ == "__main__":
    raise SystemExit(main())
