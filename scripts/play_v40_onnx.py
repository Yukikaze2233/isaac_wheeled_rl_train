#!/usr/bin/env python3
"""Replay a provenance-checked server ONNX actor in native Sim GUI or headless."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from train_v40 import add_common_arguments, launch_app, make_env, make_manifest, preflight


GROUND_ASSET_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
    "Environments/Grid/default_environment.usd"
)
GROUND_ASSET_SHA256 = "78e9a1e72a8838a13d0f65c49cd487ab92e89233cd128b057730b5b5b4ca2164"


def verify_cached_ground(path: Path) -> dict:
    """Pin the official Sim6 grid asset bytes; this is not an arbitrary terrain override."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != GROUND_ASSET_SHA256:
        raise ValueError("cached ground must match the pinned official Sim6 grid USD SHA256")
    return {"mode": "local_verified_official_asset", "path": str(path.resolve()),
            "source_url": GROUND_ASSET_URL, "sha256": digest, "size": len(raw)}


@contextmanager
def cached_ground_spawner(env_module, path: Path):
    """Scope a byte-identical ground URL replacement to this replay's construction.

    Keep the environment's actual material and all other ground settings. Restore
    the imported spawner even when construction fails; never patch training files.
    """
    verify_cached_ground(path)
    original = env_module.spawn_ground_plane

    def spawn(prim_path, cfg, **kwargs):
        local_cfg = deepcopy(cfg)
        local_cfg.usd_path = str(path.resolve())
        return original(prim_path=prim_path, cfg=local_cfg, **kwargs)

    env_module.spawn_ground_plane = spawn
    try:
        yield
    finally:
        env_module.spawn_ground_plane = original


DIAGNOSTIC_FLAGS = (
    "nonfinite", "non_wheel_contact", "knee_limit", "instantaneous_tilt",
    "low_height", "base_visual_bounds_ground", "failure_gravity", "sustained_failure",
)
TERMINATION_FLAGS = (
    "nonfinite", "non_wheel_contact", "knee_limit", "tilt", "low_height", "base_visual_bounds_ground",
)


def _record_snapshot_flags(summary: dict, snapshot: dict) -> tuple[list[bool], list[bool]]:
    """Count one pre-reset policy frame; never substitute task flags for diagnostics."""
    if snapshot["sample_kind"] != "pre_reset" or snapshot["diagnostic_flags_version"] != 1:
        raise ValueError("replay requires a versioned pre-reset diagnostic snapshot")
    diagnostics, reasons = snapshot["diagnostic_flags"], snapshot["termination_flags"]
    if set(diagnostics) != set(DIAGNOSTIC_FLAGS) or set(reasons) != set(TERMINATION_FLAGS):
        raise ValueError("incomplete snapshot diagnostic/termination flags")
    diagnostic_values = [bool(diagnostics[key].item()) for key in DIAGNOSTIC_FLAGS]
    reason_values = [bool(reasons[key].item()) for key in TERMINATION_FLAGS]
    for key, active in zip(DIAGNOSTIC_FLAGS, diagnostic_values):
        summary["diagnostic_frames"][key] += int(active)
    if bool(snapshot["terminated"].item()):
        for key, active in zip(TERMINATION_FLAGS, reason_values):
            summary["termination_flags"][key] += int(active)
    summary["diagnostic_frame_count"] += 1
    return diagnostic_values, reason_values


class KeyboardController:
    """Pure held-key state; requested (vx [m/s], wz [rad/s], height [m])."""

    KEYS = frozenset(("W", "A", "S", "D", "Q", "E", "SPACE", "MINUS", "EQUAL",
                      "NUMPAD_SUBTRACT", "NUMPAD_ADD", "LEFT_BRACKET", "RIGHT_BRACKET"))

    def __init__(self, bounds: dict, initial=(0.0, 0.0, 0.30),
                 linear_speed=1.0, angular_speed=1.0, *, exploratory=False,
                 max_linear_speed=None, max_angular_rpm=None):
        self.bounds = tuple(tuple(bounds[key]) for key in ("vx", "wz", "height"))
        for low, high in self.bounds:
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("invalid command bounds")
        self.exploratory = exploratory
        trained_caps = tuple(min(-low, high) for low, high in self.bounds[:2])
        self.max_linear_speed = trained_caps[0] if max_linear_speed is None else max_linear_speed
        self.max_angular_speed = (trained_caps[1] if max_angular_rpm is None
                                  else self.rpm_to_rad_s(max_angular_rpm))
        for cap, trained_cap in zip((self.max_linear_speed, self.max_angular_speed), trained_caps):
            if not math.isfinite(cap) or cap <= 0:
                raise ValueError("speed caps must be finite and positive")
            if not exploratory and cap > trained_cap:
                raise ValueError("caps outside saved contract require --exploratory")
        for speed, (low, high) in zip((linear_speed, angular_speed), self.bounds):
            if not math.isfinite(speed) or speed <= 0 or not low <= -speed <= speed <= high:
                raise ValueError("initial keyboard speed must be positive and inside saved contract")
        if linear_speed > self.max_linear_speed or angular_speed > self.max_angular_speed:
            raise ValueError("initial keyboard speed exceeds selected cap")
        if len(initial) != 3:
            raise ValueError("initial command must have three components")
        for value, (low, high) in zip(initial, self.bounds):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("initial command outside saved contract")
        if initial[:2] != (0.0, 0.0):
            raise ValueError("keyboard requires initial zero velocity; use keys to move")
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.height = float(initial[2])
        self.held: set[str] = set()

    @staticmethod
    def rpm_to_rad_s(rpm: float) -> float:
        if not math.isfinite(rpm):
            raise ValueError("rpm must be finite")
        return rpm * (math.tau / 60.0)

    @staticmethod
    def rad_s_to_rpm(rad_s: float) -> float:
        return rad_s * (60.0 / math.tau)

    def out_of_training_domain(self, command) -> bool:
        if len(command) != 3 or not all(math.isfinite(value) for value in command):
            raise ValueError("command must contain three finite values")
        return any(not low <= value <= high for value, (low, high) in zip(command, self.bounds))

    @property
    def metadata(self) -> dict:
        return dict(
            exploratory=self.exploratory,
            training_command_bounds=dict(zip(("vx", "wz", "height"), self.bounds)),
            max_linear_speed_m_s=self.max_linear_speed,
            max_angular_speed_rad_s=self.max_angular_speed,
            max_angular_speed_rpm=self.rad_s_to_rpm(self.max_angular_speed),
            selected_linear_speed_m_s=self.linear_speed,
            selected_angular_speed_rad_s=self.angular_speed,
            selected_angular_speed_rpm=self.rad_s_to_rpm(self.angular_speed),
            requested_out_of_training_domain=self.out_of_training_domain(self.requested),
            selected_speed_out_of_training_domain=self.out_of_training_domain(
                (self.linear_speed, self.angular_speed, self.height)),
        )

    @property
    def status_text(self) -> str:
        vx, wz, height = self.requested
        mode = "EXPLORATORY" if self.exploratory else "CONTRACT-BOUNDED"
        domain = "OOD" if self.out_of_training_domain(self.requested) else "IN-TRAINING-RANGE"
        selected_domain = "OOD" if self.metadata["selected_speed_out_of_training_domain"] else "in-range"
        return (f"Sim6 cross-sim {mode} | requested {domain}: vx={vx:.2f} m/s, "
                f"wz={wz:.4f} rad/s ({self.rad_s_to_rpm(wz):.2f} rpm), h={height:.2f} m | "
                f"speed ({selected_domain}) {self.linear_speed:.2f}/{self.max_linear_speed:g} m/s, "
                f"{self.rad_s_to_rpm(self.angular_speed):.2f}/"
                f"{self.rad_s_to_rpm(self.max_angular_speed):.2f} rpm "
                f"({self.angular_speed:.4f} rad/s) | +/- linear, [/] yaw, SPACE stop")

    @property
    def requested(self) -> tuple[float, float, float]:
        return (self.linear_speed * (("W" in self.held) - ("S" in self.held)),
                self.angular_speed * (("A" in self.held) - ("D" in self.held)), self.height)

    def press(self, key: str) -> None:
        if key == "SPACE":
            self.stop()
        elif key in self.KEYS and key not in self.held:
            self.held.add(key)
            if key in ("Q", "E"):
                low, high = self.bounds[2]
                self.height = min(high, max(low, round(self.height + (0.01 if key == "Q" else -0.01), 8)))
            elif key in ("MINUS", "EQUAL", "NUMPAD_SUBTRACT", "NUMPAD_ADD"):
                delta = -0.5 if key in ("MINUS", "NUMPAD_SUBTRACT") else 0.5
                self.linear_speed = min(self.max_linear_speed, max(0.0, self.linear_speed + delta))
            elif key in ("LEFT_BRACKET", "RIGHT_BRACKET"):
                delta = self.rpm_to_rad_s(-10.0 if key == "LEFT_BRACKET" else 10.0)
                self.angular_speed = min(self.max_angular_speed, max(0.0, self.angular_speed + delta))

    def release(self, key: str) -> None:
        self.held.discard(key)

    def stop(self) -> None:
        self.held.clear()


class NativeKeyboard:
    """Own only this Kit window's subscription; callbacks never mutate the env."""

    def __init__(self, controller: KeyboardController, record):
        import carb.input
        import carb.windowing
        import omni.appwindow

        self.controller = controller
        self.record = record
        self.window = omni.appwindow.get_default_app_window()
        self.windowing = carb.windowing.acquire_windowing_interface()
        self.last_title = None
        self.update_status()
        self.input = carb.input.acquire_input_interface()
        self.keyboard = self.window.get_keyboard()
        self.event_types = carb.input.KeyboardEventType
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.on_event)
        self.focus_subscription = self.window.get_window_focus_event_stream().create_subscription_to_pop(
            self.on_focus, name="server-final-onnx-teleop-focus")
        self.closed = False

    def update_status(self):
        title = (f"{getattr(self, 'title_prefix', 'SERVER FINAL ONNX TELEOP')} [PID {os.getpid()}] | {self.controller.status_text}"
                 + getattr(self, "runtime_status", ""))
        if title != self.last_title:
            self.windowing.set_window_title(self.window.get_window(), title)
            print("TELEOP_STATUS " + self.controller.status_text, flush=True)
            self.last_title = title

    def on_event(self, event, *args):
        if event.type not in (self.event_types.KEY_PRESS, self.event_types.KEY_RELEASE,
                              self.event_types.KEY_REPEAT):
            return True
        key = event.input.name
        if key not in self.controller.KEYS:
            return True
        if event.type == self.event_types.KEY_RELEASE:
            self.controller.release(key)
        elif event.type == self.event_types.KEY_PRESS:
            if self.window.is_focused():
                self.controller.press(key)
        # Repeats are recorded but never add speed, height or resurrect stopped keys.
        self.record("requested", event=event.type.name, key=key,
                    command=self.controller.requested, source="native_carb_keyboard")
        return False

    def on_focus(self, event):
        self.check_focus()

    def check_focus(self):
        if not self.window.is_focused() and self.controller.held:
            self.controller.stop()
            self.record("requested", event="FOCUS_LOST", key="", command=self.controller.requested,
                        source="native_appwindow_focus")

    def close(self):
        if not self.closed:
            self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
            self.focus_subscription.unsubscribe()
            self.controller.stop()
            self.closed = True


class ObservationCommandBridge:
    """Viewer-instance-only sampler hook, after reward and before the next history push.

    Even empty sampler ids reach this hook. Repeated observations in a tick must
    retain both the cached actor command and the reward command. Auto-reset calls
    the same sampler after reward and retains the latest manual override.
    """

    def __init__(self, env, controller: KeyboardController, record, before_sample=None):
        self.env = env
        self.controller = controller
        self.record = record
        self.before_sample = before_sample
        self.original = env._sample_commands
        self.last_seen = int(env.common_step_counter)
        self.applied = tuple(env._evaluation_command_override)
        self.had_instance_sampler = "_sample_commands" in vars(env)
        env._sample_commands = self.sample

    def sample(self, env_ids):
        if self.before_sample is not None:
            self.before_sample()
        tick = int(self.env.common_step_counter)
        if tick != self.last_seen:
            self.last_seen = tick
            requested = self.controller.requested
            if requested != self.applied:
                self.env._evaluation_command_override = requested
                self.env.commands[:] = self.env.commands.new_tensor(requested)
                self.applied = requested
                self.record("applied", command=requested, source="observation_boundary")
        return self.original(env_ids)

    def close(self):
        if self.had_instance_sampler:
            self.env._sample_commands = self.original
        else:
            del self.env._sample_commands


def load_policy(model: Path, expected: dict):
    """Validate immutable artifact bytes before creating a CPU-only ORT session."""
    import onnxruntime as ort

    raw = model.read_bytes()
    sidecar = json.loads(model.with_suffix(model.suffix + ".json").read_bytes())
    manifest_raw = (model.parent / "run_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if hashlib.sha256(raw).hexdigest() != sidecar["onnx_sha256"]:
        raise ValueError("ONNX SHA256 mismatch")
    if hashlib.sha256(manifest_raw).hexdigest() != sidecar["run_manifest_sha256"]:
        raise ValueError("run manifest SHA256 mismatch")
    if manifest != sidecar["run_manifest"]:
        raise ValueError("sidecar run manifest mismatch")
    for key in ("contract_id", "contract_sha256", "asset_manifest_sha256",
                "actor_obs_dim", "critic_obs_dim", "action_dim", "policy"):
        if manifest[key] != expected[key] or sidecar[key] != expected[key]:
            raise ValueError(f"artifact identity mismatch: {key}")
    if manifest["stage"] != expected["stage"]:
        raise ValueError("stage mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(raw, sess_options=options, providers=["CPUExecutionProvider"])
    for kind, nodes, name, shape in (
        ("input", session.get_inputs(), "obs_history", [1, 125]),
        ("output", session.get_outputs(), "actions", [1, 6]),
    ):
        if (len(nodes) != 1 or nodes[0].name != name or nodes[0].shape != shape
                or nodes[0].type != "tensor(float)"
                or sidecar[kind] != {"name": name, "dtype": "float32", "shape": shape}):
            raise ValueError(f"invalid ONNX {kind} signature")
    return session, sidecar


def create_report(path: Path) -> None:
    """Reserve a new report directory; never reuse or overwrite another run."""
    path.mkdir(parents=False, exist_ok=False)


def add_replay_grid(stage) -> dict:
    """World-fixed visual lines; never author collision or material friction."""
    from pxr import Gf, UsdGeom

    root = "/World/ReplayMetricGrid"
    UsdGeom.Xform.Define(stage, root)
    for name, major, width, color in (
        ("Major", True, .006, (.12, .14, .16)),
        ("Minor", False, .0015, (.32, .34, .36)),
    ):
        points = []
        for index in range(-200, 201):
            if (index % 10 == 0) != major:
                continue
            coordinate = index / 10
            points.extend((Gf.Vec3f(coordinate, -20, .002), Gf.Vec3f(coordinate, 20, .002),
                           Gf.Vec3f(-20, coordinate, .002), Gf.Vec3f(20, coordinate, .002)))
        curves = UsdGeom.BasisCurves.Define(stage, root + "/" + name)
        curves.CreateTypeAttr("linear")
        curves.CreateWrapAttr("nonperiodic")
        curves.CreateCurveVertexCountsAttr([2] * (len(points) // 2))
        curves.CreatePointsAttr(points)
        curves.CreateWidthsAttr([width])
        curves.SetWidthsInterpolation("constant")
        curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return {"prim_path": root, "major_spacing_m": 1.0, "minor_spacing_m": .1,
            "extent_m": [-20, 20], "visual_height_m": .002, "collision_enabled": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--ground-usd", type=Path,
                         help="Local copy of the SHA-pinned official Sim6 grid USD; avoids root-asset network lookup")
    parser.add_argument("--visible-grid", action="store_true",
                        help="Add local 1m/0.1m visual grid lines without changing ground collision")
    parser.add_argument("--repaired-visuals", type=Path,
                        help="SHA-pinned 15-component chassis visual reference; serial PhysX stays authoritative")
    parser.add_argument("--render-interval", type=int,
                        help="GUI render interval in physics steps; physics and policy dt stay fixed")
    parser.add_argument("--command", nargs=3, type=float, default=(0.0, 0.0, 0.30),
                        metavar=("VX", "WZ", "HEIGHT"))
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--max-wall-seconds", type=float, default=600.0)
    parser.add_argument("--keyboard", action="store_true", help="Native GUI held-key command control")
    parser.add_argument("--linear-speed", type=float, default=1.0, help="Keyboard speed [m/s]")
    parser.add_argument("--angular-speed", type=float, default=1.0, help="Keyboard yaw speed [rad/s]")
    parser.add_argument("--exploratory", action="store_true", help="Label keyboard OOD command exploration")
    parser.add_argument("--max-linear-speed", type=float, help="Selectable linear speed cap [m/s]")
    parser.add_argument("--max-angular-rpm", type=float, help="Selectable yaw speed cap [revolutions/min]")
    args = parser.parse_args(argv)
    ground_asset = None
    if args.ground_usd is not None:
        try:
            ground_asset = verify_cached_ground(args.ground_usd)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if not args.keyboard and (args.exploratory or args.max_linear_speed is not None
                              or args.max_angular_rpm is not None):
        parser.error("exploratory mode and speed caps require --keyboard")
    if args.keyboard and args.headless:
        parser.error("--keyboard requires GUI")
    if args.headless and (args.visible_grid or args.render_interval is not None):
        parser.error("visual grid and render interval require GUI")
    if args.render_interval is not None and not 1 <= args.render_interval <= 200:
        parser.error("render interval must be 1..200 physics steps")
    step_limit = 60000 if args.keyboard else 6000
    if (args.contract is None or args.num_envs != 1 or not 1 <= args.max_steps <= step_limit
            or not 300 <= args.max_wall_seconds <= 600):
        parser.error(f"require explicit saved --contract, one env, 1..{step_limit} steps, 300..600 wall seconds")
    # Stage comes from the server manifest, not the common training CLI default.
    args.stage = json.loads((args.onnx.parent / "run_manifest.json").read_bytes())["stage"]
    report, contract, asset = preflight(args)
    if ground_asset is not None:
        report["ground_asset"] = ground_asset
    if report["blockers"]:
        print(json.dumps(report, indent=2), flush=True)
        return 2
    for value, key in zip(args.command, ("vx", "wz", "height")):
        low, high = contract["commands"]["stages"][args.stage][key]
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f"command {key} outside saved contract")
    controller = None
    if args.keyboard:
        try:
            controller = KeyboardController(contract["commands"]["stages"][args.stage],
                                            tuple(args.command), args.linear_speed, args.angular_speed,
                                            exploratory=args.exploratory, max_linear_speed=args.max_linear_speed,
                                            max_angular_rpm=args.max_angular_rpm)
        except ValueError as exc:
            parser.error(str(exc))
    expected = make_manifest(contract, asset, args)
    session, sidecar = load_policy(args.onnx, expected)
    repaired_model = None
    if args.repaired_visuals is not None:
        from wheeled_tasks.v40.repaired_visuals import POLICY_SHA256, RepairedKinematics
        if sidecar["onnx_sha256"] != POLICY_SHA256:
            parser.error("repaired visual replay requires the pinned R3A final ONNX")
        repaired_model = RepairedKinematics(args.repaired_visuals)
        report["repaired_visual_sha256"] = repaired_model.hashes
    if controller is not None:
        report["teleoperation"] = controller.metadata
    if args.preflight_only:
        print(json.dumps(report, indent=2), flush=True)
        return 0
    args.report_dir = args.report_dir.resolve()
    create_report(args.report_dir)
    args.usd_cache_dir = args.report_dir / "usd_cache"
    started = time.monotonic()
    summary = dict(
        status="starting", pid=os.getpid(), argv=[sys.executable, *sys.argv],
        started_utc=datetime.now(timezone.utc).isoformat(), command=args.command,
        policy_source="SERVER FINAL ONNX; deterministic raw mean; no training",
        onnx_path=str(args.onnx.resolve()), onnx_sha256=sidecar["onnx_sha256"],
        contract_path=str(args.contract.resolve()), contract_sha256=expected["contract_sha256"],
        contract_file_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        asset_manifest_sha256=expected["asset_manifest_sha256"],
        source_versions=sidecar["run_manifest"]["target_versions"], local_preflight=report,
        cross_sim=True, provider=session.get_providers(), policy_steps=0,
        termination_resets=0, timeout_resets=0, reset_events=0, all_numeric_finite=True,
        max_action_norm=0.0, observation_noise=False, root_reset_velocity=0.0,
        max_steps=args.max_steps, max_wall_seconds=args.max_wall_seconds,
        headless=args.headless, screenshot_status="disabled_headless" if args.headless else "pending",
        termination_flags=dict.fromkeys(TERMINATION_FLAGS, 0),
        diagnostic_frames=dict.fromkeys(DIAGNOSTIC_FLAGS, 0),
        diagnostic_flags_version=1, diagnostic_frame_count=0,
        diagnostic_count_semantics="pre_reset_policy_frames_not_independent_contact_events",
        termination_count_semantics="task_reason_flags_on_terminated_frames; v2_tilt_is_sustained_failure",
        contact_semantics="rigid_body_net_force_history_peak_not_ground_pair",
        keyboard=args.keyboard, command_events=[],
        exploratory=args.exploratory, out_of_training_domain_command_frames=0,
        replay_domain="local Sim6 cross-sim; not server-native reproduction",
        teleoperation=controller.metadata if controller is not None else None,
        command_application="next observation boundary; previous reward uses previous command",
        ground_asset=ground_asset,
    )

    def save_summary():
        summary["elapsed_wall_s"] = time.monotonic() - started
        temporary = args.report_dir / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report_dir / "summary.json")

    save_summary()
    app = env = keyboard = bridge = event_stream = repaired_visuals = None
    exit_code = 0
    captures = []
    try:
        app = launch_app(args).app
        import numpy as np
        import torch

        if args.ground_usd is None:
            env = make_env(args)
        else:
            from wheeled_tasks.direct.v40_serial import env as env_module
            with cached_ground_spawner(env_module, args.ground_usd):
                env = make_env(args)
        if (env.contract_sha256 != expected["contract_sha256"]
                or env.asset_manifest_sha256 != expected["asset_manifest_sha256"]):
            raise RuntimeError("contract/assets changed during startup")
        env.set_evaluation_command(tuple(args.command))
        if args.render_interval is not None:
            env.cfg.sim.render_interval = args.render_interval
        if args.visible_grid:
            summary["visual_grid"] = add_replay_grid(env.sim.stage)
        summary["physics_device"] = str(env.device)
        obs, _ = env.reset()
        if repaired_model is not None:
            from wheeled_tasks.v40.repaired_visuals import TITLE, RepairedVisuals
            repaired_visuals = RepairedVisuals(
                env.sim.stage, repaired_model, "/World/envs/env_0/Robot", args.report_dir)
            repaired_visuals.update(env)
            summary["repaired_visuals"] = repaired_visuals.metadata
        event_stream = (args.report_dir / "command_events.csv").open("x", newline="")
        event_writer = csv.writer(event_stream)
        event_writer.writerow(["policy_tick", "kind", "event", "key", "vx", "wz", "height", "source"])

        def record(kind, *, command, source, event="", key=""):
            tick = int(env.common_step_counter)
            event_writer.writerow([tick, kind, event, key, *command, source])
            event_stream.flush()
            if event == "KEY_REPEAT":
                # Repeats cannot change controller state. Keep the raw CSV without
                # growing/re-serializing a large summary or updating the title.
                return
            entry = dict(policy_tick=tick, kind=kind, event=event, key=key,
                         command=command, source=source, exploratory=args.exploratory,
                         command_wz_rpm=KeyboardController.rad_s_to_rpm(command[1]),
                         out_of_training_domain=(controller.out_of_training_domain(command)
                                                 if controller is not None else False))
            summary["command_events"].append(entry)
            summary["command_events"] = summary["command_events"][-512:]
            summary["command_event_summary"] = "last 512 non-repeat events; complete stream in command_events.csv"
            print("TELEOP " + json.dumps(entry), flush=True)
            if controller is not None:
                summary["teleoperation"] = controller.metadata
            if keyboard is not None:
                keyboard.update_status()

        record("initial", command=tuple(args.command), source="evaluation_reset")
        if controller is not None:
            keyboard = NativeKeyboard(controller, record)
            if repaired_visuals is not None:
                keyboard.title_prefix = TITLE
                keyboard.update_status()
            bridge = ObservationCommandBridge(env, controller, record, keyboard.check_focus)
            summary["keyboard_name"] = keyboard.input.get_keyboard_name(keyboard.keyboard)
            summary["focus_clear"] = "native window focus event + observation-boundary is_focused check"
            print("KEYBOARD: Click this viewport. Hold W/S forward/back, A/D left/right yaw; "
                  "release stops that axis; SPACE clears held keys and commands zero velocity. "
                  "Q/E height +/-0.01 m (0.28..0.32). Focus loss clears keys. "
                  "-/= (or numpad -/+) linear speed -/+0.5 m/s; [/] yaw speed -/+10 rpm. "
                  "Speed selection clamps at zero/caps, one increment per press. "
                  "Commands drive the ONNX policy, not camera or kinematic pose. "
                  "Zero command is not a physical brake. Close this window to exit.", flush=True)
        viewport = None
        if not args.headless:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            viewport = get_active_viewport()
            if viewport is None or not env.sim.has_gui or not env.render_enabled:
                raise RuntimeError("native GUI/viewport/rendering unavailable")
            env.viewport_camera_controller.update_view_to_asset_root("robot")
            if repaired_visuals is not None:
                # Close three-quarter view; the controller continues tracking the live root.
                env.viewport_camera_controller.update_view_location(
                    eye=(1.15, 1.65, .85), lookat=(0., 0., -.06))
                if keyboard is None:
                    import carb.windowing
                    import omni.appwindow
                    window = omni.appwindow.get_default_app_window()
                    carb.windowing.acquire_windowing_interface().set_window_title(window.get_window(), TITLE)
            summary["viewport"] = dict(resolution=list(viewport.resolution),
                                       camera=str(viewport.camera_path), tracking="asset_root:robot")
        summary["status"] = "running"
        ready_marker = "SERVER_FINAL_ONNX_HEADLESS_READY" if args.headless else "SERVER_FINAL_ONNX_GUI_READY"
        print(ready_marker + " " + json.dumps(summary), flush=True)
        screenshot = args.report_dir / "viewport_policy_step100.png"
        movement_screenshot = args.report_dir / "viewport_after_command_change.png"
        first_movement_tick = None
        with (args.report_dir / "telemetry.csv").open("x", newline="") as stream, \
                (args.report_dir / "command_changes.csv").open("x", newline="") as changes_stream, \
                torch.inference_mode():
            changes_writer = csv.writer(changes_stream)
            changes_header = ["policy_tick", "kind", "vx", "wz_rad_s", "wz_rpm", "height", "exploratory",
                              "out_of_training_domain", "out_of_training_domain_command_frames"]
            changes_writer.writerow(changes_header)
            changes_writer.writerow([0, "initial", args.command[0], args.command[1],
                                     KeyboardController.rad_s_to_rpm(args.command[1]), args.command[2],
                                     args.exploratory, False, 0])
            changes_stream.flush()
            writer = csv.writer(stream)
            writer.writerow(["step", "sim_s", "action_norm", "x", "y", "z", "gravity_z",
                             "vx", "vy", "vz", "non_wheel_contact_n", "clearance_m", "terminated", "timeout",
                             "wz", "action_cmd_vx", "action_cmd_wz", "action_cmd_height",
                             "reward_cmd_vx", "reward_cmd_wz", "reward_cmd_height",
                             "next_cmd_vx", "next_cmd_wz", "next_cmd_height",
                             "requested_vx", "requested_wz", "requested_height", "policy_tick",
                             "exploratory", "action_command_ood", "next_command_ood",
                             "out_of_training_domain_command_frames",
                             "sample_kind", "episode_step", "episode_time_s", "sustained_failure_ticks",
                             "non_wheel_net_force_max_n",
                             *["diagnostic_" + key for key in DIAGNOSTIC_FLAGS],
                              *["termination_" + key for key in TERMINATION_FLAGS]])
            rollout_started = time.monotonic()
            for step in range(1, args.max_steps + 1):
                if not app.is_running() or time.monotonic() - started >= args.max_wall_seconds:
                    summary["stop_reason"] = "window_closed_or_wall_budget"
                    break
                tick_started = time.monotonic()
                action_command = tuple(env._evaluation_command_override)
                inputs = np.ascontiguousarray(obs["policy"].detach().cpu().numpy(), dtype=np.float32)
                if inputs.shape != (1, 125) or not np.isfinite(inputs).all():
                    raise FloatingPointError("invalid real policy observation")
                encoded_command = np.asarray(action_command, dtype=np.float32) * np.asarray(
                    contract["observations"]["scales"]["command"], dtype=np.float32)
                encoded_command = np.clip(encoded_command, -contract["observations"]["clip"],
                                          contract["observations"]["clip"])
                if not np.array_equal(inputs[0, -19:-16], encoded_command):
                    raise RuntimeError("action observation command differs from environment command")
                if step == 1:
                    summary["first_observation_sha256"] = hashlib.sha256(inputs.tobytes()).hexdigest()
                actions = session.run(["actions"], {"obs_history": inputs})[0]
                if actions.shape != (1, 6) or actions.dtype != np.float32 or not np.isfinite(actions).all():
                    raise FloatingPointError("invalid ONNX actions")
                obs, reward, terminated, timeout, _ = env.step(torch.from_numpy(actions).to(env.device))
                if repaired_visuals is not None:
                    repaired_visuals.update(env)
                snapshot = env.get_evaluation_snapshot()
                reward_command = snapshot["command"][0].cpu().tolist()
                if not np.array_equal(np.asarray(reward_command, dtype=np.float32),
                                      np.asarray(action_command, dtype=np.float32)):
                    raise RuntimeError("reward command no longer matches action observation")
                next_command = tuple(env._evaluation_command_override)
                requested = controller.requested if controller is not None else next_command
                finite_rollout = all(
                    bool(torch.isfinite(value).all())
                    for value in [obs["policy"], reward, *snapshot.values()]
                    if isinstance(value, torch.Tensor)
                ) and not bool(snapshot["diagnostic_flags"]["nonfinite"].item())
                norm = float(np.linalg.norm(actions))
                summary["max_action_norm"] = max(summary["max_action_norm"], norm)
                summary["policy_steps"] = step
                summary["sim_seconds"] = step * env.step_dt
                summary["rollout_wall_seconds"] = time.monotonic() - rollout_started
                summary["real_time_factor"] = summary["sim_seconds"] / max(summary["rollout_wall_seconds"], 1e-9)
                term, tout = bool(terminated.item()), bool(timeout.item())
                summary["termination_resets"] += int(term)
                summary["timeout_resets"] += int(tout)
                summary["reset_events"] += int(term or tout)
                if (term != bool(snapshot["terminated"].item())
                        or tout != bool(snapshot["timeout"].item())):
                    raise RuntimeError("step reset flags differ from pre-reset snapshot")
                diagnostic_values, reason_values = _record_snapshot_flags(summary, snapshot)
                pos = snapshot["root_link_pos_w_m"][0].cpu().tolist()
                vel = snapshot["root_com_lin_vel_b_m_s"][0].cpu().tolist()
                wz = snapshot["root_com_ang_vel_b_rad_s"][0, 2].item()
                # Keep summary JSON finite while preserving raw failure evidence in CSV.
                summary["latest_base_position_m"] = pos if finite_rollout else None
                summary["latest_velocity"] = dict(vx=vel[0], wz=wz) if finite_rollout else None
                summary["applied_command"] = next_command
                summary["requested_command"] = requested
                action_ood = controller.out_of_training_domain(action_command) if controller is not None else False
                next_ood = controller.out_of_training_domain(next_command) if controller is not None else False
                summary["out_of_training_domain_command_frames"] += int(action_ood)
                if next_command != action_command:
                    changes_writer.writerow([
                        int(env.common_step_counter), "command_change", next_command[0], next_command[1],
                        KeyboardController.rad_s_to_rpm(next_command[1]), next_command[2],
                        args.exploratory, next_ood, summary["out_of_training_domain_command_frames"]])
                    changes_stream.flush()
                writer.writerow([step, step * env.step_dt, norm, *pos,
                                 snapshot["projected_gravity_b"][0, 2].item(), *vel,
                                 snapshot["non_wheel_net_force_max_n"].item(),
                                 snapshot["base_visual_clearance_lower_bound_m"].item(), term, tout,
                                 wz, *action_command, *reward_command,
                                 *next_command, *requested, int(env.common_step_counter),
                                 args.exploratory, action_ood, next_ood,
                                 summary["out_of_training_domain_command_frames"],
                                 snapshot["sample_kind"], snapshot["episode_step"].item(),
                                 snapshot["episode_time_s"].item(), snapshot["sustained_failure_ticks"].item(),
                                 snapshot["non_wheel_net_force_max_n"].item(),
                                 *diagnostic_values, *reason_values])
                stream.flush()
                if not finite_rollout:
                    raise FloatingPointError("nonfinite physical rollout; diagnostic frame recorded")
                if term or tout:
                    # DirectRLEnv already auto-resets with the unchanged override.
                    print(f"RESET step={step} terminated={term} timeout={tout} "
                          f"flags={summary['termination_flags']}", flush=True)
                if step == 100 and viewport is not None:
                    summary["screenshot_request_policy_tick"] = int(env.common_step_counter)
                    summary["screenshot_sample_boundary"] = (
                        "async next viewport render; repaired visuals use latest complete post-step snapshot"
                        if repaired_visuals is not None else "async next viewport render")
                    captures.append(capture_viewport_to_file(viewport, str(screenshot), is_hdr=False))
                if next_command[:2] != (0.0, 0.0) and first_movement_tick is None:
                    first_movement_tick = step
                if (first_movement_tick is not None and step == first_movement_tick + 100
                        and viewport is not None):
                    captures.append(capture_viewport_to_file(viewport, str(movement_screenshot), is_hdr=False))
                if movement_screenshot.exists() and movement_screenshot.stat().st_size > 0:
                    summary["movement_screenshot"] = str(movement_screenshot)
                if screenshot.exists() and screenshot.stat().st_size > 0:
                    summary["screenshot_status"] = "captured"
                    summary["screenshot"] = str(screenshot)
                    summary["screenshot_bytes"] = screenshot.stat().st_size
                if step % 25 == 0:
                    save_summary()
                if step % 100 == 0:
                    if keyboard is not None:
                        keyboard.runtime_status = (f" | actual vx={vel[0]:.2f} m/s, wz={wz:.2f} rad/s"
                                                   f" | playback {summary['real_time_factor']:.2f}x")
                        keyboard.update_status()
                    print(f"REPLAY step={step} resets={summary['reset_events']} base={pos} "
                          f"action_norm={norm:.4f} screenshot={summary['screenshot_status']}", flush=True)
                if not args.headless:
                    time.sleep(max(0.0, env.step_dt - (time.monotonic() - tick_started)))
            else:
                summary["stop_reason"] = "step_budget"
            final_command = tuple(env._evaluation_command_override)
            changes_writer.writerow([
                int(env.common_step_counter), "end", final_command[0], final_command[1],
                KeyboardController.rad_s_to_rpm(final_command[1]), final_command[2], args.exploratory,
                controller.out_of_training_domain(final_command) if controller is not None else False,
                summary["out_of_training_domain_command_frames"]])
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="failed", error=repr(exc))
        if isinstance(exc, FloatingPointError):
            summary["all_numeric_finite"] = False
        traceback.print_exc()
        exit_code = 1
    finally:
        if summary["screenshot_status"] == "pending":
            summary["screenshot_status"] = "not_captured"
        save_summary()
        try:
            if keyboard is not None:
                keyboard.close()
                summary["keyboard_subscriptions_closed"] = keyboard.closed
                save_summary()
            if bridge is not None:
                bridge.close()
            if event_stream is not None:
                event_stream.close()
            if repaired_visuals is not None:
                repaired_visuals.close()
                summary["repaired_visual_visibility_restored"] = repaired_visuals.closed
                save_summary()
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
