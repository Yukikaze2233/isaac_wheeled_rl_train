#!/usr/bin/env python3
"""One candidate ONNX, native GUI scenes, WASD, and explicitly unscored replay."""
from __future__ import annotations

import argparse
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
from chassis_suite_support import SCENARIOS, control_ownership, digest, prepare_playback, write_json
from play_v5_grounded import PolicyView, begin_manual_jump


class SuiteView(PolicyView):
    """Retain the existing actual-effort/arrow view and keyboard behavior."""
    def __init__(self, env, args):
        super().__init__(env, args)
        import omni.ui as ui
        self.suite_window = ui.Window("SCENARIO / NATIVE CONTROL OWNERSHIP", width=580, height=220)
        with self.suite_window.frame:
            with ui.VStack(spacing=5):
                ui.Label(SCENARIOS[args.scenario], word_wrap=True)
                ui.Label("Candidate REC update 38000 | qualification: null | unscored\n"
                         "WASD: translation/yaw | Q/E or T/G: height | SPACE: stop\n"
                         "R resets this scene's original birth pose. P pauses.\n"
                         "To change scene, close this window and use play_chassis_suite.sh --menu.", word_wrap=True)
                self.ownership_info = ui.Label("Native controller initializing", word_wrap=True)
                if args.scenario == "jump_challenge":
                    ui.Label("Jump request challenge: J 6 cm / K 10 cm. This REC model has no demonstrated trained jump completion.", word_wrap=True)
                elif args.scenario.startswith("fallen_"):
                    ui.Label("Native script / torque blend / RL hold / RL unlocked are shown separately.\nWASD planar motion waits for the native stability release gate.", word_wrap=True)
        self.auto.model.set_value(False)

    def update(self, sample, poses, spec):
        super().update(sample, poses, spec)
        rec = sample["native_control"]
        self.ownership_info.text = (
            f"Ownership: {rec['ownership']} | FSM: {rec['phase']}\n"
            f"Pure RL: {rec['pure_rl']} | policy interval mask: {rec['policy_learning_mask']}\n"
            f"Motion released: {rec['motion_released']} | true stable peak: {rec['stable_seconds']:.3f} s\n"
            f"Applied vx/yaw: {sample['command'][0]:+.3f} / {sample['command'][1]:+.3f} | gap: {sample['gap_mm']:.3f} mm")


def validated_session(policy, contract_path, contract):
    """Identity and ABI checks run before starting the native simulator."""
    import onnxruntime as ort
    metadata = json.loads(Path(str(policy) + ".json").read_text())
    if (metadata["onnx_sha256"] != digest(policy)
            or metadata["asset_manifest_sha256"] != contract["asset_manifest_sha256"]
            or metadata["contract_sha256"] != digest(contract_path)):
        raise ValueError("ONNX, model or source contract identity mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape[-1] != 35
            or inputs[0].type != "tensor(float)" or len(outputs) != 1 or outputs[0].shape[-1] != 6):
        raise ValueError("ONNX policy interface mismatch")
    return session, metadata


def apply_requested_command(env, previous, requested, jump_active):
    """Gate REC motion, slew once, and refresh the current actor observation."""
    from wheeled_tasks.chassis.playback_command import advance_playback_command
    proposed = env.commands.new_tensor([requested])
    rec = env.recovery
    if rec is not None and bool(rec.enabled[0]) and not bool(rec.fsm.motion_released[0]):
        proposed[0, :2] = 0.
        proposed[0, 2] = .305
    applied = advance_playback_command(previous, proposed, env.policy_dt,
        env.cfg["command_slew"], env.cfg.get("motion_limits"))
    acceleration = ((applied[:, :2] - previous[:, :2]) / env.policy_dt)[0].tolist()
    previous.copy_(applied)
    env.commands.copy_(applied)
    env.command_target.copy_(applied[:, :2])
    env.command_acceleration[0] = acceleration[0]
    env.raw_height.copy_(applied[:, 2])
    if rec is not None and bool(rec.enabled[0]):
        rec.followup_command.copy_(applied)
    env.mode[0] = 4 if jump_active else int(abs(float(applied[0, 0])) + abs(float(applied[0, 1])) > .01)
    env.history.last_tick = None
    env.update_targets()
    env.update_command_reference()
    return env.get_observations(), applied[0].tolist(), acceleration


def sample_state(env, diagnostic, requested, applied, acceleration, sim_time, view, inference_ms, jump_active):
    from wheeled_tasks.chassis.playback_command import attitude_from_gravity
    from wheeled_tasks.chassis.task import Phase
    positions = env.robot.data.joint_pos.torch[0]
    poses = dict(zip(env.robot.body_names, env.robot.data.body_link_pose_w.torch[0].cpu().numpy()))
    gravity = diagnostic["gravity"][0].cpu().tolist()
    roll, pitch = attitude_from_gravity(gravity)
    sample = dict(time_s=sim_time, status="running", scenario=view.args.scenario if view else None,
        command=applied, requested_command=requested, command_acceleration_vx_yaw=acceleration,
        projected_gravity_b=gravity, base_quaternion_xyzw=poses["base_link"][3:].tolist(),
        roll_deg=math.degrees(roll), pitch_deg=math.degrees(pitch),
        pressed_keys=sorted(view.teleop.keys) if view else [],
        inference_backend="onnxruntime_cpu", inference_ms=inference_ms,
        height_m=float(diagnostic["height"][0]), velocity_m_s=diagnostic["velocity"][0].cpu().tolist(),
        omega_rad_s=diagnostic["omega"][0].cpu().tolist(), position_m=diagnostic["position"][0].cpu().tolist(),
        knee_deg=[44.93665895381104 + math.degrees(float(positions[env.knee_ids[0]])),
                  44.93665895381104 - math.degrees(float(positions[env.knee_ids[1]]))],
        compression_mm=((env.v5.s0 - positions[env.spring_ids]) * 1000).cpu().tolist(),
        spring_force_n=env.robot.data.applied_torque.torch[0, env.spring_ids].cpu().tolist(),
        motor_effort_nm=diagnostic["motor_effort"][0].cpu().tolist(),
        wheel_force_n=env.contact_force[0, env.wheel_ids].norm(dim=-1).cpu().tolist(),
        gap_mm=float(diagnostic["gap"][0]) * 1000, native_control=control_ownership(env.recovery),
        jump_active=jump_active, jump_phase=Phase(int(env.phase.phase[0])).name,
        jump_clearance_peak_m=float(env.full_tasks.clearance_peak[0]),
        jump_air_time_peak_s=float(env.full_tasks.clear_air_time_peak[0]),
        jump_com_rise_m=float(env.full_tasks.com_rise[0]), task_success=bool(diagnostic["success"][0]))
    return sample, poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=ROOT / "models/v6_recovery_candidate_38000_20261008/policy.onnx")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="flat")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="CPU identity/design preparation; no native scene")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=10., help="Bounded headless simulation duration")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--view", choices=("whole", "left"), default="whole")
    parser.add_argument("--keyboard-vx-acceleration", type=float, default=6.)
    parser.add_argument("--keyboard-yaw-acceleration", type=float, default=16.)
    parser.add_argument("--initial-vx", type=float, default=0., help="Headless requested forward speed")
    parser.add_argument("--initial-yaw", type=float, default=0., help="Headless requested yaw speed")
    parser.add_argument("--jump-at", type=float, help="Scheduled request, only in jump_challenge")
    parser.add_argument("--jump-height", type=float, choices=(.06, .10), default=.06)
    args = parser.parse_args()
    if args.list_scenarios:
        for key, label in SCENARIOS.items():
            print(f"{key:24s} {label}")
        return 0
    if not args.onnx.is_file() or not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
        parser.error("Existing ONNX and positive bounded headless duration required")
    if not (.1 <= args.keyboard_vx_acceleration <= 20 and .1 <= args.keyboard_yaw_acceleration <= 60):
        parser.error("Keyboard acceleration is outside the existing GUI limits")
    if not (math.isfinite(args.initial_vx) and abs(args.initial_vx) <= 3
            and math.isfinite(args.initial_yaw) and abs(args.initial_yaw) <= 4 * math.pi):
        parser.error("Headless command is outside the existing GUI limits")
    if args.jump_at is not None and (args.scenario != "jump_challenge" or not math.isfinite(args.jump_at) or args.jump_at < 0):
        parser.error("Scheduled jumps require jump_challenge and a finite nonnegative time")
    args.contract = args.contract or Path(str(args.onnx) + ".contract.json")
    args.output = args.output or ROOT / "reports" / ("chassis_suite_" + args.scenario + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    # One native GUI owns keyboard focus. The lock is released on process exit.
    if not args.headless and not args.preflight_only:
        import fcntl
        (ROOT / "reports").mkdir(exist_ok=True)
        gui_lock = (ROOT / "reports/.chassis_suite_gui.lock").open("w")
        try:
            fcntl.flock(gui_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A suite GUI is already open. Close it before launching another scene.")
        gui_lock.write(str(os.getpid()) + "\n")
        gui_lock.flush()
    args.checkpoint = None  # Existing PolicyView only needs this for non-ONNX runs.
    args.available_jump_heights = (.06, .10) if args.scenario == "jump_challenge" else ()
    source, config, manifest, control, seed, receipt = prepare_playback(ROOT, args.contract, args.output, args.scenario)
    session, metadata = validated_session(args.onnx, args.contract, source)
    report = dict(status="CPU_preflight_passed_scene_not_run", pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), scenario=args.scenario,
        scenario_label=SCENARIOS[args.scenario], policy_file=str(args.onnx.resolve()),
        policy_sha256=digest(args.onnx), onnx_metadata=metadata, source_contract_sha256=digest(args.contract),
        design_preflight=receipt, inference_backend="onnxruntime_cpu", physics_device=args.device,
        run_scope="interactive_unscored", qualification=None, domain_qualified=None,
        optimizer_constructed=False, formal_updates_consumed=0, state_writes_between_resets=0,
        native_recovery="script_then_torque_blend_then_RL_with_native_motion_release_gate",
        applied_command_slew=config["command_slew"], command_timing="slew_once_before_refreshed_actor_observation",
        manual_jump_events=[], samples=[])
    write_json(args.output / "report.json", report)
    print(f"Scenario: {SCENARIOS[args.scenario]}\nOutput: {args.output}\nQualification: null", flush=True)
    if args.preflight_only:
        return 0
    launcher, env, view, writer = None, None, None, None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=args.headless, device=args.device, enable_cameras=False,
            **({"visualizer": ["kit"]} if not args.headless else {})))
        import carb.settings
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        torch.set_num_threads(4)
        writer = SummaryWriter(str(args.output))
        env = FixedCaseEnv(config, manifest, control, ROOT, stage_name=config["enabled_stages"][0],
            num_envs=1, device=args.device, seed=seed, level=1., run_role="playback")
        # Interactive playback keeps native failure guards; timeout is user-owned.
        # These lifecycle overrides are explicit in the report and never graded.
        env.max_episode_length = 2**40
        env.episode_limits.fill_(env.max_episode_length)
        report["interactive_timeout_override_ticks"] = env.max_episode_length
        report["startup_report"] = env.startup_report
        carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
        carb.settings.get_settings().set_bool("/app/file/ignoreUnsavedOnExit", True)
        observations = env.reset_suite()
        view = None if args.headless else SuiteView(env, args)
        applied = env.commands.clone()
        sim_time, ticks, last_save, last_overlay, last_render = 0., 0, 0., 0., 0.
        sample, poses, jump_active, scheduled_sent = None, None, False, False
        report["status"] = "running"
        while launcher.app.is_running() and (not args.headless or sim_time < args.seconds):
            wall = time.monotonic()
            if env.sim.is_stopped():
                report["status"] = "timeline_stopped"
                break
            if view and view.reset_requested:
                observations = env.reset_suite()
                applied.copy_(env.commands)
                view.reset_requested = False
                jump_active = False
                report["status"] = "running"
            if not view or (not view.paused and env.sim.is_playing()):
                requested = view.command(sim_time, env.policy_dt) if view else [args.initial_vx, args.initial_yaw, .305]
                jump = view.jump_requested if view else None
                if view:
                    view.jump_requested = None
                if args.jump_at is not None and not scheduled_sent and sim_time >= args.jump_at:
                    jump, scheduled_sent = args.jump_height, True
                if jump is not None and not jump_active:
                    begin_manual_jump(env, source, jump)
                    jump_active = True
                    report["manual_jump_events"].append(dict(time_s=sim_time, height_m=jump, qualification=None))
                observations, command, acceleration = apply_requested_command(env, applied, requested, jump_active)
                ownership_before = control_ownership(env.recovery)
                actor_command = observations["policy"][0, :4].cpu().tolist()
                with torch.no_grad():
                    start = time.perf_counter()
                    output = session.run(None, {"obs": observations["policy"].cpu().numpy().astype(np.float32)})[0]
                    inference_ms = (time.perf_counter() - start) * 1000
                    if output.shape != (1, 6) or not np.isfinite(output).all():
                        raise RuntimeError("Invalid ONNX action")
                    observations, _, done, extras = env.step(torch.as_tensor(output, device=args.device))
                sim_time += env.policy_dt
                ticks += 1
                sample, poses = sample_state(env, extras["diagnostics"], requested, command,
                    acceleration, sim_time, view, inference_ms, jump_active)
                sample.update(scenario=args.scenario, native_control_before_policy=ownership_before,
                    actor_observed_command_vx_lateral_yaw_height_times5=actor_command)
                if bool(done[0]):
                    diagnostic = extras["diagnostics"]
                    reasons = [name for name, mask in diagnostic["reasons"].items() if bool(mask[0])]
                    if bool(diagnostic["success"][0]):
                        reasons = ["native_task_success_unscored"]
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
                    for tag, value in dict(height_command_m=command[2], height_m=sample["height_m"],
                        vx_command_m_s=command[0], vx_m_s=sample["velocity_m_s"][0],
                        yaw_command_rad_s=command[1], yaw_rad_s=sample["omega_rad_s"][2],
                        roll_deg=sample["roll_deg"], pitch_deg=sample["pitch_deg"],
                        closure_gap_mm=sample["gap_mm"], inference_ms=inference_ms,
                        recovery_phase=int(env.recovery.fsm.phase[0]) if env.recovery is not None else -1,
                        recovery_pure_rl=float(sample["native_control"]["pure_rl"])).items():
                        writer.add_scalar("Playback/" + tag, value, ticks)
            if view and sample is not None and wall - last_overlay >= .1:
                view.update(sample, poses, env.model_spec)
                last_overlay = wall
            if view and wall - last_render >= 1 / 30.:
                if env.sim.is_playing():
                    env.sim.render()
                else:
                    launcher.app.update()
                last_render = wall
                # Capture the visible startup even if a first episode failed.
                if sample is not None and not (args.output / "viewport.png").exists():
                    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                    capture_viewport_to_file(get_active_viewport(), str(args.output / "viewport.png"))
            if time.monotonic() - last_save > .5 and sample is not None:
                write_json(args.output / "runtime.json", dict(pid=os.getpid(), scenario=args.scenario,
                    qualification=None, inference_backend="onnxruntime_cpu", paused=bool(view and view.paused),
                    timeline_playing=env.sim.is_playing(), updated_at=datetime.now(timezone.utc).isoformat(), current=sample))
                write_json(args.output / "report.json", report)
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
        write_json(args.output / "report.json", report)
        if view is not None:
            view.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
