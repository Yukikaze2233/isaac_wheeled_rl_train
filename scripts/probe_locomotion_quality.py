#!/usr/bin/env python3
"""Matched short ONNX locomotion diagnostics using native fixed-case physics/PD."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_MODELS = {
    "flat": ROOT / "models/v6_flat_candidate_14020_20261003/policy.onnx",
    "speed": ROOT / "models/v6_speed_candidate_19837_20261005/policy.onnx",
}
SPEEDS = (0., .5, -.5, 1., -1., 3., -3.)


def selected_models(specifications):
    if not specifications:
        return DEFAULT_MODELS.copy()
    models = {}
    for specification in specifications:
        name, separator, path = specification.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or not path or name in models:
            raise ValueError("Use unique NAME=PATH models with filename-safe names")
        models[name] = Path(path).resolve(strict=True)
    return models


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trajectory(speed, dt=.02, acceleration=1.5):
    """Exact policy-step phases; a release requests zero at the first brake tick."""
    if not all(math.isfinite(x) for x in (speed, dt, acceleration)) or dt <= 0 or acceleration <= 0:
        raise ValueError("Finite speed and positive command timing required")
    settle = round(2. / dt)
    if speed == 0.:
        return [("settle", settle, 0.), ("stand", round(13. / dt), 0.)]
    ramp = math.ceil(abs(speed) / (acceleration * dt) - 1e-10)
    return [("settle", settle, 0.), ("acceleration", ramp, speed),
            ("steady", round(4. / dt), speed), ("braking", ramp, 0.),
            ("stop_warmup", round(2. / dt), 0.), ("stop", round(10. / dt), 0.)]


def case_name(speed):
    if speed == 0.:
        return "stand_305mm"
    return ("forward_" if speed > 0 else "backward_") + f"{abs(speed):g}".replace(".", "p")


def terminal_reasons(diagnostics, index):
    reasons = {name: True for name, mask in diagnostics["reasons"].items() if bool(mask[index])}
    if bool(diagnostics["success"][index]):
        reasons["task_success"] = True
    return reasons


def summarize_frames(frames):
    """Each phase owns its position anchor; stopping never uses episode-origin drift."""
    import numpy as np
    result = {}
    for phase in dict.fromkeys(row["phase"] for row in frames):
        rows = [row for row in frames if row["phase"] == phase]
        p = np.asarray([r["position_m"][:2] for r in rows])
        anchor = np.asarray(rows[0]["phase_anchor_xy_m"])
        v = np.asarray([r["velocity_m_s"] for r in rows])
        omega = np.asarray([r["omega_rad_s"] for r in rows])
        command = np.asarray([r["command"] for r in rows])
        roll = np.asarray([r["roll_deg"] for r in rows])
        pitch = np.asarray([r["pitch_deg"] for r in rows])
        sat = np.asarray([r["wheel_saturated"] for r in rows], dtype=float)
        delta = np.asarray([r.get("action_delta_control_order", [0.] * 6) for r in rows])
        height = np.asarray([r["height_m"] for r in rows])
        speed = np.linalg.norm(v[:, :2], axis=1)
        result[phase] = dict(samples_200hz=len(rows), duration_s=rows[-1]["time_s"] - rows[0]["time_s"] + .005,
            vx_actual_mean_m_s=float(v[:, 0].mean()), vx_command_mean_m_s=float(command[:, 0].mean()),
            vx_mae_m_s=float(np.abs(v[:, 0] - command[:, 0]).mean()),
            planar_speed_mean_m_s=float(speed.mean()), planar_speed_max_m_s=float(speed.max()),
            roll_abs_mean_deg=float(np.abs(roll).mean()), roll_abs_max_deg=float(np.abs(roll).max()),
            pitch_abs_mean_deg=float(np.abs(pitch).mean()), pitch_abs_max_deg=float(np.abs(pitch).max()),
            tilt_max_deg=max(r["tilt_deg"] for r in rows),
            omega_abs_mean_rad_s=np.abs(omega).mean(0).tolist(), omega_abs_max_rad_s=np.abs(omega).max(0).tolist(),
            height_mae_m=float(np.abs(height - command[:, 2]).mean()),
            wheel_saturated_fraction=sat.mean(0).tolist(),
            action_abs_delta_mean_control_order=np.abs(delta).mean(0).tolist(),
            action_abs_delta_max_control_order=np.abs(delta).max(0).tolist(),
            displacement_from_phase_anchor_m=float(np.linalg.norm(p[-1] - anchor)),
            peak_displacement_from_phase_anchor_m=float(np.linalg.norm(p - anchor, axis=1).max()),
            phase_anchor_xy_m=anchor.tolist(),
            final_velocity_m_s=v[-1].tolist())
        if phase == "stop":
            # A continuous final 0.5 s below 3 cm/s is reported only when present.
            final = [r for r in rows if r["time_s"] >= rows[-1]["time_s"] - .5 + 1e-9]
            final_speed = [math.hypot(*r["velocity_m_s"][:2]) for r in final]
            result[phase]["final_half_second_speed_mean_m_s"] = sum(final_speed) / len(final_speed)
            result[phase]["final_half_second_all_below_003_m_s"] = all(x <= .03 for x in final_speed)
    return result


def make_common_config(path, speeds, output):
    from train_chassis import preflight
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    source, manifest = preflight(path)
    config = fixed_suite_contract(source)
    config.update(scene_groups=[dict(name="live", fraction=1., terrain=["jump"])],
        record_diagnostics=True, auto_reset=False, monitor_applied_effort=True,
        evaluation_long_corridors=True, playback_open_ground=False, evaluation_exact_cases=True,
        flat_triangle_mesh=False, flat_half_length_m=44.,
        skill_specs={"live": dict(kind="constant", command=[0., 0., .305], mode=0)})
    config["evaluation"].update(protocol_id="keyboard_playback_unscored", episodes_per_case=1,
        canonical_case_names=["live"], promotion_case_names=[], protected_case_names=[],
        retention_case_names=[], quick_case_names=[], seed=190619,
        cases=[dict(name="live", terrain="jump", command=[0., 0., .305],
            skill=dict(kind="constant", command=[0., 0., .305], mode=0))])
    design_path = ROOT / "contracts/v6_flat_keyboard_playback_expectations_v1.json"
    design = json.loads(design_path.read_text())
    if digest(path) != design["source_contract_sha256"]:
        raise ValueError("Common scene contract differs from frozen flat playback source")
    cases = [dict(name=case_name(speed), terrain="jump", command=[0., 0., .305], reset_seed_key="quality_common_reset",
                  skill=dict(kind="constant", command=[0., 0., .305], mode=0)) for speed in speeds]
    config["scene_groups"] = [dict(name=c["name"], fraction=1. / len(cases), terrain=["jump"]) for c in cases]
    config["skill_specs"] = {c["name"]: c["skill"] for c in cases}
    config["evaluation"].update(cases=cases, canonical_case_names=[c["name"] for c in cases])
    fixture = deepcopy(design)
    fixture.update(run_scope="interactive_unscored", probe_scope="short_quality_probe_unscored", training_release=False,
        review_basis="Native flat playback physics/control inherited unchanged; explicit parallel constant-case lanes, matched reset seed, external visible command trajectories, no skill credit.",
        inherited_playback_design_sha256=digest(design_path))
    row = fixture["stages"]["flat"]["roles"]["playback"]
    for key in ("evaluation", "scene_groups", "skill_specs"):
        row["contract"][key] = deepcopy(config[key])
    row["invocation"]["num_envs"] = len(cases)
    fixture_path = output / "probe_design_expectations.json"
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n")
    config["design_preflight"] = dict(file=str(fixture_path.resolve()), sha256=digest(fixture_path), stage="flat")
    control = load_chassis_control(ROOT / config["control_math_source"])
    receipt = design_preflight(config, control, ROOT, stage=config["enabled_stages"][0],
        role="playback", seed=190619, num_envs=len(cases))
    if config["command_slew"] != {"vx_m_s2": 1.5, "yaw_rad_s2": 4.}:
        raise ValueError("Probe requires frozen 1.5/4 command slopes")
    return source, config, manifest, control, receipt


def model_metadata(path, common):
    sidecar = Path(str(path) + ".contract.json")
    source = json.loads(sidecar.read_text())
    metadata = json.loads(Path(str(path) + ".json").read_text())
    if metadata["onnx_sha256"] != digest(path) or metadata["contract_sha256"] != digest(sidecar):
        raise ValueError("ONNX metadata identity mismatch")
    for key in ("asset_manifest_sha256", "control_math_sha256", "physics_dt", "policy_dt", "v5_control",
                "actor_dim", "critic_dim", "history_length", "actor_observation_source", "policy_action_order",
                "command_slew", "motion_limits", "actuator_response"):
        if source.get(key) != common.get(key):
            raise ValueError(f"Matched scene/model ABI differs: {key}")
    return dict(onnx=str(path), onnx_sha256=digest(path), contract=str(sidecar),
        contract_sha256=digest(sidecar), metadata=metadata,
        comparison_scope="unscored; flat 3m/s cases exceed flat trained speed coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--speeds", type=float, nargs="+", default=list(SPEEDS))
    parser.add_argument("--model", action="append", help="Compare NAME=ONNX_PATH; repeat to override the default models")
    args = parser.parse_args()
    if any(x not in SPEEDS for x in args.speeds) or len(set(args.speeds)) != len(args.speeds):
        parser.error("Choose unique speeds from 0, ±0.5, ±1, ±3 m/s")
    try:
        models = selected_models(args.model)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="locomotion-quality-native-probe-v1", status="starting", pid=os.getpid(),
        started_at_utc=datetime.now(timezone.utc).isoformat(), scope="short_matched_deterministic_diagnostics",
        qualification=None, hardware_deployment_ready=False, optimizer_steps=0, formal_updates_consumed=0,
        num_envs=len(args.speeds), seed=190619, physics_device=args.device, cases={},
        stopping_anchor="position after zero applied command plus two seconds of stop warmup",
        delay_scope="fixed-suite signal_delay disabled; nominal communication profile; actual state recorded",
        state_writes_between_resets=0)
    launcher = env = writer = None
    try:
        source, config, manifest, control, receipt = make_common_config(Path(str(DEFAULT_MODELS["flat"]) + ".contract.json"), args.speeds, args.output)
        report.update(common_contract_sha256=digest(Path(str(DEFAULT_MODELS["flat"]) + ".contract.json")),
            design_preflight=receipt, command_slew=config["command_slew"], motion_limits=config["motion_limits"],
            asset_manifest_sha256=config["asset_manifest_sha256"],
            models={name: model_metadata(path, source) for name, path in models.items()},
            requested_speeds_m_s=args.speeds, runtime_config={k: config[k] for k in
                ("physics_dt", "policy_dt", "command_transport", "signal_delay", "signal_perturbations")})
        report["source_sha256"] = {name: digest(ROOT / name) for name in (
            "scripts/probe_locomotion_quality.py", "src/wheeled_tasks/chassis/env.py",
            "src/wheeled_tasks/chassis/eval_env.py", "src/wheeled_tasks/chassis/v5_control.py",
            "src/wheeled_tasks/chassis/playback_command.py", "src/wheeled_tasks/chassis/command_transport.py")}
        (args.output / "effective_contract.json").write_text(json.dumps(config, indent=2) + "\n")
        if args.preflight_only:
            report["status"] = "preflight_passed_scene_not_run"
            return 0
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, enable_cameras=False, device=args.device,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import carb.settings
        import numpy as np
        import torch
        import onnxruntime as ort
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.playback_command import advance_playback_command, attitude_from_gravity
        torch.set_num_threads(4)
        torch.manual_seed(190619)
        writer = SummaryWriter(str(args.output / "events"))
        env = FixedCaseEnv(config, manifest, control, ROOT, stage_name=config["enabled_stages"][0],
            num_envs=len(args.speeds), device=args.device, seed=190619, level=0., run_role="playback")
        env.max_episode_length = 2**40
        env.episode_limits.fill_(env.max_episode_length)
        carb.settings.get_settings().set_bool("/app/file/ignoreUnsavedOnExit", True)
        report["runtime_startup"] = deepcopy(env.startup_report)
        report["status"] = "running"
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if env.startup_report["runtime_preflight"]["status"] != "runtime_configuration_passed":
            raise RuntimeError("Native runtime preflight did not pass")
        if env.signal_delay is not None or env.usb_transport is not None:
            raise RuntimeError("Nominal comparison has an unexpected signal-age module")
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        sessions = {name: ort.InferenceSession(str(path), sess_options=options,
                    providers=["CPUExecutionProvider"]) for name, path in models.items()}
        frames, active = {}, {}
        native_update = env.robot.update

        def capture_update(dt):
            native_update(dt)
            if not active:
                return
            active["physics_index"] += 1
            values = dict(
                gravity=env.robot.data.projected_gravity_b.torch.cpu().tolist(),
                position=(env.robot.data.root_link_pose_w.torch[:, :3] - env.origins).cpu().tolist(),
                velocity=env.robot.data.root_link_lin_vel_b.torch.cpu().tolist(),
                omega=env.robot.data.root_com_ang_vel_b.torch.cpu().tolist(),
                wheel_speed=env.robot.data.joint_vel.torch[:, [env.ids[i] for i in env.v5.WHEELS]].cpu().tolist(),
                req=env.v5.requested_motor_effort[:, list(env.v5.WHEELS)].cpu().tolist(),
                bounds=env.v5.current_motor_bounds[:, list(env.v5.WHEELS)].cpu().tolist(),
                applied=env.robot.data.applied_torque.torch[:, [env.ids[i] for i in env.v5.WHEELS]].cpu().tolist(),
                commands=env.commands.cpu().tolist(), actions=env.actions.cpu().tolist(),
                action_delta=(env.actions - env.previous_actions).cpu().tolist())
            for index, lane in active["lanes"].items():
                if lane["finished"]:
                    continue
                gravity = values["gravity"][index]
                roll, pitch = attitude_from_gravity(gravity)
                position, req, bounds = values["position"][index], values["req"][index], values["bounds"][index]
                frames[index].append(dict(time_s=active["physics_index"] * env.dt, phase=lane["phase"],
                    command=values["commands"][index], requested_command=lane["requested"],
                    phase_anchor_xy_m=lane["anchor"], position_m=position, height_m=position[2],
                    velocity_m_s=values["velocity"][index], omega_rad_s=values["omega"][index],
                    projected_gravity_b=gravity, roll_deg=math.degrees(roll), pitch_deg=math.degrees(pitch),
                    tilt_deg=math.degrees(math.acos(max(-1., min(1., -gravity[2])))),
                    wheel_speed_rad_s=values["wheel_speed"][index], wheel_requested_effort_nm=req,
                    wheel_torque_limit_nm=bounds, wheel_applied_effort_nm=values["applied"][index],
                    wheel_saturated=[abs(x) >= b - 1e-6 for x, b in zip(req, bounds)],
                    wheel_limit_ratio=[abs(x) / max(b, 1e-9) for x, b in zip(req, bounds)],
                    actions_control_order=values["actions"][index], action_delta_control_order=values["action_delta"][index]))

        env.robot.update = capture_update
        began = time.monotonic()
        for model, session in sessions.items():
            report["cases"][model] = {}
            active.clear()
            frames = {index: [] for index in range(env.num_envs)}
            env.reset_suite()
            previous = env.commands.clone()
            lanes = {}
            for index, group in enumerate(env.scene_groups):
                speed = next(x for x in args.speeds if case_name(x) == group)
                steps = [(phase, vx) for phase, count, vx in trajectory(speed) for _ in range(count)]
                lanes[index] = dict(speed=speed, name=group, steps=steps, finished=False,
                    terminal=None, phase=None, anchor=None, requested=[0., 0., .305])
            transport = env.command_transport
            actual = dict(signal_delay=env.signal_delay is not None,
                legacy_usb_transport=env.usb_transport is not None,
                command_transport=transport is not None,
                command_transport_enabled=transport.enabled.cpu().tolist() if transport else None,
                actual_base_delay_ms=(transport.base_delay.cpu() * 1000).tolist() if transport else None,
                actual_loss_probability=transport.loss_probability.cpu().tolist() if transport else None)
            active.update(physics_index=0, lanes=lanes)
            for tick in range(max(len(lane["steps"]) for lane in lanes.values())):
                positions = (env.robot.data.root_link_pose_w.torch[:, :2] - env.origins[:, :2]).cpu().tolist()
                for index, lane in lanes.items():
                    if tick >= len(lane["steps"]):
                        lane["finished"] = True
                    if lane["finished"]:
                        lane["requested"] = [0., 0., .305]
                        continue
                    phase, requested_vx = lane["steps"][tick]
                    if phase != lane["phase"]:
                        lane.update(phase=phase, anchor=positions[index])
                    lane["requested"] = [requested_vx, 0., .305]
                target = previous.new_tensor([lane["requested"] for lane in lanes.values()])
                next_command = advance_playback_command(previous, target, env.policy_dt,
                    config["command_slew"], config["motion_limits"])
                previous.copy_(next_command)
                env.commands.copy_(previous)
                env.command_target.copy_(previous[:, :2])
                env.mode.copy_((previous[:, 0].abs() > .01).long())
                for index, lane in lanes.items():
                    if not lane["finished"] and lane["phase"] in ("stop_warmup", "stop") and abs(float(previous[index, 0])) > 1e-6:
                        raise RuntimeError("Stop phase began before the visible command reached zero")
                env.history.last_tick = None
                env.update_targets()
                observations = env.get_observations()
                output = session.run(None, {"obs": observations["policy"].cpu().numpy().astype(np.float32)})[0]
                if output.shape != (env.num_envs, 6) or not np.isfinite(output).all():
                    raise RuntimeError("Invalid deterministic ONNX action")
                with torch.no_grad():
                    _, _, done, extras = env.step(torch.as_tensor(output, device=args.device))
                for index, lane in lanes.items():
                    if not lane["finished"] and bool(done[index]):
                        lane.update(finished=True, terminal=terminal_reasons(extras["diagnostics"], index),
                            terminal_time_s=(tick + 1) * env.policy_dt)
                if tick % 250 == 0:
                    print(json.dumps(dict(model=model, simulated_s=(tick + 1) * env.policy_dt,
                        active_lanes=sum(not x["finished"] for x in lanes.values()), wall_s=time.monotonic() - began)), flush=True)
            active.clear()
            for index, lane in lanes.items():
                name, speed, terminal = lane["name"], lane["speed"], lane["terminal"]
                summaries = summarize_frames(frames[index])
                data_path = args.output / f"{model}_{name}.frames.jsonl"
                with data_path.open("w") as stream:
                    for row in frames[index]:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                result = dict(status="terminal" if terminal else "short_window_completed",
                    commanded_speed_m_s=speed, phases=summaries, terminal_reasons=terminal,
                    terminal_time_s=lane.get("terminal_time_s"),
                    actual_modules=actual, frames_file=str(data_path.resolve()),
                    control_timing=deepcopy(env.control_audit.last), qualification=None)
                report["cases"][model][name] = result
                for phase, values in summaries.items():
                    for metric, value in values.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"Quality/{model}/{name}/{phase}/{metric}", value, 0)
                writer.flush()
                (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps(dict(model=model, case=name, status=result["status"],
                    physical_samples=len(frames[index]), wall_s=time.monotonic() - began)), flush=True)
        report.update(status="short_matched_native_diagnostics_completed", wall_seconds=time.monotonic() - began,
            runtime_final=deepcopy(env.startup_report))
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
