"""Bounded, fresh-process ONNX playback of clean versus shared PC feedback.

The original full role is authenticated before both runtime and independent
expectations are projected. Each invocation creates one scene and one actor;
it never constructs an optimizer or grants training/deployment qualification.
"""
from copy import deepcopy
import argparse
from datetime import datetime, timezone
import faulthandler
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chassis_suite_support import digest, prepare_playback, write_json
from play_chassis_suite import validated_session
from wheeled_tasks.chassis.pc_communication import apply_pc_communication, runtime_expectation
from wheeled_tasks.chassis.preflight import design_preflight

NAMES = ("link_stand", "link_yaw_positive", "link_yaw_negative")


def observed_metrics(accumulator):
    """Unavailable observation windows carry no numerical behavior credit."""
    report = accumulator.report()
    for group in report["groups"].values():
        if group["frames"]:
            continue
        for key in (*accumulator.ERROR_NAMES, "planar_speed_mean_cm_s", "vx_rmse_m_s",
                "yaw_rmse_rad_s", "height_rmse_m", "reward_per_sim_second",
                "stand_drift_max_m", "tilt_max_deg", "closure_gap_max_m",
                "rms_motor_torque_nm", "peak_motor_torque_nm"):
            group[key] = None
        group["behavior_window_observed"] = False
    return report


def project(mapping, enabled):
    result = apply_pc_communication(mapping, enabled=enabled)
    original = result["evaluation"]["cases"][0]
    cases = []
    for name in NAMES:
        case = deepcopy(original)
        case.update(name=name, reset_seed_key="feedback_link_common_birth",
                    episode_seconds=30.)
        case.pop("communication_profile", None)
        cases.append(case)
    result["evaluation"].update(cases=cases, canonical_case_names=list(NAMES),
        episodes_per_case=2, protocol_id="shared_pc_feedback_unscored_v1")
    result.update(scene_groups=[dict(name=name, fraction=1 / 3, terrain=["flat"]) for name in NAMES],
        skill_specs={case["name"]: deepcopy(case["skill"]) for case in cases},
        flat_floor_width_m=8., playback_open_ground=False,
        evaluation_exact_cases=True, run_scope="interactive_unscored",
        evaluation_transport_mode="training_domain_prior")
    return result


def prepare(policy, output, enabled):
    source, base, manifest, control, seed, playback_receipt = prepare_playback(
        ROOT, Path(str(policy) + ".contract.json"), output / "source_projection", "flat")
    original = json.loads((output / "source_projection/CPU_preflight.json").read_text())["source_full_role"]
    design_path = Path(base["design_preflight"]["file"])
    design = json.loads(design_path.read_text())
    stage = base["design_preflight"]["stage"]
    row = design["stages"][stage]["roles"]["playback"]
    config = project(base, enabled)
    row["contract"] = project(row["contract"], enabled)
    row["invocation"]["num_envs"] = 6
    row["runtime"].update(runtime_expectation(row["contract"]))
    row["runtime"]["modules"].update(feedback_transport=enabled,
        command_transport=enabled, signal_delay=False, usb_transport=False)
    row["runtime"]["signal_delay"] = deepcopy(row["contract"]["signal_delay"])
    row["runtime"]["command_transport_model"] = "phase_aware_hold_v1" if enabled else None
    design.update(scenario="shared_pc_feedback", communication_enabled=enabled,
        allowed_changes=design["allowed_changes"] + ["feedback_transport", "command_transport",
            "signal_delay.enabled", "usb_transport.enabled", "signal_perturbations.max_delay_steps",
            "flat_floor_width_m", "runtime.feedback_consumers"],
        qualification=None, domain_qualified=None)
    path = output / "probe_expectations.json"
    write_json(path, design)
    config["design_preflight"] = dict(file=str(path.resolve()), sha256=digest(path), stage=stage)
    receipt = design_preflight(config, control, ROOT, stage=config["enabled_stages"][0],
        role="playback", seed=seed, num_envs=6)
    session, metadata = validated_session(policy, Path(str(policy) + ".contract.json"), source)
    write_json(output / "probe_contract.json", config)
    write_json(output / "CPU_preflight.json", dict(source_full_role=original, probe=receipt,
        source_playback_projection=playback_receipt,
        source_policy_sha256=digest(policy), qualification=None, scene_run=False))
    return config, manifest, control, seed, session, metadata


def run(args, config, manifest, control, seed, session, report):
    from isaaclab.app import AppLauncher
    launcher, env, writer = None, None, None
    def stage(name):
        report["initialization_stage"] = name
        write_json(args.output / "report.json", report)
        print("Feedback diagnostic: " + name, flush=True)
    faulthandler.dump_traceback_later(90., repeat=True)
    try:
        stage("launch_application")
        launcher = AppLauncher(dict(headless=True, device="cpu", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
        from wheeled_tasks.chassis.playback_command import advance_playback_command
        stage("initialize_native_cpu_threads")
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        stage("construct_native_scene")
        env = FixedCaseEnv(config, manifest, control, ROOT,
            stage_name=config["enabled_stages"][0], num_envs=6,
            device="cpu", seed=seed, level=1., run_role="playback")
        writer = SummaryWriter(str(args.output / "events"))
        stage("reset_native_scene")
        observations = env.reset_suite()
        previous, alive = env.commands.clone(), torch.ones(6, dtype=torch.bool)
        metrics = EpisodeMetrics(env.scene_groups, "cpu", env.policy_dt, warmup_seconds=2.)
        phase_metrics = {name: EpisodeMetrics(env.scene_groups, "cpu", env.policy_dt)
            for name in ("steady_yaw_3_to_9s", "settled_stop_10_6_to_12s")}
        report.update(status="running", startup_report=env.startup_report,
            scene_groups=env.scene_groups, samples=[], terminal={})
        for tick in range(600):
            t = tick * env.policy_dt
            requested = previous.new_tensor([[0., (0. if name == NAMES[0] else
                2 * math.pi * (1 if name == NAMES[1] else -1)) if 1 <= t < 9 else 0., .305]
                for name in env.scene_groups])
            applied = advance_playback_command(previous, requested, env.policy_dt,
                config["command_slew"], config.get("motion_limits"))
            env.command_acceleration.copy_((applied[:, 0] - previous[:, 0]) / env.policy_dt)
            previous.copy_(applied)
            env.commands.copy_(applied)
            env.command_target.copy_(applied[:, :2])
            env.raw_height.copy_(applied[:, 2])
            env.mode.copy_((applied[:, :2].abs().sum(-1) > .01).long())
            env.history.last_tick = None
            env.update_targets()
            env.update_command_reference()
            observations = env.get_observations()
            actions = session.run(None, {"obs": observations["policy"].cpu().numpy().astype(np.float32)})[0]
            if actions.shape != (6, 6) or not np.isfinite(actions).all():
                raise RuntimeError("Invalid deterministic ONNX output")
            active = alive.clone()
            with torch.no_grad():
                observations, _, done, extras = env.step(torch.from_numpy(actions))
            metrics.observe(extras["diagnostics"], active)
            for name, (start, end) in zip(phase_metrics, ((3., 9.), (10.6, 12.))):
                if start <= t < end:
                    phase_metrics[name].observe(extras["diagnostics"], active)
            if tick % 10 == 0 or bool((active & done).any()):
                state = extras["diagnostics"]
                report["samples"].append(dict(tick=tick + 1, time_s=(tick + 1) * .02,
                    active=active.tolist(), command=applied.tolist(),
                    velocity_m_s=state["velocity"].tolist(), omega_rad_s=state["omega"].tolist(),
                    position_m=state["position"].tolist(), height_m=state["height"].tolist(),
                    gravity=state["gravity"].tolist(), motor_effort_nm=state["motor_effort"].tolist(),
                    feedback_age_s=env.feedback_transport.age_s.tolist() if env.feedback_transport else [0.] * 6,
                    feedback_enabled=env.feedback_transport.enabled.tolist() if env.feedback_transport else [False] * 6,
                    command_enabled=env.command_transport.enabled.tolist() if env.command_transport else [False] * 6,
                    command_base_delay_s=env.command_transport.base_delay.tolist() if env.command_transport else [0.] * 6,
                    command_can_phase_s=env.command_transport.can_phase.tolist() if env.command_transport else [0.] * 6,
                    command_lost_packets=env.command_transport.lost.tolist() if env.command_transport else [0.] * 6,
                    command_age_s=env.command_transport.age.tolist() if env.command_transport else [0.] * 6))
                for tag, value in extras["log"].items():
                    if tag.startswith("/transport/"):
                        writer.add_scalar(tag, value, tick + 1)
            for lane in (active & done).nonzero(as_tuple=False).flatten().tolist():
                report["terminal"][str(lane)] = dict(time_s=(tick + 1) * .02,
                    case=env.scene_groups[lane], nonwheel_body_names=[env.robot.body_names[j]
                        for j in env.nonwheel_ids if float(env.contact_force[lane, j].norm()) > 5.],
                    reasons=[name for name, mask in extras["diagnostics"]["reasons"].items() if bool(mask[lane])])
            alive &= ~done
            if tick % 50 == 0:
                write_json(args.output / "report.json", report)
            if not bool(alive.any()):
                break
        report.update(status="completed_bounded_unscored", native_metrics=observed_metrics(metrics),
            phase_metrics={name: observed_metrics(accumulator) for name, accumulator in phase_metrics.items()},
            right_censored=alive.tolist(), policy_ticks=tick + 1, physics_ticks=(tick + 1) * 4)
    except BaseException as error:
        report.update(status="failed_unscored", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        write_json(args.output / "report.json", report)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--communication", choices=("clean", "prior"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config, manifest, control, seed, session, metadata = prepare(
        args.onnx, args.output, args.communication == "prior")
    report = dict(status="CPU_design_identity_passed_scene_not_run", qualification=None,
        at_utc=datetime.now(timezone.utc).isoformat(), source_policy_sha256=digest(args.onnx),
        source_checkpoint_sha256=metadata.get("source_checkpoint_sha256"),
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_files_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(
            (ROOT / "src/wheeled_tasks/chassis").glob("*.py"))}, script_sha256=digest(Path(__file__)),
        communication=args.communication, run_scope="interactive_unscored", policy_hz=50, physics_pd_hz=200,
        formal_updates_consumed=0, optimizer_constructed=False, actor_count=1,
        independent_process_per_model_profile=True, max_sim_seconds=12.,
        scope="Two replicas per zero-speed and +/-1rps case; no recovery or hardware qualification")
    write_json(args.output / "report.json", report)
    if not args.preflight_only:
        run(args, config, manifest, control, seed, session, report)


if __name__ == "__main__":
    main()
