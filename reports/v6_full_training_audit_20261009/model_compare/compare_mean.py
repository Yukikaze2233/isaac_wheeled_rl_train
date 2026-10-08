#!/usr/bin/env python3
"""Two REC mean actors, one native two-lane scene; unscored bounded smoke."""
from __future__ import annotations

import argparse
from collections import Counter
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
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
from chassis_suite_support import control_ownership, digest, project_scene, write_json
from play_chassis_suite import validated_session


def project_pair(mapping, seed, wide_floor=False):
    """Use the existing suite projection for each lane, independently per source."""
    left = project_scene(mapping, "fallen_left", seed)
    flat = project_scene(mapping, "flat", seed)
    cases = left["evaluation"]["cases"] + flat["evaluation"]["cases"]
    result = deepcopy(left)
    result["evaluation"].update(cases=cases, canonical_case_names=[c["name"] for c in cases],
        protocol_id="REC_two_lane_mean_smoke_unscored")
    result["scene_groups"] = [dict(name=c["name"], fraction=.5, terrain=[c["terrain"]]) for c in cases]
    result["skill_specs"] = {c["name"]: deepcopy(c["skill"]) for c in cases}
    if wide_floor:
        # Existing native collider accepts 4-8m. Extend only the observation
        # runway so lateral drift does not hide the later yaw request.
        result["flat_floor_width_m"] = 8.
    return result


def prepare(contract_path, output, wide_floor=False):
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import design_preflight

    source, manifest = preflight(contract_path)
    if tuple(source[k] for k in ("physics_dt", "policy_dt", "actor_dim", "critic_dim", "action_dim", "history_length")) != (.005, .02, 35, 81, 6, 1):
        raise ValueError("Requires frozen REC 200/50 Hz, 35/81/6 single-frame ABI")
    if (source["position_iterations"], source["velocity_iterations"], source["closure_gap_termination_m"]) != (64, 32, .003):
        raise ValueError("Requires original REC solver and closure guard")
    binding = source["design_preflight"]
    design_path = ROOT / binding["file"]
    if digest(design_path) != binding["sha256"]:
        raise ValueError("Source design hash mismatch")
    design = json.loads(design_path.read_text())
    row = design["stages"][binding["stage"]]["roles"]["full"]
    seed_rule = row["invocation"]["seed"]
    if seed_rule["rule"] != "fixed":
        raise ValueError("Requires original fixed evaluation seed")
    seed = seed_rule["value"]
    full = deepcopy(source)
    full["evaluation"] = select_evaluation_cases(source["evaluation"],
        [c["name"] for c in row["contract"]["evaluation"]["cases"]])
    full["evaluation"]["seed"] = seed
    full = fixed_suite_contract(full)
    control = load_chassis_control(ROOT / source["control_math_source"])
    stage = source["enabled_stages"][0]
    original = design_preflight(full, control, ROOT, stage=stage, role="full", seed=seed,
        num_envs=row["invocation"]["num_envs"])
    config = project_pair(full, seed, wide_floor)
    expected = deepcopy(row)
    expected["contract"] = project_pair(row["contract"], seed, wide_floor)
    expected["invocation"]["num_envs"] = 2
    projection = {key: deepcopy(value) for key, value in design.items() if key != "stages"}
    # Existing playback validator requires this exact opt-in scope. The driver
    # remains headless and bounded; it does not alter the validator or grades.
    projection.update(run_scope="interactive_unscored", qualification=None,
        domain_qualified=None, source_contract_sha256=digest(contract_path),
        source_design_sha256=binding["sha256"], source_role="full",
        review_basis="Separately project authenticated full runtime and frozen full-role expectations with existing suite project_scene; no PPO, qualification or campaign credit.",
        allowed_changes=["evaluation", "scene_groups", "skill_specs", "auto_reset", "record_diagnostics",
            "monitor_applied_effort", "flat_triangle_mesh", "flat_half_length_m", "invocation.num_envs",
            *(["flat_floor_width_m"] if wide_floor else [])],
        stages={binding["stage"]: {"roles": {"playback": expected}}})
    output.mkdir(parents=True, exist_ok=False)
    path = output / "playback_expectations.json"
    write_json(path, projection)
    config["design_preflight"] = dict(file=str(path.resolve()), sha256=digest(path), stage=binding["stage"])
    receipt = design_preflight(config, control, ROOT, stage=stage, role="playback", seed=seed, num_envs=2)
    write_json(output / "playback_contract.json", config)
    write_json(output / "CPU_preflight.json", dict(status="CPU_design_preflight_passed_scene_not_run",
        source_full_role=original, playback=receipt, qualification=None, scene_run=False))
    (output / "source").mkdir()
    (output / "source" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return source, config, manifest, control, seed, receipt


def flat_request(t):
    # A short *challenge*, not a trained-domain or comprehensive speed grade.
    # One positive translation plateau and one positive yaw plateau, each followed by stop.
    if 1. <= t < 5.:
        return [2., 0., .305], "forward_2"
    if 7. <= t < 10.:
        return [0., 2. * math.pi, .305], "yaw_2pi"
    return [0., 0., .305], "stand" if t < 1. else "stop"


def commands(env, previous, t):
    from wheeled_tasks.chassis.playback_command import advance_playback_command
    proposed = previous.new_tensor([flat_request(t)[0] if g == "flat" else [0., 0., .305]
                                    for g in env.scene_groups])
    if env.recovery is not None:
        held = env.recovery.enabled & ~env.recovery.fsm.motion_released
        proposed[held, :2] = 0.
        proposed[held, 2] = .305
    applied = advance_playback_command(previous, proposed, env.policy_dt,
        env.cfg["command_slew"], env.cfg.get("motion_limits"))
    env.command_acceleration.copy_((applied[:, 0] - previous[:, 0]) / env.policy_dt)
    previous.copy_(applied)
    env.commands.copy_(applied)
    env.command_target.copy_(applied[:, :2])
    env.raw_height.copy_(applied[:, 2])
    if env.recovery is not None:
        env.recovery.followup_command.copy_(applied)
    env.mode.copy_((applied[:, :2].abs().sum(-1) > .01).long())
    env.history.last_tick = None
    env.update_targets()
    env.update_command_reference()
    return env.get_observations(), proposed


def run(args, config, manifest, control, seed, candidates, report):
    from isaaclab.app import AppLauncher
    launcher, env, writer = None, None, None
    try:
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
        from wheeled_tasks.chassis.playback_command import attitude_from_gravity
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        env = FixedCaseEnv(config, manifest, control, ROOT, stage_name=config["enabled_stages"][0],
            num_envs=2, device=args.device, seed=seed, level=1., run_role="playback")
        writer = SummaryWriter(str(args.output / "events"))  # event file only; no TB server
        report.update(status="running", startup_report=env.startup_report, scene_groups=env.scene_groups,
            body_names=env.robot.body_names, terminal_behavior="First terminal frame counted; later frames excluded, auto_reset=false",
            native_episode_limits_unchanged=True)
        write_json(args.output / "report.json", report)
        for index, (session, metadata, identity) in enumerate(candidates):
            torch.manual_seed(seed)
            observations = env.reset_suite()  # native birth-only reset, original reset/USB RNG reseeding
            previous = env.commands.clone()
            alive = torch.ones(2, dtype=torch.bool, device=args.device)
            metrics = EpisodeMetrics(env.scene_groups, args.device, env.policy_dt, warmup_seconds=0.)
            # Reuse the native accumulator for common time windows, rather than
            # compare unequal whole-episode denominators or sparse trace means.
            windows = {key: EpisodeMetrics(env.scene_groups, args.device, env.policy_dt)
                for key in ("flat_0_5s", "flat_0_7s", "flat_brake_5_7s", "flat_zero_6p34_7s", "flat_yaw_7_10s", "flat_stop_11p58_12s")}
            outcome = dict(**identity, status="running", samples=[], native_metrics=None,
                ownership_frames={g: Counter() for g in env.scene_groups},
                command_exposure_frames={g: Counter() for g in env.scene_groups}, first_pure_rl_s={},
                first_motion_release_s={}, first_motion_release_after_pure_rl_s={},
                applied_command_abs_peak={g: [0., 0.] for g in env.scene_groups},
                terminal={}, right_censored={}, qualification=None)
            started = time.monotonic()
            for tick in range(args.steps):
                if not launcher.app.is_running():
                    raise RuntimeError("Native app closed")
                observations, requested = commands(env, previous, tick * env.policy_dt)
                observed = observations["policy"].detach().cpu().numpy().astype(np.float32)
                before = [control_ownership(env.recovery, lane) for lane in range(2)]
                actions = session.run(None, {"obs": observed})[0]
                if actions.shape != (2, 6) or not np.isfinite(actions).all():
                    raise RuntimeError("Invalid deterministic actor output")
                active = alive.clone()
                with torch.no_grad():
                    observations, _, done, extras = env.step(torch.as_tensor(actions, device=args.device))
                diagnostic = extras["diagnostics"]
                metrics.observe(diagnostic, active)
                t = tick * env.policy_dt
                flat_mask = active & active.new_tensor([g == "flat" for g in env.scene_groups])
                window_times = {"flat_0_5s": t < 5., "flat_0_7s": t < 7.,
                    "flat_brake_5_7s": 5. <= t < 7., "flat_zero_6p34_7s": 6.34 <= t < 7.,
                    "flat_yaw_7_10s": 7. <= t < 10., "flat_stop_11p58_12s": 11.58 <= t < 12.}
                for key, enabled in window_times.items():
                    if enabled:
                        windows[key].observe(diagnostic, flat_mask)
                now = (tick + 1) * env.policy_dt
                for lane, name in enumerate(env.scene_groups):
                    if not bool(active[lane]):
                        continue
                    owner = control_ownership(env.recovery, lane)
                    outcome["ownership_frames"][name][before[lane]["ownership"]] += 1
                    label = flat_request(tick * env.policy_dt)[1] if name == "flat" else "native_recovery_stand"
                    outcome["command_exposure_frames"][name][label] += 1
                    if owner["pure_rl"] and name not in outcome["first_pure_rl_s"]:
                        outcome["first_pure_rl_s"][name] = now
                    if owner["motion_released"] and name not in outcome["first_motion_release_s"]:
                        outcome["first_motion_release_s"][name] = now
                    if owner["pure_rl"] and owner["motion_released"] and name not in outcome["first_motion_release_after_pure_rl_s"]:
                        outcome["first_motion_release_after_pure_rl_s"][name] = now
                    for axis in range(2):
                        outcome["applied_command_abs_peak"][name][axis] = max(outcome["applied_command_abs_peak"][name][axis], abs(float(previous[lane, axis])))
                    gravity = diagnostic["gravity"][lane].cpu().tolist()
                    roll, pitch = attitude_from_gravity(gravity)
                    row = dict(tick=tick + 1, time_s=now, lane=lane, case=name,
                        requested_command=requested[lane].cpu().tolist(), applied_command=previous[lane].cpu().tolist(),
                        actor_command_35d_first4=observed[lane, :4].tolist(),
                        velocity_m_s=diagnostic["velocity"][lane].cpu().tolist(),
                        local_position_m=diagnostic["position"][lane].cpu().tolist(),
                        position_w_m=(diagnostic["position"][lane] + env.origins[lane]).cpu().tolist(),
                        omega_rad_s=diagnostic["omega"][lane].cpu().tolist(),
                        height_m=float(diagnostic["height"][lane]), roll_deg=math.degrees(roll), pitch_deg=math.degrees(pitch),
                        gap_mm=float(diagnostic["gap"][lane]) * 1000., native_control=owner,
                        native_control_before_policy=before[lane],
                        wheel_force_n=env.contact_force[lane, env.wheel_ids].norm(dim=-1).cpu().tolist(),
                        touching_body_names=[n for n, f in zip(env.robot.body_names,
                            env.contact_force[lane].norm(dim=-1).cpu().tolist()) if f > 5.],
                        nonwheel_touching_body_names=[env.robot.body_names[j] for j in env.nonwheel_ids
                            if float(env.contact_force[lane, j].norm()) > 5.],
                        motor_effort_nm=diagnostic["motor_effort"][lane].cpu().tolist(),
                        full_chain_success=bool(diagnostic.get("recovery_true_success", diagnostic["success"])[lane]),
                        done=bool(done[lane]), terminated=bool(diagnostic["terminated"][lane]))
                    for field in ("recovery_rl_stable_s", "recovery_followup_s", "recovery_failure_code", "recovery_handover_count"):
                        if field in diagnostic:
                            row[field] = float(diagnostic[field][lane])
                    if tick % 10 == 0 or row["done"]:
                        outcome["samples"].append(row)
                        for tag, value in dict(vx_m_s=row["velocity_m_s"][0], yaw_rad_s=row["omega_rad_s"][2],
                            height_m=row["height_m"], roll_deg=row["roll_deg"], pitch_deg=row["pitch_deg"],
                            closure_gap_mm=row["gap_mm"], stable_peak_s=owner["stable_seconds"]).items():
                            writer.add_scalar(f"MeanSmoke/{index}/{name}/{tag}", value, tick + 1)
                    if row["done"]:
                        outcome["terminal"][name] = row
                alive &= ~done
                if tick % 50 == 0:
                    write_json(args.output / f"candidate_{index:02d}_partial.json", outcome)
                if not bool(alive.any()):
                    break
            outcome.update(status="completed_bounded_unscored", native_metrics=metrics.report(),
                common_window_metrics={key: value.report()["groups"]["flat"] for key, value in windows.items()},
                policy_ticks=tick + 1, physical_ticks=(tick + 1) * 4, wall_seconds=time.monotonic() - started,
                right_censored={name: bool(alive[lane]) for lane, name in enumerate(env.scene_groups)})
            write_json(args.output / f"candidate_{index:02d}.json", outcome)
            report["candidates"].append({key: value for key, value in outcome.items() if key != "samples"})
            write_json(args.output / "report.json", report)
        report["status"] = "completed_bounded_unscored"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "report.json", report)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return int(report["status"] == "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=600, help="Per actor; sum across actors must be <=1200")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wide-floor8", action="store_true", help="Independently frozen 8m diagnostic runway; no changed task/guard predicates")
    args = parser.parse_args()
    if not 1 <= len(args.onnx) <= 2 or not 1 <= args.steps or args.steps * len(args.onnx) > 1200:
        parser.error("One or two actors, at most 1200 total policy ticks")
    contracts = [Path(str(p) + ".contract.json") for p in args.onnx]
    if len({digest(p) for p in contracts}) != 1:
        parser.error("REC candidates must have byte-identical source contracts; no silent model/source migration")
    source, config, manifest, control, seed, receipt = prepare(contracts[0], args.output, args.wide_floor8)
    candidates = []
    for policy, contract in zip(args.onnx, contracts):
        session, metadata = validated_session(policy, contract, source)
        test = session.run(None, {"obs": np.zeros((2, 35), dtype=np.float32)})[0]
        if test.shape != (2, 6) or not np.isfinite(test).all():
            raise ValueError("Two-lane CPU ORT ABI preflight failed")
        identity = dict(policy_file=str(policy.resolve()), policy_sha256=digest(policy),
            source_checkpoint_sha256=metadata.get("source_checkpoint_sha256"),
            source_contract_sha256=digest(contract), actor_task_update=metadata.get("checkpoint_infos", {}).get("task_actor_update"))
        candidates.append((session, metadata, identity))
    report = dict(status="CPU_identity_design_ORT_preflight_passed_scene_not_run",
        script_sha256=digest(Path(__file__)), started_at=datetime.now(timezone.utc).isoformat(),
        design_preflight=receipt, scene_run=False, steps_per_actor=args.steps,
        max_total_policy_ticks=args.steps * len(candidates), max_sim_seconds_per_actor=args.steps * .02,
        selected_cases=[c["name"] for c in config["evaluation"]["cases"]],
        identities=[row[2] for row in candidates], candidates=[], qualification=None, formal_updates_consumed=0,
        optimizer_constructed=False, deterministic_mean=True, inference_backend="onnxruntime_CPU",
        scope="Two-lane native smoke only; no full acceptance, no model comprehensive ranking, no hardware qualification",
        omitted_cases=["fallen_right", "fallen_belly", "fallen_back", "reverse_translation", "reverse_yaw", "rough", "stairs", "jump"],
        horizon_rule="Native episode/task/guard predicates unchanged. Unfinished episodes at short observation horizon are right-censored, not failed.",
        flat_challenge_note="+2 m/s and +2pi rad/s requested outside REC actual training range; applied slew/motion projection may reduce them",
        command_schedule=[[0, 1, 0, 0], [1, 5, 2, 0], [5, 7, 0, 0], [7, 10, 0, 2 * math.pi], [10, 12, 0, 0]],
        source_control=source["control_math_source"], command_slew=config["command_slew"], motion_limits=config.get("motion_limits"))
    report.update(diagnostic_runway_width_m=config.get("flat_floor_width_m", 4.),
        diagnostic_boundary_x_m=config["flat_half_length_m"] - 4.,
        diagnostic_boundary_y_m=config.get("flat_floor_width_m", 4.) / 2 - .3,
        wider_runway_purpose="Observe later yaw exposure; does not excuse previous lateral displacement or confer qualification",
        motion_release_diagnostic="Raw FSM can have release=true before recovery starts; only first_motion_release_after_pure_rl_s is an effective actor release timestamp")
    write_json(args.output / "report.json", report)
    print(json.dumps(dict(status=report["status"], output=str(args.output), policies=report["identities"])), flush=True)
    if args.preflight_only:
        return 0
    report["scene_run"] = True
    return run(args, config, manifest, control, seed, candidates, report)


if __name__ == "__main__":
    raise SystemExit(main())
