#!/usr/bin/env python3
"""Matched train-role reset/zero-effort/capture probe; no optimizer or qualification."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_recovery_takeover import digest, require, write_report, load_actor


def authenticate(args):
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
    cfg, manifest = preflight(args.contract)
    binding = cfg["design_preflight"]
    design_path = ROOT / binding["file"]
    require(digest(design_path) == binding["sha256"], "Independent design SHA differs")
    row = json.loads(design_path.read_text())["stages"][binding["stage"]]["roles"]["train"]["invocation"]
    rule = row["seed"]
    seed = rule if type(rule) is int else rule["root_seed"]
    control = load_chassis_control(ROOT / cfg["control_math_source"])
    stage = cfg["enabled_stages"][0]
    receipt = design_preflight(cfg, control, ROOT, stage=stage, role="train",
        num_envs=row["num_envs"], seed=seed, block_index=0)
    require((cfg["physics_dt"], cfg["policy_dt"], cfg["actor_dim"], cfg["critic_dim"]) ==
        (.005, .02, 35, 81), "Control/observation ABI differs")
    require(cfg.get("nonstandard_start", {}).get("enabled") is True, "Birth feature absent")
    require(cfg.get("release_stop", {}).get("enabled") is True, "Release-stop feature absent")
    source_path = checkpoint_contract_path(args.checkpoint)
    source = json.loads(source_path.read_text())
    for key in ("asset_manifest_sha256", "control_math_sha256", "physics_dt", "policy_dt",
                "actor_dim", "critic_dim", "action_dim", "actor_observation_source",
                "policy_action_order", "policy_architecture", "v5_control"):
        require(source.get(key) == cfg.get(key), f"Actor inference ABI differs: {key}")
    provenance = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        checkpoint_contract=str(source_path), checkpoint_contract_sha256=digest(source_path),
        checkpoint_learning_updates=checkpoint_update_count(args.checkpoint),
        actor_only_deterministic_inference=True)
    return cfg, manifest, control, stage, seed, row["num_envs"], receipt, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--actor-update", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    require(args.actor_update >= 0 and math.isfinite(args.seconds) and 2 <= args.seconds <= 10,
        "Invalid engineering clock/duration")
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    report = dict(schema="nonstandard-start-native-probe-v1", status="starting",
        scope="engineering_only_no_PPO_no_qualification", formal_updates_consumed=0,
        optimizer_steps=0, qualification=None, started_at_utc=datetime.now(timezone.utc).isoformat(),
        target_contract=str(args.contract), target_contract_sha256=digest(args.contract),
        forced_engineering_actor_update=args.actor_update, requested_physics_seconds=args.seconds,
        randomization_or_delay_overrides=[])
    write_report(path, report)
    launcher = env = payment_audit = None
    try:
        cfg, manifest, control, stage, seed, count, receipt, provenance = authenticate(args)
        report.update(design_preflight=receipt, checkpoint=provenance, num_envs=count, run_role="train", seed=seed)
        if args.preflight_only:
            report["status"] = "preflight_passed_scene_not_run"
            return 0
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, enable_cameras=False, device="cuda:0",
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from wheeled_tasks.chassis.env import ChassisEnv
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        env = ChassisEnv(cfg, manifest, control, ROOT, stage_name=stage, num_envs=count,
            device="cuda:0", level=1., seed=seed, run_role="train")
        require(env.startup_report["runtime_preflight"]["status"] == "runtime_configuration_passed",
            "Runtime design differs")
        actor, before = load_actor(env, args.checkpoint, cfg, provenance, seed)
        if "locomotion_posture" in cfg.get("dense_tracking", {}):
            from locomotion_payment_audit import LocomotionPaymentAudit
            payment_audit = LocomotionPaymentAudit(cfg)
            payment_audit.install()
        env.task_actor_update = env.stage_actor_update = env.global_actor_update = args.actor_update
        env.reset(torch.arange(count, device=env.device))
        birth = env.nonstandard_start
        release = env.skills.release_stop
        require(release is not None and release.enabled, "Release-stop runtime missing")
        extended_ids = (release.active & env.release_stop_terrain).nonzero(as_tuple=False).flatten().tolist()
        for index in extended_ids:
            surfaces = env.surfaces[index]
            require(min(s.x0 for s in surfaces) == -20. and max(s.x1 for s in surfaces) == 20.,
                    "Release-stop terrain corridor did not expand")
        friction = env.startup_report["contact_domain"]
        require(.3 <= friction["min_effective_friction"] + 1e-6
                and friction["max_effective_friction"] <= 1.200001
                and friction["low_grip_fraction"] > 0, "Low-grip randomization absent")
        selected = birth.selected.clone()
        alive_initial = selected.clone()
        require(int(selected.sum()) > 0, "No nonstandard births sampled")
        observations = env.get_observations()
        counts = dict(initial_selected=int(selected.sum()), zero_active_lane_frames=0,
            passive_spring_lane_frames=0, excluded_learning_lane_frames=0,
            released_learning_lane_frames=0, capture_lane_frames=0,
            capture_released_initial_lanes=0, terminations=0,
            release_requested_lane_frames=0, release_stationary_lane_frames=0,
            release_settled_lane_frames=0, extended_terrain_lanes=len(extended_ids),
            release_boundary_truncations=0)
        max_release_command_rates = torch.zeros(2, device=env.device)
        max_release_tilt = 0.
        initial_gap = float(env.closure_gap().max())
        require(initial_gap < .001, "Reset bank introduced a closure gap")
        max_gap = initial_gap
        first_release = None
        began = time.monotonic()
        steps = math.ceil(args.seconds / env.policy_dt)
        for tick in range(1, steps + 1):
            held = birth.holding.clone()
            capturing = birth.capturing.clone()
            previous_commands = env.commands[:, :2].clone()
            with torch.inference_mode():
                actions = actor(observations["policy"])
                observations, reward, done, extras = env.step(actions)
            if payment_audit is not None:
                payment_audit.observe(extras, env.policy_dt)
            # A terminated initial episode replaced by a nominal reset is not
            # evidence that its original capture window actually expired.
            alive_initial &= ~done
            require(all(torch.isfinite(x).all() for x in (*observations.values(), reward)), "Nonfinite native state")
            require(tuple(observations["policy"].shape) == (count, 35)
                    and tuple(observations["critic"].shape) == (count, 81), "Observation ABI changed")
            timing = env.control_audit.last
            require(timing["pd_updates"] == timing["feedback_refreshes"] == 4
                    and timing["target_decodes"] == 1, "Fresh feedback control differs")
            if held.any():
                require(env.torque[held].eq(0).all(), "Active motor effort survived hold")
                require(~extras["policy_learning_mask"][held].any(), "Forced hold entered PPO population")
                require(env.actions[held & ~done].eq(0).all(), "Forced action survived history")
                spring = env.robot.data.applied_torque.torch[:, env.spring_ids]
                counts["zero_active_lane_frames"] += int(held.sum())
                counts["passive_spring_lane_frames"] += int((spring[held].abs().sum(-1) > 0).sum())
                counts["excluded_learning_lane_frames"] += int((held & ~extras["policy_learning_mask"]).sum())
            active = alive_initial & ~held
            counts["released_learning_lane_frames"] += int((active & extras["policy_learning_mask"]).sum())
            if first_release is None and (active & extras["policy_learning_mask"]).any():
                first_release = (tick - 1) * env.policy_dt
            now_capture = birth.capturing
            require(env.commands[now_capture, 0].abs().max() <= 1.000001 if now_capture.any() else True,
                "Capture forward command escaped cap")
            require(env.commands[now_capture, 1].abs().max() <= 1.000001 if now_capture.any() else True,
                "Capture yaw command escaped cap")
            counts["capture_lane_frames"] += int(capturing.sum())
            counts["capture_released_initial_lanes"] = int((alive_initial & ~now_capture).sum())
            counts["terminations"] += int(done.sum())
            diagnostics = extras["diagnostics"]
            releasing = release.active & ~done
            if releasing.any():
                rates = (env.commands[:, :2] - previous_commands).abs() / env.policy_dt
                max_release_command_rates = torch.maximum(max_release_command_rates, rates[releasing].amax(0))
                require((rates[releasing] <= rates.new_tensor([1.5, 4.]) + 1e-4).all(),
                        "Release-stop bypassed sealed command acceleration")
                elapsed = env.episode_length_buf * env.policy_dt
                requested = releasing & (elapsed >= release.release_at)
                require(env.command_target[requested].eq(0).all(), "Released target was not exactly zero")
                max_release_tilt = max(max_release_tilt,
                    float((-diagnostics["gravity"][releasing, 2]).clamp(-1., 1.).acos().max()))
            counts["release_requested_lane_frames"] += int(diagnostics["release_requested"].sum())
            counts["release_stationary_lane_frames"] += int(diagnostics["release_stop_stationary_window"].sum())
            counts["release_settled_lane_frames"] += int(diagnostics["release_stop_settled"].sum())
            counts["release_boundary_truncations"] += int(
                (release.active & diagnostics["reasons"]["boundary"]).sum())
            max_gap = max(max_gap, float(env.closure_gap().max()))
        require(all(torch.equal(before[n], v.detach().cpu()) for n, v in actor.state_dict().items()),
            "Engineering probe changed actor")
        require(counts["zero_active_lane_frames"] > 0 and counts["passive_spring_lane_frames"] > 0
                and counts["released_learning_lane_frames"] > 0 and first_release is not None
                and math.isclose(first_release, .2, abs_tol=1e-7),
            "Passive hold or RL handoff not observed")
        require(counts["capture_released_initial_lanes"] > 0, "Capture window failed to expire")
        if args.seconds >= 10.:
            require(counts["release_requested_lane_frames"] > 0
                    and counts["release_stationary_lane_frames"] > 0,
                    "Release or unbiased stationary observation window absent")
        if payment_audit is not None:
            require(payment_audit.steps == steps, "Quality payment probe omitted native steps")
            require(payment_audit.counts["stand_still_lin_vel"] > 0,
                    "Stationary L1 cost was never numerically paid")
            report["locomotion_quality_payments"] = payment_audit.report()
        report.update(status="passed", policy_steps=steps, counts=counts,
            surviving_initial_selected_lanes=int(alive_initial.sum()),
            first_RL_release_seconds=first_release, reset_max_closure_gap_m=initial_gap,
            observed_max_closure_gap_m=max_gap, wall_seconds=time.monotonic() - began,
            actor_unchanged=True, startup=deepcopy(env.startup_report),
            contact_randomization=deepcopy(friction),
            observed_release_command_rates_m_s2_rad_s2=max_release_command_rates.tolist(),
            observed_release_max_tilt_rad=max_release_tilt,
            stop_quality_assessed=False)
        return 0
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        return 1
    finally:
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(path, report)
        if payment_audit is not None:
            payment_audit.restore()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
