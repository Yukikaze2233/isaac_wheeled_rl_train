#!/usr/bin/env python3
"""Run the full frozen scene with a fixed actor; report recovery engineering evidence.

This bounded probe consumes no PPO updates and grants no skill acceptance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_sources():
    """Bind every local chassis module before Python imports the execution path."""
    files = [*sorted((ROOT / "src/wheeled_tasks/chassis").glob("*.py")),
             ROOT / "src/wheeled_tasks/agents/v40_ppo_cfg.py", ROOT / "scripts/train_chassis.py",
             Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def source_changes(before, after):
    return [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]


def select_engineering_cases(expected_cases, *, recovery_only, stand_only):
    """Select from the independent frozen design, never the runtime catalogue."""
    selected = []
    for case in expected_cases:
        skill = case.get("skill", {})
        recovery = case.get("recovery_case") is True and skill.get("kind") == "recovery_handoff"
        if recovery_only and not recovery:
            continue
        if stand_only:
            command = skill.get("recovery_followup_command")
            if not recovery or not isinstance(command, list) or len(command) != 3 or command[:2] != [0., 0.]:
                continue
        selected.append(case["name"])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Engineering design subset must have unique nonempty case names")
    return selected


def project_evaluation(settings, names, repeats):
    """Only select cases/coverage and repetition count; do not change physics."""
    chosen = set(names)
    result = deepcopy(settings)
    if chosen - {case["name"] for case in settings["cases"]}:
        raise ValueError("Engineering subset contains cases absent from source settings")
    result["cases"] = [case for case in result["cases"] if case["name"] in chosen]
    result["episodes_per_case"] = repeats
    for key, values in list(result.items()):
        if key.endswith("_case_names"):
            if not isinstance(values, list):
                raise ValueError(f"Malformed frozen case coverage: {key}")
            result[key] = [name for name in values if name in chosen]
    return result


def project_scenes(scenes, names):
    chosen = set(names)
    result = [deepcopy(scene) for scene in scenes if scene["name"] in chosen]
    if {scene["name"] for scene in result} != chosen:
        raise ValueError("Frozen scene groups do not cover the requested subset")
    for scene in result:
        scene["fraction"] = 1. / len(result)
    return result


def materialize(args):
    from train_chassis import preflight
    from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight

    plan = resolve_plan(json.loads(args.plan.read_text()),
                        lambda name: json.loads((ROOT / name).read_text()))
    recipe = next(row for row in plan["stages"] if row["name"] == args.stage)
    source = stage_contract({}, plan, recipe, 20000)
    path = args.output / "source_contract.json"
    path.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    source, manifest = preflight(path)
    control = load_chassis_control(ROOT / source["control_math_source"])
    if args.num_envs is not None:
        # Unmodified formal train role; only the bounded zero-action execution
        # loop is an engineering probe. No actor, optimizer or PPO is invoked.
        seed = plan["training_seed"]
        receipt = design_preflight(source, control, ROOT, stage=recipe["kind"],
                                   role="train", seed=seed, num_envs=args.num_envs, block_index=0)
        (args.output / "contract.json").write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
        return source, manifest, control, recipe, seed, args.num_envs, "train", receipt

    config = fixed_suite_contract(source)
    seed = config["evaluation"]["seed"]
    original_count = len(config["evaluation"]["cases"]) * config["evaluation"]["episodes_per_case"]
    # Reject a materialization drift before deriving any subset expectations.
    original_receipt = design_preflight(config, control, ROOT, stage=recipe["kind"],
        role="full", seed=seed, num_envs=original_count)
    binding = config["design_preflight"]
    source_design_path = ROOT / binding["file"]
    if digest(source_design_path) != binding["sha256"]:
        raise ValueError("Independent source design changed during probe materialization")
    design = json.loads(source_design_path.read_text())
    source_row = design["stages"][binding["stage"]]["roles"]["full"]
    names = select_engineering_cases(source_row["contract"]["evaluation"]["cases"],
        recovery_only=args.recovery_only, stand_only=args.stand_only)
    count = len(names) * args.repeats
    # The expected row is independently projected from its authenticated
    # design source. Runtime case content is not used to construct expectations.
    expected_row = deepcopy(source_row)
    expected_row["contract"]["evaluation"] = project_evaluation(
        source_row["contract"]["evaluation"], names, args.repeats)
    expected_row["contract"]["scene_groups"] = project_scenes(
        source_row["contract"]["scene_groups"], names)
    expected_row["invocation"]["num_envs"] = count
    engineering_design = {key: deepcopy(value) for key, value in design.items() if key != "stages"}
    engineering_design["stages"] = {binding["stage"]: {"roles": {"full": expected_row}}}
    engineering_design["engineering_projection"] = {
        "scope": "nonformal_runtime_probe_no_training_no_capability_acceptance",
        "source_design_file": binding["file"], "source_design_sha256": binding["sha256"],
        "source_plan_sha256": digest(args.plan), "source_role": "full",
        "selected_case_names": names, "episodes_per_case": args.repeats,
        "formal_updates_consumed": 0,
        "allowed_changes": ["evaluation.cases", "evaluation.*_case_names",
                            "evaluation.episodes_per_case", "scene_groups", "invocation.num_envs"]}
    design_path = args.output / "engineering_design.json"
    design_path.write_text(json.dumps(engineering_design, indent=2, allow_nan=False) + "\n")
    config["evaluation"] = project_evaluation(config["evaluation"], names, args.repeats)
    config["scene_groups"] = project_scenes(config["scene_groups"], names)
    config["design_preflight"] = {**binding, "file": str(design_path.resolve()), "sha256": digest(design_path)}
    path = args.output / "contract.json"
    path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    config, checked_manifest = preflight(path)
    if checked_manifest != manifest:
        raise ValueError("Asset changed during engineering projection")
    receipt = design_preflight(config, control, ROOT, stage=recipe["kind"],
                               role="full", seed=seed, num_envs=count)
    receipt["original_full_role_verified"] = {key: original_receipt[key]
        for key in ("status", "design_sha256", "num_envs")}
    receipt["engineering_projection"] = engineering_design["engineering_projection"]
    return config, manifest, control, recipe, seed, count, "full", receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=14.)
    parser.add_argument("--plan", type=Path, default=ROOT / "contracts/v6_recovery_curriculum_training_v1.json")
    parser.add_argument("--recovery-only", action="store_true", help="Select recovery cases from the frozen full-role design")
    parser.add_argument("--stand-only", action="store_true", help="Select zero-motion recovery followups; implies --recovery-only")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num-envs", type=int, choices=(20000,), help="Unmodified S3 train-role capacity probe, no fixed-suite projection")
    parser.add_argument("--policy-steps", type=int, choices=(3, 4, 5), default=3, help="Capacity-probe steps only")
    parser.add_argument("--stage", choices=("S3", "S4", "S5", "S5T", "S5D", "S6"), default="S3")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.stand_only = bool(args.stand_only)
    args.recovery_only = args.recovery_only or args.stand_only
    if args.num_envs is not None and (args.recovery_only or args.stand_only):
        parser.error("--num-envs uses the unchanged train-role distribution; do not combine with case selection")
    if args.checkpoint is None and args.num_envs is None and not args.preflight_only:
        parser.error("Fixed-actor recovery probes require --checkpoint")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "scope": "engineering_runtime_probe_no_training_or_acceptance",
              "formal_updates_consumed": 0, "capability_acceptance": "not_assessed",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_files_sha256": snapshot_sources(),
              "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
              "source_plan_sha256": digest(args.plan), "script_sha256": digest(Path(__file__)),
              "probe_mode": "train_role_capacity" if args.num_envs else "fixed_actor_engineering_subset",
              "zero_policy_actions": args.num_envs is not None}
    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["source_snapshot_sha256"] = hashlib.sha256(json.dumps(
        report["source_files_sha256"], sort_keys=True).encode()).hexdigest()
    save()
    launcher, env = None, None
    try:
        cfg, manifest, control, recipe, seed, count, role, receipt = materialize(args)
        report.update(design_preflight=receipt, contract_sha256=digest(args.output / "contract.json"),
                      num_envs=count, stage=args.stage, run_role=role)
        if args.preflight_only:
            report["status"] = "preflight_only_no_scene"
            return 0
        save()
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": "cuda:0",
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        import torch
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
        changed = source_changes(report["source_files_sha256"], snapshot_sources())
        if changed:
            raise RuntimeError(f"Probe source changed before environment import: {changed}")
        report["environment_import_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.env import ChassisEnv
        if source_changes(report["source_files_sha256"], snapshot_sources()):
            raise RuntimeError("Probe source changed during environment import")
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        environment_class = ChassisEnv if args.num_envs else FixedCaseEnv
        env = environment_class(cfg, manifest, control, ROOT,
            stage_name=recipe["kind"], num_envs=count, device="cuda:0", level=1., seed=seed, run_role=role)
        if args.num_envs:
            if env.startup_report.get("gpu_dynamics_enabled") is not True:
                raise RuntimeError("20k capacity probe requires the actual GPU dynamics scene")
            actor = lambda observation: torch.zeros(count, 6, device="cuda:0")
            obs = env.get_observations()
        else:
            runner_cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
            runner_cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
            runner_cfg.seed, runner_cfg.device = seed, "cuda:0"
            runner = OnPolicyRunner(env, deepcopy(runner_cfg.to_dict()), log_dir=None, device="cuda:0")
            parent = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
            if parent["infos"]["asset_manifest_sha256"] != cfg["asset_manifest_sha256"]:
                raise ValueError("Recovery physical probe actor belongs to another asset")
            runner.alg.actor.load_state_dict(parent["actor_state_dict"], strict=True)
            actor = runner.alg.actor.as_onnx(verbose=False).to("cuda:0").eval()
            obs = env.reset_suite()
        alive = env.recovery.enabled.clone()
        trace, ends, learning_population = [], [], []
        indices = alive.nonzero().flatten().tolist()
        recovery_lanes = len(indices)
        if recovery_lanes == 0:
            raise RuntimeError("Recovery probe did not instantiate any recovery lanes")
        if args.num_envs:
            representatives = {}
            for index in indices:
                representatives.setdefault(env.scene_groups[index], index)
            indices = list(representatives.values())
        report.update(status="running", num_envs=count, recovery_lanes=recovery_lanes, recorded_lanes=len(indices), startup=env.startup_report)
        save()
        started = time.monotonic()
        for tick in range(args.policy_steps if args.num_envs else round(args.seconds / env.policy_dt)):
            with torch.inference_mode():
                obs, reward, done, extras = env.step(actor(obs["policy"]))
            data = extras["diagnostics"]
            learning = extras["policy_learning_mask"]
            bootstrap = extras["policy_bootstrap_mask"]
            if (learning.dtype != torch.bool or bootstrap.dtype != torch.bool
                    or learning.shape != (count,) or bootstrap.shape != (count,)):
                raise RuntimeError("Physical recovery did not emit the two distinct PPO eligibility masks")
            learning_population.append({"policy_step": tick + 1,
                "pure_rl_intervals": int(learning.sum()), "excluded_intervals": int((~learning).sum()),
                "bootstrap_states": int(bootstrap.sum())})
            if (not torch.isfinite(obs["policy"]).all() or not torch.isfinite(obs["critic"]).all()
                    or not torch.isfinite(reward).all()):
                raise RuntimeError("Nonfinite recovery physical transition")
            keys = ("recovery_phase", "recovery_failed", "recovery_failure_code", "recovery_pure_rl",
                    "recovery_motion_released", "recovery_rl_stable_s", "recovery_followup_s",
                    "recovery_handover_count", "recovery_true_success", "recovery_observed_height",
                    "recovery_observed_support", "recovery_wheel_balance_active", "recovery_platform_retained",
                    "recovery_reroute_count")
            for index in indices:
                if not bool(alive[index]):
                    continue
                row = {key: data[key][index].item() for key in keys}
                row.update(t=(tick + 1) * env.policy_dt, lane=index, group=env.scene_groups[index],
                    gravity_z=float(data["gravity"][index, 2]), height=float(data["height"][index]),
                    q=data["motor_position"][index].tolist(),
                    dq=data["motor_velocity"][index].tolist(),
                    omega=data["omega"][index].tolist(),
                    q_continuous=data["recovery_continuous_q"][index].tolist(),
                    leg_target_error=data["recovery_leg_target_error_rad"][index].tolist(),
                    wheel_target=data["recovery_wheel_velocity_target"][index].tolist(),
                    reasons=[key for key, mask in data["reasons"].items() if bool(mask[index])])
                if tick % 10 == 0:
                    trace.append(row)
                if bool(done[index]):
                    ends.append(row)
            alive &= ~done
            if tick % 50 == 0:
                report.update(elapsed_physics_s=(tick + 1) * env.policy_dt,
                              alive_recovery_lanes=int(alive.sum()), endings=ends, trace=trace)
                save()
            if not args.num_envs and not bool(alive.any()):
                break
        if args.num_envs:
            timing = env.control_audit.last
            if not (timing and timing["physics_steps"] == args.policy_steps * 20
                    and timing["pd_updates"] == timing["feedback_refreshes"] == 20
                    and timing["target_decodes"] == 1):
                raise RuntimeError("20k recovery capacity probe did not preserve the 1 kHz / 50 Hz execution cadence")
            if not any(0 < row["excluded_intervals"] < count for row in learning_population):
                raise RuntimeError("20k train-role probe did not exercise mixed RL and scripted populations")
        report.update(status="engineering_runtime_completed", runtime_seconds=time.monotonic() - started,
            endings=ends, trace=trace, unfinished_lanes=alive.nonzero().flatten().tolist(),
            learning_population=learning_population,
            control_timing=env.control_audit.last,
            capability_acceptance="not_assessed;inspect_per_pose_failures_before_course_release")
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["source_files_changed_during_run"] = source_changes(report["source_files_sha256"], snapshot_sources())
        report["source_unchanged"] = not report["source_files_changed_during_run"]
        save()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
