#!/usr/bin/env python3
"""Independent commanded multi-scene closed-chain training on Isaac Sim 6."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(path):
    c = json.loads(path.read_text())
    kinds = {"chassis-closedchain-commanded-full-v1": "coupled_fourbar_research",
             "v5-gas-spring-commanded-research-v1": "v5_gas_spring_closedchain_research",
             "v5-gas-spring-mixed-research-v1": "v5_gas_spring_closedchain_research",
             "v5-gas-spring-locomotion-research-v2": "v5_gas_spring_closedchain_research",
             "v5-gas-spring-full-stage-research-v2": "v5_gas_spring_closedchain_research"}
    if c["contract_id"] not in kinds or c["task_source"] != "upper_level_commands":
        raise ValueError("Unsupported full-task interface")
    if sum(s["updates"] for s in c["stages"]) != c["total_updates"]:
        raise ValueError("Curriculum budgets do not sum to total")
    bundle = ROOT / c["asset_directory"]
    if digest(bundle / "manifest.json") != c["asset_manifest_sha256"]:
        raise ValueError("Closed-chain manifest identity mismatch")
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        target = (bundle / name).resolve()
        if not target.is_relative_to(bundle.resolve()) or digest(target) != expected:
            raise ValueError(f"Asset dependency mismatch: {name}")
    if manifest["model_kind"] != kinds[c["contract_id"]]:
        raise ValueError("This task requires the multi-body asset")
    if version("rsl-rl-lib") != "5.5.1":
        raise ValueError("This entry requires the validated RSL-RL 5.5.1 runtime")
    return c, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "contracts/chassis_full_v1.json")
    parser.add_argument("--stage", choices=("foundation", "terrain", "speed", "up", "down", "jump", "mixed"), default="foundation")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--level", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--coverage", action="store_true", help="validation: include every stage terrain family")
    parser.add_argument("--validate-scene", type=int, default=0, metavar="STEPS")
    parents = parser.add_mutually_exclusive_group()
    parents.add_argument("--resume", type=Path)
    parents.add_argument("--transfer", type=Path, help="V5 weights only into a new compatible scene contract; fresh optimizer")
    parser.add_argument("--transfer-actor-only", action="store_true", help="Reset critic when changing the reward/task distribution")
    parser.add_argument("--publish-state", action="store_true", help="Publish atomic selected-env real physics snapshots at most 4 Hz")
    parser.add_argument("--max-runtime-seconds", type=float, default=172800.)
    args = parser.parse_args()
    if not (1 <= args.num_envs <= 4096 and 1 <= args.updates <= 100000 and 0 <= args.level <= 1
            and 0 <= args.validate_scene <= 10000 and args.max_runtime_seconds > 0):
        parser.error("Invalid bounded run settings")
    c, manifest = preflight(args.contract)
    if args.transfer_actor_only and not args.transfer:
        parser.error("--transfer-actor-only requires --transfer")
    if args.stage not in c.get("enabled_stages", [s["name"] for s in c["stages"]]):
        parser.error("This contract has not enabled the requested training stage")
    if args.preflight_only:
        print(json.dumps({"asset_verified": True, "contract": c["contract_id"],
                          "asset_directory": c["asset_directory"], "stage": args.stage,
                          "long_run_behavior_acceptance": "pending"}, indent=2))
        return 0
    if not args.research or args.run_dir is None:
        parser.error("--research and a new --run-dir are required")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    source_files = ["scripts/train_chassis.py", "src/wheeled_tasks/chassis/env.py",
                    "src/wheeled_tasks/chassis/task.py", "src/wheeled_tasks/v40/core.py",
                    "src/wheeled_tasks/v40/contract.py", "src/wheeled_tasks/agents/v40_ppo_cfg.py",
                    "src/wheeled_algo/v40_job.py", "src/wheeled_algo/chassis_export.py", c["control_math_source"]]
    if manifest["model_kind"] == "v5_gas_spring_closedchain_research":
        source_files.append("src/wheeled_tasks/chassis/v5_control.py")
        source_files.append("src/wheeled_tasks/chassis/torque_monitor.py")
    if c.get("record_diagnostics"):
        source_files.append("src/wheeled_tasks/chassis/episode_metrics.py")
    source_files.append("src/wheeled_tasks/chassis/full_curriculum.py")
    if c.get("skill_specs"):
        source_files.extend(["src/wheeled_tasks/chassis/skill_commands.py", "src/wheeled_tasks/chassis/skill_curriculum.py"])
    if c.get("actor_observation_source"):
        source_files.append("src/wheeled_tasks/chassis/scut_observation.py")
    if c.get("task_semantics"):
        source_files.extend(["src/wheeled_tasks/chassis/full_tasks.py", "src/wheeled_tasks/chassis/robustness.py"])
    source_hashes = {name: digest(ROOT / name) for name in source_files}
    for name in source_files:
        target = args.run_dir / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    identity = {"contract_id": c["contract_id"], "contract_sha256": digest(args.contract),
                "asset_manifest_sha256": c["asset_manifest_sha256"],
                "control_math_sha256": digest(ROOT / c["control_math_source"])}
    report = {"started_at": datetime.now(timezone.utc).isoformat(), **identity,
              "source_sha256": source_hashes, "stage": args.stage, "status": "starting",
               "successful_updates": 0, "parent_updates": 0, "parent_training_transitions": 0,
               "actor_updates_in_block": 0,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.run_dir / "contract.json").write_bytes(args.contract.read_bytes())
    (args.run_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    launcher, env, runner, metrics = None, None, None, None
    from wheeled_algo.v40_job import TrainingBudget, PlannedStop
    budget = TrainingBudget(args.max_runtime_seconds)
    try:
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": args.device})
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        from wheeled_tasks.v40.core import load_contract
        from wheeled_tasks.chassis.env import ChassisEnv
        env = ChassisEnv(c, manifest, load_contract(ROOT / c["control_math_source"]), ROOT,
            stage_name=args.stage, num_envs=args.num_envs, device=args.device, level=args.level,
            seed=args.seed, coverage=args.coverage)
        report["startup"] = env.startup_report
        if c.get("record_diagnostics"):
            from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
            metrics = EpisodeMetrics(env.scene_groups, args.device, c["policy_dt"])
        report["status"] = "running"
        (args.run_dir / "startup.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        with budget.signal_handlers():
            budget.begin_learning()
            if args.validate_scene:
                for _ in range(args.validate_scene):
                    budget.check()
                    env.step(torch.zeros(args.num_envs, 6, device=args.device))
                report["status"] = "scene_validated"
            else:
                from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
                from rsl_rl.runners import OnPolicyRunner
                from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
                cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
                cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
                cfg.seed, cfg.device = args.seed, args.device
                cfg.save_interval = c["save_interval"]
                cfg.num_steps_per_env = c["num_steps_per_env"]
                if "learning_rate" in c:
                    cfg.algorithm.learning_rate = c["learning_rate"]
                if "learning_rate_schedule" in c:
                    cfg.algorithm.schedule = c["learning_rate_schedule"]
                runner_config = cfg.to_dict()
                if "initial_noise_std" in c:
                    runner_config["actor"]["distribution_cfg"]["init_std"] = c["initial_noise_std"]
                (args.run_dir / "agent_config.json").write_text(json.dumps(runner_config, indent=2))
                runner = OnPolicyRunner(env, deepcopy(runner_config), log_dir=str(args.run_dir), device=args.device)
                if args.transfer:
                    checkpoint = torch.load(args.transfer, map_location="cpu", weights_only=True)
                    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                    source_contract_path = checkpoint_contract_path(args.transfer)
                    old_contract = json.loads(source_contract_path.read_text())
                    for key in ("asset_manifest_sha256", "actor_dim", "actor_frame_dim", "critic_dim", "action_dim",
                                "actor_layout", "critic_layout", "task_modes", "phases", "v5_control", "policy_dt"):
                        if args.transfer_actor_only and key in ("critic_dim", "critic_layout"):
                            continue
                        if key == "v5_control":
                            from wheeled_tasks.chassis.full_curriculum import compatible_control_transfer
                            if compatible_control_transfer(old_contract[key], c[key]):
                                continue
                        if old_contract.get(key) != c.get(key):
                            raise ValueError(f"Scene transfer changes the V5 physical/control interface: {key}")
                    if checkpoint.get("infos", {}).get("asset_manifest_sha256") != identity["asset_manifest_sha256"]:
                        raise ValueError("Transfer checkpoint asset mismatch")
                    runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
                    if not args.transfer_actor_only:
                        runner.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
                    report["transfer"] = {"checkpoint_sha256": digest(args.transfer),
                        "source_contract_sha256": digest(source_contract_path),
                        "source_updates": checkpoint["infos"]["successful_updates_total"],
                        "optimizer": "fresh", "critic": "fresh" if args.transfer_actor_only else "transferred",
                        "scope": "compatible_V5_actor_transfer" if args.transfer_actor_only else "compatible_V5_weights_scene_transfer"}
                    report["transfer"]["wheel_action_clip_old"] = old_contract["v5_control"].get("wheel_action_clip", old_contract["v5_control"]["action_clip"])
                    report["transfer"]["wheel_action_clip_new"] = c["v5_control"].get("wheel_action_clip", c["v5_control"]["action_clip"])
                if args.resume:
                    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
                    infos = checkpoint.get("infos", {})
                    if any(infos.get(k) != v for k, v in identity.items()):
                        raise ValueError("Parent checkpoint has a different task/asset/control interface")
                    runner.load(str(args.resume))
                    report["parent_updates"] = int(infos["successful_updates_total"])
                    report["parent_training_transitions"] = int(infos.get("training_transitions", 0))
                    report["parent_checkpoint_sha256"] = digest(args.resume)
                    runner.current_learning_iteration = report["parent_updates"]
                env.training_transitions = report["parent_training_transitions"]
                before_actor = {k: v.detach().clone() for k, v in runner.alg.actor.state_dict().items()}
                warmup_updates = c.get("critic_warmup_updates", 0)
                runner.alg.actor.requires_grad_(report["parent_updates"] >= warmup_updates)
                original_update, original_step, original_save = runner.alg.update, env.step, runner.save

                def update(*a, **kw):
                    actor_enabled = report["parent_updates"] + report["successful_updates"] >= warmup_updates
                    result = original_update(*a, **kw)
                    report["successful_updates"] += 1
                    report["actor_updates_in_block"] += int(actor_enabled)
                    runner.alg.actor.requires_grad_(report["parent_updates"] + report["successful_updates"] >= warmup_updates)
                    env.training_transitions = report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * cfg.num_steps_per_env
                    progress = {"updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                        "successful_updates": report["successful_updates"], "parent_updates": report["parent_updates"],
                         "num_envs": args.num_envs, "stage": args.stage,
                         "actor_updates_in_block": report["actor_updates_in_block"],
                         "optimizer_phase": "actor_and_critic" if report["parent_updates"] + report["successful_updates"] >= warmup_updates else "critic_warmup",
                        "training_transitions": report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * cfg.num_steps_per_env}
                    temp = args.run_dir / "progress.tmp"
                    temp.write_text(json.dumps(progress, indent=2) + "\n")
                    temp.replace(args.run_dir / "progress.json")
                    if env.torque_monitor is not None and report["successful_updates"] % 10 == 0:
                        monitor = env.torque_monitor.report()
                        monitor["successful_updates"] = report["successful_updates"]
                        temp = args.run_dir / "torque_monitor.tmp"
                        temp.write_text(json.dumps(monitor, indent=2, allow_nan=False))
                        temp.replace(args.run_dir / "torque_monitor.json")
                        with (args.run_dir / "torque_history.jsonl").open("a") as history:
                            history.write(json.dumps(monitor, allow_nan=False) + "\n")
                    if metrics is not None and report["successful_updates"] % 10 == 0:
                        behavior = {"successful_updates": report["successful_updates"], **metrics.report()}
                        temp = args.run_dir / "behavior_metrics.tmp"
                        temp.write_text(json.dumps(behavior, indent=2, allow_nan=False) + "\n")
                        temp.replace(args.run_dir / "behavior_metrics.json")
                        with (args.run_dir / "behavior_history.jsonl").open("a") as history:
                            history.write(json.dumps(behavior, allow_nan=False) + "\n")
                    return result

                last_publish = 0.

                def step(actions):
                    nonlocal last_publish
                    budget.check()
                    transition = original_step(actions)
                    if metrics is not None:
                        metrics.observe(transition[3]["diagnostics"])
                    if args.publish_state and time.monotonic() - last_publish >= .25:
                        state = {**identity, "wall_time_unix": time.time(), "run_id": args.run_dir.name,
                            "snapshot_kind": "post_step_after_auto_reset", "env_id": 0,
                            "sim_time_s": env.tick * c["policy_dt"], "episode_step": int(env.episode_length_buf[0]),
                            "body_names": list(env.robot.body_names), "quaternion_order": "xyzw",
                            "body_link_pose_w": env.robot.data.body_link_pose_w.torch[0].detach().cpu().tolist(),
                            "commands": env.commands[0].detach().cpu().tolist()}
                        selected = [env.scene_groups.index(name) for name in dict.fromkeys(env.scene_groups)]
                        all_poses = env.robot.data.body_link_pose_w.torch[selected].detach().cpu().tolist()
                        state["environments"] = [{"env_index": i, "scene_group": env.scene_groups[i],
                            "terrain": env.kinds[i], "origin": env.origins[i].cpu().tolist(),
                            "surfaces": [vars(s) for s in env.surfaces[i]],
                            "body_link_pose_w": poses, "commands": env.commands[i].cpu().tolist(),
                            "episode_step": int(env.episode_length_buf[i])} for i, poses in zip(selected, all_poses)]
                        temp = args.run_dir / "live_state.tmp"
                        temp.write_text(json.dumps(state, allow_nan=False))
                        temp.replace(args.run_dir / "live_state.json")
                        last_publish = time.monotonic()
                    return transition

                def save(path, infos=None):
                    original_save(path, infos={**identity, "stage": args.stage,
                        "successful_updates_total": report["parent_updates"] + report["successful_updates"],
                        "training_transitions": report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * cfg.num_steps_per_env})

                runner.alg.update, env.step, runner.save = update, step, save
                runner.learn(args.updates)
                report["actor_changed"] = any(not torch.equal(v, before_actor[k]) for k, v in runner.alg.actor.state_dict().items())
                if report["actor_updates_in_block"] > 0 and not report["actor_changed"]:
                    raise RuntimeError("No actor parameter updates")
                if not all(torch.isfinite(p).all() for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()):
                    raise RuntimeError("Nonfinite trained parameters")
                report["status"] = "completed"
    except PlannedStop as exc:
        report.update(status="stopped", reason=exc.reason)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if runner is not None and report["status"] in ("stopped", "failed"):
            runner.logger.stop_logging_writer()
        if runner is not None and report["successful_updates"] > 0 and report["status"] in ("completed", "stopped"):
            path = args.run_dir / "model_final.pt"
            runner.save(str(path))
            report["checkpoint_sha256"] = digest(path)
            try:
                from wheeled_algo.chassis_export import export_actor
                report["export"] = export_actor(runner.alg.actor, args.run_dir, identity,
                                                 env.get_observations()["policy"])
            except Exception:
                report.update(status="export_failed", export_error=traceback.format_exc())
                traceback.print_exc()
        if env is not None:
            report["metrics"] = env.summary()
            if metrics is not None:
                (args.run_dir / "behavior_metrics.json").write_text(json.dumps(metrics.report(), indent=2, allow_nan=False) + "\n")
            if env.torque_monitor is not None:
                torque_report = env.torque_monitor.report()
                (args.run_dir / "torque_monitor.json").write_text(json.dumps(torque_report, indent=2, allow_nan=False))
                report["torque_monitor_file"] = "torque_monitor.json"
            env.close()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.run_dir / "completion.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("CHASSIS_COMPLETION", report["status"], report["successful_updates"], flush=True)
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] in ("completed", "scene_validated", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
