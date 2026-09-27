#!/usr/bin/env python3
"""Compare compatible V5 actors using fixed, deterministic, first-episode trials."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "contracts/v5_locomotion_v2.json")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-case", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--export-policy", action="store_true")
    parser.add_argument("--cases", nargs="+", help="Named subset of the unchanged fixed manifest")
    parser.add_argument("--no-traces", action="store_true", help="Score cases without collecting bulky trajectory arrays")
    parser.add_argument("--source-asset-manifest", type=Path,
                        help="Explicit original manifest for cross-asset actor evaluation with an unchanged control ABI")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    from train_chassis import digest, preflight
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, grade_fixed_suite, validate_cross_asset_actor, select_evaluation_cases
    contract, manifest = preflight(args.contract)
    if args.seed is not None:
        contract["evaluation"]["seed"] = args.seed
    if args.cases:
        contract["evaluation"] = select_evaluation_cases(contract["evaluation"], args.cases)
    settings = contract["evaluation"]
    repeats = args.episodes_per_case or settings["episodes_per_case"]
    if not 1 <= repeats <= 64:
        parser.error("episodes-per-case must be in [1,64]")
    if any(not path.is_file() for path in args.checkpoint):
        parser.error("Checkpoint does not exist")
    source_manifest = None
    if args.source_asset_manifest:
        if args.export_policy:
            parser.error("Cross-asset evaluation does not export or relabel the original policy")
        source_manifest = json.loads(args.source_asset_manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": digest(args.contract), "asset_manifest_sha256": contract["asset_manifest_sha256"],
        "script_sha256": digest(Path(__file__)), "settings": settings, "episodes_per_case": repeats,
        "device": args.device,
        "deterministic_actor": True, "random_pushes": False, "case_resampling": False, "candidates": []}
    report.update(trace_recording=not args.no_traces, evaluated_case_names=[c["name"] for c in settings["cases"]],
                  coverage="selected_cases" if args.cases else "full_manifest")
    if source_manifest is not None:
        report["source_asset_manifest_sha256"] = digest(args.source_asset_manifest)
        (args.output / "source_asset_manifest.json").write_bytes(args.source_asset_manifest.read_bytes())
    sources = ["scripts/evaluate_chassis.py", "src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/chassis/eval_env.py",
               "src/wheeled_tasks/chassis/evaluation.py", "src/wheeled_tasks/chassis/episode_metrics.py",
               "src/wheeled_tasks/chassis/v5_control.py", "src/wheeled_tasks/chassis/task.py"]
    if contract.get("task_semantics"):
        sources += ["src/wheeled_tasks/chassis/full_tasks.py", "src/wheeled_tasks/chassis/robustness.py"]
    if contract.get("skill_specs"):
        sources += ["src/wheeled_tasks/chassis/skill_commands.py", "src/wheeled_tasks/chassis/skill_curriculum.py"]
    if contract.get("actor_observation_source"):
        sources.append("src/wheeled_tasks/chassis/scut_observation.py")
    if contract.get("reward_profile"):
        sources.append("src/wheeled_tasks/chassis/rewards.py")
    if contract.get("motion_limits"):
        sources.append("src/wheeled_tasks/chassis/motion_limits.py")
    if contract.get("usb_transport", {}).get("enabled"):
        sources.append("src/wheeled_tasks/chassis/usb_transport.py")
    if contract.get("command_transport", {}).get("enabled"):
        sources.append("src/wheeled_tasks/chassis/command_transport.py")
    if contract.get("contact_domain", {}).get("enabled"):
        sources.append("src/wheeled_tasks/chassis/contact_domain.py")
    if contract.get("command_reference"):
        sources.extend(["src/wheeled_tasks/chassis/references.py", "src/wheeled_tasks/chassis/policy_transfer.py"])
    for key, module in (("dynamics_randomization", "dynamics"), ("step_assist", "step_assist")):
        if contract.get(key, {}).get("enabled"):
            sources.append(f"src/wheeled_tasks/chassis/{module}.py")
    report["source_sha256"] = {name: digest(ROOT / name) for name in sources}
    for name in sources:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (args.output / "contract.json").write_bytes(args.contract.read_bytes())
    launcher, env = None, None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": args.device, "enable_cameras": False,
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
        from wheeled_tasks.v40.core import load_contract
        torch.set_num_threads(4)
        torch.manual_seed(settings["seed"])
        count = len(settings["cases"]) * repeats
        env = FixedCaseEnv(fixed_suite_contract(contract), manifest, load_contract(ROOT / contract["control_math_source"]), ROOT,
            stage_name=contract["enabled_stages"][0], num_envs=count, device=args.device, seed=settings["seed"],
            level=1. if contract.get("task_semantics") else 0.)
        cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
        cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        cfg.seed, cfg.device = settings["seed"], args.device
        runner = OnPolicyRunner(env, deepcopy(cfg.to_dict()), log_dir=None, device=args.device)
        report["startup"] = env.startup_report
        report["body_names"] = env.robot.body_names
        report["quaternion_order"] = "xyzw"
        report["trace_columns"] = ["velocity_b_xyz", "omega_b_xyz", "height", "position_xyz", "motor_effort_6"]
        report["body_pose_trace_scope"] = "representatives every four policy ticks, post-step after possible auto-reset"
        report["trace_state_scope"] = "representatives at policy rate; commands, episode_ticks, done and terminated before reset; active includes the first terminal frame"
        if contract.get("diagnostic_trace"):
            report["diagnostic_trace_layout"] = {"motor_order": list(manifest["control_joint_names"]),
                "leg_target_indices": [0, 1, 3, 4], "policy_input": "scaled35D_before_action"}
        representative_ids = [env.scene_groups.index(case["name"]) for case in settings["cases"]]
        for candidate_index, checkpoint_path in enumerate(args.checkpoint):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            source_asset_sha = checkpoint.get("infos", {}).get("asset_manifest_sha256")
            cross_asset = source_asset_sha != contract["asset_manifest_sha256"]
            source_contract_sha = None
            if cross_asset:
                if args.export_policy:
                    raise ValueError("Cross-asset evaluation cannot relabel/export an old policy")
                if source_manifest is None:
                    from wheeled_tasks.chassis.evaluation import cross_asset_checkpoint_provenance
                    proof = cross_asset_checkpoint_provenance(checkpoint_path, checkpoint["infos"], contract,
                        manifest, digest(ROOT / contract["control_math_source"]))
                    source_contract_path = Path(proof["source_contract"])
                    source_contract_sha = proof["source_contract_sha256"]
                    (args.output / f"candidate_{candidate_index:02d}_source_asset_manifest.json").write_bytes(Path(proof["source_manifest"]).read_bytes())
                else:
                    if source_asset_sha != digest(args.source_asset_manifest):
                        raise ValueError("Evaluation checkpoint asset mismatch")
                    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                    source_contract_path = checkpoint_contract_path(checkpoint_path)
                    source_contract_sha = digest(source_contract_path)
                    source_contract = json.loads(source_contract_path.read_text())
                    if (source_contract_sha != checkpoint["infos"].get("contract_sha256")
                            or source_contract.get("asset_manifest_sha256") != source_asset_sha
                            or checkpoint["infos"].get("control_math_sha256") != digest(ROOT / contract["control_math_source"])):
                        raise ValueError("Cross-asset checkpoint provenance mismatch")
                    validate_cross_asset_actor(source_contract, contract, source_manifest, manifest)
                (args.output / f"candidate_{candidate_index:02d}_source_contract.json").write_bytes(source_contract_path.read_bytes())
            actor_state, observation_migration = checkpoint["actor_state_dict"], None
            if contract.get("command_reference"):
                from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                from wheeled_tasks.chassis.policy_transfer import transfer_actor_state
                source_path = checkpoint_contract_path(checkpoint_path)
                source_contract = json.loads(source_path.read_text())
                if digest(source_path) != checkpoint["infos"]["contract_sha256"]:
                    raise ValueError("Reference evaluation requires the checkpoint's authentic contract")
                actor_state, observation_migration = transfer_actor_state(actor_state, source_contract, contract)
            runner.alg.actor.load_state_dict(actor_state, strict=True)
            actor = runner.alg.actor.as_onnx(verbose=False).to(args.device).eval()
            observations = env.reset_suite()
            height_range = contract.get("height_workspace", {}).get("height_m")
            metrics = EpisodeMetrics(env.scene_groups, args.device, contract["policy_dt"], settings["warmup_seconds"],
                warmup_by_group={c["name"]: c.get("warmup_seconds", settings["warmup_seconds"]) for c in settings["cases"]},
                height_range_m=[height_range[0], height_range[-1]] if height_range else None)
            alive = torch.ones(count, dtype=torch.bool, device=args.device)
            trajectories, poses = [], []
            trace_state = {name: [] for name in ("active", "episode_ticks", "done", "terminated", "commands")}
            if contract.get("command_reference"):
                trace_state.update({name: [] for name in ("reference_phase", "reference_height", "reference_vz",
                    "reference_ax", "physical_phase", "jump_requested", "whole_com_vz", "wheel_target_velocity",
                    "motor_velocity", "actions", "reference_height_error", "com_displacement")})
            if contract.get("diagnostic_trace"):
                trace_state.update({name: [] for name in ("policy_input", "motor_position", "leg_target_position",
                    "requested_motor_effort", "motor_effort_bounds", "planar_speed_world")})
            for tick in range(env.max_episode_length + 1):
                # Environment buffers must remain mutable when resetting between actors.
                with torch.no_grad():
                    policy_input = observations["policy"]
                    actions = actor(policy_input)
                    observations, _, done, extras = env.step(actions)
                    diagnostic = extras["diagnostics"]
                    metrics.observe(diagnostic, alive)
                    if not args.no_traces:
                        for key, values in trace_state.items():
                            value = alive if key == "active" else policy_input if key == "policy_input" else diagnostic[key]
                            values.append(value[representative_ids].cpu().numpy().copy())
                        sample = torch.cat((diagnostic["velocity"], diagnostic["omega"], diagnostic["height"][:, None],
                                            diagnostic["position"], diagnostic["motor_effort"]), -1)
                        trajectories.append(sample[representative_ids].cpu().numpy())
                        if tick % 4 == 0:
                            poses.append(env.robot.data.body_link_pose_w.torch[representative_ids].cpu().numpy())
                    alive &= ~done.bool()
                if not bool(alive.any()):
                    break
            result = grade_fixed_suite(metrics.report(), settings, repeats)
            result.update(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=digest(checkpoint_path),
                checkpoint_updates=checkpoint["infos"].get("successful_updates_total"),
                unfinished_episodes=int(alive.sum()), policy_ticks=tick + 1,
                actor_input_dim=contract["actor_dim"], actor_output_dim=contract["action_dim"])
            result.update(cross_asset_evaluation=cross_asset, source_asset_manifest_sha256=source_asset_sha,
                          evaluated_asset_manifest_sha256=contract["asset_manifest_sha256"],
                           source_contract_sha256=source_contract_sha)
            result["coverage"] = report["coverage"]
            result["observation_migration"] = observation_migration
            name = f"candidate_{candidate_index:02d}"
            if args.export_policy:
                from wheeled_algo.chassis_export import export_actor
                policy_directory = args.output / (name + "_policy")
                policy_directory.mkdir()
                identity = {"contract_id": contract["contract_id"], "contract_sha256": digest(args.contract),
                    "asset_manifest_sha256": contract["asset_manifest_sha256"],
                    "control_math_sha256": digest(ROOT / contract["control_math_source"]),
                    "source_checkpoint_contract_sha256": checkpoint["infos"]["contract_sha256"]}
                identity.update(origin_checkpoint_sha256=digest(checkpoint_path), evaluation_protocol=settings["protocol_id"])
                result["export"] = export_actor(runner.alg.actor, policy_directory, identity, observations["policy"])
                (policy_directory / "policy.onnx.contract.json").write_bytes(args.contract.read_bytes())
                result["export_directory"] = str(policy_directory.resolve())
                runner.alg.actor.to(args.device)
            if not args.no_traces:
                np.savez_compressed(args.output / f"{name}_traces.npz", values=np.asarray(trajectories), body_poses=np.asarray(poses),
                                    **{key: np.asarray(values) for key, values in trace_state.items()})
            (args.output / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            report["candidates"].append(result)
            print("V5_FIXED_EVAL", json.dumps(result, allow_nan=False), flush=True)
        report["status"] = "evaluated"
        report["best_candidate_index"] = min(range(len(report["candidates"])), key=lambda i: report["candidates"][i]["rank_lower_is_better"])
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] == "evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
