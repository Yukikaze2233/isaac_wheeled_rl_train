#!/usr/bin/env python3
"""Replay one exported V7 ordinary actor on the plane at 50/1000Hz; no PPO.

Preparation is CPU-only. Native execution requires the existing shared budget
review/ledger, and must be scheduled separately from training. This is not the
source evaluator: its original full-source identity checks remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def prepare_inputs(args):
    import numpy as np
    import onnxruntime as ort
    from export_chassis_candidate import authenticate
    from wheeled_tasks.chassis.ordinary_replay import OrdinaryReplaySetup, ROLE
    from wheeled_tasks.chassis.training_profile import profile_source_files
    from wheeled_tasks.chassis.preflight import design_preflight

    bundle = args.bundle.resolve(strict=True)
    manifest_file = bundle / "bundle_manifest.json"
    package = json.loads(manifest_file.read_bytes())
    if package.get("schema") != "chassis-candidate-bundle-v1":
        raise ValueError("Replay requires the existing exported candidate bundle")
    bindings = {str(manifest_file): sha(manifest_file)}
    for name, row in package["files"].items():
        path = (bundle / name).resolve(strict=True)
        if not path.is_relative_to(bundle) or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Candidate bundle member differs: " + name)
        bindings[str(path)] = row["sha256"]
    required = {"model.pt", "contract.json", "policy.onnx", "deployment/interface.json",
                "deployment/abi_test_vectors.json"}
    if not required <= package["files"].keys():
        raise ValueError("Incomplete ordinary deployment bundle")
    (checkpoint, saved, updates, contract_path, contract, _, _, manifest_path,
     control_path) = authenticate(bundle / "model.pt")
    setup = OrdinaryReplaySetup(tuple(args.command), args.seed,
        saved["infos"]["stage_actor_update"], saved["infos"]["global_actor_update"])
    setup.validate_constructor(contract, 1, args.seed, ROLE)
    manifest = json.loads(manifest_path.read_bytes())
    control = json.loads(control_path.read_bytes())
    preflight = design_preflight(contract, control, ROOT, stage=saved["infos"]["stage"],
        role=ROLE, seed=args.seed, num_envs=1, replay_setup=setup)
    interface = json.loads((bundle / "deployment/interface.json").read_bytes())
    onnx = bundle / "policy.onnx"
    if (interface["identity"]["onnx_sha256"] != sha(onnx)
            or interface["identity"]["source_checkpoint_sha256"] != sha(checkpoint)
            or interface["identity"]["contract_sha256"] != sha(contract_path)):
        raise ValueError("Exported actor/contract identity differs from its interface")
    names = profile_source_files(contract) + ["scripts/replay_v7_ordinary.py",
        "scripts/export_chassis_candidate.py", "src/wheeled_tasks/chassis/ordinary_replay.py",
        "src/wheeled_tasks/chassis/training_budget.py", "src/wheeled_algo/training_control.py"]
    sources = {name: sha(ROOT / name) for name in dict.fromkeys(names)}
    bindings.update({str(ROOT / name): value for name, value in sources.items()})
    directory = ROOT / contract["asset_directory"]
    bindings[str(manifest_path)] = sha(manifest_path)
    for name, digest in manifest["files_sha256"].items():
        path = (directory / name).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()) or sha(path) != digest:
            raise ValueError("Replay mechanical asset differs: " + name)
        bindings[str(path)] = digest
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"])
    if (len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1
            or session.get_inputs()[0].name != "obs" or session.get_outputs()[0].name != "actions"
            or session.get_inputs()[0].type != "tensor(float)" or session.get_outputs()[0].type != "tensor(float)"
            or session.get_inputs()[0].shape[-1] != 35 or session.get_outputs()[0].shape[-1] != 6):
        raise ValueError("ONNX replay requires the exported float32 obs35/actions6 interface")
    vectors = json.loads((bundle / "deployment/abi_test_vectors.json").read_bytes())
    obs = np.asarray([v["obs"] for v in vectors["vectors"]], dtype=np.float32)
    expected = np.asarray([v["actions"] for v in vectors["vectors"]], dtype=np.float32)
    actual = session.run(["actions"], {"obs": obs})[0]
    if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=vectors["atol"], rtol=vectors["rtol"]):
        raise ValueError("ONNX actor does not reproduce its sealed ordinary ABI vectors")
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0 or not math.isclose(
            args.duration_seconds / .02, round(args.duration_seconds / .02), abs_tol=1e-9):
        raise ValueError("Replay duration must be positive and an integer number of policy intervals")
    identity = {"contract_sha256": sha(contract_path),
        "training_source_sha256": hashlib.sha256(canonical(sources)).hexdigest(),
        "seed": args.seed, "stage": saved["infos"]["stage"], "requested_updates": 1,
        "parent_checkpoint_sha256": sha(checkpoint)}
    origin = saved["infos"]["profile_source_sha256"]
    report = {"schema": "v7-ordinary-deployment-replay-v1", "status": "prepared_cpu_only",
        "bundle": str(bundle), "bundle_manifest_sha256": sha(manifest_file),
        "checkpoint_sha256": sha(checkpoint), "onnx_sha256": sha(onnx),
        "learning_updates": updates, "training_contract_sha256": sha(contract_path),
        "training_source_sha256": saved["infos"]["training_source_sha256"],
        "replay_source_sha256": identity["training_source_sha256"], "runtime_sources": sources,
        "source_differences": {n: {"training": value, "replay": sources.get(n)}
            for n, value in origin.items() if sources.get(n) != value},
        "source_scope": "authenticated exported actor ABI; separately bound replay implementation, not native source equivalence",
        "bindings": bindings, "budget_identity": identity, "setup": setup.receipt(),
        "design_preflight": preflight, "duration_seconds": args.duration_seconds,
        "abi_vector_count": len(obs), "abi_max_abs_error": float(np.max(np.abs(actual - expected))),
        "native_started": False, "policy_steps": 0, "physical_steps": 0, "ppo_updates": 0,
        "paid_update_reservations": 0, "qualified": None, "hardware_deployment_ready": False}
    return report, setup, session, saved, contract, manifest, control


def assert_bindings(report):
    for name, expected in report["bindings"].items():
        if sha(name) != expected:
            raise ValueError("Replay input/source/asset changed: " + name)


def persist(output, report):
    temporary = output / "replay.json.tmp"
    with temporary.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "replay.json")


def execute(args):
    report, setup, session, saved, contract, manifest, control = prepare_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    # Preserve the original bytes, including the original .005 training dt.
    (args.output / "training_contract.json").write_bytes((args.bundle / "contract.json").read_bytes())
    persist(args.output, report)
    if args.prepare_only:
        return report
    from wheeled_tasks.chassis.training_budget import BudgetOwner
    from wheeled_algo.training_control import TrainingBudget
    from wheeled_tasks.chassis.ordinary_replay import ROLE
    if not args.budget_ledger.is_file():
        raise ValueError("Replay must reuse the existing shared training ledger")
    checkpoint_budget = saved["infos"]["training_budget"]
    if checkpoint_budget["attempt_id"] == args.budget_attempt_id:
        raise ValueError("Replay needs a separate zero-PPO attempt in the same ledger")
    # Exported copies are not ledger checkpoint registrations. Keep the existing
    # absolute-path registration check and compare that original with the actor.
    if sha(args.registered_checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Registered parent checkpoint differs from the exported actor checkpoint")
    report["registered_checkpoint"] = str(args.registered_checkpoint.resolve(strict=True))
    wall = TrainingBudget(args.max_runtime_seconds)
    wall.begin_learning()
    owner = launcher = env = writer = None
    error = None
    report.update(status="starting", started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid())

    def check():
        wall.check()
        if owner is not None:
            owner.check_runtime()

    try:
        owner = BudgetOwner.authenticate(args.budget_review_file, args.budget_review_sha256,
            args.budget_ledger, args.budget_attempt_id, report["budget_identity"], evaluation_only=True)
        if owner.review["gpu_meter_owner"]["kind"] != "campaign_elapsed_wall_v1":
            raise ValueError("Replay requires the existing elapsed-campaign wall meter")
        owner.verify_resume(args.registered_checkpoint, checkpoint_budget)
        report["training_budget_before"] = owner.summary()
        if report["training_budget_before"]["paid_updates_attempt"] != 0:
            raise ValueError("Replay attempt must have zero paid updates")
        assert_bindings(report)
        with wall.signal_handlers():
            check()
            from isaaclab.app import AppLauncher
            launcher = AppLauncher.__new__(AppLauncher)
            AppLauncher.__init__(launcher, {"headless": not args.gui, "device": args.device,
                "enable_cameras": False, "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
            report["native_started"] = True
            import numpy as np
            import torch
            from torch.utils.tensorboard import SummaryWriter
            from wheeled_tasks.chassis.env import ChassisEnv
            torch.set_num_threads(4)
            torch.manual_seed(args.seed)
            env = ChassisEnv.__new__(ChassisEnv)
            ChassisEnv.__init__(env, contract, manifest, control, ROOT,
                stage_name=saved["infos"]["stage"], num_envs=1, device=args.device,
                seed=args.seed, run_role=ROLE, replay_setup=setup)
            report["startup"] = env.startup_report
            writer = SummaryWriter(str(args.output / "tensorboard"))
            obs = env.get_observations()
            report["simulation_steps_before"] = env.sim.get_physics_step_count()
            report["reset_count"] = 0
            assert_bindings(report)
            for step in range(round(args.duration_seconds / .02)):
                check()
                array = obs["policy"].detach().cpu().numpy()
                if array.shape != (1, 35) or not np.isfinite(array).all():
                    raise RuntimeError("Invalid ordinary actor observation")
                actions = session.run(["actions"], {"obs": array})[0]
                if actions.shape != (1, 6) or not np.isfinite(actions).all():
                    raise RuntimeError("Invalid ordinary actor action")
                obs, reward, done, extras = env.step(torch.from_numpy(actions).to(env.device))
                report.update(policy_steps=step + 1, physical_steps=env.physical_tick,
                    last_control_readback=env.control_audit.last)
                numerical = extras.get("diagnostics", {}).get("reasons", {}).get("numerical_safety")
                if numerical is not None and bool(numerical.any()):
                    torch.save(extras["diagnostics"], args.output / "numerical_failure.pt")
                    raise RuntimeError("Source numerical safety reset during ordinary replay")
                if not all(bool(torch.isfinite(x).all()) for x in
                           (obs["policy"], reward, env.robot.data.root_link_pose_w.torch,
                            env.robot.data.joint_pos.torch, env.robot.data.joint_vel.torch)):
                    raise RuntimeError("Nonfinite native ordinary replay state")
                if env.physical_tick != (step + 1) * 20 or env.flat_episode.commands.physics_steps != (step + 1) * 4:
                    raise RuntimeError("Replay physical/source command clocks diverged")
                # Existing environment logs retain native units; no new score thresholds.
                for name, value in extras.get("log", {}).items():
                    writer.add_scalar("ordinary_replay" + name, float(value), step + 1)
                writer.add_scalar("ordinary_replay/physics_steps", env.physical_tick, step + 1)
                writer.add_scalar("ordinary_replay/reward", float(reward[0]), step + 1)
                if step == 0 or bool(done.any()) or step + 1 == round(args.duration_seconds / .02):
                    torch.save({"policy_step": step + 1, "actor_obs": array,
                        "raw_policy_action": actions, "returned_control_action": env.actions.detach().cpu(),
                        "returned_root_pose": env.robot.data.root_link_pose_w.torch.detach().cpu(),
                        "done": done.detach().cpu(), "diagnostics": extras.get("diagnostics"),
                        "returned_state_may_be_post_reset": bool(done.any())}, args.output / f"step_{step+1:04d}.pt")
                if args.gui:
                    env.sim.render()
                if bool(done.any()):
                    report["reset_count"] += int(done.sum())
                    report["status"] = "episode_ended"
                    break
            else:
                report["status"] = "window_completed"
            report["simulation_steps_after"] = env.sim.get_physics_step_count()
            if report["simulation_steps_after"] - report["simulation_steps_before"] != report["physical_steps"]:
                raise RuntimeError("Native physics step counter differs from the replay clock")
            report["termination_counts"] = dict(env.termination_counts)
            report["timeout_count"] = env.timeout_count
            assert_bindings(report)
            check()
    except BaseException as caught:
        error = caught
        report.update(status="failed", error=traceback.format_exc())
    finally:
        if writer is not None:
            try:
                writer.close()
            except BaseException as caught:
                error = error or caught
                report.update(status="failed", writer_close_error=repr(caught))
        if owner is not None:
            try:
                report["training_budget_after"] = owner.summary()
                owner.check_runtime()
                if report["training_budget_after"]["paid_updates_attempt"] != 0:
                    raise RuntimeError("Replay unexpectedly consumed PPO updates")
            except BaseException as caught:
                error = error or caught
                report.update(status="failed", budget_error=repr(caught))
            finally:
                try:
                    owner.close()
                    report["budget_close_returned"] = True
                except BaseException as caught:
                    error = error or caught
                    report.update(status="failed", budget_close_returned=False, budget_close_error=repr(caught))
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        # Save before Kit close, which may exit the process without returning.
        persist(args.output, report)
        for name, close in (("environment", env.close if env is not None and getattr(env, "sim", None) else None),
                ("application", (lambda: launcher._app.close(exit_code=int(error is not None)))
                    if launcher is not None and getattr(launcher, "_app", None) else None)):
            if close is not None:
                try:
                    close()
                except BaseException as caught:
                    error = error or caught
                    report.update(status="failed", **{name + "_close_error": repr(caught)})
        persist(args.output, report)
    if error is not None:
        raise RuntimeError("Ordinary replay failed; see replay.json") from error
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--native", action="store_true")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", type=float, nargs=3, required=True, metavar=("VX", "YAW", "HEIGHT"))
    parser.add_argument("--seed", type=int, default=190619)
    parser.add_argument("--duration-seconds", type=float, default=20.)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--budget-review-file", type=Path)
    parser.add_argument("--budget-review-sha256")
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--budget-attempt-id")
    parser.add_argument("--registered-checkpoint", type=Path,
        help="Original checkpoint path registered in the existing shared ledger, not the exported copy")
    parser.add_argument("--max-runtime-seconds", type=float, default=600.)
    args = parser.parse_args(argv)
    if args.native and not all((args.budget_review_file, args.budget_review_sha256,
                               args.budget_ledger, args.budget_attempt_id, args.registered_checkpoint)):
        parser.error("Native replay requires the shared review, SHA, ledger, separate zero-PPO attempt and registered parent checkpoint")
    report = execute(args)
    print(json.dumps({k: report[k] for k in ("status", "policy_steps", "physical_steps", "ppo_updates")}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
