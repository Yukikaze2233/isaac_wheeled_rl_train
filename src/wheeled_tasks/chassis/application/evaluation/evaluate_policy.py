#!/usr/bin/env python3
"""Evaluate formal V7 flat cases or the unchanged rough baseline rough distribution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[5]


def capture_options(args):
    selected = getattr(args, "rough_numerical_capture", False)
    if selected and (type(args.max_runtime_seconds) not in (float, int)
            or not math.isfinite(args.max_runtime_seconds) or not 0 < args.max_runtime_seconds <= 600.):
        raise ValueError("Rough numerical capture requires a finite wall cap of at most 600 seconds, including startup")
    return dict(numerical_capture=selected,
        numerical_capture_directory=args.output / "numerical_capture" if selected else None,
        diagnostics_directory=getattr(args, "output", None))


def prepare(args):
    from wheeled_tasks.chassis.application.evaluation.flat_evaluation_runtime import BoundEvaluationInputs, EvaluationActor
    bound = BoundEvaluationInputs.capture(root=args.root, checkpoint=args.checkpoint,
        contract=args.contract, protocol=args.protocol, manifest=args.manifest,
        control=args.control, overlay=args.overlay, **capture_options(args))
    bound.authenticate()
    # This constructs only a stock CPU inference model. No PPO runner or learn.
    actor = EvaluationActor.from_prepared(bound, device="cpu")
    setup = bound.build_setup()
    bound.assert_unchanged()
    report = bound.receipt()
    report["actor_preparation"] = actor.receipt()
    report["setup_type"] = type(setup).__module__ + "." + type(setup).__name__
    report["execution"] = {"kit_started": False, "native_steps": 0, "ppo_updates": 0,
        "actor_inference_calls": 0, "mechanical_reachability_verified": False,
        "execution_tree_frozen": False, "deployment_1000Hz_verified": False}
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for name, item in bound.inputs.items():
        suffix = ".pt" if name == "checkpoint" else ".json"
        target = output / "inputs" / (name + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.data)
    for name, item in bound.sources.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.data)
    bound.assert_unchanged()
    (output / "preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def native_inputs(args):
    """Bind the formal actor, unchanged contract, assets and budget code."""
    from wheeled_tasks.chassis.application.evaluation.flat_evaluation_runtime import BoundEvaluationInputs, BoundFile, canonical, sha
    from wheeled_tasks.chassis.application.configuration.flat_episode_selection import require_native_flat_admission
    from wheeled_tasks.chassis.application.configuration.control_contract import validate_chassis_control
    bound = BoundEvaluationInputs.capture(root=args.root, checkpoint=args.checkpoint,
        contract=args.contract, protocol=args.protocol, manifest=args.manifest,
        control=args.control, overlay=args.overlay, **capture_options(args))
    if bound.root != ROOT.resolve():
        raise ValueError("Native evaluation must execute from its captured source root")
    if bound.contract.get("formal_v7_flat") is not True:
        raise ValueError("Native evaluation is restricted to admitted formal_v7_flat contracts")
    require_native_flat_admission(bound.contract, bound.manifest)
    validate_chassis_control(bound.control)
    for name in ("src/wheeled_tasks/chassis/application/runtime/training_budget.py", "src/wheeled_algo/training_control.py"):
        bound.sources[name] = BoundFile.read(bound.root / name)
    bound.authenticate()
    setup = bound.build_setup()
    if bound.is_scripted_recovery:
        from wheeled_tasks.chassis.application.configuration.preflight import design_preflight
        design_preflight(bound.evaluation_config(), bound.control, bound.root,
            stage=bound.execution_stage, role=bound.run_role,
            seed=bound.protocol["environment_seed"], num_envs=bound.overlay["num_envs"],
            evaluation_setup=setup)
    directory = (bound.root / bound.contract["asset_directory"]).resolve(strict=True)
    if bound.inputs["manifest"].path != directory / "manifest.json":
        raise ValueError("Native evaluation must use the configured asset manifest")
    assets = {}
    for name, expected in bound.manifest["files_sha256"].items():
        path = (directory / name).resolve(strict=True)
        if not path.is_relative_to(directory):
            raise ValueError("Asset dependency escapes its configured directory")
        item = BoundFile.read(path)
        if item.digest != expected:
            raise ValueError("Asset dependency mismatch: " + name)
        assets[name] = item
    identity = {"contract_sha256": bound.inputs["contract"].digest,
        "training_source_sha256": sha(canonical({name: item.digest for name, item in bound.sources.items()})),
        "seed": bound.protocol["environment_seed"], "stage": bound.authentication["stage"],
        "requested_updates": 1, "parent_checkpoint_sha256": bound.inputs["checkpoint"].digest}
    return bound, assets, identity


def native(args):
    """One formal inference attempt; no optimizer or paid-update reservation."""
    from wheeled_tasks.chassis.application.evaluation.flat_evaluation_runtime import sha
    from wheeled_tasks.chassis.application.evaluation.flat_evaluation_loop import FlatEvaluationLoop, EpisodeRecordWriter
    from wheeled_tasks.chassis.application.runtime.training_budget import BudgetOwner
    from wheeled_algo.training_control import TrainingBudget, PlannedStop
    max_policy_steps = getattr(args, "max_policy_steps", None)
    if max_policy_steps is not None and (type(max_policy_steps) is not int or not 0 < max_policy_steps <= 999):
        raise ValueError("Observation step cap must be an integer in1..999")
    bound, assets, identity = native_inputs(args)
    checkpoint_budget = bound.load_checkpoint()["infos"].get("training_budget")
    if not checkpoint_budget or checkpoint_budget.get("attempt_id") == args.budget_attempt_id:
        raise ValueError("Evaluation requires a separate attempt and a registered training checkpoint")
    if not args.budget_ledger.is_file():
        raise ValueError("Evaluation must reuse the existing shared training ledger")
    wall = TrainingBudget(args.max_runtime_seconds)
    wall.begin_learning()  # Include native startup in this invocation's wall cap.
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "formal-flat-evaluation-v1", "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(), "budget_identity": identity,
        "source_identity_scope": "captured evaluator sources, not the parent training source digest",
        "preparation": bound.receipt(), "asset_bindings": {name: item.receipt() for name, item in assets.items()},
        "device": args.device, "max_runtime_seconds": args.max_runtime_seconds,
        "max_policy_steps": max_policy_steps,
        "ppo_updates": 0, "paid_update_reservations": 0, "kit_started": False,
        "native_qualification": False, "training_release": False, "cleanup_errors": []}
    if bound.numerical_capture:
        report.update(schema="rough-numerical-capture-diagnostic-v1",
            diagnostic_scope="one bounded fixed-parent sampled attempt; not a capability assessment or exact failure replay",
            repair_verified=False)
    owner = launcher = env = writer = loop = None
    error = None
    reported_steps = 0

    def check():
        nonlocal reported_steps
        wall.check()
        if owner is not None:
            owner.check_runtime()
        steps = getattr(loop, "steps", 0)
        if max_policy_steps is not None and steps >= max_policy_steps:
            raise PlannedStop("matched_observation_policy_steps")
        if steps >= reported_steps + 100:
            print(json.dumps({"evaluation_policy_steps": steps,
                "evaluation_physics_intervals": steps * 4, "ppo_updates": 0}), flush=True)
            reported_steps = steps

    def check_inputs():
        bound.assert_unchanged()
        for name, item in assets.items():
            if not item.path.is_file() or sha(item.path.read_bytes()) != item.digest:
                raise ValueError("Native asset changed: " + name)

    def persist_report():
        # Kit shutdown may terminate the process without returning to Python.
        # Publish a complete, durable receipt before invoking either SDK close.
        temporary = args.output / "evaluation.json.tmp"
        with temporary.open("w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.output / "evaluation.json")
        directory = os.open(args.output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    try:
        for category, items in (("inputs", bound.inputs), ("source", bound.sources), ("asset", assets)):
            for name, item in items.items():
                if category == "inputs":
                    name += ".pt" if name == "checkpoint" else ".json"
                path = args.output / category / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(item.data)
        (args.output / "native_preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        owner = BudgetOwner.authenticate(args.budget_review_file, args.budget_review_sha256,
            args.budget_ledger, args.budget_attempt_id, identity, evaluation_only=True)
        if owner.review["gpu_meter_owner"]["kind"] != "campaign_elapsed_wall_v1":
            raise ValueError("Formal evaluation requires the existing elapsed-campaign budget meter")
        owner.verify_resume(args.checkpoint, checkpoint_budget)
        report["training_budget_before"] = owner.summary()
        if report["training_budget_before"]["paid_updates_attempt"] != 0:
            raise ValueError("Evaluation attempt must contain zero paid updates")
        check_inputs()
        with wall.signal_handlers():
            check()
            from isaaclab.app import AppLauncher
            launcher = AppLauncher.__new__(AppLauncher)
            AppLauncher.__init__(launcher, {"headless": True, "device": args.device, "enable_cameras": False,
                "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
            report["kit_started"] = True
            check()
            import torch
            from wheeled_tasks.chassis.application.environment import ChassisEnv
            torch.set_num_threads(4)
            torch.manual_seed(bound.protocol["environment_seed"])
            if bound.numerical_capture:
                Path(bound.numerical_capture_directory).mkdir(parents=True, exist_ok=False)
            env = ChassisEnv.__new__(ChassisEnv)
            ChassisEnv.__init__(env, bound.evaluation_config(), bound.manifest, bound.control, bound.root,
                stage_name=bound.execution_stage, num_envs=bound.overlay["num_envs"],
                device=args.device, level=0., seed=bound.protocol["environment_seed"],
                coverage=False, block_index=0, run_role=bound.run_role, evaluation_setup=bound.build_setup())
            if bound.numerical_capture:
                env.enable_numerical_failure_capture()
            report["startup"] = env.startup_report
            check_inputs()
            writer = EpisodeRecordWriter(args.output / "episodes.jsonl")
            print(json.dumps({"evaluation_phase": "actor_preparation", "ppo_updates": 0}), flush=True)
            loop = FlatEvaluationLoop(bound, device=args.device, limit_check=check)
            print(json.dumps({"evaluation_phase": "rollout", "ppo_updates": 0}), flush=True)
            # Kit may replace handlers during startup. Reinstall the budget
            # handlers for rollout so an interrupt reaches durable cleanup.
            with wall.signal_handlers():
                report["evaluation"] = loop.run(env, commit_records=writer)
            check()
            check_inputs()
            report["status"] = "evaluated"
    except BaseException as caught:
        error = caught
        report.update(status="failed", error=traceback.format_exc())
    finally:
        if loop is not None:
            try:
                report["evaluation"] = loop.report()
            except BaseException as caught:
                report["cleanup_errors"].append({"phase": "loop_readback", "error": repr(caught)})
                error = error or caught
        if env is not None and hasattr(env, "evaluation_heading_hold"):
            report["heading_hold"] = env.evaluation_heading_hold.receipt()
        if writer is not None:
            try:
                writer.close()
            except BaseException as caught:
                report["cleanup_errors"].append({"phase": "episode_writer", "error": repr(caught)})
                error = error or caught
            try:
                report["episode_records"] = writer.receipt()
            except BaseException as caught:
                report["cleanup_errors"].append({"phase": "episode_readback", "error": repr(caught)})
                error = error or caught
        if owner is not None:
            try:
                report["training_budget_after"] = owner.summary()
                owner.check_runtime()
                if report["training_budget_after"]["paid_updates_attempt"] != 0:
                    raise RuntimeError("Evaluation unexpectedly consumed paid updates")
            except BaseException as caught:
                report["cleanup_errors"].append({"phase": "budget_readback", "error": repr(caught)})
                error = error or caught
            finally:
                try:
                    owner.close()
                    report["budget_close_returned"] = True
                except BaseException as caught:
                    report["budget_close_returned"] = False
                    report["cleanup_errors"].append({"phase": "budget_close", "error": repr(caught)})
                    error = error or caught
        if error is not None:
            report["status"] = "failed"
        if bound.numerical_capture:
            event = getattr(loop, "numerical_capture_event", None)
            capture = getattr(env, "startup_report", {}).get("numerical_failure_capture", {})
            report["numerical_capture_readback"] = dict(capture)
            report["diagnostic_outcome"] = ("numerical_event_detected" if event is not None or capture.get("recorded") is True
                else "not_reproduced_within_observed_window" if loop is not None and loop.steps
                else "no_native_rollout_observed")
            report["repair_verified"] = False
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        closes = (("environment", env.close if env is not None and getattr(env, "sim", None) is not None else None),
            ("application", (lambda: launcher._app.close(exit_code=int(error is not None)))
                if launcher is not None and getattr(launcher, "_app", None) is not None else None))
        report["shutdown"] = {name: {"attempted": False, "returned": False} for name, _ in closes}
        for name, close in closes:
            if close is not None:
                report["shutdown"][name]["attempted"] = True
                try:
                    persist_report()
                except BaseException as caught:
                    report["cleanup_errors"].append({"phase": "report_before_" + name, "error": repr(caught)})
                    report["status"] = "failed"
                    error = error or caught
                try:
                    close()
                    report["shutdown"][name]["returned"] = True
                except BaseException as caught:
                    report["cleanup_errors"].append({"phase": name, "error": repr(caught)})
                    report["status"] = "failed"
                    error = error or caught
        try:
            persist_report()
        except BaseException as caught:
            report["cleanup_errors"].append({"phase": "report_after_shutdown", "error": repr(caught)})
            report["status"] = "failed"
            if error is not None:
                error.add_note("Final report persistence failed: " + repr(caught))
            error = error or caught
    if error is not None:
        raise error
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--native", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    for name in ("checkpoint", "contract", "protocol", "manifest", "control", "overlay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--budget-review-file", type=Path)
    parser.add_argument("--budget-review-sha256")
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--budget-attempt-id")
    parser.add_argument("--max-runtime-seconds", type=float, default=600.)
    parser.add_argument("--max-policy-steps", type=int,
        help="Optional short observation cap; open episodes remain censored, never full-horizon passes")
    parser.add_argument("--rough-numerical-capture", action="store_true",
        help="Pinned rough016/4701 sampled zero-PPO failure capture; at most 600s, no qualification")
    args = parser.parse_args(argv)
    if args.native:
        if not all((args.budget_review_file, args.budget_review_sha256, args.budget_ledger, args.budget_attempt_id)):
            parser.error("Native evaluation requires the shared budget review, SHA, ledger and separate attempt ID")
        native(args)
        return 0
    if not args.prepare_only:
        # No CLI switch, token or callable backend bypasses this public gate.
        from wheeled_tasks.chassis.application.evaluation.flat_policy_evaluation import require_native_execution
        require_native_execution()
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
