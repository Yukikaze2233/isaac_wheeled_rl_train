#!/usr/bin/env python3
"""Independent bounded V40 mean-policy evaluation. Default is CPU preflight, NOT Sim."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from train_v40 import add_common_arguments, checked_checkpoint, launch_app, make_env, make_manifest, preflight, print_preflight
from wheeled_algo.v40_metrics import (
    LIMITATIONS, MAX_RECORDS, InvalidTrajectory, Thresholds, aggregate_cases, collect_case,
    default_cases, diagnostic_json, evaluate_trajectory, require_trained_stage,
    validate_command, write_json_exclusive,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, num_envs=1)
    parser.add_argument("--apply", action="store_true", help="Explicitly permit evaluation Sim (also needs --research and all gates)")
    parser.add_argument("--checkpoint", type=Path, help="Stock V40 checkpoint beside run_manifest.json")
    parser.add_argument("--output-dir", type=Path, help="New evaluation directory; overwrite is never allowed")
    parser.add_argument("--thresholds", type=Path, help="JSON research goals, locked/saved before rollout")
    parser.add_argument("--duration-s", type=float, help="Override research horizon, <=10s, exact .01s multiples")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX_M_S", "WZ_RAD_S", "HEIGHT_M"),
                        help="One fixed command within checkpoint's trained stage; otherwise small stage suite")
    args = parser.parse_args(argv)
    if args.num_envs != 1:
        parser.error("evaluation requires exactly one env (stop each case on first terminal)")
    if args.apply and not args.research:
        parser.error("simulation requires BOTH --apply and --research")
    if args.apply and not args.preflight_only and (args.checkpoint is None or args.output_dir is None):
        parser.error("actual evaluation requires --checkpoint and new --output-dir")
    args.preflight_only = args.preflight_only or not args.apply
    return args


def main(argv=None) -> int:
    args = parse_arguments(argv)
    report, contract, asset = preflight(args)  # Same immutable physical/asset/runtime gates as train/play.
    report.update({"evaluation_mode": "preflight_only" if args.preflight_only else "explicit_apply",
                   "policy": "deterministic mean, no exploration sampling", "limitations": LIMITATIONS})
    actor = manifest = provenance = None
    cases = []
    thresholds = None
    try:
        thresholds = Thresholds.from_json(args.thresholds)
        if args.duration_s is not None:
            thresholds = replace(thresholds, duration_s=args.duration_s)
        if not math.isclose(thresholds.duration_s/.01, round(thresholds.duration_s/.01), abs_tol=1e-7):
            raise InvalidTrajectory("duration-s must be an exact .01s policy-tick multiple")
        if args.output_dir is not None and args.output_dir.exists():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        if contract is not None:
            cases = ([{"case_id": "fixed_command", "command": list(validate_command(contract, args.stage, args.command))}]
                     if args.command is not None else default_cases(contract, args.stage))
        if not report["blockers"]:
            manifest = make_manifest(contract, asset, args)
            if args.checkpoint is not None:
                actor, provenance = checked_checkpoint(args.checkpoint, manifest)
                require_trained_stage(provenance["run_manifest"], args.stage)
            else:
                report["checkpoint_status"] = "not_tested; no checkpoint supplied"
    except Exception as exc:
        report["blockers"].append(f"evaluation rejected: {exc}")
    report["cases"] = cases
    report["thresholds"] = None if thresholds is None else thresholds.to_dict()
    report["checkpoint_provenance"] = provenance
    report["ready"] = not report["blockers"]
    print_preflight(report)
    if args.preflight_only or not report["ready"]:
        return 0 if report["ready"] else 2

    # Only now reserve a new directory. Preflight creates no output and imports no Sim.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {"schema_version": 1, "stage": args.stage, "seed": args.seed, "cases": cases,
              "thresholds": thresholds.to_dict(), "contract_id": manifest["contract_id"],
              "contract_sha256": manifest["contract_sha256"],
              "asset_manifest_sha256": manifest["asset_manifest_sha256"],
              "checkpoint_path": str(args.checkpoint.resolve()), **provenance,
              "preflight": report, "limitations": LIMITATIONS,
              "record_bound_per_case": MAX_RECORDS, "acceptance": "sampled commands only, not full domain coverage"}
    locked = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    config["evaluation_config_sha256"] = hashlib.sha256(locked).hexdigest()
    write_json_exclusive(args.output_dir / "evaluation_config.json", config)
    write_json_exclusive(args.output_dir / "thresholds.json", thresholds.to_dict())
    simulation_app = raw_env = wrapped_env = None
    summaries = []
    runtime_error = None
    try:
        launcher = launch_app(args)
        simulation_app = launcher.app
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        raw_env = make_env(args)
        if (raw_env.contract_sha256 != manifest["contract_sha256"]
                or raw_env.asset_manifest_sha256 != manifest["asset_manifest_sha256"]):
            raise RuntimeError("contract/assets changed during evaluation startup")
        # Reuse approved play/export stock-state validation, including byte hashes and full critic.
        reloaded_actor, current_provenance = checked_checkpoint(args.checkpoint, manifest)
        if current_provenance != provenance:
            raise RuntimeError("checkpoint/run manifest changed during evaluation startup")
        actor = reloaded_actor.to(args.device).eval()
        wrapped_env = RslRlVecEnvWrapper(raw_env)
        steps = round(thresholds.duration_s / contract["timing"]["policy_dt"])

        def mean_policy(observation):
            if not torch.isfinite(observation).all():
                raise InvalidTrajectory("nonfinite policy observation")
            actions = actor(observation)  # Sequential mean actor, never policy.act()/distribution.sample().
            if not torch.isfinite(actions).all():
                raise InvalidTrajectory("nonfinite mean action")
            return actions

        with torch.inference_mode():
            for case in cases:
                records = []
                try:
                    raw_env.set_evaluation_command(tuple(case["command"]))
                    wrapped_env.reset()  # Fixed command is installed before reset/first observation.
                    stop_reason = collect_case(raw_env, wrapped_env, mean_policy, steps,
                                               simulation_app.is_running, records)
                    summary = evaluate_trajectory(records, contract, args.stage, case["command"], thresholds)
                    summary["collector_stop_reason"] = stop_reason
                except Exception as exc:
                    summary = {"passed": False, "end_reason": "invalid_or_interrupted",
                               "failure_reasons": [f"{type(exc).__name__}: {exc}"], "sample_count": max(0, len(records)-1)}
                summary["case_id"] = case["case_id"]
                summaries.append(summary)
                write_json_exclusive(args.output_dir / (case["case_id"] + ".records.json"), {
                    "case": case, "evaluation_config_sha256": config["evaluation_config_sha256"],
                    "records": diagnostic_json(records),
                    "invalid_numeric_encoding": "tagged invalid_numeric objects are failure evidence, never repaired observations",
                })
                write_json_exclusive(args.output_dir / (case["case_id"] + ".summary.json"), summary)
                # Fail-fast suite: no extra rollout after any terminated/timeout/invalid/failed case.
                if not summary["passed"]:
                    break
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            try:
                if wrapped_env is not None:
                    wrapped_env.close()
                elif raw_env is not None:
                    raw_env.close()
            except Exception as exc:
                runtime_error = f"environment close failed: {exc}; previous={runtime_error}"
            completed_ids = {s["case_id"] for s in summaries}
            for case in cases:
                if case["case_id"] not in completed_ids:
                    summaries.append({"case_id": case["case_id"], "passed": False,
                                      "end_reason": "not_tested", "failure_reasons": ["suite stopped before case"]})
            suite = aggregate_cases(summaries)
            if runtime_error is not None:
                suite["passed"] = False
                suite["runtime_error"] = runtime_error
            suite.update({"evaluation_config_sha256": config["evaluation_config_sha256"],
                          "contract_sha256": manifest["contract_sha256"],
                          "asset_manifest_sha256": manifest["asset_manifest_sha256"],
                          "checkpoint_sha256": provenance["checkpoint_sha256"],
                          "run_manifest_sha256": provenance["run_manifest_sha256"],
                          "thresholds": thresholds.to_dict(), "video": "not_recorded",
                          "application_cleanup": "not_observed; summary published before app.close()"})
            # Default fast shutdown can exit the process with code 0 without returning.
            # Publish pass/fail evidence first; process exit status is not policy acceptance.
            write_json_exclusive(args.output_dir / "summary.json", suite)
            print(json.dumps({"passed": suite["passed"], "summary": str(args.output_dir / "summary.json")},
                             allow_nan=False), flush=True)
        finally:
            if simulation_app is not None:
                cleanup = {"status": "returned", "evaluation_config_sha256": config["evaluation_config_sha256"]}
                try:
                    simulation_app.close()
                except Exception as exc:
                    runtime_error = f"application close failed: {exc}; previous={runtime_error}"
                    cleanup.update(status="failed", runtime_error=runtime_error)
                write_json_exclusive(args.output_dir / "application_cleanup.json", cleanup)
    return 0 if suite["passed"] and runtime_error is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
