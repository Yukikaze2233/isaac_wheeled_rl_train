#!/usr/bin/env python3
"""Recover an export-only failure without changing the original training receipt."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/round4"), str(REPO / "scripts"), str(REPO / "src")]

from r4_common import file_record, training_progress, verify_snapshot
from wheeled_algo.v40_job import artifact_record, atomic_bytes, json_bytes, strict_json, validate_completion

PROGRESS_KEYS = ("prior_completed_updates", "invocation_completed_updates", "cumulative_completed_updates")
RUNTIME_CHANGES = {
    "src/wheeled_algo/v40_export.py", "scripts/export_v40_onnx.py",
    "scripts/round4/evaluate_policy.py", "scripts/round4/export_recovery.py",
}


def validate_precision_evidence(sidecar):
    arithmetic = {"internal_precision": "float64-internal", "input_output_precision": "float32",
                  "reference": "native PyTorch FP64 ELU, FP32 output",
                  "checkpoint_parameters_changed": False, "deployment_provider": "CPUExecutionProvider"}
    validation = sidecar.get("validation", {})
    if (sidecar.get("arithmetic") != arithmetic or validation.get("passed") is not True
            or validation.get("sample_count") != 4105 or validation.get("seed") != 40
            or validation.get("atol") != 1e-6 or validation.get("rtol") != 1e-5
            or validation.get("provider") != "CPUExecutionProvider"):
        raise ValueError("recovery precision/reference validation mismatch")
    samples = validation.get("samples", [])
    if len(samples) != 4105 or any(item.get("sample") != i for i, item in enumerate(samples)):
        raise ValueError("recovery validation samples incomplete")
    for item in [validation, *samples]:
        for key in ("max_abs_error", "max_relative_error"):
            value = item.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid recovery numerical error")
    original = validation.get("original_float32_comparison", {})
    count = original.get("failed_sample_count")
    if (original.get("sample_count") != 4105 or type(count) is not int or not 0 <= count <= 4105
            or type(original.get("first_nine_failed_samples")) is not list
            or any(type(i) is not int or not 0 <= i < 9 for i in original["first_nine_failed_samples"])
            or len(set(original["first_nine_failed_samples"])) != len(original["first_nine_failed_samples"])
            or len(original["first_nine_failed_samples"]) > count
            or type(original.get("max_abs_error")) not in (int, float)
            or not math.isfinite(original["max_abs_error"]) or original["max_abs_error"] < 0):
        raise ValueError("original FP32 arithmetic comparison missing/invalid")


def recovery_source(snapshot, experiment):
    """Require a clean committed recovery tree with byte-identical simulation sources."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("recovery source must be a clean committed checkout")
    commit = git("rev-parse", "HEAD")
    changes = git("diff", "--name-only", snapshot["git_commit"], commit).splitlines()
    if any(name not in RUNTIME_CHANGES and not name.startswith(("tests/", "docs/")) for name in changes):
        raise ValueError("recovery source changes unaudited runtime files")
    prefix = snapshot["code_directory_name"] + "/"
    for name, record in snapshot["files"].items():
        relative = name.removeprefix(prefix)
        if relative in RUNTIME_CHANGES or relative.startswith(("tests/", "docs/")):
            continue
        if file_record(REPO / relative) != record:
            raise ValueError("recovery changed frozen simulation source: " + relative)
    return {"recovery_source_commit": commit,
            "training_source_commit": snapshot["git_commit"],
            "source_snapshot_sha256": file_record(experiment / "snapshot.json")["sha256"]}


def verify_failed_training(plan_path):
    plan = strict_json(plan_path.read_bytes())
    run = Path(plan["run_dir"]).resolve()
    experiment = Path(plan["experiment"])
    snapshot = verify_snapshot(experiment)
    if (run != (Path(plan["stage_root"]) / "train").resolve()
            or Path(plan["repo"]).resolve() != (experiment / snapshot["code_directory_name"]).resolve()
            or snapshot["git_commit"] != plan["git_commit"]):
        raise ValueError("training plan/source identity mismatch")
    completion = validate_completion(strict_json((run / "completion.json").read_bytes()))
    if (completion["status"] != "export_failed" or completion["update_failed"]
            or completion["stop_reason"] != "iterations_completed"
            or not completion["requested_iterations_completed"]):
        raise ValueError("recovery requires an export-only failure after all requested updates")
    for record in completion["artifacts"]:
        if artifact_record(run, record["path"]) != record:
            raise ValueError("original training artifact changed: " + record["path"])
    manifest = strict_json((run / "run_manifest.json").read_bytes())
    progress = training_progress(plan, completion, manifest)
    if progress["cumulative_completed_updates"] != 30000:
        raise ValueError("recovery requires exactly 30000 cumulative updates")
    hashes = strict_json((run / "source_hashes.json").read_bytes()).get("source_files_sha256")
    if not hashes or any(snapshot["files"].get(snapshot["code_directory_name"] + "/" + name, {}).get("sha256") != digest
                         for name, digest in hashes.items()):
        raise ValueError("original training source hashes differ from its snapshot")
    identity = recovery_source(snapshot, experiment)
    identity.update(training_plan_sha256=file_record(plan_path)["sha256"],
                    completion_sha256=file_record(run / "completion.json")["sha256"],
                    checkpoint_sha256=file_record(run / "model_final.pt")["sha256"],
                    run_manifest_sha256=file_record(run / "run_manifest.json")["sha256"],
                    original_status="export_failed", training_progress={k: progress[k] for k in PROGRESS_KEYS})
    return plan, manifest, snapshot, identity


def prepare(plan_path, output):
    from evaluation import evaluation_request
    from evaluate_policy import validate_request
    from wheeled_algo.v40_export import export_checkpoint

    plan, manifest, _, identity = verify_failed_training(plan_path)
    run = Path(plan["run_dir"])
    output.mkdir(parents=False, exist_ok=False)
    atomic_bytes(output / "run_manifest.json", (run / "run_manifest.json").read_bytes())
    report = export_checkpoint(run / "model_final.pt", run / "run_manifest.json", output / "policy.onnx",
                               precision="float64-internal")
    if verify_failed_training(plan_path)[3] != identity:
        raise ValueError("training artifacts changed during recovery export")
    receipt = {"schema_version": 1, **identity, "arithmetic": report["arithmetic"],
               "policy_quality_verified": False,
               "artifacts": {name: file_record(output / name) for name in
                             ("policy.onnx", "policy.onnx.json", "run_manifest.json")}}
    atomic_bytes(output / "recovery.json", json_bytes(receipt))
    request = evaluation_request(plan, [plan["requested_profile"]["push"]["schedule"][-1][1], 0.0])
    request.update(training_plan=str(plan_path.resolve()), policy=str((output / "policy.onnx").resolve()),
                   policy_sha256=report["onnx_sha256"], training_progress=identity["training_progress"],
                   export_recovery=str((output / "recovery.json").resolve()),
                   export_recovery_sha256=file_record(output / "recovery.json")["sha256"])
    if plan.get("initialization") == "resume":
        request["resume_parent_checkpoint_sha256"] = plan["parent"]["checkpoint_sha256"]
    validate_request(request)
    atomic_bytes(output / "request.json", json_bytes(request))
    verify_inputs(output / "request.json", create_policy=False)


def verify_inputs(request_path, *, create_policy=True):
    from evaluate_policy import OLD_ASSET_SHA, validate_request
    from play_v40_onnx import load_policy
    from wheeled_algo.v40_ground import verify_cached_ground
    from wheeled_tasks.v40.contract import contract_digest, load_contract, validate_asset

    request_record = file_record(request_path)
    request = strict_json(request_path.read_bytes())
    validate_request(request)
    plan, manifest, snapshot, identity = verify_failed_training(Path(request["training_plan"]))
    receipt_path = Path(request["export_recovery"])
    receipt_record = file_record(receipt_path)
    receipt = strict_json(receipt_path.read_bytes())
    if (receipt_record["sha256"] != request["export_recovery_sha256"]
            or receipt.get("schema_version") != 1 or receipt.get("policy_quality_verified") is not False
            or any(receipt.get(key) != value for key, value in identity.items())
            or request["training_progress"] != identity["training_progress"]
            or Path(request["run_dir"]).resolve() != Path(plan["run_dir"]).resolve()):
        raise ValueError("export recovery receipt/training identity mismatch")
    if plan.get("initialization") == "resume" and request.get("resume_parent_checkpoint_sha256") != plan["parent"]["checkpoint_sha256"]:
        raise ValueError("resume parent mismatch")
    root = receipt_path.parent
    if Path(request["policy"]).resolve() != (root / "policy.onnx").resolve():
        raise ValueError("recovery policy must belong to its receipt")
    if set(receipt["artifacts"]) != {"policy.onnx", "policy.onnx.json", "run_manifest.json"}:
        raise ValueError("invalid recovery artifact set")
    for name, expected in receipt["artifacts"].items():
        if file_record(root / name) != expected:
            raise ValueError("recovery artifact changed: " + name)
    sidecar = strict_json((root / "policy.onnx.json").read_bytes())
    validate_precision_evidence(sidecar)
    if (sidecar.get("checkpoint_sha256") != identity["checkpoint_sha256"]
            or sidecar.get("run_manifest_sha256") != identity["run_manifest_sha256"]
            or sidecar.get("run_manifest") != manifest
            or sidecar.get("onnx_sha256") != receipt["artifacts"]["policy.onnx"]["sha256"]
            or request["policy_sha256"] != sidecar["onnx_sha256"]
            or receipt["artifacts"]["run_manifest.json"]["sha256"] != identity["run_manifest_sha256"]
            or receipt.get("arithmetic") != sidecar.get("arithmetic")):
        raise ValueError("recovered export lacks the required independent FP64 validation")
    contract = load_contract(Path(plan["run_dir"]) / "contract.json")
    asset = validate_asset(contract, allow_research=True)
    if (contract_digest(contract) != contract_digest(load_contract(REPO / snapshot["contract"]))
            or contract_digest(contract) != manifest["contract_sha256"]
            or request["contract_sha256"] != manifest["contract_sha256"]
            or asset["asset_manifest_sha256"] != manifest["asset_manifest_sha256"]
            or manifest["asset_manifest_sha256"] != OLD_ASSET_SHA):
        raise ValueError("recovery contract/physics differs from frozen training")
    ground = verify_cached_ground(Path(request["ground_usd"]))
    if (request["ground_record"] != plan["ground_record"]
            or any(ground[k] != request["ground_record"][k] for k in ("sha256", "size"))):
        raise ValueError("recovery ground identity mismatch")
    policy = load_policy(root / "policy.onnx", manifest)[0] if create_policy else None
    if file_record(request_path) != request_record or file_record(receipt_path) != receipt_record:
        raise ValueError("recovery request/receipt changed during validation")
    identity.update(request_sha256=request_record["sha256"], recovery_receipt_sha256=receipt_record["sha256"],
                    policy_sha256=sidecar["onnx_sha256"], contract_sha256=manifest["contract_sha256"],
                    asset_manifest_sha256=OLD_ASSET_SHA, arithmetic=sidecar["arithmetic"])
    return request, plan, manifest, contract, policy, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    commands.add_parser("evaluate", add_help=False)
    args, remainder = parser.parse_known_args()
    if args.command == "prepare":
        if remainder:
            parser.error("unexpected arguments: " + " ".join(remainder))
        prepare(args.plan, args.output)
        return 0
    from evaluate_policy import main as evaluate
    return evaluate(remainder, input_verifier=verify_inputs)


if __name__ == "__main__":
    raise SystemExit(main())
