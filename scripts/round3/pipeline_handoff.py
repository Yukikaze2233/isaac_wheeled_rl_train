#!/usr/bin/env python3
"""Read A final/evaluation and B1 pilot evidence; recommend, never launch/promote."""
import argparse
import hashlib
import json
from pathlib import Path

from common import file_record
from evaluate_finished import CRITERIA, summarize, training_alive, verified_final, write_json
from pipeline_common import A_SHA, verify_transfer
from start_v40_round2 import verify_run
from watch_evaluation import ALLOWED, validate_report_manifest


def verified_evaluation(directory, a_plan_path, a_plan):
    directory = Path(directory)
    marker = directory / "report_manifest.json"
    if not marker.exists():
        return None
    manifest = json.loads(marker.read_text())
    plan_sha = hashlib.sha256(Path(a_plan_path).read_bytes()).hexdigest()
    validate_report_manifest(manifest, plan_sha, directory.name)
    if (manifest.get("state") != "evaluation_completed"
            or {item["path"] for item in manifest["files"]} != ALLOWED):
        raise ValueError("A final evaluation manifest must cover all four-height evidence")
    for item in manifest["files"]:
        if file_record(directory / item["path"]) != {k: item[k] for k in ("size", "sha256")}:
            raise ValueError("A evaluation report hash mismatch")
    result = json.loads((directory / "evaluation_result.json").read_text())
    if (result["training_plan_sha256"] != plan_sha or result["contract_sha256"] != A_SHA
            or result["policy_sha256"] != file_record(Path(a_plan["run_dir"]) / "policy.onnx")["sha256"]
             or result["criteria"] != CRITERIA or result["classification"] != "final_policy_research_evaluation"
             or result["state"] != "evaluation_completed"
             or any(case.get("status") != "completed" or case.get("exit_code") != 0 for case in result["cases"])
            or sorted(case["height_m"] for case in result["cases"]) != [.29, .30, .31, .32]):
        raise ValueError("A evaluation is incomplete or differs from frozen identity/criteria")
    recomputed = []
    for height in (.29, .30, .31, .32):
        case = directory / f"h{round(height * 100):03d}"
        summary = json.loads((case / "summary.json").read_text())
        if summary["onnx_sha256"] != result["policy_sha256"] or summary["contract_sha256"] != A_SHA:
            raise ValueError("A evaluation case policy/contract mismatch")
        recomputed.append(summarize(case / "telemetry.csv", summary, height, 6000))
    return {"policy_sha256": result["policy_sha256"], "all_heights_pass": all(c["research_criteria_pass"] is True for c in recomputed),
            "cases": recomputed, "criteria": CRITERIA}


def decide(a_plan_path, evaluation, pilot_plan_path):
    a_plan = json.loads(Path(a_plan_path).read_text())
    pilot = json.loads(Path(pilot_plan_path).read_text())
    result = {"schema_version": 1, "next_action": "waiting_A_final", "B_main_started": False,
              "A_local_return_checked": False,
              "pilot_classification": "experiment_not_promoted", "policy_quality_verified": False,
              "criteria": CRITERIA, "a_run": a_plan["run_dir"], "pilot_run": pilot["run_dir"]}
    try:
        if a_plan["contract_sha256"] != A_SHA or pilot["source_run"] != a_plan["run_dir"]:
            raise ValueError("pipeline A identity mismatch")
        status_path = Path(pilot["audit_dir"]) / "worker.status.json"
        if not status_path.exists():
            result["next_action"] = "waiting_B1_pilot"
            return result
        status = json.loads(status_path.read_text())
        if status.get("resource_stop"):
            result.update(next_action="blocked_resources", reason=status["resource_stop"])
            return result
        if status.get("status") != "completed":
            result.update(next_action="blocked_pilot_or_export", reason=status.get("error", status.get("status")))
            return result
        result["pilot_checks"] = verify_run(pilot, {"name": "train", "requested_iterations": 500})
        if result["pilot_checks"]["completed_updates"] != 500:
            raise ValueError("B1 pilot did not finish 500 updates")
        verify_transfer(json.loads((Path(pilot["run_dir"]) / "run_manifest.json").read_text()), pilot)
        if training_alive(a_plan):
            return result
        final = verified_final(a_plan)
        if final is None:
            return result
        result["a_final_checkpoint_sha256"] = next(i["sha256"] for i in final["artifacts"] if i["path"] == "model_final.pt")
        report = verified_evaluation(evaluation, a_plan_path, a_plan)
        if report is None:
            result["next_action"] = "waiting_A_four_height_evaluation"
            return result
        result["a_evaluation"] = report
        if not report["all_heights_pass"]:
            result["next_action"] = "continue_A_optimization"
        else:
            result.update(next_action="prepare_B1_main_from_A_final",
                          next_parent="verified A model_final.pt; do not resume the engineering pilot",
                          rationale="A frozen four-height criteria pass and B1 engineering/export checks complete; no pilot reward promotion")
    except (OSError, ValueError, KeyError) as error:
        result.update(next_action="blocked_evidence_or_export", reason=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-plan", type=Path, required=True)
    parser.add_argument("--a-evaluation", type=Path, required=True)
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional local handoff.json; no other writes")
    args = parser.parse_args()
    result = decide(args.a_plan, args.a_evaluation, args.pilot_plan)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
