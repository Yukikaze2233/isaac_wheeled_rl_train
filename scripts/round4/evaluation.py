#!/usr/bin/env python3
"""Prepare final-policy evaluation; dispatch only an explicitly supplied, committed evaluator."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import uuid

from r4_common import JobError, atomic_bytes, file_record, json_bytes, strict_json, verify_training_run, verify_snapshot


def evaluation_request(plan, delta_xy):
    return {
        **{key: plan[key] for key in ("physics_model", "physics_asset_manifest_sha256", "repaired_dynamics_used",
                                      "explicit_user_authorization")},
        "schema_version": 2, "training_plan": str(Path(plan["audit_dir"]) / "plan.json"),
        "run_dir": plan["run_dir"], "initialization": "scratch", "parent": None,
        "ground_usd": plan["ground_usd"], "ground_record": plan["ground_record"],
        "contract_sha256": plan["contract_sha256"], "seed": 44,
        "policy": str(Path(plan["run_dir"]) / "policy.onnx"),
        "random_push_enabled": False, "policy_quality_verified": False,
        "cases": [
            {"id": f"{kind}_h{round(height * 100):03d}", "kind": kind,
             "command": [0, 0, height], "episodes": 20 if kind == "single_push" else 3,
             "episode_seconds": 20, "push": None if kind == "standing" else
             {"method": "env.apply_velocity_impulse(delta_xy, env_ids)", "time_s": 5,
              "delta_xy_m_s": delta_xy, "count_per_episode": 1}}
            for kind in ("standing", "single_push") for height in (.29, .30, .31, .32)
        ] + [
            {"id": name, "kind": "tracking", "command": command, "episodes": 2,
             "episode_seconds": 20, "push": None}
            for name, command in (
                ("tracking_vx_pos3", [3, 0, .32]), ("tracking_vx_neg3", [-3, 0, .32]),
                ("tracking_wz_pos6", [0, 6, .32]), ("tracking_wz_neg6", [0, -6, .32]))
        ],
        "recovery": {
            "deadline_s": 2, "steady_band_hold_s": 1,
            "height_error_band_m": .01, "planar_speed_band_m_s": .05,
            "tilt_band_deg": 10, "hold_must_finish_within_deadline": False,
            "denominator": "all actually disturbed episodes, including termination/nonfinite/no recovery",
            "required_counts": ["scheduled_episodes", "pre_push_failed_episodes", "disturbed_episodes",
                                "recovered_within_2s_episodes", "failed_disturbed_episodes", "censored_episodes"],
            "censored_must_not_count_as_recovered": True,
        },
        "contact_metric": "net_contact is diagnostic only; no ground-pair <=5N claim",
        "reward_threshold": None,
        "report_protocol": "report_manifest.json binds request_sha256 and files; result.json contains every case id/status/metrics; raw CSV required",
    }


def prepare(plan, delta_xy):
    checks = verify_training_run(plan)
    if not checks["formal_training_target_completed"]:
        raise JobError("evaluation requires cumulative 30000 updates and verified final export")
    root = Path(plan["stage_root"]) / "evaluation"
    if delta_xy is None:
        delta_xy = [plan["requested_profile"]["push"]["schedule"][-1][1], 0.0]
    request = evaluation_request(plan, delta_xy)
    request["training_progress"] = {key: checks[key] for key in
                                    ("prior_completed_updates", "invocation_completed_updates", "cumulative_completed_updates")}
    if plan.get("initialization") == "resume":
        request["resume_parent_checkpoint_sha256"] = plan["parent"]["checkpoint_sha256"]
    request["policy_sha256"] = file_record(Path(request["policy"]))["sha256"]
    path = root / "request.json"
    if root.exists():
        if strict_json(path.read_bytes()) != request:
            raise JobError("existing evaluation request differs; preserve its evidence")
    else:
        root.mkdir()
        atomic_bytes(path, json_bytes(request))
    return root, path, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--push-delta-v", type=float, nargs=2)
    parser.add_argument("--evaluator", help="Python script in this or a separately verified committed experiment")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.push_delta_v is not None and (not all(math.isfinite(v) for v in args.push_delta_v)
                                         or not any(args.push_delta_v)):
        parser.error("push delta-v must be finite and nonzero")
    plan = strict_json(args.plan.read_bytes())
    args.evaluator = args.evaluator or plan.get("evaluation_entry")
    # The remote completion hook and a reconnecting laptop can arrive together.
    lock = (Path(plan["audit_dir"]) / "evaluation-dispatch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    root, path, request = prepare(plan, args.push_delta_v)
    result = {"state": "awaiting_evaluator_implementation", "request": str(path),
              "request_sha256": file_record(path)["sha256"], "policy_quality_verified": False}
    if args.evaluator and (args.launch or plan.get("evaluation_entry")):
        entry = Path(args.evaluator)
        if not entry.is_absolute():
            entry = Path(plan["repo"]) / entry
        code = next((parent for parent in entry.parents if parent.name == "isaac_wheeled_rl_train"), None)
        if code is None or ".." in entry.parts or entry.suffix != ".py":
            raise JobError("evaluator must be in an isolated committed code snapshot")
        snapshot = verify_snapshot(code.parent)
        if str(entry.relative_to(code.parent)) not in snapshot["files"]:
            raise JobError("evaluator is not included in the verified source snapshot")
        result.update(evaluator=str(entry), evaluator_commit=snapshot["git_commit"],
                      evaluator_snapshot_sha256=file_record(code.parent / "snapshot.json")["sha256"])
        submitted = root / "submission.json"
        if submitted.exists():
            previous = strict_json(submitted.read_bytes())
            if previous.get("evaluator") != str(entry) or previous.get("evaluator_commit") != snapshot["git_commit"]:
                raise JobError("evaluator already submitted from another source; inspect it before retrying")
            print(json.dumps(previous))
            return 0
        # An interrupted submission is ambiguous, never silently launch a second evaluator.
        with (root / "submission.claim").open("x") as claim:
            claim.write(result["request_sha256"])
        session = "round4-eval-" + uuid.uuid4().hex
        shell = ('source "$1" || exit; shift; export LIVESTREAM=0 ENABLE_CAMERAS=0; '
                 'exec "$@"')
        argv = ["tmux", "new-session", "-d", "-s", session, "-c", str(code),
                "timeout", "--signal=TERM", "--kill-after=120s", "9000", "bash", "-c", shell,
                "round4-evaluation", plan["runtime"], plan["python"], str(entry),
                "--request", str(path), "--output", str(root / "reports")]
        reply = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        result.update(state="evaluator_submitted" if reply.returncode == 0 else "evaluator_submission_failed",
                      session=session, command=argv, stderr=reply.stderr, evaluation_success=False)
        atomic_bytes(submitted, json_bytes(result))
    elif args.evaluator:
        result["state"] = "awaiting_explicit_evaluator_launch"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
