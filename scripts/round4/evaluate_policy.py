#!/usr/bin/env python3
"""Execute the frozen final-policy standing/push/tracking matrix; never train a policy."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/round4"), str(REPO / "scripts"), str(REPO / "src")]
from r4_common import file_record, verify_training_manifest, verify_training_run, verify_snapshot
from wheeled_algo.v40_job import TrainingBudget, atomic_bytes, json_bytes, strict_json

RECOVERY = {"deadline_s": 2, "steady_band_hold_s": 1,
            "height_error_band_m": .01, "planar_speed_band_m_s": .05,
            "tilt_band_deg": 10, "hold_must_finish_within_deadline": False}
OLD_ASSET_SHA = "df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886"
TRACKING_COMMANDS = {
    "tracking_vx_pos3": [3, 0, .32], "tracking_vx_neg3": [-3, 0, .32],
    "tracking_wz_pos6": [0, 6, .32], "tracking_wz_neg6": [0, -6, .32],
}
FIELDS = ["episode", "step", "time_s", "policy_tick", "finite", "height_m", "height_error_m",
          "x_m", "y_m", "vx_m_s", "vy_m_s", "vz_m_s", "planar_speed_m_s", "wz_rad_s",
          "tilt_deg", "gravity_z", "terminated", "timeout", "non_wheel_net_force_n",
          "left_wheel_net_force_n", "right_wheel_net_force_n", "clearance_m", "action_norm",
          "diagnostic_flags_json", "termination_flags_json", "command_vx_m_s", "command_wz_rad_s"]


def validate_request(request):
    if request.get("schema_version") != 2 or request.get("random_push_enabled") is not False:
        raise ValueError("evaluation requires request protocol v2 and disabled random training pushes")
    if request.get("initialization") != "scratch" or request.get("parent") is not None or request.get("policy_quality_verified") is not False:
        raise ValueError("request must identify scratch origin without claiming policy quality")
    if any(request.get("recovery", {}).get(key) != value for key, value in RECOVERY.items()):
        raise ValueError("recovery protocol mismatch: require .01m/.05m_s/10deg, hold 1s, hold START within 2s")
    expected = {f"{kind}_h{height:03d}" for kind in ("standing", "single_push") for height in (29, 30, 31, 32)}
    expected.update(TRACKING_COMMANDS)
    cases = request.get("cases", [])
    if len(cases) != 12 or {case.get("id") for case in cases} != expected:
        raise ValueError("request must contain exactly the twelve reviewed standing/push/tracking cases")
    for case in cases:
        if case["id"] in TRACKING_COMMANDS:
            if (case["kind"] != "tracking" or case["command"] != TRACKING_COMMANDS[case["id"]]
                    or case["episodes"] != 2 or case["episode_seconds"] != 20 or case["push"] is not None):
                raise ValueError("tracking cases require the reviewed command, two 20s episodes and no push")
            continue
        height = int(case["id"][-3:]) / 100
        kind = case["kind"]
        if (case["id"] != f"{kind}_h{round(height * 100):03d}" or case["command"] != [0, 0, height]
                or case["episodes"] != (3 if kind == "standing" else 20) or case["episode_seconds"] != 20):
            raise ValueError("case command/count/duration differs from reviewed matrix")
        push = case["push"]
        if kind == "standing":
            if push is not None:
                raise ValueError("standing cases cannot include pushes")
        elif (not isinstance(push, dict) or push.get("time_s") != 5 or push.get("count_per_episode") != 1
              or push.get("method") != "env.apply_velocity_impulse(delta_xy, env_ids)"):
            raise ValueError("single-push cases require exactly one observation-boundary impulse at 5s")
        else:
            delta = push.get("delta_xy_m_s")
            if (type(delta) is not list or len(delta) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in delta)
                    or not any(delta)):
                raise ValueError("push delta must be two finite, nonzero world-XY velocity increments")


def verify_inputs(request_path, *, create_policy=True):
    """Completion, source snapshot, asset and ONNX gates all precede AppLauncher."""
    from wheeled_tasks.v40.contract import contract_digest, load_contract, validate_asset
    from wheeled_algo.v40_ground import verify_cached_ground
    from play_v40_onnx import load_policy

    request_record = file_record(request_path)
    request = strict_json(request_path.read_bytes())
    validate_request(request)
    plan = strict_json(Path(request["training_plan"]).read_bytes())
    run = Path(request["run_dir"]).resolve()
    if (run != Path(plan["run_dir"]).resolve() or run != (Path(plan["stage_root"]) / "train").resolve()
            or Path(request["policy"]).resolve() != run / "policy.onnx"
            or Path(plan["repo"]).resolve() != REPO.resolve()):
        raise ValueError("evaluation must use this training run and its frozen repository")
    snapshot = verify_snapshot(Path(plan["experiment"]))
    member = str(Path(__file__).resolve().relative_to(Path(plan["experiment"]).resolve()))
    if (member not in snapshot["files"] or not re.fullmatch(r"[0-9a-f]{40}", snapshot["git_commit"])
            or snapshot["git_commit"] != plan["git_commit"]):
        raise ValueError("evaluator is not part of the committed training snapshot")
    checks = verify_training_run(plan)
    if not checks["formal_training_target_completed"]:
        raise ValueError("evaluation requires a verified final after all 30000 updates")
    if request.get("training_progress") != {key: checks[key] for key in
            ("prior_completed_updates", "invocation_completed_updates", "cumulative_completed_updates")}:
        raise ValueError("evaluation request cumulative progress mismatch")
    if plan.get("initialization") == "resume" and request.get("resume_parent_checkpoint_sha256") != plan["parent"]["checkpoint_sha256"]:
        raise ValueError("evaluation request resume parent mismatch")
    manifest = strict_json((run / "run_manifest.json").read_bytes())
    verify_training_manifest(manifest, plan)
    if manifest["training_curriculum"]["completed_updates"] != 30000:
        raise ValueError("final curriculum clock differs from the verified 30000 updates")
    contract = load_contract(run / "contract.json")
    if contract_digest(load_contract(REPO / snapshot["contract"])) != contract_digest(contract):
        raise ValueError("run contract differs from the frozen training contract")
    hashes = strict_json((run / "source_hashes.json").read_bytes()).get("source_files_sha256")
    if type(hashes) is not dict or not hashes or any(
            snapshot["files"].get(REPO.name + "/" + name, {}).get("sha256") != digest
            for name, digest in hashes.items()):
        raise ValueError("training source hashes differ from the frozen evaluator repository")
    asset = validate_asset(contract, allow_research=True)
    if (contract_digest(contract) != request["contract_sha256"] or request["contract_sha256"] != manifest["contract_sha256"]
            or asset["asset_manifest_sha256"] != manifest["asset_manifest_sha256"]
            or manifest["asset_manifest_sha256"] != OLD_ASSET_SHA):
        raise ValueError("evaluation contract/old research asset identity mismatch")
    ground = verify_cached_ground(Path(request["ground_usd"]))
    if any(ground[key] != request["ground_record"][key] for key in ("sha256", "size")):
        raise ValueError("request ground differs from the byte-pinned official asset")
    if file_record(run / "policy.onnx")["sha256"] != request.get("policy_sha256"):
        raise ValueError("request ONNX SHA mismatch")
    if create_policy:
        session, sidecar = load_policy(run / "policy.onnx", manifest)
    else:
        # verify_run already checked the sidecar; preserve the initial ORT session.
        session, sidecar = None, strict_json((run / "policy.onnx.json").read_bytes())
    if file_record(request_path) != request_record:
        raise ValueError("evaluation request changed during verification")
    identity = {"request_sha256": request_record["sha256"], "policy_sha256": sidecar["onnx_sha256"],
                "contract_sha256": manifest["contract_sha256"], "asset_manifest_sha256": OLD_ASSET_SHA,
                "source_commit": snapshot["git_commit"],
                "source_snapshot_sha256": file_record(Path(plan["experiment"]) / "snapshot.json")["sha256"],
                "completion_sha256": file_record(run / "completion.json")["sha256"],
                "run_manifest_sha256": sidecar["run_manifest_sha256"]}
    identity["training_progress"] = request["training_progress"]
    if plan.get("initialization") == "resume":
        identity["resume_parent_checkpoint_sha256"] = plan["parent"]["checkpoint_sha256"]
    return request, plan, manifest, contract, session, identity


def binomial_information(failures, trials):
    if trials == 0:
        return {"trials": 0, "failure_rate": None, "wilson_95": None}
    z = 1.959963984540054
    p = failures / trials
    center = (p + z*z/(2*trials)) / (1 + z*z/trials)
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / (1 + z*z/trials)
    return {"trials": trials, "failure_rate": p,
            "wilson_95": [max(0., center-radius), min(1., center+radius)],
            "zero_failure_one_sided_95_upper": 1 - .05**(1/trials) if failures == 0 else None,
            "assumption": "nominal independent Bernoulli trials; repeated deterministic resets/materials may be correlated"}


def analyse_episode(rows, case, event):
    """Statistics from raw CSV samples; auto-reset never supplies recovery evidence."""
    applied = event.get("push_applied", False)
    push_time = event.get("push_time_s")
    start = recovered_start = None
    failed = event.get("ending") in {"termination", "nonfinite"}
    height_errors, speeds, tilts = [], [], []
    timeouts = samples = steady_samples = 0
    duration = 0.
    for row in rows:
        samples += 1
        duration = float(row["time_s"])
        finite = bool(int(row["finite"]))
        terminated = bool(int(row["terminated"]))
        timeouts += int(row["timeout"])
        failed |= terminated or not finite
        if finite:
            error, speed, tilt = (abs(float(row["height_error_m"])), float(row["planar_speed_m_s"]), float(row["tilt_deg"]))
            height_errors.append(error)
            speeds.append(speed)
            tilts.append(tilt)
            steady_samples += int(not terminated and error <= RECOVERY["height_error_band_m"]
                                  and speed <= RECOVERY["planar_speed_band_m_s"] and tilt <= RECOVERY["tilt_band_deg"])
        if applied and duration >= push_time:
            within = (finite and not terminated and error <= RECOVERY["height_error_band_m"]
                      and speed <= RECOVERY["planar_speed_band_m_s"] and tilt <= RECOVERY["tilt_band_deg"])
            if within:
                if start is None:
                    start = duration
                if (recovered_start is None and duration-start + 1e-9 >= RECOVERY["steady_band_hold_s"]
                        and start-push_time <= RECOVERY["deadline_s"] + 1e-9):
                    recovered_start = start-push_time
            else:
                start = None
    censored = not failed and event.get("ending") not in {"horizon", "termination", "nonfinite"}
    # A later failure/censoring never turns an earlier band entry into a successful trial.
    recovered = bool(applied and recovered_start is not None and not failed and not censored)
    return {**event, "samples": samples, "valid_samples": len(height_errors), "steady_band_samples": steady_samples,
            "observed_seconds": duration, "timeout_events": timeouts,
            "pre_push_failed": bool(case["kind"] == "single_push" and not applied and failed),
            "failed": bool(failed or (applied and not recovered and not censored)), "censored": censored,
            "recovered_within_2s": recovered, "detected_recovery_start_after_push_s": recovered_start,
            "height_abs_error_mean_m": sum(height_errors)/len(height_errors) if height_errors else None,
            "height_abs_error_max_m": max(height_errors) if height_errors else None,
            "planar_speed_max_m_s": max(speeds) if speeds else None, "tilt_max_deg": max(tilts) if tilts else None}


def tracking_statistics(rows, command):
    """Sample-weighted errors; steady means each episode's time >2s, not a pass grade."""
    rows = list(rows)
    result = {}
    for window, selected in (("full", rows), ("steady_after_2s", [r for r in rows if float(r["time_s"]) > 2])):
        valid = [r for r in selected if int(r["finite"]) and all(math.isfinite(float(r[k])) for k in ("vx_m_s", "wz_rad_s"))]
        stats = {"samples": len(selected), "valid_samples": len(valid)}
        for axis, key, target, unit in (("vx", "vx_m_s", command[0], "m/s"), ("wz", "wz_rad_s", command[1], "rad/s")):
            actual = [float(r[key]) for r in valid]
            errors = sorted(abs(value-target) for value in actual)
            p95 = None
            if errors:
                position = .95 * (len(errors)-1)
                lower = int(position)
                p95 = errors[lower] + (position-lower) * (errors[min(lower+1, len(errors)-1)]-errors[lower])
            stats[axis] = {"command": target, "unit": unit,
                           "actual_mean": sum(actual)/len(actual) if actual else None,
                           "actual_min": min(actual) if actual else None, "actual_max": max(actual) if actual else None,
                           "mae": sum(errors)/len(errors) if errors else None, "p95_abs_error": p95}
        result[window] = stats
    diagnostics = {}
    terminations = {}
    for row in rows:
        for column, counts in (("diagnostic_flags_json", diagnostics), ("termination_flags_json", terminations)):
            for name, active in json.loads(row[column] or "{}").items():
                counts[name] = counts.get(name, 0) + int(active)
    result["diagnostic_sample_counts"] = diagnostics
    result["termination_sample_counts"] = terminations
    result["contact_peak_n"] = {
        key: max((float(r[key]) for r in rows if r[key] and math.isfinite(float(r[key]))), default=None)
        for key in ("non_wheel_net_force_n", "left_wheel_net_force_n", "right_wheel_net_force_n")}
    result["semantics"] = "absolute tracking errors; P95 linear interpolation; observed pre-reset samples including failed/censored episodes; no acceptance threshold"
    return result


def summarize_csv(path, case, events):
    import itertools
    episodes = []
    with path.open(newline="") as stream:
        groups = {int(key): list(rows) for key, rows in itertools.groupby(csv.DictReader(stream), key=lambda row: row["episode"])}
    for event in events:
        rows = groups.get(event["episode"], [])
        episode = analyse_episode(rows, case, event)
        if case["kind"] == "tracking":
            episode["tracking"] = tracking_statistics(rows, case["command"])
        episodes.append(episode)
    disturbed = [episode for episode in episodes if episode.get("push_applied") is True]
    failures = sum(episode["failed"] for episode in disturbed)
    censored = sum(episode["censored"] for episode in disturbed)
    count = len(disturbed)
    valid = sum(e["valid_samples"] for e in episodes)
    metrics = {"scheduled_episodes": case["episodes"], "started_episodes": len(episodes),
               "unstarted_episodes": case["episodes"]-len(episodes),
               "pre_push_failed_episodes": sum(e["pre_push_failed"] for e in episodes),
               "disturbed_episodes": count, "recovered_within_2s_episodes": sum(e["recovered_within_2s"] for e in disturbed),
               "failed_disturbed_episodes": failures, "failed_episodes": sum(e["failed"] for e in episodes),
               "censored_episodes": sum(e["censored"] for e in episodes), "censored_disturbed_episodes": censored,
               "unverified_push_attempts": sum(e.get("push_attempted", False) and e.get("push_applied") is not True for e in episodes),
               "observed_sim_seconds": sum(e["observed_seconds"] for e in episodes),
               "requested_accumulated_seconds": case["episodes"] * case["episode_seconds"],
               "duration_semantics": "independent reset episodes including timeouts; NOT a continuous 60-second standing trial",
               "timeout_events": sum(e["timeout_events"] for e in episodes),
               "height_abs_error_mean_m": sum(e["height_abs_error_mean_m"]*e["valid_samples"] for e in episodes if e["valid_samples"])/valid if valid else None,
               "height_abs_error_max_m": max((e["height_abs_error_max_m"] for e in episodes if e["valid_samples"]), default=None),
               "planar_speed_max_m_s": max((e["planar_speed_max_m_s"] for e in episodes if e["valid_samples"]), default=None),
               "tilt_max_deg": max((e["tilt_max_deg"] for e in episodes if e["valid_samples"]), default=None),
               "steady_band_fraction_valid_samples": sum(e["steady_band_samples"] for e in episodes)/valid if valid else None,
               "resolved_episode_binomial": binomial_information(sum(e["failed"] for e in episodes), sum(not e["censored"] for e in episodes)),
               "failure_rate_bounds_all_disturbed": [failures/count, min(1., (failures+censored)/count)] if count else None,
               "resolved_disturbed_binomial": binomial_information(failures, count-censored)}
    if case["kind"] == "tracking":
        metrics["tracking"] = tracking_statistics((row for rows in groups.values() for row in rows), case["command"])
    return {"id": case["id"], "status": "completed" if len(episodes) == case["episodes"] and not any(e["censored"] for e in episodes) else "failed",
            "kind": case["kind"], "command": case["command"], "metrics": metrics, "episodes": episodes}


def snapshot_row(snapshot, episode, step, dt, command, action_norm):
    def scalar(key):
        return float(snapshot[key][0].item())
    def vector(key):
        return snapshot[key][0].detach().cpu().tolist()
    if snapshot["sample_kind"] != "pre_reset" or snapshot["state_phase"] != "pre_reset":
        raise ValueError("evaluation requires the owned pre-reset transition snapshot")
    velocity, position, gravity = vector("root_com_lin_vel_b_m_s"), vector("root_link_pos_w_m"), vector("projected_gravity_b")
    wheels = vector("wheel_net_force_max_n")
    flags = {name: bool(value[0].item()) for name, value in snapshot["diagnostic_flags"].items()}
    numbers = [*velocity, *position, *gravity, *wheels, scalar("height_m"), scalar("non_wheel_net_force_max_n"),
               scalar("base_visual_clearance_lower_bound_m"), action_norm]
    finite = all(math.isfinite(value) for value in numbers) and not flags["nonfinite"]
    norm = math.sqrt(sum(v*v for v in gravity)) if finite else 0.
    finite &= norm > 0
    tilt = math.degrees(math.acos(max(-1., min(1., -gravity[2]/norm)))) if finite else float("nan")
    return dict(zip(FIELDS, [episode, step, step*dt, snapshot["policy_tick"], int(finite), scalar("height_m"),
        scalar("height_m")-command[2], *position[:2], *velocity, math.hypot(*velocity[:2]),
        vector("root_com_ang_vel_b_rad_s")[2], tilt, gravity[2], int(scalar("terminated")), int(scalar("timeout")),
        scalar("non_wheel_net_force_max_n"), *wheels, scalar("base_visual_clearance_lower_bound_m"), action_norm,
        json.dumps(flags, sort_keys=True, separators=(",", ":")),
        json.dumps({k: bool(v[0].item()) for k, v in snapshot["termination_flags"].items()}, sort_keys=True, separators=(",", ":")),
        command[0], command[1]]))


def apply_push(env, obs, delta, *, evidence=None):
    """Fresh backend before/after evidence, then same-tick observation refresh."""
    import torch
    evidence = {} if evidence is None else evidence
    ids = torch.tensor([0], dtype=torch.long, device=env.device)
    before, pose_before = env._read_root_impulse_state(ids)
    if not torch.isfinite(before).all() or not torch.isfinite(pose_before).all():
        evidence.update(push_applied=False, ending="nonfinite", readback_before_nonfinite=True)
        raise FloatingPointError("nonfinite backend state before the scheduled impulse")
    evidence.update(requested_delta_xy_m_s=delta, before_com_velocity_w_m_s_rad_s=before.cpu().tolist()[0])
    report = env.apply_velocity_impulse(torch.tensor([delta], dtype=torch.float32, device=env.device), ids)
    after, pose_after = env._read_root_impulse_state(ids)
    realized = after[:, :2]-before[:, :2]
    if torch.isfinite(after).all() and torch.isfinite(realized).all():
        evidence.update(after_com_velocity_w_m_s_rad_s=after.cpu().tolist()[0],
                        realized_delta_xy_m_s=realized.cpu().tolist()[0])
    if (report["events_count"] != 1 or not torch.isfinite(after).all()
            or not bool((torch.linalg.vector_norm(realized, dim=-1) > 0).all())
            or not torch.allclose(realized, realized.new_tensor([delta]), atol=1e-5, rtol=0)
            or not torch.allclose(realized, report["realized_delta_xy_m_s"], atol=1e-5, rtol=0)
            or not torch.equal(before[:, 2:], after[:, 2:])
            or not torch.allclose(pose_before, pose_after, atol=1e-7, rtol=0)):
        raise RuntimeError("single impulse failed actual delta/vertical/angular/pose readback")
    # A later observation-refresh failure must not erase a verified disturbance.
    evidence.update(push_applied=True, pose_unchanged=True, vertical_angular_unchanged=True,
                    readback_source="fresh PhysX root_view before and after explicit impulse")
    refreshed = env._get_observations()
    if not torch.equal(obs["policy"], refreshed["policy"]):
        raise RuntimeError("velocity-only impulse unexpectedly changed actor history/command")
    if refreshed["critic"].shape != (1, 29) or not torch.isfinite(refreshed["critic"]).all():
        raise RuntimeError("invalid refreshed critic observation after impulse")
    if torch.equal(obs["critic"][:, 25:28], refreshed["critic"][:, 25:28]):
        raise RuntimeError("critic linear velocity is stale after the nonzero impulse")
    evidence["critic_velocity_refreshed"] = True
    return refreshed, evidence


def run_case(env, policy, case, directory, check):
    """One env, explicit episode resets; no post-reset sample can count as recovery."""
    import numpy as np
    import torch
    directory.mkdir()
    path = directory / "telemetry.csv"
    events = []
    error = None
    dt = float(env.step_dt)
    if not math.isclose(dt, .01, abs_tol=1e-12):
        raise ValueError("evaluation requires the audited 100Hz policy clock")
    steps = round(case["episode_seconds"]/dt)
    with path.open("x", newline="") as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        try:
            for episode in range(case["episodes"]):
                check()
                event = {"episode": episode, "ending": "interrupted", "push_applied": False, "push_attempted": False}
                events.append(event)
                env.set_evaluation_command(tuple(case["command"]))
                obs, _ = env.reset()
                for tick in range(steps):
                    check()
                    if env._evaluation_command_override != tuple(case["command"]):
                        raise RuntimeError("evaluation override changed; random training commands/pushes must remain disabled")
                    if case["push"] is not None and tick == round(case["push"]["time_s"]/dt):
                        event.update(push_attempted=True, push_applied=None, push_time_s=tick*dt)
                        obs, _ = apply_push(env, obs, case["push"]["delta_xy_m_s"], evidence=event)
                    inputs = np.ascontiguousarray(obs["policy"].detach().cpu().numpy(), dtype=np.float32)
                    if inputs.shape != (1, 125) or not np.isfinite(inputs).all():
                        event["ending"] = "nonfinite"
                        event["observation_nonfinite"] = True
                        break
                    actions = policy.run(["actions"], {"obs_history": inputs})[0]
                    if actions.shape != (1, 6) or actions.dtype != np.float32 or not np.isfinite(actions).all():
                        raise RuntimeError("ONNX returned invalid actions")
                    obs, reward, terminated, timeout, _ = env.step(torch.from_numpy(actions).to(env.device))
                    if env.extras.get("log", {}).get("Push/events_count", 0) != 0:
                        raise RuntimeError("random training push fired during fixed-command evaluation")
                    snapshot = env.get_evaluation_snapshot()
                    if (bool(terminated.item()) != bool(snapshot["terminated"].item())
                            or bool(timeout.item()) != bool(snapshot["timeout"].item())):
                        raise RuntimeError("transition reset flags disagree with the pre-reset snapshot")
                    if not np.array_equal(snapshot["command"][0].cpu().numpy(), np.asarray(case["command"], dtype=np.float32)):
                        raise RuntimeError("reward snapshot command differs from the requested policy command")
                    row = snapshot_row(snapshot, episode, tick+1, dt, case["command"], float(np.linalg.norm(actions)))
                    if not torch.isfinite(reward).all():
                        row["finite"] = 0
                    # Nine significant digits round-trip float32 telemetry while
                    # keeping each 20-episode CSV inside the transfer size budget.
                    writer.writerow({key: format(value, ".9g") if type(value) is float else value
                                     for key, value in row.items()})
                    if not row["finite"] or row["terminated"]:
                        event["ending"] = "nonfinite" if not row["finite"] else "termination"
                        break
                    if row["timeout"]:
                        event["ending"] = "horizon" if (tick+2)*dt >= case["episode_seconds"]-1e-9 else "early_timeout"
                        break
                else:
                    event["ending"] = "horizon"
                stream.flush()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    summary = summarize_csv(path, case, events)
    if error:
        summary.update(status="failed", error=error)
    atomic_bytes(directory / "summary.json", json_bytes(summary))
    return summary


def training_device(plan):
    argv = plan["command"]
    for index, value in enumerate(argv):
        if value == "--device":
            return argv[index+1]
        if value.startswith("--device="):
            return value.split("=", 1)[1]
    return "cuda:0"


def effective_materials(report):
    """Average-combine effective coefficients from verified wheel and ground readback."""
    ground = [b for b in report["bindings"] if b["collider"].startswith("/World/ground/")]
    if (len(ground) != 1 or ground[0]["usd_combine_modes"] != ["average", "average"]
            or report.get("passed") is not True):
        raise ValueError("effective friction requires verified ground average-combine identity")
    ground_values = ground[0]["usd_coefficients"]
    return [{"env_id": env_id, "wheel_body_path": path,
             "static_friction": (row[0][0]+ground_values[0])/2,
             "dynamic_friction": (row[0][1]+ground_values[1])/2,
             "restitution": (row[0][2]+ground_values[2])/2}
            for env_id, path, row in zip(report["wheel_env_ids"], report["wheel_body_paths"], report["wheel_physx_coefficients"])]


def publish_reports(output, request, result):
    """Publish every case, including explicitly unrun cases, then the SHA marker last."""
    by_id = {case["id"]: case for case in result["cases"]}
    for case in request.get("cases", []):
        if (case.get("id") not in TRACKING_COMMANDS
                and not re.fullmatch(r"(?:standing|single_push)_h0(?:29|30|31|32)", str(case.get("id")))):
            continue
        if case["id"] not in by_id:
            directory = output / case["id"]
            directory.mkdir(exist_ok=True)
            csv_path = directory / "telemetry.csv"
            if not csv_path.exists():
                with csv_path.open("x", newline="") as stream:
                    csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
            summary = summarize_csv(csv_path, case, [])
            summary["error"] = result.get("error", "evaluation stopped before this case")
            atomic_bytes(directory / "summary.json", json_bytes(summary))
            result["cases"].append(summary)
    atomic_bytes(output / "result.json", json_bytes(result))
    lines = ["# Round4 final-policy evaluation", "", f"Execution status: **{result['status']}**", "",
             "Standing: three independent nominal 20s episodes per height (60s requested in total), including timeouts; not continuous 60s.",
             "Recovery: height error ≤0.01m, planar speed ≤0.05m/s, tilt ≤10°, continuously for 1s; hold start ≤2s after the verified impulse.",
             "Net contact forces are diagnostics, not identified ground-pair contacts. Material values are the recorded N=1 startup readback, not assumed μ=0.5.",
             "", "| Case | Execution | Failed episodes | Pre-push failures | Disturbed | Recovered ≤2s | Failed disturbed | Censored disturbed | Observed s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case in result["cases"]:
        m = case["metrics"]
        lines.append(f"| {case['id']} | {case['status']} | {m['failed_episodes']} | {m['pre_push_failed_episodes']} | {m['disturbed_episodes']} | {m['recovered_within_2s_episodes']} | {m['failed_disturbed_episodes']} | {m['censored_disturbed_episodes']} | {m['observed_sim_seconds']:.2f} |")
    lines += ["", "## Per-case rates and tracking", ""]
    for case in result["cases"]:
        m = case["metrics"]
        lines.append(f"- **{case['id']}**: height absolute error mean/max [m] = {m['height_abs_error_mean_m']} / {m['height_abs_error_max_m']}; "
                     f"failure-rate bounds over all disturbed = {m['failure_rate_bounds_all_disturbed']}; "
                     f"resolved-episode Wilson 95% = {m['resolved_episode_binomial']['wilson_95']}.")
    lines += ["", "## Tracking (no push; descriptive errors, not acceptance)", "",
              "Steady window is time >2s independently within each episode; failed/censored samples are not discarded.", "",
              "| Case / window | Valid samples | Command vx / wz | Actual mean vx / wz | vx MAE / P95 [m/s] | wz MAE / P95 [rad/s] |",
              "|---|---:|---|---|---|---|"]
    for case in result["cases"]:
        if case["kind"] == "tracking":
            for window in ("full", "steady_after_2s"):
                stats = case["metrics"]["tracking"][window]
                vx, wz = stats["vx"], stats["wz"]
                lines.append(f"| {case['id']} / {window} | {stats['valid_samples']} | {vx['command']} / {wz['command']} | "
                             f"{vx['actual_mean']} / {wz['actual_mean']} | {vx['mae']} / {vx['p95_abs_error']} | {wz['mae']} / {wz['p95_abs_error']} |")
    lines += ["", "Confidence: 0 failures in 20 independent trials still gives a one-sided 95% upper failure bound of about 13.9%, not <1%. Repeated fixed resets/pushes/material assignments can be correlated; intervals are descriptive, not certification.",
              "Censored trials never count as recovered. JSON includes failure-rate bounds over all verified disturbed trials, and intervals for resolved trials.",
              "", "Policy quality is not automatically certified by successful report execution.", "",
              "```json", json.dumps(result.get("identity", {}), indent=2), "```", ""]
    if result.get("error"):
        lines += ["Failure: " + result["error"], ""]
    atomic_bytes(output / "evaluation_report.md", "\n".join(lines).encode())
    files = [{"path": str(path.relative_to(output)), **file_record(path)}
             for path in sorted(output.rglob("*")) if path.is_file()]
    atomic_bytes(output / "report_manifest.json", json_bytes({
        "schema_version": 1, **result["identity"], "status": result["status"], "files": files,
        "policy_quality_verified": False}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", help="Explicit physics override; default follows the training command")
    parser.add_argument("--max-wall-seconds", type=float, default=8700,
                        help="Includes validation/startup; reserves time before the dispatcher's 9000s timeout")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error("wall budget must be finite and positive")
    args.output.mkdir(parents=False, exist_ok=False)
    request, env, app = {}, None, None
    result = {"schema_version": 1, "status": "failed", "evaluation_success": False,
              "policy_quality_verified": False, "cases": [], "identity": {}}
    started = time.monotonic()
    budget = TrainingBudget(args.max_wall_seconds)
    budget.begin_learning()
    try:
        with budget.signal_handlers():
            result["identity"]["request_sha256"] = file_record(args.request)["sha256"]
            request = strict_json(args.request.read_bytes())
            request, plan, manifest, contract, policy, identity = verify_inputs(args.request)
            result["identity"] = identity
            from train_v40 import launch_app, make_env, preflight
            device = args.device or training_device(plan)
            options = SimpleNamespace(contract=Path(request["run_dir"]) / "contract.json",
                usd_cache_dir=args.output.parent / "evaluation-usd-cache", ground_usd=Path(request["ground_usd"]),
                stage=manifest["stage"], num_envs=1, seed=request["seed"], research=True, headless=True, device=device)
            report, _, _ = preflight(options)
            atomic_bytes(args.output / "preflight.json", json_bytes(report))
            if not report["ready"]:
                raise RuntimeError("evaluation runtime preflight failed: " + "; ".join(report["blockers"]))
            app = launch_app(options, budget=budget).app
            with budget.signal_handlers():
                env = make_env(options)
                if (env.contract_sha256 != manifest["contract_sha256"] or env.asset_manifest_sha256 != OLD_ASSET_SHA
                        or env.num_envs != 1 or len(env.robot.body_names) != 7):
                    raise RuntimeError("runtime environment differs from the old 7-body research asset")
                materials = env.check_round3_materials()
                if materials.get("passed") is not True:
                    raise RuntimeError("evaluation material readback failed")
                result["environment"] = {"physics_device": str(env.device), "training_device": training_device(plan),
                    "physics_device_override": device != training_device(plan), "onnx_provider": "CPUExecutionProvider",
                    "num_envs": 1, "training_num_envs": manifest["num_envs"], "seed": request["seed"],
                    "body_names": list(env.robot.body_names), "domain_randomization_report": materials,
                    "effective_materials": effective_materials(materials),
                    "material_scope": "actual N=1 startup bucket/mask and shape readback; not assumed nominal friction",
                    "random_training_pushes": False}
                def check():
                    budget.check()
                    if not app.is_running():
                        raise InterruptedError("simulation application stopped")
                for case in request["cases"]:
                    summary = run_case(env, policy, case, args.output / case["id"], check)
                    result["cases"].append(summary)
                    if summary.get("error"):
                        raise RuntimeError(summary["error"])
                if any(case["status"] != "completed" for case in result["cases"]):
                    raise RuntimeError("one or more evaluation cases were censored/incomplete")
                # Revalidate artifacts/source after a long evaluation before publication.
                _, _, _, _, _, final_identity = verify_inputs(args.request, create_policy=False)
                if final_identity != identity:
                    raise RuntimeError("source artifacts changed during evaluation")
                result.update(status="completed", evaluation_success=True)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Publish failure evidence before possibly slow Kit cleanup on TERM/timeout.
        published = result["status"] == "failed"
        if published:
            result["wall_seconds"] = time.monotonic()-started
            publish_reports(args.output, request, result)
        for resource in (env, app):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    if published:
                        print(f"cleanup failed after failure publication: {exc}", file=sys.stderr)
                    else:
                        result.update(status="failed", evaluation_success=False, error=f"cleanup failed: {type(exc).__name__}: {exc}")
        result["wall_seconds"] = time.monotonic()-started
        if not published:
            publish_reports(args.output, request, result)
    print(json.dumps({"status": result["status"], "output": str(args.output), "policy_quality_verified": False}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
