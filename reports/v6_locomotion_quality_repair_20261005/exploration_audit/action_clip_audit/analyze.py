#!/usr/bin/env python3
"""Read pinned completed traces on CPU; never load actor, simulator or optimizer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ("stand_305mm", "forward_0p5", "backward_0p5")
BOUNDS = np.array([3., 3., 3., 3., 9., 9.])
CONTROL_FROM_POLICY = [0, 1, 4, 2, 3, 5]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_files(root, pins):
    for name, expected in pins.items():
        require(sha(root/name) == expected, f"Pinned input SHA mismatch: {root/name}")


def records(path, expected):
    """One-pass large-file SHA, verified before any report is written."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            result.update(line)
            yield json.loads(line)
    require(result.hexdigest() == expected, f"Pinned trace SHA mismatch: {path}")


def array(value, shape):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), "Trace tensor ABI/nonfinite value")
    return result


def regularization(x, previous, before):
    first, second = x-previous, x-2*previous+before
    return -.01*(first**2).sum(-1)-.05*(second[..., :4]**2).sum(-1)-.01*(second[..., 4:]**2).sum(-1)


def action_stats():
    return {"env_policy_rows": 0, "leg_axis_rows": 0, "wheel_axis_rows": 0,
            "mu_leg_clip": 0, "mu_wheel_clip": 0, "sent_leg_clip": 0, "sent_wheel_clip": 0,
            "potential_sample_leg_clip": 0, "potential_sample_wheel_clip": 0,
            "mu_max_abs_policy_order": np.zeros(6), "sent_max_abs_policy_order": np.zeros(6),
            "raw_regularizer_sum": 0., "clipped_regularizer_sum": 0.,
            "raw_clip_regularizer_differing_rows": 0, "maximum_absolute_density_difference": 0.}


def physical_stats():
    return {"env_physics_rows": 0, "leg_axis_rows": 0, "wheel_axis_rows": 0,
            "request_exceeds_leg_cap": 0, "request_exceeds_wheel_cap": 0,
            "applied_touches_leg_cap": 0, "applied_touches_wheel_cap": 0,
            "requested_absmax_control_order": np.zeros(6), "applied_absmax_control_order": np.zeros(6),
            "lastsub_env_policy_rows": 0, "lastsub_tau_squared_sum": 0., "lastsub_positive_wheel_power_sum": 0.,
            "body_vx_sum": 0., "projected_body_vx_sum": 0., "heading_world_vx_sum": 0.,
            "projected_mae_to_command_sum": 0., "heading_mae_to_command_sum": 0.,
            "abs_projected_minus_heading_sum": 0., "absolute_euler_pitch_deg_sum": 0.,
            "maximum_projection_factor_difference": 0., "maximum_body_world_rotation_error": 0.,
            "maximum_gravity_quaternion_gx_difference": 0., "maximum_com_link_velocity_difference": 0.}


def scopes(case, phase):
    return ("all", case, case+"/"+phase)


def analyze_run(directory, pin, manifest):
    small = {key: value for key, value in pin["files_sha256"].items() if not key.endswith(".jsonl")}
    check_files(directory, small)
    report, config, birth, cohort = (read(directory/name) for name in
                                   ("report.json", "clean_effective_contract.json", "birth.json", "first_cohort.json"))
    runtime, receipt = read(directory/"runtime.json"), read(directory/"clean_fixture_receipt.json")
    require(report["status"] == "native_probe_completed" and report["execution_status"] == "completed"
            and report["formal_updates_consumed"] == 0 and report["actor_tensors_unchanged"] is True
            and report["source_identity_final_verified"] is True and report["sealed_inputs_final_verified"] is True
            and report["qualification"] in (None, False), "Completed inference/unchanged actor scope differs")
    require(report["source_revision"] == pin["source_revision"] and report["arm"] == pin["arm"]
            and report.get("fixture_role", "clean") == pin["fixture_role"], "Actual arm/source/role differs")
    auth = report["input_authentication"]
    for key, name in (("checkpoint", "model.pt"), ("contract", "contract.json"),
                      ("seal", "model.checkpoint.json"), ("agent_config", "agent_config.json")):
        require(auth["files"][key]["sha256"] == manifest["actor_files_sha256"][name], "Actor/seal/factory input scope differs")
    require(auth["clocks"]["learning_updates"] == 24500 and auth["clocks"]["task_actor_update"] == 4185
            and auth["optimizer_steps"] == auth["formal_updates_consumed"] == 0
            and auth["native_cpu_factory"]["raw_std_policy_order"] == manifest["expected_actor_std"], "Actor clock/std scope differs")
    source = report["source_authentication"]
    require(source["base_core_revision"] == manifest["core_revision"] and source["complete_clean_checkout"] is True,
            "Frozen core/source authentication differs")
    for name, expected in manifest["core_files_sha256"].items():
        if name.startswith("src/"):
            require(source["actual_source_files_sha256"][name] == expected, "Actual frozen core differs: "+name)
    require(receipt["effective_contract_sha256"] == pin["files_sha256"]["clean_effective_contract.json"]
            and runtime["runtime_config_validation"] == read(directory/"runtime_config_validation.json")
            and runtime["runtime_config_validation"]["status"] == "passed", "Actual effective runtime binding differs")
    require(config["scut_effort_reward_scale"] == 1. and config["reward_velocity_reference"] == "base_link_origin"
            and config["v5_control"]["action_clip"] == 3. and config["v5_control"]["wheel_action_clip"] == 9.
            and config["v5_control"]["leg_kp"] == 160. and config["v5_control"]["leg_kd"] == 2.5
            and config["auto_reset"] is False, "Measured control/reward premise differs")
    names = birth["scene_groups"]
    require(len(names) == 12 and Counter(names) == {case: 4 for case in CASES}
            and cohort["fixed_birth_denominator"] == 12 and cohort["actual_physics_tick"] == 4000
            and cohort["partitions"] == {"failure": 0, "success": 0, "boundary_truncation": 0,
                                          "timeout": 12, "right_censored_unfinished": 0}, "Complete first-cohort scope differs")
    actions, astats, pstats = {}, {}, {}
    old = np.zeros((12, 6)); older = old.copy(); old_clip = old.copy(); older_clip = old.copy()
    action_lines, physical_lines, clip_verified = 0, 0, 0
    for row in records(directory/"actions.jsonl", pin["files_sha256"]["actions.jsonl"]):
        action_lines += 1; tick = row["policy_tick"]
        require(tick == action_lines and row["first_cohort_active"] == [True]*12, "Action clock/first cohort differs")
        require(len(row["phase"]) == 12, "Action phase layout differs")
        mu, sent, sample = (array(row[key], (12, 6)) for key in ("mu", "sent_policy_action", "sample"))
        require(np.array_equal(sent, mu if pin["arm"] == "mean" else sample), "Actual sent arm differs")
        clipped = np.clip(sent, -BOUNDS, BOUNDS)
        raw_cost, clip_cost = regularization(sent, old, older), regularization(clipped, old_clip, older_clip)
        actions[tick] = clipped[:, CONTROL_FROM_POLICY]
        for lane, case in enumerate(names):
            for key in scopes(case, row["phase"][lane]):
                stat = astats.setdefault(key, action_stats()); stat["env_policy_rows"] += 1
                stat["leg_axis_rows"] += 4; stat["wheel_axis_rows"] += 2
                for label, values in (("mu", mu[lane]), ("sent", sent[lane]), ("potential_sample", sample[lane])):
                    stat[label+"_leg_clip"] += int((np.abs(values[:4]) > 3.).sum())
                    stat[label+"_wheel_clip"] += int((np.abs(values[4:]) > 9.).sum())
                    if label != "potential_sample":
                        stat[label+"_max_abs_policy_order"] = np.maximum(stat[label+"_max_abs_policy_order"], np.abs(values))
                difference = float(raw_cost[lane]-clip_cost[lane])
                stat["raw_regularizer_sum"] += float(raw_cost[lane]); stat["clipped_regularizer_sum"] += float(clip_cost[lane])
                stat["raw_clip_regularizer_differing_rows"] += int(difference != 0.)
                stat["maximum_absolute_density_difference"] = max(stat["maximum_absolute_density_difference"], abs(difference))
        older, old, older_clip, old_clip = old, sent, old_clip, clipped
    require(action_lines == 1000, "Incomplete policy trace")
    for row in records(directory/"physical_frames.jsonl", pin["files_sha256"]["physical_frames.jsonl"]):
        physical_lines += 1; lane = row["lane"]; tick = row["physics_tick"]; policy = row["policy_tick"]
        require(lane == (physical_lines-1) % 12 and tick == (physical_lines-1)//12+1
                and policy == (tick-1)//4+1 and row["first_cohort"] is True
                and row["case"] == names[lane], "Physical order/first-cohort identity differs")
        require(np.array_equal(array(row["clipped_control_actions"], (6,)), actions[policy][lane]), "Actual native clip differs")
        clip_verified += 1
        req, tau, bounds, dq = (array(row[key], (6,)) for key in
                               ("requested_motor_effort", "applied_torque6", "current_motor_bounds", "dq6"))
        require(np.array_equal(bounds, [40., 40., 4.5, 40., 40., 4.5]), "Actual physical motor cap differs")
        vb, vw, gravity, pose, com = (array(row[key], shape) for key, shape in
            (("root_link_velocity_body_m_s", (3,)), ("root_link_velocity_world_m_s", (3,)),
             ("projected_gravity_b", (3,)), ("root_pose_world_xyzw", (7,)), ("root_com_velocity_world_m_s", (3,))))
        x, y, z, w = pose[3:]; sine = np.clip(2*(w*y-z*x), -1., 1.); pitch = math.asin(sine)
        cosine = math.cos(pitch); factor = math.sqrt(max(0., 1.-gravity[0]**2))
        yaw = math.atan2(2*(w*z+x*y), 1.-2*(y*y+z*z))
        projected, heading = vb[0]*factor, vw[0]*math.cos(yaw)+vw[1]*math.sin(yaw)
        vector = np.array([x, y, z]); cross = np.cross(vector, vb)
        rotated = vb+2*(w*cross+np.cross(vector, cross))
        for key in scopes(row["case"], row["phase"]):
            stat = pstats.setdefault(key, physical_stats()); stat["env_physics_rows"] += 1
            stat["leg_axis_rows"] += 4; stat["wheel_axis_rows"] += 2
            for suffix, indices in (("leg", [0, 1, 3, 4]), ("wheel", [2, 5])):
                stat["request_exceeds_"+suffix+"_cap"] += int((np.abs(req[indices]) > bounds[indices]).sum())
                stat["applied_touches_"+suffix+"_cap"] += int((np.abs(tau[indices]) >= bounds[indices]-1e-6).sum())
            stat["requested_absmax_control_order"] = np.maximum(stat["requested_absmax_control_order"], np.abs(req))
            stat["applied_absmax_control_order"] = np.maximum(stat["applied_absmax_control_order"], np.abs(tau))
            if tick % 4 == 0:
                stat["lastsub_env_policy_rows"] += 1; stat["lastsub_tau_squared_sum"] += float((tau**2).sum())
                stat["lastsub_positive_wheel_power_sum"] += float(np.maximum(tau[[2, 5]]*dq[[2, 5]], 0.).sum())
            for label, value in (("body_vx", vb[0]), ("projected_body_vx", projected), ("heading_world_vx", heading),
                    ("projected_mae_to_command", abs(projected-row["command"][0])),
                    ("heading_mae_to_command", abs(heading-row["command"][0])),
                    ("abs_projected_minus_heading", abs(projected-heading)), ("absolute_euler_pitch_deg", abs(math.degrees(pitch)))):
                stat[label+"_sum"] += float(value)
            for label, value in (("projection_factor_difference", abs(factor-cosine)),
                    ("body_world_rotation_error", np.abs(rotated-vw).max()),
                    ("gravity_quaternion_gx_difference", abs(gravity[0]-sine)),
                    ("com_link_velocity_difference", np.abs(com-vw).max())):
                stat["maximum_"+label] = max(stat["maximum_"+label], float(value))
    require(physical_lines == clip_verified == 48000 and pstats["all"]["lastsub_env_policy_rows"] == 12000,
            "Incomplete physical/reward denominator")
    for stat in astats.values():
        n = stat["env_policy_rows"]
        stat["raw_regularizer_density_mean"] = stat["raw_regularizer_sum"]/n
        stat["clipped_regularizer_density_mean"] = stat["clipped_regularizer_sum"]/n
        stat["raw_minus_clip_density_mean"] = (stat["raw_regularizer_sum"]-stat["clipped_regularizer_sum"])/n
        for key, value in list(stat.items()):
            if isinstance(value, np.ndarray): stat[key] = value.tolist()
    for stat in pstats.values():
        n = stat["lastsub_env_policy_rows"]
        stat["motor_cost_density_mean"] = -1e-4*stat["lastsub_tau_squared_sum"]/n
        stat["wheel_power_cost_density_mean"] = -1e-4*stat["lastsub_positive_wheel_power_sum"]/n
        stat["combined_effort_density_mean"] = stat["motor_cost_density_mean"]+stat["wheel_power_cost_density_mean"]
        stat["same_trace_scut_coefficient_effort_density_mean"] = .1*stat["combined_effort_density_mean"]
        for key in list(stat):
            if key.endswith("_sum") and not key.startswith("lastsub_"):
                stat[key[:-4]+"_mean"] = stat.pop(key)/stat["env_physics_rows"]
            elif isinstance(stat[key], np.ndarray): stat[key] = stat[key].tolist()
    return {"arm": pin["arm"], "fixture_role": pin["fixture_role"], "source_revision": pin["source_revision"],
        "formal_updates_consumed": 0, "qualification": False, "actor_tensors_unchanged": True,
        "policy_lines": action_lines, "physical_lines": physical_lines,
        "actual_native_clip_verified_frames": clip_verified, "fixed_birth_denominator": 12,
        "case_replica_counts": dict(Counter(names)), "native_terminal_partitions": cohort["partitions"],
        "actions_by_case_phase": astats, "physical_by_case_phase": pstats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parents[3])
    parser.add_argument("--scut-root", type=Path)
    parser.add_argument("--output", type=Path, default=HERE/"audit.json")
    args = parser.parse_args(); root = args.root.resolve(); manifest = read(HERE/"inputs.json")
    require(not args.output.exists(), "Refusing to overwrite audit output")
    check_files(root, manifest["core_files_sha256"])
    check_files(root/manifest["actor_input_directory"], manifest["actor_files_sha256"])
    check_files(args.scut_root or root.parent/"scut-wheeled-legged-rl", manifest["scut_files_sha256"])
    base = root/"reports/v6_locomotion_quality_repair_20261005/exploration_audit"
    runs = {}
    for name, pin in manifest["runs"].items():
        runs[name] = analyze_run(base/name, pin, manifest)
        print(json.dumps({"verified_run": name, "first_cohort_physics_frames": runs[name]["physical_lines"]}), flush=True)
    result = {"schema": "same-PT-action-clip-effort-projection-audit-v1", "status": "pinned_CPU_analysis_passed",
        "qualification": False, "root_cause_verified": False, "candidate_implemented_or_deployed": False,
        "scope": "Completed 24500 inference only; 0/+/-.5m/s, N4 per case, 20s; no PPO or high-speed/full-DR claim",
        "input_manifest_sha256": sha(HERE/"inputs.json"), "analysis_script_sha256": sha(Path(__file__)),
        "formulas": {"action_regularizer_density": "-.01 sum((a-a_prev)^2)-.05 sum_leg((a-2a_prev+a_before)^2)-.01 sum_wheel((a-2a_prev+a_before)^2)",
            "history": "Actual sent actions, zero at birth, retained across phases; raw/clip differ only at native action bounds, including next two history rows",
            "effort_density": "-1e-4*(sum(actual applied torque6^2)+sum positive(actual wheel torque*dq)); last 200Hz substep before each 50Hz reward",
            "reward_payment": "densities * policy_dt(.02); effort values reconstruct these two components, not full logged reward or a learned counterfactual",
            "ours_projection": "body link-origin vx*sqrt(max(1-projected_gravity_b.x^2,0))",
            "scut_projection": "source root_lin_vel_b.x*cos(Euler pitch); no SCUT physical trace evaluated",
            "equivalence": "For unit gravity and standard Euler pitch: gx=sin(pitch), sqrt(1-gx^2)=abs(cos(pitch)); normal canonical Euler pitch matches",
            "heading_caveat": "Not full inverse heading velocity; pure horizontal world motion at zero roll/yaw gives projected vx=world vx*cos(pitch)^2",
            "velocity_origin_caveat": "Ours pins base_link_origin. SCUT calls classic root_lin_vel_b; saved run lacks SDK identity to authenticate its historical velocity origin"},
        "candidate_single_variable": {"field": "scut_effort_reward_scale", "before": 1., "after": .1,
            "per_step_reward_delta": "+9e-5*(sum(torque6^2)+sum positive(wheel torque*dq))*.02",
            "same_PT_inference_gain_expected": False, "learning_benefit_verified": False}, "runs": runs}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha(args.output), "qualification": False}))


if __name__ == "__main__":
    main()
