"""Read existing native rows only; no simulator, CUDA, PPO or source edits."""
from pathlib import Path
import gzip
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
ROOTS = {
    "source022_baseline": AUDIT / "exploration_gate_review_021/source_script_repair_native_candidate_022/native_SOURCE_SCRIPT_001/native",
    "immediate_v1": Path("/tmp/robot_rl_immediate_handoff_022/native001/native"),
}
LANES = (1708, 2216, 1877)
CONTROL_IDS = (0, 2, 8, 1, 3, 10)
CONTROL_NAMES = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
PHASE_NAMES = {0: "SELECT", 1: "FOLD", 2: "PLANT", 3: "ORBIT", 4: "THRUST", 5: "SIDE", 6: "CAPTURE", 7: "PREPARE", 8: "BLEND", 9: "RL", 10: "FAILED"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def norm(v):
    return math.sqrt(sum(a*a for a in v))


def tilt(v):
    return math.degrees(math.acos(max(-1., min(1., -v[2]/norm(v)))))


def motor(v):
    return [v[i] for i in CONTROL_IDS]


def posture(s):
    return dict(tilt_deg=tilt(s["gravity_body"]), gyro_norm_rad_s=norm(s["gyro_body"]),
                gravity_body=s["gravity_body"], gyro_body_rad_s=s["gyro_body"])


def analyze(root):
    files = ("identity.json", "report.json", "mechanical_5ms.jsonl.gz", "mechanical_summary.json", "first_birth_rows.jsonl.gz", "teaching_summary.json")
    before = {f: sha(root/f) for f in files}
    identity, report = read(root/"identity.json"), read(root/"report.json")
    mechanical, teaching = read(root/"mechanical_summary.json"), read(root/"teaching_summary.json")
    outcomes = {int(x["lane"]): x for x in rows(root/"first_birth_rows.jsonl.gz") if x.get("birth_id") == 1 and x.get("lane") in LANES}
    traces = {lane: [] for lane in LANES}
    for x in rows(root/"mechanical_5ms.jsonl.gz"):
        s = x["state"]
        if s.get("birth_id") == 1 and s.get("lane") in traces:
            traces[s["lane"]].append(x)
    lane_results = {}
    for lane, trace in traces.items():
        assert trace[0]["physical_step"] == 0
        assert all(b["physical_step"] == a["physical_step"] + 1 for a, b in zip(trace, trace[1:]))
        changes, key, handoff = [], None, None
        for i, x in enumerate(trace):
            s = x["state"]
            current = (s["phase"], s["route"], s["release_finished"])
            if current != key:
                changes.append(dict(physical_step=x["physical_step"], simulated_s=x["simulated_s"], phase=PHASE_NAMES[s["phase"]],
                                    route=s["route"], release_finished=s["release_finished"]))
                key = current
            if handoff is None and s["pure_rl"] and s["release_finished"]:
                assert i > 0 and x["applied_control"]["blend"] == 1.
                previous = trace[i-1]
                before_motor = motor(previous["state"]["applied_full_joint_effort"])
                after_motor = motor(s["applied_full_joint_effort"])
                jump = [b-a for a, b in zip(before_motor, after_motor)]
                fsm = x["applied_control"]["once_decoded_and_actual_delayed_PD"]["substep"]["fsm"]
                handoff = dict(
                    physical_step=x["physical_step"], simulated_s_after_physics=x["simulated_s"],
                    preceding_physical_step=previous["physical_step"], route=s["route"],
                    phase_before_update=PHASE_NAMES[fsm["phase_before"]], phase_after_update=PHASE_NAMES[fsm["phase_after"]],
                    truth_before_physics=posture(previous["state"]), truth_after_physics=posture(s),
                    sensor_input_tilt_deg=tilt(fsm["gravity"]), sensor_input_gyro_norm_rad_s=norm(fsm["gyro_rad_s"]),
                    sensor_input_height_m=fsm["conditional_height_m"], native_eight_gates=fsm["native_gate_masks"],
                    native_stable_pass=fsm["native_stable_pass"],
                    actual_applied_motor_before_nm=before_motor, actual_applied_motor_after_nm=after_motor,
                    actual_applied_motor_jump_nm=jump, max_abs_actual_motor_jump_nm=max(map(abs, jump)),
                    current_setter_matches_applied_motor=after_motor == motor(x["applied_control"]["full_joint_effort_target"]),
                    generated_script_target_control6=x["applied_control"]["script_targets_control6"],
                    prior_script_target_control6=previous["applied_control"]["script_targets_control6"],
                )
        o = outcomes[lane]
        assert (handoff is None and o["first_handover_reference_tick"] == -1) or (handoff is not None and handoff["physical_step"] == o["first_handover_reference_tick"])
        post = [x for x in trace if handoff is not None and x["physical_step"] >= handoff["physical_step"]]
        peak_s = max(x["state"]["native_FSM_after_reference_before_current_physics"].get("stable_peak_ticks", 0) for x in trace) * .005
        assert math.isclose(peak_s, o["recovery_stable_peak_seconds"], abs_tol=2e-7)
        lane_results[str(lane)] = dict(
            group=o["group"], lesson=o["lesson"], sampled_profile=o["sampled_profile"], phase_changes=changes,
            first_real_actor_handoff=handoff,
            before_handoff_recovery_action_phases=([p["phase"] for p in changes if handoff is not None and p["physical_step"] < handoff["physical_step"] and p["phase"] in ("FOLD", "PLANT", "SIDE", "ORBIT", "THRUST", "CAPTURE")]),
            post_handoff_max_truth_tilt_deg=max((tilt(x["state"]["gravity_body"]) for x in post), default=None),
            post_handoff_max_truth_gyro_norm_rad_s=max((norm(x["state"]["gyro_body"]) for x in post), default=None),
            last_observed_physical_step=trace[-1]["physical_step"], last_observed_simulated_s=trace[-1]["simulated_s"],
            last_observed_truth=posture(trace[-1]["state"]),
            first_birth_pin_max_gap_mm=1000 * max(x["state"]["pin_max_gap_m"] for x in trace),
            first_birth_pin_guard_3mm=mechanical["first_substep_guard_crossings"].get(str(lane)),
            outcome={k: o[k] for k in ("partition", "done", "terminated", "success", "timeout", "recovery_failed", "recovery_failure_code", "recovery_motion_released", "recovery_stable_seconds", "recovery_stable_peak_seconds", "recovery_stable_break_count", "recovery_rl_stable_s", "recovery_followup_s", "recovery_true_success", "first_handover_reference_tick", "first_motion_release_reference_tick")},
            termination_reasons=o.get("termination_reasons", []),
            censoring=("right-censored at native 12s; not episode timeout and not recovery success" if o["partition"] == "right_censored_unfinished" else "observed first native terminal"),
        )
    after = {f: sha(root/f) for f in files}
    assert before == after
    assert report["arm"] == "mean" and report["PPO_updates"] == 0 and report["physical_steps_completed"] == 2400
    return dict(root=str(root), input_sha256=before, input_bytes_unchanged_during_read=True,
                source_commit=identity["source_commit_label"], checkpoint_sha256=identity["checkpoint_sha256"],
                actual_actor_clocks=identity["observation_actor_clocks"], inference_arm=report["arm"],
                native_status=report["status"], policy_steps=report["policy_steps_completed"], physics_steps=report["physical_steps_completed"],
                PPO_updates=report["PPO_updates"], simulated_seconds=report["simulated_seconds"],
                full_4096_all_births_max_pin_gap_mm=mechanical["full_window_max_pin_gap_m"] * 1000,
                closure_guard_terminal_lanes_all_births=teaching["native_formal_probe_structural_audit"]["counts"]["closure_guard_terminal_lanes"],
                selected_first_birth_guard_crossings=mechanical["first_substep_guard_crossings"], lanes=lane_results)


def main():
    arms = {k: analyze(v) for k, v in ROOTS.items()}
    baseline, candidate = arms["source022_baseline"], arms["immediate_v1"]
    assert baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"]
    assert baseline["source_commit"] == candidate["source_commit"]
    endpoint_file = ROOTS["immediate_v1"] / "immediate_handoff_endpoint_events.jsonl.gz"
    endpoint_sha = sha(endpoint_file)
    events = {str(s["lane"]): dict(update_call=x["update_call"], native_reference_tick=x["native_reference_tick"], input_physical_step=x["input_physical_step"], **s)
              for x in rows(endpoint_file) for s in x.get("selected", []) if s["birth_id"] == 1 and s["lane"] in LANES}
    assert sha(endpoint_file) == endpoint_sha
    for lane in ("1708", "2216", "1877"):
        assert candidate["lanes"][lane]["first_real_actor_handoff"]["physical_step"] == events[lane]["update_call"]
        assert not candidate["lanes"][lane]["before_handoff_recovery_action_phases"]
    review = dict(
        schema="source022-readonly-native-posture-comparison-v1", status="native_complete_intended_full_action_handoff_not_tested",
        qualification=None, reviewed_historical_checkpoint="ownREC3076 mean; not current training actor near4875",
        units=dict(tilt="degree total inclination acos(-normalized body gravity z)", gyro="rad/s body-vector norm", torque="N.m actual robot.data.applied_torque on six rotary motor joints", pin_gap="mm", time="s simulation time"),
        control_order=CONTROL_NAMES, arms=arms, immediate_endpoint_events=events, endpoint_events_sha256=endpoint_sha,
        interpretation={
            "confirmed_endpoint_semantics_error": "V1 accepts initial SELECT->PREPARE(routeUPRIGHT), before any recovery FOLD/PLANT/SIDE/ORBIT/THRUST/CAPTURE. remaining is the residual of the current sensor-dependent goal; it does not identify full recovery-action completion.",
            "passive_phase9_excluded": "Before native passive release finishes, phase9 is a placeholder; actual state.pure_rl=false. Require release_finished and pure_rl for first actor ownership. Actual selected handoff is phys53=.265s (1708/2216), phys29=.145s (1877), not phys6=.030s.",
            "full_action_counterfactual_unmeasured": "Baseline 1708/2216 finish reroute/FOLD/SIDE/FOLD/ORBIT/THRUST/CAPTURE then final PREPARE; V1 ends the initial PREPARE and aborts before this chain. V1 outcome cannot establish whether immediate RL after completed self-righting action succeeds.",
            "two_layers_preserved": "Torque ownership changes to actor at handoff; original continuous1s sensor stable gate still gates subsequent vx/yaw release and physical evaluation. None of the three immediate selected births obtained 1s stable or command release.",
            "cross_arm_scope": "Same historical model/source/initial protocol, descriptive comparison. Earliest global intervention update4 precedes selected intervention, so post-intervention global RNG/resets can diverge; do not label whole window a strict paired counterfactual.",
            "mechanical_scope": "Selected first-birth guard absence is not global mechanical qualification. Immediate all4096/allbirth max6.052mm exceeds3mm; early selected terminations also shorten observation exposure.",
        },
        correction_design_only={
            "required": "An explicit per-birth original-phase-transition provenance must distinguish final PREPARE after a recovery action from initial SELECT->PREPARE; no control patch or rerun performed in this review.",
            "native_route_examples": "CAPTURE->PREPARE or completed PLANT->PREPARE; SIDE converted near-upright can also enter PREPARE. route!=UPRIGHT alone is insufficient, and everFOLD alone is weaker than recording the actual ending transition.",
            "avoid_new_success_gate": "After the correctly identified generated final reference endpoint, handoff must not wait for measured q, eight stable predicates,100ms ready, or200ms blend; preserve original post-RL1s motion release.",
        },
        review_process=dict(read_only_native_inputs=True, simulator_started=False, SSH=False, GPU_execution=False, PPO=False, frozen_core_or_driver_modified=False),
    )
    with (HERE / "posture_review_read001.json").open("x") as f:
        json.dump(review, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    lines = [
        "# source022 即时接管姿态只读复核", "",
        "此次 native 已完成 600 个策略步 / 2400 个物理步，0 PPO。用的是历史 ownREC3076 的 mean 策略，不代表当前训练 actor。",
        "",
        "候选接错了动作终点：1708/2216 是初始 SELECT→PREPARE（UPRIGHT）即交给 RL，此前没有 FOLD/ORBIT/CAPTURE。`remaining<.02` 是当前动态目标残差，不是完整自起动作结束。本次负结果保留，但不能据此判断完整动作结束后立即接管是否有效。",
        "",
        "| lane / 分支 | 首真实 RL 时刻 | 接管前→后真实倾角 | 后帧 gyro | 最大轴力矩跳变 | 最长连续传感稳定 | 结果 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, arm in arms.items():
        for lane in ("1708", "2216"):
            x = arm["lanes"][lane]
            h, o = x["first_real_actor_handoff"], x["outcome"]
            result = "12s 右删失，未放行" if not o["done"] else f"{x['last_observed_simulated_s']:.3f}s actor failure"
            lines.append(f"| {lane} / {name} | {h['simulated_s_after_physics']:.3f}s | {h['truth_before_physics']['tilt_deg']:.2f}→{h['truth_after_physics']['tilt_deg']:.2f}° | {h['truth_after_physics']['gyro_norm_rad_s']:.3f}rad/s | {h['max_abs_actual_motor_jump_nm']:.3f}N·m | {o['recovery_stable_peak_seconds']:.3f}s | {result} |")
    lines += ["", "首接管按实际 `release_finished && pure_rl` 认定。前置静置期的 phase9 占位不算 RL 接管，物理 6 / .030s 的记录不应用作首接管时刻。", "",
              "旧分支两 lane 在完整恢复末段 CAPTURE→PREPARE 后才接管，最后 12s 倾角约0.62° / 0.88°，但连续稳定仅0.180s / 0.155s，尚未达到1s和后续命令放行。候选两 lane 在 .300s 原生 actor_failure 终止，倾角约54° / 53°，稳定峰0，未放行。", "",
              f"力矩来自实际施加读回的六电机轴，顺序为 {', '.join(CONTROL_NAMES)}；每轴前后值及跳变量见 JSON。传感输入与物理前后真值分别记录，不能用延迟观测替代实际姿态。", "",
              "候选两选定首出生闭链峰未超3mm，但全4096 / 全出生最大仍6.052mm且有1个原生闭链终止；旧分支全局27.620mm。候选更早终止、全局干预改变后续重置，不能据此认证机械改善或严格逐lane因果效果。", "",
              "后续修正须记录每 birth 的原生末段→PREPARE 转移 provenance，排除初始 SELECT→PREPARE；不能只要求 route 非 UPRIGHT，也不能把1s稳定重新放到 actor 接管前。本复核没有修改冻结候选、driver或正式源码，也未启动仿真。", ""]
    with (HERE / "posture_review_read001.md").open("x") as f:
        f.write("\n".join(lines))
    print(json.dumps(dict(report=str(HERE / "posture_review_read001.json"), sha256=sha(HERE / "posture_review_read001.json"), markdown=str(HERE / "posture_review_read001.md"),
                          table=[dict(arm=name, lane=l, handoff_s=a["lanes"][l]["first_real_actor_handoff"]["simulated_s_after_physics"],
                                      pre_tilt_deg=a["lanes"][l]["first_real_actor_handoff"]["truth_before_physics"]["tilt_deg"],
                                      post_tilt_deg=a["lanes"][l]["first_real_actor_handoff"]["truth_after_physics"]["tilt_deg"],
                                      motor_jump_nm=a["lanes"][l]["first_real_actor_handoff"]["max_abs_actual_motor_jump_nm"],
                                      final_tilt_deg=a["lanes"][l]["last_observed_truth"]["tilt_deg"]) for name,a in arms.items() for l in ("1708","2216")]), ensure_ascii=False))


if __name__ == "__main__":
    main()
