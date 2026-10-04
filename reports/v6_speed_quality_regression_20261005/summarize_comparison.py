#!/usr/bin/env python3
"""Read-only postprocessing of native paired evaluation, with TensorBoard output.

All-replica scores remain the evaluator's scores. Additional timing and posture
statistics describe only the first representative of each case; they award no
qualification and perform no optimizer or simulator operations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np


LABELS = ("flat", "speed")
CHECKPOINT_SHA256 = (
    "7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f",
    "123841b609439939e9c9c8962c0171330892276446a692c936c8f33375ce11da",
)
PHASES = ("stationary", "acceleration", "cruise", "brake", "stop")
METRICS = ("vx_actual_mean_m_s", "vx_mae_m_s", "planar_speed_mean_m_s",
           "yaw_mae_rad_s", "height_mae_m", "tilt_max_deg",
           "stand_drift_max_m", "settled_stop_speed", "failures",
           "boundary_truncations", "frames", "observed_frames")
CURVE_CASES = {"stand_230mm", "stand_305mm", "stand_430mm", "forward_05",
               "backward_05", "rotate_1", "rotate_1_reverse", "start_stop_05",
               "start_stop_1", "forward_2", "backward_2", "forward_3", "backward_3"}
SALVAGE_SCRIPT_SHA256 = "e1c9f091211c7e0bdaed8eea0c19bc73db55051b50f97116df3d2d6cfaad6cd5"
SALVAGE_CONTRACT_SHA256 = "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7"
SALVAGE_CASE_ORDER_SHA256 = "4bd6b7f6ebc6257dedd7a76c586a41acb9246a4365b45a85c4084983a9c23127"
CAPTURE_SHAPES = {name: () for name in (
    "active", "episode_ticks", "done", "terminated", "reference_phase", "reference_height",
    "reference_vz", "reference_ax", "physical_phase", "jump_requested", "whole_com_vz",
    "reference_height_error", "com_displacement", "request_events_delivered", "accepted_request_count",
    "repeat_request_count", "cancel_request_count", "cancel_phase", "repeat_clock_reset_errors",
    "completed_jump_count", "successful_jump_count", "command_sequence_steps", "command_sequence_length")}
CAPTURE_SHAPES.update(values=(16,), commands=(3,), actions=(6,), motor_velocity=(6,),
                      wheel_target_velocity=(2,))
BROKEN_PIPE_PATTERN = (
    r'Traceback \(most recent call last\):\n  File "[^"\n]+/source_122d8db63f37/scripts/evaluate_chassis\.py", '
    r'line 273, in main\n    print\("V5_FIXED_EVAL", json\.dumps\(result, allow_nan=False\), flush=True\)\n'
    r'BrokenPipeError: \[Errno 32\] Broken pipe\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite JSON capture")
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def validate_capture(directory, report, contract, *, salvage=False):
    """Authenticate captured diagnostics without changing a failed run's status.

    Only the pinned evaluator's final stdout print may be salvaged. Candidate
    files and all source snapshots must match their original capture receipts;
    complete traces are checked separately before any output is written.
    """
    finite_json(report)
    if len(report.get("candidates", [])) != 2:
        raise ValueError("Exactly two captured candidates required")
    salvaged = report.get("status") == "failed"
    if salvaged:
        if not salvage or not re.fullmatch(BROKEN_PIPE_PATTERN, report.get("error", "")):
            raise ValueError("Only the explicit pinned final stdout BrokenPipeError can be salvaged")
        if report.get("script_sha256") != SALVAGE_SCRIPT_SHA256 or sha(directory / "contract.json") != SALVAGE_CONTRACT_SHA256:
            raise ValueError("Salvage source/contract identity differs from the frozen comparison")
    elif report.get("status") != "evaluated":
        raise ValueError("A completed native evaluation or explicitly salvaged diagnostic capture is required")
    sources = report.get("source_sha256", {})
    required_sources = {"scripts/evaluate_chassis.py", "src/wheeled_tasks/chassis/env.py",
                        "src/wheeled_tasks/chassis/eval_env.py", "src/wheeled_tasks/chassis/evaluation.py",
                        "src/wheeled_tasks/chassis/rewards.py", "src/wheeled_tasks/chassis/preflight.py"}
    if not required_sources <= sources.keys() or sources["scripts/evaluate_chassis.py"] != report.get("script_sha256"):
        raise ValueError("Missing or inconsistent source capture")
    for name, digest in sources.items():
        path = directory / "source" / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Source SHA mismatch: {name}")
    if report.get("contract_sha256") != sha(directory / "contract.json"):
        raise ValueError("Contract SHA mismatch")
    cases = [case["name"] for case in report["settings"]["cases"]]
    if len(cases) != len(set(cases)) or report.get("evaluated_case_names") != cases:
        raise ValueError("Case coverage/order mismatch")
    if salvaged and hashlib.sha256(json.dumps(cases, separators=(",", ":")).encode()).hexdigest() != SALVAGE_CASE_ORDER_SHA256:
        raise ValueError("Salvage case order differs from the frozen 87-case manifest")
    manifest = {case["name"]: case for case in contract["evaluation"]["cases"]}
    if any(case != manifest.get(case["name"]) for case in report["settings"]["cases"]):
        raise ValueError("Case definitions differ from the immutable contract")
    if ({key: value for key, value in report["settings"].items() if key != "cases"}
            != {key: value for key, value in contract["evaluation"].items() if key != "cases"}):
        raise ValueError("Evaluation settings differ from the immutable contract")
    repeats = report.get("episodes_per_case")
    if repeats != report["settings"]["episodes_per_case"] or (salvaged and repeats != 4):
        raise ValueError("Replica coverage differs from the declared capture")
    for index, candidate in enumerate(report["candidates"]):
        path = directory / f"candidate_{index:02d}.json"
        if json.loads(path.read_text()) != candidate:
            raise ValueError("Embedded and standalone candidate JSON differ")
        if (set(candidate.get("cases", {})) != set(cases) or candidate.get("unfinished_episodes") != 0
                or candidate.get("checkpoint_sha256") != CHECKPOINT_SHA256[index]
                or (salvaged and candidate.get("policy_ticks") != 1600)
                or candidate.get("evaluated_asset_manifest_sha256") != report.get("asset_manifest_sha256")
                or candidate.get("actor_input_dim") != contract.get("actor_dim")
                or candidate.get("actor_output_dim") != contract.get("action_dim")):
            raise ValueError("Candidate identity/coverage/completeness mismatch")
        for case in candidate["cases"].values():
            if case.get("episodes") != repeats or case.get("requested_episodes") != repeats:
                raise ValueError("Incomplete per-case replica capture")
    return {"status": "complete_diagnostic_capture_after_final_stdout_failure" if salvaged else "evaluated_capture",
            "original_evaluation_status": report["status"], "original_error": report.get("error"),
            "qualification_awarded": False, "source_file_count_authenticated": len(sources),
            "candidate_json_sha256": [sha(directory / f"candidate_{i:02d}.json") for i in range(2)],
            "case_count": len(cases), "replicas_per_case": repeats}


def validate_trace_arrays(arrays, ticks, cases, bodies, *, salvaged=False):
    required = set(CAPTURE_SHAPES) if salvaged else {"values", "commands", "episode_ticks", "active", "done", "terminated"}
    if not required <= arrays.keys() or "body_poses" not in arrays:
        raise ValueError("Missing captured trace arrays")
    for key, value in arrays.items():
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"Nonfinite or nonnumeric captured array: {key}")
        shape = ((ticks + 3) // 4, cases, bodies, 7) if key == "body_poses" else (ticks, cases, *CAPTURE_SHAPES.get(key, value.shape[2:]))
        if value.shape != shape:
            raise ValueError(f"Incomplete/misaligned captured array: {key}")
    for key in ("active", "done", "terminated"):
        if arrays[key].dtype.kind != "b":
            raise ValueError(f"Expected boolean capture: {key}")
    if arrays["episode_ticks"].dtype.kind not in "iu":
        raise ValueError("Expected integer episode clock capture")
    active, done = arrays["active"], arrays["done"]
    if not active[0].all() or (active[1:] & ~active[:-1]).any() or not ((active & done).sum(0) == 1).all():
        raise ValueError("Incomplete representative episode/terminal capture")
    if (active[-1] & ~done[-1]).any():
        raise ValueError("Unfinished representative capture")
    episode_ticks = arrays["episode_ticks"]
    pairs = active[1:] & active[:-1]
    if not (episode_ticks[0] == 1).all() or not ((np.diff(episode_ticks, axis=0) == 1) | ~pairs).all():
        raise ValueError("Discontinuous active episode clock capture")
    if (np.linalg.norm(arrays["body_poses"][..., 3:7], axis=-1) <= 0).any():
        raise ValueError("Invalid captured body quaternion")


def attitude_xyzw(quaternion):
    """Roll and pitch of base_link from normalized world quaternion xyzw."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("Invalid representative base quaternion")
    x, y, z, w = np.moveaxis(quaternion / norm, -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.degrees(roll), np.degrees(pitch)


def classify(commands, valid, ticks, dt):
    """Classify applied command phases and independent post-release windows.

    Reversal is braking while the command moves toward zero, then acceleration
    after the sign crossing. Stop includes every actual zero-command frame after
    motion, regardless of measured residual speed or stability.
    """
    count = len(commands)
    masks = {name: np.zeros(count, dtype=bool) for name in PHASES}
    quiet = np.zeros(count, dtype=bool)
    stop_id = np.full(count, -1, dtype=np.int64)
    derivative = np.zeros((count, 2), dtype=np.float64)
    seen_motion, zero_age, current_stop = False, 0., -1
    for index in range(count):
        continuous = index > 0 and valid[index - 1] and ticks[index] == ticks[index - 1] + 1
        if not valid[index]:
            seen_motion, zero_age, current_stop = False, 0., -1
            continue
        if not continuous:
            seen_motion, zero_age, current_stop = False, 0., -1
        current = commands[index, :2]
        previous = commands[index - 1, :2] if continuous else current
        derivative[index] = (current - previous) / dt
        moving = np.max(np.abs(current)) >= .01
        if not moving:
            phase = "stop" if seen_motion else "stationary"
            if seen_motion:
                if zero_age == 0:
                    current_stop = index
                stop_id[index] = current_stop
                zero_age += dt
                quiet[index] = zero_age > 2. + 1e-9
        else:
            zero_age, current_stop = 0., -1
            changed = np.abs(current - previous) > 1e-6
            toward_zero = changed & (np.abs(current) < np.abs(previous) - 1e-6)
            phase = "brake" if toward_zero.any() else "acceleration" if changed.any() else "cruise"
            seen_motion = True
        masks[phase][index] = True
    return masks, quiet, stop_id, derivative


def finite_mean(values):
    return float(np.mean(values)) if len(values) else None


def stats(values, commands, selected, pair_valid, dt, pose_ticks, roll, pitch, pose_valid):
    frame = values[selected]
    differences = selected & pair_valid
    delta_torque = np.diff(values[:, 10:16], axis=0)[differences[1:]]
    delta_omega = np.diff(values[:, 3:6], axis=0)[differences[1:]]
    chosen_poses = pose_valid & selected[pose_ticks]
    error = frame[:, :2] - np.column_stack((commands[selected, 0], np.zeros(selected.sum())))
    return {
        "policy_samples": int(selected.sum()), "sim_seconds": float(selected.sum() * dt),
        "vx_actual_mean_m_s": finite_mean(frame[:, 0]),
        "vx_mae_m_s": finite_mean(np.abs(error[:, 0])),
        "planar_speed_body_xy_mean_m_s": finite_mean(np.linalg.norm(frame[:, :2], axis=-1)),
        "yaw_mae_rad_s": finite_mean(np.abs(frame[:, 5] - commands[selected, 1])),
        "height_mae_m": finite_mean(np.abs(frame[:, 6] - commands[selected, 2])),
        "omega_xy_rms_rad_s": (float(np.sqrt(np.mean(frame[:, 3:5] ** 2))) if len(frame) else None),
        "representative_rms_motor_torque_nm": (np.sqrt(np.mean(frame[:, 10:16] ** 2, axis=0)).tolist()
                                                if len(frame) else None),
        "torque_step_change_rms_nm_50hz_proxy": (np.sqrt(np.mean(delta_torque ** 2, axis=0)).tolist()
                                                 if len(delta_torque) else None),
        "omega_step_change_rms_rad_s_50hz_proxy": (np.sqrt(np.mean(delta_omega ** 2, axis=0)).tolist()
                                                   if len(delta_omega) else None),
        "attitude_samples": int(chosen_poses.sum()),
        "pitch_mean_deg": finite_mean(pitch[chosen_poses]),
        "pitch_abs_mean_deg": finite_mean(np.abs(pitch[chosen_poses])),
        "pitch_abs_max_deg": (float(np.max(np.abs(pitch[chosen_poses]))) if chosen_poses.any() else None),
        "roll_mean_deg": finite_mean(roll[chosen_poses]),
        "roll_abs_mean_deg": finite_mean(np.abs(roll[chosen_poses])),
        "roll_abs_max_deg": (float(np.max(np.abs(roll[chosen_poses]))) if chosen_poses.any() else None),
    }


def representative(arrays, case_index, body_index, dt, warmup):
    values = arrays["values"][:, case_index].astype(np.float64)
    commands = arrays["commands"][:, case_index].astype(np.float64)
    ticks = arrays["episode_ticks"][:, case_index]
    valid = (arrays["active"][:, case_index].astype(bool)
             & ~arrays["done"][:, case_index].astype(bool)
             & ~arrays["terminated"][:, case_index].astype(bool)
             & (ticks * dt > warmup))
    if values.shape[-1] != 16 or commands.shape[-1] != 3:
        raise ValueError("Unknown native trace column layout")
    if not np.isfinite(values[valid]).all() or not np.isfinite(commands[valid]).all():
        raise ValueError("Nonfinite active representative trace")
    pair_valid = np.zeros(len(valid), dtype=bool)
    pair_valid[1:] = valid[1:] & valid[:-1] & (ticks[1:] == ticks[:-1] + 1)
    poses = arrays["body_poses"][:, case_index, body_index]
    pose_ticks = np.arange(len(poses)) * 4
    if pose_ticks[-1] >= len(valid):
        raise ValueError("Body-pose trace is not aligned with tick%4")
    pose_valid = valid[pose_ticks]
    roll, pitch = attitude_xyzw(poses[:, 3:7])
    masks, quiet, stop_ids, derivative = classify(commands, valid, ticks, dt)
    phases = {name: stats(values, commands, selected, pair_valid, dt,
                         pose_ticks, roll, pitch, pose_valid) for name, selected in masks.items()}
    signals = {key: arrays[key][:, case_index].astype(np.float64)
               for key in ("actions", "motor_velocity", "motor_position") if key in arrays}
    if "actions" in signals:
        delta = np.diff(signals["actions"], axis=0)
        delta_second = np.diff(delta, axis=0)
        for name, selected in masks.items():
            first = delta[(selected & pair_valid)[1:]]
            second = delta_second[(selected & pair_valid & np.roll(pair_valid, 1))[2:]]
            phases[name]["action_step_change_rms_50hz_proxy"] = (np.sqrt(np.mean(first ** 2, axis=0)).tolist()
                                                                 if len(first) else None)
            phases[name]["action_second_difference_rms_50hz_proxy"] = (np.sqrt(np.mean(second ** 2, axis=0)).tolist()
                                                                       if len(second) else None)
    stop_windows = []
    for stop_id in np.unique(stop_ids[stop_ids >= 0]):
        selected = stop_ids == stop_id
        indices = np.flatnonzero(selected)
        displacement = np.linalg.norm(values[selected, 7:9] - values[stop_id, 7:9], axis=-1)
        stop_windows.append({
            "start_time_s": float(ticks[stop_id] * dt), "policy_samples": int(selected.sum()),
            "sim_seconds": float(selected.sum() * dt),
            "max_displacement_from_stop_entry_m": float(displacement.max()),
            "last_displacement_from_stop_entry_m": float(displacement[-1]),
            "final_planar_speed_body_xy_m_s": float(np.linalg.norm(values[indices[-1], :2])),
            "quiet_samples_after_two_seconds": int((selected & quiet).sum()),
        })
    phases["stop"]["quiet_after_two_seconds"] = stats(values, commands, quiet, pair_valid,
                                                       dt, pose_ticks, roll, pitch, pose_valid)
    phases["stop"]["windows"] = stop_windows
    result = {"scope": "first representative only; not all-replica score", "phases": phases}
    return result, (values, commands, ticks, valid, masks, derivative, pose_ticks, roll, pitch, pose_valid, signals)


def write_events(path, summary, curves):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(path))
    def scalars(prefix, measured):
        for metric, value in measured.items():
            tag = f"{prefix}/{metric}"
            if metric == "settled_stop_speed" and measured.get("settled_stop_metric_valid") is False:
                continue  # Preserve raw JSON zero but never publish it as a successful stop.
            if isinstance(value, (int, float)):
                writer.add_scalar(tag, value, 0)
            elif isinstance(value, dict):
                scalars(tag, value)
            elif isinstance(value, list) and all(isinstance(number, (int, float)) for number in value):
                for axis, number in enumerate(value):
                    writer.add_scalar(f"{tag}/axis{axis}", number, 0)
    try:
        writer.add_text("Comparison/scope", json.dumps(summary["limitations"], ensure_ascii=False), 0)
        for label, candidate in summary["candidates"].items():
            for case, scores in candidate["all_replicas"].items():
                scalars(f"Comparison/{label}/{case}/all_replicas", scores)
            for case, measured in candidate["representatives"].items():
                for phase in PHASES:
                    scalars(f"Comparison/{label}/{case}/representative/{phase}", measured["phases"][phase])
            for case, data in curves[label].items():
                values, commands, ticks, valid, masks, derivative, pose_ticks, roll, pitch, pose_valid, signals = data
                for index in np.flatnonzero(valid):
                    numbers = {"vx_m_s": values[index, 0], "height_m": values[index, 6],
                               "yaw_rad_s": values[index, 5], "command_vx_m_s": commands[index, 0],
                               "command_yaw_rad_s": commands[index, 1],
                               "command_height_m": commands[index, 2],
                               "command_vx_slope_m_s2": derivative[index, 0],
                               "command_yaw_slope_rad_s2": derivative[index, 1]}
                    numbers.update({f"omega_rad_s/axis{axis}": values[index, 3 + axis] for axis in range(3)})
                    numbers.update({f"motor_torque_nm/axis{axis}": values[index, 10 + axis] for axis in range(6)})
                    for signal, values_by_axis in signals.items():
                        numbers.update({f"{signal}/axis{axis}": value
                                        for axis, value in enumerate(values_by_axis[index])})
                    for tag, value in numbers.items():
                        writer.add_scalar(f"Comparison/{label}/{case}/trace/{tag}", float(value), int(ticks[index]))
                for index in np.flatnonzero(pose_valid):
                    for tag, value in (("roll_deg", roll[index]), ("pitch_deg", pitch[index])):
                        writer.add_scalar(f"Comparison/{label}/{case}/trace/{tag}", float(value),
                                          int(ticks[pose_ticks[index]]))
        writer.flush()
    finally:
        writer.close()


def markdown(summary):
    lines = ["# 同条件 flat / speed 对照", "", "原评分直接引用冻结评测器；代表轨迹补充不改变通过、资格或模型选择。", "",
             f"原运行状态：`{summary['capture_validation']['original_evaluation_status']}`；后处理身份：`{summary['capture_validation']['status']}`。原始报告和错误保持不变，仅用于诊断。", "",
             "| 用例 | 模型 | vx MAE cm/s | yaw MAE rad/s | 高度 MAE mm | 最大倾角 ° | 停车速度 cm/s | 原通过 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    def number(value, scale=1.):
        return "—" if value is None else f"{value * scale:.3f}"
    for case in summary["case_order"]:
        for label in LABELS:
            score = summary["candidates"][label]["all_replicas"][case]
            stop_value = score.get('settled_stop_speed') if score.get('settled_stop_metric_valid', True) else None
            lines.append(f"| {case} | {label} | {number(score.get('vx_mae_m_s'), 100)} | {number(score.get('yaw_mae_rad_s'))} | {number(score.get('height_mae_m'), 1000)} | {number(score.get('tilt_max_deg'))} | {number(stop_value, 100)} | {score.get('passed')} |")
    lines += ["", "## 代表轨迹加减速与停车", "", "每组仅首副本；姿态12.5Hz，速度／角速度／力矩50Hz。零命令停车窗重新锚定位置，未停稳样本保留。", "",
              "| 用例 | 模型 | 相位 | 时长 s | pitch绝对均值 ° | roll绝对均值 ° | 平面均速 cm/s |", "|---|---|---|---:|---:|---:|---:|"]
    for case in summary["case_order"]:
        if case not in CURVE_CASES:
            continue
        for label in LABELS:
            for phase, measured in summary["candidates"][label]["representatives"][case]["phases"].items():
                if measured["policy_samples"]:
                    lines.append(f"| {case} | {label} | {phase} | {number(measured['sim_seconds'])} | {number(measured['pitch_abs_mean_deg'])} | {number(measured['roll_abs_mean_deg'])} | {number(measured['planar_speed_body_xy_mean_m_s'], 100)} |")
    lines += ["", "## 解释范围", ""] + ["- " + line for line in summary["limitations"]]
    return "\n".join(lines) + "\n"


def self_test():
    dt = .02
    commands = np.zeros((10, 3)); commands[:, 2] = .305
    commands[:, 0] = [0, .1, .2, .2, .1, 0, 0, 0, 0, 0]
    ticks = np.arange(1, 11)
    valid = np.ones(10, dtype=bool); valid[9] = False
    masks, quiet, stop_id, _ = classify(commands, valid, ticks, dt)
    assert np.flatnonzero(masks["acceleration"]).tolist() == [1, 2]
    assert np.flatnonzero(masks["cruise"]).tolist() == [3]
    assert np.flatnonzero(masks["brake"]).tolist() == [4]
    assert np.flatnonzero(masks["stop"]).tolist() == [5, 6, 7, 8]
    assert stop_id[5] == 5 and stop_id[8] == 5 and stop_id[9] == -1
    q = np.array([[0, np.sin(.1 / 2), 0, np.cos(.1 / 2)]])
    roll, pitch = attitude_xyzw(q)
    assert np.allclose(roll, 0) and np.allclose(pitch, np.degrees(.1))
    values = np.zeros((10, 1, 16)); values[:, 0, 0] = .2
    values[:, 0, 7] = 100 + np.arange(10) * .004
    poses = np.zeros((3, 1, 1, 7)); poses[..., 6] = 1
    arrays = dict(values=values, commands=commands[:, None], episode_ticks=ticks[:, None],
                  active=np.ones((10, 1), bool), done=(~valid)[:, None],
                  terminated=np.zeros((10, 1), bool), body_poses=poses,
                  actions=np.tile(np.arange(10)[:, None, None] * .1, (1, 1, 6)))
    result, _ = representative(arrays, 0, 0, dt, 0.)
    stop = result["phases"]["stop"]
    assert stop["policy_samples"] == 4 and np.isclose(stop["planar_speed_body_xy_mean_m_s"], .2)
    assert np.isclose(stop["windows"][0]["max_displacement_from_stop_entry_m"], .012)
    assert stop["windows"][0]["start_time_s"] == .12
    assert np.allclose(stop["action_step_change_rms_50hz_proxy"], .1)
    assert np.allclose(stop["action_second_difference_rms_50hz_proxy"], 0)
    print("Self-test passed: phases, terminal exclusion, quaternion, failed stop, stop-relative displacement, action differences")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--no-events", action="store_true", help="Write JSON/Markdown only")
    parser.add_argument("--all-case-curves", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--salvage-final-stdout-broken-pipe", action="store_true",
                        help="Authenticate the pinned complete capture after its final stdout print failed; diagnostic only")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.directory is None:
        parser.error("An existing evaluation directory is required")
    directory = args.directory.resolve()
    report_path = directory / "evaluation.json"
    report = json.loads(report_path.read_text())
    contract = json.loads((directory / "contract.json").read_text())
    validation = validate_capture(directory, report, contract, salvage=args.salvage_final_stdout_broken_pipe)
    if report.get("quaternion_order") != "xyzw" or "base_link" not in report["body_names"]:
        raise ValueError("Unknown body/quaternion trace layout")
    dt = float(contract["policy_dt"])
    if not np.isclose(dt, .02):
        raise ValueError("This comparison declares 50Hz policy traces")
    cases = [case["name"] for case in report["settings"]["cases"]]
    summary = {
        "schema": "native-flat-speed-readonly-comparison-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_evaluation_sha256": sha(report_path), "source_contract_sha256": sha(directory / "contract.json"),
        "postprocessor_sha256": sha(Path(__file__)),
        "capture_validation": validation,
        "episodes_per_case": report.get("episodes_per_case"), "evaluation_device": report.get("device"),
        "optimizer_steps": 0, "qualification_awarded": False,
        "policy_rate_hz": 1 / dt, "representative_attitude_rate_hz": 1 / (4 * dt),
        "case_order": cases, "candidates": {},
        "limitations": [
            "All-replica metrics and passed flags are copied from the frozen evaluator, without regrading.",
            "Temporal and attitude statistics use only the first representative of each case, not all four replicas.",
            "Representative traces exclude terminal/reset frames; posture samples are post-step xyzw base_link at tick%4==0 (12.5Hz).",
            "Commands, velocity, omega and torque are policy-rate (50Hz); step-change RMS is a low-frequency proxy, not high-frequency jitter identification.",
            "Optional native actions and motor_velocity/motor_position are used as saved, with action first/second differences at 50Hz only.",
            "Stationary/stop phases use actual applied commands. Stop-relative displacement starts at each zero-command window, not episode birth.",
            "Stop windows include failed or unstable stops; quiet residuals additionally omit the first two seconds, without success filtering.",
            "Body-XY speed is a representative trace proxy and differs from the evaluator world-horizontal planar-speed metric.",
            "Nominal cases provide the strongest paired evidence: communication perturbation generators may not all reseed between candidates for perturbed cases; do not attribute their difference solely to policy.",
            "No new hardware readiness, qualification, optimizer update or automatic model selection is produced.",
            "This is the frozen selected 87-case diagnostic subset, not the full 322-case qualification suite.",
            "The native report remains failed when only its final stdout print broke; authenticated complete capture is diagnostic-only, never relabelled completed/evaluated.",
            "All-boundary-truncated start/stop cases have no valid settled-stop metric; their legacy numeric zero is not a successful stop.",
        ],
    }
    curves = {}
    body_index = report["body_names"].index("base_link")
    for index, label in enumerate(LABELS):
        result = report["candidates"][index]
        if result["checkpoint_sha256"] != CHECKPOINT_SHA256[index]:
            raise ValueError(f"Candidate order/identity differs from frozen {label} comparison")
        path = directory / f"candidate_{index:02d}_traces.npz"
        all_replicas = {case: {key: result["cases"][case].get(key) for key in (*METRICS, "rms_motor_torque_nm", "peak_motor_torque_nm", "passed", "checks")}
                        for case in cases}
        representative_stats, curves[label] = {}, {}
        with np.load(path, allow_pickle=False) as archive:
            # Decode each needed compressed array once, instead of reloading
            # the complete population once per representative case.
            arrays = {key: archive[key] for key in archive}
        validate_trace_arrays(arrays, result["policy_ticks"], len(cases), len(report["body_names"]),
                              salvaged=report["status"] == "failed")
        for case_index, case in enumerate(cases):
            warmup = float(result["cases"][case].get("warmup_seconds", report["settings"]["warmup_seconds"]))
            measured, curve = representative(arrays, case_index, body_index, dt, warmup)
            representative_stats[case] = measured
            all_replicas[case]["settled_stop_metric_valid"] = not (
                case.startswith(("start_stop", "velocity_curve"))
                and result["cases"][case].get("boundary_truncations", 0) >= report["episodes_per_case"])
            if args.all_case_curves or case in CURVE_CASES:
                curves[label][case] = curve
        summary["candidates"][label] = {
            "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"],
            "trace_sha256": sha(path), "original_passed": result["passed"],
            "original_passed_case_count": sum(bool(result["cases"][case]["passed"]) for case in cases),
            "all_replicas": all_replicas, "representatives": representative_stats,
        }
    if not args.no_events:
        event_dir = directory / "comparison_events"
        if event_dir.exists():
            raise FileExistsError("Do not append duplicate comparison events; preserve the existing run")
        write_events(event_dir, summary, curves)
        summary["tensorboard_event_directory"] = str(event_dir)
    (directory / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (directory / "comparison_summary.md").write_text(markdown(summary))
    print(json.dumps({"summary": str(directory / "comparison_summary.json"),
                      "markdown": str(directory / "comparison_summary.md"),
                      "events": summary.get("tensorboard_event_directory")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
