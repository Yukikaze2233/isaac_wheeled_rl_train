"""Summarize authenticated native playback; keep missing windows explicit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circle(points):
    points = np.asarray(points, dtype=np.float64)
    x, y = points.T
    center_x, center_y, _ = np.linalg.lstsq(
        np.c_[2 * x, 2 * y, np.ones(len(x))], x * x + y * y, rcond=None)[0]
    distances = np.linalg.norm(points - [center_x, center_y], axis=1)
    return dict(center_xy_m=[float(center_x), float(center_y)],
        radius_m=float(distances.mean()),
        radial_rms_m=float(np.sqrt(np.mean((distances - distances.mean()) ** 2))),
        samples=len(points), descriptive_fit_only=True)


def inverse_rotate(vector, quaternion):
    xyz = -np.asarray(quaternion[:3], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    cross = np.cross(xyz, vector)
    return vector + 2 * (quaternion[3] * cross + np.cross(xyz, cross))


def main():
    report_file = HERE / "native004_four_cases/report.json"
    report = json.loads(report_file.read_text())
    process = json.loads((HERE / "native004_four_cases.process.json").read_text())
    assert report["status"] == "completed_bounded_unscored"
    assert process["completed_comparison"] and process["PPO_updates"] == 0
    keys = ("observed_sim_seconds", "planar_speed_mean_m_s", "vx_actual_mean_m_s",
            "vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "tilt_max_deg")
    expected_frames = dict(stand_2_12s=500, flat_cruise_2p34_3s=33,
        flat_zero_4p34_5s=33, flat_stationary_5_12s=350,
        yaw_cruise_3_5s=100, yaw_stop_6p58_12s=271)
    rows = []
    for candidate in report["candidates"]:
        policy = Path(candidate["policy_file"])
        assert digest(policy) == candidate["policy_sha256"]
        windows = {}
        for key, expected in expected_frames.items():
            raw = candidate["common_window_metrics"][key]
            windows[key] = dict(frames=raw["frames"], expected_frames=expected,
                complete=raw["frames"] == expected,
                **{field: raw[field] if raw["frames"] else None for field in keys})
        terminal = candidate["terminal"]
        recovery = terminal.get("recovery_left_side_90_stand")
        rows.append(dict(model=policy.parent.name, policy_file=str(policy),
            policy_sha256=candidate["policy_sha256"],
            source_contract_sha256=candidate["source_contract_sha256"],
            source_checkpoint_sha256=candidate["source_checkpoint_sha256"],
            windows=windows, qualification=None,
            yaw_terminal={key: terminal["yaw_2pi_stop"][key] for key in
                ("time_s", "terminated", "nonwheel_touching_body_names")}
                if "yaw_2pi_stop" in terminal else None,
            recovery_one_birth=dict(
                full_chain_success=bool(recovery and recovery["full_chain_success"]),
                completion_s=recovery["time_s"] if recovery else None,
                unfinished_right_censored=recovery is None)))

    trace_file = HERE / "native005_wheel_trace/candidate_01_partial.json"
    traced = json.loads(trace_file.read_text())
    trace_report = json.loads((HERE / "native005_wheel_trace/report.json").read_text())
    assert trace_report["status"] == "completed_bounded_unscored"
    # Only the locomotion cases are used for this control/geometry audit. The
    # recovery descriptor lifecycle differs after a different number of resets.
    repeated = trace_report["candidates"][-1]["common_window_metrics"]
    original = report["candidates"][-1]["common_window_metrics"]
    verified_windows = [key for key in expected_frames if not key.startswith("recovery")]
    for key in verified_windows:
        for field in ("frames", *keys):
            assert repeated[key][field] == original[key][field], (key, field)
    spin = [s for s in traced["samples"]
            if s["case"] == "yaw_2pi_stop" and 3 <= s["time_s"] < 5]
    assert len(spin) == 10
    geometry = []
    for sample in spin:
        pose = sample["body_pose_w_xyzw"]
        wheels = [inverse_rotate(np.asarray(wheel) - pose[:3], pose[3:])
                  for wheel in sample["wheel_centers_w_m"]]
        target = sample["last_physics_pd_wheel_target_rad_s"]
        dq = sample["last_physics_pd_dq_rad_s"]
        geometry.append(dict(time_s=sample["time_s"],
            foreaft_split_m=float(wheels[0][0] - wheels[1][0]),
            wheel_position_body_m=[point.tolist() for point in wheels],
            target_rim_forward_m_s=[.06 * target[0], -.06 * target[1]],
            actual_rim_forward_m_s=[.06 * dq[2], -.06 * dq[5]],
            wheel_contact_force_norm_n=sample["wheel_force_n"]))
    selected = min(spin, key=lambda s: abs(s["time_s"] - 3.42))
    point = geometry[spin.index(selected)]
    comparison = dict(schema="native-model-comparison-summary-v1",
        status="completed_bounded_unscored", raw_report_sha256=digest(report_file),
        seed=290929, policy_hz=50, physics_and_pc_feedback_pd_hz=200,
        actor_dim=35, critic_dim=81, action_dim=6,
        source_revision=report["source_revision"], models=rows,
        runtime_signal_delay_enabled=False, dynamics_randomization_enabled_fraction=0.,
        command_transport="Fixed cases have no communication profile; native object retained but profiles disabled",
        scope="One fixed birth per model/case, 12 s per independent case; mean actions; no PPO or hardware qualification",
        omitted_cases=report["omitted_cases"], qualification=None,
        invalid_fixture_excluded="native003_four_cases: overlapping floor colliders; no behavior credit",
        trace_audit=dict(raw_trace_sha256=digest(trace_file),
            exact_locomotion_window_reproduction=verified_windows,
            geometry_samples=geometry, representative=selected,
            representative_rim_velocity=point,
            body_circle=circle([s["body_pose_w_xyzw"][:2] for s in spin]),
            wheel_midpoint_circle=circle([
                np.mean(s["wheel_centers_w_m"], axis=0)[:2] for s in spin]),
            interpretation="Body planar speed contains eccentric orbiting; it is not rotation-center drift speed. Unequal mirrored wheel targets and fore-aft wheel geometry coexist; their independent causal contributions are not isolated."))
    (HERE / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")

    table = []
    for row in rows:
        w = row["windows"]
        yaw = "完成观察窗；平移 {:.1f} cm/s".format(w["yaw_cruise_3_5s"]["planar_speed_mean_m_s"] * 100) if w["yaw_cruise_3_5s"]["complete"] else "{:.2f}s 非轮触地终止".format(row["yaw_terminal"]["time_s"])
        recovery = "{:.2f}s 完成".format(row["recovery_one_birth"]["completion_s"]) if row["recovery_one_birth"]["full_chain_success"] else "12s 内未完成；右删失"
        table.append("| {} | {:.2f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {} | {} |".format(
            row["model"], w["stand_2_12s"]["planar_speed_mean_m_s"] * 100,
            w["stand_2_12s"]["yaw_mae_rad_s"],
            w["flat_cruise_2p34_3s"]["vx_mae_m_s"],
            w["flat_zero_4p34_5s"]["planar_speed_mean_m_s"] * 100,
            w["flat_stationary_5_12s"]["planar_speed_mean_m_s"] * 100, yaw, recovery))
    markdown = """# 模型回放对照与旋转平移分析

四个封存模型使用同一原生回放平台、同一出生 seed 290929；35D/81D、50 Hz 策略、200 Hz 物理及 PC 反馈 PD 160/2.5、64/32 求解器与 3 mm 闭链保护保持。每模型四个独立场景各观察 12 秒，均值动作，额外信号延迟关闭，无 PPO。固定评测的通信与动力学 profile 均为空，实际动力学随机化比例为 0。下表是本轮局部回放，不能与含探索及随机化的正式训练统计直接混合。

| 模型 | 静站平面均速 cm/s | 静站 yaw MAE rad/s | 2m/s 巡航 vx MAE m/s | 指令归零后首窗均速 cm/s | 后续停车均速 cm/s | 1圈/s旋转 | 左侧倒地脚本→RL |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
""" + "\n".join(table) + """

静站窗口 2–12s，共 500 帧；巡航 2.34–3s，只有 33 帧；指令归零后首窗 4.34–5s，共 33 帧；后续停车 5–12s，共 350 帧。1圈/s 巡航为 3–5s 共 100 帧；提前终止的模型不记缺失窗口为零误差。恢复只有一次左侧出生，12s 未完成是右删失，不能用它计算泛化成功率。

旧 flat14020 在本轮直行、制动和姿态上较好；最新 speed45500 能维持 1圈/s 旋转观察，但不是原地稳定旋转。其旋转巡航平面均速 0.6992m/s、yaw MAE 0.3131rad/s、高度 MAE 81.11mm。零指令静站也有 0.3091rad/s 旋转偏置。因此最新不能称为四模型综合最优。

## 偏心旋转与中心漂移须分开

补充读取实际 PC PD 的反馈和目标，没有改变原生力矩输出。上述六个运动窗口所有列出的数值逐项复现；恢复生命周期另有差异，不用于本次几何归因。

3.42s 纯旋转指令为 vx=0、yaw=6.2832rad/s。原始左右轮轴目标为 −8.776/−35.045rad/s；左右轴是镜像方向，按既有符号和 60mm 半径换到前进方向，目标轮缘速度为 **−0.527/+2.103m/s**，实际为 **−0.538/+2.003m/s**。两轮速度没有抵消；平面差速近似的共同分量为目标 0.788m/s、实际 0.733m/s，对应实测车体前向 0.702m/s。这个近似不是包含打滑、腿运动和质心偏置的完整动力学等式。

旋转 3–5s 稀疏样本中，两轮相对车体的前后位置差为 **46.9–58.1mm**；代表帧左右接触合力模长 **28.9/92.4N**。这支持存在不对称支撑和驱动，但接触合力不是纯竖直载荷，原始轴角也不能直接要求左右数值相等。

车体平面轨迹的描述性圆拟合半径 **113mm**，径向 RMS **31mm**；两轮中点的轨迹半径约 **118mm**。因此 0.70m/s 含绕偏心圆运动，不能当成圆心以 0.70m/s 漂走。采样仅每 200ms 一点、约两圈，分段圆心拟合只能提示中心变化，不能授予长期圆心漂移率或唯一因果结论。

## 当前惩罚与剩余归因

冻结 speed 合同已启用低前进指令下的平面绝对速度惩罚 −2.0，普通 yaw-only 仍会扣分；dense 前向宽尺度代价也未关闭。不能说完全没有平移惩罚。`no_fork` 的 50mm 前后差阈值和平方项关注轮位几何，不等于完整双腿镜像姿态精调；dense 开启时旧 precision 配置不生效。

直接现象是轮速共同分量和不对称支撑并存；尚未用干预对照分离“腿几何导致轮策略偏置”和“轮策略偏置驱动腿姿态改变”。修订应联合检查纯旋转的平移、转速、双侧几何、姿态和停车复习，按镜像几何约束而非原始轴角相等；不增加训练阻断 gate，也不凭这次回放放宽机械保护。

原始证据：native004_four_cases/report.json、native005_wheel_trace/candidate_01_partial.json；机器可读汇总 comparison_summary.json。native003 的重叠地板结果已排除。资格均保持 null，模型为研究候选。
"""
    (HERE / "README.md").write_text(markdown)
    print(json.dumps(dict(models=len(rows), status=comparison["status"],
        circle_radius_m=comparison["trace_audit"]["body_circle"]["radius_m"],
        summary=str(HERE / "comparison_summary.json"))))


if __name__ == "__main__":
    main()
