# V6 整腿反馈序号保持消融（2026-09-29）

本实验仅重放左右两包的 79/89 四个完整留出段，每段 30 s；CPU 离线、固定原六参数、无重新拟合、无训练发布。两条件均使用最终 r3 的同一组物理输入、1 ms 物理/控制步长、已记录的 50 Hz 保持参考。`training_release=false`。

**结果：单独恢复已记录的反馈序号保持，对这四段闭环角度误差的影响很小。** 右侧序号重复约 8.1%，但每次只多保持一个 tick；角 RMSE 的最大增量为 `4.2105e-6 rad`，约为对应原误差的 0.023%。这不支持把重复序号当作当前闭环模型误差的主要原因。

| 侧 / 段 | 两轴序号重复次数 / 29999 | 理想反馈两轴 RMSE（rad） | 序号保持两轴 RMSE（rad） | 保持 − 理想（rad） |
|---|---:|---:|---:|---:|
| 左 79 | 0 / 0 | 0.0197892354 / 0.0224698195 | 0.0197892354 / 0.0224698195 | 0 / 0 |
| 左 89 | 3 / 7 | 0.0265088448 / 0.0204927142 | 0.0265088327 / 0.0204927188 | −1.219e−8 / +4.665e−9 |
| 右 79 | 2422 / 2425 | 0.0169498578 / 0.0186728959 | 0.0169519541 / 0.0186771064 | +2.096e−6 / +4.210e−6 |
| 右 89 | 2464 / 2430 | 0.0210437359 / 0.0259612219 | 0.0210453374 / 0.0259638463 | +1.602e−6 / +2.624e−6 |

右侧两条件完整模拟角轨迹的 RMS 差为 `1.68e−5–1.76e−5 rad`，避免仅看误差指标相互抵消。所有结果都在 [report.json](results/report.json)，完整轨迹保留在远端同目录四个 `*-traces.npz`，本地工作拷贝不纳入Git；文件SHA见 `artifact_sha256.json`。同一指标也写入 `results/tensorboard/`。

## 反馈通道的实际语义

两包归档 `wheel-leg.cpp` 的 CAN 接收回调在 `match_then_store_status` 成功后调用主机 `steady_clock::now()`，写入 `motor_last_ns_`，并增加 `motor_receive_count_`（归档行 650–679）。状态更新将计数和时间发布为 `feedback_sequence` / `feedback_steady_ns`（行 232–250）。这些字段没有电机内部采样时间，不能据此识别 FOC 周期、采样至上报延迟或完整 USB/CAN 往返延迟。原文摘录、归档及成员 SHA 见 [telemetry_source_audit.json](telemetry_source_audit.json)。

四段中，重复序号对应的 q、dq、接收时间均严格保持不变；未发现序号回退。左 79 的 host receive age p99 约 0.72–0.73 ms，右 79/89 为 1.68–1.69 ms。age 仅作为审计值保存，未人为构造亚毫秒电机采样时刻。

## 唯一实验改动

理想条件直接调用冻结 `Replay.replay`。保持条件使用相同物理初态、积分、固定增益/阻尼/摩擦/armature/一阶 lag 和气簧模型，只改变 PD 的观测输入：某轴序号变化时缓存该轴当前**模拟** q/dq；序号未变时继续使用前次缓存。参考、PD 计算和力矩更新仍每 1 ms 执行，物理状态持续积分。每段仅起始时使用实测状态，之后没有向仿真注入记录 q/dq。

新序号仍被假设在当前控制 tick 提供当前模拟状态；因此本消融只恢复可直接验证的**更新/保持模式**，不包含未知的传感获取延迟。原拟合 lag 不变，且可能已经吸收部分采样效应。本结果不能宣称已排除所有反馈延迟，也不能把 lag 解释为已识别的电机物理常数。

本次没有增加 `torque_input` 回放；该模式直接使用已记录 host torque，本身不调用 PD 反馈。因此本消融不改变原先力矩输入回放的大误差证据，也不能替代力矩尺度、静摩擦历史和质量/质心的诊断。

## 一致性检查与来源

- 四个 fresh ideal 的角 RMSE 与现存 `left/right-final-asset-1ms/report.json` **完全相同**。
- 无重复序号的左 79 段，独立保持实现与原 Replay 的 q、dq、τ 全部逐元素完全相等。这检查了复制积分路径没有额外动力学变化。
- 每个段先验证序号/主机时间单调、重复序号的 q/dq/time 一致；报告保存实际重复次数、age 分位数及参数/输入 SHA。
- 选用源：`/home/kaiser/robot-rl-sim60/experiments/v6-capacity-check-20260929/frozen/model/纯底盘_v6_232mm/urdf`。
- 当前源 manifest 为 `9542c788…`，旧 final-asset 基线 manifest 为 `fd561ea7…`。初次严格 manifest 检查在任何回放前拒绝启动，日志保留在 `preflight-rejected.log`。随后逐项确认 `robot.xml`、`model_spec.json`、`fit_10mpa.json` 的 SHA 及控制关节/树关节/限位 schema 完全相同，再运行成对回放；两种 manifest SHA 都写入报告。没有修改资产。
- 独立脚本：[ablate_pair_feedback_sampling.py](scripts/ablate_pair_feedback_sampling.py)；使用已修复几何 getter 状态复位的 Replay 快照 `bd7e66f6…`。固定参数仍来自原 `left-fit/right-fit/parameters.json`，gain 恒为 1。
- 执行入口：[run.sh](run.sh)。远端独立目录 `/home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929/feedback-ablation-v1`。没有覆盖历史结果、原始数据或 final_r3，没有启动实机/GPU训练。
