# source022 即时接管姿态只读复核

此次 native 已完成 600 个策略步 / 2400 个物理步，0 PPO。用的是历史 ownREC3076 的 mean 策略，不代表当前训练 actor。

候选接错了动作终点：1708/2216 是初始 SELECT→PREPARE（UPRIGHT）即交给 RL，此前没有 FOLD/ORBIT/CAPTURE。`remaining<.02` 是当前动态目标残差，不是完整自起动作结束。本次负结果保留，但不能据此判断完整动作结束后立即接管是否有效。

| lane / 分支 | 首真实 RL 时刻 | 接管前→后真实倾角 | 后帧 gyro | 最大轴力矩跳变 | 最长连续传感稳定 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 1708 / source022_baseline | 4.885s | 5.78→5.79° | 0.060rad/s | 0.095N·m | 0.180s | 12s 右删失，未放行 |
| 2216 / source022_baseline | 4.920s | 9.32→9.43° | 0.387rad/s | 2.285N·m | 0.155s | 12s 右删失，未放行 |
| 1708 / immediate_v1 | 0.265s | 36.70→38.54° | 9.005rad/s | 50.638N·m | 0.000s | 0.300s actor failure |
| 2216 / immediate_v1 | 0.265s | 36.51→38.39° | 9.149rad/s | 51.319N·m | 0.000s | 0.300s actor failure |

首接管按实际 `release_finished && pure_rl` 认定。前置静置期的 phase9 占位不算 RL 接管，物理 6 / .030s 的记录不应用作首接管时刻。

旧分支两 lane 在完整恢复末段 CAPTURE→PREPARE 后才接管，最后 12s 倾角约0.62° / 0.88°，但连续稳定仅0.180s / 0.155s，尚未达到1s和后续命令放行。候选两 lane 在 .300s 原生 actor_failure 终止，倾角约54° / 53°，稳定峰0，未放行。

力矩来自实际施加读回的六电机轴，顺序为 L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3；每轴前后值及跳变量见 JSON。传感输入与物理前后真值分别记录，不能用延迟观测替代实际姿态。

候选两选定首出生闭链峰未超3mm，但全4096 / 全出生最大仍6.052mm且有1个原生闭链终止；旧分支全局27.620mm。候选更早终止、全局干预改变后续重置，不能据此认证机械改善或严格逐lane因果效果。

后续修正须记录每 birth 的原生末段→PREPARE 转移 provenance，排除初始 SELECT→PREPARE；不能只要求 route 非 UPRIGHT，也不能把1s稳定重新放到 actor 接管前。本复核没有修改冻结候选、driver或正式源码，也未启动仿真。
