# PREPARE 零通过原因：原生物理 002 的 CPU 审计

两例失败发生在自起脚本的 PREPARE，尚未交给 RL。已证实一个配置语义错位：`prepare_speed_rad_s=1` 被 PREPARE 的 `push_speed_rad_s=4` 覆盖。原始轨迹确实多数时刻以 4 rad/s 追逐目标；这值得做单变量物理对照，但现有数据不能证明改成 1 就能自起成功。

## 来源与复核

- 只读取 [native_prefix_002](../native_prefix_002/report.json) 的四姿态、seed 290929、第一出生 cohort，以及配套物理帧、策略帧、事件和事件快照。物理采集始于 **2026-10-04 23:17:43 UTC**；全局最长运行 8 s，四例均已终结，没有右截断。
- 使用 **旧 flat** actor：`7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f`，checkpoint learning_updates=14020、flat task_actor=13000。它不是正在训练的 v14 actor。
- 控制与物理源码固定为 `b4ce80964d47c0127e0b3d9929ad35a620bbf204`。从 Git 取出纯 CPU 类，逐文件 SHA 与 002 回执核对；资产 profile/lookup 同样按有效合同 SHA 核对。[evidence.json](evidence.json) 保留完整输入 SHA、1/4 秒分段、目标/实测轴角/速度、原始门和 CPU 反事实。
- 实体帧里的 `q/root` 是积分后状态，`gates/targets` 是这次力矩所用的积分前结果。审计将传感器状态前移一物理 tick，完整复算 **6172 lane×physics tick 的八个门，零不一致**；**1543 policy 样本的连续六轴角度逐值一致，最大差 0 rad**。没有用同帧积分后状态解释前帧门，也没有把释放期 raw phase=RL 当作实际 RL。
- 本次没有 Isaac、GPU、SSH、硬件、PPO、部署或修改运行源码/合同，formal_updates_consumed=0。

复算入口：

```bash
/home/yukikaze/isaacsim60-venv/bin/python reports/v6_locomotion_quality_repair_20261005/recovery_audit/prepare_gate_audit/analyze.py
```

## 哪个条件没有满足

下表分母是 **实际 script/PREPARE 物理 tick**，不是出生数或策略学习区间。右侧/俯卧的第 20 个连续 ready tick 已归属 BLEND，所以 PREPARE 内记录为 19；随后原始证据证明完整 40 tick/200 ms BLEND。两例失败从未进入 BLEND。

| 姿态 | PREPARE ticks | 八门同时通过 | 最大 ready ticks | 最长连续 tilt 通过 | 最长连续 leg_speed 通过 | 结局 |
|---|---:|---:|---:|---:|---:|---|
| 左侧 90° | 427 | 0 | 0 | 35 ms | 20 ms | 8 s 脚本超时 |
| 右侧 90° | 226 | 19 | 19 | 95 ms | 505 ms | 接管、保持、后续任务成功 |
| 俯卧 | 229 | 19 | 19 | 95 ms | 300 ms | 接管、保持、后续任务成功 |
| 仰卧 | 404 | 0 | 0 | 35 ms | 15 ms | 8 s 脚本超时 |

| PREPARE 门通过 ticks | 左侧 / 427 | 仰卧 / 404 |
|---|---:|---:|
| 有效传感器 | 427 | 404 |
| 支撑估计 | 261 | 239 |
| body_clear 估计 | 51 | 47 |
| 倾角 <8° | 14 | 13 |
| gyro norm <0.75 rad/s | 85 | 90 |
| 估计高度 0.27–0.36 m | 283 | 249 |
| 轮轴速度 ×0.06 <0.25 m/s | 157 | 151 |
| 四腿驱动轴 max speed <2 rad/s | 39 | 36 |

失败两例即使临时忽略任意一门，其余七门同时通过仍为 **0 tick**。最接近的帧也同时缺 `tilt` 和 `leg_speed`；例如左侧 4.795 s 倾角 90.32°、腿最大速度 4.06 rad/s，仰卧 4.890 s 倾角 90.55°、腿最大速度 4.60 rad/s。六项通过不等于“差一点即可接管”，更不支持放宽门。

最后策略帧直接给出 `recovery_failure_code=6`、phase=PREPARE、reroute_count=1、handover_count=0。FSM 自身的 failure_code 仍是 0，数字 6 来自外层 `RecoveryTraining` 的 8 s prelude 超时归因。其他终止原因未触发。这是脚本失败，没有 RL 学习接稳的机会。

## 真实时间轨迹和机构目标

| 姿态 | 第一次 PREPARE 输入 | 第一次翻离后重路由 | 第二次 PREPARE 输入 | 终止物理姿态 |
|---|---|---|---|---|
| 左侧 | 4.150 s：pitch +19.23°、roll 约0.26° | 5.125 s：tilt 112.62° → FOLD | 6.840 s：pitch −18.28° | 8 s：pitch +35.65°、roll 约0.27° |
| 仰卧 | 4.270 s：pitch +18.94°、roll 约0.26° | 5.235 s：tilt 121.56° → FOLD | 6.945 s：pitch −19.82° | 8 s：pitch +33.00°、roll 约0.26° |
| 右侧对照 | 4.280 s：pitch +19.09°、roll 约0.26° | 未重路由 | 5.410 s 进入 BLEND | 随后成功 |

上表事件发生在 reference 更新后、力矩发送前；首个 PREPARE **实际作用的物理帧**分别是 4.155/4.275/4.285 s。两失败末态侧倾已消除，主要剩余问题是 sagittal pitch 与速度，而不是左侧 SIDE 路线仍保持 90° roll。

实际 CONTROL 顺序为 `L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3`，对应原生 18 轴索引 `[0,2,8,1,3,10]`。`LL/RR_joint1` 是独立辅助驱动轴，不能按大小腿内角理解。连续绕圈的整数圈差由统一 encoder provider 保留，CPU 复算与原始策略诊断角度全部相同；不能将 ±2π/±4π 直接判为映射错误。

最后一次 reference 的四腿目标/连续反馈如下（rad，按 LH/LA/RH/RA）；值来自末策略帧的积分前反馈，最后物理帧姿态已再推进 5 ms：

| 姿态 | 目标 | 连续反馈 | 目标−反馈 |
|---|---|---|---|
| 左侧 | −0.7807/−0.3651/+0.7807/+0.3651 | −0.7548/−0.3320/+0.7542/+0.3314 | −0.0259/−0.0330/+0.0265/+0.0337 |
| 仰卧 | −0.7856/−0.3700/+0.7856/+0.3700 | −0.8994/−0.3606/+0.8993/+0.3606 | +0.1138/−0.0094/−0.1137/+0.0094 |

在原生 PD=160/2.5 下，失败 PREPARE 的最大实际腿力矩为 36.76/32.85 N·m，未触及 40 N·m 腿上限；轮侧 ±4.5 N·m 限幅出现 107/427、96/404 tick。大腿 target 不静止：左/仰卧分别 **411/427、378/404 tick** 恰为 4 rad/s；目标超过已配置 1 rad/s 的 tick 为 421/427、389/404。小静态跟踪误差不能代替稳定姿态与静止速度。

## 支撑估计与真实接触的区别

`support` 是 FK 高度、两轮估计高度差、IMU specific acceleration 6–16 m/s²、gyro <8 rad/s 的联合 plausibility，不是真实 contact 位。`body_clear` 在 PREPARE 还要求轮探针两轮正反向响应确认，且估计高度 >0.27 m。它的名称不能读成仿真真值“底盘没有接触”。

- 左侧/仰卧的 support false 为 166/165 tick。其中仅 specific acceleration 条件拒绝、其他支撑条件全部满足的 tick 为 **134/123**，占各自 support false 的 80.7%/74.5%。加速度是实测物理运动的前差分，不是人为注入延迟；它可能受冲击/加速影响，但这里没有证明是假阴性。
- 在 support=true 且高度 >0.27 m 的 PREPARE tick 中，轮探针仍未确认 body_clear 的为 **135/117**。它确实压缩了有效稳定窗，不过同时去掉 body_clear 也不能让其余七门过关。
- 末态物理快照两轮中心 z≈0.060 m，cached contact force 有两个非零 patch，closure gap 很小；原始快照没有绑定 native body names，不将具体 contact index 冒充已核实的 wheel/body 身份。末态轮在地面附近不证明整个 8 s 支撑估计都应为 true。
- 8 s 超时包含 1 s 被动零主动力矩释放，这是源码明确约定，不是漏计。一次重路由后只剩约 1.16/1.055 s 的第二 PREPARE。它是时间约束，不是已证实的无限脚本或 BLEND 时钟故障；单纯延长超时能否收敛未知。

## 可证实错位与下一项物理对照

冻结 `recovery.py` 中默认速度为 `prepare_speed_rad_s`，随后一行把 `preparing | thrusting` 同时覆盖为 `push_speed_rad_s`。最小候选只删除这个覆盖的 PREPARE 部分，保留 THRUST 的 4 rad/s；[candidate_prepare_slew.diff](candidate_prepare_slew.diff) 只是一份报告附件，未应用到 core：

```diff
- speed[preparing | thrusting] = self.config["push_speed_rad_s"]
+ speed[thrusting] = self.config["push_speed_rad_s"]
```

**已完成 CPU 反事实：**从 36 个真实 controller 事件快照克隆 FSM，给基线和候选完全相同的 q/dq/gravity/gyro/height/support/body_clear，推进一次 reference；不写物理状态、不改变输入。8 个 PREPARE 快照步幅约 0.020→0.005 rad，固定同一反馈时下一 PD 请求力矩差约 2.4 N·m；八门掩码相同。28 个其他阶段快照的所有 FSM tensor 状态逐值相同，包含 THRUST、CAPTURE、SIDE、ORBIT、FOLD、PLANT、BLEND。它证明了修改范围，**不证明实际物理响应或成功率改善**。

下一实验建议只做一个命名的工程物理 A/B：

1. 保留 002 的 exact 四姿态、seed、旧 flat PT/资产/几何/PD/轮平衡/释放/8 s 超时/八门/200 ms BLEND；复制冻结工程源码，认证唯一差异为上述一行，运行合同保持逐字段相同。
2. 基线和候选均从完整出生重放，不声称事件 PT 快照能恢复 PhysX 内部接触与求解器状态。保留原始 002 不覆盖；基线重放可用于排查现场差异。
3. 记录已存在的 first-cohort 分区、PREPARE pitch/gyro/真实 dq/目标slew/轮限幅、支撑五条件与探针确认、完整 20 ready tick/40 BLEND tick/1 s RL保持、超时阶段和 code。四姿态均保留，以检查原成功样本是否退化；不新设成功阈值，不把慢下来当作成功。
4. 判别问题：降低 PREPARE 目标速度后，俯仰翻离/腿速度/加速度支撑拒绝是否同时降低并产生真正连续 ready 窗？若无，则“PREPARE slew 是主要原因”不足，下一轮再独立检查轮平衡与 observer 的动态适用性；不要一次改 slew、轮增益和门限。

该 A/B 尚未执行，正在跑的 v14 PPO 没有受此审计改动。当前最严谨的结论是：**脚本在 PREPARE 持续运动，没有构造出合法接管窗；存在可修正的速度字段错位，因果充分性等待独立物理验证。**
