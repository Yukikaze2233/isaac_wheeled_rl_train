# v14 零速 yaw 偏差：冻结源码与 CPU 反事实审计

当前证据不支持“零 yaw 没有奖励”或“奖励被直立门关闭”。零速 yaw 有两个实际生效的稠密成本，偏差 ±0.088 rad/s 比零偏差少拿 **0.06547 reward/s**。该值甚至大于核验过的 SCUT 两条保存配置在相同误差、门值为 1 时的 yaw 收益差。仍然偏转，说明需要核查采样优势、任务竞争与均值动作，而不是直接宣称缺少 yaw 项或立即重启。

本审计只新增本目录的工具、回执和说明；没有启动 Isaac、GPU、PPO、硬件或远端进程，没有修改源码、运行合同、native_003 原始数据或现有训练。

## 证据身份、时钟和窗口

- 训练冻结源码：`b4ce80964d47c0127e0b3d9929ad35a620bbf204`；worker 为 `contracts/v6_task_campaign_speed_worker_v14.json`。
- 原始回放：`probe/native_003/report.json` 与 21 份 200 Hz frames，seed190619、305 mm、7 环境；实际运行 2026-10-04 22:10–22:14 UTC。
- v14 PT SHA：`430c1c395e93c886899dfeffd46b7d19a958f6f743198d30eb15ba9a9fb3c3e0`；ONNX SHA：`457358b4a9727fa33d6ed1741f98e14fef9534b96f99825f4bdf44afe6ac1021`。
- seal 的 `learning_updates=24000`，metadata 学习谱系 `17113`、累计 actor 消费 `23880`、task actor `3685`。v14 起点为 task actor `3563`，所以这是 **新增 122 次 actor 更新**的检查点，不能将 `23880` 当总学习谱系或解释为仅训练两次；也不是当前 live 最新策略。
- 三个静站比较均剔除 2 s settle 后记录完整 **13 s／2600 帧**；零命令为 `[0,0,0.305]`。下表不把运动中的累计位移当停车漂移，也不把截断窗口当完整通过。

| 完整静站窗口 | 旧 flat | 旧 speed | v14 新增 122 actor |
|---|---:|---:|---:|
| signed yaw 均值 rad/s | +0.00042 | +0.00137 | **−0.08791** |
| yaw 标准差 rad/s | 0.01292 | 0.00570 | 0.00379 |
| yaw 绝对值均值 rad/s | 0.01039 | 0.00465 | **0.08791** |
| 13 s 位移 m | 0.07586 | 0.64789 | 0.42192 |
| 统一 v14 yaw 公式重算成本 reward/s | 0.00154 | 0.00032 | **0.06542** |

最后一行是同一冻结公式在物理状态上的 **CPU 重算**，不是这三条不同训练合同的实付日志。成本均值使用逐帧计算后平均，未用均值角速度代替整段轨迹。

## 真正生效的 yaw 付款路径

冻结 `env.py:1434` 组装 `reward_terms`，随后 `apply_manual_tracking`，最后在 `env.py:1513` 调用 `apply_dense_tracking`。稠密替换函数会删除旧的 `track_yaw`、`yaw_square`、`yaw_precision`，然后重建恒定峰值和两个成本：

\[
e_\omega=\omega_z-\omega_{z,\mathrm{cmd}},\qquad
r_{\omega,b}=-\left(\sqrt{1+e_\omega^2}-1\right),\qquad
r_{\omega,f}=0.35\left[\exp\!\left(-\left(\frac{e_\omega}{0.2}\right)^2\right)-1\right].
\]

`track_yaw=1 reward/s` 为恒定峰值，不奖励继续偏转。`dense_yaw_broad` 与 `dense_yaw_precision` 的 gate 是全 1；既不随支撑状态、姿态误差关闭，也不使用静止命令的 quiet blend。旧直立门约 0.9991，且其已被替换的正 yaw 项不参与最终稠密收益。`horizontal_angular_velocity` 仅包含 roll/pitch 角速度，不含 yaw；现有 posture floor 也只约束 pitch/roll，不是 yaw 角速度成本。

| 固定 yaw 误差 rad/s | 宽域成本 reward/s | 精度成本 reward/s | 合计损失 reward/s | 每个 20 ms 策略 reward 损失 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0.01 | 0.000050 | 0.000874 | 0.000924 | 0.0000185 |
| 0.04 | 0.000800 | 0.013724 | 0.014523 | 0.0002905 |
| ±0.088 | 0.003865 | 0.061604 | **0.065469** | **0.0013094** |
| 0.15 | 0.011187 | 0.150576 | 0.161763 | 0.0032353 |

密度在最终 reward 求和时只乘一次 `policy_dt=0.02`（`env.py:1525`），没有再乘 `physics_dt=0.005`。0.088 附近成本不是平台；降低误差确实改善目标。但是解析单调性、斜率或局部曲率 **不能保证 PPO 采样出有用优势、及时学会抵消偏置**。

旧 `yaw_precision` 若单独计算会给 0.088 约 0.13475/s 的成本，但稠密模式有意替换了它。不能将“旧变量被删除”直接等同于“当前 yaw 没接”，也不能不做消融就把两套精核叠加。

## 静站项、单位和观测延迟

`stand_still_lin_vel=-2(|v_x|+|v_y|)` 是平面线速度成本；dense stationary 同样仅针对水平速度。只改变真实 `omega_z`、保持其他状态不变，直接调用冻结函数验证这三项全部不变。它们不直接支付 yaw 抑制，也不是位置／航向角锚。

原始 `omega_rad_s` 来自 `root_com_ang_vel_b` 真值，reward 也使用该真值。**−0.08791 rad/s = −5.037°/s = −0.01399 rps**，不是 −0.08791 圈/s。35D 的 gyro 输入乘 0.5，所以名义观测该分量约 −0.04396；缩放没有改变物理 reward 的 SI 单位。

native_003 使用共同 flat 诊断合同：signal delay 和噪声关闭，command transport 实例存在，但 7 条实际 enabled 均为 false。因此该回放不能证明“延迟随机化导致偏置”或证明带延迟时仍能接稳。训练 worker 的 signal delay 是启用的，两者不是同一随机化域；本审计不修改它们。

共同回放合同保留与 v14 相同的 dense yaw 设置，**未包含 v14 新增 L1／posture floor reward 字段**。该共同合同用于比较物理行为，原报告未输出 reward components，不能从本次 frames 声称新字段在这次回放已实付。

另一个测量口径缺口：frames 的线速度为 `root_link_lin_vel_b`，runtime reward 使用 `root_com_lin_vel_b`。所以回执中的平移／静站成本只标作 **link 速度代理、support=1 的反事实**；yaw 部分没有这个原点差异。完整总收益、支撑掩码、腿部 effort 和历史动作收益不能由现有 frames 精确重建。

## 左右轮输出确有偏置，但不等于根因已确认

核对冻结控制顺序 `[LH,LA,LW,RH,RA,RW]`，轮项在 index 2／5，目标为动作乘 `10 rad/s`。V6 名义 URDF 前向运动学得到左右轮轴近似分别 **+Y／−Y**；所以直行对应两轮编码器速度异号，不能看到动作一正一负就判映射错误。

| 完整静站 13 s 平均量 | 旧 flat | 旧 speed | v14 |
|---|---:|---:|---:|
| 左轮目标 rad/s | −0.08973 | +0.91477 | **+1.18072** |
| 右轮目标 rad/s | +0.18903 | −0.95749 | **−0.14506** |
| 左轮实速 rad/s | −0.06463 | +0.82078 | **+0.88701** |
| 右轮实速 rad/s | +0.06771 | −0.83276 | **−0.21279** |
| 编码器两轴实速之和 rad/s | +0.00307 | −0.01199 | **+0.67422** |
| 左轮实发力矩 N·m | −0.01503 | +0.05640 | **+0.17623** |
| 右轮实发力矩 N·m | +0.07274 | −0.07486 | **+0.04064** |
| 两轮力矩饱和比例 | 0／0 | 0／0 | **0／0** |

v14 的均值策略持续给出不对称轮目标，实际轮速与负 yaw 同时持续。仅此能排除“静站轮力矩一直被限幅卡住”，不能证明是哪项 reward、摩擦不等或哪次梯度更新造成偏置。名义 FK 也不能替代接触／打滑条件下的动态无滑滚动验证。

这种偏置在完成的三段松杆停车窗仍复现：forward0.5、backward0.5、forward1 后各完整 10 s signed yaw 分别 −0.08841、−0.08940、−0.08851 rad/s。前进0.5 的完整4 s稳速为 −0.14341；后退0.5 均值 +0.04421 但标准差 **0.58609**，不能用接近零的均值掩盖摆动。backward1、forward3、backward3 的稳速分别只有3.96／2.34／0.54 s，并由 boundary 结束，没有完整停车窗，未作通过判定。

## SCUT 源码对照：仅核验过的激活配置

本机 primary 仓库 commit `b8ff79f3df855faf9dc92f4a282bd80c42649466`。V14 继承 V13，后者继承 `wheelbipe25_v3/env.py` 的 `_get_rewards`；该函数原生 yaw 为 exp 核加平方项。保存 run 的 YAML 不是模板默认，核验到：

| 保存 run | yaw exp 权重 | yaw square 权重 | stand_still | stand_still_lin_vel | deadzone |
|---|---:|---:|---:|---:|---|
| flat_and_rotation / 2026-07-29_23-20-03 | +1 | −0.1 | **−0.0** | −2 | true，0.1 |
| rough_dash / 2026-07-19_19-15-07 | +1 | −1 | **−0.0** | 顶层 rewards 无此项 | false |

两 run 的 `ang_vel_z_sigma=.25`、`ang_vel_z_square_sigma=.5`。对0.088 rad/s误差，gate=1／普通地面状态机 scale=1 的 exp+square 损失分别 **0.03069／0.03244 reward/s**，小于当前 v14 的0.06547。SCUT 源码存在含 yaw 的 `rew_stand_still`，但上述实际保存权重为零，不能称其已额外启用零速 yaw 惩罚。

他们的 yaw tracking 也会乘姿态／高度综合门；本表是归一化公式比较，不是运行 SCUT 策略。源码存在 airborne／jump／stair 的 reward 后处理与 V14 gimbal 特例，不能笼统说他们“全部 reward 无相位化”或“不会遗忘”。`flat_orientation_x/y_v` 实际是由命令速度调节系数的 **projected-gravity 平方**，不是 roll/pitch gyro 惩罚。

## 最小后续假设与未执行实验

| 问题／假设 | 当前证据等级 | 最小下一证据 |
|---|---|---|
| yaw 奖励缺失、直立门令其不付费、单位误差 | 本轮冻结调用否定 | 保留现有 CPU 回归；未来原生监测直接记录组件，确认运行实付 |
| 均值 actor 形成持续左右轮偏置 | 行为已证实，形成原因未证实 | 最近已保存 checkpoint 同命名13 s零速窗／不同reset和至少两seed，比较是否仍偏；同时记录真实policy input、COM/link速度、支撑、组件与左右轮目标 |
| 静站线速改善与 yaw 支付竞争，整体优势仍偏向某稳定吸引态 | 仅候选机制 | 在自然训练窗口按零速组分别记录 yaw／线速／姿态成本与 action；不能靠不同模型总收益或解析曲率判决 |
| 延迟、随机化、机械不对称改变策略对补偿的期望 | 仅候选机制；本回放无延迟且未饱和 | 同 checkpoint、同轨迹、同seed一次只切名义／训练延迟 profile，再比较 signed yaw；不得热改现有worker |
| 需要更窄精核或显式零命令 yaw 项 | 尚无学习消融 | 仅在上述证据仍显示稳定偏差后冻结独立小消融；一次只改 yaw 精度或零速 yaw 成本，保持线速、正反旋转、高速姿态保护；不同时翻倍权重、改采样和LR |

目前没有可证实的 yaw 付款实现 bug，**不据此重启正在运行的 v14 PPO**。此回放已证明的退化需要继续跟踪，但不能用旧122-update回放替代最新100-update窗口的趋势，也不能通过放宽判据或把均值抵消当达标。

复算：

```bash
/home/yukikaze/isaacsim60-venv/bin/python -B reports/v6_locomotion_quality_repair_20261005/yaw_audit/analyze_yaw.py
```

工具只加载冻结 AST 中的奖励函数／SCUT 指定语句，在 CPU 上执行并生成 `evidence.json`。回执包含 SHA、函数行号、21份原始frames哈希、模型时钟、完整／右截断窗口与反事实断言，不创建新的图表数据源或本机 TensorBoard。
