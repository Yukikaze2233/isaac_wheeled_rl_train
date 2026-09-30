# V6 35D 动态稠密奖励研究冻结 v1

日期：2026-09-29。性质：可执行研究配置及 CPU 反例检查，不是实机最优权重或新资产物理轨迹验收。正式第一段 40 次更新仍须核验实测行为及各奖励项；失败不自动加预算或降低功能门槛。保持 35D actor、81D critic、6 维动作及现有 MLP。

2026-09-30运行核对：618消费封存运行确实启用下述dense基形；旧`precision_tracking.stationary_width_m_s`和`reference_reward.stationary_width_m_s`不控制最终静止奖励。本文后文“其余…stationary anchor保留”需要更正：代码仍支持该组件，但此运行物化合同没有`stationary_tracking`，实际未实例化anchor，不能称本轮已启用。静站漂移及启停补救、学习率对照和课程排序提案见[补救计划](V6_S1_REMEDIATION_AND_ORDER_20260930.md)，尚未部署。

## 来源与适配边界

本机 SCUTRobotLab 源码 commit `b8ff79f3df855faf9dc92f4a282bd80c42649466`：

- `wheelbipe_V14/env_cfg.py:869–914`：高度、平面速度、yaw 的指数核及平方项，并保留姿态、能耗、动作平滑成本。
- `wheelbipe25_v3/env.py:3505–3833`：按命令和阶段计算跟踪目标；`3952–3977` 的关节限位项针对配置的前后关节相对关系及闭链关节约束，并非把任意髋轴绝对角统一当作机械止挡。
- 原生过程参考继续由本项目 `CommandReference` 产生，不依赖 raycast 自动请求。原生跳跃 PUSH 截止、COM 首离地、真实腾空及小跳落地折扣保持。

因此，限位接触样本不应一概弃用；其反力/释放应作为接触工况解释，而非用于拟合自由运动摩擦。无物理止挡的大腿轴不得因一个任意绝对角超过软件区间而获得“机械撞限位”判定。本奖励更新不修改或放宽实际膝内角、气簧行程及闭链失败边界；资产限位修订由独立资产回执记录。

P0 A/B 的信息仍有效：宽核改善静止平移，但高度响应变差且前进过速，不能把“放宽核”作为充分修复。新版本同时检查静止漂移、偏高、半速、过速和高度/速度联合错误。

## 配置与计算

运行字段为 `dense_tracking`，配置规范是 `rewards.py::DENSE_TRACKING_V1`，版本 `v6-dense-visible35-v1`。

```text
rho(e/s) = sqrt(1 + (e/s)^2) - 1
cost_density = -broad_weight * rho(e/broad_scale)
               + fine_weight * expm1(-(e/fine_scale)^2)
r_step = policy_dt * sum(density) + success_event - failure_event
```

实现采用 `x²/(sqrt(1+x²)+1)` 避免近零相减丢失精度。广域误差不 clip；精核默认启用，不能传零权重。大误差仍保持区分，局部精核饱和不关闭广域成本；二者在零误差处导数均为零，不声称保证 PPO 梯度或收敛。

|项|误差与单位|广域尺度|广域权重 /s|精核尺度|精核权重 /s|
|---|---|---:|---:|---:|---:|
|高度|base_link 到双轮支撑面高度减可见高度命令，m|0.040|3.0|0.015|1.50|
|前向速度|原生水平投影前向速度减可见 vx，m/s|0.50|2.0|0.10|0.60|
|侧向|普通运动的侧滑速度，m/s|0.15|0.5|0.08|0.25|
|yaw|角速度减可见 yaw 命令，rad/s|1.0|1.0|0.20|0.35|
|静止/近零速|前向命令误差与侧滑的平面模长，m/s|0.10|1.5|0.04|0.75|

静止精度与普通运动成本按当前可见命令连续混合：`quiet=clamp(1-max(abs(vx_cmd)/0.10,abs(yaw_cmd)/0.20),0,1)`。近零非零命令仍以命令误差为目标，不把小速度命令拉回零。跳跃不使用静止混合。高度成本仅在支撑且非跳跃时启用；跳跃由原生 COM 相位项承担。

一次性替换原来的宽核、平方、Huber 和精度同义项，避免重复相加。SCUT 原有三个跟踪项各自的峰值 `1 reward/s` 保留一次；在正常直立、理想跟踪且没有物理成本时，总密度仍为 `5/s`（三个跟踪峰值及 pitch/roll 各一）。负成本不乘姿态 gate，不能靠倾斜隐藏误差。自旋平移使用原生 `reference_filtered-reference_target` 的模长，与已有虚拟云台参考和跨自旋周期滤波一致，替换旧额外 Gaussian 项而非叠加。其余能耗、动作平滑、姿态、机械约束及有界 stationary anchor 保留。

普通高度往返轨迹在同一高度可对应完全相同的 35D 输入，却有相反 `h_dot`；普通加速也没有独立 `ax` 槽。因此本版本不使用隐藏 `h_dot/ax` 目标，不向当前零槽偷塞新含义。普通统一零 `vz` 惩罚同时归零，避免压制所请求的升降；可见高度误差驱动响应，斜率仍作为诊断量。跳跃不同 apex 已通过原有槽 33 的 `jump_height*5` 与槽 34 的命令时钟区分；3/6/10/15/20/25/30cm 在 35D 中可见，无需扩维。台阶可见 mode/clock 的原生姿态参考保持。

`policy_dt=0.02s` 只由环境最终求和后乘一次，物理步 1ms 不重乘密度。成功事件仍 `+5`，失败事件本版本冻结为 **200**，是一次性事件成本，不是 `200*policy_dt`。

## 反例定标结果与限制

运行入口：

```bash
PYTHONPATH=src /home/yukikaze/isaacsim60-venv/bin/python scripts/calibrate_v6_rewards.py --output /tmp/v6-reward-freeze-evidence
PYTHONPATH=src /home/yukikaze/isaacsim60-venv/bin/python -m pytest -q tests/test_new_asset_rewards.py tests/test_reference_remediation.py
```

[冻结证据目录](../reports/v6_reward_freeze_20260929/)保存本次 TensorBoard event 与 `summary.json`；不建立另一套训练曲线系统。tag 前缀 `RewardCalibration/`，轨迹横轴为 50Hz 策略 tick，值为每秒密度；JSON 保存分项折扣 return。训练期间现有组件 logger 自动记录 `dense_*_broad`、`dense_*_precision`，不得用这些上升替代固定功能验收。

本次是同一受控合成状态轨迹的完整原生平地支持期成本重算，10s、gamma=0.99；正确运动额外承担四腿各 5Nm 等示例运动成本，反例不强行承担相同成本。几何/触地/姿态项在这些切片上固定，不能称为已经验证可实现的动力学轨迹。未使用旧实测电机辨识轨迹冒充 RL 观测。

|偏好行为 vs 反例|正确行为折扣 return 优势|
|---|---:|
|站立正确 vs 前向 0.03m/s 漂移|+1.3203|
|站立正确 vs 侧向 0.03m/s 漂移|+1.3203|
|站立正确 vs 305mm 命令实际 350mm|+5.9697|
|高度误差 10mm 降至 5mm|+0.8930|
|0.5m/s 正确 vs 0.25m/s 半速|+1.6348|
|0.5m/s 正确 vs 0.8m/s 过速|+1.8347|
|正确 vs 同时 350mm 高度、0.8m/s 过速|+7.8259|
|速度误差 0.10 降至 0.05m/s|+0.5494|
|高度连续扫描正确 vs 固定中位高度|+6.0161|
|高度连续扫描正确 vs 追高但漂移|+2.3960|
|启停正确 vs 延迟 0.4s|+1.0366|
|启停正确 vs 停车后残余 0.04m/s 漂移|+0.0527|

终止反例同时设置 0.20m 高度误差、5m/s 速度误差和 6π rad/s yaw 误差，其完整生存 return 为 −90.8449，time-limit 加连续价值 bootstrap 为 −91.4457。事件成本 80 时主动早失败可以更高回报，因此拒绝 80；成本 200 时第一步失败为 −200.9145，并核对 1s、5s、10s 失败均差于生存及带 bootstrap 截断。正常成功事件、失败零 bootstrap 与 time-limit bootstrap 分别计算。这里是有限误差包络反例，不声称有限 terminal cost 能覆盖无界异常状态；数值/硬机械终止继续保留。

首段正式 40 更新须在新资产上核对各高度驻留/速度误差/姿态/驱动成本、记录完整轨迹并重算；权重冻结不是物理性能通过。局部曲率、解析斜率、合成 return 优势都不能代替固定评测和实际可学习性。

## S5 有限辅助窗口

`step_assist.py::JumpAssist` 是独立于原 raycast `StepHeightAssist` 的训练辅助器。合同字段 `jump_assist` 消费 `JUMP_ASSIST_V1`。前 300 次 **actor** 更新辅助抽样概率 0.8→0.2，力幅 1→0；第 300 次及以后、评测模式均硬关。critic-only 更新不推进窗口，进行中的 episode 也受全局窗口关断约束。

选中一次后，仅在命令 PUSH、真实支撑且未处于 FLIGHT 时施加 world-up 外力，不改 root 速度、不在腾空托举。工程上限：`F≤0.35mg`；累计 `J≤0.25m√(2g*apex)`；累计施力时间 `≤0.12s`。名义质量 13.886kg 对应约 47.68N，3cm 目标冲量上限约 2.66N·s；30cm 冲量上限约 8.42N·s，而时间上限进一步限制实际最大冲量约 5.72N·s。质量按环境实际质量计算。

这些是有限教学辅助的工程上限，**不是实测执行器能力，也不是无辅助可达证据**。实际注力后 `assisted` episode 标志锁存；辅助/无辅助离地与成功分别记录，辅助成绩不能进入验收。随机抽样保留至少 20% 的无辅助概率，实际采样比例由 TensorBoard 报告。评测必须硬关，即使载入的训练 checkpoint 仍处于辅助窗口。部分 reset 只清除对应环境的冲量，重复 PUSH 不能在同一 episode 返还预算。

CPU 测试覆盖窗口、评测关断、接触/腾空门、冲量/持续时间及部分 reset。仍须在 S5 前核验外力实际施加到正确 body、其余推扰相加及 assisted 日志在完整仿真中的一致性。
