# L0 接稳后的原始运动目标桥接候选 021

这份候选只修复教学目标的传递：简单姿态出生的 L0 环境通过原生连续接稳门之后，按它所属行为组的原始前进／正反旋转目标继续运动。当前正式 source020 与 v18 合同未修改；没有部署、原生物理试验、PPO 更新或新增行为资格。

基线为冻结 source020 `7f315f30741be01d70046a11848443038c14b240`，归档 SHA-256 `ff0c57ca2255fb0e336b28be57e0a66e7923597dd5398d0bb18c4e34f91c1b87`。精确源文件与补丁身份在 `source_bindings.json`。

## 已确认的传递缺口

`recovery_prefix_curriculum.py:124` 保存每组原始 `recovery_followup_command`，但 `:148` 在每次 L0 reset 时把其前进／旋转目标清零。`apply_lesson:155` 将清零结果写入恢复模块。因此 L0 即便真正接稳并放行，仍只得到零速任务。

下游流程已经完整，不需要新增状态机：`recovery_training.py:203` 在恢复期间写入跟随目标，并在原生门放行前强制可见速度命令归零；`recovery.py:325` 在 200 Hz 下累计连续 200 个合格反馈 tick，门被打断就重新计时。`recovery_training.py:343` 另外用真实双轮接触、平台证据、姿态、角速度及高度检查接稳，传感器门提前放行会产生 code7 失败。释放后的第一段完整 50 Hz 区间才看到非零目标。

## 最小改动与显式启用

`goal_bridge_021.patch` 只修改一个候选模块。未来新冻结合同须显式加入：

```json
{"recovery_training.prefix_curriculum.goal_bridge": "native-original-followup-after-stable-gate-v1"}
```

作者函数 `prefix_settings(origin_actor_update, bridge_original_goals=True)` 可以生成该设置。当前 revision 生成器依旧使用默认 `False`；源码加入字段支持本身不会启用功能。新字段缺省时，设置内容、采样、RNG 状态及所有 L0 行为与旧 v18 一致；出现不支持的字段值则拒绝启动。

启用后，仅不再将 L0 的原始目标清零。easy 出生姿态、零力矩释放时间、抽样概率、actor origin=2294、难度时钟、原生脚本、200 ms blend、连续 1 s 门、控制器、模型、奖励及 PPO 参数均不变。L1、L2 及 1449 普通环境没有行为修改，不启用旧 `recovery_takeover` 奖励。

## 时长和成功口径

不需要恢复另一个被缩短的跟随时长：原有 `RecoveryTraining.outcome:366–372` 对所有 lesson 本来就要求释放后的连续 **1 s** 跟踪，且必须先有完整接稳证据。候选保留原条件：前向误差 ≤0.12 m/s、yaw 角速度误差 ≤0.2 rad/s；原零速组另外要求平面速度 <0.10 m/s。真实接触／平台／姿态条件失守会把连续跟随时钟清零。

因此原 forward=0.3 m/s 或 yaw=±0.6 rad/s 的 L0 环境，释放后只静站不能得到该运动组成功。L0 成功仍只记为 `prefix_stand` lesson 成功，不是倒地整链成功；只有 L2 `original_chain` 保留原始坏姿态出生，才进入整链分母。旧 L0 成功计数不能与新语义直接拼接比较，未来迁移与报告必须标记目标桥接版本，并按 lesson、原目标和释放后运动分别查看。

## CPU 验证范围

21 项 CPU fixture 使用冻结020的真实采样器、`RecoveryFSM` 和 `RecoveryTraining`，验证默认兼容、四个难度时钟的相同采样和 RNG、partial reset、1449 普通环境、真实目标经 `apply_lesson` 传递、196/200 tick 连续门边界、打断后重计、真实接触／平台／undesired 检查、完整跟随 1 s 和整链记账隔离。它们把传感器与状态作为有限输入，**没有仿真出物理接稳、证明学习有效或测试驱动性能**。

## 原生验证与部署仍缺的证据

1. 独立合同设计表／runtime 预检与源快照绑定新字段，默认 v18 及启用候选分别通过；明确新 revision，只迁移账本及完整学习状态，不改变 origin、RNG 恢复或预算。
2. 使用同 sealed actor、同源条件、4096 人口做旧／新原生配对。每次 reset 保存原始组目标和 lesson；逐 5 ms 验证门、每 20 ms 验证可见命令；只有首次真放行之后目标才改变。ordinary1449／L1／L2 保留原过程。
3. 记录真实双轮接触／平台／连续持有秒数、放行失败码、原目标、释放后连续跟随秒数、lesson 终止与整链分母。成功侧证必须来自同一旧 episode 的 producer，不能使用 auto-reset 后新 lesson。
4. 对 forward／yaw± 单独查看放行后的姿态、跟随、失稳／闭链护栏与实际付款。原生机制通过不等于行为 qualified；PPO 效果需后续完整窗口确认。

本候选沿用 L1/L2 已有的直接目标释放路径。0.3 m/s 与 ±0.6 rad/s 的一步变化是否过猛仍需物理配对证据；若确有放行冲击，再单独设计并冻结已有 `command_slew` 的有界命令过渡。当前 `override_commands` 位于普通 slew 之后，直接开普通 slew 不会自动作用于恢复目标。不得因将来渐变到位前静站较稳，就缩短跟随 1 s 或把渐变阶段计为原目标跟踪成功。
