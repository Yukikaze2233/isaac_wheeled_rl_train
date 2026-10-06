# 接管等待窗口的门尺度运动成本候选

本目录是隔离候选，尚未通过候选原生验收、未部署、未进行 PPO，不能声称改善了实车或策略成功率。原生连续 1 秒稳定门、接触与平台判据、释放逻辑和最终随动成功条件均保留。

## 已确认的付款事实

审计读取已封存的 `native_payment_mean_001` 与 `native_payment_stochastic_001`，而不是重新运行仿真。两臂均为同一历史2579 checkpoint、source019/712d701、同一出生与私人采样初始流。每臂 4096 环境×600 策略步＝2,457,600 lane intervals，2400 物理步，完整一次性事件与即时 reward 对账残差为零，PPO 更新为零。证据 SHA 与所有 first-birth、whole-interval actor-valid still-holding 分区保存在 [actual_payment_audit_021.json](actual_payment_audit_021.json)。

下表只列 first-birth L0 原目标为静止、实际仍处于等待运动放行的 actor-owned interval。两臂条件分区的成员与时长会不同，不能把均值差解释为某个训练因素的充分因果。

| 实测量 | 均值策略 | 原生随机策略 |
|---|---:|---:|
| 该条件分区 intervals | 401741 | 587424 |
| 车体平面速度均值 | 0.03419 m/s | 0.10136 m/s |
| 两轮轮缘绝对最大速度均值 | 0.22731 m/s | 0.54892 m/s |
| 四腿轴绝对最大速度均值 | 1.00622 rad/s | 2.83019 rad/s |
| 三轴 gyro 范数均值 | 0.43173 rad/s | 1.07924 rad/s |
| 现有 leg_velocity 实付密度 | −0.01805 /s | −0.09941 /s |
| 现有 wheel_velocity 实付密度 | −0.000343 /s | −0.001466 /s |
| 现有 horizontal_angular_velocity 实付密度 | −0.01477 /s | −0.06352 /s |
| 现有 dense_yaw_broad＋precision | −0.08657 /s | −0.36539 /s |
| 现有 dense_stationary_broad＋precision | −0.43573 /s | −1.36313 /s |
| 最后反馈样本八门全过比例 | 57.47% | 2.97% |

现有负项已实际付款，不能写成零信号或惩罚未启用。小的有符号均值也不表示关节运动小：审计保留各轴 mean、standard deviation、RMS 与 extrema，避免正负抵消。`pitch_velocity`/`roll_velocity` 原项是姿态误差平方项；名称不表示三轴角速度税。

旧运动税是四腿 `−.005 Σdq²`、两轮 `−1e−5 Σdq²`、横向 gyro `−.05 Σωxy²`，另有实际启用的 yaw broad/fine。单根腿轴在 2 rad/s 门上仅付 −0.02/s，单轮在 0.25 m/s 轮缘门上付约 −0.000174/s。旧税有作用，但不是按等待放行的门尺度设计。这里只据此提出候选，未证明它们是失败的唯一原因。

物理周期末 `dq/omega` 与最后 200 Hz 延迟反馈门样本不是同一输入时钟。新成本继续使用原通用运动税的真实物理周期末输入；原门仍消费原有反馈，不替换或放松。末样本门通过比例不是连续 1 秒通过率，也不能当作独立概率连乘。账本只含统计矩，不能将 `cost(mean)` 当作 `mean(cost)`。

## 最小修复语义

新字段为顶层 `recovery_hold_cost`，完整 schema 由 `validate_recovery_hold_cost` 校验。字段缺席完全走原路径；完整对象 `enabled=False` 供固定评测明确声明禁用，helper 返回原字典和原 tensor 对象，不增加奖励分量。

启用范围严格为：

```python
recovery.enabled & recovery.learning_mask & recovery.policy_motion_hold
```

`learning_mask` 是原生产者的整个策略 interval ownership；`policy_motion_hold` 是 begin-policy 的等待状态。actor 导致终止的 interval 保留付款；曾经历脚本、混合或 release-disabled 子步的 interval 不获得新 actor 税。真正放行的首个 20 ms interval 按其出生时 hold 处理；下个 interval 即恢复原命令跟踪奖励。

只原位替换三个既有 key：`leg_velocity`、`wheel_velocity`、`horizontal_angular_velocity`。全轴 gyro 税已含 yaw，所以同一 hold mask 内两项 `dense_yaw_broad`、`dense_yaw_precision` 归零，避免收两次同类静止 yaw 税。其他 lane 与放行后它们恢复原值。reward 字典 catalog、插入顺序、堆叠规约树均不变；不添加零 tensor 分量。

高度、平移、姿态等其他密度、正奖励基线、力矩/功率/动作平滑成本、一次性失败/成功事件均不变。原 `recovery_takeover` 全局研究功能仍不启用。

## 门尺度与提案系数

令三种无量纲误差为：

\[
e_{leg}=\frac{\max_{i\in\{0,1,3,4\}}|\dot q_i|}{2},\quad
e_{wheel}=\frac{.06\max_{i\in\{2,5\}}|\dot q_i|}{.25},\quad
e_{gyro}=\frac{\|\omega\|_2}{.75}.
\]

复用原 `dense_error_terms`，运算顺序不另造实现：

\[
r_b=-w_b\frac{(e/s_b)^2}{\sqrt{1+(e/s_b)^2}+1},\qquad
r_f=w_f\operatorname{expm1}[-(e/s_f)^2].
\]

| 轴 | broad_scale | broad_weight | fine_scale | fine_weight | 对应物理 broad/fine |
|---|---:|---:|---:|---:|---|
| leg | 1 | 0.25 | 0.25 | 0.10 | 2 / 0.5 rad/s |
| wheel | 1 | 0.25 | 0.40 | 0.10 | 0.25 / 0.10 m/s 轮缘 |
| gyro | 1 | 1.00 | 0.2/0.75 | 0.35 | 0.75 / 0.20 rad/s |

weight 单位是 reward/s，helper 不乘 dt，原环境统一乘 `.02` 一次。broad 不在大误差区截平，fine 只补局部精度；解析性质不构成 PPO 能学会的保证。

腿/轮权重复用已有研究成本的 0.25/0.10，作为保守试验起点；不是 bag 辨识值，也不是经验最优值。gyro fine 复用当前 yaw 0.35、0.2 rad/s，broad 按真实全轴门 `.75` 归一并保持 weight=1。这样纯 yaw 静止成本在 CPU 测试的 0–6 rad/s 范围不弱于现有 yaw broad/fine，而没有同时叠加两套 yaw 税。门上三种税分别约 −0.20355、−0.20336、−0.76421/s。它们仅描述提案曲线，不是从聚合均值推断的新实付账本。

## 必须绑定实际门与资产

`RecoveryFSM` 仅从原门算式提取三个既有字面量为类常量：`WHEEL_RADIUS_M=.06`、`WHEEL_RIM_MAX_M_S=.25`、`LEG_SPEED_MAX_RAD_S=2.`；算式、严格小于关系、乘法顺序及所有时钟原样。真实 gyro 门继续来自 `fsm.config['handover_gyro_max']`。

启用新税时，在真实 `RecoveryTraining` 构造完成后校验上述门值，并与 `ClosedChainHeightModel.wheel_radius_m` 两侧实际半径匹配。半径使用同 device/dtype 的 expected tensor 逐值比较，避免 Python float 与 float32 的伪差。任何门、左右半径、形状漂移均拒绝启动。不会依据一个复制的常数表默默给另一套门付费。

运行回执要求 `actual.modules.recovery_hold_cost=True`、完整 `actual.recovery_hold_cost`，以及 `actual.recovery_hold_cost_gate_binding` 的真实绑定成功信息。disabled 固定评测保留完整 False 对象与模块 False，不要求该绑定。最终 design/runtime/preflight、全部消费者 source snapshot 由协调 agent 的 v19 候选继承；此目录的 `reference/preflight_enabled_only_superseded.py` 只保留旧草稿，不是可应用 patch 的组成部分。

## 日志与资格边界

新成本使用既有 `extras['log']`，在 `_recovery_metrics` 和自动 reset 前记录 `/HoldCost/actor_hold_population` 及 leg/wheel/gyro 的 `gate_relative_error_mean`、`broad_density_mean`、`fine_density_mean`。分母是同一真实付款 mask，诊断数组先按 mask 归零；日志不增加 reward key。新增每策略 tick 的设备规约仍有开销，尚未测量，不能宣称免费或提高训练速度。

41 项 CPU 验证覆盖完整/disabled schema、修改门/左右半径拒绝启动、旧 FSM 240×64 流的所有诊断与时钟逐值一致、4096 lane 普通/excluded/postrelease payment 规约逐值兼容、实际 `RecoveryTraining.torque` 生成的 whole-interval ownership、terminal 保留、混合排除、首释放窗与下窗还原、正奖励/事件/catalog 原样、mask 前处理 excluded NaN、sign/maxaxis/三轴 norm、broad tail。

仍需唯一冻结候选的 4096×600 原生验证：启动回执同时确认 goal bridge 与新税；核查 1 秒真实门之后的原 forward/yaw followup 曝光和完整随动判据；对所有 interval 的付款 mask、actor/脚本排除、普通与事件原值做原生账本对账。CPU fixture 从人工控制的 FSM 状态验证谓词，不能替代真实 SCRIPT→BLEND→RL 动力学与原生付款。

后续 v19 奖励迁移须从最新不可变 seal 完整恢复 actor、critic、Adam、std、RNG 和既有出生/难度时钟，不回滚、不手改探索 std。按父任务决策保留原合同已有 `resume_critic_warmup_updates=10`；10 个 critic-only 更新计入消费、不推进 actor，v19 同合同恢复仅接续 remaining，不重复做。source020 同合同的工程迁移不另做 warmup。这是运行账本语义，当前 artifact 准备仍为 PPO=0。
