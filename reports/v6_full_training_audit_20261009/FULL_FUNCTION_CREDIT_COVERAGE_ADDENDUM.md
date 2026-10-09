# 完整功能学习信用增量审查（81ad / SCUT b8）

本次仅补审已经冻结的 v27 源码和正式静态 recipe，不重复先前 V26 对照。基准：我方 `81ad124554c4ab45cc274e2d482d46c76ba125b0`；华南虎官方 `b8ff79f3df855faf9dc92f4a282bd80c42649466`。配套逐文件 SHA 与静态环境配额见 `full_function_credit_coverage_addendum001.json`。未运行 SSH、物理回放或 PPO；当前 source029 高速学习与 source030 队列均未修改。v27 的课程存在，不等于这些未来域已经学会。

## 新结论

**找到一项实际奖励目标冲突：主动跳阶落台要求停车，而落地奖励仍追踪起跳时锁定的非零前进速度。** 它有完整源码数据流证据，需要在独立新版本修复。其余主要是目标覆盖与反馈范围差异，不能直接认定是当前策略表现不佳的已证实原因。

普通大阶并非只写了“通过”指标：实际接入 progress、抬轮、有限车体支撑、姿态、横向位置和平台事件奖励；主动跳及跳阶也确有收腿奖励。不能继续用“没有支付”概括。行进跳虽然只命名 03/06/10 三组，在真实启用的 bin 课程中仍会采到 3–30 cm；但行进速度仅为正向 0.4 m/s，不能由此宣称训练过高速行进跳。

## 1. 普通爬阶与主动跳阶是两套真实任务

| 项目 | 普通大阶 | 主动跳阶 |
|---|---|---|
| 请求/控制 | `body_climb`、mode 0，仅前向速度；无 JUMP 请求，`StepClimbTask` 不产生命令 | mode 4；`StepJumpTask.control` 用轮编码器、IMU、单束测距/预设操作者事件产生请求、飞行前进与落台停车命令 |
| 接触/完成 | 允许真实底盘接触平台顶面；每一级必须实际跨沿、双轮支撑、稳定 .35 s；连续失去轮/合法车体支撑 ≥ .08 s 算跳跃，不能拿来当普通爬阶 | 真实无支撑、跨沿轮底高度、实际平台双轮接触、姿态及平面速度 ≤ .25 m/s；与命令 FSM 的完成分开 |
| 大阶几何 | 50/100/150/200 mm 单级，再四级双阶难度，终点 200+250 mm；首踏面 .8 m、第二 .6 m，最终平台 .45 m | 单级 30/60/100/150/200/250 mm 的 manual/laser，200+250 mm 双级 manual/laser |
| 学习信用 | 真正加到 `env.step` 的总 reward，预教期间不付专项爬阶信用；普通 PPO 样本保留 | 基础参考跳跃项、contact-triggered 落地窗、前沿净空/着地区/横向及平台事件项都进总 reward |

出处：`env.py:1430,1597,1750,1816,1823`；`step_climb.py:130–224`；`step_jump.py:110–166,174–258`；`recovery_rollout.py:72–157`。PPO 排除的是脚本/混合恢复、非标准出生短暂零力矩的非 actor 阶段；普通大阶预教不是“无有效样本的空站”。

普通阶底盘支撑信用为 `.3 × 新达到的支撑时间`，每平台封顶 .1 s，因此最多 **.03 reward/平台**；真正跨阶支撑事件另为 **+2/平台**。抬轮有不饱和的大误差 pseudo-Huber 及局部细核；并非只靠指数精核。

不过，车体先搭台再收腿只是**被允许且有有限支撑信用**，不是被定义为必选顺序；`StepClimbTask` 没有“合法车体已承重时，双腿分别收短”的独立目标。这个选择避免强迫所有成功动作都必须贴底盘，但若要专门学出这一机械动作，现有 credit 仍偏间接。不能仅凭 .03 较小就武断翻倍；应先测支撑发生率、轮底 shortfall 和每轴支付，确认能否实际建立承重支点。

车体支撑只能是实际平台顶面、朝上法向和真实 contact patch，竖直立面及其他非合法碰撞仍不能豁免。该普通任务不允许用长时间腾空代替爬阶。全局 `PhaseTracker` 只看轮接触，因此“底盘承重、双轮离地”可标成 FLIGHT；普通任务自己的 support/flight violation 则计入合法车体接触。诊断与后续信用核对必须区分这两个概念，不能把 Phase.FLIGHT 数量直接当普通爬阶作弊次数。

## 2. 已证实问题：跳阶停车与锁定的落地速度相冲突

数据流如下：

1. `step_jump.py:143–159` 起跳前按着地区距离/剩余飞行时间计算 `command_vx∈[0,1.5]`；观测到自由落体后再次获得 IMU 支撑，将该目标置 0。
2. `env.py:2119–2146` 把这一目标写入 `command_target`，经已有 slew 变成 actor 可见的真实前进指令。
3. `jump_landing.py:242–256` 在新请求锁定 `goal_commands=commands`，后续不更新；`reward_terms:339–375` 同时用它计算落地 vx/yaw 误差和接触前轮滚速目标。
4. v27 integrated **stepjump patch 已启用该落地窗（3 s）**；环境调用只传 `landing_delta`，没有传明确 `StepJumpTask.enabled` 上下文。因此平台任务停稳时，仍会被要求跟踪接受请求时的前进速度。

以锁定 vx=.4 m/s、平台停车实际 vx=0、落地窗权重为 1 为例，这一速度轴额外支付约 **−1.16125 reward/s**；vx=.65 时约 −1.88024/s。保持锁定非零 vx 会减少此代价，却与实际平台速度 ≤ .25 和停车命令矛盾。这些数字是原核公式的解析值，不是新物理评测成绩，也不证明此前 GUI 坏行为全部由它引起。

**最小修复**：明确传入固定的跳阶 row mask，独立锁定“落台 vx/yaw=0”的任务目标与原始空中/接触前滚速目标；普通平地 jump/running jump 的命令快照保持原语义。不要从实时零指令覆盖所有锁定目标，也不要靠取消请求重启落地窗；窗口、真实接触门、原完成阈值不变。不能只凭 mode 4 区分二者；两种任务都是 mode 4。也不宜只以 `landing_delta != 0` 猜类型，其值随请求状态变化。该问题已经交给独立修复 agent，**81ad 与已排队产物未改**。

## 3. 收腿奖励是真的存在，但不等于完整腿长塑形

`rewards.py:644–704` 的实际参考路径支付：

`jump_tuck = 2 × airborne × exp(-((extension-.16)/.04)^2)`。

这里 `extension` 来自 `env.py:1630` 的**世界竖直方向车体到两轮平均高度差**，而非每腿的欧氏几何长度。两腿一长一短可以有相同平均值；倾斜和前后摆腿也可改变竖直投影而不等量收短机构。其他姿态/净空/无叉腿项会约束这些行为，但不能把这个项独立解释为“每条腿均正确收短”。这是可证明的目标语义差异，尚无证据它实际被当前策略利用。

此项仍为固定 40 mm 核宽，没有 actor 时钟的宽→紧退火，也没有对应不饱和的收腿大误差负项。extension=.30 m 时该正奖励密度约 9.57e−6/s，反馈区分度很小。PPO 并不对 reward 的状态导数直接反传，不能把这称为“actor 梯度字面为零”。应先做实际每腿长度/竖直距离、空中姿态、收腿误差与支付的分布核查；若需要下一版塑形，优先采用每腿几何目标的宽代价+细核，再按 component actor clock 收紧，而不是先提高饱和项权重。

现有 contact-triggered 落地窗已经有宽/细速度、姿态、垂速、冲击、接触前带方向轮速及限轮矩代价。不能再把华南虎的接触前滚速对齐概括为我们“完全没有”。

## 4. 实际曝光与后续同模型复习

下表是 4096 环境的正式静态分配，不是有效 PPO 数或成功数：

| 域 | 普通 200+250 mm | 普通大阶八组合计 | 小阶 5–40 mm | rough 行合计 | 坡/横坡行合计 |
|---|---:|---:|---:|---:|---:|
| terrain | 30 | 245 | 1022 | 22 | 66 |
| jump | 18 | 144 | 610 | 10 | 30 |
| stepjump | 14 | 112 | 486 | 10 | 30 |

源 terrain pool 为普通阶 10%、小阶 42%；引入普通阶来自 surface traversal 10→2%、surface height joint 4→2%。融合其他组件后普通阶实际约 5.93%→3.55%→2.83%。因此**基础小阶和普通大阶后续复习存在**，也同时留下 rough/坡的经验摊薄风险。不得把所有行仍在场等同于质量充分；后段必须分别核有效 PPO transition 与实际接触/命令曝光。

普通阶以 terrain component actor clock 从 1000–1200 开 50 mm，1400–1600 开 100 mm，依次至 3800–4000 全开 200+250 mm。正式边界证书根据实际剩余 actor allowance 重新绑定；不足标准 allowance 时按 `min(4000,floor(.35×allowance))` 压缩激活终点，至少留 65% allowance 全难度学习。未激活行在平台前跑 +.1/−.1/0 的 5 s 循环，专项爬阶 assessment/reward 关闭，普通学习保留。`bind_course` 在后续 jump/stepjump 保存同一引入证书，不重新归零其 terrain 时钟。静态 origin=0/allowance=15000 仅是合同生成占位，不能当已经发生的准入父值。

行进 jump 的命名 03/06/10 行在开启 `iteration_probability_schedule` 时共同采样 [3,6,10,15,20,25,30] cm，`skill_commands.py:355–365` 同步 `FullTaskSemantics` 的 actual apex/COM rise 目标。因此 15–30 cm 不是漏采。其正向前进指令固定 .4 m/s（nominal h=.305），没有高速/反向行进跳的专门覆盖。早期 3/6/10 cm 概率合计 75%，jump component actor 6000 后 20/25/30 cm 合计 65%，各低 bin 仍有 5% 下限。后续保留旧 jump component，assist 按该组件累计时钟退火，不因换域重新获得免费辅助。

## 5. 华南虎能直接借用什么，不能据此补什么

SCUT rough_dash 真正启用了 airborne 的轮居中、低轮矩、接触前方向滚速、落地垂速与冲击代价；这类相位信用值得逐窗口比较。它的普通大阶主要靠地形×前进命令绑定、全局跟踪与姿态/接触成本，并没有本项目这套“200+250 mm 双平台、合法车体顶面承重、逐平台普通完成”的任务 API。因此直接复制其 reward 表不保证学出本机构的 body-first 动作。

还要区分配置写着与实际付费：保存 rough_dash 的 `airborne_leg_length_min` 配置是 `below_target_per_leg(target=.25)`，但该名字**不在实际 rewards 权重表**。`airborne.apply_reward_term_scales` 会跳过无权重/零权重 additions，所以不能把这一可选配置称为其交付 run 已付费的收腿 reward。主动 JumpTakeoff 在所核验三份保存配置仍关闭，不能用来证明其主动原地/行进/双级跳任务已经全部学成。其几何腿长接口与我们竖直平均 extension 也不同，不应机械照抄目标数值。

官方可复核源码：[airborne reward additions](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/state_machines/airborne.py#L983)、[rough_dash 保存配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml#L878)、[普通地形命令管理](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/manager/mdp/terrain/terrain_command_manager.py)。

## 最小后续行动（不新增晋级门）

1. 独立冻结跳阶落地目标修复，CPU 核对普通 jump 快照不变、平台停车误差为零、原接触前滚速目标不变、窗口不续期；源/合同/队列通过既有可审计版本路径采纳，不能原地改排队的 81ad。
2. 普通大阶进入真实 native 学习后，分别看每一级实际跨沿、车体承重、左右轮支撑、true unsupported 时长、每 reward 支付和 actor-owned 曝光。能走到平台与以车体搭台收腿的动作质量分别报告；不能新增一个顺序 gate 代替有效信用。
3. 对同一 checkpoint 的 jump/stepjump，旁路核对每高度、辅助/无辅助、两腿几何收腿、真实空中窗口、落地后实际零指令和速度/姿态。先确认目标与信用一致，再考虑核宽塑形；不拿辅助期的腾空当已学会完整跳跃。
4. 后段同模型固定回放复习低/中/高姿态静止、低/高速加减速与旋转、rough/坡、普通与主动双级阶；报告实际曝光和连续任务结果。当前源级审查没有新的能力通过结论，也未给实车 qualification。
