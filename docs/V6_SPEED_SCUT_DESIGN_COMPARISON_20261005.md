# 高速平地设计与 SCUT 实际 rough_dash 配置对照

日期：2026-10-05。范围：源码机制核查；不代表新增行为评测完成，也不代表修复已训练。

## 1. 比较对象与结论

本机官方仓库 `scut-wheeled-legged-rl` 的 HEAD 为 `b8ff79f3df855faf9dc92f4a282bd80c42649466`。对照其实际保存的 `pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml`、同目录 `agent.yaml`，并核查该提交的 `wheelbipe25_v3/env.py` 和 `manager/mdp/isaaclab/commands.py`。保存配置是实际运行参数；当前提交源码用于解释其机制，不能用另一个模板默认值替代。

我们的比较对象是已交付 `models/v6_speed_candidate_19837_20261005/policy.onnx` 的随包 `contract.json`、`agent_config.json` 与 provenance；训练源码为 `122d8db63f376f0737e89b4caf746e3f744aee8e`。该模型来自 speed 本域 actor 3563 终点，ONNX SHA-256 为 `294d80c8618fdc4baced6587c3c36edd8d97cfba3d99afd05b3639c681e8b738`，身份仍是未评定候选、非实车资格交付。待部署 v13 的停车池和非标准姿态出生不属于这份模型已学能力。

核心结论：**姿态、gyro 和动作平滑的基础项已经继承；真正差异在训练分布、命令组织、出生接管、执行器先验以及奖励相对尺度。** 不能把当前问题概括为“漏了华南虎的姿态 reward”。

官方原始文件：

- [实际 rough_dash env.yaml](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml)
- [实际 agent.yaml](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/agent.yaml)
- [环境实现](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py)
- [命令实现](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/manager/mdp/isaaclab/commands.py)

## 2. 已继承的姿态与平滑机制

两边均为物理 `dt=0.005s`、decimation 4，即 200Hz 物理/反馈控制和 50Hz 策略；普通连续 reward 都乘策略 `dt=0.02s`。

令投影重力水平分量为 `g_x,g_y`，`v_c` 为可见前向速度命令：

\[
a(v_c)=2\exp(-v_c^2/3)+2.
\]

| 项 | SCUT 实际启用 | 我们交付 speed 实际执行 |
|---|---|---|
| 俯仰 exp | `exp(-g_x²/0.02)`，权重 +1 | 相同；普通平地姿态参考为 0 |
| 横滚 exp | `exp(-g_y²/0.01)`，权重 +1 | 相同 |
| 俯仰速度缩放平方项 | `-2(a(v_c)g_x)²` | 相同，支撑跟踪门还会作用于俯仰项 |
| 横滚速度缩放平方项 | `-2(a(v_c)g_y)²` | 相同 |
| 身体水平角速度 | `-0.05‖ω_xy‖²` | 相同 |
| 动作一阶变化 | `-0.01‖a_t-a_{t-1}‖²` | 相同 |
| 腿动作二阶变化 | `-0.05‖a_t-2a_{t-1}+a_{t-2}‖²` | 相同 |
| 轮动作二阶变化 | 系数 -0.01 | 相同 |
| 关节速度/加速度、轮速/轮加速度 | -0.005 / -5e-7 / -1e-5 / -1e-8 | 相同基础系数 |
| 额外动作低通 | 实际 `use_action_low_pass_filter=false` | 不能声称缺少其实际启用的低通 |

SCUT 的 `flat_orientation_x/y_v` 和我们 `roll/pitch_velocity` 的名字容易误解：**它们不是身体角速度项**，而是随命令速度缩放的倾角成本。真正 gyro 项是上表的 -0.05，不能沿用“华南虎有 -2 角速度而我们没有”的说法。

随前向速度增加，平方系数从静止的 32 降至 3m/s 时的 8.816。SCUT 同样采用这个取向，故“高速姿态变软”不是我们独有的遗漏。以支撑成立、单轴倾斜、零姿态参考计算，下面是相对完全水平损失的 **reward/s**；每策略步再乘 0.02：

| 倾角 | 静止俯仰损失 | 3m/s 俯仰损失 | 静止横滚损失 | 3m/s 横滚损失 |
|---|---:|---:|---:|---:|
| 5° | 0.559 | 0.383 | 0.775 | 0.599 |
| 10° | 1.743 | 1.044 | 1.916 | 1.217 |
| 15° | 3.108 | 1.555 | 3.142 | 1.589 |

这些数值证明成本存在，不能证明 PPO 一定学会水平。加减速时应允许短暂合理平衡倾斜，匀速与停车后持续偏斜应单独检查。`atan(1.5/9.81)≈8.7°` 仅是质量中心平衡倾斜的物理量级，不是要求车体强制倾斜 8.7° 的指令。

## 3. 真正不同的机制

| 机制 | SCUT 此 run 的实际配置/源码 | 我们 speed_v10 交付模型 | 含义 |
|---|---|---|---|
| 速度范围 | 基础 vx ±2.7m/s；特殊高速段 ±2–3m/s | 固定/连续平移覆盖到 ±5m/s | 我们要求更宽的速度域，不能按相同训练更新数默认同等难度 |
| 旋转范围 | 基础 yaw ±2πrad/s；两个特殊旋转段约 6.28–10.21 和 10.21–14.14rad/s | 覆盖至 3rps，即约 18.85rad/s，以及组合平移自旋 | 高速分布和任务耦合更广 |
| 命令组织 | 常驻连续空间，特殊模式互斥概率桶 | 129 个常驻命名组；名义 65% 基础/35% 扩展，各组内部固定或连续采样 | 不是旧技能被全部移走，但联合分布与每类有效样本仍需审计 |
| heading | `heading_command=true`、`rel_heading_envs=.5`、刚度 5；普通 heading 子集用 `clip(5*wrap(heading_target-heading),yaw_range)`；特殊模式绕过 heading | 直接速度/yaw 命令，无等价的世界航向目标闭环命令生成 | heading 可训练航向修正和减少横向跑出边界；它不直接替代俯仰/横滚 reward |
| 命令驻留 | 5–15s | 1–8s，另有 20% 长驻留 8–15s；停车序列另有相位安排 | 更频繁切换，留给恒速/驻停的完整窗口不同 |
| 静止配额 | `rel_standing_envs=.1`，另有非 standing 特殊零命令模式 `rel_envs=.1`，特殊模式须回合达到 5s；正常池委托父类重采样、地形覆盖也会改变最终份额 | 固定静止锚点 12% + 连续高度静止 8% = 20%，另有升降和停车组 | 我们**没有缺少永久静止采样组**；SCUT 不能简单写成“20% 环境永远零速”或“只有 10%” |
| 地形×命令 | 同一 rough_dash 策略混合粗糙、坡面、台阶等；`terrain_command_overrides` 给各地形限定速度、站高、heading 和特殊模式权限 | speed worker 4096 环境全部 flat；terrain 为后续不同 worker | 缺的是同一高速策略内地形与命令的二维绑定，不能把后续 terrain 配置当已进入这份 ONNX |
| 台阶绑定例 | 高速台阶 vx ±1.5–2.7、高度 .22–.34m；低速台阶 vx ±.5–1.5；限制非 heading yaw ±.1并关闭特殊模式 | 高速命名组与高度变体更宽；平地组不构成该二维覆盖表 | 安全可行工作区的绑定很重要，但不能机械照抄他车的高度/力矩 |
| 高速开启时钟 | 此 run 三个特殊运动段和额外零命令段 `iteration_start=0`，`iteration_end=-1`；回合准备期 5s | v10 现有所有静止/低高速组常驻，不以能力门阻断 | 源码有迭代门功能，不等于这份实际 run 使用 2000/3000/4000 解锁；跳跃状态机与助跳在此 run 关闭 |
| 非标准出生/接管 | `use_leg_random_start=true`、预设地面腿姿态出生启用；零主动力矩 .2s、接管窗口 1.5s、命令限制 vx/yaw ±1；另启用 air 出生 `.4` 模式且受地形限制 | 当前交付 speed 无这套普通 reset→短零力矩→RL 接管；v13 是待部署补强 | 这是实际缺项，补后仍需看确定性接稳质量，不能视为漂移的必然修复 |
| 小幅推扰 | vx/vy ±.25m/s、间隔 5–10s，普通训练常驻事件 | 交付 speed 普通组 `push_m_s=0`；边缘出生/降级有其他语义 | 普通姿态恢复样本不同；推扰不会自动教会原地不漂 |
| 延迟 | 观测、动作延迟实际开启；DelayBuffer 在物理链上更新，lag 在初始化/reset 重采，逐帧重采代码被注释 | 75% 环境使用物理信号延迟；obs 0–80ms/action 0–60ms 的概率混合，5ms 抖动，随 actor 0–1000 增强 | 我们不缺延迟，当前分布更宽且有帧间抖动；不能把 SCUT 的 lag 错算为策略 20ms 一帧 |
| 执行器 | 腿 IdealPD 60/2、armature .015795；轮实际速度模式，速度动作比例 10，P/阻尼 .2、轮力矩上限 5 | 腿 PC PD160/2.5、腿动态响应未 qualified；轮 P .6、上限4.5，已接入轮侧辨识 | 抗抖必须结合刚度、惯性和闭链机械结构，不能只比较无量纲 action_cost 的系数 |
| 失败确认 | roll/pitch 40°，普通接触/姿态需连续10策略步（.2s）；数值异常立即终止；有相位覆盖 | fall_confirmation=0，另有闭链 gap、膝、气簧等真实机械失败条件 | 短暂非稳态被终止的时间和成本不同；不能直接移除机械条件来仿照 |
| PPO | 5 epoch/4 minibatch、entropy .005、LR1e-4 adaptive、value_loss2 | 同 epoch/minibatch/entropy，固定LR1e-4、value_loss4 | fixed LR 是此前恢复 bug 的验证修复，不应未核查就改回 adaptive |

SCUT 延迟 YAML 为 obs `[1,4]`、act `[1,3]`；初始化/reset 的 `torch.randint(lo,hi)` 上界不含 hi，实际采到 obs 1–3、act 1–2 个物理更新。按 5ms 物理更新对应约 5–15ms 和 5–10ms 的缓冲滞后；观测采样/读取相位仍有额外年龄。这个源码核查结果修正了“他们是 20–80ms/20–60ms 策略帧延迟”的旧口径。

### 3.1 基础跟踪与姿态/高度的耦合不同

SCUT 此 run `vel_upright_gate_enabled=false`、`height_upright_gate_enabled=false`、俯仰门关闭；但 **`vel_height_gate_enabled=true`**，线性门：

\[
g_h=\operatorname{clip}\left(\frac{0.10-|h-h_c|}{0.10-0.05},0,1\right).
\]

高度误差 ≤5cm 时门为 1，≥10cm 时为 0。源码 `env.py:3886` 附近同时对速度/yaw exp 正项和 square 负项乘这个门。因此不能概括他们“没有任何基础跟踪门”，也不能不经分析照抄：关门同时豁免误差负项是与我们 dense 设计不同的取舍。当前3/5m/s组平均高度误差约3.2/3.9cm，低于其5cm全开阈值，故不能声称复制此门就会修复这些组；平均值也不能推导每一帧都在阈值内，须直接统计门的实际占用。该门检查高度误差，不检查俯仰/横滚，不能当作倾角修复机制。

我们先组装旧基础/精度项，随后 `apply_dense_tracking` 删除旧 velocity、yaw、height、stand precision，再用未裁切 broad pseudo-Huber 与局部精核替换；三个基础 peak 常量只付一次，负成本不乘直立门。ordinary velocity/yaw 在俯仰时仍有误差代价，高度/静止成本带物理支撑门。不能通过改旧 `precision_tracking.stationary_width_m_s=.03` 就修复当前生效 dense 项。

零速支撑时当前静止成本为：

\[
r_s=-1.5[\sqrt{1+(e/.10)^2}-1]
+0.75\operatorname{expm1}[-(e/.04)^2].
\]

5cm/s 残速已付约 -0.770 reward/s，即 -0.0154/策略步；10cm/s 为 -1.370 reward/s。没有“静止奖励未启用/在6cm/s后全部零信号”的证据。旧 `zero_command_velocity_scale=3` 只作用于稍后被 dense 删除的 `stand_translation`，旧 `quiet_*` 属于另一 reward profile 分支，不能据字段存在声称增强已生效。

### 3.2 终止与力矩/功率成本相对尺度不同

| 项 | SCUT 此 run 最终语义 | 我们当前最终语义 |
|---|---|---|
| 普通失败 | `termination=-200` 在统一 reward 汇总中**先乘 .02**，因此普通失败单次约 -4；airborne 状态还可能放大该项 | `termination_event_cost=200` 独立事件一次 -200，不乘 dt |
| 力矩平方税 | `-1e-5 Στ²` reward/s | `-1e-4 Στ²` reward/s，10倍 |
| 轮正功率税 | `-1e-5 Σmax(τω,0)` reward/s | `-1e-4 Σmax(τω,0)` reward/s，10倍 |
| 前向 exp 跟踪正项 | 权重1.25 | dense 基础峰1.0 + 负误差项 |
| vertical velocity | 普通 -0.5 v_z²；airborne 有相位调整 | dense ordinary_vertical_velocity_weight=0，意在不惩罚可见升降目标 |

这是**已证实的相对尺度差异**，而非发现新接口 bug：我们的事件式 -200 是合同明确声明的语义。它会改变一次失败相对持续跟踪/控制成本的权重，但尚无隔离对照证明它导致当前偏斜、抖动或爬行。不能沿用“同字段 -200所以两边一样”的结论，也不能未冻结方案就将其改成 -4。

保留 -200 已有明确依据，见 [2026-10-02 奖励参考审计 §4](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/docs/V6_POLICY_REWARD_REFERENCE_AUDIT_20261002.md:42)。该审计用原生 dense 跟踪项和 gamma=.99 比较大误差代数包络：继续10s约 -90.845，正确时间截断 bootstrap 约 -91.446；失败成本4时首步结束约 -4.914，反而优于纠错存活。失败成本200时约 -200.914，通过这一反例；100也通过，但当时没有物理行为证据支持降低。这不是全状态证明，却足以否定盲目复制 SCUT 的 -4。**本次对照不将 -200 重新列为已证实缺陷，不要求恢复旧的错误回报排序。** 后续只针对实际失败样本优势/critic residual 检查尺度，力矩税另做独立对照。

探索初始化两边均 `std=1.0`、entropy .005。我们的最后100训练更新 mean_std≈.441，但该值不能当确定性 ONNX 动作抖动的来源；ONNX 导出的是确定性 actor。训练曲线中的探索、信号延迟和 reset 与回放控制质量必须分开。

## 4. 目前可用证据与修复优先级

此前 speed 完整100更新窗口：start_stop_1 残速13.14cm/s，95.32%驻停样本超过3cm/s；230/305/430mm静止组平面均速12.62/12.84/7.58cm/s。窗口含探索、出生、命令斜坡和延迟，不能当纯稳态确定性评测。高速3/5m/s组高度误差32.2/39.2mm，缺少这一窗口完整俯仰/横滚相位统计，不能用高度误差替代倾角证据。

本机确定性零命令回放10–35s平均前向速度4.96cm/s，roll约 -0.69°、pitch约 +0.28°，已证实无探索时仍爬行；这个窗口未验证高速/制动质量。用户看到的移动偏斜与抖动需要同模型同条件轨迹记录，不应归咎于用户观感或网络导出精度。

建议按以下顺序推进，不改变网络结构：

1. 同一 ONNX 记录长静止、低中高速恒速、释放制动、正反yaw；分加速/恒速/减速/停车后统计角度、gyro、动作一二阶差分、实际轮速/力矩、支撑、饱和和样本数。失败/未停稳样本不能被指标选择过滤。
2. 先核对推理 ABI、轴映射、参考/实际命令斜坡和200Hz fresh PD；已有修复的 GUI 命令斜坡不能当模型重新学习了。
3. 对齐训练分布：常驻静止/低中高速继续保留；限定可行的高度×速度×yaw 联合区域，添加显式长驻停和足够恒速驻留；普通训练加入小推扰及逐级非标准出生接管。待部署 v13 只解决其中部分覆盖，不保证修掉基础漂移。
4. heading 命令生成作为独立候选，输出仍为当前可见 yaw_rate 指令，不将世界航向目标隐蔽塞进奖励。它针对航向/横向边界，不替代姿态成本。
5. 核对奖励相对尺度，优先量测力矩/功率税实际支付与采样优势；保持已通过大误差反例检查的单次失败200，终止尺度仅检查真实失败样本优势/critic residual，不将降为4列为高速修复优先项。姿态/动作平滑已经继承，不能仅再加一个同名项。若确定性证据显示高速匀速倾角确被容忍，再隔离试验姿态速度缩放下限；若证实高频输出振荡，再隔离试验平滑成本或训练部署一致的目标滤波。
6. 完整机构腿侧 real2sim 与 PD离散等价仍需独立验证。其他车型的60/2、惯量与限幅不可直接替换本机160/2.5；在本机仿真也不稳时，应先修仿真行为，不能仅用实车gap解释。

上述源码对照已完成；独立行为因果判定仍待完成。没有启动新的训练、没有改训练源码、没有修改 RMCS 或实车驱动。不能声称 SCUT 绝不遗忘、无相位、完全水平、或这次候选补强已学会。
