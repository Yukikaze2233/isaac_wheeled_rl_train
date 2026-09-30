# V6 S1：华南虎源码对照与训练／评测错位审计

日期：2026-09-30。状态：**审计及确定性奖励复现完成；未修改训练实现、合同或重启 Kaiser**。

本审计承接 `8c42db2` 接续结果：S1 消费 698→738，actor 学习谱系 368→408，45 项双 seed 均由 A 的 33 项通过降至 29 项，S1 7/16→4/16。无新增通过；退步为 `stand_230mm/350mm/380mm` 和 `forward_05_height_230mm`。S1 存活率仍为 100%，属于精度及保持退步。余 262 更新，不返还 A/B 或弃选分支消费。详见 [结果回执](../reports/v6_s1_continuation_20260930/result_review/README.md)。

## 1. 对照基准与结论边界

华南虎对照固定在 `scutrobotlab/wheeled-legged_RL` 的 `b8ff79f3df855faf9dc92f4a282bd80c42649466`，主要核对 V14 Flat、继承的 wheelbipe25_v3、命令管理器和 airborne/jump 状态机。不得把不同子类、注释参数和已启用配置合成一个不存在的默认方案。

用户提供的 [开源说明](https://bbs.robomaster.com/article/1942489?source=1) 本轮网页提取仅返回标题等外壳；以下机制结论来自已核对的作者源码，不声称从页面读到了完整正文。

**我们的设计确有问题，但“698 时固定档份额被切走，所以遗忘”不成立。当前最直接的实现缺陷是普通静站能够持续领取恢复／落地正奖励，恢复成功却失去奖励。** 此外，联合任务覆盖、同步相位、短 worker 的状态生命周期和总 reward 的统计总体存在错位。它们不能被笼统归为训练量不足。

源码和确定性探针证明的是激励漏洞；TensorBoard 证明该奖励在 230mm 训练中实际支付。尚无消融证明它解释全部四项回归，350/380mm 的完整因果也未闭环。

## 2. 逐机制对照

| 机制 | 华南虎固定版本源码 | 我们实际运行 | 判断 |
|---|---|---|---|
| 命令分布 | 基类有连续速度／高度；V14 Flat 用 `SpecialModeUniformVelocityCommandCfg`，普通 vx ±2.7m/s，驻留 5–15s，高度 0.20–0.42m | 同时常驻多个 skill 池；静止池本身已是固定九档＋连续高度混合 | 连续域值得保留；“我们只有九个互斥档位”不准确。连续分布也不保证不遗忘 |
| 技能推进 | `dash` 从 iteration 2000、`spin_low` 3000、`spin_mid` 4000 启用；README 列平地、落地预训练、粗糙地形等任务 | S1→S2 串行验收；本次仍在同一 S1，没有跨阶段 | 不能用“他们完全无阶段、所有技能从头全在”解释差异；应学基础分布常驻、技能渐进加入 |
| 静止份额 | 每次命令重采样以 0.1 概率选择 standing，再令其速度为零 | S1 总静止池 45%；当前九档各约 2.5% 总环境 | 0.1 是概率，不是永久固定一批环境，更不是不退化保证；我们不缺总静止量 |
| 高度观测 | V14 Flat `use_absolute_height=True`；核心 actor 高度块为 `height_cmd`，配置中的测高／raycast 不等于 actor 直接得到真值高度 | 35D 每帧包含连续高度命令，另有关节与 IMU；没有真实高度通道 | “我们的命令靠记忆、他们靠 raycast 所以连续泛化更好”没有源码依据；本轮不改网络 |
| 奖励相位 | airborne 有接触计时退出及最大持续时间；落地轨迹启用时由自身有限时钟清除，基础 Flat 默认关闭该轨迹；jump 附加项按 active/push/tuck 窗口付费 | 任意非主动跳跃模式的 LANDING 或 RECOVERY 都付落地正奖励；RECOVERY 退出取决于达到高度／姿态阈值，无该奖励独立截止时钟 | 有明确可修的奖励激励错位，见第 3 节 |
| 训练生命周期 | 训练入口一次 `runner.learn`，请求随机初始 episode length；不中途每 40 更新重新建场景 | 每 40 更新重建物理场景；恢复网络、优化器、RNG，但不恢复物理状态／命令相位 | 保留模型状态不等于连续采样；不能把评测分块需求变成短周期冷启动 |

源码：[V14 配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py)、[命令采样](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/manager/mdp/isaaclab/commands.py)、[actor 观测](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py)、[训练入口](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/scripts/rsl_rl/train.py)。这些机制降低风险，不构成华南虎在我们的严格评测上永不退步的实验证据。

## 3. P0：恢复成功反而失去正奖励

当前代码路径：

- `task.py::PhaseTracker.update`：无跳跃请求时，普通静站丢失轮接触累计至少 30ms 也能进入 FLIGHT，重新接触后进入 LANDING/RECOVERY。
- `env.py`：stable 定义包含姿态 upright 和高度误差 `<10mm`；连续 stable 满 0.5s，RECOVERY 才退出为 GROUND。
- 同文件落地奖励块：`LANDING | RECOVERY` 且 `mode != 4` 就叠加高度正奖励（权重 2）和竖直速度正奖励（权重 1）。没有按普通静站／合法落地任务隔离，也没有独立到期停止。
- dense 普通高度／静止项在 RECOVERY 仍生效，但不会移除上述正奖励。

这允许姿态或高度误差在退出阈值附近波动时长期保留额外收益；“不够稳定”成了可获收益的相位条件。不是说机器人有意钻漏洞，而是优化目标确实给出了错误比较。

### 原生代码确定性复现

[探针脚本](../reports/v6_s1_curriculum_audit_20260930/probe_recovery_reward.py) 直接复用 `PhaseTracker`、`landing_reference`、dense 误差核，并从 `env.py` 的 AST 编译实际落地奖励块。两个合成状态均 upright、支撑、速度为零，接触持续 2s；从 FLIGHT 开始，比较高度误差 9mm 与 11mm。保持其余相同奖励不变，只比较有差异的高度与落地项。

| 状态 | 最终相位 | dense 高度项 /s | 落地正奖励 /s | 两者小计 /s |
|---|---|---:|---:|---:|
| 高度误差 9mm | GROUND | −0.52849 | 0 | −0.52849 |
| 高度误差 11mm | RECOVERY | −0.73530 | +2.81189 | +2.07659 |

**更差的 11mm 状态反而多拿 2.60507 reward/s。** 这是合成激励检查，不是两条经过物理可达性验证的实机轨迹，也不是 PPO 因果消融。原始数值及源码 SHA 在 [probe 结果](../reports/v6_s1_curriculum_audit_20260930/recovery_reward_probe.json)。

运行时证据：本轮最后 5 次更新，230mm 的 `landing_height_tracking` 均值 **+0.97554/s**，`landing_velocity_tracking` **+0.47189/s**，`landing_impact` **−0.07588/s**。这是组均值，不等于每个环境都处于恢复态；但是明确证明静站组末期仍在领取落地奖励。350/380mm 同期该正奖励为零，故不能据此宣布四项退步全由这一机制导致。

华南虎的 [airborne 实现](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/state_machines/airborne.py) 用接触／时长结束 airborne，落地轨迹由 `duration_done` 清除；[jump 实现](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/state_machines/jump_takeoff.py) 有请求进入、相位窗口及退出。应借鉴其任务和时间边界，不能仅复制一个 Gaussian 形式。

建议修复契约：普通静止继续正常高度／速度跟踪；合法落地瞬态保留接触冲击、缓冲轨迹，但不能无限按 RECOVERY 标签支付正基线。落地奖励用独立、有限、不能靠不收敛续期的时钟；与普通跟踪替换或平滑交接，验证跨相位改善不会造成总回报骤降。重新失接触的重触发条件也要防止靠抖动续领奖励。修复前先记录各组相位占用、触发来源、相位时长及对应奖励，避免盲目把所有落地机制整体关闭。

## 4. 固定档并未被切走；联合行为和时间覆盖才是缺口

高度固定份额 80%→50% 的边界是 **stage actor update 200**。A 结束已经是 368，本轮 368→408，全程 50%。698 是消费计数，不能拿它当课程切换时钟。

实际每 rollout 24 步、20,000 环境，40 次更新中以下计数恒定：

| 档位 mm | 固定档环境数 | 总环境占比 |
|---:|---:|---:|
|230|460|2.300%|
|250|515|2.575%|
|280|528|2.640%|
|305|484|2.420%|
|320|490|2.450%|
|350|527|2.635%|
|380|505|2.525%|
|400|527|2.635%|
|430|466|2.330%|

230mm 合计 **441,600 transition**，不是零访问。九档设置 1–2% 下限适合后续阶段守约，但在这轮不会提高既有份额。250/305mm A 前后都未通过，不能拿它们证明“仍采到所以没退步”。

真正缺项是 **高度 × 动作 × 相位**：S1 的前进／后退／停车／yaw 主要固定在 305mm；保护评测却含 230mm 前进等组合。静站 230mm 的份额不等于 230mm 行走的份额。后续配额必须写到联合行为，而不是只有九根静态锚点。

时间维度还有两点：

1. 1,333 个 start_stop 环境共享同一 4s 分段时钟、同一出生相位。40 个 PPO batch 只有 **7 个含制动窗口、14 个含 settled_stop 窗口**，最后 5 个两者均没有。20k 并行不能自动消除这样的时间相关性。需分散相位且使物理初态、当前命令、历史目标一致，不能只随机改 elapsed 来制造虚假停车状态。
2. 每个 40-update worker 只演进 `40×24×0.02 = 19.2s`。静站回合 20s，高度扫描回合 28s；这些组本轮没有一次 done。命令保持 10/15s 并不缺，静站最后 5 个 batch 的 `continuous_zero_15s=1`；真正缺的是更长持续采样及同一个学习进程中的多轮 reset 分布。

华南虎长期 `learn` 和逐环境重采样有助于打散时序，但不能直接推导它在我们的 stop/高度专项中保证足够配额。我们的补强应可审计，而非换成完全均匀采样后假定覆盖已解决。

## 5. “训练好”这个前提也不成立

最后 5 个 batch 的 230mm 训练平均平面速度 **2.769cm/s**，高度 MAE **9.179mm**；并没有达到优质驻点状态。10s 最大漂移 10cm 的验收，对持续同向速度偏差的容许量约为 1cm/s；平均速度模长不能直接换算净漂移，但足以说明训练与评测并非“训练已极精确、评测突然完全不会”。需要追加带符号 DC 偏差、真实 10s 位移及相位条件统计，而非只看速度模长或总 reward。

RSL-RL 的 `Train/mean_reward` 统计最近已结束回合。本轮 stand、height_scan、正反 yaw 的 `done` 都为零，结束回合来自前进／后退和 start_stop。因此该总 reward 的 32.46→51.62 **不是全高度静站能力上涨的证据**。持续 rollout 的 `LearningAudit` 更适合当前问题；所有新增曲线继续写 TensorBoard，不新建独立训练曲线体系。

训练动作有探索噪声、评测用确定性策略，两者分布差异也需控制；当前只能列为待核查因素。不能仅凭组优势均值或 action KL 断言存在梯度冲突，需同冻结 actor 的诊断或单变量消融。

## 6. 补救顺序与预算

1. **先修奖励语义和仪表**：解决恢复奖励持续支付及成功退出降回报；验证普通静止、接触短暂丢失、合法落地、跳跃退出四类路径。相位比例、奖励实际门控、带符号速度偏差和位移写入已有 TensorBoard。确定性激励检查之外，还要做相同冻结 actor 的原生物理回放，确认收益变化来源。
2. **命名联合覆盖**：固定九档实际最小份额守约；明确低／中／高姿态与前后行走、yaw、停车的组合；连续域仍常驻，预算内重新分配，不把每种新增组合都重复加到总量上。静站与低姿态运动分别记账。
3. **打散停车相位，维持环境生命周期**：在状态一致的条件下错开各环境时钟。把“每 20/40 更新封存并评测”与“销毁场景”分离；用常驻 worker 到安全 checkpoint 边界停队、评测，再继续同一场景。不能为解决短 worker 直接取消回归评测或一次性盲跑全部余量。
4. **先因果核查，再花 PPO 预算**：最小优先分支只改已证实的恢复奖励问题，其余 LR、网络、PD、响应、配额不同时变，以判断该缺陷的实际贡献；联合采样／相位改善随后单独归因。仅两个短分支比较时必须写明独立变化，不再称“多改几项仍是单变量”。
5. 后续若从较好的 A 学习状态再出发，原消费仍为 **738**；actor 谱系与消费继续分开，余量仍 **262**。冻结新合同和代码、记录迁移基线后再部署；先针对本次退步及停车／yaw 的固定小套件观察，再进行现有 45 项双 seed 复核。保持原阈值，不用放宽 gate 掩盖回归。

暂不把“其余都是量没练够”“静止 reward 已被 B 证伪为无问题”“华南虎绝不会遗忘”写入结论。B 只否定了当时将静止两项权重翻倍的修法，并没有审计过本节的相位奖励漏洞。

## 7. 证据与复现

- [TensorBoard 派生摘要](../reports/v6_s1_curriculum_audit_20260930/audit_summary.json)：原始远端训练路径、导出 SHA、九档实际配额、任务分组、停车相位和 LR。
- [TensorBoard 导出](../reports/v6_s1_curriculum_audit_20260930/tensorboard_metrics.json)：942 个 tag，来自同一已封存 worker 的 event。
- [原生奖励探针及结果](../reports/v6_s1_curriculum_audit_20260930/recovery_reward_probe.json)：可在仓库根目录以 `PYTHONPATH=src /home/yukikaze/isaacsim60-venv/bin/python reports/v6_s1_curriculum_audit_20260930/probe_recovery_reward.py` 复现。
- 本次没有新 PPO 更新，没有修改运行合同、RMCS、网络或硬件标定；仍为封存待修复状态。
