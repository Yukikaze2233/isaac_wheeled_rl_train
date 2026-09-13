# V40 Transformer 策略与训练范式设计

日期：2026-09-13。状态：**研究设计，尚未实现新网络或训练器，尚未进行 Transformer 训练、仿真或推理测速。** 本文依据原论文、官方实现和现有 V40 观测/控制代码，区分文献证据、设计选择与待验证假设。

## 1. 结论

**可以完全不用 RSL-RL，也可以采用 Decision Transformer（DT）。** RSL-RL 是训练软件，Transformer 是模型结构，DT/ODT 则同时定义了条件化输入、学习目标和数据流程。三者不是同一层面的替代品。

建议确定两个可比较的候选，而非先把旧网络整体换掉：

1. **CT-S：命令条件化的小型因果 Transformer。** 以单步观测为时间 token，直接在线强化学习，或先用传感器—动作轨迹预训练，再在线改进。部署无需 reward/RTG。
2. **DT-S / ODT-S：命令与回报条件化的小型 Transformer。** 离线用轨迹学习动作，ODT 用随机探索、实际回报重标记和轨迹 replay 继续改进。主算法不需要 PPO、critic 或 GAE。

当前更适合优先验证 CT-S 的原因是：已有 GPU 并行仿真与命令跟踪任务，但尚无经核实、覆盖高低静止/换高/恢复的完整训练轨迹集。**这不是否定 DT；若目标是转向数据驱动的控制学习，DT/ODT 是有依据的独立路线，数据与 RTG 协议必须成为设计的一部分。**

Transformer 的合理收益假设是利用更长的动作—响应历史，改善对时延、轮滑瞬态和隐含动力学的适应。更大的网络不会自动修正模型归属、错误标签或不可观测的持续滑移，也不能直接证明高低站立成功。

## 2. 原始研究实际怎样训练

| 路线 | 输入/目标 | 是否在线 | critic / PPO | 已核实的边界 |
|---|---|---|---|---|
| [Decision Transformer][dt] | RTG、state、action；连续动作 MSE | 原主实验为离线 | 主算法均无 | 从 state token 预测当前 action；不是从当前 action token 抄答案 |
| [Online DT][odt] | 随机动作分布 NLL＋平均熵下界；真实 RTG 重标记 | 离线预训练＋在线微调 | 均无 | 官方实现为 tanh-Gaussian、轨迹 replay；不是 PPO finetune |
| [Humanoid Transformer][digit] | privileged teacher PPO；学生 PPO＋退火教师 KL | 学生直接在线 RL | 有 | 主方法不是纯 BC，也不是无教师的纯 PPO；v1/v2 标题和观测表有版本差异 |
| [Humanoid Next Token Prediction][hntp] | 离线 observation/action 预测，MSE | 学生主训练为离线 | 学生主目标无 PPO | 不含 RTG 的 sensorimotor modeling，不能称为原 DT |
| [LocoTransformer][loco] | 视觉与本体 token 融合＋PPO | 在线 | 有 | attention 主要沿空间/模态，不是本项目的时间记忆 |
| [GTrXL][gtrxl] | 门控残差、segment memory＋V-MPO | 在线 | 有 value，原论文非 PPO | 主要是记忆任务证据，不是 V40 实机验证 |

关键更正：不能说“DT 只能离线”。ODT 已实现离线到在线的完整流程；但其证据不等于“没有预训练数据也一定能高效从零学会机器人平衡”。

原实现也不应整套照抄：ODT 的 NLL 屏蔽 padding，但熵项直接对所有位置求均值；适配时应明确有效 token 的熵统计。LocoTransformer 仓库中的 `torchrl` 是作者自己的历史库，**不是**今天 PyTorch 官方的 `pytorch/rl`。

## 3. 机器人接口：先保留物理含义

依据 [`build_observation` / `build_critic`](../src/wheeled_tasks/v40/core.py) 和实际 run 的 v2 合同：

| 单帧 actor 输入 | 维数 |
|---|---:|
| 机身角速度 | 3 |
| 机身系重力方向 | 3 |
| 指令：vx、wz、目标高度 | 3 |
| 四个腿关节相对 nominal 的位置偏差 | 4 |
| 六个关节速度 | 6 |
| 上一周期已执行、裁剪后的归一化动作 | 6 |
| **合计** | **25** |

当前 actor 将 5 帧拼为125维。critic29 是 clean actor25＋真实机身线速度3＋实际基座高度1。**目标高度不是实际高度；额外四维仿真真值不能直接放进可部署 actor。**

新模型仍输出6维：四个腿位置控制量、两个轮速度控制量，经同一 action decoder 和反馈控制器转为力矩。actor100 Hz、物理/反馈200 Hz；每个物理子步重算反馈。v1/v2 虽然维数相同，腿/轮缩放、action clip、噪声和终止不同，必须绑定实际合同哈希。

高度目标先限定在已核对的0.28–0.32 m内。此范围是研究覆盖范围，不是声称每个姿态已经实机验证。现有机械短杆归属及低位净接触来源仍会影响后续物理解释。

## 4. CT-S：命令条件化因果 Transformer

### 4.1 具体网络

| 部分 | 首版候选 |
|---|---|
| 输入 | `history[B,16,25]`，从旧到新，当前帧最后 |
| 帧投影 | Linear(25,64)＋固定时间位置编码 |
| 主干 | 2个 pre-LN causal attention block |
| Attention | 4 heads，每头16维；只允许当前及过去 token |
| FFN | 64→128→64，GELU，残差连接 |
| Dropout | attention / FFN 均为0 |
| 输出 | 取最后 token，LayerNorm，MLP 64→64→6 |
| 探索分布 | 独立 Gaussian；6个可学习 log-std，部署只导出均值 |
| Critic | 在线 actor-critic 路线使用独立29→256→128→64→1 MLP |
| 首项消融 | context16→32，主干不变 |

每个 token 已包含前一动作，首版不重复追加 action token。16/32帧的名义覆盖为0.16/0.32秒，首尾采样间隔为0.15/0.31秒。一次输出全部6个动作，不按六个关节自回归生成，不预测一段开环动作 chunk。

按上述 Linear bias、LayerNorm affine、Gaussian log-std 计算，CT-S actor 约 **73,292 参数**；与现有 MLP 的约73,804参数接近。完整窗口 forward 的主要矩阵乘算术量约为1.144M MAC（16帧）、2.415M MAC（32帧），**不包含 softmax/LN/访存/调度，不能据此承诺100 Hz时延。**

### 4.2 记忆和条件边界

- 保留每一历史帧当时的命令。0.28→0.32 m切换时，只改变当前和未来命令，不重写过去、不清空动力学历史。
- 每个 env 独立 history，同一 policy tick 只推进一次；只清理真正 reset 的行。
- 首版沿用 repeat-first：reset 时上一动作归零，同一份首帧观测重复填满窗口，不重新采样16份噪声。
- 首版完整窗口重算，不保存 KV/learned hidden。PPO更新后旧 embedding 不再等价于新模型编码；即使权重不变，滑窗丢弃最老 KV 也不必等价于多层完整窗口重算。
- Actor 只接收当前可获得的传感/命令信息；未来状态、当前目标动作和 critic 真值不进入该动作的计算。
- 观测缩放先与基线保持一致；LayerNorm只沿 token 的特征维。暂不一起引入 running normalization、BatchNorm、RoPE、门控残差等多个变量。

### 4.3 三种训练方式

**A. 直接在线 RL。** Transformer 输出随机动作，仿真给 reward，用策略梯度训练网络本身。优先用 PyTorch 官方 TorchRL 的 `GAE` / `ClipPPOLoss` 组件，配项目自己的同步 GPU 采集与历史窗口管理；无需 RSL-RL。模块化复用损失实现，比重新维护一套完整 PPO 更符合本项目可复查需求。具体 TorchRL release 与环境兼容性在未来实现阶段固定。

**B. 轨迹预训练＋在线 RL。** 先在已有合格轨迹上做动作预测；可选添加一个明确的 observation 预测辅助目标，再用在线 RL 改善闭环表现。只用动作监督是 sequence BC，不因网络是 Transformer 就成为 DT。用于监督的旧数据不直接充当新策略的 on-policy PPO rollout。

**C. 教师正则化在线 RL。** 学生自己执行动作，在学生访问的状态上查询冻结教师，目标为 `L_PPO + lambda_teacher * KL(student || teacher)`，教师权重逐步退火。方向与上述 humanoid 论文一致。旧 V40 actor没有特权输入，不应因其 critic有真值就称它为 privileged teacher；旧策略有漂移，也不能默认其动作都是专家标签。

复旦式速度估计可以作为后续单独消融：由时序表示预测当前线速度，用仿真真值监督，部署只用估计。它可能改善表示学习，但本体感知不能保证辨认所有匀速滑移；不作为首版强制增设的损失。

## 5. DT-S / ODT-S：真正改变训练范式

### 5.1 Token 结构

先保留一个原法参照：`[RTG_t, observation25_t, action6_t]`，从 observation token 预测 action。窗口第一帧保留 previous action 是有意义的；它与显式 action token 的冗余并不造成当前动作泄漏。

面向动态命令与持续控制的候选进一步拆成：

| Token | 原始字段 | 维数 |
|---|---|---:|
| B | 归一化的剩余回报预算、剩余时长 | 2 |
| C | 当前 vx / wz / 目标高度 | 3 |
| X | 原25维去掉command的其余输入 | 22 |
| A | 当时传给控制器的归一化动作 | 6 |

每个时刻顺序为 **B、C、X、A**，从 **X token** 预测当前6维动作。causal mask允许看到当前预算、命令、状态和历史动作，不能看到当前A。真实A可用于 teacher forcing后续token，不能用于预测它自己。

首版仍用 d64、2层、4 heads、FFN128；context16，对应64 tokens。context32对应128 tokens。模型参数仍是约7–8万量级，但 token 数与 attention 成本高于CT-S；不能按参数量相近推断吞吐相近。d96/3层仅作为容量候选。

这四token方案是**本项目提案**，不是声称原DT/ODT已有command token和剩余时长字段。需要显式修改projection、mask、输出位置和数据协议；不是把state_dim改成125就完成。

### 5.2 离线DT怎样学习

训练样本是完整、时间对齐的轨迹。以一个有明确终点b的片段为例：

`G_t = sum(reward_t ... reward_(b-1))`

对有效时间步执行动作监督：

`L_DT = mean_valid(|| action_pred_t - action_label_t ||²)`

原实现还定义了state/return预测头，但主loss只训练动作；未训练的return头不能充当critic或真实RTG估计。对本项目也应先明确每个头的真实loss，不因存在模块就宣称学会世界模型。

离线验证按完整session/动力学条件分组，不能把相邻相关帧随机拆分。高分次优数据可以使用；但只有checkpoint、TensorBoard和汇总CSV，无法逆推出旧训练时的逐步轨迹。

### 5.3 ODT怎样在线改进

1. 在离线轨迹上预训练随机策略。
2. 给定目标回报，用该策略采集新的完整片段。
3. 保存实际动作、观测、reward与边界。
4. **按实际获得的reward重算RTG**，加入轨迹replay。
5. 用随机策略NLL与平均熵约束继续更新，再采新片段。

例如要求得到100，实际只得到20，新动作必须配实际20对应的RTG，不能仍标为“完成100”。官方核心目标是：

`L_policy = E_valid[-log pi(action | history, RTG)] - temperature * H(pi)`

另更新正的temperature，使平均策略熵满足目标下界。它没有PPO ratio、advantage或Q最大化。官方实现为tanh-Gaussian，密度包含Jacobian修正。

**动作适配决策：**V40 v2不是默认[-1,1]控制合同，不能复制tanh后悄悄缩窄髋/轮的控制域。原DT支持关闭action_tanh；DT-S先保留现有控制坐标的连续均值输出。ODT若保留原tanh分布，必须另定逐维映射并建立同映射基线；若使用现有Gaussian＋控制器裁剪，则属于明确的ODT-style分布适配，不能称官方算法逐项复现。raw sample与实际执行值都要保存。

### 5.4 为什么RTG必须另做设计

**command回答做什么，RTG回答希望表现多好。**要求高回报既不能指定站高/站低，也不能保证消除旧reward忽视的漂移。

对于固定终点的预算段，可以更新：`G_next = G - reward`。对于长度固定、不断向前滑动的窗口，真实关系是：

`G_next_window = G_window - reward_now + reward_at_future_window_end`

最后一项现在未知。因此不能把无限运行机器人上的滚动RTG，错误地实现为永远减reward。

可讨论的自洽协议：

- **有限预算段**：例如固定2–5秒的候选段长，输入预算及剩余时长。段尾重新设定预算，物理和动力学记忆继续；预算边界与机器人reset是不同事件，训练数据需覆盖跨边界context。若混合段长，剩余时间用统一秒尺度，不能只给各自归一化比例。
- **固定表现条件**：用预先定义的平均表现档位训练和推理，不每步扣预算。更适合持续控制，但属于RvS风格的条件化序列学习，不能只改原DT的推理输入、仍声称同一RTG语义。
- **延迟回报**：DT原论文/代码已有delayed-return模式，可在段末获得评价，中途保持条件；它仍需相应的训练标签与部署协议，不是随意忽略reward。

标准扣减RTG的部署需要可计算的reward。仿真中的真实vx、高度、接触力在实机上不一定同精度可得；仅移除critic不会消除这个需求。若部署只能估计reward，估计误差会累积进预算，必须训练/部署同定义并验证。

## 6. 高低站立和抗扰如何进入学习

这两件事由**命令条件、训练分布与评价目标**共同决定，而非某种网络名称保证。

- 高度连续采样0.28–0.32 m，同时覆盖端点与中间值；零速度、高低切换、起步/停止、纯转向、行驶换高分别有样本。
- 学会多个固定高度不等于学会换高。轨迹中需要真实命令变化，旧动作按旧命令评分，新观测带新命令。
- 不用固定关节角监督把腿锁在一套nominal姿势，否则与多高度平衡冲突。DT可以监督多种高度下的动作，但其标签必须来自相应命令下的合格行为。
- 抗扰要求轨迹/在线交互含偏离与恢复过程，且训练参数随机化真实作用于物理/控制计算。扩大历史不等于自动获得抗推数据。
- 站立要分别看高度误差、机身平移速率、累计位移、姿态与接触。低平均速度也不等于位置锁定；若要求严格回到同一世界位置，需要位置参考和可获得的状态估计。

已有独立五高度评估显示，旧策略能维持各档附近的姿态，但固定零速仍漂移；它是改进动机，不是“必须Transformer”的证据。尤其不能把旧策略高总reward直接当作DT的高质量标签。

## 7. 两条路线共享的工程边界

### 数据

至少记录：exact actor frame、当时command、raw/执行action、逐项reward、next observation、terminated/truncated、reset与日志切段、policy tick/version、合同/缩放版本、DR与扰动元数据。对齐为 `o_t → a_t → r_t,o_(t+1)`，数据保存不能只是指向环境下步会原地修改的tensor。

### 在线RL特有

- Isaac Lab内部auto-reset后返回的obs可能已是新回合；time-limit bootstrap必须取reset前的真实next critic state。
- delta中bootstrap使用`1-terminated`，GAE递推使用`1-(terminated or truncated)`；true terminal优先。
- PPO更新重建的每个完整history window必须与采样时完全一致，保留rollout前缀；可以打乱endpoint，不能打乱窗口内部时间或跨env拼窗。
- 一条长rollout一次causal forward不必等价于逐endpoint有限窗；多层局部attention也可能间接访问窗口外历史。
- old log-prob、old value、噪声实现与预处理在整轮更新中固定；首个optimizer step前应ratio≈1、KL≈0。functional attention的dropout也必须为0。

### 离线/ODT特有

- 学习使用achieved RTG，而非采集时未实现的desired RTG。
- causality mask、padding mask、loss/entropy有效位mask各自定义；当前action标签不能泄漏到其预测输入。
- 同样总回报可能来自更容易的高度/DR条件，而非更好的动作；按条件分层检查标签排序。
- 低MSE/NLL不是闭环稳定证据；ODT新增旧轨迹的replay也不能未经处理拿去做on-policy PPO。

### 部署

CT-S优先静态ONNX：`[1,L,25] → [1,6]`，外部维护预分配history；DT另带预算/command/action-history接口。critic和训练损失不导出。PyTorch融合attention无法导出时可等价分解为MatMul/Softmax，不改变mask。

100 Hz给整条actor链路10 ms，不是只给网络10 ms。后续需要目标硬件上的batch1端到端p95/p99和长时间jitter实测；本次未测速。关节反馈仍按200 Hz独立更新。

## 8. 有辨识力的后续比较设计

以下仅定义比较，不是训练启动计划：

| 比较 | 回答的问题 |
|---|---|
| 原MLP5 vs 同窗口MLP16/32 vs CT-S16/32 | 收益来自更长历史，还是attention结构？ |
| 同窗口GRU vs CT-S | 压缩记忆与直接attention何者更适合控制？ |
| command-only sequence BC vs DT | RTG条件是否真的有贡献？ |
| return-conditioned MLP vs DT | 相同标签是否必须用Transformer？ |
| DT vs ODT | 新探索数据及在线重标记能否改进？ |
| CT-S纯在线RL vs 教师辅助 | 教师是否提升样本效率，还是复制漂移？ |

比较时固定物理合同、可见信息、任务分布和评价场景；同时报告unique transitions、离线数据获取/教师成本、gradient updates、GPU-hours和部署时延。训练seed建议至少3个，保留失败seed；方法间不能只比“iter数”。

决策依据应是高低静止、换高、推后恢复、分布外动力学的物理指标，而不是attention可视化、回报均值或更低监督loss。若RTG没有贡献，去掉它；若较长history没有收益，保留较短窗口；若同预算简单网络同样好，不因“现代”保留额外复杂度。

## 9. 当前设计决策与下一份规格

已确定：**低维连续控制、显式高度命令、小型causal骨干、全部6动作一次输出、首版固定窗无cache、架构与训练后端解耦。**

推荐主候选CT-S16，context32作首项消融；DT/ODT保留为真正改变学习方式的候选，不将其误称为不能在线。若优先走DT，下一份规格先定**轨迹schema、可用于训练/实机的reward、预算horizon与动作分布映射**。若优先直接在线RL，则先定**terminal-next观测、history重建和loss convention**。

本轮只完成研究与设计，没有改变现有策略合同、实现新trainer或开始Transformer训练。原生GUI配置和实际第三轮训练安排按用户最新要求后置。

## 10. 来源与版本

- [DT原论文][dt]；[官方模型固定commit](https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/decision_transformer/models/decision_transformer.py)；[动作MSE训练](https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/decision_transformer/training/seq_trainer.py)。
- [ODT原论文][odt]；[官方实现](https://github.com/facebookresearch/online-dt/tree/c376fa113ba34bcd422da44598e8c2433c06a590)，重点`main.py`、`data.py`、`trainer.py`、`evaluation.py`与模型的SquashedNormal。
- [RvS论文](https://arxiv.org/abs/2112.10751)；[作者代码](https://github.com/scottemmons/rvs/tree/90c0ffb26a1a3dd617d4f51c726f135d06f44b00)，用于区分return-conditioned监督目标与Transformer的贡献。
- [Humanoid Transformer v1][digit]；[v2改名为Real-World Humanoid Locomotion with Reinforcement Learning](https://arxiv.org/abs/2303.03381v2)。teacher/student PPO＋KL依据论文；本轮未核实公开训练实现，不能声称其reset/cache细节已经代码验证。
- [Humanoid Next Token Prediction][hntp]，依据原论文和作者项目页；本轮未核实完整公开训练代码。
- [LocoTransformer][loco]；[官方PPO入口](https://github.com/Mehooz/vision4leg/blob/b8e33886143f94866320d1b362adc31b58d26b29/starter/ppo_locotransformer.py)。
- [GTrXL][gtrxl]；[ICML论文页](https://proceedings.mlr.press/v119/parisotto20a.html)，原算法V-MPO，不冒称PPO机器人结果。
- [PyTorch官方PPO教程](https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html)；[GAE](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.value.GAE.html)；[ClipPPOLoss](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html)。动态文档页面不是本项目已安装、已验证的版本契约。

[dt]: https://arxiv.org/abs/2106.01345
[odt]: https://arxiv.org/abs/2202.05607
[digit]: https://arxiv.org/abs/2303.03381v1
[hntp]: https://arxiv.org/abs/2402.19469
[loco]: https://arxiv.org/abs/2107.03996
[gtrxl]: https://arxiv.org/abs/1910.06764
