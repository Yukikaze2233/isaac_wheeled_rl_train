# 恢复课程过程诊断候选

这是默认关闭、尚未部署的日志候选。只复用现有 LearningAudit 和原生 pre-reset 诊断，不修改合同、奖励、FSM、稳定门、模型或 PPO 样本归属。候选环境配置显式设置 `recovery_phase_audit: true` 才分配缓冲和输出指标；当前冻结生产合同没有此开关。若以后部署，应冻结诊断配置及源修订，不能热改当前 worker。

仅支持原生 `auto_reset: true`；启用诊断而关闭 auto-reset 会在构造时拒绝。否则一个终止快照可能被重复观察，无法用下一帧认定真实新回合。

## 已有结束分母

现有 `RecoveryOutcomes/{prefix_stand,prefix_release,original_chain}/` 已记录 `ended`、`successes`、`failures`、`collection_timeouts`、`has_completed_episode`，并在 `ended>0` 时输出 `cumulative_success_fraction`。计数由原生 producer 在 auto-reset 前记录，使用实际 `level_ids`，与原行为组名称无关；一个真实回合只计一次。它们是当前 worker 的累计计数，两个端点相减才是窗口分母。此前“纯 L2 分母缺失”应区分日志已实现与本次尚未提取。

L0=`prefix_stand`，L1=`prefix_release`，L2=`original_chain`。L2 保留各原组的初始 profile，包含直立、45°、放松和已在目标平台的初态，**不是四种真实倒地姿态专属成功率**。四倒地能力仍应单独用原生固定 case 评测；不能用 L2 汇总授予完整自起能力。本候选不增加 profile 级标签或重造完成判定。

## 新增 33 个全局标量

`RecoveryPhaseAudit/{prefix_stand,prefix_release,original_chain}/` 每级 11 项；不按行为组复制。除 `policy_owned_fraction` 外均为本次 rollout 的计数。事件锁存跨 rollout 保留，真实回合结束后再清除，因此同一回合的首次事件不会在后续 rollout 重复支付计数。

| 事件后缀 | 原生依据 | 解释 |
|---|---|---|
| `script_complete` | `recovery_ever_blended` | 原生 FSM 至少一次进入 BLEND，即其实际分支已到交接边界；不保证执行所有其他分支的完整轨迹 |
| `pure_rl_started` | `recovery_handover_count > 0` | 至少一次真实进入 pure RL；即使策略区间内含 blend 而不能学习，也记录发生过交接 |
| `stable_hold` | `recovery_stable_peak_seconds >= FSM stable_seconds` | 原生物理稳定门连续满足所需时长；不替代任务后续跟踪成功判定 |
| `motion_released` | `recovery_motion_released` | producer 实际放行后续运动 |

每个事件输出 `first_{事件}`（本 rollout 首次发生的回合数）和 `ended_with_{事件}`（本 rollout 结束且曾达到该事件的回合数）。后者的分母使用同一 rollout 两端已有 `RecoveryOutcomes/.../ended` 的差值，不能除以 `first_{事件}`，也不能把未结束的回合当作失败。没有结束回合的窗口不定义完成率；累计结束计数必须限于同一 worker，不能跨重启直接相减。

`observed_transitions` 是该实际课级的 env×policy-step 数；`policy_owned_transitions` 使用 env.step 最终 `policy_learning_mask`，已经包含恢复脚本、混合区间及非标准初态的联合 mask。`policy_owned_fraction` 为两者之比；零曝光时的 0 只表示没有样本。脚本→RL 混合策略区间可以“出现首次 pure RL”却不进入 PPO；actor 导致的末步失能则仍可有 actor 所有权，不能用当前 pure-RL 相位替代最终 mask。这是控制归属曝光，不等同于 minibatch 经原生 PPO 处理后的有效梯度样本数。

所有新归约使用固定大小 tensor，无策略/物理循环内 `.item()`、`.cpu()`、`.tolist()` 或可变大小布尔索引；标量并入既有 drain 的一次 CPU 打包。默认关闭时没有新增 tensor 缓冲、观察调用或 TensorBoard 标签。必要 CPU 检查覆盖跨 rollout 首次事件、终止归属、实际课级/结束分母、不参与恢复的环境排除、默认关闭及输入/RNG 不变；没有 GPU 性能或行为收益验证，不声明已接入正式训练。
