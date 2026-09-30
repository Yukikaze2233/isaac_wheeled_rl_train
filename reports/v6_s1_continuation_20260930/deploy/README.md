# S1 从 A 正式接续部署回执

**2026-09-30 18:04最新核验：** 40次PPO于16:58完成，双seed评测17:19结束，正常封存待复核。消费738、S1剩262；通过数由A的33/45降为29/45，S1由7/16降为4/16，未晋级。当前无PPO/评测进程。以下16:32记录是启动时历史观察；完整结果见[接续结果](../result_review/README.md)。

- 冻结源码：`8c42db27253ba9b590735642424cf323b5a6c705`，无 overlay。
- 源码归档 SHA-256：`8e07242b761fea8ca440869617d00fe4468e9e8956ae0a87b15b88373c821713`。
- 启动：2026-09-30 16:24（UTC+8）。
- Kaiser tmux：`v6-s1-continue-20260930T082414Z-6699c9`。
- 运行根：`/home/kaiser/robot-rl-sim60/experiments/v6-s1-continue-20260930T082414Z-6699c9`。
- 合同：`contracts/v6_s1_continuation_v1.json`；worker SHA `fb0f7c40ef9a6b79dba4860e37d881eb9a57deef7f8d44948ef2df69bdbcebcd`。
- TensorBoard：`http://127.0.0.1:6006/#scalars`，别名 `v6-s1-continue-8c42db2`。
- 20k 环境，50Hz 策略/1kHz 物理与 PD，网络不变。
- 从 A 完整恢复；fixed LR 1e-4；40 次 actor+critic，无重复 critic 预热。消费 698→738，actor 消费 678→718；actor 学习谱系 368→408。后续继承本轮检查点即可继承这些学习，不合并 B 权重。
- 原保护与 A 双 seed 已通过项并集 39 项，原 retry 历史保留，不退款。40 次后双 seed 45 项复核，满足目标与保护才补全量出口审计；不会自动晋级或自动花完余下 262 次。

`launch.json` 只证明 tmux 启动。真实 actor 更新、学习状态恢复和当前优化进度另见 `runtime_verification.json`；不得将场景构建或只评测称为 PPO 已学习。

## 16:32（UTC+8）实际学习核验

累计消费700，本轮新增2次actor/critic更新；actor学习谱系370。恢复检查确认actor、critic和Adam逐项精确恢复，学习率保持1e-4，无critic预热。已封存首个699检查点：actor的8个权重/偏置张量全部发生变化，最大绝对变化0.004068，直接证明策略在学习。原生20k场景配置和执行器属性回读通过，GPU92%、显存13088MiB，TensorBoard新增停车分相位指标已写入。尚未运行本轮结束评测，不报告新增技能通过。

关节响应边界：本次继续使用PD160/2.5、完整重力/气簧/闭链物理；四腿主动轴的新增armature/friction/damping/lag仍为零，真实bag拟合候选未采用。原因见 `docs/V6_RESPONSE_LEG_AUDIT_20260929.md` 与 `docs/V6_LEG_RESPONSE_BLOCKERS_20260929.md`，不能把轮侧接入描述成六轴辨识完成。
