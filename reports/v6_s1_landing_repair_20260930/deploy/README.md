# S1 被动落地奖励修复部署回执

**2026-10-01最新状态：** 下述启动记录为历史。PPO已于9月30日19:52完成，20:13双seed评测结束，正常封存待审；当前无该任务训练／评测。消费778，S1余222；本轮27／28项通过，对照A为33／33，未晋级。普通行为落地奖励确实全部归零，整体能力尚未恢复。详见[结果与判决](../result_review/README.md)。

- 启动：2026-09-30 19:18（UTC+8）。
- 冻结源码：`62c451d09c686c6e6bc322710532459884e3e137`，无 overlay。
- 源码归档 SHA-256：`8fddf4d99620375744952794cc405bd0560d4d809d02cb58f86e716c946541d6`。
- Kaiser tmux：`v6-s1-continue-20260930T111823Z-58c545`。
- 运行根：`/home/kaiser/robot-rl-sim60/experiments/v6-s1-continue-20260930T111823Z-58c545`。
- 新合同：`contracts/v6_s1_landing_repair_v1.json`；worker SHA `4a7c51ecfbd8d1be3672b27d245affbb858a57a63607d3bf3c3f81dc7c5f0ca1`。
- TensorBoard：`http://127.0.0.1:6006/#scalars`，实验别名 `v6-s1-landing-repair-62c451d`。
- 20k，50Hz 策略／1kHz 物理与 PD，35D/81D 网络不变；fixed LR1e-4。
- 从 A 完整恢复学习，预算从最新封存738开始，40次actor+critic到778；actor消费718→758，actor谱系368→408。原保护39、retry和总预算不重置。
- 40次后自动跑原45项双seed评测，符合目标／保护后才补全量出口审计，封存待判，不自动晋级。
- CPU 257项检查通过、远端五种入口静态预检通过；实际实例启用、优化器恢复、PPO权重变化以 `runtime_verification.json` 为准。

修复仅涉及被动落地奖励的任务范围、有限窗口与零中心成本，以及配套日志；网络、采样、执行器／物理和验收保持原合同。普通静站恢复成功不再失去额外正基线，真正落地任务保留缓冲目标和冲击成本。完整范围见[计划](../../../docs/V6_S1_LANDING_REPAIR_20260930.md)。

`launch.json` 只证明启动；场景构建不能当作 PPO 已经更新。

## 19:26（UTC+8）实际学习核验

已新增3次actor+critic更新，累计消费741、actor学习谱系371；非仅启动或只评测。739封存检查点相对A的8个MLP权重／偏置张量及探索std均发生变化。actor、critic、Adam恢复逐项exact，实际LR保持1e-4。20k运行时预检通过，实际有限落地实例为enabled=true、0.6s窗口、0.02s奖励时钟。

TensorBoard已写1271个scalar tag，包括256个新增相位／窗口／速度诊断tag；静站与230mm组的landing paid mask及两项落地奖励前三次均为0，修复已实际生效。此事实只证明不再错付奖励，不代表驻点精度已通过。GPU实读93%，显存13088MiB/24564MiB。原始证据见[runtime_verification.json](runtime_verification.json)。

尚无本轮新增评测成绩；40次结束后的固定评测将判定是否改善及是否损失保护能力。
