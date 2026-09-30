# S1 消费1000出口评测（进行中）

2026-10-01 05:03（UTC+8）首seed290929的45项评测完成，通过20/45、S1目标4/16：stand_305mm、stand_continuous、backward_05、rotate_1。05:09实读确认seed290930正在运行；322项双seed全量审计尚未完成。本记录不是最终晋级结论。

首seed中305mm静站最大位移2.77cm，高度MAE4.65mm；连续高度静站也恢复通过。但前进命令0.5m/s实际0.325m/s，速度MAE17.53cm/s；驻停最坏10.69cm/s仍高于3cm/s。正转yaw0.1393rad/s通过，反转0.4444rad/s未过。相对940有部分静站恢复，同时运动域仍退步，不能把通过数17→20视为全面收敛。

原始报告SHA-256：`e09151148f9fca1657a061fed85ff84340d27207a113806612786074b4f08282`。评测使用model_final SHA `7dd3ed3b3c959ca31b6bb7f02488d54a7a908a9591d44b059f7c0997b6b63160`，与封存一致；settings、startup和source manifest与940相同，unfinished=0。

消费固定1000、actor谱系590、总学习谱系600；没有再启动PPO、退款或晋级，原A与全部检查点保留。等待确认seed及全量结果，按原目标和39项保护集判定。

证据：[首seed原始报告](evaluation_290929/evaluation.json)、[精简摘要](evaluation_290929/summary.json)、[PPO完成回执](../deploy/ppo_completion_verification.json)。
