# REC41393 本机 ONNX 独立审查

结论：真实 REC41393 父 checkpoint 与本机 ONNX 身份一致，25 个既有 ABI 输入上的原生均值 actor、导出 wrapper 和 ONNX CPU 输出一致。未做新物理回放或能力验收，`qualification=null`、`hardware_deployment_ready=false` 保持；不能称最强或已具备完整高速/自起/跳跃/台阶能力。

模型目录：`models/v6_recovery_candidate_41393_20261009/`。`policy.onnx` 实际 SHA 为 `4ceaa168f303f094cb6d106ab7de99ea20fe63f7220a7b7812a6fb94e3bb3dd8`，原父 `model.pt` 为 `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967`。它与 REC41374 是两个不同模型，不能借用旧模型成绩。

独立实际重算了原五输入和本机副本 SHA、官方 18 文件 inventory、交付 20 文件 inventory、两份 checksum 列表（19/21 项，含对应 manifest），以及合同、资产 manifest、轮侧 control manifest、interface 的身份绑定。两个官方 exporter 文件从实际 Git `e5bd9c1367b875d7f229b9a1930647013d023372` 和 `bd4bdd052a07cab205e764f50513bcc752bec84e` 读出，逐字节相同且与本机源码 SHA 一致。

消费/学习时钟独立核对：累计消费 **41393**，actor 消费 **41223**，REC actor **14188**，stage actor **27556**，学习谱系 **27646**，critic warmup 余 0。官方顶层 `consumed_updates=41223` 指 actor 消费，不能把它写成累计消费。ABI 保持 35→6、critic 81、单帧、50 Hz policy/200 Hz 物理 PD、160/2.5、64/32、4096。

只读调用官方 `authenticate`、`load_actor`，以 CPU 安全加载原父和 stock `MLPModel`；没有导出新模型或写入原包。实际独立数值结果为：

| 25 个既有 ABI 输入 | 最大绝对误差 |
| --- | ---: |
| 原生均值 actor → wrapper | 0 |
| 原生均值 actor → ONNX CPU | 4.917383193969727e−7 |
| ONNX CPU → 包内既有 ABI 输出 | 0 |

`atol=rtol=1e−5` 全部满足。包内 ABI `actions` 本身是官方 exporter 存下的 **ONNX 输出**，因此最后一行只证明回读一致；第二行才是独立重算的原生 actor 数值对应证据。既有 full-state receipt 的 SHA 与 9 actor/8 critic/17 Adam 身份已认证，本次没有重新执行完整 Adam/RNG 恢复事务。

独立回执：`rec41393_cpu_export_independent_review001.json`；实际 CPU 日志：`rec41393_cpu_export_independent_review001.log`。执行中 GPU/Isaac/SSH/PPO/远端写入均为 0。数值导出不证明站稳、制动、高速、完整倒地链、普通上阶或跳阶实测通过。
