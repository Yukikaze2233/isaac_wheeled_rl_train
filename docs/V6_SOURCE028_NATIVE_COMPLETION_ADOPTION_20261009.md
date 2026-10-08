# source028 已完成原生启动的显式采纳

source028 已授权从封存 REC14169 的同一 actor 继续速度、特殊地形、跳跃和跳阶，保留复习与原预算。生产源码仍为 `335542ddc753b84b2778609b92803064682198a9`，合同、PD、随机化和学习状态没有因本次启动修复改变。

原 `boundary_integrated_028.native` 在 Simulator 前因缺少显式 `--learning-rate-scale 1.0` 拒绝，exit=1、零 PPO。一次命名 CLI 修复保留这份证据，并于 UTC 2026-10-08 19:53:55–19:55:45 完成 4096 环境、一策略步/四物理步；exit=0，真实 `scene/completion.json` 为 `scene_validated`、successful/attempted=0。然而外部适配器查找不存在的 `scene/report.json`，因此保存第二份 failed 回执并拒绝启动 PPO。

实际 completion SHA 为 `09102d256f400212503b23e37bbd632585038d52a1b1c8216344ca4c8ba0d2f1`，startup SHA 为 `d305f7d360fbdfcea7f2bb66dd572a934137335703e760408a01769d5677f4ab`，selection SHA 为 `a50715201c99d4de9eb41505525c214a03536cc9c931e1479ae7c879e9ea2978`。独立读取确认原 native 全部 passed 谓词成立，真实 selection 和账本仍与原准入一致。

`scripts/adopt_known_native_cli_completion.py` 仅处理这一已证明的文件名错误：硬 pin 原五份 CLI 失败证据和七份 CLI 修复/实际完成证据；检查无 owner、无原 launch 意图/回执/日志；取得原共享锁，认证原 source/archive/account 和原准入。它发布新的 `boundary_integrated_028_completion_adoption002.intent.json`、`.native.receipt.json`、`.receipt.json`，native 采纳回执绑定真正的 completion，原两份失败回执逐字节保留。随后复用原 `q.step()` 的准入和串行 launch，不再调用 Simulator，不重跑 native，不删除或退款消费。

外部打包的 `code_manifest.json` schema 为 `frozen-known-native-completion-adoption-code-v1`，严格包含四文件 SHA：adoption 脚本、原 CLI 修复脚本和两个原 queue 脚本。原 queue 的两个读回位置修正为 `scene/completion.json`；生产冻结 source028 不热改。任一证据改变、已有采纳产物或已有 launch 都拒绝；没有自动 retry。账户锁若仍被占用，本次返回 waiting 并封存结果，不能重跑同名采纳。

CPU 测试以显式 synthetic 文件替换硬 pin，覆盖零 native 调用、原证据/账本保持、精确准入、七个字节变更、passed 谓词的反例、代码包完整性和重复启动拒绝。实际采纳及 PPO 是否启动，以远端新回执为准；后续只读核验应检查精确恢复、10 次计费 critic warmup、首个 actor seal 与唯一 owner。容量验证不授功能或实车资格，qualification 保持 null。
