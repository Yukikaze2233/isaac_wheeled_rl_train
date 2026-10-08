# source029 串行入口与真实 REC 来源更正

部署源码必须直接以实际 source028 提交 `335542ddc753b84b2778609b92803064682198a9` 为父树，只叠加本说明、派课入口、lineage认证、更正工具、两份测试、原 wrapper 认证参数和 production 来源别名这八个文件。之前基于较新本机主树的 `55ead9e` 不作为部署树；它被预检拒绝后的暂存/归档证据保留，未更正账本或启动训练。所有合同和八文件之外的代码、资产必须与真实 source028 逐字一致。

v26 lineage manifest 保留原不可变 `source_root`，不重签合同或放宽 `load_plan`。新 runtime 目录仅通过固定 `source_revision_029.json` 的 code-only 回执认证：旧源须是 manifest 原根，新根须是实际运行根；原 wrapper 验证同命令、合同 SHA、真实 commit/归档 SHA/PAX 和提取字节，别名另外限定仅八个修复文件可以变化，其余代码、合同、模型逐字及 mode 相同。`FrozenPlan` 继续读取原数据根，实际代码执行新目录。全归档验证按进程缓存，但回执、原 job、归档及旧新文件 stat 变化使缓存失效；该入口不在物理或策略热循环。

wrapper 启动前验证旧 `job.json`；启动后根据实际 `previous_job` / `source_revision_receipt` 字段配对验证留存原 job SHA 和回执字节，不把源码029误认为 wrapper 的 resume 索引。正常重入仍认证该原发布链，活跃 owner 检查仅在部署/更正事务，运行中的 worker 准入不要求无人运行。CPU 测试使用真实不同 old/new 目录和归档，覆盖更正前拒绝过时来源、更正后10critic准入、wrapper改写job后 `open_account` 幂等、同最新父 speed 来源、实际 worker admission 和四域串行 dispatch，错误保存job及更改合同均拒绝。

source028 的原生检查已经完成，采纳适配器也成功提交正式 owner；但生产 `run_task_campaign` 随后错误启动旧 REC `attempt_010`。原因是完整六域 plan 的循环仍访问 flat/REC，而 integrated 分支仅接管 speed/terrain/jump/stepjump。`authorized_REC_early_stop` 保存了入场证据，却没有阻止旧 REC 的 15000 actor 端点补跑。

本次最小入口更正在 `run_domain`：integrated v24/v25/v26 只派发 canonical `integrated_lineage.CHAIN`。flat/REC 保留原域状态、原资格和全部消费，作为历史与认证来源，不制造达标或预算完成。旧非 integrated 课程派发保持。恢复后的同 actor 串行为 speed → terrain → jump → stepjump，复习与冻结 v26 配方不变。

错误 REC 已真实多做 19 次 actor 更新，故不能恢复原 REC14169/消费41374。实际停止后的完整 PT 与 seal 为消费41393、actor消费41223、REC14188、stage27556、学习谱系27646，SHA `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967`。completion 顶层旧谱系是父值；以完整 PT/sidecar 为准。

`scripts/repair_known_integrated_origin.py` 只处理这一已知十九更新来源更正。`--check-only` 与 `--apply` 二选一；两者均取原 account lock，拒绝活跃 owner、任何已选 joint 来源/已有 joint own seal、退款、非 stopped campaign、错误父 SHA/时钟及更正目录已存在。它用原生 CPU loader 严格核验完整 actor、critic、Adam、RNG 与 seal，零仿真、零 PPO、零信号、零启动。

`--apply` 在新目录 `train/integrated_early_origin_v26_dispatch_correction029/` 独占保存 `before_account/state/completion/consumption/migration.json` 的原字节、新 stopped campaign snapshots 和 `intent.json`。新 REC record 通过 `early_stop_origin_correction={file,sha256}` 绑定该唯一 intent；`integrated_lineage._finished` 只允许该固定目录，并验证旧认证 entry、原不可变 origin、停止状态、无 joint 入场与完整十九更新时钟，原其它 early-entry 规则不变。原 `integrated_early_origin_v26` 和失败回执不改。

所有完整状态与 fresh speed 准入校验先完成，再逐文件持久化新动态 origin、原账本身份及迁移回执；最终写 `receipt.json`，schema `known-integrated-dispatch-origin-correction029-result-v1`、status `corrected`，包含新父、原文件与最终文件 SHA、全部原消费和保留十九更新、`qualification=null`。原迁移回执的字节在 `before_migration.json` 保留；本次仅更正动态来源，合同、奖励、随机化、PD、物理与预算均不改。各文件持久写入不等于跨文件原子提交；若中途失败留下 named intent，拒绝自动重试，需审核留存证据后另行处理。

CPU 回归覆盖真实六行 dispatch、旧课程作用域、只读无写、最新完整来源与重启幂等、非法父/已有 joint/退款/活跃 owner/重复更正、intent SHA/schema 与 snapshot 重定向拒绝。实际部署及首 10 critic/首 actor 恢复另由真实远端回执证明；不能把此前 REC14169 的一策略步原生预检描述为 REC14188 恢复验证。

验证记录：候选生命周期24项 CPU通过（782.39s，canonical native fixture 构建成本，非 PPO 更新耗时）；最后对应改动后的真实来源重入与 joint own 拒绝两项通过（108.09s），既有原生 lineage 43项通过（4.08s）。独立 SCUT deployment helper 的真实 CPU更正→准入→派发三项通过（135.39s）。测试设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 仅作用于这些 CPU验证进程，生产设置不改。

真实 source028 基底上，lineage 与核心来源/派课定向验证46项通过（153.09s）；随后上述真实跨 prefix 单个全生命周期测试通过（99.81s），未重跑旧长套。首次该测试因原生 fixture 使用 `run` 而 wrapper 要求 `train` 的 resolved-path 规则被拒绝，补 fixture 同目录别名后通过；没有因此修改生产 guard、合同或账本。
