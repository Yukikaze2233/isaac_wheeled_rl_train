# source029 跨源码目录准入审查

范围：只读审查，不执行 SSH、仿真、PPO、账本更正或部署。原 `deploy_source029.py` 字节保持，SHA-256 为 `3864f047ffea30989cce30db1bd697c20d5f217d70286d09c77c12198c05b963`。

## 已证实的缺陷

冻结 `contracts/v6_integrated_lineage_v26.json` 的 `source_root` 是 `source_integrated_v26_20261008`。`integrated_lineage.load_plan` 要求它与实际执行源码目录完全相同；source029 的独立目录因此无法通过 `binding_for`。此前三个 actual-admission CPU 测试使用同一个 fixture source，不能证明跨目录部署成立。

修复不能重写冻结 manifest 或合同。最小方案是在 `binding_for` 先认证同合同代码修订，再按原 immutable source_root 建立 `FrozenPlan`，保留原 `load_plan` 的严格绑定。旧根下的静态合同/学习谱系与新执行根中的字节必须相同；生产 worker 仍执行新根的 dispatch/runtime 源码，manifest、manifest SHA 和学习谱系身份保持原值。

## 最小授权边界

- 仅接受同 v26、同 run_root、`task-source-revision-v1`；回执 old_source_root 必须等于 manifest.source_root，new_source_root 必须等于实际执行根，二者独立且都归属于该 experiment。
- 复用 `chassis_remote_job.validate_source_revision` 认证原 job、只变 runner/contract 前缀的 argv、归档 commit/SHA 和解包文件。原 source028 应仍为真实 `335542ddc753b84b2778609b92803064682198a9`。
- 全部 `contracts/` 与机械资产、控制数学、物理/执行器/观测/奖励 runtime 字节不变；代码变化限于最终真实 diff 核验的 dispatch/origin/alias 修复文件闭集（新增 alias 后文件数须重新核对，不能沿用原六文件）。仅比较 campaign 合同 SHA 不足以排除额外 `env.py` 改动。
- 运行时不修改账本、合同、回执或 old source；保留最新 REC 41393 / actor 消费 41223 / REC 14188 / 谱系 27646，以及全部 19 次误派发 actor 消费。别名授权不能成为退款、重置课程或新模型选择入口。

## wrapper 启动前后都要可认证

`validate_source_revision` 当前要求回执 `previous_job_sha256 == SHA(job.json)`。原生 wrapper 在启动后替换 `job.json`，同时保存 `job.previous_<N>.json` 和 `job.source_revision_<N>.json`。因此别名认证不能只接受启动前的 live job，也不能在每次 `load_plan` 要求 noowners：那会拒绝实际 worker。

启动前使用原 job。启动后通过 live job 的 `previous_job` 与 `source_revision_receipt` 字段选择 wrapper 留存且 SHA 匹配的 previous-job 及逐字节 source-revision 快照，另外核对 live job 的新源码命令与回执一致。路径必须局限于该 experiment，拒绝任意 previous-job 路径或不同回执。既有 validator 可增加默认仍为 `root/job.json` 的显式 previous-job 参数，复用归档/命令/commit 全认证，无需修改原 job 字节或猜日志序号。

noowners 与 account lock 属于 staging/correction/launch 的一次性动作门禁，不属于运行期只读 alias。

## 保持原 helper 顺序

1. `stage`：安全解包、同合同回执、helper SHA 固定；不绑定学习状态。
2. `preflight`：真实新目录 `run_task_campaign --preflight-only`，随后 origin `--check-only`。后者是首次真正调用跨目录 `binding_for` 的关键拒绝点；此前不得 apply。
3. `correct`：再次独立锁/无 owner 验证，一次性 origin 事务。全状态 loader、冻结合同及别名认证都必须先于账本发布。
4. `launch`：再次独立锁/无 owner、真实 prepare/make/authenticate；保存 admission 和 intent 后仅一次原生 wrapper。实际首 worker 必须 speed，保留 10 次计费 critic-only，再进入 actor。

不另起模拟场景，不用旧 source028 intent，不覆写任何旧日志/回执。原生 worker 自己仍完成 native startup/readback。

## 必要小规模 CPU 覆盖

新增 fixture 必须让 old/new source 实际是不同绝对目录，保留 old-prefix manifest 字节。至少验证：未认证别名拒绝；真实 same-contract revision 接受；替换 wrapper job 后仍接受；额外合同/机械/runtime 改动拒绝；随后 `correct_origin` 与实际 helper `launch_evidence` 接受最新完整状态。恢复不足、旧 parent、已有 own-joint seal 或 owner 仍按原测试拒绝。没有新 GPU/PPO 或放宽判据。

归档大小约 210 MB，完整 archive/tree 认证应位于准入与进程启动边界，不应加入物理/策略子步。正确性优先，不为本次修复新建复杂缓存或部署框架。

## 实现独立核读

已只读核读 `serial-dispatch-source028-20261009` worktree 的 `_source029_data_root` 与 wrapper optional previous-job 参数：仅 v26/匹配旧新根的原回执；真实 job 链字段配对；原归档认证；所有非 owned 文件字节/mode 同；receipt/job/archive/旧新文件 stat 改变使准入缓存失效。`binding_for` 返回按原根读取的 `FrozenPlan`，旧 `load_plan` 不放宽。

最终候选闭集为八文件：`scripts/run_task_campaign.py`、`scripts/chassis_remote_job.py`、`scripts/repair_known_integrated_origin.py`、`src/wheeled_tasks/chassis/integrated_lineage.py`、`src/wheeled_tasks/chassis/integrated_production.py`、`tests/test_integrated_early_production.py`、`tests/test_known_integrated_origin_repair.py` 和本次 source029 设计文档。冻结提交前仍须用真实 git diff 复核该闭集；不以候选文件数代替 commit/tree 认证。

新增真实双目录 lifecycle 测试源码已经核读，覆盖原 manifest 不变、更正前旧 origin 拒绝、latest full-state 更正、wrapper 使用 resume024 留存链、open_account 幂等、new-root train command/实际 worker admission、speed first/四域 dispatch，以及缓存后错误 prior job 与合同更改拒绝。本独立审查没有重复执行长测试；最终测试退出与冻结 commit 由实施者回执提供。
