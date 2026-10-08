# source028 → source029：同 v26 合同的代码修复续训审查

日期：2026-10-09。本报告仅只读检查源码与已保存回执；没有 SSH、远端写入、创建原生场景或启动 PPO。新冻结提交、最新停止封包和一次入场来源更正事务仍须实际核验，不能由本报告替代。

## 可用入口与不能复用的入口

使用已有 `scripts/chassis_remote_job.py --resume-existing --source-revision-receipt`，回执 schema 为 `task-source-revision-v1`。该 schema 恰好用于**合同不变的代码修复**：旧/新 v26 campaign 合同 SHA 必须一致；新 command 只能替换 `run_task_campaign.py` 与 `--contract` 的冻结 source 前缀，run-dir、4096、research、device、runtime 等参数均保持原 job argv。

`queue_task_campaign_continuation.py` 固定 v15→v16；`queue_task_campaign_boundary_continuation.py` 固定 source023/v21→source028/v26 等指定目标，并已产生 v26 的一次性动作 intent。两者不是 source028→source029 的泛用队列。不得重用、清除或改写旧 `.task_campaign_boundary_v26.*intent.json` 来重新提交。029 使用独立部署 intent 与回执；若 intent 已存在而启动状态不明，先只读核对进程和当前 job，不能自动再 Popen。

## 最新状态与消费

已保存 `stop_wrong_rec028_002.json` 证实 source028 错派到 `recovery/attempt_010/train`，非 integrated speed。停止前该 worker 在 REC 原 41374 父状态上完成 19 个 actor 更新，19 个更新全部保留。root 提供待核验最新 final：

| 项 | 来源/目标值 |
|---|---|
| latest REC final | `train/recovery/attempt_010/train/model_final.pt` |
| SHA-256 | `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967` |
| 累计消费 / actor 消费 | 41393 / 41223 |
| REC actor 时钟 | 14188 |
| recovery 域累计消费 | 14502 = 14483 + 19 |
| 总余量 | 79607 = 121000 − 41393 |
| speed 历史消费 | 5093（全部继续计费） |
| 若无 joint speed own seal，首次 speed 最大新更新 | 14907 = 10 critic + 14897 actor |

最后一行来自 `integrated_lineage.remaining_budget()`，不是新增预算。v26 fresh speed 目标是 15000 actor，但旧 speed 分支占用了每域 20000 上限，因此本段在未调整预算的前提下以 `charged_cap` 结束；不能伪称完成了 fresh 15000。最新停止读回可能比停止前 progress 更新；务必按实际 final/progress/claim 重新算，而非盲填表。

## 必须先修正入场来源，不能只修派课

已有 v26 report/migration/claim 绑定 `authorized_REC_early_stop` 旧 41374；`recovery_entry_record()` 会复用它，而 `integrated_lineage._finished()` 又拒绝任何比 latest REC final 更旧的 origin。因此 source029 仅改 `run_domain` 跳过 legacy flat/recovery 仍不能合法进入 speed。

因果修复需提供**显式、一次、可认证的来源更正事务**：pin current v26 job/state/completion/claim/migration 的原 SHA、latest stopped final 与 full seal，归档旧来源与原入场快照，保留 19 个消费与完整 actor/critic/Adam/RNG，不篡改旧源码、合同、checkpoint 或原 origin 证据；更正当前 durable 入场引用及其一致性字段。新入场快照采用新路径并由新冻结源码认证。旧 `integrated_early_origin_v26` 原件保留，不能静默覆盖。事务要在无 live owner、原 wrapper final delivery ready、campaign account 原生 lock 独占时完成；失败/中断恢复由事务 receipt 决定，不重新执行模糊写入。

029 的冻结源码只做派课和来源更正认证；v26 合同/reward/采样/资产/控制/DR/预算字节一致。新 source archive 与 extracted source 独立目录，不修改 source028。

## 最小部署顺序：无额外原生建场

1. 等旧 supervisor/worker/wrapper 全部退出，`delivery.json.status=ready` 且 final/seal/contract/completion 文件 SHA 与归档清单一致；检查所有 evaluator/owner 均无。进程分类复用 `owned_processes()`，以真实 entry 判断，不能因 wrapper argv 内嵌 child 路径误计。
2. 读取实际 `job.json` 保存其完整字节和 SHA。核验其 source 为028/v26；全部已完成 progress 与 claim/state/consumption 重算一致、`rollback_refunds_consumed_updates=false`。必须保留 attempt010 的19更新，不能删除 attempt、递减 counters、退回41374父模型。
3. 冻结 source029，git archive 的 PAX `comment` pin 新40位 commit；分离新目录与新 archive，逐文件 SHA、路径安全和所有 contracts 字节一致检查。不要把 working tree 的动态内容混入冻结包。
4. 在新 source 的全新 CPU Python 进程中运行 `chassis_remote_job.validate_source_revision()` 与 `run_task_campaign --preflight-only`。后者只执行设计检查（当前30 role），不创建 AppLauncher/native scene。
5. 在 account 原生锁 `claim.with_suffix('.lock')` 下执行经过负测的来源更正事务。结束后调用 CPU `integrated_production.prepare_admission()`，使用**实际**最新 origin、全部 records 重算 allowance。还应模拟真实 `learning_source(speed)` / `train_command(speed)` 并调用 `authenticate_invocation()`；普通 `--preflight-only` 不会派课，单独成功不足以证明第一 worker 是 speed。此 CPU 演练应在只读或隔离临时状态完成，不另开 worker、不额外建场。
6. 独立029部署 intent pin 新 commit/archive/source revision receipt SHA 与前序 job/事务 SHA，然后通过已有 wrapper 唯一启动。保留 runtime 初始化，但不要再调用旧 queue 的 `validate_runtime()`。source028 已成功原生创建同 v26 场景，029 不改变 physics 合同；正常 speed worker 本身必须保留原生 startup/assertions，这不等于提前再创建一次预检场景。
7. 实读第一 worker 与首更新，下述必须全部一致；失败则封存，不自动改派课、回滚、退款或启动重复进程。

## 新 source-only receipt

新 `RUN/source_revision_029.json` 用独占创建，形状如下。实际值从最新原件读取，不复制旧028 migration receipt：

```json
{
  "schema": "task-source-revision-v1",
  "previous_job_sha256": "SHA256-of-current-source028-job.json",
  "command": ["exact-old-campaign-command-with-only-new-source-prefixes"],
  "source_commit": "new-source029-40hex-commit",
  "old_source_root": "absolute-source028-directory",
  "new_source_root": "absolute-distinct-source029-directory",
  "old_contract_sha256": "same-v26-campaign-sha",
  "new_contract_sha256": "same-v26-campaign-sha",
  "source_archive": "absolute-new-source029.tar.gz",
  "source_archive_sha256": "new-archive-sha"
}
```

既有 validator 对 code-only schema 自动沿用 `contracts/v6_task_campaign_v26.json`，不需要 `new_contract_relative_path` 或 `task-plan-revision-v27`。允许额外 provenance 说明，但它们不能替代上述认证字段。新 command 从当前 job argv 做两个有位置约束的 source-prefix 替换；无需人工重新拼装参数。

调用形状（由 root 在验证通过后执行，不是本报告已执行）：

```text
/home/kaiser/robot-rl-sim60/env/bin/python -B NEW_SOURCE/scripts/chassis_remote_job.py
  --run-root RUN --resume-existing
  --source-revision-receipt RUN/source_revision_029.json
  -- EXACT_AUTHENTICATED_NEW_CAMPAIGN_ARGV
```

外层 runtime shell 复用 `/bin/bash -c 'set -e; source "$1"; shift; exec "$@"'` 加 literal argv，runtime 文件保持原配置，不靠 SSH 客户端存活。wrapper 会独占保存 `job.previous_NNN.json` / `job.source_revision_NNN.json` 和新的 resume log；不手工覆盖旧 job 历史。

## 第一 worker / first actor 的验收（部署正确性，不是能力 gate）

- 唯一 wrapper→supervisor→worker 真实父子链，全部新029 source；没有重复训练或 evaluator。
- worker `--contract NEW_SOURCE/contracts/v6_task_campaign_speed_worker_v26.json`，run-dir 在 `train/speed/attempt_*/train`；绝无 REC attempt011。
- `--resume` pin 最新 REC41393 full seal，非旧 speed19837、旧 REC41374 或 ONNX；new selection domain=speed、mode=`rebase_from_authorized_REC`、fresh joint/domain actor=0、origin REC actor=14188。
- selection 文件/SHA 由 `make_admission()` 并 `authenticate_invocation()` 认证；allowance 含当前41393/41223与最新所有尝试，预计14907总更新。**这些是条件值，实际最终封存不同时按实际重算。**
- first full-state initial：9 actor/std、8 critic、17 Adam 参数与父完全恢复，RNG完整；新的 episode RNG 由现有接口处理。10critic warmup 计费，actor/std与任务actor时钟不变化；不能将前source028的REC19actor当作speed critic warmup。
- 从首次 actor 更新开始 actor参数与actor时钟一起变化；总消费连续（origin+critic+actor），不退款，学习谱系与消费时钟分别读实际seal。qualification仍 null，不把部署正确当作技能完成。

## 核验来源

- `scripts/chassis_remote_job.py`: `validate_source_revision`、`prepare_job_log`、`preserve_previous_job`、`finalize`。
- `scripts/queue_task_campaign_continuation.py`: pinned constructor、`owned_processes`、`validate_staged_source`、`account_ownership`。
- `scripts/queue_task_campaign_boundary_continuation.py`: pinned `OLD_COMMIT` / `TARGET_REVISIONS`、`NativeAdapter.preflight/validate_runtime`、一次动作 intent。
- `scripts/run_task_campaign.py`: `open_integrated_account`、`learning_source/train_command`、`run_domain/run_integrated_domain`、`main --preflight-only`。
- `src/wheeled_tasks/chassis/integrated_production.py`: `recovery_entry_record/admission`、`prepare_admission`、`select_source`、`authenticate_invocation`。
- `src/wheeled_tasks/chassis/integrated_lineage.py`: `_finished` latest REC检查、`select_learning_source`、`remaining_budget`。
- `reports/v6_full_training_audit_20261009/stop_wrong_rec028_002.json`：实读错误REC route与19actor消费。

本报告未认证029尚未生成的冻结 commit、来源更正事务、新receipt、remote staging、首speed worker或训练效果。
