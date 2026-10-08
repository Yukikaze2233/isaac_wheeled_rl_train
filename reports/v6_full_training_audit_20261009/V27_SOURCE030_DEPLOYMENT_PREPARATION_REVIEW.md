# Source030 部署准备：独立只读审查

审查日期：2026-10-09。对象为现有 `deploy_source029.py`、`run_readonly.py`，以及组合提交 `f627f678fae98d374c049d76a444430dda70306a` 的 wrapper/controller/准入接口。**本轮没有执行 CPU、SSH、远端写入、信号或启动。** 正式 12 个 v27 静态合同仍应由 root 生成并另行冻结，不能把当前代码提交误作最终部署归档。

结论：可以复用已有归档提取、原生 wrapper、账户锁、准入和 controller。需要将 source029 专用部署 helper 最小改写为新的 source030 helper，保留原 helper 和回执；不能仅改版本号直接运行。无需新 scheduler。

## 可以提前做什么

本机完成静态合同生成、验证、冻结及 `git archive`，归档必须包含最终合同和正确 PAX commit。新 lineage 的 `source_root` 应预先固定为实际计划提取路径，旧 v26 合同必须逐字保留；允许新增的合同文件集是具名 v27 的 12 个产物，不再要求旧 helper 的 `old_contracts == new_contracts`。

Kaiser 可在旧 speed 持续运行时接收归档，并安全提取到新的独立 source030 目录；只写新 staging 清单，记录 commit、归档 SHA、提取文件校验和 helper SHA。已有目录应检查后退出，不覆盖重提取。这一步不需要持有训练账户锁，不改变 source029 源码、job、claim、state 或训练过程。

**不可提前生成**动态 `source_revision_030.json`、实际 completed-speed 证书、parent SHA、`previous_job_sha256` 或 worker admission。静态 source030 不是 source029 alias；真实 speed 父值必须来自未来正常完成的 full seal。

## 正常边界的最小顺序

1. **静态 preflight。** 使用新冻结源码的原 `run_task_campaign.py --preflight-only` 验证静态 plan/design/source 清单，不伪造 REC、speed 父或运行中账户快照。不调用 source029 的 origin correction。
2. **未来 controller。** 从新冻结树运行现有 `hold_completed_speed_boundary.py`，`--source-root` 传实际 source029 目录，`--run-root` 传实验的 **`train` 子目录**。默认不带 `--apply` 只是只读准入；未来明确应用才使用 `--apply`。它负责在实际剩余 20–100 actor 时暂停 supervisor、让 worker 自然完成、原生 TERM/CONT 计账并等旧 wrapper 退出，最后零 PPO 补闭合正常 speed record。
3. **确认旧 wrapper 封存成功。** controller 只等待 wrapper 退出，不证明 `finalize` 成功。必须另核对 `delivery.status=ready`、无 `delivery.tar.gz.tmp`、`archiver_status=finished` 且 PID 对应旧 supervisor，并逐项核对 delivery 所列实际 speed final PT、checkpoint/contract sidecar 和 worker completion 的 SHA。不能用前一轮遗留的 ready 代替。
4. **生成动态计划修订回执。** 此时再读取稳定的实际 `job.json` 和正常 speed 父。新回执 schema 为 `task-plan-revision-v27`，旧 source root 从当前 job 的真实 source029 runner/contract 前缀解析，新 contract 为 `contracts/v6_task_campaign_v27.json`，必须有 `new_contract_relative_path`。`previous_job_sha256` 在此刻计算；命令只替换原 runner 和 contract 前缀，其它参数保持，由现有 `validate_source_revision` 验证。不要把旧 manifest 内的原始 source root 冒作当前 source029 runner 前缀。
5. **在原锁下迁移与准入。** 先确认成功 controller receipt，获取同一个 `train/native_speed_boundary_controller.lock`，再获取原账户 flock，确认无活跃 wrapper/supervisor/worker/evaluator。由新源码真实 `TaskCampaign.open_account()` 走具名 completed-speed 迁移，生成 before/entry 证书；再 `learning_source(terrain)`、`make_admission`、`authenticate_receipt`。记录真实父、消费、学习/组件时钟、terrain actor=0、10 次付费 critic 和实际 allowance，不启动 worker。启动前准入 snapshot 放在实际 `train/revisions/...` 下，不伪造 `attempt_*`。
6. **一次原生 launch。** 复用旧 helper 的 exclusive launch intent、原 native wrapper `--resume-existing --source-revision-receipt` 与原 Bash runtime 注入方式。释放账户锁后才启动 wrapper，intent 防止释放与启动之间重复提交。新 supervisor 会再次认证 entry、跳过旧 speed、创建自己的真实 terrain attempt 与 worker admission；不要由部署 helper 手写 worker 命令或 attempt 编号。
7. **只读首 actor 核验。** 新 worker 必须实际是 terrain，采用新 source/合同与独立 admission SHA，完整继承未来真实 speed 父；先 10 critic，再首 actor。只读验证器应从真实边界证书导出预期时钟，不能沿用 source029 的固定 REC14188/消费41393 首 speed 常量。启动提交只记 `first_actor_pending`，不记已学习或 qualified。

这是方案，步骤中的新 source030 helper 尚未由本次审查实现。零 PPO 闭合和账户迁移属于未来显式部署操作，不得包装成 `run_readonly.py` 的只读调用。

## 两个容易混淆的文件生命周期

`chassis_remote_job.finalize()` **不修改 `job.json`**。它读取 job、训练文件和日志，生成 delivery archive/manifest。只有新 wrapper 启动时，`preserve_previous_job()` 才先保存旧 job 和 source-revision 回执，再把 `job.json` 更新为新 supervisor PID。动态回执仍应等旧 wrapper 真实退出后生成，不能因为当前实现不写 job 就提前绑定未来父。

controller 的零 PPO native close 位于旧 wrapper 归档之后，会合法更新当前 `train/campaign_state.json`、`consumption.json`、campaign completion 和账户 claim。因此旧 delivery 中这些**可变账本文件**的 SHA 不再等于 live 值，是正常顺序，不应重新 `finalize` 覆盖旧证据。分别保留旧 delivery manifest/摘要及 controller 闭合回执，验证不变的 speed PT/sidecar/worker completion；当前正常 speed record、账户与新准入由闭合后的文件认证。若需保留旧 delivery archive，采用明确的不可变归档名，不依赖会在下一次 finalize 被替换的通用文件名。

## 现有 helper 必须改的位点

| Source029 helper 位点 | Source030 所需最小调整 |
| --- | --- |
| `EXPECTED_PARENT`、`CONTRACT`、`CHAIN` | 去除固定 REC SHA/41393 起点；目标 v27，待启动链从 terrain 开始，父由真实证书导出。 |
| `stopped_original()` | 不再要求 source028/REC stopped19；改核验实际 source029 正常 speed、成功 controller、旧 wrapper 完整封存及无 owner。 |
| `extract_new()` 与 `stage` 分支 | 保留安全提取及归档 SHA/PAX 校验；required entrypoints 改为 v27/controller。staging 不要求旧训练停止，不生成动态修订回执。 |
| `validate_staging()` | 分离静态 source 身份校验和边界后动态 receipt 校验；旧合同是字节保留的子集，仅允许具名 12 个新合同；不使用同-v26 code-only 判据。 |
| `cpu_dispatch()` | 不能对 v27 用缺少 report 的 `TaskCampaign.__new__` 假调用：跳过 speed 现在需要真实完成证书。静态期用已有 plan/design 验证，边界后走真实准入；不另造假 record。 |
| `correction_command()` / `correct` | 删除该新部署流程中的 correction 调用；已付费 REC19 与原 correction 证据继续继承，不再执行。 |
| `launch_evidence()` | `prepare_admission(v27)` 被生产接口明确拒绝。改用锁下真实 `open_account → learning_source(terrain) → make_admission/authenticate_receipt`；不伪造 REC re-entry。 |
| `launch` | 沿用原 wrapper/intent；使用 `task-plan-revision-v27`，动态 previous job 与原始完整命令。每个 mode 用新进程、先加载新 source，避免旧 PYTHONPATH/module cache 混用。 |
| `durable()` | 复用现有原子排他 `durable_json`/已验证发布方式即可，不新建日志或调度框架；已有模糊 intent/log 必须先检查。 |

## 本机与 Kaiser 的职责、等待与互斥

- **本机：**代码/合同冻结、归档、传输的发起端、结果归档；`run_readonly.py` 只用于已审查 `.remote.py` 的紧凑状态读取。它是 SSH transport，不会自动限制脚本副作用，并且有 **180 秒总超时**，不适合运行 controller、迁移、launch、长时间等待或归档。
- **Kaiser：**新目录提取及静态 preflight；controller 的实际 PID/进程树认证和信号；正常 full seal、wrapper 封存、原生零 PPO 闭合、动态 receipt、账户迁移与原 wrapper launch。有限 controller 应作为现有远端独立进程运行，生命周期不依赖短 SSH 读取会话；不增加第二层 scheduler。
- **路径：**部署 helper 的 run root 是实验目录；controller 和 lineage/admission 的 run root 是实验目录的 `train`。二者不能互换。
- **超时：**controller 默认 3,600 秒覆盖“等待接近终点 + hold + wrapper 退出”，不是仅 hold 时长。若提前很久启动，必须依据真实剩余更新与当前耗时传入足够的有限超时，或临近窗口再启动；不能沿用默认而假称会持续到几天后的终点。额外零 PPO closure 子进程当前单独限定 180 秒，失败就核对封存后重入，不追加 PPO。
- **互斥：**`owned_processes()` 仅识别四类训练 owner，不识别 controller。030 prepare 不得仅凭 no-owner 继续；须确认成功回执并取得同一个 controller flock，再持原账户 flock，锁顺序与 controller 一致。成功 launch 仍由原 account 锁和一次性 intent 保障唯一 owner。
- **异常：**超时/普通取消会 CONT 旧 supervisor，可能继续旧 terrain；失败后不得假称正常边界仍成立。SIGKILL/主机丢失须依据持久 intent 重新认证恢复。真实旧 terrain 已开始则具名边界认证拒绝，不能通过退款、删除 selection 或放宽 identity 来绕过。

本报告只给出代码级部署接线结论。root 的组合定向测试、最终 12 合同冻结、真实未来 parent、边界执行和首 terrain actor 仍各需自己的实际回执；当前 `deployment=null`、`qualification=null`。
