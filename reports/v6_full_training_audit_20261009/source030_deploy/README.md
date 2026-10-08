# Source030 部署 helper（已静态暂存，尚未接续训练）

`deploy_source030.py` 复用 source029 的严格 Git archive 提取、现有排他发布、原账户锁和 native wrapper。它没有 SSH、信号、等待循环或调度器。2026-10-09 root已通过冻结的`run_source030_static.py`实际复制归档与helper、暂存4341文件，并完成远端30角色CPU预检；只读controller-check已通过。尚未执行boundary-prepare/launch，没有训练信号、额外wrapper/PPO/Isaac。

实际回执：上级目录的`source030_transfer001.json`、`source030_static_stage001.json`、`source030_static_preflight001.json`、`source030_static_remote_readback001.json`和`source030_controller_check001.json`。静态通过不代表未来父已绑定或新增动作已通过。默认权限001归档没有部署，实际采用freeze002。

最终源码 SHA 由命令行提供，当前 root 冻结的是 `81ad124554c4ab45cc274e2d482d46c76ba125b0`；归档 SHA 也必须明确提供。应采用 root 的 **freeze002 / tar.umask=0022** 归档。helper 拒绝文件权限不是 0644/0755、目录权限不是 0755 的包，并核对提取后权限，不能使用之前默认 umask 的 001 包。

## 四个独立调用

所有 mode 必需参数：`--run-root EXPERIMENT --old-source-root ACTUAL_SOURCE029 --new-source-root EXPERIMENT/SOURCE030 --source-archive EXPERIMENT/ARCHIVE --source-commit FROZEN40HEX --source-archive-sha256 ACTUAL64HEX`。均须为真实绝对路径。helper 本身复制到实验目录中的独立 SHA 固定文件，**不要放进预留的 `EXPERIMENT/source030_deploy`**；静态 stage 排他创建该产物目录。各 mode 用新 Python 进程，运行前由操作者校验 helper SHA。

1. `--mode static-stage`：可在当前 speed 运行时执行；只创建新源码目录和 `source030_deploy/stage.json`。不读取 job，不绑定未来 parent，也不创建修订回执。保留所有旧合同字节，仅接受具名 12 个 v27 新合同。所有归档文件均验证 SHA、PAX commit、路径和权限。
2. `--mode static-preflight`：只调用新源码原 `run_task_campaign.py --preflight-only --device cpu`，600 秒有限子进程；写独立日志和结果，不造 worker 或 scene。
3. `--mode boundary-prepare --controller-intent ACTUAL_INTENT --controller-receipt-sha256 ACTUAL64HEX`：仅在另行授权、实际正常 speed 边界 controller 成功后执行。本 helper 不运行 controller。获取 controller 固定 flock，再取原账户 flock；确认无训练 owner、controller 原 supervisor/worker 对应当前 job、普通成功退出、零 PPO close、未启动下一域以及真实 speed full-final 身份。核对旧 wrapper 当前 `delivery.tar.gz` 的完整 SHA、实际归档成员及未变的 PT/seal/contract/worker completion/job，核对 archiver 的 finished/PID。然后才使用原生 `TaskCampaign.open_account → learning_source(terrain) → make_admission → authenticate_receipt`。terrain actor 从 0 起、10 critic、全部消费不退回，未来父不硬编码。写独立 snapshots/admission、`source_revision_030.json` 和 `prepare.json`。
4. `--mode launch`，带相同实际 controller 参数以及 `--runtime-script ACTUAL_RUNTIME`：重新核对 source、父、归档、准入和当前账本未变；先排他写 launch intent，释放原账户锁，再启动原 `chassis_remote_job.py --resume-existing --source-revision-receipt`。controller 锁一直保持到提交回执，原 supervisor 自行获取账户锁、创建真实 terrain attempt。此阶段仅 `first_terrain_actor_pending`，不授学习完成/qualification。

controller、lineage 使用的 run root 是 `EXPERIMENT/train`；这个 helper 的 run root 是 `EXPERIMENT`。controller 的长等待及信号职责没有搬到此 helper。`run_readonly.py` 不得用于任何以上有副作用的 mode。

## 失败与归档语义

- 动态 receipt 只在真实旧 wrapper 退出、正常 completed-speed 和原生迁移认证之后发布。旧 `job.json` 不在 finalize 中改写；新 wrapper 才保留旧 job 并写新 PID。
- controller 的零 PPO 收尾发生在旧 delivery 归档之后，当前 campaign state/consumption 可合法变化。这些旧归档条目的 SHA **不要求等于当前 live 账本**。immutable speed 产物、旧 job 以及整个旧 archive 均实际检查；当前账户由原生 v27 准入另行认证。
- 无 owner 检查不能替代 controller flock；锁忙立即拒绝，不等待或抢占。锁顺序为 controller → account；launch 在提交前先解 account，避免 supervisor 等锁死锁。
- `prepare.intent.json`、未完成的快照或 launch intent/log 出现后，失败重入明确拒绝，保留证据供人工核对，不覆盖已有 admission。原生 v27 的可恢复三写事务仍可按其协议恢复，但不得删除此工具意图后未经检查再次启动。
- `delivery.tar.gz` 是下一轮 finalize 会覆盖的通用路径。helper 保存其 manifest/完整 archive SHA 和 immutable 成员清单；root 如需保留旧归档字节，应另行保存具名不可变副本，不能宣称通用文件名永久保留。

## 本机验证范围

`/usr/bin/python3 -B -m unittest discover -s reports/v6_full_training_audit_20261009/source030_deploy -p test_deploy_source030.py -v`

19 个 CPU 测试覆盖：静态 stage 无 job/parent 依赖、归档 SHA/PAX/路径/链接/重复/权限拒绝、旧合同字节和 12 新合同闭集、真实 archive 成员核对、允许归档后可变账本合法变化、不可变 seal 变化拒绝、错误父/未完成/额外 PPO/已后续 dispatch 拒绝、owner/archiver/退款拒绝、真实本机辅助进程持有 controller flock、controller→account 锁序与释放、准备失败保留意图、原生接口适配调用顺序、模拟 launch 的锁交接和一次性提交。

测试内 PT 字节和原生 endpoint spies **只验证部署 guard/接线**，不冒称完整学习状态认证通过；真实 full-state/迁移/零 PPO 闭合 CPU 生命周期验证仍引用 B 与 root 的独立结果。模拟 Popen 不启动训练，唯一真实测试子进程只持有临时文件 flock。

保留 `cpu001.log` 的 14 通过/1 fixture 错误（测试 `put` 未接受字符串路径）；修复后 `cpu002.log` 18 通过；补权限守卫后 `cpu003.log` 19 通过。未将前两次当最终结果，也未删除负面证据。
