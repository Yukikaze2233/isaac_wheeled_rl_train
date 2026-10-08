# 未来 v25 只读学习窗口适配

本目录只有一套适配：`reader.py`＋CPU 合成测试。尚未对 Kaiser 执行，不是部署、PPO、模型或资格通过回执。当前 source023 不能通过该适配的准入；source027 正常边界实际采纳后才可绑定。

解析核直接加载已有 `learning_audit/read_recovery_window_019.py`，认证 SHA `ffefbaae2e1cbf30b520224fc286a8adbc194708570e4ce0fcb00fa7f86cdb17`。CRC 对齐、TFRecord 流式读取、scalar、stats、计数不变量原样复用；不下载 event，不用 event `read_bytes()`，不导入 torch、模拟器、模型或生产模块。

## 绑定真实证据

调用方先保存并审查实际采纳后的 authority JSON。该文件必须包含：

- `schema: "v25-readonly-actual-authority-v1"`、`reviewed_actual_adoption: true`、`source_commit: "59c201678b37cb33a03a878e734d1a263d8ab97a"`。
- 真实绝对路径 `campaign_root`、`source_root`、唯一原生 worker 的 `run_dir`。
- `owner: {pid, ppid, start_ticks, command}`，来自实际只读进程回执。适配再核对 `/proc` 与唯一 campaign worker；单 PID 不够。
- `files` 下列每项均为 `{path, sha256}`，全部为实际已保存文件，没有自动占位或成功模板。

| key | 原生文件 |
|---|---|
| revision | campaign 根的 `source_revision_027.json` |
| adoption | campaign 根的 `boundary_continuation_027.receipt.json`，实际 `launch_submitted` |
| invocation | attempt 的 `invocation.json` |
| selection | attempt 的 `integrated_source_selection.json`，与调用参数路径/SHA一致 |
| admission | train 的 `integrated_admission.json`，与 selection 字节 SHA 一致 |
| startup | train 的 `startup.json` |
| resume | train 的 `resume_verification.json` |
| design | train 的 `integrated_design.json` |
| agent | train 的 `agent_config.json` |
| worker | actual invocation 指定的静态 v25 worker 合同 |
| recipe | worker 认证的 `contracts/v6_integrated_runtime_recipe_v25.json` |

缺采纳、selection、runtime、resume、startup、reviewed owner 或任意 immutable pin 时默认拒绝。recipe SHA 固定 `69efab87fb9422f19fee7318ca3aab278488202b644e906b1be7c5ff77724902`；6 个源 producer/native 文件与 source027 inventory SHA及 startup 的 native source SHA交叉核对。

目录重建只执行已认证源码的纯 `_expand` AST 函数，再附加已保存 selection/clocks；没有新的 runtime 读回。重建目录须逐组对齐冻结 recipe 的实际 `allocation.counts`、`integrated_design.runtime_expected.integrated_curriculum` 与真实 `startup.scene_group_counts/behavior_pool_counts`。另外原样复用3个纯原生 clock函数，核 sealed selection/component clocks、startup observed 与 progress joint lineage/component clocks。不用静态 worker 的135/101/56/57组代替真实展开目录；也不根据189/233/248/262的预期数臆造人口。

不调用 `authenticate_receipt`、`attach_runtime` 或启动时账本准入：它们要求 live account 仍等于启动快照，PPO推进后合法消费已增长。只读适配认证保存的准入/设计/恢复/源与实际 progress，**不重新批准预算、谱系或运行**，不读取模型张量。

## 一次有界读取

在已有同一 Python 环境中调用，所有路径必须先绑定为实际证据：

```sh
python -B /absolute/path/future_learning_reader001/reader.py \
  --binding /absolute/path/actual-reviewed-authority.json \
  --kernel019 /absolute/path/preserved/read_recovery_window_019.py \
  --actor-end ACTUAL_COMPLETED_LOCAL_ACTOR_ENDPOINT \
  --max-read-bytes 536870912
```

适配仅将 JSON 写 stdout；调用方用已有执行器保存实际 stdout/stderr/exit/耗时。没有 SSH 包装或远端启动操作。默认512MiB，硬上限1GiB；suffix缺少实际baseline则缺测/不完整，不回退全历史读取，不自动第二次读取。exit0表示完整只读窗口，exit3表示缺 tag，exit2表示准入/CRC/计数/owner等拒绝，均不授功能资格。

`actor-end` 是当前同 worker attempt 的已完成 actor 数，窗口100个 actor，baseline必须真实存在；endpoint须匹配原生每10个 **completed optimizer** 的 Behavior cadence。消费 parent、剩余 critic、域 actor origin来自保存的 selection/resume/invocation，完成量从 progress读取；`attempted_updates` 不作为完成，不固定重加10 critic。`parent_updates`与`restored_checkpoint_updates`保留各自含义。

## 输出含义

- 动态目录全部 Behavior 计数/均值、10秒静止窗口、Timing、按实际 `kind` 选择的 start_stop/velocity_curve/release_stop 条件制动窗、全局 PPO mask 曝光。每组 rollout分母是 `24 × startup实际lane人口`，不固定552；全局4096×24。
- forward_05/backward_05的16项原生实付/优势/残差/帧数，完整100与前后50。这些组已是 **连续命令混采的旧名字**，不是固定±0.5用例；实际指令均值来自 Behavior。reward/s与stored advantage/return单位分开，实付不证明梯度。
- `RecoveryOutcomes/{prefix_stand,prefix_release,original_chain}`原生整数 baseline/end同 worker差分；逐端点与差值核 `ended=success+failure+collection_timeout`。ended=0时 success ratio=null；缺 tag不补0，超过simple_value可保证整数精度范围拒绝。混合恢复标签成功率不冒充纯L2完整链率。
- 行为是带随机化/探索/脚本前史的训练窗口；不授确定性策略能力、当前 pitch/roll 精度、硬件资格或改合同收益。前后attempt累计计数不能直接相减；owner改变须新绑定。
- PPO `Timing/iteration_wall_seconds`与整个 CPU reader耗时单独输出。仅认证实际已读suffix区域，不授全部event历史CRC。

CPU测试使用临时目录、合成 metadata/Event/owner，明确不是实际 source027 采纳回执。最终 CPU stdout/stderr/执行回执另存本目录；首次真实使用仍待根在正常边界确认采纳。

`cpu001` 是修复两处人口/clock认证缺口前的历史测试，其输入SHA仅描述当时字节；不声称可用当前 reader重算。最终当前实现以 `cpu002` 为准，旧实际回执保留。
