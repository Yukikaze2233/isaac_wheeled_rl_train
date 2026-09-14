# A 长训并行 B1 pilot：独立部署、资源保护、回传与交接

## 本轮交付范围

只新增 `scripts/round3/pipeline_*.py`、`test_pipeline.py` 和本文件。原 A 的 fixed code root、启动工具、PID 446292、模型/评估 watcher 均不修改、不停止。没有依赖 `systemd --user`、`/run/user/1000/bus`、sudo 或用户 cgroup。

本轮完成 CPU 测试和一次真实只读 A periodic capture；没有部署 B1、启动 pilot、benchmark、长训或原生 Kit。必须由主 agent 合并测试后提交完整源码，再显式调用下面的 deploy / submit。

## 固定关系和目录

```text
/home/kaiser/robot-rl-sim60/
  isaac_wheeled_rl_train/                     # A frozen 6ed8340，保持不变
  round3-runs/train-formal01/                  # A，保持不变
  experiments/b1-pilot-<B_COMMIT>/
    isaac_wheeled_rl_train/                   # B1 完整已提交源码，叶名不变
    parent/                                  # A periodic 的独立只读快照
      model_<index>.pt                        # 保留原文件名
      run_manifest.json
      contract.json
      source_hashes.json
      asset_manifest.json
      agent_config.json
    pipeline.json                            # 全文件 size/SHA、Git commit、源/目标身份
    bundle.tar.gz
    pilot/
      audit/                                 # plan/PID/log/resource/stop/exit 证据
      train/                                 # B1 本次产物
      usd-cache/
```

父 task A 的固定合同 SHA：

```text
e31ba1538be4f26d5a5c6452ca8b06e1d09f41db999118957909d13f7248e73e
```

目标 `contracts/own_v40_round3_b1.json` 必须来自指定已提交 commit，`round3.stage=B1`。训练 argv 使用算法侧约定：

```text
--stage-transfer parent/model_<index>.pt --source-checkpoint-sha256 <CAPTURED_SHA>
```

实际血缘字段是 `source_provenance.parent_checkpoint_sha256`（不是 CLI 名称的照抄），并核对 `mode=stage_transfer`、A→B1、父/目标合同 SHA、原文件名、`optimizer_reset=true`、`initial_iteration=0`、`std_preserved=true`。父快照多带 `agent_config.json`，因为算法侧验证 `BASE_ARTIFACTS`。

## 1. 本机：检查候选父 checkpoint 与部署

在本机代码仓库根目录，使用现有效 ControlMaster，不存密码、不修改认证：

```bash
HOST=kaiser@192.168.64.234
CONTROL=/tmp/opencode/kaiser-2222-control-current
ARUN=/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/train

# 只读，动态选择当前最高编号且 mtime 已超过5秒的 periodic 文件。
python scripts/round3/pipeline_deploy.py --inspect-parent \
  --a-run "$ARUN" --host "$HOST" --control-path "$CONTROL"
```

capture 对 checkpoint 和五份配套 JSON 分别做 lstat/fstat、普通非链接文件检查、两轮完整读取/SHA，中间至少间隔1秒；任何变化都拒绝。部署时下载后再次比对复制内容 SHA，最后重读同名源文件比对全部指纹和哈希。**不读取/依赖未完成的 completion，也不把 periodic 改名成 final。**

本轮只读样例捕获到 `model_3200.pt`（1,493,429 bytes），SHA 为 `0d91339c6fe585d5eec08c4eec972fa8181aba4958318189fb0ccefb843572ff`。这只是当时的样例；实际 deploy 默认重新选最新稳定文件，也可用 `--checkpoint model_N.pt` 指定同名父版本。

主 agent 提交完整 B1/env/transfer/pipeline 代码后：

```bash
COMMIT='<FULL_MERGED_B1_COMMIT>'
EXP=/home/kaiser/robot-rl-sim60/experiments/b1-pilot-$COMMIT
python scripts/round3/pipeline_deploy.py --commit "$COMMIT" \
  --a-run "$ARUN" --experiment "$EXP" --bundle "/tmp/opencode/b1-pilot-$COMMIT.tar.gz" \
  --host "$HOST" --control-path "$CONTROL" --deploy
```

省略 `--deploy` 只检查 committed source 和只读父快照、打印计划。包复用现有 `committed_files()`，不打包 dirty/untracked 算法文件；新 container 必须位于 `experiments/b1-pilot-*`，不会写 A 代码根。上传后核验 bundle 和每个文件的 SHA，最后发布 `pipeline.json`。已经存在的实验目录不覆盖，失败现场不删除。

## 2. 显式启动 pilot，并自动创建一个新的本机 watcher

建议从本机通过 `pipeline_follow.py submit` 启动：只有 `--launch` 才会真正提交远端任务，随后自动创建本机 watcher。提交默认 dry-run，不会创建 watcher。

```bash
LOCAL=/tmp/opencode/b1-pilot-$COMMIT
APLAN=/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/audit/plan.json
AEVAL=/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/audit/after_training/final-evaluation
ARETURN=/home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round3_a_20260914/final-return/artifacts

python scripts/round3/pipeline_follow.py submit \
  --experiment "$EXP" --destination "$LOCAL" \
  --host "$HOST" --control-path "$CONTROL" \
  --a-plan "$APLAN" --a-evaluation "$AEVAL" --a-return "$ARETURN" --launch
```

`LOCAL` 的父目录应存在，`LOCAL` 本身必须新建。若只需 B1 回传，省略三项 A 证据参数；可在证据齐全后另行运行 handoff。输出区分 `pilot_submitted` 和 `watcher_submitted`：若远端已提交但本机 watcher 创建失败，**不要重提 pilot**，按 `pilot-plan.json` 恢复 watcher。

远端可先独立检查 dry-run：

```bash
/home/kaiser/robot-rl-sim60/env/bin/python \
  "$EXP/isaac_wheeled_rl_train/scripts/round3/pipeline_pilot.py" --experiment "$EXP"
```

pilot 固定 **256 env × 500 updates**，使用现有 runtime，`--headless`、不启 GUI。独立 tmux + GNU timeout **7200秒**（TERM 后120秒 kill 兜底）；内部 deadline 7080秒，预留最终清理。训练软预算最多5400秒，实际还受绝对 cutoff 限制，cutoff 距内部 deadline 保留600秒保存/导出余量。所有等待都计入总任务预算，不建立无界隐藏进程。

## 资源监测与保护 A

阈值依据：用户 04:11 北京时间观测 WSL available 约10 GiB，A RSS 4.14 GiB、历史峰4.86 GiB，GPU使用约6.14/24 GiB。启动时以实时读数为准：

| 条件 | 行为 |
|---|---|
| 启动前 `MemAvailable < 6 GiB`，或 GPU free <2 GiB /查询不可用 | 等待；每次 admission 最多30分钟，仍受总2小时约束；preflight 前后都检查 |
| 运行中 `MemAvailable < 2 GiB` | **只对本 pilot 的 Popen handle 发 SIGTERM**，请求训练预算机制在安全边界保存 |
| 运行中 GPU free <1 GiB | 同样仅停止本 pilot，标记 `gpu_memory_reserve` |
| SIGTERM 后未结束 | 最多留600秒保存/导出，且不超过内部硬 deadline；最后仅清理本 worker 的进程树 |

资源日志约每5秒一条，查询本身有短 timeout，故不是硬实时监测。记录 WSL MemAvailable、本 pilot RSS/VmHWM、全卡 used/free/total 和可查询到的本 pilot 显存；WSL/WDDM 若无法把 PID 映射到显存，写 `null` 与 `unavailable_or_WSL_PID_not_mapped`，不把整卡显存冒充 pilot 占用。

使用独立 `.b1-pilot.lock` 防止多个本工具 B pilot 同时启动，**不获取或修改 A 的 `.round3-training.lock`**。复用 A 工具里的 child-subreaper/子孙回收和 `run_child()`（仅 CPU preflight），训练监控只持有自己创建的 Popen。没有按名字杀进程或向 PID 446292 发信号。

**这不是内核级内存隔离**：没有假设 user-bus/cgroup 可用，不能保证快速突增、驱动 OOM 或其他程序抢占时 A 绝对不受影响。6 GiB 启动门槛和2 GiB余量是保护性运行策略，不是显存/RSS硬配额。

关键证据：`pilot/audit/plan.json`、`worker.started.json`、`preflight.log`、`train.started.json`、`train.log`、`resources.jsonl`、`soft-stop.json`（如发生）、`worker.status.json`，以及 `pilot/train/completion.json`。

只读查看最近资源记录和状态：

```bash
ssh -S "$CONTROL" -o BatchMode=yes -o ProxyCommand=false -p 2222 "$HOST" \
  "tail -n 3 '$EXP/pilot/audit/resources.jsonl'; test ! -f '$EXP/pilot/audit/worker.status.json' || cat '$EXP/pilot/audit/worker.status.json'"
```

## 3. 回传、认证恢复和独立命令

本机 watcher 复用既有 `pull_v40_artifacts` 的 completion/指纹/允许列表和逐文件 size/SHA 校验。传回 `$LOCAL/return/train/`，不会删 remote。中断保留 attempt，恢复时使用新 attempt；已发布文件重新验 SHA。pilot 永远标 `experiment_not_promoted`，`stopped`/`export_failed` 不会伪装成完成500次更新。

恢复命令（不会再启动远端 pilot）：

```bash
tmux new-session -d -s b1-return-recovery \
  "timeout --signal=TERM --kill-after=30s 52h python scripts/round3/pipeline_follow.py watch \
    --plan '$LOCAL/pilot-plan.json' --destination '$LOCAL/return' \
    --host '$HOST' --control-path '$CONTROL' \
    --a-plan '$APLAN' --a-evaluation '$AEVAL' --a-return '$ARETURN' \
    > '$LOCAL/watcher-recovery.log' 2>&1"
```

watcher 有文件锁与最多52h预算，失败状态和 PID 写入 `watch-state.json`。ControlMaster 不可用时只记录并重试，不落盘密码、不新认证、不修改 authorized_keys；换新 control 路径后重启该 watcher 即可。不能保证本机重启/休眠后仍免密。既有 A 模型和评估 watcher 完全独立，保持原状。

## 4. 自动交接只给建议，不偷启 B 主训

配置 A 三项证据参数时，B1 watcher 完成自己的回传后继续只读观察 A：

1. A 进程退出、final completion 达到计划更新数并全部 artifact/ONNX 校验通过。
2. 四高度 `.29/.30/.31/.32` 的 report manifest 必须覆盖全部 CSV/summary 和最终报告/收据；逐文件 SHA、每个 case 的 policy/contract、报告的 plan 身份一致且 case 正常退出；重用现有 `summarize()` 从 CSV/summary 重算冻结指标（稳态 MAE≤5mm、P95≤10mm、平移均速≤0.02m/s、nonfinite=0、failure resets=0）。
3. B1 pilot 完成500更新、出口通过、stage-transfer 血缘匹配、没有资源中止。**不读取 pilot reward 来决定升级。**
4. 在建议 B1 正式准备前，还验证现有 A watcher 的本机 `local_receipt.json`、completion 和全部文件 SHA，与被评估 A final 一致。

`$LOCAL/return/handoff.json` 的 `next_action` 会明确为：等待A/评估/本机回收、`continue_A_optimization`、`blocked_resources`、`blocked_pilot_or_export`、`blocked_evidence_or_export`，或 `prepare_B1_main_from_A_final`。最后一种要求**新 run 从已验证 A final 再做 stage-transfer，不 resume 本 pilot**。`B_main_started` 始终 false；这组工具没有 B 主训启动函数，也不建立新的用户审批步骤。

可在远端单次只读生成建议：

```bash
/home/kaiser/robot-rl-sim60/env/bin/python \
  "$EXP/isaac_wheeled_rl_train/scripts/round3/pipeline_handoff.py" \
  --a-plan "$APLAN" --a-evaluation "$AEVAL" \
  --pilot-plan "$EXP/pilot/audit/plan.json" --output "$EXP/handoff.json"
```

该远端单次检查不访问本机 A 回传目录，明确 `A_local_return_checked=false`；本机 follow watcher 才补齐这项检查。

## 已验证与未验证

- `test_pipeline.py`：**12 passed**。覆盖 periodic 双读/变化拒绝/硬链接拒绝、阈值、仅 pilot SIGTERM、stage-transfer 身份、不能凭高 pilot reward 跳过 A final/四高度准则；新增四高度各6000帧合成 CSV 的真实统计路径，验证通过、高度未达标、错误 policy、遗漏 SHA 条目、发布后 CSV 变化。
- `python -m pytest -q scripts/round3/test_pipeline.py scripts/round3/test_post_evaluation.py scripts/round3/test_tools.py`：**32 passed**；四个 pipeline CLI 的 `--help` 均正常。
- 真实只读 capture：上述 `model_3200.pt` 及五份 JSON，双次 stat/hash 一致，源 A SHA/RSL5 身份正确。没有向 remote 写入或创建任务。
- 新合并 commit 的部署、B1 preflight、500-update 并行运行、真实低内存保存、完成回传及最终 handoff 尚未端到端执行，须由主 agent 提交后实际调用。本轮不宣称 pilot 或 B 主训已运行。
