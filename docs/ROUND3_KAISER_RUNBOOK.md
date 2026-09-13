# Round3-A Kaiser：部署、工程验证、长训与自动回传

## 边界与实现

本工具只负责部署和进程/产物边界，不改 PPO、env、warm-start 权重映射或评估阈值。源码必须来自**主 agent 合并测试后已提交的 Git commit**；工作树的 dirty/untracked 内容不会进入快照。`094fa33` 不含 Round3-A 合并代码，不能直接用于本次部署。

**远端代码根固定为 `/home/kaiser/robot-rl-sim60/isaac_wheeled_rl_train`，是真实目录，不建立 symlink、不使用 `source/` 子目录、不修改 GitHub 仓库名。**父模型和部署清单放在同级 `round3-a-deployment-<commit>/`，runs/audits 继续独立保存。代码来自 Git archive 的完整已提交文件内容，Git commit 和文件 SHA 由部署清单记录，不是另一个带 `.git` 的 clone。

实现复用：

- `scripts/start_v40_round2.py` 的 `run_child()` / `verify_run()`：PID、进程组超时、退出回执、completion/所有产物哈希/ONNX sidecar 校验。**没有复用 pilot→resume 流程。**
- `scripts/pull_v40_artifacts.py` 的 probe、文件指纹和 `pull_artifacts()`：只按 `completion.artifacts` 允许列表传输，逐文件 SHA-256 校验，完成清单最后发布。
- `scripts/round3/common.py`：现有 OpenSSH ControlMaster 的小型 transport adapter；使用 BatchMode 和 `ProxyCommand=false`，master 不可用时不尝试新的认证。
- `deploy.py` / `launch.py` / `watch.py`：快照、分 profile 启动和可恢复回传。

本轮仅完成工具/CPU 验证和只读 SSH/SCP 验证，**没有启动 Round3 训练、容量测试或原生 GUI**。主 agent 负责合并、全仓测试、commit/push，以及实际 launch。

## 固定父模型和阶段隔离

父目录：

```text
/home/yukikaze/Documents/workspace/robot_rl/reports/current/cqa1_round2_20260912/final
```

固定父 `model_final.pt` SHA-256：

```text
f94919f8f7ae1cfee95e45b922a9939e21c9f6ad9dff5656ff6f29dc0f4e6360
```

部署时验证 Round2 completion、合同身份及其中列出的全部文件，复制 8 个 artifact 加 completion 到快照 `parent/`。每个 profile 都对同一 `parent/model_final.pt` 执行 `--warm-start`，并验证 preflight 和 run manifest 的 `source_provenance`：父 SHA、target contract SHA、optimizer reset、初始 iteration 0。

**正式 run 不 resume smoke/capacity 权重。**smoke 的 3 次更新、capacity 的 10 次更新仅为工程产物，回传明确标记 `engineering_only_not_round3_final`。正式训练完成也不会被标为策略质量或下一阶段达标。

| profile | env | updates | 软学习预算 | 初始化余量 | 导出/清理余量 |
|---|---:|---:|---:|---:|---:|
| smoke | 2 | 3 | 180 s | 600 s | 600 s |
| capacity-256 | 256 | 10 | 900 s | 600 s | 600 s |
| capacity-1024 | 1024 | 10 | 900 s | 600 s | 600 s |
| train | 默认 1024；可选已 benchmark 的 256 | 默认 10000 | **48 h** | 30 min | 30 min |

正式 trainer 的硬上限为 **49 h**；worker 外层另加 2 min preflight 和 5 min 管理余量，并有 TERM→KILL 兜底。训练 cutoff 在 worker 开始该阶段时计算，为“当前时间 + 初始化余量 + 软学习预算”，不会以机器不关机为由无限运行。

worker 启用 Linux child-subreaper，并回收自己的残留子孙进程（包括另建 session 的 exporter）；不按进程名杀进程、不重启 tmux server。`.round3-training.lock` 只防止这些 Round3 工具互相并发，不能阻止不使用该锁的其他探针；仍由主 agent 协调 GPU 使用。

## 1. 本机：提交后打包与部署

以下在 `isaac_wheeled_rl_train-60` 根目录执行。替换用户/主机；ControlMaster 使用当时有效的路径，不存密码、不改 authorized_keys。

```bash
HOST='<USER>@<KAISER_HOST>'
PORT=2222
CONTROL=/tmp/opencode/kaiser-2222-control-current
COMMIT=$(git rev-parse HEAD)
PARENT=/home/yukikaze/Documents/workspace/robot_rl/reports/current/cqa1_round2_20260912/final
SNAP=/home/kaiser/robot-rl-sim60/round3-a-deployment-$COMMIT
BUNDLE=/tmp/opencode/r3a-$COMMIT.tar.gz

# 默认只验证/打印；不会创建文件、联网或训练。
python scripts/round3/deploy.py --commit "$COMMIT" --parent "$PARENT" \
  --bundle "$BUNDLE" --remote-dir "$SNAP" \
  --host "$HOST" --ssh-port "$PORT" --control-path "$CONTROL"

# 验证通过后显式部署；新目录、新 bundle，绝不覆盖旧快照。
python scripts/round3/deploy.py --commit "$COMMIT" --parent "$PARENT" \
  --bundle "$BUNDLE" --remote-dir "$SNAP" \
  --host "$HOST" --ssh-port "$PORT" --control-path "$CONTROL" --deploy
```

实际布局：

```text
/home/kaiser/robot-rl-sim60/
  isaac_wheeled_rl_train/              # 代码根：scripts/、src/、contracts/、assets/ 等
  round3-a-deployment-<commit>/        # 独立部署容器（--snapshot 指向这里）
    parent/                           # 经 SHA 验证的 Round2 final
    snapshot.json
    bundle.tar.gz
    deployment.json
  round3-runs/                        # 独立训练 run 和 audit
```

bundle 的源码成员前缀为 `isaac_wheeled_rl_train/`，解包直接写入上述固定代码根；parent 和清单留在部署容器。所有源码/parent 文件有 size/SHA-256，最后发布 snapshot marker。**固定代码目录已存在时明确拒绝覆盖（包括 symlink），不会自动改名或切换链接。**首次部署失败保留现场；主 agent 应先检查，不要仅换 metadata 目录盲目重试，也不要在训练中替换固定代码根。

首次使用时创建独立 run 容器（不是训练 run 本身）：

```bash
ssh -S "$CONTROL" -o BatchMode=yes -o ProxyCommand=false -p "$PORT" "$HOST" \
  'mkdir -p /home/kaiser/robot-rl-sim60/round3-runs'
```

## 2. 远端：smoke → capacity → formal

在 Kaiser WSL shell 中设置以下变量。所有 `launch.py` 命令默认 dry-run；加 `--launch` 才创建独立 tmux。它会在 worker 内重新 source 既有 runtime、执行 CPU preflight，然后才可能创建 Sim。

```bash
SNAP=/home/kaiser/robot-rl-sim60/round3-a-deployment-<FULL_COMMIT>
CODE=/home/kaiser/robot-rl-sim60/isaac_wheeled_rl_train
RUNS=/home/kaiser/robot-rl-sim60/round3-runs
PY=/home/kaiser/robot-rl-sim60/env/bin/python
ENTRY="$CODE/scripts/round3/launch.py"

# 先省略 --launch 审查计划；确认后追加 --launch。
"$PY" "$ENTRY" --snapshot "$SNAP" --run-root "$RUNS" \
  --profile smoke --label smoke01 --launch
```

本例 smoke 的验证状态路径：

```bash
SMOKE="$RUNS/smoke-smoke01/audit/worker.status.json"

# 等 smoke 完整成功，再分别、串行执行容量测试。
"$PY" "$ENTRY" --snapshot "$SNAP" --run-root "$RUNS" \
  --profile capacity-256 --label cap01 --smoke-status "$SMOKE" --launch

"$PY" "$ENTRY" --snapshot "$SNAP" --run-root "$RUNS" \
  --profile capacity-1024 --label cap01 --smoke-status "$SMOKE" --launch
```

容量状态含 `learning_elapsed_seconds`、`rollout_steps_per_env`、按实际完成更新数计算的 aggregate env-steps/s，以及所有 artifact/ONNX 验证结果。它不是 GPU 峰值显存测量；主 agent 还需结合 train.log、OOM/退出情况和必要的显存观察选择规模。10 次更新只是短容量探针，不证明 48 h 稳定。

```bash
CAP="$RUNS/capacity-1024-cap01/audit/worker.status.json"
"$PY" "$ENTRY" --snapshot "$SNAP" --run-root "$RUNS" \
  --profile train --label formal01 --num-envs 1024 --updates 10000 \
  --smoke-status "$SMOKE" --capacity-status "$CAP" --launch
```

正式启动要求 smoke 和与 env 数对应的 capacity 已成功，且 commit、目标合同、父模型一致；重新校验门禁 run 的产物，不仅信任 saved passed 标志。若选 256，则用 capacity-256 的状态。正式 updates 显式调整必须至少 100；3/10 update 应使用工程 profile。

每个 run 的 `audit/` 有不可覆盖 plan、worker PID、preflight.log、train.log、train.started.json（真实子 PID/完整 argv）、train.status.json（退出/超时）、worker.status.json；有效训练产物在同级 `train/`。tmux 提交成功、session 消失、甚至 checkpoint 文件存在都不是训练成功判据。既有 runner 的 checkpoint interval=100 保留，不由本工具覆盖。

## 3. 本机：完成后自动回传与恢复

取得该 run 的 `audit/plan.json`，并在**本机独立 tmux**启动 watcher：

```bash
# 本机：LOCAL 的父目录须已存在，路径不含 symlink；每个 run 用自己的目录。
LOCAL=/tmp/opencode/r3a-formal01
mkdir "$LOCAL"
REMOTE_JOB=/home/kaiser/robot-rl-sim60/round3-runs/train-formal01
scp -o ControlPath="$CONTROL" -o BatchMode=yes -o ProxyCommand=false -P "$PORT" \
  "$HOST:$REMOTE_JOB/audit/plan.json" "$LOCAL/plan.json"

# REPO 使用本机仓库绝对路径；占位替换后执行。
REPO=/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train-60
tmux new-session -d -s r3a-watch-formal01 \
  "timeout --signal=TERM --kill-after=30s 51h python '$REPO/scripts/round3/watch.py' \
    --plan '$LOCAL/plan.json' --destination '$LOCAL/return' \
    --host '$HOST' --ssh-port '$PORT' --control-path '$CONTROL' \
    --poll-seconds 60 --wait-seconds 180000 > '$LOCAL/watcher.log' 2>&1"
```

watcher 每 60 秒只读检查 worker、对应 PID、tmux、最近 checkpoint 文件元数据和 completion。正常途中不拉走每个 periodic checkpoint。完成后调用原有 `pull_artifacts()`：completion/schema/文件指纹检查、只传允许列表、逐文件 size/SHA 校验、completion 最后发布。

- 成功结果在 `return/artifacts/`，包含训练 completion 和 `local_receipt.json`；`watch-state.json` 记录 profile、实际/请求更新数及工程/正式分类。
- transfer 断开留下 `attempt-<UUID>/`；下次采用新 attempt 重试，不覆盖未知目录，验证完整后才 rename 为 `artifacts/`。这是可恢复的完整文件重试，不是任意 partial 文件的字节级续传。
- 同一 plan/host 的 watcher 可重新运行并复用自己的目录。已有 artifacts 会重新验 SHA；不同 run/plan 会拒绝复用。并发 watcher 有文件锁。
- `export_failed` 时可以拉回已完成 checkpoint，但不会标记为 ONNX 成功或正式目标完成。`stopped` 可能是预算下的部分更新，不能称为完成 10000 次更新。
- 本机断开、watcher 到期或退出不会停止远端 tmux 训练；训练自身仍受软预算和独立硬上限控制。
- **认证真实限制**：成功的周期性 multiplex 会话通常会重置 master 的空闲计时，但不能保证主机重启、socket 清理、网络长期断开后仍免密。master 不可用时记录 `transport_unavailable`，不询问/保存密码、不尝试新认证；相同路径恢复后可继续。若改用新 control 路径，停止旧 watcher 后用相同 plan/destination 和新路径重启，不能伪称永久自动回传。
- watcher 有 50 h 自身预算；超过预算退出并保留状态，主 agent 可恢复它。正式训练本身硬上限约 49 h，留有回传时间，但网络恢复可能仍需要人工处理。

## 4. 固定高度 deterministic eval：单独决定、单独运行

回传 SHA 验证只证明产物完整，不证明 R3A 达标。后续由主 agent 按目标合同允许的高度建立固定高度用例，示意接口：

```bash
# 仅为后续调用模板；本轮不执行，不自动触发下一 phase。
timeout --signal=TERM --kill-after=30s 15m \
  "$PY" "$CODE/scripts/evaluate_v40.py" \
    --contract "$CODE/contracts/own_v40_round3_a.json" \
    --checkpoint '<R3_RUN>/train/model_final.pt' --stage locomotion \
    --research --headless --num-envs 1 --seed 43 --duration-s 10 \
    --command 0 0 '<REVIEWED_HEIGHT>' --output-dir '<NEW_EVAL_DIRECTORY>' --apply
```

该入口的 R3A 合同适配和阈值由主 agent/评估协作者最终核对。至少区分固定高度误差、存活/终止原因、倾角、轮接触、非轮触地、速度误差和执行器统计；不由部署脚本自动宣告下一阶段达标。原生 GUI 训练显示另列待对接，长训/benchmark 不启动 GUI。

## 验证范围

- `python -m pytest -q scripts/round3/test_tools.py`：**11 passed**。覆盖固定父模型/工程标签、warm-start lineage、48 h + finalization 预算、门禁身份、禁止新认证回退、归档路径/哈希、watcher 目录恢复、只读健康检查、真实 CPU 子进程退出及 detached exporter 子孙进程回收。
- 本地 Round2 父目录的全部 9 个部署文件已通过现有 completion/size/SHA 验证。
- 用现有 ControlMaster 完成了无副作用 SSH 和只读 SCP 下载测试。
- 旧提交 `094fa33` 被部署前检查按预期拒绝；未创建训练 snapshot 或启动 GPU 任务。
- **尚未验证**合并提交后的远端解包、Round3 preflight/warm-start、真实 smoke/容量/长训、48 h 运行和真实 R3 completion 自动回传。主 agent 应先完成合并测试，再按上述阶段执行。
