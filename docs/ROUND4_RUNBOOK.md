# Round4 full：用户授权旧串联模型，从零正式训练与自动回收

## 2026-09-15 新决定与一条命令执行

用户明确授权先用**旧七刚体串联等效研究模型**开始训练，新15body模型仅预览，不参与本次physics。取消等待repaired asset的旧调度，保留初始时槽`2026-09-15T00:00:00+08:00`；已经过点时，完整源码部署和CPU验证完成即启动，不顺延到次日午夜。

授权原文和身份元数据：`docs/evidence/round4_serial_authorization_20260915.json`。

```text
physics_model = legacy_seven_body_equivalent_serial_research
physics_asset_manifest_sha256 = df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886
repaired_dynamics_used = false
explicit_user_authorization = true
initialization = scratch; parent = null; seed = 44
```

bundle/snapshot/plan及`audit/physics-identity.json`记录这些身份；部署、启动前对旧research manifest和其中所有`files_sha256`做验证，训练产物再次核对实际`asset_manifest_sha256`。不要求尚不存在的repaired mass properties，不把旧模型描述成通过修复资产验证。

### 旧 cron 已实际取消

2026-09-15 **05:28:05+08:00**，通过新有效ControlMaster `/tmp/opencode/kaiser-training-control`，只删除了：

```text
# robot-rl-schedule:round4-repaired-20260915
```

对应整条cron。当时仅有该条用户cron；取消后为空，已读回验证。原目录`/home/kaiser/robot-rl-sim60/scheduled-training/round4-repaired-20260915/`保留request、geometry confirmation、cron.log、schedule.py，增加`state-before-supersession.json`和`cancelled.json`；`state.json`为`superseded`，记录授权原文、精确被删除行、crontab前后SHA、`repaired_asset_validation_passed=false`。取消前为`waiting_repaired_asset`，本操作未启动训练或停止其他进程。

主 agent 拿到完整整合commit后，在本机仓库根执行：

```bash
python scripts/round4/schedule_serial.py start \
  --commit '<MAIN_AGENT_FULL_INTEGRATION_COMMIT>' \
  --ground-usd /home/kaiser/robot-rl-sim60/experiments/a-evaluation-recovery-20260914T070110Z/default_environment.usd \
  --destination /home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_full_20260915 \
  --host kaiser@192.168.64.234 --control-path /tmp/opencode/kaiser-training-control \
  --execute
```

如已指定评估agent的实现，同一命令加`--evaluation-entry <实际已提交评估脚本路径>`；该值写入plan，训练成功结束且子进程清理后，**远端worker自动执行评估dispatch hook**。本机关闭不影响训练、远端评估提交和报告落盘。没有evaluator时明确pending，不宣称自动评估通过；后续可用相同request协议和独立已提交评估快照接入。

这条命令依次：幂等确认旧cron取消 → 打包/验证完整commit及用户授权 → 部署独立实验 → 保持原not-before判断且过点立即提交 → 创建本机有界回传watcher。`submitted`仍不是首次更新或训练成功证明。`serial-schedule.json`在提交前写claim；失败/断线的不确定提交不会自动再开PPO。已有部署会核验复用，不覆盖源码。已有claim需按audit确认，恢复下载使用watcher而非重启训练。

新命令默认dry-run，省略`--execute`不写远端。独立取消命令为：

```bash
python scripts/round4/schedule_serial.py cancel-old \
  --host kaiser@192.168.64.234 --control-path /tmp/opencode/kaiser-training-control --execute
```

一条命令模式的本机plan在`<destination>.plan.json`，watcher日志在`<destination>.watcher.log`；本机重开后使用：

```bash
python scripts/round4/watch.py \
  --plan /home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_full_20260915.plan.json \
  --destination /home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_full_20260915 \
  --host kaiser@192.168.64.234 --control-path /tmp/opencode/kaiser-training-control
```

## 目标与当前事实

- 单次正式训练：**1024 env × 30000 updates**，`locomotion / research / headless`，随机初始化，无旧权重。
- `contracts/own_v40_round4_full.json`：完整平地鲁棒 profile、startup friction DR、按成功 PPO update 计时的 push curriculum；stand 30%、spin 10%、drive 60%，目标高度 .29–.32 m。精确物理配置以已提交合同及 inline 读回为准。
- 直接进入正式 runner。没有独立 PPO 快测、smoke/capacity prerequisite，也不调用 Round3 warm-start launch gate。
- A 已于 07:57 完成并 verified export/回收；网络地面问题修复后重评，**只有 h=.32 全部通过，h=.30 non-wheel force 持续**。这不是 Round4 的站立验收结果。旧 frozen 目录、archive、B1 工具保留。
- 此交付仅修改部署/调度管理工具、runbook及用户决定metadata；提交与实际PPO启动由主 agent 执行。

## 目录与身份

```text
/home/kaiser/robot-rl-sim60/experiments/round4-full-<40-char-commit>/
  isaac_wheeled_rl_train/       # 完整已提交源码；叶名固定
  bundle.tar.gz
  snapshot.json                # 全部源码 size/SHA、commit、scratch、parent=null
  formal/
    audit/                     # plan / effective command / PID / log / resources / worker status
    train/                     # completion / final checkpoint / ONNX / manifest / curriculum
    usd-cache/
    evaluation/
      request.json             # 明确待评估计划，不代表成功
      submission.json          # 仅配置 evaluator 后可能出现
      reports/                 # evaluator 实现的 result / CSV / report manifest
```

没有 `parent/` 模型路径。代码部署只用 `git archive <commit>`，不读取未提交训练代码。上传后验证 bundle 和每个源码 SHA，最后发布 `snapshot.json`；已有实验目录不覆盖。

## 1. 本机 CPU 检查与部署

主 agent 完成 contract/env/algo/wrapper 整合后运行相应 CPU 测试并提交；无需 GPU smoke。

```bash
python -m pytest -q scripts/round4/test_round4_tools.py scripts/round3/test_tools.py

HOST=kaiser@192.168.64.234
CONTROL=/tmp/opencode/kaiser-training-control
COMMIT='<FULL_MERGED_ROUND4_COMMIT>'
EXP=/home/kaiser/robot-rl-sim60/experiments/round4-full-$COMMIT
GROUND=/home/kaiser/robot-rl-sim60/experiments/a-evaluation-recovery-20260914T070110Z/default_environment.usd
PLAN=/tmp/opencode/round4-$COMMIT-plan.json
LOCAL=/home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_full_20260915

python scripts/round4/deploy.py --commit "$COMMIT" \
  --host "$HOST" --control-path "$CONTROL"
python scripts/round4/deploy.py --commit "$COMMIT" \
  --bundle "/tmp/opencode/round4-$COMMIT.tar.gz" \
  --host "$HOST" --control-path "$CONTROL" --deploy
```

第一条为无文件/网络写入的 dry-run。第二条仅部署。`GROUND` 是恢复 A 评估时缓存的官方 Sim6 Grid USD；正式 trainer 校验固定 SHA，wrapper 记录 size/SHA 并在 child 启动前重查。

## 2. 并发协调后，显式启动一次正式训练

**最新2026-09-15 05:27附近只读观测**：未见WSL PPO/sensitivity训练进程；MemAvailable 15281774592 bytes（约14.23GiB），GPU used/free=1889/22250MiB、利用率29%。只代表当时WSL进程/全卡快照；没有停止用户任务，不能推断Windows端全部空闲。以下为前一日历史记录。

交付前只读资源检查发现 sensitivity stage-b 正在运行：tmux `sensitivity-b-071345`，编排 PID `586958`，当时训练 PID `592541`、`593495`（各约4.6 GiB RSS）；GPU 14073/24564 MiB、91%利用率，WSL MemAvailable 7044288512 bytes。**这些是观测快照，不是当前空闲承诺**。

16:10:31（北京时间）再次只读检查：以上编排/训练PID及tmux session均已退出；MemAvailable为14911123456 bytes（约13.89GiB），GPU used/free为5674/18465 MiB，利用率15%。仍有显存占用，不能据此假定Windows端完全空闲。

主 agent 启动前重新查询并协调当前 sensitivity 工作。wrapper 只记录进程/资源快照，不停止、重启或按名称 kill 其他任务；`.round4-training.lock` 仅防本工具重复正式 worker，不是跨项目 GPU 预约锁。

```bash
ssh -S "$CONTROL" -o BatchMode=yes -o ProxyCommand=false -p 2222 "$HOST" \
  'free -b; /usr/lib/wsl/lib/nvidia-smi; ps -eo pid,ppid,etime,rss,args; tmux list-sessions'

# 远端 dry-run：校验完整源码、合同、runtime、官方缓存路径；不构造 Sim。
ssh -S "$CONTROL" -o BatchMode=yes -o ProxyCommand=false -p 2222 "$HOST" \
  "/home/kaiser/robot-rl-sim60/env/bin/python '$EXP/isaac_wheeled_rl_train/scripts/round4/launch.py' --experiment '$EXP' --ground-usd '$GROUND'"

# 正式启动。输出 submitted 只代表 tmux 提交，training_success 始终不能据此判定。
ssh -S "$CONTROL" -o BatchMode=yes -o ProxyCommand=false -p 2222 "$HOST" \
  "/home/kaiser/robot-rl-sim60/env/bin/python '$EXP/isaac_wheeled_rl_train/scripts/round4/launch.py' --experiment '$EXP' --ground-usd '$GROUND' --launch"

scp -o ControlPath="$CONTROL" -o BatchMode=yes -o ProxyCommand=false -P 2222 \
  "$HOST:$EXP/formal/audit/plan.json" "$PLAN"
```

实际核心 argv：

```text
train_v40.py --contract contracts/own_v40_round4_full.json
  --max-iterations 30000 --num-envs 1024 --stage locomotion --research --headless
  --seed 44 --device cuda:0 --ground-usd <PINNED_OFFICIAL_CACHE>
  --max-runtime-seconds 172800 --stop-at <worker-start + 48h>
  --run-dir <EXP>/formal/train --usd-cache-dir <EXP>/formal/usd-cache
```

不传 `--resume/--finetune/--warm-start/--stage-transfer`。CPU preflight 与启动物理读回由这个正式 trainer inline 执行，wrapper 仅启动这一个训练 child。

### 有限预算

| 参数 | 默认 | 作用 |
|---|---:|---|
| learning soft budget | 48h | trainer 自身学习预算 |
| absolute stop-at | worker正式阶段开始+48h | 含初始化的绝对训练截止 |
| export/finalization余量 | 30min | soft截止后保存/导出 |
| child hard timeout | 48h30min | 到时仅终止本 child group；通用 run_child TERM/KILL 回收 |
| outer timeout | 49h + TERM后最多120s | worker/子孙有限保护 |
| 本机 watcher | 52h | 训练后含评估/传输的有限重试预算 |

启动前显式 `--soft-hours 72` 可改为72h soft、72h30min child、73h outer；watcher同步改 `--wait-hours 76`。该选项记录在 plan，不能靠修改运行中 plan 延长已启动 timeout。默认48h满足约15h估算的余量；估算不作为完成证据。

## 3. 正式启动后必须观察的证据

主 agent 至少观察到以下证据再报告已正常开始训练：

1. `train.started.json` 中 PID 对应当前完整 command；`train.log` 中首次成功更新。
2. `run_manifest.json`：`training_profile=round4_full`；`initialization == source_provenance` 且 mode=scratch，随机架构/seed/profile与plan一致；无parent、无resume_provenance。
3. `domain_randomization_report.passed=true`，实际材质分桶/readback；`training_curriculum.timebase=successful_ppo_updates`、实际 course state，而非只打印配置。
4. 第一条真实 push 事件（含施加的 delta-v/目标env/时刻），随后 `model_100.pt` 与对应保存时的成功更新时钟。文件出现不是独立的策略质量判定。

可只读观察日志与manifest：

```bash
ssh -S "$CONTROL" -o BatchMode=yes -o ProxyCommand=false -p 2222 "$HOST" \
  "tail -n 80 '$EXP/formal/audit/train.log'; cat '$EXP/formal/train/run_manifest.json'"

python scripts/round4/watch.py --plan "$PLAN" --destination "$LOCAL" \
  --host "$HOST" --control-path "$CONTROL" --once
```

`watch --once` 没有terminal evidence时退出3，表示pending。完整源码hash在`<EXP>/snapshot.json`；PID/argv、资源起始快照、每30秒资源记录、训练日志、child退出状态、worker子孙回收状态均在audit。资源轮询为记录用途，没有内核内存硬隔离。

## 4. 本机持续监督与自动回传

`reports/current/` 父目录必须存在；`LOCAL`可不存在，或必须已由同一plan的watcher创建。日志写到父目录，避免先创建LOCAL破坏owner协议。

```bash
tmux new-session -d -s "round4-return-${COMMIT:0:8}" \
  "timeout --signal=TERM --kill-after=30s 52h python '$PWD/scripts/round4/watch.py' \
    --plan '$PLAN' --destination '$LOCAL' --host '$HOST' --control-path '$CONTROL' \
    > '$LOCAL.watcher.log' 2>&1"
```

同命令恢复watcher只观察与回传，不重提训练。ControlMaster失效只有限重试，不写密码、不新认证。

- `artifacts/`：复用通用 `pull_v40_artifacts` completion/allowlist/逐文件SHA验证，校验scratch/合同/seed/实际curriculum后原子发布；含local receipt。
- `audit/`：worker结束后回传完整源码snapshot、plan、effective argv、PID/status、train.log、resources等，逐文件SHA及二次读一致性；单文件上限64MiB，超限明确报错，不静默截断日志。
- `watch-state.json`：区分pending、stopped、export_failed、30000更新完成、证据拒绝、评估实现待补。
- `evaluation-request.json`：训练全部完成、导出验证、训练PID退出且worker清理成功后生成并回收。

`stopped`即使有可用checkpoint/ONNX也不称完成30000更新；不触发最终评估。自动回传不删除远端产物，不按`reward>=91`升级结论。

## 5. 评估接口：可晚于训练补齐

本次实现调度/计划与报告回收，**不实现物理评估器**。没有提供`--evaluation-entry`时，训练/模型回收照常完成，记录`awaiting_evaluator_implementation`并退出3，不伪造空success。

计划含8个case：`standing_h029/h030/h031/h032`及`single_push_h029/h030/h031/h032`。固定command关闭随机push；standing每高度3×20s；single_push每高度20个episode、5s施加一次`env.apply_velocity_impulse(delta_xy, env_ids)`。默认delta-v使用合同最终curriculum幅值沿世界+X（当前0.5m/s）；可在第一次生成请求时用`--push-delta-v vx vy`显式指定，保存后不静默改配置。

恢复要求：**2s内完成持续0.5s的steady-band hold**；默认height误差≤5mm、平面速度≤0.02m/s。0.5s/band是本计划的显式评估参数，不宣称已获实验验证。统计必须报告planned/pre-push-failed/disturbed/recovered/failed-disturbed/censored计数，分母包含所有实际受扰episode及其失败/非有限/未恢复情况；censored不能算恢复。逐episode时间线和CSV用于复查hold。

`net_contact`只作为诊断指标，不能当成ground-pair≤5N的证据。

### 后续 evaluator 的协议

```text
<PYTHON> <EVALUATOR> --request <EXP>/formal/evaluation/request.json
                    --output <EXP>/formal/evaluation/reports
```

evaluator 必须是已验证commit快照中的文件。可传本轮相对路径，也可传**另一独立完整部署**中的绝对路径，便于训练启动后补评估实现而不改训练frozen源码。dispatcher记录其commit/snapshot SHA；tmux outer为9000s，TERM后120s；存在submission claim后不自动重复执行。evaluator负责记录自己的PID/日志、子进程退出与实际物理测量。

watcher加以下参数即可在正式完成后自动提交该实现并回收报告：

```text
--evaluation-entry <RELATIVE_COMMITTED_SCRIPT_OR_ABSOLUTE_ISOLATED_SCRIPT>
```

输出最后发布`report_manifest.json`：

```json
{
  "schema_version": 1,
  "request_sha256": "<SHA256_OF_REQUEST_JSON_BYTES>",
  "files": [{"path": "result.json", "size": 123, "sha256": "<SHA256>"}]
}
```

上述JSON仅说明schema；真实清单还必须包含原始CSV（仅JSON/CSV/Markdown，≤100项，每项≤32MiB）。`result.json.cases`须包含全部8个唯一case id，每个都有`status=completed/failed`和非空`metrics`。恢复计数/hold判定由实际 evaluator 按request完成；wrapper对SHA和结构做校验，**不把传输校验当成策略质量通过**。完整文件校验且manifest二次读取一致后发布本机`evaluation-reports/`和`evaluation-receipt.json`。

## 验证范围

最新CPU回归：`python -m pytest -q scripts/round4/test_round4_tools.py scripts/round4/test_serial_schedule.py scripts/round4/test_schedule.py scripts/round3/test_tools.py`，**45 passed**。包括精确cron删除与并发变更拒绝、旧状态保留、05:03过点立即提交、claim防重复、旧research资产所有引用文件SHA，以及训练wrapper/timeout回归。`schedule_serial.py`、`launch.py`、`evaluation.py --help`正常。

2026-09-15 05:30:44+08:00再次远端读回：crontab为空，旧调度保持`superseded`，备份状态为`waiting_repaired_asset`。实际部署/PPO仍等待主agent完整commit。

CPU测试覆盖完整源码hash/路径隔离、固定scratch argv/预算、无父权重/无resume、实际读回与curriculum字段、仅一个正式child、超时仅回收持有的child、重复worker拒绝、恢复分母和hold计划。测试结果见本次交付消息。

真实deploy、正式Sim inline读回/首次更新/push/checkpoint、最终导出/回传和物理评估均由主agent后续执行；本次没有运行GPU smoke或启动native GUI。
