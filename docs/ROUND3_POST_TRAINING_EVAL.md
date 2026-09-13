# Round3-A：自动最终效果评估与报告回收

日期：2026-09-14（北京时间）。本流水线独立于正在训练的代码和既有最终模型回传任务。

## 已部署、已启动

正式训练保持原样：

- 冻结commit：`6ed83408d1d1cb02ea687e77fa80e5c20932f36c`。
- 代码：`/home/kaiser/robot-rl-sim60/isaac_wheeled_rl_train`。
- run：`/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/train`。
- 原训练PID：`446292`；session：`r3a-train-35f06fa3504049f8924f9ccd1ed02b93`。
- 未改训练源码、合同、进程或原completion；未替换本机`kaiser-round3-a-return`产物watcher。

| 新任务 | PID（启动记录） | tmux | 当前状态 |
|---|---:|---|---|
| Kaiser最终效果waiter | **452597** | `r3a-post-eval-waiter` | `waiting_training_process_exit` |
| 本机评估报告pull watcher | **31700** | `kaiser-round3-a-eval-return` | `waiting_evaluation_report` |

两者内部预算均 **183600秒/51小时**；外层timeout另留30秒退出余量。正式waiter于UTC `2026-09-13T19:16:53`启动，本机watcher约UTC19:17启动。后续状态以状态JSON为准，不能把这里的启动PID当永久活性证明。

新远端工具只放在：

`/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/audit/after_training/evaluate_finished.py`

源码新增：

- [scripts/round3/evaluate_finished.py](../scripts/round3/evaluate_finished.py)
- [scripts/round3/watch_evaluation.py](../scripts/round3/watch_evaluation.py)
- [scripts/round3/test_post_evaluation.py](../scripts/round3/test_post_evaluation.py)

部署时script SHA256：

```text
evaluate_finished.py
7f17fe054f52477ba7d35bb15a50ffd9888efe1de7703571872c1c4617804022
watch_evaluation.py
f366771e505c3c214d12df4bc6ef23e7c1316646eb8560116afd08e6b9412500
```

训练plan原文SHA：`e796d5689c6d22461812c19242873694963d133243f7034f93c9a4134de0b266`，本机/远端一致。

## 触发条件和失败处理

waiter每60秒只读检查训练PID及完成收据。**训练仍在运行时不启动正式评估**；不发送信号、不重启、不抢占训练控制。

自然退出后必须同时满足：

1. 通过现有`validate_completion`，status=`completed`、export=`verified`、完成/请求均10000次更新。
2. completion时间不早于本run的`train.started.json`，且不是未来时间。
3. 最终产物全部size/SHA符合completion；run manifest的合同、stage、训练seed和warm-start父模型血缘符合plan。
4. 合同semantic SHA、ONNX sidecar、manifest与模型SHA一致。
5. 原部署snapshot源码哈希和runtime哈希仍符合冻结记录。

若`export_failed`、`stopped`部分完成、缺失或错误metadata等，不对periodic checkpoint或partial ONNX兜底评估，发布 **`blocked_no_verified_final_policy`** 和原因。原completion不改写。
等待超期发布`expired_waiting_completion`；剩余预算不足下一项完整620秒窗口时发布`expired_before_evaluation_finished`。已验证产物后的评估失败记录为`evaluation_failed`，不改称训练失败。

## 正式评估内容

顺序运行 **0.29、0.30、0.31、0.32m**，每次：

```text
scripts/play_v40_onnx.py --research --headless --seed 44 --num-envs 1
  --onnx <正式train>/policy.onnx --contract <正式train>/contract.json
  --command 0 0 <HEIGHT> --max-steps 6000 --max-wall-seconds 580
  --report-dir <独立final-evaluation>/h029或h030/h031/h032
```

每项由`timeout --signal=TERM --kill-after=20s 620s`包裹，使用既有`sim60-runtime.sh`；没有PPO、GUI或新的观测/物理API。旧`evaluate_v40.py`未使用。

原20秒timeout保持不变，因此每高度是**累计60仿真秒，不是连续无reset60秒**。

统计来自全部真实pre-reset CSV行并与summary逐项核对：

- 每episode前2秒（`episode_time_s<=2`）为初始化；后续为统一稳态统计窗口。
- 高度绝对误差MAE、P95（线性插值），平移速率`hypot(vx,vy)`均值/P95。
- 每回合root-link世界XY首个已记录姿态到最后姿态的净位移，不跨reset相减。
- 真实`diagnostic_flags`计数、termination reasons、failure/timeout次数；不是旧的诊断/终止混用。
- 非轮净力候选帧和峰值单列。**它不是地面pair、支撑证明或真实滑移率。**

预置研究criteria：稳态高度MAE≤0.005m、P95≤0.01m、平移速率均值≤0.02m/s，nonfinite/failure为0。
`research_criteria_all_heights_pass`只表示这些研究指标，不等于实物放行或完整站立/摩擦验收。
`policy_quality_verified`不因文件SHA回传通过而变true。

未改冻结脚本来暴露`wheel_slip_diagnostics`。当前CSV不含完整轮world twist/ground-pair，**不输出虚构的滑移结果**。
冻结replay中的`SERVER FINAL ONNX`、`cross_sim=True`等为历史固定标签；本次身份以新模型实际路径、policy SHA和源/目标版本为准，报告已说明。

## 已真实完成的最小工程验证

使用已完成`smoke-smoke01/train`策略，不使用正在训练的checkpoint：

- 原smoke训练3 updates的final已完整验证；其ONNX SHA：`d3aa6088fb870a92b974272bce50d8b372e997535275092ad08fc78200b370af`。
- 实际只运行0.30m、seed44、1env、**2步/0.02仿真秒**，exit0。
- 真实CSV/summary生成并通过统计校验，8项diagnostic均0，failure/timeout均0；**steady=null，研究通过标志=null**。
- 标签：`engineering_2_steps_not_policy_quality`。不能把这2步当站立通过。
- 回收5个报告文件，逐项size/SHA验证通过，manifest SHA：
  `8f32f031f9139bb1a2ce8cd74327cf7edd48a51d1c5321963c120ede5da98144`。
- 工程评估进程452601及其tmux已自然结束。核查正式训练PID446292仍存在，未停止/重启。

CPU测试：`python -m pytest scripts/round3/test_post_evaluation.py scripts/round3/test_tools.py -q`，**20 passed**。
新增9项覆盖pending不吃periodic checkpoint、export失败/部分完成、过期completion、错误metadata、reset分段统计/位移、manifest路径与checkpoint拒绝、waiter到期落原因，以及SSH断线可重试状态。

## 报告最终落点

### Kaiser

```text
/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/audit/after_training/
  evaluate_finished.py
  r3a-post-eval-waiter.log
  final-evaluation/
    evaluation_state.json          # 运行中状态
    evaluation_result.json         # 最终各高度、分阶段、逐回合指标
    evaluation_report.md
    training_completion.json       # 成功触发时的只读收据副本
    report_manifest.json           # 最终报告白名单+SHA；最后发布
    h029/summary.json, telemetry.csv
    h030/summary.json, telemetry.csv
    h031/summary.json, telemetry.csv
    h032/summary.json, telemetry.csv
  engineering-smoke/               # 已完成2步工程验证
```

### 本机

```text
/home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round3_a_20260914/
  final-return/artifacts/           # 原模型回收watcher负责，未改
  evaluation-watcher.log
  evaluation-return/
    watch-state.json
    local_receipt.json
    reports/                       # 最终报告完整验证后原子发布
      evaluation_result.json
      evaluation_report.md
      report_manifest.json
      h029…h032/summary.json, telemetry.csv
  evaluation-smoke-return/          # 已实际回收工程证据及local_receipt
```

仅回收报告JSON、Markdown、四高度CSV/summary，不下载checkpoint、USD cache或巨大Kit日志。支持白名单文件复用，断线后的部分下载不会发布为已验证最终报告。

## 查看与断线恢复

远端运行状态：`final-evaluation/evaluation_state.json`；本机状态：`evaluation-return/watch-state.json`。

本机离线、ControlMaster失效或认证不可用时，watcher记录`transport_unavailable_retryable`，不保存密码、不改SSH key、不尝试新密码认证。恢复同一control路径后自动重试；51小时后退出并记录可重试状态。此时可用原命令重新启动**本机报告watcher**，不会重跑评估：

```bash
python scripts/round3/watch_evaluation.py \
  --host kaiser@192.168.64.234 --ssh-port 2222 \
  --control-path /tmp/opencode/kaiser-2222-control-current \
  --remote-output /home/kaiser/robot-rl-sim60/round3-runs/train-formal01/audit/after_training/final-evaluation \
  --training-plan /home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round3_a_20260914/plan.json \
  --destination /home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round3_a_20260914/evaluation-return \
  --wait-seconds 183600 --poll-seconds 60
```

可追加`--once`单次检查。运行中的同目录watcher有互斥锁，勿另起重复任务。远端评估output拒绝复用已有目录；远端中途崩溃或超期若需重跑，应先检查原因并用新output、对应新本机destination，而不是覆盖旧报告。

部署脚本SHA已记录；后处理工具独立发布，正式训练继续使用冻结的`6ed8340`源码。
