# Round4 实际续训确认 — 2026-09-17

## 结论

已于北京时间 **2026-09-17 16:54:49** 提交独立正式续训，完成正常 Sim/PhysX startup 后实际进入 PPO。**首个新 checkpoint `model_7800.pt` 已生成并验证**：累计完成 7801 updates，较暂停时新增 66 updates，actor/critic 已发生更新，Adam 历史连续且材料 mapping 未改变。

没有运行独立 PPO/GPU 快测，没有改动 env/core/contract、网络或奖励，也没有覆盖旧 run/completion。

## 父产物与恢复口径

- 父 run：`/home/kaiser/robot-rl-sim60/experiments/round4-full-fb334c4f96a098ce398f0c94754021fe7c21d0d4/formal/train`
- 父 checkpoint：上述目录的 `model_final.pt`。
- 父 SHA256：`0cbd7aec25d8bcd2058e5742df349b5e02591e411e3d7d0968b17eaf81d2f9b6`。
- 父 completion：`stopped / sigterm / export verified`，本次原训练已完成 **7735 / 30000**。
- 合同语义 SHA256：`f701f2e391fd4e48d65902632b5a166df63e5967d38a65731fa84caf5cd7d592`。
- 旧 7-body research 资产 SHA256：`df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886`。
- seed 44，1024 env；剩余调用量 **22265 updates**。
- Ground：`/home/kaiser/robot-rl-sim60/experiments/a-evaluation-recovery-20260914T070110Z/default_environment.usd`，维持原官方固定 SHA。

CPU 用真实父 checkpoint 和官方 RSL5 MLP/PPO 实例核验：actor/critic/std 逐 tensor 相等、17 组 Adam 状态逐项相等、学习率恢复为 `1e-5`。`saved_iter=7734` 是旧 RSL 循环索引，恢复时使用 checkpoint infos 的 **completed_updates=7735** 设置 runner 和课程。

恢复的是模型、std、optimizer 和课程进度，**不是 bitwise 轨迹恢复**；环境 episode 状态、push timers/masks/RNG 不承诺接续。manifest 的 `initialization/source_provenance.mode=scratch` 描述原始训练起源；本次调用另有 `resume_provenance` 绑定父 SHA 和 7735，controller plan 的 `initialization=resume` 明确区别于从零启动。

## 实际部署与进程

- 冻结代码 commit：`6916f801077b9bed77d13d28e749b32383089e23`，已 push `isaac60`。
- 新 experiment：`/home/kaiser/robot-rl-sim60/experiments/round4-full-6916f801077b9bed77d13d28e749b32383089e23`
- code leaf：上述目录下 `isaac_wheeled_rl_train`。
- 新 stage：上述 experiment 下 `resume-20260917T085448Z-f2782b`。
- **新 run**：`/home/kaiser/robot-rl-sim60/experiments/round4-full-6916f801077b9bed77d13d28e749b32383089e23/resume-20260917T085448Z-f2782b/train`
- plan：新 stage 的 `audit/plan.json`；实际参数与新截止记录在 `audit/effective-command.json`。
- 远端 tmux：`round4-resume-d0832c6ecdc947c9bcd878bbb5ff454e`。
- controller PID：**10538**；正式训练 PID：**10560**（本次确认时）。
- 相对训练预算：172800 秒（48h）；新绝对截止：**2026-09-19 16:54:49.960297 +08:00**。
- child hard budget 174600 秒，worker hard budget 176400 秒，保留最终保存/导出/清理空间。

唯一正式训练调用使用 `scripts/train_v40.py --resume <父 checkpoint> --max-iterations 22265`。没有复用过期的旧 `stop_at`。

## 正式 startup 与首个新 checkpoint 的证据

新 startup 完整 `domain_randomization_report` 与父 manifest/checkpoint **逐项相等**，物理材料 readback `passed=true`。课程初值：

| 指标 | 从父产物接续 |
|---|---:|
| completed updates | 7735 |
| push max Δv [m/s] | 0.38675000000000004 |
| vx cap [m/s] | 2.091315789473684 |
| yaw cap [rad/s] | 2.365263157894737 |

北京时间 17:02:29 已观察到训练日志 iteration 7808；`model_7800.pt` 文件已稳定存在。随后以 CPU、`weights_only=True` 和现有完整 shape/finite loader 检验该 checkpoint：

- 文件：新 run 的 `model_7800.pt`。
- SHA256：`9ef05a699285afc1a8b58bcf31c0224c9caf5d7040bba844734b46dd3c2e1782`。
- saved loop iteration：7800；**累计 completed updates：7801**，新增 66。
- 17 组 Adam step：**154700 → 156020**，每组增加 **1320 = 66 × 20**，没有重置为新 optimizer。
- actor 与 critic 权重均已变化；optimizer tensor 全部 finite；六维 std 保持学习状态，未重置为统一初值。
- 材质 mapping 与父 checkpoint 完全一致。
- push max Δv：**0.39005 m/s**。
- vx cap：**2.0947894736842105 m/s**；yaw cap：**2.379157894736842 rad/s**。
- 父 checkpoint SHA 再次验证未变。

`model_7800.pt` 对应 7801 次成功更新，是 RSL 零起始循环索引与成功更新计数的正常差异，不是多跑或少跑。

## 诚实 completion、累计 receipt 与自动评估

新增 `scripts/round4/resume.py`，复用既有 `run_child`、artifact/sidecar 验证和 trainer 的严格 resume。子 `completion.json` 必须如实记录本次请求 **22265**；绝不改写成 30000。

controller 在成功或计划性停止并完成导出后，另外写 `audit/cumulative-receipt.json`，关联父 checkpoint/父 completion SHA 与子 completion/manifest SHA，并分别列出：

- prior completed updates：7735；
- invocation completed updates：本次真实值；
- cumulative completed updates：两者之和。

只有**累计 30000、子完成目标且 export verified、子进程清理完成**，才由服务器自动提交既有 12-case evaluator。部分或失败写明确 blocked hook；原 scratch/schedule 的防旧模型门禁保留，不把 resume 当 scratch 调用。

## 新 watcher / 回传

- 本机 plan：`/home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_resumed_20260917.plan.json`。
- 本机回传目标：`/home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_resumed_20260917`。
- 本机 watcher tmux：`round4-resume-watch-20260917`，确认时 PID **34097**。
- SSH ControlPath：`/tmp/opencode/kaiser-training-control`，`kaiser@192.168.64.234:2222`。
- watcher 已实际运行，当前等待的是**新 run** 的 completion，不是旧暂停 run。

本机关闭不影响远端训练、最终导出和服务器 evaluation hook；本机 watcher/回传需要本机与 SSH 恢复后继续。可以使用同一 plan/目标目录重新启动 watcher：

```bash
/home/yukikaze/isaacsim60-venv/bin/python scripts/round4/watch.py \
  --plan /home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_resumed_20260917.plan.json \
  --destination /home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_resumed_20260917 \
  --host kaiser@192.168.64.234 --ssh-port 2222 \
  --control-path /tmp/opencode/kaiser-training-control --wait-hours 52
```

已有 watcher 存活时不要重复启动；目录锁会拒绝并行回收。

## CPU 验证

```bash
/home/yukikaze/isaacsim60-venv/bin/python -m pytest \
  scripts/round4 tests/test_round4_evaluation.py tests/v40/test_round4_launch.py -q
```

**96 passed**，另有已有 ONNX 导出弃用提示。真实父 checkpoint CPU 恢复验证和首个新 periodic checkpoint CPU 核验亦通过。
