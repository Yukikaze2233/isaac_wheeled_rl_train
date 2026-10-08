# REC41393 与 speed500：隔离均值回放驱动器

`scripts/run_rec_speed_mean_replay.py` 已实现原生启动、均值 ONNX 推理、真实命令斜坡、原生脚本→力矩混合→RL 接管、结束分母和现有指标的可执行路径。当前只验证 CPU 输入与事件归属；未启动 Isaac/GPU/PPO。`run_scope=interactive_unscored`、`qualification=null`，不做生产准入或实车发布。

运行库固定为真实 `e5bd9c1367b875d7f229b9a1930647013d023372`。每份输入先复用原 full 合同、响应、预算和封存认证，再核验 PT 均值与 ONNX 数值、35D/81D/6 单帧 ABI、控制和资产。实际父 SHA 为 `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967`，REC actor14188/消费41393；子模型必须提供真实 sealed speed actor500（消费41903、actor消费41723、谱系28156、10critic 已计费，origin 必须为此父）。不猜子文件/SHA。父 ONNX 与 speed500 封存未提供时不能执行实际比较。

固定七个原场景，各四副本，共28环境：230/305/430mm 静站与左右侧倒、正反俯卧四种真实原几何倒地链。倒地原 case/pose/20s 上限不变；原生任务成功或失败可更早结束，继续保持 done 不重复计分，未终局标右删失。不是 L0/L1/L2 混课或容易 reset 的平均完成率。

五协议按顺序执行，每个协议在一个独立原生进程内对两模型 reset 同一七场景；没有并行场景或训练。nominal 保留静站10s、倒地20s。其他四协议仅改变 stand305 的真实操作请求：正反0.5m/s 在1–5s；正反1rad/s 在1–10s。t5/t10 操作请求直接归零，原合同斜坡仍决定实际命令稍后归零。该场景自然回合上限分别17/22s，其他六场景不变。不覆写超时、native FSM 或出生姿态。

静停窗口从**实际前向和角速度命令严格同时为零**的策略区间开始，连续收500个20ms区间；不按机器人已经停止过滤样本。分别报告真实请求零时刻、实加命令零时刻、10s窗口完成时刻与提前自然终止/边界原因。窗口起点采用首个零命令区间开始的实际位置，速度与位移分母一致，先前运动行程不记成停车漂移。没有满10s就标 incomplete，不授完整停车成绩。循环按原生 per-case episode_limits 最大值运行，避免全局20s上限截掉22s yaw静停窗。复用现有 `EpisodeMetrics`，记录姿态、平面速度、高度、yaw 角速度、闭链误差和终止；原始稀疏轨迹保留 projected gravity、速度与位置。本次全出生段和固定quiet窗均采用raw warmup=0，明确区别于正式考试可能丢弃初始warmup的 MAE，不能混比正式成绩。

所有模型采用同一原生 `FixedCaseEnv.reset_suite`、出生seed290929和相同副本；额外在 reset 边界配对全局观测 RNG 与原 transport RNG/时钟。不在物理区间间写 root/joint 状态。两模型初始35D观测、出生/关节/质量/摩擦/实际通信参数 fingerprint 必须完全一致，差异立即中止，无自动重试。

控制器使用已认证 REC21 train 行：side1.5rad/s、`complete_started_blend=true`、blend0.2s、连续接稳1s、脚本上限8s。原 `RecoveryFSM` 文件 SHA `04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6` 不变，原 `native_runtime_binding` 对实际实例认证。50Hz 策略、200Hz 物理/反馈PD、160/2.5、64/32 solver 配置和3mm闭链保护必须读回；原子步审计继续核验每策略步4次 PD/求解/新反馈。28环境不能当4096吞吐测试。

独立投影仅复用原 `baseline` 角色验证器，不给 baseline 分数。原生产 full 规则（包括旧固定考试 side6）完全保留。投影明确移除原 campaign migration，并记录 diff/SHA。在这个独立0-PPO进程的 **FixedCaseEnv 构造范围内**，一次精确匹配的 config/profile/control内容与SHA/root/stage=recovery/role=baseline/seed290929/num_envs28/block0 调用使用刚重新认证的独立 receipt；其他或重复调用转回原 `design_preflight`。finally 恢复原函数。生产文件不修改、训练入口不接受这个投影。原 Env 的 `validate_runtime_state`、原 PhysX/闭链/动力学读回与独立FSM来源读回仍实际执行；尚未运行前不能称完整原生预检通过。

诊断区分首次 BLEND、首次 pureRL、首次 motion release、整段 `extras.policy_learning_mask` 的真实policy-owned秒数和终局原 `RequestProtocolMetrics` 分母；没有新增接管状态机或gate。八项物理稳定条件分别记录 last-physical-tick 的策略率快照、RL hold 时分母/通过帧，原生连续峰值仍来自全部物理子步，另保留仿真真值接稳峰值。瞬时条件比例不能替代连续1s或完整恢复成功。

随机化覆盖限制要看实际 profile，而非模块开关：这七个原 exact-case 没有新增 broad 20–80ms 传感器/动作年龄、负载或低摩擦压力profile。原 full关闭 signal_delay/signal_perturbations；没有 communication/dynamics/friction 专项 profile 时原 exact-case 使用名义参数，虽然相关模块配置可能enabled。driver 输出实际lane启用/质量/摩擦/延迟及 observation_noise_flag，不把它称广域DR验证。后续压力对照需另冻结共同profile。

## 运行入口

先在有两份真实输入后进行 CPU 准备。输出必须是新目录，PT/ONNX 的实际路径由调用者提供：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/yukikaze/isaacsim60-venv/bin/python -B scripts/run_rec_speed_mean_replay.py \
  --runtime-root /home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009 \
  --candidate "$REC_PARENT_PT" "$REC_PARENT_ONNX" \
  --candidate "$SPEED500_PT" "$SPEED500_ONNX" \
  --output /tmp/rec-speed-mean-cpu-new
```

此命令默认不启动原生场景。`--interface-smoke` 只允许两个现有同 REC21 合同导出做 CPU 接口准备，禁止配合 `--execute`，不能伪称新父子行为比较。

root 在实际500封存、独立审查与资源确认后，用另一个新输出目录运行同命令并增加 `--execute --device cpu`（或其明确安排的 `cuda:0`）。驱动器顺序启动五个原生子进程，每个比较两模型均值；stdout、启动墙钟与物理循环墙钟分开保存，任何失败就停止且不重试。无优化器/PPO/课程消费、qualification始终null。这里给出命令，不代表已获实际500输入或已运行。

产物包括 `replay_preparation.json`、每协议独立 SHA 期望表/合同/diff/receipt、每原生子进程 `native_replay.json`、每模型原指标/终局/quiet窗口 JSON 和稀疏trace。指标还写少量稳定 `Unscored/<protocol>/<scene>/<metric>` TensorBoard event；不启动本机TensorBoard服务，也不触碰6006/6007。

CPU focused checks覆盖原输入认证、7×4与原四倒地保持、一次DI/逐字段fallback/异常finally、严格t5/t10零请求、500区间quiet/提前终止、真实终局去重/final-mask与首次接管分开，以及首零区间 begin3.998→end4.000 的2mm停车位移数据流。实际 AppLauncher/PhysX/native初始fingerprint/控制读回与物理表现仍待执行，不根据CPU通过授能力。
