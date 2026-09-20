# V5 locomotion v2：基础训练记录（已结束）

最终状态：2026-09-20 02:50:57（UTC+8）完成打包，1500 更新、18,432,000 transitions，
`regression_hold_best_preserved`。三个新增 checkpoint 的固定评测均未通过，原始基线通过并保留。
顶层 `policy.onnx` 对应最后的未通过候选，不能作为已通过产物使用。
后续完整专项课程见 [SCUT-V5 v3](V5_SCUT_SKILLS_V3.md)。

2026-09-20 00:04:39（UTC+8）在Kaiser启动，冻结源码：
`003e4980d9672acee18aea3ce8d5e0fa779dbe55`。
00:09:56确认进入训练并完成11更新、135168 transitions；编排PID589540，首块训练PID592133。

## 训练内容

这是单策略主线的**平地基础细化与评测闭环**：站立102环境、平移76环境、旋转78环境，共256环境。
前向命令上限0.5 m/s、yaw上限1 rad/s，高度0.29–0.32 m。
后续地形、落地和跳跃按[重设计路线](V5_TRAINING_V2_DESIGN.md)分别实现与验收，不计为本轮已完成能力。

- 初始化：旧mixed最终19179次权重，仅迁移actor；critic和优化器新建。
- 原始actor SHA256：`51260d0c1518c697bbcf98d07af392b70161b401fc57d82d32f6adfd22137e30`。
- 合同：`contracts/v5_locomotion_v2.json`，SHA256 `d6185e3f2f9f73c4cfe986eac08dc8f98706a159d4e5787ec68686b51829fc03`。
- 物理1 kHz、策略100 Hz、TGS 32/8；仍为19体真实闭链和10 MPa气簧。
- 最大10000更新，目标122880000 transitions；**24小时是包括评测／重启在内的运行预算，不保证跑满更新上限**。

## 每个训练块如何接受

入口`run_chassis_blocks.py`先在Kaiser复核初始actor，保存`baseline_actor.pt`及独立评测。
每500次PPO更新保存权重并退出训练子进程，释放其仿真资源，再启动固定场景评测。
评测不把数据写入PPO rollout。

- 7案例×8个初始扰动实例，每例10 s，actor确定性推理，关闭随机推力和命令重采样。
- 分别报告完整回合、高度、速度、yaw、零指令漂移和终止原因。
- `best_candidate.pt`仅在排名优于此前候选／基线时保存；`model_best.pt`保留新合同下通过评测的最佳权重。
- 连续两次通过标记`foundation_accepted`；已有通过基线而连续三次失败时，停止并保留基线。
- 基础门槛不是全部技能验收，也不是实机部署许可。

## 遥测与查询

远端根目录：
`/home/kaiser/robot-rl-sim60/experiments/v5-locomotion-v2-20260919T160436Z-1d0908`。

tmux：`v5-locomotion-v2-20260919T160436Z-1d0908`。
回执：`docs/evidence/v5_locomotion_v2_launch_20260920.json`。

```bash
python scripts/check_chassis_remote.py docs/evidence/v5_locomotion_v2_launch_20260920.json
```

主要文件位于远端`train/`：

- `progress.json`：总更新、当前子进程和`training`／`fixed_evaluation`阶段。
- `block_*/behavior_metrics.json`、`behavior_history.jsonl`：分组跟踪误差与结束原因。
- `block_*/torque_monitor.json`：每物理步实际力矩统计，int64计数和float64累计。
- `baseline_evaluation/`、`evaluation_*/`：固定套件、哈希、原始代表轨迹及逐案例判定。
- `curriculum.json`：已完成训练块及门槛历史。
- `model_final.pt`：最新训练快照；是否接受以对应评测为准。

顶层实时状态为当前／最近训练块的转发。评测期间姿态缓存停止刷新是正常的，
应结合`progress.phase`识别，不能把缓存画面当作当下训练实例。
分块统计从该块的环境创建时开始，不能把当前块RMS称为整轮累计RMS。

## 回收和验证记录

远端job结束后自动打包，包含训练块、评测、日志、checkpoint、ONNX和原始轨迹。
本机独立tmux `v5-locomotion-v2-recovery`等待并逐文件校验，目标：
`reports/v5_locomotion_v2_recovered_20260920/`。

```bash
python scripts/watch_chassis_artifacts.py docs/evidence/v5_locomotion_v2_launch_20260920.json \
  --output reports/v5_locomotion_v2_recovered_20260920 --seconds 90000
```

启动前完成49项本地针对性测试和GitHub两项CI作业。
Kaiser 256环境8更新短测跑通基线评测、PPO、ONNX和后评测，74项产物回收校验通过，
位于`reports/v5_v2_kaiser_probe_recovered_01/`。其后评测单次通过，连续通过次数为1；
短测状态`budget_exhausted_gate_pending`表示尚未满足连续两次的阶段门槛。
正式运行的后续结果以其进度与评测文件为准。

2026-09-20 已完成正式轮回收：167 项文件逐项 SHA256 校验通过；
归档 SHA256 `e9b05a3c0bd8c564257ab585f9f8996d3934d3904f0102060c47ae42682b9ea4`。
本地凭据：`reports/v5_locomotion_v2_recovered_20260920/recovery.json`。
