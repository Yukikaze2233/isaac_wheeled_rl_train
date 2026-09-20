# V5 SCUT v4：弧线场地修正与保留技能的恢复训练

## v3 的真实结果

运行 `v5-scut-v3-20260920T133901Z-d01443` 已结束于 `stage_gate_pending`，
阻塞任务 `curve`，子状态 `regression_hold_best_preserved`。

- stand、height、forward_05、backward_05、start_stop_05、rotate_1：各自双种子通过，自动跳过训练。
- curve：三个 250-update 块，共 750 PPO 更新、700 次 actor 更新、9,216,000 transitions。
- 已验收 checkpoint 仍为原始 19179 模型，SHA256 `51260d0c1518c697bbcf98d07af392b70161b401fc57d82d32f6adfd22137e30`。
- 522 项产物逐文件校验完成；归档 SHA256 `a93eeb9436b8ab08f3701185ee04dd9c61289605fa88126fb2c69f2ccb2fe1d0`。
- 本地：`reports/v5_scut_recovered_20260920/`。

初始 actor 的两个弧线案例均发生 4/4 边界截断；反向弧线的速度和 yaw 误差已在门槛内，
但原 4m 宽地面对应的 ±1.7m 根部边界没有为弧线轨迹及允许的跟踪误差保留足够空间。
正向弧线同时存在真实的速度/yaw 跟踪误差，不能只扩大场地就宣布通过。

训练后，站立、启停、平移等已通过案例明显退化。750 更新候选中，三个静态站立案例的最大漂移
分别为 1.39/1.08/0.75m。此证据支持增加旧任务样本以抑制遗忘，但不构成唯一学习根因证明。

## v4 修订

可执行计划：`contracts/v5_scut_skills_v4.json`，仍为 21 项技能，增加低速弧线后共 35 阶段。

1. 平地碰撞块从 8×4m 扩为 **8×8m**，实际平地边界对应变为 ±3.7m；环境间距同步增大，避免重叠。
   地面沿纵向仍分段拼接，不使用已发现接触回归的单个 80m 薄盒。
2. 新增 `curve_low`（0.25m/s、±0.3rad/s），随后保留原 `curve`（0.5m/s、±0.6rad/s）。
3. 每阶段保持一个专项主任务，同时分配 50% 环境重新采集前序技能样本；这是当前 policy 的
   on-policy 采样，不把旧 rollout 直接输入 PPO。各旧技能的地形尺寸按原合同保存。
4. 评测块缩短到 100 更新；基础漂移、高度、速度和任务成功门槛不放宽。
5. 从前轮 `accepted_policy.pt`、`--start-stage curve_low` 继续。前六阶段不重复作为训练阶段执行，
   其案例仍包含在当前与后续阶段的回归验收中。
6. 进度文件的阶段切换改为原子写入，保留真正的叶子 worker PID；子进度尚未出现时不重复累加前序更新。

critic 预热 50 次、固定学习率 3e-5、物理 1kHz/策略 100Hz、TGS 64/32、动作和气簧接口沿用 v3。
最终综合、复杂地形、空中、落地、主动跳跃与鲁棒训练仍在完整队列中。

## 恢复连接和产物

SSH 复用 socket 使用 `/home/yukikaze/.ssh/kaiser-opencode-control`，避免依赖临时目录。
部署回执记录实际远端目录、源码提交、当前阶段与回收会话。
发布候选仍通过 `artifact_selection.json` 选择 `accepted_policy.pt/.onnx`；最新训练快照不自动视为已通过。

## 正式恢复运行

2026-09-21 01:44:46（UTC+8）在 Kaiser 启动，从 `curve_low` 接续剩余 29 阶段。
冻结源码 `b10cf1c62134234c33efe36584921e681392b9cf`，256 环境，新的 72h 上限，
剩余更新预算 92500（1,136,640,000 transitions）。01:50:03 已确认完成初始评测并进入实际 PPO：
首个更新完成、12288 transitions，训练 PID1693940，处于 critic 预热阶段。

- 回执：`docs/evidence/v5_scut_v4_launch_20260921.json`。
- 远端：`/home/kaiser/robot-rl-sim60/experiments/v5-scut-v4-20260920T174442Z-c888b5`。
- 回收使用家目录 tmux socket `/home/yukikaze/.cache/robot-rl-tmux.sock`，会话 `v5-scut-v4-recovery-20260921`。
- 验证：70 项本地测试和部署提交 GitHub CI `35526840668` 均通过；64 环境短 PPO 实际更新 actor 并通过 ONNX 校验；18 案例评测流程完成，
  无边界截断，原有基础案例均通过，尚存真实弧线跟踪误差待训练。

```bash
python scripts/check_chassis_remote.py docs/evidence/v5_scut_v4_launch_20260921.json
tmux -S /home/yukikaze/.cache/robot-rl-tmux.sock has-session -t v5-scut-v4-recovery-20260921
```
