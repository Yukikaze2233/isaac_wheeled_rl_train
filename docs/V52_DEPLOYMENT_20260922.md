# V5.2完整训练部署回执（2026-09-22）

计划与修复依据：[V52_REPAIR_TRAINING.md](V52_REPAIR_TRAINING.md)。

## 冻结身份

- 训练commit：`55a675868d66481af5e66faa3fce5406578c6d75`。
- 提交：`Add performance-gated V5.2 repair training and reward audit`。
- 执行合同：`contracts/v5_scut35_repair_v52.json`。
- 合同文件SHA-256：`adf55e47d67bb3c91b4b7123dfd8ed84d0b04cc4d8fc9524e235bfc9e06bb8cd`。
- 正式源码归档SHA-256：`5d456626a32aa5d69f424ad96573030e168019e6e1d2d8ad52e8113ec39122ad`。
- 从commit冻结，无工作区overlay；本回执提交是部署后的文档记录，不改变运行中的训练版本。

## 部署前验证

- 105项相关单元/回归测试通过；6144环境七阶段合同成功展开。
- 奖励形状调用真实reward实现，19组TensorBoard曲线已生成，数值见计划中的权重审计表。
- 同commit在Kaiser完成64环境、60次PPO工程短测。
- 前50次固定LR=5e-5的critic预热；后10次actor更新，actor权重确有变化。
- ONNX 35D→6D对照通过，最大绝对误差`4.172325134277344e-7`。
- 短测fall／非轮接触／机械越界／闭链失效计数均为0；checkpoint保存25组课程状态、64条完成episode记录。
- 自主封存的6批126个文件经归档及逐文件SHA-256校验后回收到本地。
- 短测完整编排按60次预算结束于`stage_gate_pending`，训练块为`completed`；这验证工程链路，不是技能验收通过。
- 短测回执：`reports/v52_probe_20260922/launch.json`；回收：`reports/v52_probe_recovered_20260922/`。

## 正式任务

- 启动时间：2026-09-22 21:36:00（北京时间）。
- 会话：`v5-scut35-20260922T133554Z-277650`。
- tmux socket：`/home/kaiser/robot-rl-sim60/tmux.sock`。
- 远端根目录：`/home/kaiser/robot-rl-sim60/experiments/v5-scut35-20260922T133554Z-277650`。
- 本地原始启动回执：`reports/v52_formal_20260922/launch.json`。
- 已提交的机器可读回执：[evidence/v52_launch_20260922.json](evidence/v52_launch_20260922.json)，也可直接用于查询与回收脚本。
- 6144环境、200Hz物理、50Hz策略、rollout24，总预算25336次实际更新、最长7天。
- 完整队列：flat_repair → landing → rough → spin_translate → high_speed → jump → mixed_robust。
- 起点为12486候选，仅迁移actor；正式初始化固定评测复现18/27项通过，失败项与计划记录一致。
- 21:40:29已完成首个6144环境PPO更新；supervisor、真实worker和tmux均存活。
- 21:41:58已增长至21次更新／3096576 transitions，独立封存2批；第10次恢复快照4文件已回收校验，含actor、critic、优化器和25组课程状态。
- 21:44:38累计57次更新／8404992 transitions，已完成critic预热并执行7次actor更新；各次查询连接退出后训练持续推进。

每阶段仍需固定验收过门，当前尚未完成基础修复验收。

## TensorBoard与回收

浏览器：<http://127.0.0.1:6006>。Kaiser独立会话`v52-tensorboard`，本地转发仍由
`v5-tensorboard-tunnel.service`维护；两者不承担训练生命周期。

- `prior/`：上一轮`v5-speed-resume-20260922/train`。
- `current/`：本轮正式`train/`。
- `reward_shapes/`：同commit生成的高度三档、驻留与正则切片；横轴按run名称／tag标注物理误差单位。
- 训练重点：`Curriculum/*`、`/reward/*`、`/termination/*`、`Loss/*`及独立固定评测。
- HTTP查询已确认`current/stage_00_flat_repair/block_000`及新增核宽、fall候选和机械余量指标可见。

```bash
python3 -B scripts/check_chassis_remote.py docs/evidence/v52_launch_20260922.json --brief
python3 -B scripts/sync_chassis_batches.py docs/evidence/v52_launch_20260922.json \
  --output reports/v52_batches_20260922 --once
```

训练与自动封存由远端tmux承载；客户端断开后仍运行。checkpoint恢复模型、优化器与课程状态，物理episode重新初始化。
