# V5.4 完整训练部署回执

- 训练方案：[全范围稠密训练](V54_FULL_RANGE_TRAINING.md)。
- 对接文档：[模型结构、输入输出与使用方法](V54_DEPLOYMENT_INTERFACE.md)。
- 机器可读回执：[evidence/v54_launch_20260923.json](evidence/v54_launch_20260923.json)。

## 运行身份

- 正式启动：2026-09-23 **07:31:11，北京时间**。
- 运行commit：`ed90d8efe4bc6b1e3010c61a577059b54329d3aa`。
- 训练实现commit：`e6c31de`；`ed90d8e`补充128／6144环境的逐组覆盖验证。
- 合同：`contracts/v5_full_range_v54.json`。
- 冻结源码归档SHA-256：`56257967c08e4cc38c616cda4308ff9d32e5002495687aaf7b94f7e9664ddb14`。
- 远端根目录：`/home/kaiser/robot-rl-sim60/experiments/v5-scut35-20260922T233106Z-be2cfc`。
- tmux：`v5-scut35-20260922T233106Z-be2cfc`，socket为`/home/kaiser/robot-rl-sim60/tmux.sock`。
- 6144环境，从头训练；总预算34668次实际更新（52000参考更新），最长7天。
- 五阶段：完整平地域 → 落地 → 跳跃 → 地形 → 鲁棒混合。

部署先commit再冻结，未使用工作区overlay。本文和对接示例是部署后的交付记录，不改变正在运行的训练源码。

## 验证与实测

- 101项相关训练／课程／几何测试通过；NumPy部署接口另有6项训练侧对照测试通过。
- 首次64环境探针触发空采样组检查，在训练前退出；改用128环境后全部任务组均有实例。
- 平地128环境完成60次PPO且actor确实更新，指令高度已覆盖0.21–0.35m。
- 平地ONNX对照最大绝对误差`4.768e-7`，最大闭链间隙1.922mm。
- 混合128环境完成8次PPO／critic预热更新，导出通过，最大闭链间隙1.292mm，无资源门限中止。
- 这些从头初始化探针存在预期的失败episode，验证的是工程链路，不是技能通过。
- 对接示例已加载真实导出ONNX，验证SHA、35D float32输入、6D输出及动作解码。

08:01:22查询时，正式任务累计277次更新、40845312 transitions，已进入`block_001`；
supervisor、实际worker与tmux均存活。5批83条文件记录已校验回收，10／100／200次checkpoint均保存actor、critic与优化器。

首次250更新固定评测为**1/54**，当前处于从头学习的早期阶段，尚无完整能力验收结果。
模型发布仍依据阶段验收和`artifact_selection.json`，不以ONNX导出成功替代技能验收。

## 查看和回收

TensorBoard：<http://127.0.0.1:6006>，使用`current/`查看本轮；`baseline/`、`v52/`、`v53/`保留历史对照。
服务在Kaiser的`v54-tensorboard`独立会话内运行，本地通过`v5-tensorboard-tunnel.service`转发。

```bash
python3 -B scripts/check_chassis_remote.py docs/evidence/v54_launch_20260923.json --brief
python3 -B scripts/sync_chassis_batches.py docs/evidence/v54_launch_20260923.json \
  --output reports/v54_batches_20260923 --once
```

本机重启后如转发服务不存在，可重新建立：

```bash
systemd-run --user --unit=v5-tensorboard-tunnel \
  --property=Restart=always --property=RestartSec=5 \
  --setenv="SSH_AUTH_SOCK=${SSH_AUTH_SOCK}" \
  /usr/bin/ssh -S none -N -T -o BatchMode=yes -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:6006:127.0.0.1:6006 -p 2222 kaiser@192.168.64.234
```

训练、封存、远端TensorBoard和本地转发的生命周期相互独立；已验证本地转发消失时远端训练仍持续推进。
