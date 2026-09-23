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

## 09:33运行复查：保持本轮，取消扩容

用户撤回翻倍并行请求，继续当前6144环境、原合同与原优化设置。
09:32:53检查时已完成1250次更新，并从固定评测转入下一块训练初始化。

- supervisor、实际worker及远端tmux正常；会话无附着客户端，归档器无错误记录。
- 最新1200次checkpoint已下载、验SHA并实际加载：actor/critic参数有限，优化器含17项参数状态，CPU／CUDA随机数状态已保存。
- 当前阶段合同SHA仍为`f401986bc2fe40c2e3d397e46c494539be430bcdacde1083827d7ebee7486ea6`。
- 训练吞吐约3.26万transitions/s；WSL可用内存约2.2GiB，远端磁盘可用约878GiB。
- Windows交流供电配置的睡眠、休眠超时均为0；查询未返回电池供电状态，不把配置值当作断电保证。

**运行正常不等于收敛正常。** 固定评测显示明显波动：

| 实际更新 | 通过数/54 | stand高度MAE | stand漂移 | 全范围升降MAE |
|---:|---:|---:|---:|---:|
| 250 | 1 | 20.10mm | 0.230m | 16.53mm |
| 500 | 9 | 12.80mm | 0.177m | 6.26mm |
| 750 | 8 | 32.27mm | 0.109m | 48.51mm |
| 1000 | 1 | 42.00mm | 1.598m | 67.94mm |
| 1250 | 2 | 27.76mm | 0.237m | 53.23mm |

1250次的升降案例实际高度范围约0.309–0.341m，低位覆盖仍未恢复；尚不能认为稳定收敛。
按用户要求继续该轮，每250次的固定评测和阶段门控继续执行。

连续运行保障来自远端独立进程与自动封存：本地SSH退出不影响训练。
宿主断电、关闭WSL或进程异常不属于可绝对保证的范围；当前编排没有自动异常重启功能。
实际异常需在确认旧进程退出后，从同合同最新封存checkpoint使用`--resume`恢复模型和优化器；
不能把异常恢复做成跳过技能验收门或无限重试。恢复流程见[运行与回收说明](V5_INTEGRATED_TRAINING_20260921.md)。

## 20:51正常停止与完整回收

按用户要求停止本轮，SIGTERM经supervisor传递至worker，模型保存和导出完成后退出。
最终7947次更新、1171832832 transitions，完成状态`stopped`、退出码0，145个封存批次。
完整归档844291597字节、2211文件已在`reports/v54_final_20260923/`逐文件校验回收。
归档SHA：`5fa3e277dd8a7b535ecf12b44528a56f804cdb608b6fcb36f8b6f72542e84f35`。
最终ONNX误差7.451e-7，表示导出一致性，不表示技能验收通过。
新轮采用[V5.6单阶段自适应方案](V56_ADAPTIVE_TRAINING.md)，通过候选对比选择已有actor续训。
