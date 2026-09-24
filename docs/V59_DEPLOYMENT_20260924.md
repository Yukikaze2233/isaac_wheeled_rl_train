# V5.9完整训练部署回执

训练合同、机械核对及阶段预算见[V59_FULL_TRAINING_20260924.md](V59_FULL_TRAINING_20260924.md)。
机器可读身份与核验记录见[evidence/v59_launch_20260924.json](evidence/v59_launch_20260924.json)。

## 运行身份

- 启动：2026-09-24 15:55:56 UTC（北京时间23:55:56）。
- 冻结源码：`eb4a3fd12978bda55e7636c9f95addc28cc7c302`，无未提交overlay。
- 远端：`/home/kaiser/robot-rl-sim60/experiments/v5-scut35-20260924T155549Z-4e012e`。
- 原始回执：`reports/v59_formal16384_20260924/launch.json`。
- 8阶段完整执行，合计14625更新／4374528000 transitions，运行上限72小时。
- 前五阶段16384环境，后三阶段8192环境；每minibatch49152样本。
- 策略50Hz，物理／反馈200Hz；40–110°内角，232mm气簧，0.23–0.43m完整高度目标。

## 已实际验证

16:12:12 UTC已到60更新、35次actor更新；supervisor、worker和tmux均存活。
初始化基线在Kaiser复验为clean 19/27、USB 19/27；没有物理失败，但后退3m/s仍有越界。
该基线是源策略在新资产上的复测，不是本轮训练后成绩。

第10、32、42更新快照已封存回收；加载第42快照核对：

- actor MLP参数相对于源667模型确实变化。
- actor／critic张量均有限，optimizer含17项状态。
- 前进3m/s训练边界仍为3m/s。
- checkpoint SHA为`5d23dfa3ee9dbf13a8a80cd228cccc17145c043e20497e3acc054486afd9188a`。

TensorBoard截至step52最近10次更新：吞吐中位44823.5 transitions/s，
采样时间8.563s、学习时间0.194s；这不是与旧版本的受控速度对比。
观测到GPU利用率57–89%，显存约11.2GiB，WSL仍有约9.6GiB可用内存。
目前是已实际运行的16384环境配置，尚未证明其为全局最大吞吐。

USB启用组RTT均值约74.71μs，峰值约163μs，启用环境约50.4%。
它来自分位数匹配近似先验，均值不等于用户给出的约80μs中位数。
时延逐物理步变化；进入后续阶段时启用比例提高到约70%。

## 查看与恢复

TensorBoard：<http://127.0.0.1:6006>，`current/`为V5.9，`v58/`保留上一轮。
本机持久服务`v59-tensorboard`、`v59-event-sync`、`v59-artifact-recovery`均已启用。
event每30秒镜像，封存批次每60秒回收；回收预留本机2GiB磁盘空间。
当前任务在远端tmux运行，SSH客户端退出不改变训练生命周期。

```bash
python3 -B scripts/check_chassis_remote.py docs/evidence/v59_launch_20260924.json --brief
systemctl --user status v59-tensorboard v59-event-sync v59-artifact-recovery
```

第一阶段500更新后由同一完整训练调度器自动进入flat_full，后续按合同推进。
固定评测为monitor模式；工程进程完成与模型全部能力验收分别记录。
