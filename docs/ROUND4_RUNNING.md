# Round4 运行记录

## 已经开始的这轮

- 启动时间：北京时间 **2026-09-15 06:09:08**。
- 冻结训练代码：`fb334c4f96a098ce398f0c94754021fe7c21d0d4`。
- 使用旧七刚体串联等效研究模型，资产manifest为`df5ca769…b364886`；新15组件几何不参与本轮物理解算。
- 从零初始化，seed44，**1024环境、30000更新、48步rollout**，没有加载旧checkpoint。
- 训练PID：`247538`；tmux：`round4-full-dca9a609acac463594b4a49d7c001a9e`。

此前的零点等待修复资产任务已按用户新决定取消。原定零点是本次启动的最早时间，代码验证完成时已过该时刻，因此直接启动，没有顺延到下一晚。

## 路径

```text
/home/kaiser/robot-rl-sim60/experiments/round4-full-fb334c4f96a098ce398f0c94754021fe7c21d0d4/
  isaac_wheeled_rl_train/       # 本轮实际运行的冻结代码
  snapshot.json
  formal/
    audit/train.log
    audit/train.started.json
    audit/worker.status.json   # worker结束后生成
    train/                    # checkpoint、event和最终产物
    evaluation/               # 完成后的自动评估
```

本机回收位置：

```text
/home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round4_full_20260915/
```

同级的`.plan.json`、`.submission.json`和`.watcher.log`记录部署及回收。后处理工具可独立更新，训练端继续使用上述冻结版本。

## 已观察到的实际执行

正式启动检查中，摩擦桶的物理读回通过。TensorBoard已记录真实推扰事件，请求和实际速度增量的聚合值一致，`Push/readback_error_max`为0；每次写入同时检查竖直速度、角速度和位姿保持。两项抑振奖励也有实际非零记录。

06:22左右已到iteration97，随后`model_100.pt`落盘。此时仍是从零学习的早期：最近20更新reward约35.4，高度MAE约0.077m，不能宣称已经站稳或收敛。

Kaiser同时在运行另一个Transformer架构实验。本轮当时约1.1万transitions/s，WSL可用内存约2.2GiB、swap未使用。不能将旧轮独占时约2.8万的速度直接作为本轮预计时间；并发情况变化后应重新估时。

## 实际启用与后续评估

本轮启用：启动时摩擦随机化、成功PPO更新驱动的命令/推扰课程、30/20/30/20命令桶、车体roll/pitch角速度惩罚、全零速度命令下的弱轮速死区正则。

地形、台阶与跳跃尚未启用，属于后续阶段。最终自动评估使用同一冻结代码中的`scripts/round4/evaluate_policy.py`：四高度站立、四高度单次推扰，以及双向3m/s直行和双向6rad/s旋转，共12 cases。评估执行完成与行为达标分别记录。

远端worker结束后启动评估，不依赖本机在线。本机watcher校验并回收模型和评估报告；它等待远端评估提交完成后再查询结果，避免重复提交竞争。本机重启后需恢复watcher，远端训练和报告文件保留。

## 代码检查与提交

部署前完整相关CPU检查 **1062 passed**，GitHub CI也已通过。没有额外运行PPO快测。README已更新为当前使用说明；`.gitignore`排除运行目录、日志、checkpoint、缓存及本地环境，机器人资产、合同和小型证据文件保留。
