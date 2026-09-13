# Round3-A Kaiser 运行交接

## 已实际启动

- 启动：北京时间 **2026-09-14 02:51:43**。
- 代码目录：`/home/kaiser/robot-rl-sim60/isaac_wheeled_rl_train`。
- 冻结版本：`6ed83408d1d1cb02ea687e77fa80e5c20932f36c`。
- 训练：**1024 env × 10000 updates**，seed43，官方RSL5 PPO。
- 初始化：经哈希核验的Round2最终actor/critic/std，fresh optimizer；不是从工程smoke继续学习。
- 训练PID：`446292`；tmux：`r3a-train-35f06fa3504049f8924f9ccd1ed02b93`。
- run：`/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/train`，日志位于同级`audit/train.log`。

PID是启动记录，后续活性以进程、原始event和完成回执联合判断。

## 启动验证与首次性能观察

2env×3更新的PPO/ONNX成功；1024env×10更新的容量验证成功，实测约27925 transitions/s。256env×10更新完成学习，但单样本CPU ONNX近零输出的float32消减导致严格数值校验失败；原失败回执保留，没有据此伪造通过。该checkpoint另在本机用未修改的exporter成功重新导出，未放宽容差，正式run使用成功的1024容量验证作为启动依据。

**03:23北京时间快照**：已记录903次更新，最近100次平均日志FPS29053，采样1.601s/update、学习0.0915s/update；按event墙钟测得1.730s/update，剩余约4小时22分。条件不变时预计07:45左右训练结束，再留15–30分钟完成四高度评估和回传。这是动态估计，不是终止时刻或收敛保证。

**03:30快照**：日志到1141/10000迭代标签，正式进程和tmux正常，未观察到OOM。WSL物理内存上限约15.19GiB，正式进程RSS约4.86GiB；更多并行任务的约束包含主机RAM和建场时间，不能只根据24GiB显存决定扩容。详见[容量检查](ROUND3_CAPACITY_REVIEW.md)。

## 自动检测和回收

- 本机模型回收tmux：`kaiser-round3-a-return`。
- Kaiser最终评估tmux：`r3a-post-eval-waiter`。
- 本机评估回收tmux：`kaiser-round3-a-eval-return`。
- 本机结果根：`reports/current/kaiser_round3_a_20260914/`（工作区根目录下）。
- 模型最终目录：`final-return/artifacts/`；评估最终目录：`evaluation-return/reports/`。

已真实完成smoke产物回收、size/SHA核对，以及1env×2步评估报告生成和回收测试。正式产物尚未完成时目录不会被伪装为最终成功。远端完成后执行0.29/0.30/0.31/0.32m固定零速测试，每高度累计60仿真秒，保留20秒timeout reset；输出高度误差、慢漂、位移、真实诊断与非轮净力候选。完整流程见[自动评估](ROUND3_POST_TRAINING_EVAL.md)。

若最终ONNX校验失败，原checkpoint仍会保存并回收，正式评估明确阻塞；可以在独立恢复目录重新导出和核验，不能改写原失败回执。

## 阶段范围

2026-09-14预算更新：第三轮总规划扩大为**100000次更新**，按A10000、B20000、C25000、D30000、E15000分配；详见[长期课程与晋级](ROUND3_TRAINING_PLAN.md#8-长周期预算与晋级)。当前进程只执行A的10000次，B–E待实现和独立验证后分段部署。

A实际训练0.29–0.32m高低站立、起停和原速度域行驶；30%站立、10%纯旋转、高度端点覆盖、单一零平移L1项、显式名义摩擦0.5。高速扩域、摩擦DR、冲击恢复尚属于后续阶段，不能将10000次A更新称为全部Round3完成。

当前run没有启用姿态发布，因此没有同一训练状态的实时GUI。Windows原生WebRTC通道已验证；V40 checkpoint原生回放入口已准备但尚未通过运行验收。它与实时训练状态是不同数据源，见[原生可视化状态](ROUND3_NATIVE_VIEW_STATUS.md)。
