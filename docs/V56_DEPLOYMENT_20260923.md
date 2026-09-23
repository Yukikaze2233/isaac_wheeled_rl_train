# V5.6单阶段自适应续训部署回执

**最新状态：2026-09-23 22:35北京时间，539更新时触发连续两次退化保护，训练已正常结束；182文件已完整回收。**

方案：[V5.6训练设计](V56_ADAPTIVE_TRAINING.md)。
机器可读身份：[evidence/v56_launch_20260923.json](evidence/v56_launch_20260923.json)。
网络结构、35D／6D接口与PC力矩控制链见[统一部署对接规范](V54_DEPLOYMENT_INTERFACE.md)；
此前平地移动／旋转用的[12486候选ONNX包](../models/v5_flat_12486/README.md)已独立整理，可用于接口对接与历史能力复现。

## 旧轮已停止并回收

V5.4于20:51北京时间正常结束，7947更新，退出码0。
145批封存、2211文件完整回收到`reports/v54_final_20260923/`并逐文件验证SHA。
原任务的supervisor、worker和tmux均已退出，保留最终模型、优化器、评测和TensorBoard事件。

## 已验证的续训来源

同一18案例对比：V5.4最终7947与早期500模型均0/18，原12486候选7/18。
后者接口／控制参数兼容，actor参数有限，作为新轮初始策略。
来源checkpoint SHA：`e6307dd7419c052a96285e58ce6a6624b8832e6c5986aadf568287bfcfc37f81`。

128环境短测完成60次更新：50次critic预热＋10次actor更新，新critic与新优化器实际生效。
完整59案例基线和短测后均为20/59，七项保护能力全部保留。
最大闭链间隙0.807mm，ONNX最大误差4.172e-7，128文件校验回收。
`curriculum_state`版本2、45个组及184320个真实采样帧的曝光计数均核验；TensorBoard新课程／行为tag已读取。
128环境短测的等效参考更新不足100，因此尚不应触发在线晋退级；晋退级及checkpoint恢复逻辑由针对性CPU测试验证。
工程短测结果不作为完整能力验收。

## 正式运行身份

- 首次启动：2026-09-23 **21:26:56，北京时间**；容量探针后于**21:41:11**恢复。
- 训练实现commit：`94a2035`；当前冻结commit：`29323d62f83e1f169fe163959433f4239a4d065d`，后者只增加容量探针的actor迁移及宿主内存保护。
- 99项训练／课程相关CPU测试及1项容量保护测试通过。
- 当前源码归档SHA：`b3f0dca2476e06de48f21ca3ec27302a2e97ba2af2fb96c99e4c437548e77b0c`；无overlay。
- 合同：`contracts/v5_adaptive_v56.json`。
- 当前根目录：`/home/kaiser/robot-rl-sim60/experiments/v5-scut35-20260923T134101Z-207a15`。
- 6144环境、45个平地技能组并行；13000次新更新上限、最长7天。
- actor从12486次候选迁移；新轮计数从0开始，不把继承的训练量计作本轮scratch数据。
- 一个`flat_adaptive`阶段。每250次做固定59案例评测，七项保护能力连续两次回归失败则保留模型并停机。
- 课程窗口、上限、采样概率、revision和曝光计数进入恢复快照；首次10更新及每100更新封存。

## 容量实测与连续恢复

按用户要求尝试进一步利用Kaiser主存／显存。6144环境时，WSL可用约2.4GiB且已有swap使用，
Windows可用物理内存约1.35GiB；显存约9152MiB，说明主存比显存先成为限制。
先在第39次更新正常暂停，回收103文件并验证actor、critic、优化器和课程状态。

8192环境使用相同初始actor进行独立容量探针，144.9秒内触发`host_memory_guard`：
宿主可用主存最低1009188KiB（约0.96GiB），WSL可用最低2647988KiB，显存峰值8808MiB。
探针在PPO更新开始前停止，**没有稳定8192吞吐数据，不能宣称其更快或更慢**。
30文件回收，证据：`reports/v56_capacity8192_recovered_20260923/artifacts/train/report.json`。

据此选择已经验证的6144环境。从第39更新使用同合同`--resume`恢复，
合同SHA始终为`26494ef4924a48e1e93b389b3006a948f1624fc28652911ea7cf0c97bf43f749`。
暂停前critic预热阶段吞吐中位数31707.5 transitions/s；恢复后actor+critic训练最近20点中位数29569 transitions/s。
两种优化阶段和采样状态不同，这些数字不构成8192对比结果。

21:53:22检查：累计163更新，恢复后已有113次actor更新，supervisor／worker／tmux均存活。
第100更新快照已回收、验SHA并实际加载，17项优化器参数状态、45组课程状态、CPU/CUDA RNG完整，网络参数有限。
实际已有37组发生课程调整：例如正反3rps组的学习yaw上限从1.0提高到1.5rad/s；
forward_5组当前边界占比降至0.65以增加容易样本。调整数量不是技能通过数量。

21:59:13复查：累计230更新（180次actor更新）、33914880 transitions，进程正常；第200更新快照已在远端封存。

当前瓶颈是宿主主存余量；进一步扩大环境数需要先释放宿主内存。未更改WSL内存配置或重启宿主。

## 查看与回收

TensorBoard：<http://127.0.0.1:6006>。
`current/`对应恢复后的V5.6，`v56_initial/`对应容量测试前的39更新，`v54/`对应停止的上一轮；
`baseline/`、`v52/`、`v53/`、`reward_shapes/`保留。
远端TensorBoard会话`v56-tensorboard-resume`与训练独立，本地沿用`v5-tensorboard-tunnel.service`。

```bash
python3 -B scripts/check_chassis_remote.py docs/evidence/v56_launch_20260923.json --brief
python3 -B scripts/sync_chassis_batches.py docs/evidence/v56_launch_20260923.json \
  --output reports/v56_resumed_batches_20260923 --once
```

正式恢复使用同合同封存checkpoint及`--resume`；跨合同只迁移经过验证的actor。

## 539更新退化停机与回收

| 检查点 | 通过/59 | 保护案例通过/7 | stand高度MAE | stand最大漂移 |
|---|---:|---:|---:|---:|
| 初始候选／128环境短测后 | 20 | 7 | 见对应原始评测 | 见对应原始评测 |
| 289更新 | 3 | 0 | 22.78mm | 0.504m |
| 539更新 | 3 | 1 | 0.537mm | 2.305m |

539更新仅`forward_05`仍通过保护集；stand、305mm长驻留、前后2m/s、前后3m/s均未保住。
高度精度恢复不等于站立稳定：零速下vx MAE已达0.231m/s，18秒驻留漂移达4.156m。
末次59案例中有3个案例出现物理失败、17个出现边界截断；不能把边界退出全部解释成摔倒。

课程状态确实运行，但前后2/3m/s组的学习上限仍为0.5m/s。
因此，已学命令也被初学课程压低，原目标主要依赖5%的全域探测；
这暴露了续训时已学难度排练不足的问题，但不是采样是唯一根因的因果证明。

停机状态为`regression_hold_best_preserved`，不是SSH中断或进程崩溃；不将直接重启清零退化计数当作异常恢复。
完整结果回收在`reports/v56_final_20260923/`：182文件逐项验SHA，归档SHA
`47c78823c43132a8d771c9ad100a4c2d2c33b5d89cb9dabaa38a2d9281053dda`。
最终checkpoint SHA：`ae9d715ab0b030c39d2c87850de0258b8c8390f7f030d73394aefe5cc9fd6fc1`，角色仍为未验收候选。
原12486-update模型及已交付历史ONNX继续独立保留。
