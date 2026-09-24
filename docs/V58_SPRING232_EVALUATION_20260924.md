# V5.8收尾与232mm气簧独立评测

这是40–110°新限位提出前的独立评测记录。后续新限位＋USB比较及完整合同见[V5.9](V59_FULL_TRAINING_20260924.md)。

## 已结束的远端训练

Kaiser运行`v5-scut35-20260924T130433Z-87c442`于2026-09-24 14:37 UTC完成既定667更新，
累计196706304 transitions，22批封存，完成状态`full_training_completed`。
本机新气簧评测启动时，远端本轮已自然完成；因此当前不是两条完整训练同时运行。

同一旧资产上的27case、每case四个episode结果：

| checkpoint | 通过 |
|---|---:|
| 原12486 baseline | 18/27 |
| V5.8 250 | 6/27 |
| V5.8 500 | 18/27 |
| V5.8 667 | 21/27 |

667仍未通过`height_hold_high`、`backward_1`、`backward_3`、`rotate_1rps_reverse`、
`rotate_8_reverse`、`rotate_2rps_reverse`。原先通过的backward_3退化，21/27不等于全部能力保留。

## 本机独立评测

- 设备：RTX4060 Laptop；启动时没有其他CUDA计算进程。
- systemd user unit：`v58-spring232-eval-20260924t144156z.service`。
- 启动时间：2026-09-24 14:41:56 UTC。
- 运行目录：`reports/v58_spring232_local_20260924T144156Z/`。
- 源码、合同、模型均按文件SHA冻结在该目录，`launch.json`保存完整命令和来源。
- 类型：本地跨资产预评测，工作树内容快照；不是已提交版本的正式训练部署。
- 27case×4 episodes＝108个并行环境，依次评测三个actor：原12486、V5.8 500、V5.8 667。
- 策略50Hz、物理／反馈200Hz；本次只改变安装长度基准。
- 新资产manifest：`6f43e64723f8e348120b497c08c8f53c8090f47e89b6ea63caed844c62aabaf2`。
- 旧资产manifest：`5f88012c30ca7191ee30f6ed460bc51bdd96590b63910e119010e7b3dbb6093a`。
- 合同SHA：`170d36a5abe5f26a3b192467976843139a5ebf8a9cff974073ef09d5894a35f8`。
- 启动后已观测到PhysX关节表中的新移动副限位、存活PID190165及约2443MiB显存占用。

评测新增显式`--source-asset-manifest`选项；默认资产一致性约束仍生效。
跨资产路径验证原checkpoint、原合同、原manifest及控制数学身份，并严格保持观测／动作布局、
策略／物理周期、PD增益、关节顺序与名义零位一致。报告同时保留原始和评测资产SHA，
不会把旧checkpoint改写成在新资产上训练过的模型；跨资产评测不直接导出新的部署包。
相关评测及接口校验16项测试通过。

状态与结果：

```bash
systemctl --user status v58-spring232-eval-20260924t144156z.service
```

进度日志为运行目录的`evaluation.log`；每个候选完成后生成`evaluation/candidate_00.json`等，
全部结束后生成`evaluation/evaluation.json`。这些是固定评测验收产物，不是训练曲线。
跨GPU结果存在数值差异的可能，边缘case需同设备对照再归因于气簧改动。

## 下一轮频率决策

下一轮完整训练优先保持50Hz策略，先处理机械基准、已学能力保留与实际通信时序。
100Hz作为小预算A/B候选，保持相同资产、PD频率、延迟模型与物理时长预算，
按跟踪、抗扰恢复、力矩波动和名义能力保留判断收益，再决定是否用于Kaiser完整训练。
1kHz PD可与50Hz策略配合，不构成提高策略频率的必然理由。
