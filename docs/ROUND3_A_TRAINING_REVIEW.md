# Round3-A 完整回收与训练曲线审计

审计日期：2026-09-14。冻结源码：`6ed83408d1d1cb02ea687e77fa80e5c20932f36c`。

## 结论与交接

**A 的 10000 次更新已完成，最终导出通过数值核验，完整训练文件已回收并逐项通过 size/SHA256。训练 reward 后期大体平台化，末段部分 tracking/contact 指标变差；不能据此宣布确定性策略退化，也不能把 final 自动认定为 best。**

- 当前首选**曲线候选 `model_5300.pt`**：中期高 reward 窗口，较 final 的 vx/wz/高度误差、非轮净力和零平移惩罚都较低；`model_5100.pt` 是低非轮接触候选。
- **最新后期局部较优候选 `model_9500.pt`**：供独立评估与 final 对照。全程 checkpoint 对齐的最高 trailing-100 reward 是 `model_900.pt`，应保留早期参照。
- 以上是多指标描述性筛选，没有统计显著性或独立评估保证；**`policy_quality_verified=false`**。评估 agent 决定实际 best 与下一阶段 parent。
- 本次仅 CPU 数据审计和 SSH 文件回收；没有启动仿真、GPU 或训练。评估恢复代码和本地 dirty B1 源码未改动，无 commit/push。

## 1. 回收范围、完整性与时间

工作区根目录下的独立归档：

```text
/home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round3_a_20260914/training-archive/
```

远端源：`/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/`。
仅读取已完成的 `train/`（深度不超过 2）及 `audit/` 顶层常规文件；评估使用的 `audit/after_training/` 不属于本归档范围。

| 项目 | 已验证结果 |
|---|---:|
| 原 `final-return/artifacts/` | 8 项产物逐项 SHA/size 匹配远端 completion；completion 原始字节完全一致；旧 local receipt 的绑定一致 |
| 远端训练源文件 | **124 个，226,014,019 bytes** |
| 周期 checkpoint | **101 个，150,834,993 bytes**；`0,100,…,9900,9999` 全齐 |
| 最终 checkpoint | 另有 `model_final.pt`，1,497,251 bytes |
| 原始 TensorBoard event | 1 个，39,748,198 bytes |
| export logs | stdout 441 bytes；stderr 0 bytes，均已校验 |
| audit 顶层记录 | 11 个；plan/effective-command/submission、preflight log/started/status、train log/started/status、worker started/status |
| 本地完整归档含派生数据与 seal | **149 个文件，410,729,547 bytes** |
| 完成时间 | **2026-09-14 07:57:20.888252 BJT** |
| 更新数 / 实际最后迭代标签 | **10000 / 9999** |
| learning elapsed | **18023.248708 s = 5 h 00 min 23.249 s** |
| 训练子进程 elapsed | 18338.723194 s，含初始化/收尾 |
| iteration 0→9999 event 墙钟跨度 | 18020.785759 s |
| 累计 transitions / 完成收据吞吐 | 491,520,000 / 27,271.443 transitions/s |

流程：先复核旧产物 → 远端逐文件 size/SHA inventory → 新目录 rsync → 本地逐文件 SHA → 再取远端 inventory，确认整个传输前后没有变化 → 本地封存前再次复核 payload 与旧最终产物。旧已验证文件没有覆盖。

`archive_receipt.json` 记录全部 124 个源文件的路径/size/SHA；`local_archive_seal.json` 记录封存时所有本地文件（不自含 seal）。17 项 `source_files_sha256` 全部与本地 Git 对象中的冻结 commit 匹配，不以 dirty working tree 解释历史 run。此训练归档不是完整资产/依赖环境镜像，`source_hashes.json` 本身也明确了这一边界。

### 关键 SHA256

| 文件 | SHA256 |
|---|---|
| `payload/train/completion.json` | `8854d884914a0e50b88ead6e56a785068ccfa08556111ad6e479832d652b3ac1` |
| `payload/train/model_final.pt` | `6d33498d2424bb45305c199d42261d62c39c13a39107a604a89d28ccd19f4af0` |
| `payload/train/policy.onnx` | `3d8c682cfc94ac0ee1879bdc6fd90c208e123c05e6b48de3f970f42a474c1ad9` |
| `payload/train/events.out.tfevents.1789325816.Kaiser.446292.0` | `f9149700faac989603f87d9f0c87b5e3c8114c652fae984e7a7173a97583476f` |
| `remote_inventory.json` | `17ace783d6fb5a4c9d5c844f9e2fb9f816e4dfcd35e2509ca6b148c9e441b5e8` |
| `archive_receipt.json` | `ae3a844d3594fa9534ca5e88fd9c8f2631713483c029f6aa4eab4ec1dfc27199` |
| `local_archive_seal.json` | `75041398bef82d469b66b2904689b1e2f00999e30ea17002061c7ca96743701b` |
| `scalar-audit-v2/all-scalars.csv` | `74a0b9776cbc443e9065e50331d2e79493b2b57f7ab6507ae9703b20963c1e72` |
| `scalar-audit-v2/scalar-manifest.json` | `d98500d8a9a927a8345d64332641745e08d8e5ae1b52fa2ece8830a445b4a616` |

ONNX export stdout 记录 9 个数值验证样本、最大绝对误差 `5.340576171875e-05`。这是导出一致性验证，不是 policy quality 验收。

## 2. 原始 event 全量导出与统计口径

**权威派生目录是 `scalar-audit-v2/`**，保留原始 event，不生成柱状总结图或截图。

- 严格遍历 624,713 条 TFRecord：1 个 file-version，624,712 个 scalar summary；校验每条 length/data CRC32C 和完整 EOF，无静默截断。
- 全量导出 **65 个 scalar 标签、624,712 条记录**，CSV 为 92,077,210 bytes。保留来源文件、record index、byte offset、wall time、step、tag、value 和 encoding；不抽样、不去重、不平滑。
- 以 TensorBoard 2.21.0 `EventAccumulator` 独立复读，关闭 reservoir 限制和 orphan purge，65 标签的每个 `(step, wall_time, value)` 与 CSV 精确一致。
- **50 个密集标签**覆盖 `0…9999`；Train reward/episode length 两个标签从首批 episode 完成后的 `41…9999` 连续记录，各 9959 点。
- **11 个 Episode_Reward 标签**各 7716 点，`41…9999` 内按 reset 出现，不填补空缺，不把缺点当零。
- 两个 `Train/*/time` 标签的 step 是 **72…17647 秒**，不是更新次数。原始 CSV 完整保留，迭代比较排除这两个标签。
- 所有 scalar 值均有限；各标签没有重复 step 或倒序 step。

初次派生目录 `scalar-audit/` 的统计曾混入 `/time` 秒轴，已用 `SUPERSEDED.json` 明确作废；其原始全量 CSV 与 v2 完全一致。保留初次输出用于可追溯，所有本报告结论和交接只引用 v2。

以下窗口均为**截至指定迭代标签、含端点的 100 次更新**：`901…1000`、`2401…2500`、`4901…5000`、`7401…7500`、`9900…9999`。这是 update 级日志标量的算术均值，不是原始 transition 的分位数，也不是独立确定性 rollout。Train mean reward/length 本身包含 runner 的 episode 汇总，因此窗口不代表 100 个独立 episode。

## 3. 关键窗口对比

| 指标 | →1000 | →2500 | →5000 | →7500 | 最后100，→9999 |
|---|---:|---:|---:|---:|---:|
| Train mean reward | 93.34485 | 91.83505 | 93.16048 | 92.66922 | **92.06567** |
| Train mean episode length，步 | 1991.605 | 1999.000 | 1993.759 | 1995.592 | **1997.957** |
| Policy mean std | 0.224271 | 0.297404 | 0.294006 | 0.340735 | **0.376161** |
| vx MAE，m/s | 0.121004 | 0.147000 | 0.121257 | 0.125687 | **0.131308** |
| wz MAE，rad/s | 0.081706 | 0.107560 | 0.094062 | 0.099273 | **0.112868** |
| height MAE，mm | 2.439861 | 4.252351 | 2.659562 | 2.915850 | **2.992788** |
| standing fraction，% | 30.1112 | 29.9155 | 30.0338 | 29.8920 | **29.8577** |
| 非轮接触 candidate，% | 6.0709 | 6.3177 | 4.1095 | 6.8585 | **6.8899** |
| 非轮净力统计量，N | 9.50468 | 10.84444 | 8.05470 | 15.11580 | **18.75688** |
| zero-vx L1 加权每 tick reward | -0.000373869 | -0.000426007 | -0.000382063 | -0.000412267 | **-0.000444355** |
| 瞬时 tilt diagnostic，% | 0.034566 | 0.001099 | 0.009684 | 0.042236 | **0.012817** |
| sustained tilt termination，比例 | 8.138e-7 | 0 | 6.104e-7 | 6.104e-7 | **2.035e-7** |
| nonfinite termination，比例 | 0 | 0 | 0 | 0 | **0** |
| action-rate 加权每 tick reward | -0.000107085 | -0.000167171 | -0.000167515 | -0.000217829 | **-0.000266054** |
| effort 加权每 tick reward | -0.000052658 | -0.000060593 | -0.000062830 | -0.000069763 | **-0.000077364** |

最后 100 的 spin bucket 为 **9.8414%**，低/高高度端点占 **15.4432% / 14.6179%**，与 A 的 30% 站立、10% 自旋、各 15% 高度端点设计基本相符。base visual bounds ground diagnostic 为 **0.284363%**，low-height diagnostic 为 **0.028829%**；这些不能由接近满长的 episode length 排除。

### 必须保留的指标语义

- 非轮 candidate 使用 `Diagnostic/non_wheel_contact`，阈值为刚体净力 >1 N。A 沿用 v2 termination 语义，`Termination/non_wheel_contact=0` 是被屏蔽的终止项，不能充当无接触证据。
- `Contact/non_wheel_force_n` 是每个环境在非轮刚体和 contact history 上取峰值后再聚合；**不是 ground-pair 支撑力，也不是最大瞬时冲击力的全局峰值**。
- `Diagnostic/tilt` 是瞬时超过 35°；`Termination/tilt` 是 projected gravity z > -0.1 持续超过 1 s 的实际失效，两者不能互换。
- A 的 zero-vx L1 为 `1(|cmd_vx|<0.05) × (|vx|+|vy|)`，权重 -1、policy dt 0.01。最后 100 反解的**全样本门控 L1 均值为 0.0444355 m/s**。它包括纯旋转和小非零 vx 命令，不能除以 standing fraction 当作纯站立条件 MAE；原始 event 没有条件零速 L1 或世界坐标累计漂移。
- **action saturation、knee-target saturation、torque saturation 的原始 scalar 均不存在。**只有 action-rate/effort reward 代理量，不能由其反算饱和比例，也不能由 std 上升断言发生了饱和。

## 4. 平台、退化与不确定性

1. **总体平台，末段偏弱，未见灾难性崩溃。**1000-update block reward 在 3000…7999 约为 92.42–92.71；8000…8999 为 92.30973，9000…9999 为 92.25688，最后两块仅差 -0.05285。末 100 则从前 100 的 92.31257 降至 92.06567；末 1000 内线性斜率为 -1.03815 reward/1000 updates，表明局部有先恢复再走弱的结构，不能把单个末窗口当作全程单调趋势。
2. **tracking 与探索强度有后期走弱信号。**最后 1000 相对前 1000，vx MAE 0.126289→0.130233，wz MAE 0.104809→0.105625，height MAE 3.027429→3.044222 mm，std 0.344881→0.360090。末 1000 内 vx/wz/height/std 斜率分别为 +0.007114 m/s、+0.011970 rad/s、+0.275540 mm、+0.037741 每 1000 更新。末 100 height 又比前 100 的 3.182976 mm 改善，故不是所有指标同步恶化。
3. **非轮接触没有持续收敛到零。**4000…4999 block candidate/force 为 3.4193% / 6.4904 N，最后 1000 为 5.9535% / 14.4227 N，最后 100 为 6.8899% / 18.7569 N。最后 1000 candidate 的内部斜率反而为负，但最后 100 又回升，说明存在波动；净力后期增大值得独立评估检查。
4. **稳定性指标是混合变化。**末两块 episode length 1993.533→1997.230，瞬时 tilt diagnostic 0.03613%→0.01906%，有改善；全程 nonfinite scalar 为零。它们不抵消接触/跟踪的疑点，也不能证明四高度站立与慢漂合格。

这些都是随机探索训练策略在混合命令下的聚合曲线，窗口相邻且相关，没有重复种子或置信区间。正确结论是“训练曲线平台并有局部退化信号”，不是“确定性策略已退化”或“策略质量已验证”。

**R2 与 A 不可按 reward 直接排名。**R2 最后 100 的 88.9074 与 A 的 92.0657 来自不同命令/高度采样、零平移奖励和运行栈；A 改了站立占比、自旋桶、高度端点/时序及 L1 项。只有同一独立测试矩阵下的轨迹指标才适合比较策略质量。

## 5. 交给评估 agent 的 checkpoint 候选

所有路径均在 `training-archive/payload/train/`。窗口是对应 checkpoint 标签的 trailing-100，不是对该权重进行 rollout 后得到的成绩。

| 候选 | reward | vx / wz MAE | height MAE，mm | 非轮 candidate / force | 用途 |
|---|---:|---:|---:|---:|---|
| `model_900.pt` | 93.58209 | 0.124956 / 0.079517 | 2.448208 | 5.3105% / 8.3957 N | checkpoint 对齐的最高 reward 参照 |
| `model_5100.pt` | 93.23341 | 0.123105 / 0.090091 | 2.870905 | 3.1246% / 6.2482 N | 低接触折中 |
| **`model_5300.pt`** | **93.34244** | **0.123709 / 0.091582** | **2.624134** | **5.5402% / 10.6818 N** | **当前首选中期曲线候选** |
| **`model_9500.pt`** | **92.81929** | **0.128593 / 0.099967** | **3.116098** | **6.2310% / 14.8930 N** | **最新后期局部较优对照** |
| `model_9999.pt` / final 时间窗 | 92.06567 | 0.131308 / 0.112868 | 2.992788 | 6.8899% / 18.7569 N | 最终基线，不能默认 best |

选择 5300 的理由是中期 5000/5100/5200/5300 的 reward 窗口分别为 93.1605/93.2334/93.2491/93.3424，末端仍处于较优区间；同时 5300 的表中误差和接触量均优于 final。5100 的接触更低、5300 的高度误差更低，故不存在由这些曲线唯一决定的多目标最优。9500 仅作为更晚阶段的候选，避免只比较中期与 final。

```text
model_900.pt   bdd39b4fe69db8bd932b36d016c71b143ffe3a2f9510bf510ec070e791e9bf49
model_5100.pt  dc253493b505a1039be589206002eb3545d710b9de96c9fdd428b335a049d83e
model_5300.pt  09f7774a42bea49726b87a78e8d58860078c96c650f74b1a089a3c387af1cd2a
model_9500.pt  6ecf2645fa162f804301cc8a758422724987a5884e2f679e8362a14c1e79b1e6
model_9999.pt  bbfb71534789a538623c2f6307963e841b703d926733e15ff3d830725c97cab7
```

下一轮规划建议：先由独立评估对上述候选应用相同的四高度零速、慢漂/位移与接触诊断矩阵，再决定 B1 parent；若 5300/5100 优于 final，应以哈希绑定该周期 checkpoint。曲线没有提供单纯延长相同 A 任务、增加网络容量或直接扩大训练预算的充分依据。B1 的摩擦任务实现与后续晋级由主 agent 依据独立评估决定。

## 6. 可复查产物

- 小型结果：[review JSON](evidence/round3_a_training_review_20260914.json)、[20 个关键标签 × 5 窗口 CSV](evidence/round3_a_training_windows_20260914.csv)。JSON 包含候选 SHA、末 100/1000、前 1000、斜率、文件计数、时间口径和数据边界。
- 完整本地：`scalar-audit-v2/all-scalars.csv`、`scalar-manifest.json`、`window-statistics.csv`（63 个迭代标签）、`trends.json`、`checkpoint-windows.csv`、`thousand-update-blocks.json`、`independent-csv-verification.json`、`source-commit-verification.json`。
- `recovery-tools/` 保留通过 apply_patch 创建的回收/审计脚本；CPU parser 使用独立临时 venv，未向项目运行环境安装依赖。
- checkpoint、raw event、原始训练/export logs 和大 CSV 均保留在工作区外层 reports 归档；仓库此次只新增本文件与上述两个小型 JSON/CSV，无 Git 提交。
