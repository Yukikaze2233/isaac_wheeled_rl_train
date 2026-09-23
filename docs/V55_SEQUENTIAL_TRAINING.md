# V5.5：单任务主训、并行定高与逐步组合

**当前执行方向已改为[V5.6单阶段自适应续训](V56_ADAPTIVE_TRAINING.md)**；本文保留已验证但未部署的串行备选。

执行计划：`contracts/v5_sequential_v55.json`。状态：本地方案与CPU验证，尚未部署或证明收敛。

用户随后要求评估续训：增加`contracts/v5_sequential_transfer_v55.json`作为actor迁移入口。
从同一固定评测比较的兼容候选中选择起点；首阶段也采用3e-5和50次critic预热。
若初始actor在两个固定seed上均通过站立，直接登记通过并进入并行定高阶段。
其余采样、案例和阶段门与原计划一致；不同分布之间不恢复旧优化器。

## 1. 结论与依据

V5.4在7500更新时仍为2/54通过，完整时域完成率明显低于早期。
应转向有明确归因边界的任务分解；当前数据不支持把继续同分布训练视为可靠的恢复方案。
这里的“单任务”指一个控制任务族，例如定高直线速度跟踪；正反方向和速度档位是该任务的采样点。
训练同一个actor，逐阶段学习和复训，不把多个独立策略直接拼接。

用户最新方向落实为：

1. 不同并行环境采样不同高度，episode内保持目标不变，先建立不同构型下的稳定驻留。
2. 名义高度平移／旋转各自主训，保留独立名义高度组。
3. 原地连续升降单列；随后重新主训平移／旋转，检查保留能力。
4. 多高度运动先学定高组合，再学移动中的高度过渡；后者不是前者自动成立的能力。
5. 启停、弧线、蛇形、旋转平移、恢复、跳跃、地形逐项扩展，最终做全域巩固。

### 华南虎参考的准确边界

来源：

- [华南虎开源文章](https://bbs.robomaster.com/article/1942489?source=1)，主要参考4.3奖励设计、4.6数据与泛化。
- [训练仓库](https://github.com/scutrobotlab/wheeled-legged_RL)，本地核对commit：`b8ff79f3df855faf9dc92f4a282bd80c42649466`。
- `pretrained/26_infantry/readme.md:39-47`明确列出平地模型向复杂地形／旋转平移的继续训练链路。

文章支持稠密过程参考、先定性后定量、checkpoint继续训练与探索覆盖；没有公布本方案这样的逐阶段自动验收队列。
本方案的阶段数、70/30分配、预算和回归门属于V5工程选择。

**不能把`default_height_cmd=0.22`解释成所有训练高度恒为0.22m：**

| 参考位置 | 实际含义 |
|---|---|
| `wheelbipe_V14/env_cfg.py:502,564` | 默认0.22；同时配置`height_range=[0.20,0.42]` |
| `wheelbipe25_v3/env.py:1321-1346` | 有范围则均匀采样；只有范围缺失才回退默认高度 |
| `wheelbipe_V13/env.py:157,358` | reset／命令重采样调用高度采样器，V14继承该路径 |
| `wheelbipe_V14/env_cfg.py:1046-1083` | 关闭的是额外正弦／阶跃波形，不是基础随机高度 |
| 公开轻量化flat `params/env.yaml:1525-1528` | 默认0.22，同时保存范围0.20–0.42 |
| 公开轻量化rotation_translation `params/env.yaml:1771-1774` | 同样保存默认0.22和范围0.20–0.42 |
| `wheelbipe25_v3/env.py:1251-1260` | 状态机输出的有效高度同时进入奖励与策略观测 |

因此，状态机生成目标不等于“不经过学习”。目标轨迹可由脚本定义，实现闭环跟踪的动作仍由策略学习。
并行定高覆盖可建立高度条件化，但高度输入通道存在不保证未训练构型的泛化；目标高度也不唯一决定左右腿构型。

## 2. V5.4复盘：事实、口径与假设

运行：`v5-scut35-20260922T233106Z-be2cfc`，冻结commit `ed90d8e`。
原始记录位于远端`train/stage_00_ground_full_range/evaluation_*/candidate_00.json`。
紧凑核对回执见[`evidence/v54_7500_audit_20260923.json`](evidence/v54_7500_audit_20260923.json)。

| 更新 | 通过/54 | 出现物理失败的案例 | 出现边界截断的案例 | 零有效帧案例 | rank第二分量 |
|---:|---:|---:|---:|---:|---:|
| 250 | 1 | 0 | 4 | 0 | 0.0741 |
| 500 | 9 | 2 | 6 | 0 | 0.1343 |
| 750 | 8 | 0 | 8 | 2 | 0.1481 |
| 3000 | 2 | 9 | 13 | 8 | 0.4074 |
| 5500 | 1 | 11 | 18 | 10 | 0.5370 |
| 6500 | 6 | 18 | 11 | 9 | 0.4907 |
| 7000 | 7 | 19 | 10 | 11 | 0.5093 |
| 7500 | 2 | 17 | 10 | 12 | 0.5000 |

计数按“至少一个episode触发”计算，不同原因可在一个案例中并存；rank是三元组，第二分量是平均未完成时域比例，不是死亡案例数。
500更新曾9/54，6500和7000也曾达到6/54和7/54；趋势依然不佳，但不能写成“从未回到6个”或“死亡单调增加”。

- 7500的通过项仅`rotate_1`、`rotate_1rps`。stand高度MAE 2.063mm，但漂移门未过。
- 名义高度`forward_05`在750更新的速度MAE为0.05979m/s，到7500为2.25345m/s，四次均边界截断；不是第一帧翻倒。
- `backward_05`对应0.08728→3.21218m/s，末次四次物理失败，仍有25个warmup后有效帧。
- 原始名义高度案例仍固定0.305m，跨高度案例另外新增。不能把所有变化解释成“评测换了一道题”。
- `frames=0`表示没有warmup后的有效样本。物理失败、边界截断与warmup长度必须分别看；非轮触地终止还有0.2秒起始保护。
- V5.3从原12486-update候选迁移actor，关闭性能课程；正式250更新结果为6/27，不能当作scratch的8/27去比较30倍训练效率。
- V5.4总采样权重65；`height_full`及两驻留组18/65=27.69%，全部height组24/65=36.92%，stand另占4/65。
  单个普通组1/65=1.54%；旋转包含多个档位与反向组，不能只算一份。每组实际有效帧／完成episode仍需另计。
- 高度平方项是每秒密度：`-(10e)^2`最后乘0.02。e=30mm时是−0.09/s、−0.0018/策略步；约95mm才是−0.9025/s。
- 速度MAE没有方向信息，且训练命令还经历幅值采样和slew。不能用“目标值＋MAE”还原实际速度，或据此断言所有任务执行成3m/s。
- 负平均奖励不等于负advantage，也不能直接证明共享网络梯度污染。PPO优化的是相对价值基线的优势；容量不足与任务负迁移都尚无隔离证据。

最值得优先验证的假设是任务分配与耦合难度，而不是先扩大网络或盲目降低终止代价。
高速度处是否受电机包络限制，应看实际转速、饱和比例、力矩余量与接触；命令域宽比例本身不是不可行证明。

## 3. 并行定高的实现

### 独立驻留任务

`height_parallel`：20%名义0.305m，20%低端0.21m，20%高端0.35m，40%在0.21–0.35m均匀采样。
在reset时采样，每个episode内保持指令不变。环境之间并行覆盖，下一次reset可以重新抽取。
不在仿真中途把机器人写入目标姿态；从现有合法nominal reset到目标构型的暂态也需要策略控制。

固定验收包括210/250/305/320/350mm五个驻留点，每个18秒、前6秒warmup、高度MAE≤5mm，原漂移／倾角／存活要求保留。
“目标抽到了低位”不能替代“机器人实际蹲到低位”；新增按目标高度分层的实际跟踪统计。

### 三种训练组同时可追踪

- `forward_05`等原组：始终保留名义高度，作为独立排练分布。
- `forward_05__height_train`等组：到`height_locomotion`阶段才出现，episode内定高；40%端点＋60%区间均匀采样。
- `forward_05__height_transition_train`等组：下一阶段才引入，6秒三次平滑高度过渡后驻留，保留解析竖直速度参考。

原地升降继续由`height`／`height_full`组负责。移动中变高的训练与验收单列，避免用“定高会了”推定“过渡会了”。
命令速度仍经0.6m/s²、yaw经4rad/s²的slew；每个速度任务阶段覆盖完整档位和方向，没有给高速组永久降目标。

## 4. 阶段、排练与预算

| 主训阶段 | 6144环境实际更新上限 | 新能力 |
|---|---:|---|
| stand | 1000 | 名义高度站立 |
| height_parallel | 3000 | 多环境定高驻留 |
| translation_nominal | 4000 | 名义高度前后0.5/1/2/3/4/5m/s |
| rotation_nominal | 3000 | 名义高度正反1/4/2π/8/4π/6π rad/s |
| height_servo | 3000 | 原地连续升降及端点驻留 |
| translation_revisit / rotation_revisit | 各1000 | 升降训练后再次集中训练原运动能力 |
| start_stop_nominal | 2000 | 低速及高速启停 |
| curve_nominal / weave_nominal | 各1334 | 弧线／蛇形分别主训 |
| spin_translate_nominal | 3000 | 参考系旋转平移 |
| height_locomotion | 3000 | 不同固定高度下运动 |
| height_transition_locomotion | 3000 | 移动中的高度过渡 |
| ground_consolidation | 1000 | 全部地面技能巩固 |
| push_recovery / landing | 1000 / 1667 | 推扰恢复／被动腾空与落地 |
| jump_in_place / running_jump | 3334 / 2000 | 原地跳跃／运动中跳跃 |
| slopes / rough | 1667 / 1000 | 坡面／粗糙地面 |
| step_up / step_down / stairs | 各1667 | 上阶／下阶／楼梯 |
| mixed_robust | 2667 | 综合与扰动 |

总计24阶段，73500参考更新，6144环境为49004次实际上限；整轮最长7天。
这些是预算上限，不是必须消耗的训练量。一般每250更新评测，达到250实际更新并通过双seed固定验收即可晋级；最终阶段最低500更新。

- 第一个站立任务从头训练，固定LR=1e-4；后续只迁移**已验收actor**，新critic／优化器、LR=3e-5、50次critic预热。
- 每个新增或复训阶段70%主训、30%所有已学组重新采样；这是on-policy排练，不是使用旧PPO rollout。
- 分配发生在方向／驻留派生组展开之后。合并巩固阶段按组均匀采样，鲁棒阶段另外保留5%材质场景。
- 前序固定案例全部标为anchor；迁移前检查anchor，连续两块失去已学判据就停止并保留原模型。同合同resume也保留该保护。
- 每阶段验收只包含已学及当前任务。最终保留V5.4原54个地面案例和146个综合案例的目标／门槛，另外增加五档定高与八个定高运动案例及对应扰动版本。
- 预算耗尽或回归停机是待诊断状态，不会自动跳过验收或无限重新训练。

奖励保留当前已实现的窄核＋宽伴随项、平方误差、机械余量、跳跃／落地过程项。
本轮首先检验任务顺序与采样结构；不同时引入未经隔离验证的高度—速度门控、辅助力或网络扩容。
若后续做松核→紧核，核宽与权重应作为独立实验；固定验收不会随reward宽松程度变化。

## 5. 新诊断与TensorBoard

`EpisodeMetrics`保留原验收计算，新增：

- `observed_frames`、`post_warmup_sample_fraction`：分清物理上采到的帧与验收有效帧。
- `terminal_episode_time_{min,mean,max}_s`、`episodes_ending_before_or_at_warmup`：分清早退与首帧失败。时间字段来自终止时episode时钟；随机起始时钟场景不应解释成自观测开始的墙钟时长。
- `vx_command_mean_m_s`、`vx_actual_mean_m_s`、`vx_bias_m_s`：有符号目标／响应／偏差。
- `vx_command_response_slope`：样本协方差斜率；恒定命令或无有效样本时为null，不是因果敏感度，也不替代固定案例的速度误差。
- `height_{low,mid,high}_{frames,mae_m,within_10mm_fraction}`：按**目标高度**分层，验证实际到达率。三段边界随报告给出，按工作区间等分；零曝光均值为null。

训练每10更新写入原RSL-RL TensorBoard的`Behavior/<group>/<metric>`；均值在当前训练块内累计，横轴为当前阶段实际更新数。
评测NPZ新增逐代表环境的`active`、`episode_ticks`、`done`、`terminated`、`commands`，能剔除首次终止后的自动reset轨迹。
`active`包含首次终止帧，与验收采样mask一致；body pose仍是原有post-step视图，可能已经reset，不能拿它代替终止前诊断状态。

## 6. 验证与执行

CPU测试覆盖：任务分配及复训70/30、旧合同兼容、名义组不被重定义、并行高度实际采样且整episode不变、
定高先于运动过渡、旧评测案例／机械ABI保留、warmup／终止mask、有符号速度和分层高度统计。

```bash
PYTHONPATH=src /home/yukikaze/isaacsim60-venv/bin/python -m pytest -q \
  tests/test_sequential_curriculum.py tests/test_full_range_curriculum.py \
  tests/test_integrated_curriculum.py tests/test_chassis_evaluation.py tests/test_chassis_blocks.py

/home/yukikaze/isaacsim60-venv/bin/python -B scripts/run_full_chassis.py \
  --contract contracts/v5_sequential_v55.json --run-dir reports/v55_contracts_new \
  --num-envs 6144 --research --prepare-only
```

正式部署前从提交版本冻结源码，分别完成首阶段、定高采样、移动过渡及mixed场景的工程探针；后期带权排练采用512环境检查全部组覆盖，不能仅按组数选128/256环境。
本地prepare与单元测试没有进行新的PhysX/PPO训练，不代表定高、速度或迁移已收敛。
新的分布使用新计划，不把V5.4优化器直接恢复到V5.5。运行身份、checkpoint来源与部署commit须写入新的回执。

### 本次验证记录（2026-09-23）

- 课程、编排、SCUT技能／观测、高速、V5.1–V5.3兼容与评测相关测试共91项通过。
- 新队列10项测试另行核验512／6144环境的逐组覆盖；最多138个训练组。
- 24份6144环境阶段合同展开并通过`preflight`，包含模型资产SHA和RSL-RL 5.5.1版本核对。
- 新代码展开V5.4 ground合同与远端冻结合同逐字段、序列化SHA完全一致：`f401986bc2fe40c2e3d397e46c494539be430bcdacde1083827d7ebee7486ea6`。
- Python语法与`git diff --check`通过。
- 12:48 UTC（20:48北京时间）查询原V5.4：7919更新、6144环境、142个封存批次，supervisor／worker／tmux均存活。
