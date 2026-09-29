# V6 电机响应模型接续与复习协议

状态：2026-09-29 制定；迁移校验、成对评测比较、worker/分块/全课程/远端 launch/评测入口已接通，CPU 模拟入口测试覆盖。本文不证明任何候选响应模型已通过 bag / Isaac 验证，也不表示正在运行的旧合同已切换；不能把本文命令当作已部署回执。

用户要求是保留已学习进度，在经验证的响应模型下复查和必要复习；不改 35D actor / 81D critic / 六轴 action，不重建网络，不清零消费。旧任务源为 `1f9a884368e772741c269567a1b4b3c21f371349`，Kaiser 目录为 `/home/kaiser/robot-rl-sim60/experiments/v6-new-asset-20260928T190254Z-b07eff`。该任务随后在累计消费 320 次更新后停止：回滚入口错误拒绝了 adaptive LR 相对初始 LR 大于 1 的合法比值。已修复该操作错误；本次接续选择其保留的 200 更新完整学习状态，实际已消费仍为 320，S1 剩余 680、全程剩余 8180。具体 checkpoint 与新部署源码 SHA 另由启动回执封存。

## 1. 先确认模型，再切换训练

1. bag 区分自由运动、气弹簧负载与相对膝止挡接触。四主动轴连续，不能用主动髋角越界替代相对膝几何检查。悬空辨识保留固定车体、重力、气弹簧和双轴闭链。
2. 在同一轨迹、同一 PC 控制协议下比较原模型与候选模型，并留出独立时间段或整轮 bag 验证。PD=160/2.5、轮输出速度 P=0.6、1 kHz PC 更新、50 Hz 目标更新不因拟合而悄悄改变。辨识的反馈与力矩参考端必须与训练一致。
3. 在 Isaac 中复现同一命令序列，检查目标、原始与展开后的编码器角度、速度、请求力矩、实际施加力矩。若模型只改善 MuJoCo 而未改善 Isaac，就不加入正式训练。
4. 将实际采用的参数、bag/分段清单、旧/新误差、Isaac 回放、控制合同及源码各自封存。拟合范围之外的高速力矩包络、温度和峰值持续时间仍保留为未验证范围。
5. 完成后才在旧任务更新边界封存，将新模型写进新的合同与独立实验目录。不要修改运行中进程的配置文件，不重写旧 checkpoint 的 `.contract.json`。

## 2. 什么进度保留，什么必须重新证明

| 状态 | 处理 |
|---|---|
| Actor、critic、探索参数、Adam 一阶/二阶矩 | 默认逐张量完整恢复；启动前执行 `verify_learning_state_restore` |
| 学习谱系计数与阶段课程时钟 | 从封存 checkpoint 恢复；不因换模型归零 |
| 已消费优化更新和 transitions | 从实际消费账本恢复，包含回滚掉的更新；不得退款 |
| 当前环境状态、rollout、GAE | 丢弃；在新模型下重新 reset 和采样，禁止一个 PPO batch 混入两套 plant |
| 电机滤波器、延迟队列等动态状态 | 随 fresh episodes 按明确初值重新初始化，不沿用旧 plant 状态 |
| 旧合同的通过记录 | 只作历史与复查清单，不能标为新模型通过 |
| 新模型的保护基线 | 同 checkpoint 在新模型、两个固定 seed 下重测后建立 |
| 旧通过、新模型不通过 | 标为 `response_recovery_case_names`；成为待复习目标，不能用旧成绩抵扣 |

“精度未达标”单独报告。某 case 的 survival、mechanics、tilt 仍通过，只有高度、漂移或速度误差超标时，不能改写为“该能力没学会”。完整验收通过数与能力表现、指标变化分别展示。

消费例：若加载的 checkpoint 谱系为 360，切换前实际已消费 372，接续从 360 的权重开始，但预算从 372 开始；S1 的 1000 上限只剩 628，不是 640。前 12 次被舍弃的工作仍记账。完整训练 8500 上限与各阶段上限不自动延长。

## 3. 复习由成对结果决定

使用**同一个封存 checkpoint**在原模型与新模型各运行 primary + confirmation，固定 case、seed、episode 数、命令和评测阈值相同。报告 2×2 组完整 SHA，不比较“旧模型 320 checkpoint”与“新模型训练了 40 次后的 checkpoint”来冒充纯 plant 对照。

`assess_response_retention` 接收四份 `evaluate_chassis.py` 完整报告，并检查同 checkpoint、两 seed 配对、同 suite、完整覆盖和逐 case 判分一致性。输出：

- `new_confirmed_case_names`：两 seed 都在新模型通过；仅这些可成为新的已确认通过记录。
- `response_recovery_case_names`：指定复查目标中，旧模型两 seed 通过而新模型未通过的项目。
- `precision_not_yet_met`：基本检查仍通过、精度检查失败的项目及具体检查名。
- `basic_or_mechanical_check_failures`：存活、几何、倾角或数据完整性失败，不能掩盖成精度差。
- 原/新模型的高度、速度、漂移、停车峰值等逐项均值与差值；两 seed 的小样本均值不是置信区间。

当前 S1 的站立 45%、高度 25%、平移 20%、转向 10% 本身覆盖已经教学的行为。因此第一轮最小接续可以保留该分布，在新响应模型下继续一个 40 更新 block，作为适应与复习，避免把拟合变化和课程/reward 调整同时混在一次实验中。四份基线和新模型评测先决定是否需要额外复习，不凭“曲线改了”自动给固定 200 次 warmup。

建议冻结的有限复查窗口是最多 2 个现有 40 更新 block，并且计入剩余 S1 预算；不足 80 时按剩余预算截断。每个 block 结束重测同一新模型基线，不能因为成绩不佳补回预算。若新模型未损失旧已确认能力，首个常规 block 已承担适应，不增加专门复习预算。若损失持续，保留当前进度与结果，先诊断 plant/控制尺度/值估计，不能自动进高难度阶段、无限重复或恢复旧评测合格标签。

对后期阶段，复习池须来自已经教学的行为族，不把未训练过的跳跃、高台阶偶然通过当成一项必须立刻恢复的成熟技能。当前门槛/保护范围是否放宽是独立课程变更，不能混入只改电机响应的迁移白名单。若决定改变复习配额，应显式冻结第二份课程修订并重跑设计表，复用现有 `apply_protected_rehearsal`，而不是在 runtime 静默修改 `scene_groups`。

## 4. Critic 和 optimizer 不凭直觉重置

动力学改变会改变价值估计，但“动力学变了”本身不证明必须清空 critic 或 Adam。默认完整恢复，保留前期学习；在相同回放任务中观察 value loss、returns/value 偏差、KL、entropy 和固定行为指标。

若确实出现显著 value 失配，可另做从同一个 checkpoint、同一个新模型出发的短对照：完整恢复 vs actor 冻结的 critic 适应。必须给出实际行为收益后再采用，不能只凭 value loss 下降裁决。该对照应单独记账，区分 critic-only 与 actor 更新；现有 V6 的“每次完成更新都有 actor 更新”断言不能为了跑通而删除。

当前默认接续 **critic_warmup=0**，不新建模型、不重置 optimizer、不降低学习率。若需要改变这些设置，单独记录和验证，不能把 `--transfer-actor-only` 当成保留完整学习进度。

## 5. 实现审计与接线点

本轮新增的纯辅助模块：`src/wheeled_tasks/chassis/response_migration.py`。

```python
validate_response_resume(source, target,
    source_control=source_control, target_control=target_control)
assess_response_retention(old_primary, old_confirmation,
    new_primary, new_confirmation, required_cases)
```

`response_migration` 为 `v6_actuator_response_v1`。新模型使用顶层 `actuator_response` 且必须明确 `enabled: true`。校验拒绝随迁移改变 reward、评测、预算、网络、采样池、课程时钟或 controller gains。控制合同身份改变时必须提供展开的新旧 control；只允许明示 armature 和辨识说明改变，PD、时钟、减速比、软件力矩上限严格相同。外层调用者仍负责实际文件 SHA、bag 验证回执和设计预检，辅助校验器不能证明模型已辨识成功。

入口审计和实现点如下；列出的行号是实施前定位，实际新版本行号会顺延：

| 位置 | 现状和必须处理的点 |
|---|---|
| `policy_transfer.py:60` | `validate_physics_resume` 只允许旧的 1 kHz→200 Hz 时钟迁移，不能冒用来改响应模型；新增专用 `--resume-response-change` |
| `train_chassis.py:341` | 将 response migration 加入互斥分支，并认证旧 checkpoint→旧合同→旧 control SHA |
| `train_chassis.py:348` | 当前跨合同恢复要求 `control_math_sha256` 与新 control 相同；response 分支必须改为先认证旧 hash，再逐字段验证展开 control，不能直接跳过 provenance |
| `train_chassis.py:367` | 复用 `runner.load`、`verify_learning_state_restore`、`resume_budget`；保留真实消费与完整优化器 |
| `train_chassis.py:419` | 不引入无依据的 critic-only warmup；原合同从 0 开始的课程时钟不能重新启动 |
| `run_chassis_blocks.py:387,503` | response 也应属于 `migrating`，按新 plant 对同 checkpoint 建 baseline 和 confirmation；不能复制旧评测作为新基线 |
| `run_chassis_blocks.py:536` | 目前仅 `clock_changed` 才把新基线丢失的旧通过移入恢复目标；本次 dt 相同也必须执行对应迁移逻辑 |
| `run_chassis_blocks.py:328` | 回退 checkpoint 必须属于新模型合同；原 plant baseline 不能当作新合同 full-state 恢复点。新 worker 的 `initial_state` 封存后再注册为恢复点 |
| `run_chassis_blocks.py:554` | 已有 migration 情况下若 baseline 已通过，`self.args.transfer` 可能为 `None`；应使用 `baseline_source`，并避免直接把旧合同 policy 当新合同正式产物 |
| `run_chassis_blocks.py:627` | 新资产流程强制 `actor_completed == completed`；本次保留该断言，不滥用 critic-only warmup |
| `run_full_chassis.py:143` | 仅第一恢复 stage 传 response flag；后续 block 同合同普通 resume，后续 stage 使用既有迁移逻辑 |
| `preflight.py:30` | 新增 actuator model/identity 的设计表逐字段对照、runtime 实启探针；训练、baseline、quick、full、confirmation 五种角色全部覆盖 |
| `evaluate_chassis.py:167` / `transfer_actor_state:17` | ABI 校验含 `control_math_source`，control 文件改变后旧 actor 评测需要专用具名 response 验证，不能绕开 checkpoint 来源检查 |
| `launch_v5_remote.py:66,142` | 远端 freeze/launch 也传递 response flag，记录新旧源码、contract、control、checkpoint 和 response profile hash |

特别注意：首次新模型 worker 保存的 checkpoint 应包含新合同身份和继承的 actor/critic/Adam，但原 checkpoint sidecar 原封不动保留。用新文件名封存新的 provenance，不“重贴标签”。

## 6. 可执行评测与接续形状

下面变量须来自封存回执。四份评测已可用现有命令执行；若 control 身份变化，必须先完成第 5 节所述 ABI 验证接线，否则现有 evaluator 正确地拒绝跨控制身份。不要跳过该拒绝。

```bash
# 在 Kaiser 冻结的新源码目录；CHECKPOINT 指旧任务实际封存的同一文件。
python -B scripts/evaluate_chassis.py --contract "$OLD_STAGE" --checkpoint "$CHECKPOINT" --role full --seed 290929 --no-traces --output "$REVIEW/old_primary"
python -B scripts/evaluate_chassis.py --contract "$OLD_STAGE" --checkpoint "$CHECKPOINT" --role confirmation --seed 290930 --no-traces --output "$REVIEW/old_confirmation"
python -B scripts/evaluate_chassis.py --contract "$NEW_STAGE" --checkpoint "$CHECKPOINT" --role full --seed 290929 --no-traces --output "$REVIEW/new_primary"
python -B scripts/evaluate_chassis.py --contract "$NEW_STAGE" --checkpoint "$CHECKPOINT" --role confirmation --seed 290930 --no-traces --output "$REVIEW/new_confirmation"
```

比较四份结果时 `required_cases` 使用当前阶段明确要求加上已教学、需保留的 case 名；缺项即无效，不能跳过。全 239 case 可以保留观测结果，尚未教学的能力不因失败被描述成“遗忘”。

接续命令如下。CLI 已接通；仍须先完成拟合/Isaac 验收、新合同五角色预检、提交与冻结，之后才执行：

```bash
python -B scripts/run_full_chassis.py \
  --contract "$NEW_PLAN" --resume "$CHECKPOINT" --resume-response-change \
  --start-stage S1 --consumed-updates "$SPENT_S1" \
  --global-actor-update-offset "$SPENT_PRECEDING_STAGES" \
  --num-envs 20000 --seed 290927 --research --device cuda:0 \
  --max-runtime-seconds 604800 --publish-state --run-dir "$NEW_RUN"
```

`SPENT_PRECEDING_STAGES` 对当前 S1 为 0；不是把当前 S1 的消费再加一次。`SPENT_S1` 为旧 supervisor 的实际账本，不能仅从所选旧 checkpoint 推断。新任务每次追加的优化都继续计入全程预算。采用独立 tmux 会话和冻结源码目录；不把 TensorBoard 或 SSH 转发作为生命周期依赖。

分块入口会在第一次新模型 PPO 更新前自动执行四组基线，并写 `response_retention_comparison.json`；无需再手工复制旧成绩。每个后续 block 在新模型两 seed 评测后写 `response_review.json`，分别记录已确认恢复、仍待恢复、模型变更后的真实消费量。旧合同的文件和 sidecar 不被修改。本文建议的两 block 复查窗口用于诊断，当前实现仍按原 S1 预算/晋级条件推进，未偷偷加入新的提前停止阈值。

## 7. RMCS 自起后的接管姿态也需覆盖

RMCS 当前是状态机插值自起→200 ms 力矩混合→RL 后 1 s 保持；RL 不能据此声称学会任意倒地姿态自主恢复。已有旧资产自起测试存在未通过姿态，不能把“已实现状态机”解读为所有初态可用。新 V6 与新电机响应仍需重新验证交接。

本轮不修改自起算法。待有实际或经验证 RMCS 仿真的交接记录，应保留姿态、髋膝轴角、相对膝角、关节速度、车体速度、上一个目标/力矩以及混合阶段，作为**可达的完整接管状态**回放；不可任意拼接各轴极值制造违背闭链约束的姿态。先重放 200 ms 混合与随后 1 s 的保持，在新 plant 观察力矩、止挡接触、漂移和存活，再决定加入针对性的复习出生池。当前仅有普通 S1 case 的对照，不能代替这种非标准接管验证。

没有接管状态 bank 时应明确记录缺口，不声称已完成；普通 S3 的速度扰动 reset 也不等于完整非标准自起接管复习。
