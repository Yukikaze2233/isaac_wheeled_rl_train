# V6 恢复接管课程：命名修订 v8

更新：2026-10-03。**部署状态：`pending`。** 用户已授权完成验证后部署；本文记录待冻结的 v8 方案，不代表远端已切换。实际运行身份以 [launch 回执](../reports/v6_task_campaign_20261001/deploy/launch.json)、对应源修订及首次 PPO 核验为准，取得这些证据后再更新本文状态。此前 v7、9 月 29 日 V6 工程回放和 10 月 3 日 V5 自起试验分别保留原身份。

## 1. 本次解决的问题与保留范围

恢复链依然是正常 episode reset → 释放/反馈脚本翻正 → 200 ms 力矩混合 → 纯 RL 零速接稳 → 连续稳定 1 s → 原命令下站立、前进或转向。当前训练的症状需要分开观察：脚本能否交出可接管状态、纯 RL 哪个稳定门持续失败、是否遗忘平地完整命令域。v8 为这三部分补初态难度、接稳目标和常驻复习，不把 reward 增长当作完整自起成功。

本次仅修订 recovery worker；其余 flat、speed、terrain、jump、stepjump 五个 worker 必须与 v7 引用逐值、逐 SHA 相同。物理资产、执行器响应、轮侧 15.8 比率、35D actor/81D critic、50 Hz 策略、200 Hz 仿真与反馈 PD、4096 并行、PPO 批量与 fixed LR 保持。腿侧 bag 响应仍未 qualified，G1 与硬件发布不因本课程改变。

## 2. 恢复 65% 与完整平地 35%

新 recovery worker 的固定场景布局共 **95 个命名组：54 个恢复任务组＋完整 flat worker 的 41 个命令组**。恢复占 65%，平地复习占 35%；flat 的 41 组全部保留其命令、分布与组内相对配额，按总占比缩放，覆盖驻停、多站高、平动、正反 yaw 及完整运动/请求序列。

旧 recovery worker 的额外 push/air/landing 15% 不再另占一个池；恢复主任务已提供恢复过程，腾出的配额用于完整平地复习。这只说明命名采样池重组，不表示可以删除合同继承的信号延迟、执行器或其他物理扰动。恢复与复习分母分别记录，不能用大量容易复习样本稀释完整恢复失败率。

每个环境在场景创建后保持原 group、地形类型、平台几何、env origin 和 follow-up 命令。**难度课程只在该环境正常 reset 时选择兼容初态**，不会在回合中改地形、改变组归属、传送到接管状态或注入成功轨迹。完整 41 组是覆盖保证；4096 人口的整数分配仍须通过物化合同核验。

## 3. 初态难度与绝对 actor 时钟

`origin_actor_update` 是命名迁移时冻结的非负整数，参考 recovery 域已经累积的 actor update 时钟。令 `u = max(0, task_actor_update - origin_actor_update)`；以下为相对该原点的关键帧，帧间线性插值，6000 以后保持末帧分布。

| u | easy | medium | hard |
|---:|---:|---:|---:|
| 0 | 0.90 | 0.10 | 0.00 |
| 1000 | 0.50 | 0.35 | 0.15 |
| 3000 | 0.25 | 0.35 | 0.40 |
| 6000 | 0.20 | 0.20 | 0.60 |

easy/medium 从冻结 profile sets 中选取同地形、同目标平台高度的有效初态；hard 使用该命名恢复组原来的初态。单级 200 mm、双级首层 200 mm/末层 450 mm 的平台身份不能互换。选择发生于 episode reset，整个 episode 保持；随机数来自环境现有 generator，并记录实际 profile ID、level ID、重置计数及实际人口比例。

原点不在进程启动、故障恢复或每次 reset 时重算。**actor 历史、task/stage/global 时钟、Adam 和全部消费均不清零**；新课程相对年龄与已有 actor 历史是两个量。10 次 critic-only 适应计入消费，不增加 actor 课程年龄；同合同故障恢复只完成尚未完成的适应次数。精确固定评测绕过初态重采样，仍检验其命名案例，训练比例不构成评测删项。

## 4. 纯 RL 接稳奖励

新目标仅作用于 recovery 环境中完整由 actor 负责的 policy interval，包括 actor 导致失败的终止 interval。脚本、零力矩释放及 BLEND 不进入 PPO；普通 41 组的奖励逐值保留。接管奖励在既有 dense reward 装配后应用，最后统一乘一次 `policy_dt`，不重复缩放。

新增项采用固定零中心的广域 pseudo-Huber 成本与精核成本：`-w_b (sqrt(1 + (e/s_b)^2) - 1)` 和 `w_f (exp(-(e/s_f)^2) - 1)`。两者均在零误差时为 0，偏离时为负；离开精核后广域梯度仍存在。权重和宽度是本次冻结的研究设计，不是实测最优参数，也不随本轮 actor 更新自动放宽。

| 误差 | 广域尺度 / 权重 | 精核尺度 / 权重 | 生效范围 |
|---|---|---|---|
| 水平重力投影范数，局部直立倾斜误差 | sin(12°) / 1.0 | sin(8°) / 0.5 | recovery actor interval |
| 机体角速度范数 | 0.75 rad/s / 0.5 | 0.25 rad/s / 0.25 | hold 包含 yaw，运动释放后仅 roll/pitch，yaw 继续跟踪原命令 |
| 四个主动腿轴速度，逐轴成本均值 | 2 rad/s / 0.25 | 0.5 rad/s / 0.1 | hold actor interval |
| 两轮输出轴速度，逐轴成本均值 | 0.25/0.06 rad/s / 0.25 | 0.10/0.06 rad/s / 0.1 | hold actor interval |

倾斜误差是直立附近的接稳目标，不能因为倒置时水平重力投影也为零，就称其能独立训练翻身；倒地翻正仍由传感器脚本承担。hold 的归属取 policy interval 开始时快照，防止结束时转入 FAILED 或释放运动后丢失该 interval 的责任。

高度与静止平面速度复用当前 `dense_tracking` 的同一组参数，在 recovery actor interval 内去掉会掩盖误差的接触/物理相位豁免，不另造第二套高度或 quiet 核。只对 recovery 去除三项 tracking 的常数正峰，已有 pitch/roll 核减去单位峰但保留其误差依赖；其余普通运动和物理成本保留。不会奖励“多等一会儿”：无正的时间进度奖励，成功事件仍 **+30**，真实失败事件仍 **−200**，均按事件一次计入。末步失败样本继续收费并保留终止负反馈，禁止 terminal bootstrap；脚本失败不能靠更多 PPO 修复。

## 5. 稳定门诊断与有限脚本改动

稳定门本身不放宽。FSM 暴露每环境的 `valid/support/body_clear/tilt/gyro/height/wheel_speed/leg_speed` 八项 pass mask、整体 pass、该 reference tick 的 `holding_rl` 和 `ready`、最长连续稳定时间及正连续段中断次数。计数属于 episode，局部 reset 只清对应环境；张量保持在设备，不增加每 tick 主机标量读取。TensorBoard 记录各门通过比例及其 hold 人口，人口为零的比例不得解释为全失败；最长时间的均值也不能代替完整成功率。

新合同显式启用 `complete_started_blend=true`：未到 BLEND 的脚本仍受 8 s 预算约束，已经进入 BLEND 的环境允许完成完整 200 ms，传感器无效和倾角失稳保护仍然有效。开关默认 false，旧合同保持原来的含 BLEND 8 s 行为；包装层与 FSM 的超时口径需要一致。

**CAPTURE 的 support 入口保持原样。** 最新 V5 RMCS 允许支撑确认前进入 CAPTURE，其行为必须在 V6 做独立、同条件配对仿真后再讨论。此次只移入经过明确范围核对的 BLEND 预算开关和诊断，不声称已移植全部最新 RMCS 自起实现，也不声称 V6 动态成功率已提高。

## 6. 来源、结果与资格边界

来源审计：[source_audit.json](../reports/v6_recovery_takeover_20261003/source_audit.json)。旧 V6 profiles SHA 为 `5b09a3bb27285ab7571bd133151d7118091997309ccc9d59c6b1259836dc1eef`，资产 manifest 为 `875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba`。本次初态重采样复用这份几何库，不把其中 `dynamic_recovery_validated=false` 或静载 prepare 的 `dynamic_validation_passed=false` 改成已通过。

| 证据 | 来源与实际范围 |
|---|---|
| 9/29 V6 probe07 | 源码 `3051c6d6b8da3dd6fa71d5c0200b7884e0da7879`；65/108 完整链成功，平地 61/68、台阶 4/40，31/54 命名项两次均过；工程回放，不是独立训练双 seed 正式验收 |
| v7 使用的恢复源 | `dc2bc475e1a16ce3a7b0e2271699cd8765687ddb` 与上述版本的 FSM/profile builder/静载 helper/observer 相同；wrapper 后来支持 200 Hz PD。旧 probe 的 1 kHz 物理成绩不是当前 200 Hz 的同条件复测 |
| 10/03 最新 RMCS/ActivationBench | V5 232 mm、40–110°，Python 62/80、C++ 58/80；200 Hz 控制/400 Hz 物理/50 Hz 策略。模拟 105°编码器零参考不等于 V6 stop105 资产 |

V6 probe07 原始报告 SHA `baa30c9154b89871a39cbcf8b5e7f1c23ab39ea194b199c89697f71006371e20`；10/03 Python 报告 SHA `fb3aae7436f2f5a5dfc8f3eab553f8043880508799c67ceefa9e7b5489293763`、C++ 报告 SHA `591e5322fe776fa1cb557749f9bd6977862da6928ccbdd5f7dcc19a3d89d15ea`。源码、模型、物理时钟与分母不同，不能用 V5 成功数或同输入运动学一致性作为 V6 全链验收。

## 7. 命名迁移、持续训练与待部署状态

计划只允许 **v7→v8**，recovery worker 的 `task-worker-revision-v8` 绑定原 v6 worker SHA `e2a7ecaec1985adac34c280a71b1359a6446fc12c38d9aaa6fb98ac769e4d74c` 和冻结 origin。校验器从该源通过唯一的 `apply_recovery_takeover_revision` 重建目标；额外 reward、机械、接口、预算、父来源或其他 worker 改动均不随迁移放行。

当前父树保持 `A→flat`；recovery、speed、terrain、jump 均从 flat 分支，stepjump 从 jump 继续。执行顺序保持 flat→recovery→speed→terrain→jump→stepjump；此前讨论的全累加父链仍是另一项未来修订，不混入 v8。自动 endpoint 保持 flat actor 13000，其余五域 actor 15000；各域消费上限仍 20000，总消费上限 121000，历史消费不退款。

保留无评测阻断、每 500 更新保存、现有证据 review/自动 endpoint 结束语义。自动结束仍产生 `unassessed_research_candidate`、`domain_qualified=null`，不把完成更新数当技能达标。部署需保留合法旧 seal、actor/critic/Adam、所有 attempt 和账本，冻结新源码/合同并以同 root 的认证源修订接续；不能把本机文件变化描述为活 worker 已热更新。

**截至本文：`deployment_status=pending`，新 V6 动态验收未完成。** CPU 单测、合同物化、原生运行探针、首次学习接续分别记证据；它们的作用不同，不能合并成“自起已通过”。本次文档不启动训练、不修改硬件 readiness，不追加未授权实机动作。
