# 单 actor 全功能完成审计：复习覆盖与行为证据分开

2026-10-09，只读。生产基线为 `e5bd9c1367b875d7f229b9a1930647013d023372` / source029 / v26；组合候选为 `0d6294b988c10d294cbe6758080e461ac95fdf22`，根目录 `/home/yukikaze/Documents/workspace/robot_rl/.worktrees/audit-logger-timing-20261009`。本次只读取已保存合同与摘要、调用纯静态 recipe composition；没有写训练合同/manifest，没有远端操作、重复采样测试、仿真或 PPO。

[机器可读矩阵](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/SINGLE_ACTOR_COMPLETION_EVIDENCE_MATRIX_20261009.json)保存实际 Git SHA、旧 v26 plan/recipe/lineage SHA、各原生 worker 的文件/SHA、函数入口、整数 lane、clock ownership 和未提取的行为证据。候选 v27 静态参数 `origin=0 / allowance=15000` 是 schema 示例；真正完成 speed 后的父 checkpoint、引入时钟、剩余预算和 native 准入尚需另行认证，不能把本表当成已部署回执。

## 串行推进确为同一模型，后段保留前段复习

`integrated_lineage.CHAIN` 为 **REC 父 → speed → terrain → jump → stepjump**。`select_learning_source/_checkpoint`认证父封存和完整 actor/critic/Adam；`TaskCampaign.train_command/run_integrated_domain`用实际父 PT 接续、计费 warmup，不把历史独立专项模型拼成一个模型。阶段正常完成的是时钟/预算，不自动授予功能合格。

下表为候选静态 allocator 的 **4096 lanes**。有效份额是去重、原生选中行归一化与永久静站 floor 之后的数值，不能直接引用名义 component 配比作为实际曝光。

| 段 / 行数 | 有效 component 份额 | 5–40 mm 小阶 | 普通大阶候选 |
|---|---|---:|---:|
| speed / 189 | speed 80%、REC 20% | 0 | 0 |
| terrain / 241 | terrain 60.4396%、speed 24.7253%、REC 14.8352% | **24.9231% / 1022 lanes** | **5.9341% / 245 lanes** |
| jump / 256 | jump 43.75%、terrain 23.4375%、speed 18.75%、REC 14.0625% | **14.9148% / 610 lanes** | **3.5511% / 144 lanes** |
| stepjump / 270 | stepjump 39.3782%、jump 13.9896%、terrain 18.6528%、speed 13.9896%、REC 13.9896% | **11.8700% / 486 lanes** | **2.8262% / 112 lanes** |

候选 speed recipe 与冻结 v26 **逐值相同**。三个后段中的八条 5/10/20/40 mm 上下小阶 spec、每行份额及合计整数 lanes 均与 v26 相同；新大阶没有替换它们。代价来自原生 terrain 的 surface_traversal **10%→2%** 和 surface_height_joint **4%→2%**，形成 ordinary_body_climb **10%**。坡/rough/材质行保留，但学习曝光预算确实减少；不能由“行还在”保证原能力保持。

## 两种 200+250 mm 任务独立判定

普通 `ordinary_double_200_250mm` 是 **body_climb、mode 0、无 trigger**；主动 `double_step_jump_200_250_manual/laser` 是 **mode 4、有请求**。几何都为每级增高 `[0.20,0.25] m`、踏面 `[0.8,0.6] m`、顶面离地 **0.45 m**。实际 composed spec 中两种 task selector 互斥。

`StepClimbTask.set_assessment_mask/observe/reward_terms/success`仅对放行后的普通爬阶计分；每级双轮上平台稳定≥0.35 s，连续无轮/合法车体支撑≥0.08 s会记 flight_violation。`StepJumpTask.control/observe/reward_terms/success`需要显式请求、腾空≥0.06 s、越前沿、各级真实落地稳定≥0.35 s。`env.step`分别在各自 mask 上替换 generic success，所以不能拿跳阶成绩授普通爬阶，反之亦然。

原生 terrain/jump/stepjump 合同的环境级 `success_hold_seconds` 为 **2.0 s**，在各自 task success 上继续累计。任务内每级0.35 s和环境级2.0 s是两层计数；后者消费的是完成标志，不能擅自解释为“完成后还逐物理步再次接稳2秒”。

**姿势边界仍需保留：普通 completion 允许车体支撑，但不强制必须先搭车体再收腿。** 新奖励中的 body support credit 有限、不会靠驻留持续刷分；最终还应单列 body_support、收腿轨迹和飞行违规，才能证明用户要求的搭台姿势。上述为源码判定分离，未重跑 native 物理轨迹，未授经验性完成率。

## 迭代时钟与跳跃来源

原生来源：REC worker **v21**；speed、jump、stepjump worker **v18**；小阶 terrain worker **v23**。v27仅明确派生 terrain 八条普通大阶，jump/stepjump 原生 source 与单一跳高 curriculum 没有换成第二套。`integrated_runtime.component_actor_updates`读取真实源时钟并加本段 task actor 增量；`row_settings`按 ownership 派课程，critic-only 不推进。

普通大阶按 **terrain component actor−认证引入 origin** 有限放行，最大完成放行时钟为 `min(4000,floor(0.35×实际actor allowance))`；后续复习不能重置该 origin。jump 段三个 running 行由 jump clock 采 **3/6/10/15/20/25/30 cm**；0/2000/6000 高四 bin 条件份额为25/55/80%。stepjump 复习去重后 running03/06仍是 jump clock，running10归 stepjump clock，不能把三个组都按本段零时钟解释。详细 CPU 证据沿用[既有更正报告](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/RUNNING_JUMP_SAMPLER_COVERAGE_CORRECTION.md)，本次未重复测试。

## 当前共同 speed 谱系还不能称复习已保持

已保存的 source029 第1/第2个100 actor TensorBoard窗口是探索、随机化且难度在改变的训练统计。第二窗结果不支持“恢复复习已保持”，但也不能直接判确定性遗忘或归因于新候选。

| 指标 | actor 1–100 | actor 101–200 |
|---|---:|---:|
| 静站230/305/430 mm平面均速，cm/s | 3.187 / 2.534 / 2.127 | **4.108 / 3.063 / 2.823** |
| 条件停车均速，cm/s | 2.578 | **2.993** |
| L0 已结束成功 / 分母 | 47 / 387 | **2 / 485** |
| L1 已结束成功 / 分母 | 60 / 638 | **4 / 770** |
| L2 original-chain 已结束成功 / 分母 | 47 / 1065 | **2 / 1187** |
| 全局有效 PPO 帧比例 | 95.52% | 95.58% |
| 每更新墙钟，s | 23.79 | 20.21 |

L2包含非倒地/平台出生，不能写成四种真正倒地完整自起率。全局 valid_fraction 也不等于每组/每相位/每 bin 的有效曝光。普通移动组名经过SCUT重采并不代表一直给±0.5固定命令，signed均值差不能当速度 MAE。这里复用已保存 `source029_tb100_001.summary.json` 与 `source029_tb200_001.summary.json`，未读取远端 event。

最终全功能 audit 仍缺：同条件父/子 checkpoint 均值动作回放；每 bin 无辅助跳跃与行进跳结束分母/成功率；普通小/大阶按每级与方向完成、搭台姿势；主动单双跳阶按请求源完整链；L2×真正倒地出生脚本→pureRL→连续接稳→运动放行；各任务最终 mask 的有效曝光。应等首500 actor完整 seal 并用固定协议旁路回放，再讨论保持/遗忘；可采样、时钟完成或 reward 上升均不能替代这些证据。
