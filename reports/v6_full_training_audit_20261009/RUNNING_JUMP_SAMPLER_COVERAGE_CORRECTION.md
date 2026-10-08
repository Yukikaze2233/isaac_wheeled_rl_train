# 行进跳覆盖更正：冻结 v26 的高 bin 已可采

2026-10-09。仅本机 CPU 核验；没有修改生产源、合同、模型或账本，没有 SSH、仿真或 PPO。暂停新增 moving-jump transformation，以免重复已有课程。

## 结论

实际 source029 使用的冻结 `e5bd9c1367b875d7f229b9a1930647013d023372` / v26 已对 **三个** `running_jump_03/06/10` 行采样 3/6/10/15/20/25/30 cm。组名中的 03/06/10 是名义固定用例高度，不是训练时的最高高度。

因此原审查中“行进跳没有 15–30 cm”及“下一版先增加行进 15/20、再到 25/30 cm”的结论应撤回。真实尚待补证的是分高度实际曝光与完成率、未辅助成绩、15–30 cm 行进跳固定评测，以及行进速度域：三个行的前进目标仍为 **+0.4 m/s**。本次没有证明当前 actor 已完成高跳。

## 可复算证据

冻结配方：`contracts/v6_integrated_runtime_recipe_v26.json`，SHA-256 `d1a1de9829bb1c798f627e6605d4cc57ca669b400136b568e46b9b73708df2c5`。实测使用真实 `SkillCommands.sample`、`TrainingSchedule`、`FullTaskSemantics` 与 component clock helper；仅把三条真实行投影到 CPU fixture。

- 执行器：[verify_running_jump_sampler.py](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/verify_running_jump_sampler.py)。它首先验证真实 e5 Git HEAD、配方及原生 source hash。
- 结果：[running_jump_sampler001.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/running_jump_sampler001.json)，`passed=true`，exit 0。
- 每行每时钟 10,000 个均匀 CDF 中点，jump/stepjump 两域合计 180,000 次目标采样。七个 bin 全部非零，计数逐项吻合配置概率（容差 0.0002）。这些是 CPU 合成采样数，不是 Kaiser rollout 曝光数。
- 实际源中两个既有 focused tests 通过：`test_time_bins_cover_running_high_jump_without_changing_step_or_evaluation`、`test_skill_sampler_updates_task_semantics_and_keeps_active_jump_target_on_resample`，2 passed / 0.91 s。前者同时确认主动跳阶不会被此 bin 采样覆盖，后者确认活动跳跃目标不会被重采样中途改变。

复算命令（在该冻结 worktree 中执行，不访问远端）：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /home/yukikaze/isaacsim60-venv/bin/python -B /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/verify_running_jump_sampler.py
```

## jump 域三个 running 行的真实条件分布

`running_jump_03`、`running_jump_06`、`running_jump_10` 在 jump 域均由 **jump component actor clock** 驱动。表内概率是在已选择这三条行之一、发生合法目标采样的条件下的分布，不是全部 batch 的配额或有效 PPO 帧份额。

| jump component actor 更新 | 3 cm | 6 cm | 10 cm | 15 cm | 20 cm | 25 cm | 30 cm | 15–30 cm 合计 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30% | 25% | 20% | 10% | 5% | 5% | 5% | **25%** |
| 2000 | 10% | 15% | 20% | 20% | 15% | 10% | 10% | **55%** |
| 6000 | 5% | 5% | 10% | 15% | 20% | 20% | 25% | **80%** |

0→2000→6000 之间线性插值，6000 后保持末档分布。不是高 bin 在晚期才首次解锁：0 时四个高 bin 已各有正概率。低 bin 在成熟期也保留非零份额。无需再新增第二套跳高课程。

三个行的其余原生目标相同：vx **+0.4 m/s**、yaw 0、mode 4、水平位移目标 **0.15 m**、落地半径 **1.2 m**、落地速度上限 **0.8 m/s**。这些是命令/验收参数，不是实际实现的速度和位移。CPU 还验证采样高度进入 `FullTaskSemantics.jump_apex_delta_m`，其 COM 上升目标为 `max(apex−0.015,0.005)`。每次跳跃参考与结果判定读取该每-lane 目标，而非只读取组名名义值。

## stepjump 复习域的时钟区别

去重后 ownership 不完全相同：`running_jump_03/06` 仍为 jump；`running_jump_10` 被 stepjump component 接管。故不能把 stepjump task actor 0 当成三个 running 行的 jump component actor 0。

核验器使用明确标记为合成的、通过真实 clock validator 的顺序源时钟：speed 起点 0、terrain 15000、jump 30000、stepjump 40000。该 fixture 下，stepjump 入口 03/06 的 jump clock 已为 10000，使用成熟的 80% 高 bin 分布；10 行的 stepjump clock 为 0，使用 25% 高 bin 分布。真实入场数值必须由未来完整父 checkpoint 及准入回执读取，不能把这组合成数字写成实际谱系。

## critic 不推进与固定评测 scope

CPU 在 task actor 不变时增加 10 次 critic-only 完成/其他计数，jump 组三个 clock 均保持 `[0,0,0]`，stepjump fixture 保持 `[10000,10000,0]`。对应真实 trainer 只以 `int(actor_enabled)` 增加 `actor_updates_in_block`，然后写回 task/component clock。critic warmup 的消费照计，跳高课程不因其推进。

精确固定评测禁用随机课程，三条命名用例仍分别测 **3/6/10 cm**。本次再次执行真实 sampler 的 evaluation 分支确认这一点。`move_stop_jump_resume` 有显式 move→stop→jump→resume 命令序列，被随机 bin 分支排除；它是停车后跳，不能作为行进跳覆盖证据。主动台阶跳也走自己的目标与触发，不在本次结论范围内。

## 下一步最小建议

1. 更正完整功能报告 §2 行进跳一行及 §4.3：保留当前唯一跳高课程，改为“3–30 cm 已可采、0.4 m/s 行进目标；高 bin 有效曝光/无辅助成功率与固定评测尚需核验”。
2. 跳跃段实际开始后按真实采样 `jump_bin`、policy-owned mask、触发/腾空/落地/结束回合分别核查，避免将行名或条件概率当成已练到、已学会。
3. 若确需扩到 0.2–0.8 m/s，先单独冻结有限速度域扩展及对应固定用例；复用当前 jump component clock、现有命令采样/预载/收腿/落地实现。没有证实缺口前不补重复高 bin 行，不新增隐藏 gate 或第二状态机。

源码入口（均为上述冻结根）：`training_schedule.py:286`、`skill_commands.py:343`、`integrated_runtime.py:280`、`env.py:1044`/`:1301`、`train_chassis.py:925`。结果 JSON 保存这些实际源文件 SHA；CPU 核验并未运行 PhysX，也未验证真实起跳、受辅助比例或成功完成。
