# Speed/yaw 源码只读审计

日期：2026-10-09。冻结 runtime 为 `e5bd9c1367b875d7f229b9a1930647013d023372`，tree `769725c57b8ab3b3b2e66827594c3f5c83a8c51f`；native driver 为 `bcc65de507ed0ec877bb17c9401aeff13b52da2a`。本审计未找到可证实的 yaw 目标、量纲或 mode 错配。已完成的 005 四个独立进程在固定 ±1 rad/s、.305 m 条件复现 speed597 过转；这不唯一证明 actor 遗忘或某项奖励导致偏差。机器证据、完整源 SHA/行号、20 项 CPU 检查与回放 pin 在 [JSON](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/yaw_response_source_audit001.json)。

只读了冻结源码、合同、保存配置和既有回执；本任务 0 native constructor、0 PPO、0 optimizer、0 远端操作。仅新写本 JSON/MD，不更改旧回放数据、合同、模型、工具或 031 queue；qualification=null。

## 有效合同与命令

以 [speed worker v26](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/contracts/v6_task_campaign_speed_worker_v26.json) 加 [integrated recipe v26](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/contracts/v6_integrated_runtime_recipe_v26.json) 为准。实际 `static_training_config` CPU 展开确认 reward、reference、command slew、控制、观测与延迟字段不被 speed patch 覆盖。该入口只做静态展开；实际 component clock 由真实 admission/runtime 与已完成 actor 更新绑定，不能用裸 worker 的历史 origin 或父 global 更新代替。

`rotate_1` 的 nominal 仍为 [0,1,.305]，但实际属于 24 个 SCUT ordinary 行：rough_dash 源连续采 vx ±2.7 m/s、yaw ±2π rad/s，nominal_fraction=0，standing coin=.1，保持 5–15 s；非 standing 行有 .5 heading coin。heading 目标范围 ±π，world heading 误差 wrap 后乘 stiffness=5，再 clamp 到源 yaw 范围。kind=rotate/mode=1 并非 spin_translate；该组名不代表固定 +1 请求。见 [skill_commands:296](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/skill_commands.py:296)、[heading_command:83](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/heading_command.py:83)。

sampling 先生成 raw joint/heading 命令，再按 component difficulty 缩放一次；后续 heading 更新复用缓存 scale、做可行性投影，native slew 最后作用。command slew 为 vx1.5 m/s²、yaw4 rad/s²，policy_dt=.02，即一次最多 .08 rad/s yaw 增量。

| speed component actor | vx cap m/s | yaw cap rad/s（转每秒） | moving height m |
|---|---:|---:|---|
| 0 | 2 | 6.283185（1） | .28–.35 |
| 120 | 3 | 10.210176（1.625） | .25–.38 |
| 240 | 4 | 14.137167（2.25） | .24–.40 |
| 360 及 597 | 5 | 18.849556（3） | .23–.43 |

这是缩小已声明分布的 cap，ordinary 源 yaw±2π 不会因 cap 放宽自动变成±6π；专用高转速行需另算。row ownership 解析 speed component clock，保存 component counter 加实际 task actor 增量；critic-only warmup 不推进 actor clock。见 [locomotion_difficulty:68](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/locomotion_difficulty.py:68)、[integrated_runtime:280](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/integrated_runtime.py:280)、[train_chassis.py:925](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/scripts/train_chassis.py:925)。配置覆盖不能证明 command×height 的实际曝光充分。

## 量纲、动作与 mode

35D actor 的 slot2（0-based）是未缩放 yaw command rad/s，slot6 是 body-Z gyro×.5。奖励、EpisodeMetrics、raw trace 使用未缩放 `root_com_ang_vel_b`；manual35 只重写 context。sensor delay 保存 raw IMU，随后 build_sensor18 缩放一次；commands slots0:4 不被 sensor noise/delay 改写。两个保存 agent 配置均关闭 actor obs_normalization，未发现隐含再归一化。见 [scut_observation:22](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/scut_observation.py:22)、[env:1074](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/env.py:1074)、[env:1411](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/env.py:1411)、[episode_metrics:93](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/episode_metrics.py:93)。

六轴 policy/control 排列互逆；policy wheel 输出 slots4/5 对应 control2/5。wheel action clip±9，decode×10 rad/s；轮侧 kd=.6、cap4.5 Nm，gear15.8 在该 wheel_output cap 分支没有再次放大。yaw 请求没有额外在线控制器转换成 actor 动作。yaw+1 的轮缘可行性速度为左−3.644/右+3.644 rad/s，yaw−1 反向，远低于90 rad/s，projection 不改目标。原生关节轴正负由模型定义；该差速边界不是硬件轴校准，合同仍 hardware_motor_mapping_verified=false。见 [v5_control:34](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/v5_control.py:34)、[core.py:186](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/v40/core.py:186)。

003/005 driver 在 stand_305mm lane4–7、1≤t<10 请求 ±1；apply_request 使用原 playback slew 一次，同步 commands/command_target，非零 planar 设置 mode1。stand_305mm 不在 heading owners，exact config 不启 training heading；未发现 mode 或 heading 在回放中重复改写目标。见 [driver:305](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/moving-jump-source-20261009/scripts/run_rec_speed_mean_replay.py:305)、[driver:370](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/moving-jump-source-20261009/scripts/run_rec_speed_mean_replay.py:370)。

body-Z rate 在倾斜时不等于 world-heading 导数；当前合同和指标明确采用 body-Z。未发现相反合同要求，因此记录此接口区别，不能先当作当前 ±1 过转的原因。训练 delay/noise 是另一个分布，精确回放关闭它们，也不能直接归因为 005 的即时目标量纲错误。

## 实际支付的 yaw 核与姿态项

实际 dense yaw error=`omega[:,2]−commands[:,1]`：

- broad：`−(sqrt(1+e²)−1)`，scale=1 rad/s、weight=1，无截断。
- precision：`.35×expm1(−(e/.2)²)`，scale=.2 rad/s、weight=.35。

两项均在 e=0 最优。fine 在大误差趋近−.35，但 broad 继续区分误差；yaw gate 恒为1，不乘 ordinary/mode/support/倾斜门。旧 yaw_square/yaw_precision 等先删除，旧 precision 分支在 dense enabled 时跳过，保留单次 +1 yaw tracking peak。后续 recovery hold/takeover 有单独的 recovery-enabled/ownership/hold gate，stand_305mm ordinary 行不受该特定覆盖。最终所有密度统一乘 .02 一次。见 [rewards:138](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/rewards.py:138)、[rewards:178](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/rewards.py:178)、[env:1719](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/env.py:1719)。

CPU 标量核样本：e=.052 的 broad/precision 约−.001351/−.022878；e=.416 约−.083077/−.345375；e=.741 约−.244621/−.350000。它们用于验证函数和核区间，**不是 rollout 实付支付统计**。PPO 不通过物理 reward 对状态导数反传，不能据 fine 饱和宣称 actor 梯度为零。

姿态成本使用 body projected gravity；flat ordinary 的 pitch target0，地形可见命令可有±.08rad。pitch/roll floor 的 mask 是 ordinary、非 jumping、terrain_mode0，普通 mode1 旋转也包含。horizontal angular cost 只看 omega x/y，低 vx 的 stand_still_lin_vel 惩罚平移而非改 yaw target。effort/power、动作一阶/二阶平滑和姿态项与跟踪共同支付；历史 `_v` 姿态命名不能译成角速度惩罚。源码可说明这些竞争存在，不能据权重大小证明它们导致当前过转。见 [rewards:534](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/rewards.py:534)、[rewards:562](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/rewards.py:562)。既有 [华南虎源码审计](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/scut_source_comparison.md) 的上述解释已复核，未重新浏览或执行其外部工程。

## 005 独立进程证据

[完成回执](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec_speed_mean42000_independent_motion005/completion.json) status=`independent_native_motion_completed_unscored`，SHA `f8e8f8c32e3bb17d416c4d1ff374e56520471fbe3d248c9504adbee6a5c60aca`。四 raw readback 均 `native_mean_replay_completed_unscored`，各1次 fresh-process constructor，四个不同 PID；PPO0、optimizer=false、qualification=null。这里只读取并重算父线程已完成数据，没有新增 native 执行。

统计只用 `.02+n×.2` 固定5Hz栅格，筛选 active、非零 yaw request、lane4–7，每 endpoint/protocol180样本；不将额外 FSM/event trace 行等权加入，也不是完整50Hz episode均值。

| 请求 rad/s | endpoint | applied mean | actual body-Z mean | MAE vs request | MAE vs applied | plateau actual mean |
|---|---|---:|---:|---:|---:|---:|
| +1 | REC41393 | 0.976889 | 1.002135 | 0.052130 | 0.031057 | 1.027073 |
| +1 | speed597 | 0.976889 | 1.701715 | 0.741493 | 0.724826 | 1.740202 |
| −1 | REC41393 | -0.976889 | -0.945746 | 0.056250 | 0.033773 | -0.968895 |
| −1 | speed597 | -0.976889 | -1.369367 | 0.415583 | 0.395542 | -1.410051 |

以上逐数重现旧003。plateau 另有172样本，实际 applied 精确±1；speed 的 plateau MAE 分别 .740202/.410051，所以平均 slew 不是偏差的唯一解释。旧003的 REC→speed 同 env 复用设计限制仍保留；005 的此 yaw 现象已有独立 world 复现，但 fresh process 也不证明 PhysX 全部内部状态逐项等价。此结论仅限两个冻结 endpoint、固定均值策略、所选姿态/种子/replica/请求窗，不推广至所有高速/高度或当前 live checkpoint，也不声称能力通过。

## 必要后续事项

1. 先只做 command×height×pool/mode/heading 的真实 issued/applied 曝光账，按 speed component0/120/240/360/597 分窗，记录 policy-owned 有效样本与 delay strength；不由组名或总份额推断 ±1×.305 的采样充分性。
2. 在同一批真实状态/动作上只读重算原合同各项 mask、核区间及 dt 后实付 yaw/姿态/平滑/effort，先定位目标区间的支付竞争；不凭 fine 饱和或权重先验加权重。
3. 若另行授权，以同 checkpoint、独立 world、原物理/FSM，只变一个 yaw 幅度或一个 height 做条件响应图；同步区分 body-Z、heading derivative、scaled gyro、实际轮动作与停车窗。
4. 有实付/曝光证据后才冻结一项核宽、平滑或曝光参数的独立对照，其他父模型、合同、预算、随机化和验收固定，并检查 REC/驻停/高速回退。此任务未执行或热改任何候选。

20项 native-free CPU inline检查真实 exit0，计算部分3.627428s（不含imports）。完整脚本 SHA `5d2a24adbec0775a06a5ae8580abd050e8ed0fb776be76f8a18d5ae0b8bd5ca7` 与最终检查名单保存JSON。31个 source/contract/reference 文件和17个 input文件完整pin，写后再次复算；这些CPU检查不能替代 native物理验证、实际训练曝光或能力资格。
