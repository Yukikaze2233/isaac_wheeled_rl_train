# V6 手动技能补救计划：合同 v6_manual_remediation_v1

V5.9 全队列复盘、能力缺口分析与本合同的逐段校验记录。合同本体：
`contracts/v6_manual_remediation_v1.json`（extends `v6_emergency_v1` → `v5_full_usb_v59` 链）。

## 1. 目标与范围

四个交付物，之外不做：

1. 稳的底子（stand 八项 + 锚定组跟踪，含质量/质心随机化后的鲁棒性）；
2. 会跳（`jump_cold_03` 3cm 冷启动 bin → `jump_small` 6cm → `jump_full` 10cm）；
3. 上得了台阶（手动触发状态机，3/6/10cm，进阶 15/20/25cm）；
4. 1.5 m/s 加减速（连续速度曲线 + 全幅值命令）。

明确不进本课程：≥4 m/s 高速域、坡道（评测环境 slope 秒死 bug 未修）、连续地形感知（无前向传感，
`step_assist.enabled=false`，触发时序与地形几何解耦——sim-to-real 一致性约束写死在合同里）。

## 2. 诊断摘要（为什么 V5.9 没成）

| # | 结论 | 证据 |
|---|---|---|
| 1 | 验收闸门是配置关闭而非缺失：`evaluation_mode=monitor`、`require_passing_anchors=false` | v59 合同根字段；runner 存在 `complete_gated_curriculum` 分支 |
| 2 | 跳跃从未发生：每回合 1.86s 终止于 2s 触发闸门之前，`jump_air_time_peak=0` | stage_05 五次评测；jump 组课程零变化、caps 停 0 |
| 3 | 台阶只生存不推进：0.4 m/s 命令实际 0.056；step_assist 依赖仿真真值，实车无法复现 | stage_06 评测 step_up_03 字段 |
| 4 | 加速墙：前沿课程只采样可存活域，start_stop 帽位卡 1.25，进退相抵 | stage_03/04 课程变化历史 |
| 5 | 预算语义：合同 updates 按 4096-env 参考并行度计价，环境步守恒、优化步数随并行度反比 | `full_curriculum.py:156`；stage_04 实际仅 500 updates |

## 3. 起点决策

`initialization = latest_complete_V59_actor_with_explicit_reference36_migration`
——取 V5.9 队列最终完整 checkpoint 的 actor，经 `scut35_to_reference36_zero_new_columns`
迁移到 36D 观测（新增 1 维参考列零初始化）。

备选方案的取舍记录：

- **V5.8-667**：同协议 54-case 最强（19/27），但缺 V5.9 的域排练；且 667→36D 同样要迁移，无省事优势；
- **resume 线 stage_03 终点**：stand 最干净（漂移 6.5mm/s），但域窄、且已是中间态而非完整线终点。

选 V5.9 终点的理由：全域排练在身、36D 迁移与门控由合同强制兜底；其已知风险（入场基础抖动，
stage_07 首评 stand vx_bias −0.117 m/s）交给 `command_repair` 段修复、出口闸门把关（见 §5 预案）。

## 4. 阶段表（resolve_plan + stage_contract 实测解析值）

参考并行度 4096；`updates` 列为合同参考值，`实际` = 参考值 × 4096 / 段并行度。

| # | 阶段 | kind | envs | updates(ref) | 实际 | gate promo 用例 | 评测用例 |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | command_repair | foundation | 16384 | 2000 | 500 | 10 | 58 |
| 2 | dynamic_motion | speed | 8192 | 1500 | 750 | 20 | 92 |
| 3 | load_adaptation | foundation | 16384 | 3000 | 750 | 14 | 224 |
| 4 | manual_step_basic | terrain | 8192 | 1000 | 500 | 16 | 238 |
| 5 | manual_step_target | terrain | 8192 | 2000 | 1000 | 16 | 246 |
| 6 | manual_jump | jump | 8192 | 5000 | 2500 | 18 | 260 |
| 7 | manual_mixed | mixed | 8192 | 2000 | 1000 | 24 | 428 |

合计实算 **7000 实际 updates**，与 `training_reference.schedule` 声明一致。

阶段要点：

- **command_repair**：全幅值命令（`full_amplitude_commands=true`）+ `promotion_cases`
  [stand, forward_05, backward_05, rotate_1, rotate_1_reverse] ± USB 孪生，修命令响应与静态质量；
- **dynamic_motion**：新增 `velocity_curve_05/1/3` 技能（加速度限幅 0.6 m/s² 的连续速度曲线，
  替代"每集从静止阶跃"——对着加速墙的根因），加上 forward/backward_4/5、start_stop_3/5（加速度
  1.5 m/s²，catalog_overrides 显式降陡度）、weave、3rps 旋转；
- **load_adaptation**：质量随机化段（见 §7），height_parallel/full/pulse 系列进排练；
- **manual_step_basic/target**：手动触发上/下台阶（`special_mode_activation` + `command_reference`
  的 step ramp：抬身 0.5s、请求窗 [0.4,1.4]s、保持 [5,9]s；台阶命令高 0.4m/下 0.28m 为绝对命令高）；
  target 段进阶 15/20/25cm；
- **manual_jump**：`jump_cold_03`（apex_delta 0.03，`jump_apex_frame=com_release`，全跳系已切换
  COM-at-release 口径）→ small 6cm → full 10cm → running_jump，附带 push_recovery/airborne/landing
  排练；`reference_reward.push_soft_cost_scale=0.25` 在蹬伸窗口软化 effort 惩罚；
- **manual_mixed**：全技能混合排练收口。

## 5. 门控与熔断

- 每段出口 `evaluation.mode="gate"`，promotion 用例不过则阻断流转（runner 已实现
  `accepted_capabilities` 记账与阻断分支）；`regression_patience=2`、`consecutive_passes_required=1`；
- promotion_cases 随段进阶递增（10→24），全程含 USB 孪生（`usb_evaluation_pairs` 继承自 v59）；
- **熔断条款（运行手册级，非合同字段）**：manual_jump 段若在 assist/b cold-bin 条件下 1500 实际
  updates 内 `jump_cold_03` 完成率为零，人工停机复盘，不烧完剩余预算；
- **预案**：command_repair 出口闸门两次不过 → 预算扩至 4000 ref（=1000 实际）重跑该段。

## 6. 随机化与奖励改动（相对 V5.9）

- `dynamics_randomization_file: rigid_body_startup_v1`：startup 采样，base 质量 ×[0.9,1.3]、
  腿/轮 ±10%、base 质心偏移 x±4cm / y,z±2cm，50% 环境启用（SCUT V14 事件对标，惯量事件独立关闭）；
  ——关闭实车静态漂移缺口的主补丁；
- `reference_reward`：速度 Huber 核（δ=0.5，权重 0.25）+ 宽核（σ=1.5，权重 0.5）替代单一窄核——
  加速瞬态不再被窄核重罚；spin 平移项加权 1.5；
- 观测 35→36D：`encoders_imu_command_reference36`，迁移时新列零初始化，部署侧 I/O 文档需同步
  +1 维（对接文档更新项，见 §9）。

## 7. 运行手册

1. 前置：V5.9 队列正常收尾（`mixed_robust` completion 落盘）；确认 stage_07 `model_final.pt`
   与 completion.json 的 updates/SHA；
2. Kaiser 侧以 V5.9 同款作业方式起跑：`chassis_remote_job.py` 打包本地源 +
   `run_full_chassis.py --contract contracts/v6_manual_remediation_v1.json --research`；
   launch.json 的 initialization 指向 V5.9 终点 checkpoint 路径；
3. 监控点：每段出口评测的 promotion 通过表；TB 关注 `reward/dense_jump`（manual_jump 段应显著
   非零）、`task/height_error_m`（load_adaptation 后不应劣化）、`/task/success`；
4. 产物：每段 `accepted_capabilities` 链、终段 onnx 导出按既有 `models/v5_flat_12486` 同款流程，
   I/O 合同按 36D 出新版本。

## 8. 时间账

按 Kaiser 实测吞吐（16384 envs ≈ 10s/update，8192 ≈ 3.5–5s/update）+ 每段出口评测：

command_repair ≈1.5h、dynamic_motion ≈1.5h、load_adaptation ≈2h、两个 step 段 ≈2.5h、
manual_jump ≈3.5h、manual_mixed ≈1.5h，评测与间隙 ≈3h——**全程约 16h，一天内出结果**；
熔断/回滚各预留半天。

## 9. 已知风险与待办

| 风险 | 缓解 |
|---|---|
| V5.9 终点入场基础抖动（首评 stand bias −0.117 m/s） | command_repair 段 + 硬闸门；预案 §5 |
| manual_jump 2500 实际 updates 对冷启动仍可能偏紧 | jump_cold_03 降门槛 + 熔断条款；必要时扩 ref 预算重跑 |
| 静态漂移奖励侧无专项核（stationary_tracking=null） | 依赖随机化兜底；若实车复测仍漂，补 stand 窄速度核（σ≈0.1）为 v6.1 修订 |
| 36D 观测破坏现有部署 I/O 对接 | 部署侧按 `models/v5_flat_12486/v5_policy_io.py` 同款流程出 36D 版本并更新对接文档 |
| 坡道评测环境秒死 bug 遗留 | 本课程不含坡道；bug 修复立 V6.1 待办 |

## 10. 校验记录（2026-09-26）

`resolve_plan` 全链（manual → emergency → v59 → … → scut_skills_v4）解析通过；
逐段 `stage_contract(base, plan, recipe, 16384)` 七段全过，实际 updates 合计 7000 与声明一致；
promotion/cases 计数见 §4。跳段 `jump_apex_frame=com_release` 代码支持确认
（`full_tasks.py:95,109`、`skill_curriculum.py:53`）。
