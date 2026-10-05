# 原生 PREPARE slew A/B，只读比较

本次 N4 单 seed 物理诊断：A 成功 2/4、failure 2/4；B 成功 0/4、failure 4/4。两臂 timeout、boundary、右删失均为 0。B 四个 first cohort 均在 8.000 s 以 `recovery_script` 原生 reason 终止，没有 BLEND、纯 RL、motion unlock 或 followup success。候选状态为 **candidate_not_merged_or_deployed**；qualification=false，不因配置含义正确而接入课程。

来源为 `prepare_A_physical_003` 与 `prepare_B_physical_004`。共同 H 为 `9859f41f488847640ced9d258ac8799a39ca6721`，C 为 `93d31d3f015c23978b1960ca220404e48086530f`。两个独立 clean checkout 均保留，外部 identity manifest 的 SHA、最终认证、当前 HEAD 和全部 69 个 runtime source SHA 均核验。仅 `recovery.py` 不同，候选字节恰为批准的一行替换；effective 合同只允许 `design_preflight.file` 的绝对输出路径不同。同 frozen flat PT、V8 worker、seed 290929、N4/full role、20 s 上限、200 Hz 物理/50 Hz actor、signal_delay=false、command_transport=true；正式更新和 optimizer steps 均 0，PT 与 actor 张量未变。

| 实际 case | 首次 PREPARE（A=B）s | PREPARE 前完全相同帧数 | 首个差异 tick / post 时间 s | A BLEND / 纯 RL / unlock / 成功 s | B |
|---|---:|---:|---|---|---|
| recovery_left_side_90_stand | 4.150 | 830 | 832 / 4.160 | 无，8.000 failure | 无，8.000 failure |
| recovery_right_side_90_stand | 4.280 | 856 | 858 / 4.290 | 5.410 / 5.615 / 6.680 / 7.660 | 无，8.000 failure |
| recovery_prone_belly_legs_above_stand | 3.840 | 768 | 770 / 3.850 | 4.985 / 5.190 / 6.220 / 7.200 | 无，8.000 failure |
| recovery_prone_back_legs_above_stand | 4.270 | 854 | 856 / 4.280 | 无，8.000 failure | 无，8.000 failure |

四 case 到首次 PREPARE 的 phase event 序列和时间逐项一致；进入 PREPARE 的首个实际物理帧也完全相同。首差异均在第二个 PREPARE 子步，涉及实际 script targets、post-mix/applied torque、q18/dq18、root pose、body gyro 和 root link velocity。此“一致”限于已记录的全部物理帧字段，不声称 PhysX 内部状态相同或可精确恢复。

原 A 成功的两个姿态在 B 未达到原生 ready 条件：

| case / 臂 | PREPARE 子步 | tilt 通过 | wheel_speed 通过 | leg_speed 通过 | 八门同时通过 | ready_ticks 最大 |
|---|---:|---:|---:|---:|---:|---:|
| right / A | 226 | 25 | 26 | 120 | 19 | 19 |
| right / B | 744 | 20 | 244 | 717 | 0 | 0 |
| belly / A | 229 | 32 | 136 | 124 | 19 | 19 |
| belly / B | 832 | 20 | 29 | 824 | 0 | 0 |

表中分母是实际 `phase_used=7` 且 script ownership 的物理子步。A 达到 ready 的第 20 个子步已经标为 BLEND，故不计入 PREPARE 分母；其 BLEND 入口 `ready_ticks=20`，后续均有 40 个真实 BLEND 子步。B 的 ready_ticks 最大为 0；right 的主要失败组合包含 tilt、body_clear、height、wheel_speed，belly 的 `tilt+wheel_speed` 同时失败为 567/832 子步。上述是直接读出的原生判据，未重造八门或放宽门限。

剔除进入 PREPARE 的边界帧后，四 case 的实际腿目标 slew 最大绝对值从 A 约 4.0000 rad/s 变为 B 约 1.00002 rad/s，证明候选影响了实际轨迹。B 原成功 belly 的末个 PREPARE post 状态 gravity pitch 8.466°、gyro norm 0.108 rad/s、轮 dq +5.364/−5.351 rad/s；right 为 37.023°、0.465 rad/s、+24.149/−23.772 rad/s。它们仍未取得同时通过八门的实际子步。整个 PREPARE 窗长度不同，较小 gyro 均值不能作为同时间窗改善或通用增益结论；单 seed/N4 结果也不构成动态资格。

`physical_AB_comparison.json` 保存逐 case milestone、终止/分区、对齐的 pre/post pitch/gyro/named dq18、实际 slew、原生八门计数、前缀一致性以及报告/事件/帧/快照/source manifest/source files 的 SHA。frame 的 q/dq/root/gyro 是 robot.update 后的 post 状态；pre 输入取同 lane 上一连续 post 帧，时间为当前 stamp−5 ms；gates/ready/targets 是当前 advance_reference 在物理步前的原生结果。gravity pitch 为 `atan2(g_body.x,-g_body.z)`，不将倒置状态当 Euler pitch。

比较器只有 Python 标准库依赖，不导入 Torch/Isaac、不反序列化 PT、不启动 GPU/SSH、不改控制或合同。12 个 CPU 测试通过，失败测试仅使用独立 temporary Git fixtures 和深拷贝记录，未修改真实 A/B、两个冻结 checkout 或原始事件。运行命令：

```sh
python reports/v6_locomotion_quality_repair_20261005/recovery_audit/prepare_gate_audit/compare_physical_ab.py \
  --baseline reports/v6_locomotion_quality_repair_20261005/recovery_audit/prepare_A_physical_003 \
  --candidate reports/v6_locomotion_quality_repair_20261005/recovery_audit/prepare_B_physical_004 \
  --output /path/outside/both/source/report/directories/new_comparison.json
```

CLI 拒绝覆盖既有输出；保留冻结 checkout 到身份认证完成。本文和比较 JSON 不触发部署。
