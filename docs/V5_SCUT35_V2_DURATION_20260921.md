# 历史 SCUT35 v2 各阶段预算耗时估算

当前采用7阶段综合课程，见[V5综合训练运行手册](V5_INTEGRATED_TRAINING_20260921.md)。
下表保留v2专项方案的当时预算，不作为当前运行的预计完成时间。

这是按更新上限跑满的估算，不是收敛时间预测。通过门槛可提前结束；回归保护也可能提前暂停。

实测 4096 环境，约 3.27s/update；每次进程初始化/导出约 167s。
已计入每批重新初始化、独立评测和最终确认；复杂场景以更宽范围估算。

| 阶段 | 更新上限 | 含评测预算耗时 |
|---|---:|---:|
| stand | 2000 | 183–239 min |
| height | 2000 | 183–239 min |
| forward_05 | 4000 | 364–475 min |
| backward_05 | 4000 | 364–475 min |
| start_stop_05 | 2000 | 183–239 min |
| rotate_1 | 1500 | 138–180 min |
| curve_low | 1500 | 138–180 min |
| curve | 2000 | 183–239 min |
| forward_1 | 2000 | 183–239 min |
| backward_1 | 2000 | 183–239 min |
| forward_2 | 2500 | 228–298 min |
| backward_2 | 2500 | 228–298 min |
| forward_3 | 3000 | 274–357 min |
| backward_3 | 3000 | 274–357 min |
| rotate_4 | 2500 | 228–298 min |
| rotate_8 | 3000 | 274–357 min |
| spin_translate | 4000 | 364–475 min |
| push_recovery | 2000 | 183–239 min |
| airborne | 2500 | 228–298 min |
| landing | 3000 | 274–357 min |
| slope_up | 2500 | 233–427 min |
| slope_down | 2500 | 233–427 min |
| cross_slope | 2500 | 233–427 min |
| rough | 3000 | 279–511 min |
| step_up_03 | 3000 | 279–511 min |
| step_down_05 | 2500 | 233–427 min |
| stairs | 3000 | 279–511 min |
| stairs_down | 3000 | 279–511 min |
| step_up_06 | 4000 | 371–680 min |
| step_down_10 | 3000 | 279–511 min |
| jump_small | 5000 | 464–849 min |
| jump_full | 6000 | 556–1018 min |
| running_jump | 6000 | 556–1018 min |
| mixed | 5000 | 464–849 min |
| mixed_robust | 6000 | 556–1018 min |

所有阶段均跑满：约 **165.9–262.9h**。
整体墙钟预算为 72h；未完成阶段不会被标为通过。首次模型复用和提前验收会显著缩短实际时间。
