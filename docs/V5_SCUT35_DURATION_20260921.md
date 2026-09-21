# SCUT35 各阶段预算耗时估算

这是按更新上限跑满的估算，不是收敛时间预测。通过门槛可提前结束；回归保护也可能提前暂停。

实测 4096 环境，约 3.27s/update；每次进程初始化/导出约 167s。
已计入每批重新初始化、独立评测和最终确认；复杂场景以更宽范围估算。

| 阶段 | 更新上限 | 含评测预算耗时 |
|---|---:|---:|
| stand | 2000 | 183–239 min |
| height | 500 | 47–62 min |
| forward_05 | 375 | 37–48 min |
| backward_05 | 375 | 37–48 min |
| start_stop_05 | 500 | 47–62 min |
| rotate_1 | 375 | 37–48 min |
| curve_low | 375 | 37–48 min |
| curve | 500 | 47–62 min |
| forward_1 | 500 | 47–62 min |
| backward_1 | 500 | 47–62 min |
| forward_2 | 625 | 61–80 min |
| backward_2 | 625 | 61–80 min |
| forward_3 | 750 | 71–94 min |
| backward_3 | 750 | 71–94 min |
| rotate_4 | 625 | 61–80 min |
| rotate_8 | 750 | 71–94 min |
| spin_translate | 1000 | 92–121 min |
| push_recovery | 500 | 47–62 min |
| airborne | 625 | 61–80 min |
| landing | 750 | 71–94 min |
| slope_up | 625 | 63–115 min |
| slope_down | 625 | 63–115 min |
| cross_slope | 625 | 63–115 min |
| rough | 750 | 73–135 min |
| step_up_03 | 750 | 73–135 min |
| step_down_05 | 625 | 63–115 min |
| stairs | 750 | 73–135 min |
| stairs_down | 750 | 73–135 min |
| step_up_06 | 1000 | 94–174 min |
| step_down_10 | 750 | 73–135 min |
| jump_small | 1250 | 119–219 min |
| jump_full | 1500 | 141–258 min |
| running_jump | 1500 | 141–258 min |
| mixed | 1250 | 119–219 min |
| mixed_robust | 1500 | 141–258 min |

所有阶段均跑满：约 **43.4–69.0h**。
整体墙钟预算为 72h；未完成阶段不会被标为通过。首次模型复用和提前验收会显著缩短实际时间。
