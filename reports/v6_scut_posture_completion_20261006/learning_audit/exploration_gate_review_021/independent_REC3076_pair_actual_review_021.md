# REC3076 原生 mean／stochastic 配对独立复核

冻结源 `d0dce6181df83b03fe166abcbadef3c81de2e57e`，own REC3076／消费30007，PT `286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31`。两臂真实4096×600策略／2400物理、12s、exit0、零PPO；原生初态出生、固定lesson/profile、saved std与私有采样首尾逐值一致。后续环境RNG可因自动reset分叉；App.close returned仍unknown。

| Arm | Lesson | 固定出生 | Release | Success | Failure | Boundary | Timeout | 右删失 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mean | prefix_stand | 1883 | 443 | 216 | 142 | 0 | 0 | 1525 |
| mean | prefix_release | 675 | 157 | 63 | 93 | 1 | 0 | 518 |
| mean | original_chain | 89 | 12 | 2 | 41 | 1 | 0 | 45 |
| mean | ordinary_base | 1449 | 0 | 0 | 9 | 44 | 0 | 1396 |
| stochastic | prefix_stand | 1883 | 0 | 0 | 278 | 0 | 0 | 1605 |
| stochastic | prefix_release | 675 | 0 | 0 | 138 | 0 | 0 | 537 |
| stochastic | original_chain | 89 | 0 | 0 | 43 | 1 | 0 | 45 |
| stochastic | ordinary_base | 1449 | 0 | 0 | 60 | 34 | 0 | 1355 |

恢复固定分母2647；ordinary1449独立，不进入恢复成功分母。mean恢复release612／success281，探索均0。两臂均有未完右删失，不能按12s仍存活记失败。

| Arm | L0首出生实际forward/yaw+/yaw−曝光（50Hz全interval actor-owned帧） | L0 sensor连续稳定peak最大 | L0 actual contact稳定peak最大 | 原生付款残差 | Guard次数／全窗gap峰 |
|---|---|---:|---:|---:|---:|
| mean | [77615, 2894, 1008] | 1.00000s | 11.60008s | 0 | 5／31.374mm |
| stochastic | [0, 0, 0] | 0.06000s | 0.24000s | 0 | 5／730.632mm |

两臂实际2,457,600 lane intervals、600原episode event／hold producer，原reward/s仅乘一次.02、event不乘dt。首出生逐lane关闭时计terminal后不复开；付款每组帧分母与原生diagnostics逐值对齐；事件总付款分别与success×30及failure×−200精确相符。独立float64分箱重建与原生float32逐lane相加有≤0.001248的单箱舍入差，不是native零残差被改变。完整whole-interval ownership保留，excluded即时actor-eligible付款为0。

机械资格未过：两臂各5次closure guard。mean首出生可定位4（1708@57/L2、2216@59/L2、780@99/L1、1855@107/L2）；stochastic首出生可定位3（1708@57/L2、2216@59/L2、2466@243/L2）。这些首出生guard均处脚本阶段，whole-interval actor mask=false，FSM failure_code=0，由env closure_gap物理reason终止、bootstrap=false。冻结记录没有保存max-gap argmax或后续出生guard逐lane事件，因此不能把0.73063m可靠归到某lane/帧；mean另1和stoch另2后续guard定位 unavailable。逐guard完整返回reward／density亦未保存；native −200 failure event规则与汇总计数/付款核对成立，mask排除后的actor-eligible即时付款为0。

本配对证明该seal在实际采样下未跨过连续1s接稳门，mean可以释放并提供原目标曝光；不能把阻断全归因entropy或把新cost有效付款当成质量改善因果证据。轮/gyro/leg各门marginals不是joint或连续概率。接稳峰是历史连续段，followup是terminal/censor瞬时值；真实contact与200Hz FK/IMU传感器门分开，原1s成功标准保留。

完整机器回执：`independent_REC3076_pair_actual_review_021.json`；SHA `de41e6e5f2121b45d5d1b6116b73d52b3ce0ce83b50d6c7885c092c3487f2b81`。原始与18项绑定产物前后SHA一致；未SSH／App／GPU／PPO。
