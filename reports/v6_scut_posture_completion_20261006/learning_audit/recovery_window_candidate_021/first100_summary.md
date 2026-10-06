source021 完整100 actor与不重叠两半50窗口实测

前半task3086–3135／event30016–30065；后半task3136–3185／event30066–30115。各4096×24s，共100actor／48s／9,830,400lane策略帧。实际父29996/task3075及critic10保持独立；95组760个精确计数和855个frame加权均值分割核对通过。100窗与两个50窗重叠，不相加经验。

| 指标 | 前50 | 后50 | 后−前（描述性） |
|---|---:|---:|---:|
| stand_230mm 平面均速 | 11.7995cm/s | 12.1804cm/s | +0.3809cm/s |
| stand_305mm 平面均速 | 11.2189cm/s | 10.7198cm/s | -0.4992cm/s |
| stand_430mm 平面均速 | 8.6087cm/s | 9.2306cm/s | +0.6218cm/s |
| forward_05 vx MAE | 26.7897cm/s | 27.2519cm/s | +0.4622cm/s |
| backward_05 vx MAE | 26.2344cm/s | 27.7641cm/s | +1.5297cm/s |
| rotate_1 yaw角速度MAE | 0.6322rad/s | 0.5796rad/s | -0.0526rad/s |
| rotate_1__reverse_train yaw角速度MAE | 0.5326rad/s | 0.5543rad/s | +0.0217rad/s |
| start_stop_05 settled_stop条件均速 | 12.2190cm/s | 12.5178cm/s | +0.2987cm/s |
| start_stop_1 settled_stop条件均速 | 12.7118cm/s | 11.7349cm/s | -0.9769cm/s |
| velocity_curve_1 settled_stop条件均速 | 10.6472cm/s | 10.9703cm/s | +0.3231cm/s |
| wheel门归一误差 | 2.3802 | 2.3823 | +0.0021 |
| leg门归一误差 | 1.6585 | 1.6301 | -0.0284 |
| gyro门归一误差 | 1.6748 | 1.6651 | -0.0097 |
| support observer瞬时通过率 | 55.2662% | 56.2131% | +0.9469% |
| body_clear observer瞬时通过率 | 55.2609% | 56.2063% | +0.9454% |
| tilt observer瞬时通过率 | 79.1219% | 79.2199% | +0.0980% |
| height observer瞬时通过率 | 99.6867% | 99.7444% | +0.0577% |
| wheel_speed observer瞬时通过率 | 15.7690% | 15.8503% | +0.0813% |
| leg_speed observer瞬时通过率 | 22.4453% | 23.0686% | +0.6233% |
| gyro observer瞬时通过率 | 24.7386% | 24.8508% | +0.1122% |
| 更新墙钟 | 14.6752s | 16.1479s | +1.4727s |
| Policy mean_std | 0.4537 | 0.4555 | +0.0017 |

100窗整体：230／305／430mm静站平面均速11.99／10.97／8.92cm/s；0.5前／后vx MAE27.02／27.00cm/s；正／反yaw角速度MAE0.6059／0.5435rad/s。actor有效转移9389210、SCRIPT／BLEND／passive排除441190。恢复成功仍未出现，腿／gyro门误差微降且轮误差几乎不变，普通行为改善与退步并存，未见整体好转证据。

恢复静态54组结束7255回合，成功0、失败2323、超时4932；前／后成功分别0/3546与0/3709，混合动态lessons。L0/L1/L2成功日志峰值为{'prefix_stand': 0.0, 'prefix_release': 0.0, 'original_chain': 0.0}。54组运动命令min/max全零：前True／后True／100窗True。独立L0 motion release和原目标曝光仍为unavailable，不能据此认定bridge实现错误。

support/body_clear是可部署FK／IMU observer门（recovery.py41），不是实际接触、接触力或真实离地状态。真接触在RecoveryTraining.outcome另评分。表中边际门率不能推出连续1s联合通过率；hold成本是条件密度均值，不是完整付款或优势。静止10s没有完成样本时指标不可用，不能填零；实际230／305两半都0样本，430前44／后27／合71，只是eligible完成子集。停车条件样本（前50／后50／100）为{'start_stop_05': [17554, 17494, 35048], 'start_stop_1': [21050, 21968, 43018], 'velocity_curve_1': [7857, 9070, 16927]}，整段启停速度与驻停速度分开。4932 timeout为实际结束／删失，含80次边界截断；不能从它推断无限时长下的恢复成功率，failure也不能一律解释成跌倒。

两半变化只有同一run短期描述意义。策略年龄、动态lesson比例、出生／重置／域随机化和探索都随学习变化；未做同seal mean/stochastic配对，不能将一个成本或std定为唯一原因。Policy mean_std是归一化actor动作维度的聚合，不是单轴std或实测关节／gyro噪声，也不能与reward密度／熵loss跨量纲直接比较。当前物理dt5ms/200Hz、策略dt20ms/50Hz，4096与原PD160/2.5保持；工程原生通过不替代恢复行为资格。硬件资格仍null；本文没有修改训练、预算、门判据或优化器。

原始输入SHA：
- `first50_read001/report.json`：`cd9d5be371bb1c62ff95e27813f4c14d59b18b9bf378f4cc74b21283a61ecccb`
- `first100_tail50_read002/report.json`：`9f941e3fcaf6e95ff603c8cdeaeabded9df7bfe94844c2f65508d88c9de4e0c4`
