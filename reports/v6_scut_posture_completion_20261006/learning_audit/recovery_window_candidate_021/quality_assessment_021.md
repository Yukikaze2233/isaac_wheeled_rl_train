source021 的100 actor观测仍没有出现恢复成功；新hold成本已生效，但尚未证实训练收益。腿和gyro门误差微降，轮误差基本不变，基础行为有升有降，不能描述成整体已改善。

这里是 attempt005 的 **actor11–110，即task3086–3185**，event30016–30115。父消费29996／task3075保留；critic10计消费29997–30006且没有算成actor。两半task3086–3135和3136–3185不重叠，各24s，共4096×48s＝9,830,400策略lane帧，其中9,389,210有效actor转移、441,190 SCRIPT／BLEND／passive排除转移。95组计数和frame加权分割已经独立核验。

| 指标 | 前50 | 后50 | 完整100 |
|---|---:|---:|---:|
| 54静态恢复组已结束回合的成功 | 0/3546 | 0/3709 | 0/7255 |
| 230mm静止平面均速 | 11.80cm/s | 12.18cm/s | 11.99cm/s |
| 305mm静止平面均速 | 11.22cm/s | 10.72cm/s | 10.97cm/s |
| 430mm静止平面均速 | 8.61cm/s | 9.23cm/s | 8.92cm/s |
| 0.5前进vx MAE | 26.79cm/s | 27.25cm/s | 27.02cm/s |
| 0.5后退vx MAE | 26.23cm/s | 27.76cm/s | 27.00cm/s |
| 正向yaw角速度MAE | 0.6322rad/s | 0.5796rad/s | 0.6059rad/s |
| 反向yaw角速度MAE | 0.5326rad/s | 0.5543rad/s | 0.5435rad/s |
| 轮／腿／gyro门归一误差 | 2.380／1.659／1.675 | 2.382／1.630／1.665 | 2.381／1.644／1.670 |
| 单次更新墙钟 | 14.675s | 16.148s | 15.412s |

0/7255来自54静态组，**混合L0/L1/L2，不是最终固定评测的原始整链完成率**。7255＝2323失败＋4932超时，80次边界截断包含在超时里。动态三个lesson的success日志也全0，但这些累计量是24帧rollout均值，不能把它们当作精确episode分母。54组forward/yaw命令日志的min/max全部0；当前TB没有独立L0实际释放及原目标曝光tag，因此这两项仍是unavailable，不能由组均值认定bridge实现bug或“释放次数一定为0”。

前进实际均速0.23253→0.22501m/s，目标约0.496m/s；后退−0.23453→−0.22085m/s，目标约−0.496m/s。后半确实更欠速。yaw是角速度误差，不能说成角度跟踪；现有通用日志没有signed实际yaw角速度。

| 原生settled_stop条件 | 前50速度／条件帧 | 后50速度／条件帧 | 完整100速度／条件帧 |
|---|---:|---:|---:|
| 0.5m/s启停 | 12.22cm/s／17554 | 12.52cm/s／17494 | 12.37cm/s／35048 |
| 1m/s启停 | 12.71cm/s／21050 | 11.73cm/s／21968 | 12.21cm/s／43018 |
| velocity_curve | 10.65cm/s／7857 | 10.97cm/s／9070 | 10.82cm/s／16927 |

这是驻停相位的真实条件帧加权结果，不能与整段运动速度混为一谈。10s静止评估只统计eligible完成窗口：230／305没有完成样本，结果不可用；430有前44／后27＝71个选择性样本，不能代表所有430静止状态。超时／截断也不能推断无限时间下的成功率，failure不能一律解释成跌倒。

support/body_clear是可部署FK／IMU观察器判断，**不是真实接触、接触力或真实离地**。真实接触、平台、姿态和后续目标跟随在RecoveryTraining.outcome另行评分。瞬时轮／腿／gyro边际通过率约16%／23%／25%，只能说明这些observer门经常未通过，不能合成连续1s联合概率。

新hold的broad/fine负成本在原生日志中实际出现；既有002原生观察器另验证了工程付款、事件及排除边界。这些条件密度均值不构成完整lane付款账本，也不保证PPO已经学到有效接稳。mean_std约0.45460，仅是6个归一化动作维度的全局聚合，不是每轴std、实测关节／gyro噪声或独立熵因果证据。共享网络、探索、重置／域随机化、lesson份额及策略年龄尚未分开，不能将reward、std或力矩上限定为唯一原因。

下一步是**同一个REC3076封存checkpoint的mean／stochastic配对，尚待运行**。诊断锚是首次新actor的不可变seal，不是当前最新模型，也不是历史flat工程模型；记录真实task3076与origin2294，不强制改成工程clock。两臂保持同源、同checkpoint／std、种子、首次出生分区及DR条件，0PPO，GPU串行。应观测逐轴动作噪声、200Hz逐门连续dwell／break、L0实际release与原forward／yaw曝光、observer与真contact分开评分，并核对付款与ordinary／excluded边界。结果只能用于解释同锚对探索的敏感性，不自动成为当前模型的功能资格。

该诊断必须与正在训练的PPO避免资源竞争。这里没有热改source／合同／门判据／std／Adam，没有重启训练、退款或授予硬件资格；200Hz物理反馈、50Hz策略、4096、PD160/2.5及原预算保持。腿电机bag仍未qualified。

配对锚位置：`/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_005/train/checkpoints/update_00030007/model.pt`；SHA `286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31`，消费30007／task3076。

本文件绑定的原始报告：
- `first50_read001/report.json`：`cd9d5be371bb1c62ff95e27813f4c14d59b18b9bf378f4cc74b21283a61ecccb`
- `first100_tail50_read002/report.json`：`9f941e3fcaf6e95ff603c8cdeaeabded9df7bfe94844c2f65508d88c9de4e0c4`
