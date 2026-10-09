# source029 actor 1901–2000 原窗独立算术核对

原raw94条序列，每条43302–43402共101点，9494点的时间/step/value均有限、step无缺漏或重复。43303–43402的100个actor点用于均值，43302仅用于累计counter差分。原raw与canonical summary的source均为e5；独立math.fsum复算与canonical最大差0，全部小于1e-12。

| 高度组 | rollout帧 | 平面速度 cm/s | 高度绝对误差 mm |
|---|---:|---:|---:|
| stand_230mm | 100800 | 6.053920 | 14.327684 |
| stand_305mm | 100800 | 3.700167 | 3.348044 |
| stand_430mm | 100800 | 8.762295 | 9.290538 |

原生settled-stop条件分母72068帧，加权速度4.018864 cm/s；wall均值26.208867 s/update。停车条件为奇数profile segment中越过原settle时长，使用诊断body-plane速度；不是stationary_10s完整十秒窗。

prefix_stand：62/516，failure=33，collection_timeout=421；三项严格合计ended。

prefix_release：73/757，failure=142，collection_timeout=542；三项严格合计ended。

original_chain：71/1212，failure=418，collection_timeout=723；三项严格合计ended。

本复核只读原JSON与冻结源，未独立重读远端event CRC字节；原reader回执认证的范围是受限suffix，earlier history CRC=false。未使用HTTP抽样，未改root raw/summary/docs。训练组名不是固定命令，original_chain为混合L2而非四倒地分母，signed mean不是MAE；不授评测/qualification/实车信用。

[有限JSON](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/source029_tb2000_exact001_independent_review001.json) SHA：`7a932242e3b2613a2e70c330f1c45a72ff26a012bb14ceb0945cf7cca261806a`。native/PPO/remote动作均0。
