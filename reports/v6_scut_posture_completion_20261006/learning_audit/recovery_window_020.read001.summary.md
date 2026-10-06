# Source020首个完整100更新窗

现有Kaiser原生TensorBoard事件，CRC及必需tag完整；actor2769–2868、event29689–29788，每环境48s。最近50窗是100窗的子集，不相加。带探索训练数据，不是固定均值评测或功能资格。

| 行为 | 平面均速 cm/s | 前向速度 MAE cm/s | yaw角速度 MAE rad/s | 高度 MAE mm |
| --- | ---: | ---: | ---: | ---: |
| stand_230mm | 11.86 | 7.68 | 0.4150 | 5.38 |
| stand_305mm | 10.29 | 6.19 | 0.4250 | 5.32 |
| stand_430mm | 7.63 | 6.30 | 0.3585 | 5.76 |
| forward_05 | 25.33 | 26.35 | 0.4426 | 5.52 |
| backward_05 | 24.66 | 27.20 | 0.4375 | 5.71 |
| forward_1 | 53.38 | 46.42 | 0.5442 | 5.94 |
| backward_1 | 47.75 | 51.68 | 0.4782 | 6.50 |
| rotate_1 | 11.33 | 7.50 | 0.5539 | 6.16 |

100次更新wall均值17.073s，中位16.804s；env均值13.988s，optimizer0.128s。共享GPU/启动条件未配对，不能将差异全部归因于COM。

恢复静态场景合计{'successes': 0, 'failures': 2333, 'timeouts': 4810, 'scope': '54 static scenario labels mix dynamic lessons; not original-chain rate'}。zero successes是真实观测，但场景标签不等于动态lesson或完整自起链分母。既有zero-count驻停长窗记unavailable，不能补零；lifetime最大倾角不是本窗最大。

修复候选尚未部署：gate尺度hold运动成本、L0稳定后原目标桥。source020是已运行工程修复，不能把候选奖惩写成已生效。
