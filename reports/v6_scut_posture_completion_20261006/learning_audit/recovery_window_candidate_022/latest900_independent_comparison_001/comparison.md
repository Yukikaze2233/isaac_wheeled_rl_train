source022实际完整100 actor窗口，同口径描述性比较，qualification=null。

| 指标 | source022 首100 | source022 actor801–900 | 后−首 |
|---|---:|---:|---:|
| stand_230mm 静站平面均速 | 12.4910cm/s | 12.8222cm/s | 0.3311cm/s |
| stand_305mm 静站平面均速 | 10.9214cm/s | 11.9501cm/s | 1.0286cm/s |
| stand_430mm 静站平面均速 | 8.8476cm/s | 8.8973cm/s | 0.0497cm/s |
| forward_05 vx MAE | 30.3983cm/s | 29.5989cm/s | -0.7994cm/s |
| forward_05 signed vx | 19.3568cm/s | 20.1126cm/s | 0.7558cm/s |
| forward_05 signed bias | -30.2882cm/s | -29.5255cm/s | 0.7626cm/s |
| backward_05 vx MAE | 31.5871cm/s | 32.0702cm/s | 0.4831cm/s |
| backward_05 signed vx | -18.0885cm/s | -17.6175cm/s | 0.4710cm/s |
| backward_05 signed bias | 31.5777cm/s | 32.0062cm/s | 0.4285cm/s |
| rotate_1 yaw角速度MAE | 0.6368rad/s | 0.6227rad/s | -0.0141rad/s |
| rotate_1__reverse_train yaw角速度MAE | 0.6051rad/s | 0.6510rad/s | 0.0459rad/s |
| braking 实际samples加权 samples | 98953.0000 | 96485.0000 | -2468.0000 |
| braking 实际samples加权 mean_speed_m_s | 28.5088cm/s | 27.5187cm/s | -0.9901cm/s |
| braking 实际samples加权 above_3cm_s_fraction | 98.6751% | 98.7003% | 0.0252% |
| settled_stop 实际samples加权 samples | 179052.0000 | 183509.0000 | 4457.0000 |
| settled_stop 实际samples加权 mean_speed_m_s | 11.6327cm/s | 11.9791cm/s | 0.3464cm/s |
| settled_stop 实际samples加权 above_3cm_s_fraction | 92.9948% | 92.9775% | -0.0173% |
| 恢复混组 episodes | 6944.0000 | 7488.0000 | 544.0000 |
| 恢复混组 successes | 0.0000 | 0.0000 | 0.0000 |
| 恢复混组 failures | 2076.0000 | 2120.0000 | 44.0000 |
| 恢复混组 timeouts | 4868.0000 | 5368.0000 | 500.0000 |
| iteration_wall_seconds | 19.0675s | 14.9426s | -4.1249s |
| env_step_seconds | 15.9551s | 12.2398s | -3.7153s |
| optimizer_seconds | 0.1291s | 0.1111s | -0.0180s |
| diagnostics_seconds | 0.4765s | 0.3952s | -0.0813s |
| logger_seconds | 0.3546s | 0.2420s | -0.1125s |

- Same source022, nonpaired actor1–100 versus801–900; intervening700 actor updates, changing lessons, episodes, DR and exploration prevent causal attribution.
- Standing planar speed is not ten-second drift distance; completedstationary10s windows are eligibility-selected.
- Forward/backward vx and bias are signed; yaw is angular-velocity MAE, no signed yaw mean is available.
- Exact54-group recovery ended counts mix dynamicL0/L1/L2; their success fraction is not finalL2 qualification.
- Dynamic lesson counter deltas are24-frame averaged endpoint differences, not exact episode integers.
- Stop means and above3cm fractions are weighted by actual logged phase samples across six command-transition groups, with movement frames excluded.
- JSON95-group counts/pool frame means are independently recomputed; rawTFRecord reductions remain the unchanged CRCreader authority.
- Selected latest region CRC is certified; earlier records outside512MiB suffix are not claimed verified.
- Timing windows are not paired hardware scheduling trials; no source-only speedup conclusion.
- No mechanical, deterministic behavior, ONNX or hardware qualification follows; prior guards/native27.620mm peak are not reassessed.
