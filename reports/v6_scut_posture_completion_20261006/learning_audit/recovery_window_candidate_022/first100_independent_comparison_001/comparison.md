实际同口径100 actor窗口比较；描述性结果，qualification=null。

| 指标 | source021 第五100 | source022 首100 | 新−旧 |
|---|---:|---:|---:|
| stand_230mm 静站平面均速 | 13.1224cm/s | 12.4910cm/s | -0.6314cm/s |
| stand_305mm 静站平面均速 | 10.8705cm/s | 10.9214cm/s | 0.0509cm/s |
| stand_430mm 静站平面均速 | 7.9909cm/s | 8.8476cm/s | 0.8567cm/s |
| forward_05 vx MAE | 29.8559cm/s | 30.3983cm/s | 0.5424cm/s |
| forward_05 实测signed vx | 19.8848cm/s | 19.3568cm/s | -0.5280cm/s |
| forward_05 signed bias | -29.7389cm/s | -30.2882cm/s | -0.5493cm/s |
| backward_05 vx MAE | 30.3285cm/s | 31.5871cm/s | 1.2587cm/s |
| backward_05 实测signed vx | -19.3568cm/s | -18.0885cm/s | 1.2683cm/s |
| backward_05 signed bias | 30.2597cm/s | 31.5777cm/s | 1.3180cm/s |
| rotate_1 yaw角速度MAE | 0.6866rad/s | 0.6368rad/s | -0.0498rad/s |
| rotate_1__reverse_train yaw角速度MAE | 0.5767rad/s | 0.6051rad/s | 0.0284rad/s |
| start_stop_05 驻停条件均速 | 12.3820cm/s | 12.3755cm/s | -0.0065cm/s |
| start_stop_1 驻停条件均速 | 12.2626cm/s | 11.3620cm/s | -0.9006cm/s |
| velocity_curve_1 驻停条件均速 | 11.0166cm/s | 10.7889cm/s | -0.2276cm/s |
| iteration_wall_seconds | 20.2676s | 19.0675s | -1.2000s |
| env_step_seconds | 17.0136s | 15.9551s | -1.0585s |
| optimizer_seconds | 0.1283s | 0.1291s | 0.0008s |
| diagnostics_seconds | 0.4809s | 0.4765s | -0.0043s |
| logger_seconds | 0.4019s | 0.3546s | -0.0473s |

- Nonpaired source021 actor3486–3585 versus source0223872–3971; intervening286 actor updates, new episodes/DR/exploration and10critic adaptation confound causal attribution.
- 230/305/430 planar velocity is standing speed, not ten-second drift distance. Completedstationary10s reports are eligibility-selected and can have earlier anchors.
- Forward/backward actual vx and bias are signed; yaw is angular-velocity MAE, not angular position error or a signed yaw mean.
- Exact54-group mixed ended counts include dynamically sampledL0/L1/L2; success fraction is not the finalL2 chain rate.
- Dynamic lesson cumulative counters are24-frame averaged endpoint values; their differences are not exact episode integers.
- Instantaneous marginal hold gates do not determine continuous1s joint dwell; support/body_clear are observer proxies, not actual contact proof.
- HoldCost tags are time means of per-frame actor-hold conditional densities; not fulllane payment or advantage attribution.
- Same100 width does not pair hardware scheduling; physical/env,optimizer,diagnostics,logger,wall timing means cannot establish a source-only speedup.
- No raw TFRecord series are re-read; reduced JSON pool/partition arithmetic is independent, per-tag reductions remain certified by the frozenCRC reader.
- No mechanical,behavior,ONNX or hardware qualification follows from this comparison; residual3mm guards/native27.620mm peak are not reassessed.
