# Matched native locomotion quality diagnostic

The final run completed in 265.35 s wall time using one headless Isaac Kit process, seven environments and the local RTX 4060. It consumed zero PPO updates. Both frozen ONNX actors used the same native V6 asset, 200 Hz physical feedback/PD, 50 Hz policy clock, box floor, seed 190619 and matched reset profile. The independent small playback fixture and native runtime preflight passed. This is one nominal short diagnostic, with no skill qualification or hardware acceptance.

`report.json` contains the complete runtime/config/model identities and stage statistics. `comparison_summary.json` additionally contains signed attitude means, attitude standard deviations, gyro/action RMS and lateral path displacement. Each case's `*.frames.jsonl` records all 200 Hz physical samples; action differences refer to 50 Hz clipped control actions repeated across the four physical substeps. Phase scalar results are in `events/`, without a local TensorBoard server. `validation.json` verifies all 14 traces' 200 Hz timestamps, physical command slew, zero stop commands and complete 2000-sample parking windows where available.

The command schedule was 2 s settling, 1.5 m/s² acceleration, 4 s constant command, normal release through the same command slope, then 12 s at zero visible command. The first 2 s after visible command zero form a separate warmup phase; the following 10 s form the parking window and own position anchor. The stationary case lasted 15 s, including its first 2 s settling. The same `advance_playback_command` joint velocity/yaw envelope and 4 rad/s² yaw slope were used; these straight trajectories requested yaw zero.

The old USB RTT module and signal-age module were absent. The command transport instance existed, but all seven nominal lanes were disabled by their absent communication profiles: actual loss probability was zero. This run does not assess the 80/60 ms signal-delay tail or packet-loss robustness.

| Case / metric | Flat actor | Speed v10 actor |
| --- | ---: | ---: |
| Stationary mean planar speed, 13 s analysis segment | 0.593 cm/s | 4.983 cm/s |
| Stationary peak displacement from segment anchor | 7.59 cm | 64.79 cm |
| After +0.5 m/s, 10 s parking mean planar speed | 0.567 cm/s | 4.668 cm/s |
| After −0.5 m/s, 10 s parking mean planar speed | 0.541 cm/s | 4.608 cm/s |
| After +0.5 m/s, parking peak displacement | 5.54 cm | 46.71 cm |
| After −0.5 m/s, parking peak displacement | 5.30 cm | 46.11 cm |
| +0.5 m/s constant-command pitch standard deviation | 0.59° | 4.07° |
| −0.5 m/s constant-command pitch standard deviation | 0.86° | 5.48° |
| +1 m/s constant-command pitch standard deviation | 0.47° | 5.56° |
| −1 m/s constant-command pitch standard deviation | 0.62° | 7.22° over the observed 3.3 s |

All seven flat trajectories completed. Four speed trajectories stopped at the unchanged lateral corridor boundary, y = ±3.7 m: +1 m/s at 14.88 s during parking, −1 m/s at 5.98 s during constant command, +3 m/s at 4.70 s and −3 m/s at 4.12 s during constant command. The final reported y positions were +3.701, −3.714, +3.713 and −3.711 m. These were boundary truncations; no fall/nonwheel-contact reason was recorded at those first terminals. The +3/−3 m/s constant-command observations lasted only 0.70/0.12 s, so they do not support full steady-state claims. The +1 m/s parking segment was also truncated and must not be counted as a full ten-second window.

The speed actor's observed phases had zero wheel torque saturation; its greater low/mid-speed pitch fluctuation and lateral departure are present without a wheel torque ceiling binding in this run. Flat +3 m/s braking briefly saturated the left/right wheel requests for 0.5%/0.25% of that phase. The flat actor's ±3 m/s diagnostic is beyond its trained speed coverage and gives no high-speed qualification.

`native_001` is preserved separately as a failed probe attempt: its flat recordings completed, then an incorrect diagnostic field name interrupted terminal recording. The final `native_002` used the corrected `diagnostics.reasons` field and passed all trace checks. The source hashes recorded by the final run still matched the files immediately after completion.
