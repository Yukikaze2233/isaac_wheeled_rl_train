# Actual independent yaw world review

Four fresh native processes completed with exit0. External PID/starttime/argv, constructor PID, complete pair identities, the two literal historical projection pins and strict single result validation agree. This reviewer independently invoked all four validators on CPU; no Kit/env import, native/PPO/remote action or original-file edit occurred. Each protocol's two full reset fingerprints match.

| Protocol | Model | Actual yaw mean rad/s | Request MAE rad/s | Applied MAE rad/s |
|---|---|---:|---:|---:|
| yaw_positive | REC41393 parent | 1.002135355388 | 0.052129661592 | 0.031056725203 |
| yaw_positive | speed42000 task597 | 1.701714736596 | 0.741492643911 | 0.724825850502 |
| yaw_negative | REC41393 parent | -0.945745669925 | 0.056249504165 | 0.033772797594 |
| yaw_negative | speed42000 task597 | -1.369367426574 | 0.415582837721 | 0.395542225963 |

Each candidate uses **180 samples**: stand305 lanes4–7,45 samples each, active nonzero requested yaw, only the fixed5Hz lattice `.02+n*.2` from1.02 through9.82s. Additional terminal/FSM event rows are excluded. These tracking statistics and per-lane maxima equal the historical003 values under the same filter. They show larger request/applied yaw errors for speed597 in both signs in this fixed test; they do not establish a unique training cause or general superiority.

All four native50Hz quiet windows contain **2000 frames**,4×500, from actual zero applied command10.24s to completed10s window20.24s. Episodes end22s; quiet `episodes=0` is expected because the quiet mask has already ended.

| Protocol | Model | Quiet speed cm/s | Quiet drift max m | Quiet height MAE mm | Quiet tilt max deg |
|---|---|---:|---:|---:|---:|
| yaw_positive | REC41393 parent | 2.280029577 | 0.221640676 | 1.133868381 | 1.532123804 |
| yaw_positive | speed42000 task597 | 2.456618829 | 0.196039110 | 4.724091083 | 10.000794411 |
| yaw_negative | REC41393 parent | 2.435160992 | 0.238586873 | 1.083856478 | 1.528671265 |
| yaw_negative | speed42000 task597 | 2.143820850 | 0.174528047 | 4.380370766 | 4.837440014 |

The complete quiet native metrics equal each corresponding003 result. Parent's complete native episode metrics also reproduce003; speed's stand305 metrics reproduce003, while fallen metrics change. All four worlds count **11/16 fallen full-chain successes and5/16 script_timeout failures**, left2/4,right1/4,belly4/4,back4/4. All28 first episodes ended; no censoring or failure exclusion.

For yaw_positive, fresh fallen prefixes have **0/16 lanes with a physical difference** across **527 lane/grid comparisons** before either firstBLEND; firstBLEND times equal: True. Historical paired prefixes have 16/16 differing lanes. Five fields are position,velocity,omega,gravity,height. Per-lane preBLEND and all-active maxima are kept separately in JSON.

For yaw_negative, fresh fallen prefixes have **0/16 lanes with a physical difference** across **527 lane/grid comparisons** before either firstBLEND; firstBLEND times equal: True. Historical paired prefixes have 16/16 differing lanes. Five fields are position,velocity,omega,gravity,height. Per-lane preBLEND and all-active maxima are kept separately in JSON.

Source/evidence SHA pins, actual four process receipts, per-lane tracking maxima, native quiet/full-episode summaries and complete ended denominators are in the companion JSON; large traces are referenced by SHA rather than copied. These results do not prove hidden solver/RNG equivalence, unique training causality, permanent forgetting, qualification or hardware readiness.
