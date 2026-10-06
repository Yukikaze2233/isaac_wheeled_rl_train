# 2216 mean target/payment readback

Actual exit0 and600/2400, source021/ownREC3076/SIDE1.5. Birth/private RNG/std match. Decoded4096 births,3756 events and9 global peaks equal SIDE001 bitwise. No PPO or production change; qualification null; App.close return unknown.

| Native segment | 5ms samples | Raw wheel target rim mean | Delayed target mean | Fresh wheel mean | Gyro norm mean | Max leg speed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| script | 936 | 3.4730m/s | 3.4735m/s | 0.2525m/s | 2.2212rad/s | 2.6111rad/s |
| blend | 40 | 0.4433m/s | 0.4531m/s | 0.3274m/s | 0.2800rad/s | 0.6453rad/s |
| pure_rl | 1417 | 0.4321m/s | 0.4304m/s | 0.2920m/s | 0.6794rad/s | 1.4685rad/s |

Hold inputs physical984–2399,1416 samples: joint pass295; peak16 ticks/80ms versus required200/1s;82 breaks. Nonexclusive failures wheel787/gyro596/legs332/tilt39; at breaks32/58/17/0. Other four masks passed. At12s it is unfinished right-censoring.

During hold, raw wheel target rim mean0.4322m/s; delayed0.4305; fresh actual0.2920. Delayed target exceeds.25 in1062/1416; this is not the native actual-speed gate. Max-axis wheel decode-to-delay delta mean5.647rad/s/max22.754; leg delta mean0.01609rad.

| Hold input window | Samples | Joint pass | Breaks | Wheel/gyro/leg failures | Leg/wheel/gyro20ms payment |
| --- | ---: | ---: | ---: | --- | --- |
| 4.920–5.995s | 216 | 33 | 12 | 122/77/47 | -0.1789/-0.2493/-0.8250 |
| 6.000–7.995s | 400 | 107 | 27 | 224/139/88 | -0.3302/-0.4515/-1.4406 |
| 8.000–9.995s | 400 | 76 | 20 | 223/193/94 | -0.3432/-0.4041/-1.7221 |
| 10.000–11.995s | 400 | 79 | 23 | 218/187/103 | -0.3573/-0.4471/-1.6942 |

354 actor-owned intervals/7.08s paid original density×.02: leg-1.209551,wheel-1.552033,gyro-5.681967. Script/blend paid no native hold costs. Payment uses physics4n; last gate uses4n−1. Python dt weighting is diagnostic precision; inherited original total reward/event reconciliation passed.

Mean is already unstable, so stochastic exploration alone cannot explain this lane. Requests, delayed targets, physical wheel speed and gyro/leg failures coexist; no unique cause follows. The smallest proposed identification is one hold target-delay contrast with the same historical actor and other thresholds/physics/rewards/std fixed. This is a future0PPO proposal, not an implemented repair. Do not add duplicate taxes or tune weights without critic/advantage evidence; keep1s/3mm.

No five-positive-peak, full-negative ledger or criticV conclusions are added. Original inputs and frozen candidate bytes remain unchanged.
