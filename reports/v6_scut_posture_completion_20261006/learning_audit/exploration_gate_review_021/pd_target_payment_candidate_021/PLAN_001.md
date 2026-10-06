# 2216 mean settling observation

This artifact observes the frozen source021 / own REC3076 / mean / SIDE1.5 context. It preserves the original 4096 × 600 policy / 2400 physics loop, sampled actions, private and environment RNG, decode, delay, FSM, rewards and control. Only lane2216's first birth is retained. This is a diagnostic proposal, not a new model result or qualification.

The observer wraps the already constructed original methods once and returns their original objects:

| Boundary | Native data retained | Clock |
| --- | --- | --- |
| `V5Control.decode` return | Actual decode input in CONTROL order, encoder input, returned unwrapped leg targets, wheel targets and clipped actions | Once per 20ms policy interval |
| `PhysicalSignalDelay.targets` return | Original incoming and delayed leg/wheel targets, actual target lags | Once before each 5ms PD update |
| `V5Control.motor_efforts` return | Actual continuous control q, fresh dq, delayed targets, original RL PD request, bounds and returned clipped torque, before recovery may overwrite the request | Before physics k; input state is physics k−1 |
| `RecoveryFSM.update` return | Original input sensors, native eight stable masks, stable/holding/ready state, dwell/peak/break counters, phase and motion release | Before physics k; no predicate reconstruction |
| Existing hold producer return | Same settings/diagnostics object, original mask, actual broad/fine densities and three replaced component values | After four physical steps, once per 20ms |
| Original step return | Original pre-reset diagnostics, whole-interval actor mask, returned reward/done, actual components and inherited event ledger | After physics4n; final gate consumed physics4n−1 |

The eight masks already exist as `fsm.stable_pass_masks`; no new FSM producer is required. Existing003 effort setter / post-physics / before-reset hooks and the existing payment helper remain intact. A small outer observer adds the four method-return hooks and a read-only wrapper around the already installed hold producer. It does not read or evaluate the policy again.

Payment units remain explicit: the three actual densities are `leg_velocity`, `wheel_velocity`, `horizontal_angular_velocity`; each policy payment is `.02 × density`. Broad/fine densities come from the original diagnostics, not another reward call. The final native reward and discrete events retain the inherited reconciliation. Do not treat these 50Hz end-state payments as 200Hz samples or analytic entropy as payment.

First-cohort labels and begin-policy hold are retained from the existing payment interval before reset. The returned whole-interval actor mask attributes an actor-caused terminal; final pureRL does not erase it. Target rows, direct gate rows and payments use explicit policy/physics/input-state indices, with immutable copies taken before the next original call can overwrite tensors.

CPU checks must establish one original call per boundary, identical return objects, no RNG changes, delayed versus decoded targets kept distinct, correct first terminal/reset attribution, and restoration before parent finish / Kit.close on success and exception. Native admission remains pending until root observes exit and checks original SIDE001 identity/birth/private sampling/std and complete600/2400 receipts. Changed trajectories are preserved and explicitly marked; they do not gain a reproduction or causal claim. All-lane GPU scatter-add statistics and gzip artifact hashes are recorded separately, never used as bitwise per-lane payment equality. Keep the 1s, .75rad/s and 3mm criteria unchanged.
