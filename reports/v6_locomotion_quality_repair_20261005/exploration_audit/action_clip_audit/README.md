# 24500 action clipping, effort and velocity projection audit

The completed traces expose very little action clipping, including **zero sent-action clipping in every standing phase**. They expose substantial wheel torque saturation under stochastic sampling. SCUT's selected **rough_dash** saved reward coefficients charge torque squared and positive wheel power at one tenth of ours. Its flat_and_rotation saved run uses the same effort coefficients as ours; the difference is task-specific. These are measured exposures and source differences; none establishes the cause of locomotion regression or a useful learned repair. `qualification=false`, `root_cause_verified=false`, `candidate_implemented_or_deployed=false`.

`inputs.json` pins the actor PT, seal, contract, agent config, four completed reports and their runtime/birth/cohort/config receipts, all action/physical traces, the actual frozen control/reward core, and two SCUT primary files by SHA256. `audit.json` contains every case and phase denominator, sums and derived means. `analyze.py` reads the inputs on CPU, checks all pins before writing a result, authenticates the completed inference scope, and verifies native clipping against each physical frame. It does not deserialize the PT, construct an optimizer, start a simulator or change an input.

Reproduce from the repository root, choosing a fresh output path:

```sh
CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python reports/v6_locomotion_quality_repair_20261005/exploration_audit/action_clip_audit/analyze.py --output /tmp/action_clip_audit_recheck.json
CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python reports/v6_locomotion_quality_repair_20261005/exploration_audit/action_clip_audit/validate_analysis.py --output /tmp/action_clip_audit_validation_recheck.json
```

The NumPy dependency is used for CPU arithmetic. The analysis streams each large trace once while hashing; input verification failure prevents output. Both scripts refuse to overwrite an existing output. Validation passed **10 CPU checks**, including SHA mismatch, noncompleted report, nonfinite JSON/tensor, tensor ABI and the two following history rows affected by a clipped action. Failure fixtures are temporary and never alter the observed reports.

## Authenticated scope and denominators

All four runs use the same sealed 24500 actor, PT SHA `78dc24240038910b3844cc609a26d259075239b137c008ee038de7a76b4963e7`, input task actor update 4185, and unchanged actor tensors. Formal updates consumed and optimizer steps are zero. Clean diagnostics use task clock 0; pipeline diagnostics use 4185 with the inherited physical signal delay. Role changes can change generator consumption; cross-role random streams are not assumed identical.

The actual frozen native control/reward files match `b4ce80964d47c0127e0b3d9929ad35a620bbf204`. Clean runs use harness `13439727c365ef126f19d6da35c3f2f7291ae504`; pipeline runs use `87c174e78fbec4d6d0c894a4a2d673a7a214cb46`. Each report is completed with source and sealed-input final verification. Each run has 3 cases × N4, seed 190619, 20 s, 50 Hz policy / 200 Hz physics, 1,000 policy records, **12,000 lane-policy samples and 48,000 lane-physics samples**. All 12 lanes reach native timeout; success/failure/boundary/censor counts are zero. This is a 0/±0.5 m/s diagnostic, with no high-speed or full training-DR qualification.

Per case, whole-run denominators are 4,000 lane-policy / 16,000 lane-physics samples. The following phase counts are identical across the four runs and both signed-motion cases. Leg-axis counts are 4× lane counts; wheel-axis counts are 2× lane counts. Standing phases are not substituted for complete stopping windows.

| Case / phase | Lane-policy samples | Lane-physics samples |
| --- | ---: | ---: |
| Every case: settle | 400 | 1,600 |
| Stand: stand | 2,600 | 10,400 |
| Stand: hold | 1,000 | 4,000 |
| Each signed motion: acceleration | 68 | 272 |
| Each signed motion: steady | 800 | 3,200 |
| Each signed motion: braking | 68 | 272 |
| Each signed motion: stop_warmup | 400 | 1,600 |
| Each signed motion: stop | 2,000 | 8,000 |
| Each signed motion: hold | 264 | 1,056 |

The current stand-labelled phase lasts 13 s. Complete stop samples begin after the visible command is zero and the 2 s stop warmup; each signed-motion case contributes all later 10 s assigned to `stop`. The source fixture/cohort remains the authority for phase and terminal semantics.

## Control limits and reward paths

Our `v5_control.py:28–52` clamps **leg actions to ±3, wheel actions to ±9**, then maps leg targets by 0.25 rad/action and wheel targets by 10 rad/s/action. Wheel target limits are therefore ±90 rad/s. **±4.5 Nm is the wheel torque limit, not an action limit.** Leg PD is kp=160 / kd=2.5 with ±40 Nm; wheel velocity feedback coefficient is 0.6, with the wheel-output software cap applied without another gear multiplication (`v40/core.py:197–209`). Actual frame bounds are always `[40,40,4.5,40,40,4.5]` in control order.

Our `env.py:1233–1245` stores clipped actions and clipped previous histories; `rewards.py:428–430` pays the regularizer on those histories. The selected SCUT source stores raw `actions.clone()` (`env.py:2830`), splits them into targets before target clamping (`2871–2891`), and uses the raw action histories (`3107–3108`, `3554–3561`). The selected run disables its optional action filter. Raw action regularization coefficients match ours: first difference −0.01 across six axes, second difference −0.05 on four legs and −0.01 on two wheels.

Velocity/yaw tracking clamps their errors inside exponential terms (`rewards.py:409,411`); this is separate from action clipping. The action and effort costs have no additional reward clamp. Native `env.py:1525` sums the components and multiplies by policy dt, without a generic total-reward clip in this measured ordinary-motion path.

The reconstructed action regularizer density is:

```text
−0.01 Σ(a−a_prev)² −0.05 Σleg(a−2a_prev+a_before)² −0.01 Σwheel(a−2a_prev+a_before)²
```

Reconstruction uses the actual sent action, zero history at birth, and continuous history across phases; the clipped counterpart applies native bounds before all history operations. Payment multiplies the density by 0.02 s. CPU float64 arithmetic can differ slightly from native float32 rounding, but a no-clipping sequence has exactly equal raw and clipped reconstruction.

| Run | μ leg clips / 48,000 axes | Sent leg clips / 48,000 | Sent wheel clips / 24,000 | Raw − clipped mean density | Fraction of raw cost magnitude |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean mean 002 | 0 | 0 | 0 | 0 | 0% |
| clean stochastic 003 | 2 | 13 | 0 | −0.0000578969 | 0.01582% |
| pipeline mean 004 | 7 | 7 | 0 | −0.0000000723081 | 0.000587% |
| pipeline stochastic 005 | 6 | 23 | 0 | −0.0000721010 | 0.01965% |

Every `stand_305mm/stand` phase has zero μ and sent clips. Pipeline mean's seven μ clips occur in the backward early steady phase. All completed `stop` windows have zero μ clips; pipeline stochastic has two sampled leg clips there. Potential sampled actions in a mean run are recorded but **not sent**, so their clipping does not count as physical exposure. Histories create 0/25/11/49 differing lane-policy regularizer rows respectively out of 12,000. These observations do not support action clipping as an explanation of the standing regression in these traces.

Our `rewards.py:424–425` uses −1e−4 for applied torque squared and positive wheel power; the actual `scut_effort_reward_scale` is 1 (`env.py:1487–1488`). SCUT's saved `rough_dash/2026-07-19_19-15-07/params/env.yaml:2080,2083` sets each coefficient to −1e−5. SCUT's functions use actual applied torques (`env.py:3527–3546`). This establishes a tenfold coefficient difference, without assuming equal robot dynamics or trajectories.

Effort density below is reconstructed from the **last 200 Hz substep before each 50 Hz reward**, as native payment does. Its denominator is 12,000 lane-policy samples per run. Torque saturation separately uses all 200 Hz frames, denominator 96,000 wheel-axis / 192,000 leg-axis samples. It is not a 50 Hz-only saturation measure.

| Run | Wheel cap hits / 96,000 | Leg cap hits / 192,000 | Mean torque² + positive wheel-power density | Same-trace density at 0.1 scale |
| --- | ---: | ---: | ---: | ---: |
| clean mean 002 | 9 (0.009375%) | 0 | −0.00700737 | −0.000700737 |
| clean stochastic 003 | 34,280 (35.7083%) | 962 (0.501042%) | −0.0380951 | −0.00380951 |
| pipeline mean 004 | 2,163 (2.253125%) | 0 | −0.00784122 | −0.000784122 |
| pipeline stochastic 005 | 36,072 (37.575%) | 1,156 (0.602083%) | −0.0516006 | −0.00516006 |

Standing-phase wheel cap hits are 0/7,397/875/7,997 out of 20,800 wheel-axis samples. Saturation and raw action clipping are different events. The compact per-case/per-phase data also retains requested/applied maxima and separate motor/power costs.

An independently testable reward-only candidate is the existing scalar `scut_effort_reward_scale: 1 → 0.1`. At identical trace inputs its per-step payment change is exactly `+9e−5 × [Σtorque² + Σpositive wheel power] × 0.02`. A frozen same-PT inference trajectory should be unchanged by this reward-only setting. Learning benefit requires a future controlled learning comparison; no such gain, implementation or deployment is claimed here. This is a counterfactual candidate, not the selected next repair: the later actual payment capture shows effort is small compared with static and tracking costs. Changing raw-action regularization is also reward-only and would require preserving the clipped actor previous-action ABI, but the observed tiny exposure gives it little support for explaining standing drift.

The separately verified SCUT flat run is `pretrained/26_infantry/flat_and_rotation/2026-07-29_23-20-03/params/env.yaml`, SHA256 `a729486fcb5a8a24c25eafed96709b4ec4e6b29221c615a09bda5343b19a1b21` at the same b8 revision, unchanged. It activates torque/power −1e−4 (lines 1685/1688), the same three smoothing coefficients (1692–1694) and stand_still_lin_vel −2 (1706). This supplementary read-only correction is outside the original two-file rough input manifest; it does not change its pinned analysis.

## Velocity frame and pitch projection

**Yes: ours multiplies body vx by `sqrt(1−gx²)`, and the saved SCUT source has the same pitch factor mathematically.** Ours pins `reward_velocity_reference=base_link_origin` (`env.py:1356–1357`) and applies `velocity[:,0] × sqrt(max(1−gravity[:,0]²,0))` (`rewards.py:397–398`; reference motion also does so at `441`). SCUT b8 source obtains Euler pitch and uses `root_lin_vel_b[:,0] × cos(pitch)` (`env.py:3703–3712`). With unit gravity, gx=sin(pitch), so our factor is |cos(pitch)|; canonical Euler pitch is within ±π/2, giving the same factor. Across our 192,000 observed physical frames, the largest factor difference is **2.97e−8**, consistent with recorded float precision. This applies SCUT's formula to our trace; it does not evaluate a SCUT physical rollout.

The factor is a body-x vector projection, not a full inverse rotation into horizontal heading velocity. With zero roll/yaw and purely horizontal world motion, body vx is `world vx × cos(pitch)`, and either reward formula yields `world vx × cos²(pitch)`. This common mathematical behavior is not a new difference between these source formulas.

The following compares the actual reward-source projected body velocity against independently calculated world velocity along quaternion yaw. Each cell uses the full 3,200 lane-physics frames of that signed steady phase and reports mean absolute difference in m/s.

| Run | Forward steady | Backward steady |
| --- | ---: | ---: |
| clean mean 002 | 0.00295219 | 0.00353883 |
| clean stochastic 003 | 0.00990506 | 0.01034034 |
| pipeline mean 004 | 0.00406274 | 0.01204942 |
| pipeline stochastic 005 | 0.02053049 | 0.01675751 |

Whole-trace maximum body-to-world rotation consistency error is below 5.38e−7 m/s. The observed instantaneous COM-versus-link velocity difference can reach 0.35956 m/s, so the point of velocity measurement can matter. However, the selected SCUT run's saved parameters do not authenticate the historical SDK or its `root_lin_vel_b` velocity-origin implementation. The b8 source describes current function semantics; it is not proof of the exact source/SDK used for the July run. We can authenticate our link-origin choice, but cannot claim historical SCUT origin equality or mismatch from these two files. Neither shared projection math nor this unauthenticated origin issue proves the regression cause.

`audit.json` SHA256: `11f9093b40c097134c5cc6dc863ae94d87d91c7a1526a3a6c22ef24093a96f92`. Input manifest SHA256: `901c026044eb1f7d958bcf405400c27eed276f33ffe12a78063c0d081a0ce166`. Source and output pins are preserved in `audit.json` and `validation.json`; no raw trace, model, core or entry file is copied or modified by this audit.
