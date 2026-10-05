# Frozen b4 command-domain design audit

The frozen training contract permits pure 5 m/s and pure 3 rev/s commands without a direct nominal wheel-target conflict. **It does not establish dynamic reachability at all heights.** At 5 m/s the ±90 rad/s wheel-target ceiling has little positive-drive headroom; moving height sampling reaches the full 230–430 mm domain after 360 new actor updates, independently of measured competence. Mixed population and an unblocked schedule do not guarantee retention or successful high-speed/high-height behavior.

`inputs.json` pins 17 local files and four SCUT primary files by SHA256. Eleven imported/core files were compared byte-for-byte with Git blobs from `b4ce80964d47c0127e0b3d9929ad35a620bbf204`. These include native command, difficulty, control, reward and episode-count code. The actual 24500 contract SHA is `f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3`. The current SCUT source revision is `b8ff79f3df855faf9dc92f4a282bd80c42649466`; saved July YAML and present b8 interpreter code are distinguished below.

`analyze.py` checks every input pin, calls the native CPU `project_commands`, `validate_command`, `difficulty_envelope` and spin command update, and safely parses only bounded YAML data fragments. It creates no env, simulator, optimizer or actor and reads no remote event. `audit.json` contains exact numbers, parsed configuration fragments, source/clock provenance and available failure denominators. No core, contract, entry or remote process was edited; the candidate is neither implemented nor deployed. `qualification=false`, `root_cause_verified=false`.

Reproduce from the repository root to a fresh output path:

```sh
CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python reports/v6_locomotion_quality_repair_20261005/exploration_audit/command_boundary_audit/analyze.py --output /tmp/command_boundary_audit_recheck.json
```

## Wheel targets and nominal limits

Native `motion_limits.py:5–6` uses differential-drive wheel speeds `(vx ± W*yaw/2)/r`; the actual contract has r=0.06 m, W=0.4373 m, wheel-target max=90 rad/s and lateral-product cap=2.06 m/s². The asset wheel collision cylinders have radius 0.06 m (`robot.urdf:129,251`). `v5_control.py:28–52` clips legs to ±3 actions and wheels to ±9, scales wheel targets by 10 rad/s/action, uses leg PD160/2.5 with ±40 Nm, and wheel velocity feedback coefficient 0.6 with ±4.5 Nm output-side software cap. The 15.8 gear ratio is already represented by the wheel-output convention and is not multiplied into the actor target again (`manifest.json:98–100`; `v40/core.py:197–209`).

| Nominal request | Required wheel speeds | Native sampling projection |
| --- | --- | --- |
| vx=5, yaw=0 | +83.3333 / +83.3333 rad/s | unchanged |
| vx=0, yaw=3 rev/s | −68.6909 / +68.6909 rad/s | unchanged |
| vx=5, yaw=3 rev/s | +14.6424 / +152.0243 rad/s | vx=2.96005 m/s, yaw=0.695933 rad/s |

`SkillCommands.sample` applies the joint projection after command/difficulty sampling (`skill_commands.py:291–294`). `motion_limits.py:19–23` first scales vx/yaw into the wheel-speed diamond and then scales yaw to satisfy the lateral-product cap. Therefore independent maxima in the contract are not an unprojected simultaneous ordinary command. The legacy fallback in `env.py:1028–1034` has a different 4.35 m/s cap; this worker uses populated `skill_specs` and returns through the skill path (`env.py:1005–1017`), so that fallback is not its active command limit.

The nominal pure rolling target ceiling is 5.4 m/s and pure yaw target ceiling is 24.697 rad/s (3.93065 rev/s). At 5 m/s, rolling dq=83.3333 rad/s leaves only 6.6667 rad/s between dq and the maximum target; the largest nominal positive PD request at that speed is `0.6 × 6.6667 = 4.0 Nm` per wheel, below the 4.5 Nm cap. This is before motor-strength reduction, unilateral degradation, held/delayed transport, response, slip and external load (`env.py:1271–1303`; `training_schedule.py:164–166`). The target ceiling is not a hard physical dq clamp, and these calculations do not prove a sustainable speed or acceleration.

The real command slopes are 1.5 m/s² and 4 rad/s², requiring at least 3.333 s to ramp zero→5 m/s and 4.712 s to ramp zero→3 rev/s. Resampling short requests and early physical termination can shorten actual high-command exposure. The shared pitch projection also matters mathematically: for purely horizontal world motion, zero roll/yaw and nominal no-slip wheel target speed, the reward's body-x projection reaches 5 m/s only up to approximately 15.793° pitch at the 5.4 m/s target ceiling. This restricted calculation is not a new physical test, a general slope bound, or proof of the regression cause.

## Spin translation is an explicit command-path exception

The `spin_translate_3rps` spec has a fixed world-reference translation `[0.5,0]` m/s, with sampled body vx=0 and yaw=3 rev/s. Later `skill_commands.py:510–513` sets body vx=`0.5*cos(yaw_angle)`, and `env.py:1912–1915` directly overwrites this coordinate vx and sets command acceleration to zero; that special path intentionally bypasses the 1.5 m/s² longitudinal slew and does not repeat the lateral projection.

The CPU audit calls the native update on a **synthetic command state only**, at yaw angle zero: the later request becomes vx=0.5, yaw=18.8496, nominal wheels −60.3576/+77.0243 rad/s, and `|vx*yaw|=9.42478`, greater than 2.06. The body coordinate vx derivative can have magnitude 9.42478 m/s² at constant yaw rate. Coordinate rotation is not automatically equal to physical acceleration, so this is a source-level envelope/semantics exception, not a demonstrated physical violation or root cause. A future design must explicitly define whether the ordinary lateral metric applies to the world-reference spin path; blindly adding a clamp could change the intended reference/reward problem.

## Height/command coupling

`locomotion_difficulty.py:13–21,34–49` expands by the domain actor clock from origin 3563:

| New actor updates | vx max | yaw max | Moving height interval |
| ---: | ---: | ---: | --- |
| 0 | 2 m/s | 1 rev/s | 280–350 mm |
| 120 | 3 m/s | 1.625 rev/s | 250–380 mm |
| 240 | 4 m/s | 2.25 rev/s | 240–400 mm |
| 360 | 5 m/s | 3 rev/s | 230–430 mm |

The sealed 24500 actor's task clock 4185 and saved later event clocks 4514–4603 are beyond full-domain activation at 3923. This is deterministic availability, not observed mastery. Static heights are excluded from this narrowing (`moving_spec`, `bounded_height_sampling` at 52–65). No active performance curriculum is present in the worker contract.

`forward_5__height_train`, `rotate_3rps__height_train` and analogous groups sample height independently within the current interval: nominal fraction 0; exact endpoint mass 20% total; continuous sampler doubles density in each outer 10% band (`skill_commands.py:247–267`; `training_schedule.py:199–212`). At the full interval, endpoint plus continuous outer-band probability is analytically **46.6667%**. Half begin with the requested endpoint height reference immediately; half transition the reference from 305 mm over 4 s (`skill_commands.py:261–267,494–499`). This is command-reference timing, not proof that physical birth pose equals the target. The 430 mm transition's peak reference rate is 46.875 mm/s and can overlap the speed ramp.

The already completed engineering 4096 population contains 1,561 height-train lanes (38.1104%); high-motion/high-transition pools contain 984+384 lanes (33.3984%), while stand anchors/continuous contain 468+311 lanes. This population reference authenticates layout for this contract, not a newly read current worker state or full high-command success. Full height endpoints plus speed/yaw extremes are therefore meaningful planned exposure, not rare fallback. The combination may be too demanding for the current policy, but source limits alone cannot determine that.

## SCUT saved domains and phases

Primary rough config: `rough_dash/2026-07-19_19-15-07/params/env.yaml`, SHA `da26c974517564480c8900b6adf339d1648c99902ea3d9bb86fa48387df302c4`. Primary flat config: `flat_and_rotation/2026-07-29_23-20-03/params/env.yaml`, SHA `a729486fcb5a8a24c25eafed96709b4ec4e6b29221c615a09bda5343b19a1b21`.

Both ordinary configured domains are vx±2.7 m/s, vy=0, yaw±2π rad/s, heading±π, resample 5–15 s, heading stiffness 5, standing fraction 0.1 and heading fraction 0.5 (rough1744–1765; flat1360–1381). Both global height intervals are 0.20–0.42 m, but rough uses `use_absolute_height=false` and flat true (rough1920–1925; flat1525–1530), so equal numeric ranges must not be treated as equal physical reference semantics.

| Special configured pool | rough start iteration / fraction | flat start iteration / fraction | vx / yaw |
| --- | --- | --- | --- |
| Spin 1 | 0 / 0.15 | 3000 / 0.15 | vx±0.1; abs(yaw)=1–1.625 rev/s |
| Spin 2 | 0 / 0.15 | 4000 / 0.15 | vx±0.1; abs(yaw)=1.625–2.25 rev/s |
| Dash | 0 / 0.20 | 2000 / 0.30 | abs(vx)=2–3; yaw±1 rev/s |
| Zero command | 0 / 0.10 | absent | vx=vy=yaw=0 |

Special pool admission requires episode age≥5 s; `special_mode_require_stable=false` in both saved configs, so their listed stable thresholds are not an active competence gate. All special `height_range` fields are null, inheriting the global/terrain height sampler rather than imposing a universal high-vx height cap (rough1840–1919; flat1456–1524). Iteration starts are saved numerical schedule boundaries, not proof of the exact historical effective population.

Rough has terrain-specific restrictions: high/low-speed stair height220–340 mm with |vx|1.5–2.7 or0.5–1.5; selected high stair/slope height240–320 mm. Several set `disable_special_mode=true` and nonheading yaw±0.1, but heading yaw fields are null and other stair-slope rows allow nonheading yaw±π. Flat has no terrain command overrides. Thus SCUT uses some narrower terrain-conditioned combinations, while neither saved config proves all commands physically feasible or competence-gated. Present b8 command/height interpreter source supports the configured meanings, but exact July source/SDK, actual sampled population and runtime success were not authenticated here.

## Available height-failure observations

These observations use the already saved CRC-provenance native-event difference, global event steps **24828–24917 / task actors4514–4603**, a 90 actor-update interval queried at `2026-10-05T02:22:23.376571+00:00`. They include exploration, resets, nonstandard starts, DR and delay; they are not matched deterministic evaluations. Each release group has 73,440 observed lane-policy frames. Failure denominator is ended episodes, not frames; boundary counts are a subset of timeouts (`episode_metrics.py:130–142`), not a separate additive partition.

| Release group | Nominal-height failures / ended | Height-train failures / ended | Nominal / height MAE |
| --- | --- | --- | --- |
| Translate | 22/124 (17.74%) | 26/132 (19.70%) | 9.99 / 11.62 mm |
| Yaw | 7/44 (15.91%) | 14/54 (25.93%) | 7.35 / 9.00 mm |
| Combined | 26/107 (24.30%) | 33/121 (27.27%) | 10.79 / 11.94 mm |

Available stand anchors are 230 mm19/129 (14.73%), 305 mm21/139 (15.11%), 430 mm25/151 (16.56%); each has 112,320 frames. The selected endpoint retains only 13 groups while its aggregate count covers all135. **All high-speed height-train group distributions and failure-by-height bins are unavailable in the saved compact evidence**; `complete_all_height_train_failure_distribution=null`. These limited pairs suggest additional difficulty, without proving height causes failure or characterizing 5 m/s/3 rev/s performance.

For a future frozen repair, first measure sustained high-command exposure, actual targets/dq/effort saturation, separated pitch/gyro and per-height episode failure denominators under fixed conditions. A narrow single design variable is the moving-height expansion interval, retaining static/low-speed coverage while comparing the same learning budget and actor/delay clocks; do not simultaneously alter effort weights or spin semantics. Any competence-based expansion must use actual complete-window evidence and remain separate from qualification. No automatic no-forgetting guarantee follows from nominal mixture proportions or removing gates, and no new training or physics run was performed for this audit.
