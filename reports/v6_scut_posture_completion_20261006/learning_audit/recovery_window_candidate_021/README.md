Source021 native recovery window reader. This isolated artifact reads the existing event with the frozen020→019 CRC/parser/cumulative metrics. It does not start SSH, AppLauncher, GPU, PPO, evaluation or a TensorBoard server, and does not modify training or admitted files.

The default first50 window requires actor60 completed: actor3086–3135, native event30016–30065, actual Behavior baseline30015. First100 requires actor110: actor3086–3185, event30016–30115. Parent consumption29996/task3075 remain authoritative. Ten critic-only collections29997–30006 are charged and remain in cumulative observed-frame denominators, but do not enter either selected actor interval. `--end-update` is the actor count in attempt005, excluding critic10.

`--window-mode after_critic_baseline` optionally selects actor1–50/100 from actual Behavior baseline30005. Missing any of the95-group baseline fields leaves the report pending; no zero baseline is invented. Default uses ten actor updates of baseline exposure, consistent with the previous reader.

Root copies the final script, both exact frozen readers019/020, the pinned first-PPO verification and inventory to a separate remote evidence directory. The launch argument is the immutable root `recovery_quality_launch_021.json`. Example for the default explicit first50 (substitute the directory containing the copied artifacts):

```sh
/home/kaiser/robot-rl-sim60/env/bin/python -B /REMOTE_EVIDENCE/read_recovery_window_021.py \
  --reader020 /REMOTE_EVIDENCE/read_recovery_window_020.py \
  --reader019 /REMOTE_EVIDENCE/read_recovery_window_019.py \
  --launch /home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/recovery_quality_launch_021.json \
  --first-ppo /REMOTE_EVIDENCE/first_PPO_verification_021.json \
  --inventory /REMOTE_EVIDENCE/frozen_source_inventory_021.json \
  --updates 50 --end-update 60 \
  --output /REMOTE_EVIDENCE/recovery_first50_021.json
```

Output uses exclusive creation. `read_failed` exits2; `pending_requested_actor_windows` exits0 with complete=False and exact missing tags/steps. Consumers must examine report status/complete, not only process exit. Live native cmdline, PID/PPID, process state and before/after PPID and starttime are checked. First-PPO records no historical starttime ticks, so this reader asserts read-interval ownership continuity rather than inventing a historical starttime match. Runtime population, config/module/gate binding, source hashes and immutable metadata are authenticated separately. The actual event is the only curve source.

New teaching output has all10 existing HoldCost tags, all8 original hold gates,3 hold-state tags and command means for all54 static recovery groups. A missing record blocks complete status. Dynamic L0 lesson success, original-chain success and configured zero-torque release duration remain separate. Native tags do not log L0-specific motion-release count or original-goal exposure: those fields are explicitly unavailable, even when static group command means are nonzero. Conditional HoldCost densities/gate fractions are24-policy-frame logger means; they are not exact lane sums, motor current, actor advantage, or continuous1s joint gate success probability. Signed actual yaw angular velocity remains unavailable; yaw MAE is angular-velocity error, not heading-angle error. PPO quality and hardware qualification are not awarded.

CPU fixtures only test the new clock/tag/ownership adaptation, reuse the old real TFRecord writer and metric producers, and do not repeat the entire old suite. Tested27 passed; native remote reading is still root-owned and was not run by this agent.
