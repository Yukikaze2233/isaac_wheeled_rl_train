# Yaw independent world interface review

The frozen `bcc65de507ed0ec877bb17c9401aeff13b52da2a` driver supports yaw_positive and yaw_negative through the same candidate-index/prepared-projection path. No runtime, physics, contract or original report change is required. Both original checkpoint/ONNX/seal identities, shared ABI and exact pair lineage authenticate before selection.

The two existing 003 projections passed native CPU authentication and reject the opposite yaw protocol. All four unchanged original measurements passed verify_single_measurement. Yaw stand305 has the original 22s budget: request ±1rad/s from 1s through before10s, release at10s, then the unchanged slew/limits and quiet observer. Other stand budgets remain10s; fallen budgets remain20s. Policy50Hz and physics/feedback200Hz are unchanged.

| Protocol | Existing receipt SHA256 |
|---|---|
| yaw_positive | `b6c2f058c92505400cf9a85342a759d61f110d3970879ce789f515f3cbdd7ba1` |
| yaw_negative | `162a3f08a6341ecc7706f65e71675e5315c7c8237486b03fd564db71f843919e` |

New `run_independent_motion005.py` defaults to CPU authentication only; explicit `--execute` runs yaw_positive indices0/1 then yaw_negative indices0/1 as four serial fresh Python processes. Each child receives both candidates and the corresponding same externally fixed projection pin. It records PID/starttime/argv/exit, matches constructor PID, verifies the strict single result and each protocol's pair fingerprint, and preserves failure without retry. Its unique future output is `rec_speed_mean42000_independent_motion005`. No output directory exists yet.

CPU syntax, help, actual default CLI, two-model authentication and all four command/projection/index mappings passed. Runner SHA: `4ed4570d0111263aa887b4d4fdfe7ad06966d27238a06fce03a7e9b5f4ee7894`. Only this new runner source was added; no commit or native/PPO/remote action was performed. Root can review/freeze it, then invoke:

```text
/home/yukikaze/isaacsim60-venv/bin/python -B /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/run_independent_motion005.py --adapter-commit bcc65de507ed0ec877bb17c9401aeff13b52da2a --execute
```

The previously passed strict independent test fixture covers nominal; this review adds actual yaw CPU authentication/measurement compatibility, not a fresh yaw simulation test. Old paired index1 has cumulative2200 audit steps versus1100 own ticks, so it is not relabeled as a strict single result. Actual fresh child counters will be required by the runner. Equal fingerprints will not establish hidden solver state, policy superiority, qualification or hardware readiness. Full source/input pins and minimum child flags are in the JSON.
