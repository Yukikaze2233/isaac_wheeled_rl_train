# REC3076 local native mean/stochastic diagnosis

This is a local observation of the actual own recovery seal at task actor3076,
consumption30007, stage/global16444, actor consumption29857 and lineage16514.
PT SHA is `286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31`.
The entire downloaded seal equals the actual immutable first-PPO proof. The
v19 parent contract, full actor/critic/Adam and saved CPU/env/CUDA RNG state are
checked by the unchanged native002 driver. No optimizer is instantiated.

Source021 is `d0dce6181df83b03fe166abcbadef3c81de2e57e`; its inventory and native
recovery worker remain exact. The context is `sealed_recovery_resume`. No clock
is forced to2294. Both arms use seed290927, block0,4096lanes,600policy intervals,
2400 fresh-feedback physical steps and the saved learned Gaussian/std. The
unchanged55 loop draws the private Gaussian stream in both arms and applies
only the mean versus sampled action. It restores the same saved RNG and performs
the same one native reset. Controllers, rewards, commands and success gates
come from the frozen source. Formal training continues independently on Kaiser.

The pair is prepared, with both real original-driver CPU preflights completed
under an empty CUDA visibility mask. No App, native scene or evaluation has run.
Plan and admission are in `REC3076_pair_001/invocation.json` and `admission.json`;
both CPU report/identity files are in `CPU_mean/` and `CPU_stochastic/`.

From the repository root, root may execute the reviewed pair only after its
local resource check:

```sh
/home/yukikaze/isaacsim60-venv/bin/python -B \
  reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/run_sealed_REC3076_pair_021.py \
  run --output reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/REC3076_pair_001 \
  --max-arm-wall-seconds 600
```

The already-completed CPU command was:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/yukikaze/isaacsim60-venv/bin/python -B \
  reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/run_sealed_REC3076_pair_021.py \
  preflight --output reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/REC3076_pair_001
```

Do not repeat prepare/preflight or overwrite existing outputs. Each arm gets an
exclusive request marker before starting and a parent-observed process exit
receipt afterward. Stochastic starts only after the mean PID is absent and local
GPU0 has no compute owner. The script never SSHes or signals any process.

At the600s lifecycle limit it writes the exact child PID, argv and process
starttime, leaves that child intact, and stops before the next arm. Root may
reclaim only this exclusive local diagnostic child when all600steps, payment
reconciliation and restored callbacks are already authenticated preclose.
The timeout receipt explicitly records whether that condition passed. Preserve
raw output and the limit receipt. Root's independent exit evidence must include
the matching `pid`, `command`, `starttime_ticks`, `process_exit_observed=true`
and the actually observed integer `process_exit_code`; the script does not
invent an exit code. CPU-only collection after that explicit root action is:

```sh
/home/yukikaze/isaacsim60-venv/bin/python -B \
  reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/run_sealed_REC3076_pair_021.py \
  collect-reclaimed --arm mean --exit-evidence /absolute/path/root_exit.json \
  --output reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/REC3076_pair_001
```

Then the same run command with `--arm stochastic` can run the unstarted arm.
Neither exit0 nor a root reclamation establishes that App.close returned.
All receipts retain `app_close_returned=null`.

The final pair review requires exact native identities matching CPU admission,
initial physical birth/lesson/profile hashes, initial environment RNG, group
population, private sampling stream initial/final hashes, draw counts and std.
Subsequent environment RNG may diverge after policy-dependent resets. Each arm's
original first-birth outcomes, guard failures, unfinished/censored population,
eight sensor gates, actual motion release, followup and nonzero goal interval
counts remain in the raw report/teaching/payment files and compact readback.

The external classifier calls frozen `review_output` first. Its sole final
L0-forward/both-yaw exposure rejection is recorded separately from successful
native integrity using the same frozen `validate_terminal`; every identity,
count, payment, cleanup or other rejection remains a failure. A complete arm
with zero exposure is a valid negative mechanism result. It is not granted
strict engineering exposure admission or behavior qualification. Survival at
12s remains right censoring. The pair can distinguish the applied sampling
barrier at this fixed actor and initial state; it does not establish the causal
learning benefit of the new reward or an entropy/advantage explanation.

CPU fixtures passed10tests. They bind the actual downloaded REC3076 proof and
commands; exercise positive/zero exposure, incomplete599steps, payment residual,
failure sidecar, wrong qualification/hash, changed initial scene, permitted later
reset-RNG divergence and a foreign local compute owner. Synthetic metadata
fixtures seeded from the existing completed engineering002 report are not new
REC3076 native results. No production, frozen observer or contract was edited.
