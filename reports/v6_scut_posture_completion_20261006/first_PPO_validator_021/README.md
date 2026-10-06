# Source021 first-PPO read-only verifier

Prepared against frozen source `d0dce6181df83b03fe166abcbadef3c81de2e57e`, v19 recovery-only contracts. This artifact emits a CPU inspector; it does not connect, launch, edit a contract, run GPU work or take an optimizer step. Preparation passed 73 CPU fixtures in 1.02s. Synthetic fixtures are not native training evidence.

The inspector derives the predecessor and consumption origins from the prepared receipts. It authenticates the latest stopped recovery full seal, full actor/critic/Adam/std restoration, other five domain records, complete charged ledger, parent tree, all old contract bytes and actual runtime source snapshots. A v18→v19 migration inherits the original 10 critic-only updates; same-v19 restoration uses the saved remaining count.

Native checkpoint500 only seals initial state, first update and first actor update. Accordingly this verifier observes exact initial/+1/+11 tensor states, verifies ten charged critic iterations through CRC event timing and held curriculum clocks, and explicitly reports intermediate actor tensors as unobserved. There is no TensorBoard optimizer-phase tag in this source. The optional independent native progress index supplies direct phase/clock rows1..11, byte-authenticated by SHA. Missing files/rows return `pending`; incompatible evidence returns `failed`. Neither result authorizes restart, refund or qualification.

Prepare emission after root has saved the actual admission/predecessor/revision:

```bash
/home/yukikaze/isaacsim60-venv/bin/python -B reports/v6_scut_posture_completion_20261006/first_PPO_validator_021/verify_first_PPO_021.py \
  --emit-remote-code \
  --source reports/v6_scut_posture_completion_20261006/source_repair_021/source_candidate_021.json \
  --admission reports/v6_scut_posture_completion_20261006/source_repair_021/admission_021.json \
  --predecessor reports/v6_scut_posture_completion_20261006/source_repair_021/predecessor_021.json \
  --revision reports/v6_scut_posture_completion_20261006/source_repair_021/source_revision_021.json \
  --phase-progress-capture /ACTUAL_RUNROOT/first_progress_021/index.json
```

Use the actual runroot from the prepared receipt, not the placeholder. The emitted code is intended for root's authorized read-only CPU inspection on the executing host. The parser requires the actual native `global_budget_actor_updates` minus the authenticated argv offset; it does not invent `actor_updates_consumed` in `progress.json`.

CLI supports `--mode same_v19` and `--event-prefix-mib 1..128` (default64). It reads a bounded first-event prefix rather than copying the multi-GB event file. A moving ledger/claim mismatch requires fresh coherent reads, not a training restart.

Frozen artifact hashes and boundaries are in `cpu_readiness_021.json`; helper and native-teaching independent CPU reviews are separate receipts. No actual v19 first-PPO result is claimed here.
