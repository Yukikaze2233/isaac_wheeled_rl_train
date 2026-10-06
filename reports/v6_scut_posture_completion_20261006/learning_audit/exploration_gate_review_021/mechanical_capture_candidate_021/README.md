# REC3076 mechanical capture candidate

This artifact composes the unchanged source021 native mean observer. It neither edits the source archive nor creates PPO/optimizer/worker/campaign contracts. Native execution requires a separate root authorization and exclusive local GPU0; Kaiser training continues.

The original 4096-lane, 600-policy-tick loop executes unchanged (50 Hz policy, 200 Hz PhysX, 64/32 solver, 3 mm guard). It uses the same immutable REC3076 checkpoint, full seal, source, native birth RNG and private action RNG. The dense stream covers the first 260 physical steps (1.3 s) of original birth lanes 1708 and 2216, and stops each lane at its first native terminal. A separate stream records each new global all-lane/all-pin maximum throughout 2400 steps, including the actual lane, pin, birth and physical tick. It does not assign the old unlocalized 730 mm peak to a selected lane.

`robot.set_joint_effort_target_index` supplies the exact original command before each physics interval. Original `update_robot_data` is called exactly once after the original PhysX step, then the wrapper reads actual body/joint poses and every native closed-chain endpoint, before automatic reset. Only within-policy calls with an effort token are counted; positive-dt reset refreshes are excluded explicitly. Phase and control reference belong to the interval just applied; q/dq/gaps/applied effort are postphysics data. The native requested-effort field at setter time is separately labeled script/mixed preclip override; it is not applied shaft torque. A separately persisted pure-script preclip request is unavailable. Full setter/readback efforts include the prismatic gas joints and are labeled N for those joints, N.m for rotary joints.

Current terrain contact patches come directly from the original filtered PhysX views, never stale `env.contact_force`. Sensor order and actual filter-path membership are checked against the original environment mapping. Only the two lanes' slices are copied to CPU. Normal slices are copied before the friction API reuses indices and point buffers. Normal aggregate versus original contact matrix differences are recorded, not silently asserted zero. Impulse is explicitly force times 0.005 s, not an independently measured field. Invalid slices/API failures reject capture. Counts reaching capacity flag incomplete evidence; no native overflow flag or complete self-contact pair identity has been established.

CPU fixtures use toy tensors and callbacks; they prove capture ordering, alias isolation, token/reset accounting, restoration and failure behavior, not native physics. The CPU mode additionally invokes the original immutable REC3076 admission preflight without constructing an App. Native result admission separately requires 600/2400 completion, original identity/initial births/std/private action RNG, exact first-guard signatures, preclose cleanup and no contact-capacity indicator. A native reject preserves the raw trace and does not qualify the mechanics or solver.

Run from the repository with the recorded manifest SHA and a new output directory under this candidate directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
/home/yukikaze/isaacsim60-venv/bin/python -B \
reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/mechanical_capture_candidate_021/run_capture_021.py \
--mode cpu --candidate-sha MANIFEST_SHA --output CANDIDATE_ABSOLUTE_PATH/CPU_admission_001
```

After independent review and root authorization only, substitute `--mode native`, retain GPU0 visibility and use a new `native_capture_001` output. The driver checks GPU0 has no compute owner before App startup. Root owns process timeout, complete App shutdown verification, and any later run; no second run is started by this artifact.
