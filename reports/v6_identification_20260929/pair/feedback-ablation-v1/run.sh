#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
experiment=/home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929
ablation="$experiment/feedback-ablation-v1"
python_bin=/home/kaiser/robot-rl-sim60/env/bin/python
"$python_bin" "$ablation/scripts/ablate_pair_feedback_sampling.py" \
  --experiment-root "$experiment" \
  --replay-script "$ablation/scripts/identify_pair_dynamics.py" \
  --bundle /home/kaiser/robot-rl-sim60/experiments/v6-capacity-check-20260929/frozen/model/纯底盘_v6_232mm/urdf \
  --output "$ablation/results"
