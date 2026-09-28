#!/usr/bin/env bash
set -euo pipefail
cd /home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python_bin=/home/kaiser/robot-rl-sim60/env/bin/python
for side in left right; do
  if [[ "$side" == left ]]; then run=run_20260928T073044Z; else run=run_20260928T144606Z; fi
  "$python_bin" scripts/identify_pair_dynamics.py "$run" --bundle asset-snapshot-01 --side "$side" --coordinate-sign -1 --fit-segments 9 11 13 15 --fit-torque-gain --segments 37 39 41 43 79 89 --max-evaluations 25 --output "$side-gain-ablation"
  "$python_bin" scripts/identify_pair_dynamics.py "$run" --bundle asset-snapshot-01 --side "$side" --coordinate-sign -1 --segments 37 39 41 43 79 89 --mode torque_input --max-samples 1000 --params "$side-gain-ablation/parameters.json" --output "$side-gain-ablation-torque"
done
