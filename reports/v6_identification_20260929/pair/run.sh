#!/usr/bin/env bash
set -euo pipefail
cd /home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python_bin=/home/kaiser/robot-rl-sim60/env/bin/python
"$python_bin" scripts/identify_pair_dynamics.py run_20260928T073044Z --bundle asset-snapshot-01 --side left --coordinate-sign -1 --segments 37 39 41 43 79 89 --output left-baseline
"$python_bin" scripts/identify_pair_dynamics.py run_20260928T073044Z --bundle asset-snapshot-01 --side left --coordinate-sign -1 --fit-segments 9 11 13 15 --segments 37 39 41 43 79 89 --max-evaluations 25 --output left-fit
"$python_bin" scripts/identify_pair_dynamics.py run_20260928T144606Z --bundle asset-snapshot-01 --side right --coordinate-sign -1 --segments 37 39 41 43 79 89 --output right-baseline
"$python_bin" scripts/identify_pair_dynamics.py run_20260928T144606Z --bundle asset-snapshot-01 --side right --coordinate-sign -1 --fit-segments 9 11 13 15 --segments 37 39 41 43 79 89 --max-evaluations 25 --output right-fit
