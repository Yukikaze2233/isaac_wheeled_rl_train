#!/usr/bin/env bash
set -euo pipefail
cd /home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python_bin=/home/kaiser/robot-rl-sim60/env/bin/python
for side in left right; do
  if [[ "$side" == left ]]; then run=run_20260928T073044Z; else run=run_20260928T144606Z; fi
  "$python_bin" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); (p/"parameters.json").write_text(json.dumps(json.loads((p/"fit.json").read_text())["parameters"],indent=2)+"\n")' "$side-fit"
  "$python_bin" scripts/plot_pair_identification.py --baseline "$side-baseline" --candidate "$side-fit" --side "$side" --segment 79 --output "$side-comparison.png"
  "$python_bin" scripts/identify_pair_dynamics.py "$run" --bundle asset-snapshot-01 --side "$side" --coordinate-sign -1 --segments 79 89 --physics-dt .001 --params "$side-fit/parameters.json" --output "$side-fit-1ms"
  "$python_bin" scripts/identify_pair_dynamics.py "$run" --bundle asset-snapshot-01 --side "$side" --coordinate-sign -1 --segments 79 89 --physics-dt .00025 --params "$side-fit/parameters.json" --output "$side-fit-quarter-ms"
  "$python_bin" scripts/identify_pair_dynamics.py "$run" --bundle asset-snapshot-01 --side "$side" --coordinate-sign -1 --segments 37 39 41 43 79 89 --mode torque_input --max-samples 1000 --params "$side-fit/parameters.json" --output "$side-fit-torque"
done
