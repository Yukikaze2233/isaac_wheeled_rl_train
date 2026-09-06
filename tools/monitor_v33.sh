#!/bin/bash
# Wait for the run-4/run-5 trainings, evaluate each best checkpoint, and
# promote the winner to the live-viewer policy path (runs_v33/v33_policy.onnx).
set -u
# Portable: ROOT is the robot_rl/ directory containing both repos; the venv is
# auto-detected (.venv for the minipc, .venv_mj314 for the laptop).
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [ -x "$ROOT/.venv/bin/python" ]; then
  VENV=$ROOT/.venv/bin/python
else
  VENV=$ROOT/.venv_mj314/bin/python
fi
export PYTHONPATH=$ROOT/.rl_deps:$ROOT/.rl_deps_rsl23

while [ $(pgrep -f train_mujoco | wc -l) -gt 0 ]; do sleep 60; done

RUNS="bc kv10 kv20 r5"
BEST=""
BEST_SCORE=-1000
for v in $RUNS; do
  if [ -f "$ROOT/runs_v33_$v/v33_policy.pt" ]; then
    $VENV $ROOT/isaac_wheeled_rl_train/tools/export_v33_onnx.py \
      "$ROOT/runs_v33_$v/v33_policy.pt" "$ROOT/runs_v33_$v/best.onnx" > /dev/null 2>&1
    $VENV $ROOT/eval_v33.py "$ROOT/runs_v33_$v/best.onnx" 0.3 8 > "$ROOT/runs_v33_$v/eval.txt" 2>&1
  fi
done
echo "=== training results ==="
for v in $RUNS; do
  echo "--- $v ---"; tail -2 "$ROOT/runs_v33_$v/train.log"; cat "$ROOT/runs_v33_$v/eval.txt" 2>/dev/null
  if [ -f "$ROOT/runs_v33_$v/eval.txt" ]; then
    S=$(grep survived "$ROOT/runs_v33_$v/eval.txt" | sed 's/survived=//' | cut -d's' -f1)
    Z=$(grep mean_base_z "$ROOT/runs_v33_$v/eval.txt" | sed 's/.*mean_base_z=//' | cut -d' ' -f1)
    SCORE=$(echo "$S $Z" | awk '{print $1 + ($2 - 0.32) * 20}')
    echo "score=$SCORE"
    if awk "BEGIN{exit !($SCORE > $BEST_SCORE)}"; then
      BEST_SCORE=$SCORE
      BEST=$ROOT/runs_v33_$v/best.onnx
    fi
  fi
done
if [ -n "$BEST" ]; then
  cp "$BEST" "$ROOT/runs_v33/v33_policy.onnx"
  echo "promoted: $BEST (score $BEST_SCORE) -> runs_v33/v33_policy.onnx"
fi
