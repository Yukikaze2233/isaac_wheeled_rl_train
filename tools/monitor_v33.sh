#!/bin/bash
# Wait for the two run-4 training variants, evaluate the best checkpoint of
# each, and promote the winner to the live-viewer policy path.
set -u
ROOT=/home/yukikaze/Documents/workspace/robot_rl
VENV=$ROOT/.venv_mj314/bin/python
export PYTHONPATH=$ROOT/.rl_deps:$ROOT/.rl_deps_rsl23

while [ $(pgrep -f train_mujoco | wc -l) -gt 0 ]; do sleep 60; done

for v in kv10 kv20; do
  if [ -f "$ROOT/runs_v33_$v/v33_policy.pt" ]; then
    $VENV $ROOT/isaac_wheeled_rl_train/tools/export_v33_onnx.py \
      "$ROOT/runs_v33_$v/v33_policy.pt" "$ROOT/runs_v33_$v/best.onnx" > /dev/null 2>&1
    $VENV $ROOT/eval_v33.py "$ROOT/runs_v33_$v/best.onnx" 0.3 8 > "$ROOT/runs_v33_$v/eval.txt" 2>&1
  fi
done
echo "=== run-4 results ==="
for v in kv10 kv20; do
  echo "--- $v ---"; tail -3 "$ROOT/runs_v33_$v/train.log"; cat "$ROOT/runs_v33_$v/eval.txt" 2>/dev/null
done
# promote the winner (prefer KV2.0 if both survive and kv20 base_z >= kv10)
BEST=$ROOT/runs_v33_kv10/best.onnx
if [ -f "$ROOT/runs_v33_kv20/eval.txt" ] && [ -f "$ROOT/runs_v33_kv10/eval.txt" ]; then
  Z10=$(grep mean_base_z "$ROOT/runs_v33_kv10/eval.txt" | sed 's/.*mean_base_z=//' | cut -d' ' -f1)
  Z20=$(grep mean_base_z "$ROOT/runs_v33_kv20/eval.txt" | sed 's/.*mean_base_z=//' | cut -d' ' -f1)
  S10=$(grep survived "$ROOT/runs_v33_kv10/eval.txt" | sed 's/survived=//' | cut -d's' -f1)
  S20=$(grep survived "$ROOT/runs_v33_kv20/eval.txt" | sed 's/survived=//' | cut -d's' -f1)
  # score = survival + height above crouch floor
  OK10=$(echo "$S10 $Z10" | awk '{print $1 + ($2 - 0.32) * 20}')
  OK20=$(echo "$S20 $Z20" | awk '{print $1 + ($2 - 0.32) * 20}')
  echo "scores: kv10=$OK10 kv20=$OK20"
  if awk "BEGIN{exit !($OK20 > $OK10)}"; then BEST=$ROOT/runs_v33_kv20/best.onnx; fi
fi
cp "$BEST" "$ROOT/runs_v33/v33_policy.onnx"
echo "promoted: $BEST -> runs_v33/v33_policy.onnx"
