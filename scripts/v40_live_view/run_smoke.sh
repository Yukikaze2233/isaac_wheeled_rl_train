#!/usr/bin/env bash
set -uo pipefail
run="$1"
wait="${2:-30}"
repo="$(realpath "$(dirname "$0")/../..")"
mkdir "$run" || exit 2
source "${V40_SIM_RUNTIME:?Set the existing Sim runtime script}" || exit 2
date -Is > "$run/started"
printf '%s\n' "$$" > "$run/wrapper.pid"
timeout --signal=TERM --kill-after=20s 300s \
    python "$repo/scripts/train_v40.py" \
    --contract "$repo/contracts/own_v40_v1.json" --research --headless \
    --num-envs 2 --max-iterations 3 --max-runtime-seconds 90 \
    --usd-cache-dir "$run/usd-cache" --run-dir "$run/train" \
    --live-view --live-view-python "${V40_VIEWER_PYTHON:?Set the isolated viewer Python}" \
    --live-view-wait-seconds "$wait" --live-view-seconds 180 > "$run/train.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run/exit"
date -Is > "$run/finished"
exit "$code"
