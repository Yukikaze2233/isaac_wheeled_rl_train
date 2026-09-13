#!/usr/bin/env bash
# Engineering smoke only: 2 envs, 3 PPO updates, fresh run, bounded tmux.
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
    printf 'Usage: bash %s /path/to/sim-runtime.sh /viewer-env/bin/python /existing/run-root [browser-wait-seconds]\n' "$0" >&2
    exit 2
fi
runtime="$1"
viewer="$2"
root="$3"
wait="${4:-30}"
[[ -f "$runtime" && -x "$viewer" && -d "$root" ]] || { printf 'Runtime, viewer Python, or run root missing\n' >&2; exit 2; }
[[ "$wait" =~ ^[0-9]+$ ]] && (( wait <= 120 )) || { printf 'Browser wait must be 0..120 seconds\n' >&2; exit 2; }
name="v40-live-$(date +%Y%m%dT%H%M%S)-$$"
run="$root/$name"
script="$(realpath "$(dirname "$0")/run_smoke.sh")"
printf -v command 'env V40_SIM_RUNTIME=%q V40_VIEWER_PYTHON=%q bash %q %q %q' "$runtime" "$viewer" "$script" "$run" "$wait"
tmux new-session -d -s "$name" "$command"
printf 'Engineering smoke: 2 envs, 3 PPO updates; not Round3 or hardware acceptance.\n'
printf 'Browser: http://127.0.0.1:8088 (open before bounded browser wait expires)\n'
printf 'tmux: %s\nRun: %s\n' "$name" "$run"
