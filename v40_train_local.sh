#!/usr/bin/env bash
# Local Sim 6 training: [--gui] [envs] [iterations].
set -euo pipefail
cd "$(dirname "$0")"
POSITIONAL=()
VIEW_ARGS=(--headless)
MODE=headless
for ARG in "$@"; do
  case "$ARG" in
    --gui) VIEW_ARGS=(); MODE=gui ;;
    --help|-h)
      printf '%s\n' 'Usage: v40_train_local.sh [--gui] [envs=64] [iterations=100]' \
        'Examples: v40_train_local.sh 256 100; v40_train_local.sh --gui 16 100'
      exit 0 ;;
    --*) printf 'Unknown option: %s\n' "$ARG" >&2; exit 2 ;;
    *) POSITIONAL+=("$ARG") ;;
  esac
done
if (( ${#POSITIONAL[@]} > 2 )); then
  printf '%s\n' 'Expected at most two positional arguments: envs iterations' >&2
  exit 2
fi
ENVS="${POSITIONAL[0]:-64}"; ITERS="${POSITIONAL[1]:-100}"
if [[ ! "$ENVS" =~ ^[1-9][0-9]*$ || ! "$ITERS" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'envs and iterations must be positive integers' >&2
  exit 2
fi
RUN="runs_v40/local-${MODE}-train-$(date +%m%d-%H%M%S-%N)"
exec env OMNI_KIT_ACCEPT_EULA=YES TMPDIR="$HOME/.cache/kit-tmp" LD_LIBRARY_PATH="$HOME/.local/lib/compat" \
  "${ISAACSIM_PYTHON:-/home/yukikaze/isaacsim60-venv/bin/python}" scripts/train_v40.py --research "${VIEW_ARGS[@]}" \
  --contract contracts/own_v40_v2.json --stage locomotion \
  --num-envs "$ENVS" --max-iterations "$ITERS" --max-runtime-seconds 28800 \
  --run-dir "$PWD/$RUN" --seed 40 --device cuda:0
