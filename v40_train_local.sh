#!/usr/bin/env bash
# V40 本机 headless 训练（默认 64 env × 100 迭代；参数: <envs> <iters>）
cd "$(dirname "$0")"
ENVS="${1:-64}"; ITERS="${2:-100}"
RUN="runs_v40/local-train-$(date +%m%d-%H%M%S)"
exec env OMNI_KIT_ACCEPT_EULA=YES TMPDIR="$HOME/.cache/kit-tmp" LD_LIBRARY_PATH="$HOME/.local/lib/compat" \
  /home/yukikaze/isaacsim51-venv/bin/python scripts/train_v40.py --research --headless --stage stand \
  --num-envs "$ENVS" --max-iterations "$ITERS" --max-runtime-seconds 28800 \
  --run-dir "$PWD/$RUN" --seed 40 --device cuda:0
