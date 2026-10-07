#!/usr/bin/env bash
set -euo pipefail
SUITE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE_PYTHON="${CHASSIS_SUITE_PYTHON:-/home/yukikaze/isaacsim60-venv/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="/home/yukikaze/.local/lib/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="${TMPDIR:-/home/yukikaze/.cache/kit-tmp}"
mkdir -p -- "$TMPDIR"
scenario="${1:-flat}"
if [[ "$scenario" == "--menu" ]]; then
    mapfile -t choices < <("$SUITE_PYTHON" "$SUITE_ROOT/scripts/play_chassis_suite.py" --list-scenarios)
    if [[ -n "${DISPLAY:-}" ]] && command -v zenity >/dev/null; then
        rows=()
        for choice in "${choices[@]}"; do
            rows+=("${choice%% *}" "${choice#* }")
        done
        scenario="$(zenity --list --title='REC ONNX replay scenarios' --width=820 --height=600 \
            --text='Same REC candidate in every scene; unscored. Close the current GUI before changing scenes.' \
            --column='Scenario' --column='Description' --print-column=1 "${rows[@]}")" || exit 0
    else
        PS3="Choose a scenario (same REC ONNX; close the current GUI before changing): "
        select choice in "${choices[@]}"; do
            [[ -n "${choice:-}" ]] || continue
            scenario="${choice%% *}"
            break
        done
    fi
    shift
elif [[ $# -gt 0 ]]; then
    shift
fi
cd -- "$SUITE_ROOT"
exec "$SUITE_PYTHON" scripts/play_chassis_suite.py --scenario "$scenario" "$@"
