#!/usr/bin/env bash
# Run within a dedicated tmux. Windows subprocesses also need their own timeout.
set -uo pipefail
directory="$1"
limit="$2"
shift 2
mkdir -p "$directory"
date -Is > "$directory/started"
printf '%s\n' "$$" > "$directory/controller.pid"
finish() {
    code=$?
    printf '%s\n' "$code" > "$directory/exit"
    date -Is > "$directory/finished"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timeout --signal=TERM --kill-after=30s "$limit" "$@" > "$directory/output.log" 2>&1
code=$?
exit "$code"
