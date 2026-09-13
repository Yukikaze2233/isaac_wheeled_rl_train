#!/usr/bin/env bash
set -euo pipefail
if (( $# != 3 )); then
    printf 'Usage: bash %s /path/to/uv /path/to/python3.12 /new/viewer-env\n' "$0" >&2
    exit 2
fi
uv="$1"
python="$2"
destination="$3"
if [[ -e "$destination" ]]; then
    printf 'Refusing to replace an existing viewer environment: %s\n' "$destination" >&2
    exit 2
fi
"$uv" venv --python "$python" "$destination"
"$uv" pip install --python "$destination/bin/python" -r "$(dirname "$0")/requirements.txt"
"$uv" pip check --python "$destination/bin/python"
