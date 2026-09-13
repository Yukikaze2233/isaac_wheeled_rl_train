#!/usr/bin/env bash
set -euo pipefail
root="${1:-/mnt/d/isaac60-native}"
python3 - "$root" <<'PY'
import json
from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
receipt = root / 'downloads/isaac-sim-standalone-6.0.0-windows-x86_64.zip.json'
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    try:
        record = json.loads(receipt.read_text())
        if record.get('status') == 'verified' and record.get('md5') == '4b49a4258792f09300ece31be1b6cfd9':
            break
    except (OSError, ValueError):
        pass
    time.sleep(5)
else:
    raise TimeoutError('verified download not ready within extraction startup bound')
PY
exec /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
    -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File 'D:\isaac60-native\tools\extract_sim.ps1' -TimeoutSeconds 1800
