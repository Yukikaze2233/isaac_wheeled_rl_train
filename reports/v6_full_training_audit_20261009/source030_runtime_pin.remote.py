"""Read the existing native runtime used by the future one-time boundary queue."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat

path = Path('/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh')
mode = path.lstat().st_mode
assert stat.S_ISREG(mode)
data = path.read_bytes()
print(json.dumps(dict(
    status='native_runtime_pin_read_only',
    at_utc=datetime.now(timezone.utc).isoformat(),
    runtime_file=str(path),
    runtime_sha256=hashlib.sha256(data).hexdigest(),
    runtime_mode=stat.S_IMODE(mode),
    runtime_bytes=len(data),
    runtime_source=data.decode(),
    remote_writes=0,
    PPO_updates_by_reader=0,
    qualification=None,
)))
