"""Read a completed periodic checkpoint without touching the training worker."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
source = root / 'source_serial_dispatch029_20261009'
directory = root / 'train/speed/attempt_003/train/checkpoints/update_00043500'
stat = Path('/proc/4141576/stat').read_text().rsplit(')', 1)[1].split()
assert int(stat[19]) == 113428247 and stat[0] not in ('Z', 'X')
assert str(source / 'scripts/train_chassis.py').encode() in Path('/proc/4141576/cmdline').read_bytes().split(b'\0')
files = {}
for name in ('model.pt', 'model.contract.json', 'model.checkpoint.json', 'contract.json',
             'progress.json', 'completion.json', 'agent_config.json'):
    path = directory.parent.parent / name if name == 'agent_config.json' else directory / name
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    def identity(value):
        return dict(device=value.st_dev, inode=value.st_ino, size=value.st_size,
                    mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns, mode=value.st_mode)
    assert identity(before) == identity(after)
    files[name] = dict(remote_path=str(path), sha256=digest, size=after.st_size,
                      stat_before_hash=identity(before), stat_after_hash=identity(after))
progress = json.loads((directory / 'progress.json').read_text())
completion = json.loads((directory / 'completion.json').read_text())
seal = json.loads((directory / 'model.checkpoint.json').read_text())
assert completion['status'] == 'checkpoint_sealed'
assert completion['successful_updates'] == progress['parent_updates'] + progress['successful_updates'] == 43500
assert seal['checkpoint_sha256'] == files['model.pt']['sha256']
assert seal['learning_updates'] == 43500
print(json.dumps(dict(schema='periodic-checkpoint-read-only-v1', at_utc=datetime.now(timezone.utc).isoformat(),
    directory=str(directory), files=files, progress=progress, completion=completion, seal=seal,
    worker_pid=4141576, worker_starttime=113428247, remote_writes=0, PPO_updates_by_reader=0,
    qualification=None), allow_nan=False))
