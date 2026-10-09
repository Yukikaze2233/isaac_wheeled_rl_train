"""Read native periodic seals and actual worker identity; never manufacture one."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
directory = root / 'train/speed/attempt_003/train'
progress = json.loads((directory / 'progress.json').read_text())
source = root / 'source_serial_dispatch029_20261009'
contract = json.loads((source / 'contracts/v6_task_campaign_speed_worker_v26.json').read_text())
assert contract['checkpoint_interval'] == 500
assert contract['task_campaign']['evaluation_interval'] == 0
assert progress['parent_updates'] == 41393
identity = (Path('/proc/4141576/stat')).read_text().rsplit(')', 1)[1].split()
assert int(identity[19]) == 113428247 and identity[0] not in ('Z', 'X')
argv = Path('/proc/4141576/cmdline').read_bytes().split(b'\0')
assert str(source / 'scripts/train_chassis.py').encode() in argv
seals = []
for item in sorted((directory / 'checkpoints').glob('update_*')):
    if item.name.endswith('.pending'):
        continue
    data = json.loads((item / 'progress.json').read_text())
    seals.append(dict(directory=str(item), task_actor_update=data['task_actor_update'],
        completed_consumption=data['parent_updates'] + data['successful_updates']))
candidate = directory / 'checkpoints/update_00042000'
candidate_sha = None
if candidate.is_dir():
    data = json.loads((candidate / 'progress.json').read_text())
    assert data['task_actor_update'] == 597
    assert data['parent_updates'] + data['successful_updates'] == 42000
    with (candidate / 'model.pt').open('rb') as stream:
        candidate_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
print(json.dumps(dict(status='read_only_native_checkpoint_readiness',
    at_utc=datetime.now(timezone.utc).isoformat(), worker_pid=4141576,
    worker_starttime=113428247, current_task_actor_update=progress['task_actor_update'],
    current_completed_consumption=progress['parent_updates']+progress['successful_updates'],
    periodic_consumption_interval=500, task_assessment_interval=0, seals=seals,
    next_requested_candidate=dict(directory=str(candidate), consumed_updates=42000,
        task_actor_update=597, model_sha256=candidate_sha, present=candidate_sha is not None),
    scope='Metadata readiness only; native full-seal/learning-state validation must precede replay.',
    qualification=None, remote_writes=0, PPO_updates_by_reader=0)))
