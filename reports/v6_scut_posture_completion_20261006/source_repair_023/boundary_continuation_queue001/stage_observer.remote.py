"""Stage only the frozen external observer; never launches or signals."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


payload = json.load(sys.stdin)
files = payload['files']
if set(files) != {'queue_task_campaign_boundary_continuation.py', 'queue_task_campaign_continuation.py',
                  'manifest.json', 'code_manifest.json'}:
    raise ValueError('Unexpected observer staging inventory')
content = {name: base64.b64decode(row['base64'], validate=True) for name, row in files.items()}
for name, data in content.items():
    if hashlib.sha256(data).hexdigest() != files[name]['sha256']:
        raise ValueError('Staging payload digest differs: ' + name)
manifest = json.loads(content['manifest.json'])
root = Path(manifest['run_root'])
if root != Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358'):
    raise ValueError('Unexpected external observer run root')
directory = root / 'boundary_observer_024'
job = json.loads((root / 'job.json').read_text())
if digest(root / 'job.json') != manifest['old_job_sha256']:
    raise ValueError('Predecessor job changed before observer staging')
if digest(Path(manifest['old_source_revision_receipt'])) != manifest['old_source_revision_sha256']:
    raise ValueError('Predecessor source receipt changed')
for pin in manifest['old_owners'].values():
    stat = (Path('/proc') / str(pin['pid']) / 'stat').read_text().rpartition(')')[2].split()
    if stat[0] == 'Z' or int(stat[19]) != pin['starttime']:
        raise ValueError('Pinned predecessor owner changed')
if job['pid'] != manifest['old_owners']['supervisor']['pid']:
    raise ValueError('Job no longer belongs to pinned supervisor')
for name in ('source_revision_024.json', 'boundary_continuation_024.receipt.json',
             '.task_campaign_boundary_v22.signal.intent.json', '.task_campaign_boundary_v22.launch.intent.json'):
    if (root / name).exists():
        raise ValueError('Boundary action artifact already exists: ' + name)
directory.mkdir(exist_ok=False)
for name, data in content.items():
    path = directory / name
    with path.open('xb') as stream:
        stream.write(data)
    if digest(path) != files[name]['sha256']:
        raise ValueError('Written observer digest differs')
receipt = dict(schema='external-boundary-observer-staging-v1', status='staged_not_running',
               directory=str(directory), predecessor_job_sha256=manifest['old_job_sha256'],
               files={name: {'sha256': row['sha256'], 'size': len(content[name])} for name, row in files.items()},
               staged_at_utc=datetime.now(timezone.utc).isoformat(), PPO_updates=0, signals=0,
               production_contract_changed=False, qualification=None)
(directory / 'staging_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
