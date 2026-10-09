"""Read the actual detached once queue and unchanged original training owners."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
result = dict(at_utc=datetime.now(timezone.utc).isoformat(), status='read_only_once_queue_snapshot',
              remote_writes=0, PPO_updates_by_reader=0, train_signals_by_reader=0, qualification=None)


def read(relative):
    path = root / relative
    value = json.loads(path.read_text())
    result[relative] = value
    return value


def identity(pid):
    proc = Path('/proc') / str(pid)
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    command = [b.decode() for b in (proc / 'cmdline').read_bytes().split(b'\0') if b]
    fields = (proc / 'status').read_text().splitlines()
    return dict(pid=pid, ppid=int(stat[1]), starttime=int(stat[19]), state=stat[0], command=command,
                resources=[x for x in fields if x.startswith(('VmRSS:', 'Threads:'))])


launch = read('source030_queue_launch.receipt.json')
coordinator = identity(launch['pid'])
assert coordinator['starttime'] == launch['starttime'] and coordinator['command'] == launch['command']
assert coordinator['state'] not in ('Z', 'T', 't')
result['coordinator'] = coordinator
intent = read('source030_queue/queue.intent.json')
assert intent['source_commit'] == '81ad124554c4ab45cc274e2d482d46c76ba125b0'
assert intent['coordinator']['sha256'] == '0478bfe38c673e7fdb9ad1a5a6fb6151c76848e2a4182714b8910c8c357eb339'
assert intent['inputs']['stage']['sha256'] == '870194901f935c9c024996b0e05317f008fbf3f067039355ec14c5c4fbf42177'
assert intent['controller_wait_seconds'] == 604800 and intent['retries'] == 0
status = read('source030_queue/status.json')
assert status['status'] == 'waiting_for_normal_speed_boundary', status
started = read('source030_queue/controller.started.json')
controller = identity(started['pid'])
assert controller['command'] == started['command'] and controller['ppid'] == coordinator['pid']
assert controller['state'] not in ('Z', 'T', 't')
assert controller['command'][-2:] == ['--timeout', '604800']
result['controller'] = controller
result['original_owners'] = []
for pid, start in ((4141407, 113423130), (4141412, 113423416), (4141576, 113428247)):
    item = identity(pid)
    assert item['starttime'] == start and item['state'] not in ('Z', 'T', 't')
    result['original_owners'].append(item)
assert result['original_owners'][1]['ppid'] == result['original_owners'][0]['pid']
assert result['original_owners'][2]['ppid'] == result['original_owners'][1]['pid']
job = read('job.json')
assert job['pid'] == result['original_owners'][1]['pid'] and job['command'] == result['original_owners'][1]['command']
worker = result['original_owners'][2]['command']
directory = Path(worker[worker.index('--run-dir') + 1])
result['progress.json'] = json.loads((directory / 'progress.json').read_text())
entries = {'coordinate_source030_0478bfe3.py': [], 'hold_completed_speed_boundary.py': []}
for file in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv = [b.decode() for b in file.read_bytes().split(b'\0') if b]
        for value in argv[1:]:
            if value.endswith('.py'):
                name = Path(value).name
                if name in entries and str(root) in value:
                    entries[name].append(int(file.parent.name))
                break
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
assert entries == {'coordinate_source030_0478bfe3.py': [coordinator['pid']],
                   'hold_completed_speed_boundary.py': [controller['pid']]}, entries
result['unique_once_queue_processes'] = entries
products = ('source030_speed_boundary.json', 'source030_speed_boundary.receipt.json', 'source_revision_030.json',
            'source030_deploy/prepare.intent.json', 'source030_deploy/prepare.json',
            'source030_deploy/launch.intent.json', 'source030_deploy/launch.receipt.json')
result['future_boundary_products'] = {name: (root / name).exists() for name in products}
assert not any(result['future_boundary_products'].values())
lock = root / 'source030_queue.lock'
result['queue_lock_file_exists'] = lock.is_file()
result['queue_lock_inode'] = lock.stat().st_ino
result['actual_proc_lock_entries'] = [line for line in Path('/proc/locks').read_text().splitlines()
    if len(line.split()) > 5 and line.split()[4] == str(coordinator['pid'])
    and line.split()[5].rsplit(':', 1)[-1] == str(lock.stat().st_ino)]
assert result['actual_proc_lock_entries'], 'Coordinator lock not actually owned'
for name in ('source030_queue/controller.log', 'source030_queue.launch.log'):
    path = root / name
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 3000))
        result[name] = stream.read().decode(errors='replace')
print(json.dumps(result))
