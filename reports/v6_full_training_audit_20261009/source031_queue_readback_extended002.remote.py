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


launch = read('source031_queue_launch.receipt.json')
coordinator = identity(launch['pid'])
assert coordinator['starttime'] == launch['starttime'] and coordinator['command'] == launch['command']
assert coordinator['state'] not in ('Z', 'T', 't')
result['coordinator'] = coordinator
intent = read('source031_queue/queue.intent.json')
assert intent['source_commit'] == 'ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a'
assert intent['coordinator']['sha256'] == '91f059ca0f28c8f7636620ab8d801e95eca0b24e267475811862ce86f7a8f481'
assert intent['inputs']['stage']['sha256'] == 'b7dae15f3878aa92039580387bc98e526078fc4f617064a35c18da8994f93828'
assert intent['controller_wait_seconds'] == 604800 and intent['retries'] == 0
status = read('source031_queue/status.json')
assert status['status'] == 'waiting_for_normal_speed_boundary', status
started = read('source031_queue/controller.started.json')
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
entries = {'coordinate_source031_91f059ca.py': [], 'hold_completed_speed_boundary.py': []}
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
assert entries == {'coordinate_source031_91f059ca.py': [coordinator['pid']],
                   'hold_completed_speed_boundary.py': [controller['pid']]}, entries
result['unique_once_queue_processes'] = entries
products = ('source031_speed_boundary.json', 'source031_speed_boundary.receipt.json', 'source_revision_030.json',
            'source031_deploy/prepare.intent.json', 'source031_deploy/prepare.json',
            'source031_deploy/launch.intent.json', 'source031_deploy/launch.receipt.json')
result['future_boundary_products'] = {name: (root / name).exists() for name in products}
assert not any(result['future_boundary_products'].values())
lock = root / 'source031_queue.lock'
result['queue_lock_file_exists'] = lock.is_file()
result['queue_lock_inode'] = lock.stat().st_ino
result['actual_proc_lock_entries'] = [line for line in Path('/proc/locks').read_text().splitlines()
    if len(line.split()) > 5 and line.split()[4] == str(coordinator['pid'])
    and line.split()[5].rsplit(':', 1)[-1] == str(lock.stat().st_ino)]
assert result['actual_proc_lock_entries'], 'Coordinator lock not actually owned'
for name in ('source031_queue/controller.log', 'source031_queue.launch.log'):
    path = root / name
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 3000))
        result[name] = stream.read().decode(errors='replace')
cancel = read('source031_queue/cancel_source030.receipt.json')
assert cancel['status'] == 'waiting_source030_cancelled_without_training_signal'
assert cancel['training_signals'] == 0 and cancel['PPO_updates'] == 0 and cancel['refunds'] == 0
for item in (cancel['old_controller'], cancel['old_coordinator']):
    proc = Path('/proc') / str(item['pid'])
    if proc.exists():
        old_stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        assert int(old_stat[19]) != item['starttime'], 'Old waiting owner is still present'
old_status = read('source030_queue/status.json')
assert old_status['status'] == 'attention_no_retry' and old_status['phase'] == 'controller'
old_exit = read('source030_queue/controller.execution.json')
assert old_exit['exit_code'] == -15
result['original_speed_training_unchanged'] = True
extra_products = ('source_revision_031.json', 'source031_queue/prepare.json', 'source031_queue/launch.receipt.json', 'train/revisions/v26_completed_speed_to_v27')
result['source031_additional_future_products'] = {name: (root/name).exists() for name in extra_products}
assert not any(result['source031_additional_future_products'].values())
print(json.dumps(result))
