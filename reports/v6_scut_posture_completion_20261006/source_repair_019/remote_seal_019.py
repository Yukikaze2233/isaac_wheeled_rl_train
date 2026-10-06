"""Seal only the authenticated predecessor; never launch or kill by name."""
import hashlib
import json
import os
from pathlib import Path
import signal
from datetime import datetime, timezone

run = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
old = run / 'source_4ba339299c20'
new = run / 'source_712d701473f3'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def command(pid):
    return (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
def parent(pid):
    return int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[1])
def starttime(pid):
    return int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[19])
def entry(cmd):
    if not cmd or not Path(cmd[0]).name.startswith('python'):
        return None
    return next((Path(arg) for arg in cmd[1:] if arg and not arg.startswith('-')), None)

assert not (run / 'seal_request_019.json').exists(), 'Seal was already requested; inspect'
assert not (run / 'source_repair_launch_019.json').exists(), 'Repair was already launched'
assert sha(run / 'source_712d701473f3.tar.gz') == 'fb5172d1f2a44aaf556f52e5de389cd59f5a3ef7bed79919b68d4d7c6580d247'
assert sha(old / 'contracts/v6_task_campaign_v18.json') == sha(new / 'contracts/v6_task_campaign_v18.json') == '027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022'
job = json.loads((run / 'job.json').read_text())
pid = job['pid']
assert pid == 3471702 and command(pid) == job['command']
assert entry(job['command']) == old / 'scripts/run_task_campaign.py'
assert job['command'][job['command'].index('--run-dir') + 1] == str(run / 'train')
owned = []
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        cmd = command(int(directory.name))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
    main = entry(cmd)
    if main is None or main.name not in ('chassis_remote_job.py', 'run_task_campaign.py', 'train_chassis.py', 'evaluate_chassis.py'):
        continue
    if not any(str(run / 'train') in arg or arg == str(run) for arg in cmd):
        continue
    assert main.parent == old / 'scripts', 'Unexpected source owner'
    owned.append({'pid': int(directory.name), 'ppid': parent(int(directory.name)), 'starttime': starttime(int(directory.name)), 'entry': str(main), 'command': cmd})
assert len(owned) == 3, owned
by_name = {Path(item['entry']).name: item for item in owned}
assert len(by_name) == 3 and set(by_name) == {'chassis_remote_job.py', 'run_task_campaign.py', 'train_chassis.py'}
wrapper, supervisor, worker = (by_name[name] for name in ('chassis_remote_job.py', 'run_task_campaign.py', 'train_chassis.py'))
assert wrapper['pid'] == 3471693 and supervisor['pid'] == pid and worker['pid'] == 3471715
assert supervisor['ppid'] == wrapper['pid'] and worker['ppid'] == pid
assert worker['starttime'] == 88205735
assert worker['command'][worker['command'].index('--run-dir') + 1] == str(run / 'train/recovery/attempt_002/train')
receipt = dict(schema='authenticated-seal-request-019-v1', requested_at_utc=datetime.now(timezone.utc).isoformat(),
    previous_source_commit='4ba339299c203cf2aba77183ec81f2f017dc9bf9', next_source_commit='712d701473f38ef6797725f8efdda9e6beb85e87',
    job_sha256=sha(run / 'job.json'), signal='SIGTERM', signal_target_pid=pid, owners=owned,
    progress_before_request=json.loads((run / 'train/recovery/attempt_002/train/progress.json').read_text()),
    status='requesting_native_full_state_seal_not_PPO_launch', qualification=None)
with (run / 'seal_request_019.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
# PID starttime and argv are checked again immediately before the sole signal.
assert starttime(pid) == supervisor['starttime'] and command(pid) == job['command']
os.kill(pid, signal.SIGTERM)
print(json.dumps(receipt))
