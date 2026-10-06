"""Seal only the authenticated predecessor; never launch or kill by name."""
import hashlib
import json
import os
from pathlib import Path
import signal
from datetime import datetime, timezone

run = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
old = run / 'source_712d701473f3'
new = run / 'source_7f315f30741b'
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

assert not (run / 'seal_request_020.json').exists(), 'Seal was already requested; inspect'
assert not (run / 'source_repair_launch_020.json').exists(), 'Repair was already launched'
assert sha(run / 'source_7f315f30741b.tar.gz') == 'ff0c57ca2255fb0e336b28be57e0a66e7923597dd5398d0bb18c4e34f91c1b87'
assert sha(old / 'contracts/v6_task_campaign_v18.json') == sha(new / 'contracts/v6_task_campaign_v18.json') == '027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022'
job = json.loads((run / 'job.json').read_text())
pid = job['pid']
assert pid == 3480207 and starttime(pid)==88555229 and command(pid) == job['command']
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
assert wrapper['pid'] == 3480202 and supervisor['pid'] == pid and worker['pid'] == 3480218
assert supervisor['ppid'] == wrapper['pid'] and worker['ppid'] == pid
assert wrapper['starttime']==88555049 and supervisor['starttime']==88555229
assert worker['starttime'] == 88555666
assert worker['command'][worker['command'].index('--run-dir') + 1] == str(run / 'train/recovery/attempt_003/train')
receipt = dict(schema='authenticated-seal-request-020-v1', requested_at_utc=datetime.now(timezone.utc).isoformat(),
    previous_source_commit='712d701473f38ef6797725f8efdda9e6beb85e87', next_source_commit='7f315f30741be01d70046a11848443038c14b240',
    job_sha256=sha(run / 'job.json'), signal='SIGTERM', signal_target_pid=pid, owners=owned,
    progress_before_request=json.loads((run / 'train/recovery/attempt_003/train/progress.json').read_text()),
    status='requesting_native_full_state_seal_not_PPO_launch', qualification=None)
with (run / 'seal_request_020.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
# PID starttime and argv are checked again immediately before the sole signal.
assert starttime(pid) == supervisor['starttime'] and command(pid) == job['command']
os.kill(pid, signal.SIGTERM)
print(json.dumps(receipt))
