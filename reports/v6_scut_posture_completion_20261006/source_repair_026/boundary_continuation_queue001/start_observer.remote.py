"""Start one waiting CPU observer after actual check-only admission."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
DIRECTORY = ROOT / 'boundary_observer_026'
SOCKET = Path('/home/kaiser/robot-rl-sim60/tmux.sock')
SESSION = 'joint-boundary-observer-026'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


staging = json.loads((DIRECTORY / 'staging_receipt.json').read_text())
for name, row in staging['files'].items():
    if digest(DIRECTORY / name) != row['sha256']:
        raise ValueError('Observer staging bytes changed: ' + name)
sys.path.insert(0, str(DIRECTORY))
from queue_task_campaign_boundary_continuation import BoundaryContinuationQueue, process_identity
manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
queue = BoundaryContinuationQueue(manifest)
owners = queue.owners()
queue.authenticate_owners(queue.previous_job(), owners)
status = json.loads(queue.status_path.read_text())
if status != dict(schema=manifest['schema'], manifest_sha256=queue.config_sha256,
                  status='waiting', reason='rec_normal_endpoint_not_complete'):
    raise ValueError('Actual check-only did not admit this waiting REC predecessor')
if any(path.exists() for path in (queue.state_path, queue.signal_intent, queue.intent,
                                 queue.queue_receipt, queue.source_revision_receipt)):
    raise ValueError('Observer state or boundary action already exists; never duplicate')
script = DIRECTORY / 'queue_task_campaign_boundary_continuation.py'
for path in Path('/proc').glob('[0-9]*/cmdline'):
    identity = process_identity(int(path.parent.name))
    if identity and identity['entry'] == str(script):
        raise ValueError('External observer is already active')
check = subprocess.run(['tmux', '-S', str(SOCKET), 'has-session', '-t', '=' + SESSION], capture_output=True)
if check.returncode == 0:
    raise ValueError('Named observer tmux session exists')
command = [manifest['command'][0], '-B', str(script), '--manifest', str(DIRECTORY / 'manifest.json'),
           '--poll-seconds', '20']
log = DIRECTORY / 'observer.log'
shell = 'exec ' + shlex.join(command) + ' >>' + shlex.quote(str(log)) + ' 2>&1'
launched = subprocess.run(['tmux', '-S', str(SOCKET), 'new-session', '-d', '-s', SESSION, shell],
                          text=True, capture_output=True, check=True)
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    if queue.state_path.exists():
        break
    if queue.status_path.exists() and json.loads(queue.status_path.read_text()).get('status') == 'attention':
        raise ValueError('Observer reported attention: ' + queue.status_path.read_text())
    time.sleep(.25)
if not queue.state_path.exists():
    raise ValueError('Observer first admission not yet recorded; do not launch again')
state = json.loads(queue.state_path.read_text())
status = json.loads(queue.status_path.read_text())
if state.get('staged') is not True or state.get('manifest_sha256') != queue.config_sha256:
    raise ValueError('Observer state does not prove first staged/native admission')
if status.get('status') != 'waiting' or status.get('reason') != 'rec_normal_endpoint_not_complete':
    raise ValueError('Observer is not waiting for normal REC endpoint')
matches = []
for path in Path('/proc').glob('[0-9]*/cmdline'):
    identity = process_identity(int(path.parent.name))
    if identity and identity['entry'] == str(script):
        matches.append(identity)
if len(matches) != 1 or matches[0]['command'] != command:
    raise ValueError('Cannot authenticate a single literal observer process')
queue.authenticate_owners(queue.previous_job(), queue.owners())
if any(path.exists() for path in (queue.signal_intent, queue.intent, queue.queue_receipt, queue.source_revision_receipt)):
    raise ValueError('Waiting observer unexpectedly produced a boundary action')
receipt = dict(schema='actual-external-boundary-observer-launch-v1', status='running_waiting_REC_normal_endpoint',
               tmux_session=SESSION, tmux_socket=str(SOCKET), observer=matches[0],
               observer_code_commit=json.loads((DIRECTORY / 'code_manifest.json').read_text())['observer_code_commit'],
               manifest_sha256=queue.config_sha256, status_readback=status, state_readback=state,
               admitted_at_utc=datetime.now(timezone.utc).isoformat(), production_source_changed=False,
               duplicate_PPO_launched=False, PPO_updates=0, signals=0, qualification=None)
with (DIRECTORY / 'launch_receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(receipt))
