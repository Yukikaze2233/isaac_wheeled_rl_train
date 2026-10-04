"""Read an existing campaign and sealed checkpoint; never start a worker."""
import json
from pathlib import Path
import subprocess


REMOTE = r'''
import datetime,hashlib,json,pathlib,shutil,subprocess
base=pathlib.Path('/home/kaiser/robot-rl-sim60')
root=base/'experiments/v6-task-campaign-20261001T110045Z-282358'
train=root/'train';directory=train/'speed/attempt_002/train'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=read(train/'campaign_state.json');account=read(train/'consumption.json')
progress=read(directory/'progress.json')
entries=[str(root/'source_b4ce80964d47/scripts'/n) for n in
 ('train_chassis.py','run_task_campaign.py','chassis_remote_job.py')]
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  args=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]
  entry=next((a for a in args if a in entries),None)
  if entry:
   ppid=int(next(l.split()[1] for l in (p/'status').read_text().splitlines() if l.startswith('PPid:')))
   processes.append(dict(pid=int(p.name),ppid=ppid,entry=entry,argv=args))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
single=len(processes)==3 and any(p['pid']==progress['pid'] for p in processes)
seals=[]
for p in (directory/'checkpoints').glob('update_*/model.checkpoint.json'):
 seal=read(p)
 if seal.get('sealed'):seals.append((seal['learning_updates'],p,seal))
_,p,seal=max(seals);cp=p.parent/'model.pt'
actual=sha(cp);contractsha=sha(p.parent/'contract.json')
if actual!=seal['checkpoint_sha256'] or contractsha!=seal['contract_sha256']:
 raise RuntimeError('Immutable checkpoint/contract SHA mismatch')
gpu_command=shutil.which('nvidia-smi')
if not gpu_command:
 gpu_command=next((p for p in ('/usr/lib/wsl/lib/nvidia-smi','/usr/bin/nvidia-smi')
   if pathlib.Path(p).is_file()),None)
gpu=None;gpu_error=None
try:
 if gpu_command:
  gpu=subprocess.check_output([gpu_command,'--query-gpu=utilization.gpu,memory.used,memory.total',
    '--format=csv,noheader'],text=True,timeout=5).strip()
 else:gpu_error='nvidia-smi unavailable in read-only SSH environment'
except (OSError,subprocess.SubprocessError) as e:gpu_error=str(e)
r=dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 campaign_status=state.get('status'),contract_sha256=state.get('contract_sha256'),
 campaign_consumed_updates=account['consumed_updates'],domain_consumed_updates=account['domain_consumed_updates'],
 progress=progress,processes=sorted(processes,key=lambda p:p['pid']),single_owned_process_chain=single,
 tmux_sessions=subprocess.check_output(['tmux','-S',str(base/'tmux.sock'),'list-sessions',
   '-F','#{session_name}'],text=True).splitlines(),gpu=gpu,gpu_error=gpu_error,
 latest_sealed_checkpoint=dict(path=str(cp),seal=seal,sha256=actual,contract_sha256=contractsha,hash_verified=True),
 qualification=False,scope='read_only_native_progress_and_immutable_seal_not_fixed_behavior_qualification')
if not single or r['contract_sha256']!='41319993aa53a54e32efaab725ec8b0e269f4b8b3576bc101f23c6e525c4cf73':
 raise RuntimeError('Running campaign identity/process count changed')
print(json.dumps(r,ensure_ascii=False))
'''


def main():
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-p', '2222',
         'kaiser@192.168.64.234', '/home/kaiser/robot-rl-sim60/env/bin/python -B -'],
        input=REMOTE, text=True, capture_output=True, timeout=25)
    result.check_returncode()
    receipt = json.loads(result.stdout)
    directory = Path(__file__).resolve().parent
    timestamp = receipt['observed_at_utc']
    name = 'live_' + timestamp[:10].replace('-', '') + 'T' + timestamp[11:16].replace(':', '') + 'Z.json'
    path = directory / name
    if path.exists():
        raise FileExistsError(path)
    text = json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'
    path.write_text(text)
    (directory / 'current_live_014.json').write_text(text)
    print(json.dumps(dict(file=str(path), time=timestamp,
        actor=receipt['progress']['actor_updates_in_block'],
        task_actor=receipt['progress']['task_actor_update'],
        consumed=receipt['campaign_consumed_updates'],
        domains=receipt['domain_consumed_updates'],
        gpu=receipt['gpu'], gpu_error=receipt['gpu_error'],
        single=receipt['single_owned_process_chain'],
        seal=receipt['latest_sealed_checkpoint']['seal']['learning_updates']), ensure_ascii=False))


if __name__ == '__main__':
    main()
