"""Emit authenticated graceful-stop code only; never signal from this emitter."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
admission = json.loads((OUT / 'admission_022.json').read_text())
meta = json.loads((OUT / 'source_candidate_022.json').read_text())
assert admission['status'] == 'passed' and admission['qualification'] is None
assert admission['source_commit'] == meta['source_commit'] and admission['archive_sha256'] == meta['archive_sha256']
sha = hashlib.sha256((OUT / 'admission_022.json').read_bytes()).hexdigest()
print('ADMISSION_SHA=' + repr(sha))
print(r'''
import datetime,fcntl,hashlib,json,os,pathlib,signal
r=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
old=r/'source_d0dce6181df8'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
with (r/'recovery_quality_022.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'stop_requested_022.json').exists(),'Stop was already requested: inspect instead of repeating'
 job=read(r/'job.json');revision=read(r/'source_revision_021.json')
 assert job['command']==revision['command'] and read(r/job['source_revision_receipt'])==revision
 assert revision['source_commit']=='d0dce6181df83b03fe166abcbadef3c81de2e57e'
 assert pathlib.Path(revision['new_source_root'])==old
 assert sha(pathlib.Path(revision['source_archive']))==revision['source_archive_sha256']
 def entry(a):
  for x in a[1:]:
   if x in ('-c','-m'):return None
   if x.endswith('.py'):return pathlib.Path(x)
 owners=[]
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:
   a=[x.decode() for x in p.read_bytes().split(b'\0') if x]
   e=entry(a)
   if e and e.name in ('chassis_remote_job.py','run_task_campaign.py','train_chassis.py','evaluate_chassis.py') and any(x==str(r) or x==str(r/'train') or x.startswith(str(r/'train')+'/') for x in a):
    owners.append(dict(pid=int(p.parent.name),entry=str(e),command=a))
  except OSError:pass
 expected={3506636:'chassis_remote_job.py',3506645:'run_task_campaign.py',3506652:'train_chassis.py'}
 assert {x['pid']:pathlib.Path(x['entry']).name for x in owners}==expected,'Owners changed; re-audit'
 assert all(pathlib.Path(x['entry']).parent==old/'scripts' for x in owners)
 assert next(x['command'] for x in owners if x['pid']==3506645)==job['command']
 p=r/'train/recovery/attempt_005/train/progress.json';progress=read(p)
 assert progress['pid']==3506652
 assert progress['optimizer_phase']=='actor_and_critic' and progress['task_actor_update']>3747
 assert read(r/'train/campaign_state.json')['status']=='running'
 receipt=dict(schema='source022-authenticated-graceful-stop-request-v1',admission_sha256=ADMISSION_SHA,
   previous_source_commit=revision['source_commit'],requested_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   signal='SIGTERM',target_pid=3506645,target='supervisor_only',owners=owners,progress=progress,
   qualification=None,refund=False)
 with (r/'stop_requested_022.json').open('x') as f:
  json.dump(receipt,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
 directory=os.open(r,os.O_DIRECTORY)
 try:os.fsync(directory)
 finally:os.close(directory)
 os.kill(3506645,signal.SIGTERM)
 print(json.dumps(receipt))
''')
