ADMISSION_SHA='cdf803be3858134a2d605a8004da50c241f63989dc2beb2dc5cf5657c4703069'

import datetime,fcntl,hashlib,json,os,pathlib,signal
r=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
old=r/'source_7f315f30741b'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
with (r/'recovery_quality_021.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'stop_requested_021.json').exists(),'Stop was already requested: inspect instead of repeating'
 job=read(r/'job.json');revision=read(r/'source_revision_020.json')
 assert job['command']==revision['command'] and read(r/job['source_revision_receipt'])==revision
 assert revision['source_commit']=='7f315f30741be01d70046a11848443038c14b240'
 assert pathlib.Path(revision['new_source_root'])==old
 assert sha(pathlib.Path(revision['source_archive']))==revision['source_archive_sha256']
 owners=[]
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:
   a=[x.decode() for x in p.read_bytes().split(b'\0') if x]
   entries=[x for x in a if x.endswith(('/chassis_remote_job.py','/run_task_campaign.py','/train_chassis.py','/evaluate_chassis.py'))]
   if entries and any(x==str(r) or x==str(r/'train') or x.startswith(str(r/'train')+'/') for x in a):
    assert len(entries)==1;owners.append(dict(pid=int(p.parent.name),entry=entries[0],command=a))
  except OSError:pass
 expected={3494082:'chassis_remote_job.py',3494087:'run_task_campaign.py',3494096:'train_chassis.py'}
 assert {x['pid']:pathlib.Path(x['entry']).name for x in owners}==expected,'Owners changed; re-audit'
 assert all(pathlib.Path(x['entry']).parent==old/'scripts' for x in owners)
 assert next(x['command'] for x in owners if x['pid']==3494087)==job['command']
 p=r/'train/recovery/attempt_004/train/progress.json';progress=read(p)
 assert progress['pid']==3494096
 assert progress['optimizer_phase']=='actor_and_critic' and progress['task_actor_update']>2758
 assert read(r/'train/campaign_state.json')['status']=='running'
 receipt=dict(schema='source021-authenticated-graceful-stop-request-v1',admission_sha256=ADMISSION_SHA,
   previous_source_commit=revision['source_commit'],requested_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   signal='SIGTERM',target_pid=3494087,target='supervisor_only',owners=owners,progress=progress,
   qualification=None,refund=False)
 with (r/'stop_requested_021.json').open('x') as f:
  json.dump(receipt,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
 directory=os.open(r,os.O_DIRECTORY)
 try:os.fsync(directory)
 finally:os.close(directory)
 os.kill(3494087,signal.SIGTERM)
 print(json.dumps(receipt))

