from pathlib import Path
import sys,json,os,time,datetime,hashlib
root=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
old=root/'boundary_observer_027';new=root/'boundary_observer_028'
sys.path.insert(0,str(new))
from queue_task_campaign_boundary_continuation import BoundaryContinuationQueue,process_identity,open_pidfd,send_pidfd_sigterm
q=BoundaryContinuationQueue(json.loads((new/'manifest.json').read_text()))
job=q.previous_job();q.authenticate_owners(job,q.owners())
state=json.loads((old/'launch_receipt.json').read_text());owner=state['observer']
assert (owner['pid'],owner['starttime'])==(3928723,106534391)
assert owner['entry']==str(old/'queue_task_campaign_boundary_continuation.py')
assert process_identity(owner['pid'])==owner
status=json.loads((root/'boundary_continuation_027.status.json').read_text())
assert status['status']=='waiting' and status['reason']=='rec_normal_endpoint_not_complete'
q.active_rec_floor(json.loads((root/'train/campaign_state.json').read_text()), q.owners())
for rev,idx in ((22,24),(23,25),(24,26),(25,27),(26,28)):
 for name in (f'.task_campaign_boundary_v{rev}.signal.intent.json',f'.task_campaign_boundary_v{rev}.launch.intent.json',f'source_revision_{idx:03d}.json',f'boundary_continuation_{idx:03d}.receipt.json'):
  assert not (root/name).exists(),name
before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'job.json',old/'manifest.json',root/'.task_campaign_boundary_v25.state.json',root/'boundary_continuation_027.status.json')}
fd=open_pidfd(owner['pid'])
try:
 assert process_identity(owner['pid'])==owner
 q.authenticate_owners(q.previous_job(),q.owners())
 current_status=json.loads((root/'boundary_continuation_027.status.json').read_text())
 assert current_status==status and current_status['status']=='waiting'
 q.active_rec_floor(json.loads((root/'train/campaign_state.json').read_text()), q.owners())
 for rev,idx in ((22,24),(23,25),(24,26),(25,27),(26,28)):
  for name in (f'.task_campaign_boundary_v{rev}.signal.intent.json',f'.task_campaign_boundary_v{rev}.launch.intent.json',f'source_revision_{idx:03d}.json',f'boundary_continuation_{idx:03d}.receipt.json'):
   assert not (root/name).exists(),name
 send_pidfd_sigterm(fd)
finally:os.close(fd)
deadline=time.monotonic()+20
while process_identity(owner['pid']) is not None and time.monotonic()<deadline:time.sleep(.1)
assert process_identity(owner['pid']) is None
q.authenticate_owners(q.previous_job(),q.owners())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in before.items())
x=dict(schema='waiting-boundary-observer-retirement-v1',status='retired_waiting_observer_only',observer=owner,old_waiting_status=status,retired_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preserved_files_sha256=before,external_observer_signals=1,PPO_signals=0,PPO_updates=0,qualification=None)
with (old/'retirement_receipt_028.json').open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(x))
