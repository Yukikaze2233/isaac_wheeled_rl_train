#!/usr/bin/env python3
"""CPU-only actual source022 owner/startup/progress bookkeeping snapshot."""
import argparse,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUN='/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();meta=json.loads((HERE/'source_candidate_022.json').read_text())
 program='META='+repr(meta)+'\nRUN='+repr(RUN)+r"""
import hashlib,json,pathlib
from datetime import datetime,timezone
r=pathlib.Path(RUN);src=pathlib.Path(META['source_remote']);read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
revision=read(r/'source_revision_022.json');job=read(r/'job.json');assert revision['source_commit']==META['source_commit'] and job['command']==revision['command'] and read(r/job['source_revision_receipt'])==revision
assert sha(src/'contracts/v6_task_campaign_v20.json')==META['plan_sha256']
owners=[]
def entry(a):
 for x in a[1:]:
  if x in ('-c','-m'):return None
  if x.endswith('.py'):return pathlib.Path(x)
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=[x.decode() for x in p.read_bytes().split(b'\0') if x];e=entry(argv)
  if e and e.name in ('chassis_remote_job.py','run_task_campaign.py','train_chassis.py','evaluate_chassis.py') and any(x==str(r) or x==str(r/'train') or x.startswith(str(r/'train')+'/') for x in argv):owners.append(dict(pid=int(p.parent.name),entry=str(e),command=argv))
 except OSError:pass
assert len(owners)==3 and {pathlib.Path(x['entry']).name for x in owners}=={'chassis_remote_job.py','run_task_campaign.py','train_chassis.py'} and all(pathlib.Path(x['entry']).parent==src/'scripts' for x in owners)
workers=[]
for p in (r/'train/recovery').glob('attempt_*/invocation.json'):
 q=read(p)
 if str(src/'scripts/train_chassis.py') in q.get('command',[]):workers.append(p.parent/'train')
assert len(workers)==1;worker=workers[0];progress=read(worker/'progress.json') if (worker/'progress.json').exists() else None
startup=read(worker/'native_startup.json') if (worker/'native_startup.json').exists() else None
account=read(r/'train/consumption.json');capture=r/'first_progress_capture_022/index.json'
result=dict(read_at_utc=datetime.now(timezone.utc).isoformat(),source_commit=META['source_commit'],job=job,worker=str(worker),owners=owners,progress=progress,account={k:account[k] for k in ('contract_sha256','consumed_updates','actor_updates_consumed','domain_consumed_updates','remaining_updates')},source_script_revision=startup.get('recovery_script_revision') if startup else None,phase_capture=read(capture) if capture.exists() else None,qualification=None,PPO_actions=0,scope='readonly_existing_native_owners_startup_and_progress_no_GPU_or_model_inference')
print(json.dumps(result))
"""
 q=subprocess.run(['ssh','-p','2222','-oBatchMode=yes','-oConnectTimeout=10','kaiser@192.168.64.234','/home/kaiser/robot-rl-sim60/env/bin/python','-B','-'],input=program,text=True,capture_output=True,timeout=30);assert q.returncode==0,q.stderr;d=json.loads(q.stdout);d['local_inspector_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(output=str(a.output),read_at_utc=d['read_at_utc'],worker=d['worker'],progress=d['progress'],owners=[dict(pid=x['pid'],entry=Path(x['entry']).name) for x in d['owners']],capture_status=d['phase_capture'].get('status') if d['phase_capture'] else None,qualification=None)))
if __name__=='__main__':main()
