from pathlib import Path
import os,json,hashlib,signal,datetime,sys
root=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
source=root/'source_integrated_v26_20261008'
intent=root/'boundary_wrong_rec028_stop001.intent.json'
receipt=root/'boundary_wrong_rec028_stop001.receipt.json'
def identity(pid):
 p=Path('/proc')/str(pid); stat=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,ppid=int(stat[1]),starttime=int(stat[19]),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def durable(path,data):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
assert not intent.exists() and not receipt.exists(),'existing intent never re-signal'
sup=identity(4132822);work=identity(4132982);wrapper=identity(4132792)
assert sup['starttime']==113063560 and sup['ppid']==wrapper['pid'] and wrapper['starttime']==113062488
assert work['starttime']==113068330 and work['ppid']==sup['pid']
assert str(source/'scripts/run_task_campaign.py') in sup['argv']
assert str(source/'contracts/v6_task_campaign_v26.json') in sup['argv']
assert str(root/'train/recovery/attempt_010/train') in work['argv']
assert '--integrated-source-selection-file' not in work['argv']
progress=json.loads((root/'train/recovery/attempt_010/train/progress.json').read_text())
sys.path.insert(0,str(root/'boundary_completion_adoption002'))
from queue_task_campaign_boundary_continuation import open_pidfd,send_pidfd_sigterm
fd=open_pidfd(sup['pid'])
assert identity(sup['pid'])==sup and identity(work['pid'])==work
base=dict(schema='named-wrong-rec-route-stop-v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='v26 launched recovery attempt010 instead of authorized integrated speed; seal full state before scheduler fix',supervisor=sup,worker=work,wrapper=wrapper,progress_before_signal=progress,rollback_refunds=False,worker_signal_directly=False,other_processes_signalled=False)
durable(intent,base)
send_pidfd_sigterm(fd);os.close(fd)
result=dict(base,status='SIGTERM_submitted_once_to_exact_supervisor',intent_sha256=hashlib.sha256(intent.read_bytes()).hexdigest(),remote_writes=2,signals_sent=1)
durable(receipt,result)
print(json.dumps(result))
