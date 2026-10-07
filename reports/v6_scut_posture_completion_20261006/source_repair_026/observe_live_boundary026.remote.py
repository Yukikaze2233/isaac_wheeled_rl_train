import pathlib,json,datetime,subprocess
r=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358');src=r/'source_6f07ec298e01'
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):return None
kinds={'chassis_remote_job.py':'wrapper','run_task_campaign.py':'supervisor','train_chassis.py':'worker','evaluate_chassis.py':'evaluator'}
owners=[]
for f in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=[x.decode() for x in f.read_bytes().split(b'\0') if x];entry=next((pathlib.Path(x) for x in a[1:] if x.endswith('.py')),None)
  if entry and entry.name in kinds and any(x==str(r) or x==str(r/'train') or x.startswith(str(r/'train')+'/') for x in a):
   stat=(f.parent/'stat').read_text().rsplit(')',1)[1].split();owners.append({'pid':int(f.parent.name),'ppid':int(stat[1]),'starttime_ticks':int(stat[19]),'state':stat[0],'kind':kinds[entry.name],'entry':str(entry),'command':a})
 except OSError:pass
workers=[x for x in owners if x['kind']=='worker'];tr=None
if len(workers)==1:tr=pathlib.Path(workers[0]['command'][workers[0]['command'].index('--run-dir')+1])
x={'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owners':owners,'job':read(r/'job.json'),'state':read(r/'train/campaign_state.json'),'account':read(r/'train/consumption.json'),'training_dir':str(tr) if tr else None,'progress':read(tr/'progress.json') if tr else None,'completion':read(tr/'completion.json') if tr else None,'resume_verification':read(tr/'resume_verification.json') if tr else None,'agent_config':read(tr/'agent_config.json') if tr else None,'capture':read(r/'first_progress_023/index.json'),'launch':read(r/'recovery_entropy_launch_023.json'),'log_tail':(r/'job.resume_023.log').read_bytes()[-5000:].decode(errors='replace') if (r/'job.resume_023.log').exists() else None,'PPO_actions':0,'qualification':None}

import sys,hashlib
observer_dir=r/'boundary_observer_026'
sys.path.insert(0,str(observer_dir))
from queue_task_campaign_boundary_continuation import process_identity
observer=[v for p in pathlib.Path('/proc').glob('[0-9]*/cmdline') if (v:=process_identity(int(p.parent.name))) and v['entry']==str(observer_dir/'queue_task_campaign_boundary_continuation.py')]
x['boundary_observer']={'processes':observer,'status':read(r/'boundary_continuation_026.status.json'),'state':read(r/'.task_campaign_boundary_v24.state.json'),'job_sha256':hashlib.sha256((r/'job.json').read_bytes()).hexdigest(),'actions_exist':{name:(r/name).exists() for name in ('.task_campaign_boundary_v24.signal.intent.json','.task_campaign_boundary_v24.launch.intent.json','source_revision_026.json','boundary_continuation_026.receipt.json')},'code_manifest':read(observer_dir/'code_manifest.json'),'launch_receipt':read(observer_dir/'launch_receipt.json')}
if len(observer)==1:
 p=pathlib.Path('/proc')/str(observer[0]['pid']);status=(p/'status').read_text();stat=(p/'stat').read_text().rpartition(')')[2].split()
 x['boundary_observer']['cpu_ticks']=int(stat[11])+int(stat[12]);x['boundary_observer']['memory_lines']=[s for s in status.splitlines() if s.startswith(('VmRSS:','VmSize:','Threads:'))]
x['old_boundary_observer_retirement']=read(r/'boundary_observer_025/retirement_receipt_026.json')
print(json.dumps(x))
