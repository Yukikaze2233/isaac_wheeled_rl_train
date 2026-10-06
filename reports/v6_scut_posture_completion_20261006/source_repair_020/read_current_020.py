import json,pathlib,datetime,hashlib,subprocess
r=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358');src=r/'source_7f315f30741b';w=r/'train/recovery/attempt_004/train'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
owners=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=[x.decode() for x in p.read_bytes().split(b'\0') if x]
  if any(x in [str(r),str(r/'train')] or x.startswith(str(r/'train')+'/') for x in a) and any(x.endswith('/'+n) for x in a for n in ['run_task_campaign.py','train_chassis.py','chassis_remote_job.py','evaluate_chassis.py']):owners.append(dict(pid=int(p.parent.name),command=a))
 except OSError:pass
out=dict(schema='source020-readonly-current-v1',read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),progress=read(w/'progress.json'),budget=read(r/'train/consumption.json'),job=read(r/'job.json'),owners=owners,source_sha256={n:sha(src/n) for n in ['src/wheeled_tasks/chassis/env.py','src/wheeled_tasks/chassis/ordinary_posture_curriculum.py','contracts/v6_task_campaign_v18.json']})
out['gpu']=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
out['/usr/bin/tmux']=subprocess.run(['/usr/bin/tmux','-S','/home/kaiser/robot-rl-sim60/tmux.sock','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.strip()
print(json.dumps(out))
