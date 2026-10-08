"""Write the unique observer's waiting admission after retiring the old one."""
from pathlib import Path
import json,subprocess,sys
root=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
directory=root/'boundary_observer_027'
old=json.loads((root/'boundary_observer_026/retirement_receipt_027.json').read_text())
if old['status']!='retired_waiting_observer_only' or old['PPO_signals']!=0:raise ValueError('Prior waiting observer retirement not authenticated')
command=[sys.executable,'-B',str(directory/'queue_task_campaign_boundary_continuation.py'),'--manifest',str(directory/'manifest.json'),'--check-only']
result=subprocess.run(command,capture_output=True,text=True,timeout=300)
if result.returncode:raise RuntimeError(result.stderr+result.stdout)
status=json.loads((root/'boundary_continuation_027.status.json').read_text())
if status['status']!='waiting' or status['reason']!='rec_normal_endpoint_not_complete':raise ValueError('Not waiting at normal REC endpoint')
for name in ('.task_campaign_boundary_v25.state.json','.task_campaign_boundary_v25.signal.intent.json','.task_campaign_boundary_v25.launch.intent.json','source_revision_027.json','boundary_continuation_027.receipt.json'):
 if (root/name).exists():raise ValueError('Unexpected boundary action/state exists: '+name)
print(json.dumps(dict(status='actual_check_only_waiting_admission',readback=status,command=command,exit_code=result.returncode,stdout=result.stdout,PPO_updates=0,signals=0,qualification=None)))
