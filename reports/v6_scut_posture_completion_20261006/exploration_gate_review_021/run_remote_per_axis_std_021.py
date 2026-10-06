#!/usr/bin/env python3
"""Root-only read existing event tags; no model inference or production writes."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
helper=ROOT/'read_per_axis_std_021.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='51453b3fe0dc5a8ece02fc675ef62f54f5cbd4226bf11da0658cd8ffb0407cb8'
out=ROOT/'per_axis_std_021_001'
out.mkdir(exist_ok=False)
ssh=['ssh','-p','2222','-oBatchMode=yes','-oConnectTimeout=10','kaiser@192.168.64.234']
scp=['scp','-P','2222','-oBatchMode=yes','-oConnectTimeout=10']
remote='/tmp/readonly_per_axis_std_021_001'
base='/tmp/readonly_recovery_window_021_first100_tail50_read002'
subprocess.run(ssh+['mkdir',remote],check=True,timeout=20)
subprocess.run(scp+[str(helper),'kaiser@192.168.64.234:'+remote+'/read_per_axis_std_021.py'],check=True,timeout=20)
command=['/home/kaiser/robot-rl-sim60/env/bin/python','-B',remote+'/read_per_axis_std_021.py','--reader021',base+'/read_recovery_window_021.py','--reader020',base+'/read_recovery_window_020.py','--reader019',base+'/read_recovery_window_019.py','--launch','/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/recovery_quality_launch_021.json','--first-ppo',base+'/first_PPO_verification_021.json','--inventory',base+'/frozen_source_inventory_021.json','--output',remote+'/report.json']
(out/'invocation.json').write_text(json.dumps(dict(command=command,helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='remote_CPU_readonly_existing_event_tags_no_GPU_no_PPO'),indent=2)+'\n')
with (out/'stdout.log').open('x') as stream:
 subprocess.run(ssh+command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=55)
remote_sha=subprocess.check_output(ssh+['sha256sum',remote+'/report.json'],text=True,timeout=20).split()[0]
subprocess.run(scp+['kaiser@192.168.64.234:'+remote+'/report.json',str(out/'report.json')],check=True,timeout=20)
assert hashlib.sha256((out/'report.json').read_bytes()).hexdigest()==remote_sha
(out/'download.json').write_text(json.dumps(dict(remote_report=remote+'/report.json',report_sha256=remote_sha,formal_updates_consumed=0,PPO_updates=0,qualification=None),indent=2)+'\n')
x=json.loads((out/'report.json').read_text());print(json.dumps(dict(output=str(out),status=x.get('status'),report_sha256=remote_sha)))
