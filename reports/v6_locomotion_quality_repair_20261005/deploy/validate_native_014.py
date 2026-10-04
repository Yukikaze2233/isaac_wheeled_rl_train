"""Run matched frozen native probes serially; no PPO or state refund."""
from pathlib import Path
import json,shlex,subprocess
OUT=Path(__file__).parent
BASE='/home/kaiser/robot-rl-sim60'
RUN=BASE+'/experiments/v6-task-campaign-20261001T110045Z-282358'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','2222','kaiser@192.168.64.234']
meta=json.loads((OUT/'source_candidate_014.json').read_text())
source=meta['source_remote']
state_code="from pathlib import Path;import json; r=Path("+repr(RUN)+");j=json.loads((r/'job.json').read_text());assert not Path('/proc/'+str(j['pid'])+'/cmdline').exists();print(json.dumps(json.loads((r/'train/campaign_state.json').read_text())['domains']))"
state=json.loads(subprocess.check_output(SSH+[shlex.join(['python3','-c',state_code])],text=True))
for domain,clock in [('speed',5063),('terrain',2186)]:
 output=RUN+'/nonstandard_'+domain+'_probe_014'
 command=[BASE+'/env/bin/python','-B',source+'/scripts/probe_nonstandard_start.py',
 '--contract',source+'/contracts/v6_task_campaign_'+domain+'_worker_v14.json',
 '--checkpoint',state[domain]['candidate_checkpoint'],'--actor-update',str(clock),
 '--seconds','10','--output',output]
 inner='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 && '+shlex.join(command)
 result=subprocess.run(SSH+[shlex.join(['bash','-lc',inner])],capture_output=True,text=True)
 (OUT/('native_'+domain+'_014.stdout')).write_text(result.stdout)
 (OUT/('native_'+domain+'_014.stderr')).write_text(result.stderr)
 subprocess.run(['scp','-P','2222','kaiser@192.168.64.234:'+output+'/report.json',str(OUT/('native_'+domain+'_014.json'))],check=True)
 report=json.loads((OUT/('native_'+domain+'_014.json')).read_text())
 assert result.returncode==0 and report['status']=='passed',report.get('error',report['status'])
 print(json.dumps(dict(domain=domain,status=report['status'],counts=report['counts'],wall_seconds=report['wall_seconds'])),flush=True)
