#!/usr/bin/env python3
"""Start exactly one CPU-only native bookkeeping reader before actual launch."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUN='/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 target=HERE/'first_progress_observer_dispatch_022.json';assert not target.exists();meta=json.loads((HERE/'source_candidate_022.json').read_text());admission=json.loads((HERE/'admission_022.json').read_text());assert admission['status']=='passed' and admission['source_commit']==meta['source_commit'];code=(HERE/'observe_first_progress_022.py').read_text()
 payload=dict(RUN=RUN,SOURCE=meta['source_remote'],OBSERVER_CODE=code)
 program='\n'.join(k+'='+repr(v) for k,v in payload.items())+r"""
import hashlib,json,pathlib,subprocess
from datetime import datetime,timezone
r=pathlib.Path(RUN);script=r/'observe_first_progress_022.py';out=r/'first_progress_capture_022';log=r/'first_progress_observer_022.log';receipt=r/'first_progress_observer_022.json'
assert (r/'source_revision_022.json').exists() and (r/'predecessor_022.json').exists() and not (r/'recovery_quality_launch_022.json').exists()
assert not any(p.exists() for p in (script,out,log,receipt))
with script.open('x') as f:f.write(OBSERVER_CODE)
command=['/home/kaiser/robot-rl-sim60/env/bin/python','-B',str(script),'--source-root',SOURCE,'--run-root',RUN,'--output',str(out),'--timeout-seconds','900']
with log.open('x') as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(schema='source022-first-native-progress-observer-dispatch-v1',pid=child.pid,command=command,observer_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),dispatched_at_utc=datetime.now(timezone.utc).isoformat(),read_only=True,PPO_actions=0,qualification=None)
with receipt.open('x') as f:json.dump(d,f,indent=2);f.write(chr(10))
print(json.dumps(d))
"""
 q=subprocess.run(['ssh','-p','2222','-oBatchMode=yes','-oConnectTimeout=10','kaiser@192.168.64.234','/home/kaiser/robot-rl-sim60/env/bin/python','-B','-'],input=program,text=True,capture_output=True,timeout=30);assert q.returncode==0,q.stderr;d=json.loads(q.stdout);assert d['observer_sha256']==sha(HERE/'observe_first_progress_022.py');target.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
if __name__=='__main__':main()
