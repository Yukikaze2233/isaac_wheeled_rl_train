#!/usr/bin/env python3
"""Root-owned execute-once journal for reviewed source022 emitters."""
import argparse,hashlib,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE_PYTHON='/home/kaiser/robot-rl-sim60/env/bin/python'
SSH=['ssh','-p','2222','-oBatchMode=yes','-oConnectTimeout=10','kaiser@192.168.64.234']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('stop','prepare','launch','inspect'));a=p.parse_args()
 prefix=HERE/(a.mode+'_actual_022')
 for suffix in ('.code.py','.stdout.txt','.stderr.txt','.receipt.json'):
  assert not prefix.with_suffix(suffix).exists(),'Existing actual journal: inspect; never repeat mutation blindly'
 admission=HERE/'admission_022.json';source=HERE/'source_candidate_022.json'
 d=json.loads(admission.read_text());m=json.loads(source.read_text());assert d['status']=='passed' and d['qualification'] is None and d['source_commit']==m['source_commit'] and d['archive_sha256']==m['archive_sha256']
 if a.mode=='stop':emitter=HERE/'emit_stop_previous_022.py';argv=[sys.executable,str(emitter)]
 elif a.mode in ('prepare','launch'):emitter=HERE/'deploy_script_repair_022.py';argv=[sys.executable,str(emitter),a.mode,'--emit-remote-code']
 else:
  emitter=HERE/'verify_first_PPO_022.remote.py';argv=[sys.executable,str(emitter),'--emit-remote-code','--source',str(source),'--admission',str(admission),'--predecessor',str(HERE/'predecessor_022.json'),'--revision',str(HERE/'source_revision_022.json'),'--phase-progress-capture','/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/first_progress_capture_022/index.json']
 emitted=subprocess.run(argv,capture_output=True);assert emitted.returncode==0,emitted.stderr.decode()
 code=prefix.with_suffix('.code.py')
 with code.open('xb') as stream:stream.write(emitted.stdout)
 begin=time.monotonic();started=datetime.now(timezone.utc).isoformat()
 with prefix.with_suffix('.stdout.txt').open('xb') as out,prefix.with_suffix('.stderr.txt').open('xb') as err:
  q=subprocess.run([*SSH,REMOTE_PYTHON,'-B','-'],input=emitted.stdout,stdout=out,stderr=err,timeout=180)
 receipt=dict(schema='source022-root-actual-remote-emitted-step-v1',mode=a.mode,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),exit_code=q.returncode,wall_seconds=time.monotonic()-begin,source_commit=m['source_commit'],emitter_sha256=sha(emitter),admission_sha256=sha(admission),remote_code_sha256=sha(code),stdout_sha256=sha(prefix.with_suffix('.stdout.txt')),stderr_sha256=sha(prefix.with_suffix('.stderr.txt')),qualification=None)
 prefix.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));assert q.returncode==0,'Remote action failed: preserve and inspect before further action'
if __name__=='__main__':main()
