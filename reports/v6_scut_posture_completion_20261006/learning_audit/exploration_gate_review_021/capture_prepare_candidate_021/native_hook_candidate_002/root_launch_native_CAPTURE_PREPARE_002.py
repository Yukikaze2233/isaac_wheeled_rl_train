#!/usr/bin/env python3
"""Root-owned single diagnostic child, no PPO or external process signals."""
import hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
SHA='ed03fa1d06be18b665e13d667b905c187599acc33a8ef11540b7b87a80b2f1f7'
REVIEW_SHA='a15ec4e30537ec7936d7f0812e3301a9b32d0ec5e9000cf0232a7fee13636bc7'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def main():
 review_path=HERE/'independent_narrow_source_review_002.json'
 assert digest(review_path)==REVIEW_SHA
 review=json.loads(review_path.read_text())
 assert review['status']=='no_blocker_found_within_readonly_source_scope' and review['candidate_manifest_sha256']==SHA
 assert review['native_execution_pending'] is True and review['PPO_updates']==0
 assert json.loads((HERE/'CPU_preflight_receipt_001.json').read_text())['observed_exit_code']==0
 spec=importlib.util.spec_from_file_location('root_side_bound',HERE/'run_capture_prepare_021.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 side,capture,manifest=mod.verify_candidate(SHA)
 inv=json.loads((HERE/'invocation_001.json').read_text());command=inv['native_command'];out=Path(command[command.index('--output')+1])
 assert out==HERE/'native_CAPTURE_PREPARE_002' and not out.exists()
 pair=capture.load(capture.ORCHESTRATOR,'root_side_pair');resources=pair.local_GPU_empty()
 native_env={'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMNI_KIT_ACCEPT_EULA':'YES','PYTHONDONTWRITEBYTECODE':'1'}
 plan,admission=pair.load_plan(capture.PAIR);native_cwd=admission['source_root']
 env=dict(os.environ,**native_env);begin=time.monotonic()
 with (HERE/'native_CAPTURE_PREPARE_002.stdout.log').open('x') as f:
  child=subprocess.Popen(command,cwd=native_cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
  proc=Path('/proc')/str(child.pid);s=(proc/'stat').read_text();starttime=int(s[s.rfind(')')+2:].split()[19])
  start=dict(pid=child.pid,command=command,starttime_ticks=starttime,started_at_utc=datetime.now(timezone.utc).isoformat(),candidate_manifest_sha256=SHA,independent_review_sha256=REVIEW_SHA,root_launcher_sha256=digest(Path(__file__)),resource_check=resources,environment=native_env,cwd=native_cwd,PPO_updates=0,formal_updates_consumed=0,qualification=None)
  write(HERE/'native_CAPTURE_PREPARE_002.started.json',start)
  try:code=child.wait(timeout=900)
  except subprocess.TimeoutExpired:
   write(HERE/'native_CAPTURE_PREPARE_002.lifecycle_limit.json',dict(**start,status='root_attention_not_signalled',automatic_signal_sent=False));raise
 result=dict(**start,process_exit_observed=True,process_exit_code=code,process_present_after_wait=proc.exists(),process_wall_seconds=time.monotonic()-begin,finished_at_utc=datetime.now(timezone.utc).isoformat(),App_close_returned=None)
 write(HERE/'native_CAPTURE_PREPARE_002.process.json',result)
 print(json.dumps({k:result[k] for k in ('pid','process_exit_code','process_wall_seconds','PPO_updates')}))
 if code:raise RuntimeError('Native diagnostic failed; preserve all evidence')
if __name__=='__main__':main()
