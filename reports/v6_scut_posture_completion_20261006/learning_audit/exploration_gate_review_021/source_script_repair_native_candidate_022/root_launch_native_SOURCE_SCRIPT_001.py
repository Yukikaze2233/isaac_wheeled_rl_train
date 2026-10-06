#!/usr/bin/env python3
"""Root-owned single diagnostic child, no PPO or external process signals."""
import hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
SHA='40554571a0b3656aaed22a14e862c726704c51d550a1b07602d1d6d478de7ace'
REVIEW_SHA='ca3affeeab8f5ec4f26c1ff4011c4880bbca945b1593af1b6ea4675c35525de6'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def main():
 review_path=HERE/'independent_narrow_source_review_022.json'
 assert digest(review_path)==REVIEW_SHA
 review=json.loads(review_path.read_text())
 assert review['status']=='no_observer_level_blocker_found_native_execution_pending' and review['candidate_manifest_sha256']==SHA
 assert review['all_bound_inputs_unchanged'] is True and review['PPO_updates']==0 and review['qualification'] is None
 assert json.loads((HERE/'CPU_preflight_receipt_002.json').read_text())['exit_code']==0
 spec=importlib.util.spec_from_file_location('root_side_bound',HERE/'run_source_script_native_022.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 manifest=mod.verify_candidate(SHA);capture=mod.load(mod.CAPTURE,'root_source_capture')
 inv=json.loads((HERE/'invocation_022.json').read_text());command=inv['native_command'];out=Path(command[command.index('--output')+1])
 assert out==HERE/'native_SOURCE_SCRIPT_001' and not out.exists()
 pair=capture.load(capture.ORCHESTRATOR,'root_side_pair');resources=pair.local_GPU_empty()
 native_env={'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMNI_KIT_ACCEPT_EULA':'YES','PYTHONDONTWRITEBYTECODE':'1'}
 plan,admission=pair.load_plan(capture.PAIR);native_cwd=manifest['source_root']
 env=dict(os.environ,**native_env);begin=time.monotonic()
 with (HERE/'native_SOURCE_SCRIPT_001.stdout.log').open('x') as f:
  child=subprocess.Popen(command,cwd=native_cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
  proc=Path('/proc')/str(child.pid);s=(proc/'stat').read_text();starttime=int(s[s.rfind(')')+2:].split()[19])
  start=dict(pid=child.pid,command=command,starttime_ticks=starttime,started_at_utc=datetime.now(timezone.utc).isoformat(),candidate_manifest_sha256=SHA,independent_review_sha256=REVIEW_SHA,root_launcher_sha256=digest(Path(__file__)),resource_check=resources,environment=native_env,cwd=native_cwd,PPO_updates=0,formal_updates_consumed=0,qualification=None)
  write(HERE/'native_SOURCE_SCRIPT_001.started.json',start)
  try:code=child.wait(timeout=900)
  except subprocess.TimeoutExpired:
   write(HERE/'native_SOURCE_SCRIPT_001.lifecycle_limit.json',dict(**start,status='root_attention_not_signalled',automatic_signal_sent=False));raise
 result=dict(**start,process_exit_observed=True,process_exit_code=code,process_present_after_wait=proc.exists(),process_wall_seconds=time.monotonic()-begin,finished_at_utc=datetime.now(timezone.utc).isoformat(),App_close_returned=None)
 write(HERE/'native_SOURCE_SCRIPT_001.process.json',result)
 print(json.dumps({k:result[k] for k in ('pid','process_exit_code','process_wall_seconds','PPO_updates')}))
 if code:raise RuntimeError('Native diagnostic failed; preserve all evidence')
if __name__=='__main__':main()
