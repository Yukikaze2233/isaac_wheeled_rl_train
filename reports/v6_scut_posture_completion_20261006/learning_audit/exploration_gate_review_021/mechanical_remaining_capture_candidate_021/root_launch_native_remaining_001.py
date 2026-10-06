#!/usr/bin/env python3
"""Root-owned single diagnostic child, no PPO or external process signals."""
import hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
SHA='af75f849a42e7447b35cbdb7c5d21628c27ae04a5b976a7c5d6e2897cbb698a2'
REVIEW_SHA='6c1a73974a5ccdf1d76e8334d1281e61e93eed8127ff04d42a9c3cdc2b9efa1a'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def main():
 review_path=HERE/'independent_review_001.json'
 assert digest(review_path)==REVIEW_SHA
 review=json.loads(review_path.read_text())
 assert review['status']=='ready_for_root_only_native_measurement_not_qualified' and review['candidate_manifest_sha256']==SHA
 assert review['author_App_free_preflight']['native_scene_run'] is False and review['CPU']['exit_code']==0
 spec=importlib.util.spec_from_file_location('root_side_bound',HERE/'run_remaining_capture_021.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 side,capture,manifest=mod.verify_candidate(SHA)
 inv=json.loads((HERE/'invocation.json').read_text());command=inv['native_command'];out=Path(command[command.index('--output')+1])
 assert out==HERE/'native_remaining_001' and not out.exists()
 pair=capture.load(capture.ORCHESTRATOR,'root_side_pair');resources=pair.local_GPU_empty()
 env=dict(os.environ,**inv['environment']);begin=time.monotonic()
 with (HERE/'native_remaining_001.stdout.log').open('x') as f:
  child=subprocess.Popen(command,cwd=inv['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT)
  proc=Path('/proc')/str(child.pid);s=(proc/'stat').read_text();starttime=int(s[s.rfind(')')+2:].split()[19])
  start=dict(pid=child.pid,command=command,starttime_ticks=starttime,started_at_utc=datetime.now(timezone.utc).isoformat(),candidate_manifest_sha256=SHA,independent_review_sha256=REVIEW_SHA,root_launcher_sha256=digest(Path(__file__)),resource_check=resources,PPO_updates=0,formal_updates_consumed=0,qualification=None)
  write(HERE/'native_remaining_001.started.json',start)
  try:code=child.wait(timeout=900)
  except subprocess.TimeoutExpired:
   write(HERE/'native_remaining_001.lifecycle_limit.json',dict(**start,status='root_attention_not_signalled',automatic_signal_sent=False));raise
 result=dict(**start,process_exit_observed=True,process_exit_code=code,process_present_after_wait=proc.exists(),process_wall_seconds=time.monotonic()-begin,finished_at_utc=datetime.now(timezone.utc).isoformat(),App_close_returned=None)
 write(HERE/'native_remaining_001.process.json',result)
 print(json.dumps({k:result[k] for k in ('pid','process_exit_code','process_wall_seconds','PPO_updates')}))
 if code:raise RuntimeError('Native diagnostic failed; preserve all evidence')
if __name__=='__main__':main()
