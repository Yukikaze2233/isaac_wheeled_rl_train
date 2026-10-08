from pathlib import Path
import datetime,json,os,signal,subprocess,time
root=Path(__file__).resolve().parents[3]
base=Path(__file__).resolve().parent
assert not (base/'native002_wide8').exists() and not (base/'native002_wide8.process.json').exists()
cmd=['/home/yukikaze/isaacsim60-venv/bin/python','-B',str(base/'compare_mean.py'),'--onnx',str(root/'models/v6_recovery_candidate_41374_20261009/policy.onnx'),'--onnx',str(root/'models/v6_recovery_candidate_38000_20261008/policy.onnx'),'--output',str(base/'native002_wide8'),'--steps','600','--device','cpu','--wide-floor8']
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES',LD_LIBRARY_PATH='/home/yukikaze/.local/lib/compat',TMPDIR='/home/yukikaze/.cache/kit-tmp')
def available():
 return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
assert available()>6*1024**3,'Insufficient memory; existing GUI stays untouched'
start=time.monotonic();reason=None;minimum=available()
with (base/'native002_wide8.log').open('xb') as log:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  current=available();minimum=min(minimum,current)
  if current<1.5*1024**3 or time.monotonic()-start>420:
   reason='memory_floor' if current<1.5*1024**3 else 'wall_timeout'
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(2)
record=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,pid=p.pid,exit_code=p.returncode,guard_stop_reason=reason,minimum_available_bytes=minimum,wall_seconds=time.monotonic()-start,other_processes_signalled=False,PPO_updates=0)
(base/'native002_wide8.process.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
