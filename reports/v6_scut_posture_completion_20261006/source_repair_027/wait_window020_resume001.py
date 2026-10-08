from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess,time
base=Path(__file__).resolve().parent
helper=base/'observe_live_boundary027.remote.py';code=helper.read_text()
command=['ssh','-p','2222','-o','BatchMode=yes','-o','ConnectTimeout=10','kaiser@192.168.64.234',shlex.join(['/home/kaiser/robot-rl-sim60/env/bin/python','-B','-c',code])]
output=base/'live035.json';assert not output.exists()
started=time.monotonic();print(json.dumps({'monitor_pid':os.getpid(),'target_actor_end':7400,'read_only':True}),flush=True)
while time.monotonic()-started<1800:
 at=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
 try:r=subprocess.run(command,capture_output=True,timeout=30)
 except subprocess.TimeoutExpired:
  print(json.dumps({'time':at,'observation':'timeout','terminal_training':False}),flush=True);time.sleep(60);continue
 if r.returncode:
  print(json.dumps({'time':at,'observation':'ssh_failure','exit_code':r.returncode,'terminal_training':False}),flush=True);time.sleep(60);continue
 d=json.loads(r.stdout);progress=d['progress'];b=d['boundary_observer']
 same=[(v['pid'],v['starttime_ticks']) for v in d['owners']]==[(3662137,96629148),(3662142,96629351),(3662149,96629601)]
 observer=[(v['pid'],v['starttime']) for v in b['processes']]==[(3928723,106534391)]
 actor=progress['actor_updates_in_block'];ready=actor>=7400
 print(json.dumps({'read_at_utc':d['read_at_utc'],'REC':progress['task_actor_update'],'actor':actor,'same_owners':same,'same_observer':observer,'candidate_adopted':any(b['actions_exist'].values()),'ready':ready}),flush=True)
 if ready or not same or not observer or d['completion'] is not None or any(b['actions_exist'].values()):
  sha=lambda raw:hashlib.sha256(raw).hexdigest()
  for suffix,raw in [('.stdout',r.stdout),('.stderr',r.stderr)]:
   with (base/('live035'+suffix)).open('xb') as f:f.write(raw)
  receipt=dict(schema='actual-reviewed-remote-helper-execution-v1',started_at_utc=at,helper_file=str(helper),helper_sha256=sha(helper.read_bytes()),exit_code=r.returncode,wall_seconds=time.monotonic()-tick,stdout_sha256=sha(r.stdout),stderr_sha256=sha(r.stderr),read_only_wait_monitor=True,waiting_observations_not_promoted_to_behavior=True,PPO_updates=0,GPU_calls=0,training_signals=0)
  with (base/'live035.execution.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
  with output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
  print(json.dumps({'saved':str(output),'ready':ready,'wait_seconds':time.monotonic()-started}),flush=True);break
 time.sleep(60)
else:print(json.dumps({'observation':'monitor_deadline','terminal_training':False}),flush=True)
