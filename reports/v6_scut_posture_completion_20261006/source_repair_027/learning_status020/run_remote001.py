from pathlib import Path
import datetime,hashlib,json,subprocess,time
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
SCRIPT=OUT/'latest100_actor7301_7400.reader001.remote.py'
PREFIX=OUT/'latest100_actor7301_7400.read001'
LIVE=BASE/'live035.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
live=json.loads(LIVE.read_text());progress=live['progress']
assert progress['actor_updates_in_block']>=7400 and live['completion'] is None
assert {r['kind']:(r['pid'],r['starttime_ticks']) for r in live['owners']}=={'wrapper':(3662137,96629148),'supervisor':(3662142,96629351),'worker':(3662149,96629601)}
assert [(r['pid'],r['starttime']) for r in live['boundary_observer']['processes']]==[(3928723,106534391)]
assert not any(live['boundary_observer']['actions_exist'].values())
prep=json.loads((OUT/'reader_preparation001.json').read_text())
for name,value in prep['prepared_files_SHA'].items():assert sha(OUT/name)==value
at=datetime.datetime.now(datetime.timezone.utc).isoformat()
ready=dict(schema='actual-ready-fixed100-observation-v1',checked_at_utc=at,live_file=str(LIVE),live_sha256=sha(LIVE),ready=True,end7400_actual=True,actual_actor_updates=progress['actor_updates_in_block'],actual_REC_actor_update=progress['task_actor_update'],PPO_actions=0,qualification=None,scope='Reuse sole resumed monitor020 actual live035; root exclusive event parse, no extra readiness SSH.')
with (OUT/'readiness001.json').open('x') as f:json.dump(ready,f,indent=2);f.write('\n')
args=['ssh','-p','2222','-o','BatchMode=yes','-o','ConnectTimeout=10','kaiser@192.168.64.234','/home/kaiser/robot-rl-sim60/env/bin/python -B -']
intent=dict(schema='root-exclusive-readonly-execution-intent-v1',started_at_utc=at,command=args,script_sha256=sha(SCRIPT),event_parse_limit=1,PPO_updates=0,remote_writes=0,GPU_actions=0)
with (OUT/'root_reader_ownership001.json').open('x') as f:json.dump(intent,f,indent=2);f.write('\n')
tick=time.monotonic();result=subprocess.run(args,input=SCRIPT.read_bytes(),capture_output=True,timeout=300)
for suffix,raw in [('.stdout',result.stdout),('.stderr',result.stderr)]:
 with Path(str(PREFIX)+suffix).open('xb') as f:f.write(raw)
receipt=dict(schema='actual-source023-fixed100-readonly-execution-v1',started_at_utc=at,command=args,script_file=str(SCRIPT),script_sha256=sha(SCRIPT),exit_code=result.returncode,wall_seconds=time.monotonic()-tick,stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),PPO_updates=0,remote_writes=0,GPU_calls=0,event_parse_count=1)
with Path(str(PREFIX)+'.execution.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
if result.returncode:print(result.stderr.decode());raise SystemExit(result.returncode)
raw=json.loads(result.stdout)
with (OUT/'latest100_actor7301_7400.raw001.json').open('x') as f:json.dump(raw,f,indent=2);f.write('\n')
print(json.dumps({'status':raw['status'],'complete':raw['complete'],'progress_after':{k:raw['progress_after'][k] for k in ['task_actor_update','actor_updates_in_block','successful_updates']}}),flush=True)
