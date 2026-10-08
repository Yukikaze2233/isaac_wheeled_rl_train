from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
base=Path(__file__).resolve().parent
script=base/sys.argv[1];name=sys.argv[2]
assert script.parent==base and script.name.endswith('.remote.py')
for suffix in ['.json','.stdout','.stderr','.execution.json']:assert not (base/(name+suffix)).exists()
start=time.monotonic()
r=subprocess.run(['ssh','-p','2222','-o','BatchMode=yes','-o','ConnectTimeout=10','kaiser@192.168.64.234','/home/kaiser/robot-rl-sim60/env/bin/python -B -'],input=script.read_bytes(),capture_output=True,timeout=180)
(base/(name+'.stdout')).write_bytes(r.stdout);(base/(name+'.stderr')).write_bytes(r.stderr)
(base/(name+'.execution.json')).write_text(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),exit_code=r.returncode,wall_seconds=time.monotonic()-start),indent=2)+'\n')
if r.stdout:
 d=json.loads(r.stdout);(base/(name+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:d.get(k) for k in ['status','reason','actual_origin_recovery_actor_update','authoritative_account']},ensure_ascii=False))
else:print(r.stderr.decode()[-3000:])
raise SystemExit(r.returncode)
