from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
OUT=Path(__file__).resolve().parent
script=OUT/sys.argv[1];prefix=OUT/sys.argv[2]
assert script.parent==OUT
started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
result=subprocess.run([sys.executable,'-B',str(script)],capture_output=True)
for suffix,value in (('.stdout',result.stdout),('.stderr',result.stderr)):
 with Path(str(prefix)+suffix).open('xb') as f:f.write(value)
receipt={'schema':'actual-local-fixed100-analysis-execution-v1','started_at_utc':started,'command':[sys.executable,'-B',str(script)],'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'exit_code':result.returncode,'wall_seconds':time.monotonic()-tick,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),'PPO_updates':0,'SSH_calls':0,'GPU_calls':0,'remote_writes':0}
with Path(str(prefix)+'.receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt));sys.stdout.flush();sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr)
sys.exit(result.returncode)
