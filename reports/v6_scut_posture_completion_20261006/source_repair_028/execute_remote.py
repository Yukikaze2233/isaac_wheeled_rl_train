"""Execute one literal reviewed remote helper and retain actual outputs."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,sys,time
base=Path(__file__).resolve().parent
helper=Path(sys.argv[1]).resolve();name=sys.argv[2]
assert helper.is_relative_to(base) and name.isidentifier()
code=helper.read_text();payload=Path(sys.argv[3]).read_bytes() if len(sys.argv)>3 else b""
command=["ssh","-p","2222","-o","BatchMode=yes","-o","ConnectTimeout=10","kaiser@192.168.64.234",shlex.join(["/home/kaiser/robot-rl-sim60/env/bin/python","-B","-c",code])]
out=base/(name+".stdout");err=base/(name+".stderr");receipt=base/(name+".execution.json")
assert not any(p.exists() for p in (out,err,receipt))
started=datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.monotonic()
result=subprocess.run(command,input=payload,capture_output=True,timeout=480)
out.write_bytes(result.stdout);err.write_bytes(result.stderr)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
r=dict(schema="actual-reviewed-remote-helper-execution-v1",started_at_utc=started,helper_file=str(helper),helper_sha256=sha(code.encode()),input_sha256=sha(payload),exit_code=result.returncode,wall_seconds=time.monotonic()-start,stdout_sha256=sha(result.stdout),stderr_sha256=sha(result.stderr))
receipt.write_text(json.dumps(r,indent=2)+"\n")
if result.returncode:
 print(result.stdout.decode(errors="replace")[-2500:]);print(result.stderr.decode(errors="replace")[-2500:])
else:
 value=json.loads(result.stdout);(base/(name+".json")).write_text(json.dumps(value,indent=2)+"\n");print(json.dumps(dict(status=value.get("status"),exit_code=result.returncode,wall_seconds=r["wall_seconds"],output=str(base/(name+".json")))))
raise SystemExit(result.returncode)
