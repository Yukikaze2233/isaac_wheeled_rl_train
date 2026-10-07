import datetime, hashlib, json, os, subprocess, time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent
output=base/'flat001'
if output.exists(): raise SystemExit('Refusing to overwrite an existing playback output')
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit(): continue
 try: cmd=(proc/'cmdline').read_bytes().split(b'\0')
 except (OSError, PermissionError): continue
 if any(Path(token.decode(errors='replace')).name in ('play_chassis_suite.py','play_v5_grounded.py') for token in cmd):
  raise SystemExit('A native local playback already exists; close its window before starting another')
cmd=['/home/yukikaze/isaacsim60-venv/bin/python','-B',str(root/'scripts/play_chassis_suite.py'),'--onnx',str(root/'models/v6_recovery_candidate_38000_20261008/policy.onnx'),'--scenario','flat','--output',str(output),'--device','cpu']
env=os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1', OMNI_KIT_ACCEPT_EULA='YES', LD_LIBRARY_PATH='/home/yukikaze/.local/lib/compat'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),TMPDIR='/home/yukikaze/.cache/kit-tmp')
Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (base/'flat001.launch.log').open('wb') as log:
 child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try: ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
except OSError: ticks=None
files=['scripts/play_chassis_suite.py','scripts/chassis_suite_support.py','scripts/play_v5_grounded.py','src/wheeled_tasks/chassis/preflight.py','models/v6_recovery_candidate_38000_20261008/policy.onnx','models/v6_recovery_candidate_38000_20261008/policy.onnx.contract.json']
receipt={'schema':'actual-local-onnx-gui-launch-v1','started_at_utc':started,'pid':child.pid,'starttime_ticks':ticks,'argv':cmd,'cwd':str(root),'display':env.get('DISPLAY'),'detached':True,'output':str(output),'stdout_stderr':str(base/'flat001.launch.log'),'source_file_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},'model_role':'recovery_candidate','qualification':None,'PPO_updates':0,'remote_actions':0,'runtime_verification':'pending'}
(base/'flat001.launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
