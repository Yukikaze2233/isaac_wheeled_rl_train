import pathlib,subprocess,sys,hashlib,json,concurrent.futures,shutil
root=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929')
out=root/'final-heldout-recheck-v1';out.mkdir(exist_ok=False)
(out/'scripts').mkdir()
for name in ['identify_pair_dynamics.py','audit_v5_motor_coupling.py']:
 shutil.copy2(root/'diagnostic-fix-v1/scripts'/name,out/'scripts'/name)
shutil.copy2(__file__,out/'run.py')
bundle=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-capacity-check-20260929/frozen/model/纯底盘_v6_232mm/urdf')
manifest={'scope':'fixed existing six-parameter candidates; six held-out segments repeated on final r3 with sealed script; no refit','source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'scripts').glob('*.py')},'asset_manifest_sha256':hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),'commands':[],'training_release':False}
commands={}
for side,run in [('left','run_20260928T073044Z'),('right','run_20260928T144606Z')]:
 shutil.copy2(root/f'{side}-fit/parameters.json',out/f'{side}-parameters.json')
 commands[side]=[]
 for label in ['baseline','candidate']:
  cmd=[sys.executable,str(out/'scripts/identify_pair_dynamics.py'),str(root/run),'--bundle',str(bundle),'--coordinate-sign','-1','--physics-dt','.001','--side',side,'--segments','37','39','41','43','79','89','--output',str(out/f'{side}-{label}')]
  if label=='candidate':cmd+=['--params',str(out/f'{side}-parameters.json')]
  manifest['commands'].append(cmd);commands[side].append((label,cmd))
(out/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
def execute(side):
 for label,cmd in commands[side]:
  with (out/f'{side}-{label}.log').open('x') as log:
   subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 with (out/f'{side}-completed.json').open('w') as f:json.dump({'status':'completed'},f)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(execute,commands))
print('FINAL_HELDOUT_RECHECK_COMPLETED',flush=True)
