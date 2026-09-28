import pathlib,subprocess,sys,concurrent.futures,json,hashlib
p=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-stop-comparison-20260929')
bundle=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-capacity-check-20260929/frozen/model/纯底盘_v6_232mm/urdf')
report={'source_commit':'d7b2dd7af423660b97a9be09ce1ecbc94d43b60e','source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'scripts').glob('*.py')},'commands':[]}
for side in ['left','right']:
 report['commands'].append([sys.executable,str(p/'scripts/compare_pair_knee_stops.py'),'--side',side,'--bundle',str(bundle),'--experiment-root','/home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929','--output',str(p/side)])
(p/'launch.json').write_text(json.dumps(report,indent=2)+'\n')
def execute(cmd):
 side=cmd[cmd.index('--side')+1]
 with (p/(side+'.log')).open('x') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(execute,report['commands']))
(p/'completed.json').write_text(json.dumps({'status':'completed','cases_per_side':65,'training_release':False})+'\n')
print('STOP_COMPARISON_COMPLETED',flush=True)
