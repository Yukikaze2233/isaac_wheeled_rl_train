import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/home/kaiser/robot-rl-sim60/experiments/v6-capacity-check-20260929')
source=root/'frozen'
sys.path.insert(0,str(source/'scripts'))
from benchmark_chassis_capacity import resources
cmd=[sys.executable,'-B',str(source/'scripts/train_chassis.py'),'--contract',str(source/'contracts/v6_asset_capacity_probe_v1.json'),'--stage','foundation','--num-envs','20000','--seed','918','--device','cuda:0','--research','--validate-scene','100','--updates','1','--max-runtime-seconds','3600','--run-dir',str(root/'scene')]
report={'source_commit':'6f12c81e6fee56e6e64520f9727129a3ed588e79','source_archive_sha256':hashlib.sha256((root/'source.tar.gz').read_bytes()).hexdigest(),'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'command':cmd,'scope':'physics_scene_only_no_ppo','started_unix':time.time(),'samples':[]}
for name in ['contracts/v6_asset_capacity_probe_v1.json','contracts/v6_asset_capacity_probe_expectations_v1.json','contracts/v6_asset_capacity_control_v1.json','model/纯底盘_v6_232mm/urdf/manifest.json']:
 report.setdefault('inputs_sha256',{})[name]=hashlib.sha256((source/name).read_bytes()).hexdigest()
(root/'launch.json').write_text(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2)+'\n')
with (root/'scene.log').open('x') as log:
 process=subprocess.Popen(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT)
 report['pid']=process.pid
 while process.poll() is None:
  try:
   sample=resources(include_host=len(report['samples'])%10==0);sample['unix']=time.time()
   report['samples'].append(sample)
   if sample['available_ram_kib']<2*1024**2 or sample['gpu_free_mib']<768:
    report['stop_reason']='memory_guard'; process.send_signal(signal.SIGTERM)
    try: process.wait(timeout=30)
    except subprocess.TimeoutExpired: process.kill(); process.wait()
  except Exception as error: report.setdefault('sampling_errors',[]).append(str(error))
  (root/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
  time.sleep(1)
report['exit_code']=process.returncode;report['finished_unix']=time.time()
if report['samples']:
 report['peak_gpu_mib']=max(s['gpu_used_mib'] for s in report['samples'])
 report['min_available_ram_kib']=min(s['available_ram_kib'] for s in report['samples'])
 report['peak_gpu_utilization_percent']=max(s['gpu_utilization_percent'] for s in report['samples'])
(root/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
sys.exit(process.returncode)
