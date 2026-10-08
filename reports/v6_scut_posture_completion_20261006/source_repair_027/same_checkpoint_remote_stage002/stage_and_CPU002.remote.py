from pathlib import Path
import datetime,hashlib,json,os,subprocess,tarfile,time
source=Path('/home/kaiser/robot-rl-sim60/assessments/evaluator_source_16ed8f6_20261008')
output=Path('/home/kaiser/robot-rl-sim60/assessments/cpu_preflight_39000_16ed8f6_20261008_002')
runtime='/home/kaiser/robot-rl-sim60/env/bin/python'
archive=Path('/home/kaiser/robot-rl-sim60/assessments/source027_evaluator_reports_16ed8f6_20261008.tar.gz')
manifest_path=Path('/home/kaiser/robot-rl-sim60/assessments/source_manifest003.json')
expected_manifest_sha='b7ca06db27e9cd1cfb6c337edf4b5ae88d03521dcb7a83ee9dd1d40860941f0a'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def owners():
 values={}
 for name,pid,start in [('wrapper',3662137,96629148),('supervisor',3662142,96629351),('worker',3662149,96629601),('observer',3928723,106534391)]:
  stat=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
  assert int(stat[19])==start,(name,pid)
  values[name]={'pid':pid,'starttime_ticks':int(stat[19])}
 return values
before=owners()
assert digest(manifest_path)==expected_manifest_sha and not output.exists()
man=json.loads(manifest_path.read_text())
extra=man['dependency_extension']['files']
assert len(extra)==5 and len(man['files'])==848
assert digest(archive)==man['dependency_extension']['archive_sha256']
for name,meta in man['files'].items():
 if name not in extra:assert digest(source/name)==meta['sha256'],name
 else:assert not (source/name).exists(),name
seen=set()
with tarfile.open(archive,'r:gz') as tar:
 for item in tar:
  assert item.isfile() and item.name in extra and item.name not in seen
  p=Path(item.name)
  assert not p.is_absolute() and '..' not in p.parts
  data=tar.extractfile(item).read()
  assert len(data)==extra[item.name]['bytes'] and hashlib.sha256(data).hexdigest()==extra[item.name]['sha256']
  target=source/p;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(data)
  target.chmod(extra[item.name]['mode']);seen.add(item.name)
assert seen==set(extra)
for name,meta in man['files'].items():assert digest(source/name)==meta['sha256'],name
runroot=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
checkpoint=runroot/'train/recovery/attempt_009/train/checkpoints/update_00039000/model.pt'
assert digest(checkpoint)=='3032e8ee2745a07cfe1778ad46397ca6e36318f559ca148fff40161c6691c7e1'
command=[runtime,'-B',str(source/'scripts/evaluate_integrated_checkpoint.py'),'--contract',str(source/'contracts/v6_task_campaign_v25.json'),'--campaign-run-dir',str(runroot/'train'),'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest(checkpoint),'--output',str(output),'--device','cpu','--preflight']
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
child=subprocess.run(command,cwd=source,env=env,capture_output=True,timeout=60)
receipt={'schema':'actual-remote-single-checkpoint-CPU-preflight-v2','started_at_utc':started,'command':command,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout_sha256':hashlib.sha256(child.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(child.stderr).hexdigest(),'source_commit':man['commit'],'source_files_verified':len(man['files']),'source_root':str(source),'output':str(output),'source_manifest_sha256':expected_manifest_sha,'prior_failed_execution_preserved':True,'selected_model_scope':'sealed39000_legacy_REC_research_candidate_not_integrated_allskills','PPO_actions':0,'training_signals':0,'GPU_actions':0,'qualification':None,'owner_identity_before':before,'owner_identity_after':owners()}
assert receipt['owner_identity_after']==before
for name,meta in man['files'].items():assert digest(source/name)==meta['sha256'],name
receipt['source_unchanged_after_CPU']=True
report_path=output/'assessment.json';report=json.loads(report_path.read_text()) if report_path.exists() else None
if child.returncode==0:
 assert report['status']=='cpu_preflight_only' and report['native_scene_started'] is False and report['PPO_updates']==0 and report['qualification'] is None
 assert report['inputs']['selected_checkpoint_scope']=='legacy_research_candidate'
 receipt['status']='staged_frozen_source_and_actual_CPU_preflight_passed'
 receipt['report_sha256']=digest(report_path)
else:receipt['status']='source_staged_CPU_preflight_failed'
print(json.dumps({'receipt':receipt,'report':report,'child_stdout':child.stdout.decode(errors='replace'),'child_stderr':child.stderr.decode(errors='replace')},ensure_ascii=False))
raise SystemExit(child.returncode)
