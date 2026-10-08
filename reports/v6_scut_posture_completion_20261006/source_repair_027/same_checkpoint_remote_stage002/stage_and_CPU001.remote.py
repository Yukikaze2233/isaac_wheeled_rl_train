from pathlib import Path
import datetime,hashlib,json,os,subprocess,tarfile,time
source=Path('/home/kaiser/robot-rl-sim60/assessments/evaluator_source_16ed8f6_20261008')
archive=Path('/home/kaiser/robot-rl-sim60/assessments/source027_evaluator_16ed8f6_20261008.tar.gz')
manifest_path=Path('/home/kaiser/robot-rl-sim60/assessments/source_manifest002.json')
output=Path('/home/kaiser/robot-rl-sim60/assessments/cpu_preflight_39000_16ed8f6_20261008')
runtime='/home/kaiser/robot-rl-sim60/env/bin/python'
expected_archive_sha='fce638440be91384522fa1edd35976269e88c821e6e348a40ca56fc79dbc1911'
expected_manifest_sha='7248c45179d9e36537717dd11eba38b75dec3a9d5709e87647c10f6bd0e0dce8'

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
assert digest(archive)==expected_archive_sha and digest(manifest_path)==expected_manifest_sha
man=json.loads(manifest_path.read_text())
assert man['file_count']==843 and not source.exists() and not output.exists()
source.mkdir()
observed={}
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  path=Path(member.name)
  assert not path.is_absolute() and '..' not in path.parts
  dest=source/path
  if member.isdir():dest.mkdir(parents=True,exist_ok=True);continue
  assert member.isfile() and member.name in man['files'] and member.name not in observed
  dest.parent.mkdir(parents=True,exist_ok=True)
  h=hashlib.sha256();n=0
  with tar.extractfile(member) as original,dest.open('xb') as target:
   for chunk in iter(lambda:original.read(1024*1024),b''):
    target.write(chunk);h.update(chunk);n+=len(chunk)
  expected=man['files'][member.name]
  assert n==expected['bytes'] and h.hexdigest()==expected['sha256'],member.name
  dest.chmod(member.mode&0o777)
  observed[member.name]=h.hexdigest()
assert len(observed)==843 and set(observed)==set(man['files'])
root=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
checkpoint=root/'train/recovery/attempt_009/train/checkpoints/update_00039000/model.pt'
assert digest(checkpoint)=='3032e8ee2745a07cfe1778ad46397ca6e36318f559ca148fff40161c6691c7e1'
assert digest(checkpoint.with_suffix('.contract.json'))=='eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2'
assert digest(checkpoint.with_suffix('.checkpoint.json'))=='a7a345dd54676a79c391b4e96247ae25e88fddb3dbd2d21e6c0a7778ed559ea7'
command=[runtime,'-B',str(source/'scripts/evaluate_integrated_checkpoint.py'),'--contract',str(source/'contracts/v6_task_campaign_v25.json'),'--campaign-run-dir',str(root/'train'),'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest(checkpoint),'--output',str(output),'--device','cpu','--preflight']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
child=subprocess.run(command,env=env,capture_output=True,timeout=60)
receipt={'schema':'actual-remote-single-checkpoint-CPU-preflight-v1','started_at_utc':started,'command':command,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout_sha256':hashlib.sha256(child.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(child.stderr).hexdigest(),'child_environment_overrides':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE']},'source_commit':man['commit'],'archive_sha256':expected_archive_sha,'source_files_verified':len(observed),'source_root':str(source),'output':str(output),'selected_model_scope':'sealed39000_legacy_REC_research_candidate_not_integrated_allskills','PPO_actions':0,'training_signals':0,'GPU_actions':0,'qualification':None}
report_path=output/'assessment.json'
report=json.loads(report_path.read_text()) if report_path.exists() else None
receipt['owner_identity_before']=before;receipt['owner_identity_after']=owners()
assert receipt['owner_identity_after']==before
for name,digest_value in observed.items():assert digest(source/name)==digest_value,name
receipt['source_unchanged_after_CPU']=True
if child.returncode==0:
 assert report['status']=='cpu_preflight_only' and report['native_scene_started'] is False
 assert report['PPO_updates']==0 and report['qualification'] is None
 assert report['inputs']['selected_checkpoint_scope']=='legacy_research_candidate'
 receipt['status']='staged_frozen_source_and_actual_CPU_preflight_passed'
 receipt['report_sha256']=digest(report_path)
else:receipt['status']='source_staged_CPU_preflight_failed'
print(json.dumps({'receipt':receipt,'report':report,'child_stdout':child.stdout.decode(errors='replace'),'child_stderr':child.stderr.decode(errors='replace')},ensure_ascii=False))
raise SystemExit(child.returncode)
