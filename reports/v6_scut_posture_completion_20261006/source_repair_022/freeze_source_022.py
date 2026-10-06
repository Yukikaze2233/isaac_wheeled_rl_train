"""Freeze exactly the reviewed source021 overlay, without checkout/index mutation."""
import hashlib,json,os,subprocess,tempfile,tarfile
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
b=root/'reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/contract_script_revision_candidate_022'
m=json.loads((b/'manifest_022.json').read_text());candidate=Path(m['candidate_root']);previous=m['base_commit'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert m['status']=='CPU_candidate_ready_native_not_run'
files=m['overlay_files'];pins=m['source_files_sha256']|m['contracts_sha256']|{'tests/test_recovery_script_revision.py':m['test_sha256']}
assert set(files)==set(pins) and all(sha(candidate/n)==pins[n] for n in files)
assert not (out/'source_candidate_022.json').exists(),'Already frozen; inspect existing metadata'
fd,index=tempfile.mkstemp(prefix='source022-index-');os.close(fd);os.unlink(index);env=dict(os.environ,GIT_INDEX_FILE=index)
def git(*args,input=None):return subprocess.run(['git',*args],input=input,text=True,capture_output=True,check=True,env=env).stdout.strip()
try:
 git('read-tree',previous)
 for n in files:
  old=git('ls-tree',previous,'--',n);mode=old.split()[0] if old else '100644';obj=git('hash-object','-w',str(candidate/n));git('update-index','--add','--cacheinfo',mode,obj,n)
 tree=git('write-tree')
 check=subprocess.run(['git','rev-parse','--verify','refs/codex/frozen-source-022'],text=True,capture_output=True)
 if check.returncode==0:
  commit=check.stdout.strip();assert git('rev-parse',commit+'^{tree}')==tree and git('rev-parse',commit+'^')==previous
 else:
  commit=git('commit-tree',tree,'-p',previous,input='training: slow SIDE and gate CAPTURE on current angular rate\n');git('update-ref','refs/codex/frozen-source-022',commit,'0'*40)
finally:
 Path(index).unlink(missing_ok=True)
archive=out/f'source_{commit[:12]}.tar.gz'
if not archive.exists():
 with archive.open('xb') as f:subprocess.run(['git','archive','--format=tar.gz',commit],stdout=f,check=True)
clean=Path(tempfile.mkdtemp(prefix='scut-script-frozen-022-'))
with tarfile.open(archive,'r:gz') as t:
 assert t.pax_headers['comment']==commit
 for member in t:
  rel=Path(member.name);assert not rel.is_absolute() and '..' not in rel.parts and (member.isfile() or member.isdir())
  t.extract(member,clean,filter='data')
oldinv=json.loads((out.parent/'source_repair_021/frozen_source_inventory_021.json').read_text());newinv=dict(oldinv);newinv.update(pins)
assert all(sha(clean/n)==h for n,h in newinv.items())
inv=out/'frozen_source_inventory_022.json';inv.write_text(json.dumps(newinv,indent=2)+'\n')
run='/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358'
meta={'schema':'task-recovery-script-022-source-candidate-v1','status':'frozen_CPU_ready_pending_native_not_deployed','source_commit':commit,'previous_source_commit':previous,'archive_local':str(archive),'archive_sha256':sha(archive),'archive_size_bytes':archive.stat().st_size,'clean_root':str(clean),'source_remote':run+'/source_'+commit[:12],'archive_remote':run+'/source_'+commit[:12]+'.tar.gz','contract_file':'contracts/v6_task_campaign_v20.json','plan_sha256':pins['contracts/v6_task_campaign_v20.json'],'manifest_sha256':pins['contracts/v6_task_campaign_migration_v20.json'],'changed_runtime_files':{n:{'previous':oldinv.get(n),'candidate':h} for n,h in pins.items()},'source_inventory_file':str(inv),'source_inventory_sha256':sha(inv),'candidate_manifest_sha256':sha(b/'manifest_022.json'),'qualification':None,'formal_updates_consumed':0,'source_branch_scope':'Detached immutable source021 child; main checkout, dirty files, and live remote not mutated.'}
(out/'source_candidate_022.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'source_commit':commit,'archive_sha256':meta['archive_sha256'],'archive_bytes':meta['archive_size_bytes'],'overlay_files':len(files),'main_checkout_untouched':True}))
