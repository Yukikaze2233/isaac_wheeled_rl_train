from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,shlex,subprocess,sys
OUT=Path(__file__).parent
meta=json.loads((OUT/'source_candidate_013b.json').read_text())
BASE='/home/kaiser/robot-rl-sim60'
RUN=BASE+'/experiments/v6-task-campaign-20261001T110045Z-282358'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','2222','kaiser@192.168.64.234']
code="""
import json,hashlib,pathlib,tarfile,sys
r=pathlib.Path(RUN);src=pathlib.Path(META['source_remote']);archive=pathlib.Path(META['archive_remote'])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)==META['archive_sha256']
if not src.exists():
 src.mkdir()
 with tarfile.open(archive) as t:
  for m in t:
   p=pathlib.Path(m.name)
   assert not p.is_absolute() and '..' not in p.parts and (m.isfile() or m.isdir())
   t.extract(m,src,filter='data')
j=json.loads((r/'job.json').read_text());oldcmd=j['command'];old=pathlib.Path(oldcmd[oldcmd.index('--contract')+1]).parent.parent
assert old.name=='source_2b3734aeb37a',old
command=list(oldcmd);command[command.index(str(old/'scripts/run_task_campaign.py'))]=str(src/'scripts/run_task_campaign.py')
command[command.index('--contract')+1]=str(src/'contracts/v6_task_campaign_v13.json')
receipt=dict(schema='task-plan-revision-v13',previous_job_sha256=sha(r/'job.json'),command=command,
source_commit=META['source_commit'],old_source_root=str(old),new_source_root=str(src),new_contract_relative_path='contracts/v6_task_campaign_v13.json',
old_contract_sha256=sha(old/'contracts/v6_task_campaign_v12.json'),new_contract_sha256=sha(src/'contracts/v6_task_campaign_v13.json'),
source_archive=str(archive),source_archive_sha256=sha(archive))
if (r/'source_revision_013b.json').exists():
 assert json.loads((r/'source_revision_013b.json').read_text())==receipt
else:
 with (r/'source_revision_013b.json').open('x') as f: json.dump(receipt,f,indent=2);f.write(chr(10))
sys.path.insert(0,str(src/'scripts'));from chassis_remote_job import validate_source_revision
validated=validate_source_revision(r,command,j,r/'source_revision_013b.json')
print(json.dumps(dict(status='passed',archive_integrity='all_files_passed',receipt=validated)))
"""
code='RUN='+repr(RUN)+'\nMETA='+repr(meta)+'\n'+code
prep='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && '+shlex.join([BASE+'/env/bin/python','-B','-c',code])
raw=subprocess.check_output(SSH+[shlex.join(['bash','-lc',prep])],text=True)
receipt=json.loads(raw)
(OUT/'source_revision_013b.json').write_text(json.dumps(receipt['receipt'],indent=2)+'\n')
src=meta['source_remote']
command=[BASE+'/env/bin/python','-B',src+'/scripts/run_task_campaign.py','--contract',src+'/contracts/v6_task_campaign_v13.json','--num-envs','4096','--device','cuda:0','--research','--run-dir',RUN+'/train','--preflight-only']
inner='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export OPENBLAS_NUM_THREADS=1 && '+shlex.join(command)
r=subprocess.run(SSH+[shlex.join(['bash','-lc',inner])],capture_output=True,text=True)
(OUT/'preflight_013b.stdout').write_text(r.stdout);(OUT/'preflight_013b.stderr').write_text(r.stderr)
assert r.returncode==0,r.stderr[-3000:]
receipt.update(roles=30,verified_at_utc=datetime.now(timezone.utc).isoformat(),stdout_tail=r.stdout[-1500:])
(OUT/'predeploy_013b.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
