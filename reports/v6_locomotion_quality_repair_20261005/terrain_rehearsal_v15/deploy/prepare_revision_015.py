"""Bind one frozen v15 source migration after the former owner has sealed."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-p", "2222", "kaiser@192.168.64.234"]


def main():
    meta = json.loads((OUT / "source_candidate_015.json").read_text())
    preflight = json.loads((OUT / "remote_clean_30_roles_015.json").read_text())
    native = json.loads((OUT.parent / "verification/native_population_015.json").read_text())
    tests = json.loads((OUT.parent / "verification/clean_archive_tests.json").read_text())
    assert preflight["status"] == native["status"] == tests["status"] == "passed"
    assert preflight["source_commit"] == tests["source_commit"] == meta["source_commit"]
    code = "RUN=" + repr(RUN) + "\nMETA=" + repr(meta) + "\n" + r'''
import hashlib,json,pathlib,sys
from datetime import datetime,timezone
r=pathlib.Path(RUN);src=pathlib.Path(META['source_remote'])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
job=read(r/'job.json');assert not pathlib.Path('/proc/'+str(job['pid'])+'/cmdline').exists(),'Previous owner still live'
oldcmd=job['command'];old=pathlib.Path(oldcmd[oldcmd.index('--contract')+1]).parent.parent
assert old.name=='source_b4ce80964d47'
owned=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=p.read_bytes().split(b'\0');entry=next((x for x in argv[1:] if x.endswith(b'.py')),None)
  if entry in [str(old/'scripts'/x).encode() for x in ('run_task_campaign.py','train_chassis.py')]:owned.append(str(p))
 except (OSError,PermissionError):pass
assert not owned,owned
state=read(r/'train/campaign_state.json');account=read(r/'train/consumption.json')
assert state['contract_sha256']==account['contract_sha256']==sha(old/'contracts/v6_task_campaign_v14.json')
speed_parent=pathlib.Path(state['domains']['speed']['candidate_checkpoint'])
assert read(speed_parent.with_suffix('.checkpoint.json'))['task_actor_update']>=5063,'Speed has not reached its existing natural endpoint'
sys.path.insert(0,str(src/'src'))
from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count,checkpoint_contract_path
from wheeled_tasks.chassis.task_campaign import validate_task_resume
paths=[]
for p in (r/'train/terrain').glob('attempt_*/train/**/*.checkpoint.json'):
 if any(x.endswith('.pending') for x in p.parts):continue
 s=read(p);cp=p.with_suffix('').with_suffix('.pt')
 paths.append((s['learning_updates'],cp.name=='model_final.pt',cp,s))
_,_,parent,seal=max(paths,key=lambda x:(x[0],x[1]))
assert checkpoint_update_count(parent)==seal['learning_updates']
assert seal['task_campaign']['domain']=='terrain' and 1186<=seal['task_actor_update']<2186
parent_contract=checkpoint_contract_path(parent);target=src/'contracts/v6_task_campaign_terrain_worker_v15.json'
validate_task_resume(read(parent_contract),read(target))
before=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),job=job,campaign_state=state,consumption=account,
 actual_terrain_parent=dict(path=str(parent),sha256=sha(parent),seal=seal,contract=str(parent_contract),contract_sha256=sha(parent_contract)),
 source_commit=META['source_commit'],scope='sealed_actual_own_domain_parent_not_flat_or_speed')
proof=r/'terrain_rehearsal_predecessor_015.json'
if proof.exists():assert read(proof)['actual_terrain_parent']==before['actual_terrain_parent']
else:
 with proof.open('x') as f:json.dump(before,f,indent=2);f.write(chr(10))
command=list(oldcmd);command[command.index(str(old/'scripts/run_task_campaign.py'))]=str(src/'scripts/run_task_campaign.py')
command[command.index('--contract')+1]=str(src/'contracts/v6_task_campaign_v15.json')
receipt=dict(schema='task-plan-revision-v15',previous_job_sha256=sha(r/'job.json'),command=command,
 source_commit=META['source_commit'],old_source_root=str(old),new_source_root=str(src),
 new_contract_relative_path='contracts/v6_task_campaign_v15.json',old_contract_sha256=sha(old/'contracts/v6_task_campaign_v14.json'),
 new_contract_sha256=sha(src/'contracts/v6_task_campaign_v15.json'),source_archive=META['archive_remote'],source_archive_sha256=META['archive_sha256'])
path=r/'source_revision_015.json'
if path.exists():assert read(path)==receipt
else:
 with path.open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10))
sys.path.insert(0,str(src/'scripts'))
from chassis_remote_job import validate_source_revision
validate_source_revision(r,command,job,path)
print(json.dumps(dict(status='passed',revision=receipt,predecessor=read(proof))))
'''
    command = "source " + shlex.quote(BASE + "/bin/sim60-runtime.sh") + " && export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && " + shlex.join([BASE + "/env/bin/python", "-B", "-"])
    result = subprocess.run(SSH + [shlex.join(["bash", "-lc", command])], input=code,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]
    proof = json.loads(result.stdout)
    (OUT / "source_revision_015.json").write_text(json.dumps(proof["revision"], indent=2) + "\n")
    (OUT / "predecessor_state_015.json").write_text(json.dumps(proof["predecessor"], indent=2) + "\n")
    proof.update(verified_at_utc=datetime.now(timezone.utc).isoformat(), status="frozen_source_and_sealed_parent_authenticated_not_started")
    (OUT / "predeploy_015.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(status=proof["status"], parent=proof["predecessor"]["actual_terrain_parent"], consumed=proof["predecessor"]["consumption"]["consumed_updates"])))


if __name__ == "__main__":
    main()
