#!/usr/bin/env python3
"""Launch one native r18 campaign owner; no gate or evaluator is spawned."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

from prepare_revision_018 import BASE, OUT, RUN, SSH, load_evidence, read

SESSION = "v6-task-campaign-20261001T110045Z-282358-r18"

REMOTE = r'''
import fcntl,hashlib,json,pathlib,shlex,subprocess,sys
from datetime import datetime,timezone
r=pathlib.Path(RUN);src=pathlib.Path(REVISION['new_source_root'])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(argv):
 for value in argv[1:]:
  if value in (b'-c',b'-m'):return None
  if value.endswith(b'.py'):return pathlib.Path(value.decode())
 return None
with (r/'launch_018.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'scut_posture_launch_018.json').exists(),'Remote launch receipt already exists; inspect instead of launching twice'
 assert not (r/'job.resume_018.log').exists(),'Launch log already exists; inspect ambiguous prior launch'
 assert read(r/'source_revision_018.json')==REVISION
 job=read(r/'job.json')
 assert not pathlib.Path('/proc/'+str(job['pid'])+'/cmdline').exists(),'Former owner still live'
 assert sha(r/'job.json')==REVISION['previous_job_sha256']
 owners=[]
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=p.read_bytes().split(b'\0');script=entry(argv)
   if script and script.name in ('run_task_campaign.py','train_chassis.py','evaluate_chassis.py','chassis_remote_job.py') and any(x==str(r).encode() or x==str(r/'train').encode() or x.startswith(str(r/'train').encode()+b'/') for x in argv):owners.append(str(p))
  except (OSError,PermissionError):pass
 assert not owners,owners
 predecessor=read(r/'scut_posture_predecessor_018.json')
 assert predecessor==PREDECESSOR
 assert read(r/'train/consumption.json')==predecessor['consumption']
 assert read(r/'train/campaign_state.json')==predecessor['campaign_state']
 assert job==predecessor['job']
 assert set(predecessor['actual_parents'])=={'recovery','speed','terrain','jump'}
 sys.path.insert(0,str(src/'scripts'));sys.path.insert(0,str(src/'src'))
 from chassis_remote_job import validate_source_revision
 from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count,checkpoint_contract_path
 validate_source_revision(r,REVISION['command'],job,r/'source_revision_018.json')
 for domain,bound in predecessor['actual_parents'].items():
  cp=pathlib.Path(bound['path']);seal=read(cp.with_suffix('.checkpoint.json'))
  assert cp.resolve().is_relative_to(r/'train'/domain) and seal['sealed'] is True and seal['task_campaign']['domain']==domain
  assert seal==bound['seal'] and sha(cp)==bound['sha256']==seal['checkpoint_sha256'] and sha(checkpoint_contract_path(cp))==bound['contract_sha256']==seal['contract_sha256']
  assert checkpoint_update_count(cp)==seal['learning_updates']
  latest=[]
  for path in (r/'train'/domain).glob('attempt_*/train/**/*.checkpoint.json'):
   if any(part.endswith('.pending') for part in path.parts):continue
   meta=read(path);model=path.with_suffix('').with_suffix('.pt')
   assert type(meta.get('learning_updates')) is int and meta['learning_updates']>=0
   latest.append((meta['learning_updates'],model.name=='model_final.pt',model))
  assert latest and max(latest,key=lambda row:row[:2])[2]==cp,'Prepared parent is no longer latest own seal'
 from copy import deepcopy
 from wheeled_tasks.chassis.ordinary_posture_revision import reopen_completed_domains,ORIGINS,validate_frozen_workers
 old=pathlib.Path(REVISION['old_source_root']);source=read(old/'contracts/v6_task_campaign_v15.json');target=read(src/'contracts/v6_task_campaign_v18.json')
 assert len(validate_frozen_workers(src))==5
 assert reopen_completed_domains(deepcopy(predecessor['campaign_state']),deepcopy(predecessor['consumption']['attempts']),source,target,r/'train')==predecessor['authenticated_reopening']
 socket=BASE+'/tmux.sock'
 assert subprocess.run(['tmux','-S',socket,'has-session','-t',SESSION],capture_output=True).returncode!=0,'r18 tmux already exists'
 wrapper=[BASE+'/env/bin/python','-B',str(src/'scripts/chassis_remote_job.py'),'--run-root',str(r),
  '--resume-existing','--source-revision-receipt',str(r/'source_revision_018.json'),'--',*REVISION['command']]
 inner='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && '+shlex.join(wrapper)+' > '+shlex.quote(str(r/'job.resume_018.log'))+' 2>&1'
 receipt=dict(status='launch_admitted_pending_tmux',admitted_at_utc=datetime.now(timezone.utc).isoformat(),
  tmux=SESSION,source_commit=REVISION['source_commit'],source_root=str(src),run_root=str(r),
  plan_sha256=REVISION['new_contract_sha256'],source_archive_sha256=REVISION['source_archive_sha256'],
  command=REVISION['command'],wrapper_command=wrapper,source_revision_receipt=str(r/'source_revision_018.json'),
  evidence_sha256=EVIDENCE_SHA,qualification=None,scope='continuous_independent_tasks_no_online_gate_no_evaluator')
 path=r/'scut_posture_launch_018.json'
 with path.open('x') as output:json.dump(receipt,output,indent=2);output.write(chr(10))
 result=subprocess.run(['tmux','-S',socket,'new-session','-d','-s',SESSION,'-c',str(src),'bash','-lc',inner],capture_output=True,text=True)
 receipt.update(status='launched_waiting_for_native_full_state_restore_and_real_actor_PPO' if result.returncode==0 else 'tmux_launch_failed_inspect_before_retry',
  launched_at_utc=datetime.now(timezone.utc).isoformat(),tmux_exit_code=result.returncode,tmux_stderr=result.stderr[-2000:])
 temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2)+chr(10));temporary.replace(path)
 print(json.dumps(receipt))
 assert result.returncode==0,result.stderr
'''


def remote_code(evidence, revision, predecessor):
    assert evidence["source"]["source_commit"] == revision["source_commit"] == predecessor["source_commit"]
    assert evidence["source"]["archive_sha256"] == revision["source_archive_sha256"]
    assert evidence["source"]["plan_sha256"] == revision["new_contract_sha256"]
    assert predecessor["evidence_sha256"] == evidence["evidence_sha256"]
    values = dict(RUN=RUN, BASE=BASE, SESSION=SESSION, REVISION=revision,
                  PREDECESSOR=predecessor, EVIDENCE_SHA=evidence["evidence_sha256"])
    return "\n".join(name + "=" + repr(value) for name, value in values.items()) + "\n" + REMOTE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-remote-code", action="store_true", help="Print only; do not connect or launch")
    args = parser.parse_args()
    local_receipt = OUT / "scut_posture_launch_018.json"
    if local_receipt.exists() and not args.emit_remote_code:
        raise FileExistsError("Launch receipt already exists; inspect actual owner instead of repeating")
    evidence = load_evidence()
    code = remote_code(evidence, read(OUT / "source_revision_018.json"), read(OUT / "predecessor_state_018.json"))
    if args.emit_remote_code:
        print(code)
        return
    inner = "source " + shlex.quote(BASE + "/bin/sim60-runtime.sh") + " && export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && " + shlex.join([BASE + "/env/bin/python", "-B", "-"])
    result = subprocess.run(SSH + [shlex.join(["bash", "-lc", inner])], input=code, text=True,
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr[-3000:]
    receipt = json.loads(result.stdout)
    receipt["local_received_at_utc"] = datetime.now(timezone.utc).isoformat()
    with local_receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2); stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
