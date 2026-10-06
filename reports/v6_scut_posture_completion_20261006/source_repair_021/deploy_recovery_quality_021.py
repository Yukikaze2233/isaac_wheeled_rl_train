#!/usr/bin/env python3
"""Emit one named v18→v19 continuation; no implicit SSH, signal, or retry."""
import argparse
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
PREVIOUS = "7f315f30741be01d70046a11848443038c14b240"
SESSION = "v6-task-campaign-20261001T110045Z-282358-r21"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


COMMON = r'''
import hashlib,json,pathlib,sys,tarfile
from datetime import datetime,timezone
r=pathlib.Path(RUN);src=pathlib.Path(META['source_remote']);old=r/'source_7f315f30741b'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def entry(a):
 for x in a[1:]:
  if x in ('-c','-m'):return None
  if x.endswith('.py'):return pathlib.Path(x)
def owners():
 result=[]
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:
   a=[x.decode() for x in p.read_bytes().split(b'\0') if x];e=entry(a)
   if e and e.name in ('chassis_remote_job.py','run_task_campaign.py','train_chassis.py','evaluate_chassis.py') and any(x==str(r) or x==str(r/'train') or x.startswith(str(r/'train')+'/') for x in a):result.append(dict(pid=int(p.parent.name),entry=str(e),command=a))
  except OSError:pass
 return result
sys.path[:0]=[str(src/'src'),str(src/'scripts')]
from chassis_remote_job import validate_source_revision
from wheeled_tasks.chassis.task_campaign import account_updates,validate_plan_migration,validate_task_resume,update_attempt
from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count,checkpoint_contract_path
from wheeled_tasks.chassis.recovery_quality_revision import revised_plan,revised_manifest,validate_frozen_workers
def authenticate():
 job=read(r/'job.json');revision=read(r/'source_revision_020.json')
 assert job['command']==revision['command'] and read(r/job['source_revision_receipt'])==revision
 assert revision['schema']=='task-source-revision-v1' and revision['source_commit']==PREVIOUS
 assert pathlib.Path(revision['new_source_root'])==old and sha(pathlib.Path(revision['source_archive']))==revision['source_archive_sha256']
 assert sha(r/job['previous_job'])==revision['previous_job_sha256']
 assert not owners() and not pathlib.Path('/proc/'+str(job['pid'])+'/cmdline').exists(),'Previous owner still live'
 source=read(old/'contracts/v6_task_campaign_v18.json');target=read(src/'contracts/v6_task_campaign_v19.json')
 assert target==revised_plan(src) and read(src/'contracts/v6_task_campaign_migration_v19.json')==revised_manifest(src)
 assert sha(src/'contracts/v6_task_campaign_v19.json')==META['plan_sha256']
 assert sha(src/'contracts/v6_task_campaign_migration_v19.json')==META['manifest_sha256']
 assert validate_frozen_workers(src)['domain']=='recovery'
 oldfiles={str(p.relative_to(old)):sha(p) for p in (old/'contracts').rglob('*') if p.is_file()}
 newfiles={str(p.relative_to(src)):sha(p) for p in (src/'contracts').rglob('*') if p.is_file()}
 assert all(newfiles.get(n)==s for n,s in oldfiles.items()),'Old frozen contracts changed'
 assert set(newfiles)-set(oldfiles)=={'contracts/v6_task_campaign_v19.json','contracts/v6_task_campaign_migration_v19.json','contracts/v6_task_campaign_recovery_worker_v19.json','contracts/v6_task_campaign_recovery_expectations_v19.json'}
 state=read(r/'train/campaign_state.json');account=read(r/'train/consumption.json')
 assert state['status']=='stopped' and state['contract_sha256']==account['contract_sha256']==sha(old/'contracts/v6_task_campaign_v18.json')
 assert account['rollback_refunds_consumed_updates'] is False
 assert account==read(pathlib.Path(source['budget_source_stage'])/(source['campaign_id']+'.account.json'))
 computed=account_updates(source,account['attempts'])
 assert computed=={k:account[k] for k in computed}=={k:state[k] for k in computed}
 assert account_updates(target,account['attempts'])==computed
 replay=account['attempts']
 for p in (r/'train').glob('*/attempt_*/train/progress.json'):
  replay=update_attempt(target,replay,str(p.parent.relative_to(r/'train')),p.parts[-4],read(p))
 assert replay==account['attempts'],'Uncharged final progress'
 validate_plan_migration(source,target,r/'train')
 parents={}
 for row in target['domains']:
  d=row['name'];candidates=[]
  for p in (r/'train'/d).glob('attempt_*/train/**/*.checkpoint.json'):
   if any(part.endswith('.pending') for part in p.parts):continue
   seal=read(p);cp=p.with_suffix('').with_suffix('.pt')
   assert type(seal.get('learning_updates')) is int and seal['learning_updates']>=0
   candidates.append((seal['learning_updates'],cp.name=='model_final.pt',cp))
  if not candidates:assert d=='stepjump';continue
  cp=max(candidates,key=lambda x:x[:2])[2];sealpath=cp.with_suffix('.checkpoint.json');seal=read(sealpath);contract=checkpoint_contract_path(cp)
  assert seal['sealed'] is True and seal['task_campaign']['domain']==d and cp.resolve().is_relative_to((r/'train'/d).resolve())
  assert sha(cp)==seal['checkpoint_sha256'] and sha(contract)==seal['contract_sha256'] and checkpoint_update_count(cp)==seal['learning_updates']
  validate_task_resume(read(contract),read(src/row['contract']))
  parents[d]=dict(path=str(cp),sha256=sha(cp),seal=seal,seal_sha256=sha(sealpath),contract=str(contract),contract_sha256=sha(contract))
 assert set(parents)=={'flat','recovery','speed','terrain','jump'}
 parent=parents['recovery'];cp=pathlib.Path(parent['path'])
 assert cp.name=='model_final.pt' and cp.parent==r/'train/recovery/attempt_004/train'
 completion=read(cp.parent/'completion.json');progress=read(cp.parent/'progress.json')
 assert completion['status']=='stopped' and completion['task_actor_update']==progress['task_actor_update']==parent['seal']['task_actor_update']>2758
 import torch
 from train_chassis import resume_critic_warmup
 payload=torch.load(cp,map_location='cpu',weights_only=False);infos=payload['infos']
 assert all(payload.get(k) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')) and payload['optimizer_state_dict']['state']
 assert all(isinstance(infos.get(k),torch.Tensor) and infos[k].dtype==torch.uint8 and infos[k].numel()>0 for k in ('rng_state','env_rng_state'))
 assert isinstance(infos.get('cuda_rng_states'),list) and infos['cuda_rng_states'] and all(isinstance(v,torch.Tensor) and v.dtype==torch.uint8 and v.numel()>0 for v in infos['cuda_rng_states'])
 for k in ('stage_actor_update','task_actor_update','actor_updates_consumed','learning_lineage_updates'):assert infos[k]==parent['seal'][k]
 assert infos['successful_updates_total']==parent['seal']['learning_updates']
 row=next(x for x in target['domains'] if x['name']=='recovery');worker=read(src/row['contract'])
 assert infos['contract_sha256']==parent['contract_sha256'] and infos['critic_warmup_remaining']==0
 assert resume_critic_warmup(worker,infos,contract_sha256=row['sha256'],task_change=True)==10
 parent['authenticated_infos']={k:infos[k] for k in ('contract_sha256','successful_updates_total','stage_actor_update','task_actor_update','actor_updates_consumed','learning_lineage_updates','critic_warmup_remaining')}
 return job,state,account,parents
'''

PREPARE = r'''
import fcntl,tempfile
with (r/'recovery_quality_021.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'recovery_quality_launch_021.json').exists(),'Already dispatched; inspect'
 job,state,account,parents=authenticate()
 command=list(job['command']);command[command.index(str(old/'scripts/run_task_campaign.py'))]=str(src/'scripts/run_task_campaign.py');command[command.index('--contract')+1]=str(src/'contracts/v6_task_campaign_v19.json')
 revision=dict(schema='task-plan-revision-v19',previous_job_sha256=sha(r/'job.json'),command=command,source_commit=META['source_commit'],old_source_root=str(old),new_source_root=str(src),new_contract_relative_path='contracts/v6_task_campaign_v19.json',old_contract_sha256=sha(old/'contracts/v6_task_campaign_v18.json'),new_contract_sha256=META['plan_sha256'],source_archive=META['archive_remote'],source_archive_sha256=META['archive_sha256'])
 with tempfile.NamedTemporaryFile(mode='w',dir=r,suffix='.pending',delete=False) as f:json.dump(revision,f);pending=pathlib.Path(f.name)
 try:validate_source_revision(r,command,job,pending)
 finally:pending.unlink()
 predecessor=dict(schema='task-recovery-quality-predecessor-021-v1',source_commit=META['source_commit'],previous_source_commit=PREVIOUS,evidence_sha256=EVIDENCE_SHA,job=job,campaign_state=state,consumption=account,actual_parents=parents,expected_critic_warmup=10,reopening=None,qualification=None)
 for name,value in (('predecessor_021.json',predecessor),('source_revision_021.json',revision)):
  p=r/name
  if p.exists():assert read(p)==value
  else:
   with p.open('x') as f:json.dump(value,f,indent=2);f.write(chr(10))
 print(json.dumps(dict(status='prepared_frozen_v19_latest_full_seal_not_started',revision=revision,predecessor=predecessor)))
'''

LAUNCH = r'''
import fcntl,shlex,subprocess
with (r/'recovery_quality_021.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'recovery_quality_launch_021.json').exists() and not (r/'job.resume_021.log').exists(),'Ambiguous launch; inspect, never retry blindly'
 assert read(r/'source_revision_021.json')==REVISION and read(r/'predecessor_021.json')==PREDECESSOR
 job,state,account,parents=authenticate()
 assert (job,state,account,parents)==(PREDECESSOR['job'],PREDECESSOR['campaign_state'],PREDECESSOR['consumption'],PREDECESSOR['actual_parents'])
 assert PREDECESSOR['evidence_sha256']==EVIDENCE_SHA
 validate_source_revision(r,REVISION['command'],job,r/'source_revision_021.json')
 socket=BASE+'/tmux.sock';assert subprocess.run(['tmux','-S',socket,'has-session','-t',SESSION],capture_output=True).returncode!=0
 wrapper=[BASE+'/env/bin/python','-B',str(src/'scripts/chassis_remote_job.py'),'--run-root',str(r),'--resume-existing','--source-revision-receipt',str(r/'source_revision_021.json'),'--',*REVISION['command']]
 inner='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && '+shlex.join(wrapper)+' > '+shlex.quote(str(r/'job.resume_021.log'))+' 2>&1'
 receipt=dict(schema='task-recovery-quality-launch-021-v1',status='launch_admitted_pending_tmux',admitted_at_utc=datetime.now(timezone.utc).isoformat(),source_commit=META['source_commit'],source_root=str(src),run_root=RUN,tmux=SESSION,plan_sha256=META['plan_sha256'],source_archive_sha256=META['archive_sha256'],command=REVISION['command'],wrapper_command=wrapper,evidence_sha256=EVIDENCE_SHA,parent=parents['recovery'],expected_critic_warmup=10,qualification=None)
 p=r/'recovery_quality_launch_021.json'
 with p.open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10))
 q=subprocess.run(['tmux','-S',socket,'new-session','-d','-s',SESSION,'-c',str(src),'bash','-lc',inner],capture_output=True,text=True)
 receipt.update(status='launched_waiting_for_real_restore_critic_adaptation_and_actor_PPO' if q.returncode==0 else 'tmux_launch_failed_inspect_before_retry',tmux_exit_code=q.returncode,tmux_stderr=q.stderr[-2000:],launched_at_utc=datetime.now(timezone.utc).isoformat())
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2)+chr(10));tmp.replace(p)
 print(json.dumps(receipt));assert q.returncode==0,q.stderr
'''


def emit(mode, meta, evidence, revision=None, predecessor=None):
    assert meta['previous_source_commit'] == PREVIOUS
    assert meta['contract_file'] == 'contracts/v6_task_campaign_v19.json'
    assert evidence['status'] == 'passed' and evidence['qualification'] is None
    assert evidence['source_commit'] == meta['source_commit'] and evidence['archive_sha256'] == meta['archive_sha256']
    values = dict(RUN=RUN, BASE=BASE, SESSION=SESSION, PREVIOUS=PREVIOUS, META=meta,
                  EVIDENCE_SHA=evidence['evidence_sha256'])
    if mode == 'launch':
        assert revision['schema'] == 'task-plan-revision-v19'
        assert revision['source_commit'] == predecessor['source_commit'] == meta['source_commit']
        assert predecessor['evidence_sha256'] == values['EVIDENCE_SHA']
        values.update(REVISION=revision, PREDECESSOR=predecessor)
    return '\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+COMMON+{'prepare':PREPARE,'launch':LAUNCH}[mode]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','launch'))
    p.add_argument('--emit-remote-code',action='store_true',required=True)
    p.add_argument('--source',type=Path,default=OUT/'source_candidate_021.json')
    p.add_argument('--admission',type=Path,default=OUT/'admission_021.json')
    args=p.parse_args()
    print(emit(args.mode,read(args.source),read(args.admission),read(OUT/'source_revision_021.json') if args.mode=='launch' else None,read(OUT/'predecessor_021.json') if args.mode=='launch' else None))


if __name__=='__main__':main()
