#!/usr/bin/env python3
"""Emit one named v20→v21 continuation; no implicit SSH, signal, or retry."""
import argparse
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
PREVIOUS = "723732adae37f12f8488f9537717664415a98d10"
SESSION = "v6-task-campaign-20261001T110045Z-282358-r23"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


COMMON = r'''
import math
def require(condition,message):
 if not condition:raise ValueError(message)
def finite(value):
 import torch
 if torch.is_tensor(value):return bool(torch.isfinite(value).all())
 if isinstance(value,dict):return all(finite(x) for x in value.values())
 if isinstance(value,(list,tuple)):return all(finite(x) for x in value)
 return not isinstance(value,float) or math.isfinite(value)

def full_learning_state_check(saved):
    """Require the complete native 9 actor, 8 critic and 17 Adam parameter state."""
    from collections import Counter
    import torch
    actor=saved.get('actor_state_dict', {}); critic=saved.get('critic_state_dict', {})
    optimizer=saved.get('optimizer_state_dict', {})
    require(isinstance(actor,dict) and len(actor)==9, 'Expected all 9 actor/std tensors')
    require(isinstance(critic,dict) and len(critic)==8, 'Expected all 8 critic tensors')
    tensors=list(actor.values())+list(critic.values())
    require(all(torch.is_tensor(x) and x.is_floating_point() and x.numel()>0 and bool(torch.isfinite(x).all()) for x in tensors),
            'Invalid/nonfinite actor or critic tensor')
    stdkeys=[key for key in actor if key == 'distribution.std_param']
    require(len(stdkeys)==1 and actor[stdkeys[0]].shape==(6,) and bool((actor[stdkeys[0]]>0).all()), 'Missing/invalid 6-axis actor std')
    groups=optimizer.get('param_groups', []); states=optimizer.get('state', {})
    require(isinstance(groups,list) and groups and isinstance(states,dict), 'Missing Adam group/state')
    params=[value for group in groups for value in group.get('params', [])]
    require(len(params)==17 and all(type(x) is int and x>=0 for x in params) and len(set(params))==17 and set(states)==set(params),
            'Adam does not cover exactly the unique 17 native parameters')
    require(finite(optimizer), 'Nonfinite Adam state/group')
    moments=[]
    for param in params:
        state=states[param]
        require(isinstance(state,dict) and {'step','exp_avg','exp_avg_sq'} <= set(state), 'Incomplete Adam moments/step')
        mean,square,step=state['exp_avg'],state['exp_avg_sq'],state['step']
        require(torch.is_tensor(mean) and torch.is_tensor(square) and mean.is_floating_point() and
                mean.shape==square.shape and mean.dtype==square.dtype and mean.numel()>0 and
                bool(torch.isfinite(mean).all()) and bool(torch.isfinite(square).all()) and bool((square>=0).all()),
                'Invalid Adam moment tensors')
        require((torch.is_tensor(step) and step.numel()==1 and bool(torch.isfinite(step).all()) and step.item()>=0) or
                (type(step) in (int,float) and math.isfinite(step) and step>=0), 'Invalid Adam step')
        moments.append(mean)
    require(Counter((tuple(x.shape),str(x.dtype)) for x in moments)==Counter((tuple(x.shape),str(x.dtype)) for x in tensors),
            'Adam moment shapes/dtypes do not cover actor and critic parameters')
    return {'actor_tensors':9,'critic_tensors':8,'std_axes':6,'Adam_parameter_states':17,'all_learning_tensors_finite':True}

import hashlib,json,pathlib,sys,tarfile
from datetime import datetime,timezone
r=pathlib.Path(RUN);src=pathlib.Path(META['source_remote']);old=r/'source_723732adae37'
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
from wheeled_tasks.chassis.recovery_entropy_revision import revised_plan,revised_manifest,validate_frozen_workers
def authenticate():
 job=read(r/'job.json');revision=read(r/'source_revision_022.json')
 assert job['command']==revision['command']
 assert job.get('source_revision_receipt') is None,'Expected current attempt008 same-contract wrapper'
 resumed=read(r/'same_contract_recovery_20261007_002.json')
 assert resumed['schema']=='source022-same-contract-failed-physical-recovery-v1' and resumed['qualification'] is None
 assert resumed['source_commit']==PREVIOUS and resumed['plan_sha256']==sha(old/'contracts/v6_task_campaign_v20.json')
 assert resumed['wrapper_command'][resumed['wrapper_command'].index('--')+1:]==job['command']
 assert resumed['new_critic_warmup']==0
 assert resumed['session'].endswith('-r22-recovery002')
 previous_invocation=read(r/'train/recovery/attempt_008/invocation.json');previous_command=previous_invocation['command']
 assert previous_invocation['source_checkpoint']==resumed['parent_checkpoint'] and previous_invocation['source_checkpoint_sha256']==resumed['parent_checkpoint_sha256']
 assert previous_command[previous_command.index('--resume')+1]==resumed['parent_checkpoint']
 assert int(previous_command[previous_command.index('--consumed-updates')+1])==resumed['parent_consumption']
 assert int(previous_command[previous_command.index('--actor-updates-consumed')+1])==resumed['parent_actor_consumption']
 assert int(previous_command[previous_command.index('--task-actor-update')+1])==resumed['parent_task_actor_update']
 assert sha(pathlib.Path(resumed['runtime_shell']))==resumed['runtime_shell_sha256']=='c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7'
 assert revision['schema']=='task-plan-revision-v20' and revision['source_commit']==PREVIOUS
 assert pathlib.Path(revision['new_source_root'])==old and sha(pathlib.Path(revision['source_archive']))==revision['source_archive_sha256']
 assert (r/job['previous_job']).is_file(),'Current wrapper previous job missing'
 assert sha(pathlib.Path(META['archive_remote']))==META['archive_sha256']
 assert pathlib.Path(META['archive_remote']).stat().st_size==META['archive_size_bytes']
 with tarfile.open(META['archive_remote'],'r:gz') as tar:
  assert tar.pax_headers.get('comment')==META['source_commit']
  names=set()
  for member in tar:
   rel=pathlib.Path(member.name)
   assert not rel.is_absolute() and '..' not in rel.parts and (member.isfile() or member.isdir())
   if member.isdir():continue
   assert member.name not in names;names.add(member.name)
   assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==sha(src/member.name),member.name
 assert not owners() and not pathlib.Path('/proc/'+str(job['pid'])+'/cmdline').exists(),'Previous owner still live'
 source=read(old/'contracts/v6_task_campaign_v20.json');target=read(src/'contracts/v6_task_campaign_v21.json')
 assert target==revised_plan(src) and read(src/'contracts/v6_task_campaign_migration_v21.json')==revised_manifest(src)
 assert sha(src/'contracts/v6_task_campaign_v21.json')==META['plan_sha256']
 assert sha(src/'contracts/v6_task_campaign_migration_v21.json')==META['manifest_sha256']
 assert validate_frozen_workers(src)['domain']=='recovery'
 oldfiles={str(p.relative_to(old)):sha(p) for p in (old/'contracts').rglob('*') if p.is_file()}
 newfiles={str(p.relative_to(src)):sha(p) for p in (src/'contracts').rglob('*') if p.is_file()}
 assert all(newfiles.get(n)==s for n,s in oldfiles.items()),'Old frozen contracts changed'
 assert set(newfiles)-set(oldfiles)=={'contracts/v6_task_campaign_v21.json','contracts/v6_task_campaign_migration_v21.json','contracts/v6_task_campaign_recovery_worker_v21.json','contracts/v6_task_campaign_recovery_expectations_v21.json'}
 state=read(r/'train/campaign_state.json');account=read(r/'train/consumption.json')
 assert state['status']=='stopped' and state['contract_sha256']==account['contract_sha256']==sha(old/'contracts/v6_task_campaign_v20.json')
 assert account['rollback_refunds_consumed_updates'] is False
 assert account==read(pathlib.Path(source['budget_source_stage'])/(source['campaign_id']+'.account.json'))
 computed=account_updates(source,account['attempts'])
 assert computed=={k:account[k] for k in computed}=={k:state[k] for k in computed}
 assert account_updates(target,account['attempts'])==computed
 replay=account['attempts']
 for p in (r/'train').glob('*/attempt_*/train/progress.json'):
  replay=update_attempt(target,replay,str(p.parent.relative_to(r/'train')),p.parts[-4],read(p))
 assert replay==account['attempts'],'Uncharged final progress'
 assert read(r/'archiver_status.json')['status']=='finished','Old wrapper archival not finished'
 delivery=read(r/'delivery.json');archive=r/delivery['archive']
 assert delivery['status']=='ready' and sha(archive)==delivery['archive_sha256']
 assert archive.stat().st_size==delivery['archive_bytes']
 for name in ('train/campaign_state.json','train/consumption.json','train/recovery/attempt_008/train/model_final.pt','train/recovery/attempt_008/train/model_final.checkpoint.json','train/recovery/attempt_008/train/progress.json','train/recovery/attempt_008/train/completion.json'):
  assert name in delivery['files'] and sha(r/name)==delivery['files'][name]['sha256'],'Final delivery missing/current bytes differ: '+name
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
 assert cp.name=='model_final.pt' and cp.parent==r/'train/recovery/attempt_008/train'
 completion=read(cp.parent/'completion.json');progress=read(cp.parent/'progress.json')
 assert completion['status']=='stopped' and completion['task_actor_update']==progress['task_actor_update']==parent['seal']['task_actor_update']
 import torch
 from train_chassis import resume_critic_warmup
 payload=torch.load(cp,map_location='cpu',weights_only=False);infos=payload['infos']
 parent['full_learning_state_integrity']=full_learning_state_check(payload)
 assert all(payload.get(k) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')) and payload['optimizer_state_dict']['state']
 assert all(isinstance(infos.get(k),torch.Tensor) and infos[k].dtype==torch.uint8 and infos[k].numel()>0 for k in ('rng_state','env_rng_state'))
 assert isinstance(infos.get('cuda_rng_states'),list) and infos['cuda_rng_states'] and all(isinstance(v,torch.Tensor) and v.dtype==torch.uint8 and v.numel()>0 for v in infos['cuda_rng_states'])
 for k in ('stage_actor_update','task_actor_update','actor_updates_consumed','learning_lineage_updates'):assert infos[k]==parent['seal'][k]
 assert infos['successful_updates_total']==parent['seal']['learning_updates']
 row=next(x for x in target['domains'] if x['name']=='recovery');worker=read(src/row['contract'])
 assert worker.get('entropy_coef')==0.0 and type(worker['entropy_coef']) in (int,float)
 assert infos['contract_sha256']==parent['contract_sha256'] and infos['critic_warmup_remaining']==0
 assert parent['seal']['learning_updates']==account['consumed_updates'],'Latest final seal does not include stopped consumption'
 assert parent['seal']['actor_updates_consumed']==account['actor_updates_consumed']
 assert completion['successful_updates']==progress['successful_updates']==account['attempts']['recovery/attempt_008/train']['consumed_updates']
 assert target['total_ceiling']==source['total_ceiling']==121000
 assert all(a==b for a,b in zip(source['domains'],target['domains']) if a['name']!='recovery')
 assert all(a['parent_domain']==b['parent_domain'] for a,b in zip(source['domains'],target['domains']))
 assert resume_critic_warmup(worker,infos,contract_sha256=row['sha256'],task_change=True)==10
 parent['authenticated_infos']={k:infos[k] for k in ('contract_sha256','successful_updates_total','stage_actor_update','task_actor_update','actor_updates_consumed','learning_lineage_updates','critic_warmup_remaining')}
 return job,state,account,parents
'''

PREPARE = r'''
import fcntl,tempfile
with (r/'recovery_entropy_023.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'recovery_entropy_launch_023.json').exists(),'Already dispatched; inspect'
 job,state,account,parents=authenticate()
 command=list(job['command']);command[command.index(str(old/'scripts/run_task_campaign.py'))]=str(src/'scripts/run_task_campaign.py');command[command.index('--contract')+1]=str(src/'contracts/v6_task_campaign_v21.json')
 revision=dict(schema='task-plan-revision-v21',previous_job_sha256=sha(r/'job.json'),command=command,source_commit=META['source_commit'],old_source_root=str(old),new_source_root=str(src),new_contract_relative_path='contracts/v6_task_campaign_v21.json',old_contract_sha256=sha(old/'contracts/v6_task_campaign_v20.json'),new_contract_sha256=META['plan_sha256'],source_archive=META['archive_remote'],source_archive_sha256=META['archive_sha256'])
 with tempfile.NamedTemporaryFile(mode='w',dir=r,suffix='.pending',delete=False) as f:json.dump(revision,f);pending=pathlib.Path(f.name)
 try:validate_source_revision(r,command,job,pending)
 finally:pending.unlink()
 predecessor=dict(schema='task-recovery-entropy-predecessor-023-v1',source_commit=META['source_commit'],previous_source_commit=PREVIOUS,evidence_sha256=EVIDENCE_SHA,job=job,campaign_state=state,consumption=account,actual_parents=parents,expected_critic_warmup=10,expected_entropy_coef=0.0,reopening=None,qualification=None)
 for name,value in (('predecessor_023.json',predecessor),('source_revision_023.json',revision)):
  p=r/name
  if p.exists():assert read(p)==value
  else:
   with p.open('x') as f:json.dump(value,f,indent=2);f.write(chr(10))
 print(json.dumps(dict(status='prepared_frozen_v21_latest_full_seal_not_started',revision=revision,predecessor=predecessor)))
'''

LAUNCH = r'''
import fcntl,shlex,subprocess
with (r/'recovery_entropy_023.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'recovery_entropy_launch_023.json').exists() and not (r/'job.resume_023.log').exists(),'Ambiguous launch; inspect, never retry blindly'
 assert read(r/'source_revision_023.json')==REVISION and read(r/'predecessor_023.json')==PREDECESSOR
 job,state,account,parents=authenticate()
 assert (job,state,account,parents)==(PREDECESSOR['job'],PREDECESSOR['campaign_state'],PREDECESSOR['consumption'],PREDECESSOR['actual_parents'])
 assert PREDECESSOR['evidence_sha256']==EVIDENCE_SHA
 validate_source_revision(r,REVISION['command'],job,r/'source_revision_023.json')
 socket=BASE+'/tmux.sock';assert subprocess.run(['tmux','-S',socket,'has-session','-t',SESSION],capture_output=True).returncode!=0
 wrapper=[BASE+'/env/bin/python','-B',str(src/'scripts/chassis_remote_job.py'),'--run-root',str(r),'--resume-existing','--source-revision-receipt',str(r/'source_revision_023.json'),'--',*REVISION['command']]
 inner='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && '+shlex.join(wrapper)+' > '+shlex.quote(str(r/'job.resume_023.log'))+' 2>&1'
 receipt=dict(schema='task-recovery-entropy-launch-023-v1',status='launch_admitted_pending_tmux',admitted_at_utc=datetime.now(timezone.utc).isoformat(),source_commit=META['source_commit'],source_root=str(src),run_root=RUN,tmux=SESSION,plan_sha256=META['plan_sha256'],source_archive_sha256=META['archive_sha256'],command=REVISION['command'],wrapper_command=wrapper,evidence_sha256=EVIDENCE_SHA,parent=parents['recovery'],expected_critic_warmup=10,expected_entropy_coef=0.0,runtime_shell=BASE+'/bin/sim60-runtime.sh',runtime_shell_sha256='c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7',qualification=None)
 p=r/'recovery_entropy_launch_023.json'
 with p.open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10))
 q=subprocess.run(['tmux','-S',socket,'new-session','-d','-s',SESSION,'-c',str(src),'bash','-lc',inner],capture_output=True,text=True)
 receipt.update(status='launched_waiting_for_real_restore_critic_adaptation_and_actor_PPO' if q.returncode==0 else 'tmux_launch_failed_inspect_before_retry',tmux_exit_code=q.returncode,tmux_stderr=q.stderr[-2000:],launched_at_utc=datetime.now(timezone.utc).isoformat())
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2)+chr(10));tmp.replace(p)
 print(json.dumps(receipt));assert q.returncode==0,q.stderr
'''


def emit(mode, meta, evidence, revision=None, predecessor=None):
    assert meta['previous_source_commit'] == PREVIOUS
    assert meta['contract_file'] == 'contracts/v6_task_campaign_v21.json'
    assert evidence['status'] == 'passed' and evidence['qualification'] is None
    assert evidence['source_commit'] == meta['source_commit'] and evidence['archive_sha256'] == meta['archive_sha256']
    values = dict(RUN=RUN, BASE=BASE, SESSION=SESSION, PREVIOUS=PREVIOUS, META=meta,
                  EVIDENCE_SHA=evidence['evidence_sha256'])
    if mode == 'launch':
        assert revision['schema'] == 'task-plan-revision-v21'
        assert predecessor['schema'] == 'task-recovery-entropy-predecessor-023-v1'
        assert predecessor['previous_source_commit'] == PREVIOUS
        assert predecessor['expected_entropy_coef'] == 0.0
        assert predecessor['expected_critic_warmup'] == 10 and predecessor['reopening'] is None and predecessor['qualification'] is None
        assert revision['source_commit'] == predecessor['source_commit'] == meta['source_commit']
        assert predecessor['evidence_sha256'] == values['EVIDENCE_SHA']
        values.update(REVISION=revision, PREDECESSOR=predecessor)
    return '\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+COMMON+{'prepare':PREPARE,'launch':LAUNCH}[mode]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','launch'))
    p.add_argument('--emit-remote-code',action='store_true',required=True)
    p.add_argument('--source',type=Path,default=OUT/'source_candidate_023.json')
    p.add_argument('--admission',type=Path,default=OUT/'admission_023.json')
    p.add_argument('--revision',type=Path,default=OUT/'source_revision_023.json')
    p.add_argument('--predecessor',type=Path,default=OUT/'predecessor_023.json')
    args=p.parse_args()
    print(emit(args.mode,read(args.source),read(args.admission),read(args.revision) if args.mode=='launch' else None,read(args.predecessor) if args.mode=='launch' else None))


if __name__=='__main__':main()
