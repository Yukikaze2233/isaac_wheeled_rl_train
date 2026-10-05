#!/usr/bin/env python3
"""Read v15's current native domain, completed work and immutable seal once.

No worker or scene is started. A normal automatic domain transition is reported
separately from an ownership/identity mismatch. No authority is changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
BASE = '/home/kaiser/robot-rl-sim60'

ACCOUNTING_SOURCE = r'''
def accounting_snapshot(progress, account, key, historical_consumed_updates):
    """Keep completed optimizer work, charged reservations and file skew apart."""
    completed=progress['successful_updates']; attempted=progress.get('attempted_updates',completed)
    actor=progress['actor_updates_in_block']; record=account['attempts'].get(key)
    if not (0<=actor<=completed<=attempted<=completed+1 and
            account['consumed_updates']==historical_consumed_updates+sum(account['domain_consumed_updates'].values())):
        raise ValueError('Invalid completed/inflight native accounting')
    if record is None:
        return dict(status='pending_supervisor_attempt_publication',current_attempt=key,
            worker_completed_updates=completed,worker_actor_completed_updates=actor,
            worker_attempted_updates=attempted,worker_inflight_reservations=attempted-completed,
            learning_progress_global_completed=progress['parent_updates']+completed,
            ledger_attempt_charged_updates=None,ledger_attempt_actor_completed_updates=None,
            scope='Worker and supervisor publish independently; missing attempt is not a restart instruction.')
    charged,charged_actor=record['consumed_updates'],record['actor_updates']
    if not 0<=charged_actor<=charged:raise ValueError('Invalid charged actor accounting')
    if charged==completed and charged_actor==actor:status='aligned_completed_updates'
    elif charged==completed+1 and charged_actor==actor and attempted==charged:status='inflight_reservation_charged_not_completed'
    elif charged<=completed and charged_actor<=actor:status='supervisor_ledger_lags_worker_progress'
    else:status='non_atomic_publication_snapshot'
    return dict(status=status,current_attempt=key,worker_completed_updates=completed,
        worker_actor_completed_updates=actor,worker_attempted_updates=attempted,
        worker_inflight_reservations=attempted-completed,ledger_attempt_charged_updates=charged,
        ledger_attempt_actor_completed_updates=charged_actor,
        ledger_minus_worker_completed_updates=charged-completed,
        ledger_minus_worker_actor_updates=charged_actor-actor,
        learning_progress_global_completed=progress['parent_updates']+completed,
        scope='Completed learning is not a reserved update. Worker and supervisor files publish independently; skew is not a restart signal.')
'''
exec(ACCOUNTING_SOURCE)

REMOTE = ACCOUNTING_SOURCE + r'''
from datetime import datetime,timezone
import hashlib,json,pathlib,shutil,subprocess

def read(path):return json.loads(path.read_text())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()
def actual_entry(arguments):
    for value in arguments[1:]:
        if value in ('-c','-m'):return None
        if value.endswith('.py'):return pathlib.Path(value)
    return None
def flag(arguments,name):return arguments[arguments.index(name)+1]

def inspect(settings):
    launch=settings['launch'];root=pathlib.Path(launch['run_root']);train=root/'train'
    source=pathlib.Path(launch['source_root']);predecessor=settings['predecessor']
    result=dict(schema='v15-native-live-readonly-v1',observed_at_utc=datetime.now(timezone.utc).isoformat(),
        qualification=False,source_commit=launch['source_commit'],run_root=str(root),
        scope='native_learning_accounting_and_seal_only_not_behavior_qualification',checks={},
        publication_scope='Files and /proc are independent read-only snapshots, not one atomic campaign transaction.')
    checks=result['checks']
    def check(name,passed,detail=None):checks[name]=dict(status='passed' if passed else 'failed',detail=detail)
    def pending(name,detail):checks[name]=dict(status='pending',detail=detail)
    def finish():
        statuses=[row['status'] for row in checks.values()]
        result['status']='identity_or_invariant_mismatch' if 'failed' in statuses else 'pending_publication' if 'pending' in statuses else 'read_only_snapshot_verified'
        print(json.dumps(result,allow_nan=False,ensure_ascii=False))
    try:
        revision=read(root/'source_revision_015.json');plan_path=source/'contracts/v6_task_campaign_v15.json'
        check('launch_revision_and_plan_identity',revision['source_commit']==launch['source_commit'] and
            revision['new_source_root']==str(source) and revision['new_contract_sha256']==launch['plan_sha256'] and
            revision['source_archive_sha256']==launch['source_archive_sha256'] and sha(plan_path)==launch['plan_sha256'])
        plan=read(plan_path);state=read(train/'campaign_state.json');account=read(train/'consumption.json')
    except FileNotFoundError as error:pending('published_identity_and_campaign',str(error));finish();return
    result.update(campaign_status=state.get('status'),contract_sha256=state.get('contract_sha256'),
        campaign_consumed_updates=account['consumed_updates'],campaign_actor_updates_consumed=account['actor_updates_consumed'],
        domain_consumed_updates=account['domain_consumed_updates'],remaining_updates=account.get('remaining_updates'))
    check('current_account_authority_matches_launch',state.get('contract_sha256')==account.get('contract_sha256')==launch['plan_sha256'])
    check('historical_consumption_not_refunded',account['rollback_refunds_consumed_updates'] is False and
        account['consumed_updates']>=predecessor['consumption']['consumed_updates'] and
        all(account['attempts'].get(key)==value for key,value in predecessor['consumption']['attempts'].items()))
    if any(row['status']=='failed' for row in checks.values()):finish();return
    rows=[];kinds={'chassis_remote_job.py':'wrapper','run_task_campaign.py':'supervisor','train_chassis.py':'worker','evaluate_chassis.py':'evaluator'}
    for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments=[value.decode() for value in path.read_bytes().split(b'\0') if value]
            entry=actual_entry(arguments)
            if not entry or entry.name not in kinds:continue
            if not any(value==str(root) or value==str(train) or value.startswith(str(train)+'/') for value in arguments):continue
            ppid=int(next(line.split()[1] for line in (path.parent/'status').read_text().splitlines() if line.startswith('PPid:')))
            rows.append(dict(pid=int(path.parent.name),ppid=ppid,kind=kinds[entry.name],entry_script=str(entry),argv=arguments))
        except (OSError,PermissionError,StopIteration):continue
    result['processes']=sorted(rows,key=lambda row:row['pid'])
    workers=[row for row in rows if row['kind']=='worker'];supervisors=[row for row in rows if row['kind']=='supervisor']
    wrappers=[row for row in rows if row['kind']=='wrapper'];evaluators=[row for row in rows if row['kind']=='evaluator']
    valid=len(workers)<=1 and len(supervisors)<=1 and len(wrappers)<=1 and all(
        row['entry_script']==str(source/'scripts'/({'worker':'train_chassis.py','supervisor':'run_task_campaign.py','wrapper':'chassis_remote_job.py','evaluator':'evaluate_chassis.py'}[row['kind']])) for row in rows)
    if workers:valid=valid and len(supervisors)==len(wrappers)==1 and workers[0]['ppid']==supervisors[0]['pid'] and supervisors[0]['ppid']==wrappers[0]['pid']
    elif supervisors:valid=valid and len(wrappers)==1 and supervisors[0]['ppid']==wrappers[0]['pid']
    if evaluators:valid=valid and len(supervisors)==1 and all(row['ppid']==supervisors[0]['pid'] for row in evaluators)
    check('actual_python_entry_owner_chain',valid,dict(worker_count=len(workers),supervisor_count=len(supervisors),wrapper_count=len(wrappers),
        evaluator_count=len(evaluators),classification='first_actual_python_script_not_embedded_wrapper_child'))
    result['single_owned_process_chain']=valid and len(workers)==len(supervisors)==len(wrappers)==1
    published=read(train/'progress.json') if (train/'progress.json').exists() else None
    result['supervisor_published_progress']=published
    result['progress']=None;result['accounting']=None;result['current_domain']=None;result['current_attempt']=None
    directory=None;invocation=None
    if len(workers)==1:
        worker=workers[0];directory=pathlib.Path(flag(worker['argv'],'--run-dir'))
        relative=directory.relative_to(train)
        if len(relative.parts)!=3 or relative.parts[-1]!='train' or not relative.parts[1].startswith('attempt_'):
            raise ValueError('Worker run directory is not a native domain attempt')
        domain=relative.parts[0];rows_by_name={row['name']:row for row in plan['domains']}
        if domain not in rows_by_name:raise ValueError('Worker is outside the frozen domain catalog')
        row=rows_by_name[domain];contract=pathlib.Path(flag(worker['argv'],'--contract'))
        check('current_worker_declared_contract_and_domain',contract==source/row['contract'] and sha(contract)==row['sha256'] and
            read(contract)['task_campaign']['domain']==domain)
        invocation=read(directory.parent/'invocation.json')
        check('actual_current_invocation_argv',invocation['command']==worker['argv'])
        result.update(current_domain=domain,current_attempt=str(relative),actual_worker_directory=str(directory),invocation=invocation)
        if (directory/'progress.json').exists():
            progress=read(directory/'progress.json');result['progress']=progress
            check('current_progress_owns_worker_PID',progress['pid']==worker['pid'])
            result['accounting']=accounting_snapshot(progress,account,str(relative),plan['historical_consumed_updates'])
            result['native_task_actor_update']=progress.get('task_actor_update')
        else:pending('current_worker_progress','Native worker is initializing; no progress receipt yet.')
        result['course_lifecycle']='native_worker_present'
    else:
        result['course_lifecycle']='native_evaluation' if evaluators else 'automatic_course_transition_or_waiting' if supervisors else 'no_live_campaign_owner'
        # Retain the last published domain only as historical information.
        result['last_supervisor_published_domain']=None if published is None else published.get('active_domain')
        pending('current_worker_progress','No live PPO worker in this snapshot; no course change or restart is inferred.')
    def authenticated_checkpoint(path,coverage):
        seal_path=path.with_suffix('.checkpoint.json');contract_path=path.with_suffix('.contract.json')
        seal=read(seal_path);digest=sha(path);contract_sha=sha(contract_path)
        if not seal.get('sealed') or digest!=seal['checkpoint_sha256'] or contract_sha!=seal['contract_sha256']:
            raise ValueError('Immutable checkpoint or contract SHA mismatch')
        snapshot_contract=path.parent/'contract.json'
        if snapshot_contract.exists() and sha(snapshot_contract)!=contract_sha:raise ValueError('Snapshot contract copy SHA mismatch')
        return dict(path=str(path),seal=seal,sha256=digest,contract_sha256=contract_sha,hash_verified=True,coverage=coverage)
    if directory is not None:
        checkpoints=[]
        for seal_path in (directory/'checkpoints').glob('update_*/model.checkpoint.json'):
            seal=read(seal_path)
            if seal.get('sealed'):checkpoints.append((seal['learning_updates'],seal_path.with_name('model.pt')))
        final=directory/'model_final.pt'
        if final.exists() and final.with_suffix('.checkpoint.json').exists():checkpoints.append((read(final.with_suffix('.checkpoint.json'))['learning_updates'],final))
        if checkpoints:
            _,path=max(checkpoints,key=lambda item:item[0])
            result['latest_sealed_checkpoint']=authenticated_checkpoint(path,'current_native_attempt_immutable_learning_snapshot')
            check('latest_seal_current_task_identity',result['latest_sealed_checkpoint']['seal']['task_campaign']['domain']==result['current_domain'] and
                result['latest_sealed_checkpoint']['contract_sha256']==row['sha256'])
        elif invocation:
            result['latest_sealed_checkpoint']=authenticated_checkpoint(pathlib.Path(invocation['source_checkpoint']),'authenticated_parent_only_new_attempt_has_no_sealed_update_yet')
            pending('current_attempt_sealed_update','Only the authentic parent is available; not a new completed PPO checkpoint.')
    else:result['latest_sealed_checkpoint']=None
    gpu_command=shutil.which('nvidia-smi') or next((path for path in ('/usr/lib/wsl/lib/nvidia-smi','/usr/bin/nvidia-smi') if pathlib.Path(path).is_file()),None)
    result['gpu']=None;result['gpu_error']=None
    try:
        if gpu_command:result['gpu']=subprocess.check_output([gpu_command,'--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True,timeout=5).strip()
        else:result['gpu_error']='nvidia-smi unavailable'
    except (OSError,subprocess.SubprocessError) as error:result['gpu_error']=str(error)
    try:result['tmux_sessions']=subprocess.check_output(['tmux','-S',str(settings['base'])+'/tmux.sock','list-sessions','-F','#{session_name}'],text=True,timeout=3).splitlines()
    except (OSError,subprocess.SubprocessError) as error:result['tmux_sessions']=None;result['tmux_error']=str(error)
    finish()
'''


def settings_for():
    launch=json.loads((OUT/'terrain_rehearsal_launch_015.json').read_text())
    predecessor=json.loads((OUT/'predecessor_state_015.json').read_text())
    candidate=json.loads((OUT/'source_candidate_015.json').read_text())
    if (launch['source_commit']!=candidate['source_commit'] or launch['source_archive_sha256']!=candidate['archive_sha256'] or
            launch['source_root']!=candidate['source_remote'] or launch['plan_sha256']!=candidate['plan_sha256'] or
            predecessor['source_commit']!=candidate['source_commit']):raise ValueError('Local launch/candidate/predecessor identity mismatch')
    return dict(base=BASE,launch=launch,predecessor=predecessor)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='kaiser@192.168.64.234')
    parser.add_argument('--port',type=int,default=2222)
    parser.add_argument('--timeout',type=int,default=30,choices=range(1,31))
    parser.add_argument('--emit-remote-code',action='store_true',help='Only print remote code; no connection')
    parser.add_argument('--output',type=Path,default=OUT/'current_live_015.json')
    args=parser.parse_args();settings=settings_for()
    code=REMOTE+'\nsettings=json.loads('+repr(json.dumps(settings))+')\n'+r'''
try:
    inspect(settings)
except FileNotFoundError as error:
    print(json.dumps(dict(schema='v15-native-live-readonly-v1',status='pending_publication',qualification=False,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),checks=dict(non_atomic_publication=dict(status='pending',detail=str(error))),
        scope='File or process disappeared during independent reads; no course change or restart is inferred.')))
except Exception as error:
    print(json.dumps(dict(schema='v15-native-live-readonly-v1',status='identity_or_invariant_mismatch',qualification=False,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),checks=dict(read_invariant=dict(status='failed',detail=type(error).__name__+': '+str(error))))))
'''
    if args.emit_remote_code:print(code);return
    command='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && '+shlex.join([BASE+'/env/bin/python','-B','-'])
    try:
        remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-p',str(args.port),args.host,
            shlex.join(['bash','-lc',command])],input=code,text=True,capture_output=True,timeout=args.timeout)
        if remote.returncode:raise RuntimeError(remote.stderr[-2000:])
        receipt=json.loads(remote.stdout)
    except (subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError) as error:
        receipt=dict(schema='v15-native-live-readonly-v1',status='pending_read',qualification=False,
            observed_at_utc=datetime.now(timezone.utc).isoformat(),checks=dict(remote_read=dict(status='pending',detail=str(error))))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'
    timestamp=receipt['observed_at_utc'];immutable=args.output.parent/('live_'+timestamp[:10].replace('-','')+'T'+timestamp[11:19].replace(':','')+'Z.json')
    with immutable.open('x') as stream:stream.write(text)
    args.output.write_text(text)
    print(json.dumps(dict(file=str(immutable),current_file=str(args.output),status=receipt['status'],
        domain=receipt.get('current_domain'),attempt=receipt.get('current_attempt'),accounting=receipt.get('accounting'),
        consumed=receipt.get('campaign_consumed_updates'),qualification=False),ensure_ascii=False))


if __name__=='__main__':main()
