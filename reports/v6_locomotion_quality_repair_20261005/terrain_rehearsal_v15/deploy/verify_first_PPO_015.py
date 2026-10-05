#!/usr/bin/env python3
"""One bounded read-only inspection of the native v15 terrain learning owner.

The predecessor receipt authenticates the actual own-domain seal, its restored
actor clock and the separate consumption origin. Nothing here starts training.
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
BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"

REMOTE = r'''
from datetime import datetime, timezone
import hashlib, json, math, pathlib, re, struct, sys, tarfile

def process_entry(arguments):
    # Embedded argv after wrapper "--" is not this process's Python entry.
    for value in arguments[1:]:
        if value in (b'-c', b'-m'): return None
        if value.endswith(b'.py'): return pathlib.Path(value.decode())
    return None

def inspect(settings):
    root=pathlib.Path(settings['run']); train=root/'train'
    source=pathlib.Path(settings['source']['source_remote'])
    predecessor=settings['predecessor']; bound=predecessor['actual_terrain_parent']
    parent=pathlib.Path(bound['path']); origin=predecessor['consumption']['consumed_updates']
    task_origin=bound['seal']['task_actor_update']; restored_updates=bound['seal']['learning_updates']
    result=dict(schema='v15-first-ppo-readonly-verification-v1',verified_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=settings['source']['source_commit'],qualification=False,
        scope='own_terrain_full_state_restore_and_actual_optimizer_updates_not_behavior_quality',checks={},evidence={})
    checks,evidence=result['checks'],result['evidence']
    evidence['clocks']=dict(consumption_origin=origin,restored_checkpoint_learning_updates=restored_updates,
        restored_task_actor_origin=task_origin,nonstandard_birth_origin=1186,
        first_critic_consumption_update=origin+1,first_actor_consumption_update=origin+11)
    def check(name,passed,detail=None): checks[name]=dict(status='passed' if passed else 'failed',detail=detail)
    def pending(name,detail): checks[name]=dict(status='pending',detail=detail)
    def read(path): return json.loads(path.read_text())
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
        return h.hexdigest()
    def section(name,function):
        try: function()
        except FileNotFoundError as error: pending(name,str(error))
        except Exception as error: check(name,False,f'{type(error).__name__}: {error}')
    def finish():
        statuses=[row['status'] for row in checks.values()]
        result['status']='failed' if 'failed' in statuses else 'pending' if 'pending' in statuses else 'first_actual_actor_PPO_verified'
        result['check_counts']={name:statuses.count(name) for name in ('passed','pending','failed')}
        print(json.dumps(result,allow_nan=False))
    def identity():
        revision=read(root/'source_revision_015.json')
        if revision!=settings['revision']: raise ValueError('Remote revision differs from pinned local receipt')
        archive=pathlib.Path(revision['source_archive'])
        if sha(archive)!=settings['source']['archive_sha256']: raise ValueError('Source archive SHA differs')
        with tarfile.open(archive,'r:gz') as tar:
            if tar.pax_headers.get('comment')!=settings['source']['source_commit']: raise ValueError('Archive commit differs')
            verified=0
            for member in tar:
                if not member.isfile():
                    if member.issym() or member.islnk(): raise ValueError('Archive source contains a link')
                    continue
                rel=pathlib.PurePosixPath(member.name)
                if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe archive member')
                content=tar.extractfile(member)
                if hashlib.sha256(content.read()).hexdigest()!=sha(source/member.name):
                    raise ValueError('Committed source differs: '+member.name)
                verified+=1
        check('exact_committed_source_and_plan',revision['new_source_root']==str(source) and
            revision['new_contract_sha256']==settings['source']['plan_sha256'] and
            sha(source/revision['new_contract_relative_path'])==settings['source']['plan_sha256'] and
            sha(source/'contracts/v6_task_campaign_v14.json')==revision['old_contract_sha256'],dict(verified_files=verified))
    section('exact_committed_source_and_plan',identity)
    if checks['exact_committed_source_and_plan']['status']!='passed': finish();return
    try:
        state=read(train/'campaign_state.json'); account=read(train/'consumption.json')
        plan=read(source/'contracts/v6_task_campaign_v15.json')
        config=read(source/'contracts/v6_task_campaign_terrain_worker_v15.json')
    except FileNotFoundError as error: pending('campaign_state',str(error));finish();return
    if state.get('contract_sha256')==account.get('contract_sha256')==settings['revision']['old_contract_sha256']:
        pending('v15_campaign_account','Predecessor account is still authoritative; no v15 PPO claim.');finish();return
    check('v15_campaign_account',state.get('contract_sha256')==account.get('contract_sha256')==settings['source']['plan_sha256'])
    unaffected=('flat','recovery','speed','jump','stepjump')
    old_plan=settings['old_plan']; before=predecessor['consumption']; old_state=predecessor['campaign_state']
    old_rows={row['name']:row for row in old_plan['domains']}; new_rows={row['name']:row for row in plan['domains']}
    check('five_other_worker_contracts_parent_tree_and_endpoints_unchanged',set(old_rows)==set(new_rows) and
        all(new_rows[name]==old_rows[name] for name in unaffected) and
        new_rows['terrain']['parent_domain']==old_rows['terrain']['parent_domain'] and
        new_rows['terrain']['updates']==old_rows['terrain']['updates'] and plan['domain_completion']==old_plan['domain_completion'] and
        plan.get('budget')==old_plan.get('budget') and plan['total_ceiling']==old_plan['total_ceiling'])
    check('five_other_domain_states_unchanged',all(state['domains'].get(name)==old_state['domains'].get(name) for name in unaffected))
    check('historical_consumption_preserved_no_refund_or_reopen',account['rollback_refunds_consumed_updates'] is False and
        account['consumed_updates']>=origin and account['actor_updates_consumed']>=before['actor_updates_consumed'] and
        all(account['attempts'].get(key)==value for key,value in before['attempts'].items()) and
        all(account['domain_consumed_updates'][name]==before['domain_consumed_updates'][name] for name in unaffected) and
        state['domains'].get('terrain',{}).get('reopening')==old_state['domains'].get('terrain',{}).get('reopening'))
    evidence['consumption']=dict(before=before,current=account)
    def other_models():
        models={}
        for name in unaffected:
            row=old_state['domains'].get(name,{})
            if row.get('candidate_checkpoint'):
                path=pathlib.Path(row['candidate_checkpoint']); seal=read(path.with_suffix('.checkpoint.json'))
                actual=sha(path)
                if actual!=row['candidate_checkpoint_sha256'] or actual!=seal['checkpoint_sha256'] or not seal['sealed']:
                    raise ValueError('Changed unaffected sealed model: '+name)
                models[name]=dict(path=str(path),sha256=actual)
            else: models[name]=dict(coverage='no_candidate_checkpoint_in_predecessor_state; domain state and account compared')
        evidence['unaffected_models']=models
        check('declared_other_sealed_models_unchanged',True)
    section('declared_other_sealed_models_unchanged',other_models)
    sys.path.insert(0,str(source/'src'))
    def parent_identity():
        from wheeled_tasks.chassis.task_campaign import validate_worker_revision
        seal=read(parent.with_suffix('.checkpoint.json')); contract_path=parent.with_suffix('.contract.json')
        old_config=read(contract_path)
        if not parent.is_relative_to(train/'terrain') or seal!=bound['seal'] or not seal['sealed']:
            raise ValueError('Parent is not the captured immutable own-terrain seal')
        if sha(parent)!=bound['sha256'] or sha(contract_path)!=bound['contract_sha256'] or seal['contract_sha256']!=bound['contract_sha256']:
            raise ValueError('Own terrain PT/contract/SHA mismatch')
        if seal['task_campaign']['domain']!='terrain': raise ValueError('Parent task is not terrain')
        validate_worker_revision(old_config,config)
        evidence['actual_terrain_parent']=bound
        evidence['native_worker_revision_validation']=dict(status='passed',source_contract_sha256=bound['contract_sha256'],
            target_contract_sha256=sha(source/'contracts/v6_task_campaign_terrain_worker_v15.json'))
        check('authenticated_actual_own_terrain_seal',True)
    section('authenticated_actual_own_terrain_seal',parent_identity)
    invocations=[]
    for path in (train/'terrain').glob('attempt_*/invocation.json'):
        row=read(path); command=row.get('command',[])
        if process_entry([str(value).encode() for value in command])==source/'scripts/train_chassis.py' and '--contract' in command and command[command.index('--contract')+1]==str(source/'contracts/v6_task_campaign_terrain_worker_v15.json'):
            invocations.append((path,row))
    if not invocations: pending('v15_terrain_invocation','No actual v15 terrain invocation yet.');finish();return
    if len(invocations)!=1: check('unique_v15_terrain_invocation',False,len(invocations));finish();return
    path,invocation=invocations[0]; directory=path.parent/'train'; command=invocation['command']
    def argument(flag): return command[command.index(flag)+1]
    endpoint=plan['domain_completion']['schedule']['end_actor_updates']['terrain']
    expected_updates=min(new_rows['terrain']['updates']-before['domain_consumed_updates']['terrain'],max(0,endpoint-task_origin)+10)
    check('actual_own_parent_clock_and_requested_budget',invocation['source_checkpoint']==str(parent) and
        invocation['source_checkpoint_sha256']==bound['sha256'] and int(argument('--consumed-updates'))==origin and
        int(argument('--task-actor-update'))==task_origin and
        int(argument('--actor-updates-consumed'))==before['actor_updates_consumed'] and
        invocation['requested_updates']==expected_updates and invocation['scheduled_end_actor_update']==endpoint)
    evidence.update(actual_worker_directory=str(directory),invocation=invocation)
    def runtime():
        resume=read(directory/'resume_verification.json'); startup=read(directory/'startup.json'); progress=read(directory/'progress.json')
        native=startup['startup']; observed=native['runtime_preflight']['observed']
        evidence.update(resume_verification=resume,progress=progress,runtime_preflight=native['runtime_preflight'])
        check('full_native_restore_receipt',all(resume.get(key) is True for key in ('actor_exact','critic_exact','optimizer_exact')) and
            resume['parent_checkpoint_sha256']==bound['sha256'] and resume['parent_updates']==origin and
            resume['restored_checkpoint_updates']==restored_updates and resume['task_clock']=='restored')
        groups=native['scene_group_counts']; pools=native['behavior_pool_counts']
        check('actual_native_85_positive_groups_and_pool_population',len(groups)==85 and sum(groups.values())==4096 and
            groups==settings['engineering']['counts'] and pools==settings['engineering']['pools'] and min(groups.values())>0)
        check('actual_runtime_4096_35_81_6_and_200Hz_50Hz',progress['num_envs']==4096 and progress['stage']=='terrain' and
            (config['actor_dim'],config['critic_dim'],config['action_dim'])==(35,81,6) and
            (observed['physics_dt'],observed['policy_dt'],observed['decimation'],observed['pc_control_dt'])==(.005,.02,4,.005) and
            all(observed['modules'][key] for key in ('command_transport','signal_delay','nonstandard_start','release_stop','stand_still_lin_vel','locomotion_posture_floor')))
        check('birth_origin_preserved_separate_from_restored_actor_clock',observed['nonstandard_start']['origin_actor_update']==1186 and
            config['nonstandard_start']['origin_actor_update']==1186 and progress['task_actor_update']==task_origin+progress['actor_updates_in_block'])
        check('ten_critic_updates_charged_once_without_actor_clock',progress['parent_updates']==origin and
            progress['successful_updates']-progress['actor_updates_in_block']==min(10,progress['successful_updates']) and
            progress['task_actor_update']==task_origin+progress['actor_updates_in_block'])
        admission=read(source/config['actuator_response_evidence']['file'])
        check('wheel_response_retained_no_qualified_leg_response',admission['adopted_joint_names']==['L_joint3','R_joint3'] and
            admission['leg_response'].startswith('candidate_not_adopted') and observed['actuator_response']==config['actuator_response'])
        evidence['live_PPO_substep_capture']=dict(coverage='not_persisted_in_startup_written_before_first_step',
            startup_marker=native['control_timing']['runtime_substep_verification'],
            actual_native_assertion='SubstepControlAudit.end_policy executes in every native env.step; authenticated source is unchanged')
    section('native_runtime_and_progress',runtime)
    engineering=settings['engineering']; control=engineering['control']
    check('separate_frozen_local_three_second_engineering_control_probe',engineering['status']=='passed' and
        engineering['source_commit']==settings['source']['source_commit'] and engineering['archive_sha256']==settings['source']['archive_sha256'] and
        engineering['target_contract_sha256']==sha(source/'contracts/v6_task_campaign_terrain_worker_v15.json') and
        engineering['formal_updates_consumed']==engineering['optimizer_steps']==0 and
        control['pd_updates']==control['feedback_refreshes']==4 and control['target_decodes']==1 and
        (control['physics_dt'],control['policy_dt'])==(.005,.02),dict(scope=engineering['scope'],
            control=control,not_actual_PPO_substep_capture=True,not_behavior_quality=True))
    def learning():
        import torch
        torch.set_num_threads(1)
        def equal(a,b):
            if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
            if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        def load(path,is_parent=False):
            seal=read(path.with_suffix('.checkpoint.json')); digest=sha(path); contract_sha=sha(path.with_suffix('.contract.json'))
            expected=bound['contract_sha256'] if is_parent else sha(source/'contracts/v6_task_campaign_terrain_worker_v15.json')
            if not seal['sealed'] or digest!=seal['checkpoint_sha256'] or contract_sha!=expected or seal['contract_sha256']!=expected:
                raise ValueError('Immutable PT/contract seal mismatch: '+str(path))
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['infos']['contract_sha256']!=expected or saved['infos']['successful_updates_total']!=seal['learning_updates'] or saved['infos']['task_actor_update']!=seal['task_actor_update']:
                raise ValueError('Checkpoint infos differ from actual immutable seal')
            if saved['infos']['task_campaign']['domain']!='terrain': raise ValueError('Wrong task in loaded state')
            for name in ('actor_state_dict','critic_state_dict'):
                if not all(torch.isfinite(v).all().item() for v in saved[name].values()): raise ValueError('Nonfinite '+name)
            evidence.setdefault('immutable_checkpoints',{})[str(path)]=dict(sha256=digest,seal=seal)
            return saved
        old=load(parent,True)
        if sha(parent)!=bound['sha256']: raise ValueError('Actual parent PT changed')
        initial=None
        def initial_check():
            nonlocal initial
            initial=load(directory/'initial_state/model_final.pt')
            exact={key:equal(old[key],initial[key]) for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict')}
            check('initial_actor_critic_adam_exact_values',all(exact.values()) and initial['infos']['successful_updates_total']==origin and
                initial['infos']['task_actor_update']==task_origin and initial['infos']['critic_warmup_remaining']==10,exact)
        section('initial_actor_critic_adam_exact_values',initial_check)
        def critic_check():
            saved=load(directory/f'checkpoints/update_{origin+1:08d}/model.pt')
            check('first_critic_actual_update_actor_unchanged',equal(old['actor_state_dict'],saved['actor_state_dict']) and
                not equal(old['critic_state_dict'],saved['critic_state_dict']) and saved['infos']['task_actor_update']==task_origin and
                saved['infos']['successful_updates_total']==origin+1 and saved['infos']['critic_warmup_remaining']==9)
        section('first_critic_actual_update_actor_unchanged',critic_check)
        def actor_check():
            saved=load(directory/f'checkpoints/update_{origin+11:08d}/model.pt')
            baseline=old if initial is None else initial
            changed=[key for key,value in baseline['actor_state_dict'].items() if not torch.equal(value,saved['actor_state_dict'][key])]
            evidence['first_actor_changed_tensors']=changed
            check('first_actor_actual_update_after_ten_critic_updates',bool(changed) and saved['infos']['task_actor_update']==task_origin+1 and
                saved['infos']['successful_updates_total']==origin+11 and saved['infos']['critic_warmup_remaining']==0)
        section('first_actor_actual_update_after_ten_critic_updates',actor_check)
    section('actual_checkpoint_parameter_updates',learning)
    def ownership():
        rows=[]
        kinds={'chassis_remote_job.py':'wrapper','run_task_campaign.py':'supervisor','train_chassis.py':'worker'}
        for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args=path.read_bytes().split(b'\0'); entry=process_entry(args)
                if not entry or entry.name not in kinds: continue
                if not any(value==str(train).encode() or value.startswith(str(train).encode()+b'/') or value==str(root).encode() for value in args): continue
                ppid=int(next(line.split()[1] for line in (path.parent/'status').read_text().splitlines() if line.startswith('PPid:')))
                rows.append(dict(pid=int(path.parent.name),ppid=ppid,kind=kinds[entry.name],entry_script=str(entry)))
            except (OSError,PermissionError,StopIteration): continue
        evidence['campaign_python_entry_owners']=rows
        if not rows: pending('unique_native_owner_chain','No current campaign owner; first PPO artifacts may still be valid.');return
        valid=sorted(row['kind'] for row in rows)==['supervisor','worker','wrapper']
        if valid:
            by_kind={row['kind']:row for row in rows}
            valid=(by_kind['worker']['ppid']==by_kind['supervisor']['pid'] and by_kind['supervisor']['ppid']==by_kind['wrapper']['pid'] and
                all(row['entry_script']==str(source/'scripts'/({'wrapper':'chassis_remote_job.py','supervisor':'run_task_campaign.py','worker':'train_chassis.py'}[row['kind']])) for row in rows))
        check('unique_native_owner_chain',valid,dict(classification='actual_python_entry_not_embedded_child_argv',owners=rows))
    section('unique_native_owner_chain',ownership)
    def events():
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
        paths=list(directory.glob('events.out.tfevents.*'))
        if not paths: pending('native_crc_reward_delay_timing','No native event file yet');return
        event=max(paths,key=lambda p:p.stat().st_mtime_ns); size=event.stat().st_size; offset=max(0,size-settings['tail_bytes'])
        with event.open('rb') as stream: stream.seek(offset);data=stream.read(settings['tail_bytes'])
        pattern=re.compile(rb'(?:LearningAudit/[A-Za-z0-9_]+/reward_density/(?:stand_still_lin_vel|locomotion_pitch_floor|locomotion_roll_floor)|Timing/[A-Za-z0-9_]+|(?:Episode/)?/(?:delay|transport)/[A-Za-z0-9_]+)')
        seen=set();series={};unresolved=0
        for match in pattern.finditer(data):
            located=False
            for pos in range(max(0,match.start()-128),max(0,match.start()-11)):
                length,crc=struct.unpack_from('<QI',data,pos);end=pos+16+length
                if not 0<length<1024**2 or end>len(data) or end-4<match.end():continue
                if masked_crc32c(data[pos:pos+8])!=crc:continue
                payload=data[pos+12:end-4]
                if masked_crc32c(payload)!=struct.unpack_from('<I',data,end-4)[0]:continue
                located=True
                if pos in seen:break
                seen.add(pos);item=Event.FromString(payload)
                for value in item.summary.value:
                    if pattern.fullmatch(value.tag.encode()) and value.HasField('simple_value'):
                        number=float(value.simple_value)
                        if not math.isfinite(number):raise ValueError('Nonfinite scalar '+value.tag)
                        series.setdefault(value.tag,{})[int(item.step)]=number
                break
            unresolved+=int(not located)
        stats={tag:dict(samples=len(values),first_step=min(values),last_step=max(values),minimum=min(values.values()),
            maximum=max(values.values()),latest=values[max(values)]) for tag,values in series.items()}
        evidence['native_crc_event_window']=dict(file=str(event),file_bytes=size,read_offset=offset,read_bytes=len(data),
            coverage='bounded_CRC_valid_tail_only_not_full_run_or_quality',crc_valid_records=len(seen),unlocated_matches=unresolved,scalars=stats)
        costs=all(any(tag.endswith('/'+key) and row['minimum']<0 for tag,row in stats.items()) for key in ('stand_still_lin_vel','locomotion_pitch_floor','locomotion_roll_floor'))
        delay=any('/delay/' in tag for tag in stats) and any('/transport/' in tag for tag in stats)
        timing=all('Timing/'+name in stats for name in ('iteration_wall_seconds','env_step_seconds','optimizer_seconds'))
        if costs and delay and timing:check('native_crc_reward_delay_timing',True)
        else:pending('native_crc_reward_delay_timing',dict(negative_costs_present=costs,delay_and_transport=delay,timing=timing))
    section('native_crc_reward_delay_timing',events)
    finish()
'''


def settings_for(tail_bytes):
    source=json.loads((OUT/'source_candidate_015.json').read_text())
    revision=json.loads((OUT/'source_revision_015.json').read_text())
    predecessor=json.loads((OUT/'predecessor_state_015.json').read_text())
    engineering=json.loads((OUT.parent/'verification/native_population_015.json').read_text())
    report=OUT.parent/'verification/native_report_015.json'
    if hashlib.sha256(report.read_bytes()).hexdigest()!=engineering['report_sha256']:
        raise ValueError('Local engineering report SHA differs')
    if (source['source_commit']!=revision['source_commit'] or source['archive_sha256']!=revision['source_archive_sha256'] or
            source['plan_sha256']!=revision['new_contract_sha256'] or predecessor['source_commit']!=source['source_commit']):
        raise ValueError('Local candidate/revision/predecessor identities differ')
    old_plan_path=REPO/'contracts/v6_task_campaign_v14.json'
    if hashlib.sha256(old_plan_path.read_bytes()).hexdigest()!=revision['old_contract_sha256']:
        raise ValueError('Local predecessor plan SHA differs')
    return dict(run=RUN,source=source,revision=revision,predecessor=predecessor,engineering=engineering,
        old_plan=json.loads(old_plan_path.read_text()),tail_bytes=tail_bytes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='kaiser@192.168.64.234')
    parser.add_argument('--port',type=int,default=2222)
    parser.add_argument('--timeout',type=int,default=60)
    parser.add_argument('--tail-mib',type=int,default=8,choices=range(1,33))
    parser.add_argument('--output',type=Path,default=OUT/'first_PPO_verification_015.json')
    parser.add_argument('--emit-remote-code',action='store_true',help='Only print remote code; no connection')
    args=parser.parse_args()
    settings=settings_for(args.tail_mib*1024**2)
    code=REMOTE+'\ninspect(json.loads('+repr(json.dumps(settings))+'))\n'
    if args.emit_remote_code:print(code);return
    command='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && '+shlex.join([BASE+'/env/bin/python','-B','-'])
    try:
        remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p',str(args.port),args.host,
            shlex.join(['bash','-lc',command])],input=code,capture_output=True,text=True,timeout=args.timeout)
        if remote.returncode:raise RuntimeError(remote.stderr[-2000:])
        report=json.loads(remote.stdout)
    except (subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError) as error:
        report=dict(schema='v15-first-ppo-readonly-verification-v1',status='pending',qualification=False,
            verified_at_utc=datetime.now(timezone.utc).isoformat(),checks=dict(remote_read=dict(status='pending',detail=str(error))))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=report.get('check_counts'),output=str(args.output),
        outstanding={key:value for key,value in report['checks'].items() if value['status']!='passed'}),ensure_ascii=False))


if __name__=='__main__':main()
