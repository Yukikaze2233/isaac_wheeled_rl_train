#!/usr/bin/env python3
"""Emit a bounded read-only first PPO verifier for the v18 source-only repair.

No connection or training action is implemented here. The remote inspector reads
CPU checkpoints, native startup/progress, /proc, and a CRC-authenticated event
tail. Missing evidence stays pending; failed assertions never imply permission
to restart. Clocks come from the prepared predecessor, not a fixed update count.
"""
import argparse
import json
from pathlib import Path

from deploy_source_repair_020 import BASE, COMMIT, OUT, RUN, load_evidence, read

REMOTE = r'''
from datetime import datetime,timezone
import hashlib,json,math,pathlib,re,struct,sys,tarfile
def process_entry(arguments):
 for value in arguments[1:]:
  if value in (b'-c',b'-m'):return None
  if value.endswith(b'.py'):return pathlib.Path(value.decode())
 return None
def inspect(settings):
    root=pathlib.Path(settings['run']);train=root/'train';source=pathlib.Path(settings['source']['source_remote'])
    predecessor=settings['predecessor'];bound=predecessor['actual_parents']['recovery'];parent=pathlib.Path(bound['path'])
    before=predecessor['consumption'];old_state=predecessor['campaign_state']
    origin=before['consumed_updates'];task_origin=bound['seal']['task_actor_update'];restored_updates=bound['seal']['learning_updates']
    result=dict(schema='source020-first-ppo-readonly-verification-v1',source_commit=settings['source']['source_commit'],
        verified_at_utc=datetime.now(timezone.utc).isoformat(),qualification=None,
        scope='same_v18_contract_full_state_restore_zero_new_critic_warmup_first_actor_update_not_behavior_quality',checks={},evidence={})
    checks,evidence=result['checks'],result['evidence']
    evidence['engineering_limitations']=settings['engineering_limitations']
    evidence['clocks']=dict(consumption_origin=origin,actor_consumption_origin=before['actor_updates_consumed'],
        restored_checkpoint_learning_updates=restored_updates,restored_task_actor_origin=task_origin,
        first_actor_consumption_update=origin+1,expected_new_critic_only_updates=0)
    def check(name,passed,detail=None):checks[name]=dict(status='passed' if passed else 'failed',detail=detail)
    def pending(name,detail):checks[name]=dict(status='pending',detail=detail)
    def read(path):return json.loads(path.read_text())
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024**2),b''):h.update(b)
        return h.hexdigest()
    def section(name,function):
        try:function()
        except FileNotFoundError as error:pending(name,str(error))
        except Exception as error:check(name,False,f'{type(error).__name__}: {error}')
    def finish():
        statuses=[row['status'] for row in checks.values()]
        result['status']='failed' if 'failed' in statuses else 'pending' if 'pending' in statuses else 'first_actual_actor_PPO_verified'
        result['check_counts']={name:statuses.count(name) for name in ('passed','pending','failed')}
        print(json.dumps(result,allow_nan=False))
    def identity():
        revision=read(root/'source_revision_020.json')
        if revision!=settings['revision'] or read(root/'predecessor_020.json')!=predecessor:raise ValueError('Prepared admission changed')
        if revision['schema']!='task-source-revision-v1':raise ValueError('Not a same-contract source repair')
        if revision['source_commit']!=settings['source']['source_commit']:raise ValueError('Commit differs')
        archive=pathlib.Path(revision['source_archive'])
        if sha(archive)!=settings['source']['archive_sha256']:raise ValueError('Source archive SHA differs')
        names=set()
        with tarfile.open(archive,'r:gz') as tar:
            if tar.pax_headers.get('comment')!=settings['source']['source_commit']:raise ValueError('Archive commit differs')
            for member in tar:
                rel=pathlib.PurePosixPath(member.name)
                if rel.is_absolute() or '..' in rel.parts or not (member.isfile() or member.isdir()):raise ValueError('Unsafe archive member')
                if member.isdir():continue
                if member.name in names:raise ValueError('Duplicate archive member')
                names.add(member.name)
                if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=sha(source/member.name):raise ValueError('Committed source differs: '+member.name)
        old=pathlib.Path(revision['old_source_root'])
        old_contracts={str(p.relative_to(old)):sha(p) for p in (old/'contracts').rglob('*') if p.is_file()}
        new_contracts={str(p.relative_to(source)):sha(p) for p in (source/'contracts').rglob('*') if p.is_file()}
        check('exact_frozen_source_all_contract_bytes_unchanged',revision['new_source_root']==str(source) and old_contracts==new_contracts and
            revision['old_contract_sha256']==revision['new_contract_sha256']==settings['source']['plan_sha256']==sha(source/'contracts/v6_task_campaign_v18.json'),dict(verified_files=len(names)))
        launch=read(root/'source_repair_launch_020.json');job=read(root/'job.json')
        if launch['source_commit']!=revision['source_commit'] or launch['evidence_sha256']!=settings['evidence_sha256']:raise ValueError('Launch admission differs')
        check('actual_wrapper_same_command_with_two_source_prefix_changes',job['command']==revision['command'] and
            sha(root/job['previous_job'])==revision['previous_job_sha256'] and read(root/job['source_revision_receipt'])==revision and
            predecessor['evidence_sha256']==settings['evidence_sha256'])
        evidence['job']=job;evidence['launch']=launch
    section('source_and_launch_identity',identity)
    if any(row['status']!='passed' for row in checks.values()):finish();return
    sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source/'scripts'))
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path,checkpoint_update_count
    from wheeled_tasks.chassis.task_campaign import account_updates,validate_task_resume
    plan=read(source/'contracts/v6_task_campaign_v18.json');new_rows={row['name']:row for row in plan['domains']}
    worker=new_rows['recovery'];config=read(source/worker['contract'])
    state=read(train/'campaign_state.json');account=read(train/'consumption.json')
    check('same_v18_plan_account_and_original_domain_parent_tree',state['contract_sha256']==account['contract_sha256']==before['contract_sha256']==settings['source']['plan_sha256'] and
        plan==read(pathlib.Path(settings['revision']['old_source_root'])/'contracts/v6_task_campaign_v18.json'))
    def preserved():
        calculated=account_updates(plan,account['attempts'])
        if calculated!={k:account[k] for k in calculated}:raise ValueError('Account arithmetic differs')
        if account['rollback_refunds_consumed_updates'] is not False:raise ValueError('Refund enabled')
        if account['consumed_updates']<origin or account['actor_updates_consumed']<before['actor_updates_consumed']:raise ValueError('Consumption rewound')
        if any(account['attempts'].get(key)!=value for key,value in before['attempts'].items()):raise ValueError('Old charged attempt changed')
        unaffected=tuple(name for name in new_rows if name!='recovery')
        if any(account['domain_consumed_updates'][name]!=before['domain_consumed_updates'][name] for name in unaffected):raise ValueError('Other domain was charged')
        if any(state['domains'].get(name)!=old_state['domains'].get(name) for name in unaffected):raise ValueError('Other domain record changed')
        for key in ('completion_history','reopening'):
            if state['domains']['recovery'].get(key)!=old_state['domains']['recovery'].get(key):raise ValueError('Existing recovery history changed: '+key)
        for name,captured in predecessor['actual_parents'].items():
            cp=pathlib.Path(captured['path']);seal=read(cp.with_suffix('.checkpoint.json'));contract=checkpoint_contract_path(cp)
            if sha(cp)!=captured['sha256'] or seal!=captured['seal'] or sha(cp.with_suffix('.checkpoint.json'))!=captured['seal_sha256']:raise ValueError('Captured own-domain parent changed: '+name)
            if sha(contract)!=captured['contract_sha256'] or checkpoint_update_count(cp)!=seal['learning_updates']:raise ValueError('Captured parent contract/clock changed: '+name)
        check('all_old_attempts_domain_records_history_and_full_parents_preserved_no_refund',True)
        evidence['consumption']=dict(before=before,current=account)
    section('preserved_old_state_and_consumption',preserved)
    def parent_identity():
        if parent.name!='model_final.pt' or parent.parent.parent.name!='attempt_003' or not parent.resolve().is_relative_to(train/'recovery'):raise ValueError('Not own full final parent')
        if sha(parent)!=bound['sha256'] or read(parent.with_suffix('.checkpoint.json'))!=bound['seal']:raise ValueError('Parent seal changed')
        if sha(checkpoint_contract_path(parent))!=worker['sha256'] or bound['contract_sha256']!=worker['sha256']:raise ValueError('Parent worker contract differs')
        validate_task_resume(read(checkpoint_contract_path(parent)),config)
        if predecessor['expected_critic_warmup']!=0 or bound['authenticated_infos']['critic_warmup_remaining']!=0:raise ValueError('Prepared parent requires critic warmup')
        evidence['actual_recovery_parent']=bound
        check('own_same_contract_full_parent_and_zero_new_warmup',True)
    section('own_same_contract_full_parent_and_zero_new_warmup',parent_identity)
    invocations=[]
    for path in (train/'recovery').glob('attempt_*/invocation.json'):
        row=read(path);command=row.get('command',[])
        if process_entry([str(value).encode() for value in command])==source/'scripts/train_chassis.py':invocations.append((path,row))
    if not invocations:pending('unique_new_source_recovery_invocation','No source020 worker invocation yet');finish();return
    if len(invocations)!=1:check('unique_new_source_recovery_invocation',False,len(invocations));finish();return
    path,invocation=invocations[0];directory=path.parent/'train';command=invocation['command']
    def argument(flag):return command[command.index(flag)+1]
    endpoint=plan['domain_completion']['schedule']['end_actor_updates']['recovery']
    expected_updates=min(worker['updates']-before['domain_consumed_updates']['recovery'],max(0,endpoint-task_origin))
    check('actual_new_attempt_own_parent_clocks_and_budget_without_new_critic_allowance',str(directory.relative_to(train)) not in before['attempts'] and
        pathlib.Path(argument('--run-dir'))==directory and directory.parent.name=='attempt_004' and
        invocation['source_checkpoint']==str(parent) and invocation['source_checkpoint_sha256']==bound['sha256'] and
        int(argument('--consumed-updates'))==origin and int(argument('--task-actor-update'))==task_origin and
        int(argument('--stage-actor-update'))==bound['seal']['stage_actor_update'] and
        int(argument('--actor-updates-consumed'))==before['actor_updates_consumed'] and
        argument('--contract')==str(source/worker['contract']) and argument('--resume')==str(parent) and
        invocation['requested_updates']==expected_updates and invocation['scheduled_end_actor_update']==endpoint)
    evidence.update(actual_worker_directory=str(directory),invocation=invocation)
    def runtime():
        resume=read(directory/'resume_verification.json');startup=read(directory/'startup.json');progress=read(directory/'progress.json')
        pinned=startup['source_sha256'];adapter='src/wheeled_tasks/chassis/root_kinematics.py'
        if pinned.get(adapter)!=settings['source']['changed_runtime_files'][adapter]['candidate']:raise ValueError('Actual PPO adapter source pin differs')
        if not pinned:raise ValueError('Missing actual source snapshot')
        for name,digest in pinned.items():
            rel=pathlib.PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe source snapshot path')
            if sha(source/name)!=digest or sha(directory/'source'/name)!=digest:raise ValueError('Actual PPO source snapshot differs: '+name)
        check('actual_PPO_root_adapter_and_every_snapshot_byte_authenticated',True,dict(files=len(pinned),root_adapter_sha256=pinned[adapter]))
        native=startup['startup'];preflight=native['runtime_preflight']
        if preflight['status']!='runtime_configuration_passed':raise ValueError('Native runtime preflight not passed')
        observed=preflight['observed']
        from wheeled_tasks.chassis.preflight import validate_runtime_state
        from wheeled_tasks.chassis.ordinary_posture_curriculum import runtime_expectation
        validate_runtime_state(native['design_preflight']['runtime_expected'],observed)
        evidence.update(resume_verification=resume,progress=progress,runtime_preflight=preflight)
        check('actual_native_actor_critic_adam_full_restore_receipt',all(resume.get(k) is True for k in ('actor_exact','critic_exact','optimizer_exact')) and
            resume['parent_checkpoint_sha256']==bound['sha256'] and resume['parent_updates']==origin and
            resume['restored_checkpoint_updates']==restored_updates and resume['task_clock']=='restored' and
            resume['source_contract_sha256']==resume['target_contract_sha256']==worker['sha256'] and
            resume['effective_learning_rate']==resume['restored_optimizer_learning_rate']==1e-4)
        groups=native['scene_group_counts'];pools=native['behavior_pool_counts']
        check('actual_4096_population_matches_new_source_engineering',sum(groups.values())==4096 and min(groups.values())>0 and
            groups==settings['engineering']['startup']['scene_group_counts'] and pools==settings['engineering']['startup']['behavior_pool_counts'])
        check('actual_native_35_81_6_200Hz_50Hz_control_delay_rewards_modules',progress['num_envs']==4096 and
            (config['actor_dim'],config['critic_dim'],config['action_dim'])==(35,81,6) and
            (observed['physics_dt'],observed['policy_dt'],observed['decimation'],observed['pc_control_dt'])==(.005,.02,4,.005) and
            all(observed['modules'][key] for key in ('command_transport','signal_delay','recovery_training','recovery_prefix_curriculum','stand_still_lin_vel','locomotion_posture_floor','ordinary_posture_curriculum')) and
            observed['seed']==int(argument('--seed'))==plan['seed'] and int(argument('--block-index'))==0)
        posture=observed['ordinary_posture_curriculum'];actual=posture['runtime_observed']
        check('actual_constructor_ordinary_population1449_geometry_timers',
            {k:v for k,v in posture.items() if k!='runtime_observed'}==runtime_expectation(config['ordinary_posture_curriculum']) and
            actual['eligible_count']==actual['birth_eligible_population']==1449 and actual['geometry_bound'] is True and
            actual['birth_selected_count']>0 and 0<actual['birth_abs_angle_max_rad']<=actual['birth_angle_cap_rad']+1e-6 and
            5<=actual['push_clock_min_s']<=actual['push_clock_max_s']<=10 and actual['push_clock_std_s']>0 and
            config.get('nonstandard_start') is None)
        evidence['constructor_posture_snapshot']=dict(value=posture,scope='constructor_before_checkpoint_clock_restore; actual_resumed_ramp_checked_in_PPO_events')
        check('same_contract_no_critic_only_updates_and_restored_actor_clock',progress['parent_updates']==origin and
            progress['successful_updates']==progress['actor_updates_in_block'] and progress['task_actor_update']==task_origin+progress['actor_updates_in_block'] and
            progress['stage_actor_update']==bound['seal']['stage_actor_update']+progress['actor_updates_in_block'])
        check('fixed_curriculum_origins_retained_separate_from_restored_actor_clock',
            posture['origin_actor_update']==config['ordinary_posture_curriculum']['origin_actor_update'] and
            observed['recovery_prefix_curriculum']['settings']['origin_actor_update']==config['recovery_training']['prefix_curriculum']['origin_actor_update'])
        admission=read(source/config['actuator_response_evidence']['file'])
        check('wheel_response_retained_no_qualified_leg_response',admission['adopted_joint_names']==['L_joint3','R_joint3'] and
            admission['leg_response'].startswith('candidate_not_adopted') and observed['actuator_response']==config['actuator_response'])
        evidence['live_PPO_substep_capture']=dict(coverage='not_persisted_in_startup_written_before_first_step',
            actual_assertion='SubstepControlAudit.end_policy runs in every env.step; exact frozen source authenticated')
    section('native_runtime_and_progress',runtime)
    engineering=settings['engineering'];proof=engineering['ordinary_posture_native_proof'];control=engineering['control_timing']
    check('separate_source020_native1449_quaternion_actual_push_control_probe',engineering['status']=='engineering_probe_completed' and
        engineering['target_contract_sha256']==worker['sha256'] and engineering['formal_updates_consumed']==engineering['optimizer_steps']==0 and
        engineering['policy_steps']==600 and engineering['physics_steps']==2400 and
        proof['initial']['eligible_count']==proof['final']['eligible_count']==1449 and
        proof['initial']['native_root_quaternion_readback_passed'] is proof['actual_root_rotation_observed'] is proof['actual_native_velocity_delta_observed'] is True and
        control['pd_updates']==control['feedback_refreshes']==4 and control['target_decodes']==1 and
        (control['physics_dt'],control['policy_dt'])==(.005,.02),dict(scope='separate_frozen_zero_learning_engineering_not_actual_PPO_substep_capture'))
    def learning():
        import torch
        torch.set_num_threads(1)
        def equal(a,b):
            if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
            if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        def finite(value):
            if torch.is_tensor(value):return bool(torch.isfinite(value).all())
            if isinstance(value,dict):return all(finite(v) for v in value.values())
            if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
            return not isinstance(value,float) or math.isfinite(value)
        def load(path):
            seal=read(path.with_suffix('.checkpoint.json'));digest=sha(path)
            if not seal['sealed'] or digest!=seal['checkpoint_sha256'] or sha(checkpoint_contract_path(path))!=seal['contract_sha256']:raise ValueError('Checkpoint seal differs')
            if seal['contract_sha256']!=worker['sha256']:raise ValueError('Checkpoint worker differs')
            saved=torch.load(path,map_location='cpu',weights_only=False);infos=saved['infos']
            for key in ('task_actor_update','stage_actor_update','actor_updates_consumed','learning_lineage_updates'):
                if infos[key]!=seal[key]:raise ValueError('Checkpoint infos/clock differs: '+key)
            if infos['contract_sha256']!=worker['sha256'] or infos['successful_updates_total']!=seal['learning_updates'] or infos['task_campaign']['domain']!='recovery':raise ValueError('Checkpoint infos provenance differs')
            if not all(finite(saved[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')):raise ValueError('Nonfinite learning state')
            for key in ('rng_state','env_rng_state'):
                if not torch.is_tensor(infos.get(key)) or infos[key].dtype!=torch.uint8 or infos[key].numel()==0:raise ValueError('Missing full RNG state: '+key)
            if not isinstance(infos.get('cuda_rng_states'),list) or not infos['cuda_rng_states'] or not all(torch.is_tensor(x) and x.dtype==torch.uint8 and x.numel()>0 for x in infos['cuda_rng_states']):raise ValueError('Missing full CUDA RNG state')
            evidence.setdefault('immutable_checkpoints',{})[str(path)]=dict(sha256=digest,seal=seal)
            return saved
        old=load(parent)
        evidence['RNG_restore_scope']='Parent/initial/first checkpoint RNG presence and frozen trainer restore path; reset/resampling consumes RNG before initial_state, so no false post-reset bitwise identity claim.'
        initial=None
        def initial_check():
            nonlocal initial
            initial=load(directory/'initial_state/model_final.pt')
            exact={k:equal(old[k],initial[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')}
            check('initial_actor_critic_adam_exact_zero_warmup',all(exact.values()) and initial['infos']['critic_warmup_remaining']==0 and
                initial['infos']['successful_updates_total']==origin and initial['infos']['task_actor_update']==task_origin and
                initial['infos']['actor_updates_consumed']==before['actor_updates_consumed'],exact)
        section('initial_actor_critic_adam_exact_zero_warmup',initial_check)
        def first_actor():
            saved=load(directory/f'checkpoints/update_{origin+1:08d}/model.pt');baseline=old if initial is None else initial
            changed=[key for key,value in baseline['actor_state_dict'].items() if not torch.equal(value,saved['actor_state_dict'][key])]
            evidence['first_actor_changed_tensors']=changed
            check('first_update_changes_all_nine_actor_tensors_critic_and_adam_without_new_warmup',len(changed)==len(baseline['actor_state_dict'])==9 and
                not equal(baseline['critic_state_dict'],saved['critic_state_dict']) and not equal(baseline['optimizer_state_dict'],saved['optimizer_state_dict']) and
                saved['infos']['critic_warmup_remaining']==0 and saved['infos']['task_actor_update']==task_origin+1 and
                saved['infos']['stage_actor_update']==bound['seal']['stage_actor_update']+1 and
                saved['infos']['successful_updates_total']==origin+1 and saved['infos']['actor_updates_consumed']==before['actor_updates_consumed']+1 and
                saved['infos']['learning_lineage_updates']==bound['seal']['learning_lineage_updates']+1)
        section('first_update_changes_all_nine_actor_tensors_critic_and_adam_without_new_warmup',first_actor)
    section('actual_checkpoint_parameter_updates',learning)
    def ownership():
        rows=[];kinds={'chassis_remote_job.py':'wrapper','run_task_campaign.py':'supervisor','train_chassis.py':'worker','evaluate_chassis.py':'evaluator'}
        for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args=path.read_bytes().split(b'\0');entry=process_entry(args)
                if not entry or entry.name not in kinds:continue
                if not any(x==str(root).encode() or x==str(train).encode() or x.startswith(str(train).encode()+b'/') for x in args):continue
                ppid=int(next(line.split()[1] for line in (path.parent/'status').read_text().splitlines() if line.startswith('PPid:')))
                rows.append(dict(pid=int(path.parent.name),ppid=ppid,kind=kinds[entry.name],entry_script=str(entry),command=[x.decode() for x in args if x]))
            except (OSError,PermissionError,StopIteration):continue
        evidence['campaign_python_entry_owners']=rows
        if not rows:pending('unique_native_owner_chain','No live owner; artifacts do not prove current ownership');return
        valid=sorted(row['kind'] for row in rows)==['supervisor','worker','wrapper']
        if valid:
            by_kind={row['kind']:row for row in rows};names=dict(wrapper='chassis_remote_job.py',supervisor='run_task_campaign.py',worker='train_chassis.py')
            valid=(by_kind['worker']['ppid']==by_kind['supervisor']['pid'] and by_kind['supervisor']['ppid']==by_kind['wrapper']['pid'] and
                all(row['entry_script']==str(source/'scripts'/names[row['kind']]) for row in rows))
            valid=valid and by_kind['supervisor']['command']==settings['revision']['command'] and by_kind['worker']['command']==command and by_kind['supervisor']['pid']==evidence['job']['pid']
        check('unique_native_owner_chain',valid,rows)
    section('unique_native_owner_chain',ownership)
    def events():
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
        paths=list(directory.glob('events.out.tfevents.*'))
        if not paths: pending('native_crc_reward_delay_timing','No native event file yet');return
        event=max(paths,key=lambda p:p.stat().st_mtime_ns); size=event.stat().st_size; offset=max(0,size-settings['tail_bytes'])
        with event.open('rb') as stream: stream.seek(offset);data=stream.read(settings['tail_bytes'])
        pattern=re.compile(rb'(?:LearningAudit/[A-Za-z0-9_]+/reward_density/(?:stand_still_lin_vel|locomotion_pitch_floor|locomotion_roll_floor|pitch_velocity|roll_velocity|horizontal_angular_velocity|pitch_exp|roll_exp)|Timing/[A-Za-z0-9_]+|(?:Episode/)?/(?:delay|transport|ordinary_posture)/[A-Za-z0-9_]+)')
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
        # Floors add only a coefficient deficit. Zero payment is correct when
        # the original posture cost is already stronger than that floor.
        costs=all(any(tag.endswith('/'+key) and row['minimum']<0 for tag,row in stats.items())
            for key in ('stand_still_lin_vel','pitch_velocity','roll_velocity','horizontal_angular_velocity'))
        floor_keys=('locomotion_pitch_floor','locomotion_roll_floor')
        floor_rows=[row for tag,row in stats.items() if any(tag.endswith('/'+key) for key in floor_keys)]
        floors=all(any(tag.endswith('/'+key) for tag in stats) for key in floor_keys) and all(
            row['maximum']<=0 for row in floor_rows)
        if any(row['maximum']>0 for row in floor_rows):
            check('native_crc_reward_delay_timing',False,'A deficit-floor cost paid a positive reward');return
        positive_posture=all(any(tag.endswith('/'+key) and row['maximum']>0 for tag,row in stats.items())
            for key in ('pitch_exp','roll_exp'))
        delay=any('/delay/' in tag for tag in stats) and any('/transport/' in tag for tag in stats)
        timing=all('Timing/'+name in stats for name in ('iteration_wall_seconds','env_step_seconds','optimizer_seconds'))
        posture_tags={tag:row for tag,row in stats.items() if '/ordinary_posture/' in tag}
        evidence['actual_PPO_posture_scalars']=posture_tags
        fields=('eligible_count','birth_eligible_population','relative_actor_updates','birth_angle_cap_rad',
            'push_speed_cap_m_s','birth_selected_count','birth_abs_angle_max_rad','geometry_bound',
            'push_clock_min_s','push_clock_max_s','push_clock_std_s','push_events','push_velocity_delta_max_m_s')
        present=all(any(tag.endswith('/'+key) for tag in posture_tags) for key in fields)
        effect=all(any(tag.endswith('/'+key) and row['maximum']>0 for tag,row in posture_tags.items())
            for key in ('push_events','push_velocity_delta_max_m_s','birth_selected_count','birth_abs_angle_max_rad'))
        live_fields=('eligible_count','birth_eligible_population','relative_actor_updates','birth_angle_cap_rad','push_speed_cap_m_s',
            'geometry_bound','push_clock_min_s','push_clock_max_s','push_clock_std_s')
        live_series={key:next((values for tag,values in series.items() if '/ordinary_posture/' in tag and tag.endswith('/'+key)),{}) for key in live_fields}
        common=set.intersection(*(set(values) for values in live_series.values()))
        if not common:pending('actual_PPO_population1449_restored_clock_ramp_geometry_timers','No complete CRC-valid ordinary metric tick yet')
        else:
            step=max(common);values={key:rows[step] for key,rows in live_series.items()}
            curriculum=config['ordinary_posture_curriculum'];ramp=curriculum['ramp'];relative=values['relative_actor_updates']
            alpha=min(1.,max(0.,(relative-ramp['start_actor_update'])/(ramp['end_actor_update']-ramp['start_actor_update'])))
            expected_angle=curriculum['birth']['start_roll_pitch_rad']+alpha*(curriculum['birth']['end_roll_pitch_rad']-curriculum['birth']['start_roll_pitch_rad'])
            expected_speed=curriculum['push']['start_speed_m_s']+alpha*(curriculum['push']['end_speed_m_s']-curriculum['push']['start_speed_m_s'])
            check('actual_PPO_population1449_restored_clock_ramp_geometry_timers',values['eligible_count']==values['birth_eligible_population']==1449 and
                relative>=max(0,task_origin-curriculum['origin_actor_update']) and values['geometry_bound']==1 and
                math.isclose(values['birth_angle_cap_rad'],expected_angle,abs_tol=1e-6) and math.isclose(values['push_speed_cap_m_s'],expected_speed,abs_tol=1e-6) and
                5<=values['push_clock_min_s']<=values['push_clock_max_s']<=10 and values['push_clock_std_s']>0,
                dict(event_step=step,actual=values,expected_angle_cap_rad=expected_angle,expected_speed_cap_m_s=expected_speed))
        if present and effect:check('actual_PPO_recurring_push_and_root_posture_metrics',True)
        else:pending('actual_PPO_recurring_push_and_root_posture_metrics',dict(all_actual_fields_present=present,
            positive_native_effect_observed=effect,coverage='bounded_native_CRC_tail; wait for a completed push interval'))
        if costs and floors and positive_posture and delay and timing:check('native_crc_reward_delay_timing',True)
        else:pending('native_crc_reward_delay_timing',dict(negative_base_costs_present=costs,nonpositive_deficit_floor_terms_present=floors,positive_posture_rewards=positive_posture,delay_and_transport=delay,timing=timing))
    section('native_crc_reward_delay_timing',events)
    finish()
'''


def settings_for(args):
    evidence = load_evidence({key:getattr(args,key) for key in (
        'source_candidate','clean_tests','clean_30_roles','remote_30_roles','native_domains',
        'semantic_review','paired_summary','integration_review','integration_native_review',
        'baseline_native_jump','baseline_native_stepjump')})
    source=evidence['source'];revision=read(args.revision);predecessor=read(args.predecessor)
    assert revision['schema']=='task-source-revision-v1'
    assert revision['source_commit']==predecessor['source_commit']==source['source_commit']==COMMIT
    assert revision['old_source_root']==RUN+'/source_712d701473f3'
    assert revision['new_source_root']==source['source_remote'] and revision['source_archive_sha256']==source['archive_sha256']
    assert revision['old_contract_sha256']==revision['new_contract_sha256']==predecessor['consumption']['contract_sha256']==source['plan_sha256']
    assert predecessor['evidence_sha256']==evidence['evidence_sha256'] and predecessor['expected_critic_warmup']==0
    parent=Path(predecessor['actual_parents']['recovery']['path'])
    assert parent.name=='model_final.pt' and parent.parent.parent.name=='attempt_003'
    report=read(args.native_domains)['domains']['recovery']
    engineering={k:report[k] for k in ('status','target_contract_sha256','formal_updates_consumed','optimizer_steps',
        'policy_steps','physics_steps','ordinary_posture_native_proof','control_timing')}
    engineering['startup']={k:report['startup'][k] for k in ('scene_group_counts','behavior_pool_counts')}
    return dict(run=RUN,source=source,revision=revision,predecessor=predecessor,evidence_sha256=evidence['evidence_sha256'],
        engineering=engineering,engineering_limitations=evidence['engineering_limitations'],tail_bytes=args.tail_mib*1024**2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-remote-code',action='store_true',required=True)
    parser.add_argument('--tail-mib',type=int,default=8,choices=range(1,33))
    for flag,name in (('source-candidate','source_candidate_020.json'),('clean-tests','clean_tests_020.json'),
        ('clean-30-roles','clean_30_roles_020.json'),('remote-30-roles','remote_30_roles_020.json'),
        ('native-domains','native_domains_020.json'),('revision','source_revision_020.json'),('predecessor','predecessor_020.json')):
        parser.add_argument('--'+flag,type=Path,default=OUT/name)
    performance=OUT.parent/'performance/com_static_candidate_020'
    for flag,name in (('semantic-review','native_reviews/semantic_001.review.json'),
        ('paired-summary','paired_summary_independent_CPU.json'),('integration-review','integration_review_independent_CPU.json'),
        ('integration-native-review','integration_fixed_native_001.review.json')):
        parser.add_argument('--'+flag,type=Path,default=performance/name)
    verification=OUT.parent/'verification'
    for flag,name in (('baseline-native-jump','native_jump_018_final.report.json.gz'),
        ('baseline-native-stepjump','native_stepjump_018_final.report.json.gz')):
        parser.add_argument('--'+flag,type=Path,default=verification/name)
    args=parser.parse_args();settings=settings_for(args)
    print(REMOTE+'\ninspect(json.loads('+repr(json.dumps(settings))+'))\n')


if __name__=='__main__':
    main()
