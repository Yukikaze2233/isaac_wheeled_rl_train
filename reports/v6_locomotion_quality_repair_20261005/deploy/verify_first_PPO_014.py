#!/usr/bin/env python3
"""One bounded, read-only SSH inspection of v14's first real PPO updates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
COMMIT = "b4ce80964d47c0127e0b3d9929ad35a620bbf204"
PLAN_SHA = "41319993aa53a54e32efaab725ec8b0e269f4b8b3576bc101f23c6e525c4cf73"
PARENT_SHA = "123841b609439939e9c9c8962c0171330892276446a692c936c8f33375ce11da"

REMOTE = r'''
from datetime import datetime, timezone
import hashlib, json, math, pathlib, re, struct

def process_entry_kind(arguments, source):
    # The launch wrapper carries its child's full argv after "--". Only the
    # first actual Python script is this process's entry point.
    entry=None
    for value in arguments[1:]:
        if value in (b'-c',b'-m'): return None
        if value.endswith(b'.py'):
            entry=value; break
    for script,kind in (('chassis_remote_job.py','wrapper'),('run_task_campaign.py','supervisor'),('train_chassis.py','worker')):
        if entry==str(source/'scripts'/script).encode(): return kind
    return None

def inspect(settings):
    root=pathlib.Path(settings['run']); train=root/'train'
    source=pathlib.Path(settings['source']['source_remote'])
    result=dict(schema='v14-first-ppo-readonly-verification-v1', verified_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=settings['commit'], qualification=False, scope='full_state_restore_and_real_optimizer_updates_only', checks={}, evidence={})
    checks,evidence=result['checks'],result['evidence']
    def check(name, passed, detail=None):
        checks[name]=dict(status='passed' if passed else 'failed', detail=detail)
    def pending(name, detail): checks[name]=dict(status='pending', detail=str(detail))
    def read(path): return json.loads(path.read_text())
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
        return h.hexdigest()
    def section(name, function):
        try: function()
        except FileNotFoundError as error: pending(name,error)
        except Exception as error: checks[name]=dict(status='failed',detail=f'{type(error).__name__}: {error}')
    def finish():
        statuses=[item['status'] for item in checks.values()]
        result['status']='failed' if 'failed' in statuses else 'pending' if 'pending' in statuses else 'first_actual_actor_PPO_verified'
        result['check_counts']={name:statuses.count(name) for name in ('passed','pending','failed')}
        print(json.dumps(result,allow_nan=False))
    def identity():
        revision=read(root/'source_revision_014.json')
        check('frozen_source_identity', revision['source_commit']==settings['commit'] and
            revision['new_source_root']==str(source) and revision['new_contract_sha256']==settings['plan_sha'] and
            sha(source/'contracts/v6_task_campaign_v14.json')==settings['plan_sha'] and
            sha(pathlib.Path(revision['source_archive']))==settings['source']['archive_sha256'])
    section('frozen_source_identity',identity)
    try:
        state=read(train/'campaign_state.json'); account=read(train/'consumption.json')
        plan=read(source/'contracts/v6_task_campaign_v14.json')
    except FileNotFoundError as error:
        pending('campaign_state',error); finish(); return
    if state.get('contract_sha256')==account.get('contract_sha256')==settings['old_plan_sha']:
        pending('v14_campaign_account','Authoritative campaign still belongs to deployed v12; v14 PPO has not started.')
    else:
        check('v14_campaign_account',state.get('contract_sha256')==settings['plan_sha'] and account.get('contract_sha256')==settings['plan_sha'])
    old_state=settings['old_state']; old_plan=settings['old_plan']
    unaffected=('flat','recovery','jump','stepjump')
    check('four_other_workers_and_parent_tree_unchanged', all(
        a==b if a['name'] in unaffected else a['parent_domain']==b['parent_domain']
        for a,b in zip(old_plan['domains'],plan['domains'])) and
        all(state['domains'].get(name)==old_state['domains'].get(name) for name in unaffected))
    def history():
        before=read(train/'revisions/v12_account_before_migration.json')
        check('historical_consumption_no_refund',before['consumed_updates']==23868 and
            all(account['attempts'].get(key)==value for key,value in before['attempts'].items()) and
            all(account['domain_consumed_updates'][name]==before['domain_consumed_updates'][name] for name in unaffected))
        evidence['domain_consumption']=account['domain_consumed_updates']
        evidence['campaign_consumed_updates']=account['consumed_updates']
    section('historical_consumption_no_refund',history)
    # Discover actual v14 invocation from the migrated state and its own-domain
    # invocation files. No attempt number is invented or assumed.
    speed=state['domains'].get('speed',{})
    history_record=(speed.get('completion_history') or [{}])[-1].get('record',speed)
    parent=pathlib.Path(history_record.get('candidate_checkpoint',''))
    candidates=[]
    for path in (train/'speed').glob('attempt_*/invocation.json'):
        invocation=read(path); command=invocation.get('command',[])
        if len(command)>2 and command[2]==str(source/'scripts/train_chassis.py') and '--contract' in command and command[command.index('--contract')+1]==str(source/'contracts/v6_task_campaign_speed_worker_v14.json'):
            candidates.append((path,invocation))
    if not candidates:
        pending('v14_speed_invocation','No actual v14 speed invocation exists; PPO may not have started.'); finish(); return
    path,invocation=max(candidates,key=lambda item:item[0].stat().st_mtime_ns)
    directory=path.parent/'train'; evidence['actual_worker_directory']=str(directory)
    evidence['invocation']=invocation
    command=invocation['command']
    check('own_speed_parent_and_consumption_origin',str(parent)==invocation['source_checkpoint'] and
        invocation['source_checkpoint_sha256']==settings['parent_sha'] and sha(parent)==settings['parent_sha'] and
        command[command.index('--consumed-updates')+1]=='23868' and
        command[command.index('--task-actor-update')+1]=='3563' and invocation['requested_updates']==1510)
    config=read(source/'contracts/v6_task_campaign_speed_worker_v14.json')
    def runtime():
        resume=read(directory/'resume_verification.json'); startup=read(directory/'startup.json')
        progress=read(directory/'progress.json'); observed=startup['startup']['runtime_preflight']['observed']
        evidence.update(resume_verification=resume,progress=progress,runtime_preflight=startup['startup']['runtime_preflight'])
        check('full_native_restore_receipt',all(resume.get(key) is True for key in ('actor_exact','critic_exact','optimizer_exact')) and
            resume['parent_checkpoint_sha256']==settings['parent_sha'] and resume['parent_updates']==23868 and
            resume['task_clock']=='restored')
        check('runtime_4096_35_81_and_feedback_timing',progress['num_envs']==4096 and
            (config['actor_dim'],config['critic_dim'])==(35,81) and
            (observed['physics_dt'],observed['policy_dt'],observed['decimation'],observed['pc_control_dt'])==(.005,.02,4,.005) and
            observed['modules']['command_transport'] and observed['modules']['signal_delay'] and
            observed['modules']['stand_still_lin_vel'] and observed['modules']['locomotion_posture_floor'] and
            observed['locomotion_difficulty']['settings']==config['locomotion_difficulty'])
        check('critic_adaptation_counted_once',progress['parent_updates']==23868 and
            progress['successful_updates']-progress['actor_updates_in_block']==min(10,progress['successful_updates']) and
            progress['task_actor_update']==3563+progress['actor_updates_in_block'])
        admission=read(source/config['actuator_response_evidence']['file'])
        check('wheel_response_retained_leg_response_unqualified',admission['adopted_joint_names']==['L_joint3','R_joint3'] and
            admission['leg_response'].startswith('candidate_not_adopted') and
            observed['actuator_response']==config['actuator_response'])
    section('native_runtime_and_progress',runtime)
    def learning():
        import torch
        torch.set_num_threads(1)
        def equal(a,b):
            if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
            if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        def load(path,expected_sha=None):
            seal=read(path.with_suffix('.checkpoint.json'))
            actual_sha=sha(path)
            if not seal['sealed'] or actual_sha!=seal['checkpoint_sha256'] or expected_sha and actual_sha!=expected_sha:
                raise ValueError('Immutable checkpoint hash/seal mismatch: '+str(path))
            expected_contract=sha(source/'contracts/v6_task_campaign_speed_worker_v14.json') if expected_sha is None else sha(path.with_suffix('.contract.json'))
            if seal['contract_sha256']!=expected_contract or sha(path.with_suffix('.contract.json'))!=expected_contract:
                raise ValueError('Immutable contract seal mismatch: '+str(path))
            evidence.setdefault('immutable_checkpoints',{})[str(path)]=dict(sha256=actual_sha,seal=seal)
            return torch.load(path,map_location='cpu',weights_only=False)
        old=load(parent,settings['parent_sha']); initial=load(directory/'initial_state/model_final.pt')
        exact={key:equal(old[key],initial[key]) for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict')}
        check('initial_actor_critic_adam_exact_values',all(exact.values()) and initial['infos']['successful_updates_total']==23868 and
            initial['infos']['task_actor_update']==3563 and initial['infos']['critic_warmup_remaining']==10,exact)
        critic=load(directory/'checkpoints/update_00023869/model.pt')
        check('first_critic_actual_update_actor_unchanged',equal(old['actor_state_dict'],critic['actor_state_dict']) and
            not equal(old['critic_state_dict'],critic['critic_state_dict']) and critic['infos']['task_actor_update']==3563 and
            critic['infos']['successful_updates_total']==23869 and critic['infos']['critic_warmup_remaining']==9)
        actor=load(directory/'checkpoints/update_00023879/model.pt')
        changed=[key for key,value in initial['actor_state_dict'].items() if not torch.equal(value,actor['actor_state_dict'][key])]
        evidence['first_actor_changed_tensors']=changed
        check('first_actor_actual_update_task_3564',bool(changed) and actor['infos']['task_actor_update']==3564 and
            actor['infos']['successful_updates_total']==23879 and actor['infos']['critic_warmup_remaining']==0 and
            all(torch.isfinite(v).all().item() for key in ('actor_state_dict','critic_state_dict') for v in actor[key].values()))
    section('actual_checkpoint_parameter_updates',learning)
    def ownership():
        owners=[]; wrappers=[]
        for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
            try: args=path.read_bytes().split(b'\0')
            except (OSError,PermissionError): continue
            kind=process_entry_kind(args,source)
            if kind and any(value==str(train).encode() or value.startswith(str(train).encode()+b'/') for value in args):
                try: ppid=int(next(line.split()[1] for line in (path.parent/'status').read_text().splitlines() if line.startswith('PPid:')))
                except (OSError,StopIteration): continue
                row=dict(pid=int(path.parent.name),ppid=ppid,kind=kind,
                    entry_script=next(value.decode() for value in args[1:] if value.endswith(b'.py')))
                (wrappers if kind=='wrapper' else owners).append(row)
        evidence['owned_processes']=owners
        evidence['launch_wrappers']=wrappers
        if not owners: pending('single_owned_worker_and_supervisor','No live v14 campaign owners present')
        else:
            valid=sorted(x['kind'] for x in owners)==['supervisor','worker'] and len(wrappers)<=1
            if valid:
                supervisor=next(row for row in owners if row['kind']=='supervisor')
                worker=next(row for row in owners if row['kind']=='worker')
                valid=worker['ppid']==supervisor['pid'] and (not wrappers or supervisor['ppid']==wrappers[0]['pid'])
            check('single_owned_worker_and_supervisor',valid,dict(owner_count=len(owners),wrapper_count=len(wrappers),
                classification='actual_python_entry_point_not_embedded_child_command'))
    section('single_owned_worker_and_supervisor',ownership)
    def feedback_audit():
        probe=read(root/'nonstandard_speed_probe_014/report.json')
        # Startup.json is written before the first PPO step. The frozen native
        # probe retains the actual per-step audit; do not relabel the startup
        # pending marker as observed PPO substeps.
        audit=probe['startup']['control_timing']['runtime_substep_verification']
        check('frozen_native_four_fresh_feedback_PD_updates',probe['status']=='passed' and
            audit['pd_updates']==audit['feedback_refreshes']==4 and audit['target_decodes']==1 and
            (audit['physics_dt'],audit['policy_dt'])==(.005,.02),audit)
        payment=probe['locomotion_quality_payments']
        check('frozen_native_real_reward_payments',payment['status']=='actual_calls_components_and_dt_scaled_payments_verified' and
            payment['steps']==probe['policy_steps'] and all(payment['negative_lane_frames'][key]>0 for key in
                ('stand_still_lin_vel','locomotion_pitch_floor','locomotion_roll_floor')))
    section('native_feedback_and_payment_audit',feedback_audit)
    def events():
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
        paths=list(directory.glob('events.out.tfevents.*'))
        if not paths: pending('native_crc_event_window','No native event file yet'); return
        event=max(paths,key=lambda p:p.stat().st_mtime_ns); size=event.stat().st_size
        offset=max(0,size-settings['tail_bytes'])
        with event.open('rb') as stream: stream.seek(offset); data=stream.read(settings['tail_bytes'])
        pattern=re.compile(rb'(?:LearningAudit/[A-Za-z0-9_]+/reward_density/(?:stand_still_lin_vel|locomotion_pitch_floor|locomotion_roll_floor)|Timing/[A-Za-z0-9_]+|(?:Episode/)?/(?:curriculum/locomotion|delay|transport)/[A-Za-z0-9_]+)')
        seen=set(); series={}; unresolved=0
        for match in pattern.finditer(data):
            located=False
            for pos in range(max(0,match.start()-128),max(0,match.start()-11)):
                length,crc=struct.unpack_from('<QI',data,pos); end=pos+16+length
                if not 0<length<1024**2 or end>len(data) or end-4<match.end(): continue
                if masked_crc32c(data[pos:pos+8])!=crc: continue
                payload=data[pos+12:end-4]
                if masked_crc32c(payload)!=struct.unpack_from('<I',data,end-4)[0]: continue
                located=True
                if pos in seen: break
                seen.add(pos); item=Event.FromString(payload)
                for value in item.summary.value:
                    if pattern.fullmatch(value.tag.encode()) and value.HasField('simple_value'):
                        number=float(value.simple_value)
                        if not math.isfinite(number): raise ValueError('Nonfinite native scalar '+value.tag)
                        series.setdefault(value.tag,{})[int(item.step)]=number
                break
            unresolved+=int(not located)
        stats={tag:dict(samples=len(values),first_step=min(values),last_step=max(values),minimum=min(values.values()),
            maximum=max(values.values()),latest=values[max(values)]) for tag,values in series.items()}
        evidence['native_crc_event_window']=dict(file=str(event),file_bytes=size,read_offset=offset,read_bytes=len(data),
            coverage='bounded_tail_records_only_not_a_reservoir_or_full_run',crc_valid_records=len(seen),unlocated_matches=unresolved,scalars=stats)
        required=('stand_still_lin_vel','locomotion_pitch_floor','locomotion_roll_floor')
        present=all(any(tag.endswith('/'+key) and value['minimum']<0 for tag,value in stats.items()) for key in required)
        grade=any('/curriculum/locomotion/grade' in tag for tag in stats)
        delay=any('/delay/' in tag for tag in stats) and any('/transport/' in tag for tag in stats)
        timing='Timing/iteration_wall_seconds' in stats and 'Timing/env_step_seconds' in stats and 'Timing/optimizer_seconds' in stats
        if present and grade and delay and timing: check('native_crc_reward_grade_delay_timing',True)
        else: pending('native_crc_reward_grade_delay_timing',dict(negative_costs_present=present,grade=grade,delay=delay,timing=timing))
    section('native_crc_event_window',events)
    finish()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='kaiser@192.168.64.234')
    parser.add_argument('--port', type=int, default=2222)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--tail-mib', type=int, default=8, choices=range(1,33))
    parser.add_argument('--output', type=Path, default=OUT/'first_PPO_verification_014.json')
    parser.add_argument('--emit-remote-code', action='store_true', help='Print code for local syntax inspection; do not connect')
    args = parser.parse_args()
    settings=dict(run=RUN,commit=COMMIT,plan_sha=PLAN_SHA,parent_sha=PARENT_SHA,tail_bytes=args.tail_mib*1024**2,
        old_plan_sha='e79e94872f041ddf935ad961c59093b827b0b50f3519c9234718118cf1b59298',
        source=json.loads((OUT/'source_candidate_014.json').read_text()),
        old_plan=json.loads((REPO/'contracts/v6_task_campaign_v12.json').read_text()),
        old_state=json.loads((REPO/'reports/v6_nonstandard_locomotion_20261005/deploy/predecessor_state_013.json').read_text())['campaign_state'])
    code=REMOTE+'\ninspect(json.loads('+repr(json.dumps(settings))+'))\n'
    if args.emit_remote_code:
        print(code); return
    command='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && '+shlex.join([BASE+'/env/bin/python','-B','-'])
    try:
        remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p',str(args.port),args.host,
            shlex.join(['bash','-lc',command])],input=code,capture_output=True,text=True,timeout=args.timeout)
        if remote.returncode: raise RuntimeError(remote.stderr[-2000:])
        report=json.loads(remote.stdout)
    except (subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError) as error:
        report=dict(schema='v14-first-ppo-readonly-verification-v1',status='pending',qualification=False,
            verified_at_utc=datetime.now(timezone.utc).isoformat(),checks=dict(remote_read=dict(status='pending',detail=str(error))))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=report.get('check_counts'),output=str(args.output),
        outstanding={key:value for key,value in report['checks'].items() if value['status']!='passed'}),ensure_ascii=False))


if __name__=='__main__': main()
