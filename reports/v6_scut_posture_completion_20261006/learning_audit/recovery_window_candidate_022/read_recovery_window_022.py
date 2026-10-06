#!/usr/bin/env python3
"""Source022-only identity adapter; exact021/020/019 CRC and metrics are reused.

Read-only existing native TB. No SSH, simulator, model, optimizer or TB server.
First100 actor default: native critic10 baseline, events30812..30911.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD021 = HERE.parent / 'recovery_window_candidate_021/read_recovery_window_021.py'
OLD020 = HERE.parent / 'read_recovery_window_020.py'
OLD019 = HERE.parent / 'read_recovery_window_019.py'
READER021_SHA = '5faa75e24b0dd484bdacb0273756642caab8fcc6d6e66f90c034a0bda6ac19a6'
FIRST_PPO_SHA = '3c27939ee067ef776249290e9324ddef029e8b66e397a6528905bf289d76d9e7'
INVENTORY_SHA = '17881809c24563b248a7efa0138e9a995b4d6311c55dd277c7eda0a9e4400fbe'
PREDECESSOR_SHA = 'dd1da98e9e232288218832e3a0240c7b67fc86e0e73b53329450b5ff18dda531'
REVISION_SHA = '6515ca85e47a667272921bc3d5ac931ce1d93cecd7eef4aa54e4755f70312fe9'
COMMIT = '723732adae37f12f8488f9537717664415a98d10'
ARCHIVE_SHA = '7ee249d256c039b34a803593089e1cab69acfb61582de63474e74ebff6881843'
PLAN_SHA = '5d94e35e3fe1bcfc5ae8488e4ce1b1f5779b8ce9598ee1b5642ceb879f47dfac'
WORKER = 'contracts/v6_task_campaign_recovery_worker_v20.json'
SCRIPT_BINDING = dict(source_recovery_sha256='04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6',
    side_speed_rad_s=1.5,capture_prepare_gyro_gate='native-current-tick-norm-below-existing-handover-gyro-max',handover_gyro_max_rad_s=.75)


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load021(path=OLD021):
    require(digest(path)==READER021_SHA, 'Frozen021 reader SHA differs')
    spec=importlib.util.spec_from_file_location('frozen021_for_source022',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def clock_context(predecessor, first, progress):
    """Use actual predecessor. No synthesized launch.parent or clock refund."""
    seal=predecessor['actual_parents']['recovery']['seal']
    origin,task=seal['learning_updates'],seal['task_actor_update']
    warmup=predecessor['expected_critic_warmup']
    require((origin,task,seal['actor_updates_consumed'])==(30802,3871,30652), 'Actual source022 parent clocks differ')
    require(type(warmup)is int and warmup==10, 'Source022 critic adaptation differs')
    require(first['source_commit']==predecessor['source_commit']==COMMIT and
        first['status']=='first_actual_actor_PPO_verified' and first['qualification'] is None and
        first['check_counts']==dict(passed=10,pending=0,failed=0) and
        all(row['status']=='passed' for row in first['checks'].values()), 'First-PPO admission incomplete')
    expected=dict(consumption_origin=origin,actor_consumption_origin=seal['actor_updates_consumed'],
        restored_checkpoint_learning_updates=origin,restored_task_actor_origin=task,
        expected_critic_only_updates=warmup,first_actor_consumption_update=origin+warmup+1)
    require(first['evidence']['clocks']==expected, 'First-PPO and predecessor clocks differ')
    direct=first['evidence']['direct_native_progress_phases']
    require(direct['observed_completed_updates']==list(range(1,12)) and direct['observer_status']=='complete',
        'Native first11 phase evidence absent')
    completed,actors=progress['successful_updates'],progress['actor_updates_in_block']
    require(type(completed)is int and type(actors)is int and completed>=0 and actors>=0, 'Invalid native completed counts')
    critic=min(completed,warmup)
    require(completed==critic+actors and progress['parent_updates']==origin and progress['task_actor_update']==task+actors
        and progress['stage_actor_update']==seal['stage_actor_update']+actors, 'Critic/actor partition or curriculum clock differs')
    require(progress['global_budget_actor_updates']==seal['actor_updates_consumed']+actors, 'Actor consumption origin differs')
    require(progress['optimizer_phase']==('critic_warmup' if completed<=warmup else 'actor_and_critic'), 'Native optimizer phase differs')
    return dict(consumption_origin=origin,task_actor_origin=task,actor_consumption_origin=seal['actor_updates_consumed'],
        critic_warmup_updates=warmup,critic_completed=critic,actor_completed=actors,total_completed=completed,
        first_actor_event_step=origin+warmup,first_actor_completion=origin+warmup+1,
        actual_task_actor_update=progress['task_actor_update'],native_first11_observed=True,consumption_and_actor_are_separate=True)


def make_reader(context, reader021=OLD021, reader020=OLD020, reader019=OLD019):
    adapter=load021(reader021)
    reader=adapter.make_reader(context,reader020,reader019)
    reader.COMMIT=COMMIT
    reader.adapter_reuse['source022_projection']='Exact021 tag/teaching and020/019 CRC/metric functions; only authenticated identity and private successful-count coordinates adapt.'
    reader.adapter_reuse['source021_reader_sha256']=READER021_SHA
    return adapter,reader


def authenticate(args, first, inventory, proc_root=Path('/proc')):
    adapter=load021(args.reader021)
    launch=read(args.launch); predecessor=read(args.predecessor); revision=read(args.revision)
    require(launch['source_commit']==COMMIT and launch['source_archive_sha256']==ARCHIVE_SHA and launch['plan_sha256']==PLAN_SHA,
        'Source022 actual launch header differs')
    source,root=Path(launch['source_root']).resolve(),Path(launch['run_root']).resolve()
    run=Path(first['evidence']['actual_worker_directory']).resolve()
    require(run.parts[-4:]==('train','recovery','attempt_006','train') and run.parents[3]==root, 'Actual source022 attempt/root differs')
    require(digest(args.predecessor)==PREDECESSOR_SHA and digest(args.revision)==REVISION_SHA and
        digest(root/'predecessor_022.json')==PREDECESSOR_SHA and digest(root/'source_revision_022.json')==REVISION_SHA,
        'Prepared and native predecessor/revision bytes differ')
    require(revision['source_commit']==predecessor['source_commit']==COMMIT and revision['new_source_root']==str(source) and
        revision['source_archive_sha256']==ARCHIVE_SHA and revision['new_contract_sha256']==PLAN_SHA, 'Actual022 revision differs')
    require(first['evidence']['launch']==launch, 'Actual launch changed since admitted first-PPO')
    parent=predecessor['actual_parents']['recovery'];seal=parent['seal']
    require(('parent' not in launch or launch['parent']==parent) and
        ('expected_critic_warmup' not in launch or launch['expected_critic_warmup']==10), 'Actual optional launch parent differs from predecessor')
    require(parent['sha256']==first['evidence']['invocation']['source_checkpoint_sha256'] and
        parent['path']==first['evidence']['invocation']['source_checkpoint'], 'Actual latest parent differs')
    invocation=read(run.parent/'invocation.json');argv=invocation['command']
    require(argv==first['evidence']['invocation']['command'], 'Actual invocation changed')
    def flag(name):
        require(argv.count(name)==1, 'Missing/duplicate worker flag '+name)
        return argv[argv.index(name)+1]
    require(flag('--contract')==str(source/WORKER) and flag('--run-dir')==str(run) and flag('--resume')==parent['path'] and
        flag('--num-envs')=='4096' and flag('--task-actor-update')==str(seal['task_actor_update']) and
        flag('--consumed-updates')==str(seal['learning_updates']) and flag('--actor-updates-consumed')==str(seal['actor_updates_consumed'])
        and flag('--global-actor-update-offset')=='0', 'Native invocation/parent differs')
    progress=read(run/'progress.json');context=clock_context(predecessor,first,progress)
    worker=next(row for row in first['checks']['unique_native_owner_chain_no_parallel_evaluator']['detail'] if row['kind']=='worker')
    require(progress['pid']==worker['pid'] and worker['command']==argv and progress['num_envs']==4096, 'Native worker PID/population differs')
    owner=adapter.native_owner(progress['pid'],argv,proc_root)
    require(owner['ppid']==worker['ppid'], 'Live worker parent differs')
    reader020=adapter.load020(args.reader020);metric=reader020.make_reader(args.reader019)
    names=set(reader020.SOURCE_UPDATES)|set(metric.SOURCE_HASHES)
    names.update({WORKER,'contracts/v6_task_campaign_v20.json','src/wheeled_tasks/chassis/recovery.py',
        'src/wheeled_tasks/chassis/recovery_prefix_curriculum.py','src/wheeled_tasks/chassis/recovery_script_revision.py',
        'src/wheeled_tasks/chassis/recovery_quality_revision.py','src/wheeled_tasks/chassis/task_campaign.py'})
    expected={name:inventory[name] for name in sorted(names)}
    actual={name:digest(source/name) for name in expected}
    require(actual==expected, 'Frozen022 metric/controller/contract bytes differ')
    config=read(source/WORKER);groups=[row['name'] for row in config['scene_groups']];pools=config['behavior_pool_membership']
    bases=[g for g in groups if pools[g]!='scripted_recovery']
    require(len(groups)==len(set(groups))==95 and len(bases)==41 and set(groups)==set(pools)==set(config['skill_specs']) and
        Counter(pools.values())==Counter(scripted_recovery=54,stand_anchors=9,stand_continuous=1,height=1,translation=16,transitions=6,yaw=8),
        'Native95/41 catalog differs')
    require(config['recovery_training']['prefix_curriculum']['goal_bridge']=='native-original-followup-after-stable-gate-v1' and
        config['recovery_hold_cost']['enabled'] is True and config['recovery_training']['side_speed_rad_s']==1.5, 'Teaching/script fields absent')
    startup=read(run/'startup.json');native=startup['startup'];observed=native['runtime_preflight']['observed']
    require(startup.get('pid',progress['pid'])==progress['pid'] and native['runtime_preflight']['status']=='runtime_configuration_passed'
        and (observed['physics_dt'],observed['policy_dt'],observed['decimation'],observed['seed'])==(.005,.02,4,290927)
        and observed['recovery_hold_cost']==config['recovery_hold_cost'] and observed['modules']['recovery_hold_cost'] is True,
        'Native timing/new hold runtime differs')
    require(observed['recovery_hold_cost_gate_binding']==first['checks']['actual_native_full_restore_new_modules_origins_and_clocks']['detail']['native_hold_cost_gate_binding'],
        'Actual hold gate binding differs')
    require(observed.get('recovery_script_revision')==native.get('recovery_script_revision')==SCRIPT_BINDING==first['evidence']['actual_recovery_script_revision']
        and actual['src/wheeled_tasks/chassis/recovery.py']==SCRIPT_BINDING['source_recovery_sha256'], 'Actual native022 script binding differs')
    population=native['scene_group_counts'];ordinary=native['ordinary_posture_curriculum']
    require(set(population)==set(groups) and all(type(n)is int and n>0 for n in population.values()) and sum(population.values())==4096
        and sum(population[g] for g in bases)==1449, 'Actual positive95/41 population differs')
    require(ordinary['origin_actor_update']==config['ordinary_posture_curriculum']['origin_actor_update']==2294 and
        ordinary['runtime_observed']['eligible_count']==ordinary['runtime_observed']['birth_eligible_population']==1449 and
        config['recovery_training']['prefix_curriculum']['origin_actor_update']==2294, 'Ordinary/recovery origin or coverage changed')
    events=list(run.glob('events.out.tfevents.*.'+str(progress['pid'])+'.*'))
    require(len(events)==1, 'Expected one event from authenticated source022 worker')
    metadata={'launch':args.launch,'first_PPO':args.first_ppo,'inventory':args.inventory,'predecessor':args.predecessor,
        'revision':args.revision,'native_predecessor':root/'predecessor_022.json','native_revision':root/'source_revision_022.json',
        'startup':run/'startup.json','invocation':run.parent/'invocation.json'}
    return dict(identity_verified=True,source_commit=COMMIT,source_root=str(source),worker_pid=progress['pid'],event=str(events[0]),
        groups=groups,base_groups=bases,group_pools=pools,population=population,progress=progress,clocks=context,
        source_files_sha256=actual,invocation=invocation,native_owner=owner,actual_script_binding=SCRIPT_BINDING,
        ordinary_posture_coverage=dict(available=True,native_eligible_population=1449,base_group_population=1449,nonstandard_start_leg_bank_present=False),
        metadata_paths={name:str(path) for name,path in metadata.items()},metadata_sha256={name:digest(path) for name,path in metadata.items()},
        parent_scope='Actual latest REC3871 seal, not diagnostic REC3076; immutable PT verified by admitted first-PPO, not reopened by reader.')


def select_actor_window(adapter,reader,series,authority,audit,width=100,end_update=100,mode='after_critic_baseline'):
    report=adapter.select_actor_window(reader,series,authority,audit,width,end_update,mode)
    report['reused_metric_schema']=report['schema'];report['schema']='source022-recovery-native-actor-window-summary-v1'
    report['source_commit']=COMMIT
    report['limitations']=[line for line in report['limitations'] if not line.startswith('Source021 changes')]
    report['limitations'].append('Source022 changes REC train SIDE6 to1.5 and adds current gyro to CAPTURE→PREPARE; this stochastic window does not establish causal benefit, mechanical qualification or hardware release.')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('launch','first-ppo','inventory','predecessor','revision','output'):p.add_argument('--'+name,type=Path,required=True)
    for name,path in (('reader021',OLD021),('reader020',OLD020),('reader019',OLD019)):p.add_argument('--'+name,type=Path,default=path)
    p.add_argument('--updates',type=int,choices=(50,100),default=100)
    p.add_argument('--end-update',type=int,default=100)
    p.add_argument('--window-mode',choices=('after_critic_baseline','after_ten_actor_baseline'),default='after_critic_baseline')
    p.add_argument('--max-read-mib',type=int,default=512);args=p.parse_args(argv)
    require(not args.output.exists(),'Output exists; preserve previous read')
    schema='source022-recovery-native-event-window-set-v1'
    try:
        require(digest(args.first_ppo)==FIRST_PPO_SHA and digest(args.inventory)==INVENTORY_SHA,'First-PPO/inventory admission bytes differ')
        require(1<=args.max_read_mib<=1024,'Invalid read byte cap')
        first,inventory=read(args.first_ppo),read(args.inventory)
        authority=authenticate(args,first,inventory)
        adapter,reader=make_reader(authority['clocks'],args.reader021,args.reader020,args.reader019)
        last=authority['clocks']['consumption_origin']+authority['clocks']['total_completed']-1
        wanted=reader.wanted_tags(authority)|adapter.extra_tags(authority)
        series,audit=reader.parse_events(Path(authority['event']),wanted,last,min_step=reader.BASELINE_STEP,max_read_bytes=args.max_read_mib*1024**2)
        report_window=select_actor_window(adapter,reader,series,authority,audit,args.updates,args.end_update,args.window_mode)
        require({name:digest(Path(authority['source_root'])/name) for name in authority['source_files_sha256']}==authority['source_files_sha256'],'Frozen source changed during read')
        require(digest(args.reader021)==READER021_SHA and digest(args.reader020)==adapter.READER020_SHA and
            digest(args.reader019)==adapter.load020(args.reader020).BASE_READER_SHA256,'Frozen parser/metric dependency changed during read')
        require({name:digest(path) for name,path in authority['metadata_paths'].items()}==authority['metadata_sha256'],'Immutable metadata changed during read')
        adapter.verify_owner_continuity(authority['native_owner'],adapter.native_owner(authority['worker_pid'],authority['invocation']['command']))
        complete=report_window['complete']
        report=dict(schema=schema,status='requested_actor_windows_complete' if complete else 'pending_requested_actor_windows',complete=complete,
            source_commit=COMMIT,qualification=None,windows={str(args.updates):report_window},identity=authority,parser_metric_reuse=reader.adapter_reuse,
            source_files_unchanged=True,phase_scope='Ten critic-only updates remain charged and collect physical frames; selected dense intervals contain actor updates only.',
            observation_limits='Stochastic native PPO summaries, not deterministic evaluation, mechanical qualification or causal identification.')
    except Exception as error:
        report=dict(schema=schema,status='read_failed',complete=False,source_commit=COMMIT,qualification=None,error=type(error).__name__+': '+str(error))
    report.update(queried_at_utc=datetime.now(timezone.utc).isoformat(),reader_sha256=digest(__file__),PPO_updates_by_reader=0,
        scope='read_existing_native_event_no_simulator_no_GPU_no_SSH_no_TensorBoard_server')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],complete=report['complete'],output=str(args.output),error=report.get('error'))))
    return 2 if report['status']=='read_failed' else 0

if __name__=='__main__': raise SystemExit(main())
