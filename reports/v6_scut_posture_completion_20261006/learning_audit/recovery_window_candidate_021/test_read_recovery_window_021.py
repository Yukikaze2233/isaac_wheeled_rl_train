"""Source021-only CPU fixtures: clocks, native tag classification and coverage.

TFRecord producer and cumulative metric fixtures are reused from verified019.
No AppLauncher, SSH, GPU, model or PPO is instantiated.
"""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def adapter():
    return load('source021_window_adapter', HERE/'read_recovery_window_021.py')


@pytest.fixture(scope='module')
def common():
    return load('source019_tf_writer_for_021', HERE.parent/'test_read_recovery_window_019.py')


@pytest.fixture(scope='module')
def admitted():
    first=json.loads((ROOT/'source_repair_021/first_PPO_verification_021.json').read_text())
    return deepcopy(first['evidence']['launch']), first


def progress(admitted, actors=110, successful=None):
    launch,_=admitted;seal=launch['parent']['seal']
    successful=actors+10 if successful is None else successful
    return dict(successful_updates=successful,actor_updates_in_block=actors,
        parent_updates=seal['learning_updates'],task_actor_update=seal['task_actor_update']+actors,
        global_budget_actor_updates=seal['actor_updates_consumed']+actors,
        optimizer_phase='critic_warmup' if successful<=10 else 'actor_and_critic',
        num_envs=4096,pid=3506652)


@pytest.fixture(scope='module')
def authority(adapter, admitted):
    launch,first=admitted
    receipt=json.loads((ROOT/'source_repair_021/source_candidate_021.json').read_text())
    source=Path(receipt['clean_root'])
    config=json.loads((source/adapter.WORKER).read_text())
    groups=[row['name'] for row in config['scene_groups']];pools=config['behavior_pool_membership']
    bases=[g for g in groups if pools[g]!='scripted_recovery']
    aliases=[g for g in bases if pools[g] in {'translation','transitions','yaw'}]
    remaining=[g for g in bases if g not in aliases];population={}
    for members,total in ((remaining,577),(aliases,872),([g for g in groups if g not in bases],2647)):
        q,r=divmod(total,len(members));population.update({g:q+int(i<r) for i,g in enumerate(members)})
    current=progress(admitted)
    return dict(identity_verified=True,source_commit=adapter.COMMIT,groups=groups,base_groups=bases,
        group_pools=pools,population=population,progress=current,
        clocks=adapter.clock_context(launch,first,current))


@pytest.fixture(scope='module')
def reader(adapter,authority):
    return adapter.make_reader(authority['clocks'])


@pytest.fixture(scope='module')
def full_series(adapter,common,reader,authority,tmp_path_factory):
    # All120 physical collections include ten critic rollouts. Behavior cadence
    # remains native successful%10. HoldCost uses the actual RSL Episode prefix.
    events=[];origin=authority['clocks']['consumption_origin']
    for successful in range(1,121):
        values=common.rollout_values(authority,successful)
        values.update({tag: .125 for tag in adapter.extra_tags(authority)})
        values={('Episode/'+tag if tag.startswith('/HoldCost/') else tag):v for tag,v in values.items()}
        if successful%10==0:values.update(common.behavior_values(authority,successful))
        events.append(common.scalar_event(origin+successful-1,values))
    path=common.event_file(tmp_path_factory.mktemp('native021-clock-tags'),events)
    return reader.parse_events(path,reader.wanted_tags(authority)|adapter.extra_tags(authority),
        origin+119,min_step=reader.BASELINE_STEP)


@pytest.mark.parametrize('success,actors,phase',[(1,0,'critic_warmup'),(10,0,'critic_warmup'),(11,1,'actor_and_critic'),(120,110,'actor_and_critic')])
def test_real_admitted_parent_clocks_and_first11_are_separate(adapter,admitted,success,actors,phase):
    launch,first=admitted;p=progress(admitted,actors,success)
    c=adapter.clock_context(launch,first,p)
    assert c['consumption_origin']==29996 and c['task_actor_origin']==3075
    assert c['critic_completed']==min(success,10) and c['actor_completed']==actors
    assert c['first_actor_event_step']==30006 and c['first_actor_completion']==30007
    assert p['optimizer_phase']==phase


@pytest.mark.parametrize('key,value', [('successful_updates',110),('actor_updates_in_block',120),
    ('parent_updates',30006),('task_actor_update',3195),('global_budget_actor_updates',29976),
    ('optimizer_phase','critic_warmup')])
def test_critic_never_counted_as_actor_or_relabelled_parent(adapter,admitted,key,value):
    launch,first=admitted;p=progress(admitted);p[key]=value
    with pytest.raises(ValueError):adapter.clock_context(launch,first,p)


def test_missing_native_first11_and_failed_admission_rejected(adapter,admitted):
    launch,first=deepcopy(admitted);first['evidence']['direct_native_progress_phases']['observed_completed_updates']=list(range(1,11))
    with pytest.raises(ValueError,match='first11'):adapter.clock_context(launch,first,progress(admitted))
    launch,first=deepcopy(admitted);next(iter(first['checks'].values()))['status']='failed'
    with pytest.raises(ValueError,match='admission'):adapter.clock_context(launch,first,progress(admitted))


@pytest.mark.parametrize('width,end,mode,interval,baseline,event_steps', [
    (50,60,'after_ten_actor_baseline',[3086,3135],30015,[30016,30065]),
    (100,110,'after_ten_actor_baseline',[3086,3185],30015,[30016,30115]),
    (50,50,'after_critic_baseline',[3076,3125],30005,[30006,30055]),
    (100,100,'after_critic_baseline',[3076,3175],30005,[30006,30105])])
def test_native_crc_windows_keep_critic_frames_but_select_only_actor_updates(adapter,reader,authority,full_series,width,end,mode,interval,baseline,event_steps):
    series,audit=full_series;r=adapter.select_actor_window(reader,series,authority,audit,width,end,mode)
    assert r['complete'] and r['window']['task_actor_updates_inclusive']==interval
    assert r['window']['baseline_step']==baseline and r['window']['event_steps_inclusive']==event_steps
    assert r['window']['expected_observed_frames']==width*98304
    assert r['window']['successful_update_endpoints_in_attempt']==[end-width+10,end+10]
    assert r['authoritative_clocks']['consumption_origin']==29996
    assert r['window']['new_critic_warmup_updates']==10 and r['selection']['native_zero_baseline_fabricated'] is False
    assert audit['crc_valid_records']==121
    assert r['teaching']['actual_L0_motion_release']['available'] is False
    assert r['teaching']['actual_L0_nonzero_goal_exposure']['available'] is False


@pytest.mark.parametrize('tag', ['/HoldCost/wheel/fine_density_mean','/recovery/hold_gate/leg_speed_pass_fraction',
    'LearningAudit/RECOVERY_PLACEHOLDER/command_vx_m_s'])
def test_missing_actual_hold_gate_or_group_command_blocks_window(adapter,reader,authority,full_series,tag):
    series,audit=deepcopy(full_series);group=next(g for g in authority['groups'] if g not in authority['base_groups'])
    tag=tag.replace('RECOVERY_PLACEHOLDER',group);del series[tag][30030]
    r=adapter.select_actor_window(reader,series,authority,audit,50,60)
    assert not r['complete'] and any(row['tag']==tag for row in r['missing'])


def test_missing_actual_critic_baseline_not_fabricated(adapter,reader,authority,full_series):
    series,audit=deepcopy(full_series);tag=f"Behavior/{authority['groups'][0]}/frames";del series[tag][30005]
    r=adapter.select_actor_window(reader,series,authority,audit,100,100,'after_critic_baseline')
    assert not r['complete'] and r['window']['baseline_update']==0
    assert any(row['tag']==tag and 30005 in row['missing_steps'] for row in r['missing'])


def test_pending_100_window_needs_110_actors_not_110_total(adapter,reader,authority,admitted,full_series):
    series,audit=full_series;a=deepcopy(authority);a['progress']=progress(admitted,100)
    a['clocks']=adapter.clock_context(*admitted,a['progress'])
    r=adapter.select_actor_window(reader,series,a,audit,100)
    assert not r['complete'] and r['selection']['additional_actor_updates_needed']==10
    assert r['selection']['available_actor_updates']==100


def test_only_native_HoldCost_namespace_is_canonicalized(reader):
    assert reader.canonical_tag('Episode//HoldCost/gyro/broad_density_mean')=='/HoldCost/gyro/broad_density_mean'
    assert reader.canonical_tag('Episode//recovery/hold_population')=='/recovery/hold_population'
    assert reader.canonical_tag('Unrelated/HoldCost/gyro/broad_density_mean')=='Unrelated/HoldCost/gyro/broad_density_mean'


def test_old_metric_and_crc_code_unchanged_except_tag_classifier(adapter,reader):
    old=adapter.load020().make_reader()
    for name in adapter.load020().REUSED_FUNCTIONS:
        if name!='canonical_tag':assert getattr(reader,name).__code__==getattr(old,name).__code__
    assert old.CONSUMPTION_ORIGIN==29679 and reader.CONSUMPTION_ORIGIN==29996


def fake_proc(tmp_path, worker):
    root=tmp_path/'proc';directory=root/str(worker['pid']);directory.mkdir(parents=True)
    (directory/'cmdline').write_bytes(b'\0'.join(x.encode() for x in worker['command'])+b'\0')
    fields=['S',str(worker['ppid'])]+['0']*17+['123456']+['0']*4
    (directory/'stat').write_text(str(worker['pid'])+' (python cpu fixture) '+' '.join(fields))
    return root,directory


@pytest.fixture
def actual_source_synthetic_attempt(adapter,admitted,authority,common,tmp_path):
    launch,first=deepcopy(admitted);source=Path(json.loads((ROOT/'source_repair_021/source_candidate_021.json').read_text())['clean_root'])
    root=tmp_path/'campaign';run=root/'train/recovery/attempt_005/train';run.mkdir(parents=True)
    launch.update(source_root=str(source),run_root=str(root))
    first['evidence']['launch']=deepcopy(launch);first['evidence']['actual_worker_directory']=str(run)
    argv=first['evidence']['invocation']['command']
    for flag,value in (('--contract',str(source/adapter.WORKER)),('--run-dir',str(run))):argv[argv.index(flag)+1]=value
    worker=next(row for row in first['checks']['unique_native_owner_chain_no_parallel_evaluator']['detail'] if row['kind']=='worker')
    worker['command']=deepcopy(argv)
    invocation=deepcopy(first['evidence']['invocation'])
    current=progress(admitted,60)
    observed=deepcopy(first['evidence']['runtime_preflight']['observed'])
    startup=dict(pid=current['pid'],startup=dict(runtime_preflight=dict(status='runtime_configuration_passed',observed=observed),
        scene_group_counts=authority['population'],ordinary_posture_curriculum=observed['ordinary_posture_curriculum']))
    values={root/'source_revision_021.json':dict(source_commit=adapter.COMMIT,new_source_root=str(source),source_archive_sha256=adapter.ARCHIVE_SHA,new_contract_sha256=adapter.PLAN_SHA),
        root/'predecessor_021.json':dict(actual_parents=dict(recovery=launch['parent']),expected_critic_warmup=10),
        run.parent/'invocation.json':invocation,run/'progress.json':current,run/'startup.json':startup,tmp_path/'launch.json':launch}
    for path,value in values.items():common.write_json(path,value)
    (run/f"events.out.tfevents.1700000000.fixture.{current['pid']}.0").write_bytes(common.record_bytes(common.Event(file_version='brain.Event:2')))
    proc_root,proc_dir=fake_proc(tmp_path,worker)
    args=SimpleNamespace(reader020=adapter.OLD020,reader019=adapter.OLD019,launch=tmp_path/'launch.json',
        first_ppo=ROOT/'source_repair_021/first_PPO_verification_021.json',inventory=ROOT/'source_repair_021/frozen_source_inventory_021.json')
    return args,first,json.loads(args.inventory.read_text()),proc_root,proc_dir


def test_native_owner_and_actual_source_gate_runtime_are_authenticated(adapter,actual_source_synthetic_attempt):
    args,first,inventory,proc_root,_=actual_source_synthetic_attempt
    result=adapter.authenticate(args,first,inventory,proc_root)
    assert result['native_owner']['starttime_ticks']==123456
    assert result['clocks']['actor_completed']==60 and result['clocks']['total_completed']==70
    assert result['clocks']['consumption_origin']==29996 and len(result['source_files_sha256'])>=15


@pytest.mark.parametrize('mutation', ['wrong_command','zombie','wrong_parent'])
def test_missing_or_reassigned_live_owner_rejected(adapter,actual_source_synthetic_attempt,mutation):
    args,first,inventory,proc_root,directory=actual_source_synthetic_attempt
    if mutation=='wrong_command':(directory/'cmdline').write_bytes(b'/fake/python\0')
    else:
        stat=(directory/'stat').read_text()
        stat=stat.replace(') S ',') Z ') if mutation=='zombie' else stat.replace(') S 3506645',') S 1')
        (directory/'stat').write_text(stat)
    with pytest.raises(ValueError):adapter.authenticate(args,first,inventory,proc_root)


def test_native_read_interval_reparent_or_pid_reuse_rejected(adapter):
    before=dict(pid=123,ppid=100,command=['python','reader'],starttime_ticks=10,state='S')
    after={**before,'state':'R'};adapter.verify_owner_continuity(before,after)
    for key,value in (('ppid',1),('starttime_ticks',11),('command',['python','different']),('pid',124)):
        with pytest.raises(ValueError,match='ownership changed'):
            adapter.verify_owner_continuity(before,{**after,key:value})
