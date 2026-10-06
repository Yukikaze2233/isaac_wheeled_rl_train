"""Focused CPU fixtures. Synthetic TFRecord data are never native outcomes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

@pytest.fixture(scope='module')
def adapter():return load('source022_window_candidate',HERE/'read_recovery_window_022.py')
@pytest.fixture(scope='module')
def common():return load('source019_test_writer_022',HERE.parent/'test_read_recovery_window_019.py')
@pytest.fixture(scope='module')
def admitted():
    p=REPORT/'source_repair_022'
    return json.loads((p/'predecessor_022.json').read_text()),json.loads((p/'first_PPO_verification_022.readonly002.json').read_text())

def progress(admitted,actors=100,completed=None):
    predecessor,_=admitted;s=predecessor['actual_parents']['recovery']['seal']
    completed=actors+10 if completed is None else completed
    return dict(successful_updates=completed,actor_updates_in_block=actors,parent_updates=s['learning_updates'],
        task_actor_update=s['task_actor_update']+actors,stage_actor_update=s['stage_actor_update']+actors,
        global_budget_actor_updates=s['actor_updates_consumed']+actors,
        optimizer_phase='critic_warmup' if completed<=10 else 'actor_and_critic',num_envs=4096,pid=3543217)

@pytest.mark.parametrize('completed,actors',[(1,0),(10,0),(11,1),(110,100)])
def test_actual_parent_10critic_actor_partition(adapter,admitted,completed,actors):
    c=adapter.clock_context(*admitted,progress(admitted,actors,completed))
    assert (c['consumption_origin'],c['task_actor_origin'],c['actor_consumption_origin'])==(30802,3871,30652)
    assert c['first_actor_event_step']==30812 and c['first_actor_completion']==30813
    assert c['total_completed']==completed and c['actor_completed']==actors

@pytest.mark.parametrize('key,value',[('successful_updates',100),('actor_updates_in_block',110),('parent_updates',29996),
    ('task_actor_update',3972),('stage_actor_update',0),('global_budget_actor_updates',30753),('optimizer_phase','critic_warmup')])
def test_misreported_consumption_or_actor_origin_rejected(adapter,admitted,key,value):
    p=progress(admitted);p[key]=value
    with pytest.raises(ValueError):adapter.clock_context(*admitted,p)

@pytest.mark.parametrize('change',['missing11','failedproof','wrongparent','wrongwarmup'])
def test_no_migration_or_phase_imputation(adapter,admitted,change):
    before,proof=deepcopy(admitted)
    if change=='missing11':proof['evidence']['direct_native_progress_phases']['observed_completed_updates'].pop()
    elif change=='failedproof':next(iter(proof['checks'].values()))['status']='failed'
    elif change=='wrongparent':before['actual_parents']['recovery']['seal']['task_actor_update']=3076
    else:before['expected_critic_warmup']=0
    with pytest.raises(ValueError):adapter.clock_context(before,proof,progress(admitted))

@pytest.fixture(scope='module')
def authority(adapter,admitted):
    src=Path(json.loads((REPORT/'source_repair_022/source_candidate_022.json').read_text())['clean_root'])
    cfg=json.loads((src/adapter.WORKER).read_text());groups=[r['name'] for r in cfg['scene_groups']]
    pools=cfg['behavior_pool_membership'];bases=[g for g in groups if pools[g]!='scripted_recovery'];aliases=[g for g in bases if pools[g] in {'translation','transitions','yaw'}]
    population={}
    for members,total in (([g for g in bases if g not in aliases],577),(aliases,872),([g for g in groups if g not in bases],2647)):
        q,r=divmod(total,len(members));population.update({g:q+int(i<r) for i,g in enumerate(members)})
    p=progress(admitted)
    return dict(identity_verified=True,source_commit=adapter.COMMIT,groups=groups,base_groups=bases,group_pools=pools,population=population,
        clocks=adapter.clock_context(*admitted,p),progress=p)

@pytest.fixture(scope='module')
def readers(adapter,authority):return adapter.make_reader(authority['clocks'])
@pytest.fixture(scope='module')
def series(common,readers,authority,tmp_path_factory):
    old,reader=readers;origin=authority['clocks']['consumption_origin'];events=[]
    for successful in range(1,111):
        values=common.rollout_values(authority,successful);values.update({t:.125 for t in old.extra_tags(authority)})
        values={('Episode/'+tag if tag.startswith('/HoldCost/') else tag):v for tag,v in values.items()}
        if successful%10==0:values.update(common.behavior_values(authority,successful))
        events.append(common.scalar_event(origin+successful-1,values))
    path=common.event_file(tmp_path_factory.mktemp('synthetic022-TFRecord'),events)
    return reader.parse_events(path,reader.wanted_tags(authority)|old.extra_tags(authority),30911,min_step=reader.BASELINE_STEP)

def test_exact_first100_actor_window_real_critic_baseline(adapter,readers,authority,series):
    old,reader=readers;r=adapter.select_actor_window(old,reader,*series[:1],authority,series[1])
    assert r['complete'] and r['source_commit']==adapter.COMMIT
    w=r['window'];assert w['task_actor_updates_inclusive']==[3872,3971]
    assert w['baseline_step']==30811 and w['event_steps_inclusive']==[30812,30911]
    assert w['successful_update_endpoints_in_attempt']==[10,110]
    assert w['expected_observed_frames']==9830400 and w['policy_seconds_per_environment']==48.
    assert w['baseline_update']==0 and w['end_update']==100 and w['critic_warmup_excluded']
    assert r['teaching']['actual_L0_motion_release']['available'] is False
    assert r['teaching']['actual_L0_nonzero_goal_exposure']['available'] is False

@pytest.mark.parametrize('tag',['behavior_baseline','/HoldCost/wheel/fine_density_mean','/recovery/hold_gate/leg_speed_pass_fraction'])
def test_missing_original_critic_baseline_or_native_signal_pending(adapter,readers,authority,series,tag):
    old,reader=readers;s,a=deepcopy(series)
    if tag=='behavior_baseline':tag='Behavior/'+authority['groups'][0]+'/frames';del s[tag][30811]
    else:del s[tag][30825]
    r=adapter.select_actor_window(old,reader,s,authority,a)
    assert not r['complete'] and any(m['tag']==tag for m in r['missing'])

def test_99actors_cannot_be_complete_from_110event_fixture(adapter,readers,authority,series,admitted):
    old,reader=readers;a=deepcopy(authority);a['progress']=progress(admitted,99);a['clocks']=adapter.clock_context(*admitted,a['progress'])
    r=adapter.select_actor_window(old,reader,series[0],a,series[1])
    assert not r['complete'] and r['selection']['additional_actor_updates_needed']==1

def test_reused_crc_and_all_metric_code_unchanged(adapter,readers):
    old,reader=readers;original=old.load020().make_reader()
    for n in old.load020().REUSED_FUNCTIONS:
        if n!='canonical_tag':assert getattr(reader,n).__code__==getattr(original,n).__code__
    assert original.COMMIT!=reader.COMMIT and reader.COMMIT==adapter.COMMIT
    assert reader.canonical_tag('Episode//HoldCost/gyro/fine_density_mean')=='/HoldCost/gyro/fine_density_mean'

@pytest.fixture
def synthetic_attempt(adapter,admitted,authority,common,tmp_path,monkeypatch):
    predecessor,first=deepcopy(admitted);meta=json.loads((REPORT/'source_repair_022/source_candidate_022.json').read_text());source=Path(meta['clean_root'])
    launch=deepcopy(first['evidence']['launch']);root=tmp_path/'campaign';run=root/'train/recovery/attempt_006/train';run.mkdir(parents=True)
    launch.update(source_root=str(source),run_root=str(root))
    first['evidence']['launch']=deepcopy(launch);first['evidence']['actual_worker_directory']=str(run)
    argv=first['evidence']['invocation']['command']
    for flag,value in (('--contract',str(source/adapter.WORKER)),('--run-dir',str(run))):argv[argv.index(flag)+1]=value
    worker=next(r for r in first['checks']['unique_native_owner_chain_no_parallel_evaluator']['detail'] if r['kind']=='worker');worker['command']=deepcopy(argv)
    revision=json.loads((REPORT/'source_repair_022/source_revision_022.json').read_text());revision['new_source_root']=str(source)
    current=progress(admitted);observed=deepcopy(first['evidence']['runtime_preflight']['observed'])
    startup=dict(pid=current['pid'],startup=dict(runtime_preflight=dict(status='runtime_configuration_passed',observed=observed),
        scene_group_counts=authority['population'],ordinary_posture_curriculum=observed['ordinary_posture_curriculum'],recovery_script_revision=adapter.SCRIPT_BINDING))
    paths={root/'source_revision_022.json':revision,root/'predecessor_022.json':predecessor,run.parent/'invocation.json':first['evidence']['invocation'],
        run/'progress.json':current,run/'startup.json':startup,tmp_path/'launch.json':launch}
    for path,value in paths.items():common.write_json(path,value)
    monkeypatch.setattr(adapter,'PREDECESSOR_SHA',adapter.digest(root/'predecessor_022.json'));monkeypatch.setattr(adapter,'REVISION_SHA',adapter.digest(root/'source_revision_022.json'))
    (run/f'events.out.tfevents.1700000000.fixture.{current["pid"]}.0').write_bytes(common.record_bytes(common.Event(file_version='brain.Event:2')))
    proc,procdir=load('source021_fixture_owner',HERE.parent/'recovery_window_candidate_021/test_read_recovery_window_021.py').fake_proc(tmp_path,worker)
    args=SimpleNamespace(reader021=adapter.OLD021,reader020=adapter.OLD020,reader019=adapter.OLD019,launch=tmp_path/'launch.json',
        predecessor=root/'predecessor_022.json',revision=root/'source_revision_022.json',first_ppo=REPORT/'source_repair_022/first_PPO_verification_022.readonly002.json',
        inventory=REPORT/'source_repair_022/frozen_source_inventory_022.json')
    return args,first,json.loads(args.inventory.read_text()),proc,procdir,run

def test_actual_frozen_source_and_current_runtime_read_only_auth(adapter,synthetic_attempt):
    args,first,inventory,proc,_,_=synthetic_attempt;r=adapter.authenticate(args,first,inventory,proc)
    assert r['clocks']['actor_completed']==100 and r['native_owner']['starttime_ticks']==123456
    assert r['actual_script_binding']==adapter.SCRIPT_BINDING and len(r['source_files_sha256'])>=18

@pytest.mark.parametrize('change',['fake_launch_parent','old_attempt','old_source','bad_script_binding','bad_source_pin','wrong_parent_pid','changed_launch'])
def test_022_identity_runtime_migration_negative_paths(adapter,synthetic_attempt,change):
    args,first,inventory,proc,pdir,run=synthetic_attempt
    if change in {'fake_launch_parent','old_source','changed_launch'}:
        x=json.loads(args.launch.read_text())
        if change=='fake_launch_parent':x['parent']={'seal':{}}
        elif change=='old_source':x['source_commit']='d0dce6181df83b03fe166abcbadef3c81de2e57e'
        else:x['status']='different'
        args.launch.write_text(json.dumps(x))
    elif change=='old_attempt':first['evidence']['actual_worker_directory']=str(run).replace('attempt_006','attempt_005')
    elif change=='bad_script_binding':
        p=run/'startup.json';x=json.loads(p.read_text());x['startup']['recovery_script_revision']['side_speed_rad_s']=6;p.write_text(json.dumps(x))
    elif change=='bad_source_pin':inventory['src/wheeled_tasks/chassis/recovery.py']='0'*64
    else:p=pdir/'stat';p.write_text(p.read_text().replace(') S 3543210',') S 1'))
    with pytest.raises(ValueError):adapter.authenticate(args,first,inventory,proc)

def test_actual_launch_optional_parent_is_not_synthesized(adapter,synthetic_attempt):
    args,first,inventory,proc,_,_=synthetic_attempt
    x=json.loads(args.launch.read_text());x.pop('parent',None);x.pop('expected_critic_warmup',None)
    args.launch.write_text(json.dumps(x));first['evidence']['launch']=deepcopy(x)
    result=adapter.authenticate(args,first,inventory,proc)
    assert result['clocks']['consumption_origin']==30802 and 'parent' not in x

@pytest.fixture(scope='module')
def runner():return load('source022_readonly_runner',HERE/'run_remote_reader_022.py')

def fake_manifest():
    paths=[HERE/'read_recovery_window_022.py',HERE.parent/'recovery_window_candidate_021/read_recovery_window_021.py',
        HERE.parent/'read_recovery_window_020.py',HERE.parent/'read_recovery_window_019.py',REPORT/'source_repair_022/first_PPO_verification_022.readonly002.json',
        REPORT/'source_repair_022/frozen_source_inventory_022.json',REPORT/'source_repair_022/predecessor_022.json',REPORT/'source_repair_022/source_revision_022.json']
    import hashlib
    return dict(source_commit='723732adae37f12f8488f9537717664415a98d10',files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

def test_runner_recipe_is_actual_bundle_and_one_first100(runner):
    p=runner.recipe(fake_manifest(),'CPU_recipe_001');argv=p['command']
    assert argv[argv.index('--end-update')+1]=='100' and argv[argv.index('--window-mode')+1]=='after_critic_baseline'
    assert len(p['source_files'])==8 and p['execute_count_limit']==1
    assert p['native_window']['dense_events']==[30812,30911] and p['PPO_actions']==0

@pytest.mark.parametrize('name',['../escape','with space','x;id','中文'])
def test_runner_recipe_path_injection_rejected(runner,name):
    with pytest.raises(ValueError):runner.recipe(fake_manifest(),name)

def test_runner_manifest_changed_dependency_rejected(runner):
    m=fake_manifest();m['files_sha256'][str(HERE/'read_recovery_window_022.py')]='0'*64
    with pytest.raises(ValueError):runner.recipe(m,'safe')

def test_runner_emit_only_has_no_external_calls(runner,tmp_path,monkeypatch):
    m=fake_manifest();mf=tmp_path/'manifest.json';mf.write_text(json.dumps(m));monkeypatch.setattr(runner,'HERE',tmp_path)
    # Recipe takes original source paths; only local destination is redirected.
    plan=dict(schema='fixture',command=['read-only'],PPO_actions=0)
    monkeypatch.setattr(runner,'recipe',lambda *_:plan)
    monkeypatch.setattr(runner.subprocess,'run',lambda *_a,**_k:pytest.fail('emit must not execute subprocess'))
    assert runner.main(['--manifest',str(mf),'--manifest-sha256',runner.digest(mf),'--observation','emit_fixture'])==0
    assert json.loads((tmp_path/'emit_fixture/recipe.json').read_text())==plan
