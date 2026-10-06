"""Few meaningful CPU boundaries: single producer, global cutoff, full contact records."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import types

import pytest
import torch

HERE = Path(__file__).resolve().parent
E = HERE.parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


H=load(HERE/'source_script_capture.py','source022_CPU_helper')
D=load(HERE/'run_source_script_native_022.py','source022_CPU_driver')
M=load(E/'mechanical_remaining_capture_candidate_021/mechanical_remaining_capture.py','source022_CPU_hooks')


def writer(path,value):
    path.write_text(json.dumps(value,allow_nan=False))


@pytest.mark.parametrize('failure',[None,'pd'])
def test_five_lanes_wrap_single_original_keep_returns_rng_and_restore(tmp_path,failure):
    old=load(E/'pd_target_payment_candidate_021/test_target_payment_capture.py','source022_frozen_CPU_fake')
    env,parent,reward,unused,calls,objects,actions=old.fake(tmp_path,fail=failure)
    r=env.recovery
    for lane,level,pid,name in zip(H.LANES,H.LEVELS,H.PROFILE_IDS,H.PROFILES):
        r.pose_curriculum.level_ids[lane]=level
        r.pose_curriculum.profile_ids[lane]=pid
        r.pose_curriculum.profile_names[pid]=name
    original=(env.step,env.v5.decode.__func__,env.signal_delay.targets.__func__,env.v5.motor_efforts.__func__,r.fsm.update.__func__,reward.apply_recovery_hold_costs)
    entries=[]
    rng=torch.get_rng_state().clone()
    try:
        for lane,level,pid,name in zip(H.LANES,H.LEVELS,H.PROFILE_IDS,H.PROFILES):
            path=tmp_path/str(lane);path.mkdir()
            frozen=load(E/'pd_target_payment_candidate_021/target_payment_capture.py',f'source022_CPU_private_{lane}')
            capture=H.lane_capture(frozen,lane,level,pid,name,env,parent,reward,M.HookSet(),path,writer)
            entries.append(capture);capture.install()
        if failure:
            with pytest.raises(RuntimeError,match='original pd failure'):
                env.step(actions)
            assert calls['pd']==1
        else:
            assert env.step(actions) is objects['step']
            assert calls==dict(step=1,decode=1,delay=4,pd=4,fsm=4,cost=1)
    finally:
        for capture in reversed(entries):
            assert not capture.cleanup()
    assert torch.equal(rng,torch.get_rng_state())
    assert original==(env.step,env.v5.decode.__func__,env.signal_delay.targets.__func__,env.v5.motor_efforts.__func__,r.fsm.update.__func__,reward.apply_recovery_hold_costs)
    if failure is None:
        for lane,level,pid in zip(H.LANES,H.LEVELS,H.PROFILE_IDS):
            with gzip.open(tmp_path/str(lane)/'pd_target_payment_2216.jsonl.gz','rt') as f:row=json.loads(next(f))
            assert row['lane']==lane and row['old_episode']['lesson_id']==level and row['old_episode']['profile_id']==pid
            assert row['substeps'][0]['pd']['original_RL_PD_preclip_request_nm']==[3.]*6
            assert row['substeps'][-1]['input_physical_step']==3
            assert row['hold_producer']['physical_end_step']==4


@pytest.mark.parametrize('kind',['capture','side','invalid'])
def test_actual_source_global_first_change_outside_selected_once_no_rng(tmp_path,kind):
    source=Path(json.loads((HERE.parent.parent.parent/'source_repair_022/source_candidate_022.json').read_text())['clean_root'])
    module=load(source/'src/wheeled_tasks/chassis/recovery.py','source022_CPU_real_FSM')
    fsm=module.RecoveryFSM(4096,'cpu',{'side_speed_rad_s':1.5},[0.]*6,[1.]*6,[2.]*6,[.5]*6,[1.,1.,-1.,-1.])
    lane=7
    fsm.enabled[lane]=True;fsm.phase[lane]=6 if kind!='side' else 5
    fsm.side_direction[lane]=torch.tensor([-1.,-1.,1.,1.])
    env=types.SimpleNamespace(recovery=types.SimpleNamespace(fsm=fsm))
    watch=H.SourceChangeWatch(env,M.HookSet(),tmp_path,writer,M.plain)
    original=fsm.update.__func__;watch.install()
    q=dq=torch.zeros(4096,6);g=torch.tensor([[0.,0.,-1.]]).expand(4096,3).clone()
    omega=torch.zeros(4096,3);height=torch.full((4096,),.305);support=torch.ones(4096,dtype=torch.bool)
    if kind=='capture':omega[lane,0]=1.
    if kind=='side':g[lane]=torch.tensor([0.,1.,0.])
    if kind=='invalid':g[lane]=0.
    before=torch.get_rng_state().clone();inputs=[x.clone() for x in(q,dq,g,omega,height,support)]
    result=fsm.update(q,dq,g,omega,height,support,support)
    assert result is fsm.script_targets and watch.calls==1
    assert torch.equal(before,torch.get_rng_state()) and all(torch.equal(a,b)for a,b in zip(inputs,(q,dq,g,omega,height,support)))
    assert not watch.cleanup() and not watch.cleanup() and fsm.update.__func__ is original and 'update'not in vars(fsm)
    if kind=='invalid':assert watch.first is None
    else:
        assert watch.first['reference']==1 and watch.first['input_physical_step']==0
        assert watch.first['affected_all_population_lanes']==[7] and lane not in H.LANES


def contacts():
    normal=dict(surface_index=0,count=2,force_scalar_n=[1.,2.],point_physx_m=[[1.,2.,3.],[4.,5.,6.]],
        normal_physx=[[0.,0.,1.],[0.,1.,0.]],separation_m=[-.001,-.002],
        impulse_scalar_ns_derived_force_times_dt=[.005,.01],normal_force_aggregate_n=[0.,2.,1.],
        normal_aggregate_minus_native_matrix_n=[0.,0.,0.])
    friction=dict(surface_index=0,count=2,force_physx_n=[[1.,2.,3.],[4.,5.,6.]],
        point_physx_m=[[1.,2.,3.],[4.,5.,6.]],impulse_physx_ns_derived_force_times_dt=[[.005,.01,.015],[.02,.025,.03]],
        friction_force_aggregate_n=[5.,7.,9.])
    return dict(sensor_body_rows=[dict(sensor_index=8,matrix_force_n=[[0.,2.,1.]],
        net_force_all_contacts_n=[0.,2.,1.],normal_patches=[normal],friction_patches=[friction])])


def test_full_joint_contact_multiset_accepts_only_complete_record_reorder():
    a=contacts();b=copy.deepcopy(a)
    for key in('normal_patches','friction_patches'):
        p=b['sensor_body_rows'][0][key][0]
        for field in('force_scalar_n','force_physx_n','point_physx_m','normal_physx','separation_m',
                     'impulse_scalar_ns_derived_force_times_dt','impulse_physx_ns_derived_force_times_dt'):
            if field in p:p[field].reverse()
    assert H.bits(a)!=H.bits(b) and H.compare_contacts(a,b) is None
    b['sensor_body_rows'][0]['normal_patches'][0]['normal_physx'].reverse()
    assert H.compare_contacts(a,b) is not None


@pytest.mark.parametrize('field',['force_scalar_n','point_physx_m','normal_physx','separation_m','impulse_scalar_ns_derived_force_times_dt'])
def test_no_normal_record_field_can_be_dropped_to_pass(field):
    a=contacts();b=copy.deepcopy(a);p=b['sensor_body_rows'][0]['normal_patches'][0]
    p[field][0]=[9.,9.,9.]if isinstance(p[field][0],list)else 9.
    assert H.compare_contacts(a,b) is not None


def test_contact_duplicates_and_body_filter_attribution_are_strict():
    a=contacts();b=copy.deepcopy(a)
    b['sensor_body_rows'][0]['normal_patches'][0]['count']=1
    assert H.compare_contacts(a,b) is not None
    b=copy.deepcopy(a);b['sensor_body_rows'][0]['normal_patches'][0]['force_scalar_n'].pop()
    with pytest.raises(ValueError,match='Incomplete'):H.compare_contacts(a,b)
    b=copy.deepcopy(a);b['sensor_body_rows'][0]['sensor_index']=9
    assert H.compare_contacts(a,b)=='sensor-index'
    b=copy.deepcopy(a);b['sensor_body_rows'][0]['normal_patches'][0]['surface_index']=1
    assert H.compare_contacts(a,b) is not None
    assert H.bits(-0.)!=H.bits(0.)
