"""CPU observer semantics; these fixtures never instantiate App/PhysX/PPO."""
from pathlib import Path
import importlib.util
import json
import types

import pytest
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mechanical_candidate_test',HERE/'mechanical_capture.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def box(t): return types.SimpleNamespace(torch=t)


class Contact:
    sensor_count=2
    filter_count=2
    max_contact_data_count=8
    sensor_paths=['a','b']
    filter_paths=[['g0','g1'],['g0','g1']]
    def __init__(self):
        self.count=torch.tensor([[1,0],[0,2]],dtype=torch.int64)
        self.start=torch.tensor([[2,2**32-1],[2**32-1,4]],dtype=torch.int64)
        self.force=torch.arange(8.).view(-1,1)
        self.point=torch.arange(24.).view(-1,3)
        self.normal=torch.ones(8,3)
        self.distance=torch.zeros(8,1)
        self.matrix=torch.arange(12.).view(2,2,3)
        self.net=torch.zeros(2,3)
        self.calls=[]
    def get_contact_force_matrix(self,dt): self.calls.append(('matrix',dt));return self.matrix
    def get_net_contact_forces(self,dt): self.calls.append(('net',dt));return self.net
    def get_contact_data(self,dt): self.calls.append(('normal',dt));return self.force,self.point,self.normal,self.distance,self.count,self.start
    def get_friction_data(self,dt):
        self.calls.append(('friction',dt))
        self.count.fill_(0);self.start.fill_(0)
        self.force.fill_(999);self.point.fill_(999)
        return torch.ones(8,3),torch.ones(8,3),self.count,self.start


def fake_env():
    n=4096
    pose=torch.zeros(n,2,7);pose[...,6]=1.
    data=types.SimpleNamespace(body_link_pose_w=box(pose),joint_pos=box(torch.zeros(n,2)),
        joint_vel=box(torch.zeros(n,2)),root_link_pose_w=box(torch.zeros(n,7)),
        root_link_lin_vel_w=box(torch.zeros(n,3)),root_com_ang_vel_b=box(torch.zeros(n,3)),
        projected_gravity_b=box(torch.zeros(n,3)),applied_torque=box(torch.zeros(n,2)))
    fsm=types.SimpleNamespace(script_targets=torch.zeros(n,6),phase=torch.full((n,),5),route=torch.full((n,),4),
        blend=torch.zeros(n),scripted=torch.ones(n,dtype=torch.bool))
    levels=torch.full((n,),2); profiles=torch.zeros(n,dtype=torch.long);profiles[2216]=1
    rec=types.SimpleNamespace(fsm=fsm,enabled=torch.ones(n,dtype=torch.bool),release_finished=torch.ones(n,dtype=torch.bool),
        elapsed=torch.zeros(n),pure_rl=torch.zeros(n,dtype=torch.bool),learning_mask=torch.zeros(n,dtype=torch.bool),
        policy_motion_hold=torch.ones(n,dtype=torch.bool),observed_support=torch.zeros(n,dtype=torch.bool),observed_height=torch.zeros(n),
        pose_curriculum=types.SimpleNamespace(level_ids=levels,profile_ids=profiles,profile_names=list(m.PROFILES)))
    e=types.SimpleNamespace(num_envs=n,dt=.005,policy_dt=.02,decimation=4,device='cpu',recovery=rec,body_count=2,
        cfg=dict(position_iterations=64,velocity_iterations=32,closure_gap_termination_m=.003),
        model_spec=dict(constraints=[dict(body0='b0',body1='b1',local_pos0_m=[0,0,0],local_pos1_m=[0,0,0])]),
        manifest=dict(control_joint_names=['a']*6,spring_joint_names=['spring']),
        robot=types.SimpleNamespace(data=data,body_names=['b0','b1'],joint_names=['j0','j1']),
        _closure_body_ids=torch.tensor([[0,1]]),_closure_points=torch.zeros(1,2,3),
        scene_groups=['test']*n,env_paths=['/World/env_'+str(i) for i in range(n)], origins=torch.arange(n*3.).view(n,3),
        v5=types.SimpleNamespace(requested_motor_effort=torch.zeros(n,6),current_motor_bounds=torch.ones(n,6)),
        motor_strength=torch.ones(n,6),spring_strength=torch.ones(n,2),signal_delay=None,command_transport=None)
    e.closure_gap=lambda:(data.body_link_pose_w.torch[:,0,:3]-data.body_link_pose_w.torch[:,1,:3]).norm(dim=-1)
    view=Contact();view.filter_count=1;view.sensor_count=4
    view.sensor_paths=[e.env_paths[i]+'/Robot/'+name for i in m.LANES for name in e.robot.body_names]
    view.filter_paths=[[e.env_paths[i]+'/Terrain/surface_0/geometry/mesh'] for i in m.LANES for _ in e.robot.body_names]
    view.matrix=torch.zeros(4,1,3);view.net=torch.zeros(4,3);view.count=torch.zeros(4,1,dtype=torch.long);view.start=torch.zeros(4,1,dtype=torch.long)
    e.contact_views=[(torch.tensor(m.LANES),view)]
    return e


def writer(path,value): path.write_text(json.dumps(value,allow_nan=False))


def session(tmp_path):
    env=fake_env();module=types.SimpleNamespace()
    module.update_robot_data=lambda e,dt:None
    capture=m.MechanicalCapture(env,module,tmp_path,writer,lambda value:value)
    return env,module,capture


def test_contact_copy_precedes_reused_friction_indices():
    view=Contact();rows,capacity=m.contact_slices(view,[0,1],.005,lambda v:v)
    assert rows[0]['normal_patches'][0]['count']==1
    assert rows[0]['normal_patches'][0]['force_scalar_n']==[[2.]]
    assert rows[1]['normal_patches'][1]['point_physx_m']==[[12.,13.,14.],[15.,16.,17.]]
    assert rows[0]['normal_patches'][0]['impulse_scalar_ns_derived_force_times_dt'][0][0]==pytest.approx(.01)
    assert rows[0]['friction_patches'][0]['count']==0
    assert capacity['sum_reported_normal_patch_counts']==3 and not capacity['native_overflow_flag_available']
    assert view.calls==[('matrix',.005),('net',.005),('normal',.005),('friction',.005)]


@pytest.mark.parametrize('bad',[(-1,0),(1,-1),(9,0),(2,7)])
def test_invalid_contact_slice_rejected(bad):
    view=Contact();view.count[0,0],view.start[0,0]=bad
    with pytest.raises(ValueError): m.contact_slices(view,[0],.005,lambda v:v)


@pytest.mark.parametrize('field',['filter_count','max_contact_data_count'])
def test_contact_detail_unavailable_rejected(field):
    view=Contact();setattr(view,field,0)
    with pytest.raises(ValueError,match='unavailable'): m.contact_slices(view,[0],.005,lambda v:v)


def test_contact_force_nonfinite_rejected():
    view=Contact();view.force[2]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'): m.contact_slices(view,[0],.005,lambda v:v)


def test_pin_geometry_uses_actual_body_pose_all_endpoints():
    env=fake_env();env.robot.data.body_link_pose_w.torch[1708,1,0]=.004
    world,gaps=m.pin_geometry(env)
    assert tuple(world.shape)==(4096,1,2,3)
    assert float(gaps[1708,0])==pytest.approx(.004)
    assert float(gaps[2216,0])==0


def test_pin_aggregate_must_equal_native():
    env=fake_env();env.closure_gap=lambda:torch.ones(4096)
    with pytest.raises(ValueError,match='native closure_gap'): m.pin_geometry(env)


def test_hook_restore_inherited_descriptor_without_shadow():
    class Owner:
        def method(self): return 7
    o=Owner();h=m.HookSet();prior=o.method
    h.install(o,'method',lambda:9)
    assert o.method()==9
    assert h.restore()==[] and 'method' not in vars(o) and m.same_callback(o.method,prior)
    assert h.restore()==[]


def test_hook_does_not_clobber_foreign_owner():
    o=types.SimpleNamespace(callback=lambda:1);h=m.HookSet()
    h.install(o,'callback',lambda:2);foreign=lambda:3;o.callback=foreign
    assert h.restore()==['Mechanical hook ownership changed: callback'] and o.callback is foreign


def test_physics_sample_original_update_then_contact_not_stale_cache(tmp_path):
    e,module,capture=session(tmp_path);calls=[]
    def update(env,dt): calls.append('update');env.robot.data.joint_pos.torch.add_(1)
    module.update_robot_data=update
    e.robot.set_joint_effort_target_index=lambda **kw:calls.append('effort')
    def reset(ids): calls.append('reset');module.update_robot_data(e,.005)
    e.reset=reset
    def step(actions):
        for _ in range(4):
            e.robot.set_joint_effort_target_index(target=torch.zeros(4096,2))
            module.update_robot_data(e,.005)
        diagnostic={'done':torch.zeros(4096,dtype=torch.bool),'reasons':{}}
        e.reset(torch.tensor([1708]));diagnostic['done'][1708]=True
        return None,None,None,dict(diagnostics=diagnostic)
    e.step=step
    rng=torch.get_rng_state().clone();capture.install()
    result=e.step(None)
    assert capture.physics_steps==4 and capture.policy_steps==1
    assert capture.row_counts=={1708:5,2216:5}
    assert capture.pre_reset[1708]['state']['full_q']==[4.,4.]
    assert capture.terminals[1708]['fields']['done'] is True
    assert capture.birth_ids[1708]==2 and not capture.alive[1708]
    assert torch.equal(rng,torch.get_rng_state())
    assert result[3]['diagnostics']['done'][1708]
    capture.cleanup()
    assert module.update_robot_data is update and e.step is step
    assert calls==['effort','update']*4+['reset','update']
    with gzip_rows(tmp_path/'mechanical_5ms.jsonl.gz') as rows:
        assert rows[2]['state']['full_q']==[1.,1.] and rows[2]['contacts']['env_contact_force_cache_consumed'] is False


class gzip_rows:
    def __init__(self,path): self.path=path
    def __enter__(self):
        import gzip
        with gzip.open(self.path,'rt') as stream: return [json.loads(line) for line in stream]
    def __exit__(self,*args): pass


def test_no_effort_token_is_not_physics(tmp_path):
    _,_,capture=session(tmp_path);capture.in_policy=True
    with pytest.raises(ValueError,match='lacks native interval effort'): capture.post_physics()
    assert capture.physics_steps==0
    capture.cleanup()


@pytest.mark.parametrize('change',[('position_iterations',12),('velocity_iterations',6),('closure_gap_termination_m',.004)])
def test_solver_and_guard_not_silently_changed(tmp_path,change):
    e=fake_env();e.cfg[change[0]]=change[1]
    with pytest.raises(ValueError,match='solver/3mm'): m.MechanicalCapture(e,types.SimpleNamespace(),tmp_path,writer,lambda v:v)


def test_lane_birth_wrong_level_rejected(tmp_path):
    e=fake_env();e.recovery.pose_curriculum.level_ids[1708]=0
    with pytest.raises(ValueError,match='lane/lesson/profile'): m.MechanicalCapture(e,types.SimpleNamespace(),tmp_path,writer,lambda v:v)


def test_incomplete_capture_is_preserved_failed(tmp_path):
    _,_,capture=session(tmp_path)
    with pytest.raises(ValueError,match='loop not complete'): capture.finish()
    capture.cleanup();capture.cleanup()
    value=json.loads((tmp_path/'mechanical_capture_failure.json').read_text())
    assert value['status']=='incomplete_or_hook_restore_failed' and value['qualification'] is None
    assert not value['summary']['complete'] and value['summary']['physics_steps']==0


def test_plain_detaches_native_alias():
    x=torch.tensor([1.]);row=m.plain(x);x.fill_(99)
    assert row==[1.]


def test_layout_copies_exact_two_native_origins(tmp_path):
    e,_,capture=session(tmp_path)
    expected=e.origins[list(m.LANES)].tolist()
    layout=json.loads((tmp_path/'mechanical_layout.json').read_text())
    assert layout['selected_env_origins_world_m']==expected
    e.origins.fill_(99)
    assert layout['selected_env_origins_world_m']==expected
    capture.cleanup()


def test_no_cuda_initialized_by_fixtures(): assert not torch.cuda.is_initialized()


def driver_module():
    spec=importlib.util.spec_from_file_location('mechanical_driver_test',HERE/'run_capture_021.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('fail_stage',[None,'install','finish'])
def test_composed_observer_cleanup_before_parent(tmp_path,monkeypatch,fail_stage):
    import sys
    driver=driver_module();calls=[]
    monkeypatch.setitem(sys.modules,'warp',types.SimpleNamespace(to_torch=lambda v:v))
    class Parent:
        def __init__(self,env,output): calls.append('parent_init')
        def _before_close(self): calls.append('parent_cleanup')
        def finish(self,output): calls.append('parent_finish');return {'original':True}
    class Capture:
        def __init__(self,*args): calls.append('capture_init')
        def install(self):
            calls.append('install')
            if fail_stage=='install': raise ValueError('install rejected')
        def cleanup(self): calls.append('capture_cleanup')
        def finish(self):
            calls.append('capture_finish')
            if fail_stage=='finish': raise ValueError('finish rejected')
            return {'complete':True}
    clazz=driver.make_capture_observer(Parent,types.SimpleNamespace(),types.SimpleNamespace(MechanicalCapture=Capture),writer)
    if fail_stage=='install':
        with pytest.raises(ValueError,match='install rejected'): clazz(None,tmp_path)
        assert calls[-2:]==['capture_cleanup','parent_cleanup'];return
    obj=clazz(None,tmp_path)
    if fail_stage=='finish':
        with pytest.raises(ValueError,match='finish rejected'): obj.finish(tmp_path)
        assert calls[-2:]==['capture_cleanup','parent_cleanup'];return
    value=obj.finish(tmp_path)
    assert value['original'] and value['mechanical_capture_complete']
    assert calls.index('capture_finish')<calls.index('parent_finish')
    obj._before_close();assert calls[-2:]==['capture_cleanup','parent_cleanup']


def test_driver_default_is_cpu_with_explicit_native_only():
    import ast
    tree=ast.parse((HERE/'run_capture_021.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--mode']
    assert len(calls)==1
    keywords={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}
    assert keywords['default']=='cpu' and keywords['choices']==('cpu','native')


def test_factory_does_not_import_or_resolve_env_before_native_parent(tmp_path,monkeypatch):
    import sys
    driver=driver_module();calls=[];name='wheeled_tasks.chassis.env'
    monkeypatch.delitem(sys.modules,name,raising=False)
    monkeypatch.setitem(sys.modules,'warp',types.SimpleNamespace(to_torch=lambda v:v))
    ChassisEnv=type('ChassisEnv',(),{'__module__':name})
    env=ChassisEnv();module=types.SimpleNamespace(ChassisEnv=ChassisEnv)
    class Parent:
        def __init__(self,env,output):
            assert name not in sys.modules
            calls.append('parent_constructed');monkeypatch.setitem(sys.modules,name,module)
        def _before_close(self): calls.append('parent_cleanup')
    class Capture:
        def __init__(self,env,env_module,*args):
            assert calls==['parent_constructed'] and env_module is module
            calls.append('capture_constructed')
        def install(self): calls.append('install')
        def cleanup(self): calls.append('capture_cleanup')
    clazz=driver.make_capture_observer(Parent,None,types.SimpleNamespace(MechanicalCapture=Capture),writer)
    assert name not in sys.modules and calls==[]
    observer=clazz(env,tmp_path)
    assert calls==['parent_constructed','capture_constructed','install']
    observer._before_close();assert calls[-2:]==['capture_cleanup','parent_cleanup']


@pytest.mark.parametrize('registered,correct_class',[(False,False),(True,False)])
def test_unconstructed_native_env_resolution_rejects_without_import(monkeypatch,registered,correct_class):
    import sys
    driver=driver_module();name='wheeled_tasks.chassis.env'
    ChassisEnv=type('ChassisEnv',(),{'__module__':name});env=ChassisEnv()
    monkeypatch.delitem(sys.modules,name,raising=False)
    if registered:monkeypatch.setitem(sys.modules,name,types.SimpleNamespace(ChassisEnv=object))
    with pytest.raises(ValueError,match='not constructed/registered'):driver.resolve_constructed_env_module(env)
    assert (name in sys.modules)==registered


def test_factory_has_no_native_env_import_AST():
    import ast
    tree=ast.parse((HERE/'run_capture_021.py').read_text())
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('wheeled_tasks.chassis') for n in imports)
    assert not any(isinstance(n,ast.Import) and any(x.name.startswith('wheeled_tasks.chassis') for x in n.names) for n in imports)
