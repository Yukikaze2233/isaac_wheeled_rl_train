"""Small CPU tests for the new measurement selection; no App/PhysX/PPO."""
from pathlib import Path
import ast
import gzip
import importlib.util
import json
import types

import pytest
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'mechanical_capture_candidate_021'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


m = load(HERE/'mechanical_remaining_capture.py', 'remaining_measurement_test')
d = load(HERE/'run_remaining_capture_021.py', 'remaining_driver_test')
old = load(OLD/'test_mechanical_capture.py', 'remaining_inherited_CPU_fixture')
side = load(HERE.parent/'mechanical_side_speed_candidate_021/side_speed_capture.py', 'remaining_side_CPU_helper')


def writer(path, value): path.write_text(json.dumps(value, allow_nan=False))


def env_fixture():
    old.m = m
    env = old.fake_env()
    fsm = env.recovery.fsm
    fsm.config = {'side_speed_rad_s':1.5}
    fsm.phase_ticks = torch.zeros(4096, dtype=torch.long)
    fsm.side_start = torch.zeros(4096,4)
    fsm.side_direction = torch.ones(4096,4)
    fsm.axis_signs = torch.tensor([1.,1.,-1.,-1.])
    for index, lane in enumerate(m.LANES):
        env.recovery.pose_curriculum.profile_ids[lane] = index
        env.recovery.pose_curriculum.level_ids[lane] = m.LEVELS[index]
        env.scene_groups[lane] = m.GROUPS[index]
    view = env.contact_views[0][1]
    view.sensor_count = 6
    view.matrix = torch.zeros(6,1,3); view.net = torch.zeros(6,3)
    view.count = torch.zeros(6,1,dtype=torch.long); view.start = torch.zeros(6,1,dtype=torch.long)
    return env


def test_private_copy_only_declared_measurement_edits():
    source = (OLD/'mechanical_capture.py').read_text()
    candidate = (HERE/'mechanical_remaining_capture.py').read_text()
    expected = source.replace('LANES = (1708, 2216)', 'LANES = (881, 1877, 2121)')
    expected = expected.replace('PROFILES = ("step200_wheels_split", "double_step_first_riser_split")',
        "PROFILES = ('front_down_90', 'step200_lip_near_shell', 'step200_upright_top')\nLEVELS = (2, 2, 1)\nGROUPS = ('recovery_front_down_90_forward', 'recovery_step200_lip_near_shell_stand', 'recovery_step200_prone_belly_top_forward')")
    expected = expected.replace('TRACE_STEPS = 260', 'TRACE_STEPS = 2400')
    expected = expected.replace('for lane, name in zip(LANES, PROFILES):',
        'for lane, name, level, group in zip(LANES, PROFILES, LEVELS, GROUPS):')
    expected = expected.replace('int(r.pose_curriculum.level_ids[lane]) == 2',
        'int(r.pose_curriculum.level_ids[lane]) == level\n                and env.scene_groups[lane] == group')
    expected = expected.replace('len(v)==2', 'len(v)==len(LANES)')
    expected = expected.replace("'First1.3s capture incomplete'", "'Selected firstbirth12s capture incomplete'")
    assert candidate == expected
    assert m.GUARD_M == .003 and m.LANES == (881,1877,2121) and m.LEVELS == (2,2,1)
    oldtree, newtree = ast.parse(source), ast.parse(candidate)
    for name in ('same_callback','pin_geometry','contact_slices','HookSet'):
        a=next(n for n in oldtree.body if getattr(n,'name',None)==name)
        b=next(n for n in newtree.body if getattr(n,'name',None)==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


@pytest.mark.parametrize('field', ['level','profile','group','enabled'])
def test_exact_L1_actual_birth_binding_rejects_wrong_input(tmp_path, field):
    e = env_fixture()
    if field == 'level': e.recovery.pose_curriculum.level_ids[2121] = 2
    if field == 'profile': e.recovery.pose_curriculum.profile_ids[2121] = 0
    if field == 'group': e.scene_groups[2121] = 'recovery_step200_upright_top_forward'
    if field == 'enabled': e.recovery.enabled[2121] = False
    with pytest.raises(ValueError, match='lane/lesson/profile'):
        m.MechanicalCapture(e, types.SimpleNamespace(), tmp_path, writer, lambda v:v)
    assert not list(tmp_path.iterdir())


def test_three_lane_actual_once_cadence_slice_reset_and_cleanup(tmp_path):
    module = load(HERE/'mechanical_remaining_capture.py', 'remaining_one_interval_private')
    d.add_native_reference_telemetry(module)
    e=env_fixture(); calls=[]; env_module=types.SimpleNamespace()
    def update(env, dt): calls.append('update');env.robot.data.joint_pos.torch.add_(1)
    env_module.update_robot_data=update
    def setter(**kw): calls.append('setter');return kw['target']
    e.robot.set_joint_effort_target_index=setter
    def reset(ids): calls.append('reset');env_module.update_robot_data(e,.005)
    e.reset=reset
    target=torch.zeros(4096,2)
    for index,lane in enumerate(m.LANES):
        target[lane]=torch.tensor([index+1.,index+10.])
        e.v5.requested_motor_effort[lane].fill_(index+1.)
    result=(None,torch.tensor([9.]),None,dict(diagnostics={
        'done':torch.zeros(4096,dtype=torch.bool),'reasons':{}}))
    def step(actions):
        for _ in range(4):
            assert e.robot.set_joint_effort_target_index(target=target) is target
            env_module.update_robot_data(e,.005)
        e.reset(torch.tensor([2121]));result[3]['diagnostics']['done'][2121]=True
        return result
    e.step=step
    rng=torch.get_rng_state().clone()
    capture=module.MechanicalCapture(e,env_module,tmp_path,writer,lambda v:v);capture.install()
    assert e.step('original') is result and calls==['setter','update']*4+['reset','update']
    assert torch.equal(rng,torch.get_rng_state())
    assert capture.row_counts=={881:5,1877:5,2121:5}
    assert capture.pre_reset[2121]['state']['birth_id']==1 and capture.birth_ids[2121]==2
    assert not capture.alive[2121] and capture.alive[881]
    assert capture.cleanup()==[] and capture.cleanup()==[]
    assert e.step is step and e.reset is reset and env_module.update_robot_data is update
    rows=list(d.decoded_rows(tmp_path/'mechanical_5ms.jsonl.gz'))
    layout=json.loads((tmp_path/'mechanical_layout.json').read_text())
    assert layout['lanes']==list(m.LANES) and layout['profiles']==list(m.PROFILES)
    for row in rows:
        if row['physical_step']!=1: continue
        lane=row['state']['lane'];index=m.LANES.index(lane);control=row['applied_control']
        assert control['full_joint_effort_target']==[index+1.,index+10.]
        assert control['native_requested_motor_effort_nm_at_setter']==[index+1.]*6
        assert control['side_reference_audit']['native_config_side_speed_rad_s']==1.5
        assert len(control['side_reference_audit']['axis_signs'])==4
        assert row['contacts']['env_contact_force_cache_consumed'] is False


def make_native_outputs(path):
    path.mkdir()
    for name, values in [('first_birth_rows.jsonl.gz',[{'lane':i,'value':1.} for i in range(4096)]),
                         ('events.jsonl.gz',[{'lane':881,'terminal':True,'reward':-200.}]),
                         ('mechanical_global_peaks.jsonl.gz',[{'lane':881,'gap':.027620118}])]:
        with gzip.open(path/name,'wt') as stream:
            for value in values: stream.write(json.dumps(value)+'\n')
    for name in ('identity.json','payment_summary.json','teaching_summary.json',
                 'side_override_applied.json','side_override_preclose.json'):
        writer(path/name,{'same':1.,'qualification':None})
    writer(path/'report.json',{'physics_steps':2400,'started_at_utc':'time',
        'observer_included_wall_seconds':99.,'mechanical_summary_sha256':'changed'})


@pytest.mark.parametrize('change', ['none','event','payment','report','missing_birth'])
def test_full_native_reproduction_rejects_real_changes(tmp_path, change):
    a,b=tmp_path/'a',tmp_path/'b';make_native_outputs(a);make_native_outputs(b)
    writer(b/'report.json',{'physics_steps':2400,'started_at_utc':'different',
        'observer_included_wall_seconds':19.,'mechanical_summary_sha256':'new'})
    if change=='event':
        with gzip.open(b/'events.jsonl.gz','wt') as f:f.write(json.dumps({'lane':881,'terminal':False,'reward':-200.})+'\n')
    if change=='payment':writer(b/'payment_summary.json',{'same':-1.,'qualification':None})
    if change=='report':writer(b/'report.json',{'physics_steps':2399})
    if change=='missing_birth':
        with gzip.open(b/'first_birth_rows.jsonl.gz','wt') as f:
            for i in range(4095):f.write(json.dumps({'lane':i,'value':1.})+'\n')
    if change=='none': assert d.compare_native_outputs(a,b,side.compare_bits)['tolerance_used'] is False
    else:
        with pytest.raises(ValueError,match='differs'):d.compare_native_outputs(a,b,side.compare_bits)


def test_mode_cpu_default_and_no_cuda_by_fixtures():
    tree=ast.parse((HERE/'run_remaining_capture_021.py').read_text())
    mode=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--mode')
    assert next(k.value.value for k in mode.keywords if k.arg=='default')=='cpu'
    assert not torch.cuda.is_initialized()
