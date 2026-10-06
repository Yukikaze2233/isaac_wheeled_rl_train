"""CPU ABI/cadence/state-clock and failure tests; no App, policy or CUDA call."""
import ast
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

HERE = Path(__file__).resolve().parent
EXPLORATION = HERE.parent
FROZEN = Path('/tmp/scut-recovery-frozen-021-qv1ukmbr')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = load(HERE/'target_payment_capture.py', 'target_CPU_helper')
D = load(HERE/'run_target_payment_021.py', 'target_CPU_driver')
M = load(EXPLORATION/'mechanical_capture_candidate_021/mechanical_capture.py', 'target_CPU_original_HookSet')


def writer(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def fake(tmp_path, fail=None, terminal=False):
    n = 4096
    q, dq = torch.zeros(n, 6), torch.full((n, 6), .1)
    b = lambda value: torch.full((n,), value, dtype=torch.bool)
    calls, objects = {}, {}
    def call(name):
        calls[name] = calls.get(name, 0)+1
        if fail == name:
            raise RuntimeError('original '+name+' failure')

    class V5:
        def decode(self, actions, encoder):
            call('decode')
            result = (torch.full((n, 4), 1.), torch.full((n, 2), 2.), actions.clamp(-3., 3.))
            objects['decode'] = result
            return result
        def motor_efforts(self, control_q, feedback_dq, legs, wheels):
            call('pd')
            self.requested_motor_effort = torch.full((n, 6), 3.)
            self.current_motor_bounds = torch.full((n, 6), 40.)
            result = torch.full((n, 6), 2.)
            objects['pd'] = result
            return result

    class Delay:
        cfg = {'feedback_pd': 'fresh_physics_feedback'}
        lags = torch.full((n, 6), 3, dtype=torch.long)
        def targets(self, legs, wheels):
            call('delay')
            result = (legs+.5, wheels+.7)
            objects['delay'] = result
            return result

    class FSM:
        config = dict(side_speed_rad_s=1.5, stable_seconds=1., handover_gyro_max=.75)
        phase = torch.full((n,), 9, dtype=torch.long)
        stable_pass_masks = {name:b(True) for name in H.GATES}
        stable_pass = holding_rl = ready = b(True)
        stable_ticks = torch.zeros(n, dtype=torch.long)
        stable_peak_ticks = torch.zeros(n, dtype=torch.long)
        stable_break_count = torch.zeros(n, dtype=torch.long)
        motion_released = b(False)
        targets = torch.zeros(n,6)
        def update(self, cq, cdq, gravity, gyro, height, support, clear):
            call('fsm')
            self.stable_pass_masks['wheel_speed'].fill_(calls['fsm'] % 4 != 2)
            self.stable_ticks.fill_(calls['fsm'])
            self.stable_break_count.fill_(11)
            self.stable_peak_ticks.fill_(16)
            # Simulate the later recovery request overwrite, not another PD call.
            e.v5.requested_motor_effort.fill_(99.)
            objects['fsm'] = self.targets
            return self.targets

    module = types.SimpleNamespace()
    def cost(components, motor_dq, gyro, enabled, owned, held, settings, *, diagnostics):
        call('cost')
        diagnostics.clear()
        diagnostics['mask'] = enabled & owned & held
        for axis in ('leg', 'wheel', 'gyro'):
            for name, value in (('gate_relative_error', 1.), ('broad_density', -.4), ('fine_density', -.6)):
                diagnostics[axis+'/'+name] = torch.full((n,), value)
        for name, value in zip(H.TERMS, (-1., -2., -3.)):
            components[name] = torch.full((n,), value)
        objects['cost'] = components
        return components
    module.apply_recovery_hold_costs = cost
    profiles = ['unused']*22
    profiles[21] = 'double_step_first_riser_split'
    recovery = types.SimpleNamespace(enabled=b(True), release_finished=b(True),
        fsm=FSM(), pose_curriculum=types.SimpleNamespace(level_ids=torch.full((n,),2),
            profile_ids=torch.full((n,),21),profile_names=profiles))
    parent = types.SimpleNamespace(live=b(True), payment=types.SimpleNamespace(active=None))
    e = types.SimpleNamespace(num_envs=n, dt=.005, policy_dt=.02, decimation=4, usb_transport=None,
        signal_delay=Delay(),v5=V5(),recovery=recovery,cfg={'closure_gap_termination_m':.003},
        recovery_hold_cost_settings={'enabled':True},recovery_hold_cost_diagnostics={})
    actions = torch.full((n,6),5.)
    components = {}
    def step(a):
        call('step')
        tick=calls['step']
        parent.payment.active=dict(tick=tick,enabled=recovery.enabled.clone(),
            lesson=recovery.pose_curriculum.level_ids.clone(),profile=recovery.pose_curriculum.profile_ids.clone(),
            hold=b(True),live=parent.live.clone())
        dec=e.v5.decode(a,q)
        assert dec is objects['decode']
        for k in range(4):
            delayed=e.signal_delay.targets(dec[0],dec[1]);assert delayed is objects['delay']
            pd=e.v5.motor_efforts(q,dq,*delayed);assert pd is objects['pd']
            f=recovery.fsm.update(q,dq,torch.zeros(n,3),torch.full((n,3),.2),torch.full((n,),.305),b(True),b(True))
            assert f is objects['fsm']
        result=module.apply_recovery_hold_costs(components,dq,torch.full((n,3),.3),recovery.enabled,b(True),b(True),
            e.recovery_hold_cost_settings,diagnostics=e.recovery_hold_cost_diagnostics)
        assert result is components is objects['cost']
        reward=torch.full((n,),-.12);done=b(False)
        if terminal:done[H.LANE]=True
        diagnostics=dict(reward=reward,done=done,terminated=done.clone(),success=b(False),
            commands=torch.zeros(n,3),recovery_motion_released=b(False))
        extra=dict(diagnostics=diagnostics,time_outs=b(False),policy_learning_mask=b(True),native_reward_components=components)
        result=(object(),reward,done,extra);objects['step']=result
        # Actual env has reset already; old diagnostics/payment snapshots remain.
        if terminal:
            recovery.pose_curriculum.level_ids[H.LANE]=0
            recovery.pose_curriculum.profile_ids[H.LANE]=3
        return result
    e.step=step
    capture=H.TargetPaymentCapture(e,parent,module,M.HookSet(),tmp_path,writer)
    return e,parent,module,capture,calls,objects,actions


def decoded(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def test_once_exact_objects_and_distinct_actual_targets_clock_masks(tmp_path):
    e,parent,module,capture,calls,objects,actions=fake(tmp_path)
    original_decode=e.v5.decode.__func__;original_delay=e.signal_delay.targets.__func__
    rng=torch.get_rng_state().clone();capture.install()
    returned=e.step(actions)
    assert returned is objects['step'] and calls==dict(step=1,decode=1,delay=4,pd=4,fsm=4,cost=1)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(actions,torch.full_like(actions,5.))
    assert not capture.cleanup() and not capture.cleanup()
    assert e.v5.decode.__func__ is original_decode and 'decode' not in vars(e.v5)
    assert e.signal_delay.targets.__func__ is original_delay and 'targets' not in vars(e.signal_delay)
    row=decoded(tmp_path/'pd_target_payment_2216.jsonl.gz')[0]
    assert row['decode']['leg_targets_rad']==[1.]*4 and row['decode']['wheel_targets_rad_s']==[2.]*2
    assert row['decode']['clipped_action_CONTROL_order']==[3.]*6
    for k,s in enumerate(row['substeps'],1):
        assert s['input_physical_step']==k-1 and s['postphysics_step']==k
        assert s['pd']['actual_PD_leg_targets_rad']==[1.5]*4
        assert s['delay']['delayed_legs_rad']==[1.5]*4
        assert s['pd']['original_RL_PD_preclip_request_nm']==[3.]*6
        assert s['pd']['original_RL_PD_clipped_torque_nm']==[2.]*6
        assert s['fsm']['native_gate_masks']['wheel_speed']==(k!=2)
        assert s['fsm']['stable_break_count']==11 and s['fsm']['stable_peak_ticks']==16
    assert row['hold_producer']['physical_end_step']==4 and row['hold_producer']['last_gate_input_physical_step']==3
    assert row['policy_end']['actual_component_payments']==dict(leg_velocity=-.02,wheel_velocity=-.04,horizontal_angular_velocity=-.06)
    assert D.validate_journal([row],capture.summary())['last_reference_input_physical_step']==3


def test_old_terminal_labels_whole_actor_and_no_later_birth_rows(tmp_path):
    e,parent,module,capture,calls,objects,a=fake(tmp_path,terminal=True);capture.install()
    assert e.step(a) is objects['step']
    parent.live[H.LANE]=False
    assert e.step(a) is objects['step']
    assert not capture.cleanup()
    rows=decoded(tmp_path/'pd_target_payment_2216.jsonl.gz')
    assert len(rows)==1 and capture.first_terminal['policy_tick']==1
    assert rows[0]['old_episode']['lesson_id']==2 and rows[0]['old_episode']['profile_id']==21
    assert rows[0]['policy_end']['terminated'] and rows[0]['policy_end']['whole_interval_actor_valid']
    assert calls==dict(step=2,decode=2,delay=8,pd=8,fsm=8,cost=2)
    assert D.validate_journal(rows,capture.summary())['right_censored_unfinished'] is False


@pytest.mark.parametrize('where',('step','decode','delay','pd','fsm','cost'))
def test_original_exception_same_error_rng_and_beforeclose_restore(tmp_path,where):
    e,parent,module,capture,calls,objects,a=fake(tmp_path,fail=where)
    before=(e.step,e.v5.decode.__func__,e.signal_delay.targets.__func__,e.v5.motor_efforts.__func__,e.recovery.fsm.update.__func__,module.apply_recovery_hold_costs)
    rng=torch.get_rng_state().clone();capture.install()
    with pytest.raises(RuntimeError,match='original '+where+' failure'):e.step(a)
    assert calls[where]==1 and not capture.in_step and torch.equal(rng,torch.get_rng_state())
    assert not capture.cleanup()
    after=(e.step,e.v5.decode.__func__,e.signal_delay.targets.__func__,e.v5.motor_efforts.__func__,e.recovery.fsm.update.__func__,module.apply_recovery_hold_costs)
    assert before==after
    failure=json.loads((tmp_path/'pd_target_payment_failure.json').read_text())
    assert failure['app_close_returned'] is None and not failure['observation']['complete']


def test_full600_counts_not_extra_producers_preclose_complete(tmp_path):
    e,parent,module,capture,calls,objects,a=fake(tmp_path);capture.install()
    for _ in range(600):assert e.step(a) is objects['step']
    summary=capture.finish()
    assert summary['complete'] and summary['policy_steps']==600 and summary['expected_physical_steps']==2400
    assert summary['original_calls_completed']==dict(decode=600,delay=2400,pd=2400,fsm=2400,cost=600)
    assert summary['first_birth_policy_rows']==600 and summary['right_censored_unfinished']
    assert summary['hooks_restored'] and not (tmp_path/'pd_target_payment_failure.json').exists()


def test_foreign_hook_owner_preserved_in_failure(tmp_path):
    e,parent,module,capture,calls,objects,a=fake(tmp_path);capture.install()
    foreign=lambda *a:None;e.v5.decode=foreign
    assert capture.cleanup() and e.v5.decode is foreign


def test_partial_install_restores_and_records_failure(tmp_path):
    e,parent,module,capture,calls,objects,a=fake(tmp_path)
    class FailSecond(M.HookSet):
        def install(self,owner,name,callback):
            if len(self.rows)==1:raise RuntimeError('second install failure')
            return super().install(owner,name,callback)
    capture.hooks=FailSecond()
    with pytest.raises(RuntimeError,match='second install'):capture.install()
    assert capture.closed and 'decode' not in vars(e.v5)
    assert (tmp_path/'pd_target_payment_failure.json').exists()


def test_extra_call_outside_policy_rejected_before_original(tmp_path):
    e,parent,module,capture,calls,objects,a=fake(tmp_path);capture.install()
    with pytest.raises(ValueError,match='outside admitted'):e.v5.decode(a,torch.zeros_like(a))
    assert not calls and not capture.cleanup()


def test_no_cold_native_import_and_default_cpu():
    tree=ast.parse((HERE/'run_target_payment_021.py').read_text())
    assert not any(isinstance(node,ast.ImportFrom) and (node.module or '').startswith(('isaac','wheeled_tasks')) for node in ast.walk(tree))
    mode=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--mode')
    assert next(k.value.value for k in mode.keywords if k.arg=='default')=='cpu'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('error',('constructor','parent_finish'))
def test_parent_failure_closes_own_hooks_before_parent(tmp_path,error,monkeypatch):
    order=[]
    class SmallCapture:
        def __init__(self,*args):order.append('construct')
        def install(self):
            order.append('install')
            if error=='constructor':raise RuntimeError('capture constructor failure')
        def finish(self):order.append('own_finish');return {'complete':True}
        def cleanup(self):order.append('own_cleanup');return []
    class Parent:
        def __init__(self,env,output):order.append('parent_construct')
        def finish(self,output):order.append('parent_finish');raise RuntimeError('parent late failure')
        def _before_close(self):order.append('parent_close')
    monkeypatch.setattr(H,'TargetPaymentCapture',SmallCapture)
    Observer=H.make_observer(Parent,types.SimpleNamespace(),M.HookSet,writer,lambda p:'digest')
    if error=='constructor':
        with pytest.raises(RuntimeError,match='capture constructor'):Observer(None,tmp_path)
        assert order==['parent_construct','construct','install','own_cleanup','parent_close']
    else:
        owner=Observer(None,tmp_path)
        with pytest.raises(RuntimeError,match='parent late'):owner.finish(tmp_path)
        assert order==['parent_construct','construct','install','own_finish','parent_finish','own_cleanup','parent_close']
        assert (tmp_path/'pd_target_payment_failure.json').exists()


def test_auth_accepts_real_native_classes_and_exact_inherited_update_closure():
    sys.path.insert(0,str(FROZEN/'src'))
    from wheeled_tasks.chassis.v5_control import V5Control
    from wheeled_tasks.chassis.signal_delay import PhysicalSignalDelay
    from wheeled_tasks.chassis.recovery import RecoveryFSM
    from wheeled_tasks.chassis import rewards
    env=types.SimpleNamespace(v5=V5Control.__new__(V5Control),signal_delay=PhysicalSignalDelay.__new__(PhysicalSignalDelay),
        recovery=types.SimpleNamespace(fsm=RecoveryFSM.__new__(RecoveryFSM)))
    source=EXPLORATION.parent/'probe_prefix_mean_stochastic_019.py'
    tree=ast.parse(source.read_text());klass=next(n for n in tree.body if getattr(n,'name',None)=='NativeObserver')
    init=next(n for n in klass.body if getattr(n,'name',None)=='__init__')
    update=next(n for n in init.body if getattr(n,'name',None)=='update')
    stub=ast.parse('class NativeObserver:\n    def __init__(self,r):\n        pass').body[0]
    stub.body[0].body=[update,ast.Assign(targets=[ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='callback',ctx=ast.Store())],value=ast.Name(id='update',ctx=ast.Load()))]
    ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[stub],type_ignores=[])),str(source),'exec'),ns)
    parent=ns['NativeObserver'](env.recovery)
    parent.original_update=env.recovery.fsm.update;parent.cost_prior=rewards.apply_recovery_hold_costs
    env.recovery.fsm.update=parent.callback
    import hashlib
    capture=types.SimpleNamespace(sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
    assert D.authenticate_original_callbacks(env,parent,str(FROZEN),capture) is rewards
    foreign=types.SimpleNamespace(original_update=parent.original_update,cost_prior=parent.cost_prior)
    with pytest.raises(ValueError,match='observer chain differs'):D.authenticate_original_callbacks(env,foreign,str(FROZEN),capture)


@pytest.mark.parametrize('change',('PDtarget','gate','payment_dt','reference_clock','ownership','row_gap'))
def test_readback_rejects_missing_misclocked_or_conflated_actual_fields(tmp_path,change):
    e,parent,module,capture,calls,objects,a=fake(tmp_path);capture.install();e.step(a);capture.cleanup()
    rows=decoded(tmp_path/'pd_target_payment_2216.jsonl.gz');row=rows[0]
    if change=='PDtarget':row['substeps'][0]['pd']['actual_PD_leg_targets_rad']=[1.]*4
    if change=='gate':row['substeps'][0]['fsm']['native_gate_masks'].pop('gyro')
    if change=='payment_dt':row['policy_end']['density_dt_s']=.005
    if change=='reference_clock':row['substeps'][0]['input_physical_step']=1
    if change=='ownership':row['policy_end']['whole_interval_actor_valid']=False
    if change=='row_gap':row['policy_tick']=2
    with pytest.raises((ValueError,KeyError)):D.validate_journal(rows,capture.summary())
