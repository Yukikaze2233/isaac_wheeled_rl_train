from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import math

import pytest
import torch

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


rate=load(HERE/'capture_speed_override.py','capture_rate_fixture')
native=load('/tmp/scut-script-frozen-022-pbifeg_2/src/wheeled_tasks/chassis/recovery.py','actual022_recovery_CPU_fixture')
hooks=load(AUDIT/'mechanical_remaining_capture_candidate_021/mechanical_remaining_capture.py','frozen_remaining_CPU_hook')
driver=load(HERE/'run_capture_speed_022.py','capture_rate_driver_fixture')


def fsm(count=16,config=None):
    nominal=torch.tensor([-.8,-.6,0,.8,.6,0.])
    fold=torch.tensor([-.4,-.4,0,.4,.4,0.])
    out=native.RecoveryFSM(count,'cpu',dict(side_speed_rad_s=1.5)|dict(config or {}),nominal,fold,nominal,fold,[1,1,-1,-1])
    q=torch.zeros(count,6)
    out.reset(torch.arange(count),q,True);out.phase[:]=6
    return out


def inputs(n=16):
    q=torch.zeros(n,6);dq=q.clone();g=torch.zeros(n,3);g[:,2]=-1
    gyro=torch.zeros(n,3);gyro[:,1]=2.;height=torch.full((n,),.305)
    support=torch.ones(n,dtype=torch.bool);clear=support.clone()
    return q,dq,g,gyro,height,support,clear


def writer(path,value):
    with path.open('x') as f:json.dump(value,f,allow_nan=False)


class Env:
    def __init__(self,system):
        self.recovery=SimpleNamespace(fsm=system);self.args=inputs(system.count)
        self.step_calls=self.update_calls=0;self.returned=object()
        self.prior_update=system.update
        def update(*args):
            self.update_calls+=1
            return self.prior_update(*args)
        system.update=update

    def step(self,actions):
        self.step_calls+=1
        for unused in range(4):self.recovery.fsm.update(*self.args)
        return self.returned


def install(tmp_path,system=None):
    env=Env(system or fsm());h=hooks.HookSet()
    override=rate.CaptureSpeedOverride(env,h,tmp_path,writer,hooks.plain,lambda f:None)
    original=env.recovery.fsm.update
    override.apply()
    return env,override,original


@pytest.mark.parametrize('aligned',[False,True])
@pytest.mark.parametrize('transition',[False,True])
def test_same_input_prediction_is_exact_native_current_tick(aligned,transition):
    system=fsm(config={'capture_speed_rad_s':1.});args=list(inputs())
    args[5][:]=aligned
    if transition:args[3].zero_()
    selected,ref1,ref4=rate.same_input_references(system,*args[:6])
    before=[v.clone() for v in args];rng=torch.get_rng_state().clone()
    result=system.update(*args)
    assert result is system.script_targets and torch.equal(result[:,system.LEG_IDS][selected],ref1[selected])
    assert (ref4[selected]!=ref1[selected]).any()
    assert all(torch.equal(a,b) for a,b in zip(args,before)) and torch.equal(rng,torch.get_rng_state())
    if transition and aligned:assert system.phase.eq(7).all()


def test_invalid_and_same_tick_reroute_do_not_claim_capture_intervention(tmp_path):
    system=fsm();system.unsupported_ticks[:]=29
    args=inputs();args[2][:]=torch.tensor([1.,0.,0.]);args[5][:]=False
    # lane0 is invalid; lane1 reroutes this same reference; remaining lanes stay CAPTURE.
    args[2][0].zero_();system.unsupported_ticks[2:]=0
    selected,ref1,ref4=rate.same_input_references(system,*args[:6])
    assert not selected[0] and not selected[1] and selected[2:].all()
    system.config[rate.KEY]=1.;result=system.update(*args)
    assert system.phase[0]==10 and system.phase[1]==1
    assert torch.equal(result[:,system.LEG_IDS][selected],ref1[selected])


def test_global_first_change_can_be_outside_selected_lanes_and_exact_objects(tmp_path):
    system=fsm(4096);system.phase[:]=9;system.phase[7]=6
    env,override,original=install(tmp_path,system)
    before=[v.clone() for v in env.args];rng=torch.get_rng_state().clone()
    assert env.step(object()) is env.returned
    assert env.step_calls==1 and env.update_calls==4
    assert override.first['all_population_lanes']==[7]
    assert override.first['reference']==1 and override.first['input_physical_step']==0
    assert override.policy_completed==1 and override.reference_completed==4
    assert all(torch.equal(a,b) for a,b in zip(env.args,before)) and torch.equal(rng,torch.get_rng_state())
    assert not override.cleanup() and env.recovery.fsm.update is original
    assert env.recovery.fsm.config['capture_speed_rad_s']==4.


def test_only_config_key_changed_and_600_2400_restore(tmp_path):
    env,override,original=install(tmp_path)
    prior=override.before.copy()
    for unused in range(600):assert env.step(None) is env.returned
    assert override.fsm.config==prior|{'capture_speed_rad_s':1.}
    assert env.step_calls==600 and env.update_calls==2400
    assert not override.finish();receipt=json.loads((tmp_path/'capture_speed_override_preclose.json').read_text())
    assert receipt['complete'] and receipt['config_restored'] and receipt['hooks_restored']
    assert receipt['sole_changed_keys']==['capture_speed_rad_s'] and receipt['before']==prior
    assert override.fsm.config==prior and override.fsm.update is original
    assert not override.cleanup() and 'step' not in vars(env)


def test_source_update_exception_preserved_and_partial_cleanup(tmp_path):
    system=fsm();env=Env(system);sentinel=RuntimeError('original-native-update-failed')
    def failed(*args):raise sentinel
    system.update=failed;h=hooks.HookSet()
    override=rate.CaptureSpeedOverride(env,h,tmp_path,writer,hooks.plain,lambda f:None);override.apply()
    with pytest.raises(RuntimeError) as caught:env.step(None)
    assert caught.value is sentinel
    assert override.reference_attempted==1 and override.reference_completed==0
    assert override.policy_attempted==1 and override.policy_completed==0
    assert not override.cleanup() and system.update is failed and system.config['capture_speed_rad_s']==4.


@pytest.mark.parametrize('foreign',['value','object','callback'])
def test_foreign_config_or_callback_not_overwritten(tmp_path,foreign):
    env,override,original=install(tmp_path)
    if foreign=='value':override.fsm.config['orbit_speed_rad_s']=2.
    elif foreign=='object':override.fsm.config=dict(override.fsm.config)
    else:override.fsm.update=lambda *args:None
    errors=override.cleanup();assert errors
    if foreign=='value':assert override.fsm.config['orbit_speed_rad_s']==2. and override.fsm.config['capture_speed_rad_s']==1.
    if foreign=='object':assert override.fsm.config is not override.config_object
    if foreign=='callback':assert override.fsm.update is not original


def test_incomplete_finish_rejects_and_cleans_without_native_success(tmp_path):
    env,override,original=install(tmp_path);env.step(None)
    with pytest.raises(ValueError,match='600/2400'):override.finish()
    assert not override.cleanup()
    receipt=json.loads((tmp_path/'capture_speed_override_preclose.json').read_text())
    assert not receipt['complete'] and receipt['config_restored']


def test_outer_late_parent_finish_failure_records_before_close_and_restores(tmp_path,monkeypatch):
    env=Env(fsm());original=env.recovery.fsm.update;events=[];sentinel=RuntimeError('late-parent-finish')
    class Parent:
        def __init__(self,env,output):events.append('parent_constructed')
        def _before_close(self):events.append('parent_before_close')
        def finish(self,output):
            assert env.recovery.fsm.config['capture_speed_rad_s']==4.
            assert env.recovery.fsm.update is original and 'step' not in vars(env)
            events.append('parent_finish');raise sentinel
    source=SimpleNamespace(make_source_observer=lambda *args:Parent,
        load=lambda *args:hooks,MECHANICAL='unused',require=rate.require,
        sha=lambda p:'unused')
    monkeypatch.setattr(driver,'authenticate_fsm',lambda *args:events.append('late_native_auth'))
    outer=driver.make_observer(Parent,source,None,None,rate,writer,'unused')
    obj=outer(env,tmp_path);assert events==['parent_constructed','late_native_auth']
    for unused in range(600):env.step(None)
    with pytest.raises(RuntimeError) as caught:obj.finish(tmp_path)
    assert caught.value is sentinel
    assert events[-2:]==['parent_finish','parent_before_close']
    assert (tmp_path/'capture_speed_observer_failure.json').exists()
    assert obj.capture_override.closed and obj.capture_override.fsm.config['capture_speed_rad_s']==4.
