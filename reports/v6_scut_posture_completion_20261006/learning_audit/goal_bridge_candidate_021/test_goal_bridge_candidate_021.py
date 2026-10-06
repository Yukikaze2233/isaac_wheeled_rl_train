"""CPU logic checks against exact frozen020 native producers, no App or PPO."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

HERE=Path(__file__).parent
BINDING=json.loads((HERE/'source_bindings.json').read_text())
FROZEN=Path(BINDING['base_source_root'])
sys.path.insert(0,str(FROZEN/'src'))
from wheeled_tasks.chassis.recovery import RecoveryFSM, RecoveryPhase
from wheeled_tasks.chassis.recovery_training import RecoveryTraining
from wheeled_tasks.chassis.recovery_prefix_curriculum import prefix_settings as old_settings, RecoveryPrefixCurriculum as OldPrefix

REL='src/wheeled_tasks/chassis/recovery_prefix_curriculum.py'
spec=importlib.util.spec_from_file_location('wheeled_tasks.chassis.goal_bridge_candidate_021',HERE/'candidate'/REL)
candidate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
TABLE=json.loads((FROZEN/'contracts/v6_recovery_profiles_v2.json').read_text())


def sampler_env(count=4096,ordinary=1449,seed=519):
    names=['r_stand','r_forward','r_yaw_positive','r_yaw_negative']
    commands=[[0.,0.,.305],[.3,0.,.305],[0.,.6,.305],[0.,-.6,.305]]
    specs={name:{'kind':'recovery_handoff','recovery_pose':'front_down_90',
        'recovery_followup_command':command} for name,command in zip(names,commands)}
    groups=[names[i%4] for i in range(count-ordinary)]+['ordinary']*ordinary
    joints=list(reversed(TABLE['tree_joint_names']))
    return NS(num_envs=count,device='cpu',cfg={'asset_manifest_sha256':TABLE['asset_manifest_sha256']},
        scene_groups=groups,kinds=['flat']*count,skills=NS(specs=specs),robot=NS(joint_names=joints),
        nominal=torch.zeros(count,len(joints)),generator=torch.Generator().manual_seed(seed))


def make_sampler(cls,*,bridge=False,clock=2579,count=4096,ordinary=1449):
    env=sampler_env(count,ordinary)
    settings=candidate.prefix_settings(bridge_original_goals=bridge) if cls is candidate.RecoveryPrefixCurriculum else old_settings()
    sampler=cls(env,TABLE,settings)
    sampler.reset(torch.arange(count),clock)
    return env,sampler


def outcome_rig():
    # Start at RL ownership to test the unchanged continuous release/outcome
    # predicates. This is a finite CPU sensor fixture, not a dynamic prefix.
    n=5; enabled=torch.tensor([1,1,1,1,0],dtype=torch.bool)
    env=NS(num_envs=n,device='cpu',policy_dt=.02,commands=torch.tensor([[9.,9.,9.]]*4+[[1.2,.4,.28]]),
        raw_height=torch.full((n,),-.9),command_target=torch.ones(n,2),command_acceleration=torch.ones(n))
    r=RecoveryTraining.__new__(RecoveryTraining);r.env=env;r.settings={'complete_started_blend':True}
    r.enabled=enabled;r.release_finished=torch.ones(n,dtype=torch.bool)
    q=torch.zeros(n,6); signs=torch.tensor([1.,1.,-1.,-1.])
    r.fsm=RecoveryFSM(n,'cpu',{'complete_started_blend':True},q[0],q[0],q[0],q[0],signs)
    r.fsm.reset(torch.arange(n),q,enabled);r.fsm.phase[enabled]=RecoveryPhase.RL
    r.followup_command=torch.tensor([[0.,0.,.305],[.3,0.,.305],[0.,.6,.305],[0.,-.6,.305],[0.,0.,.305]])
    r.required_platform_height=torch.zeros(n);r.elapsed=torch.zeros(n)
    for name in ('true_stable_seconds','true_stable_peak','followup_seconds'):
        setattr(r,name,torch.zeros(n))
    for name in ('premature_release','motion_release_observed','true_success'):
        setattr(r,name,torch.zeros(n,dtype=torch.bool))
    r.policy_motion_hold=torch.zeros(n,dtype=torch.bool);r.learning_mask=torch.ones(n,dtype=torch.bool)
    data=NS(q=q,dq=torch.zeros_like(q),gravity=torch.tensor([[0.,0.,-1.]]*n),gyro=torch.zeros(n,3),
        height=torch.full((n,),.305),support=torch.ones(n,dtype=torch.bool),body_clear=torch.ones(n,dtype=torch.bool),
        contact=torch.ones(n,2,dtype=torch.bool),undesired=torch.zeros(n,dtype=torch.bool),velocity=torch.zeros(n,3))
    return env,r,data


def policy(r,d):
    r.begin_policy()
    for _ in range(4):
        r.fsm.update(d.q,d.dq,d.gravity,d.gyro,d.height,d.support,d.body_clear)
    out=r.outcome(d.contact,d.undesired,d.gravity,d.gyro,d.height,d.velocity)
    r.override_commands()
    return out


def test_exact_base_source_and_unchanged_outcome_fsm_controller_bytes():
    assert hashlib.sha256((FROZEN/REL).read_bytes()).hexdigest()==BINDING['original_module_sha256']
    for name,digest in BINDING['unchanged_semantic_source_sha256'].items():
        assert hashlib.sha256((FROZEN/name).read_bytes()).hexdigest()==digest
    assert not BINDING['changed_production_files'] and not BINDING['contract_written']


def test_old_v18_settings_and_reset_outputs_remain_exact_when_not_opted_in():
    assert candidate.prefix_settings()==old_settings()
    old_env,old=make_sampler(OldPrefix);new_env,new=make_sampler(candidate.RecoveryPrefixCurriculum)
    for key in ('profile_ids','level_ids','selected_release','selected_followup','profile_reset_counts','level_reset_counts'):
        assert torch.equal(getattr(old,key),getattr(new,key))
    assert torch.equal(old_env.generator.get_state(),new_env.generator.get_state())
    assert old.runtime_state()==new.runtime_state()


@pytest.mark.parametrize('clock',[2294,3294,5294,8294])
def test_bridge_changes_only_l0_goal_same_sampler_rng_births_origins_and_other_lessons(clock):
    old_env,old=make_sampler(OldPrefix,clock=clock);new_env,new=make_sampler(candidate.RecoveryPrefixCurriculum,bridge=True,clock=clock)
    for key in ('profile_ids','level_ids','selected_release','profile_reset_counts','level_reset_counts'):
        assert torch.equal(getattr(old,key),getattr(new,key))
    assert torch.equal(old_env.generator.get_state(),new_env.generator.get_state())
    assert new.settings['origin_actor_update']==2294
    assert old_env.scene_groups==new_env.scene_groups and old_env.skills.specs==new_env.skills.specs
    l0=new.enabled & new.level_ids.eq(0);keep=~l0
    assert l0.any() and torch.equal(new.selected_followup[l0],new.original_followup[l0])
    assert torch.equal(old.selected_followup[keep],new.selected_followup[keep])
    assert new.selected_release[l0].eq(0).all()
    assert torch.equal(old_env.nominal,new_env.nominal)


def test_partial_reset_ordinary1449_and_nonreset_lanes_keep_every_buffer():
    env,s=make_sampler(candidate.RecoveryPrefixCurriculum,bridge=True,clock=8294)
    before={k:getattr(s,k).clone() for k in ('profile_ids','level_ids','selected_release','selected_followup')}
    ids=torch.tensor([0,2,4]);s.reset(ids,2294)
    kept=torch.ones(env.num_envs,dtype=torch.bool);kept[ids]=False
    for key,prior in before.items():assert torch.equal(getattr(s,key)[kept],prior[kept])
    assert int((~s.enabled).sum())==1449
    assert s.level_ids[~s.enabled].eq(-1).all() and s.profile_ids[~s.enabled].eq(-1).all()


def test_real_sampler_apply_lesson_transmits_original_motion_goal_only_to_reset_recovery_lanes():
    env,s=make_sampler(candidate.RecoveryPrefixCurriculum,bridge=True,clock=2294)
    recovery=NS(initial_release=torch.full((4096,),-1.),followup_command=torch.full((4096,3),-9.))
    lanes=s.enabled.nonzero().flatten();s.apply_lesson(recovery,lanes)
    assert torch.equal(recovery.followup_command[lanes],s.original_followup[lanes])
    assert torch.equal(recovery.initial_release[lanes],s.selected_release[lanes])
    assert recovery.followup_command[~s.enabled].eq(-9.).all()
    assert recovery.initial_release[~s.enabled].eq(-1.).all()
    ids=torch.tensor([0,2,4]);s.reset(ids,8294)
    before=recovery.followup_command.clone();s.apply_lesson(recovery,ids)
    untouched=torch.ones(4096,dtype=torch.bool);untouched[ids]=False
    assert torch.equal(recovery.followup_command[untouched],before[untouched])


@pytest.mark.parametrize('mutation',[lambda x:x.update(goal_bridge=True),lambda x:x.update(goal_bridge=False),
    lambda x:x.update(goal_bridge='early_release'),lambda x:x.update(stable_seconds=.5),
    lambda x:x.update(blend_seconds=.05),lambda x:x.update(origin_actor_update=0.)])
def test_explicit_bridge_cannot_shorten_or_bypass_native_gate(mutation):
    s=candidate.prefix_settings(bridge_original_goals=True);mutation(s)
    with pytest.raises(ValueError):candidate.validate_prefix_curriculum(s)


def test_200Hz_continuous_one_second_gate_then_original_goal_visible_and_ordinary_unchanged():
    env,r,d=outcome_rig();ordinary=env.commands[4].clone()
    for _ in range(49):
        assert not policy(r,d).any()
        assert env.commands[:4,:2].eq(0).all()
        assert not r.fsm.motion_released[:4].any()
    assert r.fsm.stable_ticks[:4].eq(196).all()
    assert not policy(r,d).any()  # Gate-ending interval gets no followup credit.
    assert r.fsm.stable_ticks[:4].eq(200).all() and r.fsm.motion_released[:4].all()
    assert torch.equal(env.commands[:4],r.followup_command[:4])
    assert not r.premature_release.any() and torch.equal(env.commands[4],ordinary)


def test_interrupted_sensor_dwell_does_not_release_after_200_total_good_ticks():
    _,r,d=outcome_rig()
    for _ in range(30):policy(r,d)
    d.dq[:4,0]=2.1;policy(r,d);d.dq.zero_()
    assert r.fsm.stable_ticks[:4].eq(0).all()
    for _ in range(20):policy(r,d)
    assert not r.fsm.motion_released[:4].any()
    for _ in range(30):policy(r,d)
    assert r.fsm.motion_released[:4].all()


def test_sensor_only_release_without_true_contact_dwell_still_fails_code7():
    _,r,d=outcome_rig();d.contact[:4,0]=False
    for _ in range(50):policy(r,d)
    assert r.fsm.motion_released[:4].all() and r.premature_release[:4].all()
    assert r.failed[:4].all() and not r.true_success.any()


def test_zero_motion_after_hold_cannot_pass_nonzero_forward_or_yaw_goal():
    _,r,d=outcome_rig()
    for _ in range(110):policy(r,d)
    assert r.true_success[:4].tolist()==[True,False,False,False]
    assert r.followup_seconds[1:4].eq(0).all()


def test_original_goal_requires_full_followup_second_and_loss_resets_it():
    _,r,d=outcome_rig()
    for _ in range(50):policy(r,d)
    d.velocity[1,0]=.3;d.gyro[2,2]=.6;d.gyro[3,2]=-.6
    for _ in range(49):assert not policy(r,d)[1:4].any()
    assert policy(r,d)[1:4].all()
    d.contact[:4,1]=False
    assert not policy(r,d)[:4].any() and r.followup_seconds[:4].eq(0).all()


def test_bridge_does_not_bypass_required_platform_evidence_or_undesired_contact():
    _,r,d=outcome_rig();r.required_platform_height[:4]=.2
    for _ in range(110):policy(r,d)
    assert r.premature_release[:4].all() and not r.true_success.any()
    _,r,d=outcome_rig();d.undesired[:4]=True
    for _ in range(110):policy(r,d)
    assert r.premature_release[:4].all() and not r.true_success.any()


def test_easy_bridge_goal_success_is_not_original_hard_chain_success():
    _,s=make_sampler(candidate.RecoveryPrefixCurriculum,bridge=True,clock=2294)
    done=s.enabled.clone();success=s.enabled.clone();zero=torch.zeros_like(done)
    s.record_outcomes(done,success,zero,zero)
    metrics=s.lesson_metrics(success)
    assert metrics['/recovery/lesson/prefix_stand/success_fraction']==1
    assert metrics['/recovery/full_chain_population']==0
    assert metrics['/recovery/full_chain_success']==0
