"""Offline tensor and contract tests, NOT physics/RL training acceptance."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.v40.contract import load_contract, validate_contract, contract_digest, audit_asset, validate_asset, make_run_manifest
from wheeled_tasks.v40.core import HistoryStack, build_observation, build_critic, decode_targets, motor_torque_limit, compute_torques, compute_reward_terms


@pytest.fixture
def c():
    return copy.deepcopy(load_contract())


def state(c, n=2):
    q = torch.tensor(c['joints']['nominal_positions']).repeat(n, 1)
    return q, torch.zeros_like(q), torch.zeros((n, 3)), torch.tensor([[0., 0., -1.]]).repeat(n, 1), torch.tensor([[0., 0., .32]]).repeat(n, 1)


def test_contract_identity_and_digest(c):
    assert c['joints']['action_order'][4] == 'R_jonit2'
    assert len(contract_digest(c)) == 64
    original = contract_digest(c)
    c['rewards']['weights']['height'] += 1
    assert original != contract_digest(c)


@pytest.mark.parametrize('path,value', [
    (('schema_version',), True), (('contract_id',), 'old-v33'),
    (('observations','actor_dim'), 35), (('observations','critic_dim'), 125),
    (('observations','history_length'), 4), (('observations','clip'), float('nan')),
    (('observations','empirical_normalization'), True), (('actions','wheel_velocity_scale'), 1000),
    (('timing','decimation'), 4), (('joints','knee_soft_margin'), 1.),
    (('actuators','wheel','gear_ratio'), 19), (('actuators','wheel','gearbox_efficiency'), 1.2),
])
def test_invalid_contract_rejected(c, path, value):
    item = c
    for key in path[:-1]: item = item[key]
    item[path[-1]] = value
    with pytest.raises(ValueError): validate_contract(c)


def test_observation_layout_and_wheel_angle_invariance(c):
    q, v, omega, gravity, command = state(c)
    q[:, 0] += 2 * math.pi
    q[:, 3] -= 4 * math.pi
    q[:, 2] = 1e6
    q[:, 5] = -2e6
    omega[:, 0] = 2
    v[:, 2] = 11
    last = torch.arange(6, dtype=torch.float32).repeat(2, 1)
    obs = build_observation(omega, gravity, command, q, v, last, c)
    assert obs.shape == (2, 25)
    torch.testing.assert_close(obs[:, :3], omega * .5)
    torch.testing.assert_close(obs[:, 3:6], gravity)
    torch.testing.assert_close(obs[:, 6:9], command * torch.tensor([1.,1.,5.]))
    torch.testing.assert_close(obs[:, 9:13], torch.zeros((2,4)), atol=1e-6, rtol=0)
    torch.testing.assert_close(obs[:, 13:19], v * .1)
    torch.testing.assert_close(obs[:, 19:25], last)


def test_knee_observation_never_wraps(c):
    q, v, omega, gravity, cmd = state(c)
    q[:, 1] += 2 * math.pi
    obs = build_observation(omega, gravity, cmd, q, v, torch.zeros_like(q), c)
    torch.testing.assert_close(obs[:, 10], torch.full((2,), 2*math.pi))


def test_bad_observations_fail_not_nan_to_zero(c):
    q, v, omega, g, cmd = state(c)
    q[0, 1] = float('nan')
    with pytest.raises(ValueError): build_observation(omega, g, cmd, q, v, torch.zeros_like(q), c)


def test_critic_is_29_and_privilege_not_actor(c):
    q, v, omega, g, cmd = state(c)
    obs = build_observation(omega,g,cmd,q,v,torch.zeros_like(q),c)
    velocity = torch.ones((2,3)) * 3
    height = torch.ones(2) * .32
    critic = build_critic(obs, velocity, height)
    assert critic.shape == (2,29)
    torch.testing.assert_close(critic[:, :25], obs)
    torch.testing.assert_close(critic[:, 25:28], velocity)
    torch.testing.assert_close(critic[:, -1], height)


def test_history_chronology_repeated_query_subset_reset_and_snapshots():
    history = HistoryStack(2, 'cpu')
    first = torch.stack((torch.ones(25), torch.ones(25)*2))
    saved = history.update(first, 0)
    assert saved.shape == (2,125)
    torch.testing.assert_close(saved.reshape(2,5,25), first[:,None,:].expand(-1,5,-1))
    second = first + 10
    updated = history.update(second, 1)
    torch.testing.assert_close(updated[:, -25:], second)
    torch.testing.assert_close(saved[:, -25:], first)
    repeated = history.update(second + 100, 1)
    torch.testing.assert_close(repeated, updated)
    history.reset(torch.tensor([1]))
    refreshed = history.update(second + 50, 1)
    torch.testing.assert_close(refreshed[0], updated[0])
    torch.testing.assert_close(refreshed[1].reshape(5,25), (second+50)[1].expand(5,-1))
    history.reset([])
    with pytest.raises(ValueError): history.reset(torch.tensor([True,False]))
    with pytest.raises(ValueError): history.reset(torch.tensor([2]))
    with pytest.raises(ValueError): history.update(first, 0)


def test_decode_preserves_hip_winding_and_clamps_knees(c):
    q, *_ = state(c)
    q[:, 0] += 2*math.pi
    q[:, 3] -= 2*math.pi
    actions = torch.tensor([[100.,100.,100.,-100.,-100.,-100.]]).repeat(2,1)
    legs, wheels, clipped = decode_targets(actions, q, c)
    assert legs.shape == (2,4) and wheels.shape == (2,2)
    assert (clipped.abs() <= 1).all()
    assert (legs[:, 0] > 2*math.pi).all() and (legs[:, 2] < -2*math.pi).all()
    for column, name in [(1,'L_joint2'),(3,'R_jonit2')]:
        lo,hi=c['joints']['knee_hard_limits'][name]
        assert (legs[:,column] >= lo+c['joints']['knee_soft_margin']-1e-6).all()
        assert (legs[:,column] <= hi-c['joints']['knee_soft_margin']+1e-6).all()
    torch.testing.assert_close(wheels, torch.tensor([[50.,-50.]]).repeat(2,1))


def test_motor_side_lookup_uses_ratio_on_speed_not_just_torque():
    config={'gear_ratio':10.,'gearbox_efficiency':.8,'curve_side':'motor',
            'motor_speed_rad_s':[0.,100.],'motor_torque_nm':[1.,0.],'effort_limit':100.}
    speeds=torch.tensor([0.,5.,10.,11.,-5.])
    torch.testing.assert_close(motor_torque_limit(speeds,config),torch.tensor([8.,4.,0.,0.,4.]))
    with pytest.raises(ValueError): motor_torque_limit(torch.tensor([float('nan')]),config)


def test_effort_uses_named_semantic_order_and_limits(c):
    q,v,*_=state(c)
    legs,wheels,_=decode_targets(torch.ones_like(q),q,c)
    tau=compute_torques(q,v,legs,wheels,c)
    torch.testing.assert_close(tau[:,[0,1,3,4]],60*(legs-q[:,[0,1,3,4]]))
    bound=c['actuators']['wheel']['effort_limit']
    assert (tau[:,[2,5]].abs() <= bound+1e-6).all()
    v[:,[2,5]]=1e4
    tau=compute_torques(q,v,legs,wheels,c)
    torch.testing.assert_close(tau[:,[2,5]],torch.zeros((2,2)))


def test_rewards_dt_once_upright_direction_and_spin_translation(c):
    q,v,w,g,cmd=state(c)
    kwargs=(torch.zeros((2,3)),w,g,torch.full((2,),.32),cmd,v,v,v,q,c)
    terms=compute_reward_terms(*kwargs)
    torch.testing.assert_close(sum(terms.values()),torch.full((2,),.05))
    tilted=g.clone();tilted[:,0]=.5
    changed=compute_reward_terms(kwargs[0],w,tilted,kwargs[3],cmd,v,v,v,q,c)
    assert (changed['upright'] < terms['upright']).all()
    moving=torch.ones((2,3));cmd[:,1]=12.566
    spin=compute_reward_terms(moving,w,g,kwargs[3],cmd,v,v,v,q,c)
    assert (spin['zero_command_translation'] < 0).all()
    empty=compute_reward_terms(moving[:0],w[:0],g[:0],kwargs[3][:0],cmd[:0],v[:0],v[:0],v[:0],q[:0],c)
    assert all(x.shape==(0,) for x in empty.values())


def synthetic_asset(tmp_path,c):
    """Schema fixture, NOT approval of the real CAD asset."""
    folder=tmp_path/c['asset']['directory'];folder.mkdir(parents=True)
    robot=ET.Element('robot',name='synthetic_schema_fixture')
    names=['base_link','L_link1','L_link2','L_link3','R_link1','R_link2','R_link3']
    masses=[10.8,.326,.45,.2,.326,.45,.2]
    for name,mass in zip(names,masses):
        link=ET.SubElement(robot,'link',name=name)
        inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass',value=str(mass))
    pairs=[]
    for side in ('L','R'):
        parent='base_link'
        for number in (1,2,3):
            name='R_jonit2' if side=='R' and number==2 else f'{side}_joint{number}'
            child=f'{side}_link{number}'
            j=ET.SubElement(robot,'joint',name=name,type='revolute' if number==2 else 'continuous')
            ET.SubElement(j,'parent',link=parent);ET.SubElement(j,'child',link=child)
            if number==2:
                lo,hi=c['joints']['knee_hard_limits'][name];ET.SubElement(j,'limit',lower=str(lo),upper=str(hi))
            pairs.append({'body1':parent,'body2':child,'joint':name,'geometry_review_supported':True,'policy_status':'nominal_proxy_filter_supported'})
            parent=child
    urdf=folder/'robot.urdf';urdf.write_bytes(ET.tostring(robot))
    mjcf=folder/'inspection.xml';mjcf.write_text('<mujoco/>')
    m={'schema_version':1,'robot_id':'own_v40','control_frame':c['control_frame'],'knee_inner_limits_deg':[35,80],
       'total_mass_kg':12.752,'urdf':'robot.urdf','mjcf':'inspection.xml',
       'nominal_joint_pos':dict(zip(c['joints']['action_order'],c['joints']['nominal_positions'])),
       'nominal_base_height_m':.32,'collision_validation':{'passed':True},'adjacent_collision_filter_pairs':pairs,
       'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (urdf,mjcf)}}
    (folder/c['asset']['manifest']).write_text(json.dumps(m))
    return folder,m


def test_asset_gate_and_export_manifest_fixture(tmp_path,c):
    folder,m=synthetic_asset(tmp_path,c)
    result=validate_asset(c,allow_research=True,repo_root=tmp_path)
    manifest=make_run_manifest(c,result)
    assert manifest['contract_sha256']==contract_digest(c)
    assert manifest['actor_obs_dim']==125 and manifest['critic_obs_dim']==29
    with pytest.raises(ValueError,match='research'):validate_asset(c,repo_root=tmp_path)
    m['adjacent_collision_filter_pairs'][0]['geometry_review_supported']=False
    (folder/c['asset']['manifest']).write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Unapproved'):validate_asset(c,True,tmp_path)


def test_corrupt_asset_and_path_escape_rejected(tmp_path,c):
    folder,m=synthetic_asset(tmp_path,c)
    (folder/'robot.urdf').write_text('corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):audit_asset(c,tmp_path)
    c['asset']['directory']='../outside'
    with pytest.raises(ValueError,match='relative'):audit_asset(c,tmp_path)


def test_research_variant_does_not_relabel_raw_geometry(c):
    result=validate_asset(c,allow_research=True)
    assert result['manifest']['collision_validation']['passed'] is True
    assert result['raw_manifest']['collision_validation']['passed'] is False
    assert result['research_approval']['hardware_deployment_approved'] is False
    raw_contract=copy.deepcopy(c)
    raw_contract['asset']['manifest']='manifest.json'
    raw=audit_asset(raw_contract)
    assert raw['manifest']['collision_validation']['passed'] is False
    with pytest.raises(ValueError,match='BLOCKED'):validate_asset(raw_contract,allow_research=True)


def test_research_filter_needs_both_bound_records(c):
    from wheeled_tasks.v40.contract import validate_filter_policy
    result=audit_asset(c)
    with pytest.raises(ValueError,match='approval'):
        validate_filter_policy(result['manifest'])
    approval=copy.deepcopy(result['research_approval'])
    approval['hardware_deployment_approved']=True
    with pytest.raises(ValueError,match='flag'):
        validate_filter_policy(result['manifest'],approval,result['raw_manifest'])
