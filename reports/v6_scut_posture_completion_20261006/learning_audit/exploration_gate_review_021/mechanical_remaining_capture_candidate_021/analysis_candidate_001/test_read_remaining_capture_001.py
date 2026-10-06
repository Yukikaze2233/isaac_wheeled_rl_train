"""Small parser/clock fixtures; all values are synthetic and confer no native credit."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remaining_analysis_CPU',HERE/'read_remaining_capture_001.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def clock_rows():
    return [dict(physical_step=i,policy_tick=None if i==0 else (i-1)//4+1,
        substep=None if i==0 else (i-1)%4+1,simulated_s=i*.005,state={'birth_id':1}) for i in range(9)]


@pytest.mark.parametrize('change',['none','skip','policy','substep','birth','time'])
def test_native_firstbirth_clock_refuses_alias(change):
    rows=clock_rows()
    if change=='skip':rows.pop(3)
    if change=='policy':rows[4]['policy_tick']=2
    if change=='substep':rows[5]['substep']=4
    if change=='birth':rows[3]['state']['birth_id']=2
    if change=='time':rows[1]['simulated_s']=.02
    if change=='none':r.validate_clocks(rows)
    else:
        with pytest.raises(ValueError,match='differs'):r.validate_clocks(rows)


def test_pin_norm_of_captured_endpoints_not_a_contact_inference():
    state={'pin_endpoints_world_m':[[[10.,20.,30.],[10.,20.,30.125]]],'pin_gaps_m':[.125]}
    value=r.pin_norm_readback(state)
    assert value[0]['endpoint_distance_m_derived']==.125 and value[0]['absolute_roundoff_difference_m']==0
    state['pin_gaps_m'][0]=.1
    with pytest.raises(ValueError,match='endpoint norm'):r.pin_norm_readback(state)


def setter_fixture():
    names=['gas','wheelR','hipL','kneeL','wheelL','hipR','kneeR']
    layout={'full_joint_names':names,'control_joint_names':['hipL','kneeL','wheelL','hipR','kneeR','wheelR'],'body_names':['shell','wheel']}
    q=[0.,1.,2.,3.,4.,5.,6.];dq=[0.,11.,12.,13.,14.,15.,16.]
    control={'feedback_full_q_prephysics':q,'feedback_full_dq_prephysics':dq,'script_targets_control6':[1.,2.,3.,4.,5.,6.],
        'native_requested_motor_effort_nm_at_setter':[10.,20.,30.,40.,50.,60.],'motor_bounds_nm':[40.,40.,4.5,40.,40.,4.5],
        'full_joint_effort_target':[100.,1.,2.,3.,4.,5.,6.],'motor_strength':[.9]*6,'spring_strength':[1.],
        'delay_lags_5ms':None,'transport_delay_s':.0001,'transport_age_s':.001,'phase':7,'route':2,
        'side_reference_audit':{'phase':7},'release_finished':True,'blend':0.}
    state={'full_q':q,'full_dq':dq,'applied_full_joint_effort':[200.,21.,22.,23.,24.,25.,26.]}
    return {'applied_control':control,'state':state},layout,{'leg_kp':160.,'leg_kd':2.5,'wheel_velocity_p':.6}


def test_actual_prephysics_axis_command_and_cap_keep_source_clock():
    row,layout,gains=setter_fixture();prior=deepcopy(row)
    prior['applied_control']['script_targets_control6'][0]=.995
    value=r.setter_readback(row,prior,layout,gains);axes=value['axes']
    assert axes[0]['actual_fresh_raw_q_before_physics_rad']==2.
    assert axes[0]['actual_fresh_dq_before_physics_rad_s']==12.
    assert axes[0]['captured_native_preclip_command_nm']==10.
    assert axes[0]['exact_setter_after_strength_transport_response_nm']==2.
    assert axes[0]['engine_applied_effort_after_physics_nm']==22.
    assert axes[2]['actual_native_instantaneous_symmetric_cap_nm']==4.5
    assert axes[0]['reference_rate_derived']==pytest.approx(1.)
    assert not value['continuous_encoder_feedback_winding_directly_saved']
    assert row['applied_control']['script_targets_control6'][0]==1.
    prior['state']['full_q'][2]=99.
    with pytest.raises(ValueError,match='preceding postphysics'):r.setter_readback(row,prior,layout,gains)


def test_net_only_contact_force_is_retained_without_self_pair_identity():
    row,layout,_=setter_fixture()
    row['contacts']={'sensor_body_rows':[{'sensor_index':7,'net_force_all_contacts_n':[1.,2.,3.]},
        {'sensor_index':8,'net_force_all_contacts_n':[0.,0.,0.]}]}
    value=r.net_only_readback(row,layout)
    assert len(value)==1 and value[0]['body_name']=='shell'
    assert value[0]['net_force_all_contacts_without_pair_identity_n']==[1.,2.,3.]


def test_original_reader_private_lane_binding_and_no_pending_publication(tmp_path):
    a=r.original_analysis()
    assert a.LANES==(881,1877,2121) and r.LEVELS==(2,2,1)
    output=HERE/'missing_actual_never_published_001'
    assert not output.exists()
    with pytest.raises(FileNotFoundError):r.main(['--process-receipt',str(tmp_path/'missing_exit'),
        '--native-review',str(tmp_path/'missing_native_review'),'--output',str(output)])
    assert not output.exists()
