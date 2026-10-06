"""CPU parsing fixtures only; these do not represent a native trajectory."""
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mechanical_analysis_cpu',HERE/'analysis_candidate_001.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def test_native_world_minus_origin_only():
    assert a.local_points([[11.,22.,33.]], [10.,20.,30.])==[[1.,2.,3.]]
    with pytest.raises(ValueError,match='shape'):a.local_points([[1.,2.]], [1.,2.,3.])


def test_sparse_contacts_preserve_actual_body_indices():
    absent=dict(sensor_index=0,matrix_force_n=[[0.,0.,0.]],net_force_all_contacts_n=[0.,0.,0.],normal_patches=[],friction_patches=[])
    normal=dict(surface_index=0,count=1,point_physx_m=[[10.,21.,32.]],force_scalar_n=[[2.]],normal_physx=[[0.,0.,1.]],separation_m=[[-.001]])
    present=dict(sensor_index=1,matrix_force_n=[[0.,0.,2.]],net_force_all_contacts_n=[0.,0.,2.],normal_patches=[normal],friction_patches=[])
    row=a.contact_readback(dict(sensor_body_rows=[absent,present],existing_view_index=0,capacity_indicator={}),[10.,20.,30.],['base','wheel'])
    result=row['sensor_body_rows_with_observed_force_or_patch'];assert len(result)==1
    assert result[0]['body_name']=='wheel'
    assert result[0]['normal_patches'][0]['point_local_m_derived_native_point_minus_env_origin']==[[0.,1.,2.]]
    assert normal['point_physx_m']==[[10.,21.,32.]]


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_native_value_is_rejected(value):
    with pytest.raises(ValueError,match='Nonfinite'):a.finite_tree({'nested':[value]})


def test_effort_and_reference_use_explicit_full_joint_map():
    layout=dict(full_joint_names=['gas','wheelR','hipL','kneeL','wheelL','hipR','kneeR'],
        control_joint_names=['hipL','kneeL','wheelL','hipR','kneeR','wheelR'],spring_joint_names=['gas'])
    control=dict(script_targets_control6=[1.,2.,3.,4.,5.,6.],native_requested_motor_effort_nm_at_setter=[10.]*6,motor_bounds_nm=[9.]*6,
        full_joint_effort_target=[100.,101.,102.,103.,104.,105.,106.],native_request_scope='preclip',target_scope='setter',
        motor_strength=1.,spring_strength=1.,delay_lags_5ms=None,transport_delay_s=None,transport_age_s=None)
    state=dict(full_q=[.1,1.,2.,3.,4.,5.,6.],full_dq=[.2,10.,20.,30.,40.,50.,60.],applied_full_joint_effort=[200.,201.,202.,203.,204.,205.,206.])
    value=a.effort_readback(control,state,layout)
    assert value['rotary_axes'][2]['actual_postphysics']==40. and value['rotary_axes'][2]['reference_unit']=='rad/s'
    assert value['rotary_axes'][0]['exact_setter_effort_nm']==102. and value['rotary_axes'][0]['engine_applied_effort_nm']==202.
    assert value['rotary_axes'][0]['abs_requested_minus_bound_nm_derived']==1.
    assert value['gas_prismatic_axes'][0]['exact_setter_axial_force_n']==100.


def test_pending_capture_does_not_publish_toy_analysis(tmp_path):
    pending=HERE/'never_captured_fixture_path_001'
    output=HERE/'never_published_fixture_result_001'
    assert not pending.exists() and not output.exists()
    with pytest.raises(FileNotFoundError):a.main(['--capture-root',str(pending),'--output',str(output)])
    assert not output.exists()
