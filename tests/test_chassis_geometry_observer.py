"""Physical frame, mirror, drift and first-episode denominator checks."""
import math
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from chassis_geometry_observer import SpinGeometryObserver, paired_geometry, validate_report
import run_rec_speed_mean_replay as driver


def geometry(angle=0., translation=(0., 0., 0.)):
    root = torch.tensor([[*translation, 0., 0., math.sin(angle / 2), math.cos(angle / 2)]], dtype=torch.float64)
    hips = torch.tensor([[[0., .184, -.001], [0., -.183, .0008]]], dtype=torch.float64)
    spans = torch.tensor([[[.08, .03, -.30], [.08, -.03, -.30]]], dtype=torch.float64)
    wheels = hips + spans
    pairs = torch.stack((hips, wheels), dim=1)
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                             [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64)
    shift = root[:, None, :3]
    hips_world, wheels_world = hips @ rotation.T + shift, wheels @ rotation.T + shift
    pairs_world = pairs @ rotation.T + root[:, None, None, :3]
    return root, wheels_world, hips_world, pairs_world


@pytest.mark.parametrize('angle,translation', ((0., (0., 0., 0.)), (.8, (5., -2., 1.)), (-2.8, (-8., 3., -.2))))
def test_world_transform_and_unequal_hip_mounts_preserve_span_mirror(angle, translation):
    result = paired_geometry(*geometry(angle, translation))
    assert result['projected_leg_length_m'][0].tolist() == pytest.approx([math.hypot(.08, .30)] * 2)
    assert result['projected_leg_length_delta_m'].item() == pytest.approx(0., abs=1e-14)
    assert result['projected_leg_angle_delta_rad'].item() == pytest.approx(0., abs=1e-14)
    assert result['paired_link_hip_relative_error_m'].max().item() == pytest.approx(0., abs=1e-14)
    # The real mount offset is visible, not silently corrected to a zero wheel-height difference.
    assert result['wheel_vertical_delta_m'].item() == pytest.approx(-.0018)


def test_length_and_angle_asymmetry_are_independently_detected():
    values = list(geometry())
    values[1] = values[1].clone()
    values[1][0, 1, 2] -= .02
    result = paired_geometry(*values)
    assert abs(result['projected_leg_length_delta_m'].item()) > .019
    assert abs(result['projected_leg_angle_delta_rad'].item()) > .01
    assert result['wheel_vertical_delta_m'].item() == pytest.approx(.0182)


def test_leg_angle_difference_wraps_at_pi():
    root, wheels, hips, links = geometry()
    wheels[0, :, 0] = torch.tensor([1e-8, -1e-8])
    wheels[0, :, 2] = hips[0, :, 2] + .3
    result = paired_geometry(root, wheels, hips, links)
    assert abs(result['projected_leg_angle_delta_rad'].item()) < 1e-6


def observation():
    result = paired_geometry(*geometry())
    result['wheel_contact_force_n'] = torch.tensor([[20., 20.]])
    data = dict(position=torch.tensor([[1., 0., .3]]), planar_speed_world=torch.tensor([1.]),
                omega=torch.tensor([[0., 0., 6.]]))
    return result, data


def observe(observer, *, time=1., active=True, policy=True, requested=None, applied=None):
    result, data = observation()
    observer.observe(time, data, torch.tensor([active]), torch.tensor([policy]),
        requested or [[0., math.tau, .305]], torch.tensor(applied or [[0., math.tau, .305]]),
        result, torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[1., 0.]]))


def test_geometric_symmetry_cannot_hide_translation():
    observer = SpinGeometryObserver(['stand_305mm'])
    observe(observer)
    report = observer.report()
    row = report['replicas'][0]
    assert row['leg_length_mae_m'] == pytest.approx(0.)
    assert row['leg_angle_mae_rad'] == pytest.approx(0.)
    assert row['planar_speed_mean_m_s'] == 1.
    assert row['root_displacement_max_m'] == 1.
    assert row['wheel_midpoint_displacement_max_m'] == 1.
    assert report['policy_trace'][0]['root_displacement_m'] == 1.
    validate_report(json.loads(json.dumps(report)), 1)


@pytest.mark.parametrize('changes', ({'active': False}, {'policy': False},
    {'requested': [[0., 0., .305]]}, {'requested': [[.5, math.tau, .305]]}))
def test_inactive_scripted_quiet_or_translating_frames_do_not_enter_spin_denominator(changes):
    observer = SpinGeometryObserver(['stand_305mm'])
    observe(observer, **changes)
    row = observer.report()['replicas'][0]
    assert row['policy_frames'] == 0 and row['planar_speed_mean_m_s'] is None
    assert observer.report()['policy_trace'] == []
    validate_report(json.loads(json.dumps(observer.report())), 1)


def test_yaw_slew_is_measured_separately_from_requested_target():
    observer = SpinGeometryObserver(['stand_305mm'])
    observe(observer, applied=[[0., 3., .305]])
    observe(observer, time=1.02)
    row = observer.report()['replicas'][0]
    assert row['policy_frames'] == 2 and row['applied_target_frames'] == 1
    assert row['requested_yaw_mae_rad_s'] == pytest.approx(math.tau - 6.)
    assert row['applied_yaw_mae_rad_s'] == pytest.approx((3. + math.tau - 6.) / 2.)


@pytest.mark.parametrize('change', ('denominator', 'mean', 'peak', 'command', 'time', 'missing', 'nonfinite'))
def test_geometry_summary_rejects_tampered_native_evidence(change):
    observer = SpinGeometryObserver(['stand_305mm'])
    observe(observer)
    observe(observer, time=1.02)
    report = json.loads(json.dumps(observer.report()))
    if change == 'denominator':
        report['replicas'][0]['policy_frames'] += 1
    elif change == 'mean':
        report['replicas'][0]['leg_length_mae_m'] = .1
    elif change == 'peak':
        report['replicas'][0]['root_displacement_max_m'] = 0.
    elif change == 'command':
        report['policy_trace'][0]['requested'][0] = .1
    elif change == 'time':
        report['policy_trace'][1]['time_s'] = report['policy_trace'][0]['time_s']
    elif change == 'missing':
        del report['policy_trace'][0]['yaw_actual_rad_s']
    else:
        report['policy_trace'][0]['yaw_actual_rad_s'] = float('nan')
    with pytest.raises(ValueError):
        validate_report(report, 1)


def test_snapshot_uses_native_center_and_force_without_writing_native_state():
    root, wheels, hips, links = geometry()
    names = [name for pair in __import__('chassis_geometry_observer').LINK_PAIRS for name in pair]
    positions = hips.repeat(1, 9, 1)
    poses = torch.cat((positions, torch.zeros(1, 18, 4, dtype=torch.float64)), dim=-1)
    state = poses.clone()
    env = SimpleNamespace(robot=SimpleNamespace(body_names=names, data=SimpleNamespace(
        root_link_pose_w=SimpleNamespace(torch=root), body_link_pose_w=SimpleNamespace(torch=poses))),
        wheel_centers=lambda: wheels, wheel_ids=[0, 1], contact_force=torch.tensor([[[3., 4., 0.], [0., 0., 6.]]]))
    result = SpinGeometryObserver(['stand_305mm']).snapshot(env)
    assert result['wheel_contact_force_n'].tolist() == [[5., 6.]]
    assert result['paired_link_hip_relative_delta_m'].shape == (1, 9, 3)
    assert torch.equal(state, poses)


@pytest.mark.parametrize('protocol,sign', (('spin_positive', 1), ('spin_negative', -1)))
def test_spin_is_explicit_only_with_fixed_release_and_original_births(protocol, sign):
    assert protocol not in driver.PROTOCOLS and protocol in driver.EXPLICIT_PROTOCOLS
    assert driver.request(protocol, 'stand_305mm', .98) == [0., 0., .305]
    assert driver.request(protocol, 'stand_305mm', 1.) == [0., sign * math.tau, .305]
    assert driver.request(protocol, 'stand_305mm', 9.98) == [0., sign * math.tau, .305]
    assert driver.request(protocol, 'stand_305mm', 10.) == [0., 0., .305]
    for scene in driver.prep.FALLEN:
        assert driver.request(protocol, scene, 4.) == [0., 0., .305]
    golden = Path(__file__).resolve().parents[1].parent / 'serial-dispatch-source028-20261009'
    if not golden.is_dir():
        pytest.skip('Frozen runtime unavailable for birth authentication')
    source = driver.prep.read(golden / driver.prep.REC)
    original = driver.prep.selected_cases(source, 'nominal')
    spun = driver.prep.selected_cases(source, protocol)
    assert spun[:1] + spun[2:] == original[:1] + original[2:]
    assert spun[1]['episode_seconds'] == 22. and spun[1]['task'] == 'survive'
