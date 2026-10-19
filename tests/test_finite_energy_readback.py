"""Physical identities and discontinuity provenance, without native stepping."""
from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.finite_energy_readback import FiniteEnergyReadback
from wheeled_tasks.chassis.numerical_failure_capture import NumericalFailureCapture


def reader(*, bodies=1, pressure=(), external=False, pushes=False, angular=True):
    return FiniteEnergyReadback(torch.full((1, bodies), 2.),
        torch.eye(3).repeat(1, bodies, 1, 1).reshape(1, bodies, 9), physics_dt_s=.005,
        angular_joints=[angular], active_joint_ids=[0], base_body_id=0, wheel_body_ids=[0],
        pressure_body_ids=pressure, external_wrench_enabled=external,
        velocity_push_column=0 if pushes else None, configured_armature=[.1])


def state(*, bodies=1, vx=0., x=0., z=0., episode=0):
    pose = torch.zeros(1, bodies, 7, dtype=torch.float64)
    pose[..., 0], pose[..., 2], pose[..., 6] = x, z, 1.
    velocity = torch.zeros(1, bodies, 3, dtype=torch.float64)
    velocity[..., 0] = vx
    return dict(q=torch.zeros(1, 1, dtype=torch.float64), dq=torch.zeros(1, 1, dtype=torch.float64),
        episode_id=torch.tensor([episode]), body_link_pose_world=pose.clone(), body_com_pose_world=pose,
        body_com_linear_velocity_world=velocity,
        body_com_angular_velocity_world=torch.zeros(1, bodies, 3, dtype=torch.float64),
        native_applied_effort=torch.zeros(1, 1, dtype=torch.float64),
        all_partner_net_contact_force=torch.zeros(1, bodies, 3, dtype=torch.float64),
        source_interval_event_counts=torch.zeros(1, 2, dtype=torch.long))


def test_free_fall_conserves_com_energy_and_cache_survives_input_mutation():
    book = reader()
    first = state(z=1.)
    first['body_com_linear_velocity_world'][..., 2] = -1.
    snapshot = deepcopy(first)
    book.observe(1, first, {})
    first['body_com_pose_world'].fill_(999.)
    dt, g = .005, 9.81
    second = state(z=1. - dt - .5 * g * dt**2)
    second['body_com_linear_velocity_world'][..., 2] = -1. - g * dt
    frame = book.observe(2, second, {})
    assert frame['metrics']['valid_rigid_increment']
    assert abs(float(frame['metrics']['rigid_energy_increment_j'])) < 1e-12
    assert abs(float(frame['metrics']['explicit_work_residual_j'])) < 1e-12
    torch.testing.assert_close(frame['previous']['body_com_pose_world'], snapshot['body_com_pose_world'])


def test_rotated_anisotropic_link_inertia_and_actual_com_height():
    inertia = torch.diag(torch.tensor([1., 2., 3.])).reshape(1, 1, 9)
    book = FiniteEnergyReadback(torch.tensor([[2.]]), inertia, physics_dt_s=.005,
        angular_joints=[True], active_joint_ids=[0], base_body_id=0, wheel_body_ids=[0])
    sample = state(z=0.)
    sample['body_com_pose_world'][..., 2] = 2.
    sample['body_link_pose_world'][..., 3:] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
    sample['body_com_angular_velocity_world'][..., 1] = 1.
    frame = book.observe(1, sample, {})
    assert float(frame['metrics']['angular_kinetic_j']) == pytest.approx(.5)
    assert float(frame['metrics']['gravity_potential_j']) == pytest.approx(39.24)


def test_native_sized_inertia_validation_without_batched_eigensolver(monkeypatch):
    def unsupported_solver(*args, **kwargs):
        raise AssertionError("Native inertia validation must not require a CUDA eigensolver")

    monkeypatch.setattr(torch.linalg, 'eigvalsh', unsupported_solver)
    mass = torch.ones(4096, 19)
    inertia = torch.tensor([[1., .1, .2], [.1, 2., .3], [.2, .3, 3.]])
    matrices = inertia.expand(4096, 19, 3, 3).reshape(4096, 19, 9).clone()
    kwargs = dict(physics_dt_s=.005, angular_joints=[True], active_joint_ids=[0],
                  base_body_id=0, wheel_body_ids=[1, 2])
    book = FiniteEnergyReadback(mass, matrices, **kwargs)
    assert book.inertia.shape == (4096, 19, 3, 3)
    matrices[-1, -1] = torch.tensor([[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]]).reshape(9)
    with pytest.raises(ValueError, match="finite positive native"):
        FiniteEnergyReadback(mass, matrices, **kwargs)
    matrices[-1, -1] = torch.diag(torch.tensor([1., 1., 0.])).reshape(9)
    with pytest.raises(ValueError, match="finite positive native"):
        FiniteEnergyReadback(mass, matrices, **kwargs)


def test_body_frame_wrench_explains_real_translational_acceleration():
    book = reader(external=True, pushes=True)
    first = state(vx=1.)
    book.observe(1, first, {})
    dt, force, mass = .005, 4., 2.
    last = state(vx=1. + force / mass * dt, x=dt * (1. + .5 * force / mass * dt))
    last['source_interval_event_counts'][0, 1] = 1  # Wrench changes do not exclude the interval.
    control = dict(source_held_force_body_n=torch.tensor([[force, 0., 0.]]),
                   source_held_torque_body_nm=torch.zeros(1, 3))
    frame = book.observe(2, last, control)
    assert frame['metrics']['valid_work_estimate']
    assert not frame['metrics']['source_velocity_push_between_samples']
    assert abs(float(frame['metrics']['explicit_work_residual_j'])) < 1e-12


def test_pressure_pair_work_uses_actual_com_displacements_and_exact_command_stamp():
    book = reader(bodies=2, pressure=(0, 1))
    first = state(bodies=2)
    first['body_com_linear_velocity_world'][0, :, 0] = torch.tensor([-1., 1.])
    book.observe(1, first, {})
    dt, force, mass = .005, 10., 2.
    last = state(bodies=2)
    last['body_com_linear_velocity_world'][0, :, 0] = torch.tensor([-1. - force / mass * dt, 1. + force / mass * dt], dtype=torch.float64)
    dx = dt * (1. + .5 * force / mass * dt)
    last['body_com_pose_world'][0, :, 0] = torch.tensor([-dx, dx], dtype=torch.float64)
    control = dict(pressure_generated_physical_tick=torch.tensor([[1]]),
        pressure_forces_world=torch.tensor([[[-force, 0., 0.], [force, 0., 0.]]]),
        pressure_torques_about_com_world=torch.zeros(1, 2, 3))
    frame = book.observe(2, last, control)
    assert frame['metrics']['valid_work_estimate']
    assert float(frame['metrics']['gas_pressure_work_estimate_j']) == pytest.approx(2 * force * dx)
    assert abs(float(frame['metrics']['explicit_work_residual_j'])) < 1e-12
    wrong_stamp = deepcopy(control)
    wrong_stamp['pressure_generated_physical_tick'].zero_()
    third = book.observe(3, last, wrong_stamp)
    assert third['metrics']['valid_rigid_increment'] and not third['metrics']['valid_work_estimate']
    assert third['metrics']['invalid_pressure_command_stamp']


def test_native_completed_effort_and_angular_wrap_are_not_sent_targets():
    book = reader()
    first = state()
    first['q'].fill_(2 * math.pi - .001)
    first['dq'].fill_(1.)
    book.observe(1, first, {})
    last = state()
    last['q'].fill_(.004)
    last['dq'].fill_(1.)
    last['native_applied_effort'].fill_(2.)
    frame = book.observe(2, last, {'sent_effort': torch.full((1, 1), 100.)})
    assert frame['metrics']['valid_work_estimate']
    assert float(frame['metrics']['native_actuator_work_estimate_j']) == pytest.approx(.01)
    assert int(frame['metrics']['angular_coordinate_wrap_corrections']) == 1


def test_large_finite_speed_keeps_rigid_jump_when_joint_work_is_aliased():
    book = reader()
    book.observe(1, state(), {})
    last = state(vx=5.)
    last['dq'].fill_(700.)
    frame = book.observe(2, last, {})
    assert frame['metrics']['valid_rigid_increment']
    assert frame['metrics']['angular_displacement_aliased']
    assert not frame['metrics']['valid_work_estimate']
    assert float(frame['peak_values'][0]) == 25.


def test_reset_and_intentional_push_do_not_become_unlabelled_physics_faults():
    book = reader(pushes=True)
    book.observe(1, state(), {})
    pushed = state(vx=5.)
    pushed['source_interval_event_counts'][0, 0] = 1
    frame = book.observe(2, pushed, {})
    assert frame['metrics']['source_velocity_push_between_samples']
    assert not frame['metrics']['valid_rigid_increment']
    newborn = state(z=1., episode=1)
    newborn['source_interval_event_counts'].copy_(pushed['source_interval_event_counts'])
    reset = book.observe(3, newborn, {})
    assert reset['metrics']['reset_between_samples'] and not reset['metrics']['valid_rigid_increment']
    report = book.summary()
    assert report['intentional_push_energy_delta_peak_j'] == 25.
    assert report['termination_changed'] is report['ppo_mask_changed'] is False


@pytest.mark.parametrize('contact_error', [False, True])
def test_energy_peak_is_captured_before_gap_or_original_numerical_threshold(tmp_path, contact_error):
    calls = []

    def contacts(*, environment_ids):
        calls.append(environment_ids.tolist())
        if contact_error:
            raise RuntimeError('Contact API fixture unavailable')
        return {'groups': [{'environment_ids': environment_ids.cpu().clone()}]}

    capture = NumericalFailureCapture(tmp_path, 1, {'physics_dt_s': .005},
        energy_readback=reader(), energy_contact_snapshot=contacts)
    first, last = state(), state(vx=5.)
    for tick, sample in ((1, first), (2, last)):
        capture.remember_before_physics(tick - 1, 0, first)
        capture.observe_after_physics(tick, 0, (torch.zeros(1, 6),
            sample['body_com_linear_velocity_world'][:, 0], torch.zeros(1, 3)), lambda: sample,
            closure_gap=torch.tensor([.001]))
    assert not capture.recorded and not capture.summary['closure']['recorded']
    saved = torch.load(capture.energy_paths['explicit_work_residual_j'], weights_only=True)
    assert saved['value'] == 25. and saved['previous_post_physics_tick'] == 1
    assert saved['previous_post_physics_state']['rigid_energy_j'].tolist() == [0.]
    assert saved['energy_metrics']['rigid_energy_j'].tolist() == [25.]
    assert saved['energy_metrics']['wheel_kinetic_energy_j'].tolist() == [[25.]]
    assert saved['termination_changed'] is saved['ppo_mask_changed'] is False
    assert saved['native_contact']['physical_tick'] == 2
    assert calls and all(ids == [0] for ids in calls)
    if contact_error:
        assert saved['native_contact']['status'] == 'error'
    else:
        assert saved['native_contact']['groups'][0]['environment_ids'].tolist() == [0]
    count = len(calls)
    capture.remember_before_physics(2, 0, last)
    capture.observe_after_physics(3, 0, (torch.zeros(1, 6),
        last['body_com_linear_velocity_world'][:, 0], torch.zeros(1, 3)), lambda: last)
    assert len(calls) == count


def test_passive_nonfinite_still_uses_the_existing_abort_and_evidence(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 1, {'physics_dt_s': .005},
        energy_readback=reader(), stop_on_nonfinite=True)
    sample = state()
    capture.remember_before_physics(0, 0, sample)
    capture.observe_after_physics(1, 0, (torch.zeros(1, 6), torch.zeros(1, 3), torch.zeros(1, 3)), lambda: sample)
    sample['dq'].fill_(float('nan'))
    capture.remember_before_physics(1, 0, sample)
    with pytest.raises(FloatingPointError):
        capture.observe_after_physics(2, 0, (torch.zeros(1, 6), torch.zeros(1, 3), torch.zeros(1, 3)),
            lambda: sample, joint_velocity=sample['dq'])
    assert capture.nonfinite_recorded
    saved_path = Path(capture.summary['nonfinite']['file'])
    assert saved_path == capture.path  # The existing first-event file is reused.
    assert torch.load(saved_path, weights_only=True)['current']['state']['dq'].isnan().all()


def test_energy_dump_failure_is_reported_without_a_new_worker_stop(tmp_path, monkeypatch):
    capture = NumericalFailureCapture(tmp_path, 1, {'physics_dt_s': .005}, energy_readback=reader())
    sample = state()
    original = deepcopy(sample)
    def unavailable(*args, **kwargs):
        raise OSError('synthetic diagnostic storage failure')
    monkeypatch.setattr(torch, 'save', unavailable)
    capture.remember_before_physics(0, 0, sample)
    capture.observe_after_physics(1, 0, (torch.zeros(1, 6), torch.zeros(1, 3), torch.zeros(1, 3)), lambda: sample)
    assert not capture.recorded and not capture.summary['worker_abort_raised']
    assert 'dump_error' in capture.summary['finite_energy']['peak_captures']['wheel_contact_force_n']
    for key,value in original.items():
        torch.testing.assert_close(sample[key],value)


def test_healthy_cold_worker_preserves_old_energy_files_and_can_reuse_run_directory(tmp_path):
    first = NumericalFailureCapture(tmp_path, 1, {'physics_dt_s': .005}, energy_readback=reader())
    sample = state()
    first.remember_before_physics(0, 0, sample)
    first.observe_after_physics(1, 0, (torch.zeros(1, 6), torch.zeros(1, 3), torch.zeros(1, 3)), lambda: sample)
    old_file = first.energy_paths['wheel_contact_force_n']
    data = old_file.read_bytes()
    second = NumericalFailureCapture(tmp_path, 1, {'physics_dt_s': .005}, energy_readback=reader())
    assert first.energy_paths.keys() == second.energy_paths.keys()
    assert set(first.energy_paths.values()).isdisjoint(second.energy_paths.values())
    assert old_file.read_bytes() == data and not second.energy.counts.any()
