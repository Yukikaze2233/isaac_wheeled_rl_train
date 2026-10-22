"""Pressure command ownership stays separate from motor gates and slider damping."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.simulation.telemetry.numerical_failure_capture import NumericalFailureCapture
from wheeled_tasks.chassis.simulation.spring_force_element import SpringPressureForces
from wheeled_tasks.chassis.simulation.telemetry.torque_monitor import TorqueMonitor


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def environment_methods():
    tree = ast.parse((ROOT / 'src/wheeled_tasks/chassis/application/environment.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ChassisEnv')
    names = {'_apply_mount_spring_pressure', '_mount_pressure_control_fields',
             '_formal_physics_control_frame'}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 'actual_pressure_runtime', 'exec'), scope)
    return scope['ChassisEnv']


def test_physics_submission_reads_current_geometry_and_adds_one_com_wrench(environment_methods):
    calls, reads = [], []
    pose, com = torch.zeros(2, 19, 7), torch.ones(2, 19, 7)
    forces, moments = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    frame = SpringPressureForces(forces, moments, torch.randn(2, 4, 3),
        torch.full((2, 2), .2), torch.full((2, 2), .01), torch.full((2, 2), 300.),
        torch.ones(2, dtype=torch.bool))

    def evaluate(actual_pose, actual_com, strength):
        reads.append((actual_pose.clone(), actual_com.clone(), strength.clone()))
        return frame

    env = environment_methods()
    env.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose),
        body_com_pose_w=SimpleNamespace(torch=com)),
        instantaneous_wrench_composer=SimpleNamespace(add_forces_and_torques_index=lambda **kw: calls.append(kw)))
    env.spring_pressure_element = SimpleNamespace(evaluate=evaluate)
    env.spring_strength = torch.ones(2, 1)
    env._spring_pressure_body_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int32)
    env._spring_pressure_generated_tick = torch.zeros(2, 1, dtype=torch.long)
    env.physical_tick = 39
    env._apply_mount_spring_pressure()
    assert len(calls) == 1 and calls[0]['is_global'] is True and 'positions' not in calls[0]
    torch.testing.assert_close(calls[0]['forces'], forces)
    torch.testing.assert_close(calls[0]['torques'], moments)
    assert env._mount_pressure_control_fields()['pressure_generated_physical_tick'].tolist() == [[39], [39]]
    pose.add_(2.)
    com.add_(3.)
    env.spring_strength.fill_(.9)
    env.physical_tick += 1
    env._apply_mount_spring_pressure()
    assert len(calls) == 2 and len(reads) == 2
    torch.testing.assert_close(reads[1][0], pose)
    torch.testing.assert_close(reads[1][1], com[..., :3])
    torch.testing.assert_close(reads[1][2], env.spring_strength)
    frame.valid[0] = False
    with pytest.raises(FloatingPointError, match='before physics'):
        env._apply_mount_spring_pressure()
    assert len(calls) == 2


def test_pressure_dump_preserves_the_command_that_actually_preceded_the_fault(tmp_path):
    capture = NumericalFailureCapture(tmp_path, 1, {})
    state = {'q': torch.zeros(1, 18), 'pressure_force_n': torch.full((1, 2), 300.),
        'pressure_generated_physical_tick': torch.tensor([[39]])}
    capture.remember_before_physics(39, 9, state)
    state['pressure_force_n'].fill_(900.)
    state['pressure_generated_physical_tick'].fill_(40)
    capture.observe_after_physics(40, 9,
        (torch.zeros(1, 6), torch.zeros(1, 3), torch.full((1, 3), 201.)), lambda: state)
    saved = torch.load(capture.path, weights_only=True)
    assert saved['current']['state']['pressure_force_n'].tolist() == [[300., 300.]]
    assert saved['current']['state']['pressure_generated_physical_tick'].tolist() == [[39]]


def test_pressure_monitor_does_not_report_slider_damping_as_full_pressure():
    monitor = TorqueMonitor(['flat'], ['LH', 'LK', 'LW', 'RH', 'RK', 'RW'],
        'cpu', 4.5, mount_pressure=True)
    zero = torch.zeros(1, 6)
    damping = torch.tensor([[-.02, .01]])
    monitor.observe(zero, zero, damping, torch.tensor([[10., -5.]]), torch.zeros(1, 2),
        zero, torch.ones(1, 6), pressure_force=torch.tensor([[300., 330.]]))
    report = monitor.report()
    assert 'damping only' in report['gas_dof_effort_scope']
    assert report['groups']['flat']['peak_mount_pressure_command_n'] == [300., 330.]
    assert max(report['groups']['flat']['peak_gas_force_n']) < .021
    legacy = TorqueMonitor(['flat'], ['LH', 'LK', 'LW', 'RH', 'RK', 'RW'], 'cpu', 4.5)
    legacy.observe(zero, zero, damping, damping, damping, zero, torch.ones(1, 6))
    assert 'mount_pressure_scope' not in legacy.report()
