"""Shared CPU fixtures for current runtime tests."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/wheeled_tasks/chassis/env.py'
BANK = json.loads((ROOT / 'reports/v7_training_review_20261011/mechanical_stop_candidates_20261011/stop107_reference_v2/asset/manifest.json').read_text())
JOINTS = BANK['tree_joint_names']

def method(name):
    tree = ast.parse(SOURCE.read_text())
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ChassisEnv'))
    return next((node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))

def mentions(node, expression):
    return expression in ast.unparse(node)

def execute(nodes, env, **values):
    scope = {'self': env, 'torch': torch, '__package__': 'wheeled_tasks.chassis', **values}
    module = ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(SOURCE), 'exec'), scope)
    return scope

def physics_env():
    module = None
    ids = [JOINTS.index(name) for name in BANK['control_joint_names']]
    spring_ids = [JOINTS.index(name) for name in ('L_spring_slide', 'R_spring_slide')]
    data = NS(joint_pos=NS(torch=torch.zeros(2, len(JOINTS))), joint_vel=NS(torch=torch.zeros(2, len(JOINTS))))
    writes, transports = ([], [])
    response = NS(enabled=True, apply_torque=lambda tau: tau + 7.0)

    def transport(tau):
        transports.append(tau.clone())
        return tau + 2.0
    robot = NS(data=data, set_joint_effort_target_index=lambda **kwargs: writes.append(kwargs['target'].clone()), write_data_to_sim=lambda: None, update=lambda dt: None)
    env = NS(_mount_pressure_control_fields=lambda: {}, spring_pressure_element=None, nonstandard_start=module, robot=robot, ids=ids, spring_ids=spring_ids, decimation=4, dt=0.005, signal_delay=None, usb_transport=None, recovery=None, _pc_feedback_transport=None, scut_reward=None, critic78=False, physical_tick=0, is_v5=True, v5=NS(WHEELS=(2, 5), motor_efforts=lambda *args: torch.full((2, 6), 10.0), spring_efforts=lambda q: torch.full_like(q, 27.0), current_motor_bounds=torch.full((2, 6), 40.0)), motor_strength=torch.full((2, 6), 0.8), spring_strength=torch.full((2, 1), 0.9), skills=None, command_transport=NS(apply_torque=transport), actuator_response=response, nominal=torch.zeros(2, len(JOINTS)), actions=torch.full((2, 6), 3.0), jump_assist=None, sim=NS(step=lambda **kwargs: None), control_audit=NS(feedback_control=lambda: None))
    return (env, writes, transports)
