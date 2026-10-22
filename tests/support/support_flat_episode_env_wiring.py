"""Shared CPU fixtures for current runtime tests."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import torch
from tests.simulation.runtime.test_flat_reward_runtime import env as reward_env, methods as reward_methods
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/wheeled_tasks/chassis/application/environment.py'
TREE = json.loads((ROOT / 'reports/v7_training_review_20261011/mechanical_stop_candidates_20261011/stop107_reference_v2/asset/manifest.json').read_text())['tree_joint_names']

def method(name):
    cls = next((n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ChassisEnv'))
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))

def execute(nodes, env, **values):
    scope = {'self': env, 'torch': torch, '__package__': 'wheeled_tasks.chassis', **values}
    exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[])), str(SOURCE), 'exec'), scope)
    return scope

def bind_finish(e):
    context = reward_methods('_finish_scut_flat_step', '_flat_heading', '_advance_flat_episode', '_flat_command_iteration')
    context['actor_update_clock'] = lambda env: 0
    for name in ('_flat_heading', '_advance_flat_episode', '_flat_command_iteration'):
        setattr(e, name, MethodType(context[name], e))
    e.finish = MethodType(context['_finish_scut_flat_step'], e)
