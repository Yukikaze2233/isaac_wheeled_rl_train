"""Shared CPU fixtures for current runtime tests."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from wheeled_tasks.chassis.task import choose_scene_groups
ROOT = Path(__file__).resolve().parents[1]

def constructor():
    tree = ast.parse((ROOT / 'src/wheeled_tasks/chassis/env.py').read_text())
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ChassisEnv'))
    return next((node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__'))

def allocate(config, count):
    scenes = choose_scene_groups(config['scene_groups'], count)
    owner = NS(rehearsal_spec=None)
    node = next((node for node in constructor().body if isinstance(node, ast.If) and 'load_fixed_flat_rehearsal' in ast.unparse(node)))
    scope = {'self': owner, 'config': config, 'root': ROOT, 'num_envs': count, 'groups': [group for group, _ in scenes], 'evaluation_setup': None, 'replay_setup': None, '__package__': 'wheeled_tasks.chassis'}
    exec(compile(ast.Module(body=[deepcopy(node)], type_ignores=[]), 'actual_plane_allocation', 'exec'), scope)
    return (owner.rehearsal_spec, scenes)
