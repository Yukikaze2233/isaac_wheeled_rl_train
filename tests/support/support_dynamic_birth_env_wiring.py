"""Shared CPU fixtures for current runtime tests."""
import ast
from copy import deepcopy
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/wheeled_tasks/chassis/application/environment.py'

def methods(*names):
    tree = ast.parse(SOURCE.read_text())
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ChassisEnv'))
    nodes = [deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {'torch': torch, '__package__': 'wheeled_tasks.chassis'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), 'exec'), scope)
    return scope
