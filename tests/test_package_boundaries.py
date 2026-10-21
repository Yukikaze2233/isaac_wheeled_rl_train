"""Import boundaries and public command paths for the modular package."""
import ast
from graphlib import TopologicalSorter
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from wheeled_tasks.chassis.source_layout import MODULES, module_source_path, package_source_files


ROOT = Path(__file__).resolve().parents[1]
CHASSIS = ROOT / 'src/wheeled_tasks/chassis'
CORE_LAYERS = {'control', 'observations', 'tasks'}
NATIVE_PACKAGES = {'isaaclab', 'isaaclab_experimental', 'isaaclab_physx', 'isaacsim', 'omni', 'pxr', 'warp', 'carb'}
CLI_PATHS = (
    'scripts/train_chassis.py', 'scripts/evaluate_flat_policy.py',
    'scripts/export_chassis_candidate.py', 'scripts/replay_v7_ordinary.py',
    'scripts/chassis_remote_job.py', 'tools/v7ctl.py',
)


def imports(path):
    module = '.'.join(path.relative_to(ROOT / 'src').with_suffix('').parts)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ''
            if node.level:
                name = importlib.util.resolve_name('.' * node.level + name, module.rpartition('.')[0])
            yield name


def test_inner_components_do_not_import_application_or_native_sdks():
    violations = []
    core_graph = {}
    for path in CHASSIS.rglob('*.py'):
        layer = path.relative_to(CHASSIS).parts[0]
        module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        if layer in CORE_LAYERS:
            core_graph[module] = set()
        for name in imports(path):
            if layer in CORE_LAYERS:
                core_graph[module].add(name)
            if layer in CORE_LAYERS | {'simulation'} and name.startswith('wheeled_tasks.chassis.application'):
                violations.append((str(path.relative_to(ROOT)), name))
            if layer in CORE_LAYERS and name.split('.')[0] in NATIVE_PACKAGES:
                violations.append((str(path.relative_to(ROOT)), name))
            if name.split('.')[0] in {'scripts', 'tools'}:
                violations.append((str(path.relative_to(ROOT)), name))
    assert violations == []
    dependencies = {name: required & core_graph.keys() for name, required in core_graph.items()}
    tuple(TopologicalSorter(dependencies).static_order())


def test_source_provenance_resolves_existing_modules_and_captures_package_initializers():
    for name in MODULES:
        assert module_source_path(name).is_file(), name
    captured = set(package_source_files())
    assert set(CLI_PATHS) <= captured
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT / 'src').rglob('*.py')} <= captured
    assert all((ROOT / name).is_file() for name in captured)


@pytest.mark.parametrize('relative_path', CLI_PATHS)
def test_public_cli_is_a_thin_application_entry_and_still_accepts_help(relative_path):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    assert len(path.read_text().splitlines()) <= 20
    assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body)
    result = subprocess.run([sys.executable, str(path), '--help'], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


def test_delay_history_can_be_used_without_importing_the_simulator():
    code = '''
import sys, torch
from wheeled_tasks.chassis.control.transport.delay_buffer import GroupedDelayBuffer
history = GroupedDelayBuffer(4, 2, 'cpu', [0, 1])
for sample in [torch.tensor([[1., 10.], [2., 20.]]), torch.tensor([[3., 30.], [4., 40.]])]:
    history.append(sample)
assert torch.equal(history.read_groups(torch.tensor([[1, 0], [0, 1]])), torch.tensor([[1., 30.], [4., 20.]]))
history.reset(torch.tensor([0]))
history.append(torch.tensor([[5., 50.], [6., 60.]]))
assert torch.equal(history.read_groups(torch.tensor([[3, 3], [1, 1]])), torch.tensor([[5., 50.], [4., 40.]]))
assert not any(name.split('.')[0] in {'isaaclab', 'isaaclab_experimental', 'isaaclab_physx', 'omni', 'pxr', 'warp'} for name in sys.modules)
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)
