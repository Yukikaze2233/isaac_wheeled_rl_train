"""Run seven legacy cases with complete archive/AST fixtures, without weakening assertions.

Original failures are retained separately. These cases supply missing fixture
dependencies only; they do not edit the production blocks or initialize Isaac.
"""
import ast
import builtins
import importlib.util
from pathlib import Path

import pytest
import torch

import test_chassis_preflight as legacy_substep
import test_recovery_resume_pose_clock as legacy_sources
import test_resume_learning_rate_entry as legacy_launch


ROOT = Path(__file__).resolve().parents[1]


def train_entry():
    spec = importlib.util.spec_from_file_location('entropy_entry_complete_fixture', ROOT/'scripts/train_chassis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def complete_legacy_archive_context(request, monkeypatch):
    """Optional pytest plugin: run the seven original cases without deselection.

    Load this module with -p test_entropy_entry_fixture_context for the broad
    archive regression. Ordinary collection still preserves the raw failures.
    """
    name = getattr(request.node, 'originalname', request.node.name)
    if request.module is legacy_sources and name == 'test_active_curriculum_and_revision_have_checkpoint_source_pins':
        entry = train_entry()

        def complete_sources(code, namespace):
            namespace['scut_curriculum_source_files'] = entry.scut_curriculum_source_files
            return builtins.exec(code, namespace)

        monkeypatch.setattr(legacy_sources, 'exec', complete_sources, raising=False)
    elif request.module is legacy_substep and name == 'test_actual_environment_substep_loop_recomputes_pd_from_updated_feedback':
        tree = ast.parse((ROOT/'src/wheeled_tasks/chassis/root_kinematics.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'update_robot_data')
        namespace = {}
        builtins.exec(compile(ast.Module(body=[function], type_ignores=[]), 'native_robot_update_helper', 'exec'), namespace)

        def complete_substep(code, scope):
            env = scope['self']
            env.signal_delay = env.nonstandard_start = env.wheel_force_peak_n = None
            scope['update_robot_data'] = namespace['update_robot_data']
            return builtins.exec(code, scope)

        monkeypatch.setattr(legacy_substep, 'exec', complete_substep, raising=False)
    elif request.module is legacy_launch and name in (
            'test_launcher_to_full_stage_preserves_explicit_scale_and_consumption',
            'test_launcher_rejects_resume_scale_for_direct_foundation_worker'):
        original_entry = legacy_launch.entry

        def archive_git(command, **kwargs):
            assert command[:1] == ['git']
            if command[1:] == ['rev-parse', 'HEAD^{commit}']:
                raw = b'723732adae37f12f8488f9537717664415a98d10\n'
            elif command[1] == 'show':
                path = (ROOT/command[2].split(':', 1)[1]).resolve()
                assert path.is_relative_to(ROOT) and path.is_file()
                raw = path.read_bytes()
            else:
                raise AssertionError('Unexpected subprocess in archive-only fixture: ' + repr(command))
            return raw.decode() if kwargs.get('text') else raw

        def archive_entry(script):
            module = original_entry(script)
            assert script == 'launch_v5_remote'
            monkeypatch.setattr(module.subprocess, 'check_output', archive_git)
            return module

        monkeypatch.setattr(legacy_launch, 'entry', archive_entry)


@pytest.mark.parametrize('arguments', [
    (False, True, 'task-worker-revision-v8', True, True),
    (True, True, 'task-worker-revision-v8', False, True),
    (False, False, 'task-worker-revision-v8', False, True),
    (False, True, 'task-worker-revision-v6', True, False),
])
def test_original_source_pin_assertions_with_production_helper_in_ast_namespace(monkeypatch, arguments):
    entry = train_entry()

    def complete_namespace(code, namespace):
        namespace['scut_curriculum_source_files'] = entry.scut_curriculum_source_files
        return builtins.exec(code, namespace)

    monkeypatch.setattr(legacy_sources, 'exec', complete_namespace, raising=False)
    legacy_sources.test_active_curriculum_and_revision_have_checkpoint_source_pins(*arguments)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('case', ['forward_scale', 'reject_direct_worker'])
def test_original_plan_only_launch_assertions_from_immutable_archive_fixture(monkeypatch, tmp_path, capsys, case):
    # This tests argv/ledger behavior, not Git authentication. The isolation
    # archive has no .git. Fail immediately if a test attempts real execution.
    original_entry = legacy_launch.entry

    def archive_git(command, **kwargs):
        assert command[:1] == ['git']
        if command[1:] == ['rev-parse', 'HEAD^{commit}']:
            value = '723732adae37f12f8488f9537717664415a98d10\n'
            return value if kwargs.get('text') else value.encode()
        if command[1] == 'show':
            name = command[2].split(':', 1)[1]
            path = (ROOT/name).resolve()
            assert path.is_relative_to(ROOT) and path.is_file()
            raw = path.read_bytes()
            return raw.decode() if kwargs.get('text') else raw
        raise AssertionError('Unexpected subprocess in plan-only archive fixture: ' + repr(command))

    def archive_entry(name):
        module = original_entry(name)
        assert name == 'launch_v5_remote'
        monkeypatch.setattr(module.subprocess, 'check_output', archive_git)
        return module

    monkeypatch.setattr(legacy_launch, 'entry', archive_entry)
    if case == 'forward_scale':
        legacy_launch.test_launcher_to_full_stage_preserves_explicit_scale_and_consumption(tmp_path, monkeypatch)
    else:
        legacy_launch.test_launcher_rejects_resume_scale_for_direct_foundation_worker(tmp_path, monkeypatch, capsys)
    assert not torch.cuda.is_initialized()


def test_original_feedback_pd_assertions_with_current_native_helper_and_optional_modules(monkeypatch):
    # Extract the real CPU-safe helper without importing Warp/Isaac. The loop
    # and every original position/torque/timing assertion remain unchanged.
    tree = ast.parse((ROOT/'src/wheeled_tasks/chassis/root_kinematics.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'update_robot_data')
    namespace = {}
    builtins.exec(compile(ast.Module(body=[function], type_ignores=[]), 'native_robot_update_helper', 'exec'), namespace)

    def complete_namespace(code, scope):
        env = scope['self']
        env.signal_delay = None
        env.nonstandard_start = None
        env.wheel_force_peak_n = None
        scope['update_robot_data'] = namespace['update_robot_data']
        return builtins.exec(code, scope)

    monkeypatch.setattr(legacy_substep, 'exec', complete_namespace, raising=False)
    legacy_substep.test_actual_environment_substep_loop_recomputes_pd_from_updated_feedback()
    assert not torch.cuda.is_initialized()
