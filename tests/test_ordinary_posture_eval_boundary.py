"""Portable regression for the ordinary-posture constructor boundary.

The source root is discovered from this file's ancestors, or selected with SCUT_REVIEW_SOURCE_ROOT for an isolated
checkout. Actual pure source modules use a private package namespace, so no
native env/warp module is imported and a pre-imported workspace package cannot
silently replace the selected checkout. The actual env constructor AST branch
executes unchanged: there is no embedded repair or source-hash pin in this test.

Enabled-training geometry binding is explicitly mocked; these tests validate
dispatch, catalog strictness, settings, mask and RNG only.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


PACKAGE = "_ordinary_boundary_repository_regression"
CONTRACT_PATH = "contracts/v6_task_campaign_recovery_worker_v18.json"
ENV_PATH = "src/wheeled_tasks/chassis/env.py"


def source_root():
    explicit = os.environ.get("SCUT_REVIEW_SOURCE_ROOT")
    candidates = [Path(explicit)] if explicit else Path(__file__).resolve().parents
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / CONTRACT_PATH).is_file() and (candidate / ENV_PATH).is_file():
            return candidate
    raise RuntimeError("Cannot find source root; set SCUT_REVIEW_SOURCE_ROOT")


@pytest.fixture(scope="module")
def source():
    root = source_root()
    before_imports = set(sys.modules)
    for name, directory in ((PACKAGE, root / "src/wheeled_tasks"),
                            (PACKAGE + ".chassis", root / "src/wheeled_tasks/chassis")):
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    posture = importlib.import_module(PACKAGE + ".chassis.ordinary_posture_curriculum")
    evaluation = importlib.import_module(PACKAGE + ".chassis.evaluation")
    task = importlib.import_module(PACKAGE + ".chassis.task")
    training = json.loads((root / CONTRACT_PATH).read_text())
    fixed = evaluation.fixed_suite_contract(training)
    assert "rotate_1_reverse" in {row["name"] for row in fixed["evaluation"]["cases"]}
    assert "rotate_1_reverse" not in fixed["skill_specs"]
    imported = set(sys.modules) - before_imports
    assert not any(name == "warp" or name.startswith("warp.")
                   or name.startswith("isaaclab") for name in imported)
    assert PACKAGE + ".chassis.env" not in sys.modules
    tree = ast.parse((root / ENV_PATH).read_text(), filename=str(root / ENV_PATH))
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Constant)
        and node.test.left.value == "ordinary_posture_curriculum"
        and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.In)
        and len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Name)
        and node.test.comparators[0].id == "config"
        and any(isinstance(statement, ast.ImportFrom)
                and statement.module == "ordinary_posture_curriculum"
                and any(alias.name == "OrdinaryPostureCurriculum" for alias in statement.names)
                for statement in node.body)]
    assert len(branches) == 1, "Expected the actual ordinary-posture constructor branch"
    block = compile(ast.fix_missing_locations(ast.Module(body=branches, type_ignores=[])),
                    str(root / ENV_PATH) + "::ordinary_constructor_only", "exec")
    return SimpleNamespace(root=root, posture=posture, task=task,
                           training=training, fixed=fixed, block=block)


def run_constructor(source, config, scenes):
    state = SimpleNamespace(scene_groups=[group for group, _ in scenes],
        kinds=[kind for _, kind in scenes], recovery=None, skills=None,
        nonstandard_start=None, ordinary_posture=None,
        robot=SimpleNamespace(joint_names=[]),
        generator=torch.Generator(device="cpu").manual_seed(481516))
    namespace = dict(__name__=PACKAGE + ".chassis.constructor_fixture",
        __package__=PACKAGE + ".chassis", config=config, self=state,
        num_envs=len(scenes), device=torch.device("cpu"), torch=torch,
        root=source.root, manifest={})
    lane_rng_before = state.generator.get_state().clone()
    default_rng_before = torch.get_rng_state().clone()
    try:
        exec(source.block, namespace)
    finally:
        assert torch.equal(state.generator.get_state(), lane_rng_before)
        assert torch.equal(torch.get_rng_state(), default_rng_before)
    return state.ordinary_posture


def watch_actual_helper(monkeypatch, source):
    calls = []
    actual = source.posture.ordinary_eligibility
    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return actual(*args, **kwargs)
    monkeypatch.setattr(source.posture, "ordinary_eligibility", counted)
    return calls


def forbid_geometry(*args, **kwargs):
    raise AssertionError("Disabled/evaluation boundary must not bind geometry")


@pytest.mark.parametrize("scenario", ["canonical_fixed", "enabled_but_exact_eval", "disabled_train_unknown"])
def test_disabled_and_exact_eval_preserve_real_object_without_catalog_access(
        source, monkeypatch, scenario):
    config = deepcopy(source.training if scenario == "disabled_train_unknown" else source.fixed)
    if scenario == "disabled_train_unknown":
        config["ordinary_posture_curriculum"]["enabled"] = False
        config.pop("evaluation_exact_cases", None)
        scenes = [("unknown_training_skill", "flat")]
    else:
        assert config["evaluation_exact_cases"] is True
        if scenario == "enabled_but_exact_eval":
            config["ordinary_posture_curriculum"]["enabled"] = True
        scenes = source.task.choose_scene_groups(config["scene_groups"], len(config["scene_groups"]))
    before = deepcopy(config)
    calls = watch_actual_helper(monkeypatch, source)
    monkeypatch.setattr(source.posture.OrdinaryPostureCurriculum, "bind_geometry", forbid_geometry)
    posture = run_constructor(source, config, scenes)
    assert calls == [] and config == before
    assert isinstance(posture, source.posture.OrdinaryPostureCurriculum)
    assert posture.enabled is False and posture.geometry_bound is False
    assert posture.eligible.shape == (len(scenes),) and posture.eligible.dtype == torch.bool
    assert posture.eligible.device == torch.device("cpu")
    assert not bool(posture.eligible.any()) and not bool(posture.selected.any())
    assert posture.birth_population == posture.birth_selected_count == posture.push_events == 0
    assert posture.runtime_state() == source.posture.runtime_expectation(config["ordinary_posture_curriculum"], enabled=False)
    assert posture.runtime_state()["settings"] == config["ordinary_posture_curriculum"]


def test_enabled_training_keeps_strict_unknown_catalog_failure(source, monkeypatch):
    assert source.training["ordinary_posture_curriculum"]["enabled"] is True
    calls = watch_actual_helper(monkeypatch, source)
    monkeypatch.setattr(source.posture.OrdinaryPostureCurriculum, "bind_geometry", forbid_geometry)
    with pytest.raises(KeyError, match="unknown_training_skill"):
        run_constructor(source, deepcopy(source.training), [("unknown_training_skill", "flat")])
    assert len(calls) == 1


def test_recovery_training_4096_actual_allocation_keeps_1449_eligible(source, monkeypatch):
    scenes = source.task.choose_scene_groups(source.training["scene_groups"], 4096)
    expected = source.posture.ordinary_eligibility(source.training,
        [group for group, _ in scenes], [kind for _, kind in scenes], torch.device("cpu"))
    assert int(expected.sum()) == 1449
    calls = watch_actual_helper(monkeypatch, source)
    binds = []
    monkeypatch.setattr(source.posture.OrdinaryPostureCurriculum, "bind_geometry",
                        lambda instance, *args: binds.append((instance, args)))
    posture = run_constructor(source, deepcopy(source.training), scenes)
    assert posture.enabled is True and torch.equal(posture.eligible, expected)
    assert posture.birth_population == 1449 and len(calls) == len(binds) == 1
    assert posture.geometry_bound is False  # Explicit mock, no geometry evidence.


def test_disabled_settings_are_still_schema_validated(source, monkeypatch):
    broken = deepcopy(source.fixed)
    broken["ordinary_posture_curriculum"]["birth"]["probability"] = .123
    calls = watch_actual_helper(monkeypatch, source)
    with pytest.raises(ValueError, match="(?i)(settings|frozen)"):
        run_constructor(source, broken, [("rotate_1_reverse", "flat")])
    assert calls == []
