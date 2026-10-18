"""Dormant solver selection reuses strict learning-state continuation guards."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.training_profile import (
    validate_formal_learning_state_resume, validate_physx_solver_selection,
    validate_profile_selection,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/v7_formal_20261012"


def contract(name="flat_tgs_continuation_055.json"):
    return json.loads((CONFIGS / name).read_text())


def constructor_nodes():
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    env = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    return env, next(node for node in env.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")


def execute(nodes, **scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "actual_environment_solver_wiring", "exec"), scope)
    return scope




@pytest.mark.parametrize("bad", [False, True, 0., 1., -1, 2, "PGS", "TGS", None])
def test_solver_selector_rejects_noninteger_or_unknown_values(bad):
    candidate = contract()
    candidate["physx_solver_type"] = bad
    with pytest.raises(ValueError, match="integer 0"):
        validate_physx_solver_selection(candidate)
    with pytest.raises(ValueError):
        validate_profile_selection(candidate)
    with pytest.raises(ValueError):
        validate_formal_learning_state_resume(contract("flat_tgs_continuation_055.json"), candidate)




@pytest.mark.parametrize("solver", [None, 0, 1])
def test_solver_force_flag_requires_its_exact_boolean_before_native_construction(solver):
    selected = contract("flat_tgs_continuation_055.json")
    if solver is not None:
        selected["physx_solver_type"] = solver
    expected = solver != 0
    selected["physx_external_forces_every_iteration"] = expected
    assert validate_profile_selection(selected)
    for wrong in ((True,) if solver == 0 else ()) + (0, 1, None, "false"):
        invalid = {**selected, "physx_external_forces_every_iteration": wrong}
        with pytest.raises(ValueError):
            validate_profile_selection(invalid)
    if solver != 0:
        assert validate_profile_selection({**selected, "physx_external_forces_every_iteration": False})
    selected.pop("physx_external_forces_every_iteration")
    with pytest.raises(ValueError):
        validate_profile_selection(selected)


















def test_actual_physx_config_consumer_preserves_default_kwargs_and_clock():
    _, init = constructor_nodes()
    start = next(i for i, node in enumerate(init.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "physics_settings" for t in node.targets))
    end = next(i for i in range(start, len(init.body)) if isinstance(init.body[i], ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == "sim" for t in init.body[i].targets))
    observed = []
    fake_sim = NS(SimulationCfg=lambda **kw: kw, SimulationContext=lambda cfg: cfg)
    for selected in (None, 0, 1):
        cfg = {"physx_external_forces_every_iteration": selected != 0,
               "gpu_found_lost_aggregate_pairs_capacity": 1024}
        if selected is not None:
            cfg["physx_solver_type"] = selected
        env = NS(dt=.005, decimation=4)
        execute(init.body[start:end + 1], self=env, config=cfg, device="cpu", sim_utils=fake_sim,
                PhysxCfg=lambda **kwargs: observed.append(kwargs) or kwargs)
        assert env.sim["dt"] == .005 and env.sim["render_interval"] == 4
    baseline = {"enable_external_forces_every_iteration": True,
                "gpu_found_lost_aggregate_pairs_capacity": 1024}
    assert observed == [baseline,
                        {"gpu_found_lost_aggregate_pairs_capacity": 1024, "solver_type": 0},
                        {**baseline, "solver_type": 1}]
    env = NS(dt=.005, decimation=4)
    execute(init.body[start:end + 1], self=env, config={}, device="cpu", sim_utils=fake_sim,
            PhysxCfg=lambda **kwargs: pytest.fail("Unselected legacy kwargs must stay absent"))
    assert env.sim["physics"] is None


def test_actual_scene_readback_checks_named_algorithm_and_retains_iterations():
    _, init = constructor_nodes()
    node = next(node for node in init.body if isinstance(node, ast.If)
                and any(isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "scene_settings"
                    for t in child.targets) for child in node.body))
    for selected, actual in ((None, "TGS"), (0, "PGS"), (1, "TGS")):
        cfg = contract()
        if selected is None:
            cfg.pop("physx_solver_type")
        else:
            cfg["physx_solver_type"] = selected
        cfg["physx_external_forces_every_iteration"] = selected != 0
        attributes = {"physxScene:solverType": actual, "physxScene:timeStepsPerSecond": 200,
                      "physxScene:enableExternalForcesEveryIteration": selected != 0}
        scene = NS(GetAttribute=lambda name: NS(Get=lambda: attributes[name]))
        env = NS(dt=.005, startup_report={})
        execute([node], self=env, config=cfg, scene=scene, formal_solver_roots=4096)
        receipt = env.startup_report["solver_configuration_readback"]
        assert receipt["solverType"] == actual
        assert (receipt["position_iterations"], receipt["velocity_iterations"]) == (12, 6)
        assert ("physx_solver_type" in receipt) == (selected is not None)
        attributes["physxScene:solverType"] = "TGS" if actual == "PGS" else "PGS"
        with pytest.raises(RuntimeError, match="physics scene properties differ"):
            execute([node], self=env, config=cfg, scene=scene, formal_solver_roots=4096)


def test_native_material_snapshot_is_once_and_only_for_explicit_selection():
    cls, _ = constructor_nodes()
    save = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_save_formal_physics_state")
    block = next(node for node in save.body if isinstance(node, ast.If)
                 and "before_first_step" in ast.unparse(node.test))
    from wheeled_tasks.chassis.privileged_runtime import WheelShapeLayout
    material = torch.tensor([[[.4, .3, 0.], [.8, .6, .2], [.8, .6, .2]]])
    calls = []
    def read_materials():
        calls.append(True)
        return material.clone()
    getter = lambda: torch.zeros(1, 2)
    view = NS(**{name: getter for name in ("get_masses", "get_inertias", "get_coms", "get_dof_max_velocities",
                                         "get_dof_stiffnesses", "get_dof_dampings")})
    env = NS(spring_pressure_element=object(), robot=NS(root_view=view, body_names=["base_link", "wheel"]),
             critic_material=NS(native_materials=read_materials,
                layout=WheelShapeLayout((1, 1), ((1, 3), (1, 3)), (1, 2), 3)))
    for config, name, expected in (({}, "before_first_step", 0),
                                   ({"physx_solver_type": 0}, "after_first_step", 0),
                                   ({"physx_solver_type": 0}, "before_first_step", 1),
                                   ({"physx_solver_type": 1}, "before_first_step", 2)):
        env.cfg = config
        values = {}
        execute([block], self=env, values=values, name=name, wp=NS(to_torch=lambda value: value))
        assert len(calls) == expected
        if name == "before_first_step" and config:
            assert torch.equal(values["native_material_properties"], material)
            assert values["native_material_layout"]["shapes_per_body"] == [1, 2]
            assert values["native_material_layout"]["columns"] == ["static_friction", "dynamic_friction", "restitution"]
            assert "native_material_properties" not in values["native_properties"]
        else:
            assert "native_material_properties" not in values
