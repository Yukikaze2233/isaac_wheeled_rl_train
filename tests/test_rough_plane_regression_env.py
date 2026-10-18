"""CPU physical-adapter checks; no Kit, GPU or native qualification."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from support_flat_episode_env_wiring import execute, method
from test_formal_shared_plane import env_for_guard, run_pre_reset_guards
from wheeled_tasks.chassis.flat_command_sampling import ScutFlatCommands
from wheeled_tasks.chassis.flat_episode_runtime import build_flat_episode_runtime
from wheeled_tasks.chassis.flat_evaluation_setup import FlatEvaluationSetup
from wheeled_tasks.chassis.formal_terrain import SharedPlane, plane_origins


ROOT = Path(__file__).resolve().parents[1]


def inputs():
    directory = ROOT / "configs/v7_formal_20261012"
    config = json.loads((directory / "mixed_tracking_repair_063.json").read_text())
    protocol = json.loads((directory / "evaluation/foundation_fixed_mean_v1.json").read_text())
    lanes = json.loads((directory / "evaluation/foundation_fixed_lanes_v1.json").read_text())
    setup = FlatEvaluationSetup(protocol, tuple(lanes["lane_case_ids"]), 300, 1800,
        run_role="source_rough_plane_regression", evaluation_surface="plane_regression_v1")
    return config, setup


def plane(n=2):
    return SharedPlane(plane_origins(n, 4.), ["/World/ground/Plane"],
                       ["/World/ground"], ["/World/ground/physicsMaterial"], 4.)


def test_real_plane_ray_intersections_and_actual_rough_reward_scan_consumer():
    ground = plane()
    starts = torch.tensor([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., float("nan")]])
    directions = torch.tensor([[1., 0., -1.], [0., 0., 1.], [1., 0., 0.], [0., 0., -1.]])
    hits = ground.raycast(starts, directions)[0]
    torch.testing.assert_close(hits[0], torch.tensor([4., 2., 0.]))
    assert torch.isinf(hits[1:]).all()
    assert not hasattr(ground, "importer") and not hasattr(ground, "warp_mesh")
    env = env_for_guard()
    env.shared_terrain, env.evaluation_surface = ground, "plane_regression_v1"
    env.cfg.update(reward_profile="scut0730_rough_v1", formal_terrain={"type": "rough", "source": "scut0730"})
    env.origins[:, 2] = 1.4  # Deliberately distinguish actual plane scan from origin fallback.
    env.robot.data.root_link_pose_w.torch[:, 2] = 1.8
    env.scut_reward.reward_height = lambda pose, origins, scan: pose[:, 2] - scan
    before = deepcopy(env.cfg)
    result = run_pre_reset_guards(env)
    assert result["rough_scan_hit_count"].eq(9).all()
    assert result["reward_ground_height_world"].eq(0).all()
    torch.testing.assert_close(result["height"], torch.full((2,), 1.8))
    assert not result["reasons"]["boundary"].any() and env.cfg == before


def test_typed_gate_selects_actual_plane_without_rewriting_rough_contract(monkeypatch):
    config, setup = inputs()
    before = deepcopy(config)
    init = method("__init__").body
    start = next(i for i, node in enumerate(init)
                 if ast.unparse(node) == "self.evaluation_surface = 'training'")
    env = NS(rehearsal_spec=None)
    kwargs = dict(config=config, evaluation_setup=setup, replay_setup=None, num_envs=480,
                  seed=setup.protocol["environment_seed"], run_role=setup.run_role)
    execute(init[start:next(i for i, n in enumerate(init) if ast.unparse(n) == "self.rehearsal_spec = None")], env, **kwargs)
    assert env.evaluation_surface == "plane_regression_v1"
    with pytest.raises(ValueError, match="typed"):
        execute(init[start:next(i for i, n in enumerate(init) if ast.unparse(n) == "self.rehearsal_spec = None")], NS(), **{**kwargs, "evaluation_setup": None})
    with pytest.raises(TypeError, match="typed setup"):
        execute(init[start:next(i for i, n in enumerate(init) if ast.unparse(n) == "self.rehearsal_spec = None")], NS(), **{**kwargs, "evaluation_setup": NS(evaluation_surface="plane_regression_v1")})
    created = []
    def create(stage, settings, *args, **kwargs):
        created.append(deepcopy(settings))
        return plane(480)
    monkeypatch.setattr("wheeled_tasks.chassis.formal_terrain.create_shared_terrain", create)
    node = next(node for node in init if isinstance(node, ast.If)
                and ast.unparse(node.test) == "config.get('formal_terrain') is not None")
    env.sim = NS(stage=object())
    context = dict(config=config, num_envs=480, device="cpu", seed=17,
        sim_utils=NS(RigidBodyMaterialCfg=lambda **kw: kw), combine="multiply",
        material_policy=NS(restitution_combine="multiply"), authored_material_paths=[])
    execute([node], env, **context)
    assert created == [{"type": "plane", "source": "scut0719", "env_spacing_m": 4.}]
    assert config == before
    # Execute the real constructor's birth-support selector: a plane has no mesh.
    birth_node = next(node for node in ast.walk(method("__init__")) if isinstance(node, ast.If)
        and any(isinstance(child, ast.ImportFrom) and child.module == "scut_rough_support"
                for child in node.body))
    env.rough_birth_support = None
    execute([birth_node], env, config=config)
    assert env.rough_birth_support is None
    env.evaluation_surface = "training"
    execute([node], env, **context)
    assert created[-1] == config["formal_terrain"]  # Default rough path unchanged.


def test_bound_mechanical_factory_uses_fixed_flat_owner_and_no_world_terrain_advance():
    config, setup = inputs()
    before = deepcopy(config)
    asset = ROOT / config["asset_directory"]
    manifest, spec = (json.loads((asset / name).read_text()) for name in ("manifest.json", "model_spec.json"))
    runtime = build_flat_episode_runtime(config, manifest, spec, ROOT, 480, "cpu", evaluation_setup=setup)
    assert type(runtime.commands) is ScutFlatCommands
    env = NS(cfg={**config, "seed": setup.protocol["environment_seed"]}, num_envs=480, physical_tick=0,
        flat_episode=runtime, episode_length_buf=torch.zeros(480, dtype=torch.long),
        generator=torch.Generator().manual_seed(setup.protocol["environment_seed"]), startup_report={},
        evaluation_surface="plane_regression_v1", _flat_heading=lambda: torch.zeros(480),
        policy_dt=.02, commands=torch.zeros(480, 3), mode=torch.zeros(480, dtype=torch.long),
        targets=torch.zeros(480, 2), jump_requested=torch.zeros(480, dtype=torch.bool), raw_height=torch.zeros(480))
    setup.apply_before_first_reset(env)
    runtime.commands.reset(torch.arange(480), torch.zeros(480), torch.zeros(480, dtype=torch.bool), env.generator)
    for _ in range(4):
        runtime.advance_physics(.005)
    env._flat_command_iteration = lambda: 0
    execute([method("_advance_flat_episode")], env)["_advance_flat_episode"](
        env, torch.zeros(480, dtype=torch.bool), torch.zeros(480, dtype=torch.bool), {})
    assert runtime.commands.fixed_request_mask.all()
    assert not hasattr(runtime.commands, "_rough")
    assert config == before
    # A raw flag/object cannot request the alternate command owner.
    with pytest.raises(ValueError, match="typed rough regression"):
        build_flat_episode_runtime(config, manifest, spec, ROOT, 480, "cpu", evaluation_setup=NS())
