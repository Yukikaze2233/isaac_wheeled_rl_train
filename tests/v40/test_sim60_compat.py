"""CPU regression checks for the actual RSL 5 model and Lab 3 quaternion boundary."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from wheeled_algo.v40_export import ExportError, export_checkpoint, load_actor_checkpoint
from wheeled_tasks.v40.contract import load_contract, make_run_manifest, validate_asset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def split_checkpoint(tmp_path):
    contract = load_contract(ROOT / "contracts/own_v40_v2.json")
    manifest = make_run_manifest(contract, validate_asset(contract, allow_research=True))
    manifest["runtime"] = {"checkpoint_format": "rsl_rl_5_split_mlp",
                           "versions": {"rsl-rl-lib": "5.5.1"},
                           "actor_class": "MLPModel", "critic_class": "MLPModel"}
    obs = TensorDict({"policy": torch.randn(4, 125), "critic": torch.randn(4, 29)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64],
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": .2, "std_type": "scalar"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[256, 128, 64])
    saved = {"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
             "infos": {key: manifest[key] for key in ("contract_id", "contract_sha256", "asset_manifest_sha256")}}
    checkpoint_path = tmp_path / "model.pt"
    manifest_path = tmp_path / "run_manifest.json"
    torch.save(saved, checkpoint_path)
    manifest_path.write_text(json.dumps(manifest))
    return actor, obs, saved, manifest, checkpoint_path, manifest_path


def test_official_rsl5_mean_actor_export(split_checkpoint, tmp_path):
    actor, obs, _, _, checkpoint, manifest = split_checkpoint
    exported, _ = load_actor_checkpoint(checkpoint, manifest)
    with torch.no_grad():
        torch.testing.assert_close(exported(obs["policy"]), actor(obs), rtol=0, atol=0)
    report = export_checkpoint(checkpoint, manifest, tmp_path / "policy.onnx")
    assert report["validation"]["max_abs_error"] < 1e-6


@pytest.mark.parametrize("corruption", ["missing_critic", "normalizer", "nan", "shape", "mixed", "provenance"])
def test_split_checkpoint_fails_closed(split_checkpoint, corruption):
    _, _, saved, manifest, checkpoint_path, manifest_path = split_checkpoint
    saved = copy.deepcopy(saved)
    if corruption == "missing_critic":
        del saved["critic_state_dict"]
    elif corruption == "normalizer":
        saved["actor_state_dict"]["obs_normalizer._mean"] = torch.zeros(125)
    elif corruption == "nan":
        saved["critic_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
    elif corruption == "shape":
        saved["actor_state_dict"]["mlp.0.weight"] = torch.zeros(256, 29)
    elif corruption == "mixed":
        saved["model_state_dict"] = {}
    else:
        del manifest["runtime"]
    torch.save(saved, checkpoint_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ExportError):
        load_actor_checkpoint(checkpoint_path, manifest_path)


def test_lab3_xyzw_clearance_matches_rotated_corners():
    path = ROOT / "src/wheeled_tasks/direct/v40_serial/env.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_base_visual_clearance")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    # A +90-degree rotation around Y sends local X to world -Z.
    quat = torch.tensor([[0., 2**-.5, 0., 2**-.5]])
    corners = torch.tensor([[-.1, -.2, -.3], [.1, .2, .3]])
    env = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(
        root_link_quat_w=SimpleNamespace(torch=quat))), _base_visual_corners=corners,
        _base_height=lambda: torch.tensor([.32]))
    torch.testing.assert_close(namespace[method.name](env), torch.tensor([.22]))


@pytest.mark.parametrize("resume", [False, True])
def test_native_rsl5_restore_uses_official_load(split_checkpoint, monkeypatch, resume):
    from rsl_rl.algorithms import PPO
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import train_v40

    actor, obs, saved, manifest, checkpoint, _ = split_checkpoint
    critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=[256, 128, 64])
    alg = PPO(actor, critic, storage=None, learning_rate=1e-4)
    saved["optimizer_state_dict"] = copy.deepcopy(alg.optimizer.state_dict())
    saved["optimizer_state_dict"]["param_groups"][0]["lr"] = 2e-4
    saved["iter"] = 7
    torch.save(saved, checkpoint)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(1)
    runner = SimpleNamespace(alg=alg, device="cpu", current_learning_iteration=99)
    train_v40.restore_checkpoint(runner, checkpoint, resume=resume, expected_manifest=manifest)
    assert runner.current_learning_iteration == (7 if resume else 0)
    assert alg.optimizer.param_groups[0]["lr"] == (2e-4 if resume else 1e-4)
    loaded = torch.load(checkpoint, weights_only=True)
    for key, value in alg.get_policy().state_dict().items():
        torch.testing.assert_close(value, loaded["actor_state_dict"][key], rtol=0, atol=0)
