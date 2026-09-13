"""CPU-only migration tests using actual RSL 5.5.1 MLPModel and PPO."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo import v40_export as exporter
from wheeled_algo import v40_job as job
from wheeled_algo import v40_warm_start as warm
from wheeled_tasks.v40.contract import contract_digest


def legacy_mlp(input_dim, output_dim):
    dims = [input_dim, 256, 128, 64, output_dim]
    layers = []
    for index, (left, right) in enumerate(zip(dims, dims[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < 3:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("round2-parent")
    contract = json.loads((ROOT / "contracts/own_v40_v2.json").read_text())
    (directory / "asset_manifest.json").write_text('{}')
    manifest = {
        "schema_version": 1, "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "asset_manifest_sha256": exporter.sha256_file(directory / "asset_manifest.json"),
        "actor_obs_dim": 125, "critic_obs_dim": 29, "action_dim": 6,
        "policy": contract["policy"], "target_versions": {"rsl-rl-lib": "3.0.1"},
    }
    for name, value in (("run_manifest.json", manifest), ("contract.json", contract),
                        ("agent_config.json", {}), ("source_hashes.json", {})):
        (directory / name).write_text(json.dumps(value))
    torch.manual_seed(42)
    state = {}
    for role, input_dim, output_dim in (("actor", 125, 6), ("critic", 29, 1)):
        state.update({f"{role}.{key}": value for key, value in
                      legacy_mlp(input_dim, output_dim).state_dict().items()})
    state["std"] = torch.tensor([0.21, 0.32, 0.43, 0.54, 0.65, 0.76])
    torch.save({"model_state_dict": state, "iter": 987,
                "optimizer_state_dict": {"state": {0: {"step": torch.tensor(987.)}}},
                "infos": {key: manifest[key] for key in
                          ("contract_id", "contract_sha256", "asset_manifest_sha256")}},
               directory / "model_final.pt")
    exporter.export_checkpoint(directory / "model_final.pt", directory / "run_manifest.json",
                               directory / "policy.onnx")
    budget = job.TrainingBudget()
    budget.completed_updates = 1
    receipt = job._receipt(budget, 1, "iterations_completed", "completed", "verified")
    job._publish_receipt(directory, receipt, job.ALLOWED_ARTIFACTS)
    return directory


@pytest.fixture
def parent(bundle, tmp_path, monkeypatch):
    directory = tmp_path / "parent"
    shutil.copytree(bundle, directory)
    checkpoint = directory / "model_final.pt"
    # Only synthetic tests replace the immutable production identity pins.
    monkeypatch.setattr(warm, "PARENT_SHA256", exporter.sha256_file(checkpoint))
    contract = json.loads((directory / "contract.json").read_text())
    monkeypatch.setattr(warm, "PARENT_CONTRACT_SHA256", contract_digest(contract))
    manifest = json.loads((directory / "run_manifest.json").read_text())
    manifest["runtime"] = {"checkpoint_format": "rsl_rl_5_split_mlp",
                           "versions": {"rsl-rl-lib": "5.5.1"},
                           "actor_class": "MLPModel", "critic_class": "MLPModel"}
    return checkpoint, contract, manifest


def runner():
    obs = TensorDict({"policy": torch.randn(8, 125), "critic": torch.randn(8, 29)}, [8])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64],
                     distribution_cfg={"class_name": "rsl_rl.modules:GaussianDistribution",
                                       "init_std": 1.0, "std_type": "scalar"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[256, 128, 64])
    storage = RolloutStorage("rl", 8, 2, obs, [6])
    return SimpleNamespace(alg=PPO(actor, critic, storage), current_learning_iteration=0), obs


@pytest.mark.parametrize("round3", [False, True])
def test_real_ppo_actor_critic_std_parity_and_fresh_state(parent, round3):
    checkpoint, contract, manifest = parent
    if round3:
        contract = json.loads((ROOT / "contracts/own_v40_round3_a.json").read_text())
        # Retain the fixture's asset identity; only the target contract changes.
        manifest["contract_sha256"] = contract_digest(contract)
    before = {path.name: exporter.sha256_file(path) for path in checkpoint.parent.iterdir()}
    split, lineage = warm.prepare_warm_start(checkpoint, contract, manifest)
    target, obs = runner()
    optimizer_before = copy.deepcopy(target.alg.optimizer.state_dict())
    warm.apply_warm_start(target, split)
    actor, source = exporter.load_actor_checkpoint(checkpoint, checkpoint.parent / "run_manifest.json")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    critic = legacy_mlp(29, 1)
    critic.load_state_dict({key.removeprefix("critic."): value for key, value in
                            saved["model_state_dict"].items() if key.startswith("critic.")})
    torch.testing.assert_close(target.alg.actor(obs), actor(obs["policy"]), rtol=0, atol=0)
    torch.testing.assert_close(target.alg.critic(obs), critic(obs["critic"]), rtol=0, atol=0)
    target.alg.actor(obs, stochastic_output=True)
    torch.testing.assert_close(target.alg.actor.distribution.std,
                               saved["model_state_dict"]["std"].expand(8, -1), rtol=0, atol=0)
    assert target.alg.optimizer.state_dict() == optimizer_before
    assert target.current_learning_iteration == 0
    assert target.alg.storage.step == 0
    assert lineage["optimizer_reset"] and lineage["mode"] == "warm_start"
    assert lineage["parent_contract_sha256"] == source["run_manifest"]["contract_sha256"]
    assert lineage["target_contract_sha256"] == contract_digest(contract)
    if round3:
        assert lineage["changed_contract_fields"] == ["commands", "rewards", "round3"]
    assert before == {path.name: exporter.sha256_file(path) for path in checkpoint.parent.iterdir()}


@pytest.mark.parametrize("field", ["commands", "rewards", "round3"])
def test_explicit_allowlist(parent, field):
    checkpoint, contract, manifest = parent
    expected_changes = [field]
    if field == "commands":
        contract[field]["stages"]["locomotion"]["vx"] = [-1.0, 1.0]
    elif field == "rewards":
        contract[field]["weights"]["velocity"] = 2.5
    else:
        contract = json.loads((ROOT / "contracts/own_v40_round3_a.json").read_text())
        expected_changes = ["commands", "rewards", "round3"]
    manifest["contract_sha256"] = contract_digest(contract)
    _, lineage = warm.prepare_warm_start(checkpoint, contract, manifest)
    assert lineage["changed_contract_fields"] == expected_changes


@pytest.mark.parametrize("field", ["timing", "observations", "joints", "actuators", "asset", "policy", "reset", "unknown"])
def test_fixed_contract_fields_rejected(parent, field):
    _, contract, manifest = parent
    if field in contract:
        contract[field]["unexpected"] = "must not migrate"
    else:
        contract[field] = {}
    manifest["contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError):
        warm.prepare_warm_start(*parent)


@pytest.mark.parametrize("name", ["model_final.pt", "contract.json", "asset_manifest.json",
                                  "run_manifest.json", "policy.onnx.json", "completion.json"])
def test_parent_tamper_rejected(parent, name):
    path = parent[0].parent / name
    if name == "model_final.pt":
        saved = torch.load(path, weights_only=True)
        saved["model_state_dict"]["critic.0.weight"][0, 0] = float("nan")
        torch.save(saved, path)
    else:
        path.write_text('{"tampered": true}')
    with pytest.raises(ValueError):
        warm.prepare_warm_start(*parent)


@pytest.mark.parametrize("dirty", ["iteration", "optimizer"])
def test_nonfresh_runner_rejected(parent, dirty):
    split, _ = warm.prepare_warm_start(*parent)
    target, _ = runner()
    if dirty == "iteration":
        target.current_learning_iteration = 1
    else:
        target.alg.optimizer.state[next(target.alg.actor.parameters())]["step"] = torch.tensor(1.)
    with pytest.raises(ValueError, match="fresh"):
        warm.apply_warm_start(target, split)


def test_ordinary_restore_gate_still_rejects_cross_contract_and_format(parent):
    checkpoint, contract, manifest = parent
    train = runpy.run_path(str(ROOT / "scripts/train_v40.py"))
    with pytest.raises(ValueError, match="runtime format"):
        train["checked_checkpoint"](checkpoint, manifest)
    contract["rewards"]["weights"]["velocity"] = 2.5
    manifest["contract_sha256"] = contract_digest(contract)
    with pytest.raises(ValueError, match="contract_sha256"):
        train["checked_checkpoint"](checkpoint, manifest)
    # Only the explicit migration gate admits the reward-only change.
    warm.prepare_warm_start(*parent)


@pytest.mark.parametrize("other", ["--resume", "--finetune"])
def test_cli_warm_start_is_mutually_exclusive(other):
    train = runpy.run_path(str(ROOT / "scripts/train_v40.py"))
    with pytest.raises(SystemExit) as error:
        train["main"](["--warm-start", "model_final.pt", other, "model_final.pt"])
    assert error.value.code == 2


def test_finite_checkpoint_substitution_rejected_by_pin(parent):
    checkpoint = parent[0]
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    saved["model_state_dict"]["actor.0.weight"][0, 0] += 0.01
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="pinned Round2"):
        warm.prepare_warm_start(*parent)
