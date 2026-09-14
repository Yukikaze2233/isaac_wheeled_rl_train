"""A -> B1 snapshot transfer and exact material-mapping resume, without Isaac."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo import v40_export as exporter
from wheeled_algo import v40_job as job
from wheeled_algo import v40_stage_transfer as transfer
from wheeled_algo.v40_warm_start import apply_warm_start
from wheeled_tasks.v40.contract import contract_digest, load_contract, make_run_manifest, validate_asset


def make_runner():
    obs = TensorDict({"policy": torch.randn(8, 125), "critic": torch.randn(8, 29)}, [8])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64],
                     distribution_cfg={"class_name": "GaussianDistribution", "std_type": "scalar"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[256, 128, 64])
    storage = RolloutStorage("rl", 8, 2, obs, [6])
    return SimpleNamespace(alg=PPO(actor, critic, storage), current_learning_iteration=0, device="cpu"), obs


def runtime_manifest(contract):
    asset = validate_asset(contract, allow_research=True)
    manifest = make_run_manifest(contract, asset)
    manifest["runtime"] = {"checkpoint_format": "rsl_rl_5_split_mlp",
                           "versions": {"rsl-rl-lib": "5.5.1"},
                           "actor_class": "MLPModel", "critic_class": "MLPModel"}
    return manifest, asset


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("stage-a-snapshot")
    contract = load_contract(ROOT / "contracts/own_v40_round3_a.json")
    manifest, asset = runtime_manifest(contract)
    shutil.copyfile(asset["manifest_path"], directory / "asset_manifest.json")
    for name, value in (("contract.json", contract), ("run_manifest.json", manifest), ("agent_config.json", {})):
        (directory / name).write_bytes(job.json_bytes(value))
    hashes = {"schema_version": 1,
              "contract_snapshot_sha256": job.sha256_file(directory / "contract.json"),
              "asset_manifest_sha256": manifest["asset_manifest_sha256"],
              "source_files_sha256": {"scripts/train_v40.py": job.sha256_file(ROOT / "scripts/train_v40.py")}}
    (directory / "source_hashes.json").write_bytes(job.json_bytes(hashes))
    source, obs = make_runner()
    # Real Adam moments must exist in the parent, to detect accidental reuse.
    (source.alg.actor(obs).square().mean() + source.alg.critic(obs).square().mean()).backward()
    source.alg.optimizer.step()
    with torch.no_grad():
        source.alg.actor.distribution.std_param.copy_(torch.tensor([.21, .32, .43, .54, .65, .76]))
    saved = {"actor_state_dict": source.alg.actor.state_dict(),
             "critic_state_dict": source.alg.critic.state_dict(),
             "optimizer_state_dict": source.alg.optimizer.state_dict(), "iter": 37,
             "infos": {key: manifest[key] for key in
                       ("contract_id", "contract_sha256", "asset_manifest_sha256")}}
    torch.save(saved, directory / "model_37.pt")
    return directory


@pytest.fixture
def parent(bundle, tmp_path):
    directory = tmp_path / "parent"
    shutil.copytree(bundle, directory)
    return directory / "model_37.pt"


@pytest.fixture
def target():
    contract = load_contract(ROOT / "contracts/own_v40_round3_b1.json")
    manifest, _ = runtime_manifest(contract)
    return contract, manifest


def prepare(parent, target, **kwargs):
    kwargs.setdefault("source_checkpoint_sha256", job.sha256_file(parent))
    return transfer.prepare_stage_transfer(parent, *target, **kwargs)


def test_periodic_transfer_real_ppo_parity_fresh_state(parent, target):
    before = {p.name: job.sha256_file(p) for p in parent.parent.iterdir()}
    split, lineage = prepare(parent, target)
    assert not (parent.parent / "completion.json").exists()
    assert lineage["source_is_final"] is False
    assert lineage["parent_completion_verified"] is False
    assert lineage["saved_iter"] == 37 and lineage["source_filename"] == "model_37.pt"
    assert lineage["mode"] == "stage_transfer" and lineage["optimizer_reset"]
    saved = torch.load(parent, weights_only=True, map_location="cpu")
    assert saved["optimizer_state_dict"]["state"]
    source, obs = make_runner()
    source.alg.load(saved, {"actor": True, "critic": True}, strict=True)
    runner, _ = make_runner()
    optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
    apply_warm_start(runner, split)
    actor, _ = exporter.load_actor_checkpoint(parent, parent.parent / "run_manifest.json")
    torch.testing.assert_close(runner.alg.actor(obs), actor(obs["policy"]), rtol=0, atol=0)
    torch.testing.assert_close(runner.alg.critic(obs), source.alg.critic(obs), rtol=0, atol=0)
    runner.alg.actor(obs, stochastic_output=True)
    source.alg.actor(obs, stochastic_output=True)
    torch.testing.assert_close(runner.alg.actor.distribution.std, source.alg.actor.distribution.std, rtol=0, atol=0)
    assert runner.alg.optimizer.state_dict() == optimizer
    assert runner.current_learning_iteration == runner.alg.storage.step == 0
    assert before == {p.name: job.sha256_file(p) for p in parent.parent.iterdir()}


@pytest.mark.parametrize("field", ["commands", "rewards", "observations", "actions", "timing", "actuators", "physics_material", "unknown"])
def test_semantic_changes_rejected(parent, target, field):
    contract, manifest = target
    location = contract["round3"] if field == "physics_material" else contract
    location[field] = {"tampered": True}
    with pytest.raises(ValueError):
        prepare(parent, (contract, manifest))


def test_a_is_not_a_b1_target(parent):
    contract = load_contract(ROOT / "contracts/own_v40_round3_a.json")
    manifest, _ = runtime_manifest(contract)
    with pytest.raises(ValueError, match="B1"):
        prepare(parent, (contract, manifest))


@pytest.mark.parametrize("file", ["model_37.pt", "run_manifest.json", "contract.json", "asset_manifest.json", "source_hashes.json", "agent_config.json"])
def test_snapshot_rechecked_since_preflight(parent, target, file):
    _, lineage = prepare(parent, target)
    path = parent.parent / file
    # Whitespace is semantically harmless JSON, but changes the frozen snapshot.
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="SHA256|snapshot|identity|original asset"):
        prepare(parent, target, source_checkpoint_sha256=lineage["parent_checkpoint_sha256"],
                expected_provenance=lineage)


def test_infos_and_finite_checks_cannot_be_bypassed_by_a_new_sha(parent, target):
    saved = torch.load(parent, weights_only=True)
    saved["infos"]["contract_sha256"] = target[1]["contract_sha256"]
    torch.save(saved, parent)
    with pytest.raises(ValueError, match="infos.contract_sha256"):
        prepare(parent, target)
    saved["infos"]["contract_sha256"] = transfer.SOURCE_CONTRACT_SHA256
    saved["critic_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
    torch.save(saved, parent)
    with pytest.raises(ValueError, match="non-finite"):
        prepare(parent, target)


def publish_final(parent):
    directory = parent.parent
    final = directory / "model_final.pt"
    shutil.copyfile(parent, final)
    exporter.export_checkpoint(final, directory / "run_manifest.json", directory / "policy.onnx")
    budget = job.TrainingBudget()
    budget.completed_updates = 1
    receipt = job._receipt(budget, 1, "iterations_completed", "completed", "verified")
    job._publish_receipt(directory, receipt, job.ALLOWED_ARTIFACTS)
    return final


def test_final_receipt_and_all_artifacts_checked(parent, target):
    final = publish_final(parent)
    _, lineage = prepare(final, target)
    assert lineage["source_is_final"] and lineage["parent_completion_verified"]
    assert "completion.json" in lineage["parent_snapshot_files"]
    _, periodic_lineage = prepare(parent, target)
    assert periodic_lineage["source_is_final"] is False
    assert periodic_lineage["parent_completion_verified"] is True
    (parent.parent / "policy.onnx").write_bytes(b"tampered")
    for checkpoint in (parent, final):
        with pytest.raises(ValueError, match="artifact mismatch"):
            prepare(checkpoint, target)


def test_final_without_receipt_is_not_completed_parent(parent, target):
    final = parent.with_name("model_final.pt")
    shutil.copyfile(parent, final)
    with pytest.raises(ValueError, match="completion receipt"):
        prepare(final, target)


def test_failed_completion_rejected(parent, target):
    publish_final(parent)
    path = parent.parent / "completion.json"
    receipt = json.loads(path.read_text())
    receipt["status"] = "training_failed"
    path.write_bytes(job.json_bytes(receipt))
    with pytest.raises(ValueError):
        prepare(parent, target)


def material_report():
    # The environment owns the schema/PhysX readback; transfer preserves it whole.
    return {"seed": 42, "bucket_table": [[.5, .5, 0.]] * 64,
            "env_bucket_ids": list(range(8)), "nominal_mask": [True] * 8,
            "readback": {"passed": True, "materials": [[.5, .5, 0.]] * 8}}


def test_material_report_copy_and_metadata_identity(target):
    contract, manifest = target
    before = copy.deepcopy(manifest)
    env = SimpleNamespace(num_envs=8, round3_material_report=material_report())
    transfer.bind_material_report(manifest, contract, env)
    assert manifest["domain_randomization_report"] == env.round3_material_report
    assert all(manifest[key] == before[key] for key in before)
    env.round3_material_report["env_bucket_ids"][0] = 63
    assert manifest["domain_randomization_report"]["env_bucket_ids"][0] == 0


def test_startup_report_does_not_change_transfer_lineage_and_is_exported(parent, target):
    contract, manifest = target
    _, lineage = prepare(parent, target)
    report = material_report()
    report["randomization_config"] = contract["round3"]["material_randomization"]
    transfer.bind_material_report(manifest, contract, SimpleNamespace(num_envs=8, round3_material_report=report))
    split, _ = prepare(parent, target, expected_provenance=lineage)
    manifest["source_provenance"] = lineage
    final = parent.with_name("model_final.pt")
    infos = {key: manifest[key] for key in ("contract_id", "contract_sha256", "asset_manifest_sha256",
                                          "round3_stage", "num_envs", "domain_randomization_report")}
    torch.save({**split, "infos": infos, "iter": 0}, final)
    (parent.parent / "run_manifest.json").write_bytes(job.json_bytes(manifest))
    sidecar = exporter.export_checkpoint(final, parent.parent / "run_manifest.json", parent.parent / "policy.onnx")
    assert sidecar["run_manifest"]["domain_randomization_report"] == report
    assert sidecar["run_manifest"]["source_provenance"] == lineage


@pytest.mark.parametrize("report", [None, {}, {"tensor": torch.ones(1)}, {"nan": float("nan")}])
def test_invalid_material_report_rejected(target, report):
    with pytest.raises(ValueError):
        transfer.bind_material_report(target[1], target[0], SimpleNamespace(num_envs=8, round3_material_report=report))


@pytest.mark.parametrize("difference", [None, "num_envs", "seed", "env_bucket_ids", "readback", "missing"])
def test_b1_resume_mapping_gate_precedes_optimizer_restore(parent, target, difference):
    contract, manifest = target
    transfer.bind_material_report(manifest, contract, SimpleNamespace(num_envs=8, round3_material_report=material_report()))
    cli = runpy.run_path(str(ROOT / "scripts/train_v40.py"))
    saved = torch.load(parent, weights_only=True)
    source, _ = make_runner()
    source.save = lambda path, infos=None: torch.save({**saved, "infos": infos}, path)
    cli["bind_checkpoint_metadata"](source, manifest)
    source.save(parent)
    (parent.parent / "run_manifest.json").write_bytes(job.json_bytes(manifest))
    current = copy.deepcopy(manifest)
    if difference == "num_envs":
        current["num_envs"] = 16
    elif difference == "seed":
        current["domain_randomization_report"]["seed"] = 43
    elif difference == "env_bucket_ids":
        current["domain_randomization_report"]["env_bucket_ids"][0] = 63
    elif difference == "readback":
        current["domain_randomization_report"]["readback"]["materials"][0] = [.3, .3, 0.]
    elif difference == "missing":
        del current["domain_randomization_report"]
    runner, _ = make_runner()
    optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
    if difference:
        with pytest.raises(ValueError, match="B1 exact resume"):
            cli["restore_checkpoint"](runner, parent, resume=True, expected_manifest=current)
        assert runner.alg.optimizer.state_dict() == optimizer
        assert runner.current_learning_iteration == 0
    else:
        cli["restore_checkpoint"](runner, parent, resume=True, expected_manifest=current)
        assert runner.current_learning_iteration == 37 and runner.alg.optimizer.state


def test_a_resume_report_gate_is_unchanged():
    transfer.require_same_material_mapping({}, {})


@pytest.mark.parametrize("args", [
    ["--stage-transfer", "model_37.pt"], ["--source-checkpoint-sha256", "a" * 64],
    *[["--stage-transfer", "model_37.pt", "--source-checkpoint-sha256", "a" * 64, flag, "model.pt"]
      for flag in ("--resume", "--finetune", "--warm-start")],
])
def test_cli_requires_paired_sha_and_exclusive_transfer(args):
    cli = runpy.run_path(str(ROOT / "scripts/train_v40.py"))
    with pytest.raises(SystemExit) as error:
        cli["main"](args)
    assert error.value.code == 2
