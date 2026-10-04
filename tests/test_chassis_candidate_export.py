"""Native deterministic export and immutable-input rejection without Isaac."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


pytest.importorskip("rsl_rl")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from rsl_rl.models import MLPModel
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chassis_candidate_export", ROOT / "scripts/export_chassis_candidate.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


@pytest.fixture
def sealed(tmp_path):
    directory = tmp_path / "train" / "checkpoints" / "update_00002345"
    directory.mkdir(parents=True)
    contract = json.loads((ROOT / "contracts/v6_task_campaign_jump_worker_v6.json").read_text())
    raw = json.dumps(contract).encode()
    (directory / "contract.json").write_bytes(raw)
    agent = {"empirical_normalization": {}, "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 64], "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": .2, "std_type": "scalar"}}}
    (directory.parents[1] / "agent_config.json").write_text(json.dumps(agent))
    actor = MLPModel(TensorDict({"policy": torch.zeros(1, 35)}, batch_size=[1]),
        agent["obs_groups"], "actor", 6, **{key: value for key, value in agent["actor"].items() if key != "class_name"})
    infos = {"contract_id": contract["contract_id"], "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "asset_manifest_sha256": contract["asset_manifest_sha256"],
        "control_math_sha256": contract["control_math_sha256"], "task_actor_update": 42,
        "successful_updates_total": 2345, "rng_state": torch.get_rng_state()}
    path = directory / "model.pt"
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": {},
        "optimizer_state_dict": {}, "infos": infos}, path)
    (directory / "completion.json").write_text(json.dumps({"status": "checkpoint_sealed", "successful_updates": 2345}))
    return path, actor, agent


def test_native_export_is_exact_and_binds_all_25_command_vectors(sealed, tmp_path):
    path, actor, _ = sealed
    output = tmp_path / "candidate"
    original = EXPORT.digest(path)
    report = EXPORT.export_candidate(path, output)
    assert report["qualified"] is None and report["hardware_deployment_ready"] is False
    vectors = json.loads((output / "deployment/abi_test_vectors.json").read_text())["vectors"]
    observations = torch.tensor([row["obs"] for row in vectors])
    with torch.no_grad():
        expected = actor(TensorDict({"policy": observations}, batch_size=[25])).numpy()
    actual = ort.InferenceSession(str(output / "policy.onnx"), providers=["CPUExecutionProvider"]).run(
        ["actions"], {"obs": observations.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert len(vectors) == 25 and any(row["jump_request"] for row in vectors)
    assert any(not row["jump_request"] for row in vectors)
    assert all(row["obs"][28:] == [1., 0., 0., 0., 0., 0., 0.]
               for row in vectors if not row["jump_request"])
    assert {row["command_reference_phase"] for row in vectors} >= {0, 4, 5, 6, 7, 8}
    assert EXPORT.digest(path) == EXPORT.digest(output / "model.pt") == original
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["native_to_wrapper_max_abs_error"] == 0.
    assert provenance["checkpoint_infos"]["task_actor_update"] == 42
    assert provenance["optimizer_steps"] == 0 and provenance["isaac_scene_started"] is False
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected_hash, filename = line.split("  ", 1)
        assert EXPORT.digest(output / filename) == expected_hash


def test_unsealed_or_architecture_mismatched_checkpoints_are_rejected(sealed):
    path, _, agent = sealed
    marker = path.parent / "completion.json"
    marker.write_text(json.dumps({"status": "running", "successful_updates": 2345}))
    with pytest.raises(ValueError, match="immutable snapshot"):
        EXPORT.authenticate(path)
    marker.write_text(json.dumps({"status": "checkpoint_sealed", "successful_updates": 2345}))
    agent["actor"]["hidden_dims"] = [128, 128, 64]
    (path.parent.parents[1] / "agent_config.json").write_text(json.dumps(agent))
    with pytest.raises(ValueError, match="architecture differs"):
        EXPORT.authenticate(path)


def test_normalization_or_stale_contract_is_rejected_before_export(sealed):
    path, _, agent = sealed
    agent["actor"]["obs_normalization"] = True
    (path.parent.parents[1] / "agent_config.json").write_text(json.dumps(agent))
    with pytest.raises(ValueError, match="disabled observation normalization"):
        EXPORT.authenticate(path)
    agent["actor"]["obs_normalization"] = False
    (path.parent.parents[1] / "agent_config.json").write_text(json.dumps(agent))
    with (path.parent / "contract.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="does not authenticate"):
        EXPORT.authenticate(path)


def test_export_never_overwrites_an_existing_candidate(sealed, tmp_path):
    path, _, _ = sealed
    output = tmp_path / "candidate"
    output.mkdir()
    sentinel = output / "policy.onnx"
    sentinel.write_bytes(b"preserved")
    with pytest.raises(ValueError, match="immutable"):
        EXPORT.export_candidate(path, output)
    assert sentinel.read_bytes() == b"preserved"


def test_speed_abi_covers_high_commands_without_changing_ordinary_context():
    contract = json.loads((ROOT / "contracts/v6_task_campaign_speed_worker_v6.json").read_text())
    manifest = json.loads((ROOT / contract["asset_directory"] / "manifest.json").read_text())
    observations, rows = EXPORT.abi_vectors(contract, manifest)
    assert observations.shape == (35, 35)
    assert {row["commands_vx_yaw_height"][0] for row in rows} >= {-5., 5.}
    assert {row["commands_vx_yaw_height"][1] for row in rows} >= {-8., 8.}
    assert all(row["obs"][28:] == [1., 0., 0., 0., 0., 0., 0.]
               for row in rows if not row["jump_request"])
