"""Synthetic serialization/parity tests only: no training, RSL import, Isaac, or physics."""
from __future__ import annotations

from collections import OrderedDict
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo import v40_export as exporter  # noqa: E402


def make_manifest(actor_hidden=(16, 8), critic_hidden=(12, 8), activation="elu"):
    return {
        "schema_version": 1, "contract_id": exporter.CONTRACT_ID,
        "contract_sha256": "a" * 64, "asset_manifest_sha256": "b" * 64,
        "actor_obs_dim": 125, "critic_obs_dim": 29, "action_dim": 6,
        "policy": {
            "class_name": "ActorCritic", "actor_hidden_dims": list(actor_hidden),
            "critic_hidden_dims": list(critic_hidden), "activation": activation,
            "empirical_normalization": False,
        },
    }


def make_mlp(n_in, hidden, n_out, activation=nn.ELU):
    # Independent fixture construction using the audited official numeric module layout.
    modules = [nn.Linear(n_in, hidden[0]), activation()]
    for a, b in zip(hidden[:-1], hidden[1:]):
        modules.extend([nn.Linear(a, b), activation()])
    modules.append(nn.Linear(hidden[-1], n_out))
    return nn.Sequential(*modules)


def make_checkpoint(manifest):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(401)
        policy = manifest["policy"]
        actor = make_mlp(125, policy["actor_hidden_dims"], 6)
        critic = make_mlp(29, policy["critic_hidden_dims"], 1)
    state = OrderedDict()
    state.update(("actor." + k, v) for k, v in actor.state_dict().items())
    state.update(("critic." + k, v) for k, v in critic.state_dict().items())
    state["std"] = torch.ones(6)
    return {
        "model_state_dict": state, "optimizer_state_dict": {"state": {}, "param_groups": []},
        "iter": 0, "infos": {key: manifest[key] for key in
                                ("contract_id", "contract_sha256", "asset_manifest_sha256")},
    }, actor


@pytest.fixture
def artifacts(tmp_path):
    manifest = make_manifest()
    checkpoint, actor = make_checkpoint(manifest)
    return {"manifest": manifest, "checkpoint": checkpoint, "actor": actor,
            "manifest_path": tmp_path / "run_manifest.json", "checkpoint_path": tmp_path / "model.pt",
            "output": tmp_path / "policy.onnx"}


def persist(data):
    data["manifest_path"].write_text(json.dumps(data["manifest"]), encoding="utf-8")
    torch.save(data["checkpoint"], data["checkpoint_path"])


def load(data):
    persist(data)
    return exporter.load_actor_checkpoint(data["checkpoint_path"], data["manifest_path"])


def export(data):
    persist(data)
    return exporter.export_checkpoint(data["checkpoint_path"], data["manifest_path"], data["output"])


def test_official_numeric_actor_keys_and_safe_load(artifacts, monkeypatch):
    real_load = torch.load
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    actor, provenance = load(artifacts)
    assert calls == [{"weights_only": True, "map_location": "cpu"}]
    assert list(actor.state_dict()) == ["0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"]
    assert not actor.training and all(not p.requires_grad for p in actor.parameters())
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in actor.parameters())
    with torch.no_grad():
        sample = torch.linspace(-2, 2, 125).reshape(1, 125)
        torch.testing.assert_close(actor(sample), artifacts["actor"](sample), rtol=0, atol=0)
    assert provenance["checkpoint_sha256"] == exporter.sha256_file(artifacts["checkpoint_path"])
    assert provenance["run_manifest_sha256"] == exporter.sha256_file(artifacts["manifest_path"])


@pytest.mark.parametrize("hidden", [(16, 8), (256, 128, 64)])
def test_export_real_onnx_parity_and_sidecar(artifacts, hidden):
    artifacts["manifest"] = make_manifest(hidden, hidden)
    artifacts["checkpoint"], artifacts["actor"] = make_checkpoint(artifacts["manifest"])
    report = export(artifacts)
    sidecar = exporter.sidecar_path(artifacts["output"])
    assert json.loads(sidecar.read_text()) == report
    assert report["onnx_sha256"] == exporter.sha256_file(artifacts["output"])
    assert report["contract_sha256"] == "a" * 64
    assert report["asset_manifest_sha256"] == "b" * 64
    assert report["input"] == {"name": "obs_history", "dtype": "float32", "shape": [1, 125]}
    assert report["output"] == {"name": "actions", "dtype": "float32", "shape": [1, 6]}
    assert report["critic_obs_dim"] == 29 and report["dynamo"] is False
    assert report["validation"]["passed"] and report["validation"]["sample_count"] == 9
    assert report["action_order"] == ["L1", "L2", "L3", "R1", "R_jonit2", "R3"]
    assert report["action_joint_names"][4] == "R_jonit2"
    assert report["preprocessing"]["empirical_normalization"] is False
    graph = onnx.load(artifacts["output"])
    assert [(x.domain, x.version) for x in graph.opset_import] == [("", 17)]
    session = ort.InferenceSession(str(artifacts["output"]), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(9876)  # independent validation set, not the exporter's seed
    for sample in [np.zeros((1, 125), np.float32), rng.standard_normal((1, 125)).astype(np.float32)]:
        with torch.no_grad():
            expected = artifacts["actor"](torch.from_numpy(sample)).numpy()
        actual = session.run(["actions"], {"obs_history": sample})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    with pytest.raises(Exception):
        session.run(["actions"], {"obs_history": np.zeros((1, 35), np.float32)})
    assert not list(artifacts["output"].parent.glob(".v40-export-*"))


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", "1"), ("schema_version", True),
    ("contract_id", "V3"), ("contract_id", "own-v40-jointspace-h5-v3"),
    ("actor_obs_dim", 35), ("actor_obs_dim", 125.0), ("critic_obs_dim", 125),
    ("critic_obs_dim", 43), ("action_dim", 5),
    ("contract_sha256", "g" * 64), ("contract_sha256", "A" * 64),
    ("asset_manifest_sha256", "short"), ("asset_manifest_sha256", None),
])
def test_bad_manifest_contract(artifacts, field, value):
    artifacts["manifest"][field] = value
    with pytest.raises(exporter.ExportError):
        export(artifacts)
    assert not artifacts["output"].exists()
    assert not exporter.sidecar_path(artifacts["output"]).exists()


@pytest.mark.parametrize("field,value", [
    ("class_name", "ActorCriticRecurrent"), ("class_name", "__import__('os').system('true')"),
    ("activation", "__import__('os')"), ("activation", "ELU"),
    ("empirical_normalization", True), ("empirical_normalization", 0),
    ("actor_hidden_dims", []), ("actor_hidden_dims", [-1]), ("actor_hidden_dims", [True]),
    ("actor_hidden_dims", [16.0, 8]), ("critic_hidden_dims", [0]),
    ("actor_hidden_dims", [32, 8]), ("critic_hidden_dims", [10, 8]),
    ("actor_hidden_dims", [16, 8, 4]),
])
def test_explicit_policy_not_inferred(artifacts, field, value):
    artifacts["manifest"]["policy"][field] = value
    with pytest.raises(exporter.ExportError):
        load(artifacts)


@pytest.mark.parametrize("field", ["schema_version", "contract_sha256", "policy"])
def test_required_manifest_fields(artifacts, field):
    del artifacts["manifest"][field]
    with pytest.raises(exporter.ExportError):
        load(artifacts)


def test_policy_defaults_not_silently_applied(artifacts):
    del artifacts["manifest"]["policy"]["actor_hidden_dims"]
    with pytest.raises(exporter.ExportError, match="requires"):
        load(artifacts)


@pytest.mark.parametrize("value", [None, [], "metadata", ("a", "b")])
def test_infos_must_be_primitive_dict(artifacts, value):
    artifacts["checkpoint"]["infos"] = value
    with pytest.raises(exporter.ExportError, match="infos"):
        load(artifacts)


def test_missing_infos(artifacts):
    del artifacts["checkpoint"]["infos"]
    with pytest.raises(exporter.ExportError, match="infos"):
        load(artifacts)


@pytest.mark.parametrize("field,value", [
    ("contract_id", "V3"), ("contract_sha256", "c" * 64),
    ("asset_manifest_sha256", "d" * 64), ("schema_version", 2),
    ("actor_obs_dim", 35), ("critic_obs_dim", 125), ("action_dim", 7),
    ("training_info", {"step": torch.tensor(1)}), ("training_info", {"bad": float("nan")}),
    ("training_info", {"tuple": (1, 2)}),
])
def test_bad_checkpoint_metadata(artifacts, field, value):
    artifacts["checkpoint"]["infos"][field] = value
    with pytest.raises(exporter.ExportError):
        load(artifacts)


@pytest.mark.parametrize("legacy_only", [False, True])
def test_legacy_ac_is_rejected_even_with_valid_metadata(artifacts, legacy_only):
    artifacts["checkpoint"]["ac"] = artifacts["checkpoint"]["model_state_dict"]
    if legacy_only:
        del artifacts["checkpoint"]["model_state_dict"]
    with pytest.raises(exporter.ExportError, match="legacy"):
        load(artifacts)


@pytest.mark.parametrize("key,shape", [
    ("actor.0.weight", (16, 35)), ("actor.2.weight", (8, 17)),
    ("actor.4.weight", (7, 8)), ("actor.4.bias", (7,)),
    ("critic.0.weight", (12, 125)), ("critic.4.weight", (2, 8)), ("std", (7,)),
])
def test_real_weight_shape_must_match(artifacts, key, shape):
    artifacts["checkpoint"]["model_state_dict"][key] = torch.zeros(shape)
    with pytest.raises(exporter.ExportError, match="shape"):
        load(artifacts)


@pytest.mark.parametrize("key", ["actor.0.weight", "critic.2.bias"])
def test_missing_model_layer(artifacts, key):
    del artifacts["checkpoint"]["model_state_dict"][key]
    with pytest.raises(exporter.ExportError, match="keys mismatch"):
        load(artifacts)


def test_extra_model_layer(artifacts):
    artifacts["checkpoint"]["model_state_dict"]["encoder.0.weight"] = torch.zeros(1)
    with pytest.raises(exporter.ExportError, match="unexpected"):
        load(artifacts)


@pytest.mark.parametrize("key,value", [
    ("actor.0.weight", float("nan")), ("critic.0.weight", float("inf")), ("std", float("-inf")),
])
def test_nonfinite_state(artifacts, key, value):
    artifacts["checkpoint"]["model_state_dict"][key].view(-1)[0] = value
    with pytest.raises(exporter.ExportError, match="non-finite"):
        load(artifacts)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.int32])
def test_state_not_silently_cast(artifacts, dtype):
    state = artifacts["checkpoint"]["model_state_dict"]
    state["actor.0.weight"] = state["actor.0.weight"].to(dtype)
    with pytest.raises(exporter.ExportError, match="float32"):
        load(artifacts)


@pytest.mark.parametrize("location,key,value", [
    ("state", "actor_obs_normalizer._mean", torch.zeros(125)),
    ("state", "critic_obs_normalizer._var", torch.ones(29)),
    ("state", "actor.1.running_mean", torch.zeros(16)),
    ("checkpoint", "obs_norm_state_dict", {}), ("checkpoint", "obs_rms", {}),
    ("checkpoint", "critic_obs_norm_state_dict", {}),
    ("policy", "actor_obs_normalization", True), ("policy", "critic_obs_normalization", True),
    ("infos", "empirical_normalization", True),
])
def test_normalization_never_silently_dropped(artifacts, location, key, value):
    target = {"state": artifacts["checkpoint"]["model_state_dict"], "checkpoint": artifacts["checkpoint"],
              "policy": artifacts["manifest"]["policy"], "infos": artifacts["checkpoint"]["infos"]}[location]
    target[key] = value
    with pytest.raises(exporter.ExportError, match="normalization"):
        load(artifacts)


def test_explicit_official_normalization_flags_false_are_allowed(artifacts):
    artifacts["manifest"]["policy"].update(actor_obs_normalization=False, critic_obs_normalization=False)
    load(artifacts)


def test_unknown_policy_transform_is_rejected(artifacts):
    artifacts["manifest"]["policy"]["last_activation"] = "tanh"
    with pytest.raises(exporter.ExportError, match="unsupported fields"):
        load(artifacts)


@pytest.mark.parametrize("name,activation", [
    ("elu", nn.ELU), ("selu", nn.SELU), ("relu", nn.ReLU), ("crelu", nn.CELU),
    ("lrelu", nn.LeakyReLU), ("tanh", nn.Tanh), ("sigmoid", nn.Sigmoid),
    ("softplus", nn.Softplus), ("gelu", nn.GELU), ("swish", nn.SiLU),
    ("mish", nn.Mish), ("identity", nn.Identity),
])
def test_activation_whitelist_matches_v301(artifacts, name, activation):
    artifacts["manifest"]["policy"]["activation"] = name
    actor, _ = load(artifacts)
    assert type(actor[1]) is activation
    oracle = make_mlp(125, [16, 8], 6, activation)
    oracle.load_state_dict(artifacts["actor"].state_dict())
    sample = torch.linspace(-1, 1, 125).reshape(1, 125)
    with torch.no_grad():
        torch.testing.assert_close(actor(sample), oracle(sample), rtol=0, atol=0)
    report = export(artifacts)
    assert report["validation"]["passed"]


@pytest.mark.parametrize("kind", ["missing", "both", "negative"])
def test_invalid_action_noise(artifacts, kind):
    state = artifacts["checkpoint"]["model_state_dict"]
    if kind == "missing":
        del state["std"]
    elif kind == "both":
        state["log_std"] = torch.zeros(6)
    else:
        state["std"][0] = -1
    with pytest.raises(exporter.ExportError):
        load(artifacts)


def test_log_std_accepted_but_not_exported(artifacts):
    state = artifacts["checkpoint"]["model_state_dict"]
    del state["std"]
    state["log_std"] = torch.zeros(6)
    actor, _ = load(artifacts)
    assert all("std" not in key for key in actor.state_dict())


@pytest.mark.parametrize("sample", [
    torch.zeros(1, 35), torch.zeros(125), torch.zeros(2, 125), torch.zeros(1, 5, 25),
    torch.zeros(1, 125, dtype=torch.float64), torch.full((1, 125), float("nan")),
    torch.full((1, 125), float("inf")),
])
def test_bad_observation_rejected(sample):
    with pytest.raises(exporter.ExportError):
        exporter.validate_observation(sample)


def test_valid_observation():
    exporter.validate_observation(torch.zeros(1, 125, dtype=torch.float32))


def test_duplicate_and_nonfinite_json_rejected(artifacts):
    persist(artifacts)
    path = artifacts["manifest_path"]
    for raw in ('{"schema_version": 1, "schema_version": 1}', '{"value": NaN}'):
        path.write_text(raw)
        with pytest.raises(exporter.ExportError):
            exporter.load_actor_checkpoint(artifacts["checkpoint_path"], path)


def test_unsafe_pickled_module_is_not_loaded(artifacts):
    persist(artifacts)
    torch.save(nn.Linear(125, 6), artifacts["checkpoint_path"])
    with pytest.raises(exporter.ExportError, match="weights_only=True"):
        exporter.load_actor_checkpoint(artifacts["checkpoint_path"], artifacts["manifest_path"])


@pytest.mark.parametrize("which", ["model", "sidecar", "dangling_symlink"])
def test_no_overwrite(artifacts, which):
    target = exporter.sidecar_path(artifacts["output"]) if which == "sidecar" else artifacts["output"]
    if which == "dangling_symlink":
        target.symlink_to(target.parent / "missing-file")
    else:
        target.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError, match="overwrite"):
        export(artifacts)
    if which == "dangling_symlink":
        assert target.is_symlink()
    else:
        assert target.read_bytes() == b"existing artifact"


def test_verification_failure_leaves_no_artifacts(artifacts, monkeypatch):
    def fail(*args, **kwargs):
        raise exporter.ExportError("intentional numerical mismatch")
    monkeypatch.setattr(exporter, "verify_onnx", fail)
    with pytest.raises(exporter.ExportError, match="mismatch"):
        export(artifacts)
    assert not artifacts["output"].exists()
    assert not exporter.sidecar_path(artifacts["output"]).exists()
    assert not list(artifacts["output"].parent.glob(".v40-export-*"))


def test_missing_ort_fails_closed(artifacts, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(exporter.ExportError, match="verification cannot be skipped"):
        export(artifacts)
    assert not artifacts["output"].exists()


def test_second_publish_race_preserves_other_file_and_rolls_back_model(artifacts, monkeypatch):
    real_link = os.link
    sidecar = exporter.sidecar_path(artifacts["output"])

    def raced_link(source, destination, **kwargs):
        if Path(destination) == sidecar:
            sidecar.write_bytes(b"concurrent artifact")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(exporter.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        export(artifacts)
    assert not artifacts["output"].exists()
    assert sidecar.read_bytes() == b"concurrent artifact"


def test_real_numerical_mismatch_rejected(artifacts):
    export(artifacts)
    altered_actor = copy.deepcopy(artifacts["actor"])
    with torch.no_grad():
        altered_actor[-1].bias.add_(1)
    with pytest.raises(exporter.ExportError, match="mismatch"):
        exporter.verify_onnx(altered_actor, artifacts["output"])


def test_nonfinite_torch_result_rejected(artifacts):
    export(artifacts)
    with torch.no_grad():
        artifacts["actor"][-1].bias[0] = float("nan")
    with pytest.raises(exporter.ExportError, match="non-finite actions"):
        exporter.verify_onnx(artifacts["actor"], artifacts["output"])


def test_nonfinite_onnx_result_rejected(artifacts):
    export(artifacts)
    graph = onnx.load(artifacts["output"])
    bias = next(value for value in graph.graph.initializer if value.name == "4.bias")
    bias.CopyFrom(onnx.numpy_helper.from_array(np.full((6,), np.nan, np.float32), name="4.bias"))
    onnx.save(graph, artifacts["output"])
    with pytest.raises(exporter.ExportError, match="ONNXRuntime.*non-finite actions"):
        exporter.verify_onnx(artifacts["actor"], artifacts["output"])


def test_finite_weights_with_overflow_fail_before_publication(artifacts):
    state = artifacts["checkpoint"]["model_state_dict"]
    state["actor.0.bias"].fill_(1)
    state["actor.2.weight"].fill_(torch.finfo(torch.float32).max)
    state["actor.4.weight"].fill_(1)
    with pytest.raises(exporter.ExportError, match="non-finite actions"):
        export(artifacts)
    assert not artifacts["output"].exists()
    assert not exporter.sidecar_path(artifacts["output"]).exists()


@pytest.mark.parametrize("bad_field", ["input_name", "output_name", "input_shape", "output_shape", "opset"])
def test_wrong_onnx_signature_or_opset_rejected(artifacts, bad_field):
    n_in = 35 if bad_field == "input_shape" else 125
    n_out = 5 if bad_field == "output_shape" else 6
    actor = nn.Linear(n_in, n_out).eval()
    torch.onnx.export(actor, (torch.zeros(1, n_in),), str(artifacts["output"]),
                      input_names=["obs" if bad_field == "input_name" else "obs_history"],
                      output_names=["out" if bad_field == "output_name" else "actions"],
                      opset_version=13 if bad_field == "opset" else 17,
                      dynamo=False, external_data=False)
    with pytest.raises(exporter.ExportError, match="ONNX"):
        exporter.verify_onnx(artifacts["actor"], artifacts["output"])


def test_cli_success_and_no_overwrite(artifacts):
    persist(artifacts)
    command = [sys.executable, str(ROOT / "scripts/export_v40_onnx.py"),
               "--checkpoint", str(artifacts["checkpoint_path"]),
               "--run-manifest", str(artifacts["manifest_path"]), "--output", str(artifacts["output"])]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["validation_samples"] == 9
    before = artifacts["output"].read_bytes()
    rejected = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
    assert rejected.returncode == 1 and "overwrite" in rejected.stderr
    assert artifacts["output"].read_bytes() == before


def test_cli_help_needs_no_isaac_or_rsl():
    spec = importlib.util.spec_from_file_location("export_v40_cli", ROOT / "scripts/export_v40_onnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0
