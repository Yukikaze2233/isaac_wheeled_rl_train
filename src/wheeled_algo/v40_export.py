"""Fail-closed mean-actor export for own-v40-jointspace-h5-v1.

Only PyTorch, ONNX and ONNXRuntime are needed; never import Isaac or rsl_rl.
The RSL-RL v3.0.1 source/key audit and interface limits are in docs/V40_EXPORT.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

CONTRACT_ID = "own-v40-jointspace-h5-v1"
ACTOR_OBS_DIM = 125
CRITIC_OBS_DIM = 29
ACTION_DIM = 6
DEFAULT_HIDDEN_DIMS = (256, 128, 64)
ACTION_ORDER = ("L1", "L2", "L3", "R1", "R_jonit2", "R3")
JOINT_NAMES = ("L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3")
OPSET = 17
VALIDATION_SEED = 40
ATOL = 1e-6
RTOL = 1e-5

# Explicit constructors, matching v3.0.1 utils.resolve_nn_activation (not v2.x).
_ACTIVATIONS = {
    "elu": nn.ELU, "selu": nn.SELU, "relu": nn.ReLU, "crelu": nn.CELU,
    "lrelu": nn.LeakyReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus, "gelu": nn.GELU, "swish": nn.SiLU,
    "mish": nn.Mish, "identity": nn.Identity,
}
_METADATA_KEYS = ("contract_id", "contract_sha256", "asset_manifest_sha256")
_POLICY_KEYS = {
    "class_name", "actor_hidden_dims", "critic_hidden_dims", "activation",
    "empirical_normalization",
}
_NORMALIZATION_FLAGS = {
    "empirical_normalization", "actor_obs_normalization", "critic_obs_normalization",
}


class ExportError(ValueError):
    """An artifact cannot satisfy the frozen export contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def _primitive_json(value: Any, location: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), f"{location}: non-finite JSON number")
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _primitive_json(item, f"{location}[{i}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            _require(type(key) is str, f"{location}: JSON keys must be strings")
            _primitive_json(item, f"{location}.{key}")
        return
    raise ExportError(f"{location}: expected primitive JSON, got {type(value).__name__}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ExportError(f"non-finite JSON constant: {value}")


def _metadata(record: dict, location: str) -> None:
    _require(record.get("contract_id") in {CONTRACT_ID, "own-v40-jointspace-h5-v2"}, f"{location}: wrong contract_id")
    for key in _METADATA_KEYS[1:]:
        value = record.get(key)
        _require(
            type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{location}.{key}: expected lowercase 64-hex SHA-256",
        )


def _reject_normalization(record: Mapping, location: str) -> None:
    for key, value in record.items():
        if type(key) is not str:
            raise ExportError(f"{location}: expected string keys")
        if any(marker in key.lower() for marker in ("norm", "rms", "running_mean", "running_var")):
            if key in _NORMALIZATION_FLAGS and value is False:
                continue
            raise ExportError(f"{location}.{key}: unsupported normalization state/setting")


def validate_manifest(manifest: dict) -> dict:
    """Validate explicit architecture; no defaults or dimensions inferred from tensors."""
    _require(type(manifest) is dict, "run manifest must be a JSON object")
    _primitive_json(manifest, "manifest")
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
             "manifest.schema_version must be integer 1")
    _metadata(manifest, "manifest")
    for key, expected in (("actor_obs_dim", ACTOR_OBS_DIM), ("critic_obs_dim", CRITIC_OBS_DIM),
                          ("action_dim", ACTION_DIM)):
        _require(type(manifest.get(key)) is int and manifest[key] == expected,
                 f"manifest.{key} must be integer {expected}")
    policy = manifest.get("policy")
    _require(type(policy) is dict, "manifest.policy must be an object")
    _require(_POLICY_KEYS <= policy.keys(), f"manifest.policy requires {sorted(_POLICY_KEYS)}")
    _require(not (policy.keys() - _POLICY_KEYS - _NORMALIZATION_FLAGS),
             "manifest.policy contains unsupported fields")
    _require(policy["class_name"] == "ActorCritic", "only stock ActorCritic is supported")
    _require(policy["empirical_normalization"] is False,
             "policy.empirical_normalization must be false")
    _reject_normalization(policy, "manifest.policy")
    _reject_normalization(manifest, "manifest")
    _require(type(policy["activation"]) is str and policy["activation"] in _ACTIVATIONS,
             f"activation must be one of {sorted(_ACTIVATIONS)}")
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        dims = policy[key]
        _require(type(dims) is list and len(dims) > 0
                 and all(type(dim) is int and dim > 0 for dim in dims),
                 f"policy.{key} must be a nonempty list of positive integers (no inference)")
    return manifest


def _read_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw, object_pairs_hook=_unique_object,
                              parse_constant=_invalid_json_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"invalid run manifest JSON: {exc}") from exc
    return validate_manifest(manifest), hashlib.sha256(raw).hexdigest()


def _hash_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return _hash_stream(stream)


def _read_checkpoint(path: Path) -> tuple[dict, str]:
    # Hash and load the same open file; reject observable concurrent mutation.
    with path.open("rb") as stream:
        digest = _hash_stream(stream)
        stream.seek(0)
        try:
            checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
        except Exception as exc:
            raise ExportError(f"safe checkpoint load failed (weights_only=True): {exc}") from exc
        stream.seek(0)
        _require(_hash_stream(stream) == digest, "checkpoint changed during export")
    _require(type(checkpoint) is dict, "checkpoint must be a stock checkpoint dict")
    _require("ac" not in checkpoint, "legacy V3 'ac' checkpoint is forbidden")
    _reject_normalization(checkpoint, "checkpoint")
    return checkpoint, digest


def _expected_shapes(prefix: str, input_dim: int, hidden: list[int], output_dim: int) -> dict:
    dims = [input_dim, *hidden, output_dim]
    shapes = {}
    for i, (fan_in, fan_out) in enumerate(zip(dims, dims[1:])):
        shapes[f"{prefix}.{2 * i}.weight"] = (fan_out, fan_in)
        shapes[f"{prefix}.{2 * i}.bias"] = (fan_out,)
    return shapes


def _validate_state(checkpoint: dict, manifest: dict) -> Mapping:
    infos = checkpoint.get("infos")
    _require(type(infos) is dict, "checkpoint.infos must be a primitive JSON dict (not absent/None)")
    _primitive_json(infos, "checkpoint.infos")
    _metadata(infos, "checkpoint.infos")
    _reject_normalization(infos, "checkpoint.infos")
    for key in _METADATA_KEYS:
        _require(infos[key] == manifest[key], f"checkpoint.infos.{key} does not match run manifest")
    # If training includes these redundant fields, they must not contradict the manifest.
    for key in ("schema_version", "actor_obs_dim", "critic_obs_dim", "action_dim", "policy"):
        if key in infos:
            _require(type(infos[key]) is type(manifest[key]) and infos[key] == manifest[key],
                     f"checkpoint.infos.{key} does not match run manifest")
    state = checkpoint.get("model_state_dict")
    _require(isinstance(state, Mapping), "checkpoint.model_state_dict must be a tensor mapping")
    _reject_normalization(state, "model_state_dict")
    policy = manifest["policy"]
    expected = _expected_shapes("actor", ACTOR_OBS_DIM, policy["actor_hidden_dims"], ACTION_DIM)
    expected.update(_expected_shapes("critic", CRITIC_OBS_DIM, policy["critic_hidden_dims"], 1))
    noise_keys = set(state) & {"std", "log_std"}
    _require(len(noise_keys) == 1, "stock ActorCritic requires exactly one of std/log_std")
    expected[next(iter(noise_keys))] = (ACTION_DIM,)
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    _require(not missing and not extra,
             f"model_state_dict keys mismatch; missing={sorted(missing)}, unexpected={sorted(extra)}")
    # Check ALL weights before allocating the architecture. Do not silently cast or drop state.
    for key, shape in expected.items():
        tensor = state[key]
        _require(type(tensor) is torch.Tensor, f"{key}: expected a plain tensor")
        _require(tuple(tensor.shape) == shape, f"{key}: expected shape {shape}, got {tuple(tensor.shape)}")
        _require(tensor.dtype == torch.float32 and tensor.device.type == "cpu"
                 and tensor.layout == torch.strided, f"{key}: expected dense CPU float32")
        _require(bool(torch.isfinite(tensor).all()), f"{key}: non-finite weights")
    if "std" in state:
        _require(bool((state["std"] > 0).all()), "std must be positive")
    return state


def _build_actor(manifest: dict, state: Mapping) -> nn.Sequential:
    policy = manifest["policy"]
    dims = [ACTOR_OBS_DIM, *policy["actor_hidden_dims"], ACTION_DIM]
    layers: list[nn.Module] = []
    activation = _ACTIVATIONS[policy["activation"]]()
    # The audited official MLP registers numeric modules and has no output activation.
    with torch.random.fork_rng(devices=[]):
        for i, (fan_in, fan_out) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(fan_in, fan_out, device="cpu", dtype=torch.float32))
            if i < len(dims) - 2:
                layers.append(activation)
    actor = nn.Sequential(*layers)
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items()
                           if key.startswith("actor.")}, strict=True)
    actor.requires_grad_(False)
    return actor.eval()


def load_actor_checkpoint(checkpoint: str | Path, run_manifest: str | Path) -> tuple[nn.Sequential, dict]:
    """Return the strictly validated CPU mean actor and byte-hash provenance."""
    manifest, manifest_hash = _read_manifest(Path(run_manifest))
    saved, checkpoint_hash = _read_checkpoint(Path(checkpoint))
    state = _validate_state(saved, manifest)
    actor = _build_actor(manifest, state)
    return actor, {
        "run_manifest": manifest,
        "checkpoint_sha256": checkpoint_hash,
        "run_manifest_sha256": manifest_hash,
        "checkpoint_layout": "RSL-RL v3.0.1 model_state_dict / ActorCritic numeric actor.* keys",
    }


def validate_observation(observation: torch.Tensor) -> None:
    """Deployment callers must apply equivalent checks; ONNX does not reject NaN itself."""
    _require(type(observation) is torch.Tensor, "obs_history must be a tensor")
    _require(tuple(observation.shape) == (1, ACTOR_OBS_DIM), "obs_history must have shape [1,125]")
    _require(observation.dtype == torch.float32 and observation.device.type == "cpu",
             "obs_history must be CPU float32")
    _require(bool(torch.isfinite(observation).all()), "obs_history contains NaN/Inf")


def _onnx_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ExportError("numpy, onnx and onnxruntime are required; verification cannot be skipped") from exc
    return np, onnx, ort


def verify_onnx(actor: nn.Module, path: str | Path) -> dict:
    """Check a fixed ONNX signature and deterministic CPU Torch/ORT parity (not physics)."""
    np, onnx, ort = _onnx_dependencies()
    model = onnx.load(str(path), load_external_data=False)
    _require(all(t.data_location != onnx.TensorProto.EXTERNAL for t in model.graph.initializer),
             "ONNX must be a self-contained file")
    _require([(entry.domain, entry.version) for entry in model.opset_import] == [("", OPSET)],
             "ONNX must use only standard opset 17")
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    _require(len(inputs) == 1 and inputs[0].name == "obs_history"
             and inputs[0].type == "tensor(float)" and inputs[0].shape == [1, ACTOR_OBS_DIM],
             "ONNX input must be obs_history float32[1,125]")
    _require(len(outputs) == 1 and outputs[0].name == "actions"
             and outputs[0].type == "tensor(float)" and outputs[0].shape == [1, ACTION_DIM],
             "ONNX output must be actions float32[1,6]")
    rng = np.random.default_rng(VALIDATION_SEED)
    scales = [0.1, 1.0, 3.0, 10.0] * 2
    observations = [np.zeros((1, ACTOR_OBS_DIM), dtype=np.float32)]
    observations += [(rng.standard_normal((1, ACTOR_OBS_DIM)) * scale).astype(np.float32)
                     for scale in scales]
    errors = []
    with torch.inference_mode():
        for index, observation in enumerate(observations):
            tensor = torch.from_numpy(observation)
            validate_observation(tensor)
            reference = actor(tensor).detach().cpu().numpy()
            actual = session.run(["actions"], {"obs_history": observation})[0]
            for label, value in (("Torch", reference), ("ONNXRuntime", actual)):
                _require(value.shape == (1, ACTION_DIM) and value.dtype == np.float32,
                         f"{label} sample {index}: expected float32[1,6]")
                _require(bool(np.isfinite(value).all()), f"{label} sample {index}: non-finite actions")
            difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
            errors.append({
                "sample": index,
                "max_abs_error": float(difference.max()),
                "max_relative_error": float((difference / np.maximum(np.abs(reference), ATOL)).max()),
            })
            _require(bool(np.allclose(actual, reference, atol=ATOL, rtol=RTOL)),
                     f"ONNXRuntime mismatch at sample {index}: {errors[-1]}")
    return {
        "passed": True, "provider": "CPUExecutionProvider", "sample_count": len(observations),
        "seed": VALIDATION_SEED, "inputs": "one zero + eight NumPy PCG64 standard-normal samples",
        "random_sample_scales": scales, "atol": ATOL, "rtol": RTOL,
        "relative_error_denominator_floor": ATOL,
        "max_abs_error": max(item["max_abs_error"] for item in errors),
        "max_relative_error": max(item["max_relative_error"] for item in errors), "samples": errors,
    }


def sidecar_path(output: str | Path) -> Path:
    return Path(str(output) + ".json")


def export_checkpoint(checkpoint: str | Path, run_manifest: str | Path, output: str | Path) -> dict:
    """Validate, export, verify, then publish model + sidecar without overwriting either."""
    output = Path(output)
    sidecar = sidecar_path(output)
    _require(output.suffix == ".onnx", "output must have the .onnx suffix")
    for path in (output, sidecar):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    actor, provenance = load_actor_checkpoint(checkpoint, run_manifest)
    np, onnx, ort = _onnx_dependencies()
    manifest = provenance["run_manifest"]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Staging on the same filesystem permits atomic no-replace publication with link().
    with tempfile.TemporaryDirectory(prefix=".v40-export-", dir=output.parent) as staging_dir:
        staged_model = Path(staging_dir) / "actor.onnx"
        staged_sidecar = Path(staging_dir) / "actor.onnx.json"
        dummy = torch.zeros(1, ACTOR_OBS_DIM, dtype=torch.float32)
        validate_observation(dummy)
        with torch.inference_mode():
            torch.onnx.export(
                actor, (dummy,), str(staged_model), input_names=["obs_history"], output_names=["actions"],
                opset_version=OPSET, dynamo=False, dynamic_axes=None, external_data=False,
                export_params=True, do_constant_folding=True,
            )
        validation = verify_onnx(actor, staged_model)
        report = {
            "schema_version": 1, **{key: manifest[key] for key in _METADATA_KEYS}, **provenance,
            "onnx_sha256": sha256_file(staged_model), "onnx_filename": output.name,
            "policy": manifest["policy"], "opset": OPSET, "dynamo": False,
            "input": {"name": "obs_history", "dtype": "float32", "shape": [1, ACTOR_OBS_DIM]},
            "output": {"name": "actions", "dtype": "float32", "shape": [1, ACTION_DIM]},
            "actor_obs_dim": ACTOR_OBS_DIM, "critic_obs_dim": CRITIC_OBS_DIM, "action_dim": ACTION_DIM,
            "action_order": list(ACTION_ORDER), "action_joint_names": list(JOINT_NAMES),
            "preprocessing": {
                "location": "external: environment / deployment, never embedded in this actor",
                "physical_observation_dim": 25, "history_frames": 5,
                "history_order": "oldest to newest; current frame is the last 25 values",
                "physical_observation_layout": [
                    "body angular velocity 3", "projected gravity 3", "commands [vx,wz,height] 3",
                    "four leg joint position offsets 4", "six joint velocities 6", "previous policy output 6",
                ],
                "empirical_normalization": False,
                "external_responsibilities": [
                    "Use the exact contract identified by contract_sha256 and the matching assets.",
                    "Apply contract coordinate transforms, units, fixed scaling/clipping and history reset rules.",
                    "Reject non-finite observations/actions and enforce fixed float32 shapes at the caller.",
                    "Map raw mean actions to joint targets, clipping, limits and actuators per contract.",
                ],
                "critic_training_only": "current physical 25 + privileged base linear velocity 3 + height 1",
                "action_semantics": "raw deterministic actor mean; no sampling, output clamp or actuator scaling",
            },
            "validation": validation,
            "versions": {"torch": str(torch.__version__), "onnx": onnx.__version__,
                         "onnxruntime": ort.__version__, "numpy": np.__version__},
            "limitations": [
                "Synthetic numerical parity only; no training, physics rollout or policy-quality claim.",
                "Contract/asset hashes are compared with checkpoint infos; source assets are not re-audited.",
                "State tensors cannot prove training library version or parameter-free activation choice; trust the run manifest.",
                "Both files and their matching hashes are required; publication of the pair is not a single transaction.",
            ],
        }
        staged_sidecar.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                  encoding="utf-8")
        os.link(staged_model, output)  # Atomic EEXIST even if another exporter raced us.
        try:
            os.link(staged_sidecar, sidecar)
        except BaseException:
            # Roll back only our model, never a concurrent writer's replacement.
            if output.is_file() and os.path.samefile(output, staged_model):
                output.unlink()
            raise
    return report
