#!/usr/bin/env python3
"""CPU-only numerical export check; does not evaluate physical skills."""
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    torch.set_num_threads(1)
    contract = json.loads((ROOT / "contract.json").read_text())
    seal = json.loads((ROOT / "source.checkpoint.json").read_text())
    agent = json.loads((ROOT / "agent_config.json").read_text())
    saved = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True)
    assert sha(ROOT / "model.pt") == seal["checkpoint_sha256"]
    assert sha(ROOT / "contract.json") == sha(ROOT / "policy.onnx.contract.json") == seal["contract_sha256"]
    assert saved["infos"]["contract_sha256"] == seal["contract_sha256"]
    assert (contract["actor_dim"], contract["critic_dim"], contract["action_dim"], contract["history_length"]) == (35, 81, 6, 1)
    assert agent["actor"]["obs_normalization"] is False and not agent["empirical_normalization"]
    config = deepcopy(agent["actor"])
    assert config.pop("class_name") == "MLPModel"
    actor = MLPModel(TensorDict({"policy": torch.zeros(1, 35)}, batch_size=[1]),
                     agent["obs_groups"], "actor", 6, **config).cpu().eval()
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    assert all(torch.equal(value, saved["actor_state_dict"][name]) for name, value in actor.state_dict().items())

    vectors = json.loads((ROOT / "deployment/abi_test_vectors.json").read_text())
    rows = vectors["vectors"]
    observations = torch.tensor([row["obs"] for row in rows], dtype=torch.float32)
    sensor = module("candidate_sensor35", ROOT / "deployment/scut_observation.py")
    reference_source = module("candidate_command_reference", ROOT / "deployment/references.py")
    count = len(rows)
    commands = torch.tensor([row["commands_vx_yaw_height"] for row in rows])
    requests = torch.tensor([row["jump_request"] for row in rows])
    clocks = torch.tensor([row["command_elapsed_s"] for row in rows])
    apex = torch.tensor([row["jump_apex_m"] for row in rows])
    reference = reference_source.CommandReference(count, "cpu", contract["command_reference"])
    reference.update(commands[:, 2], torch.zeros(count), torch.zeros(count), torch.zeros(count),
                     torch.full((count,), 7.), requests, clocks, apex)
    rebuilt = sensor.build_manual35(
        torch.tensor([row["gyro_b_rad_s"] for row in rows]),
        torch.tensor([row["gravity_b"] for row in rows]), commands,
        torch.tensor([row["motor_q_control_rad"] for row in rows]),
        torch.tensor([row["motor_dq_control_rad_s"] for row in rows]),
        torch.tensor([row["previous_clipped_action_control"] for row in rows]),
        torch.tensor(vectors["nominal_position_control_rad"]), requests, apex, reference)
    torch.testing.assert_close(rebuilt, observations, atol=0., rtol=0.)
    assert reference.phase.tolist() == [row["command_reference_phase"] for row in rows]
    assert tuple(sensor.POLICY_FROM_CONTROL) == (0, 1, 3, 4, 2, 5)

    onnx.checker.check_model(onnx.load(str(ROOT / "policy.onnx")))
    session = ort.InferenceSession(str(ROOT / "policy.onnx"), providers=["CPUExecutionProvider"])
    assert [(value.name, value.shape, value.type) for value in session.get_inputs()] == [("obs", ["batch", 35], "tensor(float)")]
    assert [(value.name, value.shape, value.type) for value in session.get_outputs()] == [("actions", ["batch", 6], "tensor(float)")]
    generator = torch.Generator().manual_seed(261008)
    synthetic = torch.randn(256, 35, generator=generator) * .25
    synthetic[:, 28:] = torch.tensor([1., 0., 0., 0., 0., 0., 0.])
    batches = (("zero_batch1", torch.zeros(1, 35)), ("abi_batch1", observations[:1]),
               ("abi_batch8", observations[:8]), ("abi_batch25", observations),
               ("synthetic_numeric_batch256", synthetic))
    results = []
    for name, batch in batches:
        with torch.no_grad():
            native = actor(TensorDict({"policy": batch}, batch_size=[len(batch)]), stochastic_output=False).numpy()
        actual = session.run(["actions"], {"obs": batch.numpy()})[0]
        assert actual.shape == (len(batch), 6) and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, native, atol=1e-5, rtol=1e-5)
        results.append({"name": name, "observations": len(batch), "input_shape": list(batch.shape),
                        "output_shape": list(actual.shape), "max_abs_error": float(np.abs(actual - native).max())})
        if name == "abi_batch25":
            np.testing.assert_allclose(actual, np.asarray([row["actions"] for row in rows], dtype=np.float32), atol=0., rtol=0.)
    assert sha(ROOT / "model.pt") == seal["checkpoint_sha256"]
    report = {"schema": "rec38000-independent-numerical-parity-v1", "passed": True,
              "model_sha256": sha(ROOT / "model.pt"), "onnx_sha256": sha(ROOT / "policy.onnx"),
              "contract_sha256": sha(ROOT / "contract.json"), "actor_factory": "rsl_rl.models.MLPModel",
              "native_actor_state_loaded_strict": True, "actor_state_equal_to_original": True,
              "deterministic_actor_mean": True, "empirical_normalization": False,
              "abi_sensor_reconstruction_exact": True, "abi_vector_count": count,
              "atol": 1e-5, "rtol": 1e-5, "batches": results,
              "max_abs_error": max(row["max_abs_error"] for row in results),
              "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                           "onnxruntime": ort.__version__, "rsl_rl": importlib.metadata.version("rsl-rl-lib")},
              "provider": "CPUExecutionProvider", "qualified": None, "optimizer_steps": 0,
              "native_scene_started": False,
              "scope": "Numerical deterministic-mean export and frozen observation ABI only; synthetic vectors are not physical scenarios or learned-skill evidence."}
    (ROOT / "numerical_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
