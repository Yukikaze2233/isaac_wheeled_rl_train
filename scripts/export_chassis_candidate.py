#!/usr/bin/env python3
"""Export a sealed chassis actor and deployment ABI without Isaac or a GPU scene."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def authenticate(checkpoint):
    """Require the existing immutable checkpoint protocol and native actor config."""
    import torch
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count

    checkpoint = checkpoint.resolve(strict=True)
    before = digest(checkpoint)
    updates = checkpoint_update_count(checkpoint)
    contract_path = checkpoint_contract_path(checkpoint)
    contract_sha = digest(contract_path)
    contract = json.loads(contract_path.read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(digest(checkpoint) == before, "Checkpoint changed during safe load")
    require(all(name in saved for name in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "infos")),
            "Candidate requires a full learning checkpoint")
    for name, expected in (("contract_sha256", contract_sha),
            ("asset_manifest_sha256", contract["asset_manifest_sha256"]),
            ("control_math_sha256", contract["control_math_sha256"])):
        require(saved["infos"].get(name) == expected, f"Checkpoint {name} does not authenticate its contract")
    require((contract["actor_dim"], contract["critic_dim"], contract["action_dim"], contract["history_length"])
            == (35, 81, 6, 1), "Exporter requires the explicit single-frame 35/81/6 interface")
    require(contract.get("manual_context35") is True, "Exporter requires the frozen manual35 command ABI")
    config_path = next((parent / "agent_config.json" for parent in checkpoint.parents
                        if (parent / "agent_config.json").is_file()), None)
    require(config_path is not None, "Native saved agent_config.json is required")
    agent = json.loads(config_path.read_text())
    actor_cfg = agent["actor"]
    architecture = contract["policy_architecture"]
    require(actor_cfg.get("class_name") == "MLPModel", "Only the native feed-forward MLPModel is supported")
    require(actor_cfg.get("obs_normalization") is False and not agent.get("empirical_normalization"),
            "This ABI requires exactly disabled observation normalization")
    require(actor_cfg.get("hidden_dims") == architecture["actor_hidden_dims"]
            and actor_cfg.get("activation") == architecture["activation"],
            "Saved actor architecture differs from the frozen contract")
    require(actor_cfg.get("distribution_cfg", {}).get("class_name") == "GaussianDistribution",
            "Only the native Gaussian deterministic mean is supported")
    require(agent.get("obs_groups", {}).get("actor") == ["policy"], "Actor observation groups changed")
    manifest_path = ROOT / contract["asset_directory"] / "manifest.json"
    control_path = ROOT / contract["control_math_source"]
    require(digest(manifest_path) == contract["asset_manifest_sha256"], "Asset manifest differs from the frozen contract")
    require(digest(control_path) == contract["control_math_sha256"], "Control identity differs from the frozen contract")
    return checkpoint, saved, updates, contract_path, contract, config_path, agent, manifest_path, control_path


def load_actor(saved, contract, agent):
    """Construct the installed stock RSL actor factory directly, without a runner."""
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    config = deepcopy(agent["actor"])
    config.pop("class_name")
    observations = TensorDict({"policy": torch.zeros(1, contract["actor_dim"])}, batch_size=[1])
    actor = MLPModel(observations, agent["obs_groups"], "actor", contract["action_dim"], **config).cpu().eval()
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    require(all(torch.isfinite(value).all().item() for value in actor.state_dict().values()), "Nonfinite actor state")
    return actor


def abi_vectors(contract, manifest):
    """Known sensor snapshots cover ordinary motion and command-clock jump phases."""
    import torch
    from wheeled_tasks.chassis.references import CommandReference
    from wheeled_tasks.chassis.scut_observation import build_manual35
    from wheeled_tasks.chassis.v5_control import V5Control

    cases = [(False, height, vx, yaw, 0., .06)
             for height, vx, yaw in ((.23, 0., 0.), (.25, .5, 0.), (.305, -.5, 0.),
                 (.43, 0., 1.), (.305, 0., -1.), (.305, .5, 1.))]
    cases += [(True, .305, vx, 0., clock, apex)
              for vx, apex in ((0., .03), (0., .10), (.6, .06))
              for clock in (0., .2, .35, .5, .8, 1.2)]
    cases.append((False, .305, .3, -.5, 0., .06))
    if contract.get("task_campaign", {}).get("domain") == "speed":
        cases += [(False, height, vx, yaw, 0., .06)
                  for height, vx, yaw in ((.23, 1., 0.), (.43, -1., 0.),
                      (.305, 3., 0.), (.305, -3., 0.), (.305, 5., 0.), (.305, -5., 0.),
                      (.305, 0., 8.), (.305, 0., -8.), (.305, 1., 2.), (.305, -1., -2.))]
    count = len(cases)
    nominal = torch.tensor([manifest["nominal_joint_pos"][name] for name in V5Control.ACTIVE])
    commands = torch.tensor([[vx, yaw, height] for jump, height, vx, yaw, clock, apex in cases])
    request = torch.tensor([case[0] for case in cases])
    elapsed = torch.tensor([case[4] for case in cases])
    apex = torch.tensor([case[5] for case in cases])
    generator = torch.Generator().manual_seed(260105)
    omega = torch.randn(count, 3, generator=generator) * .2
    gravity = torch.tensor([[0., 0., -1.]]).repeat(count, 1)
    q = nominal + torch.randn(count, 6, generator=generator) * .10
    dq = torch.randn(count, 6, generator=generator) * 2.
    previous = torch.randn(count, 6, generator=generator) * .10
    reference = CommandReference(count, "cpu", contract["command_reference"])
    reference.update(commands[:, 2], torch.zeros(count), torch.zeros(count), torch.zeros(count),
        torch.full((count,), 7.), request, elapsed, apex)
    observations = build_manual35(omega, gravity, commands, q, dq, previous, nominal, request, apex, reference)
    rows = [{"name": f"abi_{index:02d}", "jump_request": bool(case[0]),
        "commands_vx_yaw_height": commands[index].tolist(), "jump_apex_m": case[5],
        "command_elapsed_s": case[4], "command_reference_phase": int(reference.phase[index]),
        "gyro_b_rad_s": omega[index].tolist(), "gravity_b": gravity[index].tolist(),
        "motor_q_control_rad": q[index].tolist(), "motor_dq_control_rad_s": dq[index].tolist(),
        "previous_clipped_action_control": previous[index].tolist(),
        "obs": observations[index].tolist()} for index, case in enumerate(cases)]
    return observations, rows


def export_candidate(checkpoint, output):
    import importlib.metadata
    import inspect
    import numpy as np
    import onnxruntime as ort
    import torch
    from tensordict import TensorDict
    from wheeled_algo.chassis_export import export_actor

    require(not output.exists(), "Output directory already exists; candidate artifacts are immutable")
    checkpoint, saved, updates, contract_path, contract, config_path, agent, manifest_path, control_path = authenticate(checkpoint)
    actor = load_actor(saved, contract, agent)
    observations, vectors = abi_vectors(contract, json.loads(manifest_path.read_text()))
    with torch.no_grad():
        native = actor(TensorDict({"policy": observations}, batch_size=[len(observations)]), stochastic_output=False)
        wrapped = actor.as_onnx(verbose=False)(observations)
    torch.testing.assert_close(native, wrapped, atol=0., rtol=0.)
    output.mkdir(parents=True, exist_ok=False)
    domain = contract.get("task_campaign", {}).get("domain", "chassis")
    info_keys = ("contract_id", "contract_sha256", "asset_manifest_sha256", "control_math_sha256",
        "stage", "successful_updates_total", "learning_lineage_updates", "critic_warmup_remaining",
        "stage_actor_update", "global_actor_update_offset", "global_actor_update", "actor_updates_consumed",
        "task_actor_update", "confirmed_jump_bins_m", "batch_transitions", "training_transitions")
    identity = {"role": f"{domain}_research_candidate_unassessed", "qualified": None,
        "hardware_deployment_ready": False, "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": digest(checkpoint), "learning_updates": updates,
        "consumed_updates": saved["infos"].get("actor_updates_consumed"),
        "contract_id": contract["contract_id"], "contract_sha256": digest(contract_path),
        "asset_manifest_sha256": contract["asset_manifest_sha256"],
        "control_math_sha256": contract["control_math_sha256"],
        "checkpoint_infos": {key: saved["infos"][key] for key in info_keys if key in saved["infos"]},
        "native_actor_factory": "rsl_rl.models.MLPModel",
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
        "scope": "Numerical deterministic actor export and command ABI; physical and hardware skills unassessed"}
    report = export_actor(actor, output, identity, observations)
    session = ort.InferenceSession(str(output / "policy.onnx"), providers=["CPUExecutionProvider"])
    actual = session.run(["actions"], {"obs": observations.numpy()})[0]
    np.testing.assert_allclose(actual, native.numpy(), atol=1e-5, rtol=1e-5)
    for row, action in zip(vectors, actual):
        row["actions"] = action.tolist()
    deployment = output / "deployment"
    deployment.mkdir()
    for source, name in ((checkpoint, "model.pt"), (contract_path, "contract.json"),
            (contract_path, "policy.onnx.contract.json"), (config_path, "agent_config.json")):
        shutil.copyfile(source, output / name)
    for filename in ("completion.json", "progress.json"):
        source = checkpoint.parent / filename
        if source.is_file():
            shutil.copyfile(source, output / filename)
    source_seal = checkpoint.with_suffix(".checkpoint.json")
    if source_seal.is_file():
        shutil.copyfile(source_seal, output / "source.checkpoint.json")
    write_json(output / "model.checkpoint.json", {"schema": "chassis-learning-checkpoint-v1", "sealed": True,
        "checkpoint_sha256": digest(output / "model.pt"), "contract_sha256": digest(contract_path),
        "learning_updates": updates, "source_checkpoint": str(checkpoint)})
    write_json(deployment / "abi_test_vectors.json", {"atol": 1e-5, "rtol": 1e-5,
        "policy_joint_order": ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        "nominal_position_control_rad": [json.loads(manifest_path.read_text())["nominal_joint_pos"][name]
            for name in ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")],
        "vectors": vectors})
    for source in (manifest_path, control_path, ROOT / "src/wheeled_tasks/chassis/scut_observation.py",
                   ROOT / "src/wheeled_tasks/chassis/references.py", ROOT / "src/wheeled_tasks/chassis/v5_control.py"):
        shutil.copyfile(source, deployment / source.name)
    write_json(deployment / "interface.json", {"schema": "wheel-leg-policy-handoff-v1", "identity": report,
        "hardware_deployment_ready": False, "domain_qualified": None,
        "input": {"name": "obs", "dtype": "float32", "shape": ["batch", 35],
            "history_length": 1, "normalization": "frozen contract scales; final clip +/-100",
            "observation_source": contract["actor_observation_source"], "layout": contract["actor_layout"],
            "command_context": "build_manual35 from frozen scut_observation.py and command-only references.py; never contact phase",
            "command_reference": contract["command_reference"]},
        "output": {"name": "actions", "shape": ["batch", 6], "deterministic": True,
            "policy_from_control": [0, 1, 3, 4, 2, 5], "control_from_policy": [0, 1, 4, 2, 3, 5],
            "control": contract["v5_control"]},
        "policy_hz": 1. / contract["policy_dt"], "simulated_feedback_pd_hz": 1. / contract["physics_dt"],
        "integration_limits": ["RMCS flat profile fixes model SHA and low command limits; update profile identity and limits for this candidate",
            "Ordinary flat motion uses [1,0,0,0,0,0,0] context; command-clock jump context is required only for jump requests, whose skill qualification is not established by export",
            "Hardware encoder/IMU/wheel directions and soft limits still require their existing validation",
            "Training and export do not establish deterministic jump or hardware qualification"]})
    sources = [Path(__file__).resolve(), ROOT / "src/wheeled_algo/chassis_export.py",
        ROOT / "src/wheeled_tasks/chassis/scut_observation.py", ROOT / "src/wheeled_tasks/chassis/references.py",
        Path(inspect.getfile(type(actor)))]
    write_json(output / "provenance.json", {**identity, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(path): digest(path) for path in sources}, "optimizer_steps": 0,
        "isaac_scene_started": False, "native_to_wrapper_max_abs_error": float((native - wrapped).abs().max()),
        "abi_vector_count": len(vectors), "abi_onnx_max_abs_error": float(np.max(np.abs(actual - native.numpy())))})
    (output / "README.md").write_text("# Chassis research candidate\n\n"
        "This sealed checkpoint is exported for integration and review. It is unassessed for physical skills and hardware deployment.\n\n"
        "Use policy.onnx with the frozen 35D command ABI in deployment/interface.json and compare abi_test_vectors.json. "
        "For flat motion the RMCS profile needs this model identity and its intended command limits; the ordinary context is [1,0,0,0,0,0,0]. "
        "Jump requests require the frozen command-clock context and are not qualified by this export.\n\n"
        "Validate package files with sha256sum -c SHA256SUMS.\n")
    files = {str(path.relative_to(output)): {"sha256": digest(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "bundle_manifest.json", {"schema": "chassis-candidate-bundle-v1", "files": files})
    (output / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.relative_to(output)}\n"
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"))
    require(digest(checkpoint) == identity["source_checkpoint_sha256"], "Sealed input changed during export")
    return {"output": str(output.resolve()), "onnx_sha256": report["onnx_sha256"], "qualified": None,
            "hardware_deployment_ready": False, "abi_vectors": len(vectors),
            "max_abs_error": float(np.max(np.abs(actual - native.numpy())))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = export_candidate(args.checkpoint, args.output)
    except Exception as error:
        parser.exit(1, f"Chassis candidate export rejected: {error}\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
