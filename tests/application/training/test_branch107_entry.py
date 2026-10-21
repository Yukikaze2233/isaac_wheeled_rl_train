"""Check the explicit same-foundation migration and bound model metadata."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import onnxruntime as ort
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from wheeled_algo.leg_winding_export import export_actor
from wheeled_tasks.chassis.application.configuration.training_profile import (
    profile_source_files, restore_profile_clocks, validate_formal_learning_state_resume,
)
from wheeled_tasks.chassis.control.pd import V5Control

ROOT = Path(__file__).resolve().parents[3]


def contracts():
    directory = ROOT / "configs/v7_formal_20261012"
    return tuple(json.loads((directory / name).read_text()) for name in (
        "flat_tgs_continuation_055.json", "flat_tgs_continuation_055.json"))








def test_branch107_graph_and_sidecar_bind_geometry_without_changing_actor(tmp_path):
    observations = torch.zeros(3, 35)
    actor = MLPModel(TensorDict({"policy": observations}, batch_size=[3]),
                     {"actor": ["policy"]}, "actor", 6, hidden_dims=[8],
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": .2})
    expected = actor(TensorDict({"policy": observations}, batch_size=[3])).detach().numpy()
    report = export_actor(actor, tmp_path, {"leg_winding_version": "paired_branch107_v1",
                                          "hardware_deployment_ready": False}, observations)
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata["leg_winding_version"] == "paired_branch107_v1"
    assert metadata["leg_target_model_spec_sha256"] == V5Control.BRANCH107_MODEL_SPEC_SHA256
    assert session.run(["actions"], {"obs": observations.numpy()})[0] == pytest.approx(expected, abs=1e-6)
    assert report["onnx_sha256"] == hashlib.sha256((tmp_path / "policy.onnx").read_bytes()).hexdigest()
    sidecar = json.loads((tmp_path / "policy.onnx.json").read_text())
    assert sidecar["leg_target_model_spec_sha256"] == V5Control.BRANCH107_MODEL_SPEC_SHA256
    spec = importlib.util.spec_from_file_location("branch107_export_entry", ROOT / "src/wheeled_tasks/chassis/application/export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.leg_winding_version(contracts()[1]) == "paired_branch107_v1"
    invalid = contracts()[1]
    invalid["formal_v7_flat"] = False
    with pytest.raises(ValueError, match="explicit V7"):
        module.leg_winding_version(invalid)
