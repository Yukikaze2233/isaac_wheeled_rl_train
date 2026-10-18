"""Explicit reset velocity frames preserve the existing pure geometry operation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.flat_birth_placement import PhysicalFlatPlacement


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "reports/v7_training_review_20261011/mechanical_stop_candidates_20261011/stop107_reference_v2/asset"


@pytest.fixture(scope="module")
def model():
    manifest = json.loads((ASSET / "manifest.json").read_text())
    spec = json.loads((ASSET / "model_spec.json").read_text())
    return PhysicalFlatPlacement(spec, manifest, ASSET, clearance_m=0.)


def request(model):
    manifest = json.loads((ASSET / "manifest.json").read_text())
    names = model.kinematics.tree_joint_names
    q = [manifest["nominal_joint_pos"][name] for name in names]
    return {
        "q18": torch.tensor([q, q], dtype=torch.float64),
        "source_root_position_m": torch.tensor([[.1, .2, .25], [-.1, -.2, .25]], dtype=torch.float64),
        "source_root_rpy_rad": torch.tensor([[.15, -.1, 2.8], [-.1, .15, -2.7]], dtype=torch.float64),
        "source_root_velocity6": torch.zeros(2, 6, dtype=torch.float64),
        "origins": torch.tensor([[10., 20., 1.], [30., 40., 2.]], dtype=torch.float64),
        "reset_xy_world": torch.tensor([[12., 23.], [34., 45.]], dtype=torch.float64),
        "ground_z_world": torch.tensor([1., 2.], dtype=torch.float64),
    }


def test_nonzero_root_velocity_without_frame_keeps_existing_invalid_reason(model):
    values = request(model)
    values["source_root_velocity6"][0] = torch.tensor([1., 2., 3., .1, .2, .3])
    result = model.propose(**values)
    assert result.valid.tolist() == [False, True]
    assert result.reasons["nonzero_source_root_velocity"].tolist() == [True, False]
    assert torch.isnan(result.root_pose[0]).all()
    assert torch.isnan(result.root_velocity[0]).all()
    assert torch.isfinite(result.root_pose[1]).all()
    assert values["source_root_velocity6"][0].ne(0.).all()


def test_explicit_com_world_velocity_passes_unchanged_without_changing_geometry(model):
    values = request(model)
    baseline = model.propose(**values)
    velocity = torch.tensor([[1., 2., 3., .1, .2, .3], [-3., -2., -1., -.3, -.2, -.1]], dtype=torch.float64)
    values["source_root_velocity6"] = velocity.clone()
    before = deepcopy(values)
    state = torch.random.get_rng_state().clone()
    result = model.propose(**values, root_velocity_frame="com_world")
    assert result.valid.all()
    assert torch.equal(result.root_pose, baseline.root_pose)
    assert torch.equal(result.minimum_shape_clearance_m, baseline.minimum_shape_clearance_m)
    assert torch.equal(result.placement_raise_m, baseline.placement_raise_m)
    assert torch.equal(result.root_velocity, velocity)
    assert not any(bool(mask.any()) for mask in result.reasons.values())
    assert torch.equal(torch.random.get_rng_state(), state)
    for name, original in before.items():
        assert torch.equal(values[name], original)
    result.root_velocity.zero_()
    assert torch.equal(values["source_root_velocity6"], velocity)


@pytest.mark.parametrize("frame", [None, "com_world"])
def test_zero_legacy_results_are_bitwise_identical_with_explicit_supported_frame(model, frame):
    values = request(model)
    baseline = model.propose(**values)
    result = model.propose(**values, root_velocity_frame=frame)
    for actual, expected in zip(result[:-1], baseline[:-1]):
        assert torch.equal(actual, expected)
    assert result.reasons.keys() == baseline.reasons.keys()
    for key in result.reasons:
        assert torch.equal(result.reasons[key], baseline.reasons[key])


@pytest.mark.parametrize("frame", ["world", "body", "com_body", "link_world", "", 1, True, {}])
def test_unknown_frame_always_rejected_even_for_zero_velocity(model, frame):
    with pytest.raises(ValueError, match="root_velocity_frame must be None or com_world"):
        model.propose(**request(model), root_velocity_frame=frame)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_explicit_frame_nonfinite_velocity_invalidates_only_affected_lane(model, value):
    values = request(model)
    values["source_root_velocity6"][0, 2] = value
    values["source_root_velocity6"][1, 5] = .4
    result = model.propose(**values, root_velocity_frame="com_world")
    assert result.valid.tolist() == [False, True]
    assert result.reasons["nonfinite_input"].tolist() == [True, False]
    assert not result.reasons["nonzero_source_root_velocity"].any()
    assert torch.isnan(result.root_pose[0]).all()
    assert torch.isnan(result.root_velocity[0]).all()
    assert torch.isfinite(result.root_pose[1]).all()
    assert torch.equal(result.root_velocity[1], values["source_root_velocity6"][1])
