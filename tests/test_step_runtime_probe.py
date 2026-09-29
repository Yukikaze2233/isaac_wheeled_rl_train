"""CPU checks of the independent native-force comparison oracle."""
import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "scripts/probe_v6_step_runtime.py"
SPEC = importlib.util.spec_from_file_location("step_runtime_probe_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_no_contact_is_not_reported_as_a_validated_direction():
    values = torch.zeros(2, 4, 3)
    report = MODULE.compare_contact_vectors(values, values, values.sum(1))
    assert report == {"status": "not_observed", "observed_bodies": 0}


def test_nonzero_patch_vectors_must_agree_with_both_independent_views():
    values = torch.tensor([[[0., 0., 120.], [2., 0., 10.]], [[0., 0., 0.], [0., 0., 0.]]])
    report = MODULE.compare_contact_vectors(values, values.clone(), values.sum(1))
    assert report["status"] == "observed_consistent"
    assert report["observed_bodies"] == report["upward_samples"] == 1
    assert report["existing_view_error_max_n"] == 0.
    with pytest.raises(RuntimeError, match="native contact matrix"):
        MODULE.compare_contact_vectors(-values, values, values.sum(1))
    with pytest.raises(RuntimeError, match="all-body contact view"):
        MODULE.compare_contact_vectors(-values, -values, values.sum(1))


def test_missing_patch_evidence_cannot_hide_a_nonzero_existing_contact():
    values = torch.zeros(1, 2, 3)
    with pytest.raises(RuntimeError, match="all-body contact view"):
        MODULE.compare_contact_vectors(values, values, torch.tensor([[0., 0., 80.]]))


def test_wrong_plan_identity_rejected_before_materializing(tmp_path):
    with pytest.raises(ValueError, match="frozen SHA256"):
        MODULE.materialize(PATH.parents[1] / "contracts/v6_step_curriculum_training_v1.json", tmp_path, "0" * 64)
