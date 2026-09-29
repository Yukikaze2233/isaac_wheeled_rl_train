"""The native probe's evidence oracle must reject sign and classification bugs."""
import importlib.util
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.step_climb import StepClimbTask


PATH = Path(__file__).resolve().parents[1] / "scripts/probe_v6_base_contact.py"
SPEC = importlib.util.spec_from_file_location("base_contact_probe_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def task():
    return StepClimbTask(["test"], {"test": {"body_climb": {
        "rises_m": [.2], "treads_m": [.8], "body_contact_min_force_n": 20.}}}, "cpu", .001)


def test_split_native_load_is_legal_even_when_each_patch_is_below_20n():
    points = torch.tensor([[.15, -.2, .2], [.55, -.2, .2], [.15, .2, .2], [.55, .2, .2]])
    forces = torch.tensor([[0., 0., 12.2625]]).repeat(4, 1)
    result = MODULE.classify_native_sample(task(), points, forces, forces.sum(0)[None, None])
    assert result["status"] == "observed" and result["patches"] == 4
    assert result["positive_patches_below_20n"] == 4
    assert result["upward_force_n"] == pytest.approx(49.05)


def test_native_oracle_does_not_accept_empty_sign_reversed_or_sidewall_evidence():
    t = task()
    zero = torch.empty(0, 3)
    assert MODULE.classify_native_sample(t, zero, zero, torch.zeros(1, 1, 3))["status"] == "not_observed"
    points = torch.tensor([[.3, 0., .2]])
    force = torch.tensor([[0., 0., 49.]])
    with pytest.raises(RuntimeError, match="direction"):
        MODULE.classify_native_sample(t, points, -force, force[None])
    with pytest.raises(RuntimeError, match="illegal"):
        MODULE.classify_native_sample(t, torch.tensor([[0., 0., .1]]), force, force[None])
