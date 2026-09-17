"""Conservative restoring force, fit range, units and pressure interpolation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.gas_spring import GasSpringCurve


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "model/纯底盘_v5/gas_spring"


@pytest.fixture
def curve():
    return GasSpringCurve.from_json(DATA / "fit_10mpa.json")


def test_pressure_interpolation_and_fit_reproducibility(curve):
    spec = importlib.util.spec_from_file_location("fit_bkb", ROOT / "tools/fit_bkb_gas_spring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = json.loads((DATA / "catalogue_points.json").read_text())
    fit = module.fit_curves(data)
    np.testing.assert_allclose(fit["monomial_coefficients_n"], curve.coefficients_n, atol=1e-10)
    nine = np.array(fit["source_curve_fits"]["9"]["monomial_coefficients_n"])
    twelve = np.array(fit["source_curve_fits"]["12"]["monomial_coefficients_n"])
    np.testing.assert_allclose(curve.coefficients_n, (2 * nine + twelve) / 3)
    assert curve.force(0).item() == pytest.approx(280.)
    assert curve.force(torch.tensor(0.072, dtype=torch.float64)).item() == pytest.approx(418.7601, abs=1e-4)


def test_stored_energy_derivative_is_restoring_force(curve):
    compression = torch.linspace(0.001, 0.071, 100, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(curve.potential_energy(compression).sum(), compression)[0]
    torch.testing.assert_close(derivative, curve.force(compression), atol=1e-10, rtol=1e-10)
    force = curve.force(compression)
    assert (torch.diff(force) >= 0).all()
    assert curve.potential_energy(0).item() == 0
    # Extension coordinate q makes compression decrease: generalized spring effort is +F.
    q = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    energy = curve.potential_energy(0.20 - q)
    effort = -torch.autograd.grad(energy, q)[0]
    torch.testing.assert_close(effort, curve.force(0.20 - q))


@pytest.mark.parametrize("value", [-0.001, 0.073, float("nan"), float("inf")])
def test_force_never_silently_extrapolates(curve, value):
    with pytest.raises(ValueError, match="outside fitted range"):
        curve.force(value)


def test_mount_reference_is_explicit_not_catalogue_length(curve):
    with pytest.raises(TypeError):
        curve.force_from_pin_distance(0.16)
    a = curve.force_from_pin_distance(torch.tensor(0.16, dtype=torch.float64), extended_pin_distance_m=0.20)
    b = curve.force(torch.tensor(0.04, dtype=torch.float64))
    torch.testing.assert_close(a, b)


def test_does_not_claim_identified_dynamics():
    fit = json.loads((DATA / "fit_10mpa.json").read_text())
    assert not fit["physical_force_model_identified"]
    assert not fit["damping_identified"]
    assert not fit["friction_hysteresis_identified"]
    assert not fit["mounting_pin_distance_reference_identified"]


def test_v5_design_budget_and_proposed_interface_are_consistent():
    plan = json.loads((ROOT / "contracts/v5_full_training_plan.json").read_text())
    interface = plan["proposed_interface"]
    assert sum(interface["frame_fields"].values()) == interface["frame_dim"] == 46
    assert interface["frame_dim"] * interface["history"] == interface["actor_dim"] == 230
    assert sum(stage["updates"] for stage in plan["stages"]) == plan["total_updates"] == 100000
    assert plan["status"] == "design_not_executable_training_contract"
    assert plan["launched"] is False
