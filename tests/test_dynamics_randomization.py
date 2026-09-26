"""Physical property variations must retain an exact nominal comparison lane."""
import copy

import pytest
import torch

from wheeled_tasks.chassis.dynamics import RigidBodyRandomization


CONFIG = {
    "enabled": True, "sampling": "startup", "enabled_fraction": .5,
    "base_mass_scale": [.9, 1.3], "leg_mass_scale": [.9, 1.1], "wheel_mass_scale": [.9, 1.1],
    "inertia_scale": [.8, 1.2],
    "base_com_offset_m": {"x": [-.04, .04], "y": [-.02, .02], "z": [-.02, .02]},
}


def properties(count=128):
    mass = torch.tensor([[11.62, 2.622, .2, .2]]).repeat(count, 1)
    inertia = torch.tensor([[.3, .02, 0., .02, .4, .01, 0., .01, .5]]).repeat(count, 4, 1)
    com = torch.zeros(count, 4, 7)
    com[:, :, :3] = torch.tensor([.01, .02, -.03])
    com[:, :, 6] = 1.
    return mass, inertia, com, ["base_link", "leg", "L_link3", "R_link3"]


def test_nominal_lane_is_exact_and_does_not_consume_global_rng():
    data = properties()
    state = torch.get_rng_state().clone()
    sampled = RigidBodyRandomization(*data, CONFIG, 617)
    assert torch.equal(state, torch.get_rng_state())
    assert bool(sampled.enabled.any() & (~sampled.enabled).any())
    for actual, nominal in zip((sampled.masses, sampled.inertias, sampled.coms), data[:3]):
        assert torch.equal(actual[~sampled.enabled], nominal[~sampled.enabled])
    assert torch.equal(sampled.coms[:, :, 3:], data[2][:, :, 3:])
    assert torch.equal(sampled.coms[:, 1:], data[2][:, 1:])


def test_full_inertia_tensor_scales_with_mass_and_retains_positive_definiteness():
    data = properties()
    sampled = RigidBodyRandomization(*data, {**CONFIG, "enabled_fraction": 1.}, 617)
    torch.testing.assert_close(sampled.inertias, data[1] * (sampled.mass_scale * sampled.inertia_scale)[..., None])
    matrix = sampled.inertias.reshape(-1, 3, 3)
    torch.testing.assert_close(matrix, matrix.transpose(-1, -2))
    assert bool((torch.linalg.eigvalsh(matrix) > 0.).all())
    assert float(sampled.masses.sum(-1).min()) >= sampled.total_mass_bounds[0] - 1e-5
    assert float(sampled.masses.sum(-1).max()) <= sampled.total_mass_bounds[1] + 1e-5
    assert bool((sampled.com_offset.abs() <= torch.tensor([.04, .02, .02])).all())


def test_evaluation_is_nominal_unless_explicitly_profiled():
    data = properties(3)
    profiles = [None, {"base_mass_scale": 1.3, "base_com_offset_m": [.04, -.02, .02]},
                {"leg_mass_scale": .9, "wheel_mass_scale": 1.1, "inertia_scale": 1.2}]
    sampled = RigidBodyRandomization(*data, CONFIG, 617, profiles)
    assert sampled.enabled.tolist() == [False, True, True]
    assert torch.equal(sampled.masses[0], data[0][0])
    assert sampled.masses[1, 0].item() == pytest.approx(11.62 * 1.3)
    assert torch.equal(sampled.masses[1, 1:], data[0][1, 1:])
    torch.testing.assert_close(sampled.coms[1, 0, :3], data[2][1, 0, :3] + torch.tensor([.04, -.02, .02]))
    assert torch.equal(sampled.coms[2], data[2][2])


def test_sampling_is_repeatable_without_mutating_nominal_arrays():
    data = properties()
    originals = [tensor.clone() for tensor in data[:3]]
    first = RigidBodyRandomization(*data, CONFIG, 617)
    second = RigidBodyRandomization(*data, CONFIG, 617)
    assert first.summary()["sampled_properties_sha256"] == second.summary()["sampled_properties_sha256"]
    for original, actual in zip(originals, data[:3]):
        assert torch.equal(original, actual)
    different = RigidBodyRandomization(*data, CONFIG, 618)
    assert first.summary()["sampled_properties_sha256"] != different.summary()["sampled_properties_sha256"]


@pytest.mark.parametrize("override", [
    {"base_mass_scale": [0., 1.3]}, {"base_mass_scale": [1.1, 1.3]},
    {"inertia_scale": [-1., 1.]}, {"enabled_fraction": 1.1},
    {"base_com_offset_m": {"z": [.01, .02]}}, {"sampling": "per_step"},
    {"misspelled_scale": [1., 1.]},
])
def test_invalid_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        RigidBodyRandomization(*properties(), {**copy.deepcopy(CONFIG), **override}, 617)


def test_out_of_domain_evaluation_profile_is_rejected():
    with pytest.raises(ValueError):
        RigidBodyRandomization(*properties(1), CONFIG, 617, [{"base_mass_scale": 2.}])
