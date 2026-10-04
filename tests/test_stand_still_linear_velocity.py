"""Optional SCUT body-velocity density preserves the existing dense objectives."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.rewards import (
    DENSE_TRACKING_V1, apply_dense_tracking, validate_dense_tracking,
)


def configuration():
    return deepcopy(DENSE_TRACKING_V1) | {"stand_still_lin_vel": {
        "version": "scut-flat-linear-velocity-v1", "weight": -2.,
        "command_threshold_m_s": .1,
    }}


def scene(count):
    zero3 = torch.zeros(count, 3, dtype=torch.float64)
    gravity = zero3.clone()
    gravity[:, 2] = -1.
    commands = zero3.clone()
    commands[:, 2] = .305
    return dict(velocity=zero3.clone(), omega=zero3.clone(), gravity=gravity,
                height=commands[:, 2].clone(), commands=commands,
                support=torch.ones(count, dtype=torch.float64),
                ordinary=torch.ones(count, dtype=torch.bool), reference=None)


def evaluate(data, settings, *, spin_error=None):
    terms = {"motor_torque": torch.full_like(data["height"], -.25),
             "action_rate": torch.full_like(data["height"], -.1)}
    apply_dense_tracking(terms, **data, settings=settings, spin_velocity_error=spin_error)
    return terms


def test_optional_configuration_is_explicit_and_leaves_legacy_default_unchanged():
    assert "stand_still_lin_vel" not in DENSE_TRACKING_V1
    legacy = deepcopy(DENSE_TRACKING_V1)
    assert validate_dense_tracking(legacy) is legacy
    configured = configuration()
    assert validate_dense_tracking(configured) is configured
    assert "stand_still_lin_vel" not in evaluate(scene(1), legacy)


@pytest.mark.parametrize("cost", [None, False, [], {},
    {"version": "scut-flat-linear-velocity-v1", "weight": -2.},
    {"version": "scut-flat-linear-velocity-v1", "weight": -2.,
     "command_threshold_m_s": .1, "support_gate": True},
])
def test_missing_or_extra_optional_configuration_is_rejected(cost):
    settings = configuration()
    settings["stand_still_lin_vel"] = cost
    with pytest.raises(ValueError, match="stand_still_lin_vel"):
        validate_dense_tracking(settings)


@pytest.mark.parametrize("key,value", [
    ("version", "guessed"), ("version", None),
    ("weight", 0.), ("weight", 2.), ("weight", float("nan")),
    ("weight", float("inf")), ("weight", float("-inf")), ("weight", True), ("weight", "-2"),
    ("command_threshold_m_s", 0.), ("command_threshold_m_s", -.1),
    ("command_threshold_m_s", float("nan")), ("command_threshold_m_s", float("inf")),
    ("command_threshold_m_s", True), ("command_threshold_m_s", ".1"),
])
def test_invalid_version_weight_or_threshold_is_rejected(key, value):
    settings = configuration()
    settings["stand_still_lin_vel"][key] = value
    with pytest.raises(ValueError, match="stand_still_lin_vel"):
        validate_dense_tracking(settings)


def test_linear_body_cost_is_symmetric_uses_both_axes_and_is_not_projected():
    data = scene(5)
    data["velocity"][:, :2] = torch.tensor([
        [.05, .03], [-.05, -.03], [.05, -.03], [0., 0.], [.5, .3],
    ], dtype=torch.float64)
    data["gravity"][0] = torch.tensor([1., 0., 0.], dtype=torch.float64)
    cost = evaluate(data, configuration())["stand_still_lin_vel"]
    torch.testing.assert_close(cost, torch.tensor([-.16, -.16, -.16, 0., -1.6], dtype=torch.float64))


def test_threshold_is_strict_and_symmetric_in_command_velocity():
    data = scene(7)
    data["commands"][:, 0] = torch.tensor(
        [0., .099999, -.099999, .1, -.1, .100001, -.100001], dtype=torch.float64)
    data["velocity"][:, 0] = .05
    cost = evaluate(data, configuration())["stand_still_lin_vel"]
    torch.testing.assert_close(cost, torch.tensor([-.1, -.1, -.1, 0., 0., 0., 0.], dtype=torch.float64))


def test_configured_weight_and_threshold_are_used_without_changing_commands():
    settings = configuration()
    settings["stand_still_lin_vel"].update(weight=-3., command_threshold_m_s=.2)
    validate_dense_tracking(settings)
    data = scene(2)
    data["commands"][:, 0] = torch.tensor([.15, -.2], dtype=torch.float64)
    original = data["commands"].clone()
    data["velocity"][:, 0] = .05
    cost = evaluate(data, settings)["stand_still_lin_vel"]
    torch.testing.assert_close(cost, torch.tensor([-.15, 0.], dtype=torch.float64))
    assert torch.equal(data["commands"], original)


def test_rotation_and_unsupported_states_pay_but_active_jump_does_not():
    data = scene(4)
    data["velocity"][:, :2] = torch.tensor([.05, .03], dtype=torch.float64)
    data["commands"][:, 1] = torch.tensor([0., 8., -8., 0.])
    data["omega"][:, 2] = data["commands"][:, 1]
    data["support"][:] = 0.
    data["ordinary"][:] = False
    data["reference"] = SimpleNamespace(jumping=torch.tensor([False, False, False, True]))
    cost = evaluate(data, configuration())["stand_still_lin_vel"]
    torch.testing.assert_close(cost, torch.tensor([-.16, -.16, -.16, 0.], dtype=torch.float64))


def test_additive_cost_preserves_all_old_terms_and_small_commands_bit_for_bit():
    data = scene(4)
    data["commands"][:, 0] = torch.tensor([.05, -.05, 0., .5], dtype=torch.float64)
    data["commands"][:, 1] = torch.tensor([0., .15, 8., -1.], dtype=torch.float64)
    data["velocity"][:, 0] = data["commands"][:, 0]
    data["velocity"][:, 1] = .02
    data["height"] += .01
    data["ordinary"][2] = False
    data["support"][1] = .5
    spin_error = torch.tensor([0., .1, .3, .5], dtype=torch.float64)
    originals = {name: value.clone() for name, value in data.items() if isinstance(value, torch.Tensor)}
    before = evaluate(data, DENSE_TRACKING_V1, spin_error=spin_error)
    after = evaluate(data, configuration(), spin_error=spin_error)
    assert set(after) == set(before) | {"stand_still_lin_vel"}
    for name, value in before.items():
        assert torch.equal(after[name], value), name
    for name, value in originals.items():
        assert torch.equal(data[name], value), name
    torch.testing.assert_close(after["stand_still_lin_vel"],
                               torch.tensor([-.14, -.14, -.04, 0.], dtype=torch.float64))
    assert after["dense_stationary_broad"][0] < 0.
    assert after["dense_stationary_precision"][0] < 0.


def test_raw_density_is_integrated_once_by_the_caller_policy_clock():
    data = scene(1)
    data["velocity"][0, 0] = .05
    density = evaluate(data, configuration())["stand_still_lin_vel"]
    assert density.item() == pytest.approx(-.1)  # reward/s, not -.002 per 50 Hz step
    assert (density * .02).item() == pytest.approx(-.002)
    torch.testing.assert_close((density.expand(50) * .02).sum(),
                               (density.expand(100) * .01).sum())


def configuration_v2():
    settings = configuration()
    settings["stand_still_lin_vel"].update(version="scut-flat-linear-velocity-v2",
                                          exclude_world_reference_motion=True)
    return settings


def test_world_reference_exclusion_requires_explicit_versioned_configuration():
    assert validate_dense_tracking(configuration_v2()) == configuration_v2()
    settings = configuration_v2()
    del settings["stand_still_lin_vel"]["exclude_world_reference_motion"]
    with pytest.raises(ValueError, match="stand_still_lin_vel"):
        validate_dense_tracking(settings)
    settings = configuration()
    settings["stand_still_lin_vel"]["exclude_world_reference_motion"] = True
    with pytest.raises(ValueError, match="stand_still_lin_vel"):
        validate_dense_tracking(settings)


@pytest.mark.parametrize("value", [False, None, 0, 1, "true"])
def test_v2_world_reference_exclusion_cannot_be_silently_disabled(value):
    settings = configuration_v2()
    settings["stand_still_lin_vel"]["exclude_world_reference_motion"] = value
    with pytest.raises(ValueError, match="exclude_world_reference_motion=true"):
        validate_dense_tracking(settings)


def test_v2_excludes_legal_world_translation_but_not_ordinary_yaw():
    data = scene(4)
    # A fixed nonzero world velocity has vx_body=0 at one rotation phase; its
    # lateral component is motion to track, not unintended stationary drift.
    data["commands"][:, 1] = 8.
    data["velocity"][:, :2] = torch.tensor([0., .5], dtype=torch.float64)
    data["ordinary"][:] = torch.tensor([True, False, True, False])
    data["support"][:] = 0.
    data["reference"] = SimpleNamespace(jumping=torch.tensor([False, False, True, True]))
    before = evaluate(data, configuration(), spin_error=torch.zeros(4))
    after = evaluate(data, configuration_v2(), spin_error=torch.zeros(4))
    torch.testing.assert_close(before["stand_still_lin_vel"],
                               torch.tensor([-1., -1., 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(after["stand_still_lin_vel"],
                               torch.tensor([-1., 0., 0., 0.], dtype=torch.float64))
    for key, value in before.items():
        if key != "stand_still_lin_vel":
            assert torch.equal(after[key], value), key


def test_v2_preserves_v1_cost_for_all_ordinary_commands_and_signed_thresholds():
    data = scene(5)
    data["commands"][:, 0] = torch.tensor([0., .0999, -.0999, .1, -.1], dtype=torch.float64)
    data["commands"][:, 1] = torch.tensor([0., 3., -3., 0., 0.])
    data["velocity"][:, :2] = torch.tensor([.05, -.03], dtype=torch.float64)
    before = evaluate(data, configuration())
    after = evaluate(data, configuration_v2())
    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(after[key], value), key
