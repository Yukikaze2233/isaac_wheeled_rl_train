"""Actor-owned takeover costs; no script credit or renewable phase bonus."""
from copy import deepcopy
import math

import pytest
import torch

from wheeled_tasks.chassis.rewards import (
    DENSE_TRACKING_V1, RECOVERY_TAKEOVER_V1, apply_dense_tracking,
    apply_recovery_takeover, dense_error_terms, reward_terms,
    validate_recovery_takeover,
)


def configuration():
    return deepcopy(RECOVERY_TAKEOVER_V1) | {"dense_tracking": deepcopy(DENSE_TRACKING_V1)}


def scene(count=4):
    zero3, zero6 = torch.zeros(count, 3, dtype=torch.float64), torch.zeros(count, 6, dtype=torch.float64)
    gravity = zero3.clone()
    gravity[:, 2] = -1.
    command = zero3.clone()
    command[:, 2] = .305
    return dict(gravity=gravity, omega=zero3.clone(), height=command[:, 2].clone(),
                velocity=zero3.clone(), commands=command, dq=zero6.clone(),
                eligible_interval=torch.ones(count, dtype=torch.bool),
                motion_hold=torch.ones(count, dtype=torch.bool))


def ordinary_terms(data, support=None):
    n = len(data["height"])
    q = torch.zeros(n, 6, dtype=torch.float64)
    support = torch.ones(n, dtype=torch.float64) if support is None else support
    ordinary = torch.ones(n, dtype=torch.bool)
    terms = reward_terms(data["velocity"], data["omega"], data["gravity"], data["height"],
        data["commands"], data["dq"], q, q, q, q, q,
        torch.zeros(n, 2, 3, dtype=torch.float64), support, ordinary,
        torch.zeros(n, dtype=torch.bool))
    apply_dense_tracking(terms, data["velocity"], data["omega"], data["gravity"], data["height"],
        data["commands"], support, ordinary, None, DENSE_TRACKING_V1)
    return terms


def apply(data, *, support=None, settings=None):
    terms = ordinary_terms(data, support)
    result = apply_recovery_takeover(terms, **data, settings=settings or configuration())
    assert result is terms
    return result


def takeover_sum(terms, axis):
    return sum(terms[f"recovery_takeover_{axis}_{kind}"] for kind in ("broad", "precision"))


def test_versioned_configuration_is_explicit_and_reuses_current_dense_parameters():
    settings = configuration()
    assert validate_recovery_takeover(settings) is settings
    assert settings["zero_tracking_baselines"] is True
    for axis in ("height", "stationary"):
        assert settings["dense_tracking"][axis] == DENSE_TRACKING_V1[axis]


@pytest.mark.parametrize("key,value", [
    ("enabled", False), ("version", "guessed"), ("zero_tracking_baselines", 1),
    ("renewable_stability_bonus", 1.),
])
def test_invalid_or_unrecognized_configuration_is_rejected(key, value):
    settings = configuration()
    settings[key] = value
    with pytest.raises(ValueError, match="Recovery takeover"):
        validate_recovery_takeover(settings)


@pytest.mark.parametrize("axis", ["orientation", "angular_velocity", "hold_leg_velocity", "hold_wheel_velocity"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_cost_scales_and_weights_cannot_be_disabled_or_nonfinite(axis, value):
    settings = configuration()
    settings[axis]["fine_weight"] = value
    with pytest.raises(ValueError, match=axis):
        validate_recovery_takeover(settings)


@pytest.mark.parametrize("axis", ["orientation", "angular_velocity", "hold_leg_velocity", "hold_wheel_velocity"])
def test_low_error_precision_and_far_error_broad_cost_are_monotone(axis):
    settings = configuration()[axis]
    x = torch.tensor([0., .01, .1, 1., 3., 10.], dtype=torch.float64) * settings["broad_scale"]
    x.requires_grad_()
    score = sum(dense_error_terms(x, settings))
    assert score[0] == 0. and (score.diff() < 0.).all()
    derivative, = torch.autograd.grad(score.sum(), x)
    assert derivative[0] == 0. and (derivative[1:] < 0.).all()
    assert derivative[-1].abs() > .99 * settings["broad_weight"] / settings["broad_scale"]
    torch.testing.assert_close(score, sum(dense_error_terms(-x, settings)))


def test_upright_takeover_has_no_positive_density_or_changed_physical_cost():
    data = scene()
    before = ordinary_terms(data)
    assert sum(before.values()).eq(5.).all()
    terms = apply(data)
    assert all(term.eq(0.).all() for term in terms.values())
    assert sum(terms.values()).eq(0.).all()
    for name in ("motor_torque", "wheel_power", "leg_velocity", "horizontal_angular_velocity",
                 "action_rate", "no_fork", "undesired_contact"):
        assert torch.equal(terms[name], before[name])


def test_nonrecovery_script_and_blend_lanes_remain_bitwise_unchanged():
    data = scene()
    # Lane 0: ordinary; lane 1: script; lane 2: a policy interval containing a
    # blend tick; lane 3: a whole actor-owned interval, including a failed end.
    data["eligible_interval"][:] = torch.tensor([False, False, False, True])
    data["height"] += torch.tensor([.02, -.1, .04, -.03])
    data["velocity"][:, 0] = .08
    data["omega"][:] = .3
    data["dq"][:] = 3.
    before = ordinary_terms(data)
    original = {name: tensor.clone() for name, tensor in before.items()}
    apply_recovery_takeover(before, **data, settings=configuration())
    for name, old in original.items():
        assert torch.equal(before[name][:3], old[:3]), name
    for name, value in before.items():
        if name.startswith("recovery_takeover_"):
            assert value[:3].eq(0.).all(), name
    assert takeover_sum(before, "angular_velocity")[3] < 0.


def test_support_or_physical_phase_cannot_hide_takeover_height_and_static_drift():
    data = scene(2)
    data["height"] += .08
    data["velocity"][:, :2] = .08
    support = torch.tensor([1., 0.])
    before = ordinary_terms(data, support)
    assert before["dense_height_broad"][1] == before["dense_stationary_broad"][1] == 0.
    terms = apply(data, support=support)
    for axis in ("height", "stationary"):
        for kind in ("broad", "precision"):
            assert terms[f"dense_{axis}_{kind}"][0] == terms[f"dense_{axis}_{kind}"][1] < 0.


def test_reused_height_and_stationary_costs_follow_injected_dense_configuration():
    settings = configuration()
    settings["dense_tracking"]["height"]["broad_weight"] *= 2.
    settings["dense_tracking"]["stationary"]["fine_weight"] *= 3.
    validate_recovery_takeover(settings)
    data = scene(1)
    data["height"] += .02
    data["velocity"][:, 0] = .03
    actual = apply(data, support=torch.zeros(1), settings=settings)
    height = dense_error_terms(data["height"] - data["commands"][:, 2], settings["dense_tracking"]["height"])
    stationary = dense_error_terms(data["velocity"][:, 0], settings["dense_tracking"]["stationary"])
    torch.testing.assert_close(actual["dense_height_broad"], height[0])
    torch.testing.assert_close(actual["dense_stationary_precision"], stationary[1])


def test_zero_centering_retains_existing_orientation_error_dependence():
    data = scene(2)
    angle = torch.tensor([.05, .2])
    data["gravity"][:, 0] = angle.sin()
    data["gravity"][:, 2] = -angle.cos()
    before, terms = ordinary_terms(data), apply(data)
    for name in ("pitch_exp", "roll_exp"):
        torch.testing.assert_close(terms[name], before[name] - 1.)
    assert (takeover_sum(terms, "orientation").diff() < 0).all()
    assert torch.equal(terms["pitch_velocity"], before["pitch_velocity"])


def test_release_keeps_yaw_command_tracking_without_extra_yaw_damping():
    data = scene(3)
    data["motion_hold"][:] = torch.tensor([True, False, False])
    data["omega"][:, 2] = torch.tensor([1., 1., 2.])
    data["commands"][:, 1] = torch.tensor([0., 1., 1.])
    terms = apply(data)
    assert takeover_sum(terms, "angular_velocity")[0] < 0.
    assert takeover_sum(terms, "angular_velocity")[1:].eq(0.).all()
    assert terms["dense_yaw_broad"][1] == terms["dense_yaw_precision"][1] == 0.
    assert terms["dense_yaw_broad"][2] < 0.


def test_motor_settling_applies_only_during_hold_without_masking_native_costs():
    data = scene(2)
    data["motion_hold"][:] = torch.tensor([True, False])
    data["dq"][:] = torch.tensor([2., -2., 4., -2., 2., -4.])
    terms = apply(data)
    for axis in ("hold_leg_velocity", "hold_wheel_velocity"):
        assert takeover_sum(terms, axis)[0] < 0.
        assert takeover_sum(terms, axis)[1] == 0.
    assert terms["leg_velocity"][0] == terms["leg_velocity"][1] < 0.


def test_extra_hold_yaw_cost_does_not_depend_on_followup_command():
    data = scene(2)
    # A yaw followup becomes visible only after the command hold releases.
    data["omega"][:, 2] = .4
    data["commands"][:, 1] = torch.tensor([0., 1.])
    terms = apply(data)
    assert takeover_sum(terms, "angular_velocity")[0] == takeover_sum(terms, "angular_velocity")[1] < 0.


def test_eligible_terminal_interval_is_not_dropped_by_end_state_ownership():
    data = scene(2)
    data["eligible_interval"][:] = torch.tensor([True, False])
    data["gravity"][:, 0] = math.sin(math.radians(50.))
    data["gravity"][:, 2] = -math.cos(math.radians(50.))
    data["height"] -= .1
    terms = apply(data, support=torch.zeros(2))
    assert terms["dense_height_broad"][0] < 0. and terms["dense_height_broad"][1] == 0.
    assert takeover_sum(terms, "orientation")[0] < 0. and takeover_sum(terms, "orientation")[1] == 0.


def test_new_costs_use_where_to_isolate_nonlearning_nonfinite_inputs():
    data = scene(2)
    data["eligible_interval"][1] = False
    terms = ordinary_terms(data)
    for key in ("gravity", "omega", "dq"):
        data[key][1] = float("nan")
    apply_recovery_takeover(terms, **data, settings=configuration())
    for name, value in terms.items():
        if name.startswith("recovery_takeover_"):
            assert torch.isfinite(value).all() and value[1] == 0.


@pytest.mark.parametrize("key", ["eligible_interval", "motion_hold"])
def test_ownership_masks_cannot_be_float_or_broadcast(key):
    data = scene()
    terms = ordinary_terms(data)
    data[key] = data[key].float()
    with pytest.raises(ValueError, match="bool"):
        apply_recovery_takeover(terms, **data, settings=configuration())


def test_reward_helper_does_not_advance_clocks_or_mutate_sensor_inputs():
    data = scene()
    saved = {name: value.clone() for name, value in data.items()}
    first, second = apply(data), apply(data)
    for name, tensor in data.items():
        assert torch.equal(tensor, saved[name])
    for name in first:
        assert torch.equal(first[name], second[name])
    assert not any("event" in name or "progress" in name for name in first)


def test_complete_discounted_return_beats_waiting_and_early_failure_on_same_trace():
    data = scene(1)
    data["height"] += .002
    data["velocity"][:, 0] = .01
    density = float(sum(apply(data).values())[0])
    assert density <= 0.
    gamma, dt = .99, .02

    def total(steps, event=0.):
        rewards = torch.full((steps,), density * dt, dtype=torch.float64)
        rewards[-1] += event
        return float((rewards * gamma ** torch.arange(steps)).sum())

    # Same small-error density before each endpoint. Only the authenticated
    # full success event is +30; physical failure is -200. No script rewards,
    # observer phase bonus, or reset-state bootstrap is credited here.
    complete = total(100, 30.)
    waiting = total(1000)
    assert complete > waiting
    assert complete > total(400, 30.)
    for terminal_step in (1, 20, 99, 100, 400, 1000):
        assert complete > total(terminal_step, -200.)
    # A zero-centered perfect residence earns no renewable positive return.
    assert sum(apply(scene(1)).values()).item() == 0.


def test_density_has_no_internal_dt_and_environment_integrates_it_once():
    data = scene(1)
    data["omega"][:, 0] = .75
    terms = apply(data)
    expected = sum(dense_error_terms(torch.tensor([.75], dtype=torch.float64),
                                   configuration()["angular_velocity"]))
    actual = takeover_sum(terms, "angular_velocity")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close((actual.expand(50) * .02).sum(), actual[0])
    torch.testing.assert_close((actual.expand(100) * .01).sum(), actual[0])
