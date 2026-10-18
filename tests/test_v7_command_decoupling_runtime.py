"""CPU runtime payments for a disabled-by-default V7 yaw-cost candidate.

The coefficient below is an arithmetic fixture, not a training choice. These
checks do not launch physics, optimize a policy or grant a capability gate.
"""
from copy import deepcopy

import pytest
import torch

from test_scut_rehearsal_reward_runtime import (
    fill_history,
    physical_values,
    runtime as baseline_runtime,
)
from wheeled_tasks.chassis.scut_flat_reward import WEIGHTS
from wheeled_tasks.chassis.scut_flat_reward_runtime import ScutFlatRewardRuntime


VERSION = "v7-pure-translation-yaw-cost-v2"
TERM = "pure_translation_yaw_square"
ARITHMETIC_COEFFICIENT = 2.5


def settings(*, enabled=True):
    value = {"version": VERSION, "enabled": enabled}
    if enabled:
        value["coefficient_per_rad2_s2"] = ARITHMETIC_COEFFICIENT
    return value


def candidate_runtime(config, *, rough=False, mask=None,
                      rehearsal_profile="scut0719_flat_v1"):
    template = baseline_runtime(rough=rough, mask=mask,
                                rehearsal_profile=rehearsal_profile)
    return ScutFlatRewardRuntime(4, "cpu", template.body_names, {
        "root_velocity_reference": "root_rigid_body_com_in_root_link_frame",
        "wheel_position_reference": "wheel_link_origins_in_root_link_frame",
        "root_height_reference": template.height_reference,
        "undesired_link_names": [template.body_names[i] for i in template.undesired_ids],
        "terminal_link_names": [template.body_names[i] for i in template.terminal_ids],
    }, reward_profile=template.reward_profile,
        flat_rehearsal_mask=mask,
        flat_rehearsal_reward_profile=rehearsal_profile,
        command_decoupling=config)


def moving_values():
    values = physical_values()
    values["commands"][:, 0] = .5
    values["commands"][:, 1] = 0.
    values["angular_velocity_body"][:, 2] = torch.tensor([.4, -.8, .2, -1.2])
    return values


def compare_legacy_terms(actual, expected):
    """Compare exact tensor bytes, including any raw NaN payloads."""
    for field in ("raw_terms", "payments"):
        actual_values, expected_values = getattr(actual, field), getattr(expected, field)
        assert [name for name in actual_values if name in WEIGHTS] == list(WEIGHTS)
        assert list(expected_values) == list(WEIGHTS)
        for name in WEIGHTS:
            a, b = actual_values[name], expected_values[name]
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.contiguous().view(torch.uint8),
                               b.contiguous().view(torch.uint8)), (field, name)


def compare_total(actual, expected_legacy, extra):
    torch.testing.assert_close(actual.total, expected_legacy.total + extra,
                               rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.total,
                               torch.stack(list(actual.payments.values()), -1).sum(-1),
                               rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("config", [None, settings(enabled=False)])
@pytest.mark.parametrize("rough", [False, True])
def test_absent_or_disabled_candidate_is_exact_legacy_result(config, rough):
    baseline, candidate = baseline_runtime(rough=rough), candidate_runtime(config, rough=rough)
    for runtime in (baseline, candidate):
        fill_history(runtime)
    values = moving_values()
    kwargs = {"reward_ground_height_world": torch.tensor([.175, .3, .2, .9])} if rough else {}
    expected, actual = baseline.pay(**values, **kwargs), candidate.pay(**values, **kwargs)
    assert list(actual.raw_terms) == list(actual.payments) == list(WEIGHTS)
    compare_legacy_terms(actual, expected)
    assert torch.equal(actual.total, expected.total)


def test_flat_addition_uses_physical_yaw_single_dt_and_exact_legacy_payments():
    baseline, candidate = baseline_runtime(rough=False), candidate_runtime(settings())
    for runtime in (baseline, candidate):
        fill_history(runtime)
    values = moving_values()
    before = deepcopy(values)
    expected, actual = baseline.pay(**values), candidate.pay(**values)
    # Root z minus plane-origin z errors are .075, .4, .05 and .5 m.
    # The first row is also terminal: a finite terminal transition still pays.
    raw = torch.tensor([.5 * .4**2, 0., .2**2, 0.])
    extra = torch.tensor([-ARITHMETIC_COEFFICIENT * .5 * .4**2 * .02, 0., 0., 0.])
    assert list(actual.raw_terms) == list(actual.payments) == [*WEIGHTS, TERM]
    compare_legacy_terms(actual, expected)
    torch.testing.assert_close(actual.raw_terms[TERM], raw, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.payments[TERM], extra, rtol=1e-6, atol=1e-7)
    compare_total(actual, expected, extra)
    assert actual.payments["termination"][0] == -4.
    assert all(payment[2] == 0. for payment in actual.payments.values())
    assert all(torch.equal(values[key], before[key]) for key in values)


@pytest.mark.parametrize("vx,yaw,eligible", [
    (0., 0., False), (0., 1., False), (.5, .2, False),
    (.099999, 0., False), (-.099999, 0., False),
    (.1, 0., True), (-.1, 0., True),
    (.5, -0., True), (-.5, -0., True),
    (.5, .000001, False), (-.5, -.000001, False),
    (.5, .05, False), (-.5, -.05, False),
    (.5, .099999, False), (-.5, -.099999, False),
    (.5, .1, False), (-.5, -.1, False),
])
def test_only_exact_zero_yaw_selects_cost_after_the_forward_threshold(vx, yaw, eligible):
    runtime = candidate_runtime(settings())
    values = moving_values()
    values["root_pose_world"][:, 2] = values["origins"][:, 2] + values["commands"][:, 2]
    values["commands"][:, 0], values["commands"][:, 1] = vx, yaw
    values["numerical_safety_reset"].zero_()
    # Cost uses observed omega_z only for an explicit zero yaw command.
    # A nonzero turn cannot be reclassified by the base reward deadzone.
    values["angular_velocity_body"][:, 2] = .4
    actual = runtime.pay(**values)
    expected_raw = torch.full((4,), .4**2 if eligible else 0.)
    expected_payment = torch.full((4,), -.008 if eligible else 0.)
    torch.testing.assert_close(actual.raw_terms[TERM], expected_raw, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.payments[TERM], expected_payment, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("yaw", [.05, -.05])
@pytest.mark.parametrize("rough", [False, True])
def test_small_command_survives_sampler_and_actor_without_additional_yaw_payment(yaw, rough):
    from wheeled_tasks.chassis.flat_command_sampling import ScutFlatCommands
    from wheeled_tasks.chassis.scut_observation import build_scut35

    sampler = ScutFlatCommands(4, "cpu")
    ids = torch.arange(4)
    request = torch.tensor([[.5, 0., yaw]]).repeat(4, 1)
    sampler.set_fixed_requests(ids, request, torch.full((4,), .3), evaluation_only=True)
    sampler.reset(ids, torch.zeros(4), torch.zeros(4, dtype=torch.bool),
                  torch.Generator().manual_seed(290929))
    commands = torch.stack((sampler.command3[:, 0], sampler.command3[:, 2], sampler.height), -1)
    zeros6 = torch.zeros(4, 6)
    actor = build_scut35(torch.zeros(4, 3), torch.tensor([[0., 0., -1.]]).repeat(4, 1),
        commands, zeros6, zeros6, zeros6, zeros6,
        torch.zeros(4, dtype=torch.bool), torch.zeros(4), torch.zeros(4))
    assert actor.shape == (4, 35)
    assert torch.equal(actor[:, 2], request[:, 2])

    values = moving_values()
    values["commands"] = commands
    values["root_pose_world"][:, 2] = values["origins"][:, 2] + sampler.height
    values["angular_velocity_body"][:, 2] = yaw
    values["numerical_safety_reset"].zero_()
    kwargs = {"reward_ground_height_world": values["origins"][:, 2]} if rough else {}
    baseline = baseline_runtime(rough=rough).pay(**values, **kwargs)
    candidate = candidate_runtime(settings(), rough=rough)
    actual = candidate.pay(**values, **kwargs)
    compare_legacy_terms(actual, baseline)
    assert torch.equal(actual.payments[TERM], torch.zeros(4))
    assert torch.equal(actual.total, baseline.total)
    assert "actor-visible yaw==0" in candidate.summary()["command_decoupling_scope"]["command_gate"]
    # The pinned base reward still targets zero below .1 rad/s. Fixing the
    # additive cost does not claim to resolve that separate source behavior.
    assert torch.all(baseline.raw_terms["track_ang_vel_z_square"] > 0.)


@pytest.mark.parametrize("unified", [False, True])
def test_rough_and_unified_rehearsal_add_cost_without_height_gate(unified):
    mask = torch.tensor([True, False, True, False]) if unified else None
    profile = "scut0730_rough_v1" if unified else "scut0719_flat_v1"
    baseline = baseline_runtime(mask=mask, rehearsal_profile=profile)
    candidate = candidate_runtime(settings(), rough=True, mask=mask, rehearsal_profile=profile)
    for runtime in (baseline, candidate):
        fill_history(runtime)
    values, ground = moving_values(), torch.tensor([.175, .3, .2, .9])
    expected = baseline.pay(**values, reward_ground_height_world=ground)
    actual = candidate.pay(**values, reward_ground_height_world=ground)
    raw = torch.tensor([.4**2, .8**2, .2**2, 1.2**2])
    extra = torch.tensor([-.008, -.032, 0., -.072])
    compare_legacy_terms(actual, expected)
    torch.testing.assert_close(actual.raw_terms[TERM], raw, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.payments[TERM], extra, rtol=1e-6, atol=1e-7)
    compare_total(actual, expected, extra)


def test_mixed_rehearsal_cost_uses_selected_flat_plane_datum_only():
    mask = torch.tensor([True, False, True, False])
    baseline = baseline_runtime(mask=mask)
    candidate = candidate_runtime(settings(), rough=True, mask=mask)
    for runtime in (baseline, candidate):
        fill_history(runtime)
    values, ground = moving_values(), torch.tensor([.175, .3, .2, .9])
    height = values["root_pose_world"][:, 2] - values["origins"][:, 2]
    # Non-selected flat height rows are unusable on purpose; ordinary rough
    # lanes must remain ungated and must not consume this alternate datum.
    height[~mask] = float("nan")
    kwargs = dict(reward_ground_height_world=ground, flat_reward_height_m=height)
    expected, actual = baseline.pay(**values, **kwargs), candidate.pay(**values, **kwargs)
    raw = torch.tensor([.5 * .4**2, .8**2, .2**2, 1.2**2])
    extra = torch.tensor([-.004, -.032, 0., -.072])
    compare_legacy_terms(actual, expected)
    torch.testing.assert_close(actual.raw_terms[TERM], raw, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.payments[TERM], extra, rtol=1e-6, atol=1e-7)
    compare_total(actual, expected, extra)


def test_additional_payment_sanitizes_nonfinite_rows_without_changing_legacy_terms():
    baseline, candidate = baseline_runtime(rough=False), candidate_runtime(settings())
    values = moving_values()
    values["angular_velocity_body"][:, 2] = torch.tensor([float("nan"), float("inf"), .2, -.4])
    values["root_pose_world"][3, 2] = float("nan")
    expected, actual = baseline.pay(**values), candidate.pay(**values)
    compare_legacy_terms(actual, expected)
    assert torch.equal(actual.payments[TERM], torch.zeros(4))
    assert torch.isfinite(actual.total).all()
    assert all(payment[2] == 0. for payment in actual.payments.values())
    compare_total(actual, expected, torch.zeros(4))


def test_candidate_owns_configuration_and_does_not_add_heading_or_payment_history():
    config = settings()
    runtime = candidate_runtime(config)
    config["coefficient_per_rad2_s2"] = 1000.
    values = moving_values()
    first = runtime.pay(**values).payments[TERM].clone()
    values["root_pose_world"][:, :2] += 100.
    values["root_pose_world"][:, 3:7] = torch.tensor([0., 0., 1., 0.])
    second = runtime.pay(**values).payments[TERM]
    torch.testing.assert_close(first, torch.tensor([-.004, 0., 0., 0.]), rtol=1e-6, atol=1e-7)
    assert torch.equal(second, first)
    runtime.reset(torch.arange(4))
    assert torch.equal(runtime.pay(**values).payments[TERM], first)


@pytest.mark.parametrize("invalid", [
    False, [], "enabled", {},
    {"enabled": False}, {"version": "unknown", "enabled": False},
    {"version": VERSION, "enabled": 0}, {"version": VERSION, "enabled": "true"},
    {"version": VERSION, "enabled": True},
    {"version": VERSION, "enabled": False, "coefficient_per_rad2_s2": 2.5},
    {"version": VERSION, "enabled": False, "extra": None},
])
def test_runtime_rejects_unknown_incomplete_or_silently_ignored_candidate_settings(invalid):
    with pytest.raises(ValueError):
        candidate_runtime(invalid)


@pytest.mark.parametrize("coefficient", [True, False, 0., -1., float("nan"),
                                        float("inf"), -float("inf"), 1e100, 1e-100,
                                        None, "2.5", [.5]])
def test_enabled_coefficient_requires_positive_finite_numeric_scalar(coefficient):
    config = settings()
    config["coefficient_per_rad2_s2"] = coefficient
    with pytest.raises(ValueError):
        candidate_runtime(config)


def test_production_finish_routes_addition_to_ppo_reward_and_existing_logger_once():
    from test_scut_flat_reward_runtime import env

    baseline, candidate = env(), env()
    template = candidate.scut_reward
    candidate.scut_reward = ScutFlatRewardRuntime(2, "cpu", template.body_names, {
        "root_velocity_reference": "root_rigid_body_com_in_root_link_frame",
        "wheel_position_reference": "wheel_link_origins_in_root_link_frame",
        "root_height_reference": template.height_reference,
        "undesired_link_names": [template.body_names[i] for i in template.undesired_ids],
        "terminal_link_names": [template.body_names[i] for i in template.terminal_ids],
    }, command_decoupling=settings())
    for instance in (baseline, candidate):
        instance.commands[:, 0] = .5
        instance.robot.data.root_com_ang_vel_b.torch[:, 2] = .4
    _, old_reward, old_done, old_extras = baseline.finish()
    _, new_reward, new_done, extras = candidate.finish()
    extra = torch.full((2,), -.008)
    torch.testing.assert_close(new_reward, old_reward + extra, rtol=1e-6, atol=1e-7)
    assert torch.equal(old_done, new_done)
    for name in WEIGHTS:
        assert torch.equal(extras["native_reward_components"][name],
                           old_extras["native_reward_components"][name])
    torch.testing.assert_close(extras["native_reward_components"][TERM], extra)
    torch.testing.assert_close(extras["log"]["/reward/" + TERM], extra.mean())
    assert extras["native_reward_component_units"] == "already_dt_scaled_payment_per_policy_step"
