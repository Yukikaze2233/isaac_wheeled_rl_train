"""CPU evidence for the shared delay budget and delayed-ZOH interval impulse."""
import subprocess
import sys

import pytest
import torch

from wheeled_tasks.chassis.feedback_transport_config import validate_feedback_transport
from wheeled_tasks.chassis.round_trip_transport import (
    EffectiveCommandDelay, RoundTripDelayBudget, command_settings, feedback_settings,
    round_trip_transport_settings, validate_round_trip_transport,
)


def make_transport(count=3, seed=17):
    cfg = round_trip_transport_settings()
    budget = RoundTripDelayBudget(count, "cpu", cfg, seed)
    budget.reset(budget.rows)
    return budget, EffectiveCommandDelay(count, "cpu", .005, command_settings(cfg), budget)


def fix_total(budget, values):
    budget.total_s.copy_(budget.total_s.new_tensor(values))
    budget.uplink_s.copy_(budget.total_s * .5)
    budget.downlink_s.copy_(budget.total_s * .5)


def test_configuration_is_pure_and_derived_from_one_total_budget():
    cfg = round_trip_transport_settings()
    assert validate_round_trip_transport(cfg) == cfg
    assert command_settings(cfg)["delay_ms"] == [.05, 2.5]
    feedback = feedback_settings(cfg)
    assert feedback["delay_range_s"] == [.00005, .0025]
    assert feedback["jitter_s"] == 0.
    assert feedback["enabled_fraction"] == 1.
    assert validate_feedback_transport(feedback) == feedback
    cfg["total_delay_range_s"][0] = 0.
    assert round_trip_transport_settings()["total_delay_range_s"] == [.0001, .005]
    subprocess.run([sys.executable, "-B", "-c", """
import builtins
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('Torch deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
from wheeled_tasks.chassis.round_trip_transport import (
    round_trip_transport_settings, validate_round_trip_transport,
    command_settings, feedback_settings)
assert validate_round_trip_transport(round_trip_transport_settings())
assert command_settings()['delay_ms'] == [.05, 2.5]
assert feedback_settings()['delay_range_s'] == [.00005, .0025]
"""], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("key,value", [
    ("enabled", False), ("enabled", 1), ("enabled_fraction", .75),
    ("enabled_fraction", True), ("uplink_fraction", .25),
    ("total_delay_range_s", [.0001, .015]), ("total_delay_range_s", [False, .005]),
    ("total_delay_range_s", [float("nan"), .005]),
    ("distribution", "normal"), ("resample", "physics_tick"), ("units", "milliseconds"),
])
def test_configuration_rejects_altered_budget_semantics(key, value):
    cfg = round_trip_transport_settings()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_round_trip_transport(cfg)


def test_budget_draw_is_owned_uniform_all_environment_and_once_per_selected_reset():
    global_state = torch.get_rng_state().clone()
    cfg = round_trip_transport_settings()
    budget = RoundTripDelayBudget(64, "cpu", cfg, 9)
    before = budget.generator.get_state().clone()
    command = EffectiveCommandDelay(64, "cpu", .005, cfg, budget)
    assert torch.equal(before, budget.generator.get_state())
    reference = torch.Generator().manual_seed(9 ^ RoundTripDelayBudget.SEED_XOR)
    expected = .0001 + .0049 * torch.rand(64, dtype=torch.float64, generator=reference)
    budget.reset(budget.rows)
    assert torch.equal(budget.total_s, expected)
    assert bool(budget.enabled.all())
    assert bool(((budget.total_s >= .0001) & (budget.total_s <= .005)).all())
    torch.testing.assert_close(budget.uplink_s, budget.downlink_s, rtol=0, atol=0)
    torch.testing.assert_close(budget.uplink_s + budget.downlink_s, budget.total_s, rtol=0, atol=0)
    drawn = budget.generator.get_state().clone()
    command.reset(command.rows)
    command.apply_torque(torch.ones(64, 6))
    assert torch.equal(drawn, budget.generator.get_state())
    assert torch.equal(global_state, torch.get_rng_state())
    old = budget.total_s.clone()
    ids = torch.tensor([3, 41])
    budget.reset(ids)
    expected_new = .0001 + .0049 * torch.rand(2, dtype=torch.float64, generator=reference)
    torch.testing.assert_close(budget.total_s[ids], expected_new, rtol=0, atol=0)
    unchanged = torch.ones(64, dtype=torch.bool)
    unchanged[ids] = False
    assert torch.equal(budget.total_s[unchanged], old[unchanged])
    assert command.delay.data_ptr() == budget.downlink_s.data_ptr()


def test_budget_empty_reset_and_reseed_leave_unselected_state_untouched():
    budget, command = make_transport()
    state = budget.generator.get_state().clone()
    total = budget.total_s.clone()
    budget.reset(torch.tensor([], dtype=torch.long))
    command.reset(torch.tensor([], dtype=torch.long))
    assert torch.equal(state, budget.generator.get_state())
    assert torch.equal(total, budget.total_s)
    budget.reseed(17)
    budget.reset(budget.rows)
    assert torch.equal(total, budget.total_s)


def test_step_response_is_delayed_interval_average_and_preserves_pulse_impulse():
    budget, command = make_transport()
    fix_total(budget, [.0001, .002, .005])
    high = torch.arange(1, 19, dtype=torch.float32).reshape(3, 6)
    first = command.apply_torque(high)
    expected = high * (1 - budget.downlink_s[:, None].float() / .005)
    torch.testing.assert_close(first, expected)
    second = command.apply_torque(torch.zeros_like(high))
    torch.testing.assert_close(second, high - first)
    # A one-interval pulse delayed across the boundary retains its full impulse.
    torch.testing.assert_close((first + second) * .005, high * .005)
    assert torch.equal(command.apply_torque(torch.zeros_like(high)), torch.zeros_like(high))
    assert command.substeps == 1 and command.transport_dt == .005
    assert command.phase_aware is False
    assert command.delivery_model == "continuous_interval_impulse_v1"


def test_command_weighting_matches_piecewise_zoh_integral_over_sign_changes():
    budget, command = make_transport(1)
    fix_total(budget, [.0036])
    previous = 0.
    for value in [10., -7., 0., 5.]:
        actual = command.apply_torque(torch.full((1, 6), value))
        # Integrate the two constant segments directly, rather than resampling.
        expected_impulse = previous * .0018 + value * (.005 - .0018)
        assert (actual[0, 0] * .005).item() == pytest.approx(expected_impulse, abs=4e-9)
        previous = value


def test_pd_policy_feedback_origins_use_duration_weights_and_keep_cached_origins():
    budget, command = make_transport(1)
    fix_total(budget, [.002])
    value = torch.ones(1, 6)
    command.apply_torque(value, pd_generated_s=0., policy_generated_s=-.02, feedback_sample_s=-.001)
    first_origins = command.applied_origins_s.clone()
    torch.testing.assert_close(first_origins, torch.tensor([[0., -.016, -.0008]], dtype=torch.float64))
    # Submission at t=.005 retains the original PD and sensor timestamps.
    command.apply_torque(value, pd_generated_s=0., policy_generated_s=-.02, feedback_sample_s=-.001)
    assert command.applied_origins_s[0].tolist() == pytest.approx([0., -.02, -.001])
    assert command.pd_age_s.item() == pytest.approx(.010)
    command.apply_torque(value * 0, pd_generated_s=.010, policy_generated_s=.008, feedback_sample_s=.009)
    assert command.applied_origins_s[0].tolist() == pytest.approx([.008, .0024, .007])
    assert command.pd_age_s.item() == pytest.approx(.007)
    assert command.applied_at_s.item() == pytest.approx(.015)


def test_reset_discards_only_selected_command_history_without_resampling_budget():
    budget, command = make_transport(2)
    fix_total(budget, [.002, .004])
    command.apply_torque(torch.tensor([[10.] * 6, [20.] * 6]), pd_generated_s=0.)
    before = {name: getattr(command, name).clone() for name in (
        "last_command", "last_origins_s", "applied_origins_s", "applied_at_s", "pd_age_s", "sent",
        "last_target_sample_s", "last_target_policy_generated_s",
        "applied_target_sample_s", "applied_target_policy_generated_s")}
    rng = budget.generator.get_state().clone()
    command.reset(torch.tensor([0]))
    for name, snapshot in before.items():
        assert torch.equal(getattr(command, name)[1], snapshot[1]), name
    assert torch.equal(rng, budget.generator.get_state())
    assert command.tick == 1
    assert command.last_command[0].tolist() == [0.] * 6
    assert command.last_origins_s[0].tolist() == pytest.approx([.005] * 3)
    assert command.pd_age_s[0].item() == 0.
    result = command.apply_torque(torch.ones(2, 6), pd_generated_s=.005)
    assert result[:, 0].tolist() == pytest.approx([.8, 8.6])


def test_returned_or_input_torque_mutation_does_not_change_held_history():
    budget, command = make_transport(1)
    fix_total(budget, [.002])
    value = torch.full((1, 6), 10.)
    applied = command.apply_torque(value)
    applied.zero_()
    value.zero_()
    following = command.apply_torque(torch.zeros(1, 6))
    assert following[0].tolist() == pytest.approx([2.] * 6)


def test_long_clock_preserves_fractional_millisecond_origins_and_age():
    budget, command = make_transport(1)
    fix_total(budget, [.0001])
    command.tick = 100_000_000
    command.reset(command.rows)
    now = command.tick * command.dt
    command.apply_torque(torch.ones(1, 6), pd_generated_s=now - .0004,
        policy_generated_s=now - .02, feedback_sample_s=now - .0008)
    assert command.applied_origins_s.dtype == torch.float64
    assert command.pd_age_s.dtype == torch.float64
    assert command.pd_age_s.item() == pytest.approx(.005 + .99 * .0004, abs=1e-10)
    command.apply_torque(torch.ones(1, 6), pd_generated_s=now + .005,
        policy_generated_s=now, feedback_sample_s=now + .0042)
    assert command.pd_age_s.item() == pytest.approx(.005 + .01 * .0054, abs=1e-10)


def test_invalid_clock_and_unsupported_transport_phase_are_rejected():
    budget, _ = make_transport(1)
    with pytest.raises(ValueError, match="200 Hz"):
        EffectiveCommandDelay(1, "cpu", .001, command_settings(), budget)
    cfg = command_settings()
    cfg["can_phase_ms"] = [0., 1.]
    with pytest.raises(ValueError, match="canonical"):
        EffectiveCommandDelay(1, "cpu", .005, cfg, budget)
    cfg = command_settings()
    cfg["delay_ms"] = [.05, 5.]
    with pytest.raises(ValueError, match="canonical"):
        EffectiveCommandDelay(1, "cpu", .005, cfg, budget)


def test_timestamp_vectors_and_metrics_expose_original_pd_age():
    budget, command = make_transport(2)
    fix_total(budget, [.002, .004])
    command.apply_torque(torch.ones(2, 6), pd_generated_s=torch.tensor([-.001, -.002], dtype=torch.float64))
    assert command.pd_age_s.tolist() == pytest.approx([.0058, .0062])
    metrics = command.metrics() | budget.metrics()
    assert metrics["/transport/applied_pd_computation_age_ms"].item() == pytest.approx(6.)
    assert metrics["/transport/round_trip_delay_mean_s"].item() == pytest.approx(.003)
    assert metrics["/transport/downlink_enabled_fraction"].item() == 1.
    assert metrics["/transport/command_loss_fraction"].item() == 0.
    with pytest.raises(ValueError, match="timestamps"):
        command.apply_torque(torch.ones(2, 6), pd_generated_s=torch.zeros(2, 1))
    assert command.tick == 1
