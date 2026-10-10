"""Fractional one-way feedback, chronological delivery and native reset isolation."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from wheeled_tasks.chassis.feedback_transport import FractionalFeedbackTransport, shared_activation_draw
from wheeled_tasks.chassis.feedback_transport_config import validate_feedback_transport


ROOT = Path(__file__).resolve().parents[1]


def configuration(**overrides):
    return {
        "version": "complete_one_way_feedback_v1",
        "enabled": True,
        "units": "seconds",
        "scope": "engineering_prior_complete_sensor_to_host_path_not_measured_rtt",
        "delay_range_s": [.00004, .015],
        "enabled_fraction": .75,
        "jitter_s": .00002,
        "jitter_correlation": .9,
        "ordered_delivery": "monotonic_sensor_timestamp",
        "reconstruction": "linear_physics_history",
        **overrides,
    }


def sensors(count, time=0.):
    result = torch.zeros(count, 18)
    result[:, 0] = 2 * time
    result[:, 6] = 1 + 3 * time
    result[:, 12] = time
    result[:, 17] = -1.
    return result


def fixed_link(count=1, delay=.00004, seed=17):
    return FractionalFeedbackTransport(count, "cpu", .005,
        configuration(delay_range_s=[delay, delay], enabled_fraction=1., jitter_s=0.), seed)


@pytest.mark.parametrize("delay", [.00004, .00015, .0025, .0123, .015])
def test_fractional_linear_trace_matches_sensor_timestamp_including_oldest_15ms(delay):
    link = fixed_link(delay=delay)
    link.reset(link.rows, sensors(1))
    for tick in range(1, 61):
        now = tick * .005
        link.observe(sensors(1, now))
        source_time = max(0., now - delay)
        torch.testing.assert_close(link.feedback, sensors(1, source_time), rtol=1e-6, atol=1e-7)
        assert float(link.sensor_timestamp_s[0]) == pytest.approx(source_time, abs=1e-12)
        assert float(link.age_s[0]) == pytest.approx(min(now, delay), abs=1e-12)
    assert link.history.max_length == (int(torch.ceil(torch.tensor(delay / .005))) + 2)
    assert link.sensor_timestamp_s.dtype == torch.float64


def test_repeated_actor_and_pd_reads_share_one_cached_sample_and_do_not_advance_clock_or_rng():
    link = fixed_link(delay=.0123)
    link.reset(link.rows, sensors(1))
    link.observe(sensors(1, .005))
    before = link.feedback.clone()
    timestamp = link.sensor_timestamp_s.clone()
    random_state = link.generator.get_state().clone()
    cached = link.feedback
    for _ in range(20):
        assert link.feedback is cached
        torch.testing.assert_close(link.feedback, before, rtol=0, atol=0)
        link.metrics()
    assert link.physics_steps == 1
    assert torch.equal(timestamp, link.sensor_timestamp_s)
    assert torch.equal(random_state, link.generator.get_state())


@pytest.mark.parametrize("options", [{"delay_range_s": [0., 0.]}, {"enabled_fraction": 0.}, {"enabled": False}])
def test_zero_or_disabled_feedback_is_exact_identity(options):
    link = FractionalFeedbackTransport(2, "cpu", .005, configuration(**options), 3)
    link.reset(link.rows, sensors(2))
    latest = sensors(2, .005)
    latest[:, :15] = torch.arange(15).float()
    latest[:, 15:18] = torch.tensor([0., 0., -2.])
    link.observe(latest)
    assert torch.equal(link.feedback, latest)
    assert link.age_s.eq(0).all()
    assert link.sensor_timestamp_s.eq(.005).all()


def test_jitter_cannot_make_delivered_sensor_time_go_backward():
    link = FractionalFeedbackTransport(1, "cpu", .005,
        configuration(delay_range_s=[0., .015], enabled_fraction=1., jitter_s=0., jitter_correlation=0.), 3)
    link.reset(link.rows, sensors(1))
    last_source_time = 0.
    held_steps = 0
    for tick, delay in enumerate([0., .015, .015, 0., .015, 0., .015, .015], 1):
        link.base_delay_s[:] = delay
        now = tick * .005
        source_time = max(last_source_time, now - delay)
        held_steps += source_time == last_source_time
        link.observe(sensors(1, now))
        assert float(link.sensor_timestamp_s[0]) == pytest.approx(source_time, abs=1e-12)
        torch.testing.assert_close(link.feedback, sensors(1, source_time), rtol=1e-6, atol=1e-7)
        last_source_time = source_time
    assert held_steps >= 3


def test_wrapped_hip_and_wheel_encoders_follow_shortest_arc_and_gravity_is_unit_length():
    link = fixed_link(delay=.0025)
    before, after = sensors(1), sensors(1)
    before[:, [0, 4]] = torch.pi - .01
    after[:, [0, 4]] = -torch.pi + .01
    after[:, 15:18] = torch.tensor([1., 0., 0.])
    link.reset(link.rows, before)
    link.observe(after)
    angular_error = torch.atan2((link.feedback[0, [0, 4]] + torch.pi).sin(),
        (link.feedback[0, [0, 4]] + torch.pi).cos())
    torch.testing.assert_close(angular_error, torch.zeros(2), atol=2e-7, rtol=0)
    torch.testing.assert_close(link.feedback[:, 15:18], torch.tensor([[2 ** -.5, 0., -(2 ** -.5)]]))


def test_antipodal_gravity_retains_previous_unit_direction():
    link = fixed_link(delay=.0025)
    before, after = sensors(1), sensors(1)
    after[:, 17] = 1.
    link.reset(link.rows, before)
    link.observe(after)
    assert torch.equal(link.feedback[:, 15:18], before[:, 15:18])


def test_partial_reset_primes_immediately_and_preserves_other_feedback_history_and_rng():
    a, b = fixed_link(3, .0123), fixed_link(3, .0123)
    for link in (a, b):
        link.reset(link.rows, sensors(3))
        for tick in range(1, 6):
            link.observe(sensors(3, tick * .005))
    before_feedback = a.feedback[1:].clone()
    before_timestamp = a.sensor_timestamp_s[1:].clone()
    before_counts = a.history.current_length[1:].clone()
    before_history = a.history._buffer[:, 1:].clone()
    random_state = a.generator.get_state().clone()
    fresh = sensors(3, .025)
    fresh[0, :15] = 1.
    a.reset(torch.tensor([0]), fresh)
    assert torch.equal(a.feedback[0], fresh[0])
    assert torch.equal(a.feedback[1:], before_feedback)
    assert torch.equal(a.sensor_timestamp_s[1:], before_timestamp)
    assert torch.equal(a.history.current_length[1:], before_counts)
    assert torch.equal(a.history._buffer[:, 1:], before_history)
    assert torch.equal(a.generator.get_state(), random_state)
    newest = sensors(3, .030)
    newest[0, :15] = 2.
    a.observe(newest)
    b.observe(newest)
    assert torch.equal(a.feedback[0], fresh[0])
    assert float(a.age_s[0]) == pytest.approx(.005)
    assert torch.equal(a.feedback[1:], b.feedback[1:])
    assert torch.equal(a.delay_s[1:], b.delay_s[1:])


def test_partial_prime_keeps_the_reset_endpoint_for_first_subms_interpolation_without_rng_draws():
    link = fixed_link(2, .00004)
    link.reset(link.rows, sensors(2))
    link.observe(sensors(2, .005))
    before_step_rng = link.generator.get_state().clone()
    before_reset_rng = link.reset_generator.get_state().clone()
    before_pointer = link.history._pointer
    fresh = sensors(2, .005)
    fresh[0, 6] = 10.
    link.prime(torch.tensor([0]), fresh)
    assert torch.equal(before_step_rng, link.generator.get_state())
    assert torch.equal(before_reset_rng, link.reset_generator.get_state())
    assert link.history._pointer == before_pointer
    latest = sensors(2, .010)
    latest[0, 6] = 20.
    link.observe(latest)
    assert float(link.feedback[0, 6]) == pytest.approx(20 - 10 * .00004 / .005, abs=1e-6)


def test_seed_replay_jitter_and_reset_rng_are_independent_from_global_rng():
    global_rng = torch.random.get_rng_state().clone()
    a = FractionalFeedbackTransport(4096, "cpu", .005, configuration(), 617)
    b = FractionalFeedbackTransport(4096, "cpu", .005, configuration(), 617)
    for link in (a, b):
        link.reset(link.rows, sensors(4096))
    assert .72 < float(a.enabled.float().mean()) < .78
    assert torch.equal(a.base_delay_s, b.base_delay_s)
    assert a.base_delay_s.min() >= .00004 and a.base_delay_s.max() <= .015
    initial_delay = a.delay_s.clone()
    for tick in range(1, 5):
        for link in (a, b):
            link.observe(sensors(4096, tick * .005))
    assert not torch.equal(a.delay_s, initial_delay)
    assert torch.equal(a.feedback, b.feedback)
    assert torch.equal(a.delay_s, b.delay_s)
    a.reset(torch.tensor([0]), sensors(4096, .020))
    for link in (a, b):
        link.observe(sensors(4096, .025))
    assert torch.equal(a.delay_s[1:], b.delay_s[1:])
    assert torch.equal(a.feedback[1:], b.feedback[1:])
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    for link in (a, b):
        link.reseed(19)
        link.reset(link.rows, sensors(4096, .025))
        link.observe(sensors(4096, .030))
    assert torch.equal(a.feedback, b.feedback)
    assert torch.equal(a.delay_s, b.delay_s)


def test_domain_activation_and_set_enabled_preserve_other_rows_and_chronology():
    domain_draw = torch.tensor([.1, .9, .5])
    link = FractionalFeedbackTransport(3, "cpu", .005, configuration(), 7, domain_draw=domain_draw)
    link.reset(link.rows, sensors(3))
    assert torch.equal(link.enabled, torch.tensor([True, False, True]))
    link.observe(sensors(3, .005))
    other_feedback = link.feedback[[0, 2]].clone()
    source_time = link.sensor_timestamp_s[1].clone()
    link.set_enabled(torch.tensor([1]), True)
    assert link.sensor_timestamp_s[1] == source_time
    assert torch.equal(link.feedback[[0, 2]], other_feedback)
    link.set_enabled(torch.tensor([0]), False)
    assert torch.equal(link.feedback[0], sensors(3, .005)[0])
    assert link.age_s[0] == 0


def test_timestamp_precision_and_stable_tensorboard_seconds_metrics_at_long_runtime():
    link = fixed_link(delay=.00004)
    link.physics_steps = 2_000_000_000
    link.reset(link.rows, sensors(1))
    link.observe(sensors(1))
    assert float(link.age_s[0]) == pytest.approx(.00004, abs=2e-9)
    values = link.metrics()
    assert set(values) == {
        "/transport/feedback_enabled_fraction", "/transport/feedback_delay_mean_s",
        "/transport/feedback_delay_max_s", "/transport/feedback_sensor_timestamp_mean_s",
        "/transport/feedback_sensor_timestamp_min_s", "/transport/feedback_age_mean_s",
        "/transport/feedback_age_max_s",
    }
    assert all(value.ndim == 0 and torch.isfinite(value) for value in values.values())
    assert float(values["/transport/feedback_age_mean_s"]) == pytest.approx(.00004, abs=2e-9)


def test_hot_path_uses_native_cursor_without_roll_scalar_reads_or_tensor_truth(monkeypatch):
    from isaaclab_experimental.utils.buffers import CircularBuffer

    link = fixed_link(4, .0123)
    assert isinstance(link.history, CircularBuffer)
    link.reset(link.rows, sensors(4))

    def rejected(*args, **kwargs):
        raise AssertionError("Feedback hot path must not synchronize or roll native history")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", rejected)
        patch.setattr(torch.Tensor, "__bool__", rejected)
        patch.setattr(torch, "roll", rejected)
        for tick in range(1, 20):
            if tick == 8:
                link.reset(torch.tensor([0, 2]), sensors(4, tick * .005))
            link.observe(sensors(4, tick * .005))
            link.metrics()


def test_fractional_native_groups_retain_integer_read_and_pending_reset_semantics():
    from wheeled_tasks.chassis.delay_buffer import GroupedDelayBuffer

    buffer = GroupedDelayBuffer(5, 2, "cpu", [0, 1])
    buffer.append(torch.zeros(2, 2))
    buffer.append(torch.tensor([[10., 20.], [30., 40.]]))
    ages = torch.tensor([[0., .5], [.5, 0.]])
    torch.testing.assert_close(buffer.read_groups_fractional(ages), torch.tensor([[10., 10.], [15., 40.]]))
    integer_ages = torch.tensor([[1, 0], [0, 1]])
    assert torch.equal(buffer.read_groups_fractional(integer_ages), buffer.read_groups(integer_ages))
    buffer.reset(torch.tensor([0, 1]))
    buffer.prime(torch.full((2, 2), 7.), torch.tensor([0]))
    buffer.append(torch.full((2, 2), 9.))
    torch.testing.assert_close(buffer.read_groups_fractional(torch.ones(2, 2) * .5),
        torch.tensor([[8., 8.], [9., 9.]]))


@pytest.mark.parametrize("key,value", [
    ("version", "usb_rtt_v1"), ("scope", "loopback_round_trip_time"), ("units", "milliseconds"),
    ("enabled", 1), ("enabled", None), ("enabled_fraction", True), ("enabled_fraction", 1.1),
    ("enabled_fraction", float("nan")), ("delay_range_s", [-.1, .015]),
    ("delay_range_s", [.015, .00004]), ("delay_range_s", [.00004, float("inf")]),
    ("delay_range_s", [.00004, .016]),
    ("delay_range_s", [False, .015]), ("delay_range_s", [.015]),
    ("jitter_s", -.1), ("jitter_s", float("nan")), ("jitter_s", True),
    ("jitter_correlation", 1.1), ("jitter_correlation", float("inf")),
    ("ordered_delivery", "unordered"), ("reconstruction", "nearest_tick"),
    ("drop_probability_max", 0.), ("burst_probability", .1),
])
def test_validator_rejects_rtt_relabelling_units_loss_and_invalid_numeric_contracts(key, value):
    bad = configuration()
    bad[key] = value
    with pytest.raises(ValueError):
        validate_feedback_transport(bad)


def test_validator_has_no_isaac_or_torch_import_and_owns_config_copy():
    script = (
        "import sys; from wheeled_tasks.chassis.feedback_transport_config import validate_feedback_transport; "
        "assert not any(name == 'torch' or name.startswith('isaaclab') for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original = configuration()
    validated = validate_feedback_transport(original)
    validated["delay_range_s"][0] = 0.
    assert original == configuration()
    with pytest.raises(ValueError):
        validate_feedback_transport(deepcopy(original), .001)
    with pytest.raises(ValueError):
        validate_feedback_transport(deepcopy(original), True)


def test_empty_reset_and_invalid_sensor_shapes_leave_owned_streams_unchanged():
    link = fixed_link(2)
    link.reset(link.rows, sensors(2))
    initial = link.feedback.clone()
    state = link.reset_generator.get_state().clone()
    link.reset(torch.tensor([], dtype=torch.long), sensors(2))
    assert torch.equal(link.feedback, initial)
    assert torch.equal(link.reset_generator.get_state(), state)
    with pytest.raises(ValueError):
        link.observe(torch.zeros(2, 17))
    with pytest.raises(ValueError):
        link.prime(link.rows, torch.zeros(2, 18, dtype=torch.long))
    with pytest.raises(ValueError):
        FractionalFeedbackTransport(2, "cpu", .005, configuration(), 3, domain_draw=torch.zeros(3))


def test_shared_activation_draw_replays_distinct_lanes_without_advancing_global_rng():
    before = torch.random.get_rng_state().clone()
    a = shared_activation_draw(4096, "cpu", 617)
    b = shared_activation_draw(4096, "cpu", 617)
    c = shared_activation_draw(4096, "cpu", 618)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.shape == (4096,) and a.dtype == torch.float32
    assert a.min() >= 0 and a.max() < 1
    assert .72 < float((a < .75).float().mean()) < .78
    assert a.unique().numel() > 4000
    assert torch.equal(torch.random.get_rng_state(), before)


def test_shared_activation_draw_borrows_supplied_tensor_without_copying_or_mutating():
    supplied = torch.tensor([.1, .7, .9], dtype=torch.float64)
    before = supplied.clone()
    before_rng = torch.random.get_rng_state().clone()
    returned = shared_activation_draw(3, "cpu", 617, supplied)
    assert returned is supplied
    assert torch.equal(supplied, before)
    assert torch.equal(torch.random.get_rng_state(), before_rng)


def test_shared_draw_pairs_feedback_and_native_command_transport_activation_across_resets():
    from wheeled_tasks.chassis.command_transport import UsbCommandTransport

    count = 4096
    draw = shared_activation_draw(count, "cpu", 617)
    before = draw.clone()
    feedback = FractionalFeedbackTransport(count, "cpu", .005, configuration(), 617, domain_draw=draw)
    command = UsbCommandTransport(count, "cpu", .005, {
        "can_period_ms": 1., "delay_ms": [.04, 15.], "enabled_fraction": .75,
        "drop_probability_max": 0., "burst_probability": 0., "burst_packets": [1, 1],
    }, 617, domain_draw=draw)
    for ids in (torch.arange(count), torch.tensor([0, 1, 17, 1024]), torch.arange(count)):
        feedback.reset(ids, sensors(count))
        command.reset(ids)
        assert torch.equal(feedback.enabled, command.enabled)
        assert torch.equal(feedback.enabled, draw < .75)
    assert feedback.domain_draw is command.domain_draw is draw
    assert torch.equal(draw, before)
