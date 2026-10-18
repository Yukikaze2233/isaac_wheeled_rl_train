"""Phase-aware USB/CAN holds against physical event and legacy bitwise oracles."""
import hashlib

import pytest
import torch

from wheeled_tasks.chassis.command_transport import UsbCommandTransport


def config(**overrides):
    return dict(delay_ms=[.04, .16], can_period_ms=1., enabled_fraction=1.,
        drop_probability_max=0., burst_probability=0., burst_packets=[2, 3],
        delivery_model="phase_aware_hold_v1", can_phase_ms=[0., 1.],
        jitter_ms=.02, jitter_correlation=.9, **overrides)


def make(count=1, profiles=None, seed=714, **overrides):
    cfg = config()
    cfg.update(overrides)
    return UsbCommandTransport(count, "cpu", .001, cfg, seed, profiles=profiles)


def command(values):
    return torch.tensor(values, dtype=torch.float32)[:, None].expand(-1, 6)


def test_fractional_impulse_is_distinct_from_end_held_state():
    t = make(profiles=[{"delay_ms": .1, "can_phase_ms": .25}])
    assert torch.equal(t.apply_torque(command([8.])), command([6.]))
    assert torch.equal(t.held, command([8.]))
    assert torch.equal(t.apply_torque(command([-4.])), command([-1.]))
    assert torch.equal(t.held, command([-4.]))


def test_fixed_profile_default_phase_and_no_jitter_are_stable():
    t = make(profiles=[{"delay_ms": .1}])
    assert t.can_phase.item() == .0005
    assert t.jitter_half_width.item() == 0.
    assert torch.equal(t.apply_torque(command([4.])), command([2.]))
    phase = t.can_phase.clone()
    t.reset(t.rows)
    assert torch.equal(t.can_phase, phase)


@pytest.mark.parametrize("phase,delay,first,second", [
    (0., 0., 4., 8.), (0., .1, 0., 4.), (1., .1, 0., 4.),
    (.5, .5, 2., 6.), (.5, .6, 0., 2.),
])
def test_can_boundary_and_endpoint_phases(phase, delay, first, second):
    t = make(delay_ms=[0., 1.], profiles=[{"delay_ms": delay, "can_phase_ms": phase}])
    assert torch.equal(t.apply_torque(command([4.])), command([first]))
    assert torch.equal(t.apply_torque(command([8.])), command([second]))


def test_submillisecond_jitter_changes_the_actual_can_event():
    profiles = [{"delay_ms": .1, "can_phase_ms": .1,
                 "jitter_ms": .02, "jitter_correlation": 0.}] * 64
    t = make(64, profiles)
    result = t.apply_torque(command([1.] * 64))[:, 0]
    early = t.delay <= t.can_phase
    assert early.any() and (~early).any()
    assert torch.allclose(result[early], torch.full_like(result[early], .9))
    assert torch.equal(result[~early], torch.zeros_like(result[~early]))


def test_fifo_and_same_event_latest_wins_without_backlog_replay():
    t = make(delay_ms=[0., 3.], profiles=[{"delay_ms": .1, "can_phase_ms": .5}])
    results = []
    for value, delay in ((1., 2.2), (2., .1), (3., .1), (4., .1)):
        t.base_delay.fill_(delay * .001)
        t.delay.copy_(t.base_delay)
        results.append(t.apply_torque(command([value]))[0, 0].item())
    assert results == [0., 0., 1.5, 3.5]
    assert t.held[0, 0] == 4.
    assert not t.valid.any()


def test_iid_total_loss_preserves_held_and_an_already_inflight_packet():
    t = make(delay_ms=[0., 2.], profiles=[{"delay_ms": 1.2, "can_phase_ms": .5}])
    assert torch.equal(t.apply_torque(command([4.])), command([0.]))
    t.loss_probability.fill_(1.)
    assert torch.equal(t.apply_torque(command([50.])), command([2.]))
    assert torch.equal(t.apply_torque(command([-50.])), command([4.]))
    assert t.lost.item() == 2.
    assert not t.valid.any()


def test_deterministic_burst_holds_then_adopts_only_the_newest_target():
    t = make(profiles=[{"delay_ms": .1, "can_phase_ms": .5,
                        "burst_every_packets": 3, "burst_packets": 2}])
    values = [t.apply_torque(command([float(i)]))[0, 0].item() for i in range(1, 7)]
    assert values == [.5, 1.5, 2.5, 3., 3., 4.5]
    assert t.lost.item() == 2.
    assert t.held[0, 0].item() == 6.


def test_random_loss_fifo_matches_independent_scalar_event_queue():
    t = make(delay_ms=[0., 4.], jitter_ms=2., jitter_correlation=.3,
             drop_probability_max=.4, burst_probability=.13)
    phase = t.can_phase.item()
    queue, held, last_arrival = [], 0., 0.
    lost = 0.
    for tick in range(150):
        target = float((tick * 13) % 29 - 14)
        result = t.apply_torque(command([target]))[0, 0].item()
        now = tick * .001
        if t.lost.item() == lost:
            last_arrival = max(now + t.delay.item(), last_arrival)
            queue.append((last_arrival, target))
        lost = t.lost.item()
        previous = held
        arrived = [packet for packet in queue if packet[0] <= now + phase + 1e-10]
        queue = [packet for packet in queue if packet[0] > now + phase + 1e-10]
        if arrived:
            held = arrived[-1][1]
        expected = previous * phase / .001 + held * (1 - phase / .001)
        assert result == pytest.approx(expected, abs=3e-6)
        assert t.held[0, 0].item() == held


def test_expired_slot_is_not_replayed_and_future_tag_survives_housekeeping():
    t = make(profiles=[{"delay_ms": .1, "can_phase_ms": .5}])
    t.apply_torque(command([2.]))
    slot = t.tick % t.slots
    t.valid[slot] = True
    t.pending[slot] = 99.
    t.pending_tick[slot] = t.tick + t.slots
    t.loss_probability.fill_(1.)
    assert torch.equal(t.apply_torque(command([0.])), command([2.]))
    assert t.valid[slot].all()
    expired_slot = t.tick % t.slots
    t.valid[expired_slot] = True
    t.pending[expired_slot] = -99.
    t.pending_tick[expired_slot] = t.tick - 1
    assert torch.equal(t.apply_torque(command([0.])), command([2.]))
    assert not t.valid[expired_slot].any()


def test_partial_reset_cannot_leak_pending_packets_or_touch_other_rows():
    t = make(2, [{"delay_ms": 2.2, "can_phase_ms": .5}] * 2, delay_ms=[0., 3.])
    t.apply_torque(command([10., 20.]))
    unaffected = {name: getattr(t, name)[:, 1].clone()
                  for name in ("pending", "valid", "pending_tick", "generated")}
    t.reset(torch.tensor([0]))
    assert not t.valid[:, 0].any()
    assert (t.pending_tick[:, 0] == -1).all()
    for name, value in unaffected.items():
        assert torch.equal(getattr(t, name)[:, 1], value)
    t.loss_probability.fill_(1.)
    t.apply_torque(command([100., 200.]))
    assert torch.equal(t.apply_torque(command([100., 200.])), command([0., 10.]))
    assert torch.equal(t.held, command([0., 20.]))


def test_seeded_phase_jitter_and_reset_are_reproducible():
    first = make(64, drop_probability_max=.2, burst_probability=.05)
    second = make(64, drop_probability_max=.2, burst_probability=.05)
    for tick in range(20):
        if tick == 10:
            first.reset(first.rows[::3])
            second.reset(second.rows[::3])
        target = command([float(tick)] * 64)
        assert torch.equal(first.apply_torque(target), second.apply_torque(target))
        assert torch.equal(first.can_phase, second.can_phase)
        assert torch.equal(first.delay, second.delay)
    assert not torch.equal(first.can_phase, make(64, seed=715).can_phase)


def test_submillisecond_resolution_survives_long_training_clock():
    t = make(2, [{"delay_ms": .099, "can_phase_ms": .1},
                 {"delay_ms": .101, "can_phase_ms": .1}])
    t.tick = 1_000_000_000
    t.reset(t.rows)
    actual = t.apply_torque(command([1., 1.]))
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, command([.9, 0.]))


def test_disabled_rows_bypass_transport_without_populating_its_cache():
    t = make(2, [{"delay_ms": .1, "can_phase_ms": .5}, None])
    assert torch.equal(t.apply_torque(command([4., -7.])), command([2., -7.]))
    assert not t.valid[:, 1].any()
    assert t.sent[1] == t.lost[1] == t.age[1] == 0


def test_coarser_interval_preserves_phase_impulse_and_final_cache():
    t = UsbCommandTransport(1, "cpu", .003, config(), 714,
        profiles=[{"delay_ms": .1, "can_phase_ms": .25}])
    assert torch.allclose(t.apply_torque(command([4.])), command([11. / 3]))
    assert torch.equal(t.held, command([4.]))
    assert torch.allclose(t.apply_torque(command([8.])), command([23. / 3]))
    assert torch.equal(t.held, command([8.]))
    assert t.tick == 6


@pytest.mark.parametrize("overrides", [
    {"delivery_model": "unknown"}, {"can_phase_ms": [-.1, 1.]},
    {"can_phase_ms": [0., 1.1]}, {"jitter_ms": -1.}, {"jitter_ms": float("nan")},
    {"jitter_correlation": 1.1}, {"jitter_correlation": float("inf")},
])
def test_invalid_explicit_phase_or_jitter_cannot_silently_fall_back(overrides):
    with pytest.raises(ValueError):
        make(**overrides)


@pytest.mark.parametrize("dt,with_profiles,expected", [
    (.001, False, "9dc00934b9b4c70f0da08a09ef068d79c1ee9e255c2eedf9a2c19c8b6c204956"),
    (.005, False, "efbd482a50e88f36bd3aed0014a5058bed1662f83a93251e1900e9aa61f698c5"),
    (.001, True, "296f5bca8dbbc40fa6c11e3cc8024ee2dfe4d77f4bcad8bb76184b2307203e36"),
])
def test_legacy_bitwise_outputs_state_and_rng_match_prechange_capture(dt, with_profiles, expected):
    cfg = dict(delay_ms=[.04, 2.4], can_period_ms=1., enabled_fraction=.75,
        drop_probability_max=.2, burst_probability=.12, burst_packets=[2, 3])
    profiles = ([{"delay_ms": .16}, {"delay_ms": 1.2, "burst_every_packets": 4,
                 "burst_packets": 2}, None, {"delay_ms": 2.4, "drop_probability": .3}]
                if with_profiles else None)
    t = UsbCommandTransport(4, "cpu", dt, cfg, 714, profiles=profiles)
    digest = hashlib.sha256()
    for tick in range(17):
        if tick == 9:
            t.reset(torch.tensor([1, 3]))
        target = (torch.arange(24, dtype=torch.float32).reshape(4, 6) - 10) * ((tick + 1) / 7)
        digest.update(t.apply_torque(target).numpy().tobytes())
    for name in ("held", "pending", "valid", "generated", "held_generated", "last_arrival",
                 "delay", "base_delay", "loss_probability", "enabled", "burst_remaining", "sent", "lost", "age"):
        digest.update(getattr(t, name).numpy().tobytes())
    digest.update(t.generator.get_state().numpy().tobytes())
    assert digest.hexdigest() == expected
