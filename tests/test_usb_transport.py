"""RTT units, stochastic jitter, shared feedback, impulse timing and reset isolation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.usb_transport import SubstepUsbTransport


ROOT = Path(__file__).resolve().parents[1]


def configuration(**overrides):
    profile = json.loads((ROOT / "contracts/v5_usb_latency_prior_v1.json").read_text())
    return {"enabled": True, "profile": profile, "payload_size": 63,
            "enabled_fraction": 1., "uplink_fraction_range": [.5, .5], **overrides}


def sensors(count):
    result = torch.zeros(count, 18)
    result[:, 17] = -1.
    return result


def test_profile_exactly_preserves_reported_hs_units_and_quantiles():
    measured = json.loads((ROOT / "docs/evidence/usb_loopback_20260924.json").read_text())
    for row in measured["measurements"]:
        if row["usb_mode"] != "HS":
            continue
        link = SubstepUsbTransport(1, "cpu", .005, configuration(payload_size=row["size"]), 617)
        expected = torch.tensor([row[k] * 1e-6 for k in ("min_us", "p50_us", "p95_us", "p99_us", "max_us")])
        torch.testing.assert_close(link.inverse_cdf(link.probabilities), expected, rtol=1e-6, atol=1e-11)


def test_jitter_changes_with_time_and_splits_one_rtt_budget():
    link = SubstepUsbTransport(4096, "cpu", .005, configuration(uplink_fraction_range=[.25, .75]), 617)
    link.reset(link.rows, sensors(4096))
    first = link.rtt.clone()
    link.observe(sensors(4096))
    assert not torch.equal(first, link.rtt)
    assert link.rtt.min() >= 55.654e-6 - 1e-11 and link.rtt.max() <= 163.846e-6 + 1e-11
    torch.testing.assert_close(link.uplink + link.downlink, link.rtt)
    assert torch.all((link.uplink / link.rtt >= .25) & (link.uplink / link.rtt <= .75))


def test_sensor_interpolation_is_shared_and_torque_integral_matches_event_timing():
    link = SubstepUsbTransport(1, "cpu", .005, configuration(), 7)
    start, end = sensors(1), sensors(1)
    end[:, 0] = .05
    end[:, 6] = 10.
    link.reset(link.rows, start)
    link.observe(end)
    expected_q = .05 - 10. * link.uplink[0]
    assert link.feedback[0, 0] == pytest.approx(float(expected_q), abs=1e-8)
    command = torch.full((1, 6), 10.)
    delayed = link.apply_torque(command)
    # Independently integrate a zero-to-10 N*m event at the downlink arrival.
    expected_impulse = 10. * (.005 - float(link.downlink[0]))
    assert float(delayed[0, 0]) * .005 == pytest.approx(expected_impulse, abs=1e-8)
    torch.testing.assert_close(link.apply_torque(command), command)


def test_zero_delay_and_disabled_envs_are_exact_identity():
    for options in ({"rtt_scale": 0.}, {"enabled_fraction": 0.}):
        link = SubstepUsbTransport(2, "cpu", .005, configuration(**options), 3)
        before, after = sensors(2), sensors(2)
        after[:, :15] = torch.arange(15).float()
        link.reset(link.rows, before)
        link.observe(after)
        assert torch.equal(link.feedback, after)
        command = torch.randn(2, 6)
        assert torch.equal(link.apply_torque(command), command)


def test_partial_reset_flushes_old_torque_and_preserves_other_environment():
    link = SubstepUsbTransport(2, "cpu", .005, configuration(), 5)
    link.reset(link.rows, sensors(2))
    link.apply_torque(torch.full((2, 6), 4.))
    before = link.previous_torque[1].clone()
    fresh = sensors(2)
    fresh[0, 0] = .3
    link.reset(torch.tensor([0]), fresh)
    assert torch.equal(link.previous_torque[0], torch.zeros(6))
    assert torch.equal(link.previous_torque[1], before)
    assert torch.equal(link.feedback[0], fresh[0])


def test_wrapped_position_and_repeated_reads_do_not_advance_jitter():
    link = SubstepUsbTransport(1, "cpu", .005, configuration(), 4)
    before, after = sensors(1), sensors(1)
    before[:, 0], after[:, 0] = torch.pi - .01, -torch.pi + .01
    link.reset(link.rows, before)
    link.observe(after)
    assert abs(float(link.feedback[0, 0] - after[0, 0])) < .001
    first = link.feedback.clone()
    assert torch.equal(link.feedback, first)


def test_seed_replay_is_deterministic_without_touching_other_rngs():
    a = SubstepUsbTransport(3, "cpu", .005, configuration(), 3)
    b = SubstepUsbTransport(3, "cpu", .005, configuration(), 3)
    for link in (a, b):
        link.reset(link.rows, sensors(3))
        link.observe(sensors(3))
    assert torch.equal(a.rtt, b.rtt)


def test_reset_does_not_shift_other_environments_future_jitter():
    a = SubstepUsbTransport(3, "cpu", .005, configuration(), 7)
    b = SubstepUsbTransport(3, "cpu", .005, configuration(), 7)
    for link in (a, b):
        link.reset(link.rows, sensors(3))
        link.observe(sensors(3))
    a.reset(torch.tensor([0]), sensors(3))
    a.observe(sensors(3))
    b.observe(sensors(3))
    assert torch.equal(a.rtt[1:], b.rtt[1:])


@pytest.mark.parametrize("overrides", [{"rtt_scale": 100.}, {"rtt_scale": -1.},
                                       {"enabled_fraction": 1.1}, {"uplink_fraction_range": [-.1, .5]}])
def test_invalid_transport_budgets_are_rejected(overrides):
    with pytest.raises(ValueError):
        SubstepUsbTransport(1, "cpu", .005, configuration(**overrides), 1)


def test_rtt_cannot_be_mislabelled_as_one_way_or_milliseconds():
    for key, value in (("units", "milliseconds"), ("measurement_kind", "one_way")):
        cfg = deepcopy(configuration())
        cfg["profile"][key] = value
        with pytest.raises(ValueError):
            SubstepUsbTransport(1, "cpu", .005, cfg, 1)
