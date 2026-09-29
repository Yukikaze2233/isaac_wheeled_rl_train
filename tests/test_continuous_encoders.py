"""Independent angle-winding oracles, including the observed V6 PhysX wrap."""
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.continuous_encoders import ContinuousEncoderAngles
from wheeled_tasks.chassis.mechanism_observer import ClosedChainHeightModel


def lookup():
    table = json.loads((Path(__file__).resolve().parents[1] / "contracts/v6_height_lookup_v1.json").read_text())
    geometry = ClosedChainHeightModel(table, "cpu")
    limits = [[float(side["delta_rad"][0]), float(side["delta_rad"][-1])] for side in geometry.sides]
    return geometry, limits


def test_observed_raw_hip_wrap_does_not_destroy_closed_chain_fk():
    # probe-01 left_side_90, before/after the independent raw hip wrap.
    before = torch.tensor([[-5.981, -5.348, 0., 5.982, 5.349, 0.]])
    after = torch.tensor([[5.799, -6.046, 0., -5.799, 6.045, 0.]])
    geometry, limits = lookup()
    provider = ContinuousEncoderAngles(1, "cpu", limits)
    provider.reset([0], before)
    continuous = provider.update(after)
    _, raw_valid = geometry.wheel_positions(after)
    points, valid = geometry.wheel_positions(continuous)
    assert not raw_valid.any() and valid.all() and provider.valid.all()
    assert continuous[0, 0] < -math.tau and continuous[0, 3] > math.tau
    assert torch.isfinite(points).all()


def test_reset_after_release_resolves_auxiliary_branch_without_changing_hip_turn():
    raw = torch.tensor([[5.799, -6.046, 0., -5.799, 6.045, 0.]])
    original = raw.clone()
    geometry, limits = lookup()
    provider = ContinuousEncoderAngles(1, "cpu", limits)
    q = provider.reset([0], raw)
    assert provider.valid.all() and geometry.wheel_positions(q)[1].all()
    torch.testing.assert_close(q[:, [0, 3]], raw[:, [0, 3]])
    torch.testing.assert_close(torch.cos(q), torch.cos(raw), atol=1e-6, rtol=0.)
    torch.testing.assert_close(raw, original)
    provider.update(raw)
    assert provider.valid.all()  # No false increment from the initial offset.


def test_many_paired_revolutions_preserve_accumulation_and_relative_angle():
    _, limits = lookup()
    provider = ContinuousEncoderAngles(1, "cpu", limits)
    initial = torch.tensor([[-.4, .3, 0., .4, -.3, 0.]])
    provider.reset([0], initial)
    for tick in range(1, 1601):
        angle = tick * .02
        true = initial + torch.tensor([[angle, angle, 0., -angle, -angle, 0.]])
        # PhysX may wrap over a 4pi window; this is still angle-equivalent.
        raw = torch.remainder(true + math.tau, 2 * math.tau) - math.tau
        provider.update(raw)
    torch.testing.assert_close(provider.q, true, atol=1e-4, rtol=0.)
    assert provider.valid.all() and provider.q[0, 0] > 5 * math.tau


def test_true_out_of_domain_relative_pose_stays_invalid_after_reset_and_motion():
    _, limits = lookup()
    provider = ContinuousEncoderAngles(1, "cpu", limits)
    raw = torch.tensor([[0., limits[0][0] - .05, 0., 0., limits[1][1] + .05, 0.]])
    provider.reset([0], raw)
    assert not provider.valid.any()
    for _ in range(10):
        raw += torch.tensor([[.02, .02, 0., -.02, -.02, 0.]])
        provider.update(raw)
    assert not provider.valid.any()


def test_reset_is_local_and_same_feedback_repeated_twice_is_idempotent():
    _, limits = lookup()
    provider = ContinuousEncoderAngles(2, "cpu", limits)
    raw = torch.tensor([[-.4, .3, 0., .4, -.3, 0.]]).repeat(2, 1)
    provider.reset([0, 1], raw)
    moved = raw + torch.tensor([[.02, .02, 0., -.02, -.02, 0.]])
    provider.update(moved)
    saved = provider.q.clone()
    provider.update(moved)
    torch.testing.assert_close(provider.q, saved)
    provider.reset([0], raw)
    torch.testing.assert_close(provider.q[1], saved[1])
    torch.testing.assert_close(provider.last_raw[1], moved[1])


def test_nonfinite_encoder_cannot_be_reported_as_valid_geometry():
    _, limits = lookup()
    provider = ContinuousEncoderAngles(1, "cpu", limits)
    raw = torch.tensor([[-.4, .3, 0., .4, -.3, 0.]])
    provider.reset([0], raw)
    raw[0, 0] = float("nan")
    provider.update(raw)
    assert not provider.valid.any()


def test_projection_is_nonmutating_and_release_realignment_preserves_hip_turns():
    _, limits = lookup()
    provider = ContinuousEncoderAngles(1, "cpu", limits)
    raw = torch.tensor([[5.799, -6.046, 0., -5.799, 6.045, 0.]])
    provider.reset([0], raw)
    for tick in range(1, 1001):
        moved = raw + torch.tensor([[.02, .02, 0., -.02, -.02, 0.]]) * tick
        provider.update(torch.atan2(moved.sin(), moved.cos()))
    saved, last = provider.q.clone(), provider.last_raw.clone()
    next_raw = provider.last_raw + .01
    projected = provider.project(next_raw)
    torch.testing.assert_close(provider.q, saved)
    torch.testing.assert_close(provider.last_raw, last)
    torch.testing.assert_close(projected, saved + .01, atol=2e-6, rtol=0.)
    provider.realign(torch.tensor([0]))
    torch.testing.assert_close(provider.q, saved, atol=2e-6, rtol=0.)
    assert provider.q[0, 0] > 4 * math.tau and provider.valid.all()


def test_no_wrap_feedback_is_bitwise_identical_without_accumulation_drift():
    _, limits = lookup()
    provider = ContinuousEncoderAngles(2, "cpu", limits)
    nominal = torch.tensor([[-.4, .3, 0., .4, -.3, -0.]]).repeat(2, 1)
    provider.reset([0, 1], nominal)
    # Include sub-ULP-size changes, reversals and repeated samples across many
    # control ticks. The integer provider must never integrate floating noise.
    for tick in range(3000):
        raw = nominal + torch.tensor([[1., 1., 0., -1., -1., 0.]]) * (math.sin(tick * .037) * .25)
        raw[0, 2], raw[0, 5] = 0., -0.
        assert torch.equal(provider.project(raw).view(torch.int32), raw.view(torch.int32))
        assert torch.equal(provider.update(raw).view(torch.int32), raw.view(torch.int32))
    assert provider.winding.eq(0).all()
    for _ in range(1000):
        provider.update(raw)
    assert torch.equal(provider.q.view(torch.int32), raw.view(torch.int32))


@pytest.mark.parametrize("turns", [-2, -1, 1, 2])
def test_independent_two_pi_or_four_pi_wrap_is_an_integer_offset(turns):
    _, limits = lookup()
    provider = ContinuousEncoderAngles(1, "cpu", limits)
    initial = torch.tensor([[-.4, .3, 0., .4, -.3, 0.]])
    provider.reset([0], initial)
    physical = initial + .001
    raw = physical.clone()
    raw[:, 0] += turns * math.tau
    raw[:, 4] -= turns * math.tau
    before = provider.winding.clone()
    projected = provider.project(raw)
    assert torch.equal(provider.winding, before)
    q = provider.update(raw)
    torch.testing.assert_close(q, physical, atol=1e-6, rtol=0.)
    assert provider.winding.tolist() == [[-turns, 0, 0, 0, turns, 0]]
    assert torch.equal(q, projected) and provider.valid.all()
    for _ in range(1000):
        provider.update(raw)
    assert torch.equal(provider.q, projected)


def test_nonfinite_history_remains_invalid_until_local_reset():
    _, limits = lookup()
    provider = ContinuousEncoderAngles(2, "cpu", limits)
    raw = torch.tensor([[-.4, .3, 0., .4, -.3, 0.]]).repeat(2, 1)
    provider.reset([0, 1], raw)
    corrupt = raw.clone()
    corrupt[0, 0] = float("nan")
    provider.update(corrupt)
    for _ in range(3):
        provider.update(raw)
    provider.realign(torch.tensor([0]))
    assert provider.valid.tolist() == [[False, False], [True, True]]
    assert torch.isnan(provider.q[0, 0]) and provider.winding.eq(0).all()
    provider.reset([0], raw)
    assert provider.valid.all() and torch.equal(provider.q, raw)


@pytest.mark.parametrize("limits", [[[0., math.tau], [-1., 0.]], [[1., 0.], [-1., 0.]],
    [[0., float("nan")], [-1., 0.]]])
def test_ambiguous_or_invalid_frozen_branch_is_rejected(limits):
    with pytest.raises(ValueError, match="branches"):
        ContinuousEncoderAngles(1, "cpu", limits)
