"""Mechanical-angle adaptation checks, independent of native skill qualification."""
from dataclasses import replace
import math

import pytest
import torch

from wheeled_tasks.chassis.final_rough_reward import (
    V7FinalRoughPhysicalFrame, final_rough_reward, v7_final_rough_reward,
)
from test_v7_final_rough_reward import frame


LIMITS = (math.radians(40.), math.radians(107.))


def chassis_frame(source, angles):
    return V7FinalRoughPhysicalFrame(
        base=source.base, airborne=source.airborne,
        root_height_world_m=source.root_height_world_m,
        ground_height_world_m=source.ground_height_world_m,
        wheel_contact_elapsed_s=source.wheel_contact_elapsed_s,
        base_contact_elapsed_s=source.base_contact_elapsed_s,
        knee_inner_angles_rad=angles,
        wheel_positions_heading_m=source.wheel_positions_heading_m)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_inner_angle_soft_window_keeps_radian_units_and_source_weight(dtype):
    source = frame(4, dtype)
    angles = torch.deg2rad(torch.tensor(
        [[40., 107.], [46.7, 103.65], [60., 60.], [60., 20.]], dtype=dtype))
    actual = v7_final_rough_reward(chassis_frame(source, angles), knee_inner_limits_rad=LIMITS)
    # Independently specified degrees, not an affine projection onto source rear2.
    expected = torch.tensor([10.05, 0., 0., 26.7], dtype=dtype) * (math.pi / 180.)
    torch.testing.assert_close(actual.raw_terms["airborne_joint_pos_limits"], expected,
                               rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(actual.payments["airborne_joint_pos_limits"], expected * -.2,
                               rtol=1e-6, atol=5e-8)
    original = final_rough_reward(source)
    for name in original.payments:
        if name != "airborne_joint_pos_limits":
            assert torch.equal(actual.raw_terms[name], original.raw_terms[name])
            assert torch.equal(actual.payments[name], original.payments[name])
    assert len(actual.payments) == 31


def test_relative_angle_term_is_airborne_only_and_numeric_rows_still_pay_zero():
    source = frame(3)
    source = replace(source, airborne=torch.tensor([False, True, True]),
        base=replace(source.base, numerical_safety_reset=torch.tensor([False, False, True])))
    value = chassis_frame(source, torch.deg2rad(torch.tensor([[20., 140.]] * 3)))
    result = v7_final_rough_reward(value, knee_inner_limits_rad=LIMITS)
    assert result.payments["airborne_joint_pos_limits"][0] == 0.
    assert result.payments["airborne_joint_pos_limits"][1] < 0.
    assert result.total[2] == 0.
    assert all(payment[2] == 0. for payment in result.payments.values())


def test_source_and_chassis_angle_frames_cannot_be_silently_interchanged():
    source = frame()
    value = chassis_frame(source, torch.full((1, 2), math.radians(60.)))
    with pytest.raises(TypeError, match="source-coordinate"):
        final_rough_reward(value)
    with pytest.raises(TypeError, match="inner-angle"):
        v7_final_rough_reward(source, knee_inner_limits_rad=LIMITS)


@pytest.mark.parametrize("limits", [None, (), (0., 1.), (1., 1.), (1., math.pi),
                                   (1., float("nan")), (True, 2.), (40., 107.)])
def test_inner_limits_require_explicit_physical_radians(limits):
    value = chassis_frame(frame(), torch.full((1, 2), math.radians(60.)))
    with pytest.raises(ValueError, match="knee"):
        v7_final_rough_reward(value, knee_inner_limits_rad=limits)


def test_nonfinite_inner_angle_is_not_repaired_or_allowed_to_erase_other_terms():
    source = frame()
    value = chassis_frame(source, torch.tensor([[float("nan"), math.radians(60.)]]))
    result = v7_final_rough_reward(value, knee_inner_limits_rad=LIMITS)
    assert torch.isnan(result.raw_terms["airborne_joint_pos_limits"]).item()
    assert result.payments["airborne_joint_pos_limits"] == 0.
    assert result.payments["track_lin_vel_xy"] != 0.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("airborne", [False, True])
def test_relative_height_payment_is_invariant_to_world_ground_elevation(dtype, airborne):
    source = frame(3, dtype)
    ground = torch.tensor([0., .2, .4], dtype=dtype)
    source = replace(source, airborne=torch.full((3,), airborne, dtype=torch.bool),
        root_height_world_m=ground + .32, ground_height_world_m=ground,
        base=replace(source.base, target_height_m=torch.full((3,), .32, dtype=dtype)))
    adapted = v7_final_rough_reward(chassis_frame(source,
        torch.full((3, 2), math.radians(60.), dtype=dtype)), knee_inner_limits_rad=LIMITS)
    original = final_rough_reward(source)
    torch.testing.assert_close(adapted.payments["track_height_exp_tight"],
                               torch.full((3,), .02, dtype=dtype), rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(adapted.payments["track_height_square"],
                               torch.zeros(3, dtype=dtype), rtol=0., atol=1e-12)
    assert original.payments["track_height_exp_tight"][1] < .001
    assert original.payments["track_height_exp_tight"][2] < 1e-10
    for name in original.payments:
        if name not in ("airborne_joint_pos_limits", "track_height_exp_tight", "track_height_square"):
            assert torch.equal(adapted.raw_terms[name], original.raw_terms[name])
            assert torch.equal(adapted.payments[name], original.payments[name])


def test_relative_height_clipping_retains_source_bounds_without_changing_actual_height():
    source = frame(4, torch.float64)
    heights = torch.tensor([.01, .05, .45, .6], dtype=torch.float64)
    source = replace(source, root_height_world_m=heights + .4,
        ground_height_world_m=torch.full((4,), .4, dtype=torch.float64))
    value = chassis_frame(source, torch.full((4, 2), math.radians(60.), dtype=torch.float64))
    result = v7_final_rough_reward(value, knee_inner_limits_rad=LIMITS)
    assert torch.equal(result.payments["track_height_exp_tight"][:1],
                       result.payments["track_height_exp_tight"][1:2])
    assert torch.equal(result.payments["track_height_exp_tight"][2:3],
                       result.payments["track_height_exp_tight"][3:])
    assert torch.equal(value.root_height_world_m, heights + .4)
