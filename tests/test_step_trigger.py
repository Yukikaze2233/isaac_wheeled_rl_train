"""Deployable trigger and FK/IMU estimates; no simulator state is required."""
import math

import pytest
import torch

from wheeled_tasks.chassis.step_trigger import (
    SingleBeamStepTrigger, estimate_height_if_grounded, single_beam_range_2d,
)
from wheeled_tasks.chassis.task import terrain_surfaces


def trigger_update(trigger, distance=.2, *, valid=True, age=.01, forward=.4,
                   height=.2, completed=False, new_step=False):
    def tensor(value, dtype=torch.float32):
        return torch.full((trigger.count,), value, dtype=dtype)
    return trigger.update(tensor(distance), tensor(valid, torch.bool), tensor(age),
                          tensor(forward), tensor(height),
                          completed=tensor(completed, torch.bool),
                          new_step_confirmed=tensor(new_step, torch.bool))


def make_trigger(count=1):
    return SingleBeamStepTrigger(count, "cpu", .25, .5)


def test_trigger_once_through_threshold_jitter_and_missing_echo():
    trigger = make_trigger()
    assert trigger_update(trigger).item()
    for distance, valid in [(.24, True), (.26, True), (.19, True), (0., False), (1., True)]:
        assert not trigger_update(trigger, distance, valid=valid).item()
        assert trigger.busy.item()
    assert trigger.height_m.item() == pytest.approx(.2)


@pytest.mark.parametrize("changes", [
    {"valid": False}, {"age": .1}, {"age": -.01}, {"age": float("nan")},
    {"distance": 0.}, {"distance": float("nan")}, {"distance": .3},
    {"forward": 0.}, {"forward": -.3}, {"height": 0.},
])
def test_trigger_requires_fresh_echo_forward_intent_and_preset_height(changes):
    trigger = make_trigger()
    assert not trigger_update(trigger, **changes).item()
    assert not trigger.busy.item()


def test_completion_near_obstacle_does_not_rearm_without_new_step_confirmation():
    trigger = make_trigger()
    trigger_update(trigger)
    assert not trigger_update(trigger, completed=True).item()
    assert not trigger.busy.item()
    for _ in range(4):
        assert not trigger_update(trigger).item()
    assert not trigger_update(trigger, 0., valid=False).item()
    assert not trigger.armed.item()
    assert trigger_update(trigger, new_step=True, height=.25).item()
    assert trigger.height_m.item() == pytest.approx(.25)


def test_completion_requires_later_tick_even_with_new_step_and_far_echo_rearms():
    trigger = make_trigger()
    trigger_update(trigger)
    assert not trigger_update(trigger, completed=True, new_step=True).item()
    assert trigger_update(trigger).item()
    assert not trigger_update(trigger, .7, completed=True).item()
    assert trigger.armed.item()
    assert trigger_update(trigger).item()


def test_partial_reset_preserves_other_environment_latch():
    trigger = make_trigger(2)
    assert trigger_update(trigger).tolist() == [True, True]
    trigger.reset(torch.tensor([0]))
    assert trigger_update(trigger).tolist() == [True, False]


def profile(count=1):
    surfaces = terrain_surfaces("stairs", {
        "stair_heights_m": [.2, .45], "stair_tread_m": [.8, .6],
    }, 1.)
    data = torch.tensor([[s.x0, s.x1, s.z0, s.slope] for s in surfaces])
    return data[None].expand(count, -1, -1), torch.ones(count, len(surfaces), dtype=torch.bool)


def test_single_ray_distinguishes_first_and_second_risers_with_800mm_spacing():
    surfaces, valid = profile(3)
    position = torch.tensor([[-.3, .15], [-.3, .30], [-.3, .6]])
    distance, hit = single_beam_range_2d(position, torch.zeros(3), surfaces, valid)
    assert hit.tolist() == [True, True, False]
    assert distance.tolist() == pytest.approx([.3, 1.1, 0.])


def test_sensor_extrinsics_pitch_and_nearest_ground_intersection():
    surfaces, valid = profile(2)
    position = torch.tensor([[-1., .3], [-1., .3]])
    pitch = torch.tensor([0., math.pi / 4])
    distance, hit = single_beam_range_2d(
        position, pitch, surfaces, valid, sensor_origin_b=(.1, -.15),
    )
    assert hit.tolist() == [True, True]
    assert distance[0].item() == pytest.approx(.9)
    # Independently rotated sensor: z=.3-(.1+.15)/sqrt(2).
    expected = (.3 - .25 / math.sqrt(2)) * math.sqrt(2)
    assert distance[1].item() == pytest.approx(expected)


def test_ray_invalid_surface_padding_and_range_limit_do_not_make_false_hits():
    surfaces, valid = profile()
    position = torch.tensor([[-.3, .15]])
    distance, hit = single_beam_range_2d(position, torch.zeros(1), surfaces, valid, max_range_m=.2)
    assert not hit.item() and distance.item() == 0.
    valid[:, 1:] = False
    distance, hit = single_beam_range_2d(position, torch.zeros(1), surfaces, valid)
    assert not hit.item() and distance.item() == 0.


def test_fk_height_candidates_stay_per_side_and_airborne_is_invalid():
    wheels = torch.tensor([[[.1, .15, -.2], [.1, -.15, -.3]]])
    gravity = torch.tensor([[0., 0., -1.]])
    height, valid = estimate_height_if_grounded(
        wheels, gravity, .06, grounded=torch.tensor([[True, False]]),
    )
    assert height.tolist()[0] == pytest.approx([.26, .36])
    assert valid.tolist() == [[True, False]]


def test_fk_height_is_pitch_compensated_and_invalid_imu_is_masked():
    # Rotate a fixed world-down wheel vector into body coordinates at +30deg.
    angle = math.pi / 6
    gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)], [0., 0., -2.]])
    wheels = .24 * gravity[:, None, :].expand(-1, 2, -1)
    height, valid = estimate_height_if_grounded(
        wheels, gravity, .06, grounded=torch.ones(2, 2, dtype=torch.bool),
    )
    assert height[0].tolist() == pytest.approx([.3, .3])
    assert valid.tolist() == [[True, True], [False, False]]


def test_fk_radius_projects_for_roll_and_sensor_validity_remains_per_side():
    gravity = torch.tensor([[0., .5, -math.sqrt(.75)]])
    wheels = .24 * gravity[:, None, :].expand(-1, 2, -1)
    height, valid = estimate_height_if_grounded(
        wheels, gravity, .06, grounded=torch.ones(1, 2, dtype=torch.bool),
        sensor_valid=torch.tensor([[False, True]]),
    )
    assert height.tolist()[0] == pytest.approx([.24 + .06 * math.sqrt(.75)] * 2)
    assert valid.tolist() == [[False, True]]
