"""Edge classification, hold timing and mechanical outcome precedence."""
import pytest
import torch

from wheeled_tasks.chassis.skill_commands import profile_command
from wheeled_tasks.chassis.step_assist import StepHeightAssist
from wheeled_tasks.chassis.task import episode_outcome, terrain_surfaces


CONFIG = {"step_range_m": [.15, .25], "height_bias_m": .05, "height_max_m": .43,
          "hold_seconds": 2., "lookahead_m": .1, "edge_tolerance_m": 1e-5}


def inputs(count):
    return (torch.full((count,), .305), torch.zeros(count, 2), torch.ones(count, 2, dtype=torch.bool),
            torch.tensor([[1., 0.]]).repeat(count, 1), torch.ones(count, dtype=torch.bool),
            torch.ones(count, dtype=torch.bool))


def test_inclusive_endpoints_first_frame_and_wall_classification():
    assist = StepHeightAssist(6, "cpu", .02, CONFIG)
    raw, samples, valid, direction, eligible, enabled = inputs(6)
    assist.update(raw, samples, valid, direction, eligible, enabled)
    samples[:] = torch.tensor([.149, .15, .20, .25, .251, 1.])[:, None]
    height = assist.update(raw, samples, valid, direction, eligible, enabled)
    assert assist.triggered.tolist() == [False, True, True, True, False, False]
    assert assist.wall.tolist() == [False, False, False, False, True, True]
    torch.testing.assert_close(height, torch.tensor([.305, .355, .355, .355, .305, .305]))
    assist.reset(torch.arange(6))
    assist.update(raw, samples, valid, direction, eligible, enabled)
    assert not assist.triggered.any() and not assist.wall.any()


def test_hold_is_two_seconds_without_bias_accumulation_and_respects_new_raw_height():
    assist = StepHeightAssist(1, "cpu", .02, CONFIG)
    args = inputs(1)
    assist.update(*args)
    args[1][:] = .2
    assert assist.update(*args).item() == pytest.approx(.355)
    for _ in range(99):
        assert assist.update(*args).item() == pytest.approx(.355)
    assert assist.update(*args).item() == pytest.approx(.305)
    args[1][:] += .2
    args[0][:] = .42
    assert assist.update(*args).item() == pytest.approx(.43)
    args[0][:] = .3
    assert assist.update(*args).item() == pytest.approx(.35)


def test_missing_samples_stop_and_direction_change_are_not_new_edges():
    assist = StepHeightAssist(2, "cpu", .02, CONFIG)
    args = inputs(2)
    assist.update(*args)
    args[1][:] = .2
    assist.update(*args)
    args[2][0] = False
    args[1][0] = float("nan")
    args[3][1] = 0.
    args[4][1] = False
    output = assist.update(*args)
    assert torch.isfinite(output).all() and assist.active.all()
    args[1][0] = .4
    args[2][0] = True
    args[3][1] = torch.tensor([-1., 0.])
    args[4][1] = True
    assist.update(*args)
    assert not assist.triggered.any()
    assert assist.active.tolist() == [True, False]


def test_partial_reset_and_unassisted_lane_do_not_change_other_hold():
    assist = StepHeightAssist(3, "cpu", .02, CONFIG)
    args = inputs(3)
    args[-1][2] = False
    assist.update(*args)
    args[1][:] = .15
    assist.update(*args)
    assert assist.active.tolist() == [True, True, False]
    assist.reset(torch.tensor([0]))
    remaining = assist.remaining[1].clone()
    assist.update(*args, ids=torch.tensor([0]), advance=False)
    assert assist.remaining[1] == remaining and assist.active.tolist() == [False, True, False]


@pytest.mark.parametrize("height", [.03, .06, .10, .15, .20, .25])
def test_step_platform_is_solid_to_below_lower_ground(height):
    surfaces = terrain_surfaces("step_up", {"step_up_m": height}, 1.)
    for surface in surfaces:
        size, center, _ = surface.box(solid_bottom=-.1)
        assert center[2] - size[2] / 2 == pytest.approx(-.1)
        assert center[2] + size[2] / 2 == pytest.approx(surface.z0)
    assert surfaces[1].z0 - surfaces[0].z0 == pytest.approx(height)


def test_wall_is_truncation_not_success_and_does_not_hide_mechanical_failure():
    reasons = {"fall": torch.tensor([False, True, False]), "knee": torch.tensor([False, False, True]),
               "boundary": torch.zeros(3, dtype=torch.bool), "blocked": torch.ones(3, dtype=torch.bool)}
    terminated, timeouts, success = episode_outcome(reasons, torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))
    assert terminated.tolist() == [False, True, True]
    assert timeouts.tolist() == [True, False, False]
    assert not success.any()


def test_height_pulse_matches_assist_hold_and_restores_raw_command():
    profile = {"kind": "height_pulse", "height_pulse": {"bias_m": .05, "maximum_m": .43,
               "start_seconds": 2., "hold_seconds": 2., "period_seconds": 6.}}
    values = profile_command(torch.tensor([0., 1.98, 2., 3.98, 4., 7.98, 8.]), [0., 0., .305], profile)
    torch.testing.assert_close(values[:, 2], torch.tensor([.305, .305, .355, .355, .305, .305, .355]))
