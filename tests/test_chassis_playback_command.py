"""Actor-visible command timing, stop/reset behavior and projected pose telemetry."""
import math

import pytest
import torch

from wheeled_tasks.chassis.playback_command import advance_playback_command, attitude_from_gravity
from wheeled_tasks.chassis.teleop import KeyboardCommand


SLEW = {"vx_m_s2": 1.5, "yaw_rad_s2": 4.}
LIMITS = {"wheel_radius_m": .06, "track_width_m": .4373,
          "max_wheel_speed_rad_s": 90., "lateral_acceleration_m_s2": 2.06}


def test_actor_commands_follow_frozen_rates_even_when_requested_targets_jump():
    current = torch.tensor([[0., 0., .305]])
    for target in ([3., 0., .305], [-3., 0., .305], [0., 8., .305], [0., -8., .305]):
        for _ in range(12):
            following = advance_playback_command(current, torch.tensor([target]), .02, SLEW, LIMITS)
            assert torch.all((following[:, :2] - current[:, :2]).abs() <= torch.tensor([.03, .08]) + 1e-6)
            current = following
    first = advance_playback_command(torch.zeros(1, 3), torch.tensor([[3., 1., .305]]), .02, SLEW, LIMITS)
    torch.testing.assert_close(first, torch.tensor([[.03, .08, .305]]))


def test_space_stop_decelerates_applied_command_but_reset_clears_history():
    keyboard = KeyboardCommand(6., 16.)
    keyboard.vx, keyboard.yaw = 2., -1.
    keyboard.stop()
    request = torch.tensor([keyboard.advance(.02)])
    stopped = advance_playback_command(torch.tensor([[2., -1., .305]]), request, .02, SLEW, LIMITS)
    torch.testing.assert_close(stopped, torch.tensor([[1.97, -.92, .305]]))
    fresh = advance_playback_command(torch.tensor([[0., 0., .305]]), request, .02, SLEW, LIMITS)
    torch.testing.assert_close(fresh, torch.tensor([[0., 0., .305]]))


def test_feasibility_projection_cannot_bypass_yaw_rate_near_lateral_limit():
    current = torch.tensor([[.10, 18., .305], [-.10, -18., .305]])
    request = torch.tensor([[5., 18., .305], [-5., -18., .305]])
    for _ in range(80):
        following = advance_playback_command(current, request, .02, SLEW, LIMITS)
        assert torch.all((following[:, :2] - current[:, :2]).abs() <= torch.tensor([.03, .08]) + 1e-6)
        assert torch.all((following[:, 0] * following[:, 1]).abs() <= 2.06 + 1e-5)
        assert torch.all(following[:, 0].abs() + .4373 * following[:, 1].abs() / 2 <= 5.4 + 1e-5)
        current = following


@pytest.mark.parametrize("gravity,expected", [
    ([0., 0., -1.], (0., 0.)),
    ([.5, 0., -math.sqrt(.75)], (0., math.pi / 6)),
    ([0., -.5, -math.sqrt(.75)], (math.pi / 6, 0.)),
])
def test_attitude_uses_measured_projected_gravity_and_known_body_axes(gravity, expected):
    assert attitude_from_gravity(gravity) == pytest.approx(expected)


def test_nonfinite_command_is_rejected_before_actor_input():
    with pytest.raises(ValueError, match="finite"):
        advance_playback_command(torch.zeros(1, 3), torch.tensor([[float("nan"), 0., .305]]), .02, SLEW)
