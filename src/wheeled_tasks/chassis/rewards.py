"""Chassis reward densities and bounded phase-aware stationary objectives.

Reference: SCUTRobotLab (MIT), b8ff79f, V14 env_cfg.py:869-914 and
wheelbipe25_v3/env.py:3505-3833. Values returned here are per-second densities.
The V5 environment retains its own real travel/closure/contact failure rules.
"""
import torch

from .v5_control import V5Control


class StationaryAnchor:
    """Bounded residence reward; the anchor is privileged training state only."""
    def __init__(self, count, device, weight, band_m):
        if weight < 0 or band_m <= 0:
            raise ValueError("Stationary reward requires nonnegative weight and positive distance band")
        self.weight, self.band_m = weight, band_m
        self.position = torch.zeros(count, 2, device=device)
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.valid[ids] = False

    def reward(self, position, requested, supported):
        self.valid &= requested
        entering = requested & supported & ~self.valid
        self.position.copy_(torch.where(entering[:, None], position, self.position))
        self.valid |= entering
        distance_square = (position - self.position).square().sum(-1)
        return self.weight * torch.expm1(-distance_square / self.band_m**2) * requested * supported * self.valid


class CommandedHeightMargin:
    """Penalize excess stop proximity relative to a verified commanded posture."""

    def __init__(self, reference, device):
        self.heights = torch.tensor(reference["height_m"], device=device)
        self.risks = torch.tensor(reference["risk"], device=device)
        if (self.heights.ndim != 1 or len(self.heights) < 2 or self.risks.shape != self.heights.shape
                or not torch.isfinite(self.heights).all() or not torch.isfinite(self.risks).all()
                or not (self.heights.diff() > 0).all() or not ((self.risks >= 0) & (self.risks < 1)).all()):
            raise ValueError("Height margin requires increasing finite heights and risks in [0,1)")

    def excess(self, risk, height):
        height = height.clamp(self.heights[0], self.heights[-1]).contiguous()
        upper = torch.searchsorted(self.heights, height).clamp(1, len(self.heights) - 1)
        fraction = (height - self.heights[upper - 1]) / (self.heights[upper] - self.heights[upper - 1])
        allowed = torch.lerp(self.risks[upper - 1], self.risks[upper], fraction)[:, None]
        # The cost at the mechanical boundary remains one, even near an endpoint target.
        return (risk - allowed).clamp_min(0.) / (1. - allowed).clamp_min(1e-6)


def height_tracking_terms(height_error, velocity_error, support, moving, settings):
    """A wide companion preserves the original narrow kernel's precision signal."""
    return {
        "height_wide_companion": support * settings["wide_weight"] * torch.expm1(
            -height_error.square() / settings["wide_sigma_m"]**2),
        "height_velocity_tracking": support * moving * settings["velocity_weight"] * torch.exp(
            -velocity_error.square() / settings["velocity_sigma_m_s"]**2),
    }


def reward_terms(velocity, omega, gravity, height, commands, motor_velocity, motor_acceleration,
                  motor_torque, actions, previous_actions, before_previous_actions,
                  wheel_positions_b, support, ordinary_motion, undesired_contact, height_kernel_width_m=None):
    leg, wheel = list(V5Control.LEGS), list(V5Control.WHEELS)
    # Project body-forward velocity onto the horizontal plane as in V14.
    cos_pitch = torch.sqrt((1. - gravity[:, 0].square()).clamp_min(0.))
    ev = commands[:, 0] - velocity[:, 0] * cos_pitch
    ew = commands[:, 1] - omega[:, 2]
    eh = height - commands[:, 2]
    gx, gy = gravity[:, 0], gravity[:, 1]
    orientation_scale = 2. * torch.exp(-commands[:, 0].square() / 3.) + 2.
    delta = actions - previous_actions
    second = actions - 2 * previous_actions + before_previous_actions
    fork = wheel_positions_b[:, 0, 0] - wheel_positions_b[:, 1, 0]
    stationary = (commands[:, 0].abs() < .1) * ordinary_motion * support
    height_denominator = .001 if height_kernel_width_m is None else height_kernel_width_m.square()
    return {
        "track_lin_vel": ordinary_motion * torch.exp(-ev.clamp(-1., 1.).square() / .5),
        "lin_vel_square": -(ordinary_motion * (.25 * ev).square()),
        "track_yaw": torch.exp(-ew.clamp(-.8, .8).square() / .25),
        "yaw_square": -(.5 * ew).square(),
        "track_height": support * torch.exp(-eh.clamp(-.15, .15).square() / height_denominator),
        "height_square": -support * (10. * eh).square(),
        "pitch_exp": torch.exp(-gx.square() / .02),
        "roll_exp": torch.exp(-gy.square() / .01),
        "pitch_velocity": -2. * support * (orientation_scale * gx).square(),
        "roll_velocity": -2. * (orientation_scale * gy).square(),
        "stand_translation": -stationary * velocity[:, :2].abs().sum(-1),
        "leg_velocity": -.005 * motor_velocity[:, leg].square().sum(-1),
        "leg_acceleration": -5e-7 * motor_acceleration[:, leg].square().sum(-1),
        "wheel_velocity": -1e-5 * motor_velocity[:, wheel].square().sum(-1),
        "wheel_acceleration": -1e-8 * motor_acceleration[:, wheel].square().sum(-1),
        "motor_torque": -1e-4 * motor_torque.square().sum(-1),
        "wheel_power": -1e-4 * (motor_torque[:, wheel] * motor_velocity[:, wheel]).clamp_min(0).sum(-1),
        "vertical_velocity": -.5 * support * velocity[:, 2].square(),
        "horizontal_angular_velocity": -.05 * omega[:, :2].square().sum(-1),
        "action_rate": -.01 * delta.square().sum(-1),
        "leg_action_smoothness": -.05 * second[:, leg].square().sum(-1),
        "wheel_action_smoothness": -.01 * second[:, wheel].square().sum(-1),
        "no_fork": -(fork.abs() > .05).float(),
        "no_fork_square": -(5. * fork).square(),
        "undesired_contact": -2. * undesired_contact.float(),
    }
