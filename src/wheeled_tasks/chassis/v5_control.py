"""V5 motor-output coordinates and passive gas-spring forces, independent of Isaac."""
from __future__ import annotations

import torch

from wheeled_tasks.v40.core import motor_torque_limit


class V5Control:
    ACTIVE = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
    LEGS = (0, 1, 3, 4)
    WHEELS = (2, 5)

    def __init__(self, manifest, spec, fit, prior, settings, device):
        if tuple(manifest["control_joint_names"]) != self.ACTIVE:
            raise ValueError("V5 requires the two root-driven leg axes per side, not virtual knee actuation")
        self.nominal = torch.tensor([manifest["nominal_joint_pos"][n] for n in self.ACTIVE], device=device)
        self.spring_names = manifest["spring_joint_names"]
        self.s0 = torch.tensor([spec["spring_binding"][n]["compression_at_q_zero_m"] for n in self.spring_names], device=device)
        self.stroke = torch.tensor([spec["spring_binding"][n]["stroke_m"] for n in self.spring_names], device=device)
        self.coefficients = fit["monomial_coefficients_n"]
        self.wheel_prior = prior["actuators"]["wheel"]
        self.settings = settings

    def decode(self, actions, q):
        clipped = actions.clamp(-self.settings["action_clip"], self.settings["action_clip"])
        desired = self.nominal[list(self.LEGS)] + clipped[:, list(self.LEGS)] * self.settings["leg_position_scale"]
        delta = desired - q[:, list(self.LEGS)]
        legs = q[:, list(self.LEGS)] + torch.atan2(delta.sin(), delta.cos())
        wheels = clipped[:, list(self.WHEELS)] * self.settings["wheel_velocity_scale"]
        return legs, wheels, clipped

    def motor_efforts(self, q, dq, legs, wheels):
        tau = torch.zeros_like(q)
        tau[:, list(self.LEGS)] = (self.settings["leg_kp"] * (legs - q[:, list(self.LEGS)])
                                  - self.settings["leg_kd"] * dq[:, list(self.LEGS)]).clamp(-40., 40.)
        wheel = self.wheel_prior["kd"] * (wheels - dq[:, list(self.WHEELS)])
        bound = motor_torque_limit(dq[:, list(self.WHEELS)], self.wheel_prior)
        tau[:, list(self.WHEELS)] = torch.maximum(torch.minimum(wheel, bound), -bound)
        return tau

    def spring_state(self, position, velocity):
        return self.s0 - position, -velocity

    def spring_efforts(self, position):
        compression = self.s0 - position
        # Same explicit research extrapolation as the bounded MuJoCo test.
        # Force is held at the curve endpoint during numerical stop penetration;
        # mechanical limits and episode termination handle actual overtravel.
        u = (compression / self.stroke).clamp(0., 1.)
        a, b, c, d = self.coefficients
        return a + u * (b + u * (c + u * d))

    def proprioception(self, omega, gravity, commands, q, dq, previous_actions):
        delta = q[:, list(self.LEGS)] - self.nominal[list(self.LEGS)]
        delta = torch.atan2(delta.sin(), delta.cos())
        return torch.cat((omega * .5, gravity, commands * commands.new_tensor([1., 1., 5.]),
                          delta, dq * .1, previous_actions), -1).clamp(-100., 100.)
