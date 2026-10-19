"""V5 motor-output coordinates and passive gas-spring forces, independent of Isaac."""
from __future__ import annotations

import hashlib
import json

import torch

from wheeled_tasks.chassis.control_core import motor_torque_limit


class V5Control:
    ACTIVE = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
    LEGS = (0, 1, 3, 4)
    WHEELS = (2, 5)
    BRANCH107_MODEL_SPEC_SHA256 = "0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997"
    BRANCH107_MODEL_SPEC_CANONICAL_SHA256 = "38bcb0130992cfc1a39cd5b8bd83953b1c9c1d8319f4544a1f84ef945f072d5e"
    # The signed coordinates are left auxiliary-minus-hip and right
    # hip-minus-auxiliary. The sealed branch's axis-offset geometry gives
    # this inner intersection over all common rotations, not motor limits.
    BRANCH107_SIGNED_DELTA_BOUNDS_RAD = (
        (-0.14062820701680934, 1.3692123767724917),
        (-0.1406283965709381, 1.3692125038958158),
    )

    def __init__(self, manifest, spec, fit, prior, settings, device):
        if tuple(manifest["control_joint_names"]) != self.ACTIVE:
            raise ValueError("V5 requires the two root-driven leg axes per side, not virtual knee actuation")
        self.nominal = torch.tensor([manifest["nominal_joint_pos"][n] for n in self.ACTIVE], device=device)
        self.spring_names = manifest["spring_joint_names"]
        self.s0 = torch.tensor([spec["spring_binding"][n]["compression_at_q_zero_m"] for n in self.spring_names], device=device)
        self.stroke = torch.tensor([spec["spring_binding"][n]["stroke_m"] for n in self.spring_names], device=device)
        self.coefficients = fit["monomial_coefficients_n"]
        by_name = {j["name"]: j for j in spec["joints"]}
        self.knee_bounds = torch.tensor([[float(by_name[n]["limit"][k]) for k in ("lower", "upper")]
                                        for n in ("L_joint2", "R_jonit2")], device=device)
        self.wheel_prior = prior["actuators"]["wheel"]
        self.settings = settings
        self.leg_winding_version = settings.get("leg_winding_version", "independent_nearest_v1")
        if self.leg_winding_version not in (
                "independent_nearest_v1", "paired_common_v1", "paired_branch107_v1"):
            raise ValueError("Unknown leg target winding version")
        if self.leg_winding_version == "paired_branch107_v1":
            spec_sha = hashlib.sha256(json.dumps(
                spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if (manifest.get("files_sha256", {}).get("model_spec.json") != self.BRANCH107_MODEL_SPEC_SHA256
                    or manifest.get("knee_inner_limits_deg") != [40., 107.]
                    or spec_sha != self.BRANCH107_MODEL_SPEC_CANONICAL_SHA256):
                raise ValueError("Branch107 targets require the sealed 40..107 model geometry")
            self._branch107_bounds = torch.tensor(
                self.BRANCH107_SIGNED_DELTA_BOUNDS_RAD, dtype=torch.float64, device=device)
            self._branch107_sign = torch.tensor([1., -1.], dtype=torch.float64, device=device)
        wheel_clip = settings.get("wheel_action_clip", settings["action_clip"])
        self.action_bounds = torch.tensor([settings["action_clip"], settings["action_clip"], wheel_clip,
                                          settings["action_clip"], settings["action_clip"], wheel_clip], device=device)
        self.pc_kp = self.pc_kd = self.effort_scale = None

    def set_pc_parameters(self, kp, kd, effort_scale):
        """Bind actual per-environment PC gains in ACTIVE order, without copies.

        Native actuators have zero stiffness because this controller computes
        torque explicitly. The critic must see these same tensors.
        """
        if kp.ndim != 2 or kp.shape[-1] != 6 or kd.shape != kp.shape or effort_scale.shape != kp.shape:
            raise ValueError("PC parameters must use (environments,6) ACTIVE order")
        if not all(bool(torch.isfinite(value).all()) for value in (kp, kd, effort_scale)):
            raise ValueError("PC parameters must be finite")
        if bool((kp < 0).any() | (kd < 0).any() | (effort_scale <= 0).any()):
            raise ValueError("Invalid PC gains/effort scale")
        self.pc_kp, self.pc_kd, self.effort_scale = kp, kd, effort_scale

    def decode(self, actions, q):
        clipped = actions.clamp(-self.action_bounds, self.action_bounds)
        desired = self.nominal[list(self.LEGS)] + clipped[:, list(self.LEGS)] * self.settings["leg_position_scale"]
        current = q[:, list(self.LEGS)]
        if self.leg_winding_version == "paired_branch107_v1":
            legs = self._branch107_targets(desired, current)
        elif self.leg_winding_version == "paired_common_v1":
            # Both root-driven axes share a common full revolution. Choosing
            # their branches separately can reverse the relative knee motion
            # at the angular cut even when both endpoint pairs are legal.
            pairs = desired.unflatten(-1, (2, 2))
            offset = (current.unflatten(-1, (2, 2)) - pairs).mean(-1, keepdim=True)
            turns = torch.floor(offset / (2 * torch.pi) + .5)
            legs = (pairs + turns * (2 * torch.pi)).flatten(-2)
        else:
            delta = desired - current
            legs = current + torch.atan2(delta.sin(), delta.cos())
        wheels = clipped[:, list(self.WHEELS)] * self.settings["wheel_velocity_scale"]
        return legs, wheels, clipped

    def _branch107_targets(self, desired, current):
        """Optional pair-domain semantics; retain the raw mean before winding.

        Only this version uses float64 intermediate arithmetic. Native target
        rounding can move a pair difference outside the geometric interval.
        Rebuilding just its auxiliary endpoint, then taking one ULP toward the
        unchanged hip, moves that difference inward. Zero is in both intervals,
        so coarse representations may safely collapse a pair to equal values.
        This changes PD preload; it is not a passive-stop or physics repair.
        """
        if desired.dtype not in (torch.float32, torch.float64) or current.dtype not in (
                torch.float32, torch.float64):
            raise ValueError("Branch107 targets require float32 or float64 inputs")
        pairs = desired.double().unflatten(-1, (2, 2))
        feedback = current.double().unflatten(-1, (2, 2))
        mean = pairs.mean(-1)
        lo, hi = self._branch107_bounds[:, 0], self._branch107_bounds[:, 1]
        delta = (self._branch107_sign * (pairs[..., 1] - pairs[..., 0])).clamp(lo, hi)
        projected = torch.stack((mean - self._branch107_sign * delta * .5,
                                 mean + self._branch107_sign * delta * .5), -1)
        feedback_mean = feedback[..., 0] * .5 + feedback[..., 1] * .5
        turns = torch.floor((feedback_mean - mean) / (2 * torch.pi) + .5)
        native = (projected + turns[..., None] * (2 * torch.pi)).to(desired.dtype)
        hip = native[..., 0]
        actual_delta = self._branch107_sign * (native[..., 1].double() - hip.double())
        outside = (actual_delta < lo) | (actual_delta > hi)
        repaired = (hip.double() + self._branch107_sign * delta).to(desired.dtype)
        # After the inward ULP, the repaired difference lies between zero
        # and the projected delta. That whole segment is inside either bound.
        repaired = torch.nextafter(repaired, hip)
        auxiliary = torch.where(outside, repaired, native[..., 1])
        return torch.stack((hip, auxiliary), -1).flatten(-2)

    def motor_efforts(self, q, dq, legs, wheels):
        # ACTIVE is two contiguous (hip, knee, wheel) triples. Views avoid
        # gathering/scattering joint subsets on every 1 kHz control tick.
        requested = torch.empty_like(q)
        pairs = requested.unflatten(-1, (2, 3))
        kp = self.settings["leg_kp"] if self.pc_kp is None else self.pc_kp.unflatten(-1, (2, 3))[:, :, :2]
        kd = self.settings["leg_kd"] if self.pc_kd is None else self.pc_kd.unflatten(-1, (2, 3))[:, :, :2]
        pairs[:, :, :2] = (kp * (legs.unflatten(-1, (2, 2))
                                                  - q.unflatten(-1, (2, 3))[:, :, :2])
                          - kd * dq.unflatten(-1, (2, 3))[:, :, :2])
        if self.effort_scale is not None:
            pairs[:, :, :2] *= self.effort_scale.unflatten(-1, (2, 3))[:, :, :2]
        tau = torch.empty_like(q)
        tau.unflatten(-1, (2, 3))[:, :, :2] = pairs[:, :, :2].clamp(-40., 40.)
        wheel_speed = dq[:, 2::3]
        wheel_kd = self.wheel_prior["kd"] if self.pc_kd is None else self.pc_kd[:, 2::3]
        wheel = wheel_kd * (wheels - wheel_speed)
        if self.effort_scale is not None:
            wheel *= self.effort_scale[:, 2::3]
        bound = motor_torque_limit(wheel_speed, self.wheel_prior)
        tau[:, 2::3] = torch.maximum(torch.minimum(wheel, bound), -bound)
        requested[:, 2::3] = wheel
        self.requested_motor_effort = requested
        self.current_motor_bounds = torch.full_like(q, 40.)
        self.current_motor_bounds[:, 2::3] = bound
        # These three outputs must remain independent and fresh: recovery and
        # transport mutate the diagnostics, and the caller scales torque in place.
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

    def working_margin_risk(self, knee_positions, spring_positions, margin_rad):
        compression = self.s0 - spring_positions
        spring_risk = ((compression / self.stroke - .9) / .1).clamp_min(0)
        lo, hi = self.knee_bounds[:, 0], self.knee_bounds[:, 1]
        joint_risk = torch.maximum((lo + margin_rad - knee_positions) / margin_rad,
                                  (knee_positions - hi + margin_rad) / margin_rad).clamp_min(0)
        return torch.maximum(spring_risk, joint_risk)

    def proprioception(self, omega, gravity, commands, q, dq, previous_actions):
        delta = q[:, list(self.LEGS)] - self.nominal[list(self.LEGS)]
        delta = torch.atan2(delta.sin(), delta.cos())
        return torch.cat((omega * .5, gravity, commands * commands.new_tensor([1., 1., 5.]),
                          delta, dq * .1, previous_actions), -1).clamp(-100., 100.)
