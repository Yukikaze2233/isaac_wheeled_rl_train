"""Finite costs after verified jump flight and the first physical wheel contact.

All rewards are per-second, zero-centered densities. The observer owns no task,
plant, actor observation or request state; its clocks and targets only belong to
reward accounting. Defaults are an engineering candidate, not SCUT calibration.
"""
from copy import deepcopy
import math

import torch

from .task import Phase


JUMP_LANDING_V1 = {
    "enabled": True,
    "version": "jump-contact-cost-v1",
    "zero_tracking_baselines": True,
    "window_seconds": 3.,
    "absorption_seconds": .3,
    "arming_timeout_seconds": 3.,
    "task_outcome_timeout_seconds": 6.,
    "min_clear_air_seconds": .06,
    "absorption_stroke_m": .035,
    "precontact_clearance_m": .10,
    "wheel_radius_m": .06,
    "impact_bodyweight_threshold": 2.,
    "contact_force_min_seconds": .02,
    "contact_force_threshold_n": 1.,
    "precontact_wheel_torque_limit_nm": 3.,
    "height": {"broad_scale": .04, "broad_weight": 3., "fine_scale": .015, "fine_weight": 1.5},
    "velocity": {"broad_scale": .5, "broad_weight": 2., "fine_scale": .10, "fine_weight": .6},
    "lateral": {"broad_scale": .15, "broad_weight": .5, "fine_scale": .08, "fine_weight": .25},
    "yaw": {"broad_scale": 1., "broad_weight": 1., "fine_scale": .2, "fine_weight": .35},
    "orientation": {"broad_scale": .2, "broad_weight": 1., "fine_scale": .10, "fine_weight": .5},
    "angular_velocity": {"broad_scale": .75, "broad_weight": .5, "fine_scale": .25, "fine_weight": .25},
    "vertical_velocity": {"broad_scale": .5, "broad_weight": 1., "fine_scale": .10, "fine_weight": .35},
    "position": {"broad_scale": .5, "broad_weight": .5, "fine_scale": .25, "fine_weight": .1},
    "down_speed": {"broad_scale": .5, "broad_weight": .5, "fine_scale": .10, "fine_weight": .1},
    "impact": {"broad_scale": 1., "broad_weight": .5, "fine_scale": .5, "fine_weight": .1},
    "precontact_wheel_velocity": {"broad_scale": 5., "broad_weight": .5,
                                 "fine_scale": .5, "fine_weight": .25},
    "precontact_wheel_torque": {"broad_scale": 3., "broad_weight": .25,
                               "fine_scale": 1., "fine_weight": .1},
}

_AXES = ("height", "velocity", "lateral", "yaw", "orientation", "angular_velocity",
         "vertical_velocity", "position", "down_speed", "impact", "precontact_wheel_velocity",
         "precontact_wheel_torque")


def validate_jump_landing(settings):
    """Require an explicit finite clock and nonzero broad/local cost scales."""
    if not isinstance(settings, dict) or set(settings) != set(JUMP_LANDING_V1):
        raise ValueError("Jump landing requires exactly its versioned settings")
    if (settings["enabled"] is not True or settings["zero_tracking_baselines"] is not True
            or settings["version"] != JUMP_LANDING_V1["version"]):
        raise ValueError("Jump landing requires enabled zero-centered known cost version")
    for key in set(JUMP_LANDING_V1) - set(_AXES) - {"enabled", "version", "zero_tracking_baselines"}:
        value = settings[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0.:
            raise ValueError(f"Jump landing requires positive finite {key}")
    if settings["absorption_seconds"] >= settings["window_seconds"]:
        raise ValueError("Jump landing absorption must be shorter than its finite window")
    for axis in _AXES:
        cost = settings[axis]
        if not isinstance(cost, dict) or set(cost) != {"broad_scale", "broad_weight", "fine_scale", "fine_weight"}:
            raise ValueError(f"Jump landing requires explicit broad/fine cost keys for {axis}")
        for key, value in cost.items():
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0.:
                raise ValueError(f"Jump landing requires positive finite {axis}.{key}")
    return settings


def _dense_cost(error, settings):
    # Kept independent of rewards.py so that its native integration can import
    # this observer without a circular module dependency.
    square = (error / settings["broad_scale"]).square()
    broad = -settings["broad_weight"] * square / (torch.sqrt(1. + square) + 1.)
    fine = settings["fine_weight"] * torch.expm1(-(error / settings["fine_scale"]).square())
    return broad, fine


class JumpLandingWindow:
    """Latch one bounded contact window per accepted request, including bounces.

    A fresh request snapshots the visible goals. Unsupported FLIGHT, a recorded
    COM release and actual clear air must all be observed before contact can
    start the window. Cancellation before verified flight disarms it; release
    after verified flight cannot erase its outcome. A held/busy repeat cannot renew the
    clock. A later genuinely accepted request from ground may start a new jump.
    """

    def __init__(self, count, device, dt, settings):
        validate_jump_landing(settings)
        if type(count) is not int or count <= 0 or not math.isfinite(dt) or dt <= 0.:
            raise ValueError("Jump landing requires a positive count and finite timestep")
        self.settings = deepcopy(settings)
        self.dt = dt
        self.duration = settings["window_seconds"]
        self.absorption_seconds = settings["absorption_seconds"]
        self._duration_ticks = math.ceil(self.duration / dt - 1e-9)
        self._absorption_ticks = math.ceil(self.absorption_seconds / dt - 1e-9)
        self._arming_ticks = math.ceil(settings["arming_timeout_seconds"] / dt - 1e-9)
        self._outcome_ticks = math.ceil(settings["task_outcome_timeout_seconds"] / dt - 1e-9)
        self.elapsed = torch.zeros(count, device=device)
        self.request_elapsed = torch.zeros_like(self.elapsed)
        self._contact_ticks = torch.zeros(count, dtype=torch.long, device=device)
        self._request_ticks = torch.zeros_like(self._contact_ticks)
        self.wheel_contact_time = torch.zeros(count, 2, device=device)
        self.armed = torch.zeros(count, dtype=torch.bool, device=device)
        self.outcome_active = torch.zeros_like(self.armed)
        self.outcome_timeout = torch.zeros_like(self.armed)
        self.verified_flight = torch.zeros_like(self.armed)
        self.active = torch.zeros_like(self.armed)
        self.touchdown = torch.zeros_like(self.armed)
        self.expired = torch.zeros_like(self.armed)
        self.cycle_accepted = torch.zeros_like(self.armed)
        self.entry_height_clipped = torch.zeros_like(self.armed)
        self.entry_speed_clipped = torch.zeros_like(self.armed)
        self.absorption = torch.zeros_like(self.armed)
        self.airborne = torch.zeros_like(self.armed)
        self.supported = torch.zeros_like(self.armed)
        self._request_previous = torch.zeros_like(self.armed)
        self.weight = torch.zeros_like(self.elapsed)
        self.goal_height = torch.zeros_like(self.elapsed)
        self.goal_commands = torch.zeros(count, 3, device=device)
        self.goal_apex = torch.zeros_like(self.elapsed)
        self.landing_delta = torch.zeros_like(self.elapsed)
        self.origin_xy = torch.zeros(count, 2, device=device)
        self.touchdown_height = torch.zeros_like(self.elapsed)
        self.touchdown_vz = torch.zeros_like(self.elapsed)
        self.com_rise_at_touchdown = torch.zeros_like(self.elapsed)

    def _require(self, value, name, shape, *, boolean=False):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.device != self.elapsed.device or (boolean and value.dtype != torch.bool)):
            kind = "boolean " if boolean else ""
            raise ValueError(f"Jump landing {name} requires {kind}shape {shape} on the observer device")

    def _clear_cost(self, ids):
        for value in (self.elapsed, self._contact_ticks,
                      self.armed, self.verified_flight, self.active, self.touchdown, self.absorption,
                      self.airborne, self.supported, self.weight, self.goal_height, self.goal_commands,
                      self.goal_apex, self.landing_delta, self.origin_xy, self.touchdown_height,
                      self.touchdown_vz, self.com_rise_at_touchdown, self.wheel_contact_time,
                      self.entry_height_clipped, self.entry_speed_clipped):
            value[ids] = 0

    def _clear(self, ids):
        self._clear_cost(ids)
        for value in (self.request_elapsed, self._request_ticks, self.outcome_active):
            value[ids] = 0

    def reset(self, ids):
        self._clear(ids)
        self._request_previous[ids] = False
        self.expired[ids] = False
        self.cycle_accepted[ids] = False
        self.outcome_timeout[ids] = False

    def close(self, done, success=None):
        """Close episode outcomes after the caller has recorded terminal metrics."""
        self._require(done, "done", self.active.shape, boolean=True)
        finished = done if success is None else done | success
        if success is not None:
            self._require(success, "success", self.active.shape, boolean=True)
        self._clear(finished)

    def runtime_state(self):
        return {"enabled": True, "version": self.settings["version"],
                "window_seconds": self.duration,
                "absorption_seconds": self.absorption_seconds,
                "min_clear_air_seconds": self.settings["min_clear_air_seconds"],
                "policy_dt": self.dt,
                "clock_semantics": "first_verified_physical_contact_nonrenewable",
                "zero_tracking_baselines": True, "settings": deepcopy(self.settings)}

    def diagnostics(self):
        height, speed = self.reference()
        return {f"jump_landing_{name}": value.clone() for name, value in (
            ("armed", self.armed), ("verified_flight", self.verified_flight),
            ("active", self.active), ("touchdown", self.touchdown),
            ("elapsed_s", self.elapsed), ("absorption", self.absorption),
            ("expired", self.expired), ("cycle_accepted", self.cycle_accepted),
            ("outcome_active", self.outcome_active), ("outcome_timeout", self.outcome_timeout),
            ("request_elapsed_s", self.request_elapsed),
            ("goal_height_m", self.goal_height), ("reference_height_m", height),
            ("reference_vertical_velocity_m_s", speed),
            ("entry_height_clipped", self.entry_height_clipped),
            ("entry_speed_clipped", self.entry_speed_clipped),
            ("wheel_contact_time_s", self.wheel_contact_time))}

    def update(self, request, phase, contacts, com_released, clear_air_time_peak,
               raw_height, commands, goal_apex, *, landing_delta=None, origin_xy=None,
               com_rise=None, com_vz=None, height=None, accepted_event=None, terminal=None,
               success=None, timeout=None):
        """Observe once per policy tick; return the nonrenewable landing mask.

        raw_height is the raw support-relative height command; height is the
        physical support-relative base height at entry. Wheel contacts may
        precede the nominal LAND command phase. accepted_event is optional for
        protocols that expose an accepted pulse; request remains the live latch.
        """
        count = len(self.elapsed)
        for name, value in (("request", request), ("com_released", com_released)):
            self._require(value, name, (count,), boolean=True)
        for name, value in (("phase", phase), ("clear_air_time_peak", clear_air_time_peak),
                            ("raw_height", raw_height), ("goal_apex", goal_apex)):
            self._require(value, name, (count,))
        self._require(contacts, "contacts", (count, 2), boolean=True)
        self._require(commands, "commands", (count, 3))
        for name, value, shape, boolean in (
                ("landing_delta", landing_delta, (count,), False),
                ("origin_xy", origin_xy, (count, 2), False),
                ("com_rise", com_rise, (count,), False),
                ("com_vz", com_vz, (count,), False),
                ("height", height, (count,), False),
                ("accepted_event", accepted_event, (count,), True),
                ("terminal", terminal, (count,), True),
                ("success", success, (count,), True),
                ("timeout", timeout, (count,), True)):
            if value is not None:
                self._require(value, name, shape, boolean=boolean)
        stopped = torch.zeros_like(self.active)
        for value in (terminal, success, timeout):
            if value is not None:
                stopped |= value
        edge = request & ~self._request_previous
        accepted = edge if accepted_event is None else request & accepted_event
        fresh = accepted & ~self.outcome_active & ~stopped & (
            (phase == Phase.GROUND) | (phase == Phase.TAKEOFF))
        self.expired.zero_()
        self.outcome_timeout.zero_()
        self.cycle_accepted.copy_(fresh)
        self._clear(fresh)
        self.armed |= fresh
        self.outcome_active |= fresh
        self.goal_commands[fresh] = commands[fresh]
        self.goal_height[fresh] = raw_height[fresh]
        self.goal_apex[fresh] = goal_apex[fresh]
        if landing_delta is not None:
            self.landing_delta[fresh] = landing_delta[fresh]
        if origin_xy is not None:
            self.origin_xy[fresh] = origin_xy[fresh]
        supported = contacts.any(-1)
        self.verified_flight |= (self.armed & ~supported & (phase == Phase.FLIGHT)
                                 & com_released & (clear_air_time_peak >= self.settings["min_clear_air_seconds"]))
        first = self.armed & self.verified_flight & supported & ~stopped
        self.touchdown.copy_(first)
        self.active |= first
        self.armed &= ~first
        stroke = self.settings["absorption_stroke_m"]
        entry_height = raw_height if height is None else height
        self.touchdown_height[first] = entry_height[first].clamp(
            min=self.goal_height[first] - stroke, max=self.goal_height[first] + stroke)
        self.entry_height_clipped[first] = self.touchdown_height[first] != entry_height[first]
        entry_vz = (-torch.sqrt(2 * 9.81 * self.goal_apex.clamp_min(0.))
                    if com_vz is None else com_vz)
        # Bounding both Hermite endpoint position and tangent keeps its curve
        # inside the configured compression envelope for the whole .3s segment.
        minimum_vz = torch.maximum(self.touchdown_vz.new_full(self.touchdown_vz.shape,
            -stroke / self.absorption_seconds),
            3 * (self.goal_height - stroke - self.touchdown_height) / self.absorption_seconds)
        self.touchdown_vz[first] = torch.maximum(entry_vz[first], minimum_vz[first]).clamp_max(0.)
        self.entry_speed_clipped[first] = self.touchdown_vz[first] != entry_vz[first]
        if com_rise is not None:
            self.com_rise_at_touchdown[first] = com_rise[first]
        self._contact_ticks += (self.active & ~first).long()
        self._request_ticks += (self.outcome_active & ~fresh).long()
        self.elapsed.copy_((self._contact_ticks * self.dt).clamp_max(self.duration))
        self.request_elapsed.copy_(self._request_ticks * self.dt)
        expired = self.active & (self._contact_ticks >= self._duration_ticks)
        cancelled = self.armed & ~request & ~self.verified_flight
        unlanded = self.armed & (self._request_ticks >= self._arming_ticks)
        outcome_timeout = self.outcome_active & (self._request_ticks >= self._outcome_ticks)
        self._clear_cost(expired | unlanded)
        self._clear(cancelled | outcome_timeout | stopped)
        self.expired.copy_(expired)
        self.outcome_timeout.copy_(outcome_timeout & ~stopped)
        self.wheel_contact_time.copy_(torch.where(
            contacts & self.active[:, None], self.wheel_contact_time + self.dt, 0.))
        self.supported.copy_(supported & (self.active | self.armed))
        self.airborne.copy_(self.armed & self.verified_flight & ~supported)
        self.absorption.copy_(self.active & (self._contact_ticks < self._absorption_ticks))
        self.weight.copy_(self.active)
        self._request_previous.copy_(request)
        return self.weight

    def reference(self):
        """Measured-entry bounded absorption reaches latched height and zero vz."""
        duration = self.absorption_seconds
        u = (self.elapsed / duration).clamp(0., 1.)
        h0 = self.touchdown_height
        goal = self.goal_height
        tangent = self.touchdown_vz * duration
        height = ((2 * u.pow(3) - 3 * u.square() + 1) * h0
                  + (u.pow(3) - 2 * u.square() + u) * tangent
                  + (-2 * u.pow(3) + 3 * u.square()) * goal)
        speed = ((6 * u.square() - 6 * u) * (h0 - goal)
                 + (3 * u.square() - 4 * u + 1) * tangent) / duration
        return height, speed

    def reward_terms(self, height, com_vz, velocity, omega, gravity, wheel_velocity,
                     wheel_torque, contact_peak, weight_force, *, position_xy=None,
                     wheel_clearance=None):
        """Finite settling, physical contact constraints and descending air costs.

        velocity/omega use the body frame; COM vz uses the world vertical axis.
        Wheel velocity uses native left/right motor signs [+1, -1]; the rolling
        target corrects those signs. weight_force is randomized robot mass * g.
        No reward in this method is positive and policy_dt is applied by caller.
        """
        count = len(self.elapsed)
        for name, value, shape in (
                ("height", height, (count,)), ("com_vz", com_vz, (count,)),
                ("velocity", velocity, (count, 3)), ("omega", omega, (count, 3)),
                ("gravity", gravity, (count, 3)), ("wheel_velocity", wheel_velocity, (count, 2)),
                ("wheel_torque", wheel_torque, (count, 2)), ("contact_peak", contact_peak, (count, 2)),
                ("weight_force", weight_force, (count,))):
            self._require(value, name, shape)
        if position_xy is not None:
            self._require(position_xy, "position_xy", (count, 2))
        if wheel_clearance is not None:
            self._require(wheel_clearance, "wheel_clearance", (count, 2))
        h_ref, vz_ref = self.reference()
        planar_vx = velocity[:, 0] * (1. - gravity[:, 0].square()).clamp_min(0.).sqrt()
        tilt = torch.cat((gravity[:, :2], (gravity[:, 2] + 1.).clamp_min(0.)[:, None]), -1).norm(dim=-1)
        errors = {"height": height - h_ref, "vertical_velocity": com_vz - vz_ref,
                  "velocity": planar_vx - self.goal_commands[:, 0],
                  "lateral": velocity[:, 1], "yaw": omega[:, 2] - self.goal_commands[:, 1],
                  "orientation": tilt, "angular_velocity": omega[:, :2].norm(dim=-1)}
        terms = {}
        for axis, error in errors.items():
            broad, fine = _dense_cost(error, self.settings[axis])
            terms[f"jump_landing_{axis}_broad"] = self.weight * broad
            terms[f"jump_landing_{axis}_precision"] = self.weight * fine
        position_error = (torch.zeros_like(height) if position_xy is None else
                          (position_xy - self.origin_xy).norm(dim=-1))
        broad, fine = _dense_cost(position_error, self.settings["position"])
        position_mask = self.active & (self.goal_commands[:, 0].abs() < .01)
        # Platform landing zones belong to the step task rather than a circle
        # around the launch platform, matching its independent route objective.
        position_mask &= self.landing_delta == 0.
        terms["jump_landing_position_broad"] = position_mask * broad
        terms["jump_landing_position_precision"] = position_mask * fine
        contact = self.active & self.supported
        precontact = self.airborne & (com_vz < 0.)
        if wheel_clearance is not None:
            precontact &= wheel_clearance.amin(-1) <= self.settings["precontact_clearance_m"]
        wheel_target = self.goal_commands[:, 0, None] / self.settings["wheel_radius_m"]
        rolling_speed = wheel_velocity * wheel_velocity.new_tensor([1., -1.])
        qualified_force = ((self.wheel_contact_time >= self.settings["contact_force_min_seconds"])
                           & (contact_peak > self.settings["contact_force_threshold_n"]))
        impact_ratio = torch.where(qualified_force, contact_peak, 0.).amax(-1) / weight_force.clamp_min(1e-6)
        bounded = {
            "down_speed": (contact, (-com_vz).clamp_min(0.)),
            "impact": (contact, (impact_ratio
                                  - self.settings["impact_bodyweight_threshold"]).clamp_min(0.)),
            "precontact_wheel_velocity": (precontact, (rolling_speed - wheel_target).square().mean(-1).sqrt()),
            "precontact_wheel_torque": (precontact, (wheel_torque.abs()
                                         - self.settings["precontact_wheel_torque_limit_nm"]).clamp_min(0.).norm(dim=-1)),
        }
        for axis, (mask, error) in bounded.items():
            broad, fine = _dense_cost(error, self.settings[axis])
            terms[f"jump_{axis}_broad"] = mask * broad
            terms[f"jump_{axis}_precision"] = mask * fine
        return terms
