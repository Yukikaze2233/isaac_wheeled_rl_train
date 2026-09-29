"""Physical assessment for ordinary forward climbing, with no command FSM.

All inputs are privileged reward/assessment measurements. This module neither
produces commands nor adds observations to the policy. Platform tops may support
the chassis; wheel contact on each successive top completes the route.
"""
from __future__ import annotations

import math

import torch


class StepClimbTask:
    MAX_UNSUPPORTED_SECONDS = .08

    def __init__(self, groups, specs, device, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite step-climb timestep required")
        self.dt, self.count = float(dt), len(groups)
        n = self.count
        self.enabled = torch.tensor([bool(specs.get(name, {}).get("body_climb")) for name in groups], device=device)
        self.has_tasks = bool(self.enabled.any())
        self.length = torch.zeros(n, dtype=torch.long, device=device)
        self.edges = torch.zeros(n, 2, device=device)
        self.ends = torch.zeros_like(self.edges)
        self.tops = torch.zeros_like(self.edges)
        self.margin = torch.full((n,), .06, device=device)
        self.body_min_force = torch.full((n,), 20., device=device)
        self.support_hold_seconds = torch.full((n,), .10, device=device)
        self.settle_seconds = torch.full((n,), .35, device=device)
        for i, name in enumerate(groups):
            settings = specs.get(name, {}).get("body_climb")
            if not settings:
                continue
            rises, treads = settings["rises_m"], settings["treads_m"]
            if (len(rises) not in (1, 2) or len(rises) != len(treads)
                    or any(not math.isfinite(v) or v <= 0 for v in (*rises, *treads))):
                raise ValueError("Step climbs require one/two positive preset rises and treads")
            self.length[i] = len(rises)
            x, z = 0., 0.
            for j, (rise, tread) in enumerate(zip(rises, treads)):
                self.edges[i, j] = x
                x, z = x + tread, z + rise
                self.ends[i, j], self.tops[i, j] = x, z
            for field, key, default in (("margin", "landing_margin_m", .06),
                    ("body_min_force", "body_contact_min_force_n", 20.),
                    ("support_hold_seconds", "support_hold_seconds", .10),
                    ("settle_seconds", "settle_seconds", .35)):
                value = settings.get(key, default)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid body climb {key}")
                getattr(self, field)[i] = value
            if any(tread <= 2 * float(self.margin[i]) + .12 for tread in treads):
                raise ValueError("Step-climb platform has no wheel support region")
        self.rows = torch.arange(n, device=device)
        self.index = torch.zeros_like(self.length)
        self.completed = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.completed_event = torch.zeros_like(self.enabled)
        self.front_passed = torch.zeros_like(self.completed)
        self.previous_wheel_x = torch.full((n, 2), -.8, device=device)
        self.wheel_support_time = torch.zeros(n, device=device)
        self.body_support_time = torch.zeros(n, device=device)
        self.body_support_peak = torch.zeros(n, device=device)
        self.body_support_credit = torch.zeros(n, device=device)
        self.progress_rate = torch.zeros(n, device=device)
        self.free_air_seconds = torch.zeros(n, device=device)
        self.air_peak = torch.zeros(n, device=device)
        self.flight_violation = torch.zeros_like(self.enabled)
        self.body_contact_allowed = torch.zeros(n, 0, dtype=torch.bool, device=device)

    def reset(self, ids):
        for name in ("index", "wheel_support_time", "body_support_time", "body_support_peak",
                     "body_support_credit", "progress_rate", "free_air_seconds", "air_peak"):
            getattr(self, name)[ids] = 0
        for name in ("completed", "completed_event", "front_passed", "body_contact_allowed", "flight_violation"):
            getattr(self, name)[ids] = False
        self.previous_wheel_x[ids] = -.8

    def _current(self, table):
        return table[self.rows, self.index.clamp_max(1)]

    def _body_contacts(self, bottom_points, forces, *, current_only):
        if (bottom_points.ndim != 3 or bottom_points.shape != forces.shape
                or bottom_points.shape[0] != self.count or bottom_points.shape[-1] != 3):
            raise ValueError("Body bottom points and world contact forces must have shape [env, body, 3]")
        # Points must be collision-bottom/contact points, not body COM positions.
        # Upward load inside a top face distinguishes support from a riser hit.
        x, y, z = bottom_points.unbind(-1)
        top = ((x[..., None] >= self.edges[:, None, :] + .01)
               & (x[..., None] <= self.ends[:, None, :] - .01)
               & ((z[..., None] - self.tops[:, None, :]).abs() <= .025)
               & (torch.arange(2, device=x.device)[None, None, :] < self.length[:, None, None]))
        if current_only:
            top &= torch.arange(2, device=x.device)[None, None, :] == self.index[:, None, None]
        up = forces[..., 2]
        # Legality is per contact, whereas support strength is the sum of legal
        # contacts. Splitting a 40 N load over four patches must not create four
        # illegal 10 N collisions.
        load = (up > 0.) & (up >= .5 * forces.norm(dim=-1))
        return self.enabled[:, None] & (y.abs() < 1.5) & top.any(-1) & load

    def legal_body_contacts(self, body_bottom_points, body_contact_forces):
        """Per-patch legal top contacts, independent of total support strength.

        Points use terrain-local coordinates; force vectors use world axes. This
        geometric predicate needs actual collision-bottom or contact points. A
        body COM is not a substitute and would misclassify its support height.
        """
        return self._body_contacts(body_bottom_points, body_contact_forces, current_only=False)

    def observe(self, wheel_centers, wheel_contacts, stable, *, body_bottom_points=None,
                body_contact_forces=None):
        """Credit one successive platform only after both wheels support it.

        Body support is allowed and rewarded during climbing, but is optional for
        completion. Forward motion may continue throughout the settling window.
        Sustained loss of both wheel and legal chassis support is a jump, not an
        ordinary climb; this invalidates the episode even if it later lands.
        """
        active = self.enabled & (self.index < self.length)
        edge, end, top = (self._current(table) for table in (self.edges, self.ends, self.tops))
        x, bottom = wheel_centers[:, :, 0], wheel_centers[:, :, 2] - .06
        self.front_passed |= active[:, None] & (self.previous_wheel_x < edge[:, None]) & (x >= edge[:, None])
        potential = lambda value: (value.amin(-1) - edge + .4).clamp(0., .6)
        self.progress_rate = active * (potential(x) - potential(self.previous_wheel_x)) / self.dt
        self.previous_wheel_x.copy_(x)
        in_top = ((x >= (edge + self.margin)[:, None]) & (x <= (end - self.margin)[:, None])
                  & (wheel_centers[:, :, 1].abs() < 1.5)
                  & ((bottom - top[:, None]).abs() <= .035)).all(-1)
        if (body_bottom_points is None) != (body_contact_forces is None):
            raise ValueError("Both body bottom points and body contact forces are required")
        body_support = torch.zeros_like(self.enabled)
        any_body_support = torch.zeros_like(self.enabled)
        if body_bottom_points is not None:
            self.body_contact_allowed = self.legal_body_contacts(body_bottom_points, body_contact_forces)
            current_contacts = self._body_contacts(body_bottom_points, body_contact_forces, current_only=True)
            upward_force = body_contact_forces[..., 2]
            body_support = torch.where(current_contacts, upward_force, 0.).sum(-1) >= self.body_min_force
            any_body_support = torch.where(self.body_contact_allowed, upward_force, 0.).sum(-1) >= self.body_min_force
        else:
            self.body_contact_allowed.zero_()
        any_support = wheel_contacts.any(-1) | any_body_support
        self.free_air_seconds = torch.where(self.enabled & ~any_support, self.free_air_seconds + self.dt, 0.)
        self.air_peak = torch.maximum(self.air_peak, self.free_air_seconds)
        self.flight_violation |= self.enabled & (self.free_air_seconds >= self.MAX_UNSUPPORTED_SECONDS)
        active &= ~self.flight_violation
        self.progress_rate *= ~self.flight_violation
        supported = active & self.front_passed.all(-1) & in_top & wheel_contacts.all(-1) & stable
        self.wheel_support_time = torch.where(supported, self.wheel_support_time + self.dt, 0.)
        self.body_support_time = torch.where(active & body_support, self.body_support_time + self.dt, 0.)
        peak = torch.maximum(self.body_support_peak, torch.minimum(self.body_support_time, self.support_hold_seconds))
        # Finite credit for establishing chassis support, never a residence bonus.
        self.body_support_credit = active * (peak - self.body_support_peak) / self.dt
        self.body_support_peak = peak
        self.completed_event = supported & (self.wheel_support_time >= self.settle_seconds)
        self.completed[self.rows, self.index.clamp_max(1)] |= self.completed_event
        self.index += self.completed_event.long()
        self.front_passed[self.completed_event] = False
        self.wheel_support_time[self.completed_event] = 0.
        self.body_support_time[self.completed_event] = 0.
        self.body_support_peak[self.completed_event] = 0.
        return self.completed_event

    @property
    def success(self):
        return self.enabled & ~self.flight_violation & self.completed[:, 0] & ((self.length == 1) | self.completed[:, 1])

    def height_tracking_scale(self, wheel_centers):
        """Relax nominal stance height only while crossing the current riser.

        Geometry is a privileged reward gate, never a command or policy input.
        The gate opens before the edge and restores the ordinary height target
        once both wheels are on the top. It cannot disable height tracking on a
        completed course or elsewhere on the floor.
        """
        edge = self._current(self.edges)
        x = wheel_centers[:, :, 0]
        approach = ((x.amax(-1) - edge + .30) / .20).clamp(0., 1.)
        departure = ((x.amin(-1) - edge - .08) / .22).clamp(0., 1.)
        approach = approach.square() * (3 - 2 * approach)
        departure = departure.square() * (3 - 2 * departure)
        active = self.enabled & ~self.flight_violation & (self.index < self.length)
        return 1. - active * approach * (1. - departure)

    def reward_terms(self, wheel_centers, gravity):
        active = self.enabled & ~self.flight_violation & (self.index < self.length)
        edge, top = self._current(self.edges), self._current(self.tops)
        near = (wheel_centers[:, :, 0] - edge[:, None]).abs() < .16
        shortfall = (top[:, None] - (wheel_centers[:, :, 2] - .06)).clamp_min(0.)
        square = (shortfall / .12).square()
        broad = -square / (torch.sqrt(1. + square) + 1.)
        precision = torch.expm1(-(shortfall / .05).square())
        return {
            "step_climb_progress": .5 * self.progress_rate,
            "step_climb_wheel_lift": active * (near * (broad + .5 * precision)).mean(-1),
            "step_climb_body_support": .3 * self.body_support_credit,
            "step_climb_attitude": -.2 * active * gravity[:, :2].square().sum(-1),
            "step_climb_lateral": -.1 * active * (wheel_centers[:, :, 1] / .3).square().mean(-1),
            "step_climb_platform_event": self.completed_event.float() * 2. / self.dt,
        }

    def diagnostics(self):
        return {"body_climb_enabled": self.enabled.clone(), "body_climb_required_count": self.length.clone(),
                "body_climb_completed_count": self.completed.sum(-1),
                "body_climb_first_supported": self.completed[:, 0].clone(),
                "body_climb_second_supported": self.completed[:, 1].clone(),
                "body_climb_current_step": self.index.clone(),
                "body_climb_wheel_support_s": self.wheel_support_time.clone(),
                "body_climb_body_support_s": self.body_support_time.clone(),
                "body_climb_air_peak_s": self.air_peak.clone(),
                "body_climb_flight_violation": self.flight_violation.clone(),
                "body_climb_body_supported": self.body_support_time >= self.support_hold_seconds}
