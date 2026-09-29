"""Preset-platform jump requests and separate physical landing assessment.

The request controller consumes only a preset course, wheel odometry, IMU and a
single range beam. World poses and contacts are accepted only by ``observe`` for
training reward/assessment; they never advance the command state machine.
"""
from __future__ import annotations

import math

import torch


class StepJumpTask:
    def __init__(self, groups, specs, device, dt):
        self.dt, self.count = float(dt), len(groups)
        n = self.count
        self.enabled = torch.tensor([bool(specs.get(name, {}).get("step_jump")) for name in groups], device=device)
        self.has_tasks = bool(self.enabled.any())
        self.laser = torch.zeros(n, dtype=torch.bool, device=device)
        self.rises = torch.zeros(n, 2, device=device)
        self.edges = torch.zeros_like(self.rises)
        self.ends = torch.zeros_like(self.rises)
        self.tops = torch.zeros_like(self.rises)
        self.length = torch.zeros(n, dtype=torch.long, device=device)
        self.launch_distance = torch.full((n,), .25, device=device)
        self.approach_speed = torch.full((n,), .4, device=device)
        self.margin = torch.full((n,), .10, device=device)
        self.settle_seconds = torch.full((n,), .35, device=device)
        self.extra_clearance = torch.full((n,), .08, device=device)
        for i, name in enumerate(groups):
            settings = specs.get(name, {}).get("step_jump")
            if not settings:
                continue
            rises, treads = settings["rises_m"], settings["treads_m"]
            if (len(rises) not in (1, 2) or len(treads) != len(rises)
                    or any(not math.isfinite(v) or v <= 0 for v in rises + treads)
                    or settings["trigger_source"] not in ("manual", "laser")):
                raise ValueError("Step jumps require one/two positive preset rises and treads")
            self.length[i], self.laser[i] = len(rises), settings["trigger_source"] == "laser"
            x, z = 0., 0.
            for j, (rise, tread) in enumerate(zip(rises, treads)):
                self.rises[i, j], self.edges[i, j] = rise, x
                x, z = x + tread, z + rise
                self.ends[i, j], self.tops[i, j] = x, z
            for field, key in (("launch_distance", "launch_distance_m"), ("approach_speed", "approach_speed_m_s"),
                               ("margin", "landing_margin_m"), ("settle_seconds", "settle_seconds"),
                               ("extra_clearance", "landing_clearance_m")):
                value = settings[key]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid step jump {key}")
                getattr(self, field)[i] = value
            if any(tread <= 2 * float(self.margin[i]) + .12 for tread in treads):
                raise ValueError("Landing region has no wheel clearance after margins")
        self.index = torch.zeros_like(self.length)
        self.active = torch.zeros_like(self.enabled)
        self.accepted = torch.zeros_like(self.enabled)
        self.request_start = torch.full((n,), torch.inf, device=device)
        self.request_count = torch.zeros_like(self.length)
        self.command_completed = torch.zeros_like(self.length)
        self.odometry_x = torch.full((n,), -.8, device=device)
        self.target_odometry_x = torch.zeros(n, device=device)
        self.last_time = torch.zeros(n, device=device)
        self.seen_imu_flight = torch.zeros_like(self.enabled)
        self.support_time = torch.zeros(n, device=device)
        self.command_vx = torch.zeros(n, device=device)
        self.landing_delta = torch.zeros(n, device=device)
        self.apex = torch.full((n,), .06, device=device)
        self.flight_end = torch.zeros(n, device=device)
        self.assessment_index = torch.zeros_like(self.length)
        self.air_seen = torch.zeros_like(self.enabled)
        self.invalid_landing = torch.zeros_like(self.enabled)
        self.previous_unsupported = torch.zeros_like(self.enabled)
        self.last_launch_odometry = torch.full((n,), -torch.inf, device=device)
        self.air_seconds = torch.zeros(n, device=device)
        self.air_peak = torch.zeros(n, device=device)
        self.landing_hold = torch.zeros(n, device=device)
        self.completed = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.completed_event = torch.zeros_like(self.enabled)
        self.front_clear = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.previous_wheel_x = torch.full((n, 2), -.8, device=device)
        self.beam_range = torch.full((n,), torch.inf, device=device)
        self.beam_valid = torch.zeros_like(self.enabled)
        self.rows = torch.arange(n, device=device)

    def reset(self, ids):
        for name in ("active", "accepted", "seen_imu_flight", "air_seen", "completed_event", "beam_valid", "invalid_landing", "previous_unsupported"):
            getattr(self, name)[ids] = False
        for name in ("index", "request_count", "command_completed", "last_time", "support_time", "landing_delta",
                     "command_vx", "assessment_index", "air_seconds", "air_peak", "landing_hold"):
            getattr(self, name)[ids] = 0
        self.request_start[ids] = torch.inf
        self.odometry_x[ids] = -.8
        self.last_launch_odometry[ids] = -torch.inf
        self.previous_wheel_x[ids] = -.8
        self.completed[ids] = self.front_clear[ids] = False

    def _current(self, table, index=None):
        return table[self.rows, (self.index if index is None else index).clamp_max(1)]

    def control(self, time, wheel_forward_speed, support_estimate, specific_force_norm,
                beam_range, beam_valid, beam_age, *, preload_seconds=.25, preload_depth=.02,
                release_offset=.025, beam_forward_distance=None):
        """One command-clock tick; repeated reads at the same time do not advance it."""
        delta_t = (time - self.last_time).clamp(0., self.dt * 1.01)
        advanced = delta_t > 0
        self.accepted.zero_()
        self.last_time.copy_(time)
        self.odometry_x += wheel_forward_speed * delta_t
        self.beam_range, self.beam_valid = beam_range, beam_valid
        elapsed = (time - self.request_start).clamp_min(0.)
        self.seen_imu_flight |= self.active & (specific_force_norm < 3.) & advanced
        settled = (self.active & self.seen_imu_flight & support_estimate
                   & (elapsed >= self.flight_end))
        self.support_time = torch.where(settled, self.support_time + delta_t, 0.)
        finished = self.active & (self.support_time >= self.settle_seconds)
        self.active &= ~finished
        self.index += finished.long()
        self.command_completed += finished.long()
        remaining = self.enabled & (self.index < self.length)
        idle = remaining & ~self.active
        # The manual request is a prescribed operator event based on the known
        # course and encoder odometry, never a query of true robot position.
        manual = ~self.laser & (self.odometry_x >= self._current(self.edges) - self.launch_distance)
        trigger_distance = beam_range if beam_forward_distance is None else beam_forward_distance
        laser = (self.laser & beam_valid & torch.isfinite(beam_range) & (beam_range > .015)
                 & torch.isfinite(trigger_distance) & (trigger_distance > 0.)
                 & (trigger_distance <= self.launch_distance) & (beam_age <= .05) & (beam_age >= 0))
        # A persistent near return is not evidence of the next riser. Advance
        # at least half the preceding prescribed tread in encoder odometry.
        previous_tread = self._current(self.ends - self.edges, (self.index - 1).clamp_min(0))
        laser &= (self.index == 0) | (self.odometry_x - self.last_launch_odometry >= .5 * previous_tread)
        accepted = idle & support_estimate & (manual | laser) & advanced & ~finished
        self.accepted.copy_(accepted)
        self.active |= accepted
        self.request_count += accepted.long()
        self.request_start[accepted] = time[accepted]
        self.last_launch_odometry[accepted] = self.odometry_x[accepted]
        self.seen_imu_flight[accepted] = False
        self.support_time[accepted] = 0.
        rise = self._current(self.rises)
        self.landing_delta = torch.where(self.active, rise, 0.)
        self.apex = torch.where(remaining, rise + self.extra_clearance, .06)
        release = torch.sqrt(2 * 9.81 * self.apex)
        self.flight_end = (preload_seconds + 2 * (preload_depth + release_offset) / release
                           + (release + torch.sqrt((release.square() + 2 * 9.81 * (release_offset - rise)).clamp_min(0))) / 9.81)
        beam_distance = beam_range + .10 if beam_forward_distance is None else beam_forward_distance
        distance = torch.where(laser, beam_distance, self._current(self.edges) - self.odometry_x)
        landing_offset = torch.minimum(self.margin + .12, (self._current(self.ends) - self._current(self.edges)) / 2)
        self.target_odometry_x[accepted] = (self.odometry_x + distance + landing_offset)[accepted]
        elapsed = (time - self.request_start).clamp_min(0.)
        velocity = ((self.target_odometry_x - self.odometry_x)
                    / (self.flight_end - elapsed).clamp_min(.15)).clamp(0., 1.5)
        # A supported IMU recovery after observed free fall holds the landing.
        stop = self.seen_imu_flight & support_estimate & (elapsed > .25)
        self.command_vx = torch.where(self.active, torch.where(stop, 0., velocity),
                                      torch.where(remaining, self.approach_speed, 0.))
        self.command_vx *= self.enabled
        if bool(accepted.any()):
            self.assessment_index[accepted] = self.index[accepted]
            self.air_seen[accepted] = False
            self.invalid_landing[accepted] = False
            self.previous_unsupported[accepted] = False
            self.air_seconds[accepted] = self.air_peak[accepted] = self.landing_hold[accepted] = 0.
            self.front_clear[accepted] = False
        return self.accepted

    def observe(self, wheel_centers, contacts, com_vz, planar_speed, gravity, stable, request_armed):
        """Privileged physical evidence, kept separate from controller progression."""
        tracked = self.enabled & (self.request_count > 0)
        idx = self.assessment_index
        edge, end, top = (self._current(table, idx) for table in (self.edges, self.ends, self.tops))
        unsupported = ~contacts.any(-1)
        self.air_seen |= tracked & request_armed & unsupported & (com_vz > .1)
        self.air_seconds = torch.where(tracked & self.air_seen & unsupported, self.air_seconds + self.dt, 0.)
        self.air_peak = torch.maximum(self.air_peak, self.air_seconds)
        x, bottom = wheel_centers[:, :, 0], wheel_centers[:, :, 2] - .06
        touching_target = ((x >= edge[:, None]) & (x <= end[:, None])
                           & ((bottom - top[:, None]).abs() <= .035))
        self.invalid_landing |= (tracked & self.air_seen
            & (contacts & ~touching_target).any(-1))
        crossed = ((self.previous_wheel_x < edge[:, None]) & (x >= edge[:, None])
                   & (bottom >= top[:, None] - .01) & self.air_seen[:, None] & ~self.invalid_landing[:, None]
                   & (unsupported | self.previous_unsupported)[:, None])
        self.front_clear |= crossed
        self.previous_wheel_x.copy_(x)
        self.previous_unsupported.copy_(unsupported)
        in_zone = ((x >= (edge + self.margin)[:, None]) & (x <= (end - self.margin)[:, None])
                   & (wheel_centers[:, :, 1].abs() < 1.5)
                   & ((bottom - top[:, None]).abs() <= .035)).all(-1)
        landing = (tracked & ~self.invalid_landing & self.air_seen & (self.air_peak >= .06) & self.front_clear.all(-1)
                   & contacts.all(-1) & in_zone & stable & (planar_speed <= .25)
                   & (gravity[:, 2] < -math.cos(math.radians(20))))
        self.landing_hold = torch.where(landing, self.landing_hold + self.dt, 0.)
        passed = landing & (self.landing_hold >= self.settle_seconds)
        old = self.completed[self.rows, idx]
        self.completed_event = passed & ~old
        self.completed[self.rows, idx] |= passed
        return self.completed_event

    @property
    def success(self):
        return self.enabled & self.completed[:, 0] & ((self.length == 1) | self.completed[:, 1])

    def reward_terms(self, wheel_centers, gravity):
        index = self.assessment_index
        edge, end, top = (self._current(table, index) for table in (self.edges, self.ends, self.tops))
        outside = ((edge + self.margin)[:, None] - wheel_centers[:, :, 0]).clamp_min(0)
        outside += (wheel_centers[:, :, 0] - (end - self.margin)[:, None]).clamp_min(0)
        broad = torch.sqrt(1 + (outside / .2).square()) - 1
        active = self.enabled & self.active
        near = (wheel_centers[:, :, 0] - edge[:, None]).abs() < .16
        shortfall = ((top + .02)[:, None] - (wheel_centers[:, :, 2] - .06)).clamp_min(0)
        return {
            "step_jump_landing_region": active * (-1.5 * broad.mean(-1)
                + .5 * torch.expm1(-(outside / .08).square()).mean(-1)),
            "step_jump_front_clearance": -2. * active * (near * shortfall / .08).mean(-1),
            "step_jump_lateral": -1. * active * (wheel_centers[:, :, 1] / .25).square().mean(-1),
            "step_jump_platform_event": self.completed_event.float() * 2. / self.dt,
        }

    def diagnostics(self):
        return {"step_jump_enabled": self.enabled.clone(), "step_jump_requested_count": self.request_count.clone(),
                "step_jump_required_count": self.length.clone(), "step_jump_completed_count": self.completed.sum(-1),
                "step_jump_first_landed": self.completed[:, 0].clone(), "step_jump_second_landed": self.completed[:, 1].clone(),
                "step_jump_landing_delta": self.landing_delta.clone(), "step_jump_odometry_x": self.odometry_x.clone(),
                "step_jump_beam_range": self.beam_range.clone(), "step_jump_beam_valid": self.beam_valid.clone()}
