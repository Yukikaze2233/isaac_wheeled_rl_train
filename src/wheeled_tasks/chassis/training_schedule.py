"""Actor-update curricula and episode-fixed perturbations for the V6 run.

The runner supplies the cumulative PPO update count. This clock is independent
of environment count, rollout length, transition counters and episode resets.
"""
from __future__ import annotations

import math

import torch

from .task import Phase


def actor_update_clock(env):
    value = getattr(env, "global_actor_update", env.cfg.get("global_actor_update", 0))
    if not math.isfinite(float(value)) or int(value) != value or value < 0:
        raise ValueError("global_actor_update must be a nonnegative integer")
    return int(value)


def stage_actor_updates(env):
    if getattr(env, "cfg", {}).get("task_campaign"):
        value = getattr(env, "task_actor_update", 0)
        if not math.isfinite(float(value)) or int(value) != value or value < 0:
            raise ValueError("task_actor_update must be a nonnegative integer")
        return int(value)
    if hasattr(env, "stage_actor_update"):
        value = env.stage_actor_update
        if not math.isfinite(float(value)) or int(value) != value or value < 0:
            raise ValueError("stage_actor_update must be a nonnegative integer")
        return int(value)
    start = env.cfg.get("new_asset_curriculum", {}).get("stage_start_actor_update", 0)
    return max(0, actor_update_clock(env) - int(start))


def jump_bin_probabilities(settings, confirmed_bins=()):
    """Move half of passed-bin excess mass to unpassed bins, preserving floors."""
    bins = settings["bins_m"]
    probabilities = [float(p) for p in settings["initial_probabilities"]]
    floor = float(settings.get("minimum_probability", .05))
    transfer = float(settings.get("confirmed_pass_transfer_fraction", .5))
    if (not bins or len(bins) != len(probabilities) or not 0 <= transfer <= 1
            or floor < 0 or len(bins) * floor > 1
            or any(not math.isfinite(p) or p < floor for p in probabilities)
            or abs(sum(probabilities) - 1.) > 1e-6):
        raise ValueError("Jump probabilities must sum to one and respect the floor")
    passed = [any(abs(float(b) - float(c)) < 1e-8 for c in confirmed_bins) for b in bins]
    remaining = sum(not p for p in passed)
    if remaining:
        moved = 0.
        for i, is_passed in enumerate(passed):
            if is_passed:
                amount = (probabilities[i] - floor) * transfer
                probabilities[i] -= amount
                moved += amount
        probabilities = [p if passed[i] else p + moved / remaining for i, p in enumerate(probabilities)]
    return probabilities


class TrainingSchedule:
    """No simulator writes: expose command samples, request state and effort scales."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.get("new_asset_curriculum", {})
        self.enabled = bool(self.cfg.get("enabled", False)) and not env.cfg.get("evaluation_exact_cases", False)
        n, device = env.num_envs, env.device
        self.motor_scale = torch.ones(n, 6, device=device)
        self.degraded = torch.zeros(n, dtype=torch.bool, device=device)
        self.edge_reset = torch.zeros_like(self.degraded)
        self.request_stress = torch.zeros_like(self.degraded)
        self.request_active = torch.zeros_like(self.degraded)
        self.initial_request_consumed = torch.zeros_like(self.degraded)
        self.request_start = torch.full((n,), torch.inf, device=device)
        self.next_request = torch.full((n,), torch.inf, device=device)
        self.request_event = torch.zeros(n, dtype=torch.long, device=device)
        # Event codes: 0 none, 1 accepted new, 2 repeat/ignored busy new, 3 cancel.
        self.jump_bin = torch.full((n,), -1, dtype=torch.long, device=device)
        self.jump_apex = torch.full((n,), .06, device=device)
        self.request_seen_flight = torch.zeros_like(self.degraded)
        self.awaiting_ground = torch.zeros_like(self.degraded)
        self.completed_jump_count = torch.zeros(n, dtype=torch.long, device=device)
        self.accepted_request_count = torch.zeros_like(self.completed_jump_count)
        self.repeat_request_count = torch.zeros_like(self.completed_jump_count)
        self.cancel_request_count = torch.zeros_like(self.completed_jump_count)
        self.cancel_phase = torch.full_like(self.completed_jump_count, -1)
        self.repeat_clock_reset_errors = torch.zeros_like(self.completed_jump_count)
        self.explicit_cursor = torch.zeros_like(self.completed_jump_count)
        self.has_explicit_events = torch.zeros_like(self.degraded)
        specs = env.cfg.get("skill_specs", {})
        if env.cfg.get("evaluation_exact_cases"):
            specs = {c["name"]: c.get("skill", {}) for c in env.cfg.get("evaluation", {}).get("cases", [])}
        self.event_batches = []
        for name, spec in specs.items():
            events = spec.get("request_events", [])
            if not events:
                continue
            previous = -1.
            for event in events:
                if (event["type"] not in ("new", "repeat", "cancel")
                        or not math.isfinite(event["time_s"]) or event["time_s"] < max(0., previous)
                        or event.get("after_phase") not in (None, "flight", "ground_after_recovery")):
                    raise ValueError("Invalid explicit request event or phase guard")
                previous = event["time_s"]
            members = torch.tensor([i for i, group in enumerate(env.scene_groups) if group == name], device=device, dtype=torch.long)
            self.has_explicit_events[members] = True
            self.event_batches.append((members, events))

    @property
    def requests_enabled(self):
        # Exact evaluation disables stochastic masks, not deterministic request
        # semantics. A fixed case must behave identically in a subset/full suite.
        return bool(self.event_batches) or (bool(self.cfg.get("enabled", False)) and bool(self.cfg.get("request_protocol")))

    @property
    def stage_updates(self):
        return stage_actor_updates(self.env)

    def _uniform(self, count, bounds):
        low, high = map(float, bounds)
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
            raise ValueError("Command/request duration must have positive finite bounds")
        return low + (high - low) * self.env.random(count)

    def reset(self, ids, velocity=None):
        n = len(ids)
        self.motor_scale[ids] = 1.
        for tensor in (self.degraded, self.edge_reset, self.request_stress,
                       self.request_active, self.initial_request_consumed):
            tensor[ids] = False
        self.request_start[ids] = torch.inf
        self.next_request[ids] = torch.inf
        self.request_event[ids] = 0
        self.request_seen_flight[ids] = False
        self.awaiting_ground[ids] = False
        for tensor in (self.completed_jump_count, self.accepted_request_count, self.repeat_request_count,
                       self.cancel_request_count, self.repeat_clock_reset_errors, self.explicit_cursor):
            tensor[ids] = 0
        self.cancel_phase[ids] = -1
        if not self.enabled:
            return
        self.degraded[ids] = self.env.random(n) < self.cfg.get("unilateral_degradation_fraction", 0.)
        side = (self.env.random(n) >= .5).long()
        scale = float(self.cfg.get("unilateral_degradation_scale", .5))
        if not 0 <= scale <= 1:
            raise ValueError("Unilateral degradation scale must be in [0, 1]")
        for selected_side in (0, 1):
            chosen = ids[self.degraded[ids] & (side == selected_side)]
            self.motor_scale[chosen, selected_side * 3:selected_side * 3 + 3] = scale
        self.edge_reset[ids] = self.env.random(n) < self.cfg.get("edge_reset_fraction", 0.)
        if velocity is not None:
            # Velocity-only perturbations cannot rotate a wheel through the floor.
            # These are a bounded physical reset prior, not a trajectory-state bank.
            edge = self.edge_reset[ids]
            count = int(edge.sum())
            velocity[edge, 0] += (2 * self.env.random(count) - 1) * self.cfg.get("edge_reset_velocity_m_s", .3)
            velocity[edge, 3] += (2 * self.env.random(count) - 1) * self.cfg.get("edge_reset_roll_rate_rad_s", .5)
            velocity[edge, 4] += (2 * self.env.random(count) - 1) * self.cfg.get("edge_reset_pitch_rate_rad_s", .5)
            velocity[edge, 5] += (2 * self.env.random(count) - 1) * self.cfg.get("edge_reset_yaw_rate_rad_s", .5)
        protocol = self.cfg.get("request_protocol", {})
        self.request_stress[ids] = self.env.random(n) < protocol.get("stress_episode_fraction", 0.)
        if protocol:
            self.next_request[ids] = self._uniform(n, protocol.get("interval_seconds", [.5, 4.]))

    def sample_height(self, count, bounds):
        settings = self.cfg["height_sampling_schedule"]
        fraction = settings["initial_fixed_fraction"] if self.stage_updates < settings["actor_update_boundary"] else settings["final_fixed_fraction"]
        fixed = torch.as_tensor(settings["fixed_heights_m"], device=self.env.device)
        low, high = bounds
        if not 0 <= fraction <= 1 or not len(fixed) or torch.any((fixed < low) | (fixed > high)):
            raise ValueError("Height mixture must stay inside the configured full domain")
        draw = self.env.random(count)
        fixed_index = (self.env.random(count) * len(fixed)).long().clamp_max(len(fixed) - 1)
        continuous = low + (high - low) * self.env.random(count)
        return torch.where(draw < fraction, fixed[fixed_index], continuous)

    def sample_continuous_height(self, count, settings):
        """Inverse CDF for increased density near the research height boundary."""
        low, high = settings["range_m"]
        fraction = float(settings.get("boundary_fraction_of_range", 0.))
        weight = float(settings.get("boundary_weight", 1.))
        if not 0 <= fraction < .5 or not math.isfinite(weight) or weight < 1 or not low < high:
            raise ValueError("Invalid bounded-height sampling density")
        mass = self.env.random(count) * (1 + 2 * fraction * (weight - 1))
        lower_mass = fraction * weight
        middle_end = lower_mass + 1 - 2 * fraction
        unit = torch.where(mass < lower_mass, mass / weight,
               torch.where(mass < middle_end, fraction + mass - lower_mass,
                           1 - fraction + (mass - middle_end) / weight))
        return low + (high - low) * unit

    def command_duration(self, count, fallback=None):
        bounds = self.cfg.get("command_resampling_seconds", fallback)
        if not self.enabled or bounds is None:
            return None
        duration = self._uniform(count, bounds)
        long_fraction = self.cfg.get("long_command_fraction", 0.)
        if long_fraction:
            long = self._uniform(count, self.cfg["long_command_resampling_seconds"])
            duration = torch.where(self.env.random(count) < long_fraction, long, duration)
        return duration

    def sample_jump(self, ids, spec):
        settings = self.cfg.get("jump_curriculum")
        default = spec.get("semantics", {}).get("jump_apex_delta_m", .06)
        self.jump_apex[ids] = default
        self.jump_bin[ids] = -1
        if (self.enabled and settings and spec.get("kind") == "jump"
                and not spec.get("request_events") and not spec.get("command_sequence")):
            confirmed = getattr(self.env, "confirmed_jump_bins_m", self.cfg.get("confirmed_jump_bins_m", ()))
            probabilities = torch.as_tensor(jump_bin_probabilities(settings, confirmed), device=self.env.device)
            index = torch.searchsorted(probabilities.cumsum(0), self.env.random(len(ids)), right=True).clamp_max(len(probabilities) - 1)
            self.jump_bin[ids] = index
            self.jump_apex[ids] = torch.as_tensor(settings["bins_m"], device=self.env.device)[index]
        return self.jump_apex[ids]

    def update_requests(self, elapsed, phases, eligible, initial_request):
        """Return request latch; a busy repeat never restarts its reference clock.

        Caller invokes once per policy tick and uses ``request_start`` for the
        reference elapsed time. Cancelling after liftoff preserves the latch
        through landing/recovery. No phase, pose or velocity is overwritten.
        """
        if not self.requests_enabled:
            return initial_request
        protocol = self.cfg.get("request_protocol", {})
        self.request_event.zero_()
        self.request_seen_flight |= self.request_active & (phases == Phase.FLIGHT)
        recovered_contact = self.request_active & self.request_seen_flight & (phases == Phase.RECOVERY)
        self.awaiting_ground |= recovered_contact
        self.request_active[recovered_contact] = False
        completed = (self.request_active | self.awaiting_ground) & self.request_seen_flight & (phases == Phase.GROUND)
        self.completed_jump_count += completed.long()
        self.request_active[completed] = False
        self.request_seen_flight[completed] = False
        self.awaiting_ground[completed] = False
        first = eligible & initial_request & ~self.initial_request_consumed & ~self.has_explicit_events
        self.initial_request_consumed |= first
        due = eligible & self.request_stress & (elapsed >= self.next_request) & ~self.has_explicit_events
        draw = self.env.random(self.env.num_envs)
        cancel_probability = float(protocol.get("cancel_probability", .1))
        repeat_probability = float(protocol.get("repeat_probability", .1))
        new_probability = float(protocol.get("new_probability", .8))
        if (min(cancel_probability, repeat_probability, new_probability) < 0
                or abs(cancel_probability + repeat_probability + new_probability - 1) > 1e-6):
            raise ValueError("Request event probabilities must sum to one")
        cancel = due & (draw < cancel_probability)
        repeat = due & (draw >= cancel_probability) & (draw < cancel_probability + repeat_probability)
        new = first | (due & (draw >= cancel_probability + repeat_probability))
        for members, events in self.event_batches:
            # At most one explicit event per lane per policy tick. Phase guards
            # wait for the observed physical phase instead of guessing airtime.
            cursor = self.explicit_cursor[members].clone()
            for index, event in enumerate(events):
                ready = (cursor == index) & (elapsed[members] + 1e-6 >= event["time_s"])
                guard = event.get("after_phase")
                if guard == "flight":
                    ready &= phases[members] == Phase.FLIGHT
                elif guard == "ground_after_recovery":
                    verified = getattr(self.env, "successful_jump_count", torch.zeros_like(self.accepted_request_count))
                    ready &= ((self.completed_jump_count[members] > 0) & (phases[members] == Phase.GROUND)
                              & ~self.request_active[members] & (verified[members] >= self.accepted_request_count[members]))
                chosen = members[ready & eligible[members]]
                {"new": new, "repeat": repeat, "cancel": cancel}[event["type"]][chosen] = True
                self.explicit_cursor[chosen] += 1
        previous_start = self.request_start.clone()
        can_start = (phases == Phase.GROUND) & ~self.request_active
        accepted = new & can_start
        self.request_active |= accepted
        self.request_seen_flight[accepted] = False
        self.awaiting_ground[accepted] = False
        self.request_start[accepted] = elapsed[accepted]
        self.request_event[(new & ~can_start) | repeat] = 2
        self.request_event[accepted] = 1
        self.request_event[cancel] = 3
        self.accepted_request_count += accepted.long()
        repeated_busy = (new & ~can_start) | repeat
        self.repeat_request_count += repeated_busy.long()
        self.cancel_request_count += cancel.long()
        self.cancel_phase[cancel] = phases[cancel]
        self.repeat_clock_reset_errors += (repeated_busy & (self.request_start != previous_start)).long()
        cancellable = cancel & ((phases == Phase.GROUND) | (phases == Phase.TAKEOFF))
        self.request_active[cancellable] = False
        # Releasing a finished landing request in RECOVERY prevents the ordinary
        # phase tracker from treating the same held request as another takeoff.
        # It returns to GROUND through its actual stable-contact timer.
        # Leaving a task also cancels its latch. Ground before takeoff alone is
        # not evidence of completion; the flight-seen condition above is needed.
        self.request_active[~eligible] = False
        self.request_seen_flight[~eligible] = False
        self.awaiting_ground[~eligible] = False
        self.request_start[~eligible] = torch.inf
        if due.any():
            self.next_request[due] = elapsed[due] + self._uniform(int(due.sum()), protocol.get("interval_seconds", [.5, 4.]))
        return self.request_active & eligible

    def pending_explicit_requests(self):
        pending = self.has_explicit_events & (self.request_active | self.awaiting_ground)
        for members, events in self.event_batches:
            pending[members] |= self.explicit_cursor[members] < len(events)
        return pending

    def diagnostics(self):
        return {"curriculum_jump_bin": self.jump_bin.clone(),
                "curriculum_jump_apex_m": self.jump_apex.clone(),
                "curriculum_request_event": self.request_event.clone(),
                "curriculum_request_stress": self.request_stress.clone(),
                "curriculum_unilateral_degradation": self.degraded.clone(),
                "curriculum_edge_reset": self.edge_reset.clone(),
                "request_events_delivered": self.explicit_cursor.clone(),
                "accepted_request_count": self.accepted_request_count.clone(),
                "repeat_request_count": self.repeat_request_count.clone(),
                "cancel_request_count": self.cancel_request_count.clone(),
                "cancel_phase": self.cancel_phase.clone(),
                "repeat_clock_reset_errors": self.repeat_clock_reset_errors.clone(),
                "completed_jump_count": self.completed_jump_count.clone()}
