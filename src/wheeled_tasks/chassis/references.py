"""Command-clock references shared by policy inputs, reward and deployment.

No terrain, contact or simulator state is consumed by this module. Physical
phase and task acceptance remain separate measurements in the environment.
"""
import math

import torch


class CommandReference:
    NORMAL, STEP_RAISE, STEP_HOLD, STEP_LOWER, PRELOAD, PUSH, FLIGHT, LAND, RECOVER = range(9)

    def __init__(self, count, device, config):
        self.cfg = dict(config)
        positive = ("step_height_m", "step_down_height_m", "step_rise_seconds", "step_lower_seconds", "preload_seconds",
                    "preload_depth_m", "release_offset_m", "landing_seconds", "takeoff_margin_seconds",
                    "pitch_limit_rad")
        if any(not math.isfinite(config[key]) or config[key] <= 0 for key in positive):
            raise ValueError("Command-reference lengths, times and pitch limit must be positive")
        for key in ("step_request_range_s", "step_hold_range_s"):
            low, high = config[key]
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
                raise ValueError("Manual request timing must be finite, nonnegative and ordered")
        self.phase = torch.zeros(count, device=device, dtype=torch.long)
        self.height = torch.zeros(count, device=device)
        self.height_delta = torch.zeros_like(self.height)
        self.vertical_velocity = torch.zeros_like(self.height)
        self.acceleration = torch.zeros_like(self.height)
        self.pitch = torch.zeros_like(self.height)
        self.elapsed = torch.zeros_like(self.height)
        self.terrain_mode = torch.zeros_like(self.height)
        self.release_speed = torch.zeros_like(self.height)
        self.push_end = torch.zeros_like(self.height)
        self.jumping = torch.zeros(count, device=device, dtype=torch.bool)

    def reset(self, ids):
        self.phase[ids] = self.NORMAL
        for value in (self.height_delta, self.vertical_velocity, self.acceleration, self.pitch,
                      self.elapsed, self.terrain_mode, self.release_speed, self.push_end):
            value[ids] = 0.
        self.jumping[ids] = False

    @staticmethod
    def smooth_transition(time, duration):
        u = (time / duration).clamp(0., 1.)
        return 3 * u.square() - 2 * u.pow(3), 6 * u * (1 - u) / duration

    def step_preparing(self, step_mode, elapsed):
        return (step_mode != 0) & (elapsed >= 0) & (elapsed < self.cfg["step_rise_seconds"])

    def update(self, raw_height, acceleration, step_mode, step_elapsed, step_duration,
               jump_request, jump_elapsed, apex):
        cfg = self.cfg
        step = (step_mode != 0) & (step_elapsed >= 0) & (step_elapsed < step_duration + cfg["step_lower_seconds"])
        rising = step & (step_elapsed < cfg["step_rise_seconds"])
        lowering = step & (step_elapsed >= step_duration)
        rise, rise_v = self.smooth_transition(step_elapsed, cfg["step_rise_seconds"])
        lower, lower_v = self.smooth_transition(step_elapsed - step_duration, cfg["step_lower_seconds"])
        step_target = torch.maximum(raw_height, raw_height.new_full(raw_height.shape, cfg["step_height_m"]))
        step_target = torch.where(step_mode < 0, torch.minimum(raw_height,
            raw_height.new_full(raw_height.shape, cfg["step_down_height_m"])), step_target)
        bias = step_target - raw_height
        effective = raw_height + torch.where(step, bias * (rise - lower), 0.)
        step_v = torch.where(step, bias * (rise_v - lower_v), 0.)

        self.jumping = jump_request.bool()
        self.acceleration.copy_(acceleration)
        self.pitch = torch.atan(acceleration / 9.81).clamp(-cfg["pitch_limit_rad"], cfg["pitch_limit_rad"])
        t = jump_elapsed.clamp_min(0.)
        release = torch.sqrt(2 * 9.81 * apex.clamp_min(.001))
        push_duration = 2 * (cfg["preload_depth_m"] + cfg["release_offset_m"]) / release
        push_end = cfg["preload_seconds"] + push_duration
        flight_end = push_end + 2 * release / 9.81
        land_end = flight_end + cfg["landing_seconds"]
        preload = self.jumping & (t < cfg["preload_seconds"])
        push = self.jumping & (t >= cfg["preload_seconds"]) & (t < push_end)
        flight = self.jumping & (t >= push_end) & (t < flight_end)
        land = self.jumping & (t >= flight_end) & (t < land_end)
        self.phase = torch.where(step, self.STEP_HOLD, self.NORMAL)
        self.phase = torch.where(rising, self.STEP_RAISE, self.phase)
        self.phase = torch.where(lowering, self.STEP_LOWER, self.phase)
        self.phase = torch.where(self.jumping, self.RECOVER, self.phase)
        for mask, value in ((preload, self.PRELOAD), (push, self.PUSH), (flight, self.FLIGHT), (land, self.LAND)):
            self.phase = torch.where(mask, value, self.phase)

        fraction, derivative = self.smooth_transition(t, cfg["preload_seconds"])
        h_pre = raw_height - cfg["preload_depth_m"] * fraction
        v_pre = -cfg["preload_depth_m"] * derivative
        push_time = torch.minimum((t - cfg["preload_seconds"]).clamp_min(0.), push_duration)
        a_push = release / push_duration
        h_push = raw_height - cfg["preload_depth_m"] + .5 * a_push * push_time.square()
        v_push = a_push * push_time
        air_time = (t - push_end).clamp_min(0.)
        h_air = raw_height + cfg["release_offset_m"] + release * air_time - .5 * 9.81 * air_time.square()
        v_air = release - 9.81 * air_time
        # Hermite touchdown reference: incoming ballistic speed to a quiet stance.
        duration = cfg["landing_seconds"]
        u = ((t - flight_end) / duration).clamp(0., 1.)
        h_land = (raw_height + cfg["release_offset_m"] * (2 * u.pow(3) - 3 * u.square() + 1)
                  - release * duration * (u.pow(3) - 2 * u.square() + u))
        v_land = (cfg["release_offset_m"] * (6 * u.square() - 6 * u) / duration
                  - release * (3 * u.square() - 4 * u + 1))
        self.height = torch.where(self.jumping, raw_height, effective)
        self.vertical_velocity = step_v
        for mask, height, speed in ((preload, h_pre, v_pre), (push, h_push, v_push),
                                    (flight, h_air, v_air), (land, h_land, v_land)):
            self.height = torch.where(mask, height, self.height)
            self.vertical_velocity = torch.where(mask, speed, self.vertical_velocity)
        self.vertical_velocity = torch.where(self.jumping & (self.phase == self.RECOVER), 0., self.vertical_velocity)
        self.height_delta = self.height - raw_height
        self.elapsed = torch.where(self.jumping, t, torch.where(step, step_elapsed, 0.))
        self.terrain_mode = torch.where(step & ~self.jumping, step_mode, 0.)
        self.release_speed, self.push_end = release, push_end
        return torch.where(self.jumping, raw_height, effective)

    def metrics(self):
        return {"/reference/ax_m_s2": self.acceleration.abs().mean(),
                "/reference/pitch_rad": self.pitch.abs().mean(),
                "/reference/height_delta_m": self.height_delta.mean(),
                **{f"/reference/phase_{phase}_fraction": (self.phase == phase).float().mean() for phase in range(9)}}
