"""Forward terrain samples to a bounded height command; no simulator dependency."""
import math

import torch


class StepHeightAssist:
    """Temporal edge detection with per-environment history and integer hold ticks.

    Inputs are world-height samples [m] at the two forward wheel probes and unit
    planar travel directions. The caller owns sensing and the raw command. A
    missing sample invalidates history, but an existing hold expires normally.
    """

    def __init__(self, count, device, dt, config):
        self.cfg = dict(config)
        self.low, self.high = config["step_range_m"]
        self.bias, self.maximum = config["height_bias_m"], config["height_max_m"]
        self.tolerance = config.get("edge_tolerance_m", 1e-5)
        if not all(math.isfinite(v) for v in (dt, self.low, self.high, self.bias, self.maximum,
                self.tolerance, config["hold_seconds"], config["lookahead_m"])):
            raise ValueError("Step assistance requires finite parameters")
        if not (dt > 0 and 0 < self.low <= self.high and self.bias > 0 and self.maximum > self.bias
                and 0 <= self.tolerance < self.low and config["hold_seconds"] >= dt
                and config["lookahead_m"] > 0):
            raise ValueError("Invalid step detection or hold domain")
        self.hold_ticks = math.ceil(config["hold_seconds"] / dt - 1e-9)
        self.remaining = torch.zeros(count, dtype=torch.long, device=device)
        self.previous = torch.zeros(count, 2, device=device)
        self.direction = torch.zeros_like(self.previous)
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.triggered = torch.zeros_like(self.valid)
        self.wall = torch.zeros_like(self.valid)
        self.active = torch.zeros_like(self.valid)
        self.all_ids = torch.arange(count, device=device)

    def reset(self, ids):
        self.remaining[ids] = 0
        self.valid[ids] = False
        self.triggered[ids] = False
        self.wall[ids] = False
        self.active[ids] = False
        self.direction[ids] = 0.

    def update(self, raw_height, samples, valid, direction, eligible, enabled, *, ids=None, advance=True):
        ids = self.all_ids if ids is None else ids
        if advance:
            self.remaining[ids] = (self.remaining[ids] - 1).clamp_min(0)
        finite = torch.isfinite(samples[ids]).all(-1) & torch.isfinite(direction[ids]).all(-1)
        good = valid[ids].all(-1) & finite & eligible[ids]
        same_heading = (self.direction[ids] * direction[ids]).sum(-1) >= math.cos(math.pi / 4)
        direction_valid = direction[ids].norm(dim=-1) > .5
        changed_heading = (self.direction[ids].norm(dim=-1) > .5) & direction_valid & ~same_heading
        self.remaining[ids] = torch.where(changed_heading, 0, self.remaining[ids])
        delta = (samples[ids] - self.previous[ids]).amax(-1)
        observable_edge = good & self.valid[ids] & same_heading
        self.wall[ids] = observable_edge & (delta > self.high + self.tolerance)
        self.triggered[ids] = (observable_edge & enabled[ids] & ~self.wall[ids]
            & (delta >= self.low - self.tolerance) & (delta <= self.high + self.tolerance))
        self.remaining[ids] = torch.where(self.triggered[ids], self.hold_ticks, self.remaining[ids])
        self.remaining[ids] = torch.where(self.wall[ids] | ~enabled[ids], 0, self.remaining[ids])
        self.active[ids] = self.remaining[ids] > 0
        self.previous[ids] = torch.where(good[:, None], samples[ids], 0.)
        self.direction[ids] = torch.where((finite & direction_valid)[:, None], direction[ids], self.direction[ids])
        self.valid[ids] = good
        # Bias is always applied to raw input, never to last tick's effective height.
        return torch.where(self.active[ids], (raw_height[ids] + self.bias).clamp_max(self.maximum), raw_height[ids])

    def metrics(self):
        return {"/terrain/step_assist_active_fraction": self.active.float().mean(),
                "/terrain/step_detected_fraction": self.triggered.float().mean(),
                "/terrain/wall_detected_fraction": self.wall.float().mean(),
                "/terrain/forward_sample_valid_fraction": self.valid.float().mean()}
