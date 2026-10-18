"""Raw simulated accelerometer history for the direct recovery controller.

The requested time comes from the existing uplink; no second delay or RNG is
sampled. Actual stored timestamps define interpolation and episode boundaries.
This is a shared-source simulation approximation, not independent CAN packets.
"""
import math

import torch


class RecoveryAccelerationCache:
    def __init__(self, count, device, dt, maximum_age_s):
        self.dt = dt
        self.capacity = math.ceil(maximum_age_s / dt) + 2
        self.values = torch.zeros(self.capacity, count, 3, device=device)
        self.stamps_s = torch.full((self.capacity, count), torch.nan,
                                   dtype=torch.float64, device=device)
        self.rows = torch.arange(count, device=device)
        self.pointer = -1
        self.last_physical_tick = -1

    def reset(self, ids, acceleration, physical_tick):
        """Prime selected lanes at this instant, without creating a new frame."""
        self.values[:, ids] = acceleration[ids]
        self.stamps_s[:, ids] = physical_tick * self.dt

    def append(self, acceleration, physical_tick):
        if type(physical_tick) is not int or physical_tick < self.last_physical_tick:
            raise ValueError("Recovery acceleration requires a monotonic physical clock")
        if physical_tick == self.last_physical_tick:
            return
        if acceleration.shape != self.values.shape[1:]:
            raise ValueError("Recovery acceleration requires all three axes for each lane")
        self.pointer = (self.pointer + 1) % self.capacity
        self.values[self.pointer].copy_(acceleration)
        self.stamps_s[self.pointer].fill_(physical_tick * self.dt)
        self.last_physical_tick = physical_tick

    def read(self, requested_stamp_s):
        """Return the sample and its actual time after available-history clamps."""
        finite = torch.isfinite(self.stamps_s)
        earliest = torch.where(finite, self.stamps_s, torch.inf).amin(0)
        latest = torch.where(finite, self.stamps_s, -torch.inf).amax(0)
        stamp = requested_stamp_s.to(dtype=torch.float64).clamp(min=earliest, max=latest)
        older = torch.where(finite & (self.stamps_s <= stamp), self.stamps_s, -torch.inf).argmax(0)
        newer = torch.where(finite & (self.stamps_s >= stamp), self.stamps_s, torch.inf).argmin(0)
        old_time, new_time = self.stamps_s[older, self.rows], self.stamps_s[newer, self.rows]
        old_value, new_value = self.values[older, self.rows], self.values[newer, self.rows]
        span = new_time - old_time
        weight = torch.where(span > 0., (stamp - old_time) / span.clamp_min(1e-12), 0.)
        value = old_value + weight[:, None].to(old_value.dtype) * (new_value - old_value)
        available = finite.any(0) & torch.isfinite(requested_stamp_s)
        return torch.where(available[:, None], value, 0.), torch.where(available, stamp, torch.nan)
