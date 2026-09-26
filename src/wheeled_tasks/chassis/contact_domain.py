"""Startup contact domains expressed as effective wheel/ground coefficients."""
from dataclasses import replace
import math

import torch


class ContactDomain:
    def __init__(self, count, config, seed, profiles=None):
        self.cfg = dict(config)
        fraction = config["enabled_fraction"]
        low, high = config["friction_range"]
        if not all(math.isfinite(x) for x in (fraction, low, high)) or not 0 <= fraction <= 1 or not 0 < low <= .5 <= high:
            raise ValueError("Contact domain must include the nominal friction")
        generator = torch.Generator().manual_seed(seed ^ 0x46524943)
        self.domain_draw = torch.rand(count, generator=generator)
        enabled = self.domain_draw < fraction
        # Half of randomized scenes explicitly cover the requested low-grip band.
        low_grip = torch.rand(count, generator=generator) < config.get("low_grip_fraction", .5)
        sample = torch.rand(count, generator=generator)
        mu = torch.where(low_grip, low + (.4 - low) * sample, .4 + (high - .4) * sample)
        self.mu = torch.where(enabled, mu, .5)
        if profiles is not None:
            if len(profiles) != count:
                raise ValueError("One contact profile is required per environment")
            self.mu = torch.tensor([.5 if p is None else float(p["friction"]) for p in profiles])
            if not bool(torch.isfinite(self.mu).all() & (self.mu >= low).all() & (self.mu <= high).all()):
                raise ValueError("Contact evaluation profile outside declared domain")

    def surfaces(self, surfaces, index):
        # Unit robot material and multiply combine preserve the old nominal
        # contact values on every surface, including spatial material transitions.
        factor = float(self.mu[index]) / .5
        lo, hi = self.cfg["friction_range"]
        return [replace(s, friction=min(hi, max(lo, s.friction * factor))) for s in surfaces]

    def summary(self):
        return {"combine_mode": "multiply", "robot_material_friction": 1.,
                "static_equals_dynamic": True, "sampling": "startup_episode_static",
                "min_effective_friction": float(self.mu.min()), "max_effective_friction": float(self.mu.max()),
                "nominal_fraction": float((self.mu == .5).float().mean()),
                "low_grip_fraction": float((self.mu < .4).float().mean())}
