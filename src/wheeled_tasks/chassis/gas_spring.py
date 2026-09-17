"""Conservative catalogue gas-spring prior; no invented damping or installation zero."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class GasSpringCurve:
    """Cubic force in normalized compression; positive force acts toward extension."""

    coefficients_n: tuple[float, float, float, float]
    stroke_m: float
    max_fraction: float = 0.9

    def __post_init__(self):
        if (len(self.coefficients_n) != 4 or not all(math.isfinite(x) and x >= 0 for x in self.coefficients_n)
                or self.coefficients_n[0] <= 0 or not math.isfinite(self.stroke_m) or self.stroke_m <= 0
                or not math.isfinite(self.max_fraction) or not 0 < self.max_fraction <= 1):
            raise ValueError("Invalid monotone force curve/domain")

    @classmethod
    def from_json(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        if data["normalized_fit_domain"][0] != 0:
            raise ValueError("Compression origin must be full extension")
        return cls(tuple(data["monomial_coefficients_n"]), data["stroke_m"], data["normalized_fit_domain"][1])

    def _fraction(self, compression_m):
        s = torch.as_tensor(compression_m)
        if not s.is_floating_point():
            s = s.float()
        if not bool(torch.isfinite(s).all()) or bool((s < 0).any()) or bool((s > self.stroke_m * self.max_fraction).any()):
            raise ValueError("Compression outside fitted range; extrapolation is not implicit")
        return s / self.stroke_m

    def force(self, compression_m):
        """Return extension-directed force magnitude [N] for compression [m]."""
        u = self._fraction(compression_m)
        a, b, c, d = self.coefficients_n
        return a + u * (b + u * (c + u * d))

    def potential_energy(self, compression_m):
        """Stored work [J] relative to full extension; dU/ds equals force magnitude."""
        u = self._fraction(compression_m)
        a, b, c, d = self.coefficients_n
        return self.stroke_m * u * (a + u * (b / 2 + u * (c / 3 + u * d / 4)))

    def force_from_pin_distance(self, pin_distance_m, *, extended_pin_distance_m):
        """Require installed pin spacing at full extension, not the catalogue body length."""
        if not math.isfinite(extended_pin_distance_m) or extended_pin_distance_m <= 0:
            raise ValueError("An explicit positive mounting reference is required")
        return self.force(extended_pin_distance_m - torch.as_tensor(pin_distance_m))
