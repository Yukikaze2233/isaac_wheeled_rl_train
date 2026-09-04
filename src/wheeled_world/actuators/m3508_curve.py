"""Data-driven motor model (measured torque-speed curve).

The measured torque-speed curve of a real motor (rpm, torque Nm pairs from a
dyno/datalog) becomes the torque limit: instead of a constant effort_limit,
the actuator can only deliver what the real motor delivers at the current
speed — capturing FOC-end gaps (torque droop at high speed) that a constant
limit hides. Pure torch, CSV-in, testable without Isaac Sim.
"""
import csv

import torch


class CurvedMotorModel:
    """Torque limit interpolation from a measured torque-speed curve."""

    def __init__(self, speed_rad_s: torch.Tensor, torque_nm: torch.Tensor,
                 gear_ratio: float = 1.0, name: str = "motor"):
        assert speed_rad_s.numel() == torque_nm.numel() and speed_rad_s.numel() >= 2
        order = torch.argsort(speed_rad_s)
        self.speed = speed_rad_s[order]
        self.torque = torque_nm[order].clamp(min=0.0)
        self.gear_ratio = gear_ratio
        self.name = name

    @classmethod
    def from_csv(cls, path: str, speed_col: str = "speed_rpm", torque_col: str = "torque_nm",
                 gear_ratio: float = 1.0, name: str = "motor") -> "CurvedMotorModel":
        """Load a measured curve CSV (rpm converted to output-shaft rad/s)."""
        with open(path) as f:
            rows = list(csv.DictReader(f))
        rpm = torch.tensor([float(r[speed_col]) for r in rows])
        nm = torch.tensor([float(r[torque_col]) for r in rows])
        return cls(rpm * 2 * 3.14159265 / 60.0, nm, gear_ratio=gear_ratio, name=name)

    def torque_limit(self, joint_speed: torch.Tensor) -> torch.Tensor:
        """Output-shaft torque limit at the given joint speed(s) (any shape).

        Beyond the measured range the limit holds at the last measured value
        (the real motor cannot exceed its stall/peak curve either way).
        """
        flat = joint_speed.reshape(-1)
        mag = flat.abs()
        limit = torchinterp(mag, self.speed, self.torque)
        return limit.reshape(joint_speed.shape) * self.gear_ratio

    def clip_torque(self, commanded: torch.Tensor, joint_speed: torch.Tensor) -> torch.Tensor:
        """Clip commanded torques to the curve limit, preserving sign."""
        limit = self.torque_limit(joint_speed)
        return commanded.clamp(-limit, limit)


def torchinterp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation (torch-native, differentiable w.r.t. fp)."""
    x_clamped = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, x_clamped) - 1
    idx = idx.clamp(0, xp.numel() - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    f0, f1 = fp[idx], fp[idx + 1]
    t = ((x_clamped - x0) / (x1 - x0 + 1e-12)).clamp(0.0, 1.0)
    return f0 + t * (f1 - f0)
