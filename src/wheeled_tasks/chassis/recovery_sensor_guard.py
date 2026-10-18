"""Batched timestamp admission for six motors, orientation/gyro and acceleration.

Simulation may give the first seven groups one shared feedback timestamp. This
guard validates supplied metadata; it does not create independent CAN events.
Float64 second metadata is rounded to nanoseconds for the deployment comparisons.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Real

import torch


class RecoverySensorIssue(IntEnum):
    NONE = 0
    MISSING = 1
    FUTURE = 2
    EXPIRED = 3
    REVERSED = 4
    INCONSISTENT_SEQUENCE = 5
    SKEW = 6


@dataclass(frozen=True)
class RecoverySensorStatus:
    valid: torch.Tensor
    issue: torch.Tensor
    invalid_mask: torch.Tensor
    maximum_motor_age_ms: torch.Tensor
    imu_age_ms: torch.Tensor
    acceleration_age_ms: torch.Tensor
    channel_age_s: torch.Tensor


class RecoverySensorGuard:
    """Commit eight-channel history only for rows whose complete snapshot passes.

    Issue precedence is per channel; the row reports its first failing channel
    in motor 0..5, IMU 6, acceleration 7 order. Repeated snapshots remain valid
    while fresh. An unavailable age is -1 ms, matching the deployment status.
    """

    def __init__(self, count, device, *, feedback_max_age_s,
                 acceleration_max_age_s, sensor_max_skew_s):
        if type(count) is not int or count <= 0:
            raise ValueError("Recovery sensor guard needs a positive environment count")
        limits = (feedback_max_age_s, acceleration_max_age_s, sensor_max_skew_s)
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or not 0 < value <= .05 for value in limits):
            raise ValueError("Recovery sensor limits must be in (0, 50 ms]")
        self.count = count
        self.device = torch.device(device)
        self._maximum_age_s = torch.tensor(
            [float(feedback_max_age_s)] * 7 + [float(acceleration_max_age_s)],
            dtype=torch.float64, device=self.device)
        self.device = self._maximum_age_s.device
        self._maximum_skew_s = float(sensor_max_skew_s)
        self._previous_sequences = torch.zeros(count, 8, dtype=torch.long, device=self.device)
        self._previous_stamps_ns = torch.zeros(count, 8, dtype=torch.long, device=self.device)
        self._channel_bits = 2 ** torch.arange(8, dtype=torch.long, device=self.device)

    def reset(self, ids):
        """Clear selected episode histories without changing other rows."""
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long
                or ids.ndim != 1 or ids.device != self.device):
            raise ValueError("Recovery sensor reset needs long ids on the guard device")
        if torch.any((ids < 0) | (ids >= self.count)):
            raise ValueError("Recovery sensor reset ids are outside the environment batch")
        self._previous_sequences[ids] = 0
        self._previous_stamps_ns[ids] = 0

    def update(self, now_s, sequences, stamps_s):
        """Validate a float64 second clock and [environment, 8] sample metadata."""
        if (not isinstance(sequences, torch.Tensor) or sequences.dtype != torch.long
                or sequences.shape != (self.count, 8) or sequences.device != self.device):
            raise ValueError("Recovery sensor sequences must be long [environment, 8]")
        if (not isinstance(stamps_s, torch.Tensor) or stamps_s.dtype != torch.float64
                or stamps_s.shape != sequences.shape or stamps_s.device != self.device):
            raise ValueError("Recovery sensor stamps must be float64 [environment, 8]")
        if isinstance(now_s, torch.Tensor):
            if (now_s.dtype != torch.float64 or now_s.device != self.device
                    or now_s.shape not in (torch.Size([]), torch.Size([self.count]))):
                raise ValueError("Recovery clock must be float64 scalar or [environment]")
        elif isinstance(now_s, bool) or not isinstance(now_s, Real):
            raise ValueError("Recovery clock must be a scalar second value")
        now = torch.as_tensor(now_s, dtype=torch.float64, device=self.device).expand(self.count)[:, None]
        finite = torch.isfinite(now) & torch.isfinite(stamps_s)
        now_ns = torch.round(now * 1e9).long()
        stamps_ns = torch.round(stamps_s * 1e9).long()
        delta_ns = now_ns - stamps_ns
        age_available = finite & (stamps_ns > 0) & (stamps_ns <= now_ns)
        age_s = torch.where(age_available, delta_ns.double() * 1e-9, -.001)
        age_ms = torch.where(age_available, delta_ns.double() * 1e-6, -1.)
        previous = self._previous_sequences != 0
        channel_issue = torch.zeros_like(sequences)
        checks = (
            (RecoverySensorIssue.MISSING, ~finite | (now_ns <= 0) | (sequences <= 0) | (stamps_ns <= 0)),
            (RecoverySensorIssue.FUTURE, stamps_ns > now_ns),
            (RecoverySensorIssue.EXPIRED, age_s > self._maximum_age_s),
            (RecoverySensorIssue.REVERSED, previous & (
                (sequences < self._previous_sequences) | (stamps_ns < self._previous_stamps_ns))),
            (RecoverySensorIssue.INCONSISTENT_SEQUENCE, previous & (
                (sequences == self._previous_sequences) != (stamps_ns == self._previous_stamps_ns))),
        )
        for issue, failed in checks:
            channel_issue = torch.where((channel_issue == RecoverySensorIssue.NONE) & failed,
                                        int(issue), channel_issue)
        failed = channel_issue != RecoverySensorIssue.NONE
        invalid_mask = (failed * self._channel_bits).sum(-1)
        first_failed = failed.long().argmax(-1)
        issue = channel_issue.gather(1, first_failed[:, None]).squeeze(1)
        skew = ((invalid_mask == 0)
                & ((stamps_ns.amax(-1) - stamps_ns.amin(-1)).double() * 1e-9 > self._maximum_skew_s))
        invalid_mask = torch.where(skew, 0xff, invalid_mask)
        issue = torch.where(skew, int(RecoverySensorIssue.SKEW), issue)
        valid = invalid_mask == 0
        self._previous_sequences[valid] = sequences[valid]
        self._previous_stamps_ns[valid] = stamps_ns[valid]
        return RecoverySensorStatus(
            valid=valid, issue=issue, invalid_mask=invalid_mask,
            maximum_motor_age_ms=age_ms[:, :6].amax(-1),
            imu_age_ms=age_ms[:, 6], acceleration_age_ms=age_ms[:, 7],
            channel_age_s=age_s)
