"""Continuous CONTROL angles with a frozen hip/auxiliary closure branch.

Collision geometry and angle limits are unchanged. Only integer turns are
resolved. A provider must consume one chronologically ordered encoder stream;
mixing ideal state and delayed USB feedback is not a valid stream. Consecutive
physical encoder increments must be less than pi, as required by angle unwrap.
"""
from __future__ import annotations

import math

import torch


class ContinuousEncoderAngles:
    """Keep accumulated rotations; resolve each auxiliary's initial winding once.

    ``relative_limits`` is [left/right, lower/upper] for auxiliary minus hip,
    taken directly from the frozen closed-chain lookup. ``valid`` never grants
    an out-of-domain mechanism pose credit merely because it was unwrapped.
    The class does not read or write simulator state.
    """
    HIP = (0, 3)
    AUXILIARY = (1, 4)

    def __init__(self, count, device, relative_limits):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Encoder provider requires a positive environment count")
        limits = torch.as_tensor(relative_limits, dtype=torch.float32, device=device).clone()
        if (limits.shape != (2, 2) or not torch.isfinite(limits).all()
                or not ((limits[:, 1] > limits[:, 0]) & (limits[:, 1] - limits[:, 0] < math.tau)).all()):
            raise ValueError("Frozen relative-angle branches must be finite, ordered and narrower than one turn")
        self.relative_limits = limits
        self.q = torch.zeros(count, 6, device=device)
        self.last_raw = torch.zeros_like(self.q)
        self.winding = torch.zeros_like(self.q, dtype=torch.long)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids, raw_control):
        ids = torch.as_tensor(ids, device=self.q.device, dtype=torch.long)
        raw = torch.as_tensor(raw_control, device=self.q.device, dtype=self.q.dtype)
        if raw.shape == self.q.shape:
            raw = raw[ids]
        if raw.shape != (len(ids), 6):
            raise ValueError("Encoder reset requires selected or full CONTROL angles")
        turns = torch.zeros_like(raw, dtype=torch.long)
        turns[:, self.AUXILIARY] = self._branch_turns(raw)
        self.winding[ids] = turns
        self.q[ids] = self._apply_turns(raw, turns)
        self.last_raw[ids] = raw
        self.initialized[ids] = True
        return self.q

    @staticmethod
    def _integer_turns(value):
        # Invalid samples remain invalid angles. Do not cast NaN/inf to an
        # arbitrary integer that could corrupt a later encoder winding.
        finite = torch.where(torch.isfinite(value), value, 0.)
        return torch.round(finite / math.tau).to(torch.long)

    @staticmethod
    def _apply_turns(raw, turns):
        # Zero winding is a bitwise identity (also for signed zero). Adding
        # wrapped floating deltas instead would slowly perturb native PD even
        # in ordinary episodes that never cross an encoder boundary.
        return torch.where(turns == 0, raw, raw + turns.to(raw.dtype) * math.tau)

    def _branch_turns(self, q):
        difference = q[:, self.AUXILIARY] - q[:, self.HIP]
        midpoint = self.relative_limits.mean(-1)
        return self._integer_turns(midpoint - difference)

    def realign(self, ids):
        """Resolve the branch at release, retaining accumulated hip rotations.

        Use only at a new controller activation. Normal updates never silently
        reselect a mechanism branch or rewrite accumulated orbit angles.
        """
        turns = self.winding[ids].clone()
        turns[:, self.AUXILIARY] += self._branch_turns(self.q[ids])
        self.winding[ids] = turns
        aligned = self._apply_turns(self.last_raw[ids], turns)
        self.q[ids] = torch.where(torch.isfinite(self.q[ids]), aligned, self.q[ids])
        return self.q

    def _project_winding(self, raw_control):
        if raw_control.shape != self.q.shape or raw_control.device != self.q.device:
            raise ValueError("Encoder sample requires full CONTROL angles on the provider device")
        return self.winding - self._integer_turns(raw_control - self.last_raw)

    def project(self, raw_control):
        """Map a sample into the current winding without changing provider state."""
        projected = self._apply_turns(raw_control, self._project_winding(raw_control))
        # A nonfinite observation invalidates accumulated angle history until
        # an explicit reset; finite later samples cannot recover missing turns.
        return torch.where(torch.isfinite(self.q), projected, self.q)

    def update(self, raw_control):
        turns = self._project_winding(raw_control)
        projected = self._apply_turns(raw_control, turns)
        self.q.copy_(torch.where(torch.isfinite(self.q), projected, self.q))
        self.winding.copy_(turns)
        self.last_raw.copy_(raw_control)
        return self.q

    @property
    def valid(self):
        difference = self.q[:, self.AUXILIARY] - self.q[:, self.HIP]
        return (self.initialized[:, None] & torch.isfinite(self.q).all(-1)[:, None]
                & (difference >= self.relative_limits[:, 0] - 1e-6)
                & (difference <= self.relative_limits[:, 1] + 1e-6))
