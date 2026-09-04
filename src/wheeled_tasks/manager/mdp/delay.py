"""Per-env observation / action delay buffer (self-implemented).

Isaac Lab has no built-in pipeline delay; the proven solution here is a
per-env ring buffer whose lag is resampled on reset so the policy cannot
exploit a fixed frame difference. Delays are expressed in 50 Hz control steps.
"""
import torch


class DelayBuffer:
    def __init__(self, num_envs: int, dim: int, max_lag: int, device: str):
        self.max_lag = max(int(max_lag), 1)
        # history[k] = value pushed k steps ago (k=0 is the newest frame).
        self.history = torch.zeros(self.max_lag + 1, num_envs, dim, device=device)
        self.time_lag = torch.ones(num_envs, dtype=torch.long, device=device)
        self.num_envs, self.dim = num_envs, dim
        self._ptr = 0  # newest-frame slot in the ring

    def set_time_lag(self, lag: torch.Tensor | int) -> None:
        if isinstance(lag, int):
            self.time_lag.fill_(int(lag))
        else:
            self.time_lag.copy_(lag.detach().to(dtype=torch.long, device=self.time_lag.device))
        self.time_lag.clamp_(1, self.max_lag)

    def resample_uniform(self, low: int, high: int, env_ids: torch.Tensor | None = None) -> None:
        """Sample integer lag in [low, high] per env (high inclusive)."""
        ids = torch.arange(self.num_envs, device=self.history.device) if env_ids is None else env_ids
        if high <= low:
            self.time_lag[ids] = int(low)
            return
        samples = torch.randint(low, high + 1, (ids.numel(),), device=self.history.device)
        self.time_lag[ids] = samples.clamp(1, self.max_lag)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Push the current frame, return each env's frame delayed by its lag.

        Preallocated ring: write pointer advances in place (no per-step
        (L+1,N,D) allocation, which matters at 50 Hz x 4 buffers x 4096 envs).
        Slot 0 is the newest frame after the push.
        """
        self._ptr = (self._ptr - 1) % (self.max_lag + 1)
        self.history[self._ptr].copy_(x)
        # newest-first absolute indices for each env's lag
        idx = (self._ptr + self.time_lag.clamp(1, self.max_lag)) % (self.max_lag + 1)
        return self.history[idx, torch.arange(self.num_envs, device=x.device)]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero history so resets cannot leak stale observations."""
        if env_ids is None or env_ids.numel() == self.num_envs:
            self.history.zero_()
        else:
            self.history[:, env_ids] = 0.0
