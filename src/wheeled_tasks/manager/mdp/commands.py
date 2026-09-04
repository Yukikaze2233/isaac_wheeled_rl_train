"""Velocity command sampler with mutually exclusive special modes.

Special-mode command curriculum:
- base velocity commands resampled every few seconds per env;
- a configurable share of non-standing envs is bucketed into ONE special mode
  (e.g. spin_low / spin_mid / dash); buckets are disjoint and re-rolled at
  every resample;
- each mode can be gated by training iteration (iteration_start/end) so
  behaviors enter progressively instead of from step 0;
- a mode's ranges may define several disjoint intervals, e.g. spin only
  clockwise or counter-clockwise: [(2pi, 3.25pi), (-3.25pi, -2pi)].

Pure torch: unit-testable without Isaac Sim.
"""
import torch


def _sample_range(spec, n: int, device) -> torch.Tensor:
    """Uniform sample from (low, high) or width-weighted from [(l, h), ...]."""
    if isinstance(spec[0], (int, float)):
        return (spec[1] - spec[0]) * torch.rand(n, device=device) + spec[0]
    widths = torch.tensor([hi - lo for lo, hi in spec], device=device, dtype=torch.float)
    pick = torch.multinomial(widths, n, replacement=True)
    out = torch.empty(n, device=device)
    for i, (lo, hi) in enumerate(spec):
        mask = pick == i
        m = int(mask.sum())
        if m:
            out[mask] = (hi - lo) * torch.rand(m, device=device) + lo
    return out


class SpecialModeEntryCfg:
    def __init__(
        self,
        rel_envs: float,
        ranges: dict[str, tuple | list],
        iteration_start: int = 0,
        iteration_end: int = -1,
    ):
        self.rel_envs = rel_envs
        self.ranges = ranges  # keys: "vx" | "yaw_rate"
        self.iteration_start = iteration_start
        self.iteration_end = iteration_end  # -1 = never expires


class SpecialModeUniformVelocityCommandCfg:
    def __init__(
        self,
        base_ranges: dict | None = None,
        resampling_time_range: tuple[float, float] = (5.0, 15.0),
        rel_standing_envs: float = 0.1,
        rel_heading_envs: float = 0.5,
        heading_control_stiffness: float = 5.0,
        special_modes: dict[str, SpecialModeEntryCfg] | None = None,
    ):
        self.base_ranges = base_ranges or {"vx": (-1.5, 1.5), "yaw_rate": (-3.14, 3.14)}
        self.resampling_time_range = resampling_time_range
        self.rel_standing_envs = rel_standing_envs
        self.rel_heading_envs = rel_heading_envs
        self.heading_control_stiffness = heading_control_stiffness
        self.special_modes = special_modes or {}


class SpecialModeUniformVelocityCommand:
    """Owns self.command[:, 0:3] = (vx, vy=0, yaw_rate) for all envs."""

    def __init__(self, cfg: SpecialModeUniformVelocityCommandCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.command = torch.zeros(num_envs, 3, device=device)
        self.is_standing_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.is_heading_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.heading_target = torch.zeros(num_envs, device=device)
        self.special_mode_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._mode_names: list[str] = list(cfg.special_modes.keys())

        self.time_left = torch.zeros(num_envs, device=device)

    @property
    def mode_names(self) -> list[str]:
        return self._mode_names

    def mode_mask(self, name: str) -> torch.Tensor:
        """Boolean mask of envs currently bucketed into mode `name`."""
        if name not in self._mode_names:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.special_mode_id == self._mode_names.index(name)

    def update(self, dt: float, iteration: int = 0) -> torch.Tensor:
        """Advance resample timers (iteration = common_step_counter // steps_per_iter)."""
        expired = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if expired.numel() > 0:
            lo, hi = self.cfg.resampling_time_range
            self.time_left[expired] = _sample_range((lo, hi), expired.numel(), self.device)
            self._resample(expired, iteration)
        self.time_left -= dt
        return expired

    def _resample(self, env_ids: torch.Tensor, iteration: int) -> None:
        cfg = self.cfg
        n = env_ids.numel()

        # 1) standing envs: zero command, no mode
        standing = torch.rand(n, device=self.device) < cfg.rel_standing_envs
        self.is_standing_env[env_ids] = standing
        self.is_heading_env[env_ids] = False
        self.special_mode_id[env_ids] = -1
        self.command[env_ids] = 0.0

        non_standing = env_ids[~standing]
        remaining = non_standing

        # 2) special-mode buckets: disjoint, order-agnostic by construction
        for name, mode in cfg.special_modes.items():
            if iteration < mode.iteration_start:
                continue
            if 0 <= mode.iteration_end < iteration:
                continue
            count = int(mode.rel_envs * non_standing.numel() + 0.5)
            if count <= 0 or remaining.numel() == 0:
                continue
            take = min(count, remaining.numel())
            perm = torch.randperm(remaining.numel(), device=self.device)
            chosen, remaining = remaining[perm[:take]], remaining[perm[take:]]
            self.special_mode_id[chosen] = self._mode_names.index(name)
            if "vx" in mode.ranges:
                self.command[chosen, 0] = _sample_range(mode.ranges["vx"], take, self.device)
            if "yaw_rate" in mode.ranges:
                self.command[chosen, 2] = _sample_range(mode.ranges["yaw_rate"], take, self.device)

        # 3) base ranges for the rest
        if remaining.numel() > 0:
            self.command[remaining, 0] = _sample_range(cfg.base_ranges["vx"], remaining.numel(), self.device)
            self.command[remaining, 2] = _sample_range(cfg.base_ranges["yaw_rate"], remaining.numel(), self.device)
            # heading command for a share of them (env must stabilize heading)
            is_heading = torch.rand(remaining.numel(), device=self.device) < cfg.rel_heading_envs
            self.is_heading_env[remaining] = is_heading
            self.heading_target[remaining] = (2 * torch.pi) * torch.rand(remaining.numel(), device=self.device) - torch.pi

    def resample_all(self, iteration: int = 0) -> None:
        """Force a global resample (used on env start / after large resets)."""
        self.time_left.zero_()
        self.update(self.cfg.resampling_time_range[0] + 1e-3, iteration)
