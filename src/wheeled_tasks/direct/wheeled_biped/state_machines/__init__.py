"""Task state machines for the wheeled biped (self-implemented).

The manager watches env telemetry and drives a small FSM per env:

    NORMAL ──(jump trigger)──► JUMP_TAKEOFF ──(wheel contact lost)──► AIRBORNE
    AIRBORNE ──(wheel contact regained)──► NORMAL
    NORMAL ──(stair terrain flag)──► STEP_UP / STAIR

Each state contributes:
- its 7D control-mode flag vector (policy observation contract, CONTRACT.md)
- an optional height-command bias (e.g. tuck the legs in flight)
- optional reward weight overrides the env merges into its reward table

The env calls `update()` once per control step and reads `mode_flags`,
`height_bias` and `reward_overrides`. Pure torch/telemetry — no Isaac Lab
imports — so unit-testable without a simulator.
"""
import torch

# 7D flag layout per CONTRACT.md: normal/stair/slope/recover/jump/height_target/state_time
MODE_NAMES = ("normal", "stair", "slope", "recover", "jump", "height_target", "state_time")


class AirborneStateMachine:
    """Jump -> flight -> landing cycle with a fixed-duration height trajectory."""

    def __init__(self, num_envs: int, device: str, cfg: dict | None = None):
        cfg = cfg or {}
        self.num_envs = num_envs
        self.device = device
        self.body_height_threshold = cfg.get("body_height_threshold", 0.30)   # m: airborne entry
        self.min_down_vel = cfg.get("min_down_vel", 0.2)                      # m/s falling gate
        self.target_height = cfg.get("target_height", 0.30)                   # m landing target
        self.landing_duration = cfg.get("landing_duration_s", 0.3)            # s fixed traj
        self.dt = cfg.get("control_dt", 0.02)                                 # s per update
        self.state_time = torch.zeros(num_envs, device=device)                # s in current state
        self.in_air = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def state_id(self) -> torch.Tensor:
        return self.in_air.long()

    def update(self, base_height: torch.Tensor, base_vel_z: torch.Tensor,
               wheel_contact: torch.Tensor) -> None:
        dt = self.dt
        # entry: wheels leave ground while the base is well above stand height
        self.in_air = (~wheel_contact) & (base_height > self.body_height_threshold)
        self.state_time = torch.where(self.in_air, self.state_time + dt, torch.zeros_like(self.state_time))

    def mode_flags(self) -> torch.Tensor:
        """(N, 7) contract flags."""
        n = self.num_envs
        flags = torch.zeros(n, 7, device=self.device)
        flags[:, 0] = 1.0  # normal baseline
        flags[:, 4] = self.in_air.float()          # jump/airborne active
        flags[:, 5] = self.in_air.float()          # height_target follows traj
        flags[:, 6] = torch.clamp(self.state_time / max(self.landing_duration, 1e-6), 0, 1)
        return flags

    def height_bias(self) -> torch.Tensor:
        """In flight, bias the height command up (tuck/extend trajectory); 0 on ground."""
        return torch.where(self.in_air, self.target_height - self.body_height_threshold,
                           torch.zeros_like(self.state_time))

    def reward_overrides(self) -> dict:
        """Landing-window reward shaping (dense trajectory following)."""
        return {"landing_height_track": 1.0, "landing_vel_z": 0.5} if bool(self.in_air.any()) else {}


class StairStateMachine:
    """Stair / slope traversal detection driven by terrain flags.

    The flags come from mdp.terrain.terrain_flags_from_types (per-env patch
    mapping of the TerrainImporter) — or any per-env boolean the env computes.
    """

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.stair_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.slope_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, terrain_flag: torch.Tensor | None = None,
               slope_flag: torch.Tensor | None = None) -> None:
        if terrain_flag is not None:
            self.stair_active = terrain_flag.bool()
        if slope_flag is not None:
            self.slope_active = slope_flag.bool()

    def mode_flags(self) -> torch.Tensor:
        flags = torch.zeros(self.num_envs, 7, device=self.device)
        flags[:, 0] = 1.0
        flags[:, 1] = self.stair_active.float()  # stair mode
        flags[:, 2] = self.slope_active.float()  # slope mode
        return flags


class StateMachineManager:
    """Aggregates machines; policy sees one 7D flag vector (OR-composed)."""

    def __init__(self, num_envs: int, device: str, airborne_cfg: dict | None = None):
        self.airborne = AirborneStateMachine(num_envs, device, airborne_cfg)
        self.stair = StairStateMachine(num_envs, device)

    def update(self, base_height, base_vel_z, wheel_contact,
               terrain_flag=None, slope_flag=None) -> None:
        self.airborne.update(base_height, base_vel_z, wheel_contact)
        self.stair.update(terrain_flag, slope_flag)

    def mode_flags(self) -> torch.Tensor:
        flags = self.airborne.mode_flags().clone()
        stair_flags = self.stair.mode_flags()
        flags[:, 1] = torch.maximum(flags[:, 1], stair_flags[:, 1])
        flags[:, 2] = torch.maximum(flags[:, 2], stair_flags[:, 2])
        return flags

    def height_bias(self) -> torch.Tensor:
        return self.airborne.height_bias()

    def reward_overrides(self) -> dict:
        return self.airborne.reward_overrides()
