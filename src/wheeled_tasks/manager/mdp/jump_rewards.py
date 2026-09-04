"""Jump takeoff + full-trajectory reward family (flight-phase reward family).

Design insight:: never reward the apex alone — generate a
reference trajectory for the whole jump and track it densely:
    crouch -> takeoff impulse -> flight -> quadratic landing trajectory
This module provides the trajectory generator and the four dense reward terms
used during the jump window:
    track_h_traj    height follows the reference curve
    track_vz_traj   vertical velocity follows the reference derivative
    wheel_zero_torque_exp  wheels free-spinning in flight
    precontact_speed_exp   wheels spun up to match ground speed before touch
Pure torch: unit-testable without Isaac Sim. The env feeds telemetry in and
merges the returned reward terms while the jump window is active.
"""
import torch


class JumpTrajectory:
    """Quadratic landing reference: h(0)=h0, h(T)=hf, h'(T)=0 (the landing reference
    semantics — a *reference* the policy tracks, gravity is the physics sim's job).

        h(t) = h0 + 2Δ·(t/T) − Δ·(t/T)²,   Δ = hf − h0
        v(t) = 2Δ/T − 2Δ·t/T²
    """

    def __init__(self, h0: torch.Tensor, target_height: float, duration: float = 0.3):
        self.h0 = h0
        self.hf = target_height
        self.T = max(duration, 1e-3)
        self.delta = self.hf - self.h0

    def height(self, t: torch.Tensor) -> torch.Tensor:
        s = (t.clamp(0.0, self.T)) / self.T
        return self.h0 + 2.0 * self.delta * s - self.delta * s * s

    def vel_z(self, t: torch.Tensor) -> torch.Tensor:
        s = (t.clamp(0.0, self.T)) / self.T
        return (2.0 * self.delta / self.T) * (1.0 - s)


def jump_window_rewards(
    in_jump: torch.Tensor,          # (N,) bool: jump window active
    t_in_window: torch.Tensor,      # (N,) s since window start
    traj: JumpTrajectory,
    base_height: torch.Tensor,      # (N,) m
    base_vel_z: torch.Tensor,       # (N,) m/s
    wheel_speed: torch.Tensor,      # (N, 2) rad/s
    wheel_contact: torch.Tensor,    # (N,) bool
    root_speed_x: torch.Tensor,     # (N,) m/s forward speed
    wheel_radius: float = 0.06,
    sigma_h: float = 0.01,
    sigma_vz: float = 0.25,
    sigma_zero_torque: float = 1.5,
    sigma_precontact: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Dense jump-window rewards; all terms are zero outside the window."""
    mask = in_jump.float()
    zeros = torch.zeros_like(mask)

    h_ref = traj.height(t_in_window)
    vz_ref = traj.vel_z(t_in_window)
    track_h = torch.exp(-torch.square(base_height - h_ref) / sigma_h**2) * mask
    track_vz = torch.exp(-torch.square(base_vel_z - vz_ref) / sigma_vz**2) * mask

    # wheels free in flight: speed ~ 0 reward wheels free-spinning in flight
    wheel_free = (~wheel_contact).float() * mask
    wheel_speed_norm = wheel_speed.abs().mean(dim=-1)
    zero_torque_r = torch.exp(-wheel_speed_norm / sigma_zero_torque) * wheel_free

    # pre-contact spin-up: wheel surface speed should approach ground speed
    target_wheel = (root_speed_x / wheel_radius).abs()
    precontact = wheel_free * (t_in_window > 0.5 * traj.T).float()
    spin_r = torch.exp(-torch.square(target_wheel - wheel_speed_norm) / sigma_precontact**2) * precontact

    return {
        "track_h_traj": torch.where(in_jump, track_h, zeros),
        "track_vz_traj": torch.where(in_jump, track_vz, zeros),
        "air_wheel_zero_torque_exp": zero_torque_r,
        "precontact_speed_exp": spin_r,
    }
