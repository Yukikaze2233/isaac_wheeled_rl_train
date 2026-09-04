"""Unit tests for the pure-torch mdp components (delay buffer, command sampler)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from wheeled_tasks.manager.mdp import (
    DelayBuffer,
    SpecialModeEntryCfg,
    SpecialModeUniformVelocityCommand,
    SpecialModeUniformVelocityCommandCfg,
)


def test_delay_buffer():
    torch.manual_seed(0)
    buf = DelayBuffer(num_envs=4, dim=3, max_lag=4, device="cpu")
    frames = [torch.randn(4, 3) for _ in range(6)]
    out = None
    for f in frames:
        out = buf.compute(f)
    # all lags = 1 by default -> output equals previous frame
    assert torch.allclose(out, frames[-2]), "lag=1 must return previous frame"

    buf.resample_uniform(1, 4)  # per-env random lags in [1, 4]
    # push a known ramp and verify each env gets its own lag back
    buf.reset()
    ramp = []
    for k in range(6):
        v = torch.full((4, 3), float(k))
        out = buf.compute(v)
        ramp.append(out.clone())
    lags = buf.time_lag.tolist()
    for i, lag in enumerate(lags):
        assert torch.allclose(ramp[-1][i], torch.full((3,), float(6 - 1 - lag))), \
            f"env {i}: lag {lag} mismatch"
    print(f"delay OK (lags={lags})")


def test_command_sampler():
    torch.manual_seed(0)
    modes = {
        "spin": SpecialModeEntryCfg(rel_envs=0.5, ranges={"vx": ((-0.1, 0.1),), "yaw_rate": ((6.0, 8.0),)}),
        "dash": SpecialModeEntryCfg(rel_envs=0.5, ranges={"vx": ((2.0, 3.0),), "yaw_rate": ((-1.0, 1.0),)},
                                    iteration_start=100),
    }
    cfg = SpecialModeUniformVelocityCommandCfg(
        base_ranges={"vx": (-1.0, 1.0), "yaw_rate": (-3.0, 3.0)},
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        special_modes=modes,
    )
    cmd = SpecialModeUniformVelocityCommand(cfg, num_envs=2000, device="cpu")
    cmd.resample_all(iteration=0)  # dash not yet active
    n_spin = int(cmd.mode_mask("spin").sum())
    assert 0.35 * 2000 <= n_spin <= 0.65 * 2000, f"spin bucket ratio off: {n_spin}"
    assert int(cmd.mode_mask("dash").sum()) == 0, "dash must be gated by iteration"

    cmd.resample_all(iteration=200)  # both active
    spin = cmd.mode_mask("spin")
    dash = cmd.mode_mask("dash")
    assert not torch.any(spin & dash), "mode buckets must be disjoint"
    assert torch.all(cmd.command[spin, 2].abs() >= 6.0), "spin yaw-rate out of range"
    assert torch.all(cmd.command[dash, 0] >= 2.0), "dash vx out of range"
    print(f"commands OK (spin={int(spin.sum())}, dash={int(dash.sum())})")


if __name__ == "__main__":
    test_delay_buffer()
    test_command_sampler()
    print("MDP TESTS PASSED")
