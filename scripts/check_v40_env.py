#!/usr/bin/env python3
"""Default: CPU preflight only. Explicit --apply --research: <=2 envs, <=40 steps, no training."""
from __future__ import annotations

import argparse
import json

from train_v40 import add_common_arguments, launch_app, make_env, preflight, print_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--apply", action="store_true", help="Explicit permission for bounded simulator checks")
    parser.add_argument("--max_steps", "--max-steps", type=int, default=10)
    args = parser.parse_args(argv)
    if not 1 <= args.num_envs <= 2:
        parser.error("check_env num_envs must be in [1,2]")
    if not 1 <= args.max_steps <= 40:
        parser.error("check_env max_steps must be in [1,40]")
    if args.apply and not args.research:
        parser.error("simulation checks require BOTH --apply and --research")
    report, _, _ = preflight(args)
    report["note"] = "No training; collision approval cannot be waived by --apply or --research"
    print_preflight(report)
    if args.preflight_only or not args.apply or not report["ready"]:
        return 0 if report["ready"] else 2

    launcher = launch_app(args)
    simulation_app = launcher.app
    env = None
    try:
        import torch
        env = make_env(args)
        obs, _ = env.reset()
        joint_limits_at_reset = env.check_physics_joint_limits()
        steps = 0
        terminated_count = 0
        for _ in range(args.max_steps):
            if not simulation_app.is_running():
                break
            assert obs["policy"].shape == (args.num_envs, 125)
            assert obs["critic"].shape == (args.num_envs, 29)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
            # Read-only repeated observation retrieval must not push another history frame.
            before = obs["policy"].clone()
            repeated = env._get_observations()["policy"]
            assert torch.equal(before, repeated), "same-tick history changed"
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((args.num_envs, 6), device=env.device))
            assert torch.isfinite(reward).all()
            assert terminated.dtype == torch.bool and truncated.dtype == torch.bool
            assert env.contact_sensor.data.net_forces_w.shape == (args.num_envs, 7, 3)
            steps += 1
            terminated_count += int(terminated.sum().item())
            if terminated.any():
                break  # Diagnostic failures must not become a long auto-reset rollout.
        print(json.dumps({"simulation_steps": steps, "terminated_count": terminated_count,
                          "collision_filter_usd": env.collision_filter_report,
                          "joint_limits_usd": env.joint_limit_usd_report,
                          "joint_limits_physx_at_reset": joint_limits_at_reset,
                          "joint_limits_physx_after_steps": env.check_physics_joint_limits(),
                          "scope": "bounded environment wiring check, not policy quality or hardware approval"},
                         allow_nan=False))
        return 0 if steps == args.max_steps and terminated_count == 0 else 3
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
