#!/usr/bin/env python3
"""Bounded deterministic V40 playback with strict checkpoint/asset identity checks."""
from __future__ import annotations

import argparse
from pathlib import Path

from train_v40 import add_common_arguments, checked_checkpoint, launch_app, make_env, make_manifest, preflight, print_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, help="Stock V40 checkpoint beside its run_manifest.json")
    parser.add_argument("--max_steps", "--max-steps", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("max_steps must be positive")
    if not args.preflight_only and args.checkpoint is None:
        parser.error("--checkpoint is required for playback")
    report, contract, asset = preflight(args)
    actor = None
    if not report["blockers"]:
        try:
            manifest = make_manifest(contract, asset, args)
            if args.checkpoint:
                actor, _ = checked_checkpoint(args.checkpoint, manifest)
        except Exception as exc:
            report["blockers"].append(f"checkpoint rejected: {exc}")
    report["ready"] = not report["blockers"]
    print_preflight(report)
    if args.preflight_only or not report["ready"]:
        return 0 if report["ready"] else 2

    launcher = launch_app(args)
    simulation_app = launcher.app
    env = None
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        env = make_env(args)
        if env.contract_sha256 != manifest["contract_sha256"] or env.asset_manifest_sha256 != manifest["asset_manifest_sha256"]:
            raise RuntimeError("contract/asset changed during playback startup")
        env = RslRlVecEnvWrapper(env)
        actor = actor.to(args.device).eval()
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args.max_steps):
                if not simulation_app.is_running():
                    break
                actions = actor(obs["policy"])
                if not torch.isfinite(actions).all():
                    raise RuntimeError("actor produced non-finite actions")
                obs, _, _, _ = env.step(actions)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
