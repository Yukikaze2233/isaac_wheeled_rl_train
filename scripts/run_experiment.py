#!/usr/bin/env python3
"""Run one registered experiment (local toy mode runs anywhere; server mode
prints the Isaac Lab command).

    python3 scripts/run_experiment.py --list
    python3 scripts/run_experiment.py --exp exp001_him_latent16 --iterations 60 \
        --log-dir runs/exp001
    python3 scripts/run_experiment.py --exp exp001_him_latent16 --resume \
        --log-dir runs/exp001        # continue from the latest checkpoint

Local mode trains the branch on ToyReachExtEnv (CPU) with per-iteration
metrics appended to <log-dir>/metrics.jsonl — the same loop the server uses,
swapping the toy env for the Isaac Lab task named in the spec.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch  # noqa: E402

from toy_env_ext import ToyReachExtEnv  # noqa: E402
from wheeled_algo.algorithms import ExtTrainCfg  # noqa: E402
from wheeled_algo.experiments import REGISTRY, branch_trainer, get  # noqa: E402


class EpisodeProbe(ToyReachExtEnv):
    """Toy env subclass that tracks per-episode reward for the metrics log."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.episodes: list[float] = []
        self._per_env = torch.zeros(self.num_envs, device=self.device)

    def step(self, action):
        obs, reward, done, extras = super().step(action)
        self._per_env += reward
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() > 0:
            self.episodes.extend(self._per_env[ids].tolist())
            self._per_env[ids] = 0.0
        return obs, reward, done, extras

    def mean(self):
        return sum(self.episodes[-50:]) / max(len(self.episodes[-50:]), 1)

    def mean(self):
        return sum(self.episodes[-50:]) / max(len(self.episodes[-50:]), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, help="experiment name (see --list)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--backend", choices=["local", "isaaclab"], default="local")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.list:
        for name in REGISTRY:
            spec = REGISTRY[name]
            print(f"{name:24s} [{spec.branch:9s}] {spec.description}")
        return
    assert args.exp, "--exp required (see --list)"
    spec = get(args.exp)
    log_dir = args.log_dir or os.path.join("runs", spec.name)
    os.makedirs(log_dir, exist_ok=True)

    if args.backend == "isaaclab":
        print(f"server training for {spec.name}:")
        print(f"  ./isaaclab.sh -p scripts/train.py --task {spec.isaaclab_task} "
              f"--num_envs 4096 --max_iterations 20000 --headless "
              f"--log-dir {log_dir}   # {spec.isaaclab_note}")
        return

    torch.manual_seed(args.seed)
    env = EpisodeProbe(num_envs=128, max_ep_steps=40, device="cpu", seed=args.seed,
                       **spec.env_kwargs)
    cfg = ExtTrainCfg(num_steps_per_env=16, learning_rate=3e-4, num_learning_epochs=3,
                      **spec.overrides)
    trainer_cls = branch_trainer(spec.branch)
    critic_dim = env.obs_dim + 1
    trainer = trainer_cls(env, obs_dim=env.obs_dim, priv_dim=critic_dim,
                          hist_len=env.hist_len, action_dim=env.num_actions,
                          device="cpu", cfg=cfg)

    start_iter = 0
    ckpt_path = os.path.join(log_dir, "checkpoint.pt")
    if args.resume and os.path.exists(ckpt_path):
        start_iter = trainer.loop.load(ckpt_path)
        print(f"resumed from iteration {start_iter}")

    probe = env
    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    metrics_f = open(metrics_path, "a" if args.resume else "w")

    def hook(it: int, stats: dict):
        metrics_f.write(json.dumps({
            "iteration": it, "ep_reward": probe.mean(),
            "extra": stats["extra"], "kl": stats["kl"], "lr": trainer.loop.learning_rate,
        }) + "\n")
        metrics_f.flush()

    trainer.loop.learn(iterations=args.iterations, metrics_hook=hook)
    trainer.loop.save(ckpt_path)
    metrics_f.close()
    print(f"done: {spec.name} at iteration {trainer.loop.current_iteration}, "
          f"ep_reward={probe.mean():.2f}, artifacts in {log_dir}/")


if __name__ == "__main__":
    main()
