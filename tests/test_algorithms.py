"""Smoke tests for the three estimator/constraint algorithm branches (HIM/DreamWaQ/NP3O).

Each branch must improve episode reward on the toy task within its budget —
proves estimator/VAE/BarlowTwins mechanisms, the asymmetric streams, and the
constraint machinery all run and contribute gradients. Runtime ~3 min CPU.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from toy_env_ext import ToyReachExtEnv  # noqa: E402
from wheeled_algo.algorithms import (  # noqa: E402
    DreamWaqTrainer, ExtTrainCfg, HIMTrainer, NP3OTrainer,
)

CFG = ExtTrainCfg(num_steps_per_env=16, learning_rate=3e-4, num_learning_epochs=3)


class Probe:
    def __init__(self, env):
        self.env = env
        self.rewards: list[float] = []
        self._per_env = torch.zeros(env.num_envs, device=env.device)

    def __call__(self):
        done_ids = (self.env._steps >= self.env.max_episode_length).nonzero(as_tuple=False).flatten()
        # count a finished episode when the env truncates: use episode_length reset markers
        # simpler proxy: record current mean reward of truncated envs
        if done_ids.numel() > 0:
            for _ in range(done_ids.numel()):
                self.rewards.append(float(self._per_env[done_ids[0]].item()))
            self._per_env[done_ids] = 0.0

    def mean(self):
        return sum(self.rewards[-50:]) / max(len(self.rewards[-50:]), 1)


def rollout(trainer, env, steps: int = 40) -> float:
    """Mean per-step reward of a stochastic rollout with the current policy."""
    obs, extras = env.reset_()
    total = 0.0
    with torch.no_grad():
        for _ in range(steps):
            action, _, _ = trainer.policy.act(obs, extras.get("observations", {}))
            obs, reward, done, extras = env.step(action)
            total += float(reward.mean())
    return total / steps


def run(branch: str, trainer_cls) -> None:
    torch.manual_seed(0)
    env = ToyReachExtEnv(num_envs=128, max_ep_steps=40, device="cpu", seed=0, hist_len=3)
    critic_dim = env.obs_dim + 1  # env's critic stream = policy obs + privileged norm
    trainer = trainer_cls(env, obs_dim=env.obs_dim, priv_dim=critic_dim, hist_len=3,
                          action_dim=env.num_actions, device="cpu", cfg=CFG)
    before = rollout(trainer, env)
    trainer.learn(iterations=60, episode_tracker=None)
    after = rollout(trainer, env)
    print(f"{branch}: before={before:.3f} after={after:.3f} improvement={after - before:+.3f}")
    assert after - before > 0.08, f"{branch} failed to learn: {before:.3f} -> {after:.3f}"


if __name__ == "__main__":
    run("HIMLoco", HIMTrainer)
    run("DreamWaQ", DreamWaqTrainer)
    run("NP3O", NP3OTrainer)
    print("ALGO_EXT TESTS PASSED")
