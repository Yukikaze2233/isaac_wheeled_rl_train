"""rsl_rl smoke test: stock OnPolicyRunner (2.3.x) must learn the toy task.

Runs the exact stack used in training (rsl_rl runner + asymmetric critic
extras) on a toy VecEnv, no Isaac Sim required. Pass criterion: mean episode
reward improves significantly; checkpoint round-trip works (export_onnx.py
relies on the saved format).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from toy_env import ToyReachRslEnv  # noqa: E402


class EpisodeRewardProbe(ToyReachRslEnv):
    """Tracks per-episode reward externally (2.3.x runner no longer exposes it)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.episode_rewards: list[float] = []
        self._per_env = torch.zeros(self.num_envs, device=self.device)

    def step(self, action):
        obs, reward, done, extras = super().step(action)
        self._per_env += reward
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.episode_rewards.extend(self._per_env[done_ids].tolist())
            self._per_env[done_ids] = 0.0
        return obs, reward, done, extras


def main():
    torch.manual_seed(0)
    env = EpisodeRewardProbe(num_envs=256, max_ep_steps=50, device="cpu", seed=0)

    # same schema/numbers as WheeledBipedFlatPPORunnerCfg, scaled down
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 2.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.02,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [128, 64],
            "critic_hidden_dims": [128, 64],
            "activation": "elu",
        },
        "num_steps_per_env": 24,
        "max_iterations": 200,
        "save_interval": 10_000,
        "empirical_normalization": False,
    }

    runner = OnPolicyRunner(env, train_cfg, log_dir="/tmp/rsl_smoke_logs", device="cpu")
    runner.learn(num_learning_iterations=200)

    rewards = env.episode_rewards
    assert len(rewards) >= 40, f"too few finished episodes to evaluate: {len(rewards)}"
    first = sum(rewards[:20]) / 20
    last = sum(rewards[-20:]) / 20
    print(f"first20={first:.3f} last20={last:.3f} improvement={last - first:+.3f}")
    assert last > first + 1.0, f"rsl_rl failed to learn: {first:.3f} -> {last:.3f}"

    # checkpoint round-trip
    ckpt_path = "/tmp/rsl_smoke_model.pt"
    runner.save(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in ckpt, "checkpoint must contain model_state_dict (export_onnx.py relies on it)"
    print("RSL_RL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
