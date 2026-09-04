"""Branch-architecture smoke: branch ActorCritic classes resolved via the stock
rsl_rl OnPolicyRunner (class-name injection), trained on frame-stacked toy env.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from toy_env_ext import ToyReachExtEnv  # noqa: E402
from wheeled_algo.runners.branch_runners import make_runner  # noqa: E402

HIST = 3


def run(branch_class: str) -> float:
    torch.manual_seed(0)
    env = ToyReachExtEnv(num_envs=128, max_ep_steps=40, device="cpu", seed=0, hist_len=HIST)
    env.stack_policy_obs = True
    base, priv = env.obs_dim, env.obs_dim + 1

    train_cfg = {
        "algorithm": {"class_name": "PPO", "value_loss_coef": 2.0,
                      "use_clipped_value_loss": True, "clip_param": 0.2,
                      "entropy_coef": 0.005, "num_learning_epochs": 4,
                      "num_mini_batches": 4, "learning_rate": 3e-4,
                      "schedule": "adaptive", "gamma": 0.99, "lam": 0.95,
                      "desired_kl": 0.02, "max_grad_norm": 1.0},
        "policy": {"class_name": branch_class, "init_noise_std": 1.0,
                   "activation": "elu", "base_obs_dim": base, "hist_len": HIST,
                   "actor_hidden_dims": [128, 64], "critic_hidden_dims": [128, 64]},
        "num_steps_per_env": 16, "max_iterations": 40, "save_interval": 10_000,
        "empirical_normalization": False,
    }
    runner = make_runner(env, train_cfg, log_dir="/tmp/branch_smoke", device="cpu")
    runner.learn(num_learning_iterations=40)
    obs, extras = env.get_observations()
    total = 0.0
    with torch.no_grad():
        for _ in range(40):
            a = runner.alg.policy.act(obs, **{})[0] if False else runner.alg.policy.act(obs)
            obs, r, d, extras = env.step(a)
            total += float(r.mean())
    return total / 40


if __name__ == "__main__":
    for cls in ("ActorCriticHIM", "ActorCriticDreamWaq", "ActorCriticBarlowTwins"):
        score = run(cls)
        print(f"{cls}: rollout={score:.3f}")
        assert score > 0.35, f"{cls} degraded: {score:.3f}"
    print("RSL_RL BRANCH TESTS PASSED")
