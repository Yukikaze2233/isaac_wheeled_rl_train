#!/usr/bin/env python3
"""Train the wheeled-biped flat policy with rsl_rl (Isaac Sim required).

Run inside the Isaac Lab environment, from the repo root:
    ./isaaclab.sh -p scripts/train.py --task WheeledBiped-Flat-v0 \
        --num_envs 4096 --max_iterations 20000 --headless
"""
import argparse
import os
import sys

# local src/ import so `pip install -e .` is optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train WheeledBiped flat policy (rsl_rl)")
parser.add_argument("--task", default="WheeledBiped-Flat-v0", type=str)
parser.add_argument("--num_envs", default=4096, type=int)
parser.add_argument("--max_iterations", default=20000, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--log_dir", default="logs/wheeled_biped_flat", type=str)
parser.add_argument("--checkpoint", default=None, type=str, help="resume from .pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---- heavy imports after app launch ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import wheeled_tasks.direct.wheeled_biped  # noqa: F401,E402  (registers the task)
from wheeled_tasks.agents import WheeledBipedFlatPPORunnerCfg  # noqa: E402
from wheeled_tasks.direct.wheeled_biped.wheeled_biped_env_cfg import (  # noqa: E402
    WheeledBipedFlatEnvCfg,
    WheeledBipedRoughEnvCfg,
)

gym.register(id="WheeledBiped-Flat-v0", entry_point="wheeled_tasks.direct.wheeled_biped.wheeled_biped_env:WheeledBipedEnv")
gym.register(id="WheeledBiped-Rough-v0", entry_point="wheeled_tasks.direct.wheeled_biped.wheeled_biped_env:WheeledBipedEnv")


def main():
    torch.manual_seed(args.seed)

    env_cfg_cls = WheeledBipedRoughEnvCfg if "Rough" in args.task else WheeledBipedFlatEnvCfg
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not args.headless else None)
    # Isaac Lab -> rsl_rl bridge: maps our dict streams {"policy","critic"}
    # onto rsl_rl's obs_groups for asymmetric actor-critic.
    env = RslRlVecEnvWrapper(env, obs_groups={"policy": ["policy"], "critic": ["critic"]})

    agent_cfg = WheeledBipedFlatPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.log_dir, device=env.device)
    if args.checkpoint:
        runner.load(args.checkpoint)
    runner.learn(num_learning_iterations=args.max_iterations)
    runner.save(os.path.join(args.log_dir, "model_final.pt"))
    env.close()


if __name__ == "__main__":
    main()
    sim_app.close()
