#!/usr/bin/env python3
"""Play a trained rsl_rl checkpoint in the Isaac Sim viewer (no training).

    ./isaaclab.sh -p scripts/play.py --checkpoint logs/wheeled_biped_flat/model_final.pt \
        --num_envs 4
Command is a fixed vx/wz pair; live keyboard teleop is an extension point.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Play WheeledBiped policy (rsl_rl checkpoint)")
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", default=4, type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402

import wheeled_tasks.direct.wheeled_biped  # noqa: F401,E402
from wheeled_tasks.direct.wheeled_biped.wheeled_biped_env_cfg import WheeledBipedFlatEnvCfg  # noqa: E402

gym.register(id="WheeledBiped-Flat-v0", entry_point="wheeled_tasks.direct.wheeled_biped.wheeled_biped_env:WheeledBipedEnv")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"]
    num_obs, num_actions, hidden = actor_dims_from_ckpt(state)

    agent = ActorCritic(num_obs, num_obs, num_actions, hidden, hidden, "elu", 1.0)
    agent.load_state_dict(state)
    agent.to(device)
    agent.eval()

    env_cfg = WheeledBipedFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make("WheeledBiped-Flat-v0", cfg=env_cfg, render_mode="rgb_array")

    obs, _ = env.reset()
    obs = obs["policy"] if isinstance(obs, dict) else obs
    while sim_app.is_running():
        with torch.no_grad():
            action = agent.act_inference(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        obs = obs["policy"] if isinstance(obs, dict) else obs
        if torch.any(terminated | truncated):
            obs, _ = env.reset()
            obs = obs["policy"] if isinstance(obs, dict) else obs
    env.close()


if __name__ == "__main__":
    main()
    sim_app.close()
