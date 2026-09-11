"""Explicit registration; importing wheeled_tasks alone never selects a legacy task."""
import gymnasium as gym

TASK_ID = "Own-V40-Serial-Direct-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="wheeled_tasks.direct.v40_serial.env:V40Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "wheeled_tasks.direct.v40_serial.env_cfg:V40EnvCfg",
            "rsl_rl_cfg_entry_point": "wheeled_tasks.agents.v40_ppo_cfg:V40PPORunnerCfg",
        },
    )
