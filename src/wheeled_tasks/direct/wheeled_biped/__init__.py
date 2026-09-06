from .env_cfg import (
    WheeledBipedFlatEnvCfg,
    WheeledBipedRoughEnvCfg,
    WheeledBipedV33FlatEnvCfg,
)

__all__ = ["WheeledBipedFlatEnvCfg", "WheeledBipedRoughEnvCfg", "WheeledBipedV33FlatEnvCfg"]

import gymnasium as gym  # noqa: E402

gym.register(
    id="WheeledBiped-V33-Flat-v0",
    entry_point="wheeled_tasks.direct.wheeled_biped.env:WheeledBipedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "wheeled_tasks.direct.wheeled_biped.env_cfg:WheeledBipedV33FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "wheeled_tasks.agents.rsl_rl_ppo_cfg:WheeledBipedFlatPPORunnerCfg",
    },
)
