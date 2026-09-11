"""Agent configs: rsl_rl runner configurations .

PPO baseline follows the the flat task values (num_steps 24, adaptive-KL lr,
[256,128,64] MLP). New algorithm branches (DreamWaQ / HIM / NP3O) plug in here
the same way: a new RunnerCfg subclass + runner_class string.
"""
from isaaclab.utils import configclass as _configclass_module
if callable(_configclass_module):  # Isaac Lab <= 2.3
    configclass = _configclass_module
else:  # Isaac Lab >= 3.0: configclass 子包化，装饰器在子模块内
    from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class WheeledBipedFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 20000
    save_interval: int = 500
    experiment_name: str = "wheeled_biped_flat"
    empirical_normalization: bool = False
    # asymmetric actor-critic: actor sees "policy", critic also sees "critic"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
