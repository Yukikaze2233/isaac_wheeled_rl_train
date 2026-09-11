"""Isaac Lab 2.3.0 / RSL-RL 3.0.1 stock feed-forward PPO baseline."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class V40PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    # Fudan's 48 steps at100Hz and SCUT's 24 at50Hz both span0.48s.
    num_steps_per_env = 48
    # SCUT V14 reference ceiling; actual runs remain time-budget bounded.
    max_iterations = 20000
    save_interval = 100
    experiment_name = "v40_serial"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = None  # The controller clips; policy likelihood uses raw samples.
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        # Wheel scale50*.2=10rad/s initial std; do not copy SCUT scale10*std1 blindly.
        init_noise_std=0.2,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        entropy_coef=0.005,
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
