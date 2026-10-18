# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)





@configclass
class Wheelbipe25V3FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "wheelbipe25_v3_flat_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
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
    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.01,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )
    
    

@configclass
class DreamWaqPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_class = "OnPolicyDreamWaqRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "dreamwaq_flat_direct"

    policy = dict(
        class_name="ActorCriticDreamWaq",
        init_noise_std = 1.0,
        activation = 'elu',
        # AdaBoot configuration (for DreamWaq paper reproduction)
        # available: "off" | "reward_cv" | "uncertainty" | "hybrid"
        # paper-style: p_boot = 1 - tanh(CV(R))
        adaboot_mode = "off",
        # clamp of alpha / p_boot (safety)
        adaboot_min = 0.0,
        adaboot_max = 1.0,
        # only used in uncertainty / hybrid mode
        adaboot_temperature = 1.0,
        adaboot_bias = 0.0,
        cenet_encoder_hidden_dims=[256, 128, 64],
        cenet_decoder_hidden_dims=[64, 128, 256],
    )
    algorithm = dict(
        class_name="PPODreamWaq",
        value_loss_coef = 4.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.005,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4, #5.e-4
        vae_learning_rate = 1.e-3,
        num_adaptation_module_substeps = 1,
        kl_weight = 1.0,
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.,
        # AdaBoot reward-window statistics
        adaboot_reward_window_size = 1024,
        # p_boot = 1 - tanh(adaboot_reward_cv_scale * CV + adaboot_reward_cv_offset)
        # smaller scale -> larger p_boot
        adaboot_reward_cv_scale = 0.25,
        adaboot_reward_cv_offset = 0.,
        adaboot_pboot_min = 0.0,
        adaboot_pboot_max = 1.0,
    )



@configclass
class HIMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_class = "OnPolicyHIMRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "him_flat_direct"

    policy = dict(
        class_name="ActorCriticHIM",
        init_noise_std = 1.0,
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        activation = 'elu',
    )
    algorithm = dict(
        class_name="PPOHIM",
        value_loss_coef = 4.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.005,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4, #5.e-4
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.,
    )

@configclass
class NP3OPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_class = "OnConstraintPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "np3o_flat_direct"

    policy = dict(
        class_name="ActorCriticBarlowTwins",
        init_noise_std = 1.0,
        continue_from_last_std = True,
        scan_encoder_dims = [128, 64, 32],
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        #priv_encoder_dims = [64, 20],
        priv_encoder_dims = [],
        activation = 'elu', # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm',
        # rnn_hidden_size = 512,
        # rnn_num_layers = 1,

        tanh_encoder_output = False,
        num_costs = 1,

        teacher_act = True,
        imi_flag = True,
        hist_encoder = False
    )
    algorithm = dict(
        class_name="NP3O",
        value_loss_coef = 4.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.005,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4, #5.e-4
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.,
        cost_value_loss_coef = 0.1,
        cost_viol_loss_coef = 0.1,
        dagger_update_freq = 20
    )

    

    
    
@configclass
class WheelbipeV13FlatPPORunnerCfg(Wheelbipe25V3FlatPPORunnerCfg):
    experiment_name = "wheelbipe_v13_flat_direct"

    # RND 需要 rnd_state 映射；rsl_rl 的 resolve_rnd_config 会据此拼 RND 输入维并在内部将 weight *= step_dt
    # obs_groups = {
    #     "policy": ["policy"],
    #     "critic": ["critic"],
    #     "rnd_state": ["policy"],
    # }

    algorithm = RslRlPpoAlgorithmCfg(
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
        # rnd_cfg=RslRlRndCfg(
        #     weight=1.0,
        #     weight_schedule=None,
        #     reward_normalization=False,
        #     state_normalization=False,
        #     learning_rate=1.0e-4,
        #     num_outputs=1,
        #     predictor_hidden_dims=[-1],
        #     target_hidden_dims=[-1],
        # ),
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True,
        #     use_mirror_loss=False,
        #     data_augmentation_func=compute_symmetric_states,
        #     mirror_loss_coeff=0.0,
        # ),
    )






@configclass
class WheelbipeV13RoughPPORunnerCfg(Wheelbipe25V3FlatPPORunnerCfg):
    experiment_name = "wheelbipe_v13_rough_direct"

@configclass
class WheelbipeV14FlatPPORunnerCfg(WheelbipeV13FlatPPORunnerCfg):
    experiment_name = "wheelbipe_v14_2_flat_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_hidden_dims=[512, 256, 128],
        # critic_hidden_dims=[512, 256, 128],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        # actor_hidden_dims=[128, 64, 32],
        # critic_hidden_dims=[128, 64, 32],
        activation="elu",
    )

@configclass
class WheelbipeV14FlatDreamWaqPPORunnerCfg(DreamWaqPPORunnerCfg):
    experiment_name = "wheelbipe_v14_flat_dreamwaq_direct"
    max_iterations = 10000



@configclass
class WheelbipeV14FlatHIMPPORunnerCfg(HIMPPORunnerCfg):
    experiment_name = "wheelbipe_v14_flat_him_direct"
    max_iterations = 5000

@configclass
class WheelbipeV14FlatNP3OBarlowPPORunnerCfg(NP3OPPORunnerCfg):
    experiment_name = "wheelbipe_v14_flat_np3o_barlow_direct"
    max_iterations = 3000
    policy = dict(
        class_name="ActorCriticBarlowTwins",
        init_noise_std=1.0,
        continue_from_last_std=True,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[],
        activation="elu",
        tanh_encoder_output=False,
        num_costs=5,
        teacher_act=True,
        imi_flag=True,
        hist_encoder=False,
    )
    algorithm = dict(
        class_name="NP3O",
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
        cost_value_loss_coef=0.1,
        cost_viol_loss_coef=0.1,
        dagger_update_freq=20,
    )

@configclass
class WheelbipeV14RoughPPORunnerCfg(WheelbipeV13RoughPPORunnerCfg):
    experiment_name = "wheelbipe_v14_2_rough_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_hidden_dims=[512, 256, 128],
        # critic_hidden_dims=[512, 256, 128],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        # actor_hidden_dims=[128, 64, 32],
        # critic_hidden_dims=[128, 64, 32],
        activation="elu",
    )
    # algorithm = dict(
    #     class_name="PPOTaction",
    #     value_loss_coef=4.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.005,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-4,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )
