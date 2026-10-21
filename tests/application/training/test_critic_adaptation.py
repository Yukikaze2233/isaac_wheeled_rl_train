"""Changed-input critic initialization, frozen policy and exact restart state."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from wheeled_algo.recovery_rollout import RecoveryRolloutMask
from wheeled_tasks.chassis.application.training.critic_adaptation import (
    KEY, remaining_updates, selection, validate_pair, verify_frozen_actor,
)
from wheeled_tasks.chassis.application.evaluation.flat_evaluation_runtime import _stock_actor
from wheeled_tasks.chassis.application.training.policy_transfer import formal_stage_load_kwargs, verify_learning_state_restore
from wheeled_tasks.chassis.application.configuration.training_profile import effective_profile_algorithm, validate_profile_selection


ROOT = Path(__file__).resolve().parents[3]


def config():
    return json.loads((ROOT / 'configs/v7_formal_20261012/motion_retention_review.json').read_text())


def parent_infos():
    recipe = config()[KEY]
    return {'stage': 'foundation', 'contract_sha256': recipe['source_contract_sha256'],
            'successful_updates_total': 14841, 'critic_warmup_remaining': 0}


def algorithm():
    observations = TensorDict({'policy': torch.zeros(8, 35), 'critic': torch.zeros(8, 78)}, batch_size=[8])
    actor = _stock_actor(None)
    critic = MLPModel(observations, {'critic': ['critic']}, 'critic', 1,
        hidden_dims=[256, 128, 64], activation='elu', obs_normalization=False)
    cfg = effective_profile_algorithm(config())
    cfg.pop('class_name')
    return PPO(actor, critic, RolloutStorage('rl', 8, 24, observations, [6], 'cpu'), device='cpu', **cfg)


def test_source_completed_warmup_does_not_skip_new_coordinate_adaptation():
    cfg = config()
    assert remaining_updates(cfg, parent_infos()) == cfg[KEY]['critic_only_updates_before_review']
    kwargs = formal_stage_load_kwargs(cfg, parent_infos(), stage='terrain')
    assert kwargs['load_cfg'] == {'actor': True, 'critic': False, 'optimizer': False,
                                  'iteration': False, 'rnd': False}
    spec = importlib.util.spec_from_file_location('critic_resume_entry', ROOT / 'src/wheeled_tasks/chassis/application/training/train.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert entry.resume_critic_warmup(cfg, parent_infos()) == cfg[KEY]['critic_only_updates_before_review']


def test_same_course_restarts_keep_remaining_counter_and_optimizer():
    cfg = config()
    for remaining in [0, 7, cfg[KEY]['critic_only_updates_before_review']]:
        infos = {'stage': 'terrain', KEY: cfg[KEY], 'critic_warmup_remaining': remaining}
        assert remaining_updates(cfg, infos) == remaining
        assert formal_stage_load_kwargs(cfg, infos, stage='terrain') == {}
    for invalid in [None, True, -1, 51]:
        with pytest.raises(ValueError):
            remaining_updates(cfg, {'stage': 'terrain', KEY: cfg[KEY], 'critic_warmup_remaining': invalid})
    with pytest.raises(ValueError, match='saved configuration'):
        remaining_updates(cfg, {'stage': 'terrain', 'critic_warmup_remaining': 0})


@pytest.mark.parametrize('field,value', [('initialization', 'inherit'),
    ('critic_only_updates_before_review', 0), ('critic_only_updates_before_review', True),
    ('after_window', 'automatic_actor_release')])
def test_unsafe_or_ambiguous_adaptation_selection_is_rejected(field, value):
    cfg = config()
    cfg[KEY][field] = value
    with pytest.raises(ValueError):
        validate_profile_selection(cfg)


def test_actual_native_loader_keeps_actor_std_and_verifies_fresh_critic():
    source, target = algorithm(), algorithm()
    with torch.no_grad():
        for parameter in source.critic.parameters():
            parameter.add_(.2)
        source.actor.distribution.std_param.fill_(.071)
    checkpoint = deepcopy(source.save())
    checkpoint['infos'] = parent_infos()
    fresh = deepcopy(target.critic.state_dict())
    target.load(checkpoint, **formal_stage_load_kwargs(config(), parent_infos(), stage='terrain'))
    receipt = verify_learning_state_restore(target, checkpoint, fresh_optimizer_learning_rate=1e-4,
        fresh_critic_state=fresh)
    assert receipt['actor_exact'] and receipt['critic_exact'] is False
    assert target.actor.distribution.std_param.eq(.071).all()
    assert all(torch.equal(value, fresh[name]) for name, value in target.critic.state_dict().items())
    with pytest.raises(ValueError, match='critic_state_dict'):
        verify_learning_state_restore(target, checkpoint, fresh_optimizer_learning_rate=1e-4)


def test_real_ppo_critic_only_step_keeps_actor_std_and_actor_adam_empty():
    alg = algorithm()
    alg.actor.requires_grad_(False)
    alg.schedule = 'fixed'
    actor_before, critic_before = deepcopy(alg.actor.state_dict()), deepcopy(alg.critic.state_dict())
    masking = RecoveryRolloutMask(alg, num_envs=8)
    observations = TensorDict({'policy': torch.randn(8, 35), 'critic': torch.randn(8, 78)}, batch_size=[8])
    learning = torch.tensor([False, False, True, True, True, True, True, True])
    with torch.inference_mode():
        for _ in range(24):
            alg.act(observations)
            observations = TensorDict({'policy': torch.randn(8, 35), 'critic': torch.randn(8, 78)}, batch_size=[8])
            alg.process_env_step(observations, torch.ones(8), torch.zeros(8, dtype=torch.bool),
                {'policy_learning_mask': learning, 'policy_bootstrap_mask': learning})
        alg.compute_returns(observations)
    alg.update()
    assert verify_frozen_actor(alg, actor_before)['actor_optimizer_state_entries'] == 0
    assert any(not torch.equal(value, critic_before[name]) for name, value in alg.critic.state_dict().items())
    assert alg.learning_rate == 1e-4 and alg.schedule == 'fixed'
    assert masking.metrics['valid_transitions'] == 24 * 6
    checkpoint = deepcopy(alg.save())
    checkpoint['infos'] = {'stage': 'terrain', KEY: config()[KEY], 'critic_warmup_remaining': 49}
    target = algorithm()
    assert formal_stage_load_kwargs(config(), checkpoint['infos'], stage='terrain') == {}
    target.load(checkpoint, load_cfg=None, strict=True)
    assert verify_learning_state_restore(target, checkpoint)['optimizer_exact']
    assert remaining_updates(config(), checkpoint['infos']) == 49


def test_pair_check_rejects_a_different_source_coordinate_convention():
    source = json.loads((ROOT / 'configs/v7_formal_20261012/recovery_after10000_056.json').read_text())
    assert validate_pair(source, config())['warmup_window_is_not_qualification']
    source['critic_height_reference'] = 'root_link_world_z'
    with pytest.raises(ValueError):
        validate_pair(source, config())


def test_more_critic_adaptation_keeps_prior_work_and_does_not_restart_optimizer():
    from wheeled_tasks.chassis.application.configuration.small_rough_course import validate_resume
    old = config()
    extended = deepcopy(old)
    extended[KEY]['critic_only_updates_before_review'] = 100
    infos = {'stage': 'terrain', KEY: old[KEY], 'stage_actor_update': 0, 'critic_warmup_remaining': 0}
    assert remaining_updates(extended, infos) == 50
    assert formal_stage_load_kwargs(extended, infos, stage='terrain') == {}
    assert validate_resume(old, extended)['optimizer'] == 'inherited'
    infos['critic_warmup_remaining'] = 13
    assert remaining_updates(extended, infos) == 63
    infos['stage_actor_update'] = 1
    with pytest.raises(ValueError, match='already completed actor updates'):
        remaining_updates(extended, infos)
    with pytest.raises(ValueError, match='only increase'):
        validate_resume(extended, old)
