"""Exercise the installed native PPO, including actual Adam and normalizers."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask


def observation(count, offset=0.0):
    return TensorDict({"actor": torch.arange(count * 3).reshape(count, 3).float() / 20 + offset,
                       "critic": torch.arange(count * 4).reshape(count, 4).float() / 30 + offset},
                      batch_size=[count])


def algorithm(*, steps=6, count=2, per_batch=False, normalize_obs=False):
    torch.manual_seed(321)
    obs = observation(count)
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = MLPModel(obs, groups, "actor", 2, hidden_dims=[8], obs_normalization=normalize_obs,
                     distribution_cfg={"class_name": GaussianDistribution, "init_std": 0.5})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[8], obs_normalization=normalize_obs)
    storage = RolloutStorage("rl", count, steps, obs, [2])
    return PPO(actor, critic, storage, num_learning_epochs=2, num_mini_batches=2,
               gamma=.9, lam=1., normalize_advantage_per_mini_batch=per_batch)


def collect(alg, learning, bootstrap=None, dones=None, state_value=None):
    torch.manual_seed(190)
    steps, count = learning.shape
    if bootstrap is None:
        bootstrap = torch.ones_like(learning)
    if dones is None:
        dones = torch.zeros_like(learning)
    for step in range(steps):
        obs = observation(count, step / 10)
        with torch.no_grad():
            alg.act(obs)
        rewards = torch.arange(count).float() + step + 1
        extras = {"policy_learning_mask": learning[step], "policy_bootstrap_mask": bootstrap[step]}
        if state_value is not None:
            extras["policy_state_value_mask"] = state_value[step]
        alg.process_env_step(observation(count, (step + 1) / 10), rewards, dones[step], extras)
    return observation(count, steps / 10)


@pytest.mark.parametrize("key", ["policy_learning_mask", "policy_bootstrap_mask"])
@pytest.mark.parametrize("invalid", [None, torch.ones(2), torch.ones(2, 1, dtype=torch.bool)])
def test_enabled_masks_are_mandatory_bool_per_environment(key, invalid):
    alg = algorithm()
    attach_recovery_rollout_mask(alg, num_envs=2)
    extras = {"policy_learning_mask": torch.ones(2, dtype=torch.bool),
              "policy_bootstrap_mask": torch.ones(2, dtype=torch.bool)}
    extras[key] = invalid
    with pytest.raises(ValueError, match=key):
        alg.process_env_step(observation(2), torch.ones(2), torch.zeros(2), extras)
    assert alg.storage.step == 0


def test_segmented_gae_respects_both_handoffs_autoreset_and_rollout_end():
    alg = algorithm(per_batch=True)
    adapter = attach_recovery_rollout_mask(alg, num_envs=2)
    mask = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1]], dtype=torch.bool)
    bootstrap = torch.ones_like(mask)
    bootstrap[1, 0] = False
    bootstrap[-1, 1] = False  # final observation is scripted: must not bootstrap
    done = torch.zeros_like(mask)
    done[4, 0] = True  # physical reset also splits two adjacent valid intervals
    obs = collect(alg, mask, bootstrap, done)
    last = alg.critic(obs).detach().squeeze(-1)
    alg.compute_returns(obs)
    # Rewards are [1..6] / [2..7]. These are explicit, independent segment sums.
    expected = torch.tensor([[1 + .9 * 2, 2], [2, 0], [0, 0],
                             [0, 5 + .9 * 6 + .81 * 7], [5, 6 + .9 * 7], [6, 7]])
    expected[5, 0] += .9 * last[0]
    torch.testing.assert_close(alg.storage.returns.squeeze(-1), expected)
    assert torch.equal(alg.storage.advantages.squeeze(-1)[~mask], torch.zeros(4))
    assert adapter.metrics == {"valid_transitions": 8, "script_or_blend_transitions": 4,
                               "valid_fraction": 2 / 3, "minimum_mini_batch_population": 4}


@pytest.mark.parametrize("per_batch", [False, True])
def test_poison_script_data_cannot_change_advantages_losses_adam_lr_or_normalizers(per_batch):
    baseline = algorithm(per_batch=per_batch, normalize_obs=True)
    poisoned = algorithm(per_batch=per_batch, normalize_obs=True)
    mask = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1]], dtype=torch.bool)
    for alg in (baseline, poisoned):
        attach_recovery_rollout_mask(alg, num_envs=2)
        obs = collect(alg, mask)
    invalid = ~mask
    st = poisoned.storage
    for name in ("rewards", "values", "actions", "actions_log_prob"):
        getattr(st, name)[invalid] = torch.nan
    for tensor in st.observations.values():
        tensor[invalid] = torch.nan
    for tensor in st.distribution_params:
        tensor[invalid] = torch.nan
    for alg in (baseline, poisoned):
        alg.compute_returns(obs)
    torch.testing.assert_close(baseline.storage.advantages, poisoned.storage.advantages)
    original_storages = [alg.storage for alg in (baseline, poisoned)]
    outputs = []
    for alg in (baseline, poisoned):
        torch.manual_seed(442)
        outputs.append(alg.update())
    assert outputs[0] == outputs[1]
    assert baseline.learning_rate == poisoned.learning_rate
    for key, value in baseline.actor.state_dict().items():
        assert torch.equal(value, poisoned.actor.state_dict()[key]), key
    for key, value in baseline.critic.state_dict().items():
        assert torch.equal(value, poisoned.critic.state_dict()[key]), key
    for key, state in baseline.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, poisoned.optimizer.state_dict()["state"][key][name])
    for alg, original in zip((baseline, poisoned), original_storages):
        assert alg.storage is original and original.step == 0
        # Native normalizer has seen only eight eligible observations.
        assert alg.actor.obs_normalizer.count == 8
        assert alg.critic.obs_normalizer.count == 8


def test_all_valid_adapter_is_bitwise_identical_to_unmodified_native_ppo():
    native, masked = algorithm(normalize_obs=True), algorithm(normalize_obs=True)
    attach_recovery_rollout_mask(masked, num_envs=2)
    mask = torch.ones(6, 2, dtype=torch.bool)
    for alg in (native, masked):
        obs = collect(alg, mask)
        alg.compute_returns(obs)
    assert torch.equal(native.storage.advantages, masked.storage.advantages)
    for alg in (native, masked):
        torch.manual_seed(87)
        output = alg.update()
        if alg is native:
            expected = output
    assert output == expected
    assert native.learning_rate == masked.learning_rate
    for key, value in native.actor.state_dict().items():
        assert torch.equal(value, masked.actor.state_dict()[key]), key
    for key, value in native.critic.state_dict().items():
        assert torch.equal(value, masked.critic.state_dict()[key]), key


def test_timeout_bootstraps_pre_reset_state_once_without_leaking_reset_rewards():
    alg = algorithm(steps=3)
    attach_recovery_rollout_mask(alg, num_envs=2, require_terminal_observation=True)
    obs = observation(2)
    terminal = observation(2, 10.)
    with torch.no_grad():
        alg.act(obs)
        starting_value = alg.transition.values[0, 0].item()
        final_value = alg.critic(terminal)[0, 0].item()
    assert final_value != pytest.approx(starting_value)
    raw_rewards = torch.tensor([2., 3.])
    timeout = torch.tensor([True, False])
    info = {'policy_learning_mask': torch.ones(2, dtype=torch.bool),
        'policy_bootstrap_mask': ~timeout, 'time_outs': timeout.clone(),
        'policy_terminal_critic_observation': terminal['critic'].clone()}
    # The unrelated lane has no terminal-state observation and must be isolated.
    info['policy_terminal_critic_observation'][1] = torch.nan
    alg.process_env_step(observation(2, -100.), raw_rewards, timeout, info)
    expected = 2. + alg.gamma * final_value
    assert alg.storage.rewards[0, 0, 0].item() == pytest.approx(expected)
    assert torch.equal(raw_rewards, torch.tensor([2., 3.]))
    assert torch.equal(info['time_outs'], timeout)
    for step in (1, 2):
        with torch.no_grad():
            alg.act(observation(2, step / 10))
        valid = torch.tensor([step == 2, True])
        alg.process_env_step(observation(2, (step + 1) / 10), torch.full((2,), 1000.),
            torch.zeros(2, dtype=torch.bool), {'policy_learning_mask': valid,
                'policy_bootstrap_mask': valid, 'time_outs': torch.zeros(2, dtype=torch.bool)})
    alg.compute_returns(observation(2, .3))
    assert alg.storage.returns[0, 0, 0].item() == pytest.approx(expected)


@pytest.mark.parametrize('bad', ['missing', 'nonfinite', 'shape', 'not_done'])
def test_required_terminal_observation_rejects_invalid_timeouts_before_storage(bad):
    alg = algorithm()
    attach_recovery_rollout_mask(alg, num_envs=2, require_terminal_observation=True)
    with torch.no_grad():
        alg.act(observation(2))
    info = {'policy_learning_mask': torch.ones(2, dtype=torch.bool),
        'policy_bootstrap_mask': torch.ones(2, dtype=torch.bool),
        'time_outs': torch.tensor([True, False]),
        'policy_terminal_critic_observation': observation(2, 1.)['critic'].clone()}
    done = info['time_outs'].clone()
    if bad == 'missing': info.pop('policy_terminal_critic_observation')
    if bad == 'nonfinite': info['policy_terminal_critic_observation'][0, 0] = torch.nan
    if bad == 'shape': info['policy_terminal_critic_observation'] = torch.zeros(2, 5)
    if bad == 'not_done': done[0] = False
    with pytest.raises(ValueError, match='[Tt]erminal|[Tt]imeout'):
        alg.process_env_step(observation(2, .1), torch.ones(2), done, info)
    assert alg.storage.step == 0


def test_failed_transition_keeps_last_reliable_value_without_future_reward_leak():
    alg = algorithm(per_batch=True)
    attach_recovery_rollout_mask(alg, num_envs=2)
    learning = torch.ones(6, 2, dtype=torch.bool)
    learning[2:4, 0] = False
    reliable = learning.clone()
    reliable[2, 0] = True  # s_2 is reliable; physical transition2 failed.
    bootstrap = torch.ones_like(learning)
    bootstrap[2:4, 0] = False
    obs = collect(alg, learning, bootstrap, state_value=reliable)
    st = alg.storage
    last_good_value = st.values[2, 0, 0].item()
    final_value = alg.critic(obs).detach().squeeze(-1)
    st.rewards[2:4, 0] = float("nan")
    st.values[3, 0] = float("nan")
    alg.compute_returns(obs)
    expected = torch.tensor([
        1 + .9 * (2 + .9 * last_good_value),
        2 + .9 * last_good_value,
        0., 0., 5 + .9 * 6 + .81 * final_value[0],
        6 + .9 * final_value[0],
    ])
    torch.testing.assert_close(st.returns[:, 0, 0], expected)
    assert torch.equal(st.advantages[2:4, 0], torch.zeros(2, 1))
    # The other environment has its original uninterrupted return.
    assert st.returns[0, 1, 0] == pytest.approx(
        sum((.9 ** i) * (i + 2) for i in range(6)) + .9 ** 6 * final_value[1].item())


@pytest.mark.parametrize("per_batch", [False, True])
def test_reliable_value_boundary_excludes_failed_data_from_actual_ppo(per_batch):
    baseline, poisoned = (algorithm(per_batch=per_batch, normalize_obs=True) for _ in range(2))
    learning = torch.ones(6, 2, dtype=torch.bool)
    learning[2:4, 0] = False
    reliable = learning.clone()
    reliable[2, 0] = True
    bootstrap = torch.ones_like(learning)
    bootstrap[2:4, 0] = False
    for alg in (baseline, poisoned):
        attach_recovery_rollout_mask(alg, num_envs=2)
        obs = collect(alg, learning, bootstrap, state_value=reliable)
    st = poisoned.storage
    for name in ("rewards", "actions", "actions_log_prob"):
        getattr(st, name)[~learning] = torch.nan
    st.values[~reliable] = torch.nan  # The reliable pre-failure value is retained.
    for value in (*st.observations.values(), *st.distribution_params):
        value[~learning] = torch.nan
    outputs = []
    for alg in (baseline, poisoned):
        alg.compute_returns(obs)
        torch.manual_seed(442)
        outputs.append(alg.update())
    assert outputs[0] == outputs[1]
    for component in ("actor", "critic"):
        for key, value in getattr(baseline, component).state_dict().items():
            assert torch.equal(value, getattr(poisoned, component).state_dict()[key]), key
        assert getattr(poisoned, component).obs_normalizer.count == 10
    for key, state in baseline.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, poisoned.optimizer.state_dict()["state"][key][name])


def test_cannot_learn_from_explicitly_unreliable_starting_state():
    alg = algorithm()
    attach_recovery_rollout_mask(alg, num_envs=2)
    with pytest.raises(ValueError, match="reliable starting state"):
        alg.process_env_step(observation(2), torch.ones(2), torch.zeros(2), {
            "policy_learning_mask": torch.ones(2, dtype=torch.bool),
            "policy_bootstrap_mask": torch.ones(2, dtype=torch.bool),
            "policy_state_value_mask": torch.zeros(2, dtype=torch.bool)})
    assert alg.storage.step == 0


def test_all_script_rollout_fails_before_optimizer_update():
    alg = algorithm()
    attach_recovery_rollout_mask(alg, num_envs=2)
    before = deepcopy(alg.actor.state_dict())
    obs = collect(alg, torch.zeros(6, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="script-only rollouts cannot update"):
        alg.compute_returns(obs)
    with pytest.raises(ValueError, match="returns must be computed"):
        alg.update()
    assert not alg.optimizer.state
    assert all(torch.equal(value, alg.actor.state_dict()[key]) for key, value in before.items())


def test_ineligible_final_observation_cannot_poison_valid_returns():
    alg = algorithm(per_batch=True)
    attach_recovery_rollout_mask(alg, num_envs=2)
    mask = torch.ones(6, 2, dtype=torch.bool)
    bootstrap = torch.ones_like(mask)
    bootstrap[-1, 0] = False
    obs = collect(alg, mask, bootstrap)
    obs["critic"][0] = torch.nan
    alg.compute_returns(obs)
    # The last elapsed interval was RL, but its returned state starts a script.
    # All preceding valid transitions end at its actual reward, without V(script).
    assert bool(torch.isfinite(alg.storage.returns).all())
    assert alg.storage.returns[-1, 0, 0] == 6
    assert alg.storage.returns[-2, 0, 0] == pytest.approx(5 + .9 * 6)


def test_per_batch_normalization_rejects_singleton_population():
    alg = algorithm(per_batch=True)
    attach_recovery_rollout_mask(alg, num_envs=2)
    mask = torch.zeros(6, 2, dtype=torch.bool)
    mask[-1] = True
    obs = collect(alg, mask)
    with pytest.raises(ValueError, match="need at least 4"):
        alg.compute_returns(obs)


def test_mixed_last_interval_excluded_then_new_rollout_starts_with_real_rl_state():
    alg = algorithm()
    attach_recovery_rollout_mask(alg, num_envs=2)
    mask = torch.ones(6, 2, dtype=torch.bool)
    mask[-1] = False
    obs = collect(alg, mask)  # blend completes: returned state is RL eligible
    alg.compute_returns(obs)
    assert torch.equal(alg.storage.returns[-2].squeeze(-1), torch.tensor([5., 6.]))
    assert not bool(alg.storage.returns[-1].any())
    alg.update()
    obs = collect(alg, torch.ones_like(mask))
    with pytest.raises(ValueError, match="returns must be computed"):
        alg.update()
    alg.compute_returns(obs)
    assert bool(torch.isfinite(alg.storage.advantages).all())
    alg.update()


def test_fixed_dimensions_and_single_attachment_are_enforced():
    alg = algorithm()
    with pytest.raises(ValueError, match="fixed num_envs"):
        attach_recovery_rollout_mask(alg, num_envs=3)
    attach_recovery_rollout_mask(alg, num_envs=2)
    with pytest.raises(ValueError, match="already attached"):
        attach_recovery_rollout_mask(alg, num_envs=2)
    alg.storage.num_envs = 3
    with pytest.raises(ValueError, match="fixed dimensions"):
        alg.compute_returns(observation(2))


def test_training_entry_opt_in_uses_actual_environment_configuration():
    path = Path(__file__).resolve().parents[2] / "src/wheeled_tasks/chassis/application/training/train.py"
    spec = importlib.util.spec_from_file_location("recovery_training_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    alg = algorithm()
    native_update = alg.update
    for config in ({}, {"recovery_training": {"enabled": False}}):
        assert entry.configure_recovery_rollout(alg, SimpleNamespace(cfg=config, num_envs=2)) is None
        assert alg.update == native_update
    adapter = entry.configure_recovery_rollout(alg, SimpleNamespace(
        cfg={"recovery_training": {"enabled": True}}, num_envs=2))
    assert alg.update.__self__ is adapter
    mask = torch.ones(6, 2, dtype=torch.bool)
    mask[:2, 0] = False
    obs = collect(alg, mask)
    alg.compute_returns(obs)
    alg.update()
    assert adapter.metrics["valid_transitions"] == 10
