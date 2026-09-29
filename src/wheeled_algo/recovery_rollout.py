"""Exclude scripted recovery intervals while retaining the native PPO update.

The environment reports whether the *whole elapsed interval* used the actor,
and whether the returned observation is eligible for RL bootstrapping. These
are separate at a script-to-RL handoff or an autoreset into scripted recovery.
Only GAE segmentation and the storage presented to native PPO are adapted;
sampled actions, likelihoods, optimizer, losses, and adaptive LR are untouched.
"""

from copy import copy
from importlib.metadata import version

import torch


class RecoveryRolloutMask:
    """An adapter for the project's pinned, feed-forward RSL-RL PPO storage."""

    def __init__(self, algorithm, *, num_envs: int):
        from rsl_rl.algorithms import PPO
        from rsl_rl.storage import RolloutStorage

        if version("rsl-rl-lib") != "5.5.1" or type(algorithm) is not PPO:
            raise TypeError("Recovery masking requires native RSL-RL 5.5.1 PPO")
        storage = algorithm.storage
        if type(storage) is not RolloutStorage or storage.training_type != "rl":
            raise TypeError("Recovery masking requires native RL RolloutStorage")
        if (algorithm.actor.is_recurrent or algorithm.critic.is_recurrent
                or algorithm.rnd or algorithm.symmetry or algorithm.is_multi_gpu):
            raise ValueError("Recovery masking supports single-device feed-forward PPO only")
        if num_envs != storage.num_envs or num_envs < 1 or storage.step != 0:
            raise ValueError("Attach recovery masking to empty storage with fixed num_envs")
        if algorithm.num_mini_batches < 1 or algorithm.num_learning_epochs < 1:
            raise ValueError("PPO must have positive mini-batch and epoch counts")
        if hasattr(algorithm, "_recovery_rollout_mask"):
            raise ValueError("Recovery masking is already attached")

        self.algorithm, self.storage = algorithm, storage
        self.shape = (storage.num_transitions_per_env, num_envs)
        self._validate_storage()
        self.learning = torch.zeros(self.shape, dtype=torch.bool, device=storage.device)
        self.bootstrap = torch.zeros_like(self.learning)
        self.metrics: dict[str, int | float] = {}
        self._returns_ready = False
        self._process = algorithm.process_env_step
        self._returns = algorithm.compute_returns
        self._update = algorithm.update
        algorithm.process_env_step = self.process_env_step
        algorithm.compute_returns = self.compute_returns
        algorithm.update = self.update
        algorithm._recovery_rollout_mask = self

    def _validate_storage(self):
        st = self.storage
        if self.algorithm.storage is not st or (st.num_transitions_per_env, st.num_envs) != self.shape:
            raise ValueError("Recovery rollout storage identity or fixed dimensions changed")
        if tuple(st.observations.batch_size) != self.shape or tuple(st.actions.shape[:2]) != self.shape:
            raise ValueError("Unsupported recovery rollout observation/action shape")
        for name in ("rewards", "values", "dones", "returns", "advantages", "actions_log_prob"):
            if tuple(getattr(st, name).shape) != (*self.shape, 1):
                raise ValueError(f"Unsupported recovery rollout {name} shape")

    def _mask(self, extras, key):
        mask = extras.get(key)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"Recovery rollout requires bool tensor infos['{key}']")
        if tuple(mask.shape) != (self.shape[1],) or mask.device != self.storage.rewards.device:
            raise ValueError(f"Recovery rollout {key} must have fixed shape [num_envs] on storage device")
        return mask

    def process_env_step(self, obs, rewards, dones, extras):
        self._validate_storage()
        # Validate before native PPO mutates the transition or storage cursor.
        learning = self._mask(extras, "policy_learning_mask")
        bootstrap = self._mask(extras, "policy_bootstrap_mask")
        index = self.storage.step
        if index >= self.shape[0]:
            raise OverflowError("Recovery rollout buffer is full")
        self.learning[index].copy_(learning)
        self.bootstrap[index].copy_(bootstrap)
        self._returns_ready = False
        self._process(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        self._validate_storage()
        st, alg = self.storage, self.algorithm
        if st.step != self.shape[0]:
            raise ValueError("Recovery GAE requires a complete rollout")
        valid_count = int(self.learning.sum().item())
        minimum = 2 * alg.num_mini_batches
        if valid_count < minimum:
            raise ValueError(
                f"Insufficient pure-RL transitions: {valid_count}; need at least {minimum} "
                "for native PPO mini-batches (script-only rollouts cannot update)")
        total = self.learning.numel()
        self.metrics = {
            "valid_transitions": valid_count,
            "script_or_blend_transitions": total - valid_count,
            "valid_fraction": valid_count / total,
            "minimum_mini_batch_population": valid_count // alg.num_mini_batches,
        }
        if bool(self.learning.all()) and bool(self.bootstrap.all()):
            self._returns(obs)
            self._returns_ready = True
            return

        critic_state = alg.critic.get_hidden_state()
        last_values = alg.critic(obs).detach()
        alg.critic.reset(hidden_state=critic_state)
        valid = self.learning.unsqueeze(-1)
        # torch.where, rather than multiplication by zero, isolates even NaN/Inf
        # from excluded observations, rewards, or value predictions.
        values = torch.where(valid, st.values, 0.0)
        rewards = torch.where(valid, st.rewards, 0.0)
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.shape[0])):
            next_valid = self.bootstrap[step].unsqueeze(-1)
            if step + 1 < self.shape[0]:
                next_valid = next_valid & valid[step + 1]
                next_values = values[step + 1]
            else:
                next_values = last_values
            continues = valid[step] & next_valid & ~st.dones[step].bool()
            future = torch.where(continues, alg.gamma * next_values + alg.gamma * alg.lam * advantage, 0.0)
            advantage = torch.where(valid[step], rewards[step] - values[step] + future, 0.0)
            st.returns[step].copy_(advantage + values[step])
        advantages = st.returns - values
        if not alg.normalize_advantage_per_mini_batch:
            population = advantages[valid]
            advantages = torch.where(valid, (advantages - population.mean()) / (population.std() + 1e-8), 0.0)
        st.advantages = advantages
        self._returns_ready = True

    def update(self):
        self._validate_storage()
        st, alg = self.storage, self.algorithm
        if not self._returns_ready or st.step != self.shape[0]:
            raise ValueError("Recovery returns must be computed before native PPO update")
        self._returns_ready = False
        if bool(self.learning.all()):
            return self._update()

        # The native storage generator and PPO losses operate on an ordinary
        # one-step storage of the valid population. Native normalizer updates
        # consequently see exactly the same valid observations as the losses.
        selected = self.learning.flatten().nonzero(as_tuple=True)[0]
        compact = copy(st)
        compact.num_transitions_per_env, compact.num_envs, compact.step = 1, selected.numel(), 1
        compact.observations = st.observations.flatten(0, 1)[selected].unsqueeze(0)
        for name in ("rewards", "actions", "dones", "values", "actions_log_prob", "returns", "advantages"):
            setattr(compact, name, getattr(st, name).flatten(0, 1)[selected].unsqueeze(0))
        compact.distribution_params = tuple(p.flatten(0, 1)[selected].unsqueeze(0) for p in st.distribution_params)
        alg.storage = compact
        try:
            result = self._update()
        finally:
            alg.storage = st
        st.clear()
        return result


def attach_recovery_rollout_mask(algorithm, *, num_envs: int) -> RecoveryRolloutMask:
    """Attach before training wrappers; both boolean masks become mandatory."""
    return RecoveryRolloutMask(algorithm, num_envs=num_envs)
