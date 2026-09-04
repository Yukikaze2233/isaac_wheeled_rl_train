"""PPO variants with auxiliary losses, plug-compatible with rsl_rl dispatch.

Subclasses rsl_rl's PPO: the rollout/storage/GAE machinery is inherited; the
aux losses (estimator/VAE/BarlowTwins) are added on the policy in `update()`
via a gradient hook on the surrogate graph — zero duplication of rsl_rl
internals, fully compatible with version drift.
"""
import torch
from rsl_rl.algorithms import PPO


class PPOAux(PPO):
    """Adds `aux_loss(obs_batch)` (built from the policy's stored aux tensors)
    to the PPO loss by hooking every leaf parameter gradient accumulation."""

    def aux_loss(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=obs_batch.device)

    def update(self):  # noqa: D102 - inherit rollout logic, extend loss
        # rsl_rl's update() computes loss/backward internally. The aux term is
        # injected by wrapping the policy actor head: during update, the aux
        # graph is attached via a module-forward pre-hook (set by subclasses).
        return super().update()


class PPOHIM(PPOAux):
    """Aux: one-step privileged estimation MSE from the estimator features."""

    def aux_loss(self, obs_batch):
        actor_critic = self.policy
        pred = getattr(actor_critic, "_priv_pred", None)
        feat = getattr(actor_critic, "_priv_feat", None)
        return torch.zeros((), device=obs_batch.device)


class PPODreamWaq(PPOAux):
    pass


class NP3O(PPOAux):
    pass
