"""ActorCritic variants for rsl_rl runner dispatch (frame-stacked history)."""
from .actor_critic_ext import ActorCriticBarlowTwins, ActorCriticDreamWaq, ActorCriticHIM

__all__ = ["ActorCriticHIM", "ActorCriticDreamWaq", "ActorCriticBarlowTwins"]
