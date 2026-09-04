"""Runner facade per algorithm branch (runner_class dispatch pattern)."""
from .on_policy_runner_ext import ExtOnPolicyRunner, TRAINER_ALIASES, resolve_runner_class

__all__ = ["ExtOnPolicyRunner", "TRAINER_ALIASES", "resolve_runner_class"]
