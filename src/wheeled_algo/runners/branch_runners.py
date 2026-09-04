"""runner_class dispatch over the STOCK rsl_rl OnPolicyRunner.

The runner resolves policy/algorithm class names via eval() in its own module
namespace; `register_branches()` injects our classes there (the standard way
to extend the runner without forking it), so the standard
train.py path runs every branch:

    train_cfg["policy"]["class_name"]    = "ActorCriticHIM"      (etc.)
    train_cfg["algorithm"]["class_name"] = "PPO"                 (aux via PPOAux)
    runner = branch_runners.make_runner(env, train_cfg)
"""
import rsl_rl.runners.on_policy_runner as _opr

from wheeled_algo.modules import ActorCriticBarlowTwins, ActorCriticDreamWaq, ActorCriticHIM

_ALIASES = {
    "ActorCriticHIM": ActorCriticHIM,
    "ActorCriticDreamWaq": ActorCriticDreamWaq,
    "ActorCriticBarlowTwins": ActorCriticBarlowTwins,
    # plain frame-stack / GRU paths use the stock ActorCritic
    "ActorCritic": None,
}

_registered = False


def register_branches() -> None:
    """Inject branch classes into the runner's eval namespace (idempotent)."""
    global _registered
    for name, cls in _ALIASES.items():
        if cls is not None:
            setattr(_opr, name, cls)
    _registered = True


def make_runner(env, train_cfg: dict, log_dir=None, device="cpu"):
    """Construct a stock rsl_rl OnPolicyRunner for any registered branch."""
    register_branches()
    from rsl_rl.runners import OnPolicyRunner
    return OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
