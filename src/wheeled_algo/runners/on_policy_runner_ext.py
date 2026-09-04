"""Branch-dispatching runner (runner_class string dispatch).

The registry resolves `runner_class = eval(cfg["runner_class"])` into
custom OnPolicyRunner subclasses; this module provides the equivalent for the
algo_ext branches: one runner facade per branch, resolved from the train cfg
by class-name string, sharing the ExtPPOLoop training core and the
save/load/learn API so run_experiment and the server path stay interchangeable.
"""
import torch

from ..algorithms import DreamWaqTrainer, ExtTrainCfg, HIMTrainer, NP3OTrainer
from ..experiments.ppo_hist import PPOHistTrainer

TRAINER_ALIASES = {
    "PPOHistRunner": PPOHistTrainer,
    "HIMRunner": HIMTrainer,
    "DreamWaqRunner": DreamWaqTrainer,
    "NP3ORunner": NP3OTrainer,
}


class ExtOnPolicyRunner:
    """rsl_rl-flavored facade: construct from (env, train_cfg dict), then
    learn/save/load — identical call shape to rsl_rl's OnPolicyRunner."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir

        runner_class = train_cfg.get("runner_class", "PPOHistRunner")
        assert runner_class in TRAINER_ALIASES, \
            f"unknown runner_class {runner_class!r}; options: {sorted(TRAINER_ALIASES)}"

        obs, extras = env.get_observations()
        critic = extras.get("observations", {}).get("critic", obs)
        ext_cfg = ExtTrainCfg(**train_cfg.get("ext_cfg", {}))
        self.trainer = TRAINER_ALIASES[runner_class](
            env, obs_dim=obs.shape[1], priv_dim=critic.shape[1],
            hist_len=int(train_cfg.get("hist_len", 3)),
            action_dim=env.num_actions, device=device, cfg=ext_cfg,
        )

    @property
    def current_iteration(self) -> int:
        return self.trainer.loop.current_iteration

    def learn(self, num_learning_iterations: int, metrics_hook=None) -> None:
        self.trainer.loop.learn(iterations=num_learning_iterations, metrics_hook=metrics_hook)

    def save(self, path: str) -> None:
        self.trainer.loop.save(path)

    def load(self, path: str) -> int:
        return self.trainer.loop.load(path)

    @torch.no_grad()
    def get_inference_policy(self, device: str | None = None):
        """Returns a callable obs -> mean action (play/export path)."""
        policy = self.trainer.policy.to(device or self.device)
        policy.eval()

        def infer(obs: torch.Tensor, streams: dict | None = None) -> torch.Tensor:
            action, _, _ = policy.act(obs, streams)
            return action

        return infer


def resolve_runner_class(name: str):
    """Class-name string dispatch."""
    return TRAINER_ALIASES[name]
