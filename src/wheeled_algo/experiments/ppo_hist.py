"""Frame-stack branch (frame-stack baseline branch): plain PPO over flattened history.

Completes the controlled-variable set against the estimator branches — same
history input, no estimator/VAE/twins machinery. The estimator branches are
supposed to beat this if their implicit estimation adds value.
"""
import torch
import torch.nn as nn

from wheeled_algo.algorithms.ppo_base import ExtPPOLoop, ExtTrainCfg, HistoryRoller


class PPOHistActorCritic(HistoryRoller, nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 hidden: tuple[int, ...] = (128, 64), latent_dim: int = 0, init_noise_std: float = 1.0):
        super().__init__()
        self.hist_len = hist_len
        self.obs_dim = obs_dim
        self.hist_dim = obs_dim * hist_len
        self.latent_dim = latent_dim  # kept for API parity; unused

        def mlp(inp: int, dims: tuple[int, ...], out: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = inp
            for d in dims:
                layers += [nn.Linear(prev, d), nn.ELU()]
                prev = d
            layers.append(nn.Linear(prev, out))
            return nn.Sequential(*layers)

        self.actor = mlp(self.hist_dim, hidden, action_dim)
        self.critic = mlp(self.hist_dim, hidden, 1)  # symmetric: history as critic input
        self.log_std = nn.Parameter(torch.log(torch.full((action_dim,), init_noise_std)))
        self._init_roller(obs_dim)

    def _push_hist(self, obs: torch.Tensor) -> torch.Tensor:
        return self.roll(obs)

    def act(self, obs: torch.Tensor, streams: dict | None = None):
        hist = self._push_hist(obs)
        mean = self.actor(hist)
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy()

    def update_distribution(self, obs: torch.Tensor, hist_flat: torch.Tensor | None = None):
        x = hist_flat if hist_flat is not None else obs.repeat(1, self.hist_len)
        mean = self.actor(x)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def evaluate(self, priv_obs: torch.Tensor, hist_flat: torch.Tensor | None = None) -> torch.Tensor:
        # critic eats the same flattened POLICY history as the actor; when no
        # history batch is supplied, rebuild it from the rolling buffer or the
        # policy slice of the given observation (priv streams are wider).
        if hist_flat is not None:
            x = hist_flat
        elif self._hist is not None and self._hist.shape[0] == priv_obs.shape[0]:
            x = self._hist
        else:
            x = priv_obs[:, :self.obs_dim].repeat(1, self.hist_len)
        return self.critic(x).squeeze(-1)

    def extra_loss(self, obs, priv, hist, priv_hist) -> torch.Tensor:
        return obs.new_zeros(())  # no auxiliary losses on this branch


class PPOHistTrainer:
    def __init__(self, env, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 device: str = "cpu", cfg: ExtTrainCfg | None = None):
        self.policy = PPOHistActorCritic(obs_dim, priv_dim, hist_len, action_dim)
        self.loop = ExtPPOLoop(env, self.policy, cfg or ExtTrainCfg(), device)

    def learn(self, iterations: int, episode_tracker=None):
        self.loop.learn(iterations, episode_tracker=episode_tracker)
