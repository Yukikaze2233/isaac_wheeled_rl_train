"""HIMLoco branch: history-based one-step privileged estimation.

Core mechanism (following the HIMLoco design): an estimator consumes a
short observation history and is trained (plain MSE, detached target) to
predict the privileged information one step ahead; the actor receives the
estimated latent alongside the current observation, so deployment needs only
the history that onboard sensors already provide.
"""
import torch
import torch.nn as nn

from .ppo_base import ExtPPOLoop, ExtTrainCfg, HistoryRoller


class HIMEstimator(nn.Module):
    def __init__(self, hist_dim: int, obs_dim: int, priv_dim: int, latent_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.to_latent = nn.Linear(hidden, latent_dim)
        self.to_priv = nn.Linear(hidden, priv_dim)

    def forward(self, hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(hist)
        return self.to_latent(h), self.to_priv(h)


class HIMActorCritic(HistoryRoller, nn.Module):
    """Actor: current obs + estimated latent; Critic: privileged obs + true latent."""

    def __init__(self, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 hidden: tuple[int, ...] = (128, 64), latent_dim: int = 16, init_noise_std: float = 1.0):
        super().__init__()
        self.estimator = HIMEstimator(obs_dim * hist_len, obs_dim, priv_dim, latent_dim)
        self.latent_dim = latent_dim

        def mlp(inp: int, dims: tuple[int, ...], out: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = inp
            for d in dims:
                layers += [nn.Linear(prev, d), nn.ELU()]
                prev = d
            layers.append(nn.Linear(prev, out))
            return nn.Sequential(*layers)

        self.actor = mlp(obs_dim + latent_dim, hidden, action_dim)
        self.critic = mlp(priv_dim + latent_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.log(torch.full((action_dim,), init_noise_std)))
        self.hist_len = hist_len
        self._init_roller(obs_dim)  # rolling history buffer, lazily sized

    # ---- rolling history ----------------------------------------------------
    def _push_hist(self, obs: torch.Tensor) -> torch.Tensor:
        return self.roll(obs)

    # ---- policy API ---------------------------------------------------------
    def act(self, obs: torch.Tensor, streams: dict | None = None):
        hist = self._push_hist(obs)
        flat = hist.reshape(hist.shape[0], -1)
        with torch.no_grad():
            latent, priv_pred = self.estimator(flat)
        mean = self.actor(torch.cat([obs, latent.detach()], dim=-1))
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy()

    def update_distribution(self, obs: torch.Tensor, hist_flat: torch.Tensor | None = None):
        if hist_flat is None:
            latent = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
        else:
            latent, _ = self.estimator(hist_flat)
        mean = self.actor(torch.cat([obs, latent], dim=-1))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def evaluate(self, priv_obs: torch.Tensor, hist_flat: torch.Tensor | None = None) -> torch.Tensor:
        latent = self.estimator(hist_flat)[0] if hist_flat is not None else torch.zeros(priv_obs.shape[0], self.latent_dim, device=priv_obs.device)
        return self.critic(torch.cat([priv_obs, latent], dim=-1)).squeeze(-1)

    def extra_loss(self, obs, priv, hist, priv_hist) -> torch.Tensor:
        """Estimator MSE: predict privileged info from history (detached target)."""
        if hist is None or priv_hist is None:
            return obs.new_zeros(())
        latent, priv_pred = self.estimator(hist)
        # one-step target: the CURRENT privileged state (target = what sensors
        # will confirm next step), detached so the estimator does not fight PPO
        target = priv_hist[:, -1, :].detach()
        return (priv_pred - target).square().mean()


class HIMTrainer:
    def __init__(self, env, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 device: str = "cpu", cfg: ExtTrainCfg | None = None):
        self.policy = HIMActorCritic(obs_dim, priv_dim, hist_len, action_dim)
        self.loop = ExtPPOLoop(env, self.policy, cfg or ExtTrainCfg(), device)

    def learn(self, iterations: int, episode_tracker=None):
        self.loop.learn(iterations, episode_tracker=episode_tracker)
