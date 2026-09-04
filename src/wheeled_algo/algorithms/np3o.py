"""NP3O branch: near-constrained PPO with BarlowTwins representation.

Core mechanisms (NP3O-style, practice level):
- BarlowTwins: two skewed views of the observation history are embedded and
  their cross-correlation matrix is pushed to the identity — representation
  learning without negative pairs.
- Constrained RL: a cost critic learns the expected constraint violation and
  a Lagrangian multiplier auto-tunes the penalty so the average cost stays
  near the configured limit (nP3O-style safe exploration).
"""
import torch
import torch.nn as nn

from .ppo_base import ExtPPOLoop, ExtTrainCfg, HistoryRoller


class BarlowTwinsEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden: int = 128, proj_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(), nn.Linear(hidden, latent_dim))
        self.projector = nn.Sequential(nn.Linear(latent_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ELU(),
                                       nn.Linear(proj_dim, proj_dim))
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def twins_loss(self, x1: torch.Tensor, x2: torch.Tensor, offdiag: float = 0.01) -> torch.Tensor:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        zn1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        zn2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        cross = zn1.T @ zn2 / x1.shape[0]
        diag = torch.pow(torch.diagonal(cross) - 1.0, 2).sum()
        off = offdiag * torch.pow(cross - torch.diag(torch.diagonal(cross)), 2).sum()
        return diag + off


class NP3OActorCritic(HistoryRoller, nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 hidden: tuple[int, ...] = (128, 64), latent_dim: int = 16, init_noise_std: float = 1.0):
        super().__init__()
        self.twins = BarlowTwinsEncoder(obs_dim * hist_len, latent_dim)
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
        self.cost_critic = mlp(priv_dim + latent_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.log(torch.full((action_dim,), init_noise_std)))
        self.hist_len = hist_len
        self._init_roller(obs_dim)
        # constraint tuning (nP3O): multiplier chases the cost limit
        self.cost_limit = 0.5
        self.lagrangian = nn.Parameter(torch.zeros(()), requires_grad=False)

    def _push_hist(self, obs: torch.Tensor) -> torch.Tensor:
        return self.roll(obs)

    def act(self, obs: torch.Tensor, streams: dict | None = None):
        hist = self._push_hist(obs)
        with torch.no_grad():
            latent = self.twins(hist.reshape(hist.shape[0], -1))
        mean = self.actor(torch.cat([obs, latent], dim=-1))
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy()

    def update_distribution(self, obs: torch.Tensor, hist_flat: torch.Tensor | None = None):
        if hist_flat is None:
            latent = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
        else:
            latent = self.twins(hist_flat)
        mean = self.actor(torch.cat([obs, latent], dim=-1))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def evaluate(self, priv_obs: torch.Tensor, hist_flat: torch.Tensor | None = None) -> torch.Tensor:
        latent = self.twins(hist_flat) if hist_flat is not None else torch.zeros(priv_obs.shape[0], self.latent_dim, device=priv_obs.device)
        return self.critic(torch.cat([priv_obs, latent], dim=-1)).squeeze(-1)

    def evaluate_cost(self, priv_obs: torch.Tensor, hist_flat: torch.Tensor | None = None) -> torch.Tensor:
        latent = self.twins(hist_flat) if hist_flat is not None else \
            torch.zeros(priv_obs.shape[0], self.latent_dim, device=priv_obs.device)
        return self.cost_critic(torch.cat([priv_obs, latent], dim=-1)).squeeze(-1)

    def extra_loss(self, obs, priv, hist, priv_hist) -> torch.Tensor:
        """BarlowTwins on two skewed history views + constraint value loss."""
        if hist is None or priv_hist is None:
            return obs.new_zeros(())
        x = hist
        x2 = torch.cat([x[:, x.shape[1] // 2:], x[:, :x.shape[1] // 2]], dim=-1)  # skewed view
        twins = self.twins.twins_loss(x, x2)

        # cost critic: expected violation given privileged info (detached target)
        with torch.no_grad():
            target_cost = self._instant_cost(priv).detach()
        cost_pred = self.evaluate_cost(priv)
        cost_loss = (cost_pred - target_cost).square().mean()

        # Lagrangian update: push mean cost toward the limit (nP3O multiplier rule)
        mean_cost = target_cost.mean()
        self.lagrangian += 0.05 * (mean_cost - self.cost_limit)
        self.lagrangian.clamp_(0.0, 10.0)
        return twins + 0.1 * cost_loss

    def surrogate_penalty(self, obs: torch.Tensor, hist_flat: torch.Tensor | None = None,
                          priv_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Hook called by the training loop inside the PPO loss: penalize the
        expected constraint cost with the auto-tuned Lagrangian (nP3O)."""
        if hist_flat is not None:
            latent = self.twins(hist_flat)
        else:
            latent = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
        critic_in = priv_obs if priv_obs is not None else obs  # cost critic is privileged
        cost = self.cost_critic(torch.cat([critic_in, latent], dim=-1)).squeeze(-1)
        return self.lagrangian * cost.mean()

    def _instant_cost(self, priv: torch.Tensor) -> torch.Tensor:
        """Constraint demo on the toy env: stay tilted (gravity x) within limits.
        On the real task this maps to tilt/height/contact violation terms."""
        return priv[:, -1]  # last privileged dim (toy: position norm)


class NP3OTrainer:
    def __init__(self, env, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 device: str = "cpu", cfg: ExtTrainCfg | None = None):
        self.policy = NP3OActorCritic(obs_dim, priv_dim, hist_len, action_dim)
        self.loop = ExtPPOLoop(env, self.policy, cfg or ExtTrainCfg(), device)

    def learn(self, iterations: int, episode_tracker=None):
        self.loop.learn(iterations, episode_tracker=episode_tracker)
