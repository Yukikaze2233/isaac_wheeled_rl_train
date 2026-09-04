"""DreamWaQ branch: implicit estimation via a CENet-style VAE on history.

Core mechanism: an encoder-decoder (CENet) embeds the observation history
into a latent that is trained as a variational autoencoder to reconstruct the
privileged stream; the actor consumes the latent (implicit estimation instead
of explicit prediction). Losses added to PPO: reconstruction MSE + KL.
"""
import torch
import torch.nn as nn

from .ppo_base import ExtPPOLoop, ExtTrainCfg, HistoryRoller


class CENet(nn.Module):
    """Encoder-decoder for privileged estimation from history (VAE)."""

    def __init__(self, hist_dim: int, priv_dim: int, latent_dim: int = 16,
                 enc: tuple[int, ...] = (256, 128, 64), dec: tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = hist_dim
        for d in enc:
            layers += [nn.Linear(prev, d), nn.ELU()]
            prev = d
        self.encoder = nn.Sequential(*layers)
        self.to_mu = nn.Linear(prev, latent_dim)
        self.to_logvar = nn.Linear(prev, latent_dim)

        layers = []
        prev = latent_dim
        for d in dec:
            layers += [nn.Linear(prev, d), nn.ELU()]
            prev = d
        layers.append(nn.Linear(prev, priv_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(hist)
        mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-8.0, 8.0)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def forward(self, hist: torch.Tensor):
        z, mu, logvar = self.encode(hist)
        return self.decoder(z), z, mu, logvar


class DreamWaqActorCritic(HistoryRoller, nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 hidden: tuple[int, ...] = (128, 64), latent_dim: int = 16, init_noise_std: float = 1.0):
        super().__init__()
        self.cenet = CENet(obs_dim * hist_len, priv_dim, latent_dim)
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
        self._init_roller(obs_dim)
        # loss weights (kl_weight applies to the VAE term)
        self.kl_weight = 1.0

    def _push_hist(self, obs: torch.Tensor) -> torch.Tensor:
        return self.roll(obs)

    def act(self, obs: torch.Tensor, streams: dict | None = None):
        hist = self._push_hist(obs)
        with torch.no_grad():
            recon, z, _, _ = self.cenet(hist.reshape(hist.shape[0], -1))
        mean = self.actor(torch.cat([obs, z], dim=-1))
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy()

    def update_distribution(self, obs: torch.Tensor, hist_flat: torch.Tensor | None = None):
        if hist_flat is None:
            latent = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
        else:
            _, z, _, _ = self.cenet(hist_flat)
            latent = z
        mean = self.actor(torch.cat([obs, latent], dim=-1))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def evaluate(self, priv_obs: torch.Tensor, hist_flat: torch.Tensor | None = None) -> torch.Tensor:
        latent = self.cenet(hist_flat)[1] if hist_flat is not None else torch.zeros(priv_obs.shape[0], self.latent_dim, device=priv_obs.device)
        return self.critic(torch.cat([priv_obs, latent], dim=-1)).squeeze(-1)

    def extra_loss(self, obs, priv, hist, priv_hist) -> torch.Tensor:
        """VAE loss: reconstruction of the privileged stream + KL to N(0,1)."""
        if hist is None or priv_hist is None:
            return obs.new_zeros(())
        recon, z, mu, logvar = self.cenet(hist)
        target = priv_hist[:, -1, :]
        recon_loss = (recon - target.detach()).square().mean()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean() / target.shape[-1]
        return recon_loss + self.kl_weight * kl


class DreamWaqTrainer:
    def __init__(self, env, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 device: str = "cpu", cfg: ExtTrainCfg | None = None):
        self.policy = DreamWaqActorCritic(obs_dim, priv_dim, hist_len, action_dim)
        self.loop = ExtPPOLoop(env, self.policy, cfg or ExtTrainCfg(), device)

    def learn(self, iterations: int, episode_tracker=None):
        self.loop.learn(iterations, episode_tracker=episode_tracker)
