"""Recurrent branch (GRU-memory branch): GRU memory over the history.

Actor consumes the current observation plus the GRU hidden state evolved over
the observation-history sequence; the current-observation path stays intact
(residual design of the reference Exp059). Fits the ExtPPOLoop API.
"""
import torch
import torch.nn as nn

from .ppo_base import ExtPPOLoop, ExtTrainCfg


class GRUActorCritic(nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 hidden: tuple[int, ...] = (128, 64), latent_dim: int = 64,
                 init_noise_std: float = 1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.hist_len = hist_len
        self.latent_dim = latent_dim  # GRU hidden size
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=latent_dim, batch_first=True)

        def mlp(inp: int, dims: tuple[int, ...], out: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = inp
            for d in dims:
                layers += [nn.Linear(prev, d), nn.ELU()]
                prev = d
            layers.append(nn.Linear(prev, out))
            return nn.Sequential(*layers)

        self.actor = mlp(obs_dim + latent_dim, hidden, action_dim)  # residual: obs path kept
        self.critic = mlp(priv_dim + latent_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.log(torch.full((action_dim,), init_noise_std)))
        self._hidden = None

    def _encode_hist(self, obs: torch.Tensor, hist_seq: torch.Tensor | None) -> torch.Tensor:
        """hist_seq: (N, H, obs_dim) oldest->newest; falls back to zeros."""
        if hist_seq is None:
            return torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
        out, _ = self.gru(hist_seq)
        return out[:, -1]  # last step hidden

    def act(self, obs: torch.Tensor, streams: dict | None = None):
        hist_seq = (streams or {}).get("policy_hist")  # (N, H, obs_dim) oldest->newest
        with torch.no_grad():
            hidden = self._encode_hist(obs, hist_seq)
        mean = self.actor(torch.cat([obs, hidden], dim=-1))
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy()

    def update_distribution(self, obs: torch.Tensor, hist_flat: torch.Tensor | None = None,
                            hist_seq: torch.Tensor | None = None):
        if hist_seq is not None:
            hidden = self._encode_hist(obs, hist_seq)
        elif hist_flat is not None:
            # reshape flat (N, H*obs) back to a sequence for the GRU
            seq = hist_flat.reshape(obs.shape[0], self.hist_len, self.obs_dim)
            hidden = self._encode_hist(obs, seq)
        else:
            hidden = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
        mean = self.actor(torch.cat([obs, hidden], dim=-1))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def evaluate(self, priv_obs: torch.Tensor, hist_flat: torch.Tensor | None = None) -> torch.Tensor:
        if hist_flat is not None:
            seq = hist_flat.reshape(priv_obs.shape[0], self.hist_len, self.obs_dim)
            hidden = self._encode_hist(priv_obs, seq)
        else:
            hidden = torch.zeros(priv_obs.shape[0], self.latent_dim, device=priv_obs.device)
        return self.critic(torch.cat([priv_obs, hidden], dim=-1)).squeeze(-1)

    def extra_loss(self, obs, priv, hist, priv_hist) -> torch.Tensor:
        return obs.new_zeros(())


class GRUTrainer:
    def __init__(self, env, obs_dim: int, priv_dim: int, hist_len: int, action_dim: int,
                 device: str = "cpu", cfg: ExtTrainCfg | None = None):
        self.policy = GRUActorCritic(obs_dim, priv_dim, hist_len, action_dim)
        self.loop = ExtPPOLoop(env, self.policy, cfg or ExtTrainCfg(), device)

    def learn(self, iterations: int, episode_tracker=None):
        self.loop.learn(iterations, episode_tracker=episode_tracker)
