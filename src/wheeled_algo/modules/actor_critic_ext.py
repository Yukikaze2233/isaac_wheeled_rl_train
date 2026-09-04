"""rsl_rl-integrated ActorCritic branches (frame-stacked history contract).

Each class is an rsl_rl `ActorCritic` subclass resolved by the runner's
`eval(class_name)` dispatch. The ENV stacks history into the policy observation
(obs_dim = base_dim x H, oldest->newest); these classes reshape internally:

    frame-stacked obs -> (H, base) sequence -> memory encoder -> latent
    actor: last_frame + latent (residual current-obs path)
    critic: privileged obs + latent

The aux losses (estimator/VAE/twins) are implemented on the PPO side
(algorithms/ppo_ext.py) so the policy stays a pure forward model.
"""
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic as RslActorCritic


def _mlp(inp, dims, out, act=nn.ELU):
    layers, prev = [], inp
    for d in dims:
        layers += [nn.Linear(prev, d), act()]
        prev = d
    layers.append(nn.Linear(prev, out))
    return nn.Sequential(*layers)


class _StackedHistoryBase(RslActorCritic):
    """Shared: frame-stacked obs reshaping + memory encoder + latent paths."""

    def __init__(self, num_obs, num_priv, num_actions, *, base_obs_dim, hist_len,
                 latent_dim=16, encoder="estimator", hidden=(128, 64),
                 activation="elu", init_noise_std=1.0, **kwargs):
        # rsl_rl ActorCritic builds actor for num_obs; we rebuild heads below
        super().__init__(num_obs, num_priv, num_actions,
                         activation=activation, init_noise_std=init_noise_std)
        self.base_obs_dim = base_obs_dim
        self.priv_base_dim = num_priv // hist_len  # num_priv arrives frame-stacked
        self.hist_len = hist_len
        self.latent_dim = latent_dim
        hist_dim = base_obs_dim * hist_len
        if encoder == "estimator":     # HIM: deterministic one-step estimator
            self.memory = _mlp(hist_dim, (128, 128), latent_dim)
            self.priv_head = nn.Linear(latent_dim, num_priv)
            self._priv_feat = None
        elif encoder == "cenet":       # DreamWaQ: VAE (mu/logvar heads)
            body = _mlp(hist_dim, (128, 128), latent_dim)
            self.memory = nn.Sequential(body[:-1])
            self.to_mu = nn.Linear(128, latent_dim)
            self.to_logvar = nn.Linear(128, latent_dim)
            self.decoder = _mlp(latent_dim, (64, 128), num_priv)
        elif encoder == "twins":       # NP3O: BarlowTwins encoder
            self.memory = _mlp(hist_dim, (128,), latent_dim)
        self.encoder_name = encoder
        # critic-side encoder: same latent space, but over the PRIVILEGED
        # history frames (separate weights — input dim differs from policy obs)
        self.priv_memory = _mlp(self.priv_base_dim * hist_len, (128,), latent_dim)
        # heads consume last frame + latent
        self.actor = _mlp(base_obs_dim + latent_dim, hidden, num_actions)
        self.critic = _mlp(num_priv + latent_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.log(torch.full((num_actions,), init_noise_std)))

    def _frames(self, obs: torch.Tensor) -> torch.Tensor:
        """(N, base*H) -> (N, H, base), oldest->newest."""
        return obs[:, : self.base_obs_dim * self.hist_len].reshape(
            obs.shape[0], self.hist_len, self.base_obs_dim)

    def _last_frame(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.base_obs_dim * (self.hist_len - 1): self.base_obs_dim * self.hist_len]

    def _latent(self, obs: torch.Tensor) -> torch.Tensor:
        flat = self._frames(obs).reshape(obs.shape[0], -1)
        return self.memory(flat)


    def update_distribution(self, observations):
        latent = self._latent(observations)
        mean = self.actor(torch.cat([self._last_frame(observations), latent], dim=-1))
        # rsl_rl's act() consumes the self.distribution attribute (base-class
        # convention); also returned for compatibility with our ExtPPOLoop.
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        return self.distribution

    def _priv_frames(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return critic_obs.reshape(critic_obs.shape[0], self.hist_len, self.priv_base_dim)

    def evaluate(self, critic_observations, **kwargs):
        # critic obs is frame-stacked with the SAME H (env stacks both streams);
        # latent is recomputed from the privileged history per minibatch row.
        flat = self._priv_frames(critic_observations).reshape(critic_observations.shape[0], -1)
        latent = self.priv_memory(flat)
        return self.critic(torch.cat([critic_observations, latent], dim=-1)).squeeze(-1)


class ActorCriticHIM(_StackedHistoryBase):
    encoder_name = "estimator"


class ActorCriticDreamWaq(_StackedHistoryBase):
    encoder_name = "cenet"


class ActorCriticBarlowTwins(_StackedHistoryBase):
    encoder_name = "twins"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.projector = nn.Sequential(nn.Linear(self.latent_dim, 32), nn.BatchNorm1d(32),
                                       nn.ELU(), nn.Linear(32, 32))

    def twins_loss(self, obs_batch: torch.Tensor, offdiag: float = 0.01) -> torch.Tensor:
        x = self._frames(obs_batch).reshape(obs_batch.shape[0], -1)
        x2 = torch.cat([x[:, x.shape[1] // 2:], x[:, :x.shape[1] // 2]], dim=-1)
        z1, z2 = self.projector(self.memory(x)), self.projector(self.memory(x2))
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        cross = z1.T @ z2 / x.shape[0]
        diag = torch.pow(torch.diagonal(cross) - 1.0, 2).sum()
        off = offdiag * torch.pow(cross - torch.diag(torch.diagonal(cross)), 2).sum()
        return diag + off


class ActorCriticRecurrentGRU(_StackedHistoryBase):
    encoder_name = None  # placeholder; GRU variant uses its own encoder

    def __init__(self, num_obs, num_priv, num_actions, *, base_obs_dim, hist_len,
                 latent_dim=64, **kw):
        raise NotImplementedError("use ExtOnPolicyRunner GRU branch (sequence GRU path)")
