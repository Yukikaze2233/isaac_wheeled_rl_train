"""Toy environment adapted to the rsl_rl 2.3.x VecEnv protocol.

Lets the stock rsl_rl OnPolicyRunner train a policy locally (CPU, no Isaac
Sim) so the training stack wiring — runner loop, asymmetric critic extras,
checkpointing — is regression-tested without a GPU.

rsl_rl 2.3.x protocol consumed here:
    env.get_observations() -> (policy_obs, extras)   extras["observations"]["critic"]
    env.step(actions)      -> (policy_obs, rewards, dones, extras)
    env.reset_()           -> (policy_obs, extras)
    env.num_envs / num_actions / max_episode_length / episode_length_buf
"""
import torch


class ToyReachRslEnv:
    """2-D double integrator drives to the origin; IsaacLab-style auto-reset."""

    def __init__(self, num_envs: int = 256, max_ep_steps: int = 50, device: str = "cpu", seed: int = 0):
        self.num_envs = num_envs
        self.num_actions = 2
        self.max_episode_length = max_ep_steps
        self.device = device
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self._pos = None
        self._vel = None
        self._steps = None
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset_()

    def reset_(self):
        self._pos = 2.0 * torch.rand((self.num_envs, 2), generator=self.rng, device=self.device) - 1.0
        self._vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._steps = torch.zeros(self.num_envs, device=self.device)
        self.episode_length_buf.zero_()
        return self.get_observations()

    def get_observations(self):
        zero_cmd = torch.zeros(self.num_envs, 2, device=self.device)
        policy = torch.cat([self._pos, self._vel, zero_cmd], dim=-1)
        # asymmetric critic: true position norm as privileged signal
        critic = torch.cat([policy, self._pos.norm(dim=-1, keepdim=True)], dim=-1)
        return policy, {"observations": {"critic": critic}}

    def step(self, action: torch.Tensor):
        acc = torch.clamp(action, -1.0, 1.0)
        self._vel = self._vel + 0.1 * acc
        self._pos = self._pos + 0.1 * self._vel
        self._steps += 1
        self.episode_length_buf += 1

        reward = torch.exp(-self._pos.square().sum(-1)) - 0.1 * acc.square().sum(-1)
        terminated = self._pos.norm(dim=-1) < 0.05
        truncated = self._steps >= self.max_episode_length
        done = terminated | truncated

        # auto-reset done envs (IsaacLab DirectRLEnv semantics)
        if torch.any(done):
            ids = done.nonzero(as_tuple=False).flatten()
            self._pos[ids] = 2.0 * torch.rand((ids.numel(), 2), generator=self.rng, device=self.device) - 1.0
            self._vel[ids] = 0.0
            self._steps[ids] = 0
            self.episode_length_buf[ids] = 0

        obs, extras = self.get_observations()
        extras["time_outs"] = truncated
        return obs, reward, done, extras

    def close(self):
        pass
