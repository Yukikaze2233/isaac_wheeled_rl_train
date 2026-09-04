"""Toy env with observation history + privileged streams for algo_ext tests.

Same reach task as toy_env.ToyReachRslEnv, but get_observations/step also
provide the extended streams the algorithm branches consume:
    extras["observations"]["policy_hist"]  (N, hist_len, obs_dim)
    extras["observations"]["priv_hist"]    (N, hist_len, priv_dim)
"""
import torch


class _Space:
    def __init__(self, dim: int):
        self.shape = (dim,)


class ToyReachExtEnv:
    def __init__(self, num_envs: int = 256, max_ep_steps: int = 50, device: str = "cpu",
                 seed: int = 0, hist_len: int = 3):
        self.num_envs = num_envs
        self.num_actions = 2
        self.max_episode_length = max_ep_steps
        self.device = device
        self.hist_len = hist_len
        self.obs_dim = 6
        self.priv_dim = 3
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.action_space = _Space(2)
        self._pos = None
        self._vel = None
        self._steps = None
        self._hist = None
        self._priv_hist = None
        self.stack_policy_obs = False
        self._stack_buf = None
        self._stack_priv = None
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset_()

    def reset_(self):
        self._pos = 2.0 * torch.rand((self.num_envs, 2), generator=self.rng, device=self.device) - 1.0
        self._vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._steps = torch.zeros(self.num_envs, device=self.device)
        self.episode_length_buf.zero_()
        self._stack_buf = None
        self._stack_priv = None
        return self.get_observations()

    def _streams(self, policy: torch.Tensor, critic: torch.Tensor) -> dict:
        if getattr(self, "stack_policy_obs", False):
            # policy obs already carries the stacked history; rsl_rl runner
            # only consumes the critic group from extras
            return {"critic": critic}
        frame = policy.unsqueeze(1)
        self._hist = frame.repeat(1, self.hist_len, 1) if self._hist is None else \
            torch.cat([self._hist[:, 1:], frame], dim=1)
        pframe = critic.unsqueeze(1)
        self._priv_hist = pframe.repeat(1, self.hist_len, 1) if self._priv_hist is None else \
            torch.cat([self._priv_hist[:, 1:], pframe], dim=1)
        return {
            "critic": critic,
            "policy_hist": self._hist,
            "priv_hist": self._priv_hist,
        }

    def get_observations(self):
        zero_cmd = torch.zeros(self.num_envs, 2, device=self.device)
        policy = torch.cat([self._pos, self._vel, zero_cmd], dim=-1)
        critic = torch.cat([policy, self._pos.norm(dim=-1, keepdim=True)], dim=-1)
        policy, critic = self._maybe_stack(policy, critic)
        return policy, {"observations": self._streams(policy, critic)}

    def _maybe_stack(self, policy, critic):
        """Frame-stack policy AND critic streams (rsl_rl branch contract)."""
        if not getattr(self, "stack_policy_obs", False):
            return policy, critic
        if self._stack_buf is None:
            self._stack_buf = policy.repeat(1, self.hist_len)
            self._stack_priv = critic.repeat(1, self.hist_len)
        else:
            self._stack_buf = torch.cat([self._stack_buf[:, policy.shape[1]:], policy], dim=-1)
            self._stack_priv = torch.cat([self._stack_priv[:, critic.shape[1]:], critic], dim=-1)
        return self._stack_buf, self._stack_priv

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
        if torch.any(done):
            ids = done.nonzero(as_tuple=False).flatten()
            self._pos[ids] = 2.0 * torch.rand((ids.numel(), 2), generator=self.rng, device=self.device) - 1.0
            self._vel[ids] = 0.0
            self._steps[ids] = 0
            self.episode_length_buf[ids] = 0

        policy, extras = self.get_observations()
        extras["time_outs"] = truncated
        return policy, reward, done, extras

    def close(self):
        pass
