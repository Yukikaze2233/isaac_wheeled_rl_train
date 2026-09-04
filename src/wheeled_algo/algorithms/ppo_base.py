"""Compact asymmetric-PPO training loop shared by the algorithm extensions.

Mirrors the rsl_rl on-policy contract (collect 24 steps -> GAE -> minibatch
updates with adaptive-KL LR) and adds an `extra_loss(obs_hist, priv, actions)`
hook for the estimator/VAE/constraint losses each branch contributes.

Env protocol (same family as rsl_rl 2.3.x, extended):
    get_observations() -> (policy_obs, extras)   extras["observations"]["critic"]
    step(actions)      -> (policy_obs, rewards, dones, extras)
Plus, when the env provides them (real wheeled-biped task):
    extras["observations"]["policy_hist"]  (N, H, obs_dim)  observation history
    extras["observations"]["priv_hist"]    (N, H, priv_dim) privileged history
"""
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ExtTrainCfg:
    num_steps_per_env: int = 24
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 2.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    desired_kl: float = 0.02
    max_lr: float = 1e-2
    min_lr: float = 1e-5


class HistoryRoller:
    """Shared flat-history roller: obs -> (hist_len * obs_dim) buffer.

    Five branches (him/dreamwaq/np3o/ppo_hist/recurrent) roll the same way;
    they mixin this and call `roll(obs)` instead of each keeping a copy.
    """

    hist_len: int

    def _init_roller(self, obs_dim: int) -> None:
        self._obs_dim = obs_dim
        self._hist: torch.Tensor | None = None

    def roll(self, obs: torch.Tensor) -> torch.Tensor:
        if self._hist is None:
            self._hist = obs.repeat(1, self.hist_len)
        else:
            self._hist = torch.cat([self._hist[:, self._obs_dim:], obs], dim=-1)
        return self._hist


class ExtPPOLoop:
    """Rollout + GAE + PPO updates for an ActorCritic that exposes
    `act(obs)` -> (action, log_prob, entropy) and `evaluate(priv_obs)`."""

    def __init__(self, env, policy: nn.Module, cfg: ExtTrainCfg, device: str):
        self.env = env
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.learning_rate = cfg.learning_rate
        self.current_iteration = 0
        # multi-process training: launched with torch.distributed env vars
        # (mirrors rsl_rl's --distributed; grad sync below, no rank clipping)
        self.distributed = False
        if "WORLD_SIZE" in os.environ and torch.distributed.is_available():
            if torch.distributed.is_initialized() and int(os.environ["WORLD_SIZE"]) > 1:
                self.distributed = True
                self.world_size = torch.distributed.get_world_size()

    # ---- environment access ------------------------------------------------
    def _env_obs(self):
        obs, extras = self.env.get_observations()
        return obs, extras.get("observations", {})

    # ---- collection --------------------------------------------------------
    def collect(self, n_steps: int):
        cfg = self.cfg
        obs, streams = self._env_obs()
        storage = {"obs": [], "priv": [], "hist": [], "priv_hist": [],
                   "actions": [], "logp": [], "rewards": [], "dones": [], "values": []}
        for _ in range(n_steps):
            with torch.no_grad():
                action, logp, _ = self.policy.act(obs, streams)
                value = self.policy.evaluate(streams.get("critic", obs))
            nxt_obs, rewards, dones, nxt_streams = self.env.step(action)
            storage["obs"].append(obs)
            storage["priv"].append(streams.get("critic", obs))
            if "policy_hist" in nxt_streams:
                storage["hist"].append(nxt_streams["policy_hist"])
            if "priv_hist" in nxt_streams:
                storage["priv_hist"].append(nxt_streams["priv_hist"])
            storage["actions"].append(action)
            storage["logp"].append(logp)
            storage["rewards"].append(rewards)
            storage["dones"].append(dones.float())
            storage["values"].append(value)
            obs, streams = nxt_obs, nxt_streams.get("observations", {})
        # stack the per-step lists into (T, N, ...) tensors before GAE
        for k in ("obs", "priv", "actions", "logp", "rewards", "dones", "values"):
            storage[k] = torch.stack(storage[k])
        with torch.no_grad():
            last_value = self.policy.evaluate(streams.get("critic", obs))

        # GAE
        T, N = storage["rewards"].shape[0], storage["rewards"].shape[1]
        advantages = torch.zeros_like(storage["rewards"])
        lastgaelam = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else storage["values"][t + 1]
            not_done = 1.0 - storage["dones"][t]
            delta = storage["rewards"][t] + cfg.gamma * next_v * not_done - storage["values"][t]
            lastgaelam = delta + cfg.gamma * cfg.lam * not_done * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + storage["values"]
        storage["advantages"] = advantages
        storage["returns"] = returns
        return storage, last_value

    # ---- update ------------------------------------------------------------
    def update(self, storage) -> dict:
        cfg = self.cfg
        T, N = storage["rewards"].shape[:2]

        def flat(k):
            v = storage[k]
            if isinstance(v, list):  # optional streams (hist/priv_hist) accumulate as lists
                v = torch.stack(v) if v else None
            return v.reshape(T * N, *v.shape[2:])
        obs, priv, act, logp_old = flat("obs"), flat("priv"), flat("actions"), flat("logp")
        ret, adv = flat("returns"), flat("advantages")
        hist = flat("hist") if storage["hist"] else None
        priv_hist = flat("priv_hist") if storage["priv_hist"] else None

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = (T * N) // cfg.num_mini_batches
        perm = torch.randperm(T * N, device=self.device)

        stats = {"value": 0.0, "surr": 0.0, "extra": 0.0, "kl": 0.0}
        n_updates = 0
        for _ in range(cfg.num_learning_epochs):
            for start in range(0, T * N, batch):
                sel = perm[start:start + batch]
                dist = self.policy.update_distribution(obs[sel], hist[sel] if hist is not None else None)
                logp = dist.log_prob(act[sel]).sum(-1)
                ratio = (logp - logp_old[sel]).exp()
                surr = -torch.min(ratio * adv[sel],
                                  torch.clamp(ratio, 1 - cfg.clip_param, 1 + cfg.clip_param) * adv[sel]).mean()
                value = self.policy.evaluate(priv[sel], hist[sel] if hist is not None else None)
                v_loss = (value - ret[sel]).square().mean()
                extra = self.policy.extra_loss(obs[sel], priv[sel],
                                               hist[sel] if hist is not None else None,
                                               priv_hist[sel] if priv_hist is not None else None)
                entropy = dist.entropy().sum(-1).mean()
                penalty = getattr(self.policy, "surrogate_penalty", None)
                constraint = (penalty(obs[sel], hist[sel] if hist is not None else None, priv[sel])
                              if penalty is not None else obs.new_zeros(()))
                loss = (surr + cfg.value_loss_coef * v_loss - cfg.entropy_coef * entropy
                        + extra + constraint)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                if self.distributed:
                    for pg in self.optimizer.param_groups:
                        for param in pg["params"]:
                            if param.grad is not None:
                                torch.distributed.all_reduce(param.grad)
                                param.grad /= self.world_size
                self.optimizer.step()

                with torch.no_grad():
                    kl = (logp_old[sel] - logp).mean().abs().item()
                stats["value"] += v_loss.item()
                stats["surr"] += surr.item()
                stats["extra"] += float(extra)
                stats["kl"] += kl
                n_updates += 1
        for k in stats:
            stats[k] /= max(n_updates, 1)

        if stats["kl"] > cfg.desired_kl * 2.0:
            self.learning_rate = max(cfg.min_lr, self.learning_rate * 0.5)
        elif stats["kl"] < cfg.desired_kl / 2.0:
            self.learning_rate = min(cfg.max_lr, self.learning_rate * 1.5)
        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate
        return stats

    def learn(self, iterations: int, episode_tracker=None, log_every: int = 20, metrics_hook=None):
        start = self.current_iteration
        for it in range(start, start + iterations):
            storage, _ = self.collect(self.cfg.num_steps_per_env)
            stats = self.update(storage)
            self.current_iteration = it + 1
            if episode_tracker is not None:
                episode_tracker()
            if metrics_hook is not None:
                metrics_hook(it, dict(stats))
            if it % log_every == 0:
                ep = episode_tracker.mean() if episode_tracker is not None else float("nan")
                print(f"it={it:4d} ep_reward={ep:9.2f} extra={stats['extra']:8.4f} "
                      f"kl={stats['kl']:.4f} lr={self.learning_rate:.2e}")

    # ---- iterative training: checkpoint round-trip --------------------------
    def save(self, path: str) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.current_iteration,
            "lr": self.learning_rate,
        }, path)

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.learning_rate = ckpt.get("lr", self.learning_rate)
        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate
        self.current_iteration = ckpt.get("iteration", 0)
        return self.current_iteration
