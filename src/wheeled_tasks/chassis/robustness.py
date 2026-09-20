"""Seeded policy-rate sensor/action delay and modest research observation noise."""
import torch

from wheeled_tasks.v40.core import HistoryStack


class V5SignalPerturbations:
    def __init__(self, count, device, config, generator):
        self.cfg, self.generator = config, generator
        self.max_lag = int(config.get("max_delay_steps", 2))
        self.observations = HistoryStack(count, device, self.max_lag + 1, 46)
        self.actions = HistoryStack(count, device, self.max_lag + 1, 6)
        self.obs_lag = torch.zeros(count, dtype=torch.long, device=device)
        self.act_lag = torch.zeros_like(self.obs_lag)
        self.rows = torch.arange(count, device=device)
        self.enabled = torch.ones(count, dtype=torch.bool, device=device)
        self.sensor_indices = list(range(6)) + list(range(9, 19)) + list(range(25, 29)) + list(range(38, 46))
        self.noise = torch.zeros(46, device=device)
        self.noise[:3] = .0025
        self.noise[3:6] = .005
        self.noise[9:13] = .003
        self.noise[13:19] = .005
        self.noise[25:27] = .0005 / .08
        self.noise[27:29] = .005 / 1.2
        self.noise *= config.get("noise_scale", 1.)

    def reset(self, ids):
        self.observations.reset(ids)
        self.actions.reset(ids)
        self.obs_lag[ids] = torch.randint(0, self.max_lag + 1, (len(ids),), generator=self.generator, device=self.rows.device)
        self.act_lag[ids] = torch.randint(0, self.max_lag + 1, (len(ids),), generator=self.generator, device=self.rows.device)
        enabled = torch.rand(len(ids), generator=self.generator, device=self.rows.device) < self.cfg.get("enabled_fraction", 1.)
        self.set_enabled(ids, enabled)

    def set_enabled(self, ids, enabled):
        self.enabled[ids] = enabled
        self.obs_lag[ids] *= enabled
        self.act_lag[ids] *= enabled

    def action(self, values, tick):
        history = self.actions.update(values, tick).reshape(len(values), self.max_lag + 1, 6)
        return history[self.rows, self.max_lag - self.act_lag]

    def observation(self, clean, tick):
        fresh = ~self.observations.initialized
        update = torch.ones_like(fresh) if self.observations.last_tick != tick else fresh
        values = self.observations.buffer[:, -1].clone()
        if bool(update.any()):
            noise = torch.randn((int(update.sum()), 46), generator=self.generator, device=clean.device)
            values[update] = clean[update] + noise * self.noise * self.enabled[update, None]
        history = self.observations.update(values, tick).reshape(len(clean), self.max_lag + 1, 46)
        delayed = history[self.rows, self.max_lag - self.obs_lag]
        result = clean.clone()
        result[:, self.sensor_indices] = delayed[:, self.sensor_indices]
        return result.clamp(-100., 100.)
