"""PC gain/strength priors around this robot's identified nominal parameters."""
import torch


class PcParameters:
    def __init__(self, count, device, settings, wheel_kd, config, generator):
        self.cfg, self.generator = dict(config), generator
        self.nominal_kp = torch.tensor([settings["leg_kp"], settings["leg_kp"], 0.,
            settings["leg_kp"], settings["leg_kp"], 0.], device=device)
        self.nominal_kd = torch.tensor([settings["leg_kd"], settings["leg_kd"], wheel_kd,
            settings["leg_kd"], settings["leg_kd"], wheel_kd], device=device)
        self.kp = self.nominal_kp.expand(count, 6).clone()
        self.kd = self.nominal_kd.expand(count, 6).clone()
        self.strength = torch.ones_like(self.kp)
        self.last_sample_policy_tick = torch.full((count,), -720, device=device, dtype=torch.long)
        if config.get("enabled") and config.get("version") != "pc_relative_priors_v1":
            raise ValueError("PC DR requires a versioned prior")
        self.reset(torch.arange(count, device=device), global_policy_tick=0)

    def reset(self, ids, *, global_policy_tick):
        if type(global_policy_tick) is not int or global_policy_tick < 0:
            raise ValueError("PC DR reset needs the global integer policy/environment tick")
        if not self.cfg.get("enabled"):
            return
        # Reference reset cooldown is720 policy steps =14.4s. Initial reset is
        # always sampled. Short episodes retain parameters until cooldown.
        elapsed = global_policy_tick - self.last_sample_policy_tick[ids]
        selected = ids[elapsed >= 720]
        shape = (len(selected), 6)
        kp_scale = .75 + .5 * torch.rand(shape, device=self.kp.device, generator=self.generator)
        kd_scale = .75 + .5 * torch.rand(shape, device=self.kp.device, generator=self.generator)
        strength = torch.ones(shape, device=self.kp.device)
        strength[:, [0, 1, 3, 4]] = .8 + .3 * torch.rand((len(selected), 4),
            device=self.kp.device, generator=self.generator)
        strength[:, [2, 5]] = .9 + .2 * torch.rand((len(selected), 2),
            device=self.kp.device, generator=self.generator)
        self.kp[selected] = self.nominal_kp * kp_scale
        self.kd[selected] = self.nominal_kd * kd_scale
        self.strength[selected] = strength
        self.last_sample_policy_tick[selected] = global_policy_tick
