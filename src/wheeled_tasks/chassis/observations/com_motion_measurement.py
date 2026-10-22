"""Policy-rate world root-rigid-body COM derivatives for the runtime logger.

These measurements do not modify control, rewards or learning masks. They are
backward differences at 20 ms, not physical-substep peaks, the whole robot's
mass-weighted COM, or settling tests. Every tag describes one policy frame;
the existing logger averages these tags across its reporting window. In
particular, frame maxima are not rollout peaks and frame RMS averages are not
the RMS of all rollout samples. Episode records carry cumulative measurements.
"""
import torch


class ComMotionMeasurement:
    def __init__(self, velocity):
        self.velocity = torch.zeros_like(velocity)
        self.acceleration = torch.zeros_like(velocity)
        self.valid = torch.zeros(len(velocity), dtype=torch.bool, device=velocity.device)
        self.acceleration_valid = torch.zeros_like(self.valid)
        self.episode = torch.zeros(len(velocity), dtype=torch.long, device=velocity.device)

    def observe(self, velocity, episode, valid):
        current_valid = valid & torch.isfinite(velocity).all(-1)
        pair = current_valid & self.valid & (episode == self.episode)
        acceleration = (velocity - self.velocity) / .02
        pair &= torch.isfinite(acceleration).all(-1)
        triple = pair & self.acceleration_valid
        jerk = (acceleration - self.acceleration) / .02
        triple &= torch.isfinite(jerk).all(-1)
        result = {}
        for name, values, mask, units in (
                ("acceleration", acceleration, pair, "m_s2"),
                ("jerk", jerk, triple, "m_s3")):
            squared = torch.where(mask[:, None], values, 0.).double().square().sum(-1)
            count = mask.sum()
            result[f"/motion/world_root_COM_{name}_samples_per_policy_frame"] = count
            result[f"/motion/world_root_COM_{name}_frame_rms_{units}"] = (squared.sum() / count.clamp_min(1)).sqrt()
            result[f"/motion/world_root_COM_{name}_frame_max_{units}"] = squared.max().sqrt()
        self.velocity.copy_(velocity)
        self.acceleration.copy_(acceleration)
        self.valid.copy_(current_valid)
        self.acceleration_valid.copy_(pair)
        self.episode.copy_(episode)
        return result
