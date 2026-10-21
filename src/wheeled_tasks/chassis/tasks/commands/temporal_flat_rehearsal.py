"""Static lane requests for drive, stop, reverse and stop in one episode."""
from copy import deepcopy
import hashlib

import torch


class TemporalFlatRequests:
    """Use the existing integer episode clock; add no RNG or mutable timer."""

    def __init__(self, num_envs, device, ids, requests, phase_policy_ticks, episode_policy_ticks):
        if (not isinstance(requests, torch.Tensor) or requests.dtype != torch.float32
                or requests.device != device or requests.shape != (ids.numel(), 3)
                or not ids.numel() or not bool(torch.isfinite(requests).all())
                or bool((requests[:, 0].abs() < .1).any())
                or bool((requests[:, 0].abs() > 3.).any())
                or bool((requests[:, 1] != 0.).any())
                or bool(((requests[:, 2] < .20) | (requests[:, 2] > .42)).any())):
            raise ValueError("Temporal requests require signed drive speed, zero yaw and a reviewed flat height")
        if (not isinstance(phase_policy_ticks, (list, tuple)) or len(phase_policy_ticks) != 4
                or any(type(value) is not int or value <= 0 for value in phase_policy_ticks)
                or type(episode_policy_ticks) is not int or episode_policy_ticks <= 0
                or sum(phase_policy_ticks) > episode_policy_ticks):
            raise ValueError("All four temporal phases must fit inside the actual episode horizon")
        self.ids = ids.clone()
        self.mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.mask[ids] = True
        self.requests = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.requests[ids] = requests
        self.phase_ends = torch.tensor(phase_policy_ticks, dtype=torch.long, device=device).cumsum(0) * 4
        self.sign = torch.tensor([1., 0., -1., 0.], dtype=torch.float32, device=device)
        self.identity = {
            "version": "v7-temporal-flat-requests-v1", "lanes": ids.numel(),
            "lane_ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            "requests_sha256": hashlib.sha256(requests.cpu().numpy().tobytes()).hexdigest(),
            "request_columns": ["body_vx_m_s", "body_yaw_rad_s", "root_height_m"],
            "phase_names": ["drive", "stop", "reverse", "final_stop"],
            "phase_policy_ticks": list(phase_policy_ticks),
            "episode_policy_ticks": episode_policy_ticks,
            "clock": "existing episode 5ms ticks; boundaries aligned to 20ms actor intervals",
            "request_behavior": "instant request changes; final zero held until the real episode ends",
            "scope": "command-only local adaptation; no reset, command slew, reward change or new PPO mask",
            "birth_owner": "caller must allocate nominal lanes; this component checks only ground-override conflicts",
        }

    def phases(self, episode_physics_ticks):
        return (episode_physics_ticks[:, None] >= self.phase_ends[:-1]).sum(-1)

    def validate_reset(self, ids, ground):
        if bool((self.mask[ids] & ground).any()):
            raise ValueError("Temporal rehearsal requires nominal births without the source ground override")

    def publish(self, owner, ids):
        selected = ids[self.mask[ids]]
        phase = self.phases(owner.episode_physics_ticks[selected])
        command = torch.zeros(selected.numel(), 3, dtype=torch.float32, device=owner.device)
        command[:, 0] = self.requests[selected, 0] * self.sign[phase]
        owner.generator_command3[selected] = command
        owner.command3[selected] = command
        owner.height[selected] = self.requests[selected, 2]

    def published_phases(self, owner):
        # At payment, physics advanced four ticks but advance() has not yet
        # published the next command. After reset/advance this offset is zero.
        unpublished = owner.physics_steps - owner._last_command_physics_step
        ticks = (owner.episode_physics_ticks - unpublished).clamp_min(0)
        return torch.where(self.mask, self.phases(ticks), -1)

    def summary(self):
        return deepcopy(self.identity)
