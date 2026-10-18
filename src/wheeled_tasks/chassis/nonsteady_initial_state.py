"""Source flat birth requests, before mapping to this chassis's mechanism.

Reference: pinned reference b8ff79f3, 0719 env.yaml and wheelbipe25_v3
_custom_reset_random. The source has independent left/right virtual lengths
and angles, ground births from probability .3+.2, and nonzero active joint
velocities on *all* births. This module samples those requests; it does not
copy the source IK, its root heights, or its joint zeros into our articulation.

The requests need an explicitly reviewed full-chain mechanical adapter before
any native write. A source virtual foot height is L*cos(angle), not a world
wheel-bottom clearance. Source +/-50 rad/s is wheel-output velocity. All wheel
requests here use output units; this sampler performs no gearbox conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


RESET_VERSION = "scut0719-flat-birth-requests-v1"
CONTROL_JOINT_NAMES = ("L_joint1", "LL_joint1", "L_joint3",
                       "R_joint1", "RR_joint1", "R_joint3")


@dataclass(frozen=True)
class FlatBirthRequests:
    env_ids: torch.Tensor
    ground_birth: torch.Tensor
    # -1 is ordinary, 0 positive-angle ground, 1 negative-angle ground.
    leg_mode: torch.Tensor
    virtual_leg_length_m: torch.Tensor
    virtual_leg_angle_rad: torch.Tensor
    virtual_foot_height_m: torch.Tensor
    wheel_angle_rad: torch.Tensor
    control_velocity_rad_s: torch.Tensor
    source_root_position_m: torch.Tensor
    source_root_rpy_rad: torch.Tensor
    source_root_velocity6: torch.Tensor

    def diagnostics(self):
        """Return detached snapshots; none of these are accepted native states."""
        return {name: value.clone() for name, value in vars(self).items()}


class FlatBirthSampler:
    """Distribution-level source reset sampler with caller-owned environment RNG.

    Own RNG ordering is versioned, not historical bitwise reproduction: source
    reset also consumes RNG in command, DR and SDK events. There is no ordinary
    nominal-birth fallback and no filtering/clipping of requested source lengths.
    Root velocities are zero while the six active joint velocities are random.
    """
    def __init__(self, num_envs, device):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("Birth sampler requires a positive environment count")
        self.num_envs = num_envs
        self.device = torch.empty(0, device=device).device

    def _ids(self, ids):
        if not isinstance(ids, torch.Tensor):
            if not isinstance(ids, (list, tuple)) or any(type(value) is not int for value in ids):
                raise ValueError("Birth IDs require explicit integer lane IDs")
            ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if (ids.dtype != torch.long or ids.ndim != 1 or ids.device != self.device
                or ids.unique().numel() != ids.numel()
                or bool(((ids < 0) | (ids >= self.num_envs)).any())):
            raise ValueError("Birth IDs must be unique, in-range long[K] on the sampler device")
        return ids

    def _uniform(self, shape, low, high, generator):
        return torch.empty(shape, dtype=torch.float32, device=self.device).uniform_(low, high, generator=generator)

    def sample(self, ids, generator):
        ids = self._ids(ids)
        if not isinstance(generator, torch.Generator) or torch.device(generator.device) != self.device:
            raise ValueError("Birth sampling needs a torch.Generator on the sampler device")
        count = ids.numel()
        # Reset event orientation is overwritten only on ground births. Position
        # values are source-relative requests, never this chassis's native pose.
        rpy = torch.empty(count, 3, dtype=torch.float32, device=self.device)
        rpy[:, :2] = self._uniform((count, 2), -.15, .15, generator)
        rpy[:, 2] = self._uniform((count,), -3.14, 3.14, generator)
        position = torch.zeros(count, 3, dtype=torch.float32, device=self.device)
        position[:, 2] = .38
        length = self._uniform((count, 2), .13, .32, generator)
        angle = self._uniform((count, 2), -math.pi / 2, 3 * math.pi / 4, generator)
        ground = self._uniform((count,), 0., 1., generator) < .5
        mode = torch.full((count, 2), -1, dtype=torch.int8, device=self.device)
        ground_ids = ground.nonzero(as_tuple=False).flatten()
        n = ground_ids.numel()
        if n:
            # Archive modes are .3 and .2; conditional leg choices are .6/.4.
            # Each leg is sampled independently, not one shared pose class.
            draws = self._uniform((n, 2), 0., .5, generator)
            ground_mode = (draws >= .3).to(torch.int8)
            ground_length = torch.empty(n, 2, dtype=torch.float32, device=self.device)
            ground_angle = torch.empty_like(ground_length)
            for index, (sign, lower, upper) in enumerate(((1., -.06, .12), (-1., -.06, 0.))):
                selected = ground_mode == index
                selected_count = int(selected.sum())
                height = self._uniform((selected_count,), lower, upper, generator)
                sampled_length = self._uniform((selected_count,), .14, .36, generator)
                # This source clamp is inert for the archived h/L domain. It is
                # not a mechanical reachability clamp for our chassis.
                ratio = (height / sampled_length.clamp_min(1e-6)).clamp(-1. + 1e-6, 1. - 1e-6)
                ground_length[selected] = sampled_length
                ground_angle[selected] = sign * torch.acos(ratio)
            length[ground_ids] = ground_length
            angle[ground_ids] = ground_angle
            mode[ground_ids] = ground_mode
            position[ground_ids, 2] = .25
            rpy[ground_ids] = 0.
        wheel = self._uniform((count, 2), -2 * math.pi, 2 * math.pi, generator)
        velocity = torch.empty(count, 6, dtype=torch.float32, device=self.device)
        velocity[:, [0, 1, 3, 4]] = self._uniform((count, 4), -math.pi / 2, math.pi / 2, generator)
        velocity[:, [2, 5]] = self._uniform((count, 2), -50., 50., generator)
        return FlatBirthRequests(ids.clone(), ground, mode, length, angle,
            length * angle.cos(), wheel, velocity, position, rpy,
            torch.zeros(count, 6, dtype=torch.float32, device=self.device))

    @staticmethod
    def summary():
        return {"reset_version": RESET_VERSION,
                "pinned_source_commit": "b8ff79f3df855faf9dc92f4a282bd80c42649466",
                "ground_probability": .5, "conditional_positive_probability_per_leg": .6,
                "ordinary_length_range_m": [.13, .32], "ground_length_range_m": [.14, .36],
                "leg_active_velocity_range_rad_s": [-math.pi / 2, math.pi / 2],
                "wheel_output_velocity_range_rad_s": [-50., 50.],
                "control_joint_names": list(CONTROL_JOINT_NAMES),
                "source_root_pose_is_native_pose": False,
                "own_mechanical_mapping_required": True,
                "runtime_integration_verified": False, "native_qualified": False,
                "historical_bitwise_replay": False}
