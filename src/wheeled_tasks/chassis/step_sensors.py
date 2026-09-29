"""Simulation adapters for the signals available to a preset step-jump controller.

World velocity and terrain enter only here to synthesize an accelerometer and a
single beam. The controller receives no position, contact or terrain-height truth.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch

from .mechanism_observer import ClosedChainHeightModel
from .recovery_observer import SimulatedImu
from .step_trigger import estimate_height_if_grounded, single_beam_range_2d


class StepTaskSensors:
    def __init__(self, count, device, dt, settings, root, seed, *, evaluation=False):
        path = root / settings["height_lookup_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != settings["height_lookup_sha256"]:
            raise ValueError("Encoder height lookup differs from frozen contract")
        self.model = ClosedChainHeightModel(json.loads(path.read_text()), device)
        self.cfg, self.evaluation = settings, evaluation
        self.imu = SimulatedImu(device, count, dt)
        self.generator = torch.Generator(device=device).manual_seed(seed + 47231)
        self.force_norm = torch.full((count,), 9.81, device=device)
        self.support = torch.zeros(count, dtype=torch.bool, device=device)
        self.height = torch.zeros(count, 2, device=device)
        self.height_valid = torch.zeros(count, 2, dtype=torch.bool, device=device)
        self.range = torch.zeros(count, device=device)
        self.forward_distance = torch.zeros(count, device=device)
        self.range_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.sample_time = torch.full((count,), -torch.inf, device=device)
        self.last_time = torch.full((count,), -torch.inf, device=device)
        self.range_age = torch.full((count,), torch.inf, device=device)

    def reset(self, ids, velocity_world):
        self.imu.previous_velocity[ids] = velocity_world[ids]
        self.force_norm[ids] = 9.81
        self.last_time[ids] = self.sample_time[ids] = -torch.inf
        self.range_valid[ids] = self.support[ids] = False

    def update(self, time, q, dq, omega, gravity, pose, velocity_world,
               local_position, surfaces, surface_valid):
        advanced = time > self.last_time
        # Repeated observation/reset reads must not turn a dynamic IMU sample
        # into an artificial zero-acceleration support estimate.
        previous = self.imu.previous_velocity.clone()
        force = self.imu.sample(velocity_world, pose[:, 3:])
        self.imu.previous_velocity[~advanced] = previous[~advanced]
        self.force_norm = torch.where(advanced, force.norm(dim=-1), self.force_norm)
        self.last_time.copy_(time)
        points, valid = self.model.wheel_positions(q)
        candidate = ((self.force_norm >= 6.) & (gravity[:, 2] < -math.cos(math.radians(30)))
                     & (omega.norm(dim=-1) < 2.))
        self.height, self.height_valid = estimate_height_if_grounded(
            points, gravity, self.model.wheel_radius_m,
            grounded=candidate[:, None].expand(-1, 2), sensor_valid=valid,
            wheel_axle_b=self.model.wheel_axles_b)
        self.support = (self.height_valid.all(-1) & (self.height.amin(-1) > .08)
                        & (self.height.amax(-1) < .65)
                        & ((self.height[:, 0] - self.height[:, 1]).abs() < .12))
        due = time - self.sample_time >= self.cfg["range_sample_period_s"] - 1e-6
        if bool(due.any()):
            pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
            value, hit = single_beam_range_2d(local_position[:, [0, 2]], pitch,
                surfaces, surface_valid, sensor_origin_b=self.cfg["sensor_origin_b_m"],
                sensor_pitch_rad=self.cfg["sensor_pitch_rad"], max_range_m=self.cfg["range_max_m"])
            # The 2D corridor sensor is valid only while aligned with its lane.
            x, y, z, w = pose[:, 3:].unbind(-1)
            heading_x, heading_y = 1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)
            hit &= (heading_x > 0) & (heading_y.abs() < .15) & (local_position[:, 1].abs() < 1.5)
            if not self.evaluation:
                value += torch.randn(value.shape, generator=self.generator, device=value.device) * self.cfg["range_noise_std_m"]
                hit &= torch.rand(value.shape, generator=self.generator, device=value.device) >= self.cfg["range_dropout_probability"]
            hit &= value > .015
            angle = self.cfg["sensor_pitch_rad"]
            endpoint_b = torch.stack((self.cfg["sensor_origin_b_m"][0] + value * math.cos(angle),
                torch.zeros_like(value), self.cfg["sensor_origin_b_m"][1] - value * math.sin(angle)), -1)
            # A downward-looking beam can hit the floor before a low step.
            # Reject it using encoder/IMU geometry, never the raycast surface ID.
            above_support = self.height - (gravity * endpoint_b).sum(-1, keepdim=True)
            hit &= self.height_valid.all(-1) & (above_support.amin(-1) > self.cfg["ground_echo_margin_m"])
            forward = pitch.cos() * endpoint_b[:, 0] + pitch.sin() * endpoint_b[:, 2]
            self.range[due], self.range_valid[due] = value[due], hit[due]
            self.forward_distance[due] = forward[due]
            self.sample_time[due] = time[due]
        self.range_age = time - self.sample_time
        return .03 * (dq[:, 2] - dq[:, 5])

    def diagnostics(self):
        return {"encoder_height_if_grounded": self.height.clone(),
                "encoder_height_valid": self.height_valid.clone(),
                "step_support_estimate": self.support.clone(),
                "step_specific_force_norm": self.force_norm.clone()}
