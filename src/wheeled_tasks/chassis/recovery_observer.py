"""Recovery observations reconstructed from the installed IMU and six encoders.

The mechanism table is a simulation asset. A real robot needs a separately
measured table and verified motor-to-output coordinates before using this path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


class MotorEncoderAlignment:
    """Four independent 1:1 DM-to-output axes, with an explicit encoder-zero pose."""

    LEGS = (0, 1, 3, 4)

    def __init__(self, model_at_encoder_zero, signs, device, count, dt, feedback_period=None,
                 position_resolution=None, velocity_resolution=None):
        zero = torch.as_tensor(model_at_encoder_zero, dtype=torch.float32, device=device)
        sign = torch.as_tensor(signs, dtype=torch.float32, device=device)
        if zero.shape != (4,) or not bool(torch.isfinite(zero).all()):
            raise ValueError("Four finite output angles are required at the DM encoder-zero pose")
        if sign.shape != (4,) or not bool(((sign == 1.) | (sign == -1.)).all()):
            raise ValueError("Four measured drive directions (+1 or -1) are required")
        if feedback_period is not None and feedback_period <= 0:
            raise ValueError("A measured positive feedback period is required")
        self.zero, self.sign = zero, sign
        self.device, self.count, self.dt = device, count, dt
        self.period = feedback_period
        self.position_resolution = position_resolution
        self.velocity_resolution = velocity_resolution
        self.previous_raw = torch.zeros(count, 4, device=device)
        self.continuous = torch.zeros(count, 4, device=device)
        self.initialized = False

    def reset(self):
        self.initialized = False

    def sample(self, simulated_q, simulated_dq):
        raw = (simulated_q[:, self.LEGS] - self.zero) * self.sign
        motor_dq = simulated_dq[:, self.LEGS] * self.sign
        if self.position_resolution is not None:
            raw = torch.round(raw / self.position_resolution) * self.position_resolution
        if self.velocity_resolution is not None:
            motor_dq = torch.round(motor_dq / self.velocity_resolution) * self.velocity_resolution
        if self.period is not None:
            raw = torch.remainder(raw + self.period / 2, self.period) - self.period / 2
        if not self.initialized:
            self.previous_raw = raw.clone()
            self.continuous = raw.clone()
            self.initialized = True
        else:
            step = raw - self.previous_raw
            if self.period is not None:
                step -= self.period * torch.round(step / self.period)
            self.continuous += step
            self.previous_raw = raw.clone()
        model_q = simulated_q.clone()
        model_dq = simulated_dq.clone()
        model_q[:, self.LEGS] = self.zero + self.sign * self.continuous
        model_dq[:, self.LEGS] = self.sign * motor_dq
        return model_q, model_dq, raw


@dataclass
class RecoveryObservation:
    height_if_wheels_grounded: torch.Tensor
    wheel_height_difference: torch.Tensor
    wheel_world_omega_y: torch.Tensor
    spring_positions: torch.Tensor
    knee_inner_deg: torch.Tensor
    geometry_valid: torch.Tensor
    height_rate: torch.Tensor
    wheel_speed_estimate: torch.Tensor
    impact: torch.Tensor
    alignment_candidate: torch.Tensor
    contact_candidate: torch.Tensor
    body_contact_suspected: torch.Tensor
    supported: torch.Tensor
    probe_confirmed: torch.Tensor
    settled: torch.Tensor


class SimulatedImu:
    """Generate an accelerometer sample; the controller never receives root velocity."""

    def __init__(self, device, count, dt, noise=False):
        self.previous_velocity = torch.zeros(count, 3, device=device)
        self.dt = dt
        self.noise = noise
        self.generator = torch.Generator(device=device).manual_seed(190619)
        self.gyro_bias = torch.zeros(count, 3, device=device)

    def reset(self, velocity):
        self.previous_velocity = velocity.clone()
        if self.noise:
            self.gyro_bias = torch.randn(velocity.shape, generator=self.generator, device=velocity.device) * .005

    def measure_attitude(self, gravity, gyro):
        if not self.noise:
            return gravity, gyro
        gyro = (gyro + self.gyro_bias
                + torch.randn(gyro.shape, generator=self.generator, device=gyro.device) * .01)
        gravity = gravity + torch.randn(gravity.shape, generator=self.generator, device=gravity.device) * .003
        return torch.nn.functional.normalize(gravity, dim=-1), gyro

    def sample(self, velocity_world, orientation):
        acceleration_world = (velocity_world - self.previous_velocity) / self.dt
        self.previous_velocity = velocity_world.clone()
        specific_force_world = acceleration_world + velocity_world.new_tensor([0., 0., 9.81])
        inverse = orientation.clone()
        inverse[:, :3] *= -1
        acceleration = RecoveryObserver.rotate(inverse, specific_force_world)
        if self.noise:
            acceleration += torch.randn(acceleration.shape, generator=self.generator,
                                        device=acceleration.device) * .15
        return acceleration


class WheelSupportProbe:
    """Small alternating wheel efforts with encoder/gyro response checks."""

    def __init__(self, device, count, dt):
        self.device, self.count, self.dt = device, count, dt
        self.previous_velocity = torch.zeros(count, 2, device=device)
        self.previous_gyro = torch.zeros(count, 3, device=device)
        self.previous_pulse = torch.zeros(count, 2, device=device)
        self.evidence = torch.zeros(count, 2, 2, dtype=torch.bool, device=device)
        self.age = torch.full((count, 2, 2), math.inf, device=device)
        self.bad_samples = torch.zeros(count, 2, 2, dtype=torch.long, device=device)
        self.quiet_time = torch.zeros(count, device=device)
        self.lost_time = torch.zeros(count, device=device)

    def reset(self):
        self.previous_velocity.zero_()
        self.previous_gyro.zero_()
        self.previous_pulse.zero_()
        self.evidence.zero_()
        self.age.fill_(math.inf)
        self.bad_samples.zero_()
        self.quiet_time.zero_()
        self.lost_time.zero_()

    def observe(self, wheel_velocity, gyro, plausible):
        acceleration = (wheel_velocity - self.previous_velocity) / self.dt
        angular_acceleration = (gyro - self.previous_gyro).norm(dim=-1) / self.dt
        valid = (self.previous_pulse.abs() >= .14) & plausible[:, None]
        loaded = valid & (acceleration.abs() < 100) & (angular_acceleration[:, None] < 90)
        self.quiet_time = torch.where(self.previous_pulse.abs().any(dim=1), 0., self.quiet_time + self.dt)
        self.age += self.dt
        for side in range(2):
            for direction in range(2):
                tested = valid[:, side] & ((self.previous_pulse[:, side] > 0) == bool(direction))
                self.bad_samples[:, side, direction] = torch.where(
                    tested & ~loaded[:, side], self.bad_samples[:, side, direction] + 1,
                    torch.where(tested, 0, self.bad_samples[:, side, direction]))
                self.evidence[:, side, direction] &= self.bad_samples[:, side, direction] < 2
                self.evidence[:, side, direction] |= tested & loaded[:, side]
                self.age[:, side, direction] = torch.where(
                    tested & loaded[:, side], 0., self.age[:, side, direction])
        self.previous_velocity = wheel_velocity.clone()
        self.previous_gyro = gyro.clone()
        self.previous_pulse.zero_()
        self.lost_time = torch.where(plausible, 0., self.lost_time + self.dt)
        self.evidence &= (self.lost_time < .05)[:, None, None]
        return plausible & (self.evidence & (self.age < .6)).all(dim=(1, 2))

    def command(self, tick, eligible, confirmed):
        pulse = torch.zeros(self.count, 2, device=self.device)
        stage = (tick // 3) % 4
        pulse[:, stage // 2] = .18 if stage % 2 == 0 else -.18
        pulse *= (eligible & ~confirmed)[:, None]
        self.previous_pulse = pulse.clone()
        return pulse


class RecoveryObserver:
    def __init__(self, spec, reference, device, count, dt):
        from v5_mechanism import fk

        self.device, self.count, self.dt = device, count, dt
        self.sides = []
        bodies = {body["name"]: body for body in spec["bodies"]}
        joints = {joint["name"]: joint for joint in spec["joints"]}
        theta0 = math.pi - 2.3573
        for side, hip_name, knee_name, wheel_name in (
            ("left", "L_joint1", "L_joint2", "L_link3"),
            ("right", "R_joint1", "R_jonit2", "R_link3"),
        ):
            table = reference["spring_lookup"][side]
            hip = joints[hip_name]
            hip_origin = np.asarray(hip["origin"])[:3, 3]
            hip_axis = np.asarray(hip["origin"])[:3, :3] @ np.asarray(hip["axis"])
            points, knee_axes, wheel_axes = [], [], []
            for inner in table["inner_knee_deg"]:
                knee_q = math.radians(inner) - theta0 if side == "left" else theta0 - math.radians(inner)
                frames = fk(spec, {knee_name: knee_q})
                wheel = bodies[wheel_name]["collisions"][0]
                wheel_frame = frames[wheel_name] @ np.asarray(wheel["origin"])
                points.append(wheel_frame[:3, 3])
                knee = joints[knee_name]
                knee_axes.append(frames[knee["parent"]][:3, :3] @ np.asarray(knee["origin"])[:3, :3] @ np.asarray(knee["axis"]))
                wheel_joint = joints["L_joint3" if side == "left" else "R_joint3"]
                wheel_axes.append(frames[wheel_joint["parent"]][:3, :3]
                                  @ np.asarray(wheel_joint["origin"])[:3, :3]
                                  @ np.asarray(wheel_joint["axis"]))
            tensor = lambda data: torch.as_tensor(data, dtype=torch.float32, device=device)
            self.sides.append({
                "delta": tensor(table["delta_rad"]),
                "inner": tensor(table["inner_knee_deg"]),
                "slider": tensor(table["slider_q_m"]),
                "point": tensor(np.stack(points)),
                "knee_axis": tensor(np.stack(knee_axes)),
                "wheel_axis": tensor(np.stack(wheel_axes)),
                "hip_origin": tensor(hip_origin),
                "hip_axis": tensor(hip_axis),
                "sign": 1.0 if side == "left" else -1.0,
            })
        self.last_height = torch.zeros(count, device=device)
        self.height_rate = torch.zeros(count, device=device)
        self.last_omega = torch.zeros(count, 3, device=device)
        self.support_time = torch.zeros(count, device=device)
        self.impact_age = torch.full((count,), math.inf, device=device)
        self.alignment_time = torch.zeros(count, device=device)
        self.probe = WheelSupportProbe(device, count, dt)

    def reset(self):
        self.last_height.zero_()
        self.height_rate.zero_()
        self.last_omega.zero_()
        self.support_time.zero_()
        self.impact_age.fill_(math.inf)
        self.alignment_time.zero_()
        self.probe.reset()

    @staticmethod
    def rotate(quaternion, vector):
        xyz = quaternion[..., :3]
        t = 2 * torch.linalg.cross(xyz, vector)
        return vector + quaternion[..., 3:] * t + torch.linalg.cross(xyz, t)

    def _interpolate(self, side, delta):
        knots = side["delta"]
        upper = torch.searchsorted(knots, delta.contiguous()).clamp(1, len(knots) - 1)
        lower = upper - 1
        amount = ((delta - knots[lower]) / (knots[upper] - knots[lower])).clamp(0, 1)

        def sample(name):
            table = side[name]
            shape = (self.count,) + (1,) * (table.ndim - 1)
            blend = amount.reshape(shape)
            return table[lower] + blend * (table[upper] - table[lower])

        slope = (side["inner"][upper] - side["inner"][lower]) / (knots[upper] - knots[lower])
        return sample("inner"), sample("slider"), sample("point"), sample("knee_axis"), sample("wheel_axis"), slope

    def update(self, q, dq, gravity, omega, orientation, acceleration=None):
        wheel_points, wheel_omegas, spring_positions, knee_angles = [], [], [], []
        geometry_valid = torch.ones(self.count, dtype=torch.bool, device=self.device)
        for index, side in enumerate(self.sides):
            hip, auxiliary, wheel = (0, 1, 2) if index == 0 else (3, 4, 5)
            delta = q[:, auxiliary] - q[:, hip]
            geometry_valid &= (delta >= side["delta"][0] - .002) & (delta <= side["delta"][-1] + .002)
            inner, slider, point, knee_axis, wheel_axis, slope = self._interpolate(side, delta)
            knee_angles.append(inner)
            spring_positions.append(slider)
            axis = side["hip_axis"].expand(self.count, -1)
            origin = side["hip_origin"]
            cosine, sine = torch.cos(q[:, hip])[:, None], torch.sin(q[:, hip])[:, None]

            def rotate_joint(v):
                return cosine * v + sine * torch.linalg.cross(axis, v) + (1 - cosine) * axis * (axis * v).sum(-1, keepdim=True)

            wheel_points.append(origin + rotate_joint(point - origin))
            passive_speed = side["sign"] * slope * (dq[:, auxiliary] - dq[:, hip]) * math.pi / 180
            wheel_omegas.append(omega + axis * dq[:, hip, None]
                                 + rotate_joint(knee_axis) * passive_speed[:, None]
                                 + rotate_joint(wheel_axis) * dq[:, wheel, None])
        wheel_points = torch.stack(wheel_points, dim=1)
        grounded_height = .06 + (wheel_points * gravity[:, None, :]).sum(-1)
        height = grounded_height.mean(-1)
        relative_height = -(gravity * (wheel_points[:, 0] - wheel_points[:, 1])).sum(-1)
        wheel_omegas = torch.stack(wheel_omegas, dim=1)
        world_omega_y = self.rotate(orientation[:, None, :].expand(-1, 2, -1), wheel_omegas)[:, :, 1]
        raw_height_rate = (height - self.last_height) / self.dt
        self.height_rate += (raw_height_rate - self.height_rate) * (self.dt / (.05 + self.dt))
        height_rate = self.height_rate
        self.last_height = height.clone()
        gyro_acceleration = (omega - self.last_omega) / self.dt
        self.last_omega = omega.clone()
        tilt = torch.acos((-gravity[:, 2]).clamp(-1, 1))
        if acceleration is None:
            impact = gyro_acceleration.norm(dim=-1) > 20
        else:
            impact = ((acceleration.norm(dim=-1) > 16)
                      | ((tilt > math.radians(20)) & (gyro_acceleration.norm(dim=-1) > 20)))
        self.impact_age = torch.where(impact, 0., self.impact_age + self.dt)
        wheel_velocity = dq[:, [2, 5]]
        planar_velocity = .06 * (wheel_velocity[:, 0] - wheel_velocity[:, 1]) * .5
        plausible_alignment = (geometry_valid & (height > .20) & (height < .43)
                               & (relative_height.abs() < .03) & (tilt < math.radians(65)))
        if acceleration is not None:
            plausible_alignment &= acceleration.norm(dim=-1) > 6.
        self.alignment_time = torch.where(
            plausible_alignment, (self.alignment_time + self.dt).clamp_max(.2),
            (self.alignment_time - 2 * self.dt).clamp_min(0.))
        alignment_candidate = (geometry_valid & (self.alignment_time >= .03))
        if acceleration is not None:
            alignment_candidate &= acceleration.norm(dim=-1) > 3.
        supported = (geometry_valid & (height > .27) & (height < .36)
                     & (relative_height.abs() < .03) & (tilt < math.radians(20))
                     & (omega.norm(dim=-1) < 1.5)
                     & (dq[:, [0, 1, 3, 4]].abs().amax(-1) < 6)
                     & (wheel_velocity.abs().amax(-1) < 10)
                     & ~impact)
        probe_confirmed = self.probe.observe(wheel_velocity, omega, supported)
        contact_candidate = (supported | probe_confirmed | (geometry_valid & (height > .12)
                           & (tilt < math.radians(145)) & (self.impact_age < .15)))
        body_contact_suspected = geometry_valid & (height < .27) & (tilt < math.radians(65))
        self.support_time = torch.where(supported, self.support_time + self.dt, 0.)
        settled = (supported & probe_confirmed & (self.probe.quiet_time >= .05)
                   & (self.support_time >= .10)
                   & (tilt < math.radians(8)) & (omega.norm(dim=-1) < .75)
                   & (planar_velocity.abs() < .25) & (height_rate.abs() < .15))
        return RecoveryObservation(
            height, relative_height, world_omega_y, torch.stack(spring_positions, dim=1),
            torch.stack(knee_angles, dim=1), geometry_valid, height_rate, planar_velocity,
            impact, alignment_candidate, contact_candidate, body_contact_suspected,
            supported, probe_confirmed, settled)
