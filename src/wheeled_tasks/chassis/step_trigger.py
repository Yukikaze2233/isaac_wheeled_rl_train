"""Single-beam step requests and conditional encoder/IMU height estimates.

The deployable trigger consumes only range status, command intent and explicitly
provided completion/new-obstacle confirmations. The ray function is a simulator
sensor adapter: it returns one nearest range, never terrain heights to policy.
"""
from __future__ import annotations

import math

import torch


class SingleBeamStepTrigger:
    """Latch one upward-step request until explicit completion and rearming."""

    def __init__(self, count, device, trigger_distance_m, rearm_distance_m,
                 forward_speed_min_m_s=.05, max_sample_age_s=.05):
        values = (trigger_distance_m, rearm_distance_m,
                  forward_speed_min_m_s, max_sample_age_s)
        if (type(count) is not int or count <= 0
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or not 0 < trigger_distance_m < rearm_distance_m
                or forward_speed_min_m_s < 0 or max_sample_age_s < 0):
            raise ValueError("Invalid single-beam trigger thresholds")
        self.count, self.device = count, torch.device(device)
        self.trigger_distance_m, self.rearm_distance_m = trigger_distance_m, rearm_distance_m
        self.forward_speed_min_m_s = forward_speed_min_m_s
        self.max_sample_age_s = max_sample_age_s
        self.busy = torch.zeros(count, dtype=torch.bool, device=device)
        self.armed = torch.ones_like(self.busy)
        self.completion_seen = torch.zeros_like(self.busy)
        self.height_m = torch.zeros(count, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.busy[ids] = False
        self.armed[ids] = True
        self.completion_seen[ids] = False
        self.height_m[ids] = 0.

    def update(self, range_m, range_valid, range_age_s, forward_command_m_s,
               preset_height_m, *, completed=None, new_step_confirmed=None):
        """Return a one-tick event; absent echoes never complete or rearm a step.

        A completion received while busy can rearm for a later call. It never
        emits another request in that same call, even with a new-step signal.
        """
        tensors = (range_m, range_valid, range_age_s, forward_command_m_s, preset_height_m)
        if any(value.shape != (self.count,) or value.device != self.device for value in tensors):
            raise ValueError("Step trigger inputs must have one value per environment on its device")
        if range_valid.dtype != torch.bool:
            raise ValueError("Range validity must be boolean")
        completed = torch.zeros_like(self.busy) if completed is None else completed
        new_step_confirmed = torch.zeros_like(self.busy) if new_step_confirmed is None else new_step_confirmed
        if any(value.shape != self.busy.shape or value.device != self.device or value.dtype != torch.bool
               for value in (completed, new_step_confirmed)):
            raise ValueError("Completion and new-step confirmations must be boolean per environment")
        usable = (range_valid & torch.isfinite(range_m) & (range_m > 0)
                  & torch.isfinite(range_age_s) & (range_age_s >= 0)
                  & (range_age_s <= self.max_sample_age_s))
        was_busy = self.busy.clone()
        finished = was_busy & completed
        self.busy &= ~finished
        self.completion_seen |= finished
        clear = usable & (range_m >= self.rearm_distance_m)
        self.armed |= ~self.busy & self.completion_seen & (clear | new_step_confirmed)
        event = (self.armed & ~self.busy & ~was_busy & usable
                 & (range_m <= self.trigger_distance_m)
                 & torch.isfinite(forward_command_m_s)
                 & (forward_command_m_s > self.forward_speed_min_m_s)
                 & torch.isfinite(preset_height_m) & (preset_height_m > 0))
        self.height_m = torch.where(event, preset_height_m.to(self.height_m.dtype), self.height_m)
        self.busy |= event
        self.armed &= ~event
        self.completion_seen &= ~event
        return event


def estimate_height_if_grounded(wheel_positions_b, unit_gravity_b, wheel_radius_m,
                               *, grounded, sensor_valid=None, wheel_axle_b=(0., 1., 0.)):
    """Return separate body-height candidates and validity for the two wheels.

    Wheel centers come from encoder FK, gravity from IMU, and ``grounded`` is an
    external support assumption/estimate. FK alone cannot detect support. The
    circular-wheel radius is projected perpendicular to its body-frame axle;
    default axles are lateral (+Y). No left/right averaging is performed.
    """
    if (wheel_positions_b.ndim != 3 or wheel_positions_b.shape[1:] != (2, 3)
            or unit_gravity_b.shape != (len(wheel_positions_b), 3)
            or grounded.shape != wheel_positions_b.shape[:2] or grounded.dtype != torch.bool
            or unit_gravity_b.device != wheel_positions_b.device or grounded.device != wheel_positions_b.device):
        raise ValueError("Expected wheel centers [N,2,3], gravity [N,3], grounded [N,2]")
    gravity = unit_gravity_b[:, None, :]
    radius = torch.as_tensor(wheel_radius_m, device=wheel_positions_b.device, dtype=wheel_positions_b.dtype)
    axle = torch.as_tensor(wheel_axle_b, device=wheel_positions_b.device, dtype=wheel_positions_b.dtype)
    if radius.shape not in ((), (2,), wheel_positions_b.shape[:2]) or axle.shape not in ((3,), (2, 3), wheel_positions_b.shape):
        raise ValueError("Radius or wheel axle shape cannot broadcast per side")
    gravity_ok = torch.isfinite(gravity).all(-1) & ((gravity.square().sum(-1) - 1).abs() <= 1e-3)
    axle_ok = torch.isfinite(axle).all(-1) & ((axle.square().sum(-1) - 1).abs() <= 1e-3)
    geometry_valid = (torch.isfinite(wheel_positions_b).all(-1) & gravity_ok & axle_ok
                      & torch.isfinite(radius) & (radius > 0))
    radial_drop = radius * (1 - (gravity * axle).sum(-1).square()).clamp_min(0).sqrt()
    height = (wheel_positions_b * gravity).sum(-1) + radial_drop
    valid = geometry_valid & grounded & (height > 0)
    if sensor_valid is not None:
        if sensor_valid.shape != valid.shape or sensor_valid.dtype != torch.bool or sensor_valid.device != valid.device:
            raise ValueError("FK/IMU sensor validity must be boolean [N,2]")
        valid &= sensor_valid
    return torch.where(geometry_valid, height, 0.), valid


def single_beam_range_2d(body_position_xz, pitch_rad, surface_data, surface_valid, *,
                         sensor_origin_b=(0., 0.), sensor_pitch_rad=0.,
                         max_range_m=4., min_range_m=0.):
    """Raycast one beam against an X/Z support profile, including exposed risers.

    Pitch is a right-handed +Y rotation: positive pitch tilts body +X downward.
    Extrinsics are body-local (x,z) and a relative +Y sensor rotation. Surfaces
    contain x0,x1,z0,slope in their first four fields and are ordered along X;
    any cross-slope field describes a different Y slice and is not a height scan.
    No hit returns range zero with hit=False, never a descent/completion event.
    """
    count = len(body_position_xz)
    if (body_position_xz.shape != (count, 2) or pitch_rad.shape != (count,)
            or surface_data.ndim != 3 or surface_data.shape[0] != count or surface_data.shape[2] < 4
            or surface_data.shape[1] == 0 or surface_valid.shape != surface_data.shape[:2]
            or surface_valid.dtype != torch.bool
            or any(value.device != body_position_xz.device for value in (pitch_rad, surface_data, surface_valid))):
        raise ValueError("Expected body [N,2], pitch [N], surfaces [N,S,>=4], validity [N,S]")
    if not all(math.isfinite(value) for value in (min_range_m, max_range_m)) or not 0 <= min_range_m < max_range_m:
        raise ValueError("Invalid single-beam range limits")
    origin = torch.as_tensor(sensor_origin_b, device=body_position_xz.device, dtype=body_position_xz.dtype)
    angle = torch.as_tensor(sensor_pitch_rad, device=body_position_xz.device, dtype=body_position_xz.dtype)
    if origin.shape not in ((2,), (count, 2)) or angle.shape not in ((), (count,)):
        raise ValueError("Sensor extrinsics must broadcast per environment")
    c, s = pitch_rad.cos(), pitch_rad.sin()
    ox = body_position_xz[:, 0] + c * origin[..., 0] + s * origin[..., 1]
    oz = body_position_xz[:, 1] - s * origin[..., 0] + c * origin[..., 1]
    dx, dz = (pitch_rad + angle).cos(), -(pitch_rad + angle).sin()
    x0, x1, z0, slope = surface_data[..., :4].unbind(-1)
    finite = surface_valid & torch.isfinite(surface_data[..., :4]).all(-1) & (x1 > x0)
    denominator = dz[:, None] - slope * dx[:, None]
    nonparallel = denominator.abs() > 1e-8
    t = (z0 + slope * (ox[:, None] - x0) - oz[:, None]) / torch.where(nonparallel, denominator, 1.)
    x = ox[:, None] + t * dx[:, None]
    top_hit = finite & nonparallel & (x >= x0) & (x <= x1) & (t >= min_range_m) & (t <= max_range_m)
    distances = torch.where(top_hit, t, torch.inf)
    if surface_data.shape[1] > 1:
        boundary = x0[:, 1:]
        previous_top = z0[:, :-1] + slope[:, :-1] * (boundary - x0[:, :-1])
        current_top = z0[:, 1:]
        vertical_t = (boundary - ox[:, None]) / torch.where(dx.abs() > 1e-8, dx, 1.)[:, None]
        z = oz[:, None] + vertical_t * dz[:, None]
        vertical_hit = (finite[:, 1:] & finite[:, :-1] & ((x1[:, :-1] - boundary).abs() <= 1e-6)
            & ((previous_top - current_top).abs() > 1e-8) & (dx.abs() > 1e-8)[:, None]
            & (z >= torch.minimum(previous_top, current_top)) & (z <= torch.maximum(previous_top, current_top))
            & (vertical_t >= min_range_m) & (vertical_t <= max_range_m))
        distances = torch.cat((distances, torch.where(vertical_hit, vertical_t, torch.inf)), dim=1)
    distance = distances.amin(-1)
    hit = torch.isfinite(distance) & torch.isfinite(body_position_xz).all(-1) & torch.isfinite(pitch_rad)
    return torch.where(hit, distance, 0.), hit
