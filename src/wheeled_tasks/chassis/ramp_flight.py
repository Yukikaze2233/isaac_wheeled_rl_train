"""Local ramp geometry and physical evidence for a requested lip jump.

This observer does not issue commands or change the policy's contact phase.
It requires a native accepted JUMP pulse and verifies the physical route.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from .task import Surface


@dataclass(frozen=True)
class RampFlightGeometry:
    """Metres in a local route frame; the launch lip is at signed x = 0."""

    direction: int = 1
    entry_height_m: float = .203
    lip_height_m: float = .553
    ramp_run_m: float = .350 / math.tan(math.radians(17.))
    ramp_width_m: float = .860
    gap_m: float = .650123
    landing_height_m: float = .200
    landing_width_m: float = 1.150
    landing_length_m: float = 1.500
    approach_length_m: float = 2.
    floor_width_m: float = 4.

    def __post_init__(self):
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError("Ramp direction must be +1 or -1")
        for field in fields(self):
            if field.name == "direction":
                continue
            value = getattr(self, field.name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Ramp {field.name} must be finite and positive")
        if not self.entry_height_m < self.lip_height_m:
            raise ValueError("Ramp lip must be higher than its entry")
        if self.landing_height_m >= self.lip_height_m:
            raise ValueError("Ramp landing platform must be below the lip")
        if self.floor_width_m < max(self.ramp_width_m, self.landing_width_m):
            raise ValueError("Ramp floor must cover the ramp and landing widths")

    @classmethod
    def from_limits(cls, limits, level=1.):
        """Use explicitly authored geometry; only gap width follows terrain level."""
        if type(level) not in (int, float) or not math.isfinite(level) or not 0 <= level <= 1:
            raise ValueError("Ramp terrain level must be in [0, 1]")
        data = limits.get("ramp_flight_geometry")
        if not isinstance(data, dict):
            raise ValueError("Ramp flight requires explicit ramp_flight_geometry")
        allowed = {field.name for field in fields(cls)} | {"gap_start_m"}
        if data.keys() - allowed:
            raise ValueError("Unknown ramp geometry fields")
        end = data.get("gap_m", cls.gap_m)
        start = data.get("gap_start_m", end)
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or not 0 < start <= end):
            raise ValueError("Ramp gap curriculum requires 0 < start <= end")
        values = {key: value for key, value in data.items() if key != "gap_start_m"}
        values["gap_m"] = start + level * (end - start)
        return cls(**values)

    @property
    def widths(self):
        return (self.floor_width_m, self.ramp_width_m, self.ramp_width_m, self.landing_width_m)

    def surfaces(self):
        """Floor, elevated approach, incline and opposite platform, in that order."""
        run, gap, length = self.ramp_run_m, self.gap_m, self.landing_length_m
        planes = [Surface(-run - self.approach_length_m - 1., gap + length + .5),
                  Surface(-run - self.approach_length_m, -run, self.entry_height_m),
                  Surface(-run, 0., self.entry_height_m,
                          (self.lip_height_m - self.entry_height_m) / run),
                  Surface(gap, gap + length, self.landing_height_m)]
        if self.direction == 1:
            return planes
        return [Surface(-surface.x1, -surface.x0, surface.height(surface.x1),
                        -surface.slope, surface.friction, surface.cross_slope)
                for surface in planes]

    def meshes(self):
        """Static solids with exact lip edges, avoiding a rotated box's overhang.

        The floor remains below the gap. Every mesh uses the native static
        triangle-mesh collision path; no visual mesh is needed for training.
        """
        meshes = []
        # Outward winding, including the sloping upper face. USD accepts quads.
        faces = (0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4,
                 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7)
        for surface, width in zip(self.surfaces(), self.widths):
            points = [(x, y, -.1) for x, y in
                      ((surface.x0, -width / 2), (surface.x1, -width / 2),
                       (surface.x1, width / 2), (surface.x0, width / 2))]
            points += [(x, y, surface.height(x)) for x, y, _ in points]
            meshes.append((points, faces))
        return meshes

    def support_height(self, xy):
        """Bound support by x and width; falling into the gap sees the floor."""
        result = torch.zeros(xy.shape[:-1], device=xy.device, dtype=xy.dtype)
        for surface, width in zip(self.surfaces(), self.widths):
            inside = ((xy[..., 0] >= surface.x0) & (xy[..., 0] <= surface.x1)
                      & (xy[..., 1].abs() <= width / 2))
            result = torch.maximum(result, torch.where(inside, surface.height(xy[..., 0]), 0.))
        return result


class RampFlightEvidence:
    """Physics-step route accounting, without request synthesis or action gates.

    Surface contact slots must match RampFlightGeometry.surfaces. The caller
    supplies fresh physical state, the native accepted JUMP event and a
    command-relative stability predicate. Reset at the route or jump reset.
    """

    def __init__(self, geometries, device, dt, *, stable_seconds=1., min_air_seconds=.06):
        if (any(type(value) not in (int, float) for value in (dt, stable_seconds, min_air_seconds))
                or not math.isfinite(dt) or dt <= 0 or not math.isfinite(stable_seconds)
                or stable_seconds < 1. or not math.isfinite(min_air_seconds) or min_air_seconds < .06):
            raise ValueError("Ramp evidence requires positive dt, >=1s stability and >=60ms air")
        self.geometries, self.dt = tuple(geometries), dt
        if any(item is not None and not isinstance(item, RampFlightGeometry) for item in self.geometries):
            raise ValueError("Ramp lanes require RampFlightGeometry or None")
        self.enabled = torch.tensor([item is not None for item in self.geometries], device=device)
        if not self.geometries or not bool(self.enabled.any()):
            raise ValueError("Ramp evidence requires at least one ramp lane")
        fallback = RampFlightGeometry()
        geometry = [item or fallback for item in self.geometries]
        self.direction = torch.tensor([item.direction for item in geometry], device=device)
        self.gap = torch.tensor([item.gap_m for item in geometry], device=device)
        self.platform_height = torch.tensor([item.landing_height_m for item in geometry], device=device)
        self.platform_end = self.gap + torch.tensor([item.landing_length_m for item in geometry], device=device)
        self.platform_half_width = torch.tensor([item.landing_width_m / 2 for item in geometry], device=device)
        self.ramp_half_width = torch.tensor([item.ramp_width_m / 2 for item in geometry], device=device)
        self.ramp_run = torch.tensor([item.ramp_run_m for item in geometry], device=device)
        self.lip_height = torch.tensor([item.lip_height_m for item in geometry], device=device)
        self.ramp_slope = torch.tensor([(item.lip_height_m - item.entry_height_m) / item.ramp_run_m
                                       for item in geometry], device=device)
        self.air_required = math.ceil(min_air_seconds / dt - 1e-9)
        self.stable_required = math.ceil(stable_seconds / dt - 1e-9)
        count = len(self.geometries)
        self.air_ticks = torch.zeros(count, dtype=torch.long, device=device)
        self.air_peak_ticks = torch.zeros_like(self.air_ticks)
        self.stable_ticks = torch.zeros_like(self.air_ticks)
        self.stable_peak_ticks = torch.zeros_like(self.air_ticks)
        self.pending = torch.zeros(count, dtype=torch.bool, device=device)
        self.jump_accepted = torch.zeros_like(self.pending)
        self.previous_ramp_support = torch.zeros_like(self.pending)
        self.verified_flight = torch.zeros_like(self.pending)
        self.flight_event_pending = torch.zeros_like(self.pending)
        self.crossed_gap = torch.zeros_like(self.pending)
        self.landed = torch.zeros_like(self.pending)
        self.failed = torch.zeros_like(self.pending)
        self.floor_contact = torch.zeros_like(self.pending)
        self.wrong_first_contact = torch.zeros_like(self.pending)
        self.invalid_state = torch.zeros_like(self.pending)
        self.release_position = torch.zeros(count, 3, device=device)
        self.release_velocity = torch.zeros_like(self.release_position)
        self.touchdown_position = torch.zeros_like(self.release_position)
        self.touchdown_velocity = torch.zeros_like(self.release_position)
        self.touchdown_seen = torch.zeros_like(self.pending)

    def reset(self, ids):
        for value in (self.air_ticks, self.air_peak_ticks, self.stable_ticks,
                      self.stable_peak_ticks, self.pending, self.jump_accepted, self.previous_ramp_support,
                      self.verified_flight, self.flight_event_pending, self.crossed_gap,
                      self.landed, self.failed, self.floor_contact, self.wrong_first_contact,
                      self.invalid_state,
                      self.release_position, self.release_velocity, self.touchdown_position,
                      self.touchdown_velocity, self.touchdown_seen):
            value[ids] = 0

    @property
    def success(self):
        return self.enabled & self.landed & ~self.failed & (self.stable_ticks >= self.stable_required)

    def consume_flight_events(self):
        """Return each verified physical takeoff once, including short flights."""
        result = self.flight_event_pending.clone()
        self.flight_event_pending.zero_()
        return result

    def observe(self, wheel_centers, contacts, surface_contacts, com_position,
                com_velocity, stable, body_contact, *, accepted_jump_event):
        count = len(self.enabled)
        for name, value, shape, boolean in (
                ("wheel_centers", wheel_centers, (count, 2, 3), False),
                ("contacts", contacts, (count, 2), True),
                ("surface_contacts", surface_contacts, (count, 2, 4), True),
                ("com_position", com_position, (count, 3), False),
                ("com_velocity", com_velocity, (count, 3), False),
                ("stable", stable, (count,), True),
                ("body_contact", body_contact, (count,), True),
                ("accepted_jump_event", accepted_jump_event, (count,), True)):
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.device != self.enabled.device or (boolean and value.dtype != torch.bool)):
                raise ValueError(f"Ramp {name} requires {'boolean ' if boolean else ''}{shape} on the observer device")
        # Keep the physics observer on device. The environment's finite-state
        # guard remains responsible for aborting an invalid plant transition.
        finite = (torch.isfinite(wheel_centers).all(-1).all(-1)
                  & torch.isfinite(com_position).all(-1) & torch.isfinite(com_velocity).all(-1))
        self.invalid_state |= self.enabled & ~finite
        self.failed |= self.invalid_state
        supported = contacts.any(-1)
        x = wheel_centers[..., 0] * self.direction[:, None]
        near_lip = ((x >= -.30) & (x <= .10)
                    & (wheel_centers[..., 1].abs() <= self.ramp_half_width[:, None])).all(-1)
        ramp_top = self.lip_height[:, None] + x * self.ramp_slope[:, None]
        on_top = (wheel_centers[..., 2] - .06 - ramp_top).abs() <= .035
        top_support = (surface_contacts[..., 2] & contacts & on_top).any(-1)
        on_ramp = ((x >= -self.ramp_run[:, None]) & (x <= .10)
                   & (wheel_centers[..., 1].abs() <= self.ramp_half_width[:, None])).all(-1)
        forward = com_velocity[:, 0] * self.direction > .1
        # Accept preload lead on the incline. Physical release must still occur
        # near its lip; an accepted command by itself never creates air credit.
        self.jump_accepted |= (self.enabled & accepted_jump_event & finite & on_ramp
                               & (top_support | self.previous_ramp_support)
                               & forward & ~self.failed & ~body_contact)
        ramp_support = top_support & near_lip & supported
        departure = (self.enabled & self.previous_ramp_support & ~supported
                     & self.jump_accepted
                     & ~self.verified_flight & ~self.failed
                     & (x.amin(-1) >= -.30) & (x.amax(-1) < self.gap)
                     & forward & ~body_contact)
        self.release_position[departure] = com_position[departure]
        self.release_velocity[departure] = com_velocity[departure]
        self.pending |= departure
        # A bounce before the minimum air interval starts a new physical event.
        bounced = self.pending & supported & ~self.verified_flight
        self.pending[bounced] = False
        clear = (wheel_centers[..., 2] - .06 > .01).all(-1)
        self.air_ticks = torch.where(self.pending & ~supported & clear, self.air_ticks + 1, 0)
        self.air_peak_ticks = torch.maximum(self.air_peak_ticks, self.air_ticks)
        verified = (self.pending & (self.air_ticks >= self.air_required)
                    & ~self.verified_flight & ~self.failed)
        self.verified_flight |= verified
        self.flight_event_pending |= verified
        self.crossed_gap |= self.verified_flight & (x.amin(-1) >= self.gap)
        after_release = self.pending | self.verified_flight
        self.floor_contact |= after_release & surface_contacts[..., 0].any(-1)
        valid_feet = ((x >= self.gap[:, None]) & (x <= self.platform_end[:, None])
                      & (wheel_centers[..., 1].abs() <= self.platform_half_width[:, None])
                      & ((wheel_centers[..., 2] - .06 - self.platform_height[:, None]).abs() <= .025))
        platform_contacts = surface_contacts[..., 3] & contacts & valid_feet
        first_contact = self.verified_flight & supported & ~self.touchdown_seen
        self.touchdown_seen |= first_contact
        self.touchdown_position[first_contact] = com_position[first_contact]
        self.touchdown_velocity[first_contact] = com_velocity[first_contact]
        self.wrong_first_contact |= first_contact & ~platform_contacts.any(-1)
        self.landed |= first_contact & platform_contacts.any(-1)
        self.failed |= (self.enabled & after_release &
                        (self.floor_contact | self.wrong_first_contact | body_contact))
        settled = (self.enabled & self.landed & self.crossed_gap & ~self.failed
                   & platform_contacts.all(-1) & stable)
        self.stable_ticks = torch.where(settled, self.stable_ticks + 1, 0)
        self.stable_peak_ticks = torch.maximum(self.stable_peak_ticks, self.stable_ticks)
        self.previous_ramp_support.copy_(ramp_support & self.enabled)

    def diagnostics(self):
        return {"ramp_flight_jump_accepted": self.jump_accepted.clone(),
                "ramp_flight_verified": self.verified_flight.clone(),
                "ramp_flight_crossed_gap": self.crossed_gap.clone(),
                "ramp_flight_landed": self.landed.clone(),
                "ramp_flight_success": self.success,
                "ramp_flight_failed": self.failed.clone(),
                "ramp_flight_floor_contact": self.floor_contact.clone(),
                "ramp_flight_wrong_first_contact": self.wrong_first_contact.clone(),
                "ramp_flight_invalid_state": self.invalid_state.clone(),
                "ramp_flight_air_peak_s": self.air_peak_ticks * self.dt,
                "ramp_flight_stable_peak_s": self.stable_peak_ticks * self.dt,
                "ramp_flight_release_position_m": self.release_position.clone(),
                "ramp_flight_release_velocity_m_s": self.release_velocity.clone(),
                "ramp_flight_touchdown_position_m": self.touchdown_position.clone(),
                "ramp_flight_touchdown_velocity_m_s": self.touchdown_velocity.clone(),
                "ramp_flight_touchdown_observed": self.touchdown_seen.clone()}
