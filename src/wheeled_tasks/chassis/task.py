"""Pure tensor task phases and deterministic terrain descriptions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch


class Phase(IntEnum):
    GROUND = 0
    TAKEOFF = 1
    FLIGHT = 2
    LANDING = 3
    RECOVERY = 4


class PhaseTracker:
    """Filtered ground contact drives transitions; at most one transition per tick."""

    def __init__(self, count, device, dt=0.01):
        self.dt = dt
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.time = torch.zeros(count, device=device)
        self.air_time = torch.zeros_like(self.time)
        self.contact_time = torch.zeros_like(self.time)
        self.stable_time = torch.zeros_like(self.time)
        self.flew = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.phase[ids] = Phase.GROUND
        for value in (self.time, self.air_time, self.contact_time, self.stable_time):
            value[ids] = 0
        self.flew[ids] = False

    def update(self, wheel_contact, request, stable):
        old = self.phase.clone()
        supported = wheel_contact.any(-1)
        self.air_time = torch.where(supported, 0., self.air_time + self.dt)
        self.contact_time = torch.where(supported, self.contact_time + self.dt, 0.)
        self.stable_time = torch.where(supported & stable, self.stable_time + self.dt, 0.)
        self.time += self.dt
        self.phase[(old == Phase.GROUND) & request & wheel_contact.all(-1)] = Phase.TAKEOFF
        airborne = ((old == Phase.GROUND) | (old == Phase.TAKEOFF)) & (self.air_time >= 0.03)
        self.phase[airborne] = Phase.FLIGHT
        self.flew |= airborne & ((old == Phase.TAKEOFF) | request)
        self.phase[(old == Phase.FLIGHT) & supported] = Phase.LANDING
        self.phase[(old == Phase.LANDING) & (self.contact_time >= 0.06)] = Phase.RECOVERY
        self.phase[(old == Phase.RECOVERY) & (self.stable_time >= 0.5)] = Phase.GROUND
        changed = self.phase != old
        self.time[changed] = 0.
        return changed


@dataclass(frozen=True)
class Surface:
    """A support plane segment, expressed in local terrain coordinates [m]."""

    x0: float
    x1: float
    z0: float = 0.
    slope: float = 0.
    friction: float = 0.5
    cross_slope: float = 0.

    def height(self, x, y=0.):
        return self.z0 + (x - self.x0) * self.slope + y * self.cross_slope

    def box(self):
        if self.cross_slope:
            if self.slope:
                raise ValueError("Compound slope collision geometry is not supported")
            theta = math.atan(self.cross_slope)
            return ((self.x1 - self.x0, 4. / math.cos(theta), .1),
                    ((self.x0 + self.x1) / 2, .05 * math.sin(theta), self.z0 - .05 * math.cos(theta)),
                    (math.cos(theta / 2), math.sin(theta / 2), 0., 0.))
        theta = math.atan(self.slope)
        thickness = 0.1
        length = (self.x1 - self.x0) / math.cos(theta)
        top_mid = self.height((self.x0 + self.x1) / 2)
        # Translate along the plane normal so the upper face matches height(x).
        position = ((self.x0 + self.x1) / 2 + math.sin(theta) * thickness / 2,
                    0., top_mid - math.cos(theta) * thickness / 2)
        return (length, 4., thickness), position, (math.cos(theta / 2), 0., -math.sin(theta / 2), 0.)


def terrain_surfaces(kind: str, limits: dict, level: float, index=0) -> list[Surface]:
    if not 0 <= level <= 1:
        raise ValueError("terrain level must be in [0, 1]")
    if kind in ("flat", "jump"):
        return [Surface(-4., 4.)]
    if kind == "material":
        return [Surface(-4., -0.5), Surface(-0.5, 1., friction=0.5 - 0.2 * level),
                Surface(1., 4., friction=0.5 + 0.2 * level)]
    if kind == "cross_slope":
        slope = math.tan(math.radians(limits["slope_deg"] * level)) * (1 if index % 2 else -1)
        return [Surface(-4., 4., cross_slope=slope)]
    if kind in ("slope", "slope_up", "slope_down"):
        sign = {"slope_up": 1, "slope_down": -1}.get(kind, 1 if index % 2 else -1)
        slope = math.tan(math.radians(limits["slope_deg"] * level)) * sign
        return [Surface(-4., -1.), Surface(-1., 1., slope=slope), Surface(1., 4., 2 * slope)]
    if kind == "rough":
        result = [Surface(-4., -1.)]
        for i in range(20):
            z = limits["roughness_m"] * level * (0.5 + 0.5 * math.sin(i * 2.31 + index))
            result.append(Surface(-1. + i * 0.25, -0.75 + i * 0.25, z))
        return result
    if kind in ("low_step", "step_up", "step_down"):
        height = limits[{"low_step": "low_step_m", "step_up": "step_up_m",
                         "step_down": "step_down_m"}[kind]] * level
        return ([Surface(-4., 0., height), Surface(0., 4.)] if kind == "step_down" else
                [Surface(-4., 0.), Surface(0., 4., height)])
    if kind == "platform":
        height = limits["step_up_m"] * level
        return [Surface(-4., 0.), Surface(0., 2., height), Surface(2., 4.)]
    if kind == "stairs_down":
        rise = limits["stair_rise_m"] * level
        return [Surface(-4., 0., 3 * rise), Surface(0., .6, 2 * rise),
                Surface(.6, 1.2, rise), Surface(1.2, 4.)]
    if kind == "stairs":
        rise = limits["stair_rise_m"] * level
        return [Surface(-4., 0.), Surface(0., 0.6, rise), Surface(0.6, 1.2, 2 * rise),
                Surface(1.2, 4., 3 * rise)]
    raise ValueError(f"Unknown terrain: {kind}")


def choose_terrains(stage, count, base_fraction=0.4, coverage=False):
    families = list(stage["terrain"])
    if coverage:
        # Deterministic validation includes every family, independent of training proportions.
        return [(["flat"] + families)[i % (len(families) + 1)] for i in range(count)]
    base_count = math.ceil(count * base_fraction)
    return ["flat" if i < base_count else families[(i - base_count) % len(families)] for i in range(count)]


def choose_scene_groups(groups, count):
    """Allocate simultaneous skill groups, with deterministic per-group terrain coverage."""
    if not groups or not math.isclose(sum(g["fraction"] for g in groups), 1., abs_tol=1e-9):
        raise ValueError("Scene fractions must sum to one")
    if count < len(groups):
        raise ValueError("Need at least one environment per scene group")
    result = []
    for index, group in enumerate(groups):
        size = count - len(result) if index == len(groups) - 1 else int(count * group["fraction"])
        if size < 1 or not group["terrain"]:
            raise ValueError("Empty scene group")
        result.extend((group["name"], group["terrain"][i % len(group["terrain"])]) for i in range(size))
    return result


def phase_reward_masks(phase):
    ground = phase == Phase.GROUND
    return {"ground": ground, "takeoff": phase == Phase.TAKEOFF,
            "flight": phase == Phase.FLIGHT,
            "landing": (phase == Phase.LANDING) | (phase == Phase.RECOVERY)}
