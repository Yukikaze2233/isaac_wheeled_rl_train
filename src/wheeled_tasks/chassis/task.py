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

    def update(self, wheel_contact, request, stable, *, recovery_takeoff=False):
        old = self.phase.clone()
        supported = wheel_contact.any(-1)
        self.air_time = torch.where(supported, 0., self.air_time + self.dt)
        self.contact_time = torch.where(supported, self.contact_time + self.dt, 0.)
        self.stable_time = torch.where(supported & stable, self.stable_time + self.dt, 0.)
        self.time += self.dt
        self.phase[(old == Phase.GROUND) & request & wheel_contact.all(-1)] = Phase.TAKEOFF
        if recovery_takeoff:
            self.phase[(old == Phase.RECOVERY) & request & wheel_contact.all(-1)] = Phase.TAKEOFF
        airborne = ((old == Phase.GROUND) | (old == Phase.TAKEOFF)) & (self.air_time >= 0.03)
        self.phase[airborne] = Phase.FLIGHT
        self.flew |= airborne & ((old == Phase.TAKEOFF) | request)
        self.phase[(old == Phase.FLIGHT) & supported] = Phase.LANDING
        self.phase[(old == Phase.LANDING) & (self.contact_time >= 0.06)] = Phase.RECOVERY
        self.phase[(old == Phase.RECOVERY) & (self.stable_time >= 0.5) & ~(request & recovery_takeoff)] = Phase.GROUND
        changed = self.phase != old
        self.time[changed] = 0.
        return changed


class FallConfirmation:
    """Debounce recoverable posture excursions without delaying critical falls."""

    def __init__(self, count, device, dt, hold_seconds=0.):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(hold_seconds) or hold_seconds < 0:
            raise ValueError("Positive timestep and nonnegative finite fall duration required")
        self.required_ticks = max(1, math.ceil(hold_seconds / dt - 1e-9))
        self.ticks = torch.zeros(count, dtype=torch.long, device=device)
        self.candidate = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.ticks[ids] = 0
        self.candidate[ids] = False

    def update(self, gravity_z, height, flight):
        self.candidate = (gravity_z > -.5) | ((height < .15) & ~flight)
        self.ticks = torch.where(self.candidate, self.ticks + 1, 0)
        critical = (gravity_z >= 0.) | ((height < .10) & ~flight)
        return critical | (self.ticks >= self.required_ticks)


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

    def box(self, width=4., *, solid_bottom=None):
        if solid_bottom is not None and not self.slope and not self.cross_slope:
            if not math.isfinite(solid_bottom) or solid_bottom >= self.z0:
                raise ValueError("Solid platform bottom must be below the support surface")
            return ((self.x1 - self.x0, width, self.z0 - solid_bottom),
                    ((self.x0 + self.x1) / 2, 0., (self.z0 + solid_bottom) / 2), (1., 0., 0., 0.))
        if self.cross_slope:
            if self.slope:
                raise ValueError("Compound slope collision geometry is not supported")
            theta = math.atan(self.cross_slope)
            return ((self.x1 - self.x0, width / math.cos(theta), .1),
                    ((self.x0 + self.x1) / 2, .05 * math.sin(theta), self.z0 - .05 * math.cos(theta)),
                    (math.cos(theta / 2), math.sin(theta / 2), 0., 0.))
        theta = math.atan(self.slope)
        thickness = 0.1
        length = (self.x1 - self.x0) / math.cos(theta)
        top_mid = self.height((self.x0 + self.x1) / 2)
        # Translate along the plane normal so the upper face matches height(x).
        position = ((self.x0 + self.x1) / 2 + math.sin(theta) * thickness / 2,
                    0., top_mid - math.cos(theta) * thickness / 2)
        return (length, width, thickness), position, (math.cos(theta / 2), 0., -math.sin(theta / 2), 0.)


def corridor_mesh(half_length, width, segment_length=8.):
    """A flat triangle mesh with bounded face size, avoiding elongated convex hulls."""
    if half_length <= 0 or width <= 0 or segment_length <= 0:
        raise ValueError("Positive corridor dimensions required")
    count = math.ceil(2 * half_length / segment_length)
    points = [(float(-half_length + 2 * half_length * i / count), y, 0.)
              for i in range(count + 1) for y in (-width / 2, width / 2)]
    indices = [vertex for i in range(count) for vertex in
               (2 * i, 2 * i + 2, 2 * i + 1, 2 * i + 1, 2 * i + 2, 2 * i + 3)]
    return points, indices


def terrain_surfaces(kind: str, limits: dict, level: float, index=0) -> list[Surface]:
    if not 0 <= level <= 1:
        raise ValueError("terrain level must be in [0, 1]")
    if kind in ("stairs", "stairs_down") and any(
            name in limits for name in ("stair_heights_m", "stair_tread_m")):
        # Heights are cumulative top elevations, not individual riser heights.
        heights = limits.get("stair_heights_m")
        if "stair_heights_m" not in limits:
            rise = limits.get("stair_rise_m")
            if type(rise) not in (int, float) or not math.isfinite(rise) or rise <= 0:
                raise ValueError("stair_rise_m must be finite and positive")
            heights = [rise, 2 * rise, 3 * rise]
        if (not isinstance(heights, (list, tuple)) or not heights
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in heights)
                or any(first >= second for first, second in zip(heights, heights[1:]))):
            raise ValueError("stair_heights_m must contain strictly increasing positive cumulative tops")
        tread = limits.get("stair_tread_m", .6)
        treads = [tread] * len(heights) if type(tread) in (int, float) else tread
        if (not isinstance(treads, (list, tuple)) or len(treads) != len(heights)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in treads)):
            raise ValueError("stair_tread_m must be positive or have one positive length per top")
        heights = [value * level for value in heights]
        top = heights[-1] if kind == "stairs_down" else 0.
        levels = list(reversed([0., *heights[:-1]])) if kind == "stairs_down" else heights
        result, x = [Surface(-4., 0., top)], 0.
        for height, length in zip(levels, treads):
            if not math.isfinite(x + length) or x + length <= x:
                raise ValueError("Stair treads must have finite distinguishable endpoints")
            result.append(Surface(x, x + length, height))
            x += length
        # The final tread is a bounded landing region. Continue at the same
        # height: an unmeasured terminal drop must not be invented by the scene.
        exit_x = max(4., x + .6)
        if not math.isfinite(exit_x) or exit_x <= x:
            raise ValueError("Stair exit must have a finite positive length")
        result.append(Surface(x, exit_x, levels[-1]))
        return result
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
    """Use largest remainders with one env minimum; break ties in group order."""
    fractions = [group.get("fraction") for group in groups]
    if any(type(value) not in (int, float) or not 0 < value <= 1 or not math.isfinite(value)
           for value in fractions):
        raise ValueError("Scene fractions must be finite and positive")
    if not groups or not math.isclose(math.fsum(fractions), 1., abs_tol=1e-9):
        raise ValueError("Scene fractions must sum to one")
    if type(count) is not int or count < len(groups):
        raise ValueError("Need at least one environment per scene group")
    quotas = [count * value for value in fractions]
    sizes = [max(1, math.floor(quota)) for quota in quotas]
    remaining = count - sum(sizes)
    priority = sorted((index for index, quota in enumerate(quotas) if quota > sizes[index]),
                      key=lambda index: (-(quotas[index] - sizes[index]), index))
    if not 0 <= remaining <= len(priority):
        raise ValueError("Scene fractions cannot fill the environment count within floor/ceil quotas")
    for index in priority[:remaining]:
        sizes[index] += 1
    result = []
    for group, size in zip(groups, sizes):
        if size < 1 or not group["terrain"]:
            raise ValueError("Empty scene group")
        result.extend((group["name"], group["terrain"][i % len(group["terrain"])]) for i in range(size))
    return result


def phase_reward_masks(phase):
    ground = phase == Phase.GROUND
    return {"ground": ground, "takeoff": phase == Phase.TAKEOFF,
            "flight": phase == Phase.FLIGHT,
             "landing": (phase == Phase.LANDING) | (phase == Phase.RECOVERY)}


def episode_outcome(reasons, success, horizon):
    """Mechanical failures take precedence over route success and collection cuts."""
    truncated = reasons["boundary"] | reasons.get("blocked", False)
    terminated = torch.stack([value for name, value in reasons.items() if name not in ("boundary", "blocked")]).any(0)
    success = success & ~terminated & ~reasons.get("blocked", torch.zeros_like(success))
    timeouts = (horizon | truncated) & ~terminated & ~success
    return terminated, timeouts, success
