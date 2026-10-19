"""Continuous motion requests with measured progress and bounded phase times.

Only commands change. Failed phases still train; neither rewards, physical
state nor PPO masks are changed by this command owner.
"""
from copy import deepcopy
import hashlib
import math

import torch


VERSION = "motion-retention-course-v1"
AXIS_VERSION = "continuous-pure-axis-v1"
PHASES = ("initial_hold", "accelerate", "speed_change", "brake", "reverse", "final_hold")
LESSONS = ("stationary", "same_direction_change", "stop_then_reverse", "direct_reversal")
ROUTES = ((0,), (0, 1, 2, 3, 5), (0, 1, 3, 4, 5), (0, 1, 2, 3, 5))
DEFAULT_SETTINGS = {
    "speed_edges_m_s": [.05, .5, 1.35, 2., 3.],
    "height_range_m": [.20, .42],
    "lesson_probabilities": [.25, .25, .25, .25],
    "phase_timeout_range_ticks": [[50, 100], [150, 225], [100, 200], [200, 300], [150, 225]],
    "minimum_phase_ticks": [50, 50, 50, 50, 50],
    "settled_ticks": [25, 25, 25, 50, 25, 50],
    "minimum_final_hold_ticks": 100,
    "tracking_relative_tolerance": .10,
    "tracking_floor_m_s": .01,
    "stop_planar_speed_m_s": .05,
    "lateral_speed_m_s": .05,
    "yaw_tolerance_rad_s": .15,
}


def validate_settings(settings, episode_ticks=999):
    """Validate explicit candidate gates, which are not acceptance thresholds."""
    if (not isinstance(settings, dict)
            or set(settings) not in (set(DEFAULT_SETTINGS), set(DEFAULT_SETTINGS) | {"axis_coverage"})):
        raise ValueError("Motion course requires the complete explicit settings")
    edges = settings["speed_edges_m_s"]
    height = settings["height_range_m"]
    probabilities = settings["lesson_probabilities"]
    if (not isinstance(edges, list) or len(edges) != 5
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in edges)
            or not .05 <= edges[0] < edges[-1] <= 3.
            or any(a >= b for a, b in zip(edges, edges[1:]))
            or not isinstance(height, list) or len(height) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in height)
            or not .20 <= height[0] < height[1] <= .42
            or not isinstance(probabilities, list) or len(probabilities) != len(LESSONS)
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in probabilities)
            or not math.isclose(sum(probabilities), 1., abs_tol=1e-9)):
        raise ValueError("Motion course needs ordered continuous ranges and explicit lesson probabilities")
    ranges = settings["phase_timeout_range_ticks"]
    minimum = settings["minimum_phase_ticks"]
    settled = settings["settled_ticks"]
    final = settings["minimum_final_hold_ticks"]
    if (not isinstance(ranges, list) or len(ranges) != 5
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(type(x) is not int or x <= 0 for x in pair)
                   or pair[0] > pair[1] for pair in ranges)
            or not isinstance(minimum, list) or len(minimum) != 5
            or any(type(x) is not int or x <= 0 for x in minimum)
            or not isinstance(settled, list) or len(settled) != 6
            or any(type(x) is not int or x <= 0 for x in settled)
            or type(final) is not int or final < settled[-1]
            or any(max(minimum[i], settled[i]) > ranges[i][0] for i in range(5))
            or any(sum(ranges[phase][1] for phase in route if phase < 5) + final > episode_ticks
                   for route in ROUTES[1:])):
        raise ValueError("Motion phases must leave the minimum final hold inside the actual episode")
    for key in ("tracking_relative_tolerance", "tracking_floor_m_s", "stop_planar_speed_m_s",
                "lateral_speed_m_s", "yaw_tolerance_rad_s"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Motion progress tolerances must be positive and finite")
    if settings["tracking_relative_tolerance"] >= 1 or settings["tracking_floor_m_s"] >= edges[0]:
        raise ValueError("A stopped robot must not qualify for the minimum nonzero speed")
    if "axis_coverage" in settings:
        axis = settings["axis_coverage"]
        if (not isinstance(axis, dict)
                or set(axis) != {"version", "rotation_fraction_of_moving_lessons", "yaw_edges_rad_s"}
                or axis["version"] != AXIS_VERSION
                or type(axis["rotation_fraction_of_moving_lessons"]) not in (int, float)
                or not math.isfinite(axis["rotation_fraction_of_moving_lessons"])
                or not 0 < axis["rotation_fraction_of_moving_lessons"] <= 1):
            raise ValueError("Pure-axis coverage needs an explicit version and moving-lesson fraction")
        yaw = axis["yaw_edges_rad_s"]
        if (not isinstance(yaw, list) or len(yaw) != 5
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in yaw)
                or yaw[0] != 0 or not yaw[-1] <= 4.5 * math.pi
                or any(a >= b for a, b in zip(yaw, yaw[1:]))):
            raise ValueError("Pure-axis yaw bands must continuously cover zero through the declared maximum")
    return deepcopy(settings)


class MotionTransitionRequests:
    """State-aware sequences on a disjoint nominal flat population."""

    mutable = True
    _LONG = ("lesson", "phase", "next_phase", "phase_ticks", "stable_ticks", "deadline", "band", "observed_physics_tick")
    _FLOAT = ("targets", "height", "entry_vx", "entry_target")
    _BOOL = ("qualified_prefix", "hold_settled", "hold_broken", "entry_passed")

    def __init__(self, count, device, ids, settings, episode_ticks=999):
        self.settings = validate_settings(settings, episode_ticks)
        self.axis_coverage = self.settings.get("axis_coverage")
        self._state_long = self._LONG + (("axis",) if self.axis_coverage is not None else ())
        self._state_float = self._FLOAT + (("entry_yaw", "entry_target_yaw") if self.axis_coverage is not None else ())
        self.device = torch.device(device)
        self.ids = ids.clone()
        self.mask = torch.zeros(count, dtype=torch.bool, device=device)
        self.mask[ids] = True
        self.lesson = torch.zeros(count, dtype=torch.long, device=device)
        self.phase = torch.zeros_like(self.lesson)
        self.next_phase = torch.zeros_like(self.lesson)
        self.phase_ticks = torch.zeros_like(self.lesson)
        self.stable_ticks = torch.zeros_like(self.lesson)
        self.band = torch.zeros_like(self.lesson)
        self.axis = torch.zeros_like(self.lesson)
        self.deadline = torch.ones(count, 5, dtype=torch.long, device=device)
        self.observed_physics_tick = torch.zeros_like(self.lesson)
        self.targets = torch.zeros(count, 3, device=device)
        self.height = torch.full((count,), self.settings["height_range_m"][0], device=device)
        self.entry_vx = torch.zeros(count, device=device)
        self.entry_target = torch.zeros_like(self.entry_vx)
        self.entry_yaw = torch.zeros_like(self.entry_vx)
        self.entry_target_yaw = torch.zeros_like(self.entry_vx)
        self.qualified_prefix = torch.ones_like(self.mask)
        self.hold_settled = torch.zeros_like(self.mask)
        self.hold_broken = torch.zeros_like(self.mask)
        self.entry_passed = torch.zeros_like(self.mask)
        self._settled = torch.tensor(self.settings["settled_ticks"], device=self.device)
        self._minimum = torch.tensor(self.settings["minimum_phase_ticks"] + [0], device=self.device)
        self.paid = {}
        self.identity = {
            "version": VERSION, "lanes": ids.numel(), "settings": deepcopy(self.settings),
            "lane_ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            "phase_names": list(PHASES), "episode_policy_ticks": episode_ticks,
            "scope": "commands only; progress gates never filter PPO or change rewards",
            "gate_units": "20 ms policy intervals; tolerances are curriculum gates, not acceptance",
        }
        if self.axis_coverage is not None:
            self.identity["rotation_progress"] = (
                "measured yaw tracking during nonzero requests; zero-target holds require linear and angular stop; "
                "root-point speed is recorded separately and is not a fitted rotation-center drift gate")

    def validate_reset(self, ids, ground):
        if bool((self.mask[ids] & ground).any()):
            raise ValueError("Motion course requires nominal births without the ground override")

    def reset(self, ids, generator):
        ids = ids[self.mask[ids]]
        n = ids.numel()
        if not n:
            return
        random = torch.rand(n, 10, device=self.device, generator=generator)
        edges = self.targets.new_tensor(self.settings["speed_edges_m_s"])
        band = (random[:, :3] * 4).long().clamp_max(3)
        magnitudes = edges[band] + random[:, 3:6] * (edges[band + 1] - edges[band])
        sign = torch.where(random[:, 6] < .5, -1., 1.)
        cumulative = self.targets.new_tensor(self.settings["lesson_probabilities"]).cumsum(0)
        lesson = (random[:, 7, None] >= cumulative[:-1]).sum(-1)
        # The second nonzero target may be a slower/faster same-direction
        # request or a direct reversal. The last drive follows a real stop.
        magnitudes[:, 0] *= sign
        magnitudes[:, 1] *= torch.where(lesson == 3, -sign, sign)
        magnitudes[:, 2] *= -sign
        if self.axis_coverage is not None:
            # Reuse the previously unused draw to preserve other owners' RNG stream.
            rotation = (lesson != 0) & (random[:, 9]
                < self.axis_coverage["rotation_fraction_of_moving_lessons"])
            yaw_edges = edges.new_tensor(self.axis_coverage["yaw_edges_rad_s"])
            # Open lower endpoints avoid a zero target in the first yaw band.
            yaw = yaw_edges[band] + (1. - random[:, 3:6]) * (yaw_edges[band + 1] - yaw_edges[band])
            yaw[:, 0] *= sign
            yaw[:, 1] *= torch.where(lesson == 3, -sign, sign)
            yaw[:, 2] *= -sign
            magnitudes = torch.where(rotation[:, None], yaw, magnitudes)
            self.axis[ids] = rotation.long()
        self.targets[ids] = magnitudes
        self.lesson[ids], self.band[ids] = lesson, band[:, 0]
        low, high = self.settings["height_range_m"]
        self.height[ids] = low + random[:, 8] * (high - low)
        ranges = torch.tensor(self.settings["phase_timeout_range_ticks"], device=self.device)
        draws = torch.rand(n, 5, device=self.device, generator=generator)
        self.deadline[ids] = ranges[:, 0] + (draws * (ranges[:, 1] - ranges[:, 0] + 1)).long()
        for name in ("phase", "next_phase", "phase_ticks", "stable_ticks", "observed_physics_tick",
                     "entry_vx", "entry_target", "entry_yaw", "entry_target_yaw"):
            getattr(self, name)[ids] = 0
        self.qualified_prefix[ids] = True
        self.hold_settled[ids] = False
        self.hold_broken[ids] = False
        self.entry_passed[ids] = False

    def _target(self):
        result = torch.zeros_like(self.height)
        for phase, column in ((1, 0), (2, 1), (4, 2)):
            result = torch.where(self.phase == phase, self.targets[:, column], result)
        return torch.where(self.lesson == 0, 0., result)

    def publish(self, owner, ids):
        ids = ids[self.mask[ids]]
        self.phase[ids] = self.next_phase[ids]
        request = torch.zeros(ids.numel(), 3, device=self.device)
        target = self._target()[ids]
        request[:, 0] = torch.where(self.axis[ids] == 0, target, 0.)
        request[:, 2] = torch.where(self.axis[ids] == 1, target, 0.)
        owner.command3[ids] = request
        owner.generator_command3[ids] = request
        owner.height[ids] = self.height[ids]

    def observe(self, owner, velocity, omega, done, valid):
        """Consume the completed interval before command advance and reset."""
        if any(x.shape != (owner.num_envs, 3) for x in (velocity, omega)):
            raise ValueError("Motion measurements require body velocity and angular velocity [N,3]")
        # Every native interval must be observed once, even if it terminates.
        if bool((self.mask & (owner.episode_physics_ticks - self.observed_physics_tick != 4)).any()):
            raise ValueError("Motion progress requires exactly one measurement per policy interval")
        self.observed_physics_tick.copy_(owner.episode_physics_ticks)
        phase = self.phase.clone()
        target = self._target()
        valid = valid & torch.isfinite(velocity).all(-1) & torch.isfinite(omega).all(-1)
        speed = velocity[:, :2].norm(dim=-1)
        tolerance = torch.maximum(target.abs() * self.settings["tracking_relative_tolerance"],
                                  target.new_tensor(self.settings["tracking_floor_m_s"]))
        tracking = ((velocity[:, 0] - target).abs() <= tolerance) & (
            velocity[:, 1].abs() <= self.settings["lateral_speed_m_s"])
        linear = tracking & (omega[:, 2].abs() <= self.settings["yaw_tolerance_rad_s"])
        # A tolerance floor must not let a stopped robot pass a tiny spin target.
        yaw_tolerance = torch.minimum(torch.maximum(
            target.abs() * self.settings["tracking_relative_tolerance"],
            target.new_tensor(self.settings["yaw_tolerance_rad_s"])), target.abs() * .5)
        # Root-point orbital speed is not fitted rotation-center drift. Keep
        # it observable without inventing a drift threshold for phase progress.
        angular = (omega[:, 2] - target).abs() <= yaw_tolerance
        holding_good = (speed <= self.settings["stop_planar_speed_m_s"]) & (
            omega[:, 2].abs() <= self.settings["yaw_tolerance_rad_s"])
        good = valid & torch.where(target == 0., holding_good,
                                   torch.where(self.axis == 1, angular, linear))
        self.phase_ticks += self.mask.long()
        self.stable_ticks = torch.where(self.mask & good, self.stable_ticks + 1, 0)
        settled = self._settled[phase]
        minimum = self._minimum[phase]
        passed = good & (self.stable_ticks >= settled) & (self.phase_ticks >= minimum)
        holding = self.mask & ((self.lesson == 0) | (phase == 5))
        self.hold_broken |= holding & self.hold_settled & ~good
        self.hold_settled |= holding & passed
        timed_out = self.phase_ticks >= self.deadline.gather(1, phase.clamp_max(4)[:, None]).squeeze(1)
        advance = self.mask & (self.lesson != 0) & (phase < 5) & ~done & (passed | timed_out)
        new_phase = (phase + 1).clamp_max(5)
        new_phase = torch.where((self.lesson == 2) & (phase == 1), 3, new_phase)
        new_phase = torch.where((self.lesson != 2) & (phase == 3), 5, new_phase)
        self.paid = {
            "phase": torch.where(self.mask, phase, -1), "lesson": self.lesson.clone(),
            "target_vx": torch.where(self.axis == 0, target, 0.), "measured_vx": velocity[:, 0].clone(),
            "planar_speed": speed.clone(), "height": self.height.clone(),
            "entry_vx": self.entry_vx.clone(), "entry_target": self.entry_target.clone(),
            "entry_passed": self.entry_passed.clone(),
            "phase_first": self.mask & (self.phase_ticks == 1),
            "tracking": self.mask & good, "valid": self.mask & valid,
            "phase_passed": advance & passed, "phase_timed_out": advance & ~passed,
            "hold_broken": self.hold_broken.clone(),
            "qualified_sequence": self.mask & self.qualified_prefix & passed & ~self.hold_broken
                                  & ((phase == 5) | (self.lesson == 0)),
        }
        if self.axis_coverage is not None:
            self.paid.update(axis=self.axis.clone(), target_yaw=torch.where(self.axis == 1, target, 0.),
                             measured_yaw=omega[:, 2].clone(), entry_yaw=self.entry_yaw.clone(),
                             entry_target_yaw=self.entry_target_yaw.clone())
        self.qualified_prefix &= ~advance | passed
        self.entry_vx = torch.where(advance, velocity[:, 0], self.entry_vx)
        self.entry_target = torch.where(advance, torch.where(self.axis == 0, target, 0.), self.entry_target)
        self.entry_yaw = torch.where(advance, omega[:, 2], self.entry_yaw)
        self.entry_target_yaw = torch.where(advance, torch.where(self.axis == 1, target, 0.), self.entry_target_yaw)
        self.entry_passed = torch.where(advance, passed, self.entry_passed)
        self.next_phase = torch.where(advance, new_phase, self.phase)
        self.phase_ticks = torch.where(advance, 0, self.phase_ticks)
        self.stable_ticks = torch.where(advance, 0, self.stable_ticks)
        self.hold_settled &= ~advance
        self.hold_broken &= ~advance

    def published_phases(self, owner):
        return torch.where(self.mask, self.phase, -1)

    def summary(self):
        return deepcopy(self.identity)

    def state_dict(self):
        return {name: getattr(self, name).detach().cpu().clone()
                for name in self._state_long + self._state_float + self._BOOL}

    def prepare_state(self, state):
        from .flat_command_sampling import _restart_tensor
        if not isinstance(state, dict) or set(state) != set(self._state_long + self._state_float + self._BOOL):
            raise ValueError("Incomplete motion restart state")
        checked = {name: _restart_tensor(state[name], getattr(self, name).shape,
                   getattr(self, name).dtype, name).to(self.device) for name in state}
        axis = checked["axis"] if self.axis_coverage is not None else self.axis
        maximum = (torch.where(axis == 1, self.targets.new_tensor(self.axis_coverage["yaw_edges_rad_s"][-1]),
                               self.settings["speed_edges_m_s"][-1]) if self.axis_coverage is not None
                   else self.targets.new_full(self.height.shape, self.settings["speed_edges_m_s"][-1]))
        if (bool(((axis < 0) | (axis > 1)).any())
                or bool(((checked["lesson"] == 0) & (axis != 0)).any())
                or bool((axis[~self.mask] != 0).any())
                or bool(((checked["phase"] < 0) | (checked["phase"] > 5)).any())
                or bool(((checked["next_phase"] < 0) | (checked["next_phase"] > 5)).any())
                or bool(((checked["lesson"] < 0) | (checked["lesson"] >= len(LESSONS))).any())
                or any(bool((checked[k] < 0).any()) for k in self._LONG)
                or bool((checked["band"] > 3).any())
                or bool((checked["targets"].abs() > maximum[:, None]).any())
                or bool(((checked["height"] < self.settings["height_range_m"][0])
                         | (checked["height"] > self.settings["height_range_m"][1])).any())
                or bool((checked["deadline"] <= 0).any())
                or bool((checked["next_phase"] < checked["phase"]).any())
                or bool((checked["next_phase"] > checked["phase"] + 2).any())):
            raise ValueError("Motion restart state is outside the declared domains")
        return checked

    def commit_state(self, prepared):
        for name, value in prepared.items():
            getattr(self, name).copy_(value)
        self.paid = {}
