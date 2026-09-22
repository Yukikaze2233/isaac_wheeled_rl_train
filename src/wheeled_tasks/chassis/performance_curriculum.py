"""Per-skill performance progression using completed episodes, not reward sums."""
from collections import deque
from copy import deepcopy
import math

import torch


class PerformanceCurriculum:
    def __init__(self, groups, specs, settings, device):
        self.cfg = dict(settings)
        self.names = list(dict.fromkeys(groups))
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.group_ids = torch.tensor([self.indices[name] for name in groups], device=device)
        self.specs = specs
        self.scales = list(settings["height_scales"])
        self.widths = list(settings.get("height_kernel_widths_m", [math.sqrt(.001)] * len(self.scales)))
        self.window_size = settings["window_episodes"]
        if self.window_size < 1 or not self.scales or min(self.scales) <= 0:
            raise ValueError("Positive curriculum window and height scales required")
        if len(self.widths) != len(self.scales) or min(self.widths) <= 0:
            raise ValueError("Every height scale requires a positive kernel width")
        self.height_pass = list(settings.get("height_pass_m_by_level", [settings["height_pass_m"]] * len(self.scales)))
        self.height_regress = list(settings.get("height_regress_m_by_level", [settings["height_regress_m"]] * len(self.scales)))
        if (len(self.height_pass) != len(self.scales) or len(self.height_regress) != len(self.scales)
                or any(not math.isfinite(p) or not math.isfinite(r) or not 0 < p < r
                       for p, r in zip(self.height_pass, self.height_regress))):
            raise ValueError("Each height level requires finite positive pass/regression thresholds with hysteresis")
        levels = settings["spin_levels_rad_s"]
        if not levels or levels != sorted(set(levels)) or levels[0] <= 0:
            raise ValueError("Spin levels must be positive and strictly increasing")
        self.spin_levels = {}
        self.windows = {name: deque(maxlen=self.window_size) for name in self.names}
        self.state = {}
        for name in self.names:
            target = abs(specs[name]["command"][1])
            caps = sorted(set([min(value, target) for value in levels] + [target]))
            if specs[name]["kind"] != "rotate":
                caps = [target]
            self.spin_levels[name] = caps
            self.state[name] = {"height_level": 0, "spin_level": 0, "last_change": 0., "episodes": 0}
        self.height_scale = torch.full((len(self.names),), float(self.scales[0]), device=device)
        self.env_height_scale = torch.full((len(groups),), float(self.scales[0]), device=device)
        self.height_width = torch.full((len(self.names),), float(self.widths[0]), device=device)
        self.env_height_width = torch.full((len(groups),), float(self.widths[0]), device=device)
        self.error_sum = torch.zeros(len(groups), 3, dtype=torch.float64, device=device)
        self.frames = torch.zeros(len(groups), dtype=torch.int64, device=device)
        self.total_frames = torch.zeros_like(self.frames)

    def reset_episodes(self, ids):
        self.env_height_scale[ids] = self.height_scale[self.group_ids[ids]]
        self.env_height_width[ids] = self.height_width[self.group_ids[ids]]
        self.error_sum[ids] = 0.
        self.frames[ids] = 0
        self.total_frames[ids] = 0

    def limit_command(self, name, command):
        result = list(command)
        state = self.state[name]
        if self.specs[name]["kind"] == "rotate":
            cap = self.spin_levels[name][state["spin_level"]]
            result[1] = math.copysign(min(abs(result[1]), cap), result[1])
        return result

    def observe(self, height_error, velocity_error, yaw_error, supported, done, failed, reference_update):
        supported = supported.bool()
        errors = torch.stack((height_error.abs(), velocity_error.abs(), yaw_error.abs()), -1)
        self.error_sum += errors.double() * supported[:, None]
        self.frames += supported
        self.total_frames += 1
        ids = done.nonzero(as_tuple=False).flatten()
        if not len(ids):
            return
        mean = self.error_sum[ids] / self.frames[ids, None].clamp_min(1)
        mean = torch.where(self.frames[ids, None] > 0, mean, torch.full_like(mean, 10.))
        supported_fraction = self.frames[ids].double() / self.total_frames[ids].clamp_min(1)
        success = (~failed[ids] & (supported_fraction >= self.cfg.get("support_fraction_min", .8))).double()
        rows = torch.cat((self.group_ids[ids, None].double(), mean, success[:, None]), -1).cpu().tolist()
        touched = set()
        for group, height, velocity, yaw, survived in rows:
            name = self.names[int(group)]
            self.windows[name].append([height, velocity, yaw, survived])
            self.state[name]["episodes"] += 1
            touched.add(name)
        self.reset_episodes(ids)
        for name in touched:
            self._advance(name, reference_update)

    def _advance(self, name, reference_update):
        state, window = self.state[name], self.windows[name]
        if len(window) < self.window_size or reference_update - state["last_change"] < self.cfg["min_reference_updates"]:
            return
        height, velocity, yaw, survival = [sum(row[i] for row in window) / len(window) for i in range(4)]
        caps = self.spin_levels[name]
        cap = caps[state["spin_level"]]
        changed = False
        if (state["spin_level"] + 1 < len(caps) and height <= self.cfg["spin_height_pass_m"]
                and velocity <= self.cfg["spin_velocity_pass_m_s"] and yaw <= max(.15, .075 * cap)
                and survival >= self.cfg["survival_min"]):
            state["spin_level"] += 1
            state["height_level"] = 0
            changed = True
        elif survival < .8 and state["spin_level"] > 0:
            state["spin_level"] -= 1
            state["height_level"] = 0
            changed = True
        elif height <= self.height_pass[state["height_level"]] and survival >= self.cfg["survival_min"]:
            if state["height_level"] + 1 < len(self.scales):
                state["height_level"] += 1
                changed = True
        elif height > self.height_regress[state["height_level"]] or survival < .9:
            changed = state["height_level"] != 0
            state["height_level"] = 0
        if changed:
            state["last_change"] = float(reference_update)
            self.height_scale[self.indices[name]] = self.scales[state["height_level"]]
            self.height_width[self.indices[name]] = self.widths[state["height_level"]]
            window.clear()

    def state_dict(self):
        return {"version": 1, "groups": {name: {**deepcopy(self.state[name]), "window": list(self.windows[name])}
                                         for name in self.names}}

    def load_state_dict(self, saved):
        if saved.get("version") != 1 or set(saved["groups"]) != set(self.names):
            raise ValueError("Curriculum checkpoint group/schema mismatch")
        for name in self.names:
            state = deepcopy(saved["groups"][name])
            window = state.pop("window")
            if not (0 <= state["height_level"] < len(self.scales)
                    and 0 <= state["spin_level"] < len(self.spin_levels[name])):
                raise ValueError("Invalid restored curriculum level")
            self.state[name] = state
            self.windows[name] = deque(window, maxlen=self.window_size)
            self.height_scale[self.indices[name]] = self.scales[state["height_level"]]
            self.height_width[self.indices[name]] = self.widths[state["height_level"]]
        self.error_sum.zero_()
        self.frames.zero_()
        self.total_frames.zero_()
        self.env_height_scale.copy_(self.height_scale[self.group_ids])
        self.env_height_width.copy_(self.height_width[self.group_ids])

    def report(self):
        result = {}
        for name in self.names:
            state, window = self.state[name], self.windows[name]
            prefix = "Curriculum/" + name + "/"
            result[prefix + "height_scale"] = self.scales[state["height_level"]]
            result[prefix + "height_kernel_width_m"] = self.widths[state["height_level"]]
            result[prefix + "height_pass_m"] = self.height_pass[state["height_level"]]
            result[prefix + "yaw_cap_rad_s"] = self.spin_levels[name][state["spin_level"]]
            result[prefix + "completed_episodes"] = state["episodes"]
            result[prefix + "window_episodes"] = len(window)
            if window:
                result[prefix + "height_mae_m"] = sum(row[0] for row in window) / len(window)
                result[prefix + "velocity_mae_m_s"] = sum(row[1] for row in window) / len(window)
                result[prefix + "yaw_mae_rad_s"] = sum(row[2] for row in window) / len(window)
                result[prefix + "supported_completion_fraction"] = sum(row[3] for row in window) / len(window)
        return result
