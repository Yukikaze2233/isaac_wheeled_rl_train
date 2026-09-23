"""Performance-driven command frontiers with explicit full-domain exposure."""
from collections import deque
from copy import deepcopy
import math

import torch


class AdaptiveCommandCurriculum:
    """Change commands at reset, using completed current-frontier episodes only."""

    def __init__(self, groups, specs, settings, device):
        self.cfg = deepcopy(settings)
        self.names = list(dict.fromkeys(groups))
        self.indices = {name: index for index, name in enumerate(self.names)}
        self.group_ids = torch.tensor([self.indices[name] for name in groups], device=device)
        self.specs = specs
        self.window_size = settings["window_episodes"]
        self.initial = settings["initial_caps"]
        self.increments = settings["cap_increments"]
        if (not isinstance(self.window_size, int) or self.window_size < 1
                or len(self.initial) != 2 or len(self.increments) != 2
                or any(not math.isfinite(x) or x <= 0 for x in self.initial + self.increments)):
            raise ValueError("Adaptive commands require a positive window and finite two-axis caps/increments")
        if not (0 < settings["full_domain_fraction"] < 1
                and 0 < settings["frontier_fraction_min"] <= settings["frontier_fraction_initial"]
                <= settings["frontier_fraction_max"] < 1 - settings["full_domain_fraction"]
                and 0 < settings["survival_regress"] < settings["survival_pass"] <= 1):
            raise ValueError("Invalid adaptive command mixture or completion hysteresis")
        for key in ("policy_dt", "min_reference_updates", "settle_seconds", "minimum_scored_seconds",
                    "height_pass_m", "height_regress_m", "velocity_pass_m_s", "yaw_pass_rad_s", "fraction_step"):
            if not math.isfinite(settings[key]) or settings[key] <= 0:
                raise ValueError("Adaptive timing and tracking tolerances must be finite and positive")
        if settings["height_regress_m"] <= settings["height_pass_m"]:
            raise ValueError("Height regression must have hysteresis")
        self.maximum = {name: [abs(x) for x in specs[name]["command"][:2]] for name in self.names}
        self.state = {name: {"caps": [min(a, b) for a, b in zip(self.initial, self.maximum[name])],
                            "frontier_fraction": settings["frontier_fraction_initial"],
                            "revision": 0, "last_change": 0., "episodes": 0,
                            "easy_episodes": 0, "frontier_episodes": 0, "full_domain_episodes": 0,
                            "stale_frontier_episodes": 0} for name in self.names}
        self.windows = {name: deque(maxlen=self.window_size) for name in self.names}
        count = len(groups)
        self.env_height_scale = torch.ones(count, device=device)
        self.env_height_width = torch.full((count,), math.sqrt(.001), device=device)
        self.error_sum = torch.zeros(count, 3, dtype=torch.float64, device=device)
        self.frames = torch.zeros(count, dtype=torch.int64, device=device)
        self.age = torch.zeros_like(self.frames)
        self.env_revision = torch.zeros_like(self.frames)
        # 0: easy, 1: frontier, 2: full domain. Latched for the entire episode.
        self.env_pool = torch.zeros_like(self.frames)
        self.pool_frames = torch.zeros(len(self.names), 3, dtype=torch.int64, device=device)
        self.spin = torch.tensor([specs[name]["kind"] == "spin_translate" for name in groups], device=device)

    def limit_command(self, name, command):
        # Per-environment mixtures are applied after the ordinary sampler.
        return list(command)

    def sample_commands(self, name, ids, commands, random):
        state = self.state[name]
        pick = random(len(ids))
        full = pick < self.cfg["full_domain_fraction"]
        frontier = (~full) & (pick < self.cfg["full_domain_fraction"] + state["frontier_fraction"])
        cap = commands.new_tensor(state["caps"])
        # Easy samples keep both command directions and low-amplitude control alive.
        easy_cap = commands.new_tensor([min(a, b) for a, b in zip(self.initial, self.maximum[name])])
        easy = easy_cap * (.4 + .6 * random(len(ids)))[:, None]
        magnitude = torch.where(frontier[:, None], cap, easy)
        bounded = commands[:, :2].sign() * torch.minimum(commands[:, :2].abs(), magnitude)
        commands[:, :2] = torch.where(full[:, None], commands[:, :2], bounded)
        self.env_pool[ids] = torch.where(full, 2, torch.where(frontier, 1, 0))
        self.env_revision[ids] = state["revision"]
        return commands

    def reset_episodes(self, ids):
        self.error_sum[ids] = 0.
        self.frames[ids] = 0
        self.age[ids] = 0

    def observe(self, height_error, velocity_error, yaw_error, supported, done, failed, reference_update,
                reference_velocity_error=None):
        self.age += 1
        self.pool_frames.view(-1).scatter_add_(0, self.group_ids * 3 + self.env_pool, torch.ones_like(self.env_pool))
        if reference_velocity_error is not None:
            velocity_error = torch.where(self.spin, reference_velocity_error, velocity_error)
        eligible = self.age * self.cfg["policy_dt"] > self.cfg["settle_seconds"]
        valid = supported.bool() & eligible
        errors = torch.stack((height_error.abs(), velocity_error.abs(), yaw_error.abs()), -1)
        self.error_sum += errors.double() * valid[:, None]
        self.frames += valid
        ids = done.nonzero(as_tuple=False).flatten()
        if not len(ids):
            return
        mean = self.error_sum[ids] / self.frames[ids, None].clamp_min(1)
        minimum = math.ceil(self.cfg["minimum_scored_seconds"] / self.cfg["policy_dt"])
        support_fraction = self.frames[ids] / (self.age[ids] - round(self.cfg["settle_seconds"] / self.cfg["policy_dt"])).clamp_min(1)
        completed = (~failed[ids] & (self.frames[ids] >= minimum)
                     & (support_fraction >= self.cfg.get("support_fraction_min", .8)))
        mean = torch.where((self.frames[ids] >= minimum)[:, None], mean, torch.full_like(mean, 10.))
        rows = torch.cat((self.group_ids[ids, None], self.env_pool[ids, None], self.env_revision[ids, None],
                          mean, completed[:, None]), -1).cpu().tolist()
        touched = set()
        for group, pool, revision, height, velocity, yaw, survived in rows:
            name = self.names[int(group)]
            state = self.state[name]
            state["episodes"] += 1
            state[("easy_episodes", "frontier_episodes", "full_domain_episodes")[int(pool)]] += 1
            if pool != 1:
                continue
            if revision != state["revision"]:
                state["stale_frontier_episodes"] += 1
                continue
            self.windows[name].append([height, velocity, yaw, survived])
            touched.add(name)
        self.reset_episodes(ids)
        for name in touched:
            self._adapt(name, reference_update)

    def _adapt(self, name, reference_update):
        state, window = self.state[name], self.windows[name]
        if len(window) < self.window_size or reference_update - state["last_change"] < self.cfg["min_reference_updates"]:
            return
        height, velocity, yaw, survival = [sum(row[i] for row in window) / len(window) for i in range(4)]
        vx_cap, yaw_cap = state["caps"]
        velocity_limit = max(self.cfg["velocity_pass_m_s"], .075 * vx_cap)
        yaw_limit = max(self.cfg["yaw_pass_rad_s"], .075 * yaw_cap)
        passed = (survival >= self.cfg["survival_pass"] and height <= self.cfg["height_pass_m"]
                  and velocity <= velocity_limit and yaw <= yaw_limit)
        regressed = (survival < self.cfg["survival_regress"] or height > self.cfg["height_regress_m"]
                     or velocity > 2 * velocity_limit or yaw > 2 * yaw_limit)
        old = (list(state["caps"]), state["frontier_fraction"])
        if passed:
            state["caps"] = [min(limit, cap + step) for cap, step, limit in
                             zip(state["caps"], self.increments, self.maximum[name])]
            state["frontier_fraction"] = min(self.cfg["frontier_fraction_max"], state["frontier_fraction"] + self.cfg["fraction_step"])
        elif regressed:
            state["caps"] = [max(min(initial, limit), cap - step) for cap, step, initial, limit in
                             zip(state["caps"], self.increments, self.initial, self.maximum[name])]
            state["frontier_fraction"] = max(self.cfg["frontier_fraction_min"], state["frontier_fraction"] - self.cfg["fraction_step"])
        if old != (state["caps"], state["frontier_fraction"]):
            state["revision"] += 1
            state["last_change"] = float(reference_update)
            window.clear()

    def state_dict(self):
        return {"version": 2, "kind": "adaptive_commands",
                "groups": {name: {**deepcopy(self.state[name]), "window": list(self.windows[name]),
                                  "exposure_frames": self.pool_frames[self.indices[name]].cpu().tolist()}
                           for name in self.names}}

    def load_state_dict(self, saved):
        if saved.get("version") != 2 or saved.get("kind") != "adaptive_commands" or set(saved["groups"]) != set(self.names):
            raise ValueError("Adaptive command checkpoint schema/group mismatch")
        for name in self.names:
            state = deepcopy(saved["groups"][name])
            window = state.pop("window")
            exposure = state.pop("exposure_frames")
            if (len(state["caps"]) != 2 or any(not math.isfinite(cap) or not min(initial, maximum) <= cap <= maximum
                    for cap, initial, maximum in zip(state["caps"], self.initial, self.maximum[name]))
                    or not self.cfg["frontier_fraction_min"] <= state["frontier_fraction"] <= self.cfg["frontier_fraction_max"]
                    or len(exposure) != 3 or any(not isinstance(n, int) or n < 0 for n in exposure)
                    or len(window) > self.window_size or any(len(row) != 4 or not all(math.isfinite(x) for x in row) for row in window)):
                raise ValueError("Invalid adaptive command checkpoint state")
            self.state[name] = state
            self.windows[name] = deque(window, maxlen=self.window_size)
            self.pool_frames[self.indices[name]] = torch.tensor(exposure, device=self.group_ids.device)
        self.reset_episodes(torch.arange(len(self.group_ids), device=self.group_ids.device))

    def report(self):
        result = {}
        for name in self.names:
            state, window = self.state[name], self.windows[name]
            prefix = "Curriculum/" + name + "/"
            result.update({prefix + "vx_cap_m_s": state["caps"][0], prefix + "yaw_cap_rad_s": state["caps"][1],
                           prefix + "frontier_fraction": state["frontier_fraction"],
                           prefix + "full_domain_fraction": self.cfg["full_domain_fraction"],
                           prefix + "window_episodes": len(window), prefix + "revision": state["revision"]})
            for key in ("episodes", "easy_episodes", "frontier_episodes", "full_domain_episodes", "stale_frontier_episodes"):
                result[prefix + key] = state[key]
            exposure = self.pool_frames[self.indices[name]].cpu().tolist()
            for index, pool in enumerate(("easy", "frontier", "full_domain")):
                result[prefix + pool + "_frames"] = exposure[index]
                result[prefix + pool + "_frame_fraction"] = exposure[index] / max(sum(exposure), 1)
            if window:
                for index, metric in enumerate(("height_mae_m", "velocity_mae_m_s", "yaw_mae_rad_s", "completion_fraction")):
                    result[prefix + metric] = sum(row[index] for row in window) / len(window)
        return result
