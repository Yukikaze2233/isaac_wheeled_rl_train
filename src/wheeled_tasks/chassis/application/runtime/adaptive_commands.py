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
        self.retained = set(settings.get("retained_groups", []))
        if self.retained - set(self.names):
            raise ValueError("Unknown retained curriculum group")
        self.frontier_evidence = settings.get("frontier_evidence")
        self.state_version = 3 if self.frontier_evidence or self.retained else 2
        self.frontier_range = settings.get("frontier_amplitude_range", [1., 1.])
        if not (len(self.frontier_range) == 2 and 0 < self.frontier_range[0] <= self.frontier_range[1] <= 1):
            raise ValueError("Frontier amplitude range must lie in (0,1]")
        self.height_course = settings.get("height_course")
        if self.height_course:
            widths, scales, passes = (self.height_course[key] for key in ("widths_m", "scales", "pass_m"))
            if (not widths or not len(widths) == len(scales) == len(passes)
                    or any(not math.isfinite(x) or x <= 0 for x in widths + scales + passes)):
                raise ValueError("Height course requires aligned positive finite widths, scales and thresholds")
        self.state = {name: {"caps": [min(a, b) for a, b in zip(self.initial, self.maximum[name])],
                            "frontier_fraction": settings["frontier_fraction_initial"],
                            "revision": 0, "last_change": 0., "episodes": 0,
                            "easy_episodes": 0, "frontier_episodes": 0, "full_domain_episodes": 0,
                            "stale_frontier_episodes": 0} for name in self.names}
        for name in self.retained:
            self.state[name]["caps"] = list(self.maximum[name])
        if self.state_version == 3:
            for state in self.state.values():
                state["last_regression_change"] = 0.
                state["unqualified_episodes"] = 0
        self._changes = []
        self.windows = {name: deque(maxlen=self.window_size) for name in self.names}
        if self.height_course:
            for state in self.state.values():
                state["height_level"] = 0
        count = len(groups)
        self.env_height_scale = torch.ones(count, device=device)
        self.env_height_width = torch.full((count,), math.sqrt(.001), device=device)
        self.group_height_scale = torch.ones(len(self.names), device=device)
        self.group_height_width = torch.full((len(self.names),), math.sqrt(.001), device=device)
        if self.height_course:
            self.env_height_scale.fill_(self.height_course["scales"][0])
            self.env_height_width.fill_(self.height_course["widths_m"][0])
            self.group_height_scale.fill_(self.height_course["scales"][0])
            self.group_height_width.fill_(self.height_course["widths_m"][0])
        self.env_caps = torch.zeros(count, 2, device=device)
        self.error_sum = torch.zeros(count, 3, dtype=torch.float64, device=device)
        self.frames = torch.zeros(count, dtype=torch.int64, device=device)
        self.age = torch.zeros_like(self.frames)
        self.supported_frames = torch.zeros_like(self.frames)
        self.boundary_frames = torch.zeros_like(self.frames)
        self.steady_ticks = torch.zeros_like(self.frames)
        self.previous_command = torch.zeros(count, 2, device=device)
        self.command_sum = torch.zeros(count, 2, dtype=torch.float64, device=device)
        self.episode_horizon = torch.tensor([specs[name].get("episode_seconds", 20.) for name in groups], device=device)
        if self.frontier_evidence:
            evidence = self.frontier_evidence
            if not (0 < evidence["minimum_command_fraction"] <= 1 and evidence["steady_seconds"] > 0
                    and 0 < evidence["minimum_episode_fraction"] <= 1):
                raise ValueError("Invalid frontier evidence requirements")
        self.env_revision = torch.zeros_like(self.frames)
        # 0: easy, 1: frontier, 2: full domain. Latched for the entire episode.
        self.env_pool = torch.zeros_like(self.frames)
        self.pool_frames = torch.zeros(len(self.names), 3, dtype=torch.int64, device=device)
        self.spin = torch.tensor([specs[name]["kind"] == "spin_translate" for name in groups], device=device)

    def limit_command(self, name, command):
        # Per-environment mixtures are applied after the ordinary sampler.
        return list(command)

    def sample_commands(self, name, ids, commands, random, reset=True):
        state = self.state[name]
        if name in self.retained:
            self.env_pool[ids] = 2
            self.env_caps[ids] = commands.new_tensor(self.maximum[name])
            self.env_revision[ids] = state["revision"]
            return commands
        pick = random(len(ids))
        full = pick < self.cfg["full_domain_fraction"]
        frontier = (~full) & (pick < self.cfg["full_domain_fraction"] + state["frontier_fraction"])
        pool = torch.where(full, 2, torch.where(frontier, 1, 0))
        reset_mask = torch.full_like(full, reset) if isinstance(reset, bool) else reset.bool()
        if not self.cfg.get("latch_command_pool", False):
            reset_mask = torch.ones_like(full)
        self.env_pool[ids] = torch.where(reset_mask, pool, self.env_pool[ids])
        self.env_revision[ids] = torch.where(reset_mask, state["revision"], self.env_revision[ids])
        self.env_caps[ids] = torch.where(reset_mask[:, None], commands.new_tensor(state["caps"]), self.env_caps[ids])
        full, frontier = self.env_pool[ids] == 2, self.env_pool[ids] == 1
        cap = self.env_caps[ids]
        if self.frontier_range != [1., 1.]:
            low, high = self.frontier_range
            cap = cap * (low + (high - low) * random(len(ids)))[:, None]
        # Easy samples keep both command directions and low-amplitude control alive.
        easy_cap = commands.new_tensor([min(a, b) for a, b in zip(self.initial, self.maximum[name])])
        easy = easy_cap * (.4 + .6 * random(len(ids)))[:, None]
        magnitude = torch.where(frontier[:, None], cap, easy)
        bounded = commands[:, :2].sign() * torch.minimum(commands[:, :2].abs(), magnitude)
        commands[:, :2] = torch.where(full[:, None], commands[:, :2], bounded)
        return commands

    def reset_episodes(self, ids):
        if not len(ids):
            return
        self.error_sum[ids] = 0.
        self.frames[ids] = 0
        self.age[ids] = 0
        self.supported_frames[ids] = 0
        self.boundary_frames[ids] = 0
        self.steady_ticks[ids] = 0
        self.previous_command[ids] = 0.
        self.command_sum[ids] = 0.
        if self.height_course:
            self.env_height_scale[ids] = self.group_height_scale[self.group_ids[ids]]
            self.env_height_width[ids] = self.group_height_width[self.group_ids[ids]]

    def observe(self, height_error, velocity_error, yaw_error, supported, done, failed, reference_update,
                reference_velocity_error=None, commands=None):
        self.age += 1
        self.pool_frames.view(-1).scatter_add_(0, self.group_ids * 3 + self.env_pool, torch.ones_like(self.env_pool))
        if reference_velocity_error is not None:
            velocity_error = torch.where(self.spin, reference_velocity_error, velocity_error)
        eligible = self.age * self.cfg["policy_dt"] > self.cfg["settle_seconds"]
        valid = supported.bool() & eligible
        self.supported_frames += valid
        if self.frontier_evidence:
            if commands is None:
                raise ValueError("Frontier scoring requires the commands actually seen by the policy")
            active_axes = self.env_caps > 0
            actual = commands[:, :2]
            at_boundary = ((actual.abs() >= self.env_caps * self.frontier_evidence["minimum_command_fraction"])
                           | ~active_axes).all(-1)
            steady = (((actual - self.previous_command).abs() < 1e-5) | ~active_axes).all(-1)
            self.steady_ticks = torch.where(steady, self.steady_ticks + 1, 0)
            self.previous_command.copy_(actual)
            self.boundary_frames += at_boundary
            valid &= at_boundary & (self.steady_ticks * self.cfg["policy_dt"] >= self.frontier_evidence["steady_seconds"])
            self.command_sum += actual.double().abs() * valid[:, None]
        errors = torch.stack((height_error.abs(), velocity_error.abs(), yaw_error.abs()), -1)
        self.error_sum += errors.double() * valid[:, None]
        self.frames += valid
        ids = done.nonzero(as_tuple=False).flatten()
        if not len(ids):
            return
        mean = self.error_sum[ids] / self.frames[ids, None].clamp_min(1)
        minimum = math.ceil(self.cfg["minimum_scored_seconds"] / self.cfg["policy_dt"])
        support_fraction = self.supported_frames[ids] / (self.age[ids] - round(self.cfg["settle_seconds"] / self.cfg["policy_dt"])).clamp_min(1)
        completed = (~failed[ids] & (self.frames[ids] >= minimum)
                     & (support_fraction >= self.cfg.get("support_fraction_min", .8)))
        mean = torch.where((self.frames[ids] >= minimum)[:, None], mean, torch.full_like(mean, 10.))
        qualified = torch.ones_like(completed)
        evidence_rows = None
        if self.frontier_evidence:
            enough_time = self.age[ids] * self.cfg["policy_dt"] >= self.episode_horizon[ids] * self.frontier_evidence["minimum_episode_fraction"]
            qualified = failed[ids] | (enough_time & (self.boundary_frames[ids] > 0))
            completed &= enough_time
            evidence_rows = torch.cat((self.frames[ids, None] * self.cfg["policy_dt"],
                self.age[ids, None] * self.cfg["policy_dt"], self.command_sum[ids] / self.frames[ids, None].clamp_min(1)), -1).cpu().tolist()
        rows = torch.cat((self.group_ids[ids, None], self.env_pool[ids, None], self.env_revision[ids, None],
                          mean, completed[:, None]), -1).cpu().tolist()
        touched = set()
        qualified_rows = qualified.cpu().tolist()
        for index, (group, pool, revision, height, velocity, yaw, survived) in enumerate(rows):
            name = self.names[int(group)]
            state = self.state[name]
            state["episodes"] += 1
            state[("easy_episodes", "frontier_episodes", "full_domain_episodes")[int(pool)]] += 1
            if pool != 1:
                continue
            if revision != state["revision"]:
                state["stale_frontier_episodes"] += 1
                continue
            if not qualified_rows[index]:
                state["unqualified_episodes"] += 1
                continue
            self.windows[name].append([height, velocity, yaw, survived] + (evidence_rows[index] if evidence_rows is not None else []))
            touched.add(name)
        self.reset_episodes(ids)
        for name in touched:
            self._adapt(name, reference_update)

    def _adapt(self, name, reference_update):
        state, window = self.state[name], self.windows[name]
        if len(window) < self.window_size or reference_update - state["last_change"] < self.cfg["min_reference_updates"]:
            return
        height, velocity, yaw, survival = [sum(row[i] for row in window) / len(window) for i in range(4)]
        scored_seconds = sum(row[4] for row in window) / len(window) if self.frontier_evidence else None
        vx_cap, yaw_cap = state["caps"]
        if self.frontier_evidence:
            vx_cap, yaw_cap = [sum(row[index] for row in window) / len(window) for index in (6, 7)]
        velocity_limit = max(self.cfg["velocity_pass_m_s"], .075 * vx_cap)
        if self.specs[name]["kind"] in ("stand", "height"):
            velocity_limit = self.cfg.get("stationary_velocity_pass_m_s", velocity_limit)
        yaw_limit = max(self.cfg["yaw_pass_rad_s"], .075 * yaw_cap)
        passed = (survival >= self.cfg["survival_pass"] and height <= self.cfg["height_pass_m"]
                  and velocity <= velocity_limit and yaw <= yaw_limit)
        regressed = (survival < self.cfg["survival_regress"] or height > self.cfg["height_regress_m"]
                     or velocity > 2 * velocity_limit or yaw > 2 * yaw_limit)
        old = (list(state["caps"]), state["frontier_fraction"], state.get("height_level", 0))
        if passed:
            state["caps"] = [min(limit, cap + step) for cap, step, limit in
                             zip(state["caps"], self.increments, self.maximum[name])]
            state["frontier_fraction"] = min(self.cfg["frontier_fraction_max"], state["frontier_fraction"] + self.cfg["fraction_step"])
            if self.height_course and height <= self.height_course["pass_m"][state["height_level"]]:
                state["height_level"] = min(state["height_level"] + 1, len(self.height_course["widths_m"]) - 1)
        elif regressed:
            regression_ready = reference_update - state.get("last_regression_change", 0.) >= self.cfg.get("min_regression_reference_updates", 0.)
            if regression_ready:
                state["caps"] = [max(min(initial, limit), cap - step) for cap, step, initial, limit in
                                 zip(state["caps"], self.increments, self.initial, self.maximum[name])]
                if self.state_version == 3 and state["caps"] != old[0]:
                    state["last_regression_change"] = float(reference_update)
            state["frontier_fraction"] = max(self.cfg["frontier_fraction_min"], state["frontier_fraction"] - self.cfg["fraction_step"])
            if self.height_course:
                state["height_level"] = max(0, state["height_level"] - 1)
        if old != (state["caps"], state["frontier_fraction"], state.get("height_level", 0)):
            state["revision"] += 1
            state["last_change"] = float(reference_update)
            window.clear()
            if self.height_course:
                self.group_height_scale[self.indices[name]] = self.height_course["scales"][state["height_level"]]
                self.group_height_width[self.indices[name]] = self.height_course["widths_m"][state["height_level"]]
            if self.cfg.get("snapshot_on_change") and (old[0] != state["caps"] or old[2] != state.get("height_level", 0)):
                self._changes.append({"group": name, "reference_update": float(reference_update), "revision": state["revision"],
                    "old_caps": old[0], "new_caps": list(state["caps"]), "height_level": state.get("height_level", 0),
                    "reason": "advance" if passed else "regress", "window_mean": [height, velocity, yaw, survival],
                    "scored_command_mean": [vx_cap, yaw_cap], "scored_seconds_mean": scored_seconds})

    def drain_changes(self):
        result, self._changes = self._changes, []
        return result

    def state_dict(self):
        return {"version": self.state_version, "kind": "adaptive_commands",
                "groups": {name: {**deepcopy(self.state[name]), "window": list(self.windows[name]),
                                  "exposure_frames": self.pool_frames[self.indices[name]].cpu().tolist()}
                           for name in self.names}}

    def inherit_frontiers(self, saved):
        """Keep learned ranges across scene changes, but restart evidence windows."""
        if saved.get("version") not in (2, 3) or saved.get("kind") != "adaptive_commands":
            raise ValueError("Cannot inherit an incompatible command curriculum")
        merged = self.state_dict()
        inherited = set(self.names) & set(saved["groups"])
        for name in inherited:
            old = saved["groups"][name]
            if len(old["caps"]) != 2 or any(not math.isfinite(value) or value < 0 for value in old["caps"]):
                raise ValueError("Invalid inherited command frontier")
            target = merged["groups"][name]
            target["caps"] = [min(maximum, max(min(initial, maximum), cap))
                              for cap, initial, maximum in zip(old["caps"], self.initial, self.maximum[name])]
            fraction = old["frontier_fraction"]
            if not math.isfinite(fraction):
                raise ValueError("Invalid inherited command mixture")
            target["frontier_fraction"] = max(self.cfg["frontier_fraction_min"], min(self.cfg["frontier_fraction_max"], fraction))
            if self.height_course:
                target["height_level"] = min(len(self.height_course["widths_m"]) - 1, max(0, old.get("height_level", 0)))
        self.load_state_dict(merged)
        return len(inherited)

    def load_state_dict(self, saved):
        if saved.get("version") != self.state_version or saved.get("kind") != "adaptive_commands" or set(saved["groups"]) != set(self.names):
            raise ValueError("Adaptive command checkpoint schema/group mismatch")
        for name in self.names:
            state = deepcopy(saved["groups"][name])
            window = state.pop("window")
            exposure = state.pop("exposure_frames")
            if self.height_course and not 0 <= state.get("height_level", -1) < len(self.height_course["widths_m"]):
                raise ValueError("Invalid adaptive height course level")
            if (len(state["caps"]) != 2 or any(not math.isfinite(cap) or not min(initial, maximum) <= cap <= maximum
                    for cap, initial, maximum in zip(state["caps"], self.initial, self.maximum[name]))
                    or not self.cfg["frontier_fraction_min"] <= state["frontier_fraction"] <= self.cfg["frontier_fraction_max"]
                    or len(exposure) != 3 or any(not isinstance(n, int) or n < 0 for n in exposure)
                    or len(window) > self.window_size or any(len(row) != (8 if self.frontier_evidence else 4)
                        or not all(math.isfinite(x) for x in row) for row in window)):
                raise ValueError("Invalid adaptive command checkpoint state")
            self.state[name] = state
            if self.height_course:
                self.group_height_scale[self.indices[name]] = self.height_course["scales"][state["height_level"]]
                self.group_height_width[self.indices[name]] = self.height_course["widths_m"][state["height_level"]]
            self.windows[name] = deque(window, maxlen=self.window_size)
            self.pool_frames[self.indices[name]] = torch.tensor(exposure, device=self.group_ids.device)
        self.reset_episodes(torch.arange(len(self.group_ids), device=self.group_ids.device))

    def report(self):
        result = {}
        for name in self.names:
            state, window = self.state[name], self.windows[name]
            prefix = "Curriculum/" + name + "/"
            result[prefix + "retained_group"] = int(name in self.retained)
            if self.state_version == 3:
                result[prefix + "unqualified_episodes"] = state["unqualified_episodes"]
            result.update({prefix + "vx_cap_m_s": state["caps"][0], prefix + "yaw_cap_rad_s": state["caps"][1],
                           prefix + "frontier_fraction": state["frontier_fraction"],
                           prefix + "full_domain_fraction": self.cfg["full_domain_fraction"],
                           prefix + "window_episodes": len(window), prefix + "revision": state["revision"]})
            if self.height_course:
                level = state["height_level"]
                result[prefix + "height_level"] = level
                result[prefix + "height_width_m"] = self.height_course["widths_m"][level]
                result[prefix + "height_scale"] = self.height_course["scales"][level]
            for key in ("episodes", "easy_episodes", "frontier_episodes", "full_domain_episodes", "stale_frontier_episodes"):
                result[prefix + key] = state[key]
            exposure = self.pool_frames[self.indices[name]].cpu().tolist()
            for index, pool in enumerate(("easy", "frontier", "full_domain")):
                result[prefix + pool + "_frames"] = exposure[index]
                result[prefix + pool + "_frame_fraction"] = exposure[index] / max(sum(exposure), 1)
            if window:
                for index, metric in enumerate(("height_mae_m", "velocity_mae_m_s", "yaw_mae_rad_s", "completion_fraction")):
                    result[prefix + metric] = sum(row[index] for row in window) / len(window)
                if self.frontier_evidence:
                    for index, metric in enumerate(("scored_seconds", "observed_seconds", "scored_abs_vx_command", "scored_abs_yaw_command"), 4):
                        result[prefix + metric] = sum(row[index] for row in window) / len(window)
        return result
