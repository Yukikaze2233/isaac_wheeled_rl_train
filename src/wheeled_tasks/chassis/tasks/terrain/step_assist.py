"""Forward terrain samples to a bounded height command; no simulator dependency."""
import math

import torch


JUMP_ASSIST_V1 = {
    "enabled": True,
    "actor_updates": 300,
    "probability_start": .8,
    "probability_end": .2,
    "force_body_weight_fraction": .35,
    "impulse_release_momentum_fraction": .25,
    "max_application_seconds": .12,
    "scope": "training_only_engineering_scaffold_not_identified_actuator_force",
}


class JumpAssist:
    """Finite early-training support impulse; never apply force in flight.

    The schedule is driven by committed actor updates, not episode age or
    critic-only updates. Selection is once per reset; force decays to zero even
    for episodes spanning update 300. The environment owns physical phase and
    applies the returned world-up force to the base. Evaluations always pass
    evaluation=True, which hard-disables both selection and application.
    """
    def __init__(self, count, device, config):
        self.cfg = dict(config)
        self.component_settings = None
        if not isinstance(config["actor_updates"], int) or config["actor_updates"] <= 0:
            raise ValueError("Jump assistance requires a positive actor-update window")
        if not 0 <= config["probability_end"] <= config["probability_start"] <= .8:
            raise ValueError("Jump assistance must retain at least 20% unassisted sampling")
        for key in ("force_body_weight_fraction", "impulse_release_momentum_fraction", "max_application_seconds"):
            if not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"Jump assistance requires positive finite {key}")
        if config["impulse_release_momentum_fraction"] >= 1.:
            raise ValueError("Assistance cannot supply the whole requested release momentum")
        self.selected = torch.zeros(count, device=device, dtype=torch.bool)
        self.assisted = torch.zeros_like(self.selected)
        self.impulse = torch.zeros(count, device=device)
        self.application_seconds = torch.zeros_like(self.impulse)
        self.last_force = torch.zeros_like(self.impulse)

    def configure_components(self, lane_components, settings):
        """Cache per-lane physical parameters; physics substeps stay vectorized."""
        if len(lane_components) != len(self.selected) or set(lane_components) - {None} - set(settings):
            raise ValueError("Jump assist ownership must cover every lane")
        self.component_settings = {name: dict(value) for name, value in settings.items()}
        for value in settings.values():
            JumpAssist(1, self.selected.device, value)  # Reuse the native validator once.
        self.component_masks = {name: torch.tensor([owner == name for owner in lane_components],
            device=self.selected.device, dtype=torch.bool) for name in settings}
        self.scoped = {}
        for key in ("enabled", "actor_updates", "probability_start", "probability_end",
                    "force_body_weight_fraction", "impulse_release_momentum_fraction", "max_application_seconds"):
            self.scoped[key] = self.impulse.new_tensor([settings[owner][key] if owner is not None
                else (1 if key == "actor_updates" else 0) for owner in lane_components])

    def actor_clock_tensor(self, clocks):
        if not isinstance(clocks, dict) or set(clocks) != set(self.component_settings):
            raise ValueError("Jump assist clocks must match the bound components")
        result = torch.zeros_like(self.impulse)
        for name, value in clocks.items():
            if type(value) is not int or value < 0:
                raise ValueError("Jump assistance needs nonnegative component actor clocks")
            result[self.component_masks[name]] = value
        return result

    def schedule(self, actor_updates, evaluation=False):
        if self.component_settings is not None:
            if isinstance(actor_updates, dict):
                actor_updates = self.actor_clock_tensor(actor_updates)
            if not torch.is_tensor(actor_updates) or actor_updates.shape != self.impulse.shape:
                raise ValueError("Scoped jump assistance needs one actor clock per lane")
            fraction = (actor_updates / self.scoped["actor_updates"]).clamp(0., 1.)
            probability = self.scoped["probability_start"] + fraction * (
                self.scoped["probability_end"] - self.scoped["probability_start"])
            enabled = self.scoped["enabled"].bool() & (actor_updates < self.scoped["actor_updates"])
            if evaluation:
                enabled = torch.zeros_like(enabled)
            return probability * enabled, (1. - fraction) * enabled
        if actor_updates < 0:
            raise ValueError("Actor updates cannot be negative")
        if evaluation or not self.cfg["enabled"] or actor_updates >= self.cfg["actor_updates"]:
            return 0., 0.
        fraction = actor_updates / self.cfg["actor_updates"]
        probability = self.cfg["probability_start"] + fraction * (
            self.cfg["probability_end"] - self.cfg["probability_start"])
        return probability, 1. - fraction

    def reset(self, ids, actor_updates, random_values, *, evaluation=False):
        probability, _ = self.schedule(actor_updates, evaluation)
        self.selected[ids] = random_values < (probability[ids] if torch.is_tensor(probability) else probability)
        self.assisted[ids] = False
        self.impulse[ids] = 0.
        self.application_seconds[ids] = 0.
        self.last_force[ids] = 0.

    def force(self, mass, apex, pushing, supported, airborne, dt, actor_updates, *, evaluation=False):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Jump assistance requires positive finite physics dt")
        _, amplitude = self.schedule(actor_updates, evaluation)
        parameters = self.scoped if self.component_settings is not None else self.cfg
        release_speed = (2 * 9.81 * apex.clamp_min(0.)).sqrt()
        maximum_impulse = parameters["impulse_release_momentum_fraction"] * mass * release_speed
        remaining = (maximum_impulse - self.impulse).clamp_min(0.)
        remaining_seconds = (parameters["max_application_seconds"] - self.application_seconds).clamp_min(0.)
        force = amplitude * parameters["force_body_weight_fraction"] * mass * 9.81
        force = torch.minimum(force, remaining / dt) * (remaining_seconds / dt).clamp_max(1.)
        active = self.selected & pushing.bool() & supported.bool() & ~airborne.bool()
        self.last_force.copy_(force * active)
        applied = self.last_force > 0.
        self.assisted |= applied
        self.impulse += self.last_force * dt
        self.application_seconds += torch.where(applied, remaining_seconds.clamp_max(dt), 0.)
        return self.last_force

    def metrics(self):
        return {"/jump_assist/selected_fraction": self.selected.float().mean(),
            "/jump_assist/assisted_fraction": self.assisted.float().mean(),
            "/jump_assist/force_N": self.last_force.mean(),
            "/jump_assist/impulse_N_s": self.impulse.mean()}


class StepHeightAssist:
    """Temporal edge detection with per-environment history and integer hold ticks.

    Inputs are world-height samples [m] at the two forward wheel probes and unit
    planar travel directions. The caller owns sensing and the raw command. A
    missing sample invalidates history, but an existing hold expires normally.
    """

    def __init__(self, count, device, dt, config):
        self.cfg = dict(config)
        self.low, self.high = config["step_range_m"]
        self.bias, self.maximum = config["height_bias_m"], config["height_max_m"]
        self.tolerance = config.get("edge_tolerance_m", 1e-5)
        if not all(math.isfinite(v) for v in (dt, self.low, self.high, self.bias, self.maximum,
                self.tolerance, config["hold_seconds"], config["lookahead_m"])):
            raise ValueError("Step assistance requires finite parameters")
        if not (dt > 0 and 0 < self.low <= self.high and self.bias > 0 and self.maximum > self.bias
                and 0 <= self.tolerance < self.low and config["hold_seconds"] >= dt
                and config["lookahead_m"] > 0):
            raise ValueError("Invalid step detection or hold domain")
        self.hold_ticks = math.ceil(config["hold_seconds"] / dt - 1e-9)
        self.remaining = torch.zeros(count, dtype=torch.long, device=device)
        self.previous = torch.zeros(count, 2, device=device)
        self.direction = torch.zeros_like(self.previous)
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.triggered = torch.zeros_like(self.valid)
        self.wall = torch.zeros_like(self.valid)
        self.active = torch.zeros_like(self.valid)
        self.all_ids = torch.arange(count, device=device)

    def reset(self, ids):
        self.remaining[ids] = 0
        self.valid[ids] = False
        self.triggered[ids] = False
        self.wall[ids] = False
        self.active[ids] = False
        self.direction[ids] = 0.

    def update(self, raw_height, samples, valid, direction, eligible, enabled, *, ids=None, advance=True):
        ids = self.all_ids if ids is None else ids
        if advance:
            self.remaining[ids] = (self.remaining[ids] - 1).clamp_min(0)
        finite = torch.isfinite(samples[ids]).all(-1) & torch.isfinite(direction[ids]).all(-1)
        good = valid[ids].all(-1) & finite & eligible[ids]
        same_heading = (self.direction[ids] * direction[ids]).sum(-1) >= math.cos(math.pi / 4)
        direction_valid = direction[ids].norm(dim=-1) > .5
        changed_heading = (self.direction[ids].norm(dim=-1) > .5) & direction_valid & ~same_heading
        self.remaining[ids] = torch.where(changed_heading, 0, self.remaining[ids])
        delta = (samples[ids] - self.previous[ids]).amax(-1)
        observable_edge = good & self.valid[ids] & same_heading
        self.wall[ids] = observable_edge & (delta > self.high + self.tolerance)
        self.triggered[ids] = (observable_edge & enabled[ids] & ~self.wall[ids]
            & (delta >= self.low - self.tolerance) & (delta <= self.high + self.tolerance))
        self.remaining[ids] = torch.where(self.triggered[ids], self.hold_ticks, self.remaining[ids])
        self.remaining[ids] = torch.where(self.wall[ids] | ~enabled[ids], 0, self.remaining[ids])
        self.active[ids] = self.remaining[ids] > 0
        self.previous[ids] = torch.where(good[:, None], samples[ids], 0.)
        self.direction[ids] = torch.where((finite & direction_valid)[:, None], direction[ids], self.direction[ids])
        self.valid[ids] = good
        # Bias is always applied to raw input, never to last tick's effective height.
        return torch.where(self.active[ids], (raw_height[ids] + self.bias).clamp_max(self.maximum), raw_height[ids])

    def metrics(self):
        return {"/terrain/step_assist_active_fraction": self.active.float().mean(),
                "/terrain/step_detected_fraction": self.triggered.float().mean(),
                "/terrain/wall_detected_fraction": self.wall.float().mean(),
                "/terrain/forward_sample_valid_fraction": self.valid.float().mean()}
