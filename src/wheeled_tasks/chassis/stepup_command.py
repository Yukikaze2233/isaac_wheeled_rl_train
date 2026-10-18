"""Explicit step/wall events to a latched V7 height command.

No detector, terrain truth, motor trajectory or learned-skill claim. A policy
tick advances once; reading observations does not advance the hold clock.
Formal learning-state resumes reset native episodes, cancelling old holds
and pending events rather than claiming same-episode command continuity.
"""
from collections.abc import Mapping
from copy import deepcopy
import math


VERSION = "v7-stepup-command-v1"


def validate_stepup_command(settings):
    if settings is None:
        return None
    if (not isinstance(settings, Mapping) or settings.get("version") != VERSION
            or type(settings.get("enabled")) is not bool):
        raise ValueError("V7 StepUp requires its version and explicit boolean enabled")
    fields = {"version", "enabled"}
    if settings["enabled"]:
        fields |= {"event_source", "height_bias_m", "hold_seconds", "maximum_height_m"}
        if settings.get("event_source") != "external_step_wall_events":
            raise ValueError("V7 StepUp requires explicit external step/wall events")
        for name in ("height_bias_m", "hold_seconds", "maximum_height_m"):
            value = settings.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"V7 StepUp requires finite explicit {name}")
        if (settings["height_bias_m"] < 0. or settings["hold_seconds"] < 0.
                or not .2 <= settings["maximum_height_m"] <= .42):
            raise ValueError("V7 StepUp requires nonnegative bias/hold and the existing 0.2..0.42m command domain")
        ticks = settings["hold_seconds"] / .02
        if (not math.isfinite(ticks) or round(ticks) > 2**63 - 1
                or settings["hold_seconds"] > 0. and round(ticks) == 0
                or not math.isclose(ticks, round(ticks), rel_tol=0., abs_tol=1e-9)):
            raise ValueError("V7 StepUp hold must be an explicit multiple of 20 ms")
    if set(settings) != fields:
        raise ValueError("V7 StepUp contains missing or unsupported fields")
    return deepcopy(dict(settings))


class StepUpCommand:
    def __init__(self, count, device, settings):
        import torch
        if type(count) is not int or count <= 0:
            raise ValueError("V7 StepUp requires a positive integer lane count")
        self.torch = torch
        self.settings = validate_stepup_command(settings)
        if not self.settings or not self.settings["enabled"]:
            raise ValueError("Construct V7 StepUp only when explicitly enabled")
        self.remaining = torch.zeros(count, dtype=torch.long, device=device)
        self.value = torch.zeros(count, dtype=torch.float32, device=device)
        self.pending_step = torch.zeros(count, dtype=torch.bool, device=device)
        self.pending_wall = torch.zeros_like(self.pending_step)
        self.step_event = torch.zeros_like(self.pending_step)
        self.wall_event = torch.zeros_like(self.pending_step)
        self.last_policy_tick = None
        self.hold_ticks = round(self.settings["hold_seconds"] / .02)

    @property
    def active(self):
        return self.remaining > 0

    def submit(self, step, wall):
        for value in (step, wall):
            if (not isinstance(value, self.torch.Tensor) or value.dtype != self.torch.bool
                    or value.shape != self.pending_step.shape or value.device != self.pending_step.device):
                raise ValueError("V7 StepUp events require one explicit boolean per lane on the command device")
        self.pending_step |= step
        self.pending_wall |= wall

    def reset(self, ids):
        for value in (self.remaining, self.value, self.pending_step, self.pending_wall,
                      self.step_event, self.wall_event):
            value[ids] = 0

    def advance(self, raw_height, eligible, *, policy_tick):
        if (type(policy_tick) is not int or policy_tick < 0
                or self.last_policy_tick is not None and policy_tick != self.last_policy_tick + 1):
            raise ValueError("V7 StepUp advances once at each consecutive policy tick")
        if (raw_height.shape != self.value.shape or raw_height.device != self.value.device
                or not raw_height.is_floating_point() or not bool(self.torch.isfinite(raw_height).all())
                or eligible.shape != self.pending_step.shape or eligible.dtype != self.torch.bool
                or eligible.device != self.pending_step.device):
            raise ValueError("V7 StepUp requires finite current raw heights and explicit lane eligibility")
        self.remaining.sub_(1).clamp_min_(0)
        self.wall_event.copy_(self.pending_wall & eligible)
        self.step_event.copy_(self.pending_step & eligible & ~self.wall_event)
        cancel = self.wall_event | ~eligible
        self.remaining[cancel] = 0
        self.value[cancel] = 0.
        if self.hold_ticks:
            self.value[self.step_event] = (raw_height[self.step_event]
                + self.settings["height_bias_m"]).clamp_max(self.settings["maximum_height_m"])
            self.remaining[self.step_event] = self.hold_ticks
        self.pending_step.zero_()
        self.pending_wall.zero_()
        self.last_policy_tick = policy_tick
        return self.effective_height(raw_height)

    def effective_height(self, raw_height):
        return self.torch.where(self.active, self.value, raw_height)

    def diagnostics(self):
        return {"hold_remaining_policy_ticks": self.remaining.clone(),
                "latched_height_m": self.value.clone(),
                "height_override_active": self.active.clone(), "step_event": self.step_event.clone(),
                "wall_event": self.wall_event.clone()}

    def metrics(self):
        return {"/terrain/stepup_returned_height_hold_fraction": self.active.float().mean(),
                "/terrain/stepup_returned_step_event_fraction": self.step_event.float().mean(),
                "/terrain/stepup_returned_wall_event_fraction": self.wall_event.float().mean()}
