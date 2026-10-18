"""Pinned natural airborne state and contact history, without command sampling.

The owner does not select a formal profile. Within an episode,
state advances at most once per lane/counter using float32 20 ms additions,
including source timer freezing. Selective reset permits a newborn update at
the same global counter. It does not produce ground scans, map chassis contacts,
sample terrain commands, change actor context or certify a learned jump.
"""
from dataclasses import dataclass
from types import MappingProxyType

import torch


VERSION = "v7-source-airborne-state-v1"
POLICY_DT_S = .02
ENTER = MappingProxyType({"wheel_radius": .06, "body_height_threshold": .3,
                          "wheel_clearance_threshold": .08, "duration_s": .02})
EXIT = MappingProxyType({"wheel_contact_force_threshold": 20.,
    "wheel_contact_height_threshold": .15, "wheel_contact_duration_s": .3,
    "base_contact_force_threshold": 5., "base_contact_duration_s": .25,
    "max_duration_s": 1.})


@dataclass(frozen=True)
class AirbornePhysicalObservation:
    body_relative_ground_height_m: torch.Tensor      # [N], raw world z minus base scan
    wheel_relative_ground_heights_m: torch.Tensor    # [N,2], link origins minus own scans
    wheel_contact_forces_world_n: torch.Tensor       # [N,2,3], current net vectors
    exit_body_contact_forces_world_n: torch.Tensor   # [N,K,3], mapped undesired set, K>0
    commands3: torch.Tensor                          # [N,3], before entry override
    terrain_allowed: torch.Tensor                    # [N], source final is all true
    stair_active: torch.Tensor                       # [N], actual stair FSM state


@dataclass(frozen=True)
class AirborneTransition:
    updated: torch.Tensor
    entered: torch.Tensor
    exited: torch.Tensor
    state: dict[str, torch.Tensor]


class SourceAirborneState:
    def __init__(self, count, device):
        if type(count) is not int or count <= 0:
            raise ValueError("Airborne state requires a positive integer lane count")
        self.device = torch.device(device)
        self.count = count
        self.active = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.device = self.active.device
        self.force_enter_request = torch.zeros_like(self.active)
        self.last_update_step = torch.full((count,), -1, dtype=torch.long, device=self.device)
        self.enter_time = torch.zeros(count, device=self.device, dtype=torch.float32)
        self.wheel_contact_elapsed_s = torch.zeros(count, 2, device=self.device, dtype=torch.float32)
        self.base_contact_elapsed_s = torch.zeros_like(self.enter_time)
        self.current_duration_s = torch.zeros_like(self.enter_time)
        self.maximum_duration_s = torch.zeros_like(self.enter_time)
        self.entry_command3 = torch.zeros(count, 3, device=self.device, dtype=torch.float32)

    def _boolean(self, value, name):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.count,)
                or value.dtype != torch.bool or value.device != self.device):
            raise ValueError(f"Airborne {name} requires one explicit boolean per lane")

    def _validate(self, observation):
        if not isinstance(observation, AirbornePhysicalObservation):
            raise TypeError("Airborne state requires its complete physical observation")
        shapes = {"body_relative_ground_height_m": (self.count,),
            "wheel_relative_ground_heights_m": (self.count, 2),
            "wheel_contact_forces_world_n": (self.count, 2, 3), "commands3": (self.count, 3)}
        for name, shape in shapes.items():
            value = getattr(observation, name)
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.dtype != torch.float32 or value.device != self.device):
                raise ValueError(f"Airborne {name} requires native float32 shape {shape}")
        forces = observation.exit_body_contact_forces_world_n
        if (not isinstance(forces, torch.Tensor) or forces.ndim != 3
                or forces.shape[0] != self.count or forces.shape[1] < 1 or forces.shape[2] != 3
                or forces.dtype != torch.float32 or forces.device != self.device):
            raise ValueError("Airborne exit-body forces require the explicit nonempty mapped body set")
        self._boolean(observation.terrain_allowed, "terrain allowance")
        self._boolean(observation.stair_active, "stair state")
        # Source comparisons retain native nonfinite values. The enclosing
        # physical safety/reset owner must classify them; this observer must
        # not invent repaired heights, forces or a missing contact sensor.

    def request_force_enter(self, mask):
        self._boolean(mask, "force-entry request")
        self.force_enter_request |= mask

    def reset(self, ids):
        for value in (self.active, self.force_enter_request, self.enter_time,
                self.wheel_contact_elapsed_s, self.base_contact_elapsed_s,
                self.current_duration_s, self.maximum_duration_s):
            value[ids] = 0
        self.last_update_step[ids] = -1
        # Source reset leaves entry_command intact. A grounded observation
        # clears it; a new entry replaces it. The current five additions do
        # not read this cache. This is not an exact native restart promise.

    def snapshot(self):
        return {"airborne": self.active.clone(), "enter_time_s": self.enter_time.clone(),
            "wheel_contact_elapsed_s": self.wheel_contact_elapsed_s.clone(),
            "base_contact_elapsed_s": self.base_contact_elapsed_s.clone(),
            "current_duration_s": self.current_duration_s.clone(),
            "maximum_duration_s": self.maximum_duration_s.clone(),
            "entry_command3": self.entry_command3.clone(),
            "pending_force_enter": self.force_enter_request.clone(),
            "last_update_step": self.last_update_step.clone()}

    def observe(self, observation: AirbornePhysicalObservation, *, policy_tick, observation_valid=None):
        if type(policy_tick) is not int or policy_tick < 0:
            raise ValueError("Airborne policy counter must be a nonnegative integer")
        self._validate(observation)
        update = self.last_update_step != policy_tick
        if observation_valid is not None:
            self._boolean(observation_valid, "physical observation validity")
            # Explicit physical/control ownership may exclude this observation.
            # Lifecycle resets belong to the caller. This does not filter PPO,
            # fabricate support, clear a failure or introduce a termination.
            update &= observation_valid
        empty = torch.zeros_like(update)
        if not bool(update.any()):
            return AirborneTransition(update.clone(), empty.clone(), empty, self.snapshot())
        previous = self.active.clone()
        natural = (observation.terrain_allowed & ~observation.stair_active
            & (observation.body_relative_ground_height_m > ENTER["body_height_threshold"])
            & (observation.wheel_relative_ground_heights_m
               > ENTER["wheel_radius"] + ENTER["wheel_clearance_threshold"]).all(-1))
        self.enter_time.copy_(torch.where(update,
            torch.where(natural & ~previous, self.enter_time + POLICY_DT_S, 0.), self.enter_time))
        natural_ready = natural & (self.enter_time >= ENTER["duration_s"])
        forced = observation.terrain_allowed & ~observation.stair_active & self.force_enter_request
        entered = update & (natural_ready | forced) & ~previous
        self.enter_time[entered] = 0.
        self.entry_command3[entered] = observation.commands3[entered]
        self.force_enter_request[update] = False
        wheel_stable = observation.wheel_contact_forces_world_n.norm(dim=-1) > EXIT["wheel_contact_force_threshold"]
        low = observation.wheel_relative_ground_heights_m < EXIT["wheel_contact_height_threshold"]
        wheel_countable = wheel_stable & low
        body_stable = (observation.exit_body_contact_forces_world_n.norm(dim=-1)
                       > EXIT["base_contact_force_threshold"]).any(-1)
        self.wheel_contact_elapsed_s[entered] = 0.
        self.base_contact_elapsed_s[entered] = 0.
        active_for_exit = update & previous & ~entered
        self.wheel_contact_elapsed_s.copy_(torch.where(update[:, None],
            torch.where(active_for_exit[:, None],
                torch.where(wheel_countable, self.wheel_contact_elapsed_s + POLICY_DT_S,
                    torch.where(low, self.wheel_contact_elapsed_s, 0.)),
                self.wheel_contact_elapsed_s), self.wheel_contact_elapsed_s))
        self.base_contact_elapsed_s.copy_(torch.where(update,
            torch.where(active_for_exit & body_stable, self.base_contact_elapsed_s + POLICY_DT_S,
                self.base_contact_elapsed_s), self.base_contact_elapsed_s))
        maximum_exit = active_for_exit & (self.current_duration_s + POLICY_DT_S >= EXIT["max_duration_s"])
        exit_mask = previous & ((self.wheel_contact_elapsed_s >= EXIT["wheel_contact_duration_s"]).any(-1)
            | (self.base_contact_elapsed_s >= EXIT["base_contact_duration_s"]) | maximum_exit
            | observation.stair_active)
        self.active.copy_(torch.where(update, (previous | entered) & ~exit_mask, previous))
        self.entry_command3[update & ~self.active] = 0.
        self.current_duration_s.copy_(torch.where(update,
            torch.where(self.active, self.current_duration_s + POLICY_DT_S, 0.), self.current_duration_s))
        self.maximum_duration_s.copy_(torch.maximum(self.maximum_duration_s, self.current_duration_s))
        self.last_update_step[update] = policy_tick
        return AirborneTransition(update.clone(), entered, update & previous & ~self.active, self.snapshot())
