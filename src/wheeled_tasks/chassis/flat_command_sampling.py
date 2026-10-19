"""CPU/Torch command runtime for the pinned reference 0719 flat archive.

The three generator columns are [body vx (m/s), body vy (m/s), yaw (rad/s)].
Height is a separate root-link height target in metres. This module does not
choose or write mechanical reset poses. The caller supplies the ground-birth
mask after its separately reviewed mechanical reset selection.

Ordering: reset -> observations -> begin_policy -> four repetitions of
apply torque using ``holding`` then advance_physics -> pay the interval reward
using the old command -> advance -> reset done lanes -> observations. Reset
does not overwrite the completed interval's learning mask.

CommandTerm.reset in the *current* SDK only resamples; it does not postprocess.
We preserve its mutable cache semantics: the first standing draw leaves that
cache untouched, and ground restore saves that cache, not an invented newly
sampled command. Historical July SDK binaries were not archived. This is a
versioned current-SDK mapping, not a claim of bitwise historical replay.
"""
from __future__ import annotations

import math
import json
from copy import deepcopy

import torch


COMMAND_VERSION = "scut0719_flat_commands_current_sdk_v1"
SOURCE_COMMIT = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
SOURCE_ARCHIVE = "pretrained/26_infantry/flat_and_rotation/2026-07-19_09-14-50/params/env.yaml"
POLICY_DT_S = .02
PHYSICS_DT_S = .005
COMMAND_STATE_VERSION = "own_flat_command_restart_v1"
# Explicit inventory: adding mutable state must also update the restart schema.
_COMMAND_TENSORS = {
    **{name: (torch.float32, 3) for name in ("generator_command3", "command3",
        "generator_after_resample3", "generator_after_postprocess3",
        "ground_restore_command3", "ground_temporary_command3", "fixed_request_command3")},
    **{name: (torch.float32, None) for name in ("height", "heading_target", "time_left", "fixed_request_height")},
    **{name: (torch.bool, None) for name in ("generator_cache_postprocessed", "is_heading", "is_standing",
        "standing_first_draw", "ground_birth", "holding", "learning_mask", "ground_override_pending",
        "paid_override_active", "fixed_request_mask")},
    "standing_second_draw": (torch.int8, None),
    **{name: (torch.long, None) for name in ("special_mode_id", "command_counter",
        "_height_seen_command_counter", "episode_physics_ticks")},
}
_COMMAND_CLOCKS = ("physics_steps", "_last_command_physics_step")
_GROUND_SENTINELS = ("ground_restore_command3", "ground_temporary_command3")


def _restart_tensor(value, shape, dtype, name, *, device="cpu"):
    """Strict checkpoint transport; conversion happens only after validation."""
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device != torch.device(device) or value.dtype != dtype
            or tuple(value.shape) != tuple(shape) or value.requires_grad):
        raise ValueError(f"Restart {name} requires {dtype} {tuple(shape)} on {device}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"Restart {name} must be finite")
    return value.detach().clone()


def _restart_rng(generator, value):
    expected = generator.get_state()
    state = _restart_tensor(value, expected.shape, torch.uint8, "generator state")
    # Validate the actual Torch engine format without touching caller/global RNG.
    probe = torch.Generator(device=generator.device)
    try:
        probe.set_state(state)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Invalid Torch restart generator state") from exc
    return state
# Tuple order is the archived order, not activation order.
SPECIAL_MODES = (
    ("spin_low", 3000, .15, ((-.1, .1),), ((-3.25 * math.pi, -2 * math.pi), (2 * math.pi, 3.25 * math.pi))),
    ("spin_mid", 4000, .15, ((-.1, .1),), ((-4.5 * math.pi, -3.25 * math.pi), (3.25 * math.pi, 4.5 * math.pi))),
    ("dash", 2000, .30, ((-3., -2.), (2., 3.)), ((-2 * math.pi, 2 * math.pi),)),
)


def _wrap_to_pi(value):
    # Same [-pi, pi] endpoint convention as the current IsaacLab helper.
    wrapped = torch.remainder(value + math.pi, 2 * math.pi)
    return torch.where((wrapped == 0.) & (value > 0.), math.pi, wrapped - math.pi)


class ScutFlatCommands:
    """Fixed archived sampler; all randomness uses the supplied env generator."""

    # Static training ownership is configuration, not mutable restart state.
    __slots__ = ("_training_rehearsal_mask", "_temporal_rehearsal", "__dict__")
    special_modes = SPECIAL_MODES

    def __init__(self, num_envs, device, physics_dt_s=PHYSICS_DT_S):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("reference commands require a positive integer environment count")
        if physics_dt_s != PHYSICS_DT_S:
            raise ValueError("reference flat command v1 requires the reviewed 5 ms physical clock")
        self._training_rehearsal_mask = None
        self._temporal_rehearsal = None
        self.num_envs = num_envs
        self.generator_command3 = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.device = self.generator_command3.device
        self.command3 = torch.zeros_like(self.generator_command3)
        self.generator_after_resample3 = torch.zeros_like(self.command3)
        self.generator_after_postprocess3 = torch.zeros_like(self.command3)
        self.generator_cache_postprocessed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.height = torch.full((num_envs,), .22, dtype=torch.float32, device=self.device)
        self.heading_target = torch.zeros_like(self.height)
        self.is_heading = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.is_standing = torch.zeros_like(self.is_heading)
        self.standing_first_draw = torch.zeros_like(self.is_heading)
        # -1 means the lane did not receive the second draw; 0/1 is its result.
        self.standing_second_draw = torch.full((num_envs,), -1, dtype=torch.int8, device=self.device)
        self.special_mode_id = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.command_counter = torch.zeros_like(self.special_mode_id)
        # Source env's observer is distinct from the generator's counter and
        # is not cleared by reset. On the first compute after initial reset,
        # counter 1 > observer 0 causes the source to sample height once more.
        self._height_seen_command_counter = torch.zeros_like(self.command_counter)
        self.time_left = torch.zeros_like(self.height)
        self.episode_physics_ticks = torch.zeros_like(self.special_mode_id)
        self.ground_birth = torch.zeros_like(self.is_heading)
        self.holding = torch.zeros_like(self.is_heading)
        self.learning_mask = torch.ones_like(self.is_heading)
        self.ground_override_pending = torch.zeros_like(self.is_heading)
        self.paid_override_active = torch.zeros_like(self.is_heading)
        self.ground_restore_command3 = torch.full_like(self.command3, torch.nan)
        self.ground_temporary_command3 = torch.full_like(self.command3, torch.nan)
        self.fixed_request_mask = torch.zeros_like(self.is_heading)
        self.fixed_request_command3 = torch.zeros_like(self.command3)
        self.fixed_request_height = torch.full_like(self.height, .22)
        self.physics_steps = 0
        self._last_command_physics_step = 0

    def _restart_identity(self):
        return {"command_config": self.summary(), "num_envs": self.num_envs,
                "device": str(self.device), "torch_version": str(torch.__version__),
                "policy_dt_s": POLICY_DT_S, "physics_dt_s": PHYSICS_DT_S}

    def _restart_inventory(self):
        if set(vars(self)) != set(_COMMAND_TENSORS) | set(_COMMAND_CLOCKS) | {"num_envs", "device"}:
            raise ValueError("Command restart inventory changed; version the complete state schema")
        for name, (dtype, width) in _COMMAND_TENSORS.items():
            tensor = getattr(self, name)
            shape = (self.num_envs,) if width is None else (self.num_envs, width)
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != self.device
                    or tensor.requires_grad or not tensor.is_contiguous()):
                raise ValueError(f"Destination command buffer {name} changed shape/dtype/device")

    def state_dict(self, generator):
        """Capture this owner and caller RNG, not native physics or PPO state.

        Inactive ground buffers use NaN internally. Their snapshot encoding is
        finite zero plus ground_override_pending, and restores the exact sentinel.
        Fixed request buffers are configuration identity, never silently replaced.
        """
        self._restart_inventory()
        generator = self._generator(generator)
        tensors = {}
        for name, (dtype, width) in _COMMAND_TENSORS.items():
            value = getattr(self, name)
            shape = (self.num_envs,) if width is None else (self.num_envs, width)
            if name in _GROUND_SENTINELS:
                if not bool(torch.isnan(value[~self.ground_override_pending]).all()):
                    raise ValueError(f"Inactive command {name} must retain its NaN sentinel")
                value = torch.where(self.ground_override_pending[:, None], value, 0.)
            tensors[name] = _restart_tensor(value, shape, dtype, name, device=self.device).cpu()
        result = {"schema": COMMAND_STATE_VERSION, "identity": self._restart_identity(),
                  "tensors": tensors, "clocks": {name: getattr(self, name) for name in _COMMAND_CLOCKS},
                  "generator_state": generator.get_state().clone()}
        if getattr(self._temporal_rehearsal, "mutable", False):
            result["motion_course"] = self._temporal_rehearsal.state_dict()
        self._prepare_restart(result, generator)
        return result

    def _prepare_restart(self, state, generator):
        self._restart_inventory()
        generator = self._generator(generator)
        mutable = getattr(self._temporal_rehearsal, "mutable", False)
        keys = {"schema", "identity", "tensors", "clocks", "generator_state"}
        if mutable:
            keys.add("motion_course")
        if (not isinstance(state, dict) or set(state) != keys
                or state["schema"] != COMMAND_STATE_VERSION
                or json.dumps(state["identity"], sort_keys=True, allow_nan=False)
                    != json.dumps(self._restart_identity(), sort_keys=True, allow_nan=False)):
            raise ValueError("Command restart schema/config/device identity mismatch")
        values, clocks = state["tensors"], state["clocks"]
        if not isinstance(values, dict) or set(values) != set(_COMMAND_TENSORS):
            raise ValueError("Command restart tensor inventory mismatch")
        if (not isinstance(clocks, dict) or set(clocks) != set(_COMMAND_CLOCKS)
                or any(type(value) is not int or value < 0 for value in clocks.values())
                or not 0 <= clocks["physics_steps"] - clocks["_last_command_physics_step"] <= 4):
            raise ValueError("Command restart physical clocks are invalid")
        checked = {name: _restart_tensor(values[name], (self.num_envs,) if width is None else (self.num_envs, width),
                    dtype, name) for name, (dtype, width) in _COMMAND_TENSORS.items()}
        for name in ("fixed_request_mask", "fixed_request_command3", "fixed_request_height"):
            if not torch.equal(checked[name], getattr(self, name).detach().cpu()):
                raise ValueError("Command restart fixed evaluation request identity mismatch")
        if (bool((checked["height"] <= 0).any()) or bool((checked["fixed_request_height"] <= 0).any())
                or bool((checked["fixed_request_command3"][:, 1] != 0).any())
                or bool(((checked["standing_second_draw"] < -1) | (checked["standing_second_draw"] > 1)).any())
                or bool(((checked["special_mode_id"] < -1) | (checked["special_mode_id"] >= len(self.special_modes))).any())
                or any(bool((checked[name] < 0).any()) for name in
                       ("command_counter", "_height_seen_command_counter", "episode_physics_ticks"))):
            raise ValueError("Command restart domain/counters are invalid")
        expected_hold = checked["ground_birth"] & (checked["episode_physics_ticks"] < 40)
        if not torch.equal(checked["holding"], expected_hold):
            raise ValueError("Command restart holding clock disagrees with ground birth")
        pending = checked["ground_override_pending"]
        if bool((pending & ~checked["ground_birth"]).any()):
            raise ValueError("Command restart override must belong to a ground birth")
        for name in _GROUND_SENTINELS:
            if bool((checked[name][~pending] != 0.).any()):
                raise ValueError("Inactive ground restart buffers require zero encoding")
            checked[name] = torch.where(pending[:, None], checked[name], torch.nan)
        rng = _restart_rng(generator, state["generator_state"])
        # Allocate all target-device copies before any visible state is changed.
        prepared = {name: value.to(self.device).clone() for name, value in checked.items()}
        motion = self._temporal_rehearsal.prepare_state(state["motion_course"]) if mutable else None
        return prepared, deepcopy(clocks), rng, motion

    def validate_state_dict(self, state, generator):
        """Validate without changing this owner or any RNG."""
        self._prepare_restart(state, generator)

    def _commit_restart(self, prepared, generator):
        tensors, clocks, rng, motion = prepared
        generator.set_state(rng)
        for name, value in tensors.items():
            getattr(self, name).copy_(value)
        for name, value in clocks.items():
            setattr(self, name, value)
        if motion is not None:
            self._temporal_rehearsal.commit_state(motion)

    def load_state_dict(self, state, generator):
        """Validate the complete payload first; rejected loads have no effects."""
        prepared = self._prepare_restart(state, generator)
        self._commit_restart(prepared, generator)

    @staticmethod
    def _iteration(iteration):
        if type(iteration) is not int or iteration < 0:
            raise ValueError("reference command iteration must be a nonnegative integer")
        return iteration

    def _generator(self, generator):
        if not isinstance(generator, torch.Generator) or torch.device(generator.device) != self.device:
            raise ValueError("A torch.Generator on the command device is required")
        return generator

    def _ids(self, ids):
        if not isinstance(ids, torch.Tensor):
            if not isinstance(ids, (list, tuple)) or any(type(index) is not int for index in ids):
                raise ValueError("Reset ids must be integer list/tuple or long[K] tensor; bool is not an ID")
            ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.dtype != torch.long or ids.ndim != 1 or ids.device != self.device:
            raise ValueError("Reset ids must be long[K] on the command device")
        if ids.numel() and (bool(((ids < 0) | (ids >= self.num_envs)).any())
                           or ids.unique().numel() != ids.numel()):
            raise ValueError("Reset ids must be unique and in bounds")
        return ids

    def _vector(self, value, name, dtype, ids=None):
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.device != self.device:
            raise ValueError(f"{name} must have dtype {dtype} on the command device")
        if value.shape == (self.num_envs,):
            return value if ids is None else value[ids]
        if ids is not None and value.shape == (ids.numel(),):
            return value
        raise ValueError(f"{name} must have shape [N] or selected [K]")

    def _uniform(self, count, low, high, generator):
        return torch.empty(count, dtype=torch.float32, device=self.device).uniform_(low, high, generator=generator)

    def _sample_ranges(self, ranges, count, generator):
        """the reference length-weighted mixture of sorted disjoint closed intervals."""
        if len(ranges) == 1:
            low, high = ranges[0]
            if low == high:
                return torch.full((count,), low, dtype=torch.float32, device=self.device)
            return self._uniform(count, low, high, generator)
        lengths = torch.tensor([hi - lo for lo, hi in ranges], dtype=torch.float32, device=self.device)
        probabilities = lengths / lengths.sum()
        choices = torch.multinomial(probabilities, count, replacement=True, generator=generator)
        values = torch.empty(count, dtype=torch.float32, device=self.device)
        for index, (low, high) in enumerate(ranges):
            mask = choices == index
            n = int(mask.sum())
            if n:
                values[mask] = self._uniform(n, low, high, generator)
        return values

    def _resample(self, ids, iteration, generator, *, special_modes=None):
        # A mixed terrain owner may select the archived flat schedule for an
        # explicit lane group without mutating the owner's rough schedule.
        special_modes = self.special_modes if special_modes is None else special_modes
        count = ids.numel()
        if not count:
            return
        self.time_left[ids] = self._uniform(count, 5., 15., generator)
        self.command_counter[ids] += 1
        fixed_ids = ids[self.fixed_request_mask[ids]]
        if fixed_ids.numel():
            self.generator_command3[fixed_ids] = self.fixed_request_command3[fixed_ids]
            self.is_standing[fixed_ids] = False
            self.standing_first_draw[fixed_ids] = False
            self.standing_second_draw[fixed_ids] = -1
            self.is_heading[fixed_ids] = False
            self.special_mode_id[fixed_ids] = -1
            self.generator_after_resample3[fixed_ids] = self.generator_command3[fixed_ids]
            self.generator_cache_postprocessed[fixed_ids] = False
        ids = ids[~self.fixed_request_mask[ids]]
        count = ids.numel()
        if not count:
            return
        first = self._uniform(count, 0., 1., generator) <= .1
        self.standing_first_draw[ids] = first
        self.is_standing[ids] = first
        self.standing_second_draw[ids] = -1
        self.special_mode_id[ids] = -1
        nonstanding = ids[~first]
        ready = self.episode_physics_ticks[nonstanding] >= 1000  # exactly 5 seconds
        candidate, delayed = nonstanding[ready], nonstanding[~ready]
        active = [i for i, (_, start, _, _, _) in enumerate(special_modes) if iteration >= start]
        if candidate.numel() and active:
            order = torch.randperm(len(active), device=self.device, generator=generator).tolist()
            draw = self._uniform(candidate.numel(), 0., 1., generator)
            cumulative = 0.
            for index in order:
                mode = active[index]
                upper = cumulative + special_modes[mode][2]
                selected = (draw >= cumulative) & (draw < upper)
                self.special_mode_id[candidate[selected]] = mode
                cumulative = upper
        for mode, (_, _, _, vx_ranges, yaw_ranges) in enumerate(special_modes):
            selected = candidate[self.special_mode_id[candidate] == mode]
            n = selected.numel()
            if n:
                self.generator_command3[selected, 0] = self._sample_ranges(vx_ranges, n, generator)
                self.generator_command3[selected, 1] = 0.
                self.generator_command3[selected, 2] = self._sample_ranges(yaw_ranges, n, generator)
        normal = torch.cat((delayed, candidate[self.special_mode_id[candidate] < 0]))
        n = normal.numel()
        if n:
            # Match the parent's second complete sample, including its second
            # standing draw. Do not collapse the two draws into one Bernoulli.
            self.generator_command3[normal, 0] = self._uniform(n, -2.7, 2.7, generator)
            self.generator_command3[normal, 1] = self._uniform(n, 0., 0., generator)
            self.generator_command3[normal, 2] = self._uniform(n, -2 * math.pi, 2 * math.pi, generator)
            self.heading_target[normal] = self._uniform(n, -math.pi, math.pi, generator)
            self.is_heading[normal] = self._uniform(n, 0., 1., generator) <= .5
            second = self._uniform(n, 0., 1., generator) <= .1
            self.is_standing[normal] = second
            self.standing_second_draw[normal] = second.to(torch.int8)
        # In particular, first-draw standing lanes retain their previous cache.
        self.generator_after_resample3[ids] = self.generator_command3[ids]
        self.generator_cache_postprocessed[ids] = False

    def _sample_height(self, ids, generator):
        fixed_ids = ids[self.fixed_request_mask[ids]]
        self.height[fixed_ids] = self.fixed_request_height[fixed_ids]
        source_ids = ids[~self.fixed_request_mask[ids]]
        if source_ids.numel():
            self.height[source_ids] = self._uniform(source_ids.numel(), .2, .42, generator)

    def set_fixed_requests(self, ids, command3, height, *, evaluation_only=False):
        """Named evaluation adaptation; takes effect on the next resample/reset.

        The caller must use a dedicated evaluation entry. Fixed lanes omit
        ordinary/special/heading/standing selection; ground capture remains.
        This is not a source-distribution training option or a command clamp.
        """
        if evaluation_only is not True:
            raise ValueError("Fixed requests require the explicit evaluation-only adaptation")
        self._store_fixed_requests(ids, command3, height)

    def set_training_rehearsal_requests(self, ids, command3, height):
        """V7 training adaptation; called by the explicit rehearsal allocator.

        Uses the same fixed-request mechanics without claiming an evaluation
        role. Physics, ground capture and PPO eligibility remain unchanged.
        """
        ids = self._ids(ids)
        self._store_fixed_requests(ids, command3, height)
        if ids.numel():
            if self._training_rehearsal_mask is None:
                self._training_rehearsal_mask = torch.zeros_like(self.fixed_request_mask)
            self._training_rehearsal_mask[ids] = True

    def set_temporal_training_requests(self, ids, requests, *, phase_policy_ticks, episode_policy_ticks):
        """Bind explicit nominal lane sequences before the first physical reset."""
        from .temporal_flat_rehearsal import TemporalFlatRequests
        ids = self._ids(ids)
        if (self._temporal_rehearsal is not None or self.physics_steps != 0
                or bool(self.command_counter.any()) or bool(self.fixed_request_mask[ids].any())):
            raise ValueError("Temporal requests require fresh, disjoint training ownership")
        candidate = TemporalFlatRequests(self.num_envs, self.device, ids, requests,
            phase_policy_ticks, episode_policy_ticks)
        self._temporal_rehearsal = candidate

    def set_motion_training_requests(self, ids, settings, *, episode_policy_ticks=999):
        """Install the reviewed continuous transition course on nominal lanes."""
        from .motion_transition_course import MotionTransitionRequests
        ids = self._ids(ids)
        if (not ids.numel() or self._temporal_rehearsal is not None or self.physics_steps
                or bool(self.command_counter.any()) or bool(self.fixed_request_mask[ids].any())):
            raise ValueError("Motion requests require fresh, disjoint training ownership")
        self._temporal_rehearsal = MotionTransitionRequests(
            self.num_envs, self.device, ids, settings, episode_policy_ticks)

    def observe_motion_interval(self, velocity, omega, done, valid):
        if getattr(self._temporal_rehearsal, "mutable", False):
            self._temporal_rehearsal.observe(self, velocity, omega, done, valid)

    def _store_fixed_requests(self, ids, command3, height):
        ids = self._ids(ids)
        if self._temporal_rehearsal is not None and bool(self._temporal_rehearsal.mask[ids].any()):
            raise ValueError("Fixed and temporal request lanes must be disjoint")
        if not isinstance(command3, torch.Tensor) or command3.dtype != torch.float32 or command3.device != self.device:
            raise ValueError("Fixed command3 must be float32 on the command device")
        if command3.shape == (self.num_envs, 3):
            command3 = command3[ids]
        elif command3.shape != (ids.numel(), 3):
            raise ValueError("Fixed command3 must have shape [N,3] or selected [K,3]")
        height = self._vector(height, "fixed height", torch.float32, ids)
        if (not bool(torch.isfinite(command3).all()) or not bool(torch.isfinite(height).all())
                or bool((command3[:, 1] != 0.).any()) or bool((height <= 0.).any())):
            raise ValueError("Fixed requests require finite body-forward/yaw, vy=0, and positive height")
        self.fixed_request_mask[ids] = True
        self.fixed_request_command3[ids] = command3
        self.fixed_request_height[ids] = height

    def clear_fixed_requests(self, ids):
        """Return selected lanes to source sampling on the next resample/reset."""
        ids = self._ids(ids)
        self.fixed_request_mask[ids] = False

    def reset(self, ids, heading, ground_mask, generator, iteration=0):
        """Resample selected lanes, retaining current-SDK raw-cache semantics.

        ``heading`` is validated but deliberately not used to postprocess the
        reset cache: source heading targets are independent uniform draws.
        No mechanical pose/velocity, RNG state, or completed PPO mask is reset.
        """
        ids = self._ids(ids)
        self._vector(heading, "heading", torch.float32, ids)
        ground = self._vector(ground_mask, "ground_mask", torch.bool, ids)
        if self._temporal_rehearsal is not None:
            self._temporal_rehearsal.validate_reset(ids, ground)
        generator = self._generator(generator)
        iteration = self._iteration(iteration)
        self.episode_physics_ticks[ids] = 0
        self.command_counter[ids] = 0
        self.ground_birth[ids] = ground
        self.holding[ids] = ground
        self.ground_override_pending[ids] = ground
        self.ground_restore_command3[ids] = torch.nan
        self.ground_temporary_command3[ids] = torch.nan
        self._resample(ids, iteration, generator)
        self._sample_height(ids, generator)
        if getattr(self._temporal_rehearsal, "mutable", False):
            self._temporal_rehearsal.reset(ids, generator)
        ground_ids = ids[ground]
        if ground_ids.numel():
            self.ground_restore_command3[ground_ids] = self.generator_command3[ground_ids]
            n = ground_ids.numel()
            self.ground_temporary_command3[ground_ids, 0] = self._uniform(n, -1., 1., generator)
            self.ground_temporary_command3[ground_ids, 1] = 0.
            self.ground_temporary_command3[ground_ids, 2] = self._uniform(n, -1., 1., generator)
        self._publish(ids)

    def _postprocess(self, heading):
        selected = self.is_heading & (self.special_mode_id < 0)
        error = _wrap_to_pi(self.heading_target[selected] - heading[selected])
        self.generator_command3[selected, 2] = (5. * error).clamp(-2 * math.pi, 2 * math.pi)
        self.generator_command3[self.is_standing] = 0.
        self.generator_after_postprocess3.copy_(self.generator_command3)
        self.generator_cache_postprocessed.fill_(True)

    def _publish(self, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        pending = self.ground_override_pending[ids]
        active = pending & (self.episode_physics_ticks[ids] < 300)  # exactly 1.5 seconds
        expired_ids = ids[pending & ~active]
        if expired_ids.numel():
            # Restore the saved mutable cache once, after heading/standing
            # postprocessing, exactly in the source hook's priority position.
            self.generator_command3[expired_ids] = self.ground_restore_command3[expired_ids]
            self.generator_cache_postprocessed[expired_ids] = False
            self.ground_override_pending[expired_ids] = False
            self.ground_restore_command3[expired_ids] = torch.nan
            self.ground_temporary_command3[expired_ids] = torch.nan
        self.command3[ids] = self.generator_command3[ids]
        active_ids = ids[active]
        self.command3[active_ids] = self.ground_temporary_command3[active_ids]
        if self._temporal_rehearsal is not None:
            self._temporal_rehearsal.publish(self, ids)

    def begin_policy(self):
        """Local PPO ownership adaptation, not an upstream sampler mask."""
        self.learning_mask.copy_(~self.holding)
        # The final temporary-command interval starts at physical tick 296
        # and ends at tick 300. Snapshot the command actually published for
        # that interval, before advancing physics or resetting any lane.
        self.paid_override_active.copy_(self.ground_override_pending & (self.episode_physics_ticks < 300))
        return self.learning_mask

    def advance_physics(self, dt=PHYSICS_DT_S):
        """Call AFTER applying this tick's torque; passive forces remain active."""
        if dt != PHYSICS_DT_S:
            raise ValueError("reference command physical clock must advance by 5 ms")
        self.learning_mask &= ~self.holding
        self.episode_physics_ticks += 1
        self.holding.copy_(self.ground_birth & (self.episode_physics_ticks < 40))
        self.physics_steps += 1

    def advance(self, dt, heading, iteration, generator):
        """Advance command state after paying the old interval's reward."""
        if dt != POLICY_DT_S:
            raise ValueError("reference command policy clock must advance by 20 ms")
        if self.physics_steps - self._last_command_physics_step != 4:
            raise ValueError("Each command advance requires exactly four new physical ticks")
        heading = self._vector(heading, "heading", torch.float32)
        generator = self._generator(generator)
        iteration = self._iteration(iteration)
        self.time_left -= dt
        ids = (self.time_left <= 0.).nonzero(as_tuple=False).flatten()
        self._resample(ids, iteration, generator)
        self._postprocess(heading)
        height_ids = (self.command_counter > self._height_seen_command_counter).nonzero(as_tuple=False).flatten()
        self._sample_height(height_ids, generator)
        self._height_seen_command_counter.copy_(self.command_counter)
        self._publish()
        self._last_command_physics_step = self.physics_steps

    def resample(self, ids, heading, generator, iteration):
        """Reject a second command owner; normal source flow is reset/advance."""
        raise RuntimeError("Forced reference command resampling is unsupported; use reset or timed advance only")

    def bootstrap_mask(self, terminated):
        terminated = self._vector(terminated, "terminated", torch.bool)
        return ~self.holding & ~terminated

    def diagnostics(self):
        """Current owner state plus the last interval's paid ownership mask.

        ``paid_override_active`` is captured at begin_policy and survives
        reset, like learning_mask. ``override_active`` is the current clock
        state; returned observations must not substitute it for paid state.
        Episode identity remains the caller's.
        """
        result = {"base_request_command3": self.generator_command3.clone(),
                "command3": self.command3.clone(), "height": self.height.clone(),
                "ground": self.ground_birth.clone(), "holding": self.holding.clone(),
                "override_active": (self.ground_override_pending & (self.episode_physics_ticks < 300)).clone(),
                "paid_override_active": self.paid_override_active.clone(),
                "special_mode_id": self.special_mode_id.clone(),
                "standing_first_draw": self.standing_first_draw.clone(),
                "standing_second_draw": self.standing_second_draw.clone(),
                "is_standing": self.is_standing.clone(), "is_heading": self.is_heading.clone(),
                "heading_target": self.heading_target.clone(),
                "episode_ticks": (self.episode_physics_ticks // 4).clone(),
                "episode_physics_ticks": self.episode_physics_ticks.clone(),
                "command_counter": self.command_counter.clone(), "time_left_s": self.time_left.clone(),
                "height_seen_command_counter": self._height_seen_command_counter.clone(),
                "learning_mask": self.learning_mask.clone(),
                "fixed_evaluation_request": self.fixed_request_mask.clone(),
                "generator_after_resample3": self.generator_after_resample3.clone(),
                "generator_after_postprocess3": self.generator_after_postprocess3.clone(),
                "generator_cache_postprocessed": self.generator_cache_postprocessed.clone()}
        if self._training_rehearsal_mask is not None:
            result["fixed_training_rehearsal"] = self._training_rehearsal_mask.clone()
            result["fixed_evaluation_request"] &= ~self._training_rehearsal_mask
        if self._temporal_rehearsal is not None:
            result["temporal_training_rehearsal"] = self._temporal_rehearsal.mask.clone()
            result["temporal_published_phase"] = self._temporal_rehearsal.published_phases(self)
            if getattr(self._temporal_rehearsal, "mutable", False):
                result["motion_course"] = {key: value.clone() for key, value in self._temporal_rehearsal.paid.items()}
        return result

    def metrics(self):
        """Policy-rate tensor values for the existing TensorBoard logger."""
        prefix = "/source_command/"
        values = {"ground_birth_fraction": self.ground_birth.float().mean(),
                  "holding_fraction": self.holding.float().mean(),
                  "temporary_override_fraction": (self.ground_override_pending & (self.episode_physics_ticks < 300)).float().mean(),
                  "generator_standing_fraction": self.is_standing.float().mean(),
                  "standing_first_draw_fraction": self.standing_first_draw.float().mean(),
                  "standing_second_draw_positive_fraction": (self.standing_second_draw == 1).float().mean(),
                  "fixed_evaluation_fraction": self.fixed_request_mask.float().mean(),
                  "actual_vx_mean_m_s": self.command3[:, 0].mean(),
                  "actual_yaw_abs_mean_rad_s": self.command3[:, 2].abs().mean(),
                  "height_mean_m": self.height.mean(),
                  "learning_fraction": self.learning_mask.float().mean()}
        if self._training_rehearsal_mask is not None:
            values["fixed_training_rehearsal_fraction"] = self._training_rehearsal_mask.float().mean()
            values["fixed_evaluation_fraction"] = (
                self.fixed_request_mask & ~self._training_rehearsal_mask).float().mean()
        values.update({name + "_fraction": (self.special_mode_id == index).float().mean()
                       for index, (name, *_) in enumerate(self.special_modes)})
        result = {prefix + name: value for name, value in values.items()}
        if self._temporal_rehearsal is not None:
            phase = self._temporal_rehearsal.published_phases(self)
            for index, name in enumerate(self._temporal_rehearsal.identity["phase_names"]):
                result[f"/temporal_request/{name}_actor_owned_intervals"] = (
                    (phase == index) & self.learning_mask).sum()
        return result

    def validate_state(self):
        """Explicit startup/debug finite check; not a per-step GPU synchronization."""
        for name in ("generator_command3", "command3", "height", "heading_target", "time_left"):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"SCUT command state {name} is not finite")
        for name in ("ground_restore_command3", "ground_temporary_command3"):
            if not bool(torch.isfinite(getattr(self, name)[self.ground_override_pending]).all()):
                raise ValueError(f"Active SCUT command state {name} is not finite")

    def summary(self):
        result = {"version": COMMAND_VERSION, "source_commit": SOURCE_COMMIT, "source_archive": SOURCE_ARCHIVE,
                "columns": ["body_vx_m_s", "body_vy_m_s", "yaw_rate_rad_s"],
                "height_range_m": [.2, .42], "resampling_time_range_s": [5., 15.],
                "standing_draws": [.1, .1], "second_draw_scope": "remaining normal pool only",
                "heading_probability_normal": .5, "heading_gain": 5.,
                "special_ready_episode_s": 5., "special_requires_stable": False,
                "special_modes": [{"name": name, "iteration_start": start, "iteration_end": -1,
                    "probability_within_first_nonstanding_ready_pool": probability,
                    "vx_ranges_m_s": [list(pair) for pair in vx],
                    "yaw_ranges_rad_s": [list(pair) for pair in yaw]}
                    for name, start, probability, vx, yaw in self.special_modes],
                "ground_hold_s": .2, "ground_temporary_command_s": 1.5,
                "ground_temporary_vx_range_m_s": [-1., 1.], "ground_temporary_yaw_range_rad_s": [-1., 1.],
                "ground_selection": "external reviewed mechanical sampler supplies bool mask",
                "clock": "integer 5ms physical ticks; 20ms command timer; external absolute iteration",
                "rng": "only caller torch.Generator; historical random-stream equivalence not claimed",
                "reset_cache": "current SDK reset resamples without postprocess; first standing retains mutable cache",
                "height_sampling": "reset plus generator-counter observer; observer survives reset, including initial first-compute resample",
                "ground_restore": "once after postprocess, restore saved pre-override generator cache",
                "historical_sdk_qualification": False,
                "local_adaptations": ["integer physical clock for aligned source time thresholds",
                    "PPO whole-interval learning and returned-state bootstrap ownership masks",
                    "explicit evaluation-only fixed request lanes, when configured"],
                "mechanical_reset_implemented": False, "native_qualified": False}
        if self._temporal_rehearsal is not None:
            result["temporal_training_rehearsal"] = self._temporal_rehearsal.summary()
        if self._training_rehearsal_mask is not None:
            import hashlib
            result["fixed_training_rehearsal"] = {
                "mask_sha256": hashlib.sha256(
                    self._training_rehearsal_mask.detach().cpu().numpy().tobytes()).hexdigest(),
                "scope": "explicit static training allocation; not fixed evaluation"}
        return result
