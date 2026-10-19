"""Pinned final rough command composition on caller-owned Torch buffers.

The shared sampler supplies normal/standing/heading draws and physical clocks.
Terrain overrides are sampled, not clamps. No mechanical birth, airborne entry
command resample or StepUp event is created here. Those have separate owners.
"""
from copy import deepcopy
import hashlib
import json

import torch

from .flat_command_sampling import (
    ScutFlatCommands, COMMAND_STATE_VERSION, _COMMAND_TENSORS, _COMMAND_CLOCKS,
    _GROUND_SENTINELS, _restart_tensor, SPECIAL_MODES, SOURCE_ARCHIVE, _wrap_to_pi,
)


VERSION = "v7-final-rough-commands-v1"
STATE_VERSION = "v7-final-rough-command-restart-v1"
AIR_STATE_VERSION = "v7-final-rough-command-restart-v2-air"
AIR_LIMIT_VERSION = "v7-final-rough-air-command-limits-v1"
_AIR_TENSORS = {"mode_index": (torch.long, None), "remaining_s": (torch.float32, None),
    "last_policy_step": (torch.long, None), "frame_active": (torch.bool, None),
    "applied_public_command": (torch.float32, 3)}
_FIELDS = ("lin_vel_x", "lin_vel_y", "ang_vel_z_heading", "ang_vel_z_non_heading")
_EXTRA_TENSORS = {
    "terrain_key_ids": (torch.long, None), "terrain_profile_ids": (torch.long, None),
    "profile_switch_count": (torch.long, None), "reset_heading": (torch.float32, None),
    "latest_heading": (torch.float32, None),
    "axis_mask": (torch.bool, None), "override_values": (torch.float32, 4),
    "override_masks": (torch.bool, 4),
}


def _ranges(value):
    """The source manager sorts and merges adjacent range segments."""
    pairs = [value] if isinstance(value[0], (int, float)) else value
    merged = []
    for low, high in sorted((min(float(a), float(b)), max(float(a), float(b))) for a, b in pairs):
        if merged and low <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return tuple(tuple(pair) for pair in merged)


def final_special_modes(commands):
    """Resolve the final mode pool shared by both terrain command owners."""
    if (tuple(commands["resampling_time_range"]) != (5., 15.)
            or commands["rel_standing_envs"] != .1 or commands["rel_heading_envs"] != .5
            or commands["heading_control_stiffness"] != 5.
            or commands["special_mode_min_episode_time"] != 5.
            or commands["special_mode_require_stable"] is not False
            or tuple(commands["ranges"]["lin_vel_x"]) != (-2.7, 2.7)
            or any(mode["iteration_end"] != -1 for mode in commands["special_modes"].values())):
        raise ValueError("Final command definition differs from the shared sampler assumptions")
    return tuple((name, mode["iteration_start"], mode["rel_envs"],
        _ranges(mode["ranges"]["lin_vel_x"]), _ranges(mode["ranges"]["ang_vel_z"]))
        for name, mode in commands["special_modes"].items())


class FinalRoughCommands(ScutFlatCommands):
    """Final source profiles with the existing public body-vx/yaw/height ABI."""
    __slots__ = ("_final",)

    @property
    def special_modes(self):
        return self._final["special_modes"]

    @property
    def reset_heading_candidates_deg(self):
        return tuple(self._final["definition"]["birth"]["reset_heading_axis_aligned_candidates_deg"])

    def __init__(self, num_envs, device, *, terrain_origins, env_origins):
        from .final_rough_definition import load_final_rough_definition
        super().__init__(num_envs, device)
        definition = load_final_rough_definition()
        terrain = definition["terrain"]["terrain_generator"]
        commands, birth = definition["commands"], definition["birth"]
        grid = torch.as_tensor(terrain_origins, dtype=torch.float32, device=self.device)
        origins = torch.as_tensor(env_origins, dtype=torch.float32, device=self.device)
        rows, columns = terrain["num_rows"], terrain["num_cols"]
        if (grid.shape != (rows, columns, 3) or origins.shape != (num_envs, 3)
                or not bool(torch.isfinite(grid).all()) or not bool(torch.isfinite(origins).all())):
            raise ValueError("Final commands require complete native terrain and ordered environment origins")
        special_modes = final_special_modes(commands)
        keys = tuple(terrain["sub_terrains"])
        proportions = torch.tensor([terrain["sub_terrains"][key]["proportion"] for key in keys],
                                   dtype=torch.float32, device=self.device)
        cumulative = torch.cumsum(proportions / proportions.sum(), 0)
        fractions = torch.arange(columns, dtype=torch.float32, device=self.device) / columns + 1e-3
        column_key_ids = (fractions[:, None] >= cumulative[None, :]).sum(-1).clamp_max(len(keys) - 1)
        base = {"height_range": _ranges(commands["height_range"]),
            "lin_vel_x": _ranges(commands["ranges"]["lin_vel_x"]),
            "lin_vel_y": _ranges(commands["ranges"]["lin_vel_y"]),
            "ang_vel_z_heading": _ranges(commands["ranges"]["ang_vel_z"]),
            "ang_vel_z_non_heading": _ranges(commands["ranges"]["ang_vel_z"]),
            "reset_heading_axis_aligned_only": birth["reset_heading_axis_aligned_only"]}
        profiles, profile_ids, flags, disable_air, disable_ground, disable_special = [], [], [], [], [], []
        for key in keys:
            override = definition["overrides"].get(key, {})
            profile, enabled = dict(base), [False] * 4
            for field in ("height_range", "lin_vel_x", "lin_vel_y"):
                if override.get(field) is not None:
                    profile[field] = _ranges(override[field])
                    if field in _FIELDS:
                        enabled[_FIELDS.index(field)] = True
            if override.get("ang_vel_z") is not None:
                for field in _FIELDS[2:]:
                    profile[field], enabled[_FIELDS.index(field)] = _ranges(override["ang_vel_z"]), True
            for field in _FIELDS[2:]:
                if override.get(field) is not None:
                    profile[field], enabled[_FIELDS.index(field)] = _ranges(override[field]), True
            if override.get("reset_heading_axis_aligned_only") is not None:
                profile["reset_heading_axis_aligned_only"] = override["reset_heading_axis_aligned_only"]
            if profile not in profiles:
                profiles.append(profile)
            profile_ids.append(profiles.index(profile))
            flags.append(enabled)
            disable_air.append(bool(override.get("disable_predefined_reset_air")))
            disable_ground.append(bool(override.get("disable_predefined_reset_ground")))
            disable_special.append(bool(override.get("disable_special_mode")))
        self._final = dict(definition=definition, keys=keys, profiles=profiles, special_modes=special_modes,
            grid_shape=(rows, columns), cell_size=tuple(terrain["size"]), env_origins=origins.detach().clone(),
            start_xy=grid[0, 0, :2] - torch.tensor(terrain["size"], device=self.device) * .5,
            column_key_ids=column_key_ids, key_profile_ids=torch.tensor(profile_ids, device=self.device),
            flags=torch.tensor(flags, dtype=torch.bool, device=self.device),
            disable_air=torch.tensor(disable_air, dtype=torch.bool, device=self.device),
            disable_ground=torch.tensor(disable_ground, dtype=torch.bool, device=self.device),
            disable_special=torch.tensor(disable_special, dtype=torch.bool, device=self.device),
            terrain_key_ids=torch.full((num_envs,), -1, dtype=torch.long, device=self.device),
            terrain_profile_ids=torch.full((num_envs,), -1, dtype=torch.long, device=self.device),
            profile_switch_count=torch.zeros(num_envs, dtype=torch.long, device=self.device),
            reset_heading=torch.zeros(num_envs, dtype=torch.float32, device=self.device),
            latest_heading=torch.zeros(num_envs, dtype=torch.float32, device=self.device),
            axis_mask=torch.zeros(num_envs, dtype=torch.bool, device=self.device),
            override_values=torch.zeros(num_envs, 4, dtype=torch.float32, device=self.device),
            override_masks=torch.zeros(num_envs, 4, dtype=torch.bool, device=self.device),
            origin_sha256=hashlib.sha256(grid.cpu().numpy().tobytes() + origins.cpu().numpy().tobytes()).hexdigest())
        # The final environment initializes its terrain manager with a forced
        # command reset and synchronizes observer1 before the first episode.
        self._height_seen_command_counter.fill_(1)

    def set_source_flat_rehearsal(self, ids, *, source_iteration_offset=4500):
        """Install the existing flat sampler on static lanes before any reset."""
        ids = self._ids(ids)
        offset = self._iteration(source_iteration_offset)
        if (self.physics_steps or bool((self.command_counter != 0).any())
                or "source_flat_rehearsal" in self._final):
            raise ValueError("Source flat rehearsal must be configured once before the first reset")
        if not ids.numel() or bool(self.fixed_request_mask[ids].any()):
            raise ValueError("Source flat rehearsal needs nonempty lanes without fixed requests")
        mask = torch.zeros_like(self.fixed_request_mask)
        mask[ids] = True
        # This distribution is sampled; fixed training requests have their
        # own inherited mask and may coexist on disjoint plane lanes.
        self._final["source_flat_mask"] = mask
        self._final["source_flat_rehearsal"] = {"source_iteration_offset": offset}
        # Final terrain initializes observer1; the retained flat sampler starts
        # observer0 and draws height again on its first command compute.
        self._height_seen_command_counter[ids] = 0

    def enable_air_birth_command_limits(self, *, birth_definition_sha256):
        """Bind final-source reset limits before the first physical episode.

        This enables command ownership, not a sampler or a training stage.
        Reset must receive the explicit accepted mode and the same definition
        identity; nominal, fixed, scripted and retained-flat births stay separate.
        """
        from .final_rough_birth import FinalRoughBirthSampler
        if (self.physics_steps or bool(self.command_counter.any())
                or "air_birth_limits" in self._final):
            raise ValueError("Air birth command limits require a fresh owner before the first reset")
        sampler = FinalRoughBirthSampler(self.num_envs, self.device)
        if birth_definition_sha256 != sampler.definition_sha256:
            raise ValueError("Air birth command limits require the authenticated birth definition")
        raw = sampler.summary()["air_configuration"]["modes"]
        modes = [raw] if isinstance(raw, dict) else raw
        if len(modes) != 1 or modes[0]["iteration_start"] != 0 or modes[0]["iteration_end"] != -1:
            raise ValueError("Air command ownership requires the frozen always-active source mode")
        self._final["air_birth_limits"] = {
            "definition_sha256": birth_definition_sha256, "modes": deepcopy(modes),
            "mode_index": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "remaining_s": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "last_policy_step": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "frame_active": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "applied_public_command": torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)}

    def _air_reset_modes(self, ids, ground, modes, definition_sha256):
        """Validate accepted reset metadata before commands or RNG are mutated."""
        state = self._final.get("air_birth_limits")
        if state is None:
            if modes is not None or definition_sha256 is not None:
                raise ValueError("Explicit air birth metadata requires its configured command owner")
            return None
        if self.physics_steps % 4:
            raise ValueError("Air birth reset must occur at a complete20ms policy boundary")
        if modes is None:
            if definition_sha256 is not None:
                raise ValueError("Air definition identity requires ordered accepted birth modes")
            return torch.full((len(ids),), -1, dtype=torch.long, device=self.device)
        if (not isinstance(modes, torch.Tensor) or modes.dtype != torch.long
                or modes.device != self.device or modes.shape != (len(ids),)
                or definition_sha256 != state["definition_sha256"]
                or bool(((modes < -1) | (modes >= len(state["modes"]))).any())):
            raise ValueError("Air reset requires ordered accepted modes and the same birth definition")
        selected = modes >= 0
        excluded = ground | self.fixed_request_mask[ids] | ~self.air_birth_allowed(ids)
        if "source_flat_rehearsal" in self._final:
            excluded |= self._final["source_flat_mask"][ids]
        if bool((selected & excluded).any()):
            raise ValueError("Air limits cannot own ground, fixed, retained-flat or terrain-disabled births")
        return modes

    def _reset_air_birth_limits(self, ids, modes):
        if modes is None:
            return
        state = self._final["air_birth_limits"]
        state["mode_index"][ids] = modes
        state["remaining_s"][ids] = 0.
        state["last_policy_step"][ids] = -1
        state["frame_active"][ids] = False
        state["applied_public_command"][ids] = 0.
        for index, mode in enumerate(state["modes"]):
            selected = ids[modes == index]
            state["remaining_s"][selected] = mode["command_limits"]["duration_s"]

    def apply_air_birth_command_limits(self, public_commands, *, ids=None):
        """Apply last to actual vx/yaw/height commands, without changing raw caches.

        One physical policy step owns one active snapshot. Repeated publication
        may reapply limits after other overrides but never ages the clock twice.
        The float32 countdown follows the source's subtraction; unlike repeated
        source observation getters, all consumers within this frame share the
        snapshot even at the expiry boundary.
        """
        state = self._final.get("air_birth_limits")
        if state is None:
            return
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else self._ids(ids)
        if (not isinstance(public_commands, torch.Tensor)
                or public_commands.shape != (self.num_envs, 3)
                or public_commands.dtype != torch.float32 or public_commands.device != self.device
                or not bool(torch.isfinite(public_commands[ids]).all())
                or self.physics_steps % 4):
            raise ValueError("Air limits require finite public vx/yaw/height at a complete policy boundary")
        step = self.physics_steps // 4
        last = state["last_policy_step"][ids]
        if bool((last > step).any()) or bool(((last >= 0) & (step - last > 1)).any()):
            raise ValueError("Air command publication skipped or reversed a physical policy step")
        fresh = ids[last != step]
        state["frame_active"][fresh] = state["remaining_s"][fresh] > 0.
        state["remaining_s"][fresh] = (state["remaining_s"][fresh] - .02).clamp_min(0.)
        state["last_policy_step"][fresh] = step
        active = state["frame_active"][ids]
        for index, mode in enumerate(state["modes"]):
            selected = ids[active & (state["mode_index"][ids] == index)]
            limits = mode["command_limits"]
            for column, key in enumerate(("lin_vel_x", "ang_vel_z", "height")):
                public_commands[selected, column] = public_commands[selected, column].clamp(*limits[key])
        state["applied_public_command"][ids] = public_commands[ids]

    def _store_fixed_requests(self, ids, command3, height):
        ids = self._ids(ids)
        if ("source_flat_rehearsal" in self._final
                and bool(self._final["source_flat_mask"][ids].any())):
            raise ValueError("Source flat rehearsal lanes cannot become fixed requests")
        super()._store_fixed_requests(ids, command3, height)

    def set_training_rehearsal_requests(self, ids, command3, height):
        if ("source_flat_rehearsal" in self._final
                and (self.physics_steps or bool((self.command_counter != 0).any()))):
            raise ValueError("Mixed fixed training rehearsal must be configured before the first reset")
        super().set_training_rehearsal_requests(ids, command3, height)

    def _resample(self, ids, iteration, generator):
        settings = self._final.get("source_flat_rehearsal")
        if settings is None:
            return super()._resample(ids, iteration, generator)
        selected = self._final["source_flat_mask"][ids]
        super()._resample(ids[~selected], iteration, generator)
        super()._resample(ids[selected], iteration + settings["source_iteration_offset"],
                          generator, special_modes=SPECIAL_MODES)

    def profile_indices(self, world_positions, ids=None):
        """Read profiles at world XY, retaining classified nonfinite lane keys."""
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else self._ids(ids)
        if (not isinstance(world_positions, torch.Tensor) or world_positions.shape != (self.num_envs, 3)
                or world_positions.device != self.device or world_positions.dtype != torch.float32):
            raise ValueError("Final terrain lookup requires native float32 world positions[N,3]")
        state = self._final
        finite = torch.isfinite(world_positions[ids, :2]).all(-1)
        xy = torch.where(finite[:, None], world_positions[ids, :2], state["env_origins"][ids, :2])
        col = torch.floor((xy[:, 1] - state["start_xy"][1]) / state["cell_size"][1]).long().clamp(0, state["grid_shape"][1] - 1)
        keys = state["column_key_ids"][col]
        old = state["terrain_key_ids"][ids]
        keys = torch.where(~finite & (old >= 0), old, keys)
        plane = self.fixed_request_mask[ids]
        if "source_flat_rehearsal" in state:
            plane = plane | state["source_flat_mask"][ids]
        keys = torch.where(plane, state["keys"].index("plane_for_rm"), keys)
        return keys, state["key_profile_ids"][keys]

    def _sync_terrain(self, ids, world_positions):
        keys, profiles = self.profile_indices(world_positions, ids)
        changed = ids[profiles != self._final["terrain_profile_ids"][ids]]
        self._final["terrain_key_ids"][ids], self._final["terrain_profile_ids"][ids] = keys, profiles
        return changed

    def _query_keys(self, ids, world_positions, use_env_origins):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else self._ids(ids)
        positions = self._final["env_origins"] if use_env_origins else world_positions
        keys, profiles = self.profile_indices(positions, ids)
        return ids, keys, profiles

    def axis_reset_mask(self, ids=None, *, world_positions=None, use_env_origins=True):
        ids, _, profiles = self._query_keys(ids, world_positions, use_env_origins)
        values = torch.tensor([p["reset_heading_axis_aligned_only"] for p in self._final["profiles"]],
                              dtype=torch.bool, device=self.device)
        return values[profiles] & ~self.fixed_request_mask[ids]

    def air_birth_allowed(self, ids=None, *, world_positions=None, use_env_origins=True):
        """Terrain permission only; this does not select or create an air birth."""
        ids, keys, _ = self._query_keys(ids, world_positions, use_env_origins)
        return ~self._final["disable_air"][keys] & ~self.fixed_request_mask[ids]

    def ground_birth_allowed(self, ids=None, *, world_positions=None, use_env_origins=True):
        ids, keys, _ = self._query_keys(ids, world_positions, use_env_origins)
        return ~self._final["disable_ground"][keys] & ~self.fixed_request_mask[ids]

    def _sample_height(self, ids, generator):
        flat_ids = None
        if "source_flat_rehearsal" in self._final:
            selected = self._final["source_flat_mask"][ids]
            flat_ids, ids = ids[selected], ids[~selected]
        fixed = ids[self.fixed_request_mask[ids]]
        self.height[fixed] = self.fixed_request_height[fixed]
        ids = ids[~self.fixed_request_mask[ids]]
        keys = self._final["terrain_key_ids"][ids]
        for key in torch.unique(keys).tolist():
            selected = ids[keys == key]
            ranges = (self._final["definition"]["commands"]["height_range"] if key < 0 else
                self._final["profiles"][int(self._final["key_profile_ids"][key])]["height_range"])
            self.height[selected] = self._sample_ranges(_ranges(ranges), selected.numel(), generator)
        if flat_ids is not None:
            ScutFlatCommands._sample_height(self, flat_ids, generator)

    def _resample_overrides(self, ids, generator):
        state = self._final
        state["override_masks"][ids] = False
        ids = ids[~self.fixed_request_mask[ids]]
        keys = state["terrain_key_ids"][ids]
        for index, field in enumerate(_FIELDS):
            for key in torch.unique(keys).tolist():
                if key < 0 or not state["flags"][key, index]:
                    continue
                selected = ids[keys == key]
                ranges = state["profiles"][int(state["key_profile_ids"][key])][field]
                state["override_values"][selected, index] = self._sample_ranges(ranges, selected.numel(), generator)
                state["override_masks"][selected, index] = True

    def _sample_normal(self, ids, generator):
        """Parent normal sample after clearing an already assigned special mode."""
        n = ids.numel()
        if not n:
            return
        self.generator_command3[ids, 0] = self._uniform(n, -2.7, 2.7, generator)
        self.generator_command3[ids, 1] = self._uniform(n, 0., 0., generator)
        self.generator_command3[ids, 2] = self._uniform(n, -2 * torch.pi, 2 * torch.pi, generator)
        self.heading_target[ids] = self._uniform(n, -torch.pi, torch.pi, generator)
        self.is_heading[ids] = self._uniform(n, 0., 1., generator) <= .5
        standing = self._uniform(n, 0., 1., generator) <= .1
        self.is_standing[ids] = standing
        self.standing_second_draw[ids] = standing.to(torch.int8)

    def _disable_special_modes(self, ids, heading, generator):
        keys = self._final["terrain_key_ids"][ids]
        affected = ids[(keys >= 0) & self._final["disable_special"][keys.clamp_min(0)] & (self.special_mode_id[ids] >= 0)]
        if affected.numel():
            self.special_mode_id[affected] = -1
            self._sample_normal(affected[~self.is_standing[affected]], generator)
            self._postprocess_terrain(heading)

    def _sync_heading_target(self, ids, heading):
        aligned = ids[self._final["axis_mask"][ids]]
        if aligned.numel():
            self.heading_target[aligned] = self._final["reset_heading"][aligned]
            self._postprocess_terrain(heading)

    def _postprocess_terrain(self, heading):
        """Final-only extra hooks must not postprocess the flat reset cache."""
        if "source_flat_rehearsal" not in self._final:
            return self._postprocess(heading)
        terrain = ~self._final["source_flat_mask"]
        selected = terrain & self.is_heading & (self.special_mode_id < 0)
        error = _wrap_to_pi(self.heading_target[selected] - heading[selected])
        self.generator_command3[selected, 2] = (5. * error).clamp(-2 * torch.pi, 2 * torch.pi)
        self.generator_command3[terrain & self.is_standing] = 0.
        self.generator_after_postprocess3[terrain] = self.generator_command3[terrain]
        self.generator_cache_postprocessed[terrain] = True

    def _apply_overrides(self, ids):
        values, masks = self._final["override_values"], self._final["override_masks"]
        for index in (0, 1):
            selected = ids[masks[ids, index]]
            self.command3[selected, index] = values[selected, index]
        selected = ids[masks[ids, 2] & self.is_heading[ids]]
        self.command3[selected, 2] = values[selected, 2]
        selected = ids[masks[ids, 3] & ~self.is_heading[ids]]
        self.command3[selected, 2] = values[selected, 3]
        heading_ids = ids[self.is_heading[ids]]
        self.command3[heading_ids, 2] = self.generator_command3[heading_ids, 2]

    def _publish(self, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        if "source_flat_rehearsal" in self._final:
            selected = self._final["source_flat_mask"][ids]
            ScutFlatCommands._publish(self, ids[selected])
            ids = ids[~selected]
        # Pre-state-machine terrain overrides, followed by the inherited
        # ground temporary/expiry priority, then the final terrain pass.
        self.command3[ids] = self.generator_command3[ids]
        zero = ids[~self.is_heading[ids] & self._final["axis_mask"][ids] & ~self._final["override_masks"][ids, 3]]
        self.command3[zero, 2] = 0.
        self._apply_overrides(ids)
        pending = self.ground_override_pending[ids]
        active = pending & (self.episode_physics_ticks[ids] < 300)
        expired = ids[pending & ~active]
        if expired.numel():
            self.command3[expired] = self.ground_restore_command3[expired]
            self.generator_command3[expired] = self.ground_restore_command3[expired]
            self.generator_cache_postprocessed[expired] = False
            self.ground_override_pending[expired] = False
            self.ground_restore_command3[expired] = torch.nan
            self.ground_temporary_command3[expired] = torch.nan
        self.command3[ids[active]] = self.ground_temporary_command3[ids[active]]
        self._apply_overrides(ids)
        changed = (self.command3[ids] != self.generator_command3[ids]).any(-1)
        self.generator_cache_postprocessed[ids[changed]] = False
        self.generator_command3[ids] = self.command3[ids]

    def reset(self, ids, heading, ground_mask, generator, iteration=0, *,
              air_mode_index=None, birth_definition_sha256=None):
        ids = self._ids(ids)
        selected_heading = self._vector(heading, "heading", torch.float32, ids)
        ground = self._vector(ground_mask, "ground_mask", torch.bool, ids)
        generator, iteration = self._generator(generator), self._iteration(iteration)
        air_modes = self._air_reset_modes(ids, ground, air_mode_index, birth_definition_sha256)
        if heading.shape == (self.num_envs,):
            self._final["latest_heading"].copy_(heading)
        else:
            self._final["latest_heading"][ids] = selected_heading
        heading = self._final["latest_heading"]
        self._sync_terrain(ids, self._final["env_origins"])
        self._final["axis_mask"][ids] = self.axis_reset_mask(ids)
        self._final["reset_heading"][ids] = heading[ids]
        self.episode_physics_ticks[ids] = 0
        self.command_counter[ids] = 0
        self.ground_birth[ids] = ground
        self.holding[ids] = ground
        self.ground_override_pending[ids] = ground
        self.ground_restore_command3[ids] = torch.nan
        self.ground_temporary_command3[ids] = torch.nan
        self._reset_air_birth_limits(ids, air_modes)
        self._resample(ids, iteration, generator)
        self._resample_overrides(ids, generator)
        self._sample_height(ids, generator)
        ground_ids = ids[ground]
        if ground_ids.numel():
            self.ground_restore_command3[ground_ids] = self.generator_command3[ground_ids]
            self.ground_temporary_command3[ground_ids, 0] = self._uniform(ground_ids.numel(), -1., 1., generator)
            self.ground_temporary_command3[ground_ids, 1] = 0.
            self.ground_temporary_command3[ground_ids, 2] = self._uniform(ground_ids.numel(), -1., 1., generator)
        self._sync_heading_target(ids, heading)
        self._disable_special_modes(ids, heading, generator)
        # Native observation refresh clones the whole generator after a reset.
        # Heading synchronization may have postprocessed non-reset lanes too.
        if "source_flat_rehearsal" in self._final:
            # Keep the final whole-terrain refresh, but leave another flat
            # lane's ground-expiry cache for its own next command advance.
            publish = ~self._final["source_flat_mask"]
            publish[ids] = True
            self._publish(publish.nonzero().flatten())
        else:
            self._publish()

    def advance(self, dt, heading, iteration, generator, *, world_positions):
        if dt != .02 or self.physics_steps - self._last_command_physics_step != 4:
            raise ValueError("Final commands advance after exactly four 5ms ticks and the paid interval")
        # Validate the entire caller frame before consuming RNG or clocks.
        self.profile_indices(world_positions)
        heading = self._vector(heading, "heading", torch.float32)
        generator, iteration = self._generator(generator), self._iteration(iteration)
        self._final["latest_heading"].copy_(heading)
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.time_left -= dt
        self._resample((self.time_left <= 0.).nonzero().flatten(), iteration, generator)
        self._postprocess(heading)
        height_ids = (self.command_counter > self._height_seen_command_counter).nonzero().flatten()
        self._sync_terrain(height_ids, world_positions)
        self._sample_height(height_ids, generator)
        self._height_seen_command_counter.copy_(self.command_counter)
        self._sync_heading_target(height_ids, heading)
        self._resample_overrides(height_ids, generator)
        self._disable_special_modes(height_ids, heading, generator)
        changed = self._sync_terrain(all_ids, world_positions)
        changed = changed[~self.fixed_request_mask[changed]]
        if changed.numel():
            self.command_counter[changed] = 0
            self._resample(changed, iteration, generator)
            self._height_seen_command_counter.copy_(self.command_counter)
            self._sync_heading_target(changed, heading)
            self._sample_height(changed, generator)
            self._resample_overrides(changed, generator)
            self._final["profile_switch_count"][changed] += 1
        self._disable_special_modes(all_ids, heading, generator)
        self._publish()
        self._last_command_physics_step = self.physics_steps

    def summary(self):
        from .final_rough_definition import SOURCE_COMMIT
        result = super().summary()
        result.pop("source_archive", None)
        result.update(version=VERSION, source_commit=SOURCE_COMMIT,
            source_configuration="registered_final_rough_v1", terrain_keys=list(self._final["keys"]),
            column_keys=[self._final["keys"][i] for i in self._final["column_key_ids"].tolist()],
            terrain_profiles=deepcopy(self._final["profiles"]),
            terrain_scope="origins_at_reset_world_xy_during_episode", terrain_switch_hold_steps=0,
            terrain_velocity_overrides=True, terrain_disable_special_modes=True,
            reset_heading_candidates_deg=list(self._final["definition"]["birth"]["reset_heading_axis_aligned_candidates_deg"]),
            reset_heading_mask_scope="terrain_permission_at_episode_reset_then_latched",
            partial_reset_heading="selected_actual_reset_yaw_plus_other_lanes_last_policy_yaw",
            predefined_air_birth_permission_only=True, predefined_air_birth_sampler_implemented=False,
            predefined_air_birth_configuration=deepcopy(self._final["definition"]["birth"]["predefined_reset_air"]),
            height_observer_initial_count=1,
            height_sampling="reset plus generator-counter observer; final constructor synchronizes observer1, later resets preserve it",
            airborne_entry_command_resampling_implemented=False, stepup_command_link_implemented=False,
            source_final_environment_equivalent=False, origin_sha256=self._final["origin_sha256"])
        result["local_adaptations"].extend(("caller RNG stream differs from the native global stream",
            "fixed rehearsal uses the actual plane profile and nominal birth permission",
            "classified nonfinite world XY retains its previous profile until reset"))
        result["definition_sha256"] = hashlib.sha256(json.dumps(self._final["definition"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if "source_flat_rehearsal" in self._final:
            result["source_flat_training_rehearsal"] = {
                "source_archive": SOURCE_ARCHIVE, "plane_profile": "plane_for_rm",
                "height_range_m": [.2, .42], "height_observer_initial_count": 0,
                "source_iteration_offset": self._final["source_flat_rehearsal"]["source_iteration_offset"],
                "special_mode_iteration_starts": [mode[1] for mode in SPECIAL_MODES],
                "iteration_clock": "inherited flat offset plus rough local iteration",
                "rng_group_order": "rough lanes then source flat lanes, one caller generator",
                "mask_sha256": hashlib.sha256(self._final["source_flat_mask"].detach().cpu().numpy().tobytes()).hexdigest(),
                "scope": "source command distribution only; no birth or skill qualification"}
        if "air_birth_limits" in self._final:
            air = self._final["air_birth_limits"]
            result["air_birth_command_limits"] = {"version": AIR_LIMIT_VERSION,
                "definition_sha256": air["definition_sha256"],
                "modes": deepcopy(air["modes"]), "public_layout": ["vx_m_s", "yaw_rad_s", "height_m"],
                "clock": "float32 source countdown once per complete physical policy step",
                "priority": "after command/state-machine overrides; raw generator and height remain intact",
                "local_adaptation": "one active snapshot per frame, including duplicate publications at expiry",
                "automatic_birth_sampling_enabled": False}
        return result

    def diagnostics(self):
        result = {**super().diagnostics(), **{name: self._final[name].clone() for name in _EXTRA_TENSORS}}
        if "source_flat_rehearsal" in self._final:
            result["source_flat_training_rehearsal"] = self._final["source_flat_mask"].clone()
        if "air_birth_limits" in self._final:
            result["air_birth_command_limits"] = {name: self._final["air_birth_limits"][name].clone()
                                                  for name in _AIR_TENSORS}
        return result

    def metrics(self):
        result = super().metrics()
        if "source_flat_rehearsal" in self._final:
            result["/source_command/source_flat_training_rehearsal_fraction"] = (
                self._final["source_flat_mask"].float().mean())
        if "air_birth_limits" in self._final:
            state = self._final["air_birth_limits"]
            selected = state["mode_index"] >= 0
            prefix = "/source_birth/air_command/"
            result[prefix + "current_episode_fraction"] = selected.float().mean()
            result[prefix + "frame_active_fraction"] = state["frame_active"].float().mean()
            result[prefix + "remaining_mean_s"] = state["remaining_s"].sum() / selected.sum().clamp_min(1)
            for column, key in enumerate(("vx_m_s", "yaw_rad_s", "height_m")):
                result[prefix + "applied_" + key] = (
                    state["applied_public_command"][selected, column].sum() / selected.sum().clamp_min(1))
        return result

    def state_dict(self, generator):
        self._restart_inventory()
        values = {}
        for name in _COMMAND_TENSORS:
            value = getattr(self, name)
            if name in _GROUND_SENTINELS:
                if not bool(torch.isnan(value[~self.ground_override_pending]).all()):
                    raise ValueError("Inactive final command ground sentinel changed")
                value = torch.where(self.ground_override_pending[:, None], value, 0.)
            values[name] = value.detach().cpu().clone()
        source = dict(schema=COMMAND_STATE_VERSION, identity=self._restart_identity(), tensors=values,
            clocks={name: getattr(self, name) for name in _COMMAND_CLOCKS},
            generator_state=self._generator(generator).get_state().clone())
        state = dict(schema=STATE_VERSION, source=source,
            final={name: self._final[name].detach().cpu().clone() for name in _EXTRA_TENSORS})
        if "air_birth_limits" in self._final:
            air = self._final["air_birth_limits"]
            state["schema"] = AIR_STATE_VERSION
            state["air_limits"] = {"schema": AIR_LIMIT_VERSION,
                "definition_sha256": air["definition_sha256"],
                "tensors": {name: air[name].detach().cpu().clone() for name in _AIR_TENSORS}}
        self._prepare_restart(state, generator)
        return state

    def _prepare_air_restart(self, value, source):
        state = self._final["air_birth_limits"]
        if (not isinstance(value, dict) or set(value) != {"schema", "definition_sha256", "tensors"}
                or value["schema"] != AIR_LIMIT_VERSION
                or value["definition_sha256"] != state["definition_sha256"]
                or not isinstance(value["tensors"], dict) or set(value["tensors"]) != set(_AIR_TENSORS)):
            raise ValueError("Complete bound air command restart state required")
        checked = {}
        for name, (dtype, width) in _AIR_TENSORS.items():
            shape = (self.num_envs,) if width is None else (self.num_envs, width)
            target = state[name]
            if (not isinstance(target, torch.Tensor) or target.shape != shape or target.dtype != dtype
                    or target.device != self.device or target.requires_grad or not target.is_contiguous()):
                raise ValueError("Air command destination buffer changed: " + name)
            checked[name] = _restart_tensor(value["tensors"][name], shape, dtype, name)
        modes, remaining = checked["mode_index"], checked["remaining_s"]
        last, active = checked["last_policy_step"], checked["frame_active"]
        published = checked["applied_public_command"]
        if bool(((modes < -1) | (modes >= len(state["modes"]))).any()):
            raise ValueError("Air command restart mode is outside its source definition")
        duration = torch.tensor([mode["command_limits"]["duration_s"] for mode in state["modes"]], dtype=torch.float32)
        upper = torch.where(modes >= 0, duration[modes.clamp_min(0)], 0.)
        excluded = source[0]["ground_birth"].cpu() | self.fixed_request_mask.cpu() | ~self.air_birth_allowed().cpu()
        if "source_flat_rehearsal" in self._final:
            excluded |= self._final["source_flat_mask"].cpu()
        if (bool(((remaining < 0.) | (remaining > upper)).any())
                or bool(((last < -1) | (last > source[1]["physics_steps"] // 4)).any())
                or bool(((modes >= 0) & excluded).any())
                or bool((active & ((modes < 0) | (last < 0))).any())
                or bool((published[last < 0] != 0.).any())
                or bool(((last < 0) & (remaining != upper)).any())):
            raise ValueError("Air command restart clock, ownership or publication is invalid")
        for index, mode in enumerate(state["modes"]):
            selected = active & (modes == index)
            for column, key in enumerate(("lin_vel_x", "ang_vel_z", "height")):
                low, high = mode["command_limits"][key]
                if bool(((published[selected, column] < low) | (published[selected, column] > high)).any()):
                    raise ValueError("Air command restart publication violates its active source limits")
        return {name: tensor.to(self.device).clone() for name, tensor in checked.items()}

    def _prepare_restart(self, state, generator):
        with_air = "air_birth_limits" in self._final
        fields = {"schema", "source", "final"} | ({"air_limits"} if with_air else set())
        if (not isinstance(state, dict) or set(state) != fields
                or state["schema"] != (AIR_STATE_VERSION if with_air else STATE_VERSION)
                or not isinstance(state["final"], dict)
                or set(state["final"]) != set(_EXTRA_TENSORS)):
            raise ValueError("Complete final command restart schema required; old owners are incompatible")
        source = ScutFlatCommands._prepare_restart(self, state["source"], generator)
        checked = {name: _restart_tensor(state["final"][name],
            (self.num_envs,) if width is None else (self.num_envs, width), dtype, name)
            for name, (dtype, width) in _EXTRA_TENSORS.items()}
        keys = checked["terrain_key_ids"]
        if bool(((keys < -1) | (keys >= len(self._final["keys"]))).any()) or bool((checked["profile_switch_count"] < 0).any()):
            raise ValueError("Invalid final command terrain keys or switch counts")
        expected = self._final["key_profile_ids"].cpu()[keys.clamp_min(0)]
        expected[keys < 0] = -1
        if not torch.equal(expected, checked["terrain_profile_ids"]):
            raise ValueError("Final command terrain keys and resolved profiles disagree")
        masks = self._final["flags"].cpu()[keys.clamp_min(0)].clone()
        masks[(keys < 0) | self.fixed_request_mask.cpu()] = False
        if not torch.equal(masks, checked["override_masks"]):
            raise ValueError("Final command override masks disagree with the active terrain profile")
        prepared = source, {name: value.to(self.device).clone() for name, value in checked.items()}
        return (*prepared, self._prepare_air_restart(state["air_limits"], source)) if with_air else prepared

    def _commit_restart(self, prepared, generator):
        source, extra, *air = prepared
        ScutFlatCommands._commit_restart(self, source, generator)
        for name, value in extra.items():
            self._final[name].copy_(value)
        if air:
            for name, value in air[0].items():
                self._final["air_birth_limits"][name].copy_(value)
