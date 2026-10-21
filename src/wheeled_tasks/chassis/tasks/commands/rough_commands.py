"""Pinned 0730 terrain height profiles around the unchanged source sampler.

The effective saved terrain_command_overrides contains height overrides only.
The extra stair_for_rm_rot in rough_terrain_command_overrides_cfg is an inactive
class template, not one of this archive's nine generated terrain keys.
"""
import hashlib

import torch

from wheeled_tasks.chassis.tasks.commands.restart_state import _restart_tensor
from wheeled_tasks.chassis.tasks.commands.flat_command_sampling import ScutFlatCommands, SPECIAL_MODES, COMMAND_STATE_VERSION, _COMMAND_TENSORS, _COMMAND_CLOCKS, _GROUND_SENTINELS, SOURCE_ARCHIVE
from wheeled_tasks.chassis.tasks.terrain.rough_definition import ENV_FILE, ENV_SHA, _read_bound, column_names


VERSION = "scut0730_rough_commands_v1"
_STATE_TENSORS = ("terrain_key_ids", "terrain_profile_ids", "profile_switch_count")


def command_definition():
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    for tag in ("python/tuple", "python/object/apply:builtins.slice"):
        Loader.add_constructor("tag:yaml.org,2002:" + tag,
                               lambda loader, node: loader.construct_sequence(node))
    env = yaml.load(_read_bound(ENV_FILE, ENV_SHA), Loader=Loader)
    generator = env["terrain"]["terrain_generator"]
    keys = tuple(generator["sub_terrains"])
    overrides = env["terrain_command_overrides"]
    ranges, profile_ids, profiles = [], [], []
    for key in keys:
        override = overrides.get(key, {})
        if any(value is not None for name, value in override.items() if name != "height_range"):
            raise ValueError("0730 adapter requires the saved height-only terrain overrides")
        height = override.get("height_range") or [env["height_range"]]
        if len(height) != 1:
            raise ValueError("0730 height profile must be one interval")
        pair = tuple(height[0])
        ranges.append(pair)
        if pair not in profiles:
            profiles.append(pair)
        profile_ids.append(profiles.index(pair))
    if (env["terrain_command_switch_hold_steps"] != 0 or env["reset_heading_axis_aligned_only"]
            or any(mode["iteration_start"] != 0 or mode["iteration_end"] != -1
                   or mode["jump_takeoff_enabled"] or any(mode[k] is not None for k in
                       ("height_range", "height_wave", "height_step")) for mode in env["commands"]["special_modes"])):
        raise ValueError("0730 sampler assumptions differ from the bound source")
    return dict(keys=keys, column_key_ids=tuple(keys.index(name) for name in column_names(env["terrain"])),
                height_ranges=tuple(ranges), profile_ids=tuple(profile_ids),
                tile_size_m=tuple(generator["size"]), rows=generator["num_rows"], columns=generator["num_cols"])


class ScutRoughCommands(ScutFlatCommands):
    """0730 sampler; terrain profiles follow current world position at runtime.

    Additional restart state is explicit. Slots preserve the base owner's
    complete flat tensor inventory and leave flat behavior unchanged.
    """
    __slots__ = ("_rough",)

    @property
    def special_modes(self):
        return self._rough.get("final_modes", tuple(
            (name, 0, probability, vx, yaw) for name, _, probability, vx, yaw in SPECIAL_MODES))

    def __init__(self, num_envs, device, *, terrain_origins, env_origins):
        super().__init__(num_envs, device)
        definition = command_definition()
        terrain_origins = torch.as_tensor(terrain_origins, dtype=torch.float32, device=self.device)
        origins = torch.as_tensor(env_origins, dtype=torch.float32, device=self.device)
        if (terrain_origins.shape != (10, 14, 3) or origins.shape != (num_envs, 3)
                or not bool(torch.isfinite(terrain_origins).all()) or not bool(torch.isfinite(origins).all())):
            raise ValueError("0730 commands require complete generated and ordered environment origins")
        self._rough = dict(definition=definition, env_origins=origins.detach().clone(),
            start_xy=terrain_origins[0, 0, :2] - torch.tensor(definition["tile_size_m"], device=self.device) * .5,
            column_key_ids=torch.tensor(definition["column_key_ids"], dtype=torch.long, device=self.device),
            key_profile_ids=torch.tensor(definition["profile_ids"], dtype=torch.long, device=self.device),
            terrain_key_ids=torch.full((num_envs,), -1, dtype=torch.long, device=self.device),
            terrain_profile_ids=torch.full((num_envs,), -1, dtype=torch.long, device=self.device),
            profile_switch_count=torch.zeros(num_envs, dtype=torch.long, device=self.device),
            source_flat_rehearsal=None,
            origin_sha256=hashlib.sha256(origins.cpu().numpy().tobytes()
                + terrain_origins.cpu().numpy().tobytes()).hexdigest())

    def configure_final_motion_modes(self):
        """Bind the reviewed final pool before reset, preserving terrain geometry."""
        from wheeled_tasks.chassis.tasks.commands.final_rough_commands import final_special_modes
        from wheeled_tasks.chassis.tasks.terrain.final_rough_definition import load_final_rough_definition
        if (self.physics_steps or bool((self.command_counter != 0).any())
                or "final_modes" in self._rough):
            raise ValueError("Final motion modes must be configured once before the first reset")
        self._rough["final_modes"] = final_special_modes(load_final_rough_definition()["commands"])

    def set_source_flat_rehearsal(self, ids, *, source_iteration_offset=4500):
        """Bind static training lanes to the complete 0719 command sampler.

        Call before the first reset. This is constructor configuration, not a
        live switch or fixed request. Full restart identity binds the lane mask
        and inherited flat iteration, while mutable buffers/RNG use the normal
        complete command checkpoint.
        """
        ids = self._ids(ids)
        offset = self._iteration(source_iteration_offset)
        if (self.physics_steps or bool((self.command_counter != 0).any())
                or self._rough["source_flat_rehearsal"] is not None):
            raise ValueError("Source flat rehearsal must be configured once before the first reset")
        if not ids.numel() or bool(self.fixed_request_mask[ids].any()):
            raise ValueError("Source flat rehearsal needs nonempty lanes without fixed requests")
        if self._temporal_rehearsal is not None and bool(self._temporal_rehearsal.mask[ids].any()):
            raise ValueError("Source flat and temporal request lanes must be disjoint")
        mask = torch.zeros_like(self.fixed_request_mask)
        mask[ids] = True
        self._rough["source_flat_rehearsal"] = {"source_iteration_offset": offset, "mask": mask}

    def _store_fixed_requests(self, ids, command3, height):
        ids = self._ids(ids)
        if (self._rough["source_flat_rehearsal"] is not None
                and bool(self._rough["source_flat_rehearsal"]["mask"][ids].any())):
            raise ValueError("Source flat rehearsal lanes cannot become fixed requests")
        super()._store_fixed_requests(ids, command3, height)

    def set_temporal_training_requests(self, ids, requests, *, phase_policy_ticks, episode_policy_ticks):
        ids = self._ids(ids)
        config = self._rough["source_flat_rehearsal"]
        if config is not None and bool(config["mask"][ids].any()):
            raise ValueError("Source flat and temporal request lanes must be disjoint")
        super().set_temporal_training_requests(ids, requests,
            phase_policy_ticks=phase_policy_ticks, episode_policy_ticks=episode_policy_ticks)

    def set_motion_training_requests(self, ids, settings, *, episode_policy_ticks=999):
        ids = self._ids(ids)
        config = self._rough["source_flat_rehearsal"]
        if config is not None and bool(config["mask"][ids].any()):
            raise ValueError("Dynamic flat and motion request lanes must be disjoint")
        super().set_motion_training_requests(ids, settings, episode_policy_ticks=episode_policy_ticks)

    def _resample(self, ids, iteration, generator):
        config = self._rough["source_flat_rehearsal"]
        if config is None:
            return super()._resample(ids, iteration, generator)
        selected = config["mask"][ids]
        # One caller-owned RNG, deterministic group order, no new RNG or seed.
        # Historical courses retain their archived starts. The reviewed final
        # pool is shared by dynamic flat and rough without changing either clock.
        super()._resample(ids[~selected], iteration, generator)
        super()._resample(ids[selected], iteration + config["source_iteration_offset"],
                          generator, special_modes=self._rough.get("final_modes", SPECIAL_MODES))

    def _sync_terrain(self, ids, world_positions):
        r = self._rough
        finite = torch.isfinite(world_positions[ids, :2]).all(-1)
        y = torch.where(finite, world_positions[ids, 1], r["env_origins"][ids, 1])
        col = torch.floor((y - r["start_xy"][1]) / 9.).long().clamp(0, 13)
        keys = r["column_key_ids"][col]
        # Numeric terminal lanes are reset immediately after command advance.
        # Preserve their old profile; raw native diagnostics remain untouched.
        old = r["terrain_key_ids"][ids]
        keys = torch.where(~finite & (old >= 0), old, keys)
        flat = torch.zeros_like(ids, dtype=torch.bool)
        if self._training_rehearsal_mask is not None:
            flat |= self._training_rehearsal_mask[ids]
        if r["source_flat_rehearsal"] is not None:
            flat |= r["source_flat_rehearsal"]["mask"][ids]
        if self._temporal_rehearsal is not None:
            flat |= self._temporal_rehearsal.mask[ids]
        if (self._training_rehearsal_mask is not None
                or r["source_flat_rehearsal"] is not None or self._temporal_rehearsal is not None):
            plane_key = r["definition"]["keys"].index("plane_for_rm_rot")
            keys = torch.where(flat, plane_key, keys)
        profiles = r["key_profile_ids"][keys]
        changed = ids[profiles != r["terrain_profile_ids"][ids]]
        r["terrain_key_ids"][ids] = keys
        r["terrain_profile_ids"][ids] = profiles
        return changed

    def _sample_height(self, ids, generator):
        source_flat_ids = ids[:0]
        if self._rough["source_flat_rehearsal"] is not None:
            selected = self._rough["source_flat_rehearsal"]["mask"][ids]
            source_flat_ids, ids = ids[selected], ids[~selected]
        fixed = ids[self.fixed_request_mask[ids]]
        self.height[fixed] = self.fixed_request_height[fixed]
        ids = ids[~self.fixed_request_mask[ids]]
        keys = self._rough["terrain_key_ids"][ids]
        # Match source TerrainCommandManager's sorted terrain-key grouping,
        # including distinct keys that resolve to the same height interval.
        for key in torch.unique(keys).tolist():
            selected = ids[keys == key]
            low, high = ((.2, .42) if key < 0 else self._rough["definition"]["height_ranges"][key])
            self.height[selected] = self._uniform(selected.numel(), low, high, generator)
        if source_flat_ids.numel():
            ScutFlatCommands._sample_height(self, source_flat_ids, generator)

    def reset(self, ids, heading, ground_mask, generator, iteration=0):
        ids = self._ids(ids)
        self._sync_terrain(ids, self._rough["env_origins"])
        super().reset(ids, heading, ground_mask, generator, iteration)

    def _publish(self, ids=None):
        super()._publish(ids)
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        if self._rough["source_flat_rehearsal"] is not None:
            ids = ids[~self._rough["source_flat_rehearsal"]["mask"][ids]]
        # V13's final hook restores heading yaw after the ground override and
        # synchronizes the actual command back into the mutable generator cache.
        heading_ids = ids[self.is_heading[ids]]
        self.command3[heading_ids, 2] = self.generator_command3[heading_ids, 2]
        overwritten = (self.generator_command3[ids] != self.command3[ids]).any(-1)
        self.generator_cache_postprocessed[ids[overwritten]] = False
        self.generator_command3[ids] = self.command3[ids]

    def advance(self, dt, heading, iteration, generator, *, world_positions):
        if dt != .02 or self.physics_steps - self._last_command_physics_step != 4:
            raise ValueError("0730 command advance needs exactly four 5ms ticks")
        if (not isinstance(world_positions, torch.Tensor) or world_positions.shape != (self.num_envs, 3)
                or world_positions.device != self.device):
            raise ValueError("0730 runtime terrain lookup requires current world root positions [N,3]")
        heading = self._vector(heading, "heading", torch.float32)
        generator = self._generator(generator)
        iteration = self._iteration(iteration)
        self.time_left -= dt
        self._resample((self.time_left <= 0.).nonzero(as_tuple=False).flatten(), iteration, generator)
        self._postprocess(heading)
        height_ids = (self.command_counter > self._height_seen_command_counter).nonzero(as_tuple=False).flatten()
        # Source height observer synchronizes profiles before the switch hook.
        # A simultaneous timer resample therefore does not force a second draw.
        self._sync_terrain(height_ids, world_positions)
        self._sample_height(height_ids, generator)
        self._height_seen_command_counter.copy_(self.command_counter)
        changed = self._sync_terrain(torch.arange(self.num_envs, device=self.device), world_positions)
        changed = changed[~self.fixed_request_mask[changed]]
        if changed.numel():
            # CommandTerm.reset clears only its resampling counter; robot
            # episode age (and the 5s special readiness) continues unchanged.
            self.command_counter[changed] = 0
            self._resample(changed, iteration, generator)
            self._height_seen_command_counter.copy_(self.command_counter)
            self._sample_height(changed, generator)
            self._rough["profile_switch_count"][changed] += 1
        self._publish()
        self._last_command_physics_step = self.physics_steps

    def summary(self):
        result = super().summary()
        result.update(version=VERSION, source_archive=ENV_FILE, source_env_sha256=ENV_SHA,
            terrain_profiles={key: list(height) for key, height in
                zip(self._rough["definition"]["keys"], self._rough["definition"]["height_ranges"])},
            terrain_scope="env_origins_at_reset_current_world_xy_during_episode",
            profile_switch="immediate_generator_reset_and_height_draw_only_when_resolved_profile_changes",
            terrain_switch_hold_steps=0, terrain_velocity_overrides=False,
            terrain_disable_special_modes=False, heading_reset_axis_aligned=False,
            origins_sha256=self._rough["origin_sha256"])
        result["local_adaptations"].append("nonfinite terminal position retains terrain profile until reset")
        if "final_modes" in self._rough:
            from wheeled_tasks.chassis.tasks.terrain.final_rough_definition import SOURCE_COMMIT
            result["mode_pool"] = {
                "version": "final-motion-mode-pool-v1", "source_commit": SOURCE_COMMIT,
                "scope": "final_special_modes_on_retained_small_terrain_and_dynamic_flat",
                "probability_denominator": "eligible_nonstanding_resampling_pool"}
        if self._rough["source_flat_rehearsal"] is not None:
            result["source_flat_training_rehearsal"] = {
                "source_archive": SOURCE_ARCHIVE,
                "plane_profile": "plane_for_rm_rot",
                "height_range_m": [.2, .42],
                "source_iteration_offset": self._rough["source_flat_rehearsal"]["source_iteration_offset"],
                "special_mode_iteration_starts": [mode[1] for mode in
                    self._rough.get("final_modes", SPECIAL_MODES)],
                "iteration_clock": "inherited flat offset plus rough local iteration",
                "rng_group_order": "rough lanes then source flat lanes, one caller generator",
                "mask_sha256": hashlib.sha256(
                    self._rough["source_flat_rehearsal"]["mask"].cpu().numpy().tobytes()).hexdigest(),
                "scope": "source-distribution training rehearsal; fixed evaluation remains separate"}
        if self._training_rehearsal_mask is not None:
            # Base restart identity already binds summary(). This static mask
            # is validated, never restored over the constructor's allocation.
            result["fixed_training_rehearsal"] = {
                "plane_profile": "plane_for_rm_rot",
                "mask_sha256": hashlib.sha256(
                    self._training_rehearsal_mask.detach().cpu().numpy().tobytes()).hexdigest(),
                "scope": "explicit static training allocation; not fixed evaluation or measured terrain exposure"}
        return result

    def diagnostics(self):
        result = dict(super().diagnostics(), **{name: self._rough[name].clone() for name in _STATE_TENSORS})
        if self._rough["source_flat_rehearsal"] is not None:
            result["source_flat_training_rehearsal"] = self._rough["source_flat_rehearsal"]["mask"].clone()
        return result

    def metrics(self):
        result = super().metrics()
        if self._rough["source_flat_rehearsal"] is not None:
            mask = self._rough["source_flat_rehearsal"]["mask"]
            result["/source_command/source_flat_training_rehearsal_fraction"] = mask.float().mean()
            heights = self.height[mask]
            # Diagnostic bins reuse the source terrain interval boundaries;
            # they do not create height levels or change uniform sampling.
            prefix = "/source_command/source_flat_rehearsal_height_"
            result[prefix + "mean_m"] = heights.mean()
            result[prefix + "low_fraction"] = (heights < .24).float().mean()
            result[prefix + "middle_fraction"] = ((heights >= .24) & (heights < .38)).float().mean()
            result[prefix + "high_fraction"] = (heights >= .38).float().mean()
        result["/source_command/terrain_profile_switch_count"] = self._rough["profile_switch_count"].sum()
        for key, name in enumerate(self._rough["definition"]["keys"]):
            result["/source_command/terrain_fraction/" + name] = (self._rough["terrain_key_ids"] == key).float().mean()
        return result

    def _restart_inventory(self):
        super()._restart_inventory()
        expected = {"definition", "env_origins", "start_xy", "column_key_ids", "key_profile_ids",
                    "origin_sha256", "source_flat_rehearsal", *_STATE_TENSORS}
        if "final_modes" in self._rough:
            expected.add("final_modes")
        if set(self._rough) != expected:
            raise ValueError("Rough command restart inventory changed; version its complete state")
        for name in _STATE_TENSORS:
            value = self._rough[name]
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                    or value.dtype != torch.long or value.device != self.device
                    or value.layout != torch.strided or not value.is_contiguous() or value.requires_grad):
                raise ValueError("Rough command state buffer changed shape/dtype/device")

    def state_dict(self, generator):
        self._restart_inventory()
        values = {}
        for name in _COMMAND_TENSORS:
            value = getattr(self, name)
            if name in _GROUND_SENTINELS:
                if not bool(torch.isnan(value[~self.ground_override_pending]).all()):
                    raise ValueError("Inactive rough ground sentinel changed")
                value = torch.where(self.ground_override_pending[:, None], value, 0.)
            values[name] = value.detach().cpu().clone()
        source = dict(schema=COMMAND_STATE_VERSION, identity=self._restart_identity(), tensors=values,
            clocks={name: getattr(self, name) for name in _COMMAND_CLOCKS},
            generator_state=self._generator(generator).get_state().clone())
        if getattr(self._temporal_rehearsal, "mutable", False):
            source["motion_course"] = self._temporal_rehearsal.state_dict()
        state = dict(schema=VERSION, source=source,
                     rough={name: self._rough[name].detach().cpu().clone() for name in _STATE_TENSORS})
        self._prepare_restart(state, generator)
        return state

    def _prepare_restart(self, state, generator):
        if (not isinstance(state, dict) or set(state) != {"schema", "source", "rough"}
                or state["schema"] != VERSION or not isinstance(state["rough"], dict)
                or set(state["rough"]) != set(_STATE_TENSORS)):
            raise ValueError("Complete 0730 terrain command restart schema required")
        source = ScutFlatCommands._prepare_restart(self, state["source"], generator)
        checked = {name: _restart_tensor(state["rough"][name], (self.num_envs,), torch.long, name)
                   for name in _STATE_TENSORS}
        keys, profiles = checked["terrain_key_ids"], checked["terrain_profile_ids"]
        if (bool(((keys < -1) | (keys >= 9)).any()) or bool((checked["profile_switch_count"] < 0).any())):
            raise ValueError("Invalid terrain command restart keys or switch counts")
        expected = self._rough["key_profile_ids"].cpu()[keys.clamp_min(0)]
        expected[keys < 0] = -1
        if not torch.equal(expected, profiles):
            raise ValueError("Terrain key and resolved profile restart disagree")
        return source, {name: value.to(self.device).clone() for name, value in checked.items()}

    def _commit_restart(self, prepared, generator):
        source, extra = prepared
        ScutFlatCommands._commit_restart(self, source, generator)
        for name, value in extra.items():
            self._rough[name].copy_(value)
