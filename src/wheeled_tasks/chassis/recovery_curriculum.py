"""Episode-fixed recovery reset sampling; scene membership never changes."""
from __future__ import annotations

import math

import torch


LEVELS = ("easy", "medium", "hard")
SCHEMA = "v6-recovery-pose-curriculum-v1"


def validate_pose_curriculum(settings):
    """Validate the explicit schedule without importing a physics engine."""
    required = {"schema", "enabled", "clock", "origin_actor_update", "sampling",
                "interpolation", "keyframes", "profile_sets"}
    if (not isinstance(settings, dict) or set(settings) != required
            or settings["schema"] != SCHEMA or settings["enabled"] is not True
            or settings["clock"] != "domain_actor_updates"
            or settings["sampling"] != "episode_reset_only"
            or settings["interpolation"] != "linear"):
        raise ValueError("Recovery pose curriculum requires the explicit reset-only schema")
    origin = settings["origin_actor_update"]
    if type(origin) is not int or origin < 0:
        raise ValueError("Recovery curriculum origin must be an absolute nonnegative actor update")
    points = settings["keyframes"]
    if not isinstance(points, list) or not points:
        raise ValueError("Recovery curriculum needs probability keyframes")
    previous = -1
    for point in points:
        if not isinstance(point, dict) or set(point) != {"actor_update", "probabilities"}:
            raise ValueError("Invalid recovery probability keyframe fields")
        update, values = point["actor_update"], point["probabilities"]
        if (type(update) is not int or update <= previous or not isinstance(values, dict)
                or set(values) != set(LEVELS)
                or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                       for p in values.values())
                or not math.isclose(sum(values.values()), 1., abs_tol=1e-9, rel_tol=0.)):
            raise ValueError("Recovery probabilities must sum to one at increasing actor keyframes")
        previous = update
    if points[0]["actor_update"] != 0:
        raise ValueError("Recovery probability schedule must start at relative actor update zero")
    pools = settings["profile_sets"]
    if not isinstance(pools, dict) or not pools:
        raise ValueError("Recovery curriculum needs terrain-compatible profile sets")
    surfaces = set()
    for name, pool in pools.items():
        if (not isinstance(name, str) or not name or not isinstance(pool, dict)
                or set(pool) != {"terrain", "required_platform_height_m", "easy", "medium"}):
            raise ValueError("Invalid recovery profile-set fields")
        terrain, height = pool["terrain"], pool["required_platform_height_m"]
        if (terrain not in ("flat", "step_up", "stairs") or type(height) not in (int, float)
                or not math.isfinite(height) or height < 0 or (terrain == "flat") != (height == 0)):
            raise ValueError("Invalid recovery profile-set terrain or support height")
        surface = (terrain, float(height))
        if surface in surfaces:
            raise ValueError("Recovery profile-set surfaces must be unique")
        surfaces.add(surface)
        for level in ("easy", "medium"):
            names = pool[level]
            if (not isinstance(names, list) or not names
                    or any(not isinstance(value, str) or not value for value in names)
                    or len(set(names)) != len(names)):
                raise ValueError("Recovery profile sets require nonempty unique profile names")


def pose_probabilities(settings, actor_update):
    """Use the frozen absolute origin; construction before restored time is easy."""
    if type(actor_update) is not int or actor_update < 0:
        raise ValueError("Recovery pose sampling needs a nonnegative domain actor update")
    relative = max(0, actor_update - settings["origin_actor_update"])
    points = settings["keyframes"]
    for left, right in zip(points, points[1:]):
        if relative < right["actor_update"]:
            alpha = (relative - left["actor_update"]) / (right["actor_update"] - left["actor_update"])
            return tuple((1 - alpha) * left["probabilities"][level]
                         + alpha * right["probabilities"][level] for level in LEVELS)
    return tuple(points[-1]["probabilities"][level] for level in LEVELS)


class RecoveryResetCurriculum:
    """Sample only reset lanes from precompiled pose banks using env.generator.

    ``reset`` returns bank IDs aligned with its input IDs, with -1 for ordinary
    lanes. The caller gathers the public banks for nonnegative IDs and performs
    its normal reset state write. No physics, scene, command or FSM writes occur
    here. ``profile_ids`` and ``level_ids`` retain actual selections per episode;
    ordinary lane IDs remain -1. Banks use the simulator's full joint order.
    """

    def __init__(self, env, table, settings):
        validate_pose_curriculum(settings)
        if env.cfg.get("evaluation_exact_cases"):
            raise ValueError("Exact recovery evaluation must bypass pose curriculum sampling")
        if table.get("asset_manifest_sha256") != env.cfg.get("asset_manifest_sha256"):
            raise ValueError("Recovery pose bank belongs to a different asset")
        self.env = env
        # An external config mutation must not silently move the frozen clock.
        from copy import deepcopy
        self.settings = deepcopy(settings)
        self.profile_names = tuple(table["profiles"])
        if not self.profile_names:
            raise ValueError("Recovery pose bank is empty")
        self.profile_index = {name: index for index, name in enumerate(self.profile_names)}
        n, device = env.num_envs, env.device
        joint_names = env.robot.joint_names
        if set(table.get("tree_joint_names", ())) != set(joint_names):
            raise ValueError("Recovery pose bank joint set differs from the asset")
        if len(env.scene_groups) != n or len(env.kinds) != n:
            raise ValueError("Recovery curriculum requires fixed complete scene membership")
        roots, joints, releases, goals, endings, surfaces = [], [], [], [], [], []
        for name, profile in table["profiles"].items():
            if (profile.get("name") != name or profile.get("enabled") is not True
                    or profile.get("geometry_valid") is not True
                    or profile.get("state_writes") != "once_at_episode_reset_only"):
                raise ValueError(f"Recovery pose is not a validated reset candidate: {name}")
            pose = profile["root_position_m"] + profile["root_quaternion_xyzw"]
            q = profile["full_joint_positions"]
            release = profile.get("initial_release_s", 0.)
            height = profile.get("required_platform_height_m", 0.)
            ending = profile.get("min_ending_x", -math.inf)
            terrain = profile.get("terrain", "flat")
            if (len(pose) != 7 or any(not math.isfinite(value) for value in pose)
                    or not math.isclose(sum(value * value for value in pose[3:]), 1., abs_tol=1e-5)
                    or set(q) != set(joint_names) or any(not math.isfinite(value) for value in q.values())
                    or not math.isfinite(release) or release < 0 or not math.isfinite(height) or height < 0
                    or (not math.isfinite(ending) and ending != -math.inf)
                    or terrain not in ("flat", "step_up", "stairs")
                    or (terrain == "flat") != (height == 0)):
                raise ValueError(f"Invalid recovery pose bank values: {name}")
            if (any(profile.get("root_linear_velocity_m_s", [0., 0., 0.]))
                    or any(profile.get("root_angular_velocity_rad_s", [0., 0., 0.]))
                    or any(profile.get("full_joint_velocities", {}).values())):
                raise ValueError("Recovery reset bank currently requires zero initial velocities")
            roots.append(pose)
            joints.append([q[key] for key in joint_names])
            releases.append(release)
            goals.append(height)
            endings.append(ending)
            surfaces.append((terrain, float(height)))
        tensor = lambda values: torch.tensor(values, device=device, dtype=env.nominal.dtype)
        self.root_pose_bank = tensor(roots)
        self.joint_pose_bank = tensor(joints)
        self.initial_release_bank = tensor(releases)
        self.required_platform_height_bank = tensor(goals)
        self.min_ending_x_bank = tensor(endings)
        pools = {}
        for pool in settings["profile_sets"].values():
            surface = (pool["terrain"], float(pool["required_platform_height_m"]))
            pools[surface] = []
            for level in ("easy", "medium"):
                indices = []
                for name in pool[level]:
                    index = self.profile_index.get(name)
                    if index is None or surfaces[index] != surface:
                        raise ValueError(f"Recovery curriculum profile/terrain mismatch: {name}")
                    indices.append(index)
                pools[surface].append(indices)
        maximum = max(len(indices) for levels in pools.values() for indices in levels)
        candidates = torch.full((n, 3, maximum), -1, dtype=torch.long)
        counts = torch.zeros(n, 3, dtype=torch.long)
        enabled = torch.zeros(n, dtype=torch.bool)
        for lane, group in enumerate(env.scene_groups):
            spec = env.skills.specs.get(group, {})
            name = spec.get("recovery_pose")
            if name is None:
                continue
            index = self.profile_index.get(name)
            if index is None or surfaces[index][0] != env.kinds[lane]:
                raise ValueError(f"Recovery lane profile/terrain mismatch: {name}")
            if surfaces[index] not in pools:
                raise ValueError(f"Recovery lane has no compatible curriculum surface: {name}")
            for level, options in enumerate([*pools[surfaces[index]], [index]]):
                candidates[lane, level, :len(options)] = torch.tensor(options)
                counts[lane, level] = len(options)
            enabled[lane] = True
        self.candidates = candidates.to(device)
        self.candidate_counts = counts.to(device)
        self.enabled_count = int(enabled.sum())
        self.enabled = enabled.to(device)
        self.profile_ids = torch.full((n,), -1, dtype=torch.long, device=device)
        self.level_ids = torch.full_like(self.profile_ids, -1)
        self.profile_reset_counts = torch.zeros(len(self.profile_names), dtype=torch.long, device=device)
        self.level_reset_counts = torch.zeros(3, dtype=torch.long, device=device)
        self._cached_actor_update = None
        self._cdf = None

    def reset(self, ids, actor_update):
        """Return chosen bank IDs; preserve all lanes not present in this reset."""
        probabilities = pose_probabilities(self.settings, actor_update)
        if self._cached_actor_update != actor_update:
            self._cdf = torch.tensor(probabilities, dtype=torch.float64, device=self.env.device).cumsum(0)
            self._cdf[-1] = 1.
            self._cached_actor_update = actor_update
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.env.device)
        if ids.ndim != 1:
            raise ValueError("Recovery reset IDs must be one-dimensional")
        active = self.enabled[ids]
        lanes = ids[active]
        result = torch.full_like(ids, -1)
        if not len(lanes):
            return result
        draws = torch.rand((len(lanes), 2), device=self.env.device, generator=self.env.generator)
        levels = torch.searchsorted(self._cdf, draws[:, 0].contiguous(), right=True).clamp_max(2)
        option = (draws[:, 1] * self.candidate_counts[lanes, levels]).long()
        chosen = self.candidates[lanes, levels, option]
        self.profile_ids[lanes], self.level_ids[lanes] = chosen, levels
        self.profile_reset_counts += torch.bincount(chosen, minlength=len(self.profile_names))
        self.level_reset_counts += torch.bincount(levels, minlength=3)
        result[active] = chosen
        return result

    def metrics(self, actor_update):
        """Leave counters on-device; only the existing logger converts scalars."""
        probabilities = pose_probabilities(self.settings, actor_update)
        prefix = "/recovery/curriculum/"
        result = {prefix + "relative_actor_updates": float(max(
            0, actor_update - self.settings["origin_actor_update"]))}
        for index, level in enumerate(LEVELS):
            result[prefix + level + "_probability"] = float(probabilities[index])
            result[prefix + level + "_actuallevel_fraction"] = (
                (self.level_ids == index).sum() / max(1, self.enabled_count))
            result[prefix + "level_reset_count/" + level] = self.level_reset_counts[index].clone()
        for index, name in enumerate(self.profile_names):
            result[prefix + "profile_reset_count/" + name] = self.profile_reset_counts[index].clone()
        return result

    def diagnostics(self):
        """Capture actual episode selections before auto-reset replaces them."""
        return {"recovery_actual_profile_id": self.profile_ids.clone(),
                "recovery_actual_level_id": self.level_ids.clone()}
