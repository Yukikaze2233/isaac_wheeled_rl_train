"""Lane ownership for retained flat, temporal, recovery and small rough skills."""
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .contact_generation_distance import (
    KEY as CONTACT_KEY, SUPPORTED_VERSIONS as CONTACT_VERSIONS,
    LEG_LINK_VERSION, ROUGH_WHEEL_VERSION, UPPER_LINK_VERSION, VERSION as BASE_CONTACT_VERSION,
)

KEY = "formal_small_rough_mix"
VERSION = "v7-small-rough-flat-recovery-v1"
MOTION_VERSION = "motion-retention-course-v1"
FINAL_MOTION_VERSION = "motion-retention-final-payments-v2"
MOTION_VERSIONS = (MOTION_VERSION, FINAL_MOTION_VERSION)
ROUGH_BIRTH_KEY = "formal_rough_nonsteady_birth"
ROUGH_BIRTH_VERSION = "rough-continuous-source-birth-v1"
ROUGH_ANCHOR_KEY = "formal_rough_birth_anchor"
ROUGH_ANCHOR_VERSION = "terrain-center-v1"
INACTIVE_QUIET_REWARD_KEYS = frozenset((
    "quiet_velocity_weight", "quiet_wheel_weight", "quiet_wheel_deadband_m_s",
))
ROLES = ("fixed_flat", "dynamic_flat", "temporal_flat", "recovery", "source_rough")


def selection(config):
    """Validate the local adaptation; caller supplies fractions and dwell times."""
    if KEY not in config:
        if ROUGH_BIRTH_KEY in config or ROUGH_ANCHOR_KEY in config:
            raise ValueError("Continuous rough births require the explicit mixed motion course")
        return None
    value = config[KEY]
    fields = {"version", "fractions", "temporal", "source_flat_iteration_offset"}
    final = isinstance(value, Mapping) and value.get("version") == FINAL_MOTION_VERSION
    reward_profile = "v7_final_rough_v1" if final else "scut0730_rough_v1"
    rehearsal_profile = "v7_final_rough_v1" if final else "scut0719_flat_v1"
    if isinstance(value, Mapping) and value.get("version") in MOTION_VERSIONS:
        fields.add("geometry_seed")
    if (not isinstance(value, Mapping)
            or set(value) != fields
            or value["version"] not in (VERSION, *MOTION_VERSIONS)
            or config.get("reward_profile") != reward_profile
            or config.get("curriculum_stage") != "terrain"
            or config.get("enabled_stages") != ["terrain"]
            or config.get("formal_terrain") != {"type": "rough", "source": "scut0730"}
            or config.get("formal_rough_initial_level") != 0
            or config.get("formal_foundation_nominal_birth") is not False
            or config.get("formal_learning_state_resume") is not True
            or config.get(CONTACT_KEY) not in CONTACT_VERSIONS
            or any(key in config for key in ("formal_flat_refresh", "formal_scripted_recovery_mix",
                                             "formal_final_rough_stage", "formal_stepup_command"))):
        raise ValueError("Mixed small rough requires the explicit retained-skill terrain course")
    if ROUGH_BIRTH_KEY in config and (
            value["version"] not in MOTION_VERSIONS or config[ROUGH_BIRTH_KEY] != ROUGH_BIRTH_VERSION):
        raise ValueError("Continuous rough births require their explicit motion-course selector")
    if ROUGH_ANCHOR_KEY in config and (
            config.get(ROUGH_ANCHOR_KEY) != ROUGH_ANCHOR_VERSION
            or config.get(ROUGH_BIRTH_KEY) != ROUGH_BIRTH_VERSION):
        raise ValueError("Centered rough births require the explicit continuous-birth selector")
    fractions = value["fractions"]
    if (not isinstance(fractions, Mapping) or set(fractions) != set(ROLES)
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1
                   for v in fractions.values())
            or not math.isclose(math.fsum(fractions.values()), 1., abs_tol=1e-9)):
        raise ValueError("Mixed course requires five positive fractions summing to one")
    from .fixed_flat_rehearsal import (FIXED17_VERSION, FIXED17_CASE_FILE, FIXED17_CASE_SHA,
                                      DYNAMIC_NONSTEADY_BIRTH_VERSION)
    if (config.get("formal_flat_rehearsal") != {
            "version": FIXED17_VERSION, "fraction": fractions["fixed_flat"],
            "cases_file": FIXED17_CASE_FILE, "cases_sha256": FIXED17_CASE_SHA,
            "allocation": "equal_case_lanes", "reward_profile": rehearsal_profile}
            or config.get("formal_dynamic_nonsteady_birth") != DYNAMIC_NONSTEADY_BIRTH_VERSION):
        raise ValueError("Mixed course must retain fixed17 and explicit dynamic flat births")
    offset = value["source_flat_iteration_offset"]
    temporal = value["temporal"]
    if type(offset) is not int or offset < 4000:
        raise ValueError("Mixed course must preserve previously unlocked motion modes")
    if final and "formal_tracking_reward_repair" in config:
        raise ValueError("Final motion payments replace the historical tracking repair")
    if value["version"] in MOTION_VERSIONS:
        from .motion_transition_course import validate_settings
        from .actor_sensor_noise import validate_actor_sensor_noise
        from .dynamics import validate_episode_spring_force_randomization
        from .critic_adaptation import selection as critic_adaptation_selection
        validate_settings(temporal, episode_ticks=(1000 if "formal_recovery_episode_window" in config else 999))
        if type(value["geometry_seed"]) is not int or not 0 <= value["geometry_seed"] < 2**32:
            raise ValueError("Motion course requires a separate uint32 terrain geometry seed")
        if (validate_actor_sensor_noise(config) is None
                or validate_episode_spring_force_randomization(config) is None
                or critic_adaptation_selection(config) is None):
            raise ValueError("Motion course requires explicit noise, spring randomization and critic adaptation")
    elif (not isinstance(temporal, Mapping)
            or set(temporal) != {"speeds_m_s", "height_m", "phase_policy_ticks"}
            or not isinstance(temporal["speeds_m_s"], list) or not temporal["speeds_m_s"]
            or any(type(v) not in (int, float) or not math.isfinite(v) or not .1 <= v <= 3.
                   for v in temporal["speeds_m_s"])
            or len(set(temporal["speeds_m_s"])) != len(temporal["speeds_m_s"])
            or type(temporal["height_m"]) not in (int, float)
            or not .2 <= temporal["height_m"] <= .42
            or not isinstance(temporal["phase_policy_ticks"], list)
            or len(temporal["phase_policy_ticks"]) != 4
            or any(type(t) is not int or t <= 0 for t in temporal["phase_policy_ticks"])
            or sum(temporal["phase_policy_ticks"]) != 999):
        raise ValueError("Mixed course requires paired speeds and four phases filling the 999-interval horizon")
    if "formal_recovery_episode_window" in config:
        from .flat_episode_selection import validate_recovery_episode_window
        if value["version"] not in MOTION_VERSIONS:
            raise ValueError("Extended mixed episode timing requires the motion course")
        validate_recovery_episode_window(config)
    scenes = config.get("scene_groups", [])
    if not scenes or any(not isinstance(s, Mapping) for s in scenes):
        raise ValueError("Mixed course requires explicit scene groups")
    totals = dict.fromkeys(ROLES, 0.)
    names = []
    for scene in scenes:
        name = scene.get("name")
        role = "recovery" if "recovery_pose" in scene else name
        if (role not in ROLES or not isinstance(name, str) or name in names
                or role == "recovery" and "recovery_pose" not in scene
                or scene.get("terrain") != (["rough"] if role == "source_rough" else ["flat"])
                or type(scene.get("fraction")) not in (int, float)
                or not math.isfinite(scene["fraction"]) or not 0 < scene["fraction"] < 1):
            raise ValueError("Mixed scene identity, physical surface and fraction must agree")
        names.append(name)
        totals[role] += scene["fraction"]
    if any(not math.isclose(totals[k], fractions[k], abs_tol=1e-9) for k in ROLES):
        raise ValueError("Scene fractions differ from the reviewed mixed allocation")
    return deepcopy(dict(value))


def allocate(config, root, num_envs, groups):
    """Return one allocation reused for commands, geometry, birth and payment."""
    settings = selection(config)
    if settings is None or len(groups) != num_envs:
        raise ValueError("Mixed allocation requires actual environment scene membership")
    scenes = {s["name"]: s for s in config["scene_groups"]}
    roles = ["recovery" if "recovery_pose" in scenes[g] else g for g in groups]
    counts = Counter(roles)
    for name in ROLES:
        expected = settings["fractions"][name] * num_envs
        if expected != int(expected) or counts[name] != int(expected):
            raise ValueError("Mixed training needs exact integer role allocations")
    fixed = config["formal_flat_rehearsal"]
    raw = (Path(root) / fixed["cases_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != fixed["cases_sha256"]:
        raise ValueError("Fixed17 working points changed")
    cases = json.loads(raw)["cases"]
    fixed_ids = [i for i, role in enumerate(roles) if role == "fixed_flat"]
    temporal_ids = [i for i, role in enumerate(roles) if role == "temporal_flat"]
    if (len(fixed_ids) < len(cases) or not temporal_ids
            or settings["version"] == VERSION
            and len(temporal_ids) % (2 * len(settings["temporal"]["speeds_m_s"]))):
        raise ValueError("Mixed lanes must cover fixed17 and equal paired temporal requests")
    indices = [-1] * num_envs
    for ordinal, lane in enumerate(fixed_ids):
        indices[lane] = ordinal % len(cases)
    plane_ids = [i for i, role in enumerate(roles) if role != "source_rough"]
    columns = math.ceil(math.sqrt(len(plane_ids)))
    rows = math.ceil(len(plane_ids) / columns)
    origins = {lane: ((i % columns - (columns - 1) / 2) * 4.,
                      300. + (i // columns - (rows - 1) / 2) * 4., 0.)
               for i, lane in enumerate(plane_ids)}
    return {
        "settings": deepcopy(fixed), "small_rough_mix": settings,
        "case_ids": tuple(case["case_id"] for case in cases),
        "case_requests": tuple(tuple(case["base_request"]) for case in cases),
        "case_index": tuple(indices), "population_index": tuple(ROLES.index(r) for r in roles),
        "flat_lane_mask": tuple(r != "source_rough" for r in roles),
        "origins_by_env": origins,
        "lane_counts": {case["case_id"]: indices.count(i) for i, case in enumerate(cases)},
        "population_lane_counts": dict(counts), "rehearsal_lanes": len(fixed_ids),
        "actual_lane_fraction": len(fixed_ids) / num_envs,
        "rough_lanes": counts["source_rough"], "background_group": "non_rehearsal",
        **({"episode_policy_ticks": 1000} if "formal_recovery_episode_window" in config else {}),
    }


def validate_resume(old, new):
    """Change this course's population and payments, retaining physical inputs."""
    from .training_profile import validate_profile_selection, formal_rough_stage_transition
    from .flat_episode_selection import validate_scut_flat_recovery
    settings = selection(new)
    validate_profile_selection(old)
    validate_profile_selection(new)
    if not validate_scut_flat_recovery(new):
        raise ValueError("Mixed course requires retained scripted recovery")
    changed = {key for key in old.keys() | new.keys()
               if (key in old) != (key in new) or old.get(key) != new.get(key)}
    if KEY in old and old[KEY]["version"] != settings["version"]:
        raise ValueError("Changed motion objectives require the authenticated foundation parent")
    material_changed = "wheel_material_priors" in changed
    before_material, after_material = old.get("wheel_material_priors"), new.get("wheel_material_priors")
    low_grip_extension = (before_material == {
        "enabled": True, "version": "v7_low_grip_wheel_buckets_v2"}
        and after_material == {"enabled": True, "version": "wheel_low_grip_01_v3"})
    baseline_entry = (KEY not in old and settings["version"] == FINAL_MOTION_VERSION
        and before_material == {"enabled": True, "version": "v7_low_grip_wheel_buckets_v2"}
        and after_material == {"enabled": True, "version": "wheel_shape_buckets_v1"})
    if material_changed and not (low_grip_extension or baseline_entry):
        raise ValueError("Motion materials only admit explicit0.2-to-0.1 expansion or baseline final-course entry")
    # The selected payment owner returns before the legacy quiet-cost branch.
    removed_inactive = {key for key in INACTIVE_QUIET_REWARD_KEYS if key in old and key not in new}
    if changed & INACTIVE_QUIET_REWARD_KEYS and (
            settings["version"] not in MOTION_VERSIONS
            or changed & INACTIVE_QUIET_REWARD_KEYS != removed_inactive):
        raise ValueError("Inactive quiet settings may only be removed from the motion course")
    repair_key = "formal_tracking_reward_repair"
    repair_added = repair_key in changed
    if repair_added:
        from .tracking_reward_repair import validate_forward_migration
        validate_forward_migration(old.get(repair_key), new.get(repair_key))
    continuing = KEY in old
    if continuing:
        selection(old)
        allowed = {"formal_physics_trace_directory", "formal_physics_trace_substeps", "formal_stage_policy"}
        if ROUGH_BIRTH_KEY in changed:
            if ROUGH_BIRTH_KEY in old or new.get(ROUGH_BIRTH_KEY) != ROUGH_BIRTH_VERSION:
                raise ValueError("Continuous rough births require an explicit forward migration")
            allowed.add(ROUGH_BIRTH_KEY)
        if ROUGH_ANCHOR_KEY in changed:
            if ROUGH_ANCHOR_KEY in old or new.get(ROUGH_ANCHOR_KEY) != ROUGH_ANCHOR_VERSION:
                raise ValueError("Centered rough births require an explicit forward migration")
            allowed.add(ROUGH_ANCHOR_KEY)
        if "formal_critic_adaptation" in changed:
            from .critic_adaptation import validate_extension
            validate_extension(old, new)
            allowed.add("formal_critic_adaptation")
        if CONTACT_KEY in changed:
            if (old[CONTACT_KEY], new[CONTACT_KEY]) not in (
                    (BASE_CONTACT_VERSION, UPPER_LINK_VERSION),
                    (UPPER_LINK_VERSION, LEG_LINK_VERSION),
                    (LEG_LINK_VERSION, ROUGH_WHEEL_VERSION)):
                raise ValueError("Mixed contact repair requires the next explicit contact revision")
            allowed.add(CONTACT_KEY)
    else:
        if (old.get("formal_scripted_recovery_mix") != "v7-script-plus-dynamic-entry-v1"
                or not formal_rough_stage_transition(new, old.get("curriculum_stage"), "terrain")
                or new.get("formal_stage_optimizer") != "scut_finetune_v1"
                or new.get("stages") != [{**s, "name": "terrain", "terrain": ["rough"]}
                                         for s in old.get("stages", [])]):
            raise ValueError("Mixed terrain entry requires the completed direct-recovery foundation and explicit stage clocks")
        allowed = {KEY, "formal_flat_refresh", "formal_scripted_recovery_mix", "reward_profile",
            "scut_reward_mapping", "critic_height_reference", "curriculum_stage", "enabled_stages",
            "stages", "scene_groups", "formal_stage_transition", "formal_stage_optimizer",
            "formal_stage_policy", "formal_terrain", "formal_rough_initial_level", "evaluation",
            "formal_flat_rehearsal", "formal_physics_trace_directory", "formal_contact_generation_distance",
            "formal_physics_trace_substeps",
            "research_training_authorized"}
        if settings["version"] in MOTION_VERSIONS:
            allowed.update(("actor_sensor_noise", "observation_noise_enabled", "signal_perturbations",
                            "episode_spring_force_randomization", "policy_signal_delay", "formal_critic_adaptation",
                            "formal_recovery_episode_window", ROUGH_BIRTH_KEY, ROUGH_ANCHOR_KEY))
            if settings["version"] == FINAL_MOTION_VERSION:
                allowed.add("physx_friction_offset_threshold_m")
            if "formal_critic_adaptation" in new:
                from .critic_adaptation import validate_pair
                validate_pair(old, new)
            legacy = old.get("signal_perturbations", {})
            if (legacy.get("enabled") is not True or legacy.get("enabled_fraction") != 1.
                    or old.get("pc_relative_priors", {}).get("enabled") is not True
                    or old.get("contact_domain", {}).get("enabled") is not False):
                raise ValueError("Motion course noise replacement requires the existing full-population strength prior")
        before, after = dict(old["scut_reward_mapping"]), dict(new["scut_reward_mapping"])
        before.pop("root_height_reference")
        after.pop("root_height_reference")
        if before != after:
            raise ValueError("Mixed course cannot change the body/contact reward mapping")
        previous = {s["name"]: s["recovery_pose"] for s in old["scene_groups"] if "recovery_pose" in s}
        current = {s["name"]: s["recovery_pose"] for s in new["scene_groups"] if "recovery_pose" in s}
        if previous != current:
            raise ValueError("Mixed course must retain every existing recovery profile")
    if repair_added:
        allowed.add(repair_key)
    if material_changed:
        allowed.add("wheel_material_priors")
    allowed.update(removed_inactive)
    if changed - allowed:
        raise ValueError(f"Mixed course changes unrelated physical or learning semantics: {sorted(changed - allowed)}")
    return {"kind": settings["version"], "stage_transition": not continuing,
            "changed_contract_fields": sorted(changed),
            "source_birth_population_changed": not continuing or ROUGH_BIRTH_KEY in changed,
            "rough_birth_anchor_changed": ROUGH_ANCHOR_KEY in changed,
            "wheel_material_domain_changed": material_changed,
            "removed_inactive_reward_fields": sorted(removed_inactive),
            "optimizer": ("inherited" if continuing else "fresh_Adam_actor_std_preserved_critic_fresh"
                if "formal_critic_adaptation" in new else "fresh_Adam_actor_critic_std_preserved"),
            "new_native_episodes": True, "budget_counter_reset": False}


def install(spec, commands, recovery_mask):
    """Install disjoint requests before the first reset, after native reordering."""
    import torch
    from .rough_commands import ScutRoughCommands
    if not isinstance(commands, ScutRoughCommands):
        raise ValueError("Mixed course requires the small rough command owner")
    roles = torch.tensor(spec["population_index"], device=commands.device)
    if (not isinstance(recovery_mask, torch.Tensor) or recovery_mask.dtype != torch.bool
            or recovery_mask.device != commands.device
            or not torch.equal(recovery_mask, roles == ROLES.index("recovery"))):
        raise ValueError("Mixed recovery ownership differs from actual scripted lanes")
    fixed_ids = (roles == ROLES.index("fixed_flat")).nonzero(as_tuple=False).flatten()
    requests = torch.tensor([spec["case_requests"][spec["case_index"][i]]
                             for i in fixed_ids.cpu().tolist()], device=commands.device)
    command3 = torch.zeros_like(requests)
    command3[:, 0] = requests[:, 0]
    command3[:, 2] = requests[:, 1]
    commands.set_training_rehearsal_requests(fixed_ids, command3, requests[:, 2])
    value = spec["small_rough_mix"]
    if value["version"] == FINAL_MOTION_VERSION:
        commands.configure_final_motion_modes()
    source = (roles == ROLES.index("dynamic_flat")) | recovery_mask
    commands.set_source_flat_rehearsal(source.nonzero(as_tuple=False).flatten(),
                                      source_iteration_offset=value["source_flat_iteration_offset"])
    ids = (roles == ROLES.index("temporal_flat")).nonzero(as_tuple=False).flatten()
    temporal = value["temporal"]
    if value["version"] in MOTION_VERSIONS:
        commands.set_motion_training_requests(ids, temporal,
            episode_policy_ticks=spec.get("episode_policy_ticks", 999))
    else:
        speeds = [sign * speed for speed in temporal["speeds_m_s"] for sign in (1., -1.)]
        requests = torch.tensor([[speeds[i % len(speeds)], 0., temporal["height_m"]]
                                 for i in range(len(ids))], dtype=torch.float32, device=commands.device)
        commands.set_temporal_training_requests(ids, requests,
            phase_policy_ticks=temporal["phase_policy_ticks"], episode_policy_ticks=999)
    readback = {"source_flat_iteration_offset": value["source_flat_iteration_offset"],
                "population_lane_counts": spec["population_lane_counts"],
                "clock": "explicit_parent_global_offset_plus_local_rough_actor_updates",
                "birth_or_skill_qualification": False}
    if value["version"] in MOTION_VERSIONS:
        readback["motion_course"] = commands._temporal_rehearsal.summary()
    return readback


class MixedCourseExposure:
    """Count the final PPO loss population only after successful optimization.

    Rows use pre-reset command phases. Returned observations may already belong
    to the next episode and must not relabel the transition that PPO received.
    """

    def __init__(self, spec, steps, device, *, rough_grid=None):
        import torch
        self.roles = torch.tensor(spec["population_index"], dtype=torch.long, device=device)
        self.version = spec["small_rough_mix"]["version"]
        self.motion = self.version in MOTION_VERSIONS
        if self.motion:
            from .motion_transition_course import PHASES
            self.phase_names = PHASES
            self.speed_edges = tuple(spec["small_rough_mix"]["temporal"]["speed_edges_m_s"])
            self.axis_coverage = spec["small_rough_mix"]["temporal"].get("axis_coverage")
            self.motion_yaw_tolerance_rad_s = spec["small_rough_mix"]["temporal"]["yaw_tolerance_rad_s"]
        else:
            self.phase_names = ("drive", "stop", "reverse", "final_stop")
        self.steps = steps
        self.rows = []
        self.counts = dict.fromkeys((*ROLES, *self.phase_names), 0)
        self.updates = 0
        self.temporal_ids = (self.roles == ROLES.index("temporal_flat")).nonzero(as_tuple=False).flatten()
        self.required_frames = (None if self.motion else torch.tensor(
            spec["small_rough_mix"]["temporal"]["phase_policy_ticks"], dtype=torch.long, device=device))
        self.episode_ids = torch.full_like(self.temporal_ids, -1)
        self.episode_frames = torch.zeros((len(self.temporal_ids), len(self.phase_names)), dtype=torch.long, device=device)
        self.episode_clean = torch.zeros(len(self.temporal_ids), dtype=torch.bool, device=device)
        self.complete_loss_sequences = 0
        self.motion_counts = {}
        self.episode_policy_ticks = spec.get("episode_policy_ticks", 999)
        self.rough = None
        if rough_grid is not None:
            from .rough_course_exposure import RoughCourseExposure
            self.rough = RoughCourseExposure(
                self.roles == ROLES.index("source_rough"), rough_grid, steps)

    def observe(self, index, info):
        import torch
        if index == 0:
            self.rows = []
        if index != len(self.rows) or not 0 <= index < self.steps:
            raise ValueError("Mixed exposure requires consecutive actual rollout rows")
        diagnostics = info["diagnostics"]
        owner = diagnostics["source_flat_episode"]
        phase = owner["temporal_published_phase"]
        if (phase.dtype != torch.long or phase.shape != self.roles.shape
                or phase.device != self.roles.device):
            raise ValueError("Mixed exposure requires the pre-reset temporal phase vector")
        row = {"phase": phase.detach().clone()}
        if self.motion:
            row["motion"] = {key: value.detach().clone() for key, value in owner["motion_course"].items()}
            row["episode_tick"] = owner["episode_ticks"][self.temporal_ids].clone()
        for name, value, dtype in (
                ("episode_id", owner["episode_id"], torch.long),
                ("done", diagnostics["done"], torch.bool),
                ("timeout", diagnostics["timeout"], torch.bool),
                ("boundary", diagnostics["boundary"], torch.bool),
                ("terminated", diagnostics["terminated"], torch.bool)):
            if value.dtype != dtype or value.shape != self.roles.shape or value.device != self.roles.device:
                raise ValueError("Mixed episode exposure requires pre-reset episode and terminal vectors")
            row[name] = value[self.temporal_ids].detach().clone()
        self.rows.append(row)
        if self.rough is not None:
            self.rough.observe(index, diagnostics)

    def prepare(self, loss_mask):
        import torch
        if (len(self.rows) != self.steps or loss_mask.dtype != torch.bool
                or loss_mask.shape != (self.steps, self.roles.numel())
                or loss_mask.device != self.roles.device):
            raise ValueError("Mixed exposure requires the complete final PPO loss mask")
        phases = torch.stack([row["phase"] for row in self.rows])
        temporal = self.roles == ROLES.index("temporal_flat")
        if (bool((phases[:, ~temporal] != -1).any())
                or bool(((phases[:, temporal] < 0) | (phases[:, temporal] >= len(self.phase_names))).any())):
            raise ValueError("Temporal phases differ from static population ownership")
        values = [loss_mask[:, self.roles == i].sum() for i in range(len(ROLES))]
        values += [(loss_mask & (phases == i)).sum() for i in range(len(self.phase_names))]
        episode_ids = self.episode_ids.clone()
        frames = self.episode_frames.clone()
        clean = self.episode_clean.clone()
        completed = torch.zeros((), dtype=torch.long, device=self.roles.device)
        for index, row in enumerate(self.rows):
            new = row["episode_id"] != episode_ids
            frames[new] = 0
            episode_ids = row["episode_id"]
            phase = phases[index, self.temporal_ids]
            frames.scatter_add_(1, phase[:, None], loss_mask[index, self.temporal_ids, None].long())
            horizon = row["done"] & row["timeout"] & ~row["boundary"] & ~row["terminated"]
            if self.motion:
                clean = torch.where(new, row["episode_tick"] == 1, clean)
                clean &= loss_mask[index, self.temporal_ids] & row["motion"]["valid"][self.temporal_ids]
                completed += (horizon & (row["episode_tick"] >= self.episode_policy_ticks)
                              & clean & row["motion"]["qualified_sequence"][self.temporal_ids]).sum()
            else:
                completed += ((frames == self.required_frames).all(-1) & horizon).sum()
        return {"counts": torch.stack(values).detach(), "completed": completed,
                "episode_ids": episode_ids.clone(), "episode_frames": frames, "episode_clean": clean,
                **({"rough": self.rough.prepare(loss_mask)} if self.rough is not None else {}),
                "motion_counts": {name: (loss_mask & members).sum().detach()
                                  for name, members in self._motion_groups().items()}}

    def _motion_groups(self):
        """Measured entries and continuous bins; no request-only success counts."""
        import torch
        if not self.motion:
            return {}
        data = {name: torch.stack([row["motion"][name] for row in self.rows])
                for name in self.rows[0]["motion"]}
        owned = self.roles[None, :] == ROLES.index("temporal_flat")
        valid = owned & data["valid"]
        from .motion_transition_course import LESSONS
        groups = {"motion/" + name: valid & (data["lesson"] == i)
                  for i, name in enumerate(LESSONS)}
        groups.update({"motion/tracking_frames": valid & data["tracking"],
                       "motion/hold_broken_frames": valid & data["hold_broken"],
                       "motion/invalid_frames": owned & ~data["valid"]})
        axes = [("motion", valid, data["target_vx"], data["entry_target"], data["entry_vx"],
                 self.speed_edges, (("forward", 1), ("backward", -1)), self.speed_edges[0])]
        if self.axis_coverage is not None:
            linear, rotation = valid & (data["axis"] == 0), valid & (data["axis"] == 1)
            axes[0] = ("motion", linear, *axes[0][2:])
            axes.append(("motion/rotation", rotation, data["target_yaw"], data["entry_target_yaw"],
                         data["entry_yaw"], tuple(self.axis_coverage["yaw_edges_rad_s"]),
                         (("positive", 1), ("negative", -1)),
                         self.motion_yaw_tolerance_rad_s))
            groups.update({"motion/translation_frames": linear, "motion/rotation_frames": rotation,
                           "motion/translation_tracking_frames": linear & data["tracking"],
                           "motion/rotation_tracking_frames": rotation & data["tracking"]})
        # Each axis retains its own units. Bins label continuous requests;
        # a zero-target hold must not count in either signed rotation pool.
        for prefix, members, target, entry_target, entry_speed, edges, directions, near_zero in axes:
            changing = members & (data["phase"] == 2)
            same_direction = target * entry_target > 0
            for name, condition in (
                    ("accelerating", same_direction & (target.abs() > entry_target.abs())),
                    ("decelerating", same_direction & (target.abs() < entry_target.abs())),
                    ("direct_reversal", target * entry_target < 0)):
                groups[f"{prefix}/change/{name}"] = changing & condition
            groups[f"{prefix}/reverse_after_verified_stop"] = (
                members & (data["phase"] == 4) & data["phase_first"] & data["entry_passed"])
            for direction, sign in directions:
                request = target * sign
                for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
                    upper = request <= hi if i == len(edges) - 2 else request < hi
                    groups[f"{prefix}/request/{direction}/band{i}"] = (
                        members & (request > 0) & (request >= lo) & upper)
            for phase, name in ((3, "brake"), (2, "speed_change")):
                entry = members & data["phase_first"] & (data["phase"] == phase)
                groups[f"{prefix}/entry/{name}/requested"] = entry
                groups[f"{prefix}/entry/{name}/near_zero"] = entry & (entry_speed.abs() < near_zero)
                groups[f"{prefix}/entry/{name}/above_maximum"] = entry & (entry_speed.abs() > edges[-1])
                for direction, sign in directions:
                    groups[f"{prefix}/entry/{name}/{direction}"] = (
                        entry & (entry_speed * sign > 0) & (entry_speed * sign >= edges[0]))
                for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
                    magnitude = entry_speed.abs()
                    upper = magnitude <= hi if i == len(edges) - 2 else magnitude < hi
                    groups[f"{prefix}/entry/{name}/actual_band{i}"] = entry & (magnitude >= lo) & upper
                groups[f"{prefix}/entry/{name}/from_target"] = entry & (
                    entry_speed * entry_target.sign() >= .8 * entry_target.abs()) & (entry_target != 0.)
        for i, (lo, hi) in enumerate(((.20, .27), (.27, .35), (.35, .420001))):
            groups[f"motion/height/band{i}"] = valid & (data["height"] >= lo) & (data["height"] < hi)
        for i, name in enumerate(self.phase_names):
            groups[f"motion/pass/{name}"] = valid & data["phase_passed"] & (data["phase"] == i)
            groups[f"motion/timeout/{name}"] = owned & data["phase_timed_out"] & (data["phase"] == i)
        return groups

    def commit(self, prepared):
        import torch
        rough_counts = self.rough.commit(prepared["rough"]) if self.rough is not None else {}
        values = prepared["counts"].cpu().tolist()
        for name, value in zip(self.counts, values):
            self.counts[name] += value
        self.updates += 1
        self.complete_loss_sequences += int(prepared["completed"])
        self.episode_ids, self.episode_frames = prepared["episode_ids"], prepared["episode_frames"]
        self.episode_clean = prepared["episode_clean"]
        names = list(prepared["motion_counts"])
        motion = (dict(zip(names, torch.stack(
            [prepared["motion_counts"][name] for name in names]).cpu().tolist())) if names else {})
        for key, value in motion.items():
            self.motion_counts[key] = self.motion_counts.get(key, 0) + value
        self.rows = []
        return {**dict(zip(self.counts, values)), **motion, **rough_counts}

    def optimization_metrics(self, storage, loss_mask, fixed_indices=None, fixed_names=()):
        """Read pre-update critic/advantage statistics on actual loss samples.

        The native optimizer clears storage. Detach these reductions before
        that call and publish only after its successful return. These are
        learning diagnostics, not measurements of physical task success.
        """
        import torch
        if len(self.rows) != self.steps or loss_mask.shape != (self.steps, self.roles.numel()):
            raise ValueError("Optimization diagnostics require the complete mixed rollout")
        phases = torch.stack([row["phase"] for row in self.rows])
        groups = {"role/" + name: self.roles[None, :] == index
                  for index, name in enumerate(ROLES)}
        groups.update({"phase/" + name: phases == index
                       for index, name in enumerate(self.phase_names)})
        groups.update(self._motion_groups())
        if fixed_indices is not None:
            if fixed_indices.shape != self.roles.shape or fixed_indices.device != self.roles.device:
                raise ValueError("Fixed-case diagnostics require native lane order")
            groups.update({"fixed/" + name: fixed_indices[None, :] == index
                           for index, name in enumerate(fixed_names)})
        values = storage.values.squeeze(-1).detach()
        returns = storage.returns.squeeze(-1).detach()
        advantages = storage.advantages.squeeze(-1).detach()
        if any(value.shape != loss_mask.shape for value in (values, returns, advantages)):
            raise ValueError("Mixed diagnostics require native scalar rollout values")
        residual = returns - values
        result = {}
        for name, members in groups.items():
            selected = loss_mask & members
            count = selected.sum()
            denominator = count.clamp_min(1)
            prefix = "MixedCourse/optimization/" + name + "/"
            result[prefix + "transitions"] = count.detach()
            for key, value in (("value_mean", values), ("target_return_mean", returns),
                               ("gae_residual_mean", residual),
                               ("normalized_advantage_mean", advantages),
                               ("positive_advantage_fraction", (advantages > 0).float())):
                result[prefix + key] = (torch.where(selected, value, 0.).sum() / denominator).detach()
            result[prefix + "value_residual_rms"] = (
                torch.where(selected, residual.square(), 0.).sum() / denominator).sqrt().detach()
            target_mean = torch.where(selected, returns, 0.).sum() / denominator
            residual_mean = torch.where(selected, residual, 0.).sum() / denominator
            target_variance = torch.where(selected, (returns - target_mean).square(), 0.).sum() / denominator
            residual_variance = torch.where(selected, (residual - residual_mean).square(), 0.).sum() / denominator
            ev_valid = (count > 1) & (target_variance > 1e-10)
            result[prefix + "target_variance"] = target_variance.detach()
            result[prefix + "explained_variance_valid"] = ev_valid.detach()
            result[prefix + "explained_variance"] = torch.where(ev_valid,
                1 - residual_variance / target_variance.clamp_min(1e-10), 0.).detach()
        if self.motion and self.axis_coverage is not None:
            data = {name: torch.stack([row["motion"][name] for row in self.rows])
                    for name in ("planar_speed", "target_yaw", "measured_yaw")}
            selected = loss_mask & groups["motion/rotation_frames"] & (data["target_yaw"] != 0)
            count = selected.sum()
            prefix = "MixedCourse/command/rotation/"
            result[prefix + "nonzero_request_transitions"] = count.detach()
            result[prefix + "root_reference_planar_speed_mean_m_s"] = (
                torch.where(selected, data["planar_speed"], 0.).sum() / count.clamp_min(1)).detach()
            result[prefix + "yaw_error_mean_rad_s"] = (
                torch.where(selected, (data["measured_yaw"] - data["target_yaw"]).abs(), 0.).sum()
                / count.clamp_min(1)).detach()
        return result

    def state_dict(self):
        result = {"version": self.version, "successful_actor_updates": self.updates,
                "direct_loss_transitions": dict(self.counts),
                "complete_temporal_loss_sequences": self.complete_loss_sequences,
                "scope": "unique optimized transitions; phases partition temporal lanes; full sequences require every phase frame and horizon completion, not tracking success"}
        if self.motion:
            result.update(motion_counts=dict(self.motion_counts),
                scope="unique optimized transitions; completed motion episodes require all intervals and measured phase gates; cold resumes restart episodes")
        if self.episode_policy_ticks != 999:
            result["episode_policy_ticks"] = self.episode_policy_ticks
        if self.rough is not None:
            result["rough_exposure"] = self.rough.state_dict()
        return result

    def load_state_dict(self, state):
        counts = state.get("direct_loss_transitions", {})
        motion = state.get("motion_counts") if self.motion else None
        if self.motion and (not isinstance(motion, dict)
                            or any(type(v) is not int or v < 0 for v in motion.values())):
            raise ValueError("Motion exposure checkpoint has invalid counts")
        if (state.get("version") != self.version or set(counts) != set(self.counts)
                or state.get("episode_policy_ticks", 999) != self.episode_policy_ticks
                or type(state.get("successful_actor_updates")) is not int
                or state["successful_actor_updates"] < 0
                or type(state.get("complete_temporal_loss_sequences")) is not int
                or state["complete_temporal_loss_sequences"] < 0
                or any(type(v) is not int or v < 0 for v in counts.values())
                or sum(counts[k] for k in self.phase_names)
                   != counts["temporal_flat"]):
            raise ValueError("Mixed exposure checkpoint has inconsistent counts")
        self.counts = {key: counts[key] for key in self.counts}
        self.updates = state["successful_actor_updates"]
        self.complete_loss_sequences = state["complete_temporal_loss_sequences"]
        if self.motion:
            self.motion_counts = dict(motion)
        if self.rough is not None:
            saved = state.get("rough_exposure")
            if saved is None:
                self.rough.historical_unclassified_actor_transitions = counts["source_rough"]
            else:
                self.rough.load_state_dict(saved)
                if (int(self.rough.totals[0]) + self.rough.historical_unclassified_actor_transitions
                        != counts["source_rough"]):
                    raise ValueError("Rough grid exposure differs from actual role loss totals")
