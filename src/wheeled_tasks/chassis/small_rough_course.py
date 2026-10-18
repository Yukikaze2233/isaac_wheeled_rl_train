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
    LEG_LINK_VERSION, UPPER_LINK_VERSION, VERSION as BASE_CONTACT_VERSION,
)

KEY = "formal_small_rough_mix"
VERSION = "v7-small-rough-flat-recovery-v1"
ROLES = ("fixed_flat", "dynamic_flat", "temporal_flat", "recovery", "source_rough")


def selection(config):
    """Validate the local adaptation; caller supplies fractions and dwell times."""
    if KEY not in config:
        return None
    value = config[KEY]
    if (not isinstance(value, Mapping)
            or set(value) != {"version", "fractions", "temporal", "source_flat_iteration_offset"}
            or value["version"] != VERSION
            or config.get("reward_profile") != "scut0730_rough_v1"
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
            "allocation": "equal_case_lanes", "reward_profile": "scut0719_flat_v1"}
            or config.get("formal_dynamic_nonsteady_birth") != DYNAMIC_NONSTEADY_BIRTH_VERSION):
        raise ValueError("Mixed course must retain fixed17 and explicit dynamic flat births")
    offset = value["source_flat_iteration_offset"]
    temporal = value["temporal"]
    if (type(offset) is not int or offset < 4000
            or not isinstance(temporal, Mapping)
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
    if len(fixed_ids) < len(cases) or len(temporal_ids) % (2 * len(settings["temporal"]["speeds_m_s"])):
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
    }


def validate_resume(old, new):
    """Change this course's population and payments, retaining physical inputs."""
    from .training_profile import validate_profile_selection, formal_rough_stage_transition
    from .flat_episode_selection import validate_scut_flat_recovery
    selection(new)
    validate_profile_selection(old)
    validate_profile_selection(new)
    if not validate_scut_flat_recovery(new):
        raise ValueError("Mixed course requires retained scripted recovery")
    changed = {key for key in old.keys() | new.keys()
               if (key in old) != (key in new) or old.get(key) != new.get(key)}
    repair_key = "formal_tracking_reward_repair"
    repair_added = repair_key in changed
    if repair_added:
        from .tracking_reward_repair import VERSION as repair_version
        if repair_key in old or new.get(repair_key) != repair_version:
            raise ValueError("Mixed tracking repair must be an explicit forward migration")
    continuing = KEY in old
    if continuing:
        selection(old)
        allowed = {"formal_physics_trace_directory", "formal_stage_policy"}
        if CONTACT_KEY in changed:
            if (old[CONTACT_KEY], new[CONTACT_KEY]) not in (
                    (BASE_CONTACT_VERSION, UPPER_LINK_VERSION),
                    (UPPER_LINK_VERSION, LEG_LINK_VERSION)):
                raise ValueError("Mixed contact repair requires the next explicit paired-link revision")
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
            "research_training_authorized"}
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
    if changed - allowed:
        raise ValueError(f"Mixed course changes unrelated physical or learning semantics: {sorted(changed - allowed)}")
    return {"kind": VERSION, "stage_transition": not continuing,
            "changed_contract_fields": sorted(changed),
            "source_birth_population_changed": not continuing,
            "optimizer": "inherited" if continuing else "fresh_Adam_actor_critic_std_preserved",
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
    source = (roles == ROLES.index("dynamic_flat")) | recovery_mask
    commands.set_source_flat_rehearsal(source.nonzero(as_tuple=False).flatten(),
                                      source_iteration_offset=value["source_flat_iteration_offset"])
    ids = (roles == ROLES.index("temporal_flat")).nonzero(as_tuple=False).flatten()
    temporal = value["temporal"]
    speeds = [sign * speed for speed in temporal["speeds_m_s"] for sign in (1., -1.)]
    requests = torch.tensor([[speeds[i % len(speeds)], 0., temporal["height_m"]]
                             for i in range(len(ids))], dtype=torch.float32, device=commands.device)
    commands.set_temporal_training_requests(ids, requests,
        phase_policy_ticks=temporal["phase_policy_ticks"], episode_policy_ticks=999)
    return {"source_flat_iteration_offset": value["source_flat_iteration_offset"],
            "population_lane_counts": spec["population_lane_counts"],
            "clock": "explicit_parent_global_offset_plus_local_rough_actor_updates",
            "birth_or_skill_qualification": False}


class MixedCourseExposure:
    """Count the final PPO loss population only after successful optimization.

    Rows use pre-reset command phases. Returned observations may already belong
    to the next episode and must not relabel the transition that PPO received.
    """

    def __init__(self, spec, steps, device):
        import torch
        self.roles = torch.tensor(spec["population_index"], dtype=torch.long, device=device)
        self.steps = steps
        self.rows = []
        self.counts = dict.fromkeys((*ROLES, "drive", "stop", "reverse", "final_stop"), 0)
        self.updates = 0
        self.temporal_ids = (self.roles == ROLES.index("temporal_flat")).nonzero(as_tuple=False).flatten()
        self.required_frames = torch.tensor(spec["small_rough_mix"]["temporal"]["phase_policy_ticks"],
                                           dtype=torch.long, device=device)
        self.episode_ids = torch.full_like(self.temporal_ids, -1)
        self.episode_frames = torch.zeros((len(self.temporal_ids), 4), dtype=torch.long, device=device)
        self.complete_loss_sequences = 0

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

    def prepare(self, loss_mask):
        import torch
        if (len(self.rows) != self.steps or loss_mask.dtype != torch.bool
                or loss_mask.shape != (self.steps, self.roles.numel())
                or loss_mask.device != self.roles.device):
            raise ValueError("Mixed exposure requires the complete final PPO loss mask")
        phases = torch.stack([row["phase"] for row in self.rows])
        temporal = self.roles == ROLES.index("temporal_flat")
        if (bool((phases[:, ~temporal] != -1).any())
                or bool(((phases[:, temporal] < 0) | (phases[:, temporal] > 3)).any())):
            raise ValueError("Temporal phases differ from static population ownership")
        values = [loss_mask[:, self.roles == i].sum() for i in range(len(ROLES))]
        values += [(loss_mask & (phases == i)).sum() for i in range(4)]
        episode_ids = self.episode_ids.clone()
        frames = self.episode_frames.clone()
        completed = torch.zeros((), dtype=torch.long, device=self.roles.device)
        for index, row in enumerate(self.rows):
            frames[row["episode_id"] != episode_ids] = 0
            episode_ids = row["episode_id"]
            phase = phases[index, self.temporal_ids]
            frames.scatter_add_(1, phase[:, None], loss_mask[index, self.temporal_ids, None].long())
            horizon = row["done"] & row["timeout"] & ~row["boundary"] & ~row["terminated"]
            completed += ((frames == self.required_frames).all(-1) & horizon).sum()
        return {"counts": torch.stack(values).detach(), "completed": completed,
                "episode_ids": episode_ids.clone(), "episode_frames": frames}

    def commit(self, prepared):
        values = prepared["counts"].cpu().tolist()
        for name, value in zip(self.counts, values):
            self.counts[name] += value
        self.updates += 1
        self.complete_loss_sequences += int(prepared["completed"])
        self.episode_ids, self.episode_frames = prepared["episode_ids"], prepared["episode_frames"]
        self.rows = []
        return dict(zip(self.counts, values))

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
                       for index, name in enumerate(("drive", "stop", "reverse", "final_stop"))})
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
        return result

    def state_dict(self):
        return {"version": VERSION, "successful_actor_updates": self.updates,
                "direct_loss_transitions": dict(self.counts),
                "complete_temporal_loss_sequences": self.complete_loss_sequences,
                "scope": "unique optimized transitions; phases partition temporal lanes; full sequences require every phase frame and horizon completion, not tracking success"}

    def load_state_dict(self, state):
        counts = state.get("direct_loss_transitions", {})
        if (state.get("version") != VERSION or set(counts) != set(self.counts)
                or type(state.get("successful_actor_updates")) is not int
                or state["successful_actor_updates"] < 0
                or type(state.get("complete_temporal_loss_sequences")) is not int
                or state["complete_temporal_loss_sequences"] < 0
                or any(type(v) is not int or v < 0 for v in counts.values())
                or sum(counts[k] for k in ("drive", "stop", "reverse", "final_stop"))
                   != counts["temporal_flat"]):
            raise ValueError("Mixed exposure checkpoint has inconsistent counts")
        self.counts = {key: counts[key] for key in self.counts}
        self.updates = state["successful_actor_updates"]
        self.complete_loss_sequences = state["complete_temporal_loss_sequences"]
