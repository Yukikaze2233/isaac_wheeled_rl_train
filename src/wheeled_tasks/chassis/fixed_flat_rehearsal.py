"""Explicit flat rehearsal inside V7 rough or scripted-plane recovery.

The fraction is a caller-supplied design choice, not an upstream reference value.
Allocation is deterministic and consumes no simulator/command RNG. Actual PPO
exposure is recorded separately after successful optimization.
"""
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path


VERSION = "v7_fixed15_plane_rehearsal_v1"
SOURCE_VERSION = "v7_source0719_plane_rehearsal_v2"
CASE_FILE = "configs/v7_formal_20261012/evaluation/foundation_fixed_mean_v1.json"
CASE_SHA = "e6e25cdc3fb508c5b1135dd0407dd4e5ddb67541a4bbdbb9d99968d0e670380c"
FIXED17_VERSION = "v7_fixed17_plane_rehearsal_v2"
FIXED17_CASE_FILE = "configs/v7_formal_20261012/evaluation/foundation_rehearsal17_v1.json"
FIXED17_CASE_SHA = "5935316741326e25289a6ff5c5bef5eaf61b15e9263a36abc55f8d1c9a0680be"
FLAT_REFRESH_VERSION = "v7_flat_robustness_refresh_v1"
FLAT_MIXED_REFRESH_VERSION = "v7_flat_mixed_robustness_refresh_v1"
DYNAMIC_NONSTEADY_BIRTH_VERSION = "v7-dynamic-source-birth-v1"
SCRIPTED_RECOVERY_MIX_VERSION = "v7-script-plus-dynamic-entry-v1"
_SCRIPTED_MIX_ASSET_SHA = "088e699266fa3dc666379a911b23a30e2038ab3d7f113cab4dcd4bdd0a14279b"
_SCRIPTED_MIX_GEOMETRY = {
    "profiles_file": "reports/v7_formal_training_20261012/recovery107_candidate/recovery_profiles_107_v2.json",
    "profiles_sha256": "b966a5f65246775bd185d27871c3f42783ec25ee73e8df2d6ee3efbc769b9400",
    "height_lookup_file": "reports/v7_formal_training_20261012/recovery107_candidate/height_lookup_107.json",
    "height_lookup_sha256": "91c43e8ea7da828155537ee3e64c3f45ccde7a475d002065da4829205027dfa1",
}
_SCRIPTED_MIX_PROFILES = frozenset((
    "upright", "crouched", "upright_release", "crouched_release", "left_side_90", "right_side_90",
    "pitch_forward_45", "pitch_backward_45", "front_down_90", "back_down_90",
    "front_down_left_forward_right_back_90", "front_down_left_back_right_forward_90",
    "back_down_left_forward_right_back_90", "back_down_left_back_right_forward_90",
    "prone_belly_legs_above", "prone_back_legs_above",
))


def validate_scripted_flat_mix(config, root=None):
    """Select frozen direct recovery beside dynamic births; grant no launch authority."""
    field = "formal_scripted_recovery_mix"
    if field not in config:
        return None
    if (type(config[field]) is not str or config[field] != SCRIPTED_RECOVERY_MIX_VERSION
            or not validate_dynamic_nonsteady_birth(config)
            or config.get("formal_learning_state_resume") is not True
            or config.get("asset_manifest_sha256") != _SCRIPTED_MIX_ASSET_SHA
            or config.get("v5_control", {}).get("leg_winding_version") != "paired_branch107_v1"
            or config.get("reward_profile") != "scut0719_flat_v1"
            or config.get("formal_terrain") != {
                "type": "plane", "source": "scut0719", "env_spacing_m": 4.}
            or config.get("enabled_stages") != ["foundation"]
            or (config.get("physics_dt"), config.get("policy_dt")) != (.005, .02)
            or (config.get("position_iterations"), config.get("velocity_iterations")) != (12, 6)
            or any(key in config for key in ("formal_stage_transition", "formal_stage_optimizer"))
            or config.get("critic_warmup_updates", 0) != 0
            or config.get("resume_critic_warmup_updates", 0) != 0):
        raise ValueError("Scripted flat mix requires the same branch107 dynamic foundation, clocks and flat reward")
    settings = config.get("recovery_training", {})
    if (not isinstance(settings, Mapping)
            or any(settings.get(key) != value for key, value in _SCRIPTED_MIX_GEOMETRY.items())):
        raise ValueError("Scripted flat mix requires the frozen107 recovery profiles and height lookup")
    from .flat_episode_selection import validate_scut_flat_recovery
    from .recovery_training import validate_direct_takeover
    if not validate_direct_takeover(settings):
        raise ValueError("Scripted flat mix requires the existing direct recovery selector")
    receipt = validate_scut_flat_recovery(config, root)
    if (receipt is None
            or not set(receipt["profiles_by_scene"].values()) <= _SCRIPTED_MIX_PROFILES):
        raise ValueError("Scripted flat mix requires the frozen flat recovery profile names")
    script_fraction = math.fsum(scene["fraction"] for scene in config["scene_groups"]
                                if "recovery_pose" in scene)
    ordinary_fraction = math.fsum(scene["fraction"] for scene in config["scene_groups"]
                                  if "recovery_pose" not in scene)
    # The fixed quarter is part of the ordinary cohort; retain dynamic births.
    if not 0. < script_fraction < .75 or ordinary_fraction <= .25:
        raise ValueError("Scripted flat mix must retain the fixed quarter and ordinary dynamic lanes")
    return {**receipt, "version": SCRIPTED_RECOVERY_MIX_VERSION,
            "script_fraction": script_fraction, "ordinary_fraction": ordinary_fraction,
            "dynamic_source_fraction": ordinary_fraction - .25,
            "launch_authorization": "not_granted_by_selector"}


def validate_dynamic_nonsteady_birth(config):
    """Select source births only for the existing dynamic background lanes."""
    if "formal_small_rough_mix" in config:
        from .small_rough_course import selection
        return selection(config) is not None
    field = "formal_dynamic_nonsteady_birth"
    if field not in config:
        if (config.get("formal_flat_refresh") == FLAT_MIXED_REFRESH_VERSION
                and config.get("formal_foundation_nominal_birth") is False):
            raise ValueError("Mixed flat nonsteady births require an explicit dynamic birth selector")
        return False
    expected_rehearsal = {
        "version": FIXED17_VERSION, "fraction": .25,
        "cases_file": FIXED17_CASE_FILE, "cases_sha256": FIXED17_CASE_SHA,
        "allocation": "equal_case_lanes", "reward_profile": "scut0719_flat_v1"}
    if (type(config[field]) is not str or config[field] != DYNAMIC_NONSTEADY_BIRTH_VERSION
            or config.get("formal_v7_flat") is not True
            or config.get("curriculum_stage") != "foundation"
            or config.get("formal_flat_refresh") != FLAT_MIXED_REFRESH_VERSION
            or config.get("formal_foundation_nominal_birth") is not False
            or config.get("formal_flat_rehearsal") != expected_rehearsal):
        raise ValueError("V7 dynamic birth requires the same mixed foundation with fixed17 25% nominal rehearsal")
    return True


def validate_fixed_flat_rehearsal(config):
    if "formal_small_rough_mix" in config:
        from .small_rough_course import selection
        selection(config)
        return dict(config["formal_flat_rehearsal"])
    scripted_mix = validate_scripted_flat_mix(config)
    validate_dynamic_nonsteady_birth(config)
    from .scut_reward_selection import V7_FINAL_ROUGH_PROFILE
    final_rough = config.get("reward_profile") == V7_FINAL_ROUGH_PROFILE
    flat_mixed_refresh = config.get("formal_flat_refresh") == FLAT_MIXED_REFRESH_VERSION
    flat_refresh = config.get("formal_flat_refresh") in (FLAT_REFRESH_VERSION, FLAT_MIXED_REFRESH_VERSION)
    if "formal_flat_refresh" in config:
        recovery = config.get("recovery_training", {})
        scenes = config.get("scene_groups", [])
        if (not flat_refresh
                or config.get("reward_profile") != "scut0719_flat_v1"
                or config.get("formal_terrain") != {
                    "type": "plane", "source": "scut0719", "env_spacing_m": 4.}
                or config.get("curriculum_stage") != "foundation"
                or config.get("enabled_stages") != ["foundation"]
                or not isinstance(recovery, Mapping)
                or (scripted_mix is None and (recovery.get("enabled", False)
                    or "source_flat_handoff" in recovery
                    or any(isinstance(scene, Mapping) and "recovery_pose" in scene for scene in scenes)))
                or not isinstance(config.get("formal_flat_rehearsal"), Mapping)
                or config["formal_flat_rehearsal"].get("version") != FIXED17_VERSION):
            raise ValueError("V7 fixed17 flat refresh requires ordinary foundation training on the existing plane")
    if "formal_flat_rehearsal" not in config:
        return None
    value = config["formal_flat_rehearsal"]
    if isinstance(value, Mapping) and value.get("version") == FIXED17_VERSION:
        if (set(value) != {"version", "fraction", "cases_file", "cases_sha256", "allocation", "reward_profile"}
                or value["cases_file"] != FIXED17_CASE_FILE or value["cases_sha256"] != FIXED17_CASE_SHA
                or value["allocation"] != "equal_case_lanes"
                or type(value["fraction"]) not in (int, float)
                or value["fraction"] != (1. if flat_refresh and not flat_mixed_refresh else .25)
                or value["reward_profile"] != (V7_FINAL_ROUGH_PROFILE if final_rough else
                                               "scut0719_flat_v1" if flat_refresh else "scut0730_rough_v1")
                or config.get("reward_profile") != value["reward_profile"]):
            raise ValueError("V7 fixed17 requires pinned cases and the versioned flat or rough rehearsal fraction")
        return dict(value)
    if isinstance(value, Mapping) and value.get("version") == SOURCE_VERSION:
        if (set(value) != {"version", "fraction", "cases_file", "cases_sha256", "allocation",
                          "source_iteration_offset", "reward_profile"}
                or value["cases_file"] != CASE_FILE or value["cases_sha256"] != CASE_SHA
                or value["allocation"] != "source_command_distribution"
                or (value["reward_profile"] != V7_FINAL_ROUGH_PROFILE if final_rough else
                    value["reward_profile"] not in ("scut0719_flat_v1", "scut0730_rough_v1"))
                or (not final_rough and config.get("reward_profile") != "scut0730_rough_v1")
                or type(value["source_iteration_offset"]) is not int
                or value["source_iteration_offset"] < 4000
                or type(value["fraction"]) not in (int, float)
                or not math.isfinite(value["fraction"]) or not 0. < value["fraction"] < 1.):
            raise ValueError("Source flat rehearsal requires completed flat special coverage and explicit 0719 or 0730 rewards")
        return dict(value)
    if (not isinstance(value, Mapping)
            or set(value) != {"version", "fraction", "cases_file", "cases_sha256", "allocation"}
            or value["version"] != VERSION
            or value["cases_file"] != CASE_FILE or value["cases_sha256"] != CASE_SHA
            or value["allocation"] != "equal_case_lanes"
            or type(value["fraction"]) not in (int, float)
            or not math.isfinite(value["fraction"]) or not 0. < value["fraction"] < 1.):
        raise ValueError("Fixed flat rehearsal requires the pinned 15 working points and an explicit fraction")
    return dict(value)


def load_fixed_flat_rehearsal(config, root, num_envs, *, ordinary_lane_ids=None,
                            scripted_plane_lane_ids=None):
    """Allocate fixed requests and optional explicit flat script origins.

    Script lane IDs come from the caller's actual scene allocation. This only
    composes geometry; source commands, births and stage admission stay separate.
    """
    settings = validate_fixed_flat_rehearsal(config)
    if settings is None:
        if scripted_plane_lane_ids is not None:
            raise ValueError("Scripted plane origins require explicit fixed rehearsal allocation")
        return None
    scripted_mix = validate_scripted_flat_mix(config, root)
    from .scut_reward_selection import V7_FINAL_ROUGH_PROFILE
    final_rough = config.get("reward_profile") == V7_FINAL_ROUGH_PROFILE
    source_sampling = settings["version"] == SOURCE_VERSION
    flat_mixed_refresh = config.get("formal_flat_refresh") == FLAT_MIXED_REFRESH_VERSION
    flat_refresh = config.get("formal_flat_refresh") in (FLAT_REFRESH_VERSION, FLAT_MIXED_REFRESH_VERSION)
    case_count = 17 if settings["version"] == FIXED17_VERSION else 15
    if type(num_envs) is not int or num_envs < (2 if source_sampling else case_count + (not flat_refresh)):
        raise ValueError(f"Rehearsal needs at least one lane for each of {case_count} cases and one background lane")
    raw = (Path(root) / settings["cases_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != settings["cases_sha256"]:
        raise ValueError("Fixed rehearsal working-point bytes changed")
    cases = json.loads(raw)["cases"]
    if len(cases) != case_count or len({case["case_id"] for case in cases}) != case_count:
        raise ValueError(f"Expected the {case_count} pinned fixed working points")
    if source_sampling:
        # The pinned 15 cases remain evaluation inputs, never training targets.
        cases = [{"case_id": "source_flat0719"}]
    count = math.floor(num_envs * settings["fraction"] + .5)
    if not len(cases) <= count <= (num_envs if flat_refresh else num_envs - 1):
        raise ValueError(f"Selected fraction cannot allocate all {case_count} cases and retain background lanes")
    plane_recovery = config.get("reward_profile") == "scut0719_flat_v1" and (not flat_refresh or scripted_mix is not None)
    if scripted_plane_lane_ids is not None:
        from .flat_episode_selection import SCUT_FLAT_RECOVERY
        recovery = config.get("recovery_training", {})
        if (not final_rough or settings["version"] != FIXED17_VERSION
                or ordinary_lane_ids is not None or flat_refresh
                or not isinstance(recovery, Mapping) or recovery.get("enabled") is not True
                or recovery.get("source_flat_handoff") != SCUT_FLAT_RECOVERY
                or not isinstance(scripted_plane_lane_ids, (list, tuple))
                or not scripted_plane_lane_ids
                or any(type(i) is not int or not 0 <= i < num_envs for i in scripted_plane_lane_ids)
                or len(set(scripted_plane_lane_ids)) != len(scripted_plane_lane_ids)):
            raise ValueError("Final rough scripted plane allocation requires explicit disjoint birth ownership")
        script_ids = set(scripted_plane_lane_ids)
        eligible = [i for i in range(num_envs) if i not in script_ids]
        if len(eligible) <= count:
            raise ValueError("Scripted plane allocation must retain fixed working points and rough lanes")
    elif flat_refresh and scripted_mix is None:
        if ordinary_lane_ids is not None and (
                not isinstance(ordinary_lane_ids, (list, tuple))
                or any(type(i) is not int for i in ordinary_lane_ids)
                or sorted(ordinary_lane_ids) != list(range(num_envs))):
            raise ValueError("All-flat refresh requires every actual lane to be ordinary")
        eligible = list(range(num_envs))
    elif plane_recovery:
        from .flat_episode_selection import validate_scut_flat_recovery
        if validate_scut_flat_recovery(config) is None:
            raise ValueError("Plane rehearsal requires explicit scripted recovery")
        if (ordinary_lane_ids is None or not isinstance(ordinary_lane_ids, (list, tuple))
                or any(type(i) is not int or not 0 <= i < num_envs for i in ordinary_lane_ids)
                or len(set(ordinary_lane_ids)) != len(ordinary_lane_ids)):
            raise ValueError("Plane rehearsal requires the actual ordinary scene lane indices")
        eligible = sorted(ordinary_lane_ids)
        if len(eligible) < count:
            raise ValueError("Ordinary lanes cannot fill the requested fixed rehearsal allocation")
        if scripted_mix is not None and not count < len(eligible) < num_envs:
            raise ValueError("Scripted flat mix requires actual script and ordinary dynamic lanes")
    else:
        if ordinary_lane_ids is not None:
            raise ValueError("Ordinary-only allocation belongs to scripted plane recovery")
        eligible = list(range(num_envs))
    # Uniformly thin eligible scene lanes; rough retains its exact old lane
    # selection. Round-robin cases differ by at most one, without RNG draws.
    selected = [eligible[math.floor((i + .5) * len(eligible) / count)] for i in range(count)]
    indices = [-1] * num_envs
    for index, lane in enumerate(selected):
        indices[lane] = index % len(cases)
    # Collision groups isolate dynamics, but coincident articulated clones
    # still overload the GPU broadphase. Spread births as on the flat stage.
    plane_ids = (sorted(set(selected) | script_ids)
                 if scripted_plane_lane_ids is not None else selected)
    columns = math.ceil(math.sqrt(len(plane_ids)))
    rows = math.ceil(len(plane_ids) / columns)
    eligible_set = set(eligible) if scripted_mix is not None else None
    origins = {lane: ((i % columns - (columns - 1) / 2) * 4.,
                      300. + (i // columns - (rows - 1) / 2) * 4., 0.)
               for i, lane in enumerate(plane_ids)} if not (plane_recovery or flat_refresh) else None
    return {"settings": settings, "case_ids": tuple(case["case_id"] for case in cases),
            "case_requests": (() if source_sampling else tuple(tuple(case["base_request"]) for case in cases)),
            "case_index": tuple(indices),
            "origins_by_env": origins,
            "lane_counts": {case["case_id"]: indices.count(i) for i, case in enumerate(cases)},
            **({"background_lanes": num_envs - count} if plane_recovery else {"rough_lanes": num_envs - count}),
            "rehearsal_lanes": count,
            "actual_lane_fraction": count / num_envs,
            **({"background_group": "non_rehearsal", "plane_recovery": True,
                "ordinary_lane_count": len(eligible), "recovery_lane_count": num_envs - len(eligible),
                "ordinary_unrehearsed_lane_count": len(eligible) - count}
               if plane_recovery else {}),
            **({"flat_refresh": True, "background_lanes": num_envs - count, "rough_lanes": 0,
                "background_group": "ordinary_unrehearsed", "ordinary_lane_count": num_envs,
                "recovery_lane_count": 0, "ordinary_unrehearsed_lane_count": num_envs - count}
               if flat_refresh and scripted_mix is None else {}),
            **({"flat_mixed_refresh": True} if flat_mixed_refresh else {}),
            **({"flat_refresh": True, "rough_lanes": 0,
                "scripted_recovery_mix": SCRIPTED_RECOVERY_MIX_VERSION,
                "recovery_lane_mask": tuple(lane not in eligible_set for lane in range(num_envs))}
               if scripted_mix is not None else {}),
            **({"uniform_final_rough": True} if final_rough else {}),
            **({"scripted_plane_lane_mask": tuple(i in script_ids for i in range(num_envs)),
                "scripted_plane_lanes": len(script_ids), "plane_lanes": len(plane_ids),
                "background_group": "non_rehearsal",
                "rough_lanes": num_envs - len(plane_ids)}
               if scripted_plane_lane_ids is not None else {})}


def reorder_fixed_flat_rehearsal(spec, native_order):
    if spec is None:
        return None
    if sorted(native_order) != list(range(len(spec["case_index"]))):
        raise ValueError("Native lane order must be a complete permutation")
    return {**spec, "case_index": tuple(spec["case_index"][i] for i in native_order),
            **({key: tuple(spec[key][i] for i in native_order)
                for key in ("population_index", "flat_lane_mask")}
               if "small_rough_mix" in spec else {}),
            **({"recovery_lane_mask": tuple(spec["recovery_lane_mask"][i] for i in native_order)}
               if spec.get("scripted_recovery_mix") else {}),
            **({"scripted_plane_lane_mask": tuple(spec["scripted_plane_lane_mask"][i] for i in native_order)}
               if "scripted_plane_lane_mask" in spec else {})}


def install_fixed_flat_rehearsal(spec, commands, *, recovery_mask=None,
                               source_flat_iteration_offset=None):
    """Install only the selected training lanes before the first reset."""
    import torch
    if len(spec["case_index"]) != commands.num_envs:
        raise ValueError("Rehearsal allocation differs from the actual command owner")
    ids = [i for i, case in enumerate(spec["case_index"]) if case >= 0]
    if "scripted_plane_lane_mask" in spec:
        from .final_rough_commands import FinalRoughCommands
        expected = torch.tensor(spec["scripted_plane_lane_mask"], dtype=torch.bool, device=commands.device)
        if (not isinstance(recovery_mask, torch.Tensor) or recovery_mask.dtype != torch.bool
                or recovery_mask.shape != (commands.num_envs,) or recovery_mask.device != commands.device
                or not torch.equal(recovery_mask, expected) or bool(recovery_mask[ids].any())):
            raise ValueError("Final rough plane membership must match actual disjoint recovery lanes")
        if type(source_flat_iteration_offset) is not int or source_flat_iteration_offset < 0:
            raise ValueError("Scripted plane commands require an explicit nonnegative inherited flat clock")
        if (not isinstance(commands, FinalRoughCommands) or commands.physics_steps
                or bool(commands.command_counter.any()) or bool(commands.fixed_request_mask.any())
                or "source_flat_training_rehearsal" in commands.summary()):
            raise ValueError("Mixed plane commands require a fresh final rough owner before the first reset")
    elif source_flat_iteration_offset is not None:
        raise ValueError("An inherited scripted flat clock requires the explicit mixed plane allocation")
    if spec["settings"]["version"] == SOURCE_VERSION:
        if recovery_mask is not None:
            raise ValueError("Source flat rehearsal is restricted to ordinary rough training")
        commands.set_source_flat_rehearsal(ids,
            source_iteration_offset=spec["settings"]["source_iteration_offset"])
        return
    if spec.get("plane_recovery"):
        if (not isinstance(recovery_mask, torch.Tensor) or recovery_mask.dtype != torch.bool
                or recovery_mask.shape != (commands.num_envs,) or recovery_mask.device != commands.device
                or bool(recovery_mask[ids].any())):
            raise ValueError("Fixed plane rehearsal must exclude every actual recovery lane")
        if spec.get("scripted_recovery_mix") and not torch.equal(recovery_mask, torch.tensor(
                spec["recovery_lane_mask"], dtype=torch.bool, device=commands.device)):
            raise ValueError("Scripted flat mix ordinary allocation differs from actual recovery ownership")
    if spec.get("flat_refresh") and not spec.get("scripted_recovery_mix") and recovery_mask is not None:
        if (not isinstance(recovery_mask, torch.Tensor) or recovery_mask.dtype != torch.bool
                or recovery_mask.shape != (commands.num_envs,) or recovery_mask.device != commands.device
                or bool(recovery_mask.any())):
            raise ValueError("All-flat refresh cannot include recovery lanes")
    requests = [spec["case_requests"][spec["case_index"][i]] for i in ids]
    command3 = torch.tensor([[vx, 0., yaw] for vx, yaw, _ in requests],
                            dtype=torch.float32, device=commands.device)
    height = torch.tensor([height for _, _, height in requests],
                          dtype=torch.float32, device=commands.device)
    commands.set_training_rehearsal_requests(
        torch.tensor(ids, dtype=torch.long, device=commands.device), command3, height)
    if "scripted_plane_lane_mask" in spec:
        commands.set_source_flat_rehearsal(expected.nonzero(as_tuple=False).flatten(),
            source_iteration_offset=source_flat_iteration_offset)
        return {"source_flat_iteration_offset": source_flat_iteration_offset,
                "scripted_plane_lanes": spec["scripted_plane_lanes"],
                "fixed_training_lanes": len(ids),
                "clock": "explicit_parent_global_offset_plus_local_rough_actor_updates",
                "birth_or_skill_qualification": False}


def rehearsal_summary(spec):
    keys = ("settings", "lane_counts", "rough_lanes", "background_lanes", "ordinary_lane_count",
            "recovery_lane_count", "ordinary_unrehearsed_lane_count", "rehearsal_lanes", "actual_lane_fraction")
    result = {key: spec[key] for key in keys if key in spec} | {
        "scope": "fixed lane allocation; not measured effective PPO exposure",
        "commands": "fixed working points, ordinary actor context, source ground capture retained",
        "reward": ("unchanged scut0719 flat reward for every population" if spec.get("plane_recovery")
                   else "unchanged scut0730 rough reward for both populations"),
        **({"background": "non_rehearsal includes ordinary source and recovery pure-RL loss samples",
            "geometry": "existing infinite plane and unchanged environment origins"}
           if spec.get("plane_recovery") else {}),
        "evaluation": "independent fixed/source evaluation does not install training rehearsal",
        "case_generalization": "these working points are rehearsed; regression is not unseen-command generalization"}
    if "small_rough_mix" in spec:
        final_payments = spec["settings"]["reward_profile"] == "v7_final_rough_v1"
        result.update(population_lane_counts=spec["population_lane_counts"],
            commands="fixed17, continuous dynamic flat, motion transitions, recovery followup and small rough",
            geometry="finite flat mesh for all four flat roles; separate generated rough mesh",
            reward=("unified31 chassis payments on every population" if final_payments else
                    "flat payments for every flat role; small rough payments for rough lanes only"),
            source_iteration_offset=spec["small_rough_mix"]["source_flat_iteration_offset"],
            temporal=spec["small_rough_mix"]["temporal"])
        return result
    if spec["settings"]["version"] == SOURCE_VERSION:
        result.update(commands="flat baseline source sampler; continuous uniform height 0.20–0.42m for ordinary and special commands",
            reward=("unchanged rough baseline rough reward on every lane, including source-flat rehearsal"
                    if spec["settings"]["reward_profile"] == "scut0730_rough_v1" else
                    "explicit local mixture: flat baseline on selected flat lanes, rough baseline on all other lanes"),
            case_generalization="fixed15 evaluation is unchanged; continuous source training covers the same domain, not a held-out domain",
            source_iteration_offset=spec["settings"]["source_iteration_offset"])
    elif spec["settings"]["version"] == FIXED17_VERSION:
        result.update(commands="V7 fixed17 working points, including signed 2.25 rps spin; equal case lanes",
                      reward=("unchanged flat reward on every ordinary lane" if spec.get("flat_refresh") else
                              "unchanged unified rough reward on every lane, including plane rehearsal"))
        if spec.get("flat_refresh"):
            result.update(flat_refresh=(FLAT_MIXED_REFRESH_VERSION if spec.get("flat_mixed_refresh")
                                        else FLAT_REFRESH_VERSION),
                          background=("original dynamic source commands without fixed overrides; ordinary PPO samples"
                                      if spec.get("flat_mixed_refresh") else
                                      "none; every lane is an ordinary fixed working point"),
                          geometry="existing infinite plane and unchanged environment origins")
    if spec.get("uniform_final_rough"):
        result["reward"] = "unified V7 final rough 31-term chassis reward on every lane"
    if "scripted_plane_lane_mask" in spec:
        result.update(geometry="one finite plane shared by disjoint fixed and scripted births",
            scripted_plane_lanes=spec["scripted_plane_lanes"], plane_lanes=spec["plane_lanes"],
            background="rough and scripted post-handoff intervals; fixed requests counted separately",
            role_lane_counts={"fixed_nominal": spec["rehearsal_lanes"],
                              "scripted_recovery": spec["scripted_plane_lanes"],
                              "rough": spec["rough_lanes"]},
            geometry_population_scope="preclone assignment only; no command or stage admission")
    if spec.get("scripted_recovery_mix"):
        result.update(scripted_recovery_mix=SCRIPTED_RECOVERY_MIX_VERSION,
            reward="unchanged 26-term flat reward on every lane",
            background="existing direct scripted recovery and ordinary dynamic source births",
            role_lane_counts={"fixed_nominal": spec["rehearsal_lanes"],
                              "scripted_recovery": spec["recovery_lane_count"],
                              "dynamic_source": spec["ordinary_unrehearsed_lane_count"]})
    return result
