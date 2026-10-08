"""Opt-in, local synthesis of joint runtime recipes; no production entry uses it.

Sources, the actual learning source and its clocks are authenticated separately.
The result is an integration candidate, never a native-startup receipt. Per-row
settings/clock routing still requires explicit wiring in the existing runtime.
No checkpoint, account, optimizer, physics or FSM is mutated by this module.
"""
from copy import deepcopy
import json
from pathlib import Path

from . import integrated_lineage as lineage
from . import integrated_sampling as sampling
from . import scut_reuse_curriculum as reuse
from .locomotion_difficulty import difficulty_settings, validate_difficulty, difficulty_envelope
from .nonstandard_start import nonstandard_start_settings, validate_nonstandard_start
from .ordinary_posture_curriculum import posture_settings, validate_posture_settings
from .recovery_prefix_curriculum import validate_prefix_curriculum
from .release_stop import validate_release_stop
from .training_schedule import stage_actor_updates, jump_bin_probabilities

SCHEMA = "v6-integrated-runtime-candidate-v1"
CLOCK_SCHEMA = "v6-integrated-component-clocks-v1"
STAGES = tuple(dict(lineage.CHAIN))
AUDIT = ("reports/v6_scut_posture_completion_20261006/source_repair_026/production_integration_audit001.json",
         "46d0a2ecb27d1717b8845375b349738efebd8d6e255fffbb30420201cc05d305")
SMALL_STAIR_NAMES = ("small_stairs_up_005mm", "small_stairs_down_005mm",
                     "small_stairs_up_010mm", "small_stairs_down_010mm",
                     "small_stairs_up_020mm", "small_stairs_down_020mm",
                     "small_stairs_up_040mm", "small_stairs_down_040mm")
NATIVE_CALL_SITES = {
    "recovery_prefix": "recovery_training.RecoveryTraining.initial_state/metrics",
    "posture": "env.ChassisEnv.reset: OrdinaryPostureCurriculum.reset; startup expectation and metrics",
    "nonstandard": "env.ChassisEnv.reset: NonstandardStart.reset; startup expectation and metrics",
    "difficulty": "skill_commands.SkillCommands.sample/verify_difficulty_samples/difficulty_metrics",
    "height_and_requests": "training_schedule.TrainingSchedule.actor_updates/sample_height/reset/event sampling",
    "jump_assist": "env.ChassisEnv.reset/step: JumpAssist.reset/apply",
    "terrain_runout": "terrain_runout.validate_runout/route_runout_selection and env geometry/boundary readback",
    "clocks": "train_chassis save/load: matching integrated_runtime_clocks in infos and per-file seal",
}


def _number(value, name):
    return lineage._integer(value, name)


def _pin(root, file, sha):
    path = lineage._within(file, root)
    if lineage.digest(path) != lineage._sha(sha):
        raise ValueError("Runtime source/reference SHA differs: " + file)
    return dict(file=file, sha256=sha)


def _references(root, configs):
    """Pin every declared file/SHA pair, including nested profile/parent/design files."""
    pins = {}

    def visit(value):
        if isinstance(value, dict):
            for key, file in value.items():
                sha_key = "sha256" if key == "file" else key[:-5] + "_sha256" if key.endswith("_file") else None
                if sha_key in value and isinstance(file, str):
                    pin = _pin(root, file, value[sha_key])
                    if file in pins and pins[file] != pin:
                        raise ValueError("Runtime file has conflicting source bindings")
                    pins[file] = pin
                visit(file)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for config in configs.values():
        visit(config)
        file = config["control_math_source"]
        pins[file] = _pin(root, file, config["control_math_sha256"])
    return pins


def _verified_selection(binding, selection):
    mode = selection.get("mode")
    kwargs = ({"own_candidates": [selection["checkpoint"]]} if mode == "resume_joint_own" else
              {"parent": selection["checkpoint"], "parent_record": selection["parent_record"]}
              if mode == "inherit_completed_joint_parent" else {})
    if mode not in ("resume_joint_own", "inherit_completed_joint_parent", "rebase_from_normal_REC", "rebase_from_authorized_REC"):
        raise ValueError("Runtime needs an authenticated joint learning-source selection")
    verified = lineage.select_learning_source(binding, selection["domain"], selection["entry_admission"], **kwargs)
    if verified != selection:
        raise ValueError("Runtime selection differs from the actual source clocks")
    return verified


def _clock_document(selection, starts, joint_actor, domain_actor):
    domain = selection["domain"]
    origin = _number(selection.get("origin_recovery_actor_update", 15000), "REC origin actor clock")
    components = {"recovery": origin + joint_actor,
                  **{component: joint_actor - start for component, start in starts.items()}}
    result = dict(schema=CLOCK_SCHEMA, lineage_id=selection["lineage_id"], plan_sha256=selection["plan_sha256"],
                origin_checkpoint_sha256=selection["origin_checkpoint_sha256"], domain=domain,
                joint_actor_updates=joint_actor, domain_actor_updates=domain_actor,
                component_start_joint_actor_updates=deepcopy(starts), component_actor_updates=components)
    if "origin_recovery_actor_update" in selection:
        result["origin_recovery_actor_update"] = origin
    return result


def _validate_clock_document(document, selection):
    domain = selection["domain"]
    joint_actor = _number(selection["fresh_joint_actor_updates"], "joint learning actor clock")
    domain_actor = _number(selection["fresh_domain_actor_updates"], "domain learning actor clock")
    if not isinstance(document, dict):
        raise ValueError("Joint source lacks sealed component clocks")
    starts = document.get("component_start_joint_actor_updates")
    expected_stages = STAGES[:STAGES.index(domain) + 1]
    if not isinstance(starts, dict) or set(starts) != set(expected_stages):
        raise ValueError("Component clocks do not cover the learned joint stages")
    ordered = [_number(starts[component], "component joint start") for component in expected_stages]
    if (ordered[0] != 0 or any(a > b for a, b in zip(ordered, ordered[1:]))
            or starts[domain] + domain_actor != joint_actor):
        raise ValueError("Component starts differ from fresh joint/domain learning clocks")
    expected = _clock_document(selection, starts, joint_actor, domain_actor)
    if document != expected:
        raise ValueError("Component clock metadata differs from the actual joint learning state")
    return deepcopy(expected)


def source_clocks(binding, selection):
    """Derive new-domain clocks only from the actual sealed, completed joint parent."""
    selection = _verified_selection(binding, selection)
    domain = selection["domain"]
    if selection["mode"] in ("rebase_from_normal_REC", "rebase_from_authorized_REC"):
        return _clock_document(selection, {"speed": 0}, 0, 0)
    import torch

    path = lineage._within(selection["checkpoint"]["path"], binding.run_root)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    seal = json.loads(path.with_suffix(".checkpoint.json").read_text())
    document = saved["infos"].get("integrated_runtime_clocks")
    if seal.get("integrated_runtime_clocks") != document:
        raise ValueError("Source seal and full-state component clocks differ")
    if selection["mode"] == "resume_joint_own":
        return _validate_clock_document(document, selection)
    parent_meta = saved["infos"]["integrated_lineage"]
    parent_selection = dict(selection, domain=parent_meta["domain"],
                            fresh_joint_actor_updates=parent_meta["joint_actor_updates"],
                            fresh_domain_actor_updates=parent_meta["domain_actor_updates"])
    parent = _validate_clock_document(document, parent_selection)
    starts = deepcopy(parent["component_start_joint_actor_updates"])
    starts[domain] = parent["joint_actor_updates"]
    # No planned endpoint, requested budget or old independent clock supplies a base.
    return _clock_document(selection, starts, parent["joint_actor_updates"], 0)


def advance_clock_document(document, actor_delta):
    """Compute a proposed save value; the worker must seal it with actual counters."""
    actor_delta = _number(actor_delta, "actor-only component increment")
    selection = dict(domain=document["domain"], lineage_id=document["lineage_id"],
                     plan_sha256=document["plan_sha256"], origin_checkpoint_sha256=document["origin_checkpoint_sha256"],
                     fresh_joint_actor_updates=document["joint_actor_updates"],
                     fresh_domain_actor_updates=document["domain_actor_updates"])
    if "origin_recovery_actor_update" in document:
        selection["origin_recovery_actor_update"] = document["origin_recovery_actor_update"]
    state = _validate_clock_document(document, selection)
    return _clock_document(selection, state["component_start_joint_actor_updates"],
                           state["joint_actor_updates"] + actor_delta, state["domain_actor_updates"] + actor_delta)


def _component_runtime(sources, clocks):
    result = {}
    for component in clocks["component_actor_updates"]:
        source = sources[component]
        fields = ("ordinary_posture_curriculum", "nonstandard_start", "locomotion_difficulty",
                  "new_asset_curriculum", "jump_assist")
        settings = {field: deepcopy(source[field]) for field in fields if field in source}
        if component != "recovery":
            settings["ordinary_posture_curriculum"] = posture_settings(0)
            if "nonstandard_start" in settings:
                settings["nonstandard_start"] = nonstandard_start_settings(0)
        if component == "speed":
            settings["locomotion_difficulty"] = difficulty_settings(0)
        validate_posture_settings(settings["ordinary_posture_curriculum"])
        if "nonstandard_start" in settings:
            validate_nonstandard_start(settings["nonstandard_start"])
        validate_difficulty(settings.get("locomotion_difficulty"))
        jump_bin_probabilities(settings["new_asset_curriculum"]["jump_curriculum"],
                               actor_update=clocks["component_actor_updates"][component])
        result[component] = settings
    return result


def _row_ownership(candidate, sources):
    ordinary_rec = {name for name, spec in sources["recovery"]["skill_specs"].items()
                    if spec["kind"] != "recovery_handoff"}
    speed_names = set(sources["speed"]["skill_specs"])
    owners = {}
    for name, spec in candidate["skill_specs"].items():
        component = candidate["integrated_sampling_candidate"]["row_provenance"][name]["component"]
        owners[name] = dict(component=component,
                           posture_component="recovery" if name in ordinary_rec else component,
                           difficulty_component="speed" if name in speed_names else None,
                           height_component="recovery" if name in ordinary_rec else component,
                           nonstandard_component=component if component != "recovery" else None,
                           request_component=component,
                           terrain_runout=bool(spec.get("terrain_training_row")))
    return owners


def apply(config, root, repository, *, binding=None, selection=None, role="train", count=4096):
    """Compose a pinned local candidate; all non-training roles are exact no-ops."""
    if role != "train":
        return deepcopy(config)
    sources = sampling.load_sources(root)
    domain = config.get("task_campaign", {}).get("domain")
    if domain not in STAGES or config != sources[domain]:
        raise ValueError("Runtime synthesis requires the exact frozen native source worker")
    if binding is None or selection is None or selection.get("domain") != domain:
        raise ValueError("Runtime synthesis requires its actual authenticated lineage source")
    clocks = source_clocks(binding, selection)
    actual_contract = json.loads(lineage.checkpoint_contract_path(selection["checkpoint"]["path"]).read_text())
    if any(actual_contract.get(key) != config.get(key) for key in lineage.PHYSICAL_KEYS):
        raise ValueError("Runtime source and selected learning state have different physical/policy identity")
    audit_pin = _pin(root, *AUDIT)
    references = _references(root, sources)
    # These eight frozen native names are the exact preview002 binding list.
    # Explicitly select the forward interval; do not infer a band from terrain kind.
    terrain_bindings = ({name: dict(override="low_speed_stair_for_rm", vx_interval=0)
                         for name in SMALL_STAIR_NAMES} if "terrain" in clocks["component_actor_updates"] else {})
    candidate = reuse.apply(sampling.apply(config, root, count=count), repository,
                           terrain_bindings=terrain_bindings)
    components = _component_runtime(sources, clocks)
    recovery = deepcopy(sources["recovery"]["recovery_training"])
    prefix = recovery["prefix_curriculum"]
    prefix["origin_actor_update"] = 0
    prefix["keyframes"] = [dict(actor_update=0, probabilities=deepcopy(prefix["keyframes"][-1]["probabilities"]))]
    prefix["release_schedule"] = [dict(actor_update=0, seconds=prefix["release_schedule"][-1]["seconds"])]
    validate_prefix_curriculum(prefix)
    candidate.update(recovery_training=recovery, recovery_hold_cost=deepcopy(sources["recovery"]["recovery_hold_cost"]),
                     entropy_coef=sources["recovery"]["entropy_coef"],
                     ordinary_posture_curriculum=posture_settings(0),
                     nonstandard_start=nonstandard_start_settings(0), locomotion_difficulty=difficulty_settings(0),
                     release_stop=deepcopy(sources["speed"]["release_stop"]),
                     episode_seconds=max(sources[c]["episode_seconds"] for c in components))
    validate_release_stop(candidate["release_stop"])
    from .response_migration import validate_recovery_configuration, validate_step_task_configuration
    from .rewards import validate_recovery_hold_cost
    from .skill_commands import validate_planar_sampling
    validate_recovery_configuration(candidate["recovery_training"], root)
    validate_step_task_configuration(candidate["step_jump_task"], root)
    validate_recovery_hold_cost(candidate["recovery_hold_cost"])
    for spec in candidate["skill_specs"].values():
        validate_planar_sampling(spec)
    if "terrain" in components:
        candidate["terrain_route_runout"] = deepcopy(sources["terrain"]["terrain_route_runout"])
    if "jump" in components:
        from .jump_landing import validate_jump_landing
        candidate["jump_landing"] = deepcopy(sources["jump"]["jump_landing"])
        validate_jump_landing(candidate["jump_landing"])
        candidate["jump_cycle_reset_on_accepted"] = sources["jump"]["jump_cycle_reset_on_accepted"]
    # Preserve physical/FSM configuration, exams, caps and original stage identifiers.
    # Top-level curricula are only placeholders until component routing is wired.
    candidate["integrated_runtime_candidate"] = dict(schema=SCHEMA, domain=domain, role="train",
        runtime_ready=False, qualification=None, PPO_updates=0,
        source_bindings=deepcopy(candidate["integrated_sampling_candidate"]["source_bindings"]),
        scut_source_bindings={profile: deepcopy(data["files"]) for profile, data in reuse.read_sources(repository).items()},
        scut_terrain_bindings=deepcopy(terrain_bindings),
        production_integration_audit=audit_pin, referenced_files=references,
        learning_source=deepcopy(selection), clocks_at_source=clocks, component_runtime=components,
        row_ownership=_row_ownership(candidate, sources),
        recovery_prefix_strategy="legal_single_keyframe_frozen_mature_source_endpoint",
        shared_signal_delay_clock="fresh_joint_actor_updates",
        pending_native_call_sites=deepcopy(NATIVE_CALL_SITES),
        readiness_requirements=["explicit production entry and frozen migration/contract/preflight",
            "per-row component settings/clocks installed in all listed native call sites",
            "terrain runout opt-in for mixed domains; real geometry/boundary readback",
            "source and component clocks written/restored in native infos plus checkpoint seal",
            "native runtime validation on the actual admitted normal-boundary source"],
        budget="unchanged; no endpoint/allowance is inferred by runtime synthesis")
    return candidate


def component_actor_updates(env, component):
    """Future opt-in call-site helper; existing configs retain their native clock."""
    native_clock = stage_actor_updates(env)
    audit = env.cfg.get("integrated_runtime_candidate")
    if audit is None:
        return native_clock
    if audit.get("schema") != SCHEMA or audit.get("domain") != env.cfg["task_campaign"]["domain"]:
        raise ValueError("Component clock belongs to another runtime recipe")
    clocks = audit["clocks_at_source"]
    _validate_clock_document(clocks, audit["learning_source"])
    if component not in clocks["component_actor_updates"] or native_clock < clocks["domain_actor_updates"]:
        raise ValueError("Unknown component or regressed task actor clock")
    return clocks["component_actor_updates"][component] + native_clock - clocks["domain_actor_updates"]


def row_settings(env, group, field, *, height_sampling=False):
    """Resolve existing schema and its scalar clock before native per-row sampling."""
    audit = env.cfg.get("integrated_runtime_candidate")
    if audit is None:
        return env.cfg.get(field), stage_actor_updates(env)
    routing = {"ordinary_posture_curriculum": "posture_component", "nonstandard_start": "nonstandard_component",
               "locomotion_difficulty": "difficulty_component", "new_asset_curriculum": "request_component",
               "jump_assist": "request_component"}
    if field not in routing or group not in audit["row_ownership"]:
        raise ValueError("Unbound joint row/runtime field")
    if height_sampling and field != "new_asset_curriculum":
        raise ValueError("Height clock routing only applies to native asset curriculum")
    owner = "height_component" if height_sampling else routing[field]
    component = audit["row_ownership"][group][owner]
    if component is None:
        return None, stage_actor_updates(env)
    return deepcopy(audit["component_runtime"][component].get(field)), component_actor_updates(env, component)


def row_difficulty_envelope(env, group):
    settings, clock = row_settings(env, group, "locomotion_difficulty")
    return difficulty_envelope(settings, clock) if settings is not None else None
