"""SHA-bound jump workers with permanent native locomotion rehearsal.

These contracts borrow command/coverage mechanisms from SCUT. Active jump and
raised-platform task definitions remain this project's own unqualified tasks.
No PPO parameter, task clock or fixed evaluation criterion is changed here.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .locomotion_revision import FLAT_FILE, FLAT_SHA, ROOT, read_bound
from .nonstandard_start import nonstandard_start_settings, runtime_expectation as birth_runtime
from .scut_base_tracking import apply_training_tracking, update_runtime_expectation

SCHEMA = "task-worker-revision-v17"
SCOPE = "scut_style_permanent_full_flat_rehearsal_native_nonstandard_birth_and_terrain_bound_jump_tasks"
SOURCES = dict(
    jump=("contracts/v6_task_campaign_jump_worker_v12.json",
          "bece6ea0672b13e7613118fda375ce27e03a0fab39ce687c8fff99c66ef93b9d"),
    stepjump=("contracts/v6_task_campaign_stepjump_worker_v6.json",
              "47cd1e6369ef7a0398fd44fb2792eaab6381c87359c67de352d062643e2371ac"))
ORIGINS = dict(jump=5290, stepjump=0)
FLAT_POOLS = dict(stand_anchors=.07, stand_continuous=.03, height=.03,
                  basic_translation=.07, basic_transitions=.06, basic_yaw=.04)
TASK_POOLS = dict(jump=dict(flat_jumps=.55, request_sequences=.15),
                  stepjump=dict(single_step_jump=.40, double_step_jump=.20, flat_jump_rehearsal=.10))
ALIASES = dict(translation="basic_translation", transitions="basic_transitions", yaw="basic_yaw")


def worker_file(domain):
    if domain not in SOURCES:
        raise ValueError("Jump revision requires jump or stepjump")
    return f"contracts/v6_task_campaign_{domain}_worker_v17.json"


def design_file(domain):
    worker_file(domain)
    return f"contracts/v6_task_campaign_{domain}_design_expectations_v17.json"


def revision_binding(domain):
    file, sha = SOURCES[domain]
    return dict(schema=SCHEMA, source_contract_file=file, source_contract_sha256=sha)


def sampling_patch(domain, root=ROOT):
    """Keep task terrain binding and join all 41 independently frozen flat groups."""
    source = read_bound(*SOURCES[domain], root)
    flat = read_bound(FLAT_FILE, FLAT_SHA, root)
    groups, specs, membership = [], {}, {}

    def add(row, spec, pool):
        name = row["name"]
        if name in specs:
            raise ValueError("Duplicate jump rehearsal group: " + name)
        groups.append(deepcopy(row))
        specs[name] = deepcopy(spec)
        membership[name] = pool

    for row in source["scene_groups"]:
        pool = source["behavior_pool_membership"][row["name"]]
        if pool in TASK_POOLS[domain]:
            add(row, source["skill_specs"][row["name"]], pool)
    for row in flat["scene_groups"]:
        pool = flat["behavior_pool_membership"][row["name"]]
        add(row, flat["skill_specs"][row["name"]], ALIASES.get(pool, pool))
    pools = {**TASK_POOLS[domain], **FLAT_POOLS}
    sizes = Counter(membership.values())
    for row in groups:
        row["fraction"] = pools[membership[row["name"]]] / sizes[membership[row["name"]]]
    expected_count = 56 if domain == "jump" else 57
    if (len(groups) != expected_count or len(specs) != len(groups)
            or set(flat["skill_specs"]) - set(specs)
            or set(source["skill_specs"]) - set(specs)
            or not math.isclose(math.fsum(row["fraction"] for row in groups), 1., abs_tol=1e-12)):
        raise ValueError("Jump revision lost frozen task or locomotion coverage")
    for ancestor in (source, flat):
        if any(specs[name] != spec for name, spec in ancestor["skill_specs"].items()):
            raise ValueError("Jump revision changed an inherited behavior")
    return dict(scene_groups=groups, skill_specs=specs,
                behavior_pool_membership=membership, behavior_pool_fractions=pools)


def sampling_metadata(domain):
    return dict(schema="scut-jump-coverage-v1", scope=SCOPE,
        flat_membership_file=FLAT_FILE, flat_membership_sha256=FLAT_SHA,
        flat_rehearsal_fraction=.30, static_fraction=.10,
        allocation_clock="constant_all_actor_updates",
        terrain_command_binding="inherited_skill_specs_with_native_manual_or_single_beam_request",
        nonstandard_birth_scope="eligible_ordinary_rehearsal_only_never_active_jump_or_request",
        active_jump_evidence="project_specific_source_reference_not_SCUT_active_jump_qualification",
        raised_landing_scope=("legacy_COM_displacement_reference_and_per_riser_physical_assessment"
                              if domain == "stepjump" else "v12_verified_contact_finite_costs"),
        outcome="unassessed_research_candidate")


def _bind_parent(config, domain, root):
    if domain == "stepjump":
        parent = worker_file("jump")
        raw = (Path(root) / parent).read_bytes()
        config["task_campaign"]["source_contract_file"] = parent
        config["task_campaign"]["source_contract_sha256"] = hashlib.sha256(raw).hexdigest()


def apply_revision(config, root=ROOT):
    """Enable new births without restarting the original bins/assist clocks."""
    meta = config.get("task_campaign", {})
    domain = meta.get("domain")
    if domain not in SOURCES or meta.get("clock") != "domain_actor_updates":
        raise ValueError("Jump revision requires the unchanged domain actor clock")
    exact = bool(config.get("evaluation_exact_cases"))
    if not exact:
        config.update(sampling_patch(domain, root))
    apply_training_tracking(config)
    config["nonstandard_start"] = nonstandard_start_settings(ORIGINS[domain], enabled=not exact)
    config["jump_cycle_reset_on_accepted"] = True
    meta["worker_revision_id"] = f"v6-task-campaign-{domain}-worker-v17"
    meta["revision_migration"] = revision_binding(domain)
    meta["sampling_revision"] = sampling_metadata(domain)
    _bind_parent(config, domain, root)


def revised_design(domain, root=ROOT):
    """Patch frozen independent roles; exact evaluation scenes stay untouched."""
    source = read_bound(*SOURCES[domain], root)
    binding = source["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_scut_jump_design_sha256"] = binding["sha256"]
    design["jump_curriculum_review"] = sampling_metadata(domain)
    for row in design["stages"][domain]["roles"].values():
        apply_revision(row["contract"], root)
        row["runtime"]["jump_cycle_reset_on_accepted"] = row["contract"]["jump_cycle_reset_on_accepted"]
        row["runtime"]["modules"]["nonstandard_start"] = row["contract"]["nonstandard_start"]["enabled"]
        row["runtime"]["nonstandard_start"] = birth_runtime(
            row["contract"]["nonstandard_start"], row["contract"]["policy_dt"])
        update_runtime_expectation(row["contract"], row["runtime"])
    return design


def build_worker(domain, root=ROOT):
    """Build against an already written design; stepjump binds the new jump bytes."""
    config = deepcopy(read_bound(*SOURCES[domain], root))
    apply_revision(config, root)
    file = design_file(domain)
    config["design_preflight"] = dict(file=file,
        sha256=hashlib.sha256((Path(root) / file).read_bytes()).hexdigest(), stage=domain)
    return config


def plan_sampling_pools(domain, root=ROOT):
    patch = sampling_patch(domain, root)
    return [dict(name=pool, fraction=fraction, skills=[row["name"] for row in patch["scene_groups"]
        if patch["behavior_pool_membership"][row["name"]] == pool], cross_height=False)
        for pool, fraction in patch["behavior_pool_fractions"].items()]
