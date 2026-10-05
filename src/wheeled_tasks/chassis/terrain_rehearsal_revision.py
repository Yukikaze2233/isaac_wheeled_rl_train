"""Exact terrain sampling revision; learning state and task clocks stay native."""
from collections import Counter
from copy import deepcopy
import math

from .locomotion_revision import FLAT_FILE, FLAT_SHA, ROOT, read_bound

SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v14.json"
SOURCE_PLAN_SHA = "41319993aa53a54e32efaab725ec8b0e269f4b8b3576bc101f23c6e525c4cf73"
SOURCE_WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v14.json"
SOURCE_WORKER_SHA = "35014b99f4c8ac7423df386bdddf809699c10fc75be56e6a73002f8f532be1d0"
LEGACY_WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v6.json"
LEGACY_WORKER_SHA = "4337842f9689ec2a4ecd0994dfb39ef821c95a9868a91f445de52f2a0afc456e"
SPEED_FILE = "contracts/v6_task_campaign_speed_worker_v14.json"
SPEED_SHA = "f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3"
PLAN_FILE = "contracts/v6_task_campaign_v15.json"
WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v15.json"
DESIGN_FILE = "contracts/v6_task_campaign_terrain_expectations_v15.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v15.json"
SCHEMA = "task-worker-revision-v15"
SCOPE = "terrain_sampling_full_flat_rehearsal_bounded_high_translation_and_surface_height_joint"
SAMPLING_KEYS = ("scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions")
POOL_FRACTIONS = dict(untriggered_steps=.42, surface_traversal=.10, surface_height_joint=.04,
    stand_anchors=.09, stand_continuous=.06, height=.025, basic_translation=.06,
    basic_yaw=.045, basic_transitions=.06, high_translation=.05, release_stop=.05)
SURFACES = ("slope_up", "slope_down", "cross_slope", "rough", "material_transfer")


def revision_binding():
    return dict(schema=SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
        source_contract_sha256=SOURCE_WORKER_SHA, accepted_resume_sources=[
            dict(source_contract_file=file, source_contract_sha256=sha)
            for file, sha in ((SOURCE_WORKER_FILE, SOURCE_WORKER_SHA),
                              (LEGACY_WORKER_FILE, LEGACY_WORKER_SHA))])


def sampling_patch(root=ROOT):
    """Join SHA-bound memberships; high translation is explicitly flat only."""
    terrain = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    flat = read_bound(FLAT_FILE, FLAT_SHA, root)
    speed = read_bound(SPEED_FILE, SPEED_SHA, root)
    groups, specs, membership = [], {}, {}

    def add(row, spec, pool):
        name = row["name"]
        if name in specs:
            raise ValueError("Duplicate terrain rehearsal group: " + name)
        groups.append(deepcopy(row))
        specs[name] = deepcopy(spec)
        membership[name] = pool

    for row in terrain["scene_groups"]:
        pool = terrain["behavior_pool_membership"][row["name"]]
        if pool in ("untriggered_steps", "surface_traversal"):
            add(row, terrain["skill_specs"][row["name"]], pool)
    aliases = dict(translation="basic_translation", yaw="basic_yaw", transitions="basic_transitions")
    for row in flat["scene_groups"]:
        pool = flat["behavior_pool_membership"][row["name"]]
        add(row, flat["skill_specs"][row["name"]], aliases.get(pool, pool))
    high = {f"{direction}_{value}{height}" for direction in ("forward", "backward")
        for value in (2, 3, 4, 5) for height in ("", "__height_train")}
    for row in speed["scene_groups"]:
        if row["name"] in high:
            add(dict(row, terrain=["flat"]), speed["skill_specs"][row["name"]], "high_translation")
    for row in terrain["scene_groups"]:
        if terrain["behavior_pool_membership"][row["name"]] == "release_stop":
            add(row, terrain["skill_specs"][row["name"]], "release_stop")
    original = {row["name"]: row for row in terrain["scene_groups"]}
    for name in SURFACES:
        spec = deepcopy(terrain["skill_specs"][name])
        spec.update(planar_command_sampling=dict(vx_range_m_s=[.2, .8],
            yaw_range_rad_s=[-.1, .1], nominal_fraction=.25), height_sampling=dict(
                range_m=[.26, .36], nominal_fraction=0., endpoint_fraction=.2,
                transition_seconds=4., start_at_target_fraction=.5,
                boundary_fraction_of_range=.1, boundary_weight=2.), height_transition_seconds=4.)
        add(dict(original[name], name=name + "__height_train"), spec, "surface_height_joint")
    sizes = Counter(membership.values())
    for row in groups:
        row["fraction"] = POOL_FRACTIONS[membership[row["name"]]] / sizes[membership[row["name"]]]
    if (len(specs) != 85 or set(flat["skill_specs"]) - set(specs)
            or set(terrain["skill_specs"]) - set(specs)
            or not math.isclose(math.fsum(row["fraction"] for row in groups), 1., abs_tol=1e-12)):
        raise ValueError("Terrain rehearsal lost its frozen coverage or population")
    for source in (flat, terrain):
        if any(specs[name] != spec for name, spec in source["skill_specs"].items()):
            raise ValueError("Terrain rehearsal changed an inherited behavior")
    return dict(scene_groups=groups, skill_specs=specs, behavior_pool_membership=membership,
        behavior_pool_fractions=deepcopy(POOL_FRACTIONS))


def apply_rehearsal_revision(config, root=ROOT):
    meta = config.get("task_campaign", {})
    if meta.get("domain") != "terrain" or meta.get("clock") != "domain_actor_updates":
        raise ValueError("Terrain rehearsal requires its original independent terrain account")
    if not config.get("evaluation_exact_cases"):
        config.update(sampling_patch(root))
    meta["revision_migration"] = revision_binding()


def revised_design(root=ROOT):
    """Transform independently frozen roles; fixed-case sampling stays exact."""
    source = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    binding = source["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_terrain_rehearsal_design_sha256"] = binding["sha256"]
    for row in design["stages"]["terrain"]["roles"].values():
        apply_rehearsal_revision(row["contract"], root)
    return design


def plan_sampling_pools(root=ROOT):
    patch = sampling_patch(root)
    return [dict(name=pool, fraction=fraction, skills=[row["name"] for row in patch["scene_groups"]
        if patch["behavior_pool_membership"][row["name"]] == pool], cross_height=False)
        for pool, fraction in POOL_FRACTIONS.items()]
