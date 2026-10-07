"""Terrain-only 5--40 mm teaching rows, composed with the native runout recipe.

This revision starts from the authenticated v21/v18 inputs. Reusing the v22
geometry recipe does not create a v22 checkpoint or adopt a staged campaign.
Fixed examinations and all other task workers retain their original contents.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from . import terrain_runout_revision as runout
from .nonstandard_start_revision import ROOT, read_bound


SOURCE_PLAN_FILE, SOURCE_PLAN_SHA = runout.SOURCE_PLAN_FILE, runout.SOURCE_PLAN_SHA
SOURCE_WORKER_FILE, SOURCE_WORKER_SHA = runout.SOURCE_WORKER_FILE, runout.SOURCE_WORKER_SHA
SOURCE_DESIGN_FILE, SOURCE_DESIGN_SHA = runout.SOURCE_DESIGN_FILE, runout.SOURCE_DESIGN_SHA
OWN_SOURCE_WORKER_FILE = runout.OWN_SOURCE_WORKER_FILE
OWN_SOURCE_WORKER_SHA = runout.OWN_SOURCE_WORKER_SHA
PLAN_FILE = "contracts/v6_task_campaign_v23.json"
WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v23.json"
DESIGN_FILE = "contracts/v6_task_campaign_terrain_expectations_v23.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v23.json"
SCHEMA = "task-plan-revision-v23"
WORKER_SCHEMA = "task-worker-small-stair-revision-v23"
SCOPE = "terrain_training_untriggered_small_stairs_and_inherited_native_runout_only"
CRITIC_ADAPTATION_UPDATES = runout.CRITIC_ADAPTATION_UPDATES
BINS = ((5, .35), (10, .30), (20, .20), (40, .15))
POOL = "untriggered_steps"
SETTINGS = dict(pool=POOL, pool_fraction=.42,
    rise_bins_mm=[height for height, _ in BINS],
    bin_probabilities=[fraction for _, fraction in BINS],
    direction_probabilities=dict(up=.5, down=.5), step_count=3,
    tread_m=.6, vx_range_m_s=[.3, .75], yaw_range_rad_s=[-.02, .02],
    height_range_m=[.29, .34], episode_seconds=32.,
    geometry_update="startup_only_static_rows", command_semantics="native_untriggered",
    fixed_evaluation="unchanged_high_steps_retained_for_later_specialist_training")
digest = runout.digest


def revision_binding():
    return dict(schema=WORKER_SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
        source_contract_sha256=SOURCE_WORKER_SHA, source_design_file=SOURCE_DESIGN_FILE,
        source_design_sha256=SOURCE_DESIGN_SHA, small_stairs=deepcopy(SETTINGS))


def sampling_patch(root=ROOT):
    """Replace one pool; retain every unrelated native row and its exact quota."""
    source = runout._source_worker(root)
    if source["behavior_pool_fractions"].get(POOL) != SETTINGS["pool_fraction"]:
        raise ValueError("Small stairs require the original 42 percent terrain pool")
    template = source["skill_specs"]["rolling_stairs_3x04"]
    groups, specs, membership = [], {}, {}
    inserted = False
    for original in source["scene_groups"]:
        name = original["name"]
        pool = source["behavior_pool_membership"][name]
        if pool != POOL:
            groups.append(deepcopy(original))
            specs[name] = deepcopy(source["skill_specs"][name])
            membership[name] = pool
            continue
        if inserted:
            continue
        inserted = True
        for height_mm, fraction in BINS:
            rise = height_mm / 1000.
            for direction, terrain, mode in (("up", "stairs", 2), ("down", "stairs_down", 3)):
                name = f"small_stairs_{direction}_{height_mm:03d}mm"
                groups.append(dict(name=name, terrain=[terrain],
                    fraction=SETTINGS["pool_fraction"] * fraction / 2))
                spec = deepcopy(template)
                # Both directions use the existing ordinary rolling skill.
                # kind=stairs_down would activate timed manual-height requests.
                spec.update(kind="rolling_stairs", mode=mode,
                    episode_seconds=SETTINGS["episode_seconds"], terrain_level=1.,
                    terrain_limits=dict(stair_rise_m=rise,
                        stair_heights_m=[rise, 2 * rise, 3 * rise],
                        stair_tread_m=SETTINGS["tread_m"]),
                    terrain_training_row=dict(grade=f"small_{height_mm:03d}mm",
                        geometry_scale=1., final_target_row=False,
                        scope="training_only_fixed_evaluation_unchanged"))
                spec["planar_command_sampling"].update(
                    vx_range_m_s=deepcopy(SETTINGS["vx_range_m_s"]),
                    yaw_range_rad_s=deepcopy(SETTINGS["yaw_range_rad_s"]))
                spec["height_sampling"]["range_m"] = deepcopy(SETTINGS["height_range_m"])
                specs[name], membership[name] = spec, POOL
    pools = deepcopy(source["behavior_pool_fractions"])
    if (not inserted or len(groups) != 101 or len(specs) != len(groups)
            or not math.isclose(math.fsum(row["fraction"] for row in groups), 1., abs_tol=1e-12)):
        raise ValueError("Small stairs lost native population coverage")
    for pool, fraction in pools.items():
        actual = math.fsum(row["fraction"] for row in groups if membership[row["name"]] == pool)
        if not math.isclose(actual, fraction, abs_tol=1e-12):
            raise ValueError("Small stairs changed a permanent pool quota: " + pool)
    return dict(scene_groups=groups, skill_specs=specs,
        behavior_pool_membership=membership, behavior_pool_fractions=pools)


def apply_revision(config, root=ROOT):
    runout.apply_revision(config)
    if not config.get("evaluation_exact_cases"):
        config.update(sampling_patch(root))
    config["task_campaign"].update(worker_revision_id="v6-task-campaign-terrain-worker-v23",
        small_stair_revision_migration=revision_binding())


def revised_design(root=ROOT):
    from .terrain_runout_runtime import runtime_expectation

    inherited = runout._source_worker(root)
    design = runout.revised_design(root)
    design["reviewed_change"] = SCOPE
    design["inherited_small_stair_design_sha256"] = SOURCE_DESIGN_SHA
    design["small_stair_curriculum_review"] = dict(deepcopy(SETTINGS),
        supersedes="untriggered_steps_only_previous_easy_medium_full_rows",
        other_pools="all_rows_specs_membership_and_fractions_preserved",
        high_step_training="deferred_not_deleted_from_fixed_evaluation",
        intermediate_checkpoint="none_direct_v21_to_v23",
        inherited_runout_recipe=runout.WORKER_SCHEMA)
    for row in design["stages"]["terrain"]["roles"].values():
        apply_revision(row["contract"], root)
        effective = deepcopy(inherited)
        effective.update(row["contract"])
        runtime = runtime_expectation(effective, num_envs=row["invocation"]["num_envs"])
        row["runtime"]["modules"]["terrain_route_runout"] = runtime["enabled"]
        row["runtime"]["terrain_route_runout"] = runtime
    return design


def canonical_worker(domain="terrain", root=ROOT):
    if domain != "terrain":
        raise ValueError("Small stairs cannot change another task worker")
    root = Path(root)
    if not runout._same_json(json.loads((root / DESIGN_FILE).read_text()), revised_design(root)):
        raise ValueError("Small stair design differs from its exact inherited recipe")
    worker = runout._source_worker(root)
    apply_revision(worker, root)
    worker["design_preflight"] = dict(file=DESIGN_FILE,
        sha256=digest(root / DESIGN_FILE), stage="terrain")
    return worker


def own_domain_source_binding(source, root=ROOT):
    binding = runout.own_domain_source_binding(source, root)
    # This records configuration composition, not fictitious v22 learning.
    binding["inheritance_path"].append(WORKER_SCHEMA)
    binding["intermediate_checkpoint"] = "none_direct_v21_to_v23"
    return binding


def validate_worker_revision(source, target, root=ROOT):
    binding = own_domain_source_binding(source, root)
    if not runout._same_json(target, canonical_worker(root=root)):
        raise ValueError("Small stairs change fields outside the exact own-domain recipe")
    return binding


def validate_frozen_workers(root=ROOT):
    root = Path(root)
    for row in runout._source_plan(root)["domains"]:
        worker = read_bound(row["contract"], row["sha256"], root)
        design = worker["design_preflight"]
        read_bound(design["file"], design["sha256"], root)
    if not runout._same_json(json.loads((root / WORKER_FILE).read_text()), canonical_worker(root=root)):
        raise ValueError("Small stair frozen worker changed undeclared fields")
    return dict(domain="terrain", source_file=SOURCE_WORKER_FILE,
        source_sha256=SOURCE_WORKER_SHA, target_file=WORKER_FILE,
        target_sha256=digest(root / WORKER_FILE), design_file=DESIGN_FILE,
        design_sha256=digest(root / DESIGN_FILE))


def plan_sampling_pools(root=ROOT):
    plan = runout._source_plan(root)
    pools = next(row for row in plan["domains"] if row["name"] == "terrain")["sampling_pools"]
    patch = sampling_patch(root)
    next(pool for pool in pools if pool["name"] == POOL)["skills"] = [
        row["name"] for row in patch["scene_groups"]
        if patch["behavior_pool_membership"][row["name"]] == POOL]
    return pools


def revised_plan(root=ROOT):
    root = Path(root)
    plan = runout._source_plan(root)
    plan["contract_id"] = "v6-task-campaign-v23"
    row = next(row for row in plan["domains"] if row["name"] == "terrain")
    row.update(contract=WORKER_FILE, sha256=digest(root / WORKER_FILE),
               sampling_pools=plan_sampling_pools(root))
    plan["revision_migration"].update(manifest_file=MANIFEST_FILE,
                                     source_plan_sha256=SOURCE_PLAN_SHA)
    return plan


def revised_manifest(root=ROOT):
    root = Path(root)
    source, target = runout._source_plan(root), revised_plan(root)
    raw = (json.dumps(target, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return dict(schema=SCHEMA, source_plan_file=SOURCE_PLAN_FILE,
        source_plan_sha256=SOURCE_PLAN_SHA, target_plan_file=PLAN_FILE,
        target_plan_sha256=hashlib.sha256(raw).hexdigest(),
        run_directory=target["revision_migration"]["run_directory"], geometry_scope=SCOPE,
        terrain_route_runout=deepcopy(runout.SETTINGS), small_stairs=deepcopy(SETTINGS),
        intermediate_checkpoint="none_direct_v21_to_v23",
        critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
        learning_state="actor_critic_Adam_std_preserved", task_clock="restored",
        consumption="preserved_no_refund", domain_reopening="preserve_existing_no_new_reopening",
        own_domain_predecessors=[dict(file=file, sha256=sha) for file, sha in (
            (SOURCE_WORKER_FILE, SOURCE_WORKER_SHA), (OWN_SOURCE_WORKER_FILE, OWN_SOURCE_WORKER_SHA))],
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
                      target_file=b["contract"], target_sha256=b["sha256"])
                 for a, b in zip(source["domains"], target["domains"], strict=True)],
        designs=[dict(domain="terrain", file=DESIGN_FILE, sha256=digest(root / DESIGN_FILE))])


def freeze_contracts(root=ROOT):
    """Create only four new immutable files; never start or restart training."""
    root = Path(root)
    paths = (DESIGN_FILE, WORKER_FILE, PLAN_FILE, MANIFEST_FILE)
    if any((root / name).exists() for name in paths):
        raise ValueError("Small stairs cannot replace existing frozen contracts")
    source = runout._source_plan(root)
    for row in source["domains"]:
        worker = read_bound(row["contract"], row["sha256"], root)
        design = worker["design_preflight"]
        read_bound(design["file"], design["sha256"], root)
    read_bound(OWN_SOURCE_WORKER_FILE, OWN_SOURCE_WORKER_SHA, root)
    design = revised_design(root)
    for name, make in ((DESIGN_FILE, lambda root: design), (WORKER_FILE, canonical_worker),
                       (PLAN_FILE, revised_plan), (MANIFEST_FILE, revised_manifest)):
        with (root / name).open("x") as stream:
            stream.write(json.dumps(make(root=root), indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    validate_frozen_workers(root)
    return dict(schema=SCHEMA, output_sha256={name: digest(root / name) for name in paths},
        critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
        learning_source="latest_authenticated_own_domain_seal", consumption="preserved_no_refund",
        qualification=None, deployment=False, native_runtime_verified=False)
