"""Frozen SCUT-style terrain rows and terrain-bound native command sampling.

Geometry is built once at worker startup. These are persistent difficulty rows,
not a success-rate curriculum or a promise that difficult rows are qualified.
"""
from copy import deepcopy
import math

from .nonstandard_start_revision import ROOT, read_bound

SOURCE_WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v15.json"
SOURCE_WORKER_SHA = "843c848d28557907fbe3a842b271d0a7736fdf8f56b97ec0f51ba98591d15a41"
WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v17.json"
DESIGN_FILE = "contracts/v6_task_campaign_terrain_expectations_v17.json"
SCHEMA = "task-worker-revision-v17"
SCOPE = "scut_style_static_terrain_rows_continuous_terrain_bound_commands_permanent_flat_rehearsal"
TERRAIN_POOLS = frozenset(("untriggered_steps", "surface_traversal", "surface_height_joint"))
GRADES = (("easy", 1 / 3, .25), ("medium", 2 / 3, .35), ("full", 1., .40))
GEOMETRY_FIELDS = ("slope_deg", "roughness_m", "low_step_m", "step_up_m", "step_down_m", "stair_rise_m")


def revision_binding():
    return dict(schema=SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
                source_contract_sha256=SOURCE_WORKER_SHA)


def _commands(spec, grade):
    """Bind forward route motion to its geometry; preserve ordinary-climb mode."""
    index = {"easy": 0, "medium": 1, "full": 2}[grade]
    kind = spec["kind"]
    if kind == "body_climb":
        vx = ([.15, .35], [.15, .45], [.15, .60])[index]
        height = ([.29, .34], [.28, .37], [.27, .40])[index]
        yaw = [-.02, .02]
    elif kind in ("rolling_up", "rolling_stairs", "step_down", "stairs_down"):
        vx = ([.15, .45], [.20, .75], [.25, 1.20])[index]
        height = ([.29, .33], [.27, .36], [.26, .38])[index]
        yaw = [-.02, .02]
    else:
        vx = ([.20, .60], [.20, 1.00], [.20, 1.50])[index]
        height = ([.29, .33], [.27, .36], [.26, .38])[index]
        yaw = ([-.05, .05], [-.10, .10], [-.20, .20])[index]
    spec.update(planar_command_sampling=dict(vx_range_m_s=vx,
                    yaw_range_rad_s=yaw, nominal_fraction=.25),
                height_sampling=dict(range_m=height, nominal_fraction=.25,
                    endpoint_fraction=.20, transition_seconds=4.,
                    start_at_target_fraction=.5, boundary_fraction_of_range=.1,
                    boundary_weight=2.), height_transition_seconds=4.,
                command_resampling_seconds=[1., 8.])


def _graded_spec(original, grade, scale, terrain_kind):
    spec = deepcopy(original)
    limits = spec["terrain_limits"]
    for key in GEOMETRY_FIELDS:
        if key in limits:
            limits[key] *= scale
    if "stair_heights_m" in limits:
        limits["stair_heights_m"] = [height * scale for height in limits["stair_heights_m"]]
    if "body_climb" in spec:
        # The assessment and collision geometry must describe the same tops.
        spec["body_climb"]["rises_m"] = [height * scale for height in spec["body_climb"]["rises_m"]]
    # Explicitly neutralize the global tier multiplier; limits already contain
    # the physical grade, including the exact full-size two-platform target.
    spec["terrain_level"] = scale if terrain_kind == "material" else 1.
    spec["terrain_training_row"] = dict(grade=grade, geometry_scale=scale,
        final_target_row=grade == "full", scope="training_only_fixed_evaluation_unchanged")
    _commands(spec, grade)
    return spec


def sampling_patch(root=ROOT):
    source = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    groups, specs, membership = [], {}, {}
    for original in source["scene_groups"]:
        name = original["name"]
        pool = source["behavior_pool_membership"][name]
        if pool not in TERRAIN_POOLS:
            groups.append(deepcopy(original))
            specs[name] = deepcopy(source["skill_specs"][name])
            membership[name] = pool
            continue
        for grade, scale, fraction in GRADES:
            graded_name = name if grade == "full" else name + "__terrain_" + grade
            groups.append(dict(deepcopy(original), name=graded_name,
                               fraction=original["fraction"] * fraction))
            specs[graded_name] = _graded_spec(source["skill_specs"][name], grade, scale,
                                            original["terrain"][0])
            membership[graded_name] = pool
    pools = deepcopy(source["behavior_pool_fractions"])
    if (len(groups) != 129 or set(source["skill_specs"]) - set(specs)
            or len(specs) != len(groups)
            or not math.isclose(math.fsum(row["fraction"] for row in groups), 1., abs_tol=1e-12)):
        raise ValueError("Terrain revision lost native coverage or population")
    for pool, fraction in pools.items():
        actual = math.fsum(row["fraction"] for row in groups if membership[row["name"]] == pool)
        if not math.isclose(actual, fraction, abs_tol=1e-12):
            raise ValueError("Terrain revision changed a permanent pool quota: " + pool)
    return dict(scene_groups=groups, skill_specs=specs,
                behavior_pool_membership=membership, behavior_pool_fractions=pools)


def apply_revision(config, root=ROOT):
    """Apply only the reviewed sampling patch; exact evaluations stay untouched."""
    meta = config.get("task_campaign", {})
    if meta.get("domain") != "terrain" or meta.get("clock") != "domain_actor_updates":
        raise ValueError("Terrain revision requires the unchanged independent actor clock")
    if not config.get("evaluation_exact_cases"):
        config.update(sampling_patch(root))
    meta["revision_migration"] = revision_binding()


def revised_design(root=ROOT):
    source = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    binding = source["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_static_terrain_design_sha256"] = binding["sha256"]
    design["terrain_curriculum_review"] = dict(
        mechanism="prebuilt_persistent_difficulty_rows_not_success_rate_promotion",
        row_fractions={grade: fraction for grade, _, fraction in GRADES},
        geometry_scales={grade: scale for grade, scale, _ in GRADES},
        primary_task_fraction=.56, permanent_flat_rehearsal_fraction=.44,
        collision_geometry_update="startup_only_no_mid_episode_writes",
        command_sampling="native_continuous_planar_and_reset_height_sampling",
        command_resampling_seconds=[1., 8.], inherited_long_command_tail=True,
        lower_rows_not_final_target_credit=True,
        full_double_step=dict(rises_m=[.2, .25], cumulative_tops_m=[.2, .45],
                              treads_m=[.8, .6]),
        fixed_evaluation="all_cases_and_thresholds_unchanged")
    for row in design["stages"]["terrain"]["roles"].values():
        apply_revision(row["contract"], root)
    return design


def plan_sampling_pools(root=ROOT):
    patch = sampling_patch(root)
    return [dict(name=pool, fraction=fraction,
        skills=[row["name"] for row in patch["scene_groups"]
                if patch["behavior_pool_membership"][row["name"]] == pool],
        cross_height=False) for pool, fraction in patch["behavior_pool_fractions"].items()]
