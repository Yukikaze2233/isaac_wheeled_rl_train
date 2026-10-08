"""Opt-in sampling-only candidate from the immutable v23 source contracts.

This module does not install runtime settings, alter evaluation, or train a
policy. Recovery prefixes and terrain runout require a separate integration.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


SOURCE_BINDINGS = {
    "flat": ("contracts/v6_task_campaign_flat_worker_v6.json", "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7", 41),
    "recovery": ("contracts/v6_task_campaign_recovery_worker_v21.json", "eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2", 95),
    "speed": ("contracts/v6_task_campaign_speed_worker_v18.json", "42d41d69b6040c0cc550bf8c65bcbfbc369e7f9efec280e2b71b942918ca0090", 135),
    "terrain": ("contracts/v6_task_campaign_terrain_worker_v23.json", "d03df2511996de16d2cc6fadaa355965c2abf938646976e1b10e496083322a9a", 101),
    "jump": ("contracts/v6_task_campaign_jump_worker_v18.json", "1f148d9ac6a09d66694c63acef7ae869277129fb9fea342c8c4c5b581d8a4c64", 56),
    "stepjump": ("contracts/v6_task_campaign_stepjump_worker_v18.json", "4b3c3708b317d5174a963de95d4234d89ab295158994fc01b6d9e44e6618ecb4", 57),
}
COMPONENTS = {
    "speed": (("speed", .8), ("recovery", .2)),
    "terrain": (("terrain", .6), ("speed", .25), ("recovery", .15)),
    "jump": (("jump", .4), ("terrain", .25), ("speed", .2), ("recovery", .15)),
    "stepjump": (("stepjump", .35), ("jump", .15), ("terrain", .2), ("speed", .15), ("recovery", .15)),
}
# Reviewed exceptions preserve both native trajectories, with index changes only.
TERRAIN_RENAMES = {name: "terrain_" + name for name in (
    "release_stop_combined", "release_stop_combined__height_train",
    "release_stop_translate", "release_stop_translate__height_train",
    "release_stop_yaw", "release_stop_yaw__height_train")}
STATIONARY_POOLS = frozenset({"stand_anchors", "stand_continuous"})
SAMPLING_FIELDS = ("scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions")
RUNTIME_FIELDS = ("recovery_training", "terrain_route_runout", "release_stop",
    "ordinary_posture_curriculum", "new_asset_curriculum", "jump_assist",
    "step_jump_task", "command_transport", "signal_delay", "stages",
    "entropy_coef", "recovery_hold_cost", "locomotion_difficulty")
ABI_FIELDS = ("asset_directory", "asset_manifest_sha256", "control_math_source",
    "control_math_sha256", "actor_dim", "critic_dim", "action_dim", "history_length",
    "physics_dt", "policy_dt", "position_iterations", "velocity_iterations",
    "closure_gap_termination_m", "v5_control")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def load_sources(root):
    """Read exact source025/v23 native contracts; reject modified source bytes."""
    sources = {}
    for domain, (filename, expected, count) in SOURCE_BINDINGS.items():
        data = (Path(root) / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source SHA mismatch: {domain}: {filename}")
        config = json.loads(data)
        if len(config["scene_groups"]) != count or len(config["skill_specs"]) != count:
            raise ValueError(f"Native source group count changed: {domain}")
        sources[domain] = config
    return sources


def _native_rows(sources):
    """Validate all native rows before component exclusion can hide a conflict."""
    catalog, domain_rows = {}, {}
    for domain, config in sources.items():
        if any(config.get(field) != sources["flat"].get(field) for field in ABI_FIELDS):
            raise ValueError(f"Native ABI incompatible: {domain}")
        rows = {}
        for source_row in config["scene_groups"]:
            old_name = source_row["name"]
            name = TERRAIN_RENAMES.get(old_name, old_name) if domain == "terrain" else old_name
            row = deepcopy(source_row)
            row["name"] = name
            spec = deepcopy(config["skill_specs"][old_name])
            pool = config["behavior_pool_membership"][old_name]
            if name in rows or not math.isfinite(row["fraction"]) or row["fraction"] <= 0:
                raise ValueError(f"Duplicate or invalid native row: {domain}:{name}")
            terrain = {key: value for key, value in row.items() if key not in {"name", "fraction"}}
            if name in catalog:
                previous = catalog[name]
                if spec != previous["spec"] or terrain != previous["terrain"]:
                    raise ValueError(f"Native spec/terrain conflict: {name}: {previous['domain']} vs {domain}")
            else:
                catalog[name] = {"spec": spec, "terrain": terrain, "domain": domain}
            rows[name] = {"row": row, "spec": spec, "pool": pool,
                          "source_name": old_name, "native_fraction": source_row["fraction"]}
        if set(config["skill_specs"]) != {item["source_name"] for item in rows.values()}:
            raise ValueError(f"Uncovered native specs: {domain}")
        domain_rows[domain] = rows
    return domain_rows


def _compose(sources, domain, count=4096):
    if domain not in COMPONENTS:
        raise ValueError(f"Unsupported integrated target: {domain}")
    native = _native_rows(sources)
    rows, specs, membership, provenance, components = [], {}, {}, {}, []
    for source, fraction in COMPONENTS[domain]:
        selected = [(name, item) for name, item in native[source].items()
                    if name not in specs and (source != "recovery" or item["spec"]["kind"] == "recovery_handoff")]
        if not selected:
            raise ValueError(f"Empty integrated component: {source}")
        total = math.fsum(item["native_fraction"] for _, item in selected)
        components.append({"source": source, "nominal_fraction": fraction,
                           "selected_names": [name for name, _ in selected]})
        for name, item in selected:
            row = deepcopy(item["row"])
            row["fraction"] = fraction * item["native_fraction"] / total
            rows.append(row)
            specs[name] = deepcopy(item["spec"])
            membership[name] = item["pool"]
            provenance[name] = {"component": source, "source_name": item["source_name"],
                "source_file": SOURCE_BINDINGS[source][0], "source_sha256": SOURCE_BINDINGS[source][1],
                "source_pool": item["pool"], "source_fraction": item["native_fraction"],
                "native_spec_sha256": _digest(item["spec"]),
                "native_terrain_sha256": _digest({k: v for k, v in item["row"].items() if k not in {"name", "fraction"}}),
                "nominal_component_row_fraction": row["fraction"],
                "renamed": name != item["source_name"]}

    stationary = [row for row in rows if membership[row["name"]] in STATIONARY_POOLS]
    before = math.fsum(row["fraction"] for row in stationary)
    floor_applied = before < .1
    if floor_applied:
        flat = native["flat"]
        anchor_weights = {name: item["native_fraction"] for name, item in flat.items()
                          if item["pool"] in STATIONARY_POOLS}
        if set(anchor_weights) != {row["name"] for row in stationary}:
            raise ValueError("Stationary floor must preserve all native flat anchors")
        total = math.fsum(anchor_weights.values())
        for row in rows:
            name = row["name"]
            row["fraction"] = (.1 * anchor_weights[name] / total if name in anchor_weights
                               else row["fraction"] * .9 / (1 - before))
    pools = defaultdict(float)
    for row in rows:
        pools[membership[row["name"]]] += row["fraction"]
        provenance[row["name"]]["effective_fraction"] = row["fraction"]
    if not math.isclose(math.fsum(row["fraction"] for row in rows), 1., abs_tol=1e-12):
        raise ValueError("Integrated fractions must sum to one")
    # Exercise the existing allocator, including its real floor/ceil constraints.
    from .task import choose_scene_groups
    allocation = Counter(name for name, _ in choose_scene_groups(rows, count))
    expected = set().union(*(set(native[source]) if source != "recovery" else
        {name for name, item in native[source].items() if item["spec"]["kind"] == "recovery_handoff"}
        for source, _ in COMPONENTS[domain]))
    if set(specs) != expected or set(allocation) != expected or min(allocation.values()) < 1:
        raise ValueError("Integrated native group coverage or allocation failed")
    for component in components:
        component["effective_fraction"] = math.fsum(provenance[name]["effective_fraction"]
                                                     for name in component["selected_names"])
    source_coverage = {source: {item["source_name"]: name for name, item in native[source].items()
        if name in specs} for source, _ in COMPONENTS[domain]}
    runtime_differences = {}
    for field in RUNTIME_FIELDS:
        values = {source: sources[source].get(field) for source, _ in COMPONENTS[domain]}
        if len({_digest(value) for value in values.values()}) > 1:
            runtime_differences[field] = {source: deepcopy(value) for source, value in values.items()}
    audit = {"schema": "v23-integrated-sampling-candidate-v1", "domain": domain,
        "source_bindings": {source: {"file": filename, "sha256": sha, "groups": number}
                            for source, (filename, sha, number) in SOURCE_BINDINGS.items()},
        "components": components, "row_provenance": provenance, "source_coverage": source_coverage,
        "reviewed_terrain_renames": deepcopy(TERRAIN_RENAMES),
        "stationary_floor": {"applied": floor_applied, "minimum_fraction": .1,
            "before_fraction": before, "after_fraction": math.fsum(row["fraction"] for row in stationary),
            "flat_source_sha256": SOURCE_BINDINGS["flat"][1]},
        "allocation": {"environment_count": count, "groups": len(allocation),
                       "minimum_count": min(allocation.values()), "counts": dict(allocation)},
        "pending_runtime_configuration": runtime_differences,
        "runtime_ready": False, "qualification": None, "PPO_updates": 0,
        "scope": "Sampling-only candidate; native specs/terrain preserved. Prefix/profile clocks, runout, release-stop and other runtime configuration must be integrated separately; no skill or deployment claim."}
    return {"scene_groups": rows, "skill_specs": specs, "behavior_pool_membership": membership,
            "behavior_pool_fractions": dict(pools)}, audit


def apply(config, root, *, role="train", count=4096, ordinary_stairs=False):
    """Return an opt-in candidate; every non-train role remains exactly unchanged."""
    result = deepcopy(config)
    if role != "train":
        return result
    domain = config["task_campaign"]["domain"]
    sources = load_sources(root)
    transform = None
    native_terrain = sources["terrain"]
    if type(ordinary_stairs) is not bool:
        raise ValueError("Ordinary stair synthesis requires an explicit boolean opt-in")
    if ordinary_stairs:
        if domain == "speed":
            raise ValueError("Ordinary stairs cannot change the completed speed recipe")
        from .ordinary_large_stair import ROWS, sampling_candidate
        source_file = "src/wheeled_tasks/chassis/ordinary_large_stair.py"
        sources["terrain"] = sampling_candidate(native_terrain, enabled=True,
            origin_actor_update=0, actor_allowance=15000)
        transform = dict(schema="v27-ordinary-terrain-source-transform-v1",
            file=source_file, sha256=hashlib.sha256((Path(root) / source_file).read_bytes()).hexdigest(),
            input_source=dict(file=SOURCE_BINDINGS["terrain"][0], sha256=SOURCE_BINDINGS["terrain"][1],
                              groups=SOURCE_BINDINGS["terrain"][2]),
            derived_groups=len(sources["terrain"]["scene_groups"]),
            added_names=[row[0] for row in ROWS],
            static_parameters=dict(origin_actor_update=0, actor_allowance=15000),
            clock_status="schema_only_actual_admission_parameters_required",
            qualification=None)
    patch, audit = _compose(sources, domain, count)
    if transform is not None:
        audit["source_transforms"] = dict(terrain=transform)
        native_rows = {row["name"]: row for row in native_terrain["scene_groups"]}
        for row in audit["row_provenance"].values():
            if row["component"] != "terrain":
                continue
            name = row["source_name"]
            row["source_transform"] = "terrain"
            row["source_kind"] = ("derived_row" if name in transform["added_names"] else
                "reweighted_native_row" if row["source_fraction"] != native_rows[name]["fraction"] else "native_row")
            row["immutable_source_fraction"] = native_rows[name]["fraction"] if name in native_rows else None
            if name in transform["added_names"]:
                row.update(source_name=None, derived_name=name,
                    derived_spec_sha256=row.pop("native_spec_sha256"),
                    derived_terrain_sha256=row.pop("native_terrain_sha256"))
        # Never enlarge immutable input coverage to claim that its new rows existed.
        audit["source_coverage"]["terrain"] = {
            name: value for name, value in audit["source_coverage"]["terrain"].items() if name in native_rows}
        audit["scope"] = ("Train-only terrain source derivation; new rows and reweighted native rows "
            "bind their immutable v23 input and explicit transform. Static clocks require actual admission; "
            "no skill or deployment claim.")
    result.update(patch)
    result["integrated_sampling_candidate"] = audit
    return result
