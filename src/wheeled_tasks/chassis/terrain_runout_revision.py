"""Candidate terrain-only runout binding; no campaign entry is registered here.

The v18 configuration recipe and the real v15 own-domain learning source remain
separate. Freezing new contracts never manufactures an intermediate checkpoint,
reopens a domain, starts training, or changes an existing contract.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .nonstandard_start_revision import ROOT, read_bound
from .terrain_runout import validate_runout


SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v21.json"
SOURCE_PLAN_SHA = "119c77c21da31238ac1af285d0e8df1c627816b0d9645fa099899a6496f11755"
SOURCE_WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v18.json"
SOURCE_WORKER_SHA = "d3c0658b7b5b51aa89fc08d1503e91ae57a4682173bcb076bdbd4d83ede3f611"
SOURCE_DESIGN_FILE = "contracts/v6_task_campaign_terrain_expectations_v18.json"
SOURCE_DESIGN_SHA = "183df8d257d8b5d3135e8d2d677af203870bd5906ce17606828df5310caee472"
OWN_SOURCE_WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v15.json"
OWN_SOURCE_WORKER_SHA = "843c848d28557907fbe3a842b271d0a7736fdf8f56b97ec0f51ba98591d15a41"
PLAN_FILE = "contracts/v6_task_campaign_v22.json"
WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v22.json"
DESIGN_FILE = "contracts/v6_task_campaign_terrain_expectations_v22.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v22.json"
SCHEMA = "task-plan-revision-v22"
WORKER_SCHEMA = "task-worker-terrain-runout-revision-v22"
SCOPE = "terrain_training_native_route_terminal_support_and_positive_boundary_only"
CRITIC_ADAPTATION_UPDATES = 10
SETTINGS = dict(enabled=True, support_end_x_m=6.2, positive_boundary_x_m=5.6)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serialized(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _same_json(before, after):
    # Python equality treats True as 1. Exact frozen JSON must not accept that.
    return _serialized(before) == _serialized(after)


def _source_plan(root):
    plan = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    terrain = next(row for row in plan["domains"] if row["name"] == "terrain")
    if (terrain["contract"], terrain["sha256"]) != (SOURCE_WORKER_FILE, SOURCE_WORKER_SHA):
        raise ValueError("Terrain runout requires the unchanged v21 terrain configuration source")
    return plan


def _source_worker(root):
    _source_plan(root)
    worker = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    expected = dict(schema="task-worker-revision-v18",
                    source_contract_file=OWN_SOURCE_WORKER_FILE,
                    source_contract_sha256=OWN_SOURCE_WORKER_SHA)
    if worker.get("task_campaign", {}).get("revision_migration") != expected:
        raise ValueError("Terrain runout lost the complete v18 to real v15 source binding")
    if worker.get("design_preflight") != dict(file=SOURCE_DESIGN_FILE,
                                              sha256=SOURCE_DESIGN_SHA, stage="terrain"):
        raise ValueError("Terrain runout requires the independently bound v18 design")
    return worker


def revision_binding():
    return dict(schema=WORKER_SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
                source_contract_sha256=SOURCE_WORKER_SHA,
                source_design_file=SOURCE_DESIGN_FILE,
                source_design_sha256=SOURCE_DESIGN_SHA,
                terrain_route_runout=deepcopy(SETTINGS))


def apply_revision(config):
    metadata = config.get("task_campaign", {})
    expected = dict(schema="task-worker-revision-v18",
                    source_contract_file=OWN_SOURCE_WORKER_FILE,
                    source_contract_sha256=OWN_SOURCE_WORKER_SHA)
    if (metadata.get("domain") != "terrain"
            or metadata.get("clock") != "domain_actor_updates"
            or metadata.get("revision_migration") != expected):
        raise ValueError("Terrain runout requires the existing v18 terrain actor account")
    config["terrain_route_runout"] = dict(SETTINGS,
        enabled=not bool(config.get("evaluation_exact_cases")))
    # Keep the primary v18 binding for inherited clocks, ABI and fixed exams.
    metadata.update(worker_revision_id="v6-task-campaign-terrain-worker-v22",
                    terrain_runout_revision_migration=revision_binding())


def revised_design(root=ROOT):
    from .terrain_runout_runtime import runtime_expectation

    inherited = _source_worker(root)
    design = read_bound(SOURCE_DESIGN_FILE, SOURCE_DESIGN_SHA, root)
    design["reviewed_change"] = SCOPE
    design["inherited_terrain_runout_design_sha256"] = SOURCE_DESIGN_SHA
    for role, row in design["stages"]["terrain"]["roles"].items():
        config = row["contract"]
        if bool(config.get("evaluation_exact_cases")) != (role != "train"):
            raise ValueError("Terrain runout source design role and exact exam disagree")
        apply_revision(config)
        # A design contract is an independently pinned expectation subset, not
        # the complete materialized worker (for example it omits long layout).
        # Fill that context from the immutable old worker, never a new caller.
        effective = deepcopy(inherited)
        effective.update(config)
        runtime = runtime_expectation(effective, num_envs=row["invocation"]["num_envs"])
        row["runtime"]["modules"]["terrain_route_runout"] = runtime["enabled"]
        row["runtime"]["terrain_route_runout"] = runtime
    return design


def canonical_worker(domain="terrain", root=ROOT):
    if domain != "terrain":
        raise ValueError("Terrain runout cannot change another task worker")
    root = Path(root)
    if not _same_json(json.loads((root / DESIGN_FILE).read_text()), revised_design(root)):
        raise ValueError("Terrain runout design differs from its full inherited recipe")
    worker = _source_worker(root)
    apply_revision(worker)
    if validate_runout(worker) is None:
        raise ValueError("Terrain runout training requires active separated support geometry")
    worker["design_preflight"] = dict(file=DESIGN_FILE,
        sha256=digest(root / DESIGN_FILE), stage="terrain")
    return worker


def own_domain_source_binding(source, root=ROOT):
    """Authenticate a real own-domain source, without inventing v18 weights.

    The complete inherited v18 contract is SHA-bound, and its unchanged primary
    binding names the exact v15 source. The campaign registry must additionally
    call its existing full v15-to-v18 recipe validator for that inheritance;
    the returned path label identifies that chain and does not replace it.
    """
    inherited = _source_worker(root)
    earlier = read_bound(OWN_SOURCE_WORKER_FILE, OWN_SOURCE_WORKER_SHA, root)
    if _same_json(source, inherited):
        file, sha, path = SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, [WORKER_SCHEMA]
    elif _same_json(source, earlier):
        file, sha = OWN_SOURCE_WORKER_FILE, OWN_SOURCE_WORKER_SHA
        path = ["task-worker-revision-v18", WORKER_SCHEMA]
    else:
        raise ValueError("Terrain runout requires an exact v18 or authenticated real v15 own-domain source")
    return dict(source_contract_file=file, source_contract_sha256=sha,
                inherited_contract_file=SOURCE_WORKER_FILE,
                inherited_contract_sha256=SOURCE_WORKER_SHA,
                inheritance_path=path, critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
                actor_critic_optimizer="preserved", task_clock="restored",
                consumption="preserved_no_refund")


def validate_worker_revision(source, target, root=ROOT):
    binding = own_domain_source_binding(source, root)
    if not _same_json(target, canonical_worker(root=root)):
        raise ValueError("Terrain runout changes fields outside its exact own-domain recipe")
    return binding


def validate_frozen_workers(root=ROOT):
    root = Path(root)
    source = _source_plan(root)
    for row in source["domains"]:
        worker = read_bound(row["contract"], row["sha256"], root)
        design = worker["design_preflight"]
        read_bound(design["file"], design["sha256"], root)
    if not _same_json(json.loads((root / WORKER_FILE).read_text()), canonical_worker(root=root)):
        raise ValueError("Terrain runout frozen worker changed undeclared fields")
    return dict(domain="terrain", source_file=SOURCE_WORKER_FILE,
                source_sha256=SOURCE_WORKER_SHA, target_file=WORKER_FILE,
                target_sha256=digest(root / WORKER_FILE), design_file=DESIGN_FILE,
                design_sha256=digest(root / DESIGN_FILE))


def revised_plan(root=ROOT):
    root = Path(root)
    plan = _source_plan(root)
    plan["contract_id"] = "v6-task-campaign-v22"
    row = next(row for row in plan["domains"] if row["name"] == "terrain")
    row.update(contract=WORKER_FILE, sha256=digest(root / WORKER_FILE))
    plan["revision_migration"].update(manifest_file=MANIFEST_FILE,
                                      source_plan_sha256=SOURCE_PLAN_SHA)
    return plan


def revised_manifest(root=ROOT):
    root = Path(root)
    source, target = _source_plan(root), revised_plan(root)
    raw = (json.dumps(target, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return dict(schema=SCHEMA, source_plan_file=SOURCE_PLAN_FILE,
        source_plan_sha256=SOURCE_PLAN_SHA, target_plan_file=PLAN_FILE,
        target_plan_sha256=hashlib.sha256(raw).hexdigest(),
        run_directory=target["revision_migration"]["run_directory"], geometry_scope=SCOPE,
        terrain_route_runout=deepcopy(SETTINGS),
        critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
        learning_state="actor_critic_Adam_std_preserved",
        task_clock="restored", consumption="preserved_no_refund",
        domain_reopening="preserve_existing_no_new_reopening",
        own_domain_predecessors=[dict(file=file, sha256=sha) for file, sha in (
            (SOURCE_WORKER_FILE, SOURCE_WORKER_SHA),
            (OWN_SOURCE_WORKER_FILE, OWN_SOURCE_WORKER_SHA))],
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
                      target_file=b["contract"], target_sha256=b["sha256"])
                 for a, b in zip(source["domains"], target["domains"], strict=True)],
        designs=[dict(domain="terrain", file=DESIGN_FILE, sha256=digest(root / DESIGN_FILE))])


def freeze_contracts(root=ROOT):
    """Exclusively create four new contracts; no launch or account mutation."""
    root = Path(root)
    paths = (DESIGN_FILE, WORKER_FILE, PLAN_FILE, MANIFEST_FILE)
    if any((root / name).exists() for name in paths):
        raise ValueError("Terrain runout cannot replace existing frozen contracts")
    # Authenticate all unchanged inputs before creating the first output.
    source = _source_plan(root)
    for row in source["domains"]:
        worker = read_bound(row["contract"], row["sha256"], root)
        design = worker["design_preflight"]
        read_bound(design["file"], design["sha256"], root)
    read_bound(OWN_SOURCE_WORKER_FILE, OWN_SOURCE_WORKER_SHA, root)
    design = revised_design(root)
    for name, make in ((DESIGN_FILE, lambda root: design), (WORKER_FILE, canonical_worker),
                       (PLAN_FILE, revised_plan), (MANIFEST_FILE, revised_manifest)):
        value = make(root=root)
        with (root / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    validate_frozen_workers(root)
    return dict(schema=SCHEMA, output_sha256={name: digest(root / name) for name in paths},
                critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
                learning_source="latest_authenticated_own_domain_seal",
                consumption="preserved_no_refund", qualification=None,
                deployment=False, native_runtime_verified=False)
