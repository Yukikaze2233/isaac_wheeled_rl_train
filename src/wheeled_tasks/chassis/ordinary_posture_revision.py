"""Direct v15 continuation with ordinary tilt births and gentle push lessons.

The unlaunched v17 curriculum is inherited as a canonical authoring recipe,
never as a learning checkpoint or a refunded campaign predecessor.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import scut_curriculum_revision as previous
from .nonstandard_start_revision import ROOT, read_bound

SOURCE_PLAN_FILE = previous.SOURCE_PLAN_FILE
SOURCE_PLAN_SHA = previous.SOURCE_PLAN_SHA
PLAN_FILE = "contracts/v6_task_campaign_v18.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v18.json"
SCHEMA = "task-plan-revision-v18"
WORKER_SCHEMA = "task-worker-revision-v18"
SCOPE = "ordinary_small_root_tilt_and_native_gentle_push_with_canonical_v17_task_repairs"
DOMAINS_TO_REVISE = ("recovery", "speed", "terrain", "jump", "stepjump")
ORIGINS = dict(recovery=2294, speed=5063, terrain=2186)
TEACHING_ORIGINS = dict(ORIGINS, jump=5552, stepjump=0)
ENDPOINTS = dict(previous.ENDPOINTS, speed=15000)
CRITIC_ADAPTATION_UPDATES = 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_file(domain):
    if domain not in DOMAINS_TO_REVISE:
        raise ValueError("Ordinary posture revision preserves the sealed flat worker")
    return f"contracts/v6_task_campaign_{domain}_worker_v18.json"


def design_file(domain):
    worker_file(domain)
    return f"contracts/v6_task_campaign_{domain}_expectations_v18.json"


def source_binding(domain, root=ROOT):
    plan = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    row = next(row for row in plan["domains"] if row["name"] == domain)
    worker_file(domain)
    return row["contract"], row["sha256"]


def apply_revision(config, root=ROOT):
    from .ordinary_posture_curriculum import posture_settings

    meta = config.get("task_campaign", {})
    domain = meta.get("domain")
    source_file, source_sha = source_binding(domain, root)
    if meta.get("clock") != "domain_actor_updates":
        raise ValueError("Ordinary posture lessons require the independent actor clock")
    config["ordinary_posture_curriculum"] = posture_settings(
        TEACHING_ORIGINS[domain], enabled=not bool(config.get("evaluation_exact_cases")))
    meta["worker_revision_id"] = f"v6-task-campaign-{domain}-worker-v18"
    meta["revision_migration"] = dict(schema=WORKER_SCHEMA,
        source_contract_file=source_file, source_contract_sha256=source_sha)
    if domain == "stepjump":
        parent = worker_file("jump")
        meta.update(source_contract_file=parent, source_contract_sha256=digest(Path(root) / parent))


def _base_worker(domain, root):
    if domain == "speed":
        return read_bound(*source_binding(domain, root), root)
    return previous.canonical_worker(domain, root)


def revised_design(domain, root=ROOT):
    from .ordinary_posture_curriculum import runtime_expectation

    base = _base_worker(domain, root)
    binding = base["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_task_repair_design_sha256"] = binding["sha256"]
    for row in design["stages"][domain]["roles"].values():
        config, runtime = row["contract"], row["runtime"]
        apply_revision(config, root)
        settings = config["ordinary_posture_curriculum"]
        runtime["modules"]["ordinary_posture_curriculum"] = settings["enabled"]
        runtime["ordinary_posture_curriculum"] = runtime_expectation(settings)
    return design


def canonical_worker(domain, root=ROOT):
    root = Path(root)
    design = revised_design(domain, root)
    file = design_file(domain)
    if json.loads((root / file).read_text()) != design:
        raise ValueError("Ordinary posture independent design differs: " + domain)
    worker = _base_worker(domain, root)
    apply_revision(worker, root)
    worker["design_preflight"] = dict(file=file, sha256=digest(root / file), stage=domain)
    return worker


def _recipe(domain):
    source_file, source_sha = source_binding(domain)
    return (source_file, source_sha, worker_file(domain), design_file(domain),
            apply_revision, lambda root: revised_design(domain, root),
            lambda root: plan_sampling_pools(domain, root))


def validate_frozen_workers(root=ROOT):
    root = Path(root)
    previous.validate_frozen_workers(root)
    receipts = []
    for domain in DOMAINS_TO_REVISE:
        file, design = worker_file(domain), design_file(domain)
        if json.loads((root / file).read_text()) != canonical_worker(domain, root):
            raise ValueError("Ordinary posture worker differs from exact recipe: " + domain)
        source_file, source_sha = source_binding(domain, root)
        receipts.append(dict(domain=domain, source_file=source_file, source_sha256=source_sha,
            target_file=file, target_sha256=digest(root / file), design_file=design,
            design_sha256=digest(root / design)))
    return receipts


def plan_sampling_pools(domain, root=ROOT):
    plan = previous.revised_plan(root)
    return deepcopy(next(row["sampling_pools"] for row in plan["domains"] if row["name"] == domain))


def revised_plan(root=ROOT):
    root = Path(root)
    plan = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    plan["contract_id"] = "v6-task-campaign-v18"
    plan["domain_completion"]["schedule"]["end_actor_updates"] = deepcopy(ENDPOINTS)
    for row in plan["domains"]:
        domain = row["name"]
        if domain in DOMAINS_TO_REVISE:
            file = worker_file(domain)
            row.update(contract=file, sha256=digest(root / file), sampling_pools=plan_sampling_pools(domain, root))
    plan["revision_migration"].update(manifest_file=MANIFEST_FILE, source_plan_sha256=SOURCE_PLAN_SHA)
    return plan


def revised_manifest(root=ROOT):
    root = Path(root)
    source, target = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root), revised_plan(root)
    raw = (json.dumps(target, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return dict(schema=SCHEMA, source_plan_file=SOURCE_PLAN_FILE, source_plan_sha256=SOURCE_PLAN_SHA,
        target_plan_file=PLAN_FILE, target_plan_sha256=hashlib.sha256(raw).hexdigest(),
        run_directory=target["revision_migration"]["run_directory"], curriculum_scope=SCOPE,
        reopen_actor_origins=deepcopy(ORIGINS),
        reopen_actor_endpoints={name: ENDPOINTS[name] for name in ORIGINS},
        critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)],
        designs=[dict(domain=d, file=design_file(d), sha256=digest(root / design_file(d))) for d in DOMAINS_TO_REVISE])


def reopen_completed_domains(report, records, source_plan, target_plan, root):
    import sys
    return previous.reopen_completed_domains(report, records, source_plan, target_plan, root,
                                             recipe=sys.modules[__name__])
