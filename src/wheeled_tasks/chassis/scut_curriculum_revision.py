"""Canonical v15-to-v17 curriculum recipes and append-only branch reopening."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .nonstandard_start_revision import ROOT, read_bound

SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v15.json"
SOURCE_PLAN_SHA = "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61"
PLAN_FILE = "contracts/v6_task_campaign_v17.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v17.json"
SCHEMA = "task-plan-revision-v17"
SCOPE = "scut_style_native_recovery_prefix_static_terrain_rows_and_permanent_jump_rehearsal"
DOMAINS_TO_REVISE = ("recovery", "terrain", "jump", "stepjump")
ORIGINS = dict(recovery=2294, terrain=2186)
ENDPOINTS = dict(flat=13000, recovery=15000, speed=5063, terrain=15000,
                 jump=15000, stepjump=15000)
CRITIC_ADAPTATION_UPDATES = 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recipe(domain):
    if domain == "recovery":
        from . import recovery_prefix_revision as module
        return (module.SOURCE_WORKER_FILE, module.SOURCE_WORKER_SHA, module.WORKER_FILE,
                module.DESIGN_FILE, module.apply_prefix_revision, module.revised_design,
                module.plan_sampling_pools)
    if domain == "terrain":
        from . import scut_terrain_revision as module
        return (module.SOURCE_WORKER_FILE, module.SOURCE_WORKER_SHA, module.WORKER_FILE,
                module.DESIGN_FILE, module.apply_revision, module.revised_design,
                module.plan_sampling_pools)
    if domain in ("jump", "stepjump"):
        from . import scut_jump_revision as module
        file, sha = module.SOURCES[domain]
        return (file, sha, module.worker_file(domain), module.design_file(domain),
                module.apply_revision, lambda root: module.revised_design(domain, root),
                lambda root: module.plan_sampling_pools(domain, root))
    raise ValueError("SCUT curriculum revises only four named task workers")


def canonical_worker(domain, root=ROOT):
    """Regenerate the whole object from a SHA-bound old worker and old design."""
    root = Path(root)
    source_file, source_sha, _, design_file, apply, design_recipe, _ = _recipe(domain)
    design = design_recipe(root)
    stored_design = json.loads((root / design_file).read_text())
    if stored_design != design:
        raise ValueError("SCUT curriculum independent design differs: " + domain)
    worker = read_bound(source_file, source_sha, root)
    apply(worker, root)
    worker["design_preflight"] = dict(file=design_file, sha256=digest(root / design_file), stage=domain)
    return worker


def validate_frozen_workers(root=ROOT):
    """File hashes alone cannot authorize a reward, clock or physics mutation."""
    root = Path(root)
    receipts = []
    for domain in DOMAINS_TO_REVISE:
        source_file, source_sha, worker_file, design_file, *_ = _recipe(domain)
        if json.loads((root / worker_file).read_text()) != canonical_worker(domain, root):
            raise ValueError("SCUT curriculum worker differs from its exact recipe: " + domain)
        receipts.append(dict(domain=domain, source_file=source_file, source_sha256=source_sha,
            target_file=worker_file, target_sha256=digest(root / worker_file),
            design_file=design_file, design_sha256=digest(root / design_file)))
    return receipts


def revised_plan(root=ROOT):
    root = Path(root)
    plan = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    plan["contract_id"] = "v6-task-campaign-v17"
    plan["domain_completion"]["schedule"]["end_actor_updates"] = deepcopy(ENDPOINTS)
    for row in plan["domains"]:
        domain = row["name"]
        if domain in DOMAINS_TO_REVISE:
            _, _, file, _, _, _, pools = _recipe(domain)
            row.update(contract=file, sha256=digest(root / file), sampling_pools=pools(root))
    plan["revision_migration"].update(manifest_file=MANIFEST_FILE, source_plan_sha256=SOURCE_PLAN_SHA)
    return plan


def revised_manifest(root=ROOT):
    """Pin all six workers and all four changed independent design tables."""
    root = Path(root)
    source = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    target = revised_plan(root)
    target_raw = (json.dumps(target, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return dict(schema=SCHEMA, source_plan_file=SOURCE_PLAN_FILE, source_plan_sha256=SOURCE_PLAN_SHA,
        target_plan_file=PLAN_FILE, target_plan_sha256=hashlib.sha256(target_raw).hexdigest(),
        run_directory=target["revision_migration"]["run_directory"], curriculum_scope=SCOPE,
        reopen_actor_origins=deepcopy(ORIGINS),
        reopen_actor_endpoints={name: ENDPOINTS[name] for name in ORIGINS},
        critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)],
        designs=[dict(domain=domain, file=_recipe(domain)[3], sha256=digest(root / _recipe(domain)[3]))
                 for domain in DOMAINS_TO_REVISE])


def _latest_sealed(branch):
    candidates = []
    for path in branch.glob("attempt_*/train/**/*.checkpoint.json"):
        if any(part.endswith(".pending") for part in path.parts):
            continue
        meta = json.loads(path.read_text())
        count = meta.get("learning_updates")
        if type(count) is not int or count < 0:
            raise ValueError("SCUT reopening snapshot has an invalid learning clock")
        checkpoint = path.with_suffix("").with_suffix(".pt").resolve()
        candidates.append((count, checkpoint.name == "model_final.pt", checkpoint))
    if not candidates:
        raise ValueError("SCUT reopening has no immutable own-domain learning state")
    return max(candidates, key=lambda row: row[:2])[2]


def reopen_completed_domains(report, records, source_plan, target_plan, root):
    """Atomically reopen two latest own-domain states without replacing history.

    The run-directory root owns checkpoints. Canonical frozen contract files
    come from task_campaign.ROOT, matching the existing authenticated dispatcher.
    New critic adaptation is charged; it does not advance the actor clock.
    """
    from .full_curriculum import checkpoint_contract_path, checkpoint_update_count
    from .task_campaign import ROOT as contract_root, account_updates

    if (source_plan != read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, contract_root)
            or target_plan != revised_plan(contract_root)):
        raise ValueError("SCUT reopening requires the exact v15 to v17 plan pair")
    validate_frozen_workers(contract_root)
    account = account_updates(target_plan, records)
    revised = deepcopy(report)
    tokens = list(revised.get("consumed_review_tokens", []))
    reopenings = {}
    additional = 0
    for domain, origin in ORIGINS.items():
        source_file, source_sha, *_ = _recipe(domain)
        source_worker = read_bound(source_file, source_sha, contract_root)
        current = revised.get("domains", {}).get(domain)
        if not isinstance(current, dict):
            raise ValueError("SCUT reopening requires both completed own-domain branches")
        already = current.get("reopening", {}).get("schema") == "task-domain-reopen-v17"
        if already:
            history = current.get("completion_history", [])
            if not history or history[-1].get("plan_revision") != "v15":
                raise ValueError("SCUT reopening lost its completed v15 history")
            predecessor = history[-1].get("record", {})
        else:
            predecessor = current
        if (predecessor.get("status") not in (
                "domain_schedule_finished_unassessed", "domain_review_finished_unassessed")
                or predecessor.get("domain_qualified") is not None
                or not predecessor.get("candidate_checkpoint")):
            raise ValueError("SCUT reopening accepts only completed unassessed own-domain branches")
        checkpoint = Path(predecessor["candidate_checkpoint"]).resolve(strict=True)
        branch = Path(root).resolve() / domain
        if not checkpoint.is_relative_to(branch):
            raise ValueError("SCUT reopening cannot change its own-domain learning source")
        count = checkpoint_update_count(checkpoint)
        seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
        if (seal.get("task_actor_update") != origin or type(seal.get("task_actor_update")) is not int
                or seal.get("task_campaign") != source_worker["task_campaign"]
                or seal.get("contract_sha256") != source_sha
                or digest(checkpoint) != predecessor.get("candidate_checkpoint_sha256")
                or json.loads(checkpoint_contract_path(checkpoint).read_text()) != source_worker
                or count > account["consumed_updates"]
                or type(seal.get("actor_updates_consumed")) is not int
                or not 0 <= seal["actor_updates_consumed"] <= account["actor_updates_consumed"]):
            raise ValueError("SCUT reopening must restore the completed exact actor and contract")
        added = ENDPOINTS[domain] - origin
        reopening = dict(schema="task-domain-reopen-v17", source_plan_sha256=SOURCE_PLAN_SHA,
            source_contract_sha256=source_sha, source_checkpoint=str(checkpoint),
            source_checkpoint_sha256=digest(checkpoint), origin_actor_update=origin,
            end_actor_update=ENDPOINTS[domain], actor_updates_added=added,
            critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
            consumption="preserved_no_refund", outcome="unassessed_research_candidate")
        if already:
            if current["reopening"] != reopening:
                raise ValueError("SCUT reopening no longer authenticates its completed source")
        else:
            if _latest_sealed(branch) != checkpoint:
                raise ValueError("SCUT reopening cannot rewind a newer own-domain sealed state")
            charge = added + CRITIC_ADAPTATION_UPDATES
            if account["domain_consumed_updates"][domain] + charge > 20000:
                raise ValueError("SCUT reopening cannot exceed the charged domain ceiling")
            additional += charge
            history = deepcopy(current.get("completion_history", []))
            history.append(dict(plan_revision="v15", record=deepcopy(current)))
            for key in ("finish_review", "pending_review"):
                token = current.get(key, {}).get("token")
                if token and token not in tokens:
                    tokens.append(token)
            current.update(status="domain_reopened_unassessed", completion_history=history,
                reopening=reopening, domain_qualified=None, assessment_status="not_run",
                accepted_stage=None, deployment_checkpoint=None)
            for key in ("finish_review", "pending_review", "schedule_completion"):
                current.pop(key, None)
        reopenings[domain] = deepcopy(reopening)
    if account["consumed_updates"] + additional > 121000:
        raise ValueError("SCUT reopening cannot exceed the charged campaign ceiling")
    revised["consumed_review_tokens"] = tokens
    report.clear()
    report.update(revised)
    return dict(schema="task-domain-reopening-v17", domains=deepcopy(ORIGINS),
        end_actor_updates={name: ENDPOINTS[name] for name in ORIGINS}, reopening=reopenings,
        consumption="preserved_no_refund", prior_results="append_only_completion_history")
