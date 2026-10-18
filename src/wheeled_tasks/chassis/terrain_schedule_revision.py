"""Plan-only terrain extension from its completed native v15 learning state."""
from copy import deepcopy
import json
from pathlib import Path

from .nonstandard_start_revision import read_bound

SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v15.json"
SOURCE_PLAN_SHA = "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61"
PLAN_FILE = "contracts/v6_task_campaign_v16.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v16.json"
WORKER_FILE = "contracts/v6_task_campaign_terrain_worker_v15.json"
WORKER_SHA = "843c848d28557907fbe3a842b271d0a7736fdf8f56b97ec0f51ba98591d15a41"
SCOPE = "ordinary_terrain_actor_endpoint_extension_existing_learning_state_no_refund"
ORIGINS = dict(terrain=2186)
ENDPOINTS = dict(flat=13000, recovery=15000, speed=5063, terrain=15000,
                 jump=15000, stepjump=15000)


def revised_plan(root):
    plan = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    plan["contract_id"] = "v6-task-campaign-v16"
    plan["domain_completion"]["schedule"]["end_actor_updates"] = deepcopy(ENDPOINTS)
    plan["revision_migration"].update(manifest_file=MANIFEST_FILE,
                                      source_plan_sha256=SOURCE_PLAN_SHA)
    return plan


def reopen_completed_terrain(report, records, source_plan, target_plan, root):
    """Append one completed branch to history; never replace its learning source."""
    from .full_curriculum import checkpoint_contract_path, checkpoint_update_count
    from .task_campaign import ROOT, account_updates, digest

    if (source_plan != read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT)
            or target_plan != revised_plan(ROOT)):
        raise ValueError("Terrain extension requires the exact v15 to v16 plan pair")
    worker = read_bound(WORKER_FILE, WORKER_SHA, ROOT)
    account = account_updates(target_plan, records)
    revised = deepcopy(report)
    current = revised.get("domains", {}).get("terrain")
    if not isinstance(current, dict):
        raise ValueError("Terrain extension requires its completed ordinary branch")
    already_reopened = current.get("reopening", {}).get("schema") == "task-domain-reopen-v16"
    if already_reopened:
        history = current.get("completion_history", [])
        if not history or history[-1].get("plan_revision") != "v15":
            raise ValueError("Terrain reopening lost its completed v15 history")
        predecessor = history[-1].get("record", {})
    else:
        predecessor = current
    if (predecessor.get("status") not in (
            "domain_schedule_finished_unassessed", "domain_review_finished_unassessed")
            or predecessor.get("domain_qualified") is not None
            or not predecessor.get("candidate_checkpoint")):
        raise ValueError("Only the completed unassessed terrain branch can be reopened")
    checkpoint = Path(predecessor["candidate_checkpoint"]).resolve(strict=True)
    branch = Path(root).resolve() / "terrain"
    if not checkpoint.is_relative_to(branch):
        raise ValueError("Terrain extension cannot change its own-domain learning source")
    count = checkpoint_update_count(checkpoint)
    seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
    if (type(seal.get("task_actor_update")) is not int
            or seal["task_actor_update"] != ORIGINS["terrain"]
            or seal.get("task_campaign") != worker["task_campaign"]
            or seal.get("contract_sha256") != WORKER_SHA
            or digest(checkpoint) != predecessor.get("candidate_checkpoint_sha256")
            or json.loads(checkpoint_contract_path(checkpoint).read_text()) != worker
            or count > account["consumed_updates"]
            or type(seal.get("actor_updates_consumed")) is not int
            or not 0 <= seal["actor_updates_consumed"] <= account["actor_updates_consumed"]):
        raise ValueError("Terrain extension must restore its completed v15 actor and contract")
    added = ENDPOINTS["terrain"] - ORIGINS["terrain"]
    reopening = dict(schema="task-domain-reopen-v16", source_plan_sha256=SOURCE_PLAN_SHA,
        source_contract_sha256=WORKER_SHA, source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=digest(checkpoint), origin_actor_update=ORIGINS["terrain"],
        end_actor_update=ENDPOINTS["terrain"], actor_updates_added=added,
        repeat_critic_adaptation=False, consumption="preserved_no_refund",
        outcome="unassessed_research_candidate")
    receipt = dict(schema="task-domain-reopening-v16", domains=deepcopy(ORIGINS),
        end_actor_updates=dict(terrain=ENDPOINTS["terrain"]), reopening=deepcopy(reopening),
        consumption="preserved_no_refund", prior_results="append_only_completion_history")
    if already_reopened:
        if current["reopening"] != reopening:
            raise ValueError("Terrain reopening no longer authenticates its completed source")
        return receipt

    candidates = []
    for path in branch.glob("attempt_*/train/**/*.checkpoint.json"):
        if any(part.endswith(".pending") for part in path.parts):
            continue
        metadata = json.loads(path.read_text())
        learning = metadata.get("learning_updates")
        if type(learning) is not int or learning < 0:
            raise ValueError("Terrain learning snapshot has an invalid clock")
        candidate = path.with_suffix("").with_suffix(".pt")
        candidates.append((learning, candidate.name == "model_final.pt", candidate.resolve()))
    if not candidates or max(candidates, key=lambda row: row[:2])[2] != checkpoint:
        raise ValueError("Terrain extension cannot rewind a newer own-domain sealed state")
    if (account["domain_consumed_updates"]["terrain"] + added > 20000
            or account["consumed_updates"] + added > 121000):
        raise ValueError("Terrain extension cannot exceed the existing charged ceilings")
    history = deepcopy(current.get("completion_history", []))
    history.append(dict(plan_revision="v15", record=deepcopy(current)))
    tokens = list(revised.get("consumed_review_tokens", []))
    for key in ("finish_review", "pending_review"):
        token = current.get(key, {}).get("token")
        if token and token not in tokens:
            tokens.append(token)
    current.update(status="domain_reopened_unassessed", completion_history=history,
        reopening=reopening, domain_qualified=None, assessment_status="not_run",
        accepted_stage=None, deployment_checkpoint=None)
    for key in ("finish_review", "pending_review", "schedule_completion"):
        current.pop(key, None)
    revised["consumed_review_tokens"] = tokens
    report.clear()
    report.update(revised)
    return receipt
