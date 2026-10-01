"""Frozen task-domain research with one non-refundable campaign account."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

REVISION = "v6-task-campaign-v1"
CAMPAIGN_ID = "v6-task-domains-20261001"
DOMAINS = ("flat", "recovery", "speed", "terrain", "jump", "stepjump")
ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan):
    if (plan.get("contract_id") != REVISION or plan.get("campaign_id") != CAMPAIGN_ID
            or plan.get("historical_consumed_updates") != 1000
            or plan.get("historical_actor_consumed_updates") != 980
            or plan.get("total_ceiling") != 121000 or plan.get("target_num_envs") != 20000
            or plan.get("seed") != 290927 or plan.get("evaluation_seeds") != [290929, 290930]
            or plan.get("rollback_refunds_consumed_updates", False) is not False
            or plan.get("automatic_promotion", False) is not False):
        raise ValueError("Task campaign changes its frozen budget, identity or research boundary")
    domains = plan.get("domains", [])
    if tuple(row["name"] for row in domains) != DOMAINS:
        raise ValueError("Task campaign domain order differs")
    seen = {"A"}
    for row in domains:
        if (row["updates"] != 20000 or row["parent_domain"] not in seen
                or not row["required_case_names"]
                or len(set(row["required_case_names"])) != len(row["required_case_names"])):
            raise ValueError("Each domain needs a 20000-update cap, earlier parent and explicit unique gates")
        seen.add(row["name"])
    if plan["historical_consumed_updates"] + sum(r["updates"] for r in domains) != plan["total_ceiling"]:
        raise ValueError("Task caps disagree with global consumption ceiling")


def validate_task_resume(source, target):
    meta = target.get("task_campaign", {})
    if (meta.get("campaign_id") != CAMPAIGN_ID or meta.get("domain") not in DOMAINS
            or meta.get("updates") != 20000 or meta.get("clock") != "domain_actor_updates"):
        raise ValueError("Missing frozen task migration identity")
    same_domain = source.get("task_campaign", {}).get("domain") == meta["domain"]
    if same_domain:
        if source != target:
            raise ValueError("Same-domain recovery cannot change its contract")
    else:
        path = ROOT / meta["source_contract_file"]
        if digest(path) != meta["source_contract_sha256"] or json.loads(path.read_text()) != source:
            raise ValueError("Task learning origin differs from its frozen source contract")
    # Curriculum/reward domains may differ. The physical, actuator and neural
    # interface remains identical; accepted capability is never inferred here.
    for key in ("asset_manifest_sha256", "control_math_sha256", "physics_dt", "policy_dt",
                "actor_dim", "critic_dim", "action_dim", "history_length", "policy_architecture",
                "v5_control", "actuator_response", "actuator_response_evidence", "policy_action_order",
                "actor_layout", "critic_layout", "actor_observation_source"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Task migration changes physical/policy identity: {key}")
    if (target.get("physics_dt"), target.get("policy_dt"), target.get("actor_dim"),
            target.get("critic_dim"), target.get("target_num_envs")) != (.001, .02, 35, 81, 20000):
        raise ValueError("Task campaign requires 20k, 35/81 and 50Hz/1kHz")
    if target.get("learning_rate") != 1e-4 or target.get("learning_rate_schedule") != "fixed":
        raise ValueError("Task campaign fixed learning rate changed")
    return dict(kind="task_domain_recovery" if same_domain else "task_domain_migration",
                actor_critic_optimizer="preserved", environment_state="new_episodes",
                task_clock="restored" if same_domain else "new_domain_zero",
                accepted_stage=None, hardware_release=False)


def validate_worker(config, args):
    from .full_curriculum import checkpoint_contract_path, checkpoint_update_count
    meta = config["task_campaign"]
    if (not args.resume_task_change or not args.resume or args.num_envs != 20000
            or args.seed != 290927 or args.learning_rate_scale != 1.
            or args.global_actor_update_offset != 0 or args.block_index != 0
            or type(args.updates) is not int or not 0 < args.updates <= 20000
            or type(args.consumed_updates) is not int or args.consumed_updates < 1000
            or args.consumed_updates + args.updates > 121000):
        raise ValueError("Task worker changes its invocation or campaign budget")
    checkpoint_update_count(args.resume)
    source = json.loads(checkpoint_contract_path(args.resume).read_text())
    validate_task_resume(source, config)
    seal = json.loads(Path(args.resume).with_suffix(".checkpoint.json").read_text())
    same = source.get("task_campaign", {}).get("domain") == meta["domain"]
    task_clock = seal.get("task_actor_update") if same else 0
    if (type(task_clock) is not int or task_clock < 0 or args.task_actor_update != task_clock
            or args.stage_actor_update != seal["stage_actor_update"]
            or args.actor_updates_consumed < seal["actor_updates_consumed"]
            or task_clock + args.updates > 20000):
        raise ValueError("Task worker changes its task/lineage clock or consumption")


def account_updates(plan, attempts):
    """Every attempt remains charged, including updates absent from a snapshot."""
    caps = {row["name"]: row["updates"] for row in plan["domains"]}
    by_domain = dict.fromkeys(caps, 0)
    actor = plan["historical_actor_consumed_updates"]
    for record in attempts.values():
        count, actor_count = record["consumed_updates"], record.get("actor_updates", 0)
        domain = record["domain"]
        if (domain not in caps or type(count) is not int or type(actor_count) is not int
                or count < 0 or not 0 <= actor_count <= count):
            raise ValueError("Invalid charged task attempt")
        by_domain[domain] += count
        actor += actor_count
    if any(by_domain[name] > caps[name] for name in caps):
        raise ValueError("Task budget exhausted")
    total = plan["historical_consumed_updates"] + sum(by_domain.values())
    if total > plan["total_ceiling"]:
        raise ValueError("Campaign budget exhausted")
    return dict(consumed_updates=total, actor_updates_consumed=actor,
                domain_consumed_updates=by_domain, remaining_updates=plan["total_ceiling"] - total)


def update_attempt(plan, attempts, name, domain, record):
    result = deepcopy(attempts)
    old = result.get(name, dict(domain=domain, consumed_updates=0, actor_updates=0))
    completed = int(record.get("successful_updates", 0))
    # A worker may publish an update reservation before optimization. Retain
    # such work after an abrupt exit rather than refunding an uncertain update.
    count = max(completed, int(record.get("attempted_updates", completed)))
    actors = int(record.get("actor_updates_in_block", 0))
    if old["domain"] != domain or count < old["consumed_updates"] or actors < old["actor_updates"]:
        raise ValueError("Task consumption cannot move backwards")
    result[name] = dict(domain=domain, consumed_updates=count, actor_updates=actors)
    account_updates(plan, result)
    return result


def assess_domain(results, required):
    if len(results) != 2 or any(set(required) - set(r["cases"]) for r in results):
        raise ValueError("invalid_evaluation_coverage")
    failed = sorted(n for n in required if not all(r["cases"][n]["passed"] for r in results))
    return dict(domain_qualified=not failed, required_failed=failed,
                accepted_stage=None, deployment_checkpoint=None,
                scope="individual_simulation_task_domain_not_integrated_policy")


def validate_task_budget(config, root):
    path = Path(root) / config["budget_spec_file"]
    if digest(path) != config["budget_spec_sha256"]:
        raise ValueError("Task budget specification SHA differs")
    budget = json.loads(path.read_text())
    domain = config["task_campaign"]["domain"]
    expected = dict.fromkeys(DOMAINS, 20000)
    if (budget.get("schema") != "task-domain-budget-v1"
            or budget.get("campaign_id") != CAMPAIGN_ID
            or budget.get("historical_consumed_updates") != 1000
            or budget.get("historical_actor_consumed_updates") != 980
            or budget.get("total_ceiling") != 121000 or budget.get("domains") != expected
            or budget.get("rollback_refunds_consumed_updates") is not False
            or domain not in expected or config.get("budget_stage_id") != domain
            or config.get("curriculum_stage") != domain or config.get("total_updates") != 20000
            or sum(stage["updates"] for stage in config["stages"]) != 20000):
        raise ValueError("Task-domain budget must preserve 1000 historical plus six 20000-update caps")
    return dict(schema=budget["schema"], campaign_id=CAMPAIGN_ID, domain=domain,
                domain_update_ceiling=20000, historical_consumed_updates=1000,
                total_ceiling=121000, budget_spec_sha256=digest(path), rollback_refunds_consumed_updates=False)
