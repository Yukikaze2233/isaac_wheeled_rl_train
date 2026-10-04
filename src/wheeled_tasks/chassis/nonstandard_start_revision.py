"""Named v13 restart teaching for completed ordinary flat/terrain branches."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v12.json"
SOURCE_PLAN_SHA = "e79e94872f041ddf935ad961c59093b827b0b50f3519c9234718118cf1b59298"
PLAN_FILE = "contracts/v6_task_campaign_v13.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v13.json"
SCHEMA = "task-worker-revision-v13"
SCOPE = "ordinary_nonstandard_pose_reset_200ms_zero_torque_release_and_5pct_command_release_stop_teaching"
ORIGINS = dict(speed=3563, terrain=1186)
ENDPOINTS = dict(flat=13000, recovery=15000, speed=5063, terrain=2186, jump=15000, stepjump=15000)
SOURCES = dict(
    speed=("contracts/v6_task_campaign_speed_worker_v10.json",
        "37837b286de10a66404ef155ae84ead613a90f76375d4abe6874a042eae6b406"),
    terrain=("contracts/v6_task_campaign_terrain_worker_v6.json",
        "4337842f9689ec2a4ecd0994dfb39ef821c95a9868a91f445de52f2a0afc456e"))


def worker_file(domain):
    if domain not in SOURCES:
        raise ValueError("Nonstandard start revision is limited to speed and terrain")
    return f"contracts/v6_task_campaign_{domain}_worker_v13.json"


def design_file(domain):
    worker_file(domain)
    return f"contracts/v6_task_campaign_{domain}_expectations_v13.json"


def read_bound(name, sha, root=ROOT):
    raw = (Path(root) / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"Nonstandard start frozen predecessor SHA differs: {name}")
    return json.loads(raw)


def revision_binding(domain):
    file, sha = SOURCES[domain]
    return dict(schema=SCHEMA, source_contract_file=file, source_contract_sha256=sha,
        origin_actor_update=ORIGINS[domain])


def apply_start_revision(config):
    from .nonstandard_start import nonstandard_start_settings, validate_nonstandard_start
    from .release_stop import apply_release_stop_contract
    domain = config.get("task_campaign", {}).get("domain")
    if domain not in SOURCES or config["task_campaign"]["clock"] != "domain_actor_updates":
        raise ValueError("Nonstandard start requires an unchanged ordinary task domain and actor clock")
    settings = nonstandard_start_settings(ORIGINS[domain], enabled=not config.get("evaluation_exact_cases", False))
    validate_nonstandard_start(settings)
    config["nonstandard_start"] = settings
    config["task_campaign"]["revision_migration"] = revision_binding(domain)
    apply_release_stop_contract(config, domain)


def plan_sampling_pools(domain, root=ROOT):
    source = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    pools = deepcopy(next(row for row in source["domains"] if row["name"] == domain)["sampling_pools"])
    for row in pools:
        row["fraction"] *= .95
    config = read_bound(*SOURCES[domain], root)
    apply_start_revision(config)
    names = [row["name"] for row in config["scene_groups"]
        if config["behavior_pool_membership"][row["name"]] == "release_stop"]
    pools.append(dict(name="release_stop", fraction=.05, skills=names, cross_height=False))
    return pools


def runtime_expectation(config):
    from .nonstandard_start import runtime_expectation as observe_start
    return observe_start(config["nonstandard_start"], config["policy_dt"])


def release_stop_runtime_expectation(config):
    from .release_stop import release_stop_runtime_expectation as observe_release
    return observe_release(config["release_stop"], config["policy_dt"])


def revised_design(domain, root=ROOT):
    source = read_bound(*SOURCES[domain], root)
    binding = source["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_nonstandard_start_design_sha256"] = binding["sha256"]
    for row in design["stages"][domain]["roles"].values():
        apply_start_revision(row["contract"])
        row["runtime"]["modules"]["nonstandard_start"] = row["contract"]["nonstandard_start"]["enabled"]
        row["runtime"]["nonstandard_start"] = runtime_expectation(row["contract"])
        row["runtime"]["modules"]["release_stop"] = row["contract"]["release_stop"]["enabled"]
        row["runtime"]["release_stop"] = release_stop_runtime_expectation(row["contract"])
    return design


def reopen_completed_domains(report, records, source_plan, target_plan, root):
    """Reopen exactly two authenticated completed branches, once and without refund."""
    from .full_curriculum import checkpoint_contract_path, checkpoint_update_count
    from .task_campaign import account_updates
    pair = (source_plan["contract_id"], target_plan["contract_id"])
    if pair not in (("v6-task-campaign-v12", "v6-task-campaign-v13"),
                    ("v6-task-campaign-v12", "v6-task-campaign-v14")):
        raise ValueError("Ordinary restart accounting requires its exact deployed v12 predecessor")
    revision = target_plan["contract_id"].rsplit("-", 1)[1]
    account = account_updates(target_plan, records)
    if account["consumed_updates"] + sum(ENDPOINTS[name] - value + 10 for name, value in ORIGINS.items()) > 121000:
        raise ValueError("Nonstandard restart cannot exceed the existing total campaign ceiling")
    revised_report = deepcopy(report)
    consumed_tokens = list(revised_report.get("consumed_review_tokens", []))
    for domain, origin in ORIGINS.items():
        record = revised_report["domains"].get(domain)
        if not isinstance(record, dict):
            raise ValueError("Nonstandard restart requires both completed ordinary branches")
        reopening = dict(schema="task-domain-reopen-" + revision, source_plan_sha256=SOURCE_PLAN_SHA,
            source_contract_sha256=SOURCES[domain][1], origin_actor_update=origin,
            end_actor_update=ENDPOINTS[domain], actor_updates_added=ENDPOINTS[domain] - origin,
            consumption="preserved_no_refund", outcome="unassessed_research_candidate")
        if record.get("reopening") == reopening:
            if not record.get("completion_history"):
                raise ValueError("Reopened branch lost its completed predecessor history")
            predecessor = record["completion_history"][-1]
            if (predecessor.get("plan_revision") != "v12"
                    or predecessor.get("record", {}).get("candidate_checkpoint") != record.get("candidate_checkpoint")):
                raise ValueError("Reopened branch no longer authenticates its completed predecessor")
            record = predecessor["record"]
            already_reopened = True
        else:
            already_reopened = False
        if (record.get("status") not in ("domain_schedule_finished_unassessed", "domain_review_finished_unassessed")
                or record.get("domain_qualified") is not None or not record.get("candidate_checkpoint")):
            raise ValueError("Only completed unassessed ordinary branches can be reopened")
        checkpoint = Path(record["candidate_checkpoint"]).resolve(strict=True)
        if not checkpoint.is_relative_to(Path(root).resolve() / domain):
            raise ValueError("Nonstandard restart cannot change its own-domain learning source")
        checkpoint_update_count(checkpoint)
        seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
        if (seal.get("task_actor_update") != origin
                or seal.get("contract_sha256") != SOURCES[domain][1]
                or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record.get("candidate_checkpoint_sha256")
                or read_bound(*SOURCES[domain]) != json.loads(checkpoint_contract_path(checkpoint).read_text())):
            raise ValueError("Nonstandard restart must resume its exact completed actor and contract")
        if account["domain_consumed_updates"][domain] + ENDPOINTS[domain] - origin + 10 > 20000:
            raise ValueError("Nonstandard restart cannot exceed its existing charged domain ceiling")
        if already_reopened:
            continue
        previous = deepcopy(record)
        history = deepcopy(record.get("completion_history", []))
        history.append(dict(plan_revision="v12", record=previous))
        for key in ("finish_review", "pending_review"):
            token = record.get(key, {}).get("token")
            if token and token not in consumed_tokens:
                consumed_tokens.append(token)
        record.update(status="domain_reopened_unassessed", completion_history=history, reopening=reopening,
            domain_qualified=None, assessment_status="not_run", accepted_stage=None, deployment_checkpoint=None)
        # These immutable results remain in completion_history. They no longer
        # govern the newly authorized teaching window.
        for key in ("finish_review", "pending_review", "schedule_completion"):
            record.pop(key, None)
    revised_report["consumed_review_tokens"] = consumed_tokens
    report.clear()
    report.update(revised_report)
    return dict(schema="task-domain-reopening-" + revision, domains=deepcopy(ORIGINS),
        end_actor_updates={name: ENDPOINTS[name] for name in ORIGINS},
        consumption="preserved_no_refund", prior_results="append_only_completion_history")
