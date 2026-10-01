"""Frozen task-domain research with one non-refundable campaign account."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

REVISION = "v6-task-campaign-v1"
REVISION_V2 = "v6-task-campaign-v2"
REVISION_V3 = "v6-task-campaign-v3"
REVISION_V4 = "v6-task-campaign-v4"
REVISION_V5 = "v6-task-campaign-v5"
CONTINUOUS_REVISIONS = (REVISION_V2, REVISION_V3, REVISION_V4, REVISION_V5)
CONTINUOUS_MODE = "uninterrupted_cap_then_next"
REVIEW_MODE = "continuous_review_or_cap_then_next"
CAMPAIGN_ID = "v6-task-domains-20261001"
DOMAINS = ("flat", "recovery", "speed", "terrain", "jump", "stepjump")
ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan):
    resized = plan.get("contract_id") == REVISION_V5
    if (plan.get("contract_id") not in (REVISION, *CONTINUOUS_REVISIONS) or plan.get("campaign_id") != CAMPAIGN_ID
            or plan.get("historical_consumed_updates") != 1000
            or plan.get("historical_actor_consumed_updates") != 980
            or plan.get("total_ceiling") != 121000 or plan.get("target_num_envs") != (4096 if resized else 20000)
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
    if plan["contract_id"] in CONTINUOUS_REVISIONS:
        completion = plan.get("domain_completion", {})
        if (completion.get("execution_mode") != (REVIEW_MODE if resized else CONTINUOUS_MODE)
                or completion.get("assessment_interval_actor_updates") != 0
                or completion.get("assessment_mode") != "explicit_offline_only"
                or completion.get("stop_if_all_required_pass_both_seeds") is not False
                or completion.get("cap_result") != "unassessed_research_candidate"):
            raise ValueError("Task v2 must train continuously without an assessment gate")
        if resized and (completion.get("review_interval_actor_updates") != 500
                or completion.get("early_finish") != "explicit_evidence_review"
                or completion.get("review_blocks_training") is not False):
            raise ValueError("Task v5 needs a nonblocking review; 20000 is a ceiling")


def validate_plan_migration(source, target, run_directory):
    """Authenticate the one named revision without trusting mutable run receipts."""
    validate_plan(source)
    validate_plan(target)
    if (source["contract_id"], target["contract_id"]) not in ((REVISION, REVISION_V2), (REVISION_V2, REVISION_V3), (REVISION_V3, REVISION_V4), (REVISION_V4, REVISION_V5)):
        raise ValueError("Only a named consecutive task revision is authorized")
    revision = target["contract_id"].rsplit("-", 1)[1]
    binding = target["revision_migration"]
    manifest = json.loads((ROOT / binding["manifest_file"]).read_text())
    for role, value in (("source", source), ("target", target)):
        path = ROOT / manifest[f"{role}_plan_file"]
        if digest(path) != manifest[f"{role}_plan_sha256"] or json.loads(path.read_text()) != value:
            raise ValueError("Task revision frozen plan SHA/content differs")
    if (str(Path(run_directory).resolve()) != manifest["run_directory"]
            or binding["run_directory"] != manifest["run_directory"]
            or binding["source_plan_sha256"] != manifest["source_plan_sha256"]
            or manifest.get("schema") != "task-plan-revision-" + revision):
        raise ValueError("Task revision run identity differs")
    allowed = {"contract_id", "domain_completion", "domains", "revision_migration"}
    if target["contract_id"] == REVISION_V5:
        allowed.update(("target_num_envs", "budget_spec_file", "budget_spec_sha256", "runtime_estimate"))
        validate_resized_budget(target)
        if target["runtime_estimate"] != dict(seconds_per_update=None,
                basis="4096x24_requires_native_wall_measurement; caps_not_required_run_length"):
            raise ValueError("Task v5 cannot assume an unmeasured wall-clock acceleration")
    if {k:v for k,v in source.items() if k not in allowed} != {k:v for k,v in target.items() if k not in allowed}:
        raise ValueError("Task revision changes budget, learning origin or campaign identity")
    for before, after, pin in zip(source["domains"], target["domains"], manifest["workers"], strict=True):
        if ({k:v for k,v in before.items() if k not in ("contract", "sha256")} !=
                {k:v for k,v in after.items() if k not in ("contract", "sha256")}):
            raise ValueError("Task revision changes domain scope or budget")
        for role, row in (("source", before), ("target", after)):
            path = ROOT / row["contract"]
            if (pin[f"{role}_file"] != row["contract"] or pin[f"{role}_sha256"] != row["sha256"]
                    or digest(path) != row["sha256"]):
                raise ValueError("Task revision worker SHA differs")
        validate_worker_revision(json.loads((ROOT / before["contract"]).read_text()),
                                 json.loads((ROOT / after["contract"]).read_text()))
    return dict(schema="task-plan-revision-" + revision, campaign_id=CAMPAIGN_ID,
                source_plan_sha256=manifest["source_plan_sha256"],
                target_plan_sha256=manifest["target_plan_sha256"], run_directory=manifest["run_directory"],
                actor_critic_optimizer="preserved", task_clock="restored", consumption="preserved_no_refund")


def validate_worker_revision(source, target):
    binding = target.get("task_campaign", {}).get("revision_migration", {})
    if binding.get("schema") not in ("task-worker-revision-v2", "task-worker-revision-v3", "task-worker-revision-v4", "task-worker-revision-v5"):
        raise ValueError("Unknown named task worker revision")
    path = ROOT / binding.get("source_contract_file", "")
    if not path.is_file() or digest(path) != binding.get("source_contract_sha256") or json.loads(path.read_text()) != source:
        raise ValueError("Task worker revision source is not its frozen predecessor")
    expected = deepcopy(source)
    if binding.get("schema") in ("task-worker-revision-v4", "task-worker-revision-v5"):
        if binding["schema"] == "task-worker-revision-v5":
            resize_reviewed_worker(expected)
        else:
            repair_coverage_and_events(expected)
        for key in ("source_contract_file", "source_contract_sha256", "revision_migration"):
            expected["task_campaign"][key] = deepcopy(target["task_campaign"][key])
        expected["design_preflight"] = deepcopy(target["design_preflight"])
        for field, hash_field in (("source_contract_file", "source_contract_sha256"),):
            if digest(ROOT / target["task_campaign"][field]) != target["task_campaign"][hash_field]:
                raise ValueError("Reward/coverage revision parent SHA differs")
        design = target["design_preflight"]
        if digest(ROOT / design["file"]) != design["sha256"] or expected != target:
            raise ValueError("Reward/coverage revision changed fields outside the reviewed patch")
        return
    if binding.get("schema") == "task-worker-revision-v3":
        validate_coarse_control_change(source, target)
        for key in ("physics_dt", "control_math_source", "control_math_sha256", "design_preflight"):
            expected[key] = deepcopy(target[key])
        for key in ("source_contract_file", "source_contract_sha256", "revision_migration"):
            expected["task_campaign"][key] = deepcopy(target["task_campaign"][key])
        for field, hash_field in (("source_contract_file", "source_contract_sha256"),):
            if digest(ROOT / target["task_campaign"][field]) != target["task_campaign"][hash_field]:
                raise ValueError("Coarse revision parent SHA differs")
        design = target["design_preflight"]
        if digest(ROOT / design["file"]) != design["sha256"] or expected != target:
            raise ValueError("Coarse revision changed fields outside timing/control identity")
        return
    # Every permitted revision field has an explicit value. Everything else,
    # including reward, mechanics, case thresholds and domain sampling, is exact.
    expected["checkpoint_interval"] = 500
    old_meta = expected["task_campaign"]
    new_meta = target["task_campaign"]
    for key in ("source_contract_file", "source_contract_sha256"):
        old_meta[key] = new_meta[key]
    old_meta.update(execution_mode=CONTINUOUS_MODE, evaluation_interval=0,
                    revision_migration=deepcopy(binding))
    parent = ROOT / new_meta["source_contract_file"]
    if digest(parent) != new_meta["source_contract_sha256"]:
        raise ValueError("Task revision parent SHA differs")
    expected["design_preflight"] = deepcopy(target["design_preflight"])
    design = ROOT / expected["design_preflight"]["file"]
    if digest(design) != expected["design_preflight"]["sha256"]:
        raise ValueError("Task revision design SHA differs")
    if old_meta["domain"] in ("jump", "stepjump"):
        curriculum = expected["new_asset_curriculum"]["jump_curriculum"]
        curriculum.pop("confirmed_pass_transfer_fraction", None)
        curriculum.update(clock="domain_actor_updates", iteration_probability_schedule=jump_probability_schedule())
    if expected != target:
        raise ValueError("Task worker revision changed fields outside the frozen scheduling patch")


def jump_probability_schedule():
    return [dict(actor_update=0, probabilities=[.30,.25,.20,.10,.05,.05,.05]),
            dict(actor_update=2000, probabilities=[.10,.15,.20,.20,.15,.10,.10]),
            dict(actor_update=6000, probabilities=[.05,.05,.10,.15,.20,.20,.25])]


def repair_coverage_and_events(config):
    """Reviewed v4 patch; pool/height/terrain quotas and tracking weights are exact."""
    for spec in config["skill_specs"].values():
        if spec.get("sample_amplitude"):
            spec["amplitude_range"] = [.05, 1.]
        if spec.get("kind") == "curve" and all(spec["command"][:2]):
            vx, yaw = map(abs, spec["command"][:2])
            spec["planar_command_sampling"] = dict(vx_range_m_s=[-vx, vx],
                yaw_range_rad_s=[-yaw, yaw], nominal_fraction=.5)
    # Retain the failure cost: copying SCUT's dt-scaled -4 permits early failure
    # under our unbounded broad costs. Completion must beat collecting its tail.
    config.update(success_event_bonus=30., resume_critic_warmup_updates=10,
        reward_units=dict(densities="per_second_times_policy_dt",
                          termination_event_cost="once_per_true_failure",
                          success_event_bonus="once_per_success"))


def validate_resized_budget(config):
    path = ROOT / config["budget_spec_file"]
    previous = json.loads((ROOT / "contracts/v6_task_campaign_budget_v1.json").read_text())
    expected = dict(previous, schema="task-domain-budget-v2", update_unit="one_PPO_optimizer_update",
        transition_accounting="sum_actual_num_envs_times_rollout_steps_per_attempt_no_refund",
        rollout_steps_per_env=24, previous_num_envs=20000, current_num_envs=4096)
    if (config["budget_spec_file"] != "contracts/v6_task_campaign_budget_v2.json"
            or digest(path) != config["budget_spec_sha256"] or json.loads(path.read_text()) != expected):
        raise ValueError("4096 revision changes the budget beyond its explicit mixed-batch units")


def resize_reviewed_worker(config):
    """V5 changes sampling work per update; control, reward and task clocks stay exact."""
    path = ROOT / "contracts/v6_task_campaign_budget_v2.json"
    config.update(target_num_envs=4096, num_mini_batches=4, ppo_minibatch_samples=24576,
        budget_spec_file=str(path.relative_to(ROOT)),
        budget_spec_sha256=digest(path))
    config["task_campaign"]["execution_mode"] = REVIEW_MODE
    validate_resized_budget(config)


def validate_domain_review(plan, root, row, request, consumed):
    """Authenticate an operator's early finish; review never grants skill qualification."""
    from .full_curriculum import checkpoint_update_count
    if (plan["contract_id"] != REVISION_V5 or request.get("schema") != "task-domain-review-v1"
            or request.get("campaign_id") != CAMPAIGN_ID
            or request.get("plan_sha256") != digest(ROOT / "contracts/v6_task_campaign_v5.json")
            or request.get("domain") != row["name"] or request.get("decision") != "finish_domain"
            or not isinstance(request.get("reason"), str) or not request["reason"].strip()):
        raise ValueError("Review does not identify the active frozen task/domain/decision")
    root = Path(root).resolve()
    checkpoint = Path(request["checkpoint"]).resolve(strict=True)
    if (not checkpoint.is_relative_to(root / row["name"]) or "checkpoints" not in checkpoint.relative_to(root).parts
            or digest(checkpoint) != request["checkpoint_sha256"]):
        raise ValueError("Review must name an immutable checkpoint from the active domain")
    count = checkpoint_update_count(checkpoint)
    seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
    if (count > consumed or seal.get("contract_sha256") != row["sha256"]
            or seal.get("task_campaign", {}).get("domain") != row["name"]):
        raise ValueError("Review checkpoint identity/consumption differs from the active task")
    evidence = request.get("evidence", [])
    if len(evidence) != 2 or {item.get("kind") for item in evidence} != {"curve_review", "behavior_review"}:
        raise ValueError("Early finish requires curve and behavior evidence, not total reward alone")
    for item in evidence:
        path = Path(item["file"]).resolve(strict=True)
        if (not path.is_relative_to(root / "reviews") or path.suffix != ".json"
                or digest(path) != item["sha256"]):
            raise ValueError("Review evidence is outside the immutable review receipt directory or SHA differs")
        data = json.loads(path.read_text())
        if (data.get("checkpoint_sha256") != request["checkpoint_sha256"]
                or not isinstance(data.get("observations"), dict) or not data["observations"]):
            raise ValueError("Review evidence must describe this checkpoint's observed curves/behavior")
    return dict(**request, consumed_updates_at_review=consumed, domain_qualified=None,
                continuation="seal_latest_full_learning_state; preserve_all_consumption")


def validate_coarse_control_change(source, target):
    """Authenticate the explicit approximation while retaining every motor parameter."""
    from .control_contract import load_chassis_control, COARSE_SCOPE, COARSE_TIMING
    if (source.get("physics_dt"), target.get("physics_dt")) != (.001, .005):
        raise ValueError("Coarse revision must change 1 ms to 5 ms")
    if target.get("task_campaign", {}).get("revision_migration", {}).get("schema") not in ("task-worker-revision-v3", "task-worker-revision-v4", "task-worker-revision-v5"):
        raise ValueError("Coarse physics identity requires a named v3 revision")
    controls = []
    for config in (source, target):
        path = ROOT / config["control_math_source"]
        if digest(path) != config["control_math_sha256"]:
            raise ValueError("Coarse control SHA differs")
        controls.append(load_chassis_control(path))
    expected = deepcopy(controls[0])
    expected.update(control_id="v6-wheel-response-control-200hz-v1", timing=COARSE_TIMING,
                    simulation_timing_approximation=COARSE_SCOPE)
    if controls[1] != expected:
        raise ValueError("Coarse migration changes motor parameters or undeclared timing")


def validate_task_resume(source, target):
    meta = target.get("task_campaign", {})
    if (meta.get("campaign_id") != CAMPAIGN_ID or meta.get("domain") not in DOMAINS
            or meta.get("updates") != 20000 or meta.get("clock") != "domain_actor_updates"):
        raise ValueError("Missing frozen task migration identity")
    same_domain = source.get("task_campaign", {}).get("domain") == meta["domain"]
    if same_domain:
        if source != target:
            if not meta.get("revision_migration"):
                raise ValueError("Same-domain recovery cannot change its contract")
            validate_worker_revision(source, target)
    else:
        path = ROOT / meta["source_contract_file"]
        if digest(path) != meta["source_contract_sha256"] or json.loads(path.read_text()) != source:
            raise ValueError("Task learning origin differs from its frozen source contract")
    # Curriculum/reward domains may differ. The physical, actuator and neural
    # interface remains identical; accepted capability is never inferred here.
    coarse_change = source.get("physics_dt") != target.get("physics_dt")
    if coarse_change:
        validate_coarse_control_change(source, target)
    for key in ("asset_manifest_sha256", "control_math_sha256", "physics_dt", "policy_dt",
                "actor_dim", "critic_dim", "action_dim", "history_length", "policy_architecture",
                "v5_control", "actuator_response", "actuator_response_evidence", "policy_action_order",
                "actor_layout", "critic_layout", "actor_observation_source"):
        if coarse_change and key in ("physics_dt", "control_math_sha256"):
            continue
        if source.get(key) != target.get(key):
            raise ValueError(f"Task migration changes physical/policy identity: {key}")
    revision = target.get("task_campaign", {}).get("revision_migration", {}).get("schema")
    coarse = revision in ("task-worker-revision-v3", "task-worker-revision-v4", "task-worker-revision-v5")
    resized = revision == "task-worker-revision-v5"
    if (target.get("physics_dt"), target.get("policy_dt"), target.get("actor_dim"),
            target.get("critic_dim"), target.get("target_num_envs")) != (.005 if coarse else .001, .02, 35, 81, 4096 if resized else 20000):
        raise ValueError("Task campaign changes its frozen environment count, network or control clocks")
    if resized:
        validate_resized_budget(target)
        if (target.get("num_steps_per_env"), target.get("num_mini_batches"),
                target.get("ppo_minibatch_samples"), target.get("num_learning_epochs")) != (24, 4, 24576, 5):
            raise ValueError("Task v5 requires 4096x24, four minibatches and five epochs")
    if target.get("learning_rate") != 1e-4 or target.get("learning_rate_schedule") != "fixed":
        raise ValueError("Task campaign fixed learning rate changed")
    return dict(kind="task_domain_recovery" if same_domain else "task_domain_migration",
                actor_critic_optimizer="preserved", environment_state="new_episodes",
                task_clock="restored" if same_domain else "new_domain_zero",
                accepted_stage=None, hardware_release=False)


def validate_worker(config, args):
    from .full_curriculum import checkpoint_contract_path, checkpoint_update_count
    meta = config["task_campaign"]
    if (not args.resume_task_change or not args.resume or args.num_envs != config["target_num_envs"]
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
    resized = config.get("task_campaign", {}).get("revision_migration", {}).get("schema") == "task-worker-revision-v5"
    if resized:
        validate_resized_budget(config)
    if (budget.get("schema") != ("task-domain-budget-v2" if resized else "task-domain-budget-v1")
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
