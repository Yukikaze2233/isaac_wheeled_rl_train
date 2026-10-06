#!/usr/bin/env python3
"""Independent commanded multi-scene closed-chain training on Isaac Sim 6."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed_call(timer, phase, function, *args, **kwargs):
    if timer is None:
        return function(*args, **kwargs)
    with timer.measure(phase):
        return function(*args, **kwargs)


def write_progress(directory, progress, timer=None):
    def write():
        temporary = directory / "progress.tmp"
        with temporary.open("w") as stream:
            json.dump(progress, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(directory / "progress.json")
    timed_call(timer, "progress_io", write)


def task_confirmed_bins(previous, decision, allowed):
    confirmed = decision.get("confirmed_jump_bins_m", previous)
    if (not isinstance(confirmed, list) or any(type(x) not in (int, float) or x not in allowed for x in confirmed)
            or not set(previous).issubset(confirmed)):
        raise ValueError("Task assessment cannot withdraw or invent confirmed jump bins")
    return sorted(set(confirmed))


def task_assessment_due(config, task_update):
    """An explicitly zero interval disables requests as well as waiting."""
    task = config.get("task_campaign", {})
    interval = task.get("evaluation_interval", 0)
    return bool(interval) and task_update > 0 and task_update % interval == 0


def resume_critic_warmup(config, infos, *, contract_sha256=None, task_change=False, reward_change=False):
    """Adapt once to a named contract change; a fault resumes the remaining work."""
    if task_change and contract_sha256 is None:
        raise ValueError("Task adaptation requires the frozen contract SHA")
    revision = config.get("task_campaign", {}).get("revision_migration", {})
    resize_only = (task_change and revision.get("schema") == "task-worker-revision-v5"
        and infos.get("contract_sha256") == revision.get("source_contract_sha256")
        and infos.get("task_campaign", {}).get("domain") == config["task_campaign"]["domain"])
    if reward_change or task_change and infos.get("contract_sha256") != contract_sha256 and not resize_only:
        return config.get("resume_critic_warmup_updates", 0)
    return infos.get("critic_warmup_remaining", max(0,
        config.get("critic_warmup_updates", 0) - infos["successful_updates_total"]))


def configure_learning_rate(algorithm, contract_rate, scale=None):
    """Keep PPO's scalar aligned with restored Adam, or apply an explicit ratio.

    Recovery's decay factor is bounded by the orchestrator. The resulting
    ratio to the contract's initial LR can exceed one after adaptive updates.
    """
    rate = algorithm.optimizer.param_groups[0]["lr"]
    if scale is not None:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Recovery learning-rate scale must be finite and positive")
        rate = contract_rate * scale
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Optimizer learning rate must be finite and positive")
    if scale is not None:
        for group in algorithm.optimizer.param_groups:
            group["lr"] = rate
    algorithm.learning_rate = rate
    return rate


def validate_entropy_coef(value):
    """Accept a contract scalar without changing the native distribution."""
    if type(value) not in (int, float):
        raise ValueError("entropy_coef must be a finite nonnegative scalar")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("entropy_coef must be a finite nonnegative scalar") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("entropy_coef must be a finite nonnegative scalar")
    return value


def verify_entropy_coef(algorithm, expected):
    """Read back the native loss coefficient; never reset std or Adam."""
    try:
        actual = validate_entropy_coef(algorithm.entropy_coef)
    except ValueError as error:
        raise RuntimeError("Native PPO entropy coefficient is invalid") from error
    if actual != expected:
        raise RuntimeError("Native PPO entropy coefficient differs from its configuration")
    return actual


def configure_recovery_rollout(algorithm, env):
    """Exclude any opted-in interval whose motor output is externally owned."""
    if not (env.cfg.get("recovery_training", {}).get("enabled", False)
            or env.cfg.get("nonstandard_start", {}).get("enabled", False)):
        return None
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    return attach_recovery_rollout_mask(algorithm, num_envs=env.num_envs)


def nonstandard_start_source_files(config, root=ROOT, *, require_release_stop=True):
    """Audit the exact pose bank and the ordinary birth/release teaching modules."""
    if "nonstandard_start" not in config:
        return []
    from wheeled_tasks.chassis.nonstandard_start import validate_nonstandard_start
    from wheeled_tasks.chassis.release_stop import validate_release_stop
    settings = config["nonstandard_start"]
    validate_nonstandard_start(settings)
    if require_release_stop or "release_stop" in config:
        validate_release_stop(config.get("release_stop"))
    if (settings["asset_manifest_sha256"] != config["asset_manifest_sha256"]
            or digest(Path(root) / settings["profiles_file"]) != settings["profiles_sha256"]):
        raise ValueError("Nonstandard birth source audit changed its frozen asset or pose bank")
    return ["src/wheeled_tasks/chassis/nonstandard_start.py",
            "src/wheeled_tasks/chassis/nonstandard_start_revision.py",
            "src/wheeled_tasks/chassis/release_stop.py",
            "src/wheeled_algo/recovery_rollout.py", settings["profiles_file"]]


def scut_curriculum_source_files(config, root=ROOT):
    """Audit the exact v17/v18/v19/v20 recipe and its native teaching dependencies."""
    meta = config.get("task_campaign", {})
    binding = meta.get("revision_migration", {})
    schema = binding.get("schema")
    if schema not in ("task-worker-revision-v17", "task-worker-revision-v18", "task-worker-revision-v19", "task-worker-revision-v20"):
        return []
    if schema == "task-worker-revision-v20":
        from wheeled_tasks.chassis import recovery_script_revision as recipe
        canonical = recipe.canonical_worker(root=root)
        if (meta.get("domain") != "recovery"
                or binding != recipe.revision_binding()
                or config.get("design_preflight") != canonical["design_preflight"]):
            raise ValueError("Recovery script source audit differs from its exact revision/design binding")
        source = json.loads((Path(root) / recipe.SOURCE_WORKER_FILE).read_text())
        inherited = scut_curriculum_source_files(source, root)
        return list(dict.fromkeys(inherited + [
            "src/wheeled_tasks/chassis/recovery_script_revision.py",
            "src/wheeled_tasks/chassis/rewards.py", "src/wheeled_tasks/chassis/env.py",
            "src/wheeled_tasks/chassis/recovery_training.py", "src/wheeled_tasks/chassis/recovery.py",
            "src/wheeled_tasks/chassis/preflight.py",
            recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE,
            recipe.SOURCE_DESIGN_FILE, recipe.DESIGN_FILE]))
    if schema == "task-worker-revision-v19":
        from wheeled_tasks.chassis import recovery_quality_revision as recipe
        canonical = recipe.canonical_worker(root=root)
        if (meta.get("domain") != "recovery"
                or binding != recipe.revision_binding()
                or config.get("design_preflight") != canonical["design_preflight"]):
            raise ValueError("Recovery quality source audit differs from its exact revision/design binding")
        source = json.loads((Path(root) / recipe.SOURCE_WORKER_FILE).read_text())
        inherited = scut_curriculum_source_files(source, root)
        return list(dict.fromkeys(inherited + [
            "src/wheeled_tasks/chassis/recovery_quality_revision.py",
            "src/wheeled_tasks/chassis/rewards.py", "src/wheeled_tasks/chassis/env.py",
            "src/wheeled_tasks/chassis/recovery_training.py", "src/wheeled_tasks/chassis/recovery.py",
            "src/wheeled_tasks/chassis/preflight.py",
            recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE,
            recipe.SOURCE_DESIGN_FILE, recipe.DESIGN_FILE]))
    from wheeled_tasks.chassis import scut_curriculum_revision as scut_recipe
    if schema == "task-worker-revision-v18":
        from wheeled_tasks.chassis import ordinary_posture_revision as recipe
    else:
        recipe = scut_recipe
    domain = meta.get("domain")
    canonical = recipe.canonical_worker(domain, root)
    if binding != canonical["task_campaign"]["revision_migration"]:
        raise ValueError("SCUT teaching source audit differs from its exact revision binding")
    sources = ["src/wheeled_tasks/chassis/scut_curriculum_revision.py",
        "src/wheeled_tasks/chassis/scut_base_tracking.py",
        "src/wheeled_tasks/chassis/locomotion_quality_revision.py",
        "src/wheeled_tasks/chassis/locomotion_revision.py",
        "src/wheeled_tasks/chassis/nonstandard_start_revision.py",
        binding["source_contract_file"]]
    source = json.loads((Path(root) / binding["source_contract_file"]).read_text())
    sources.append(source["design_preflight"]["file"])
    if schema == "task-worker-revision-v18":
        if config.get("design_preflight") != canonical["design_preflight"]:
            raise ValueError("Ordinary posture source audit differs from its independent design binding")
        sources.extend(["src/wheeled_tasks/chassis/ordinary_posture_curriculum.py",
            "src/wheeled_tasks/chassis/ordinary_posture_revision.py",
            recipe.SOURCE_PLAN_FILE, canonical["design_preflight"]["file"]])
        if domain in scut_recipe.DOMAINS_TO_REVISE:
            _, _, inherited_worker, inherited_design, *_ = scut_recipe._recipe(domain)
            sources.extend((inherited_worker, inherited_design))
    if domain == "recovery":
        sources.extend("src/wheeled_tasks/chassis/" + name + ".py" for name in
            ("recovery_prefix_revision", "recovery_prefix_curriculum", "recovery_curriculum"))
    elif domain == "terrain":
        sources.extend("src/wheeled_tasks/chassis/" + name + ".py" for name in
            ("scut_terrain_revision", "terrain_rehearsal_revision"))
    elif domain in ("jump", "stepjump"):
        sources.extend("src/wheeled_tasks/chassis/" + name + ".py" for name in
            ("scut_jump_revision", "jump_landing_revision", "jump_landing"))
    elif domain != "speed":
        raise ValueError("SCUT teaching source audit requires a named revised task domain")
    sources.extend(nonstandard_start_source_files(config, root,
        require_release_stop=domain in ("speed", "terrain")))
    return list(dict.fromkeys(sources))


def wait_for_task_assessment(directory, checkpoint, task_update, consumed, check_budget,
                             *, sleep=time.sleep):
    """Yield this GPU without discarding the live training scene or its episodes."""
    request = {"schema": "task-assessment-request-v1", "token": uuid.uuid4().hex,
        "task_actor_update": task_update, "consumed_updates": consumed,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint)}
    temporary = directory / "assessment_request.tmp"
    temporary.write_text(json.dumps(request, indent=2) + "\n")
    temporary.replace(directory / "assessment_request.json")
    response = directory / "assessment_decision.json"
    while True:
        check_budget()
        if response.exists():
            decision = json.loads(response.read_text())
            if decision.get("token") == request["token"]:
                if (decision.get("checkpoint_sha256") != request["checkpoint_sha256"]
                        or decision.get("decision") not in ("continue", "domain_qualified_stop")):
                    raise ValueError("Task assessment decision does not match its immutable checkpoint")
                evaluations = decision.get("evaluations", [])
                if (len(evaluations) != 2
                        or {e["seed"] for e in evaluations} != {290929, 290930}
                        or any(digest(Path(e["file"])) != e["sha256"] for e in evaluations)):
                    raise ValueError("Task assessment requires two authenticated fixed evaluations")
                if (decision["decision"] == "domain_qualified_stop"
                        and decision.get("domain_qualified") is not True):
                    raise ValueError("Early task stop requires a qualified domain")
                return decision
        sleep(1.)


def preflight(path, *, asset_directory=None):
    c = json.loads(path.read_text())
    if "entropy_coef" in c:
        validate_entropy_coef(c["entropy_coef"])
    if asset_directory is not None:
        c["asset_directory"] = str(asset_directory)
    kinds = {"chassis-closedchain-commanded-full-v1": "coupled_fourbar_research",
             "v5-gas-spring-commanded-research-v1": "v5_gas_spring_closedchain_research",
             "v5-gas-spring-mixed-research-v1": "v5_gas_spring_closedchain_research",
             "v5-gas-spring-locomotion-research-v2": "v5_gas_spring_closedchain_research",
             "v5-gas-spring-full-stage-research-v2": "v5_gas_spring_closedchain_research",
             "v6-gas-spring-full-stage-research-v1": "v6_gas_spring_closedchain_research"}
    if c["contract_id"] not in kinds or c["task_source"] != "upper_level_commands":
        raise ValueError("Unsupported full-task interface")
    if c["contract_id"] == "v6-gas-spring-full-stage-research-v1":
        if c.get("requires_design_preflight") is not True or not isinstance(c.get("design_preflight"), dict):
            raise ValueError("V6 requires an independent design_preflight binding")
    if sum(s["updates"] for s in c["stages"]) != c["total_updates"]:
        raise ValueError("Curriculum budgets do not sum to total")
    bundle = ROOT / c["asset_directory"]
    if digest(bundle / "manifest.json") != c["asset_manifest_sha256"]:
        raise ValueError("Closed-chain manifest identity mismatch")
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        target = (bundle / name).resolve()
        if not target.is_relative_to(bundle.resolve()) or digest(target) != expected:
            raise ValueError(f"Asset dependency mismatch: {name}")
    if manifest["model_kind"] != kinds[c["contract_id"]]:
        raise ValueError("This task requires the multi-body asset")
    if version("rsl-rl-lib") != "5.5.1":
        raise ValueError("This entry requires the validated RSL-RL 5.5.1 runtime")
    return c, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "contracts/chassis_full_v1.json")
    parser.add_argument("--stage", choices=("foundation", "terrain", "speed", "up", "down", "jump", "mixed"), default="foundation")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--level", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--actor-updates-consumed", type=int, default=0)
    parser.add_argument("--stage-actor-update", type=int)
    parser.add_argument("--task-actor-update", type=int)
    parser.add_argument("--global-actor-update-offset", type=int, default=0)
    parser.add_argument("--confirmed-jump-bins-json", default="[]")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--coverage", action="store_true", help="validation: include every stage terrain family")
    parser.add_argument("--validate-scene", type=int, default=0, metavar="STEPS")
    parents = parser.add_mutually_exclusive_group()
    parents.add_argument("--resume", type=Path)
    parents.add_argument("--transfer", type=Path, help="V5 weights only into a new compatible scene contract; fresh optimizer")
    parser.add_argument("--resume-physics-change", action="store_true",
                        help="Explicit1kHz-to200Hz learning-state resume with strict semantic validation")
    parser.add_argument("--resume-reward-change", action="store_true",
                        help="Explicit precision-reward migration preserving actor/critic/Adam")
    parser.add_argument("--resume-budget-change", action="store_true",
                        help="Extend a declared budget while preserving learning state and acceptance")
    parser.add_argument("--resume-response-change", action="store_true",
                        help="Validated actuator-response change preserving actor/critic/Adam and consumption")
    parser.add_argument("--resume-diagnostic", action="store_true",
                        help="Exact learning-state fork into a separate paired diagnostic ledger")
    parser.add_argument("--resume-task-change", action="store_true",
                        help="Frozen task-domain migration with separate consumption and curriculum clocks")
    parser.add_argument("--consumed-updates", type=int, help="Same-batch budget already spent, including discarded updates")
    parser.add_argument("--learning-rate-scale", type=float,
                        help="Explicit positive ratio to the contract's initial LR; may exceed one after adaptive updates")
    parser.add_argument("--transfer-actor-only", action="store_true", help="Reset critic when changing the reward/task distribution")
    parser.add_argument("--publish-state", action="store_true", help="Publish atomic selected-env real physics snapshots at most 4 Hz")
    parser.add_argument("--profile", action="store_true", help="Save bounded PPO sampling CPU profile before Kit shutdown")
    parser.add_argument("--max-runtime-seconds", type=float, default=172800.)
    args = parser.parse_args()
    if not (1 <= args.num_envs <= 20000 and 1 <= args.updates <= 100000 and 0 <= args.level <= 1
            and 0 <= args.validate_scene <= 10000 and args.max_runtime_seconds > 0):
        parser.error("Invalid bounded run settings")
    c, manifest = preflight(args.contract)
    if args.transfer_actor_only and not args.transfer:
        parser.error("--transfer-actor-only requires --transfer")
    if args.resume_physics_change and not args.resume:
        parser.error("--resume-physics-change requires --resume")
    if (args.resume_reward_change or args.resume_budget_change or args.resume_response_change or args.resume_diagnostic or args.resume_task_change or args.consumed_updates is not None) and not args.resume:
        parser.error("Reward migration and consumed-update rebasing require --resume")
    if sum((args.resume_physics_change, args.resume_reward_change, args.resume_budget_change, args.resume_response_change, args.resume_diagnostic, args.resume_task_change)) > 1:
        parser.error("Choose exactly one explicit resume migration")
    if args.resume_response_change and args.consumed_updates is None:
        parser.error("Response migration requires explicit consumed-update accounting")
    if args.resume_diagnostic and (args.consumed_updates is not None or args.learning_rate_scale is not None):
        parser.error("Diagnostic forks use the study ledger and the preserved learning rate")
    if args.learning_rate_scale is not None and (
            not math.isfinite(args.learning_rate_scale) or args.learning_rate_scale <= 0):
        parser.error("Recovery learning-rate scale must be finite and positive")
    if c.get("remediation"):
        from wheeled_tasks.chassis.remediation import validate_worker
        validate_worker(c, args)
    if c.get("continuation"):
        from wheeled_tasks.chassis.continuation import validate_worker
        validate_worker(c, args)
    if c.get("task_campaign"):
        from wheeled_tasks.chassis.task_campaign import validate_worker
        validate_worker(c, args)
    elif args.resume_task_change or args.task_actor_update is not None:
        parser.error("Task migration and its clock require a frozen task campaign")
    if args.stage not in c.get("enabled_stages", [s["name"] for s in c["stages"]]):
        parser.error("This contract has not enabled the requested training stage")
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    control = load_chassis_control(ROOT / c["control_math_source"])
    design_receipt = design_preflight(c, control, ROOT, stage=args.stage, role="train",
        seed=args.seed, num_envs=args.num_envs, block_index=args.block_index)
    if min(args.block_index, args.actor_updates_consumed, args.global_actor_update_offset, args.stage_actor_update or 0) < 0:
        parser.error("Block and actor consumption clocks must be nonnegative")
    confirmed_bins = json.loads(args.confirmed_jump_bins_json)
    allowed_bins = c.get("new_asset_curriculum", {}).get("jump_curriculum", {}).get("bins_m", [])
    if (not isinstance(confirmed_bins, list) or any(type(x) not in (int, float)
            or x not in allowed_bins for x in confirmed_bins)):
        parser.error("Confirmed jump bins must belong to the frozen curriculum")
    if c.get("engineering_probe_only"):
        if not args.preflight_only and args.validate_scene != c["capacity_probe"]["policy_steps"]:
            parser.error("Engineering capacity contract requires its scene-only step count; PPO is forbidden")
        if args.resume or args.transfer:
            parser.error("Engineering capacity probe cannot resume or transfer a policy")
    if args.preflight_only:
        print(json.dumps({"asset_verified": True, "contract": c["contract_id"],
                          "asset_directory": c["asset_directory"], "stage": args.stage,
                          "design_preflight": design_receipt,
                          "long_run_behavior_acceptance": "pending"}, indent=2))
        return 0
    if not args.research or args.run_dir is None:
        parser.error("--research and a new --run-dir are required")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    source_files = ["scripts/train_chassis.py", "src/wheeled_tasks/chassis/env.py",
                    "src/wheeled_tasks/chassis/root_kinematics.py",
                    "src/wheeled_tasks/chassis/preflight.py",
                    "src/wheeled_tasks/chassis/control_contract.py",
                    "src/wheeled_tasks/chassis/task.py", "src/wheeled_tasks/v40/core.py",
                    "src/wheeled_tasks/v40/contract.py", "src/wheeled_tasks/agents/v40_ppo_cfg.py",
                    "src/wheeled_algo/v40_job.py", "src/wheeled_algo/chassis_export.py", c["control_math_source"]]
    if c.get("design_preflight"):
        source_files.append(c["design_preflight"]["file"])
    from wheeled_tasks.chassis.preflight import GAS_SPRING_MODEL_KINDS
    if manifest["model_kind"] in GAS_SPRING_MODEL_KINDS:
        source_files.append("src/wheeled_tasks/chassis/v5_control.py")
        source_files.append("src/wheeled_tasks/chassis/torque_monitor.py")
    if c.get("record_diagnostics"):
        source_files.append("src/wheeled_tasks/chassis/episode_metrics.py")
    source_files.append("src/wheeled_tasks/chassis/full_curriculum.py")
    if c.get("skill_specs"):
        source_files.extend(["src/wheeled_tasks/chassis/skill_commands.py", "src/wheeled_tasks/chassis/skill_curriculum.py",
                             "src/wheeled_tasks/chassis/integrated_curriculum.py"])
    if c.get("checkpoint_snapshots"):
        source_files.append("scripts/chassis_checkpoints.py")
    if c.get("actor_observation_source"):
        source_files.append("src/wheeled_tasks/chassis/scut_observation.py")
    if c.get("reward_profile"):
        source_files.append("src/wheeled_tasks/chassis/rewards.py")
    if c.get("performance_curriculum"):
        source_files.append("src/wheeled_tasks/chassis/performance_curriculum.py")
        if c["performance_curriculum"].get("kind") == "adaptive_commands":
            source_files.append("src/wheeled_tasks/chassis/adaptive_commands.py")
    if c.get("motion_limits"):
        source_files.append("src/wheeled_tasks/chassis/motion_limits.py")
    if c.get("usb_transport", {}).get("enabled"):
        source_files.append("src/wheeled_tasks/chassis/usb_transport.py")
    if c.get("command_transport", {}).get("enabled"):
        source_files.append("src/wheeled_tasks/chassis/command_transport.py")
    if "signal_delay" in c:
        source_files.append("src/wheeled_tasks/chassis/signal_delay.py")
    if c.get("contact_domain", {}).get("enabled"):
        source_files.append("src/wheeled_tasks/chassis/contact_domain.py")
    if c.get("command_reference"):
        source_files.append("src/wheeled_tasks/chassis/references.py")
    if c.get("command_reference") or args.transfer or args.resume_physics_change or args.resume_response_change or args.resume_task_change:
        source_files.append("src/wheeled_tasks/chassis/policy_transfer.py")
    if c.get("response_migration"):
        source_files.append("src/wheeled_tasks/chassis/response_migration.py")
    source_files.append("src/wheeled_tasks/chassis/actuator_response.py")
    if c.get("actuator_response", {}).get("enabled"):
        evidence_file = c["actuator_response_evidence"]["file"]
        source_files.append(evidence_file)
        source_files.extend(json.loads((ROOT / evidence_file).read_text())["files_sha256"])
    if c.get("diagnostic_study"):
        source_files.append("src/wheeled_tasks/chassis/diagnostic_curriculum.py")
    if c.get("continuation"):
        source_files.append("src/wheeled_tasks/chassis/continuation.py")
    if c.get("task_campaign"):
        source_files.extend(["src/wheeled_tasks/chassis/task_campaign.py",
                             "src/wheeled_tasks/chassis/training_timing.py"])
        if "locomotion_difficulty" in c:
            source_files.extend(["src/wheeled_tasks/chassis/locomotion_difficulty.py",
                                 "src/wheeled_tasks/chassis/graded_locomotion_revision.py"])
        if c["task_campaign"].get("revision_migration", {}).get("schema") == "task-worker-revision-v8":
            source_files.append("src/wheeled_tasks/chassis/recovery_revision.py")
        if c["task_campaign"].get("revision_migration", {}).get("schema") == "task-worker-revision-v12":
            source_files.extend(["src/wheeled_tasks/chassis/jump_landing.py",
                                 "src/wheeled_tasks/chassis/jump_landing_revision.py"])
        if c["task_campaign"].get("revision_migration", {}).get("schema") in (
                "task-worker-revision-v13", "task-worker-revision-v14", "task-worker-revision-v15"):
            source_files.extend(nonstandard_start_source_files(c))
        if c["task_campaign"].get("revision_migration", {}).get("schema") in (
                "task-worker-revision-v14", "task-worker-revision-v15"):
            source_files.append("src/wheeled_tasks/chassis/locomotion_quality_revision.py")
        if c["task_campaign"].get("revision_migration", {}).get("schema") == "task-worker-revision-v15":
            source_files.extend(["src/wheeled_tasks/chassis/terrain_rehearsal_revision.py",
                                 "src/wheeled_tasks/chassis/locomotion_revision.py"])
        source_files.extend(scut_curriculum_source_files(c))
    if c.get("remediation") or c.get("continuation") or c.get("task_campaign"):
        source_files.extend(["src/wheeled_tasks/chassis/remediation.py",
                             "src/wheeled_tasks/chassis/learning_audit.py"])
    for key, module in (("dynamics_randomization", "dynamics"), ("step_assist", "step_assist")):
        if c.get(key, {}).get("enabled"):
            source_files.append(f"src/wheeled_tasks/chassis/{module}.py")
    if c.get("cross_asset_source_manifest_sha256"):
        source_files.append("src/wheeled_tasks/chassis/evaluation.py")
    if c.get("task_semantics"):
        source_files.extend(["src/wheeled_tasks/chassis/full_tasks.py", "src/wheeled_tasks/chassis/robustness.py"])
    if c.get("new_asset_curriculum", {}).get("enabled"):
        source_files.extend(["src/wheeled_tasks/chassis/training_schedule.py",
            "scripts/validate_v6_functional_probes.py", "src/wheeled_tasks/chassis/step_assist.py"])
    if c.get("recovery_training", {}).get("enabled"):
        source_files.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in
            ("recovery", "recovery_training", "continuous_encoders", "mechanism_observer", "recovery_observer"))
        if (c["recovery_training"].get("pose_curriculum", {}).get("enabled")
                and not c.get("evaluation_exact_cases")):
            source_files.append("src/wheeled_tasks/chassis/recovery_curriculum.py")
        if c["recovery_training"].get("prefix_curriculum", {}).get("enabled"):
            source_files.extend(["src/wheeled_tasks/chassis/recovery_prefix_curriculum.py",
                "src/wheeled_tasks/chassis/recovery_curriculum.py"])
        source_files.append("src/wheeled_algo/recovery_rollout.py")
        source_files.extend(c["recovery_training"][key] for key in ("profiles_file", "height_lookup_file"))
    source_files = list(dict.fromkeys(source_files))
    source_hashes = {name: digest(ROOT / name) for name in source_files}
    for name in source_files:
        target = args.run_dir / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    identity = {"contract_id": c["contract_id"], "contract_sha256": digest(args.contract),
                "asset_manifest_sha256": c["asset_manifest_sha256"],
                "control_math_sha256": digest(ROOT / c["control_math_source"])}
    report = {"started_at": datetime.now(timezone.utc).isoformat(), **identity,
              "source_sha256": source_hashes, "stage": args.stage, "status": "starting",
               "successful_updates": 0, "attempted_updates": 0, "parent_updates": 0, "parent_training_transitions": 0,
               "actor_updates_in_block": 0,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.run_dir / "contract.json").write_bytes(args.contract.read_bytes())
    (args.run_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    launcher, env, runner, metrics, profiler, timing = None, None, None, None, None, None
    from wheeled_algo.v40_job import TrainingBudget, PlannedStop
    budget = TrainingBudget(args.max_runtime_seconds)
    try:
        if c.get("new_asset_curriculum", {}).get("enabled"):
            from validate_v6_functional_probes import run_probes
            functional = run_probes(args.seed)
            functional["contract_sha256"] = identity["contract_sha256"]
            (args.run_dir / "functional_preflight.json").write_text(json.dumps(functional, indent=2) + "\n")
            if not functional["passed"]:
                raise RuntimeError("Executable module probes failed")
            report["functional_preflight"] = "passed"
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": args.device,
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        from wheeled_tasks.chassis.env import ChassisEnv
        env = ChassisEnv(c, manifest, control, ROOT,
            stage_name=args.stage, num_envs=args.num_envs, device=args.device, level=args.level,
            seed=args.seed, coverage=args.coverage, block_index=args.block_index)
        report["startup"] = env.startup_report
        if c.get("record_diagnostics"):
            from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
            height_range = c.get("height_workspace", {}).get("height_m")
            metrics = EpisodeMetrics(env.scene_groups, args.device, c["policy_dt"],
                height_range_m=[height_range[0], height_range[-1]] if height_range else None)
        report["status"] = "running"
        (args.run_dir / "startup.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        with budget.signal_handlers():
            budget.begin_learning()
            if args.validate_scene:
                scene_started = time.monotonic()
                for _ in range(args.validate_scene):
                    budget.check()
                    env.step(torch.zeros(args.num_envs, 6, device=args.device))
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                scene_seconds = time.monotonic() - scene_started
                report["scene_capacity"] = {
                    "scope": "physics_only_no_ppo_allocation", "policy_steps": args.validate_scene,
                    "physics_steps_per_env": args.validate_scene * env.decimation,
                    "wall_seconds": scene_seconds,
                    "environment_transitions_per_second": args.num_envs * args.validate_scene / scene_seconds}
                from torch.utils.tensorboard import SummaryWriter
                with SummaryWriter(str(args.run_dir)) as scene_writer:
                    for name, value in report["scene_capacity"].items():
                        if isinstance(value, (int, float)):
                            scene_writer.add_scalar(f"SceneCapacity/{name}", value, args.validate_scene)
                report["status"] = "scene_validated"
            else:
                from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
                from rsl_rl.runners import OnPolicyRunner
                from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
                cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
                cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
                cfg.seed, cfg.device = args.seed, args.device
                cfg.save_interval = c["save_interval"]
                cfg.num_steps_per_env = c["num_steps_per_env"]
                if "num_mini_batches" in c:
                    cfg.algorithm.num_mini_batches = c["num_mini_batches"]
                if "learning_rate" in c:
                    cfg.algorithm.learning_rate = c["learning_rate"]
                if "learning_rate_schedule" in c:
                    cfg.algorithm.schedule = c["learning_rate_schedule"]
                if "entropy_coef" in c:
                    cfg.algorithm.entropy_coef = validate_entropy_coef(c["entropy_coef"])
                if args.stage in ("terrain", "mixed"):
                    cfg.algorithm.value_loss_coef = 2.
                runner_config = cfg.to_dict()
                runner_config["logger"] = "tensorboard"
                if "initial_noise_std" in c:
                    runner_config["actor"]["distribution_cfg"]["init_std"] = c["initial_noise_std"]
                (args.run_dir / "agent_config.json").write_text(json.dumps(runner_config, indent=2))
                runner = OnPolicyRunner(env, deepcopy(runner_config), log_dir=str(args.run_dir), device=args.device)
                report["entropy_coef"] = verify_entropy_coef(runner.alg, cfg.algorithm.entropy_coef)
                report["entropy_coef_source"] = "contract" if "entropy_coef" in c else "runner_default"
                if args.transfer:
                    checkpoint = torch.load(args.transfer, map_location="cpu", weights_only=True)
                    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                    source_contract_path = checkpoint_contract_path(args.transfer)
                    old_contract = json.loads(source_contract_path.read_text())
                    if checkpoint.get("infos", {}).get("contract_sha256") != digest(source_contract_path):
                        raise ValueError("Source checkpoint contract provenance mismatch")
                    from wheeled_tasks.chassis.policy_transfer import is_reference_migration, transfer_actor_state
                    reference_migration = is_reference_migration(old_contract, c)
                    if reference_migration and not args.transfer_actor_only:
                        raise ValueError("Observation migration requires fresh critic and optimizer")
                    asset_migration = None
                    if checkpoint.get("infos", {}).get("asset_manifest_sha256") != identity["asset_manifest_sha256"]:
                        if not args.transfer_actor_only:
                            raise ValueError("Cross-asset migration requires an actor-only transfer")
                        from wheeled_tasks.chassis.evaluation import cross_asset_checkpoint_provenance
                        asset_migration = cross_asset_checkpoint_provenance(args.transfer, checkpoint["infos"], c,
                            manifest, identity["control_math_sha256"])
                    for key in ("asset_manifest_sha256", "actor_dim", "actor_frame_dim", "critic_dim", "action_dim",
                                "actor_layout", "critic_layout", "task_modes", "phases", "v5_control", "policy_dt",
                                "policy_action_order", "actor_observation_source", "history_length"):
                        if args.transfer_actor_only and key in ("critic_dim", "critic_layout"):
                            continue
                        if reference_migration and key in ("actor_dim", "actor_frame_dim", "actor_layout", "actor_observation_source"):
                            continue
                        if key == "asset_manifest_sha256" and asset_migration is not None:
                            continue
                        if key == "v5_control":
                            from wheeled_tasks.chassis.full_curriculum import compatible_control_transfer
                            if compatible_control_transfer(old_contract[key], c[key]):
                                continue
                        if old_contract.get(key) != c.get(key):
                            raise ValueError(f"Scene transfer changes the V5 physical/control interface: {key}")
                    if asset_migration is None and checkpoint.get("infos", {}).get("asset_manifest_sha256") != identity["asset_manifest_sha256"]:
                        raise ValueError("Transfer checkpoint asset mismatch")
                    actor_state, observation_migration = transfer_actor_state(checkpoint["actor_state_dict"], old_contract, c)
                    runner.alg.actor.load_state_dict(actor_state, strict=True)
                    if c.get("transfer_noise_floor"):
                        with torch.no_grad():
                            runner.alg.actor.distribution.std_param.clamp_(min=c["transfer_noise_floor"])
                    if not args.transfer_actor_only:
                        runner.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
                    report["transfer"] = {"checkpoint_sha256": digest(args.transfer),
                        "source_contract_sha256": digest(source_contract_path),
                        "source_updates": checkpoint["infos"]["successful_updates_total"],
                        "optimizer": "fresh", "critic": "fresh" if args.transfer_actor_only else "transferred",
                        "scope": "compatible_V5_actor_transfer" if args.transfer_actor_only else "compatible_V5_weights_scene_transfer"}
                    report["transfer"]["exploration_std_floor"] = c.get("transfer_noise_floor")
                    report["transfer"]["asset_migration"] = asset_migration
                    report["transfer"]["observation_migration"] = observation_migration
                    report["transfer"]["wheel_action_clip_old"] = old_contract["v5_control"].get("wheel_action_clip", old_contract["v5_control"]["action_clip"])
                    report["transfer"]["wheel_action_clip_new"] = c["v5_control"].get("wheel_action_clip", c["v5_control"]["action_clip"])
                    if c.get("transfer_curriculum") == "shared_frontiers" and env.performance_curriculum is not None:
                        saved_course = checkpoint["infos"].get("curriculum_state")
                        if saved_course is not None:
                            report["transfer"]["curriculum_groups_inherited"] = env.performance_curriculum.inherit_frontiers(saved_course)
                            env.resample_commands(torch.arange(args.num_envs, device=args.device), reset_height=True)
                if args.resume:
                    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
                    infos = checkpoint.get("infos", {})
                    migrating = args.resume_physics_change or args.resume_reward_change or args.resume_budget_change or args.resume_response_change or args.resume_diagnostic or args.resume_task_change
                    if migrating:
                        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                        from wheeled_tasks.chassis.policy_transfer import validate_physics_resume, validate_reward_resume, validate_budget_resume
                        from wheeled_tasks.chassis.diagnostic_curriculum import validate_diagnostic_resume
                        source_path = checkpoint_contract_path(args.resume)
                        if infos.get("contract_sha256") != digest(source_path):
                            raise ValueError("Resume source contract provenance mismatch")
                        branch = c.get("continuation") or c.get("remediation")
                        if branch and digest(source_path) != branch["source_contract_sha256"]:
                            raise ValueError("Source contract differs from the sealed learning origin")
                        for key in ("asset_manifest_sha256", "control_math_sha256"):
                            if args.resume_response_change and key == "control_math_sha256":
                                # Both original and target control identities are
                                # authenticated below before the precise diff.
                                continue
                            if args.resume_task_change and key == "control_math_sha256":
                                if infos.get(key) != json.loads(source_path.read_text())[key]:
                                    raise ValueError("Task source control identity differs from checkpoint")
                                # The named task validator below authenticates both controls
                                # and permits only the declared feedback/physics clock change.
                                continue
                            if infos.get(key) != identity[key]:
                                raise ValueError(f"Learning-state resume changes {key}")
                        resume_key = ("task_resume" if args.resume_task_change else "response_resume" if args.resume_response_change else "diagnostic_resume" if args.resume_diagnostic else "budget_resume" if args.resume_budget_change else
                                      "reward_resume" if args.resume_reward_change else "physics_resume")
                        validate = (validate_diagnostic_resume if args.resume_diagnostic else validate_budget_resume if args.resume_budget_change else
                                    validate_reward_resume if args.resume_reward_change else validate_physics_resume)
                        if args.resume_task_change:
                            from wheeled_tasks.chassis.task_campaign import validate_task_resume
                            validate = validate_task_resume
                        if args.resume_response_change:
                            from wheeled_tasks.chassis.response_migration import response_checkpoint_provenance
                            _, _, report[resume_key] = response_checkpoint_provenance(args.resume, infos, c, ROOT)
                        else:
                            report[resume_key] = validate(json.loads(source_path.read_text()), c)
                        if args.resume_diagnostic:
                            plan = c["diagnostic_study"]["plan"]
                            if (digest(source_path) != plan["source_contract_sha256"]
                                    or digest(args.resume) != plan["source_checkpoint_sha256"]
                                    or infos["successful_updates_total"] != plan["source_updates"]):
                                raise ValueError("Diagnostic arms must start from the same authenticated750 checkpoint")
                        report[resume_key].update(source_contract_sha256=digest(source_path),
                            target_contract_sha256=identity["contract_sha256"])
                    elif any(infos.get(k) != v for k, v in identity.items()):
                        raise ValueError("Parent checkpoint has a different task/asset/control interface")
                    runner.load(str(args.resume))
                    if "entropy_coef" in infos:
                        parent_entropy = validate_entropy_coef(infos["entropy_coef"])
                        if not migrating and parent_entropy != cfg.algorithm.entropy_coef:
                            raise ValueError("Same-contract resume changed its saved entropy coefficient")
                    from wheeled_tasks.chassis.policy_transfer import verify_learning_state_restore, resume_budget
                    restored = verify_learning_state_restore(runner.alg, checkpoint)
                    report.update(resume_budget(infos, args.consumed_updates, args.num_envs * cfg.num_steps_per_env))
                    report["parent_checkpoint_sha256"] = digest(args.resume)
                    if args.resume_diagnostic:
                        report["diagnostic_origin"] = {"source_formal_updates": report["parent_updates"],
                            "source_training_transitions": report["parent_training_transitions"],
                            "checkpoint_sha256": report["parent_checkpoint_sha256"],
                            "arm": c["diagnostic_study"]["arm"]}
                        report.update(parent_updates=0, parent_training_transitions=0)
                    elif infos.get("diagnostic_origin"):
                        report["diagnostic_origin"] = infos["diagnostic_origin"]
                    if migrating:
                        report[resume_key].update(restored)
                    (args.run_dir / "resume_verification.json").write_text(json.dumps({
                            **(report[resume_key] if migrating else restored),
                            "parent_checkpoint_sha256": report["parent_checkpoint_sha256"],
                            "parent_updates": report["parent_updates"],
                            "restored_checkpoint_updates": report["restored_checkpoint_updates"],
                            "learning_lineage_updates": report["learning_lineage_updates"],
                            "requested_learning_rate_scale": args.learning_rate_scale,
                            "restored_optimizer_learning_rate": runner.alg.optimizer.param_groups[0]["lr"],
                            "diagnostic_origin": report.get("diagnostic_origin"),
                            "parent_training_transitions": report["parent_training_transitions"]}, indent=2) + "\n")
                    runner.current_learning_iteration = report["parent_updates"]
                    if "rng_state" in infos:
                        torch.set_rng_state(infos["rng_state"])
                        env.generator.set_state(infos["env_rng_state"])
                        if infos.get("cuda_rng_states"):
                            torch.cuda.set_rng_state_all(infos["cuda_rng_states"])
                    if env.performance_curriculum is not None:
                        if infos.get("curriculum_state") is None:
                            raise ValueError("Performance curriculum resume requires its saved state")
                        env.performance_curriculum.load_state_dict(infos["curriculum_state"])
                env.training_transitions = report["parent_training_transitions"]
                saved_clock = infos.get("stage_actor_update", args.actor_updates_consumed) if args.resume else 0
                if args.stage_actor_update is not None and args.stage_actor_update != saved_clock:
                    raise ValueError("Requested curriculum clock differs from the resumed checkpoint")
                stage_actor_start = saved_clock
                env.stage_actor_update = stage_actor_start
                env.global_actor_update = args.global_actor_update_offset + stage_actor_start
                task_actor_start = 0
                if c.get("task_campaign"):
                    same_task = infos.get("task_campaign", {}).get("domain") == c["task_campaign"]["domain"]
                    task_actor_start = infos.get("task_actor_update") if same_task else 0
                    if type(task_actor_start) is not int or args.task_actor_update != task_actor_start:
                        raise ValueError("Task curriculum clock differs from the sealed learning origin")
                    env.task_actor_update = task_actor_start
                    report.update(task_campaign=c["task_campaign"], task_actor_update=task_actor_start)
                env.confirmed_jump_bins_m = confirmed_bins
                report.update(stage_actor_update=stage_actor_start,
                    global_actor_update_offset=args.global_actor_update_offset,
                    confirmed_jump_bins_m=confirmed_bins)
                if args.resume or confirmed_bins:
                    # Constructor resets precede checkpoint loading. Re-sample
                    # at the restored curriculum clock, not at iteration zero.
                    ids = torch.arange(args.num_envs, device=args.device)
                    if (args.resume and env.recovery is not None
                            and env.recovery.pose_curriculum is not None):
                        env.reset(ids)
                    else:
                        env.resample_commands(ids, reset_height=True)
                        env.update_targets()
                        env.update_command_reference()
                if "locomotion_difficulty" in c:
                    probe = env.skills.verify_difficulty_samples()
                    report["locomotion_difficulty_resume_probe"] = probe
                    (args.run_dir / "locomotion_difficulty_resume_probe.json").write_text(
                        json.dumps(probe, indent=2, allow_nan=False) + "\n")
                before_actor = {k: v.detach().clone() for k, v in runner.alg.actor.state_dict().items()}
                warmup_updates = c.get("critic_warmup_updates", 0)
                if args.resume:
                    remaining = resume_critic_warmup(c, infos, task_change=args.resume_task_change,
                        reward_change=args.resume_reward_change, contract_sha256=identity["contract_sha256"])
                    if args.resume_diagnostic:
                        remaining = c["diagnostic_study"]["plan"]["critic_adaptation_updates"]
                    warmup_updates = report["parent_updates"] + remaining
                report["learning_rate"] = configure_learning_rate(
                    runner.alg, cfg.algorithm.learning_rate, args.learning_rate_scale)
                report["initial_learning_rate"] = report["learning_rate"]
                report["initial_entropy_coef"] = verify_entropy_coef(runner.alg, cfg.algorithm.entropy_coef)
                if args.resume:
                    receipt_path = args.run_dir / "resume_verification.json"
                    receipt = json.loads(receipt_path.read_text())
                    receipt.update(effective_learning_rate=report["initial_learning_rate"],
                                   learning_rate_override_applied=args.learning_rate_scale is not None,
                                   parent_entropy_coef=infos.get("entropy_coef"),
                                   effective_entropy_coef=report["initial_entropy_coef"],
                                   entropy_coef_source=report["entropy_coef_source"])
                    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                if args.resume_diagnostic and runner.alg.learning_rate != c["diagnostic_study"]["plan"]["learning_rate"]:
                    raise ValueError("Diagnostic source learning rate differs from the fixed study")
                runner.alg.actor.requires_grad_(report["parent_updates"] >= warmup_updates)
                recovery_rollout = configure_recovery_rollout(runner.alg, env)
                original_update, original_step, original_save = runner.alg.update, env.step, runner.save
                pending_assessment = None
                if c.get("task_campaign"):
                    from wheeled_tasks.chassis.training_timing import TrainingTiming
                    timing = TrainingTiming()
                learning_schedule = runner.alg.schedule
                policy_probe = None
                learning_audit = None
                if c.get("remediation") or c.get("continuation") or c.get("task_campaign"):
                    from wheeled_tasks.chassis.learning_audit import LearningAudit
                    learning_audit = LearningAudit(env, runner.alg)
                if c.get("diagnostic_logging"):
                    from wheeled_tasks.chassis.diagnostic_curriculum import PolicyUpdateProbe
                    policy_probe = PolicyUpdateProbe(runner.alg.actor, runner.alg.storage, env.skills.batches)

                def update(*a, **kw):
                    nonlocal pending_assessment
                    report["attempted_updates"] = report["successful_updates"] + 1
                    if c.get("task_campaign"):
                        # Reserve before optimizer work so an abrupt process loss
                        # cannot silently refund a possibly completed update.
                        previous = args.run_dir / "progress.json"
                        reservation = json.loads(previous.read_text()) if previous.exists() else {
                            "pid": os.getpid(), "parent_updates": report["parent_updates"],
                            "successful_updates": report["successful_updates"], "actor_updates_in_block": report["actor_updates_in_block"]}
                        reservation["attempted_updates"] = report["attempted_updates"]
                        write_progress(args.run_dir, reservation, timing)
                    actor_enabled = report["parent_updates"] + report["successful_updates"] >= warmup_updates
                    verify_entropy_coef(runner.alg, cfg.algorithm.entropy_coef)
                    # A frozen actor has near-zero KL; adaptive scheduling would
                    # otherwise inflate the learning rate during critic warmup.
                    runner.alg.schedule = learning_schedule if actor_enabled else "fixed"
                    if policy_probe is not None:
                        policy_probe.capture()
                    if learning_audit is not None:
                        timed_call(timing, "diagnostics", learning_audit.capture)
                    result = timed_call(timing, "optimizer", original_update, *a, **kw)
                    report["successful_updates"] += 1
                    report["actor_updates_in_block"] += int(actor_enabled)
                    if (c.get("remediation") or c.get("continuation") or c.get("task_campaign")) and (runner.alg.learning_rate != c["learning_rate"]
                            or any(g["lr"] != c["learning_rate"] for g in runner.alg.optimizer.param_groups)):
                        raise RuntimeError("Fixed remediation learning rate changed during optimization")
                    env.stage_actor_update = stage_actor_start + report["actor_updates_in_block"]
                    env.global_actor_update = args.global_actor_update_offset + env.stage_actor_update
                    if c.get("task_campaign"):
                        env.task_actor_update = task_actor_start + report["actor_updates_in_block"]
                        report["task_actor_update"] = env.task_actor_update
                    report.update(stage_actor_update=env.stage_actor_update,
                        global_actor_update=env.global_actor_update,
                        actor_updates_consumed=args.actor_updates_consumed + report["actor_updates_in_block"])
                    # The optimizer has returned: retain its real clocks even
                    # if a post-update invariant fails. Never seal that state.
                    report["entropy_coef"] = verify_entropy_coef(runner.alg, cfg.algorithm.entropy_coef)
                    if recovery_rollout is not None:
                        report["recovery_rollout"] = dict(recovery_rollout.metrics)
                        report["policy_learning_transitions_in_block"] = (
                            report.get("policy_learning_transitions_in_block", 0)
                            + recovery_rollout.metrics["valid_transitions"])
                        report["script_or_blend_transitions_in_block"] = (
                            report.get("script_or_blend_transitions_in_block", 0)
                            + recovery_rollout.metrics["script_or_blend_transitions"])
                        if runner.logger.writer is not None:
                            for name, value in recovery_rollout.metrics.items():
                                runner.logger.writer.add_scalar(f"RecoveryRollout/{name}", value,
                                    report["parent_updates"] + report["successful_updates"] - 1)
                    if policy_probe is not None:
                        diagnostic = {**policy_probe.metrics(), **env.diagnostic_reward_metrics()}
                        values = torch.stack(list(diagnostic.values())).detach().cpu().tolist()
                        for tag, value in zip(diagnostic, values):
                            runner.logger.writer.add_scalar(tag, value, report["parent_updates"] + report["successful_updates"] - 1)
                    if learning_audit is not None:
                        audit = timed_call(timing, "diagnostics", learning_audit.drain)
                        if runner.logger.writer is not None:
                            for tag, value in audit.items():
                                runner.logger.writer.add_scalar(tag, value, report["parent_updates"] + report["successful_updates"] - 1)
                    runner.alg.actor.requires_grad_(report["parent_updates"] + report["successful_updates"] >= warmup_updates)
                    env.training_transitions = report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * cfg.num_steps_per_env
                    progress = {"updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                        "successful_updates": report["successful_updates"], "parent_updates": report["parent_updates"],
                        "attempted_updates": report["attempted_updates"],
                         "num_envs": args.num_envs, "stage": args.stage,
                         "actor_updates_in_block": report["actor_updates_in_block"],
                         "stage_actor_update": env.stage_actor_update,
                         "global_actor_update": env.global_actor_update,
                         "global_budget_actor_updates": args.global_actor_update_offset + args.actor_updates_consumed + report["actor_updates_in_block"],
                         "optimizer_phase": "actor_and_critic" if actor_enabled else "critic_warmup",
                         "entropy_coef": report["entropy_coef"],
                         "entropy_coef_source": report["entropy_coef_source"],
                        "diagnostic_origin": report.get("diagnostic_origin"),
                        "remediation": c.get("remediation"), "continuation": c.get("continuation"),
                        "training_transitions": report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * cfg.num_steps_per_env}
                    if c.get("task_campaign"):
                        progress.update(task_actor_update=env.task_actor_update, task_campaign=c["task_campaign"])
                    if recovery_rollout is not None:
                        progress.update({name: report[name] for name in (
                            "recovery_rollout", "policy_learning_transitions_in_block",
                            "script_or_blend_transitions_in_block")})
                    write_progress(args.run_dir, progress, timing)
                    if env.performance_curriculum is not None and report["successful_updates"] % 10 == 0:
                        if runner.logger.writer is not None:
                            for tag, value in env.performance_curriculum.report().items():
                                runner.logger.writer.add_scalar(tag, value, report["parent_updates"] + report["successful_updates"] - 1)
                        temp = args.run_dir / "curriculum_state.tmp"
                        temp.write_text(json.dumps(env.performance_curriculum.state_dict(), indent=2, allow_nan=False))
                        temp.replace(args.run_dir / "curriculum_state.json")
                    completed = report["parent_updates"] + report["successful_updates"]
                    changes = env.performance_curriculum.drain_changes() if hasattr(env.performance_curriculum, "drain_changes") else []
                    if changes:
                        progress["curriculum_changes"] = changes
                        with (args.run_dir / "curriculum_changes.jsonl").open("a") as history:
                            history.write(json.dumps({"successful_updates": completed, "changes": changes}, allow_nan=False) + "\n")
                    assessment_due = task_assessment_due(c, getattr(env, "task_actor_update", 0))
                    snapshot = None
                    if c.get("checkpoint_snapshots") and (
                            completed % c.get("checkpoint_interval", 100) == 0
                            or completed == c.get("checkpoint_first_update", 10)
                            or c.get("task_campaign") and report["successful_updates"] == 1
                            or c.get("task_campaign") and actor_enabled and report["actor_updates_in_block"] == 1
                            or changes or assessment_due):
                        from chassis_checkpoints import seal_checkpoint
                        snapshot = timed_call(timing, "checkpoint", seal_checkpoint,
                            args.run_dir, completed, runner.save, args.contract, progress)
                    if env.torque_monitor is not None and report["successful_updates"] % 10 == 0:
                        monitor = env.torque_monitor.report()
                        monitor["successful_updates"] = report["successful_updates"]
                        temp = args.run_dir / "torque_monitor.tmp"
                        temp.write_text(json.dumps(monitor, indent=2, allow_nan=False))
                        temp.replace(args.run_dir / "torque_monitor.json")
                        with (args.run_dir / "torque_history.jsonl").open("a") as history:
                            history.write(json.dumps(monitor, allow_nan=False) + "\n")
                    if metrics is not None and report["successful_updates"] % 10 == 0:
                        behavior = {"successful_updates": report["successful_updates"], **metrics.report()}
                        if runner.logger.writer is not None:
                            for group, values in behavior["groups"].items():
                                for name, value in values.items():
                                    if isinstance(value, (int, float)):
                                        runner.logger.writer.add_scalar(f"Behavior/{group}/{name}", value, completed - 1)
                        temp = args.run_dir / "behavior_metrics.tmp"
                        temp.write_text(json.dumps(behavior, indent=2, allow_nan=False) + "\n")
                        temp.replace(args.run_dir / "behavior_metrics.json")
                        with (args.run_dir / "behavior_history.jsonl").open("a") as history:
                            history.write(json.dumps(behavior, allow_nan=False) + "\n")
                    if assessment_due:
                        if snapshot is None:
                            raise RuntimeError("Task assessment requires an immutable snapshot")
                        # Log the completed PPO update before yielding the GPU.
                        pending_assessment = (snapshot / "model.pt", env.task_actor_update, completed)
                    return result

                last_publish = 0.

                def step(actions):
                    nonlocal last_publish
                    budget.check()
                    transition = timed_call(timing, "env_step", original_step, actions)
                    if learning_audit is not None:
                        timed_call(timing, "diagnostics", learning_audit.observe,
                            transition[3]["diagnostics"], transition[3]["native_reward_components"])
                    if c.get("evaluation", {}).get("regression_recovery"):
                        transition[3]["log"].update({
                            "/recovery/learning_rate": runner.alg.learning_rate,
                            "/recovery/discarded_updates_charged": report.get("discarded_updates_charged", 0)})
                    if metrics is not None:
                        timed_call(timing, "diagnostics", metrics.observe, transition[3]["diagnostics"])
                    if args.publish_state and time.monotonic() - last_publish >= .25:
                        state = {**identity, "wall_time_unix": time.time(), "run_id": args.run_dir.name,
                            "snapshot_kind": "post_step_after_auto_reset", "env_id": 0,
                            "sim_time_s": env.tick * c["policy_dt"], "episode_step": int(env.episode_length_buf[0]),
                            "body_names": list(env.robot.body_names), "quaternion_order": "xyzw",
                            "body_link_pose_w": env.robot.data.body_link_pose_w.torch[0].detach().cpu().tolist(),
                            "commands": env.commands[0].detach().cpu().tolist()}
                        selected = [env.scene_groups.index(name) for name in dict.fromkeys(env.scene_groups)]
                        all_poses = env.robot.data.body_link_pose_w.torch[selected].detach().cpu().tolist()
                        state["environments"] = [{"env_index": i, "scene_group": env.scene_groups[i],
                            "terrain": env.kinds[i], "origin": env.origins[i].cpu().tolist(),
                            "surfaces": [vars(s) for s in env.surfaces[i]],
                            "body_link_pose_w": poses, "commands": env.commands[i].cpu().tolist(),
                            "episode_step": int(env.episode_length_buf[i])} for i, poses in zip(selected, all_poses)]
                        temp = args.run_dir / "live_state.tmp"
                        temp.write_text(json.dumps(state, allow_nan=False))
                        temp.replace(args.run_dir / "live_state.json")
                        last_publish = time.monotonic()
                    return transition

                def save(path, infos=None):
                    entropy_coef = verify_entropy_coef(runner.alg, cfg.algorithm.entropy_coef)
                    timed_call(timing, "checkpoint", original_save, path, infos={**identity, "stage": args.stage,
                         "entropy_coef": entropy_coef,
                         "successful_updates_total": report["parent_updates"] + report["successful_updates"],
                         "learning_lineage_updates": report.get("learning_lineage_updates", 0) + report["successful_updates"],
                         "critic_warmup_remaining": max(0, warmup_updates - report["parent_updates"] - report["successful_updates"]),
                         "stage_actor_update": env.stage_actor_update,
                         "global_actor_update_offset": args.global_actor_update_offset,
                         "global_actor_update": env.global_actor_update,
                         "actor_updates_consumed": args.actor_updates_consumed + report["actor_updates_in_block"],
                         **({"task_campaign": c["task_campaign"], "task_actor_update": env.task_actor_update}
                            if c.get("task_campaign") else {}),
                         "confirmed_jump_bins_m": confirmed_bins,
                         "batch_transitions": args.num_envs * cfg.num_steps_per_env,
                         "diagnostic_origin": report.get("diagnostic_origin"),
                        "training_transitions": report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * cfg.num_steps_per_env,
                        "rng_state": torch.get_rng_state(), "env_rng_state": env.generator.get_state(),
                        "curriculum_state": env.performance_curriculum.state_dict() if env.performance_curriculum is not None else None,
                        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})

                    checkpoint_path = Path(path)
                    checkpoint_path.with_suffix(".contract.json").write_bytes(args.contract.read_bytes())
                    checkpoint_path.with_suffix(".checkpoint.json").write_text(json.dumps({
                        "schema": "chassis-learning-checkpoint-v1", "sealed": True,
                        "checkpoint_sha256": digest(checkpoint_path),
                        "contract_sha256": identity["contract_sha256"],
                        "entropy_coef": entropy_coef,
                        "learning_updates": report["parent_updates"] + report["successful_updates"],
                        "learning_lineage_updates": report.get("learning_lineage_updates", 0) + report["successful_updates"],
                        "stage_actor_update": env.stage_actor_update,
                        "global_actor_update_offset": args.global_actor_update_offset,
                        "global_actor_update": env.global_actor_update,
                        "actor_updates_consumed": args.actor_updates_consumed + report["actor_updates_in_block"],
                        **({"task_campaign": c["task_campaign"], "task_actor_update": env.task_actor_update}
                           if c.get("task_campaign") else {}),
                        "confirmed_jump_bins_m": confirmed_bins,
                        "gate_result": "not_assessed", "role": "learning_state",
                        "budget_consumption_is_separate_from_checkpoint_lineage": True}, indent=2) + "\n")

                runner.alg.update, env.step, runner.save = update, step, save
                if timing is not None:
                    original_log, original_returns = runner.logger.log, runner.alg.compute_returns

                    def compute_returns(*a, **kw):
                        return timed_call(timing, "returns", original_returns, *a, **kw)

                    def log(*a, **kw):
                        nonlocal pending_assessment, confirmed_bins
                        timed_call(timing, "logger", original_log, *a, **kw)
                        decision = None
                        if pending_assessment is not None:
                            checkpoint, task_update, consumed = pending_assessment
                            if args.device.startswith("cuda"):
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                            decision = timed_call(timing, "assessment_wait", wait_for_task_assessment,
                                args.run_dir, checkpoint, task_update, consumed, budget.check)
                            pending_assessment = None
                            report["latest_assessment"] = decision
                            confirmed_bins = task_confirmed_bins(confirmed_bins, decision, allowed_bins)
                            env.confirmed_jump_bins_m = confirmed_bins
                            report["confirmed_jump_bins_m"] = confirmed_bins
                        values = timing.finish_iteration()
                        report["latest_timing"] = values
                        if runner.logger.writer is not None:
                            iteration = report["parent_updates"] + report["successful_updates"] - 1
                            for tag, value in values.items():
                                runner.logger.writer.add_scalar(tag, value, iteration)
                        if decision and decision["decision"] == "domain_qualified_stop":
                            report["stop_reason"] = "domain_qualified_stop"
                            raise PlannedStop("domain_qualified_stop")

                    runner.alg.compute_returns, runner.logger.log = compute_returns, log
                if c.get("evaluation", {}).get("regression_recovery") or c.get("diagnostic_study"):
                    # A same-contract full-state rollback point exists even when
                    # this stage started from an actor-only scene transfer.
                    initial = args.run_dir / "initial_state"
                    initial.mkdir()
                    runner.save(str(initial / "model_final.pt"))
                    (initial / "contract.json").write_bytes(args.contract.read_bytes())
                    (initial / "completion.json").write_text(json.dumps({
                        "status": "completed", "scope": "initial_learning_state",
                        "parent_updates": report["parent_updates"], "successful_updates": 0,
                        "checkpoint_sha256": digest(initial / "model_final.pt")}, indent=2) + "\n")
                if args.profile:
                    import cProfile
                    profiler = cProfile.Profile()
                    profiler.enable()
                runner.learn(args.updates)
                report["actor_changed"] = any(not torch.equal(v, before_actor[k]) for k, v in runner.alg.actor.state_dict().items())
                if report["actor_updates_in_block"] > 0 and not report["actor_changed"]:
                    raise RuntimeError("No actor parameter updates")
                if not all(torch.isfinite(p).all() for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()):
                    raise RuntimeError("Nonfinite trained parameters")
                report["status"] = "completed"
    except PlannedStop as exc:
        report.update(status="stopped", reason=exc.reason)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(args.run_dir / "profile.pstats"))
        if runner is not None:
            # Adaptive PPO can change LR within a block. Recovery must consume
            # the optimizer's final value, not the value at block entry.
            report["learning_rate"] = runner.alg.optimizer.param_groups[0]["lr"]
            native_entropy = runner.alg.entropy_coef
            try:
                report["entropy_coef"] = validate_entropy_coef(native_entropy)
            except ValueError:
                report.update(status="failed", entropy_coef=None,
                              entropy_coef_status="invalid_native_value",
                              entropy_coef_invalid_native_value=repr(native_entropy))
                report.setdefault("error", "Native PPO entropy coefficient is invalid at completion")
        if runner is not None and report["status"] in ("stopped", "failed"):
            runner.logger.stop_logging_writer()
        if runner is not None and report["successful_updates"] > 0 and report["status"] in ("completed", "stopped"):
            path = args.run_dir / "model_final.pt"
            runner.save(str(path))
            report["checkpoint_sha256"] = digest(path)
            try:
                from wheeled_algo.chassis_export import export_actor
                report["export"] = export_actor(runner.alg.actor, args.run_dir, identity,
                                                 env.get_observations()["policy"])
            except Exception:
                report.update(status="export_failed", export_error=traceback.format_exc())
                traceback.print_exc()
        if env is not None:
            report["metrics"] = env.summary()
            if metrics is not None:
                (args.run_dir / "behavior_metrics.json").write_text(json.dumps(metrics.report(), indent=2, allow_nan=False) + "\n")
            if env.torque_monitor is not None:
                torque_report = env.torque_monitor.report()
                (args.run_dir / "torque_monitor.json").write_text(json.dumps(torque_report, indent=2, allow_nan=False))
                report["torque_monitor_file"] = "torque_monitor.json"
            env.close()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if timing is not None:
            report["timing"] = timing.summary()
            (args.run_dir / "timing.json").write_text(json.dumps(report["timing"], indent=2, allow_nan=False) + "\n")
        report["final_learning_lineage_updates"] = report.get("learning_lineage_updates", 0) + report["successful_updates"]
        (args.run_dir / "completion.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("CHASSIS_COMPLETION", report["status"], report["successful_updates"], flush=True)
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] in ("completed", "scene_validated", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
