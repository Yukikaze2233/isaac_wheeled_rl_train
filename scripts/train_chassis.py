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

from wheeled_tasks.chassis.training_validation import (
    validate_numerical_failure_policy,
    validate_restored_policy_request,
    validate_restored_sampled_policy_request,
    bind_sampled_policy_inputs,
    validate_restored_sampled_policy,
    validate_restored_policy,
    prepare_rough_scene_validation,
    rough_scene_validation_frame,
    nonsteady_scene_validation_frame,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhysxCapacityErrorLatch:
    """Stop before learning from a step for which PhysX dropped interactions.

    The callback signature and add/remove_logger API follow Isaac Sim's bundled
    isaacsim.code_editor.python_server extension. Carbonite's
    LoggingSettingsUtils.h maps /log/async=false to synchronous logger delivery;
    callbacks only latch evidence, then Python raises outside the native callback.
    """

    def __init__(self, logging, error_level):
        self._logging = logging
        self._error_level = error_level
        self.summary = {"logging_async": False, "count": 0, "first_error": None,
                        "raised_at": None}
        self._handle = logging.add_logger(self.on_log)

    @classmethod
    def install(cls):
        import carb.logging
        import carb.settings
        carb.settings.get_settings().set_bool("/log/async", False)
        carb.logging.configure_logging()
        return cls(carb.logging.acquire_logging(), carb.logging.LEVEL_ERROR)

    def on_log(self, source, level, filename, line_number, message):
        text = message.lower()
        if level < self._error_level or "physx" not in (source + " " + message).lower():
            return
        if not ("pxgpudynamicsmemoryconfig" in text
                or "simulation will miss interactions" in text
                or "buffer overflow" in text
                or ("capacity" in text and any(word in text for word in
                    ("increase", "exceed", "overflow", "insufficient")))):
            return
        self.summary["count"] += 1
        if self.summary["first_error"] is None:
            self.summary["first_error"] = {"source": source, "level": level,
                "filename": filename, "line_number": line_number, "message": message}

    def raise_if_failed(self, boundary):
        error = self.summary["first_error"]
        if error is not None:
            if self.summary["raised_at"] is None:
                self.summary["raised_at"] = boundary
            raise RuntimeError(f"PhysX capacity error at {boundary}: {error['message']}")

    def wrap_step(self, step):
        def checked_step(*args, **kwargs):
            self.raise_if_failed("before_env_step")
            transition = step(*args, **kwargs)
            self.raise_if_failed("after_env_step")
            return transition
        return checked_step

    def close(self):
        if self._handle is not None:
            self._logging.remove_logger(self._handle)
            self._handle = None


def timed_call(timer, phase, function, *args, **kwargs):
    if timer is None:
        return function(*args, **kwargs)
    with timer.measure(phase):
        return function(*args, **kwargs)


def log_scalar_batch(writer, values, step):
    """Batch drained Python scalars without changing tags or writer backends."""
    if writer is None or not values:
        return
    from itertools import islice
    from tensorboard.compat.proto.summary_pb2 import Summary
    from torch.utils.tensorboard import SummaryWriter

    if type(writer) is not SummaryWriter:
        for tag, value in values.items():
            writer.add_scalar(tag, value, step)
        return
    items = iter(values.items())
    while batch := list(islice(items, 512)):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)
                                 for tag, value in batch])
        writer._get_file_writer().add_summary(summary, step)


def incoming_action_clipping_metrics(config, actions, loss_population_mask):
    """Read raw policy-order storage once/update, using the direct-loss mask.

    Four leg axes precede two wheels. Incoming actions have already passed the
    entry's finite check, but may differ from delayed/held applied commands.
    Target excess measures the SAME raw clamp in rad/rad_s, before leg angle
    wrapping, not a second target clamp. Empty populations are explicitly marked.
    No inputs are mutated and no counters survive an update.
    """
    import torch
    control = config["v5_control"]
    leg_bound = control["action_clip"]
    wheel_bound = control.get("wheel_action_clip", leg_bound)
    bounds = actions.new_tensor([leg_bound] * 4 + [wheel_bound] * 2)
    mask = loss_population_mask.reshape(-1)
    excess = torch.where(mask[:, None],
        (actions.detach().reshape(-1, 6).abs() - bounds).clamp_min(0), 0.)
    # Exactly 13 aggregate scalars cross to the host; never save a rollout copy.
    values = torch.cat((mask.sum().reshape(1).double(), (excess > 0).sum(0).double(),
                        excess.sum(0, dtype=torch.float64))).cpu().tolist()
    samples, counts, excess = int(values[0]), values[1:7], values[7:]
    prefix = "ActionClipping/incoming_direct_loss/"
    result = {prefix + "loss_transitions": samples,
              prefix + "has_loss_samples": int(samples > 0)}
    for group, start, stop, scale, unit in (
            ("legs", 0, 4, abs(control["leg_position_scale"]), "rad"),
            ("wheels", 4, 6, abs(control["wheel_velocity_scale"]), "rad_s")):
        denominator = samples * (stop - start)
        count = int(sum(counts[start:stop]))
        tag = prefix + group + "/"
        result.update({tag + "axis_samples": denominator, tag + "raw_clip_count": count,
            tag + "raw_clip_fraction": count / max(denominator, 1),
            tag + "target_clip_excess_mean_" + unit: sum(excess[start:stop]) * scale / max(denominator, 1)})
    return result


def fixed_flat_rehearsal_assignment(spec, num_envs, device):
    """One worker's static lane labels; nominal allocation is not PPO exposure."""
    import torch
    names = tuple(spec["case_ids"])
    indices = tuple(spec["case_index"])
    background = spec.get("background_group", "rough")
    all_flat = (background == "ordinary_unrehearsed" and spec.get("flat_refresh") is True
                and not spec.get("flat_mixed_refresh")
                and len(indices) == num_envs and all(type(i) is int and i >= 0 for i in indices))
    mixed_flat = (background == "ordinary_unrehearsed" and spec.get("flat_refresh") is True
                  and spec.get("flat_mixed_refresh") is True and num_envs == 4096
                  and len(names) == 17 and len(indices) == num_envs
                  and indices.count(-1) == 3072 and spec.get("rough_lanes") == 0
                  and spec.get("recovery_lane_count") == 0 and spec.get("ordinary_lane_count") == num_envs
                  and spec.get("background_lanes") == spec.get("ordinary_unrehearsed_lane_count") == 3072
                  and spec.get("rehearsal_lanes") == 1024 and spec.get("actual_lane_fraction") == .25
                  and spec.get("origins_by_env") is None
                  and spec.get("settings", {}).get("version") == "v7_fixed17_plane_rehearsal_v2"
                  and spec["settings"].get("fraction") == .25
                  and spec["settings"].get("reward_profile") == "scut0719_flat_v1"
                  and sorted(indices.count(i) for i in range(17)) == [60] * 13 + [61] * 4)
    if (background not in ("rough", "non_rehearsal") and not (all_flat or mixed_flat)
            or not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or background in names
            or len(indices) != num_envs
            or any(type(i) is not int or not -1 <= i < len(names) for i in indices)):
        raise ValueError("Fixed flat rehearsal requires unique case names and fixed native-order lane indices")
    groups = (background, *names)
    counts = [indices.count(i) for i in range(-1, len(names))]
    return torch.tensor([i + 1 for i in indices], dtype=torch.int64, device=device), {
        "scope": "current_worker_successful_actor_updates_only_not_resumed",
        "population": "unique transitions in recovery_rollout.loss_population_mask; no epoch multiplier",
        **({"background_commands": "original dynamic source commands without fixed overrides; included in ordinary PPO"}
           if mixed_flat else {}),
        "nominal_lane_counts": dict(zip(groups, counts)),
        "nominal_lane_fractions": {name: n / num_envs for name, n in zip(groups, counts)},
        "successful_actor_updates": 0,
        "last_update_direct_loss_transitions": dict.fromkeys(groups, 0),
        "worker_direct_loss_transitions": dict.fromkeys(groups, 0),
    }


def fixed_flat_rehearsal_counts(loss_mask, lane_indices, bucket_count):
    """GPU reduction only; commit after PPO succeeds, never from a failed rollout."""
    import torch
    if (loss_mask.dtype != torch.bool or loss_mask.ndim != 2
            or lane_indices.dtype != torch.int64 or lane_indices.shape != (loss_mask.shape[1],)
            or lane_indices.device != loss_mask.device):
        raise ValueError("Rehearsal exposure requires the actual time-by-native-lane loss mask")
    result = torch.zeros(bucket_count, dtype=torch.int64, device=loss_mask.device)
    return result.scatter_add_(0, lane_indices, loss_mask.sum(0, dtype=torch.int64))


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


def next_formal_assessment_update(config, restored_learning_updates):
    """Choose one strictly future learning boundary, never a paid-budget clock."""
    if config.get("formal_v7_flat") is not True:
        return None
    settings = config.get("formal_stage_policy", {})
    mode = settings.get("assessment_mode", "pause_for_assessment")
    if type(mode) is not str or mode not in ("pause_for_assessment", "record_and_continue"):
        raise ValueError("Formal assessment_mode requires pause_for_assessment or record_and_continue")
    first = settings.get("first_assessment_update")
    interval = settings.get("assessment_interval_updates")
    if any(type(value) is not int or value <= 0 for value in (first, interval)):
        raise ValueError("Formal assessment requires positive first_assessment_update and assessment_interval_updates")
    if type(restored_learning_updates) is not int or restored_learning_updates < 0:
        raise ValueError("Formal assessment requires the restored nonnegative learning clock")
    return first if restored_learning_updates < first else first + (
        (restored_learning_updates - first) // interval + 1) * interval


def resume_critic_warmup(config, infos, *, contract_sha256=None, task_change=False, reward_change=False):
    """Adapt once to a named contract change; a fault resumes the remaining work."""
    if "formal_critic_adaptation" in config:
        from wheeled_tasks.chassis.critic_adaptation import remaining_updates
        return remaining_updates(config, infos)
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


def completion_learning_rate(optimizer):
    """Read a JSON-safe final rate, rejecting empty/inconsistent Adam groups."""
    rates = [group.get("lr") for group in optimizer.param_groups]
    if not rates or any(not isinstance(rate, (int, float)) or isinstance(rate, bool)
            or not math.isfinite(rate) or rate <= 0 for rate in rates) or any(
            rate != rates[0] for rate in rates):
        raise ValueError(f"Invalid optimizer learning rates at completion: {rates!r}")
    return rates[0]


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
            or env.cfg.get("nonstandard_start", {}).get("enabled", False)
            or getattr(env, "flat_episode", None) is not None):
        return None
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    return attach_recovery_rollout_mask(algorithm, num_envs=env.num_envs,
        require_terminal_observation="formal_recovery_episode_window" in env.cfg)


def log_recovery_outcomes(writer, recovery, step):
    """Write producer episode counts once, outside the physics/rollout loop."""
    curriculum = getattr(recovery, "pose_curriculum", None)
    snapshot = getattr(curriculum, "rollout_outcome_metrics", None)
    if writer is None or snapshot is None:
        return
    for tag, value in snapshot().items():
        writer.add_scalar(tag, value, step)


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
    """Audit exact teaching recipes and their separately bound optional revisions."""
    meta = config.get("task_campaign", {})
    if ("small_stair_revision_migration" in meta
            or "terrain_runout_revision_migration" in meta or "terrain_route_runout" in config):
        if "small_stair_revision_migration" in meta:
            from wheeled_tasks.chassis import small_stair_revision as recipe
        else:
            from wheeled_tasks.chassis import terrain_runout_revision as recipe

        source = recipe.read_bound(recipe.SOURCE_WORKER_FILE, recipe.SOURCE_WORKER_SHA, root)
        recipe.validate_worker_revision(source, config, root=root)
        recipe.validate_frozen_workers(root=root)
        exact = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
        if (exact(json.loads((Path(root) / recipe.PLAN_FILE).read_text())) != exact(recipe.revised_plan(root=root))
                or exact(json.loads((Path(root) / recipe.MANIFEST_FILE).read_text())) != exact(recipe.revised_manifest(root=root))):
            raise ValueError("Terrain source audit differs from its frozen plan or manifest")
        inherited = scut_curriculum_source_files(source, root)
        return list(dict.fromkeys(inherited + [
            "src/wheeled_tasks/chassis/terrain_runout.py",
            "src/wheeled_tasks/chassis/terrain_runout_revision.py",
            *(["src/wheeled_tasks/chassis/small_stair_revision.py"]
               if "small_stair_revision_migration" in meta else []),
            "src/wheeled_tasks/chassis/terrain_runout_runtime.py",
            "src/wheeled_tasks/chassis/task.py", "src/wheeled_tasks/chassis/env.py",
            "src/wheeled_tasks/chassis/preflight.py", "src/wheeled_tasks/chassis/evaluation.py",
            "src/wheeled_tasks/chassis/task_campaign.py", "scripts/train_chassis.py",
            "scripts/run_task_campaign.py", "scripts/chassis_remote_job.py",
            recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE, recipe.SOURCE_DESIGN_FILE,
            recipe.OWN_SOURCE_WORKER_FILE,
            recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE]))
    if "entropy_revision_migration" in meta:
        from wheeled_tasks.chassis import recovery_entropy_revision as recipe
        if (meta.get("domain") != "recovery"
                or meta["entropy_revision_migration"] != recipe.revision_binding()):
            raise ValueError("Recovery entropy source audit differs from its exact revision binding")
        validate_entropy_coef(meta["entropy_revision_migration"]["entropy_coef"])
        validate_entropy_coef(config.get("entropy_coef"))
        canonical = recipe.canonical_worker(root=root)
        if config != canonical:
            raise ValueError("Recovery entropy source audit differs from its full canonical worker")
        recipe.validate_frozen_workers(root=root)
        if (json.loads((Path(root) / recipe.PLAN_FILE).read_text()) != recipe.revised_plan(root=root)
                or json.loads((Path(root) / recipe.MANIFEST_FILE).read_text()) != recipe.revised_manifest(root=root)):
            raise ValueError("Recovery entropy source audit differs from its frozen plan or manifest")
        source = recipe.read_bound(recipe.SOURCE_WORKER_FILE, recipe.SOURCE_WORKER_SHA, root)
        inherited = scut_curriculum_source_files(source, root)
        return list(dict.fromkeys(inherited + [
            "src/wheeled_tasks/chassis/recovery_entropy_revision.py",
            recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE, recipe.SOURCE_DESIGN_FILE,
            recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE]))
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
        raise ValueError("reference teaching source audit differs from its exact revision binding")
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
        raise ValueError("reference teaching source audit requires a named revised task domain")
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


def preflight(path, *, asset_directory=None, physics_probe=False):
    c = json.loads(path.read_text())
    from wheeled_tasks.chassis.curriculum_selection import load_curriculum_definition
    load_curriculum_definition(c, ROOT)
    from wheeled_tasks.chassis.training_profile import validate_profile_selection
    validate_profile_selection(c)
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
    if c["contract_id"] == "v6-gas-spring-full-stage-research-v1" and not c.get("formal_v7_flat"):
        if c.get("requires_design_preflight") is not True or not isinstance(c.get("design_preflight"), dict):
            raise ValueError("V6 requires an independent design_preflight binding")
    if sum(s["updates"] for s in c["stages"]) != c["total_updates"]:
        raise ValueError("Curriculum budgets do not sum to total")
    bundle = ROOT / c["asset_directory"]
    if digest(bundle / "manifest.json") != c["asset_manifest_sha256"]:
        raise ValueError("Closed-chain manifest identity mismatch")
    manifest = json.loads((bundle / "manifest.json").read_text())
    from wheeled_tasks.chassis.flat_episode_selection import require_native_flat_admission
    require_native_flat_admission(c, manifest, physics_probe=physics_probe)
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
    parser.add_argument("--contract", type=Path, required=True, help="Training configuration JSON")
    parser.add_argument("--stage", choices=("foundation", "terrain", "speed", "up", "down", "jump", "mixed"), default="foundation")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--level", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--actor-updates-consumed", type=int)
    parser.add_argument("--stage-actor-update", type=int)
    parser.add_argument("--task-actor-update", type=int)
    parser.add_argument("--integrated-source-selection-file", type=Path)
    parser.add_argument("--integrated-source-selection-sha256")
    parser.add_argument("--global-actor-update-offset", type=int)
    parser.add_argument("--confirmed-jump-bins-json", default="[]")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--coverage", action="store_true", help="validation: include every stage terrain family")
    parser.add_argument("--validate-scene", type=int, default=0, metavar="STEPS")
    parser.add_argument("--validate-restored-policy", type=int, default=0, metavar="STEPS",
                        help="After full scripted-recovery resume, check bounded mean-policy steps with the formal 4096 environments; exit without PPO or checkpoints")
    parser.add_argument("--validate-restored-sampled-policy", type=int, default=0, metavar="STEPS",
                        help="After full branch107 resume sample24 physical policy steps, or48 for mount pressure; no PPO or checkpoints")
    parser.add_argument("--capture-numerical-failure", action="store_true",
                        help="Formal training: save 64 prior physical states at the first 5ms numerical crossing; abort worker by default")
    parser.add_argument("--numerical-failure-policy", choices=("abort-worker", "episode-reset"),
                        default="abort-worker",
                        help="Formal PPO only: abort worker, or preserve source per-episode numerical resets with capture; non-finite failures always abort")
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
    parser.add_argument("--rng-mode", choices=("inherit", "fresh_fork"), default="inherit",
                        help="Explicit diagnostic fork or first motion-course handoff; ordinary restarts inherit RNG")
    parser.add_argument("--resume-task-change", action="store_true",
                        help="Frozen task-domain migration with separate consumption and curriculum clocks")
    parser.add_argument("--consumed-updates", type=int, help="Same-batch budget already spent, including discarded updates")
    parser.add_argument("--budget-review-file", type=Path, help="Hash-pinned shared V7 budget evidence; not native authorization")
    parser.add_argument("--budget-review-sha256")
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--budget-attempt-id")
    parser.add_argument("--learning-rate-scale", type=float,
                        help="Explicit positive ratio to the contract's initial LR; may exceed one after adaptive updates")
    parser.add_argument("--transfer-actor-only", action="store_true", help="Reset critic when changing the reward/task distribution")
    parser.add_argument("--publish-state", action="store_true", help="Publish atomic selected-env real physics snapshots at most 4 Hz")
    parser.add_argument("--profile", action="store_true", help="Save bounded PPO sampling CPU profile before Kit shutdown")
    parser.add_argument("--max-runtime-seconds", type=float, default=172800.)
    args = parser.parse_args()
    requested_profile_clocks = {key: getattr(args, key) for key in
        ("stage_actor_update", "actor_updates_consumed", "global_actor_update_offset")}
    # Preserve the legacy defaults; new exact restarts distinguish omission
    # from an explicit request to overwrite a nonzero checkpoint clock.
    for key in ("actor_updates_consumed", "global_actor_update_offset"):
        if getattr(args, key) is None:
            setattr(args, key, 0)
    if not (1 <= args.num_envs <= 20000 and 1 <= args.updates <= 100000 and 0 <= args.level <= 1
            and 0 <= args.validate_scene <= 10000 and args.max_runtime_seconds > 0):
        parser.error("Invalid bounded run settings")
    c, manifest = preflight(args.contract, physics_probe=bool(args.validate_scene
        or args.validate_restored_policy or args.validate_restored_sampled_policy))
    numerical_failure_policy = validate_numerical_failure_policy(args, c)
    restored_policy_validation = validate_restored_policy_request(args, c, ROOT)
    sampled_policy_validation = validate_restored_sampled_policy_request(args, c)
    read_only_validation = bool(args.validate_scene or restored_policy_validation or sampled_policy_validation)
    if not args.validate_scene:
        next_formal_assessment_update(c, 0)  # Validate the declared cadence before Kit.
    if c.get("formal_closure_observation_only"):
        if not (c.get("formal_v7_flat") is True and args.validate_scene):
            parser.error("Closure observation is restricted to physics-only scene diagnosis; PPO is forbidden")
    if c.get("formal_learning_state_resume") and not args.resume:
        parser.error("Formal repair must resume its registered learning checkpoint")
    from wheeled_tasks.chassis.curriculum_selection import require_curriculum_launch
    curriculum_selection = require_curriculum_launch(c, resume=args.resume, transfer=args.transfer,
        num_envs=args.num_envs, rollout_steps=24 if "training_profile" in c else None)
    from wheeled_tasks.chassis.scut_reward_selection import validate_reward_selection
    reward_selection = validate_reward_selection(c, stage_name=args.stage)
    from wheeled_tasks.chassis.small_rough_course import FINAL_MOTION_VERSION, MOTION_VERSIONS
    final_motion = c.get("formal_small_rough_mix", {}).get("version") == FINAL_MOTION_VERSION
    from wheeled_tasks.chassis.training_profile import uses_profile, validate_profile_selection
    if validate_profile_selection(c):
        if args.transfer:
            parser.error("The78D campaign starts from zero; legacy weight transfer is forbidden")
        if args.resume and any((args.resume_physics_change, args.resume_reward_change,
                args.resume_budget_change, args.resume_response_change, args.resume_diagnostic, args.resume_task_change)):
            parser.error("The78D profile only permits an exact same-contract restart from its own checkpoint")
        if args.learning_rate_scale is not None:
            parser.error("The78D profile preserves its adaptive optimizer rate; explicit LR scaling requires separate review")
        if (not args.preflight_only and not args.validate_scene and not restored_policy_validation
                and not sampled_policy_validation and c.get("research_training_authorized") is not True):
            parser.error("This78D candidate has not passed the user-requested training-contract review")
        if not args.preflight_only and not args.validate_scene and args.num_envs != 4096:
            parser.error("The reviewed78D train entry requires4096 environments, including restored-policy checks")
    from wheeled_tasks.chassis.training_budget import require_budget_arguments
    budget_selected = require_budget_arguments(c, review_file=args.budget_review_file,
        review_sha256=args.budget_review_sha256, ledger_path=args.budget_ledger,
        attempt_id=args.budget_attempt_id, resume=args.resume, consumed_updates=args.consumed_updates)
    from wheeled_tasks.chassis.diagnostic_budget_selection import select_budget_scope
    budget_scope = select_budget_scope(c)
    if budget_scope == "v7_diagnostic81":
        expected_num_envs = c["diagnostic_study"]["num_envs"]
        if args.num_envs != expected_num_envs or c.get("target_num_envs") != expected_num_envs:
            parser.error("V7 diagnostic environment count must match its frozen study and target contract")
    if args.rng_mode == "fresh_fork" and not (args.resume and args.resume_diagnostic):
        from wheeled_tasks.chassis.small_rough_course import KEY, selection
        settings = selection(c) if KEY in c else None
        if (not args.resume or args.stage != "terrain" or settings is None
                or settings["version"] not in MOTION_VERSIONS):
            parser.error("Fresh RNG requires an authenticated diagnostic fork or first motion-course handoff")
    contract_config = c
    integrated_receipt = None
    integrated_design = None
    if c.get("integrated_production") or args.integrated_source_selection_file is not None or args.integrated_source_selection_sha256 is not None:
        from wheeled_tasks.chassis import integrated_production
        integrated_receipt = integrated_production.authenticate_invocation(c, args, ROOT)
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
    if c.get("integrated_production"):
        c = integrated_production.static_training_config(contract_config, ROOT)
    design_receipt = design_preflight(c, control, ROOT, stage=args.stage, role="train",
        seed=args.seed, num_envs=args.num_envs, block_index=args.block_index,
        physics_probe=read_only_validation)
    if c.get("integrated_production"):
        from wheeled_tasks.chassis.integrated_training import make_design_hook
        c = integrated_production.attach_runtime(contract_config, integrated_receipt, ROOT)
        integrated_design = make_design_hook(design_receipt, c, integrated_receipt, args)
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
        if budget_selected:
            parser.error("V7 budget preflight cannot certify GPU occupancy: authenticated meter owner is unavailable")
        print(json.dumps({"asset_verified": True, "contract": c["contract_id"],
                          "asset_directory": c["asset_directory"], "stage": args.stage,
                          "design_preflight": design_receipt,
                          "reward_selection": reward_selection,
                          "long_run_behavior_acceptance": "pending"}, indent=2))
        return 0
    if not args.research or args.run_dir is None:
        parser.error("--research and a new --run-dir are required")
    if args.capture_numerical_failure and not c.get("formal_v7_flat"):
        parser.error("--capture-numerical-failure requires the formal source training path")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    if c.get("formal_v7_flat"):
        c["formal_physics_trace_directory"] = str(args.run_dir)
        if args.validate_scene:
            c["formal_physics_trace_ticks"] = [1, 2, 3, 5, 10, 25, 50, args.validate_scene]
            if not c.get("formal_foundation_nominal_birth"):
                # Resolve first drive and the source's 40-tick motor-off
                # boundary in the existing physics-only diagnosis.
                c["formal_physics_trace_substeps"] = [1, 2, 4, 36, 39, 40, 41, 42, 44, 48]
    source_files = ["scripts/train_chassis.py", "src/wheeled_algo/training_control.py", "src/wheeled_tasks/chassis/env.py",
                    "src/wheeled_tasks/chassis/training_validation.py",
                    "src/wheeled_tasks/chassis/curriculum_selection.py",
                    "src/wheeled_tasks/chassis/autonomous_curriculum.py",
                    "src/wheeled_tasks/chassis/root_kinematics.py",
                    "src/wheeled_tasks/chassis/preflight.py",
                    "src/wheeled_tasks/chassis/control_contract.py",
                    "src/wheeled_tasks/chassis/task.py", "src/wheeled_tasks/chassis/control_core.py",
                    "src/wheeled_tasks/agents/chassis_ppo_cfg.py",
                    "src/wheeled_algo/chassis_export.py", c["control_math_source"]]
    if args.capture_numerical_failure:
        source_files.append("src/wheeled_tasks/chassis/numerical_failure_capture.py")
    if args.publish_state:
        source_files.append("src/wheeled_tasks/chassis/state_snapshot.py")
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
    if c.get("scut_heading_command", {}).get("enabled") and not c.get("evaluation_exact_cases"):
        from wheeled_tasks.chassis.heading_command import heading_source_files
        source_files.extend(heading_source_files(c, ROOT))
    if c.get("checkpoint_snapshots"):
        source_files.append("scripts/chassis_checkpoints.py")
    if c.get("actor_observation_source"):
        source_files.append("src/wheeled_tasks/chassis/scut_observation.py")
    source_files.append("src/wheeled_tasks/chassis/training_profile.py")
    from wheeled_tasks.chassis.training_profile import profile_source_files, profile_identity
    source_files.extend(profile_source_files(c))
    from wheeled_tasks.chassis.scut_reward_selection import reward_source_files
    source_files.extend(reward_source_files(c))
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
        source_files.extend(["src/wheeled_tasks/chassis/signal_delay.py",
                             "src/wheeled_tasks/chassis/delay_buffer.py"])
    if "feedback_transport" in c:
        source_files.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in
            ("feedback_transport", "feedback_transport_config", "pc_communication", "delay_buffer"))
    if c.get("contact_domain", {}).get("enabled"):
        source_files.append("src/wheeled_tasks/chassis/contact_domain.py")
    if c.get("command_reference"):
        source_files.append("src/wheeled_tasks/chassis/references.py")
    if (c.get("command_reference") or c.get("formal_stage_optimizer") or args.transfer
            or args.resume_physics_change or args.resume_response_change or args.resume_task_change):
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
        if not c.get("integrated_production"):
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
    if c.get("jump_actuation_measurement", {}).get("enabled"):
        source_files.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in
            ("env", "jump_actuation_evidence", "step_jump", "training_schedule", "v5_control", "root_kinematics"))
    if c.get("jump_physics_trace", {}).get("enabled"):
        source_files.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in
            ("env", "jump_physics_trace", "jump_actuation_evidence", "skill_commands", "root_kinematics"))
    if c.get("reward_episode_measurement", {}).get("enabled"):
        source_files.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in
            ("env", "reward_episode_journal", "rewards", "scut_flat_reward", "scut_flat_reward_runtime"))
    if c.get("new_asset_curriculum", {}).get("enabled"):
        source_files.extend(["src/wheeled_tasks/chassis/training_schedule.py",
            "src/wheeled_tasks/chassis/step_assist.py"])
    if c.get("recovery_training", {}).get("enabled"):
        source_files.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in
            ("recovery", "recovery_training", "recovery_evidence", "continuous_encoders", "mechanism_observer", "recovery_observer"))
        if (c["recovery_training"].get("pose_curriculum", {}).get("enabled")
                and not c.get("evaluation_exact_cases")):
            source_files.append("src/wheeled_tasks/chassis/recovery_curriculum.py")
        if c["recovery_training"].get("prefix_curriculum", {}).get("enabled"):
            source_files.extend(["src/wheeled_tasks/chassis/recovery_prefix_curriculum.py",
                "src/wheeled_tasks/chassis/recovery_curriculum.py"])
        source_files.append("src/wheeled_algo/recovery_rollout.py")
        source_files.extend(c["recovery_training"][key] for key in ("profiles_file", "height_lookup_file"))
    if c.get("integrated_production"):
        source_files.extend(integrated_production.production_source_files(contract_config, integrated_receipt, ROOT))
        source_files.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in
            ("integrated_training", "ordinary_posture_curriculum", "nonstandard_start",
             "step_assist", "terrain_runout", "training_schedule"))
    if budget_selected:
        source_files.extend(["src/wheeled_tasks/chassis/training_budget.py",
                             "src/wheeled_tasks/chassis/diagnostic_budget_selection.py"])
    from wheeled_tasks.chassis.diagnostic_budget_selection import (
        diagnostic_source_files, diagnostic_source_identity,
    )
    source_files.extend(diagnostic_source_files(c))
    from wheeled_tasks.chassis.curriculum_selection import curriculum_source_files
    source_files.extend(curriculum_source_files(c))
    source_files = list(dict.fromkeys(source_files))
    source_hashes = {}
    for name in source_files:
        data = (ROOT / name).read_bytes()
        source_hashes[name] = hashlib.sha256(data).hexdigest()
        target = args.run_dir / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    identity = {"contract_id": c["contract_id"], "contract_sha256": digest(args.contract),
                "asset_manifest_sha256": c["asset_manifest_sha256"],
                "control_math_sha256": digest(ROOT / c["control_math_source"]),
                **profile_identity(c, source_hashes),
                **diagnostic_source_identity(c, source_hashes)}
    report = {"started_at": datetime.now(timezone.utc).isoformat(), **identity,
              "source_sha256": source_hashes, "stage": args.stage, "status": "starting",
              "reward_selection": reward_selection,
               "successful_updates": 0, "attempted_updates": 0, "parent_updates": 0, "parent_training_transitions": 0,
               "actor_updates_in_block": 0,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    if c.get("integrated_production"):
        (args.run_dir / "integrated_admission.json").write_bytes(args.integrated_source_selection_file.read_bytes())
        (args.run_dir / "integrated_design.json").write_text(json.dumps(integrated_design, indent=2, allow_nan=False) + "\n")
        report["integrated_admission_sha256"] = args.integrated_source_selection_sha256
        from wheeled_tasks.chassis.integrated_training import learning_metadata
        report.update(learning_metadata(integrated_receipt, args.task_actor_update))
    (args.run_dir / "contract.json").write_bytes(args.contract.read_bytes())
    (args.run_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    launcher, env, runner, metrics, profiler, timing = None, None, None, None, None, None
    physx_errors = None
    budget_owner = None
    recovery_evidence = None
    source_birth_exposure = None
    diagnostic_budget_binding = None
    from wheeled_algo.training_control import TrainingBudget, PlannedStop
    budget = TrainingBudget(args.max_runtime_seconds)
    from wheeled_algo.training_control import TrainingControl
    training_control = TrainingControl(args.run_dir)
    try:
        if budget_selected:
            from wheeled_tasks.chassis.training_budget import BudgetOwner
            budget_identity = {
                "contract_sha256": identity["contract_sha256"],
                "training_source_sha256": identity["training_source_sha256"],
                "seed": args.seed, "stage": args.stage, "requested_updates": args.updates,
                "parent_checkpoint_sha256": (digest(args.resume)
                    if c.get("formal_learning_state_resume") and args.resume else None),
            }
            budget_resume_checkpoint = args.resume
            if budget_scope == "v7_diagnostic81":
                from wheeled_tasks.chassis.diagnostic_budget_selection import bind_diagnostic_budget
                source_contract_file = None
                if args.resume_diagnostic:
                    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                    source_contract_file = checkpoint_contract_path(args.resume)
                budget_identity["parent_checkpoint_sha256"] = c["diagnostic_study"]["plan"]["source_checkpoint_sha256"]
                diagnostic_budget_binding = bind_diagnostic_budget(c,
                    review_file=args.budget_review_file, review_sha256=args.budget_review_sha256,
                    ledger_path=args.budget_ledger, attempt_id=args.budget_attempt_id,
                    actual_identity=budget_identity, resume=args.resume,
                    resume_diagnostic=args.resume_diagnostic, source_contract_file=source_contract_file,
                    consumed_updates=args.consumed_updates, transfer=args.transfer,
                    lr_scale=args.learning_rate_scale, other_migrations=(args.resume_physics_change,
                        args.resume_reward_change, args.resume_budget_change,
                        args.resume_response_change, args.resume_task_change))
                budget_resume_checkpoint = diagnostic_budget_binding["budget_resume_checkpoint"]
                report["diagnostic_budget_selection"] = diagnostic_budget_binding
            budget_owner = BudgetOwner.authenticate(args.budget_review_file, args.budget_review_sha256,
                args.budget_ledger, args.budget_attempt_id, budget_identity,
                resume_checkpoint=budget_resume_checkpoint,
                evaluation_only=bool(args.validate_scene or restored_policy_validation or sampled_policy_validation))
            budget.max_runtime_seconds = budget_owner.execution_wall_limit(args.max_runtime_seconds)
            report["effective_invocation_wall_cap_seconds"] = budget.max_runtime_seconds
            if budget_resume_checkpoint is not None:
                budget_owner.verify_resume(budget_resume_checkpoint)
            report["training_budget"] = budget_owner.summary()
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": args.device,
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        physx_errors = PhysxCapacityErrorLatch.install()
        # Keep a live reference even if ChassisEnv construction raises before
        # assigning env: completion.json must still contain the native evidence.
        report["physx_capacity_errors"] = physx_errors.summary
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        from wheeled_tasks.chassis.env import ChassisEnv
        env = ChassisEnv(c, manifest, control, ROOT,
            stage_name=args.stage, num_envs=args.num_envs, device=args.device, level=args.level,
            seed=args.seed, coverage=args.coverage, block_index=args.block_index,
            physics_probe=read_only_validation,
            **({"integrated_design_preflight": integrated_design} if c.get("integrated_production") else {}))
        physx_errors.raise_if_failed("after_env_initialization")
        env.step = physx_errors.wrap_step(env.step)
        if args.capture_numerical_failure:
            capture_policy = {"stop_on_first_event": numerical_failure_policy == "abort-worker"}
            if numerical_failure_policy == "episode-reset":
                capture_policy["stop_on_nonfinite"] = True
            env.enable_numerical_failure_capture(**capture_policy,
                closure_observation_threshold_m=(.01 if c.get("formal_flat_refresh")
                    == "v7_flat_mixed_robustness_refresh_v1" else .03))
        report["startup"] = env.startup_report
        if c.get("record_diagnostics"):
            from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
            from wheeled_tasks.chassis.fixed_flat_rehearsal import FIXED17_VERSION
            height_range = c.get("height_workspace", {}).get("height_m")
            fixed_spec = getattr(env, "rehearsal_spec", None)
            metrics = EpisodeMetrics(env.scene_groups, args.device, c["policy_dt"],
                height_range_m=[height_range[0], height_range[-1]] if height_range else None,
                fixed_rehearsal=(fixed_spec if fixed_spec is not None
                    and fixed_spec["settings"]["version"] == FIXED17_VERSION else None))
        report["status"] = "running"
        (args.run_dir / "startup.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        with budget.signal_handlers():
            budget.begin_learning()
            if args.validate_scene:
                scene_started = time.monotonic()
                nonsteady_readback_steps = set()
                if "formal_dynamic_nonsteady_birth" in c:
                    nonsteady_readback_steps = {step for step in (1, 9, 10, 11, 12)
                                                if step <= args.validate_scene}
                    report["nonsteady_scene_validation"] = {
                        "requested_policy_steps": sorted(nonsteady_readback_steps), "readback_files": [],
                        "scope": "changed-birth ABI and motor-off ownership; no PPO allocation or skill pass"}
                if (c.get("formal_terrain", {}).get("type") == "plane"
                        and c.get("formal_foundation_nominal_birth") is True):
                    report["plane_contact_probe"] = env.prepare_formal_plane_contact_probe()
                rough_readback_steps = set()
                if c.get("formal_terrain", {}).get("type") == "rough":
                    report["rough_scene_validation"] = prepare_rough_scene_validation(env)
                    rough_readback_steps = {step for step in (1, 2, 3, 5, 10, args.validate_scene)
                                            if step <= args.validate_scene}
                    report["rough_scene_validation"].update(
                        requested_policy_steps=sorted(rough_readback_steps), readback_files=[])
                from torch.utils.tensorboard import SummaryWriter
                survivors = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
                report["scene_diagnostics"] = {"action_mode": "zero_policy_action_nominal_PD",
                    "steps_completed": 0, "first_episode_survivors": args.num_envs,
                    "closure_reset_disabled_for_observation": bool(c.get("formal_closure_observation_only") or c.get("formal_closure_monitor_only")),
                    "scope": "physics_diagnosis_only_not_training_acceptance"}
                with SummaryWriter(str(args.run_dir)) as scene_writer:
                    for step_index in range(1, args.validate_scene + 1):
                        budget.check()
                        if budget_owner is not None:
                            budget_owner.check_runtime()
                        observations, _, done, info = env.step(torch.zeros(args.num_envs, 6, device=args.device))
                        if "formal_dynamic_nonsteady_birth" in c:
                            frame = nonsteady_scene_validation_frame(env, observations, done, info, step_index)
                            if step_index in nonsteady_readback_steps:
                                name = f"nonsteady_scene_step_{step_index:04d}.pt"
                                torch.save(frame, args.run_dir / name)
                                report["nonsteady_scene_validation"]["readback_files"].append(name)
                        if step_index in rough_readback_steps:
                            name = f"rough_scene_step_{step_index:04d}.pt"
                            torch.save(rough_scene_validation_frame(env, observations, done, info, step_index),
                                       args.run_dir / name)
                            report["rough_scene_validation"]["readback_files"].append(name)
                        survivors &= ~done.bool()
                        diagnostics = info.get("diagnostics", {})
                        values = {"first_episode_surviving_fraction": float(survivors.float().mean())}
                        if "gap" in diagnostics:
                            gap = diagnostics["gap"]
                            quantiles = torch.quantile(gap, gap.new_tensor([.5, .95, .99, 1.]))
                            values.update({f"closure_{name}_m": float(value) for name, value in
                                           zip(("median", "p95", "p99", "max"), quantiles)})
                            values["closure_above_3mm_fraction"] = float((gap > .003).float().mean())
                            if bool(survivors.any()):
                                values["survivors_closure_max_m"] = float(gap[survivors].max())
                        for name, mask in diagnostics.get("reasons", {}).items():
                            values[f"reset_{name}_fraction"] = float(mask.float().mean())
                        for name, value in values.items():
                            scene_writer.add_scalar(f"ScenePhysics/{name}", value, step_index)
                        for name, value in info.get("log", {}).items():
                            if name.startswith(("/sensor_noise/", "/domain/spring_force_scale_", "/policy_delay/")):
                                scene_writer.add_scalar(f"SceneSignals/{name.lstrip('/')}", float(value), step_index)
                        report["scene_diagnostics"].update(steps_completed=step_index,
                            first_episode_survivors=int(survivors.sum()), last_step=values)
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
                if (c.get("formal_terrain", {}).get("type") == "plane"
                        and c.get("formal_foundation_nominal_birth") is True):
                    report["plane_contact_probe"] = env.formal_plane_contact_probe_result()
                    if not report["plane_contact_probe"]["passed"]:
                        raise RuntimeError("Shared plane contact outside the old lane did not pass")
                report["status"] = "scene_validated"
            else:
                from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
                from rsl_rl.runners import OnPolicyRunner
                from wheeled_tasks.agents.chassis_ppo_cfg import ChassisPPORunnerCfg
                cfg = handle_deprecated_rsl_rl_cfg(ChassisPPORunnerCfg(), "5.5.1")
                cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
                cfg.seed, cfg.device = args.seed, args.device
                cfg.save_interval = c["save_interval"]
                cfg.num_steps_per_env = c["num_steps_per_env"]
                if "num_mini_batches" in c:
                    cfg.algorithm.num_mini_batches = c["num_mini_batches"]
                if "num_learning_epochs" in c:
                    cfg.algorithm.num_learning_epochs = c["num_learning_epochs"]
                if "learning_rate" in c:
                    cfg.algorithm.learning_rate = c["learning_rate"]
                if "learning_rate_schedule" in c:
                    cfg.algorithm.schedule = c["learning_rate_schedule"]
                if "entropy_coef" in c:
                    cfg.algorithm.entropy_coef = validate_entropy_coef(c["entropy_coef"])
                if "value_loss_coef" in c:
                    cfg.algorithm.value_loss_coef = c["value_loss_coef"]
                elif args.stage in ("terrain", "mixed") and not uses_profile(c):
                    cfg.algorithm.value_loss_coef = 2.
                runner_config = cfg.to_dict()
                runner_config["logger"] = "tensorboard"
                if "initial_noise_std" in c:
                    runner_config["actor"]["distribution_cfg"]["init_std"] = c["initial_noise_std"]
                from wheeled_tasks.chassis.training_profile import configure_runner, verify_runner, verify_learning_rate
                runner_config = configure_runner(runner_config, c, stage=args.stage)
                effective_algorithm = runner_config["algorithm"]
                rollout_steps = runner_config["num_steps_per_env"]
                (args.run_dir / "agent_config.json").write_text(json.dumps(runner_config, indent=2))
                runner = OnPolicyRunner(env, deepcopy(runner_config), log_dir=str(args.run_dir), device=args.device)
                profile_readback = verify_runner(runner.alg, runner_config, c)
                if profile_readback is not None:
                    (args.run_dir / "runner_profile_readback.json").write_text(json.dumps(profile_readback, indent=2) + "\n")
                report["entropy_coef"] = verify_entropy_coef(runner.alg, effective_algorithm["entropy_coef"])
                report["entropy_coef_source"] = ("training_profile" if uses_profile(c)
                    else "contract" if "entropy_coef" in c else "runner_default")
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
                course_rng_metadata = {}
                fresh_critic_state = None
                if args.resume:
                    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
                    infos = checkpoint.get("infos", {})
                    if "formal_critic_adaptation" in c and infos.get("stage") == "foundation":
                        if digest(args.resume) != c["formal_critic_adaptation"]["source_checkpoint_sha256"]:
                            raise ValueError("Critic adaptation requires the sealed14841 checkpoint")
                    if budget_owner is not None and (diagnostic_budget_binding is None
                            or diagnostic_budget_binding["mode"] != "first_parent_fork"):
                        if "training_budget" not in infos:
                            raise ValueError("V7 registered checkpoint lacks its shared budget receipt")
                        budget_owner.verify_resume(args.resume, infos["training_budget"])
                    migrating = args.resume_physics_change or args.resume_reward_change or args.resume_budget_change or args.resume_response_change or args.resume_diagnostic or args.resume_task_change
                    if migrating:
                        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                        from wheeled_tasks.chassis.policy_transfer import validate_physics_resume, validate_reward_resume, validate_budget_resume
                        from wheeled_tasks.chassis.diagnostic_curriculum import validate_diagnostic_resume
                        if budget_scope == "v7_diagnostic81":
                            from wheeled_tasks.chassis.diagnostic_budget_selection import validate_budgeted_diagnostic_resume
                            validate_diagnostic_resume = validate_budgeted_diagnostic_resume
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
                            report[resume_key] = validate(json.loads(source_path.read_text()),
                                contract_config if c.get("integrated_production") else c)
                        if args.resume_diagnostic:
                            plan = c["diagnostic_study"]["plan"]
                            if (digest(source_path) != plan["source_contract_sha256"]
                                    or digest(args.resume) != plan["source_checkpoint_sha256"]
                                    or infos["successful_updates_total"] != plan["source_updates"]):
                                raise ValueError("Diagnostic arms must start from the same authenticated750 checkpoint")
                        report[resume_key].update(source_contract_sha256=digest(source_path),
                            target_contract_sha256=identity["contract_sha256"])
                    elif c.get("formal_learning_state_resume"):
                        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                        from wheeled_tasks.chassis.training_profile import validate_formal_learning_state_resume
                        source_path = checkpoint_contract_path(args.resume)
                        if infos.get("contract_sha256") != digest(source_path):
                            raise ValueError("V7 parent contract provenance mismatch")
                        report["formal_repair_resume"] = validate_formal_learning_state_resume(
                            json.loads(source_path.read_text()), c)
                    elif any(infos.get(k) != v for k, v in identity.items()):
                        raise ValueError("Parent checkpoint has a different task/asset/control interface")
                    from wheeled_tasks.chassis.training_profile import restore_profile_clocks
                    profile_clocks = restore_profile_clocks(c, infos, stage=args.stage,
                        requested=requested_profile_clocks)
                    if profile_clocks is not None:
                        if ("formal_small_rough_mix" in c or c.get("formal_final_rough_stage")
                                and "scripted_plane_lane_mask" in (getattr(env, "rehearsal_spec", None) or {})):
                            source_clock = env.startup_report.get("rough_recovery_source_command_clock")
                            if (not isinstance(source_clock, dict)
                                    or type(source_clock.get("source_flat_iteration_offset")) is not int
                                    or source_clock["source_flat_iteration_offset"]
                                    != profile_clocks["global_actor_update_offset"]):
                                raise ValueError("Installed rough source-flat clock differs from the authenticated parent clock")
                            report["rough_recovery_source_clock_authenticated"] = {
                                "global_actor_update_offset": profile_clocks["global_actor_update_offset"],
                                "stage_actor_update": profile_clocks["stage_actor_update"],
                                "global_actor_update": profile_clocks["global_actor_update"],
                                "source": "restored_registered_parent_profile_clocks"}
                        for key in requested_profile_clocks:
                            setattr(args, key, profile_clocks[key])
                        report["profile_resume_clocks"] = profile_clocks
                    from wheeled_tasks.chassis.policy_transfer import formal_stage_load_kwargs
                    resume_load_kwargs = formal_stage_load_kwargs(c, infos, stage=args.stage)
                    from wheeled_tasks.chassis.policy_transfer import formal_resume_rng_selection
                    resume_rng_selection = ({"mode": args.rng_mode, "phase": "diagnostic_fork"}
                        if args.resume_diagnostic else formal_resume_rng_selection(
                            c, infos, stage=args.stage, mode=args.rng_mode, seed=args.seed))
                    if "sampling_seed" in resume_rng_selection:
                        course_rng_metadata = {"sampling_seed": resume_rng_selection["sampling_seed"],
                            "terrain_geometry_seed": resume_rng_selection["geometry_seed"]}
                    if resume_load_kwargs.get("load_cfg", {}).get("critic") is False:
                        fresh_critic_state = {key: value.detach().clone()
                            for key, value in runner.alg.critic.state_dict().items()}
                    runner.load(str(args.resume), **resume_load_kwargs)
                    if "entropy_coef" in infos:
                        parent_entropy = validate_entropy_coef(infos["entropy_coef"])
                        if not migrating and parent_entropy != effective_algorithm["entropy_coef"]:
                            raise ValueError("Same-contract resume changed its saved entropy coefficient")
                    from wheeled_tasks.chassis.policy_transfer import verify_learning_state_restore, resume_budget
                    restored = verify_learning_state_restore(runner.alg, checkpoint,
                        fresh_optimizer_learning_rate=(effective_algorithm["learning_rate"]
                            if resume_load_kwargs else None), fresh_critic_state=fresh_critic_state)
                    report.update(resume_budget(infos, args.consumed_updates, args.num_envs * rollout_steps))
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
                            "rng_selection": resume_rng_selection,
                            "diagnostic_origin": report.get("diagnostic_origin"),
                            "parent_training_transitions": report["parent_training_transitions"]}, indent=2) + "\n")
                    runner.current_learning_iteration = report["parent_updates"]
                    from wheeled_tasks.chassis.training_profile import restore_rng
                    report["rng_restore"] = restore_rng(torch, env, infos,
                        seed=args.seed, mode=resume_rng_selection["mode"])
                    report["rng_restore"]["selection"] = resume_rng_selection
                    (args.run_dir / "rng_restore.json").write_text(json.dumps(report["rng_restore"], indent=2) + "\n")
                    if env.performance_curriculum is not None:
                        if infos.get("curriculum_state") is None:
                            raise ValueError("Performance curriculum resume requires its saved state")
                        env.performance_curriculum.load_state_dict(infos["curriculum_state"])
                env.training_transitions = report["parent_training_transitions"]
                saved_clock = infos.get("stage_actor_update", args.actor_updates_consumed) if args.resume else 0
                if args.resume and report.get("formal_repair_resume", {}).get("stage_transition"):
                    saved_clock = report["profile_resume_clocks"]["stage_actor_update"]
                if args.stage_actor_update is not None and args.stage_actor_update != saved_clock:
                    raise ValueError("Requested curriculum clock differs from the resumed checkpoint")
                stage_actor_start = saved_clock
                env.stage_actor_update = stage_actor_start
                env.global_actor_update = args.global_actor_update_offset + stage_actor_start
                task_actor_start = 0
                if c.get("task_campaign"):
                    same_task = infos.get("task_campaign", {}).get("domain") == c["task_campaign"]["domain"]
                    task_actor_start = (integrated_receipt["selection"]["fresh_domain_actor_updates"]
                        if c.get("integrated_production") else infos.get("task_actor_update") if same_task else 0)
                    if type(task_actor_start) is not int or args.task_actor_update != task_actor_start:
                        raise ValueError("Task curriculum clock differs from the sealed learning origin")
                    env.task_actor_update = task_actor_start
                    report.update(task_campaign=c["task_campaign"], task_actor_update=task_actor_start)
                env.confirmed_jump_bins_m = confirmed_bins
                report.update(stage_actor_update=stage_actor_start,
                    global_actor_update_offset=args.global_actor_update_offset,
                    confirmed_jump_bins_m=confirmed_bins)
                formal_assessment_update = next_formal_assessment_update(c, report["parent_updates"])
                if formal_assessment_update is not None:
                    report["formal_assessment_schedule"] = {
                        "clock": "learning_updates", "restored_learning_updates": report["parent_updates"],
                        "restored_stage_actor_update": stage_actor_start,
                        "next_assessment_update": formal_assessment_update}
                if args.resume or confirmed_bins:
                    # Constructor resets precede checkpoint loading. Re-sample
                    # at the restored curriculum clock, not at iteration zero.
                    ids = torch.arange(args.num_envs, device=args.device)
                    if args.resume and c.get("formal_learning_state_resume"):
                        if sampled_policy_validation:
                            # Diagnostics belong to the restored native window,
                            # not the frozen contract checked before loading.
                            env.cfg["formal_physics_trace_substeps"] = [1, 2, 4, 5, 6, 7, 36, 39, 40, 41, 42, 44, 48, 96]
                            if env.cfg.get("gas_pressure_application") == "main_link_mounts_v1":
                                env.cfg["formal_physics_trace_substeps"] += [104, 112, 128, 144, 160, 168, 169, 170, 172, 176, 178, 179, 180, 181, 184, 188, 192] + list(range(145, 160))
                        env.reset(ids)
                    elif (args.resume and env.recovery is not None
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
                    if c.get("integrated_production"):
                        from wheeled_tasks.chassis.integrated_training import critic_warmup_remaining
                        remaining = critic_warmup_remaining(integrated_receipt, infos)
                    else:
                        remaining = resume_critic_warmup(c, infos, task_change=args.resume_task_change,
                            reward_change=args.resume_reward_change, contract_sha256=identity["contract_sha256"])
                    if args.resume_diagnostic:
                        remaining = c["diagnostic_study"]["plan"]["critic_adaptation_updates"]
                    warmup_updates = report["parent_updates"] + remaining
                report["learning_rate"] = configure_learning_rate(
                    runner.alg, effective_algorithm["learning_rate"], args.learning_rate_scale)
                verify_learning_rate(runner.alg, c)
                report["initial_learning_rate"] = report["learning_rate"]
                report["initial_entropy_coef"] = verify_entropy_coef(runner.alg, effective_algorithm["entropy_coef"])
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
                if (c.get("v5_control", {}).get("leg_winding_version") == "paired_branch107_v1"
                        and args.capture_numerical_failure):
                    rng = report.get("rng_restore", {})
                    fresh_course = (rng.get("mode") == "fresh_fork"
                        and rng.get("parent_rng_restored") is False
                        and rng.get("selection", {}).get("phase") == "new_course_entry"
                        and not sampled_policy_validation
                        and (not restored_policy_validation
                            or c.get("formal_small_rough_mix", {}).get("version") in MOTION_VERSIONS
                            and "formal_critic_adaptation" in c))
                    inherited = rng.get("mode") == "inherit" and rng.get("parent_rng_restored") is True
                    if not (inherited or fresh_course):
                        raise ValueError("Branch107 capture requires authenticated RNG; matched validations must inherit the parent stream")
                    report["physical_capture_rng_scope"] = (
                        "fresh_motion_course_not_paired_parent_replay" if fresh_course else "inherited_parent_stream")
                    bind_sampled_policy_inputs(runner.alg, env, args.run_dir)
                if sampled_policy_validation:
                    def check_sampled_validation_runtime():
                        budget.check()
                        if budget_owner is not None:
                            budget_owner.check_runtime()
                    check_report = report["restored_sampled_policy_validation"] = {}
                    try:
                        validate_restored_sampled_policy(runner, env, args.validate_restored_sampled_policy,
                            check_report, check_runtime=check_sampled_validation_runtime)
                        check_report["learning_state_unchanged"] = verify_learning_state_restore(runner.alg, checkpoint,
                            fresh_optimizer_learning_rate=(effective_algorithm["learning_rate"]
                                if resume_load_kwargs else None), fresh_critic_state=fresh_critic_state)
                    finally:
                        (args.run_dir / "restored_sampled_policy_validation.json").write_text(
                            json.dumps(check_report, indent=2, allow_nan=False) + "\n")
                    raise PlannedStop("restored_sampled_policy_validation_completed")
                if restored_policy_validation:
                    def check_restored_validation_runtime():
                        budget.check()
                        if budget_owner is not None:
                            budget_owner.check_runtime()
                    check_report = report["restored_policy_validation"] = {}
                    try:
                        validate_restored_policy(runner, env, args.validate_restored_policy,
                            check_report, check_runtime=check_restored_validation_runtime, behavior_metrics=metrics)
                        check_report["learning_state_unchanged"] = verify_learning_state_restore(runner.alg, checkpoint,
                            fresh_optimizer_learning_rate=(effective_algorithm["learning_rate"]
                                if resume_load_kwargs else None), fresh_critic_state=fresh_critic_state)
                    finally:
                        (args.run_dir / "restored_policy_validation.json").write_text(
                            json.dumps(check_report, indent=2, allow_nan=False) + "\n")
                    raise PlannedStop("restored_policy_validation_completed")
                runner.alg.actor.requires_grad_(report["parent_updates"] >= warmup_updates)
                recovery_rollout = configure_recovery_rollout(runner.alg, env)
                mixed_exposure = None
                if "formal_small_rough_mix" in c:
                    from wheeled_tasks.chassis.small_rough_course import MixedCourseExposure
                    mixed_exposure = MixedCourseExposure(env.rehearsal_spec, rollout_steps, env.device,
                        rough_grid=(env.flat_episode.commands._rough
                            if c["formal_small_rough_mix"]["version"] in MOTION_VERSIONS else None))
                    if infos.get("mixed_course_exposure_state") is not None:
                        mixed_exposure.load_state_dict(infos["mixed_course_exposure_state"])
                rehearsal_indices = None
                if getattr(env, "rehearsal_spec", None) is not None:
                    if recovery_rollout is None:
                        raise RuntimeError("Fixed flat rehearsal exposure requires the actual PPO loss mask")
                    rehearsal_indices, report["fixed_flat_rehearsal"] = fixed_flat_rehearsal_assignment(
                        env.rehearsal_spec, args.num_envs, env.device)
                env.reward_episode_journal = None
                if c.get("reward_episode_measurement", {}).get("enabled"):
                    from wheeled_tasks.chassis.reward_episode_journal import EpisodePaymentJournal
                    env.reward_episode_journal = EpisodePaymentJournal(env.scene_groups, env.device,
                        gamma=runner.alg.gamma, source_identity={
                            "training_identity": identity,
                            "producer": env.startup_report["reward_episode_measurement"],
                            "actual_algorithm_gamma": float(runner.alg.gamma)})
                    report["reward_episode_counting_scope"] = "current_worker_physical_payments_not_PPO_targets"
                    report["reward_episode_parent_state_present"] = (
                        "reward_episode_measurement_state" in infos if args.resume else False)
                recovery_evidence = None
                if env.recovery is not None and env.recovery.has_tasks:
                    from wheeled_tasks.chassis.recovery_evidence import RecoveryEvidenceJournal
                    if recovery_rollout is None:
                        raise RuntimeError("Recovery evidence requires the actual final PPO mask")
                    recovery_evidence = RecoveryEvidenceJournal(num_envs=args.num_envs,
                        steps_per_rollout=rollout_steps, device=env.device,
                        source_identity=env.recovery.source_identity())
                    report["recovery_evidence_counting_scope"] = "current_worker_only_not_resumed"
                    report["recovery_evidence_parent_state_present"] = (
                        "recovery_evidence_state" in infos if args.resume else False)
                if "formal_dynamic_nonsteady_birth" in c:
                    from wheeled_tasks.chassis.fixed_flat_rehearsal import validate_dynamic_nonsteady_birth
                    from wheeled_tasks.chassis.source_birth_exposure import SourceBirthExposure
                    validate_dynamic_nonsteady_birth(c)
                    if recovery_rollout is None or c.get("record_diagnostics") is not True:
                        raise RuntimeError("Dynamic birth exposure requires diagnostics and the actual PPO loss mask")
                    dynamic_source_mask = getattr(env, "_dynamic_birth_source_mask", None)
                    if dynamic_source_mask is None:
                        if "formal_scripted_recovery_mix" in c:
                            raise RuntimeError("Scripted mix exposure requires the actual ordinary dynamic source mask")
                        dynamic_source_mask = ~env._dynamic_birth_nominal_mask
                    source_birth_exposure = SourceBirthExposure(num_envs=args.num_envs,
                        steps_per_rollout=rollout_steps, device=env.device,
                        source_dynamic_mask=dynamic_source_mask,
                        stream_id=str(args.run_dir.resolve()), defer_tensor_validation=True)
                    parent_birth_state = infos.get("source_birth_exposure_state") if args.resume else None
                    if parent_birth_state is not None:
                        if "formal_small_rough_mix" in c:
                            repair = report.get("formal_repair_resume", {})
                            # Sampling history survives the explicit critic reset;
                            # a fresh stream must not inherit live episode cursors.
                            fresh_motion_entry = (
                                c["formal_small_rough_mix"].get("version") in MOTION_VERSIONS
                                and "formal_critic_adaptation" in c
                                and repair.get("stage_transition") is True
                                and infos.get("stage") == "foundation"
                                and fresh_critic_state is not None
                                and restored.get("actor_exact") is True
                                and restored.get("critic_initialization") == "verified_fresh_before_parent_load"
                                and restored.get("optimizer_state_entries") == 0
                                and report.get("rng_restore", {}).get("mode") == "fresh_fork"
                                and report["rng_restore"].get("parent_rng_restored") is False
                                and report["rng_restore"].get("selection", {}).get("phase") == "new_course_entry")
                            allow_change = (repair.get("source_birth_population_changed") is True
                                and (fresh_motion_entry or
                                    all(restored.get(key) is True for key in ("actor_exact", "critic_exact"))
                                    and report.get("rng_restore", {}).get("parent_rng_restored") is True))
                            source_birth_exposure.restore_state(parent_birth_state,
                                allow_population_change=allow_change)
                        elif "formal_scripted_recovery_mix" in c:
                            from wheeled_tasks.chassis.fixed_flat_rehearsal import SCRIPTED_RECOVERY_MIX_VERSION
                            mix_receipt = report.get("formal_repair_resume", {}).get("formal_scripted_recovery_mix", {})
                            allow_population_change = (
                                c.get("formal_scripted_recovery_mix") == SCRIPTED_RECOVERY_MIX_VERSION
                                and c.get("formal_learning_state_resume") is True
                                and args.resume is not None
                                and mix_receipt.get("enabled_before") is False
                                and mix_receipt.get("enabled_after") is True
                                and mix_receipt.get("population_changed") is True
                                and all(restored.get(key) is True for key in (
                                    "actor_exact", "critic_exact", "optimizer_exact"))
                                and report.get("rng_restore", {}).get("mode") == "inherit"
                                and report["rng_restore"].get("parent_rng_restored") is True)
                            source_birth_exposure.restore_state(parent_birth_state,
                                allow_population_change=allow_population_change)
                        else:
                            source_birth_exposure.restore_state(parent_birth_state)
                    report["source_birth_exposure_parent_state_restored"] = parent_birth_state is not None
                    report["source_birth_exposure_counting_scope"] = (
                        "cumulative_observed_workers_new_native_episodes_per_worker")
                    if "formal_rough_nonsteady_birth" in c:
                        report["source_birth_exposure_population"] = {
                            "scope": "current continuous flat and rough births; inherited counts retain original population",
                            "actual_membership": env._dynamic_birth_runtime_state(),
                            "per_terrain_actor_exposure": "mixed_course_exposure_state.rough_exposure"}
                elif "final_rough_birth_adapter" in c:
                    from wheeled_tasks.chassis.source_birth_exposure import SourceBirthExposure, STATE_VERSION
                    if recovery_rollout is None or c.get("record_diagnostics") is not True:
                        raise RuntimeError("Final birth exposure requires diagnostics and the actual PPO loss mask")
                    flat = env.flat_episode
                    dynamic_source_mask = ~flat.commands.fixed_request_mask
                    if getattr(env.recovery, "has_tasks", False):
                        dynamic_source_mask &= ~env.recovery.enabled
                    binding = flat.final_birth_exposure_binding(dynamic_source_mask)
                    source_birth_exposure = SourceBirthExposure(num_envs=args.num_envs,
                        steps_per_rollout=rollout_steps, device=env.device,
                        source_dynamic_mask=dynamic_source_mask,
                        stream_id=str(args.run_dir.resolve()), defer_tensor_validation=True, **binding)
                    parent_birth_state = infos.get("source_birth_exposure_state") if args.resume else None
                    inherited_flat_history = False
                    if args.resume is not None and parent_birth_state is None:
                        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                        parent_path = checkpoint_contract_path(args.resume)
                        if digest(parent_path) != infos.get("contract_sha256"):
                            raise ValueError("Birth exposure parent contract provenance differs")
                        parent_contract = json.loads(parent_path.read_text())
                        if any(key in parent_contract for key in (
                                "formal_dynamic_nonsteady_birth", "final_rough_birth_adapter")):
                            raise ValueError("Parent contract requires its saved birth exposure journal")
                    if parent_birth_state is not None:
                        if not isinstance(parent_birth_state, dict):
                            raise ValueError("Final exposure parent state must be a complete snapshot")
                        if parent_birth_state.get("version") == STATE_VERSION:
                            repair = report.get("formal_repair_resume", {})
                            if (repair.get("stage_transition") is not True
                                    or repair.get("optimizer_initialization") != "fresh_first_handoff"
                                    or c.get("formal_stage_transition") != {
                                        "from": "foundation", "to": "terrain", "reset_stage_actor_clock": True}
                                    or infos.get("stage") != "foundation"
                                    or not all(restored.get(key) is True for key in ("actor_exact", "critic_exact"))
                                    or restored.get("optimizer_initialization") != "fresh_on_foundation_to_terrain_only"
                                    or restored.get("optimizer_state_entries") != 0
                                    or report.get("rng_restore", {}).get("mode") != "inherit"
                                    or report["rng_restore"].get("parent_rng_restored") is not True):
                                raise ValueError("Flat history migration requires the reviewed first-stage learning-state handoff")
                            from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                            parent_path = checkpoint_contract_path(args.resume)
                            if digest(parent_path) != infos.get("contract_sha256"):
                                raise ValueError("Flat history parent contract provenance differs")
                            source_birth_exposure.inherit_flat_history(parent_birth_state,
                                parent_contract_json=parent_path.read_bytes().decode("utf-8"),
                                parent_checkpoint_sha256=report["parent_checkpoint_sha256"])
                            inherited_flat_history = True
                        else:
                            source_birth_exposure.restore_state(parent_birth_state)
                            if "source_birth_exposure_flat_history" not in infos:
                                raise ValueError("Final exposure restart must retain its explicit inherited-history field")
                            source_birth_exposure.restore_flat_history(infos["source_birth_exposure_flat_history"])
                    report["source_birth_exposure_parent_state_restored"] = (
                        parent_birth_state is not None and not inherited_flat_history)
                    history = source_birth_exposure.checkpoint_flat_history()
                    report["source_birth_exposure_flat_history"] = (None if history is None else {
                        key: history[key] for key in (
                            "version", "parent_checkpoint_sha256", "parent_contract_sha256", "source_state_sha256")})
                    report["source_birth_exposure_counting_scope"] = (
                        "final_classification_workers_only; legacy_flat_history_retained_separately")
                    report["final_birth_exposure_binding"] = {
                        "definition_sha256": binding["final_birth_definition_sha256"],
                        "groups": list(source_birth_exposure.groups),
                        "source_dynamic_lanes": int(dynamic_source_mask.sum()),
                        "final_birth_lanes": int(binding["final_birth_lane_mask"].sum()),
                        "skill_qualification": "not_assessed"}
                curriculum_runtime = None
                if curriculum_selection is not None:
                    from wheeled_tasks.chassis.curriculum_runtime import CurriculumRuntime
                    if budget_owner is None or recovery_rollout is None:
                        raise RuntimeError("V7 exposure requires the real paid owner and final PPO mask")
                    curriculum_runtime = CurriculumRuntime(env.curriculum_controller.definition,
                        definition_sha256=curriculum_selection["definition_sha256"], budget_owner=budget_owner,
                        training_seeds=[args.seed], num_envs=args.num_envs, steps_per_rollout=rollout_steps,
                        device=env.device, curriculum=env.curriculum_controller)
                    env.curriculum_runtime = curriculum_runtime
                original_update, original_step, original_save = runner.alg.update, env.step, runner.save
                consecutive_closure_failures = 0
                pending_assessment = None
                pending_checkpoint = None
                if c.get("task_campaign"):
                    from wheeled_tasks.chassis.training_timing import TrainingTiming
                    timing = TrainingTiming()
                learning_schedule = runner.alg.schedule
                policy_probe = None
                learning_audit = None
                if c.get("remediation") or c.get("continuation") or c.get("task_campaign"):
                    from wheeled_tasks.chassis.learning_audit import LearningAudit
                    learning_audit = LearningAudit(env, runner.alg)
                if c.get("diagnostic_logging") or final_motion:
                    from wheeled_tasks.chassis.diagnostic_curriculum import PolicyUpdateProbe
                    if final_motion:
                        from wheeled_tasks.chassis.small_rough_course import ROLES
                        probe_batches = [((mixed_exposure.roles == i).nonzero(as_tuple=False).flatten(),
                            {"group_name": role}) for i, role in enumerate(ROLES)]
                    else:
                        probe_batches = env.skills.batches
                    policy_probe = PolicyUpdateProbe(runner.alg.actor, runner.alg.storage, probe_batches)

                def update(*a, _budget_token=None, **kw):
                    nonlocal pending_assessment, pending_checkpoint
                    report["attempted_updates"] = report["successful_updates"] + 1
                    if c.get("task_campaign"):
                        # Reserve before optimizer work so an abrupt process loss
                        # cannot silently refund a possibly completed update.
                        previous = args.run_dir / "progress.json"
                        reservation = json.loads(previous.read_text()) if previous.exists() else {
                            "pid": os.getpid(), "parent_updates": report["parent_updates"],
                            "successful_updates": report["successful_updates"], "actor_updates_in_block": report["actor_updates_in_block"]}
                        reservation["attempted_updates"] = report["attempted_updates"]
                        if c.get("integrated_production"):
                            reservation.update(learning_metadata(integrated_receipt, env.task_actor_update))
                        write_progress(args.run_dir, reservation, timing)
                    actor_enabled = report["parent_updates"] + report["successful_updates"] >= warmup_updates
                    verify_entropy_coef(runner.alg, effective_algorithm["entropy_coef"])
                    # A frozen actor has near-zero KL; adaptive scheduling would
                    # otherwise inflate the learning rate during critic warmup.
                    runner.alg.schedule = learning_schedule if actor_enabled else "fixed"
                    before_rate = verify_learning_rate(runner.alg, c, actor_enabled=actor_enabled)
                    if policy_probe is not None:
                        policy_probe.capture()
                    if learning_audit is not None:
                        timed_call(timing, "diagnostics", learning_audit.capture)
                    loss_mask = (recovery_rollout.loss_population_mask()
                        if c.get("formal_v7_flat") or rehearsal_indices is not None else None)
                    clipping = (incoming_action_clipping_metrics(c, runner.alg.storage.actions,
                        loss_mask) if c.get("formal_v7_flat") else None)
                    rehearsal_counts = (fixed_flat_rehearsal_counts(loss_mask, rehearsal_indices,
                        len(report["fixed_flat_rehearsal"]["nominal_lane_counts"]))
                        if rehearsal_indices is not None and actor_enabled else None)
                    mixed_counts = (mixed_exposure.prepare(loss_mask)
                                    if mixed_exposure is not None and actor_enabled else None)
                    mixed_optimization = (mixed_exposure.optimization_metrics(
                        runner.alg.storage, loss_mask, rehearsal_indices,
                        tuple(report["fixed_flat_rehearsal"]["nominal_lane_counts"])
                        if rehearsal_indices is not None else ())
                        if mixed_exposure is not None and loss_mask is not None else None)
                    result = timed_call(timing, "optimizer", original_update, *a, **kw)
                    report["successful_updates"] += 1
                    report["actor_updates_in_block"] += int(actor_enabled)
                    # Commit all consumption and actor clocks before any invariant
                    # can fail. Critic-only work is paid but never advances actors.
                    env.stage_actor_update = stage_actor_start + report["actor_updates_in_block"]
                    env.global_actor_update = args.global_actor_update_offset + env.stage_actor_update
                    if c.get("task_campaign"):
                        env.task_actor_update = task_actor_start + report["actor_updates_in_block"]
                        report["task_actor_update"] = env.task_actor_update
                    report.update(stage_actor_update=env.stage_actor_update,
                        global_actor_update=env.global_actor_update,
                        actor_updates_consumed=args.actor_updates_consumed + report["actor_updates_in_block"])
                    if c.get("integrated_production"):
                        report.update(learning_metadata(integrated_receipt, env.task_actor_update))
                    if budget_owner is not None:
                        budget_owner.complete_update(_budget_token, actor_optimized=actor_enabled)
                        report["training_budget"] = budget_owner.summary()
                    if mixed_counts is not None:
                        counts = mixed_exposure.commit(mixed_counts)
                        report["mixed_course_exposure_state"] = mixed_exposure.state_dict()
                        log_scalar_batch(runner.logger.writer,
                            {**{f"MixedCourse/direct_loss/{k}": v for k, v in counts.items()},
                             "MixedCourse/complete_temporal_loss_sequences": mixed_exposure.complete_loss_sequences},
                            report["parent_updates"] + report["successful_updates"] - 1)
                    if mixed_optimization is not None:
                        metric_names = tuple(mixed_optimization)
                        metric_values = torch.stack(tuple(mixed_optimization.values())).detach().cpu().tolist()
                        mixed_optimization = dict(zip(metric_names, metric_values))
                        mixed_optimization["MixedCourse/optimization/actor_enabled"] = int(actor_enabled)
                        log_scalar_batch(runner.logger.writer, mixed_optimization,
                            report["parent_updates"] + report["successful_updates"] - 1)
                        if not actor_enabled:
                            report["critic_only_value_readback"] = mixed_optimization
                    report["learning_rate"] = verify_learning_rate(runner.alg, c,
                        actor_enabled=actor_enabled, before_rate=before_rate)
                    # The optimizer has returned: retain its real clocks even
                    # if a post-update invariant fails. Never seal that state.
                    report["entropy_coef"] = verify_entropy_coef(runner.alg, effective_algorithm["entropy_coef"])
                    if "formal_critic_adaptation" in c and not actor_enabled:
                        from wheeled_tasks.chassis.critic_adaptation import verify_frozen_actor
                        report["critic_only_actor_invariant"] = verify_frozen_actor(runner.alg, before_actor)
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
                        diagnostic = policy_probe.metrics()
                        if not reward_selection["scut_source_selected"]:
                            diagnostic.update(env.diagnostic_reward_metrics())
                        log_scalar_batch(runner.logger.writer, diagnostic,
                            report["parent_updates"] + report["successful_updates"] - 1)
                    if learning_audit is not None:
                        audit = timed_call(timing, "diagnostics", learning_audit.drain)
                        if runner.logger.writer is not None:
                            timed_call(timing, "audit_logger", log_scalar_batch,
                                runner.logger.writer, audit,
                                report["parent_updates"] + report["successful_updates"] - 1)
                    if clipping is not None:
                        log_scalar_batch(runner.logger.writer, clipping,
                            report["parent_updates"] + report["successful_updates"] - 1)
                    if rehearsal_counts is not None:
                        # The native optimizer clears storage; only the detached
                        # aggregate survives, and failed optimizer calls never commit it.
                        rehearsal = report["fixed_flat_rehearsal"]
                        groups = tuple(rehearsal["nominal_lane_counts"])
                        counts = rehearsal_counts.cpu().tolist()
                        rehearsal["successful_actor_updates"] += 1
                        rehearsal["last_update_direct_loss_transitions"] = dict(zip(groups, counts))
                        for name, count in zip(groups, counts):
                            rehearsal["worker_direct_loss_transitions"][name] += count
                        total = sum(counts)
                        values = {}
                        for name, count in zip(groups, counts):
                            values[f"FixedFlatRehearsal/direct_loss/{name}/transitions"] = count
                            values[f"FixedFlatRehearsal/direct_loss/{name}/fraction"] = count / max(total, 1)
                            if rehearsal["successful_actor_updates"] == 1:
                                values[f"FixedFlatRehearsal/nominal_lane_fraction/{name}"] = (
                                    rehearsal["nominal_lane_fractions"][name])
                        log_scalar_batch(runner.logger.writer, values,
                            report["parent_updates"] + report["successful_updates"] - 1)
                    timed_call(timing, "diagnostics", log_recovery_outcomes,
                        runner.logger.writer, env.recovery,
                        report["parent_updates"] + report["successful_updates"] - 1)
                    runner.alg.actor.requires_grad_(
                        report["parent_updates"] + report["successful_updates"] >= warmup_updates
                        and (actor_enabled or "formal_critic_adaptation" not in c))
                    env.training_transitions = report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * rollout_steps
                    progress = {"updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                        "successful_updates": report["successful_updates"], "parent_updates": report["parent_updates"],
                        "attempted_updates": report["attempted_updates"],
                         "num_envs": args.num_envs, "stage": args.stage,
                         "actor_updates_in_block": report["actor_updates_in_block"],
                         "stage_actor_update": env.stage_actor_update,
                         "global_actor_update": env.global_actor_update,
                         "global_budget_actor_updates": (args.actor_updates_consumed + report["actor_updates_in_block"]
                             if c.get("formal_v7_flat") is True else args.global_actor_update_offset + args.actor_updates_consumed + report["actor_updates_in_block"]),
                         "optimizer_phase": "actor_and_critic" if actor_enabled else "critic_warmup",
                         "entropy_coef": report["entropy_coef"],
                         "entropy_coef_source": report["entropy_coef_source"],
                        "diagnostic_origin": report.get("diagnostic_origin"),
                        "remediation": c.get("remediation"), "continuation": c.get("continuation"),
                        "training_transitions": report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * rollout_steps}
                    if c.get("task_campaign"):
                        progress.update(task_actor_update=env.task_actor_update, task_campaign=c["task_campaign"])
                    if c.get("integrated_production"):
                        progress.update(learning_metadata(integrated_receipt, env.task_actor_update))
                    if recovery_rollout is not None:
                        progress.update({name: report[name] for name in (
                            "recovery_rollout", "policy_learning_transitions_in_block",
                            "script_or_blend_transitions_in_block")})
                    if rehearsal_indices is not None:
                        progress["fixed_flat_rehearsal"] = deepcopy(report["fixed_flat_rehearsal"])
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
                    formal_due = c.get("formal_v7_flat") is True and completed >= formal_assessment_update
                    snapshot = None
                    if formal_due or c.get("checkpoint_snapshots") and (
                            completed % c.get("checkpoint_interval", 100) == 0
                            or completed == c.get("checkpoint_first_update", 10)
                            # Keep each worker's first paid update recoverable
                            # before a later rollout can fail.
                            or report["successful_updates"] == 1
                            or c.get("task_campaign") and actor_enabled and report["actor_updates_in_block"] == 1
                            or changes or assessment_due):
                        if (formal_due or curriculum_runtime is not None or recovery_evidence is not None
                                or source_birth_exposure is not None):
                            # Seal only after all optimizer postchecks and the
                            # matching real-loss exposure commit have succeeded.
                            pending_checkpoint = (completed, progress)
                        else:
                            from chassis_checkpoints import seal_checkpoint
                            snapshot = timed_call(timing, "checkpoint", seal_checkpoint,
                                args.run_dir, completed, runner.save, args.contract, progress,
                                budget_owner=budget_owner)
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
                            for prefix, behavior_groups in (
                                    ("Behavior", behavior["groups"]),
                                    ("FixedFlatRehearsal/Behavior",
                                     behavior.get("fixed_rehearsal", {}).get("groups", {}))):
                                for group, values in behavior_groups.items():
                                    for name, value in values.items():
                                        if isinstance(value, (int, float)):
                                            runner.logger.writer.add_scalar(f"{prefix}/{group}/{name}", value, completed - 1)
                        temp = args.run_dir / "behavior_metrics.tmp"
                        temp.write_text(json.dumps(behavior, indent=2, allow_nan=False) + "\n")
                        temp.replace(args.run_dir / "behavior_metrics.json")
                        with (args.run_dir / "behavior_history.jsonl").open("a") as history:
                            history.write(json.dumps(behavior, allow_nan=False) + "\n")
                    if assessment_due:
                        if snapshot is None and pending_checkpoint is None:
                            raise RuntimeError("Task assessment requires an immutable snapshot")
                        # Log the completed PPO update before yielding the GPU.
                        pending_assessment = (snapshot / "model.pt" if snapshot is not None else None,
                            env.task_actor_update, completed)
                    return result

                def paid_update(*a, **kw):
                    nonlocal pending_checkpoint, pending_assessment, formal_assessment_update
                    physx_errors.raise_if_failed("before_paid_update")
                    from wheeled_tasks.chassis.training_budget import BudgetLimit
                    token = None
                    actor_enabled = report["parent_updates"] + report["successful_updates"] >= warmup_updates
                    if budget_owner is not None:
                        try:
                            token = budget_owner.reserve_update(phase="actor" if actor_enabled else "critic")
                        except BudgetLimit as exc:
                            if recovery_evidence is not None:
                                recovery_evidence.discard_rollout()
                            if source_birth_exposure is not None:
                                source_birth_exposure.discard_rollout()
                            raise PlannedStop(str(exc)) from exc
                    try:
                        if curriculum_runtime is not None:
                            curriculum_runtime.bind_paid_update(token,
                                final_learning_mask=recovery_rollout.learning,
                                optimized_population_mask=recovery_rollout.loss_population_mask())
                        if recovery_evidence is not None:
                            recovery_evidence.bind_loss_population(
                                final_learning_mask=recovery_rollout.learning,
                                optimized_population_mask=recovery_rollout.loss_population_mask())
                        if source_birth_exposure is not None:
                            source_birth_exposure.bind_loss_population(
                                final_learning_mask=recovery_rollout.learning,
                                optimized_population_mask=recovery_rollout.loss_population_mask())
                        result = update(*a, _budget_token=token, **kw)
                        if curriculum_runtime is not None:
                            curriculum_runtime.finish_paid_update(token)
                        if recovery_evidence is not None:
                            recovery_evidence.finish_update(actor_optimized=actor_enabled)
                        if source_birth_exposure is not None:
                            source_birth_exposure.finish_update(actor_optimized=actor_enabled)
                    except BaseException as exc:
                        if token is not None:
                            try:
                                budget_owner.fail_update(token, f"{type(exc).__name__}: {exc}")
                                report["training_budget"] = budget_owner.summary()
                            except BaseException as cleanup_error:
                                exc.add_note(f"Paid failure receipt could not be finalized: {type(cleanup_error).__name__}: {cleanup_error}")
                        if curriculum_runtime is not None:
                            try:
                                curriculum_runtime.fail_paid_update(token)
                            except BaseException as cleanup_error:
                                exc.add_note(f"Exposure failure cleanup could not be finalized: {type(cleanup_error).__name__}: {cleanup_error}")
                        if recovery_evidence is not None:
                            try:
                                recovery_evidence.fail_update()
                            except BaseException as cleanup_error:
                                exc.add_note(f"Recovery evidence cleanup could not be finalized: {type(cleanup_error).__name__}: {cleanup_error}")
                        if source_birth_exposure is not None:
                            try:
                                source_birth_exposure.fail_update()
                            except BaseException as cleanup_error:
                                exc.add_note(f"Birth exposure cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
                        pending_checkpoint = None
                        pending_assessment = None
                        raise
                    # Publication failure cannot turn already committed learning
                    # into an unlearned update or refund its paid reservation.
                    if curriculum_runtime is not None:
                        values = env.curriculum_controller.metrics()
                        report["v7_curriculum_measurements"] = values
                        if runner.logger.writer is not None:
                            for name, value in values.items():
                                runner.logger.writer.add_scalar(name, value,
                                    report["parent_updates"] + report["successful_updates"] - 1)
                    if recovery_evidence is not None:
                        values = recovery_evidence.metrics()
                        report["recovery_evidence_measurements"] = values
                        if runner.logger.writer is not None:
                            for name, value in values.items():
                                runner.logger.writer.add_scalar(name, value,
                                    report["parent_updates"] + report["successful_updates"] - 1)
                    if source_birth_exposure is not None:
                        values = source_birth_exposure.metrics()
                        report["source_birth_exposure_measurements"] = values
                        if runner.logger.writer is not None:
                            for name, value in values.items():
                                runner.logger.writer.add_scalar(name, value,
                                    report["parent_updates"] + report["successful_updates"] - 1)
                            if not report.get("inherited_flat_exposure_published", False):
                                history_values = source_birth_exposure.flat_history_metrics()
                                for name, value in history_values.items():
                                    runner.logger.writer.add_scalar(name, value, report["parent_updates"] - 1)
                                if history_values:
                                    report["inherited_flat_exposure_published"] = True
                    episode_book = getattr(env, "reward_episode_journal", None)
                    if episode_book is not None:
                        report["reward_episode_measurements"] = episode_book.summary()
                        if runner.logger.writer is not None:
                            for name, value in episode_book.metrics().items():
                                runner.logger.writer.add_scalar(name, value,
                                    report["parent_updates"] + report["successful_updates"] - 1)
                    if pending_checkpoint is not None:
                        from chassis_checkpoints import seal_checkpoint
                        completed, progress = pending_checkpoint
                        pending_checkpoint = None
                        if curriculum_runtime is not None:
                            progress["v7_curriculum_measurements"] = env.curriculum_controller.metrics()
                        if recovery_evidence is not None:
                            progress["recovery_evidence_measurements"] = recovery_evidence.metrics()
                        if source_birth_exposure is not None:
                            progress["source_birth_exposure_measurements"] = source_birth_exposure.metrics()
                        snapshot = timed_call(timing, "checkpoint", seal_checkpoint,
                            args.run_dir, completed, runner.save, args.contract, progress,
                            budget_owner=budget_owner)
                        if pending_assessment is not None and pending_assessment[0] is None:
                            pending_assessment = (snapshot / "model.pt", *pending_assessment[1:])
                    if ("formal_critic_adaptation" in c and not actor_enabled
                            and report["parent_updates"] + report["successful_updates"] >= warmup_updates):
                        completed = report["parent_updates"] + report["successful_updates"]
                        checkpoint = control_checkpoint(completed)
                        assessment = {"status": "awaiting_review", "qualification": "not_assessed",
                            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
                            "stage_actor_update": env.stage_actor_update,
                            "global_actor_update": env.global_actor_update,
                            "optimizer_updates": completed, "actor_updates_in_block": report["actor_updates_in_block"],
                            "actor_invariant": report["critic_only_actor_invariant"],
                            "value_fit": report.get("critic_only_value_readback"),
                            "automatic_actor_release": False}
                        (args.run_dir / "critic_adaptation_review.json").write_text(
                            json.dumps(assessment, indent=2, allow_nan=False) + "\n")
                        report["critic_adaptation_review"] = assessment
                        raise PlannedStop("critic_adaptation_window_review")
                    if (actor_enabled and c.get("formal_v7_flat") is True
                            and report["parent_updates"] + report["successful_updates"] >= formal_assessment_update):
                        checkpoint = snapshot / "model.pt"
                        assessment = {"status": "assessment_pending", "clock": "learning_updates",
                            "boundary_update": formal_assessment_update,
                            "learning_updates": report["parent_updates"] + report["successful_updates"],
                            "stage_actor_update": env.stage_actor_update,
                            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
                            "training_budget": report["training_budget"],
                            "gate_result": "not_assessed", "automatic_promotion": False}
                        if c["formal_stage_policy"].get("assessment_mode", "pause_for_assessment") == "record_and_continue":
                            assessment.update(status="assessment_requested", assessment_mode="record_and_continue",
                                              training_continues=True)
                            requests = args.run_dir / "assessment_requests"
                            requests.mkdir(exist_ok=True)
                            request = requests / f"update_{assessment['learning_updates']}.tmp"
                            request.write_text(json.dumps(assessment, indent=2, allow_nan=False) + "\n")
                            request.replace(request.with_suffix(".json"))
                            latest = args.run_dir / "assessment_request.tmp"
                            latest.write_text(json.dumps(assessment, indent=2, allow_nan=False) + "\n")
                            latest.replace(args.run_dir / "assessment_request.json")
                            report["latest_assessment_request"] = assessment
                            formal_assessment_update = next_formal_assessment_update(c, assessment["learning_updates"])
                            report["formal_assessment_schedule"]["next_assessment_update"] = formal_assessment_update
                        else:
                            request = args.run_dir / "assessment_pending.tmp"
                            request.write_text(json.dumps(assessment, indent=2, allow_nan=False) + "\n")
                            request.replace(args.run_dir / "assessment_pending.json")
                            report["assessment_pending"] = assessment
                            report["stop_reason"] = "assessment_pending"
                            # Return normally from the paid optimizer wrapper. The
                            # logger boundary checks this only after SDK logging.
                            budget.request_stop()
                    return result

                last_publish = 0.
                if args.publish_state:
                    from wheeled_tasks.chassis.state_snapshot import read_selected_physics, select_group_environments
                    selected, selected_indices = select_group_environments(env.scene_groups, env.device)

                def publish_state():
                    nonlocal last_publish
                    wall_time = time.time()
                    snapshot = read_selected_physics(selected_indices,
                        poses=env.robot.data.body_link_pose_w.torch, origins=env.origins,
                        commands=env.commands, episode_steps=env.episode_length_buf)
                    state = {**identity, "wall_time_unix": wall_time, "run_id": args.run_dir.name,
                        "snapshot_kind": "post_step_after_auto_reset", "env_id": 0,
                        "sim_time_s": env.tick * c["policy_dt"], "episode_step": int(snapshot.episode_steps[0]),
                        "body_names": list(env.robot.body_names), "quaternion_order": "xyzw",
                        "body_link_pose_w": snapshot.poses[0], "commands": snapshot.commands[0]}
                    state["environments"] = [{"env_index": i, "scene_group": env.scene_groups[i],
                        "terrain": env.kinds[i], "origin": snapshot.origins[row],
                        "surfaces": [vars(s) for s in env.surfaces[i]],
                        "body_link_pose_w": snapshot.poses[row], "commands": snapshot.commands[row],
                        "episode_step": int(snapshot.episode_steps[row])} for row, i in enumerate(selected)]
                    temp = args.run_dir / "live_state.tmp"
                    temp.write_text(json.dumps(state, allow_nan=False))
                    temp.replace(args.run_dir / "live_state.json")
                    last_publish = time.monotonic()

                def step(actions):
                    nonlocal last_publish, consecutive_closure_failures
                    budget.check()
                    if budget_owner is not None:
                        from wheeled_tasks.chassis.training_budget import BudgetLimit
                        try:
                            budget_owner.check_runtime()
                        except BudgetLimit as exc:
                            raise PlannedStop(str(exc)) from exc
                    if curriculum_runtime is not None and runner.alg.storage.step == 0:
                        curriculum_runtime.begin_rollout(actor_enabled=(
                            report["parent_updates"] + report["successful_updates"] >= warmup_updates))
                    if recovery_evidence is not None and runner.alg.storage.step == 0:
                        recovery_evidence.begin_rollout(actor_enabled=(
                            report["parent_updates"] + report["successful_updates"] >= warmup_updates))
                    if (source_birth_exposure is not None
                            and runner.alg.storage.step == 0):
                        source_birth_exposure.begin_rollout(actor_enabled=(
                            report["parent_updates"] + report["successful_updates"] >= warmup_updates))
                    transition = timed_call(timing, "env_step", original_step, actions)
                    if mixed_exposure is not None:
                        mixed_exposure.observe(runner.alg.storage.step, transition[3])
                    if source_birth_exposure is not None:
                        source_birth_exposure.observe_transition(step_index=runner.alg.storage.step,
                            policy_interval_id=env.tick, physical_tick=env.physical_tick, info=transition[3])
                    if c.get("formal_v7_flat"):
                        observations, reward, _, info = transition
                        if (not bool(torch.isfinite(actions).all())
                                or not bool(torch.isfinite(reward).all())
                                or any(not bool(torch.isfinite(value).all()) for value in observations.values())):
                            raise FloatingPointError("Formal V7 non-finite action, reward or observation")
                        numerical = info.get("diagnostics", {}).get("reasons", {}).get("numerical_safety")
                        if (numerical_failure_policy == "abort-worker"
                                and numerical is not None and bool(numerical.any())):
                            raise FloatingPointError("Formal V7 native numerical safety termination")
                        closure = info.get("diagnostics", {}).get("reasons", {}).get("closure_gap")
                        majority_failed = closure is not None and bool(closure.sum() > args.num_envs // 2)
                        consecutive_closure_failures = consecutive_closure_failures + 1 if majority_failed else 0
                        if not c.get("formal_closure_monitor_only") and consecutive_closure_failures >= 8:
                            raise PlannedStop("Physical failure: more than half of lanes exceed 3 mm closure for eight consecutive policy steps")
                    episode_book = getattr(env, "reward_episode_journal", None)
                    if episode_book is not None:
                        timed_call(timing, "reward_episode_audit", episode_book.observe_transition,
                            transition[1], transition[2],
                            payments=transition[3]["final_reward_payments"],
                            payload=transition[3]["reward_episode_transition"])
                    if curriculum_runtime is not None:
                        payload = transition[3]["curriculum_transition"]
                        curriculum_runtime.observe_transition(step_index=runner.alg.storage.step,
                            node_indices=payload["node_indices"], episode_ids=payload["episode_ids"],
                            category_indices=payload["category_indices"],
                            returned_episode_ids=payload["returned_episode_ids"],
                            learning_mask=transition[3]["policy_learning_mask"], done=transition[2],
                            episode_complete=payload["episode_complete"],
                            key_phase_events=payload["key_phase_events"],
                            key_phase_occurrence_ids=payload["key_phase_occurrence_ids"])
                    if recovery_evidence is not None:
                        recovery_evidence.observe_transition(step_index=runner.alg.storage.step,
                            payload=transition[3]["recovery_transition"],
                            source_contact_or_attitude=transition[3].get("diagnostics", {}).get(
                                "reasons", {}).get("source_attitude_or_contact"))
                    if learning_audit is not None:
                        timed_call(timing, "diagnostics", learning_audit.observe,
                            transition[3]["diagnostics"], transition[3]["native_reward_components"],
                            transition[3])
                    if c.get("evaluation", {}).get("regression_recovery"):
                        transition[3]["log"].update({
                            "/recovery/learning_rate": runner.alg.learning_rate,
                            "/recovery/discarded_updates_charged": report.get("discarded_updates_charged", 0)})
                    if metrics is not None:
                        timed_call(timing, "diagnostics", metrics.observe, transition[3]["diagnostics"])
                    if args.publish_state and time.monotonic() - last_publish >= .25:
                        timed_call(timing, "state_publish", publish_state)
                    return transition

                def save(path, infos=None, *, budget_publication_path=None):
                    entropy_coef = verify_entropy_coef(runner.alg, effective_algorithm["entropy_coef"])
                    joint_metadata = (learning_metadata(integrated_receipt, env.task_actor_update,
                        completed=report["status"] == "completed", successful_updates=report["successful_updates"]) if c.get("integrated_production") else {})
                    if joint_metadata:
                        report.update(joint_metadata)
                    if budget_publication_path is not None and budget_owner is None:
                        raise ValueError("Deferred checkpoint publication requires its budget owner")
                    paid_checkpoint = (budget_owner.prepare_checkpoint(
                        path if budget_publication_path is None else budget_publication_path)
                        if budget_owner is not None else None)
                    paid_metadata = {"training_budget": paid_checkpoint} if paid_checkpoint is not None else {}
                    curriculum_metadata = {}
                    recovery_metadata = ({"recovery_evidence_state": recovery_evidence.checkpoint_state()}
                        if recovery_evidence is not None else {})
                    birth_metadata = ({"source_birth_exposure_state": source_birth_exposure.checkpoint_state()}
                        if source_birth_exposure is not None else {})
                    if mixed_exposure is not None:
                        birth_metadata["mixed_course_exposure_state"] = mixed_exposure.state_dict()
                    if (source_birth_exposure is not None
                            and source_birth_exposure.final_birth_definition_sha256 is not None):
                        birth_metadata["source_birth_exposure_flat_history"] = source_birth_exposure.checkpoint_flat_history()
                    jump_measurement = getattr(env, "jump_actuation_measurement", None)
                    jump_metadata = ({"jump_actuation_measurement_state": jump_measurement.checkpoint_state()}
                        if jump_measurement is not None else {})
                    jump_trace = getattr(env, "jump_physics_trace", None)
                    jump_trace_metadata = ({"jump_physics_trace_state": jump_trace.checkpoint_state()}
                        if jump_trace is not None else {})
                    episode_book = getattr(env, "reward_episode_journal", None)
                    episode_metadata = ({"reward_episode_measurement_state": episode_book.checkpoint_state()}
                        if episode_book is not None else {})
                    if curriculum_runtime is not None:
                        curriculum_metadata = {
                            "v7_curriculum_state": curriculum_runtime.checkpoint_state(
                                budget_receipt=paid_checkpoint, env_episode_ids=env.flat_episode.episode_id),
                            "v7_flat_episode_state": env.flat_episode.state_dict(env.generator),
                            "native_physical_exact_resume_qualified": False}
                    timed_call(timing, "checkpoint", original_save, path, infos={**identity, **joint_metadata, **paid_metadata, "stage": args.stage,
                         **curriculum_metadata,
                         **course_rng_metadata,
                         **recovery_metadata,
                         **birth_metadata,
                         **jump_metadata,
                         **jump_trace_metadata,
                         **episode_metadata,
                         **({"round_trip_rng_state": env.round_trip_transport.generator.get_state(),
                             "round_trip_transport_configuration": env.round_trip_transport.cfg}
                            if getattr(env, "round_trip_transport", None) is not None else {}),
                         **({"profile_source_sha256": source_hashes} if uses_profile(c) else {}),
                         **({"formal_critic_adaptation": c["formal_critic_adaptation"]}
                            if "formal_critic_adaptation" in c else {}),
                         "entropy_coef": entropy_coef,
                         "successful_updates_total": report["parent_updates"] + report["successful_updates"],
                         "learning_lineage_updates": report.get("learning_lineage_updates", 0)
                            + (report["actor_updates_in_block"] if "formal_critic_adaptation" in c
                               else report["successful_updates"]),
                         "critic_warmup_remaining": max(0, warmup_updates - report["parent_updates"] - report["successful_updates"]),
                         "stage_actor_update": env.stage_actor_update,
                         "global_actor_update_offset": args.global_actor_update_offset,
                         "global_actor_update": env.global_actor_update,
                         "actor_updates_consumed": args.actor_updates_consumed + report["actor_updates_in_block"],
                         **({"task_campaign": c["task_campaign"], "task_actor_update": env.task_actor_update}
                            if c.get("task_campaign") else {}),
                         "confirmed_jump_bins_m": confirmed_bins,
                         "batch_transitions": args.num_envs * rollout_steps,
                         "diagnostic_origin": report.get("diagnostic_origin"),
                        "training_transitions": report["parent_training_transitions"] + report["successful_updates"] * args.num_envs * rollout_steps,
                        "rng_state": torch.get_rng_state(), "env_rng_state": env.generator.get_state(),
                        "curriculum_state": env.performance_curriculum.state_dict() if env.performance_curriculum is not None else None,
                        "cuda_rng_states": (torch.cuda.get_rng_state_all()
                            if args.device.startswith("cuda") and torch.cuda.is_available() else [])})

                    checkpoint_path = Path(path)
                    checkpoint_path.with_suffix(".contract.json").write_bytes(args.contract.read_bytes())
                    checkpoint_path.with_suffix(".checkpoint.json").write_text(json.dumps({
                        "schema": "chassis-learning-checkpoint-v1", "sealed": True, **joint_metadata, **paid_metadata,
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
                    if budget_owner is not None and budget_publication_path is None:
                        budget_owner.register_checkpoint(checkpoint_path, paid_checkpoint)
                    return paid_checkpoint

                runner.alg.update, env.step, runner.save = paid_update, step, save
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
                if c.get("formal_v7_flat") is True:
                    formal_original_log = runner.logger.log

                    def formal_log(*a, **kw):
                        result = formal_original_log(*a, **kw)
                        if report.get("assessment_pending") is not None:
                            budget.check()
                        return result

                    runner.logger.log = formal_log
                def control_checkpoint(completed):
                    from chassis_checkpoints import seal_checkpoint
                    control_root = args.run_dir / "control"
                    saved = control_root / "checkpoints" / f"update_{completed:08d}" / "model.pt"
                    if saved.exists():
                        if budget_owner is not None:
                            budget_owner.verify_resume(saved)
                        return saved
                    snapshot = seal_checkpoint(control_root, completed, runner.save,
                        args.contract, {"parent_updates": report["parent_updates"],
                                        "successful_updates": report["successful_updates"]},
                        budget_owner=budget_owner)
                    return snapshot / "model.pt"

                def control_flush():
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    if runner.logger.writer is not None:
                        runner.logger.writer.flush()

                def control_limits():
                    budget.check()
                    if budget_owner is not None:
                        from wheeled_tasks.chassis.training_budget import BudgetLimit
                        try:
                            budget_owner.check_runtime()
                        except BudgetLimit as exc:
                            raise PlannedStop(str(exc)) from exc

                def control_stop():
                    budget.request_stop()
                    budget.check()

                control_original_log = runner.logger.log

                def controlled_log(*a, **kw):
                    result = control_original_log(*a, **kw)
                    training_control.boundary(report["parent_updates"] + report["successful_updates"],
                        checkpoint=control_checkpoint, flush=control_flush,
                        check_limits=control_limits, stop=control_stop)
                    return result

                runner.logger.log = controlled_log
                if c.get("evaluation", {}).get("regression_recovery") or c.get("diagnostic_study") or budget_owner is not None:
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
                if report.get("assessment_pending") is not None:
                    budget.check()  # Also handle a boundary at learn's final iteration.
                report["status"] = "completed"
    except PlannedStop as exc:
        report.update(status="stopped", reason="assessment_pending" if report.get("assessment_pending") else exc.reason)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        try:
            if profiler is not None:
                profiler.disable()
                profiler.dump_stats(str(args.run_dir / "profile.pstats"))
            if recovery_evidence is not None:
                try:
                    recovery_evidence.discard_rollout()
                    report["recovery_evidence_measurements"] = recovery_evidence.metrics()
                    report["recovery_evidence_state"] = recovery_evidence.checkpoint_state()
                except Exception:
                    report.update(status="failed", recovery_evidence_error=traceback.format_exc())
            if source_birth_exposure is not None:
                try:
                    source_birth_exposure.discard_rollout()
                    report["source_birth_exposure_measurements"] = source_birth_exposure.metrics()
                    report["source_birth_exposure_state"] = source_birth_exposure.checkpoint_state()
                except Exception:
                    report.update(status="failed", source_birth_exposure_error=traceback.format_exc())
            jump_measurement = getattr(env, "jump_actuation_measurement", None)
            if jump_measurement is not None:
                try:
                    report["jump_actuation_measurement"] = jump_measurement.summary()
                except Exception:
                    report.update(status="failed", jump_actuation_measurement_error=traceback.format_exc())
            jump_trace = getattr(env, "jump_physics_trace", None)
            if jump_trace is not None:
                try:
                    report["jump_physics_trace"] = jump_trace.summary()
                except Exception:
                    report.update(status="failed", jump_physics_trace_error=traceback.format_exc())
            episode_book = getattr(env, "reward_episode_journal", None)
            if episode_book is not None:
                try:
                    report["reward_episode_measurements"] = episode_book.summary()
                except Exception:
                    report.update(status="failed", reward_episode_measurement_error=traceback.format_exc())
            if runner is not None:
                # Adaptive PPO can change LR within a block. Recovery must consume
                # the optimizer's final value, not the value at block entry.
                try:
                    report["learning_rate"] = completion_learning_rate(runner.alg.optimizer)
                except ValueError as error:
                    report.update(status="failed", learning_rate=None,
                        learning_rate_status="invalid_native_value", learning_rate_invalid_native_value=str(error))
                    report.setdefault("error", str(error))
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
                if report.get("integrated_admission_sha256") is not None:
                    final_metadata = learning_metadata(integrated_receipt, env.task_actor_update,
                        completed=report["status"] == "completed", successful_updates=report["successful_updates"])
                    report.update(final_metadata)
                    progress_path = args.run_dir / "progress.json"
                    final_progress = json.loads(progress_path.read_text())
                    final_progress.update(final_metadata)
                    write_progress(args.run_dir, final_progress, timing)
                runner.save(str(path))
                report["checkpoint_sha256"] = digest(path)
                try:
                    from wheeled_algo.chassis_export import export_actor
                    export_identity = identity
                    if c.get("v5_control", {}).get("leg_winding_version") in ("paired_common_v1", "paired_branch107_v1"):
                        if c.get("formal_v7_flat") is not True:
                            raise ValueError("Paired export requires the explicit V7 profile")
                        from wheeled_algo.leg_winding_export import export_actor
                        export_identity = {**identity, "leg_winding_version": c["v5_control"]["leg_winding_version"]}
                    report["export"] = export_actor(runner.alg.actor, args.run_dir, export_identity,
                                                     env.get_observations()["policy"])
                except Exception:
                    report.update(status="export_failed", export_error=traceback.format_exc())
                    traceback.print_exc()
            if env is not None:
                capture = getattr(env, "_numerical_failure_capture", None)
                if capture is not None and capture.summary["worker_abort_raised"]:
                    # The native state was deliberately not reset or stepped
                    # again. Its NaN/Inf belongs in the raw PT, not strict JSON.
                    report["numerical_failure_capture"] = dict(capture.summary)
                    report["metrics"] = {"available": False,
                        "reason": "first_substep_numerical_abort_before_policy_end",
                        "raw_state_file": str(capture.path),
                        "existing_metric_files": "last previously written completed updates; not final native state"}
                else:
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
            if budget_owner is not None:
                report["training_budget"] = budget_owner.summary()
            (args.run_dir / "completion.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            training_control.finish(report["status"], reason=report.get("reason"),
                learning_updates=report["parent_updates"] + report["successful_updates"],
                final_checkpoint=str(args.run_dir / "model_final.pt") if report.get("checkpoint_sha256") else None)
            print("CHASSIS_COMPLETION", report["status"], report["successful_updates"], flush=True)
            # Kit close can exit the process directly; release the existing
            # lease first and preserve failure status in the native exit code.
            if budget_owner is not None:
                closing_owner, budget_owner = budget_owner, None
                closing_owner.close()
            if launcher is not None:
                if physx_errors is not None:
                    physx_errors.close()
                launcher.app.close(exit_code=0 if report["status"] in
                    ("completed", "scene_validated", "stopped") else 1)
        finally:
            if budget_owner is not None:
                pending_cleanup_error = sys.exc_info()[1]
                try:
                    budget_owner.close()
                except BaseException as close_error:
                    if pending_cleanup_error is None:
                        raise
                    # Preserve the original failure; the kernel still releases
                    # the flock at process exit, and no paid work is refunded.
                    pending_cleanup_error.add_note(f"Budget lease close also failed: {close_error!r}")
    return 0 if report["status"] in ("completed", "scene_validated", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
